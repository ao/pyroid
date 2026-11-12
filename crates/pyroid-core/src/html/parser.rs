// Derived from libxml2 2.14.6 (MIT), © Daniel Veillard and contributors; modified.
//! A port of the pull-parser path of libxml2 2.14.6 HTMLparser.c (`htmlCtxtReadMemory` ->
//! `htmlParseDocument`), with the SAX2 tree builder (SAX2.c) folded in, for the configuration
//! lxml 6.1.3 uses for `HTMLParser(collect_ids=False, default_doctype=False, encoding="utf-8",
//! remove_comments=True, remove_pis=True)`:
//!
//! - options RECOVER | NONET | COMPACT | NODEFDTD (no HUGE, no NOBLANKS, so keepBlanks = 1; no
//!   NOIMPLIED, no HTML5);
//! - `sax->comment` and `sax->processingInstruction` are NULL (comments are parsed and dropped;
//!   the 2.14 tokenizer turns `<?...>` into a bogus comment), `sax->cdataBlock` is NULL (lxml's
//!   strip_cdata), so raw text goes through `sax->characters`;
//! - the input always has XML_INPUT_HAS_ENCODING (lxml passes an encoding name on the str and
//!   the bytes path), so `<meta charset>` never switches encodings and the tokenizer sees UTF-8
//!   ([`super::lxml`] turns either input into that UTF-8).
//!
//! The tree is built directly in lxml's text/tail model: libxml2 appends text to the last text
//! child of the current node or creates one, and lxml's `.text`/`.tail` concatenate adjacent
//! text nodes, so appending to `.text` (no child yet) or to the last child's `.tail` gives the
//! same strings. No comment, PI, CDATA or entity nodes are created in this configuration.
//! Top-level elements after the first (libxml2 starts a second `<html>` for content after
//! `</html>`) are siblings of the root, as in lxml, where `getnext()` on the root returns them.
//!
//! Function names in comments refer to the C functions; behaviour follows the C code, including
//! its oddities (for example a discarded `<body/>` pops the current element).
//!
//! Limits, as in libxml2 without XML_PARSE_HUGE: the 256-element node stack (`nodePush`) and
//! the 10,000,000-byte text node (`xmlSAX2Text`) halt the parse; attribute values, comments and
//! DOCTYPE parts longer than their limits fail as in `htmlParseData` ("value too long");
//! XML_MAX_LOOKUP_LIMIT halts the parse when `xmlParserGrow` is called with more than
//! 10,000,000 bytes between the buffer's base and the current position, which happens when one
//! token (a text chunk, an attribute value, a comment, a run of blanks) spans that much input.
//! libxml2 reads the input in 4000-byte pieces and shrinks the buffer between tokens, so the
//! last limit depends on where the buffer's base and end are at each grow call: for inputs over
//! 10,000,000 bytes the parser emulates them ([`Feed`] says how the buffer fills). Smaller inputs
//! can never reach the limit, and are parsed as one buffer.

use std::sync::LazyLock;

use super::entities::{ENT_ALPHA, ENT_STRINGS, ENT_VALUES};
use super::tables::{
    BOOLEAN_ATTRS, DATA_PLAINTEXT, DATA_RCDATA, DATA_SCRIPT, DATA_SCRIPT_ESC1, DATA_SCRIPT_ESC2,
    ELEMENTS, END_PRIORITY, END_PRIORITY_DEFAULT, START_CLOSE,
};
use super::tree::{Attr, Document, Kind, NIL, Name, NodeId, names, static_name};

const HTML_PARSER_BUFFER_SIZE: usize = 100;
/// XML_MAX_TEXT_LENGTH: attribute values, comments, text nodes (no HTML_PARSE_HUGE).
const XML_MAX_TEXT_LENGTH: usize = 10_000_000;
/// XML_MAX_NAME_LENGTH: DOCTYPE name and literals.
const XML_MAX_NAME_LENGTH: usize = 50_000;
/// XML_MAX_LOOKUP_LIMIT: `xmlParserGrow` halts when `cur - base` exceeds it.
pub const XML_MAX_LOOKUP_LIMIT: usize = 10_000_000;
/// nodePush: the node stack cannot grow beyond 256 entries without XML_PARSE_HUGE.
const MAX_DEPTH: usize = 256;
/// INPUT_CHUNK (parserInternals.h): GROW reads when fewer bytes than this are left.
const INPUT_CHUNK: usize = 250;
/// LINE_LEN (parserInternals.c): `xmlParserShrink` keeps this many bytes before `cur`.
const LINE_LEN: usize = 80;
/// MINLEN (xmlIO.c): bytes per `xmlParserInputBufferGrow` read.
pub const MINLEN: usize = 4000;

const INSERT_INITIAL: u8 = 1;
const INSERT_IN_HEAD: u8 = 3;
const INSERT_IN_BODY: u8 = 10;

type Mask = [u32; 2];
const MASK_DQ: Mask = [0, 1 << (b'"' - 32)];
const MASK_SQ: Mask = [0, 1 << (b'\'' - 32)];
const MASK_GT: Mask = [0, 1 << (b'>' - 32)];
const MASK_DASH: Mask = [0, 1 << (b'-' - 32)];
const MASK_WS_GT: Mask = [
    1 << 0x09 | 1 << 0x0A | 1 << 0x0C | 1 << 0x0D,
    1 /* ' ' */ | 1 << (b'>' - 32),
];
const MASK_DQ_GT: Mask = [0, 1 << (b'"' - 32) | 1 << (b'>' - 32)];
const MASK_SQ_GT: Mask = [0, 1 << (b'\'' - 32) | 1 << (b'>' - 32)];

const FFFD: &str = "\u{FFFD}";

/// `htmlStartClose` and the element table by [`Name`] id.
struct ElementInfo {
    /// `html40ElementTable`: (`empty`, `dataMode`).
    elem: Vec<Option<(bool, u8)>>,
    priority: Vec<i32>,
    boolean: Vec<bool>,
    /// Bit `new` of `closes[old]`: a start tag `new` closes an open `old` (ids below 256).
    closes: Vec<[u64; 4]>,
}

static INFO: LazyLock<ElementInfo> = LazyLock::new(|| {
    let n = names::STATIC_COUNT as usize;
    let mut info = ElementInfo {
        elem: vec![None; n],
        priority: vec![END_PRIORITY_DEFAULT; n],
        boolean: vec![false; n],
        closes: vec![[0; 4]; n],
    };
    // Every name in the tables is a static name below 256 (checked by a test).
    for &(name, empty, mode) in &ELEMENTS {
        if let Some(t) = static_name(name) {
            info.elem[t.0 as usize] = Some((empty, mode));
        }
    }
    for &(name, p) in &END_PRIORITY {
        if let Some(t) = static_name(name) {
            info.priority[t.0 as usize] = p;
        }
    }
    for name in BOOLEAN_ATTRS {
        if let Some(t) = static_name(name) {
            info.boolean[t.0 as usize] = true;
        }
    }
    for &(old, new) in &START_CLOSE {
        if let (Some(o), Some(t)) = (static_name(old), static_name(new)) {
            if t.0 < 256 {
                info.closes[o.0 as usize][(t.0 / 64) as usize] |= 1 << (t.0 % 64);
            }
        }
    }
    info
});

/// htmlTagLookup: (`empty`, `dataMode`) of a known element.
#[inline]
fn tag_lookup(t: Name) -> Option<(bool, u8)> {
    if t.is_static() {
        INFO.elem[t.0 as usize]
    } else {
        None
    }
}

/// htmlCheckAutoClose: does a start tag `newtag` close an open `oldtag`?
#[inline]
fn check_auto_close(newtag: Name, oldtag: Name) -> bool {
    oldtag.is_static()
        && newtag.0 < 256
        && (INFO.closes[oldtag.0 as usize][(newtag.0 / 64) as usize] >> (newtag.0 % 64)) & 1 == 1
}

/// htmlGetEndPriority.
#[inline]
fn end_priority(t: Name) -> i32 {
    if t.is_static() {
        INFO.priority[t.0 as usize]
    } else {
        END_PRIORITY_DEFAULT
    }
}

/// htmlIsBooleanAttr (names reaching it are ASCII-lowercased already).
#[inline]
fn is_boolean_attr(t: Name) -> bool {
    t.is_static() && INFO.boolean[t.0 as usize]
}

/// Bytes the `htmlParseCharData` accelerator stops at. The C accelerator stops at NUL, LF, CR,
/// '&', '-', '<' and bytes >= 0x80 ([`C_STOP`]); this one also skips the bytes the switch after
/// it would only step over in the current mode: LF (line counting only), '&' outside data and
/// RCDATA, '-' outside the script escape states, and >= 0x80 when the input is known to be
/// valid UTF-8 (`htmlValidateUtf8` only rejects ill-formed UTF-8). Indexed by
/// `[valid_utf8][mode]`. The buffer emulation uses [`C_STOP`], so that grow calls happen where
/// libxml2 makes them.
static CHARDATA_STOP: [[[bool; 256]; 7]; 2] = [stop_tables(false), stop_tables(true)];

const fn stop_tables(valid_utf8: bool) -> [[bool; 256]; 7] {
    let mut out = [[false; 256]; 7];
    let mut mode = 0;
    while mode < 7 {
        let mut t = [false; 256];
        let mut c = 0x80;
        while c < 256 {
            t[c] = !valid_utf8;
            c += 1;
        }
        t[0] = true;
        t[b'\r' as usize] = true;
        t[b'<' as usize] = true;
        if mode == 0 || mode == DATA_RCDATA as usize {
            t[b'&' as usize] = true;
        }
        if mode == DATA_SCRIPT_ESC1 as usize || mode == DATA_SCRIPT_ESC2 as usize {
            t[b'-' as usize] = true;
        }
        out[mode] = t;
        mode += 1;
    }
    out
}

/// Bytes `htmlParseData` has to look at, by input validity (`[valid_utf8][byte]`): its
/// terminators, '&' when references are expanded, NUL, CR, and bytes >= 0x80 unless the input
/// is valid UTF-8. The others it only steps over.
type DataStop = [[bool; 256]; 2];

const fn data_stop_table(mask: Mask, refs: bool) -> DataStop {
    let mut out = [[false; 256]; 2];
    let mut v = 0;
    while v < 2 {
        let mut c = 0;
        while c < 256 {
            #[allow(clippy::cast_possible_truncation)] // c < 256
            let b = c as u8;
            out[v][c] = (b < 64 && (mask[(b / 32) as usize] >> (b & 31)) & 1 == 1)
                || (refs && b == b'&')
                || b == 0
                || b == b'\r'
                || (b >= 0x80 && v == 0);
            c += 1;
        }
        v += 1;
    }
    out
}

static STOP_DQ_REFS: DataStop = data_stop_table(MASK_DQ, true);
static STOP_SQ_REFS: DataStop = data_stop_table(MASK_SQ, true);
static STOP_WS_GT_REFS: DataStop = data_stop_table(MASK_WS_GT, true);
static STOP_GT: DataStop = data_stop_table(MASK_GT, false);
static STOP_DASH: DataStop = data_stop_table(MASK_DASH, false);
static STOP_WS_GT: DataStop = data_stop_table(MASK_WS_GT, false);
static STOP_DQ_GT: DataStop = data_stop_table(MASK_DQ_GT, false);
static STOP_SQ_GT: DataStop = data_stop_table(MASK_SQ_GT, false);
/// Every byte (no acceleration).
static STOP_ALL: DataStop = [[true; 256]; 2];

fn data_stop(mask: Mask, refs: bool) -> &'static DataStop {
    match (mask, refs) {
        (MASK_DQ, true) => &STOP_DQ_REFS,
        (MASK_SQ, true) => &STOP_SQ_REFS,
        (MASK_WS_GT, true) => &STOP_WS_GT_REFS,
        (MASK_GT, false) => &STOP_GT,
        (MASK_DASH, false) => &STOP_DASH,
        (MASK_WS_GT, false) => &STOP_WS_GT,
        (MASK_DQ_GT, false) => &STOP_DQ_GT,
        (MASK_SQ_GT, false) => &STOP_SQ_GT,
        _ => &STOP_ALL,
    }
}

/// The `htmlParseCharData` accelerator mask as in C.
static C_STOP: [bool; 256] = {
    let mut t = [false; 256];
    let mut c = 0x80;
    while c < 256 {
        t[c] = true;
        c += 1;
    }
    t[0] = true;
    t[b'\n' as usize] = true;
    t[b'\r' as usize] = true;
    t[b'&' as usize] = true;
    t[b'-' as usize] = true;
    t[b'<' as usize] = true;
    t
};

#[inline]
fn is_ws(c: u8) -> bool {
    c == 0x20 || ((0x09..=0x0D).contains(&c) && c != 0x0B)
}

#[inline]
fn is_letter(c: u8) -> bool {
    (0x61..=0x7a).contains(&(c | 0x20))
}

#[inline]
fn is_alnum(c: u8) -> bool {
    is_letter(c) || c.is_ascii_digit()
}

#[inline]
fn is_hex(c: u8) -> bool {
    c.is_ascii_digit() || ((c | 0x20) >= b'a' && (c | 0x20) <= b'f')
}

#[inline]
fn mask_match(m: Mask, c: u8) -> bool {
    c < 64 && (m[(c / 32) as usize] >> (c & 31)) & 1 == 1
}

/// htmlValidateUtf8 with partial = 0: the length of the valid UTF-8 sequence at `s[0]`, or None.
#[inline]
fn validate_utf8(s: &[u8]) -> Option<usize> {
    let c = u32::from(s[0]);
    let len = s.len();
    if c < 0xC2 {
        None
    } else if c < 0xE0 {
        (len >= 2 && s[1] & 0xC0 == 0x80).then_some(2)
    } else if c < 0xF0 {
        if len < 3 {
            return None;
        }
        let v = (c << 16) | u32::from(s[1]) << 8 | u32::from(s[2]);
        let bad =
            (v & 0x00_C0C0) != 0x00_8080 || (v & 0x0F_2000) == 0 || (v & 0x0F_2000) == 0x0D_2000;
        (!bad).then_some(3)
    } else {
        if len < 4 {
            return None;
        }
        let v = c << 24 | u32::from(s[1]) << 16 | u32::from(s[2]) << 8 | u32::from(s[3]);
        let bad = (v & 0x00C0_C0C0) != 0x0080_8080 || !(0xF090_0000..0xF490_0000).contains(&v);
        (!bad).then_some(4)
    }
}

/// htmlC1Remap.
const C1_REMAP: [u32; 32] = [
    0x20AC, 0x0081, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039,
    0x0152, 0x008D, 0x017D, 0x008F, 0x0090, 0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014,
    0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x009D, 0x017E, 0x0178,
];

/// htmlCodePointToUtf8: the character a numeric reference stands for.
fn code_point_char(mut c: u32) -> char {
    if (0x80..0xA0).contains(&c) {
        c = C1_REMAP[(c - 0x80) as usize];
    }
    // htmlCodePointToUtf8 maps 0, surrogates and values past U+10FFFF to U+FFFD.
    char::from_u32(c)
        .filter(|&ch| ch != '\0')
        .unwrap_or('\u{FFFD}')
}

/// strncmp(a, b, n) where `a(k)` reads the input zero-padded (the C buffer is NUL-terminated).
#[allow(clippy::many_single_char_names)] // strncmp's
fn strncmp(a: impl Fn(usize) -> u8, b: &[u8], n: usize) -> i32 {
    let mut k = 0;
    while k < n {
        let x = a(k);
        let y = b.get(k).copied().unwrap_or(0);
        if x != y {
            return i32::from(x) - i32::from(y);
        }
        if x == 0 {
            return 0;
        }
        k += 1;
    }
    0
}

/// htmlFindEntityPrefix: `s` starts after the '&' and ends with the buffer. Returns (bytes of
/// the name consumed, including a ';' if present, replacement text).
#[allow(clippy::many_single_char_names, clippy::comparison_chain)] // follows the C function
fn find_entity_prefix(s: &[u8], is_attr: bool) -> Option<(usize, &'static str)> {
    const F_SEMICOLON: u8 = 0x80;
    const F_SUBTABLE: u8 = 0x40;
    let slen = s.len();
    if slen < 2 {
        return None;
    }
    let first = s[0];
    if !is_letter(first) {
        return None;
    }
    let get = |k: usize| if k < slen { s[k] } else { 0 };
    let f = (first & 63) as usize;
    let mut left = ENT_ALPHA[f * 3] as usize | (ENT_ALPHA[f * 3 + 1] as usize) << 8;
    let mut right = left + ENT_ALPHA[f * 3 + 2] as usize;
    let mut found: Option<usize> = None;
    let mut match_len = 0usize;
    let mut soff = 1usize;
    while left < right {
        let mid = left + (right - left) / 2;
        let bo = ENT_VALUES[mid] as usize;
        let bytes = &ENT_STRINGS[bo..];
        let len = (bytes[0] & !(F_SEMICOLON | F_SUBTABLE)) as usize;
        let mut cmp = i32::from(get(soff)) - i32::from(bytes[1]);
        if cmp == 0 {
            if slen < len {
                cmp = strncmp(|k| get(soff + 1 + k), &bytes[2..], slen - 1);
                if cmp == 0 {
                    // Prefix can never match
                    break;
                }
            } else {
                cmp = strncmp(|k| get(soff + 1 + k), &bytes[2..], len.wrapping_sub(1));
            }
        }
        if cmp < 0 {
            right = mid;
        } else if cmp > 0 {
            left = mid + 1;
        } else {
            // cmp == 0: the name matches this entry.
            let term = if soff + len < slen { s[soff + len] } else { 0 };
            let alnum = is_alnum(term);
            let is_term = term == b';'
                || ((bytes[0] & F_SEMICOLON) != 0 && (!is_attr || (!alnum && term != b'=')));
            if is_term {
                found = Some(bo + len + 1);
                match_len = soff + len;
                if term == b';' {
                    match_len += 1;
                }
            }
            if bytes[0] & F_SUBTABLE != 0 {
                if is_term {
                    found = found.map(|m| m + 2);
                }
                if alnum && soff + len < slen {
                    left = mid + bytes[len + 1] as usize;
                    right = left + bytes[len + 2] as usize;
                    soff += len;
                    continue;
                }
            }
            break;
        }
    }
    let m = found?;
    let rlen = ENT_STRINGS[m] as usize;
    // The table holds UTF-8 (checked by a test).
    let r = std::str::from_utf8(&ENT_STRINGS[m + 1..m + 1 + rlen]).unwrap_or(FFFD);
    Some((match_len, r))
}

/// Replacement text of one chunk boundary (entity, character reference, U+FFFD, newline).
#[derive(Clone, Copy)]
enum Repl {
    None,
    Static(&'static str),
    Char(char),
}

impl Repl {
    #[inline]
    fn len(self) -> usize {
        match self {
            Repl::None => 0,
            Repl::Static(s) => s.len(),
            Repl::Char(c) => c.len_utf8(),
        }
    }

    #[inline]
    fn as_str(self, buf: &mut [u8; 4]) -> &str {
        match self {
            Repl::None => "",
            Repl::Static(s) => s,
            Repl::Char(c) => c.encode_utf8(buf),
        }
    }
}

/// htmlParseHTMLName's fixed buffer: a character that does not fit is dropped, later ones may
/// still fit (`nbchar + size <= HTML_PARSER_BUFFER_SIZE`).
struct NameBuf {
    b: [u8; HTML_PARSER_BUFFER_SIZE],
    n: usize,
}

impl NameBuf {
    #[inline]
    fn new() -> Self {
        NameBuf {
            b: [0; HTML_PARSER_BUFFER_SIZE],
            n: 0,
        }
    }

    #[inline]
    fn push(&mut self, s: &[u8]) {
        if self.n + s.len() <= HTML_PARSER_BUFFER_SIZE {
            self.b[self.n..self.n + s.len()].copy_from_slice(s);
            self.n += s.len();
        }
    }

    /// The name: ASCII, whole validated UTF-8 sequences and U+FFFD, so always UTF-8.
    #[inline]
    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.b[..self.n]).unwrap_or(FFFD)
    }
}

/// How libxml2's input buffer fills from lxml's buffer. Only inputs over
/// [`XML_MAX_LOOKUP_LIMIT`] bytes need `ends`; `error_read` also decides whether a decoding
/// error happened in the first read (which keeps a leading BOM).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Feed {
    /// Length of the parse input after each 4000-byte read of the raw buffer, when a decoder
    /// converts it (lxml's Latin-1, UCS-2 and UCS-4 str buffers). Empty: no decoder, each read
    /// adds 4000 bytes of the input itself (bytes input, or an ASCII str).
    pub ends: Vec<usize>,
    /// The 1-based read whose conversion fails (a lone surrogate on the str path): the input
    /// ends there, and every later grow call fails without checking the lookup limit.
    pub error_read: Option<usize>,
}

/// The result of a parse.
#[derive(Debug)]
pub struct Parsed {
    /// The tree; `root` is the first element child of the document node ([`NIL`] if none,
    /// lxml's "Document is empty"). Later top-level elements are the root's siblings.
    pub doc: Document,
    /// The parser stopped early (depth, text length or lookup limit), as `xmlHaltParser`.
    pub halted: bool,
    /// libxml2 returned a document. It does not when the parse halts before
    /// `startDocument` (more than 10,000,000 leading blanks), or after a memory error (more
    /// attributes in one tag than `xmlGrowCapacity` allows); lxml then raises XMLSyntaxError.
    pub document: bool,
    /// The libxml2 error message behind `halted` or `!document`, if any.
    pub error: Option<&'static str>,
}

/// The `maxatts` value at which `htmlParseStartTag` can no longer grow its attribute array
/// (`xmlGrowCapacity(maxatts, .., 11, HTML_MAX_ATTRS)`), whatever earlier documents left.
const MAXATTS_LIMIT: u64 = {
    const MAX: u64 = 100_000_000; // HTML_MAX_ATTRS
    let mut maxatts: u64 = 0;
    loop {
        // xmlGrowCapacity(capacity = maxatts, min = 11, max = MAX)
        let new_size = if maxatts == 0 {
            11
        } else if maxatts >= MAX {
            break;
        } else {
            let extra = maxatts.div_ceil(2);
            if maxatts > MAX - extra {
                MAX
            } else {
                maxatts + extra
            }
        };
        maxatts = new_size * 2;
    }
    maxatts
};

#[allow(clippy::struct_excessive_bools)] // libxml2's parser context flags
struct Parser<'a> {
    inp: &'a [u8],
    /// `inp` as a str when it is valid UTF-8 (then text is sliced from it with no copy check).
    inp_str: Option<&'a str>,
    cur: usize,
    /// `input->end`: how much of the input libxml2 has read (all of it unless emulating).
    end: usize,
    /// Emulate libxml2's buffer (`base`, `end`, reads) for the lookup limit.
    emulate: bool,
    /// `input->base`, as an offset into `inp`.
    base: usize,
    reads: usize,
    feed: &'a Feed,
    /// `buf->error`: the decoder failed; grows fail from then on.
    buf_error: bool,
    /// nodeTab (nameTab is the same list of tags until the parser halts).
    stack: Vec<NodeId>,
    html: u8,
    depth: i32,
    end_check_state: u8,
    stopped: bool,
    error: Option<&'static str>,
    started: bool,
    oom: bool,
    /// The whole input is well-formed UTF-8 (see `CHARDATA_STOP`).
    valid_utf8: bool,
    doc: Document,
    last_top: NodeId,
}

/// Parse UTF-8 `inp` as `htmlCtxtReadMemory` does with lxml's options; `feed` describes how
/// libxml2 reads it (see [`Feed`]).
pub fn parse(inp: &[u8], feed: &Feed) -> Parsed {
    let inp_str = std::str::from_utf8(inp).ok();
    let emulate = inp.len() > XML_MAX_LOOKUP_LIMIT;
    let mut p = Parser {
        inp,
        inp_str,
        cur: 0,
        end: if emulate { 0 } else { inp.len() },
        emulate,
        base: 0,
        reads: 0,
        feed,
        buf_error: false,
        stack: Vec::with_capacity(32),
        html: INSERT_INITIAL,
        depth: 0,
        end_check_state: 0,
        stopped: false,
        error: None,
        started: false,
        oom: false,
        valid_utf8: inp_str.is_some(),
        doc: Document::with_capacity(inp.len() / 40 + 16),
        last_top: NIL,
    };
    p.parse_document();
    Parsed {
        halted: p.stopped,
        document: p.started && !p.oom,
        error: p.error,
        doc: p.doc,
    }
}

impl<'a> Parser<'a> {
    /// The byte at `i`, or 0 at and past the end of the buffer (libxml2 NUL-terminates it).
    #[inline]
    fn b(&self, i: usize) -> u8 {
        if i < self.end { self.inp[i] } else { 0 }
    }

    #[inline]
    fn cur_b(&self) -> u8 {
        self.b(self.cur)
    }

    #[inline]
    fn nxt(&self, n: usize) -> u8 {
        self.b(self.cur + n)
    }

    #[inline]
    fn upp(&self, n: usize) -> u8 {
        self.nxt(n).to_ascii_uppercase()
    }

    fn upper_is(&self, from: usize, word: &[u8]) -> bool {
        word.iter()
            .enumerate()
            .all(|(k, &w)| self.upp(from + k) == w)
    }

    /// `inp[a..b]` as text. The tokenizer only cuts at ASCII bytes and around validated UTF-8
    /// sequences, so this never fails; a debug build checks.
    #[inline]
    fn text(&self, a: usize, b: usize) -> &'a str {
        let s = match self.inp_str {
            Some(s) => s.get(a..b),
            None => std::str::from_utf8(&self.inp[a..b]).ok(),
        };
        debug_assert!(s.is_some(), "chunk {a}..{b} is not UTF-8");
        s.unwrap_or(FFFD)
    }

    #[inline]
    fn tag_of(&self, id: NodeId) -> Name {
        self.doc.nodes[id as usize].tag
    }

    /// `ctxt->name`.
    #[inline]
    fn name(&self) -> Option<Name> {
        self.stack.last().map(|&id| self.tag_of(id))
    }

    fn halt(&mut self, error: &'static str) {
        if !self.stopped {
            self.error = Some(error);
        }
        self.stopped = true;
    }

    // ---- input buffer (emulated for inputs over XML_MAX_LOOKUP_LIMIT) --------------------

    /// One `xmlParserInputBufferGrow` read; false when the decoder failed.
    fn read(&mut self) -> bool {
        let ends = &self.feed.ends;
        if ends.is_empty() {
            if self.end < self.inp.len() {
                self.reads += 1;
                self.end = self.inp.len().min(MINLEN * self.reads);
            }
        } else if self.reads < ends.len() {
            self.end = ends[self.reads].min(self.inp.len());
            self.reads += 1;
            if self.feed.error_read == Some(self.reads) {
                self.buf_error = true;
                return false;
            }
        }
        true
    }

    /// `xmlParserGrow` with `input->cur` at `cur`; false when it returns -1.
    fn grow(&mut self, cur: usize) -> bool {
        if !self.emulate {
            return true;
        }
        if self.buf_error {
            return false;
        }
        if cur.saturating_sub(self.base) > XML_MAX_LOOKUP_LIMIT {
            self.halt("Resource limit exceeded: Buffer size limit exceeded, try XML_PARSE_HUGE");
            return false;
        }
        if self.end.saturating_sub(cur) >= INPUT_CHUNK {
            return true;
        }
        self.read()
    }

    /// The GROW macro.
    #[inline]
    fn grow_macro(&mut self, cur: usize) {
        if self.emulate && self.end.saturating_sub(cur) < INPUT_CHUNK {
            self.grow(cur);
        }
    }

    /// The SHRINK macro (`xmlParserShrink` keeps LINE_LEN bytes before `cur`).
    #[inline]
    fn shrink_macro(&mut self, cur: usize) {
        if self.emulate
            && cur.saturating_sub(self.base) > 2 * INPUT_CHUNK
            && self.end.saturating_sub(cur) < 2 * INPUT_CHUNK
        {
            self.base = cur - LINE_LEN;
        }
    }

    // ---- SAX2 tree building ---------------------------------------------------------------

    /// Link a new top-level element after the previous one (children of the document node).
    fn link_top(&mut self, id: NodeId) {
        if self.last_top == NIL {
            self.doc.root = id;
        } else {
            self.doc.nodes[self.last_top as usize].next = id;
            self.doc.nodes[id as usize].prev = self.last_top;
        }
        self.last_top = id;
    }

    /// htmlnamePush + `sax->startElement` (xmlSAX2StartHtmlElement). Neither checks
    /// `disableSAX`, so a start tag whose attribute parsing halted the parser is still added.
    fn push(&mut self, name: Name, attrs: Vec<Attr>) {
        if self.html < INSERT_IN_HEAD && name == names::HEAD {
            self.html = INSERT_IN_HEAD;
        }
        if self.html < INSERT_IN_BODY && name == names::BODY {
            self.html = INSERT_IN_BODY;
        }
        if self.stack.len() >= MAX_DEPTH {
            // nodePush fails: the node is unlinked and freed.
            self.halt("Excessive depth in document: 256, use XML_PARSE_HUGE option");
            return;
        }
        let id = self.doc.push(Kind::Element, name);
        self.doc.nodes[id as usize].attrs = attrs;
        match self.stack.last() {
            Some(&parent) => self.doc.link_last(parent, id),
            None => self.link_top(id),
        }
        self.stack.push(id);
    }

    /// `sax->endElement` + htmlnamePop.
    #[inline]
    fn pop(&mut self) {
        self.stack.pop();
    }

    /// `sax->characters` (xmlSAX2Characters -> xmlSAX2Text). Callers check `disableSAX`.
    fn characters(&mut self, s: &str) {
        if self.stopped || s.is_empty() {
            return;
        }
        let Some(&node) = self.stack.last() else {
            return; // ctxt->node == NULL: text at document level is dropped
        };
        let last = self.doc.nodes[node as usize].last;
        let slot = if last == NIL {
            &mut self.doc.nodes[node as usize].text
        } else {
            &mut self.doc.nodes[last as usize].tail
        };
        match slot {
            Some(t) => {
                if s.len() > XML_MAX_TEXT_LENGTH || t.len() > XML_MAX_TEXT_LENGTH - s.len() {
                    self.halt("Text node too long, try XML_PARSE_HUGE");
                    return;
                }
                t.push_str(s);
            }
            None => *slot = Some(s.to_owned()),
        }
    }

    // ---- implied elements and auto-close ------------------------------------------------

    /// htmlAutoClose
    fn auto_close(&mut self, newtag: Name) {
        while let Some(n) = self.name() {
            if !check_auto_close(newtag, n) {
                break;
            }
            self.pop();
        }
    }

    /// htmlAutoCloseOnClose
    fn auto_close_on_close(&mut self, newtag: Name) {
        let priority = end_priority(newtag);
        let mut found = false;
        for &id in self.stack.iter().rev() {
            let t = self.tag_of(id);
            if t == newtag {
                found = true;
                break;
            }
            if end_priority(t) > priority {
                return;
            }
        }
        if !found {
            return;
        }
        while let Some(n) = self.name() {
            if n == newtag {
                break;
            }
            self.pop();
        }
    }

    /// htmlCheckImplied
    fn check_implied(&mut self, newtag: Name) {
        if newtag == names::HTML {
            return;
        }
        if self.stack.is_empty() {
            self.push(names::HTML, Vec::new());
        }
        if newtag == names::BODY || newtag == names::HEAD {
            return;
        }
        let head_tag = matches!(
            newtag,
            names::SCRIPT | names::STYLE | names::META | names::LINK | names::TITLE | names::BASE
        );
        if self.stack.len() <= 1 && head_tag {
            if self.html >= INSERT_IN_HEAD {
                return;
            }
            self.push(names::HEAD, Vec::new());
        } else if !matches!(newtag, names::NOFRAMES | names::FRAME | names::FRAMESET) {
            if self.html >= INSERT_IN_BODY {
                return;
            }
            if self
                .stack
                .iter()
                .any(|&id| matches!(self.tag_of(id), names::BODY | names::HEAD))
            {
                return;
            }
            self.push(names::BODY, Vec::new());
        }
    }

    /// htmlStartCharData
    fn start_char_data(&mut self) {
        if self.name() == Some(names::HEAD) {
            self.auto_close(names::P);
        }
        self.check_implied(names::P);
    }

    /// htmlCharDataSAXCallback (mode is irrelevant: cdataBlock is NULL, keepBlanks is 1).
    fn char_data_cb(&mut self, mut s: &str) {
        if self.stopped {
            return;
        }
        if matches!(self.name(), None | Some(names::HTML | names::HEAD)) {
            let i = s.bytes().take_while(|&c| is_ws(c)).count();
            if i > 0 {
                self.characters(&s[..i]);
                s = &s[i..];
            }
            if s.is_empty() {
                return;
            }
            self.start_char_data();
            if self.stopped {
                return;
            }
        }
        self.characters(s);
    }

    // ---- tokenizer ------------------------------------------------------------------------

    /// htmlSkipBlankChars
    fn skip_blanks(&mut self) -> usize {
        let start = self.cur;
        let inp = self.inp;
        let mut cur = self.cur;
        if !self.stopped {
            loop {
                while cur < self.end && is_ws(inp[cur]) {
                    cur += 1;
                }
                if cur < self.end || !self.emulate {
                    break;
                }
                // avail == 0: GROW, and stop if nothing came in.
                self.grow_macro(cur);
                if self.stopped || cur >= self.end {
                    break;
                }
            }
        }
        self.cur = cur;
        if cur - start > 8 {
            self.grow_macro(cur);
        }
        cur - start
    }

    /// htmlParseHTMLName: lowercased ASCII, at most 100 bytes kept, NUL and invalid UTF-8
    /// become U+FFFD.
    fn parse_html_name_into(&mut self, attr: bool, buf: &mut NameBuf) {
        let inp = self.inp;
        let stop = if attr { b'=' } else { b' ' };
        let mut i = self.cur;
        let mut eof = !self.emulate;
        loop {
            if !eof && self.end - i < 32 {
                let avail = self.end - i;
                self.shrink_macro(i);
                self.grow(i);
                if self.end - i == avail {
                    eof = true;
                }
            }
            let end = self.end;
            if i >= end {
                break;
            }
            let c = inp[i];
            let mut size = 1;
            if buf.n != 0 && (c == b'/' || c == b'>' || c == stop || is_ws(c)) {
                break;
            }
            if c == 0 {
                buf.push(FFFD.as_bytes());
            } else if c < 0x80 {
                buf.push(&[c.to_ascii_lowercase()]);
            } else if let Some(n) = validate_utf8(&inp[i..end]) {
                size = n;
                buf.push(&inp[i..i + n]);
            } else {
                buf.push(FFFD.as_bytes());
            }
            i += size;
        }
        self.cur = i;
        self.shrink_macro(i);
    }

    fn parse_html_name(&mut self, attr: bool) -> Name {
        let mut buf = NameBuf::new();
        self.parse_html_name_into(attr, &mut buf);
        self.doc.intern(buf.as_str())
    }

    /// Start of a numeric character reference at `inp[i] == '&'`: (base, bytes consumed).
    #[inline]
    fn ncr_start(&self, i: usize) -> Option<(u32, usize)> {
        let inp = self.inp;
        let avail = self.end - i;
        let mut j = 1;
        if j < avail && inp[i + j] == b'#' {
            j += 1;
            if j < avail {
                if inp[i + j] | 0x20 == b'x' {
                    j += 1;
                    if j < avail && is_hex(inp[i + j]) {
                        return Some((16, 3));
                    }
                } else if inp[i + j].is_ascii_digit() {
                    return Some((10, 2));
                }
            }
        }
        None
    }

    /// htmlParseData: attribute values, comments, DOCTYPE parts. None when the value exceeds
    /// `max_len` ("value too long"); the input is then left at the start of the failing chunk.
    /// With `keep` false only the length is tracked (comments and DOCTYPE parts are dropped).
    #[allow(clippy::too_many_lines)] // one C function, kept in one piece to compare with it
    fn parse_data(
        &mut self,
        mask: Mask,
        comment: bool,
        refs: bool,
        max_len: usize,
        keep: bool,
    ) -> Option<String> {
        let inp = self.inp;
        let mut out = String::new();
        let mut used = 0usize;
        let mut term_skip: isize = -1;
        let mut eof = !self.emulate;
        // Under the buffer emulation bytes >= 0x80 stop the accelerator, so that it never
        // stops inside a character at `end - 64`.
        let stop = &data_stop(mask, refs)[usize::from(self.valid_utf8 && !self.emulate)];
        while !self.stopped {
            let chunk = self.cur;
            let mut i = chunk;
            let mut repl = Repl::None;
            let mut skip = 0usize;
            let mut ncr = 0u32;
            let mut ncr_size = 0usize;
            let mut cp = 0u32;
            // libxml2 visits every character and grows at most once per visit.
            let mut grown_at = usize::MAX;
            loop {
                if self.stopped {
                    break;
                }
                if !eof && self.end - i <= 64 && grown_at != i {
                    let avail = self.end - i;
                    grown_at = i;
                    self.grow(i);
                    if self.end - i == avail {
                        eof = true;
                    }
                }
                let end = self.end;
                if i >= end {
                    term_skip = 0;
                    break;
                }
                // Accelerator: skip the bytes the loop below would only step over, and stop at
                // `end - 64` while the buffer can still grow, where libxml2 grows. Not after
                // a grow halted the parser: libxml2 then takes one more character and stops.
                if ncr == 0 && !self.stopped {
                    let limit = if eof { end } else { end.saturating_sub(64) };
                    while i < limit && !stop[inp[i] as usize] {
                        i += 1;
                    }
                    if i >= end || (i == limit && !eof && grown_at != i) {
                        continue;
                    }
                }
                let c = inp[i];
                let mut size = 1;
                if mask_match(mask, c) {
                    if comment {
                        let avail = end - i;
                        if avail < 2 {
                            term_skip = 1;
                        } else if inp[i + 1] == b'-' {
                            if avail < 3 {
                                term_skip = 2;
                            } else if inp[i + 2] == b'>' {
                                term_skip = 3;
                            } else if inp[i + 2] == b'!' {
                                if avail < 4 {
                                    term_skip = 3;
                                } else if inp[i + 3] == b'>' {
                                    term_skip = 4;
                                }
                            }
                        }
                        if term_skip >= 0 {
                            break;
                        }
                    } else {
                        term_skip = 0;
                        break;
                    }
                }
                if ncr != 0 {
                    let lc = c | 0x20;
                    let digit = if c.is_ascii_digit() {
                        u32::from(c - b'0')
                    } else if ncr == 16 && (b'a'..=b'f').contains(&lc) {
                        u32::from(lc - b'a') + 10
                    } else {
                        if c == b';' {
                            i += 1;
                            ncr_size += 1;
                        }
                        break;
                    };
                    cp = cp * ncr + digit;
                    if cp >= 0x11_0000 {
                        cp = 0x11_0000;
                    }
                    ncr_size += 1;
                    i += size;
                    continue;
                }
                match c {
                    b'&' if refs => {
                        if i + 1 < end && inp[i + 1] == b'#' {
                            if let Some((base, n)) = self.ncr_start(i) {
                                ncr = base;
                                size = n;
                                ncr_size = n;
                                cp = 0;
                            }
                        } else if let Some((n, r)) = find_entity_prefix(&inp[i + 1..end], true) {
                            skip = n + 1;
                            repl = Repl::Static(r);
                            break;
                        }
                    }
                    0 => {
                        skip = 1;
                        repl = Repl::Static(FFFD);
                        break;
                    }
                    b'\r' => {
                        skip = 1;
                        if self.b(i + 1) != 0x0A {
                            repl = Repl::Static("\n");
                        }
                        break;
                    }
                    c if c >= 0x80 => {
                        if let Some(n) = validate_utf8(&inp[i..end]) {
                            size = n;
                        } else {
                            skip = 1;
                            repl = Repl::Static(FFFD);
                            break;
                        }
                    }
                    _ => {}
                }
                i += size;
            }
            // next_chunk:
            if ncr_size > 0 {
                skip = ncr_size;
                i -= ncr_size;
                repl = Repl::Char(code_point_char(cp));
            }
            let extra = (i - chunk) + repl.len();
            if extra > max_len - used {
                return None;
            }
            if keep {
                out.push_str(self.text(chunk, i));
                out.push_str(repl.as_str(&mut [0; 4]));
            }
            used += extra;
            self.cur = i + skip;
            self.shrink_macro(self.cur);
            if term_skip >= 0 {
                break;
            }
        }
        if term_skip > 0 {
            self.cur += term_skip.unsigned_abs();
        }
        Some(out)
    }

    /// htmlParseCharData with partial = 0: text, RCDATA, RAWTEXT, script data, PLAINTEXT.
    /// The SAX callback runs per chunk, and chunk boundaries matter (htmlCharDataSAXCallback
    /// treats leading whitespace of each chunk specially in html/head).
    #[allow(clippy::too_many_lines)] // one C function, kept in one piece to compare with it
    fn parse_char_data(&mut self) {
        let inp = self.inp;
        let mut mode = self.end_check_state;
        let mut complete = false;
        let mut eof = !self.emulate;
        while !self.stopped {
            let chunk = self.cur;
            let mut i = chunk;
            let mut repl = Repl::None;
            let mut skip = 0usize;
            let mut done = false;
            let mut ncr = 0u32;
            let mut ncr_size = 0usize;
            let mut cp = 0u32;
            loop {
                if self.stopped {
                    break;
                }
                if self.end - i <= 64 {
                    if !eof {
                        let avail = self.end - i;
                        self.grow(i);
                        if self.end - i == avail {
                            eof = true;
                        }
                    }
                    if i >= self.end {
                        done = true;
                        break;
                    }
                }
                let end = self.end;
                if ncr == 0 {
                    let stop = if self.emulate {
                        &C_STOP
                    } else {
                        &CHARDATA_STOP[usize::from(self.valid_utf8)][mode as usize]
                    };
                    while i < end && !stop[inp[i] as usize] {
                        i += 1;
                    }
                    if (!eof && end - i <= 64) || i >= end {
                        continue;
                    }
                }
                let c = inp[i];
                let mut size = 1;
                if ncr != 0 {
                    let lc = c | 0x20;
                    let digit = if c.is_ascii_digit() {
                        u32::from(c - b'0')
                    } else if ncr == 16 && (b'a'..=b'f').contains(&lc) {
                        u32::from(lc - b'a') + 10
                    } else {
                        if c == b';' {
                            i += 1;
                            ncr_size += 1;
                        }
                        break;
                    };
                    cp = cp * ncr + digit;
                    if cp >= 0x11_0000 {
                        cp = 0x11_0000;
                    }
                    ncr_size += 1;
                    i += size;
                    continue;
                }
                match c {
                    b'<' => {
                        if mode == 0 {
                            done = true;
                            complete = true;
                            break;
                        }
                        if mode != DATA_PLAINTEXT {
                            let avail = end - i;
                            let mut j = 1;
                            if j < avail {
                                if mode == DATA_SCRIPT && inp[i + j] == b'!' {
                                    j += 1;
                                    if j < avail && inp[i + j] == b'-' {
                                        j += 1;
                                        if j < avail && inp[i + j] == b'-' {
                                            mode = DATA_SCRIPT_ESC1;
                                        }
                                    }
                                } else {
                                    let mut solidus = false;
                                    if inp[i + j] == b'/' {
                                        j += 1;
                                        solidus = true;
                                    }
                                    if solidus || mode == DATA_SCRIPT_ESC1 {
                                        let name = self.name().map_or("", |t| self.doc.name(t));
                                        let name = name.as_bytes();
                                        let mut k = 0;
                                        while j < avail
                                            && k < name.len()
                                            && name[k] == (inp[i + j] | 0x20)
                                        {
                                            k += 1;
                                            j += 1;
                                        }
                                        if k == name.len() && j < avail {
                                            let c2 = inp[i + j];
                                            if c2 == b'>' || c2 == b'/' || is_ws(c2) {
                                                if mode == DATA_SCRIPT_ESC1 && !solidus {
                                                    mode = DATA_SCRIPT_ESC2;
                                                } else if mode == DATA_SCRIPT_ESC2 {
                                                    mode = DATA_SCRIPT_ESC1;
                                                } else {
                                                    complete = true;
                                                    done = true;
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    b'-' => {
                        if mode == DATA_SCRIPT_ESC1 || mode == DATA_SCRIPT_ESC2 {
                            let avail = end - i;
                            if 1 < avail && inp[i + 1] == b'-' && 2 < avail && inp[i + 2] == b'>' {
                                mode = DATA_SCRIPT;
                            }
                        }
                    }
                    b'&' => {
                        if mode == 0 || mode == DATA_RCDATA {
                            if i + 1 < end && inp[i + 1] == b'#' {
                                if let Some((base, n)) = self.ncr_start(i) {
                                    ncr = base;
                                    size = n;
                                    ncr_size = n;
                                    cp = 0;
                                }
                            } else if let Some((n, r)) = find_entity_prefix(&inp[i + 1..end], false)
                            {
                                skip = n + 1;
                                repl = Repl::Static(r);
                                break;
                            }
                        }
                    }
                    0 => {
                        skip = 1;
                        repl = Repl::Static(FFFD);
                        break;
                    }
                    b'\r' => {
                        skip = 1;
                        if self.b(i + 1) != 0x0A {
                            repl = Repl::Static("\n");
                        }
                        break;
                    }
                    c if c >= 0x80 => {
                        if let Some(n) = validate_utf8(&inp[i..end]) {
                            size = n;
                        } else {
                            skip = 1;
                            repl = Repl::Static(FFFD);
                            break;
                        }
                    }
                    _ => {}
                }
                i += size;
            }
            // next_chunk:
            if ncr_size > 0 {
                skip = ncr_size;
                i -= ncr_size;
                repl = Repl::Char(code_point_char(cp));
            }
            if i > chunk {
                self.cur = i;
                let s = self.text(chunk, i);
                self.char_data_cb(s);
            }
            self.cur = i + skip;
            if repl.len() > 0 {
                let mut buf = [0; 4];
                self.char_data_cb(repl.as_str(&mut buf));
            }
            self.shrink_macro(self.cur);
            if done {
                break;
            }
        }
        self.end_check_state = if complete { 0 } else { mode };
    }

    /// htmlParseComment (`sax->comment` is NULL: parse and drop).
    fn parse_comment(&mut self, bogus: bool) {
        if bogus {
            let _ = self.parse_data(MASK_GT, false, false, XML_MAX_TEXT_LENGTH, false);
            if self.cur_b() == b'>' {
                self.cur += 1;
            }
        } else if self.cur_b() == b'>' {
            self.cur += 1;
        } else if self.cur_b() == b'-' && self.nxt(1) == b'>' {
            self.cur += 2;
        } else {
            let _ = self.parse_data(MASK_DASH, true, false, XML_MAX_TEXT_LENGTH, false);
        }
    }

    /// htmlParseDoctypeLiteral: whether a literal was parsed (the value is dropped).
    fn parse_doctype_literal(&mut self) -> bool {
        let q = self.cur_b();
        let mask = match q {
            b'"' => MASK_DQ_GT,
            b'\'' => MASK_SQ_GT,
            _ => return false,
        };
        self.cur += 1;
        let r = self.parse_data(mask, false, false, XML_MAX_NAME_LENGTH, false);
        if self.cur_b() == q {
            self.cur += 1;
        }
        r.is_some()
    }

    /// htmlSkipBogusDoctype. Its SHRINK sees `input->cur` as of the last grow.
    fn skip_bogus_doctype(&mut self) {
        let inp = self.inp;
        if !self.emulate {
            let rest = &inp[self.cur.min(self.end)..self.end];
            self.cur = match memchr::memchr(b'>', rest) {
                Some(k) => self.cur + k + 1,
                None => self.end,
            };
            return;
        }
        let mut i = self.cur;
        let mut c_cur = self.cur;
        let mut eof = false;
        while !self.stopped {
            if !eof && self.end - i <= 64 {
                let avail = self.end - i;
                c_cur = i;
                self.grow(i);
                if self.end - i == avail {
                    eof = true;
                }
            }
            if i >= self.end {
                break;
            }
            let c = inp[i];
            i += 1;
            if c == b'>' {
                break;
            }
            self.shrink_macro(c_cur);
        }
        self.cur = i;
    }

    /// htmlParseDocTypeDecl. The DTD node it creates is a sibling of the root and invisible to
    /// lxml's root element, so only the input position matters.
    fn parse_doctype(&mut self) {
        self.cur += 9;
        self.skip_blanks();
        if self.cur < self.end && self.cur_b() != b'>' {
            let _ = self.parse_data(MASK_WS_GT, false, false, XML_MAX_NAME_LENGTH, false);
            self.skip_blanks();
        }
        if self.upper_is(0, b"PUBLIC") {
            self.cur += 6;
            self.skip_blanks();
            if self.parse_doctype_literal() {
                self.skip_blanks();
                self.parse_doctype_literal();
            }
        } else if self.upper_is(0, b"SYSTEM") {
            self.cur += 6;
            self.skip_blanks();
            self.parse_doctype_literal();
        }
        self.skip_bogus_doctype();
    }

    /// htmlParseAttValue
    fn parse_att_value(&mut self) -> Option<String> {
        let q = self.cur_b();
        if q == b'"' || q == b'\'' {
            self.cur += 1;
            let mask = if q == b'"' { MASK_DQ } else { MASK_SQ };
            let r = self.parse_data(mask, false, true, XML_MAX_TEXT_LENGTH, true);
            if self.cur_b() == q {
                self.cur += 1;
            }
            r
        } else {
            self.parse_data(MASK_WS_GT, false, true, XML_MAX_TEXT_LENGTH, true)
        }
    }

    /// htmlParseAttribute
    fn parse_attribute(&mut self) -> (Name, Option<String>) {
        let name = self.parse_html_name(true);
        self.skip_blanks();
        let mut val = None;
        if self.cur_b() == b'=' {
            self.cur += 1;
            self.skip_blanks();
            val = self.parse_att_value();
        }
        (name, val)
    }

    /// htmlParseStartTag
    fn parse_start_tag(&mut self) {
        self.end_check_state = 0;
        self.cur += 1;
        self.grow_macro(self.cur);
        let name = self.parse_html_name(false);
        self.auto_close(name);
        self.check_implied(name);
        let mut discard = false;
        if !self.stack.is_empty() && name == names::HTML {
            discard = true;
            self.depth += 1;
        }
        if self.stack.len() != 1 && name == names::HEAD {
            discard = true;
            self.depth += 1;
        }
        if name == names::BODY {
            for k in 0..self.stack.len() {
                if self.tag_of(self.stack[k]) == names::BODY {
                    discard = true;
                    self.depth += 1;
                }
            }
        }
        self.skip_blanks();
        // Attributes in order; a repeated name is dropped (the first one wins), and a missing
        // value is the name for HTML boolean attributes (xmlSAX2HtmlAttribute), else empty.
        let mut attrs: Vec<Attr> = Vec::new();
        let mut seen: Option<rustc_hash::FxHashSet<Name>> = None;
        // Attributes parsed, duplicates included (`nbatts / 2`).
        let mut parsed: u64 = 0;
        while self.cur < self.end
            && self.cur_b() != b'>'
            && !(self.cur_b() == b'/' && self.nxt(1) == b'>')
            && !self.stopped
        {
            if self.cur_b() == b'/' {
                self.cur += 1;
                self.skip_blanks();
                continue;
            }
            self.grow_macro(self.cur);
            let (n, v) = self.parse_attribute();
            if 2 * parsed + 4 > MAXATTS_LIMIT {
                // xmlGrowCapacity fails: htmlErrMemory, and libxml2 returns no document.
                self.oom = true;
                self.halt("Memory allocation failed");
            } else {
                parsed += 1;
                // Linear search for the first 16 names, then a set.
                let dup = if let Some(set) = &mut seen {
                    !set.insert(n)
                } else {
                    let dup = attrs.iter().any(|a| a.name == n);
                    if attrs.len() >= 16 {
                        let mut set: rustc_hash::FxHashSet<Name> =
                            attrs.iter().map(|a| a.name).collect();
                        set.insert(n);
                        seen = Some(set);
                    }
                    dup
                };
                if !dup {
                    let value = match v {
                        Some(v) => v,
                        None if is_boolean_attr(n) => self.doc.name(n).to_owned(),
                        None => String::new(),
                    };
                    attrs.push(Attr { name: n, value });
                }
            }
            self.skip_blanks();
        }
        if self.cur >= self.end || discard {
            return;
        }
        self.push(name, attrs);
    }

    /// htmlParseEndTag
    fn parse_end_tag(&mut self) {
        self.end_check_state = 0;
        self.cur += 2;
        if self.cur >= self.end {
            self.start_char_data();
            if !self.stopped {
                self.characters("</");
            }
            return;
        }
        if self.cur_b() == b'>' {
            self.cur += 1;
            return;
        }
        if !is_letter(self.cur_b()) {
            self.parse_comment(true);
            return;
        }
        let mut buf = NameBuf::new();
        self.parse_html_name_into(false, &mut buf);
        // A name the document has never seen cannot be on the stack.
        let name = self.doc.lookup(buf.as_str());
        self.skip_blanks();
        while self.cur < self.end
            && self.cur_b() != b'>'
            && !(self.cur_b() == b'/' && self.nxt(1) == b'>')
            && !self.stopped
        {
            if self.cur_b() == b'/' {
                self.cur += 1;
                self.skip_blanks();
                continue;
            }
            self.grow_macro(self.cur);
            let _ = self.parse_attribute();
            self.skip_blanks();
        }
        if self.cur_b() == b'>' {
            self.cur += 1;
        } else if self.cur_b() == b'/' && self.nxt(1) == b'>' {
            self.cur += 2;
        } else {
            return;
        }
        let Some(name) = name else {
            return; // "Unexpected end tag"
        };
        if self.depth > 0 && matches!(name, names::HTML | names::BODY | names::HEAD) {
            self.depth -= 1;
            return;
        }
        if !self.stack.iter().any(|&id| self.tag_of(id) == name) {
            return; // "Unexpected end tag"
        }
        self.auto_close_on_close(name);
        if self.name() == Some(name) {
            self.pop();
        }
    }

    /// htmlParseElementInternal
    fn parse_element_internal(&mut self) {
        self.parse_start_tag();
        let Some(name) = self.name() else {
            return;
        };
        let info = tag_lookup(name);
        if self.cur_b() == b'/' && self.nxt(1) == b'>' {
            self.cur += 2;
            self.pop();
            return;
        }
        if self.cur_b() != b'>' {
            return;
        }
        self.cur += 1;
        if let Some((empty, data_mode)) = info {
            if empty {
                self.pop();
                return;
            }
            self.end_check_state = data_mode;
        }
    }

    /// htmlParseContent
    fn parse_content(&mut self) {
        self.grow_macro(self.cur);
        while !self.stopped && self.cur < self.end {
            let mode = self.end_check_state;
            if mode == 0 && self.cur_b() == b'<' {
                let n1 = self.nxt(1);
                if n1 == b'/' {
                    self.parse_end_tag();
                } else if n1 == b'!' {
                    if self.upper_is(2, b"DOCTYPE") {
                        self.parse_doctype();
                    } else if self.nxt(2) == b'-' && self.nxt(3) == b'-' {
                        self.cur += 4;
                        self.parse_comment(false);
                    } else {
                        self.cur += 2;
                        self.parse_comment(true);
                    }
                } else if n1 == b'?' {
                    self.cur += 1;
                    self.parse_comment(true);
                } else if is_letter(n1) {
                    self.parse_element_internal();
                } else {
                    self.start_char_data();
                    if !self.stopped {
                        self.characters("<");
                    }
                    self.cur += 1;
                }
            } else {
                self.parse_char_data();
            }
            self.shrink_macro(self.cur);
            self.grow_macro(self.cur);
        }
        // htmlAutoCloseOnEnd only emits endElement events: no effect on the tree.
    }

    /// Comments and bogus comments (`<?...>`) before and after the DOCTYPE.
    fn parse_misc(&mut self) {
        while self.cur_b() == b'<' {
            if self.nxt(1) == b'!' && self.nxt(2) == b'-' && self.nxt(3) == b'-' {
                self.cur += 4;
                self.parse_comment(false);
            } else if self.nxt(1) == b'?' {
                self.cur += 1;
                self.parse_comment(true);
            } else {
                break;
            }
            self.skip_blanks();
        }
    }

    /// htmlParseDocument
    fn parse_document(&mut self) {
        // xmlDetectEncoding with XML_INPUT_HAS_ENCODING: grow once, then only skip a UTF-8 BOM,
        // and only when at least four bytes are available.
        let grown = if self.emulate {
            self.grow(0)
        } else {
            self.feed.error_read != Some(1)
        };
        if grown && self.end >= 4 && self.inp.starts_with(b"\xEF\xBB\xBF") {
            self.cur = 3;
        }
        self.skip_blanks();
        if self.stopped {
            return; // no startDocument, so no document
        }
        self.started = true;
        self.parse_misc();
        if self.cur_b() == b'<' && self.nxt(1) == b'!' && self.upper_is(2, b"DOCTYPE") {
            self.parse_doctype();
        }
        self.skip_blanks();
        self.parse_misc();
        self.parse_content();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(s: &str, attr: bool) -> Option<(usize, String)> {
        find_entity_prefix(s.as_bytes(), attr).map(|(n, r)| (n, r.to_owned()))
    }

    #[test]
    fn entities() {
        assert_eq!(ent("amp;x", false), Some((4, "&".into())));
        assert_eq!(ent("ampx", false), Some((3, "&".into())));
        assert_eq!(ent("ampx", true), None);
        assert_eq!(ent("amp=", true), None);
        assert_eq!(ent("amp ", true), Some((3, "&".into())));
        assert_eq!(ent("notin;", false), Some((6, "\u{2209}".into())));
        assert_eq!(ent("notit;", false), Some((3, "\u{ac}".into())));
        assert_eq!(ent("nbsp", false), Some((4, "\u{a0}".into())));
        assert_eq!(
            ent("CounterClockwiseContourIntegral;", false),
            Some((32, "\u{2233}".into()))
        );
        assert_eq!(ent("foo;", false), None);
        assert_eq!(ent("a", false), None);
    }

    #[test]
    fn entity_replacements_are_utf8() {
        // Every entry the lookup can land on: a length byte with flags, the name, two
        // sub-table bytes if flagged, then the replacement (as `found` computes it).
        for &bo in &ENT_VALUES {
            let bo = bo as usize;
            let len = (ENT_STRINGS[bo] & 0x3F) as usize;
            let mut r = bo + len + 1;
            if ENT_STRINGS[bo] & 0x40 != 0 {
                r += 2;
            }
            let rlen = ENT_STRINGS[r] as usize;
            assert!((1..=8).contains(&rlen), "entry at {bo}");
            assert!(std::str::from_utf8(&ENT_STRINGS[r + 1..r + 1 + rlen]).is_ok());
        }
    }

    #[test]
    fn utf8_validation() {
        assert_eq!(validate_utf8("é".as_bytes()), Some(2));
        assert_eq!(validate_utf8("\u{20ac}".as_bytes()), Some(3));
        assert_eq!(validate_utf8("\u{1f600}".as_bytes()), Some(4));
        assert_eq!(validate_utf8(b"\xED\xA0\x80"), None);
        assert_eq!(validate_utf8(b"\xE0\x80\x80"), None);
        assert_eq!(validate_utf8(b"\xF4\x90\x80\x80"), None);
        assert_eq!(validate_utf8(b"\xC3"), None);
    }

    #[test]
    fn numeric_references() {
        assert_eq!(code_point_char(0x41), 'A');
        assert_eq!(code_point_char(0x80), '\u{20AC}');
        assert_eq!(code_point_char(0), '\u{FFFD}');
        assert_eq!(code_point_char(0xD800), '\u{FFFD}');
        assert_eq!(code_point_char(0x11_0000), '\u{FFFD}');
        assert_eq!(code_point_char(0xFFFE), '\u{FFFE}');
    }

    #[test]
    fn tables_use_static_names() {
        for &(name, _, _) in &ELEMENTS {
            let t = static_name(name).unwrap_or_else(|| panic!("element {name} is not static"));
            assert!(t.0 < 256, "{name}");
        }
        for &(old, new) in &START_CLOSE {
            for n in [old, new] {
                let t = static_name(n).unwrap_or_else(|| panic!("{n} is not static"));
                assert!(t.0 < 256, "{n}");
            }
        }
        for &(name, _) in &END_PRIORITY {
            assert!(static_name(name).is_some(), "{name}");
        }
        for name in BOOLEAN_ATTRS {
            assert!(static_name(name).is_some(), "{name}");
        }
        // libxml2 looks both tables up with bsearch: they must be sorted for it.
        for w in ELEMENTS.windows(2) {
            assert!(w[0].0 < w[1].0, "{w:?}");
        }
        for w in START_CLOSE.windows(2) {
            assert!(w[0] < w[1], "{w:?}");
        }
    }

    #[test]
    fn lookups() {
        assert_eq!(tag_lookup(names::BR), Some((true, 0)));
        assert_eq!(tag_lookup(names::SCRIPT), Some((false, DATA_SCRIPT)));
        assert_eq!(tag_lookup(names::SOURCE), None);
        assert!(check_auto_close(names::P, names::P));
        assert!(check_auto_close(names::DIV, names::HEAD));
        assert!(!check_auto_close(names::SPAN, names::P));
        assert_eq!(end_priority(names::TABLE), 190);
        assert_eq!(end_priority(names::SPAN), 100);
        assert!(is_boolean_attr(names::CHECKED));
        assert!(!is_boolean_attr(names::CLASS));
    }

    // 0 -> 11 slots (maxatts 22), 33 (66), 99 (198), ...: the first maxatts at or past
    // HTML_MAX_ATTRS is 105,225,318, so the 52,612,658th attribute of a tag fails.
    const _: () = assert!(MAXATTS_LIMIT == 105_225_318);

    fn dump(p: &Parsed) -> String {
        let mut s = String::new();
        if p.doc.root != NIL {
            p.doc.dump(p.doc.root, true, &mut s);
        }
        s
    }

    #[test]
    fn small_documents() {
        let p = parse(b"<p>a&amp;b<br>c", &Feed::default());
        assert!(p.document && !p.halted);
        assert_eq!(
            dump(&p),
            "[\"S\", \"html\"]\n[\"S\", \"body\"]\n[\"S\", \"p\"]\n[\"T\", \"a&b\"]\n\
             [\"S\", \"br\"]\n[\"E\"]\n[\"L\", \"c\"]\n[\"E\"]\n[\"E\"]\n[\"E\"]\n"
        );
        let p = parse(b"   ", &Feed::default());
        assert!(p.document);
        assert_eq!(p.doc.root, NIL);
        let p = parse(
            b"<div x=1 X=2 checked>t</div></html><p>after",
            &Feed::default(),
        );
        let root = p.doc.root;
        let body = p.doc.first_child(root).unwrap();
        let div = p.doc.first_child(body).unwrap();
        assert_eq!(
            p.doc.attrs(div).collect::<Vec<_>>(),
            vec![("x", "1"), ("checked", "checked")]
        );
        // Content after </html> starts a second top-level html element, the root's sibling.
        let second = p.doc.next(root).unwrap();
        assert_eq!(p.doc.tag(second), names::HTML);
        assert_eq!(p.doc.parent(second), None);
    }

    #[test]
    fn depth_limit_halts() {
        let s = "<div>".repeat(300);
        let p = parse(s.as_bytes(), &Feed::default());
        assert!(p.halted && p.document);
        let mut depth = 0;
        let mut c = p.doc.root;
        while let Some(f) = p.doc.first_child(c) {
            c = f;
            depth += 1;
        }
        assert_eq!(depth, 255);
    }
}
