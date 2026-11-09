// Derived from lxml 6.1.3 (BSD-3-Clause), © Infrae; modified.
//! lxml 6.1.3's side of `lxml.html.fromstring(s, parser=HTMLParser(collect_ids=False,
//! default_doctype=False, encoding="utf-8", remove_comments=True, remove_pis=True))`: the
//! str and bytes input handling of `etree.fromstring` (parser.pxi `_parseUnicodeDoc` and
//! `_parseDoc`), and `lxml.html.fromstring`'s fragment handling (lxml/html/__init__.py).
//!
//! Input paths. A Python str goes to libxml2 as its PEP 393 buffer, with an encoding name
//! chosen by the string's kind: "UTF-8" (ASCII), "ISO-8859-1" (1 byte per character),
//! "UTF-16LE" (2 bytes) or "UTF-32LE" (4 bytes). Bytes go with the parser's encoding,
//! "utf-8". libxml2 converts to UTF-8 before tokenizing, so both paths give the parser the
//! UTF-8 encoding of the string, except for lone surrogates: on the str path the decoder stops
//! with an encoding error there, which ends the input at that point (a high and a low
//! surrogate in a 2-byte-kind string decode as one astral character); on the bytes path
//! (trafilatura passes `s.encode("utf-8", "surrogatepass")`) the three bytes of each surrogate
//! are invalid UTF-8 and become three U+FFFD. An error within libxml2's first 4000-byte read
//! also keeps a leading BOM. For inputs over 10 MB the parser also needs to know how libxml2's
//! buffer fills from the raw buffer, which depends on the kind ([`super::parser::Feed`]).
//!
//! Fragment handling assigns `.text` and `.tail` when it merges several bodies or heads; lxml
//! rejects strings with C0 controls or U+FFFE/U+FFFF there (ValueError), which character
//! references and raw control characters can produce.

use std::fmt;

use super::parser::{self, Feed, MINLEN, XML_MAX_LOOKUP_LIMIT};
use super::tree::{Document, NIL, NodeId, names};

/// What lxml raises, so callers can follow `except` clauses exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LxmlError {
    /// `lxml.etree.ParserError` (an `lxml.etree.LxmlError`, not a ValueError): "Document is
    /// empty", raised by `lxml.html.document_fromstring` when the parse has no root element.
    ParserError(String),
    /// `ValueError`: a str with an XML encoding declaration, or a `.text`/`.tail` assignment
    /// in the fragment handling that lxml rejects.
    ValueError(String),
    /// `lxml.etree.XMLSyntaxError`: libxml2 returned no document (the parse halted before the
    /// document started, after more than 10,000,000 leading blanks, or ran out of attribute
    /// slots in one tag). The message is libxml2's, without lxml's line and column suffix.
    XmlSyntaxError(String),
    /// Not an lxml error: [`fromstring_wtf8`] got bytes that are not surrogate-tolerant UTF-8
    /// (the field is the offset of the first bad byte). A Python str never produces this.
    InvalidInput(usize),
    /// Not an lxml error: lxml's buffer for the input would pass `i32::MAX` bytes (a str's
    /// PEP 393 buffer, or the bytes). lxml then reads it through its file-like parser, which
    /// is not ported; hand such inputs to lxml.
    InputTooLarge,
}

impl LxmlError {
    /// Whether Python sees a `ValueError` (Trafilatura's `load_html` catches it apart).
    pub fn is_value_error(&self) -> bool {
        matches!(self, LxmlError::ValueError(_))
    }
}

impl fmt::Display for LxmlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LxmlError::ParserError(m) | LxmlError::ValueError(m) | LxmlError::XmlSyntaxError(m) => {
                f.write_str(m)
            }
            LxmlError::InvalidInput(at) => {
                write!(f, "invalid surrogate-tolerant UTF-8 at byte {at}")
            }
            LxmlError::InputTooLarge => f.write_str("input over 2 GiB in lxml's buffer"),
        }
    }
}

impl std::error::Error for LxmlError {}

const ENCODING_DECLARATION: &str = "Unicode strings with encoding declaration are not supported. \
     Please use bytes input or XML fragments without declaration.";
const NOT_XML_COMPATIBLE: &str = "All strings must be XML compatible: Unicode or ASCII, no NULL \
     bytes or control characters";

/// `lxml.html.fromstring(s, parser=...)` for a Python str `s` (see the module docs).
pub fn fromstring_str(s: &str) -> Result<Document, LxmlError> {
    fromstring_wtf8_checked(s.as_bytes(), false)
}

/// [`fromstring_str`] for a str that may hold lone surrogates, given as its surrogatepass
/// UTF-8 (`s.encode("utf-8", "surrogatepass")`; a high and a low surrogate in a row are two
/// code points, as in Python). Equal to [`fromstring_str`] on valid UTF-8.
pub fn fromstring_wtf8(s: &[u8]) -> Result<Document, LxmlError> {
    crate::ftfy::validate_wtf8(s).map_err(LxmlError::InvalidInput)?;
    fromstring_wtf8_checked(s, true)
}

/// `lxml.html.fromstring(b, parser=...)` for bytes `b`.
pub fn fromstring_bytes(b: &[u8]) -> Result<Document, LxmlError> {
    if b.len() > INT_MAX {
        return Err(LxmlError::InputTooLarge);
    }
    let full = looks_like_full_html(b, true);
    let p = parser::parse(b, &Feed::default());
    finish(p, full)
}

/// INT_MAX: the largest buffer lxml hands to `htmlCtxtReadMemory`.
const INT_MAX: usize = i32::MAX as usize;

fn fromstring_wtf8_checked(s: &[u8], may_have_surrogates: bool) -> Result<Document, LxmlError> {
    // The PEP 393 buffer holds at most 4 bytes per code point, and a code point takes at least
    // one byte here.
    if s.len() > INT_MAX / 4 {
        let (unit, _) = str_kind(s);
        if CodePoints(s).count().saturating_mul(unit) > INT_MAX {
            return Err(LxmlError::InputTooLarge);
        }
    }
    // lxml.html.fromstring tests the original string; etree.fromstring then rejects a str
    // with an encoding declaration before parsing.
    let full = looks_like_full_html(s, false);
    if s.starts_with(b"<?xml") && has_encoding_declaration(s) {
        return Err(LxmlError::ValueError(ENCODING_DECLARATION.to_owned()));
    }
    let surrogates = may_have_surrogates && s.windows(2).any(|w| w[0] == 0xED && w[1] >= 0xA0);
    let p = if surrogates {
        let (input, feed) = str_path_input(s);
        parser::parse(&input, &feed)
    } else if s.len() > XML_MAX_LOOKUP_LIMIT {
        parser::parse(s, &str_feed(s))
    } else {
        parser::parse(s, &Feed::default())
    };
    finish(p, full)
}

/// Iterator over the code points of surrogate-tolerant UTF-8 (surrogates included). Callers
/// check the input first; a cut-off sequence at the end is dropped.
#[derive(Clone)]
struct CodePoints<'a>(&'a [u8]);

impl Iterator for CodePoints<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        let s = self.0;
        let c = u32::from(*s.first()?);
        let n = if c < 0x80 {
            1
        } else if c < 0xE0 {
            2
        } else if c < 0xF0 {
            3
        } else {
            4
        };
        if n > s.len() {
            self.0 = &[];
            return None;
        }
        let cont = |k: usize| u32::from(s[k]) & 0x3F;
        let cp = match n {
            1 => c,
            2 => ((c & 0x1F) << 6) | cont(1),
            3 => ((c & 0x0F) << 12) | (cont(1) << 6) | cont(2),
            _ => ((c & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
        };
        self.0 = &s[n..];
        Some(cp)
    }
}

fn is_surrogate(c: u32) -> bool {
    (0xD800..0xE000).contains(&c)
}

/// The PEP 393 kind lxml passes: bytes per code point in the str's buffer, and whether
/// libxml2 needs a decoder for it (only an ASCII str is passed as "UTF-8").
fn str_kind(s: &[u8]) -> (usize, bool) {
    let max = CodePoints(s).max().unwrap_or(0);
    if max < 0x80 {
        (1, false)
    } else if max < 0x100 {
        (1, true)
    } else if max < 0x1_0000 {
        (2, true)
    } else {
        (4, true)
    }
}

/// The [`Feed`] of a str without surrogates (needed only past the lookup limit): the UTF-8
/// length after each 4000-byte read of its PEP 393 buffer.
fn str_feed(s: &[u8]) -> Feed {
    let (unit, decoded) = str_kind(s);
    if !decoded {
        return Feed::default();
    }
    let mut ends = Vec::with_capacity(s.len() / MINLEN + 2);
    let mut raw = 0usize;
    let mut out = 0usize;
    for c in CodePoints(s) {
        raw += unit;
        while raw > MINLEN * (ends.len() + 1) {
            ends.push(out);
        }
        out += utf8_len(c);
    }
    ends.push(out);
    Feed {
        ends,
        error_read: None,
    }
}

fn utf8_len(c: u32) -> usize {
    match c {
        0..0x80 => 1,
        0x80..0x800 => 2,
        0x800..0x1_0000 => 3,
        _ => 4,
    }
}

/// The UTF-8 libxml2 parses on the str path for a string with lone surrogates (see the module
/// docs), and how its buffer fills. The raw buffer holds 2 (UCS-2) or 4 (UCS-4) bytes per code
/// point; UTF16LEToUTF8 leaves a high surrogate in the last unit of a read for the next read.
fn str_path_input(s: &[u8]) -> (Vec<u8>, Feed) {
    let cps: Vec<u32> = CodePoints(s).collect();
    let kind4 = cps.iter().any(|&c| c > 0xFFFF);
    let unit = if kind4 { 4 } else { 2 };
    let mut out = String::with_capacity(s.len());
    let mut ends = Vec::new();
    let mut error_raw_end: Option<usize> = None; // raw offset past the unit(s) the decoder failed on
    let mut i = 0;
    // `ends` is only needed when the result is long enough for the lookup limit; recording it
    // costs one comparison per code point.
    let record = s.len() > XML_MAX_LOOKUP_LIMIT;
    let push = |ends: &mut Vec<usize>, raw_end: usize, out_len: usize| {
        if record {
            while raw_end > MINLEN * (ends.len() + 1) {
                ends.push(out_len);
            }
        }
    };
    while i < cps.len() {
        let c = cps[i];
        if is_surrogate(c) {
            if kind4 || c >= 0xDC00 {
                // iconv EILSEQ, or an unpaired low surrogate.
                error_raw_end = Some((i + 1) * unit);
                break;
            }
            match cps.get(i + 1) {
                Some(&d) if (0xDC00..0xE000).contains(&d) => {
                    let cp = 0x1_0000 + ((c - 0xD800) << 10) + (d - 0xDC00);
                    push(&mut ends, (i + 2) * unit, out.len());
                    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                    i += 2;
                    continue;
                }
                Some(_) => {
                    // A high surrogate not followed by a low one.
                    error_raw_end = Some((i + 2) * unit);
                    break;
                }
                None => break, // incomplete pair at the end: only an error when flushing at EOF
            }
        }
        push(&mut ends, (i + 1) * unit, out.len());
        out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
        i += 1;
    }
    let error_read = error_raw_end.map(|e| e.div_ceil(MINLEN));
    if record {
        if let Some(k) = error_read {
            // Reads before the failing one, then the failing read with what it converted.
            while ends.len() + 1 < k {
                ends.push(out.len());
            }
            ends.truncate(k - 1);
        }
        ends.push(out.len());
    }
    (out.into_bytes(), Feed { ends, error_read })
}

/// Python's `str.isspace` and `\s` in a str pattern (`Py_UNICODE_ISSPACE`).
fn py_isspace(c: u32) -> bool {
    matches!(
        c,
        0x09..=0x0D
            | 0x1C..=0x20
            | 0x85
            | 0xA0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

fn py_strip_is_empty(s: &str) -> bool {
    s.chars().all(|c| py_isspace(c as u32))
}

/// lxml apihelpers.pxi `__RE_XML_ENCODING.match`:
/// `^(<\?xml[^>]+)\s+encoding\s*=\s*["\'][^"\']*["\'](\s*\?>|)` (the last group always
/// matches).
fn has_encoding_declaration(s: &[u8]) -> bool {
    if !s.starts_with(b"<?xml") {
        return false;
    }
    let cps: Vec<u32> = CodePoints(s).collect();
    let n = cps.len();
    let is = |q: usize, c: char| cps.get(q) == Some(&(c as u32));
    let quote = |c: u32| c == u32::from(b'"') || c == u32::from(b'\'');
    // `[^>]+` covers cps[5..p] with p >= 6, so p stays before the first '>'. `\s+` starts at
    // p; every p in one run of blanks leads to the same place, so each run is tried once.
    let gt = cps
        .iter()
        .skip(5)
        .position(|&c| c == u32::from(b'>'))
        .map_or(n, |k| k + 5);
    let last_quote = cps.iter().rposition(|&c| quote(c));
    let mut p = 6;
    while p < gt {
        if !py_isspace(cps[p]) {
            p += 1;
            continue;
        }
        let mut q = p;
        while q < n && py_isspace(cps[q]) {
            q += 1;
        }
        p = q;
        if !"encoding".chars().enumerate().all(|(k, c)| is(q + k, c)) {
            continue;
        }
        q += 8;
        while q < n && py_isspace(cps[q]) {
            q += 1;
        }
        if !is(q, '=') {
            continue;
        }
        q += 1;
        while q < n && py_isspace(cps[q]) {
            q += 1;
        }
        // `["\'][^"\']*["\']`: a quote here and another one somewhere after it.
        if q < n && quote(cps[q]) && last_quote.is_some_and(|l| l > q) {
            return true;
        }
    }
    false
}

/// lxml.html `_looks_like_full_html_unicode` / `_bytes`: `^\s*<(?:html|!doctype)` with re.I.
fn looks_like_full_html(s: &[u8], bytes_pattern: bool) -> bool {
    let space = |c: u32| {
        if bytes_pattern {
            matches!(c, 0x09..=0x0D | 0x20)
        } else {
            py_isspace(c)
        }
    };
    let rest: Vec<u32> = if bytes_pattern {
        s.iter()
            .map(|&b| u32::from(b))
            .skip_while(|&c| space(c))
            .take(9)
            .collect()
    } else {
        CodePoints(s).skip_while(|&c| space(c)).take(9).collect()
    };
    let lower = |k: usize| {
        rest.get(k).map(|&c| {
            u8::try_from(c)
                .ok()
                .filter(u8::is_ascii)
                .map_or(0, |b| b.to_ascii_lowercase())
        })
    };
    let word = |w: &[u8]| w.iter().enumerate().all(|(k, &b)| lower(k) == Some(b));
    word(b"<html") || word(b"<!doctype")
}

/// lxml `_utf8` / `_is_valid_xml_utf8`: what a `.text`/`.tail` assignment accepts (no C0
/// controls but TAB, LF and CR; no U+FFFE, U+FFFF; surrogates cannot occur in parser output).
fn xml_compatible(s: &str) -> bool {
    s.chars().all(|c| {
        matches!(c, '\t' | '\n' | '\r') || (c >= ' ' && c != '\u{fffe}' && c != '\u{ffff}')
    })
}

/// A text slot: an element's `.text` (`false`) or `.tail` (`true`).
type Slot = (NodeId, bool);

/// `x.text = (x.text or '') + s` (or `.tail`). lxml checks the whole new string; `checked`
/// remembers the last slot whose earlier content passed, so merging many bodies into one slot
/// stays linear.
fn append_text(
    d: &mut Document,
    slot: Slot,
    s: &str,
    checked: &mut Option<Slot>,
) -> Result<(), LxmlError> {
    let (id, tail) = slot;
    let t = if tail {
        &mut d.nodes[id as usize].tail
    } else {
        &mut d.nodes[id as usize].text
    };
    if *checked != Some(slot) {
        if !xml_compatible(t.as_deref().unwrap_or("")) {
            return Err(LxmlError::ValueError(NOT_XML_COMPATIBLE.to_owned()));
        }
        *checked = Some(slot);
    }
    if !xml_compatible(s) {
        return Err(LxmlError::ValueError(NOT_XML_COMPATIBLE.to_owned()));
    }
    t.get_or_insert_with(String::new).push_str(s);
    Ok(())
}

/// lxml.html `drop_tree`: remove `el`, joining its tail to the previous sibling or the parent.
fn drop_tree(d: &mut Document, el: NodeId, checked: &mut Option<Slot>) -> Result<(), LxmlError> {
    let parent = d.nodes[el as usize].parent;
    if let Some(tail) = d.nodes[el as usize].tail.take().filter(|t| !t.is_empty()) {
        let slot = match d.prev(el) {
            None => (parent, false),
            Some(prev) => (prev, true),
        };
        let r = append_text(d, slot, &tail, checked);
        d.nodes[el as usize].tail = Some(tail);
        r?;
    }
    d.unlink(el);
    Ok(())
}

/// `target.extend(source)`: move the children of `source` (with their tails) to `target`.
fn extend(d: &mut Document, target: NodeId, source: NodeId) {
    while let Some(c) = d.first_child(source) {
        d.unlink(c);
        d.link_last(target, c);
    }
}

/// lxml.html.defs.block_tags: general block tags, list tags, table tags, partial form tags.
const BLOCK_TAGS: [super::tree::Name; 40] = [
    names::ADDRESS,
    names::BLOCKQUOTE,
    names::CENTER,
    names::DEL,
    names::DIV,
    names::H1,
    names::H2,
    names::H3,
    names::H4,
    names::H5,
    names::H6,
    names::HR,
    names::INS,
    names::ISINDEX,
    names::NOSCRIPT,
    names::P,
    names::PRE,
    names::DIR,
    names::DL,
    names::DT,
    names::DD,
    names::LI,
    names::MENU,
    names::OL,
    names::UL,
    names::TABLE,
    names::CAPTION,
    names::COLGROUP,
    names::COL,
    names::THEAD,
    names::TFOOT,
    names::TBODY,
    names::TR,
    names::TD,
    names::TH,
    names::FIELDSET,
    names::FORM,
    names::LEGEND,
    names::OPTGROUP,
    names::OPTION,
];

/// lxml.html `_contains_block_level_tag`: `el` or an element below it is a block tag.
fn contains_block_level_tag(d: &Document, el: NodeId) -> bool {
    let mut c = el;
    while c != NIL {
        if d.is_element(c) && BLOCK_TAGS.contains(&d.tag(c)) {
            return true;
        }
        c = d.traverse_next(el, c);
    }
    false
}

/// The children of `el` with tag `tag` (`el.findall(tag)`).
fn children_named(d: &Document, el: NodeId, tag: super::tree::Name) -> Vec<NodeId> {
    d.children(el)
        .filter(|&c| d.is_element(c) && d.tag(c) == tag)
        .collect()
}

/// `lxml.html.document_fromstring` and the rest of `lxml.html.fromstring` on the parse result;
/// `is_full_html` is `_looks_like_full_html_*` of the original input.
fn finish(p: parser::Parsed, is_full_html: bool) -> Result<Document, LxmlError> {
    if !p.document {
        return Err(LxmlError::XmlSyntaxError(
            p.error.unwrap_or("Document is not well formed").to_owned(),
        ));
    }
    let mut d = p.doc;
    if d.root == NIL {
        return Err(LxmlError::ParserError("Document is empty".to_owned()));
    }
    if is_full_html {
        return Ok(d);
    }
    let html = d.root;
    let mut checked: Option<Slot> = None;
    let bodies = children_named(&d, html, names::BODY);
    let body = bodies.first().copied();
    if let Some(body) = body {
        for &other in &bodies[1..] {
            if let Some(t) = d.nodes[other as usize]
                .text
                .take()
                .filter(|t| !t.is_empty())
            {
                let slot = match d.last_child(body) {
                    Some(last) => (last, true),
                    None => (body, false),
                };
                let r = append_text(&mut d, slot, &t, &mut checked);
                d.nodes[other as usize].text = Some(t);
                r?;
            }
            extend(&mut d, body, other);
            drop_tree(&mut d, other, &mut checked)?;
        }
    }
    let heads = children_named(&d, html, names::HEAD);
    if let Some(&head) = heads.first() {
        for &other in &heads[1..] {
            extend(&mut d, head, other);
            drop_tree(&mut d, other, &mut checked)?;
        }
        return Ok(d);
    }
    let Some(body) = body else {
        return Ok(d);
    };
    if d.len(body) == 1 {
        let only = d.nodes[body as usize].first;
        if d.text(body).is_none_or(py_strip_is_empty) && d.tail(only).is_none_or(py_strip_is_empty)
        {
            d.root = only;
            return Ok(d);
        }
    }
    let tag = if contains_block_level_tag(&d, body) {
        names::DIV
    } else {
        names::SPAN
    };
    d.set_tag(body, tag);
    d.root = body;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_declaration() {
        let has = |s: &str| has_encoding_declaration(s.as_bytes());
        assert!(has("<?xml version=\"1.0\" encoding=\"utf-8\"?><html>"));
        assert!(!has("<?xml encoding='x'"));
        assert!(has("<?xml  encoding='x'"));
        assert!(has("<?xml v encoding = 'x\""));
        assert!(!has("<?xml version=\"1.0\"?><html>"));
        assert!(!has("<?xmlencoding='x'"));
        assert!(!has("<?xml version='1'> encoding='x'"));
        assert!(!has("<?XML encoding='x'"));
        assert!(!has("<?xml v\u{3000}encoding='x"));
        assert!(has("<?xml v\u{3000}encoding\u{a0}=\u{2028}'x'"));
    }

    #[test]
    fn full_html() {
        let f = |s: &str, b| looks_like_full_html(s.as_bytes(), b);
        assert!(f(" \n<HTML>", false));
        assert!(f("\u{a0}<!DocType html>", false));
        assert!(!f("\u{a0}<!DocType html>", true));
        assert!(!f("<head><html>", false));
        assert!(f("\x0b<html", true));
    }

    #[test]
    fn surrogates() {
        // "a😀b" as surrogatepass UTF-8 (2-byte kind: the pair is decoded)
        let (input, feed) = str_path_input(b"a\xED\xA0\xBD\xED\xB8\x80b");
        assert_eq!(input, "a\u{1f600}b".as_bytes());
        assert_eq!(feed.error_read, None);
        let (input, feed) = str_path_input(b"a\xED\xA0\xBDb");
        assert_eq!(input, b"a");
        assert_eq!(feed.error_read, Some(1));
        // A lone surrogate past the first read: the input still ends there.
        let mut s = "x".repeat(3000).into_bytes();
        s.extend_from_slice(b"\xED\xB0\x80tail");
        let (input, feed) = str_path_input(&s);
        assert_eq!(input.len(), 3000);
        assert_eq!(feed.error_read, Some(2));
    }

    #[test]
    fn feeds() {
        // Latin-1: 4000 characters per read, 2 UTF-8 bytes each here.
        let s = "é".repeat(5000);
        let f = str_feed(s.as_bytes());
        assert_eq!(f.ends, vec![8000, 10000]);
        // UCS-4: 1000 characters per read.
        let s = "😀".repeat(1500);
        assert_eq!(str_feed(s.as_bytes()).ends, vec![4000, 6000]);
        assert_eq!(str_feed(b"ascii").ends, Vec::<usize>::new());
    }

    fn dump(d: &Document) -> String {
        let mut s = String::new();
        d.dump(d.root, true, &mut s);
        s
    }

    #[test]
    fn fragments() {
        let d = fromstring_str("<p>Hello <b>world</b>").unwrap();
        assert_eq!(d.tag_name(d.root), "p");
        assert_eq!(d.text_content(d.root), "Hello world");
        assert_eq!(d.tag_name(d.parent(d.root).unwrap()), "body");
        let d = fromstring_str("text <b>x</b>").unwrap();
        assert_eq!(d.tag_name(d.root), "span");
        let d = fromstring_str("<p>a</p><p>b</p>").unwrap();
        assert_eq!(d.tag_name(d.root), "div");
        let d = fromstring_str("<title>t</title><p>x").unwrap();
        assert_eq!(d.tag_name(d.root), "html");
        let d = fromstring_str("<!doctype html><p>x").unwrap();
        assert_eq!(d.tag_name(d.root), "html");
        assert_eq!(
            dump(&fromstring_str("<html><body>a</body><body>b<i>c</i></body>").unwrap()),
            dump(&fromstring_str("<html><body>a</body><body>b<i>c</i></body>").unwrap())
        );
    }

    #[test]
    fn errors() {
        assert_eq!(
            fromstring_str("").unwrap_err(),
            LxmlError::ParserError("Document is empty".into())
        );
        assert_eq!(
            fromstring_str(" \n\t").unwrap_err(),
            LxmlError::ParserError("Document is empty".into())
        );
        assert!(
            fromstring_str("<?xml version='1.0' encoding='utf-8'?><p>x")
                .unwrap_err()
                .is_value_error()
        );
        assert!(fromstring_bytes(b"<?xml version='1.0' encoding='utf-8'?><p>x").is_ok());
        assert_eq!(
            fromstring_wtf8(b"<p>\xff").unwrap_err(),
            LxmlError::InvalidInput(3)
        );
        // A second body whose text holds a control character: lxml's .tail setter refuses it.
        let e = fromstring_str("<body><p>a</p></body><body>&#1;x</body>").unwrap_err();
        assert!(e.is_value_error(), "{e:?}");
    }

    #[test]
    fn surrogate_paths() {
        // str path: the input ends at the lone surrogate; bytes path: three U+FFFD.
        let s = b"<p>ab\xED\xA0\x80cd</p>";
        let d = fromstring_wtf8(s).unwrap();
        assert_eq!(d.text_content(d.root), "ab");
        let d = fromstring_bytes(s).unwrap();
        assert_eq!(d.text_content(d.root), "ab\u{fffd}\u{fffd}\u{fffd}cd");
    }
}
