// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! A byte-identical Rust port of [ftfy] 6.3.1 ("fixes text for you").
//!
//! ftfy repairs mojibake (text decoded with the wrong encoding) and a list of
//! smaller glitches. This module reproduces `ftfy.fix_text` and friends
//! exactly, including the heuristics' quirks, so that the output is the same
//! string ftfy 6.3.1 returns on the same CPython version.
//!
//! ```
//! use pyroid_core::ftfy::{fix_text, TextFixerConfig};
//!
//! // ftfy.fix_text uses `explain=False` when no config is given.
//! let config = TextFixerConfig { explain: false, ..TextFixerConfig::default() };
//! assert_eq!(fix_text("âœ” No problems", &config), "✔ No problems");
//! assert_eq!(fix_text("ＬＯＵＤ　ＮＯＩＳＥＳ", &config), "LOUD NOISES");
//! ```
//!
//! # Unicode versions
//!
//! ftfy's output depends on the interpreter's `unicodedata` version (for
//! normalization and for the regex classes `\s`, `\w`, `\d`). Set
//! [`TextFixerConfig::unicode_version`] to the version of the CPython whose
//! output you need; the Python bindings do this automatically.
//!
//! # Surrogates
//!
//! Python strings may contain lone surrogates, which Rust strings cannot. The
//! `*_wtf8` functions accept and return bytes in the encoding Python's
//! `"surrogatepass"` handler produces (UTF-8, with each surrogate code unit
//! encoded on its own), which is how the bindings pass such strings through.
//!
//! [ftfy]: https://github.com/rspeer/python-ftfy

mod badness;
mod chardata;
mod codecs;
mod features;
mod fixes;
mod normalize;
#[rustfmt::skip]
mod tables;
mod text;
#[rustfmt::skip]
mod unicode_tables;

use std::borrow::Cow;
use std::fmt;
use std::num::NonZeroUsize;

use chardata::{
    LATIN_1, N_ENCODINGS, class_of, decode_windows_1252, encode, encoding_name, is_sloppy,
    possible_encodings,
};
use tables::{CLUE_CONT, CLUE_FIRST2, CLUE_FIRST3, CLUE_FIRST4, CLUE_STRICT};
use text::{cp_len, cp_offset, decode_at, decode_before};

pub use text::validate as validate_wtf8;

/// The `unicodedata` version whose behaviour to reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum UnicodeVersion {
    /// Unicode 14.0 (CPython 3.11).
    V14_0,
    /// Unicode 15.0 (CPython 3.12).
    V15_0,
    /// Unicode 15.1 (CPython 3.13).
    V15_1,
    /// Unicode 16.0 (CPython 3.14).
    V16_0,
    /// Unicode 17.0 (CPython 3.15).
    V17_0,
}

impl UnicodeVersion {
    /// The newest supported version.
    pub const LATEST: Self = Self::V17_0;

    /// Parse `unicodedata.unidata_version` (e.g. `"15.0.0"`).
    #[must_use]
    pub fn from_unidata_version(v: &str) -> Option<Self> {
        Some(match v {
            "14.0.0" => Self::V14_0,
            "15.0.0" => Self::V15_0,
            "15.1.0" => Self::V15_1,
            "16.0.0" => Self::V16_0,
            "17.0.0" => Self::V17_0,
            _ => return None,
        })
    }

    /// The version as `unicodedata.unidata_version` spells it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V14_0 => "14.0.0",
            Self::V15_0 => "15.0.0",
            Self::V15_1 => "15.1.0",
            Self::V16_0 => "16.0.0",
            Self::V17_0 => "17.0.0",
        }
    }
}

/// ftfy's `unescape_html` option: `"auto"`, `True` or `False`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnescapeHtml {
    /// Decode entities unless the segment contains a literal `<`.
    Auto,
    /// Always decode entities.
    Always,
    /// Never decode entities.
    Never,
}

/// A Unicode normalization form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Normalization {
    /// `"NFC"`
    Nfc,
    /// `"NFD"`
    Nfd,
    /// `"NFKC"`
    Nfkc,
    /// `"NFKD"`
    Nfkd,
}

impl Normalization {
    /// The form's name, as `unicodedata.normalize` spells it.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nfc => "NFC",
            Self::Nfd => "NFD",
            Self::Nfkc => "NFKC",
            Self::Nfkd => "NFKD",
        }
    }

    /// Parse `"NFC"`, `"NFD"`, `"NFKC"` or `"NFKD"`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "NFC" => Self::Nfc,
            "NFD" => Self::Nfd,
            "NFKC" => Self::Nfkc,
            "NFKD" => Self::Nfkd,
            _ => return None,
        })
    }
}

/// ftfy's `TextFixerConfig`. `Default` gives ftfy's defaults, including
/// `explain: true`; note that `ftfy.fix_text(text)` with no config uses
/// `explain=False`.
///
/// As in ftfy, when `explain` is false, `fix_text` / `fix_and_explain` fix
/// encodings with the *default* values of `restore_byte_a0`,
/// `replace_lossy_sequences`, `decode_inconsistent_utf8` and
/// `fix_c1_controls` (ftfy calls `fix_encoding(text)` without the config);
/// `fix_c1_controls` still controls the separate C1 fixer step.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TextFixerConfig {
    pub unescape_html: UnescapeHtml,
    pub remove_terminal_escapes: bool,
    pub fix_encoding: bool,
    pub restore_byte_a0: bool,
    pub replace_lossy_sequences: bool,
    pub decode_inconsistent_utf8: bool,
    pub fix_c1_controls: bool,
    pub fix_latin_ligatures: bool,
    pub fix_character_width: bool,
    pub uncurl_quotes: bool,
    pub fix_line_breaks: bool,
    pub fix_surrogates: bool,
    pub remove_control_chars: bool,
    pub normalization: Option<Normalization>,
    /// Longest segment, in code points, that is fixed at once.
    pub max_decode_length: NonZeroUsize,
    pub explain: bool,
    /// Which CPython `unicodedata` version to match (not an ftfy option).
    pub unicode_version: UnicodeVersion,
}

impl Default for TextFixerConfig {
    fn default() -> Self {
        Self {
            unescape_html: UnescapeHtml::Auto,
            remove_terminal_escapes: true,
            fix_encoding: true,
            restore_byte_a0: true,
            replace_lossy_sequences: true,
            decode_inconsistent_utf8: true,
            fix_c1_controls: true,
            fix_latin_ligatures: true,
            fix_character_width: true,
            uncurl_quotes: true,
            fix_line_breaks: true,
            fix_surrogates: true,
            remove_control_chars: true,
            normalization: Some(Normalization::Nfc),
            max_decode_length: NonZeroUsize::new(1_000_000).expect("nonzero"),
            explain: true,
            unicode_version: UnicodeVersion::LATEST,
        }
    }
}

/// The kind of an [`ExplanationStep`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// `"encode"`: str to bytes with the named encoding.
    Encode,
    /// `"decode"`: bytes to str with the named encoding.
    Decode,
    /// `"transcode"`: bytes to bytes with the named fixer.
    Transcode,
    /// `"apply"`: str to str with the named fixer.
    Apply,
    /// `"normalize"`: Unicode normalization with the named form.
    Normalize,
}

impl Action {
    /// The action's name in ftfy's explanations.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Encode => "encode",
            Self::Decode => "decode",
            Self::Transcode => "transcode",
            Self::Apply => "apply",
            Self::Normalize => "normalize",
        }
    }
}

/// One step of an explanation, like ftfy's `ExplanationStep`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExplanationStep {
    pub action: Action,
    pub parameter: &'static str,
}

impl ExplanationStep {
    const fn new(action: Action, parameter: &'static str) -> Self {
        Self { action, parameter }
    }
}

impl fmt::Display for ExplanationStep {
    /// Formats like ftfy's `repr`: `('encode', 'latin-1')`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "('{}', '{}')", self.action.as_str(), self.parameter)
    }
}

/// Fixed text plus its explanation (`None` when `explain` is false).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplainedText<T> {
    pub text: T,
    pub explanation: Option<Vec<ExplanationStep>>,
}

/// The `*_wtf8` functions were given bytes that are not UTF-8 (even allowing
/// encoded surrogates). The field is the offset of the first bad byte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidWtf8(pub usize);

impl fmt::Display for InvalidWtf8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid UTF-8 at byte {}", self.0)
    }
}

impl std::error::Error for InvalidWtf8 {}

// ---------------------------------------------------------------------------
// fix_encoding
// ---------------------------------------------------------------------------

/// The sub-options `fix_encoding` reads from its config.
#[derive(Clone, Copy)]
struct EncodingOptions {
    restore_byte_a0: bool,
    replace_lossy_sequences: bool,
    decode_inconsistent_utf8: bool,
    fix_c1_controls: bool,
}

const DEFAULT_ENCODING_OPTIONS: EncodingOptions = EncodingOptions {
    restore_byte_a0: true,
    replace_lossy_sequences: true,
    decode_inconsistent_utf8: true,
    fix_c1_controls: true,
};

impl TextFixerConfig {
    fn encoding_options(&self) -> EncodingOptions {
        EncodingOptions {
            restore_byte_a0: self.restore_byte_a0,
            replace_lossy_sequences: self.replace_lossy_sequences,
            decode_inconsistent_utf8: self.decode_inconsistent_utf8,
            fix_c1_controls: self.fix_c1_controls,
        }
    }
}

type Plan<'a> = Option<&'a mut Vec<ExplanationStep>>;

#[inline]
fn record(plan: &mut Plan<'_>, step: ExplanationStep) {
    if let Some(p) = plan.as_deref_mut() {
        p.push(step);
    }
}

/// One `UTF8_DETECTOR_RE` iteration at `q`: the end of the matched sequence.
#[inline]
fn utf8_clue_seq(text: &[u8], q: usize) -> Option<usize> {
    let (c0, n0) = decode_at(text, q);
    let clue = class_of(c0).clue;
    if clue & (CLUE_FIRST2 | CLUE_FIRST3 | CLUE_FIRST4) == 0 {
        return None;
    }
    // Ends of the following 1..=3 continuation characters.
    let mut ends = [0usize; 3];
    let mut k = 0;
    let mut p = q + n0;
    while k < 3 && p < text.len() {
        let (c, n) = decode_at(text, p);
        if class_of(c).clue & CLUE_CONT == 0 {
            break;
        }
        p += n;
        ends[k] = p;
        k += 1;
    }
    // Alternatives in order: first_of_2, first_of_3, first_of_4.
    if clue & CLUE_FIRST2 != 0 && k >= 1 {
        Some(ends[0])
    } else if clue & CLUE_FIRST3 != 0 && k >= 2 {
        Some(ends[1])
    } else if clue & CLUE_FIRST4 != 0 && k >= 3 {
        Some(ends[2])
    } else {
        None
    }
}

/// The next `UTF8_DETECTOR_RE` match at or after `from`.
fn utf8_detector_find(text: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut pos = from;
    while pos < text.len() {
        let behind_ok = pos == 0 || class_of(decode_before(text, pos)).clue & CLUE_STRICT == 0;
        if behind_ok {
            let mut end = pos;
            while end < text.len() {
                match utf8_clue_seq(text, end) {
                    Some(e) => end = e,
                    None => break,
                }
            }
            if end > pos {
                return Some((pos, end));
            }
        }
        pos += decode_at(text, pos).1;
    }
    None
}

/// ftfy's `decode_inconsistent_utf8`.
fn decode_inconsistent_utf8(text: &[u8], v: UnicodeVersion) -> Option<Vec<u8>> {
    let total = cp_len(text);
    let mut out: Option<Vec<u8>> = None;
    let mut copied = 0;
    let mut from = 0;
    while let Some((s, e)) = utf8_detector_find(text, from) {
        let sub = &text[s..e];
        if cp_len(sub) < total && badness::is_bad(sub, v) {
            // `ftfy.fix_encoding(substr)`, i.e. with the default config.
            if let Some(fixed) = fix_encoding_loop(sub, DEFAULT_ENCODING_OPTIONS, v, None, None) {
                let buf = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
                buf.extend_from_slice(&text[copied..s]);
                buf.extend_from_slice(&fixed);
                copied = e;
            }
        }
        from = e;
    }
    let mut buf = out?;
    buf.extend_from_slice(&text[copied..]);
    (buf != text).then_some(buf)
}

/// ftfy's `_fix_encoding_one_step_and_explain`. Returns the new text if the
/// step produced one (which may equal the input).
fn fix_encoding_one_step(
    text: &[u8],
    o: EncodingOptions,
    v: UnicodeVersion,
    plan: &mut Plan<'_>,
    known: Option<(bool, u16)>,
) -> Option<Vec<u8>> {
    if text.is_empty() {
        return None;
    }
    let (ascii, possible) = known.unwrap_or_else(|| possible_encodings(text));
    if ascii || !badness::is_bad(text, v) {
        return None;
    }

    for enc in 0..N_ENCODINGS {
        if possible & (1 << enc) == 0 {
            continue;
        }
        let mut bytes = encode(enc, text).expect("possible_encoding implies encodable");
        let mut transcoded: [Option<&'static str>; 2] = [None, None];
        if o.restore_byte_a0 && encoding_name(enc) != "macroman" && fixes::has_altered_utf8(&bytes)
        {
            let replaced = fixes::restore_byte_a0(&bytes);
            if replaced != bytes {
                transcoded[0] = Some("restore_byte_a0");
                bytes = replaced;
            }
        }
        if o.replace_lossy_sequences && is_sloppy(enc) {
            let replaced = fixes::replace_lossy_sequences(&bytes);
            if replaced != bytes {
                transcoded[1] = Some("replace_lossy_sequences");
                bytes = replaced;
            }
        }
        let variants = bytes.iter().any(|&b| b == 0xED || b == 0xC0);
        let fixed = if variants {
            codecs::decode_utf8_variants(&bytes).map(Cow::into_owned)
        } else {
            codecs::decode_utf8(&bytes).map(<[u8]>::to_vec)
        };
        if let Some(fixed) = fixed {
            record(
                plan,
                ExplanationStep::new(Action::Encode, encoding_name(enc)),
            );
            for name in transcoded.into_iter().flatten() {
                record(plan, ExplanationStep::new(Action::Transcode, name));
            }
            let decoding = if variants { "utf-8-variants" } else { "utf-8" };
            record(plan, ExplanationStep::new(Action::Decode, decoding));
            return Some(fixed);
        }
    }

    if o.decode_inconsistent_utf8 && utf8_detector_find(text, 0).is_some() {
        if let Some(fixed) = decode_inconsistent_utf8(text, v) {
            record(
                plan,
                ExplanationStep::new(Action::Apply, "decode_inconsistent_utf8"),
            );
            return Some(fixed);
        }
    }

    // ftfy checks `"windows-1252" in possible_1byte_encodings`, which is never
    // true (the list holds "sloppy-windows-1252"), so Latin-1 text always
    // tries a strict windows-1252 decoding here.
    if possible & (1 << LATIN_1) != 0 {
        let bytes = encode(LATIN_1, text).expect("latin-1 possible");
        if let Some(fixed) = decode_windows_1252(&bytes) {
            if fixed != text {
                record(plan, ExplanationStep::new(Action::Encode, "latin-1"));
                record(plan, ExplanationStep::new(Action::Decode, "windows-1252"));
                return Some(fixed);
            }
        }
    }

    if o.fix_c1_controls && fixes::has_c1(text) {
        // Recorded even when nothing changes, as in ftfy.
        record(
            plan,
            ExplanationStep::new(Action::Transcode, "fix_c1_controls"),
        );
        return fixes::fix_c1_controls(text);
    }
    None
}

/// ftfy's `fix_encoding_and_explain` loop. `None` if unchanged.
fn fix_encoding_loop(
    text: &[u8],
    o: EncodingOptions,
    v: UnicodeVersion,
    mut plan: Plan<'_>,
    known: Option<(bool, u16)>,
) -> Option<Vec<u8>> {
    let mut cur: Cow<'_, [u8]> = Cow::Borrowed(text);
    let mut known = known;
    while let Some(next) = fix_encoding_one_step(&cur, o, v, &mut plan, known.take()) {
        if next[..] == cur[..] {
            break;
        }
        cur = Cow::Owned(next);
    }
    match cur {
        Cow::Borrowed(_) => None,
        Cow::Owned(v) => Some(v),
    }
}

// ---------------------------------------------------------------------------
// fix_and_explain / fix_text
// ---------------------------------------------------------------------------

/// ftfy's `fix_and_explain` on one segment. `None` if unchanged.
fn fix_segment(seg: &[u8], cfg: &TextFixerConfig, mut steps: Plan<'_>) -> Option<Vec<u8>> {
    let v = cfg.unicode_version;
    let unescape = match cfg.unescape_html {
        UnescapeHtml::Auto => !seg.contains(&b'<'),
        UnescapeHtml::Always => true,
        UnescapeHtml::Never => false,
    };
    // With explain=False, ftfy calls `fix_encoding(text)` with default options.
    let enc_opts = if cfg.explain {
        cfg.encoding_options()
    } else {
        DEFAULT_ENCODING_OPTIONS
    };
    let mut cur: Cow<'_, [u8]> = Cow::Borrowed(seg);
    loop {
        let mut work: Option<Vec<u8>> = None;
        // Which fixers can fire; refreshed whenever a step changes the text.
        let mut feat = features::scan(&cur);
        // ftfy's `_try_fix`: apply if enabled, record the step if it changed.
        macro_rules! try_fix {
            ($enabled:expr, $trigger:expr, $name:literal, $f:expr) => {
                if $enabled && feat.has($trigger) {
                    let input: &[u8] = work.as_deref().unwrap_or(&cur);
                    if let Some(next) = $f(input) {
                        if next[..] != input[..] {
                            record(&mut steps, ExplanationStep::new(Action::Apply, $name));
                            feat = features::scan(&next);
                            work = Some(next);
                        }
                    }
                }
            };
        }
        try_fix!(
            unescape,
            features::AMP,
            "unescape_html",
            fixes::unescape_html
        );
        if cfg.fix_encoding && feat.encoding_candidate() {
            let input: &[u8] = work.as_deref().unwrap_or(&cur);
            let plan = if cfg.explain {
                steps.as_deref_mut()
            } else {
                None
            };
            let known = Some((!feat.has(features::NON_ASCII), feat.encodings));
            if let Some(next) = fix_encoding_loop(input, enc_opts, v, plan, known) {
                feat = features::scan(&next);
                work = Some(next);
            }
        }
        try_fix!(
            cfg.fix_c1_controls,
            features::C1,
            "fix_c1_controls",
            fixes::fix_c1_controls
        );
        try_fix!(
            cfg.fix_latin_ligatures,
            features::LIGATURE,
            "fix_latin_ligatures",
            fixes::fix_latin_ligatures
        );
        try_fix!(
            cfg.fix_character_width,
            features::WIDTH,
            "fix_character_width",
            fixes::fix_character_width
        );
        try_fix!(
            cfg.uncurl_quotes,
            features::CURLY,
            "uncurl_quotes",
            fixes::uncurl_quotes
        );
        try_fix!(
            cfg.fix_line_breaks,
            features::LINE_BREAK,
            "fix_line_breaks",
            fixes::fix_line_breaks
        );
        try_fix!(
            cfg.fix_surrogates,
            features::SURROGATE,
            "fix_surrogates",
            fixes::fix_surrogates
        );
        try_fix!(
            cfg.remove_terminal_escapes,
            features::ESC,
            "remove_terminal_escapes",
            |t| { fixes::remove_terminal_escapes(t, v) }
        );
        try_fix!(
            cfg.remove_control_chars,
            features::CONTROL,
            "remove_control_chars",
            fixes::remove_control_chars
        );
        if let Some(form) = cfg.normalization {
            let trigger = if form == Normalization::Nfc {
                features::NFC_ACTIVE
            } else {
                features::NON_ASCII
            };
            if feat.has(trigger) {
                let input: &[u8] = work.as_deref().unwrap_or(&cur);
                if let Some(next) = normalize::normalize(input, form, v) {
                    record(
                        &mut steps,
                        ExplanationStep::new(Action::Normalize, form.as_str()),
                    );
                    work = Some(next);
                }
            }
        }
        match work {
            None => break,
            Some(next) if next[..] == cur[..] => break,
            Some(next) => cur = Cow::Owned(next),
        }
    }
    match cur {
        Cow::Borrowed(_) => None,
        Cow::Owned(v) => Some(v),
    }
}

/// ftfy's `fix_text` on surrogate-tolerant bytes (already validated).
fn fix_text_bytes<'a>(text: &'a [u8], config: &TextFixerConfig) -> Cow<'a, [u8]> {
    if features::is_quiet_ascii(text) {
        return Cow::Borrowed(text);
    }
    let max = config.max_decode_length.get();
    let mut cfg: Cow<'_, TextFixerConfig> = Cow::Borrowed(config);
    let mut out: Option<Vec<u8>> = None;
    let mut pos = 0;
    while pos < text.len() {
        let mut end = memchr::memchr(b'\n', &text[pos..]).map_or(text.len(), |k| pos + k + 1);
        if end - pos > max && cp_len(&text[pos..end]) > max {
            end = pos + cp_offset(&text[pos..end], max);
        }
        let seg = &text[pos..end];
        if cfg.unescape_html == UnescapeHtml::Auto && seg.contains(&b'<') {
            cfg.to_mut().unescape_html = UnescapeHtml::Never;
        }
        // fix_text discards the explanation, but `explain` still changes
        // which encoding options apply, so pass a sink when it is set.
        let mut sink = Vec::new();
        let steps = if cfg.explain { Some(&mut sink) } else { None };
        match fix_segment(seg, &cfg, steps) {
            Some(fixed) => {
                let buf = out.get_or_insert_with(|| {
                    let mut b = Vec::with_capacity(text.len() + 16);
                    b.extend_from_slice(&text[..pos]);
                    b
                });
                buf.extend_from_slice(&fixed);
            }
            None => {
                if let Some(buf) = out.as_mut() {
                    buf.extend_from_slice(seg);
                }
            }
        }
        pos = end;
    }
    match out {
        Some(v) => Cow::Owned(v),
        None => Cow::Borrowed(text),
    }
}

fn explained_bytes<'a>(text: &'a [u8], config: &TextFixerConfig) -> ExplainedText<Cow<'a, [u8]>> {
    let mut steps = Vec::new();
    let sink = if config.explain {
        Some(&mut steps)
    } else {
        None
    };
    let fixed = fix_segment(text, config, sink);
    ExplainedText {
        text: fixed.map_or(Cow::Borrowed(text), Cow::Owned),
        explanation: config.explain.then_some(steps),
    }
}

fn encoding_explained_bytes<'a>(
    text: &'a [u8],
    config: &TextFixerConfig,
) -> ExplainedText<Cow<'a, [u8]>> {
    if !config.fix_encoding {
        return ExplainedText {
            text: Cow::Borrowed(text),
            explanation: Some(Vec::new()),
        };
    }
    let mut steps = Vec::new();
    let fixed = fix_encoding_loop(
        text,
        config.encoding_options(),
        config.unicode_version,
        Some(&mut steps),
        None,
    );
    ExplainedText {
        text: fixed.map_or(Cow::Borrowed(text), Cow::Owned),
        explanation: Some(steps),
    }
}

fn into_str(b: Cow<'_, [u8]>) -> Cow<'_, str> {
    match b {
        Cow::Borrowed(b) => Cow::Borrowed(std::str::from_utf8(b).expect("valid input")),
        Cow::Owned(v) => Cow::Owned(String::from_utf8(v).expect("fixes never create surrogates")),
    }
}

fn checked(text: &[u8]) -> Result<&[u8], InvalidWtf8> {
    text::validate(text).map(|()| text).map_err(InvalidWtf8)
}

/// ftfy's `fix_text`: fix `text` line by line (segments of at most
/// `max_decode_length` code points).
#[must_use]
pub fn fix_text<'a>(text: &'a str, config: &TextFixerConfig) -> Cow<'a, str> {
    into_str(fix_text_bytes(text.as_bytes(), config))
}

/// ftfy's `fix_text_segment`: fix `text` as one segment.
#[must_use]
pub fn fix_text_segment<'a>(text: &'a str, config: &TextFixerConfig) -> Cow<'a, str> {
    into_str(explained_bytes(text.as_bytes(), config).text)
}

/// ftfy's `fix_and_explain`: fix `text` as one segment and explain the fix.
#[must_use]
pub fn fix_and_explain(text: &str, config: &TextFixerConfig) -> ExplainedText<String> {
    let r = explained_bytes(text.as_bytes(), config);
    ExplainedText {
        text: into_str(r.text).into_owned(),
        explanation: r.explanation,
    }
}

/// ftfy's `fix_encoding`: only the mojibake-fixing steps.
#[must_use]
pub fn fix_encoding<'a>(text: &'a str, config: &TextFixerConfig) -> Cow<'a, str> {
    into_str(encoding_explained_bytes(text.as_bytes(), config).text)
}

/// ftfy's `fix_encoding_and_explain`.
#[must_use]
pub fn fix_encoding_and_explain(text: &str, config: &TextFixerConfig) -> ExplainedText<String> {
    let r = encoding_explained_bytes(text.as_bytes(), config);
    ExplainedText {
        text: into_str(r.text).into_owned(),
        explanation: r.explanation,
    }
}

/// [`fix_text`] on surrogate-tolerant UTF-8 (see the module docs).
///
/// # Errors
/// [`InvalidWtf8`] if `text` is not UTF-8, even allowing encoded surrogates.
pub fn fix_text_wtf8<'a>(
    text: &'a [u8],
    config: &TextFixerConfig,
) -> Result<Cow<'a, [u8]>, InvalidWtf8> {
    Ok(fix_text_bytes(checked(text)?, config))
}

/// [`fix_text_segment`] on surrogate-tolerant UTF-8.
///
/// # Errors
/// [`InvalidWtf8`] if `text` is not UTF-8, even allowing encoded surrogates.
pub fn fix_text_segment_wtf8<'a>(
    text: &'a [u8],
    config: &TextFixerConfig,
) -> Result<Cow<'a, [u8]>, InvalidWtf8> {
    Ok(explained_bytes(checked(text)?, config).text)
}

/// [`fix_and_explain`] on surrogate-tolerant UTF-8.
///
/// # Errors
/// [`InvalidWtf8`] if `text` is not UTF-8, even allowing encoded surrogates.
pub fn fix_and_explain_wtf8<'a>(
    text: &'a [u8],
    config: &TextFixerConfig,
) -> Result<ExplainedText<Cow<'a, [u8]>>, InvalidWtf8> {
    Ok(explained_bytes(checked(text)?, config))
}

/// [`fix_encoding`] on surrogate-tolerant UTF-8.
///
/// # Errors
/// [`InvalidWtf8`] if `text` is not UTF-8, even allowing encoded surrogates.
pub fn fix_encoding_wtf8<'a>(
    text: &'a [u8],
    config: &TextFixerConfig,
) -> Result<Cow<'a, [u8]>, InvalidWtf8> {
    Ok(encoding_explained_bytes(checked(text)?, config).text)
}

/// [`fix_encoding_and_explain`] on surrogate-tolerant UTF-8.
///
/// # Errors
/// [`InvalidWtf8`] if `text` is not UTF-8, even allowing encoded surrogates.
pub fn fix_encoding_and_explain_wtf8<'a>(
    text: &'a [u8],
    config: &TextFixerConfig,
) -> Result<ExplainedText<Cow<'a, [u8]>>, InvalidWtf8> {
    Ok(encoding_explained_bytes(checked(text)?, config))
}

/// ftfy's `badness.is_bad`: whether the text looks like it contains mojibake.
#[must_use]
pub fn is_bad(text: &str, version: UnicodeVersion) -> bool {
    badness::is_bad(text.as_bytes(), version)
}

/// ftfy's `badness.badness`: the number of mojibake-looking sequences.
#[must_use]
pub fn badness(text: &str, version: UnicodeVersion) -> usize {
    badness::badness(text.as_bytes(), version)
}

#[cfg(test)]
mod perf_probe {
    //! Where the time goes (ignored; run with --ignored and PYROID_FTFY_BENCH).
    use super::*;
    use std::time::Instant;

    fn docs() -> Vec<String> {
        let path = std::env::var("PYROID_FTFY_BENCH").expect("PYROID_FTFY_BENCH");
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .take(20000)
            .filter_map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).ok()?;
                v["in"].as_str().map(str::to_owned)
            })
            .collect()
    }

    fn time(name: &str, n: usize, mut f: impl FnMut() -> usize) {
        let t = Instant::now();
        let mut acc = 0;
        for _ in 0..3 {
            acc += f();
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / 3.0 / n as f64;
        eprintln!("{name:>28}: {us:7.2} us/doc  (acc {acc})");
    }

    #[test]
    #[ignore = "profiling probe"]
    fn probe() {
        let docs = docs();
        let n = docs.len();
        let cfg = TextFixerConfig {
            explain: false,
            ..TextFixerConfig::default()
        };
        let v = cfg.unicode_version;
        let lines: Vec<&[u8]> = docs
            .iter()
            .flat_map(|d| d.as_bytes().split_inclusive(|&b| b == b'\n'))
            .collect();
        time("fix_text", n, || {
            docs.iter().map(|d| fix_text(d, &cfg).len()).sum()
        });
        time("is_quiet_ascii(doc)", n, || {
            docs.iter()
                .map(|d| usize::from(features::is_quiet_ascii(d.as_bytes())))
                .sum()
        });
        time("scan(lines)", n, || {
            lines.iter().map(|l| features::scan(l).bits as usize).sum()
        });
        let cand: Vec<&&[u8]> = lines
            .iter()
            .filter(|l| features::scan(l).encoding_candidate())
            .collect();
        eprintln!("lines {} candidates {}", lines.len(), cand.len());
        time("possible(cand)", n, || {
            cand.iter().map(|l| possible_encodings(l).1 as usize).sum()
        });
        time("is_bad(cand)", n, || {
            cand.iter()
                .map(|l| usize::from(badness::is_bad(l, v)))
                .sum()
        });
        let bad: Vec<&&&[u8]> = cand.iter().filter(|l| badness::is_bad(l, v)).collect();
        eprintln!("bad candidates {}", bad.len());
        time("fix_encoding_loop(cand)", n, || {
            cand.iter()
                .map(|l| {
                    fix_encoding_loop(l, DEFAULT_ENCODING_OPTIONS, v, None, None)
                        .map_or(0, |x| x.len())
                })
                .sum()
        });
        time("fix_segment(lines)", n, || {
            lines
                .iter()
                .map(|l| fix_segment(l, &cfg, None).map_or(0, |x| x.len()))
                .sum()
        });
        time("uncurl(lines)", n, || {
            lines
                .iter()
                .map(|l| fixes::uncurl_quotes(l).map_or(0, |x| x.len()))
                .sum()
        });
        time("width(lines)", n, || {
            lines
                .iter()
                .map(|l| fixes::fix_character_width(l).map_or(0, |x| x.len()))
                .sum()
        });
        time("nfc(lines)", n, || {
            lines
                .iter()
                .map(|l| normalize::normalize(l, Normalization::Nfc, v).map_or(0, |x| x.len()))
                .sum()
        });
        let mut counts = std::collections::BTreeMap::new();
        let ecfg = TextFixerConfig {
            explain: true,
            ..cfg.clone()
        };
        for l in &lines {
            let mut steps = Vec::new();
            fix_segment(l, &ecfg, Some(&mut steps));
            for s in steps {
                *counts.entry(s.to_string()).or_insert(0) += 1;
            }
        }
        eprintln!("steps applied: {counts:?}");
    }
}
