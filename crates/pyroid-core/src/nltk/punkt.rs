// Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
//! NLTK's Punkt sentence tokenizer (`nltk.tokenize.punkt`), with given parameters.
//!
//! [`Punkt`] is `PunktSentenceTokenizer` holding a [`Params`]
//! (`PunktParameters`) and a [`LangVars`] (`PunktLanguageVars`, of which only
//! `sent_end_chars` varies). Each method returns what NLTK 3.10.4 returns:
//!
//! | method                                 | NLTK                                          |
//! |----------------------------------------|-----------------------------------------------|
//! | [`Punkt::span_tokenize`]               | `span_tokenize(text, realign_boundaries)`     |
//! | [`Punkt::tokenize`]                    | `tokenize(text, realign_boundaries)`          |
//! | [`Punkt::text_contains_sentbreak`]     | `text_contains_sentbreak(text)`               |
//! | [`Punkt::sentences_from_tokens`]       | `sentences_from_tokens(tokens)` (lengths)     |
//! | [`Punkt::sentences_from_text_legacy`]  | `sentences_from_text_legacy(text)`            |
//! | [`Punkt::debug_decisions`]             | `debug_decisions(text)`                       |
//!
//! Offsets are byte offsets into the text; [`char_offsets`] turns them into
//! the code point offsets NLTK reports.
//!
//! NLTK compiles the Punkt patterns with the `regex` module, so `\s`, `\S`,
//! `\w` and `\d` in them are `regex`'s classes ([`rx_space`], [`rx_word`],
//! [`rx_digit`]: U+001C..U+001F are not `\s`), while `str.strip()`,
//! `str.rstrip()`, `str.lower()`, `str.isupper()` and `str.islower()` are
//! CPython's (U+001C..U+001F are whitespace there; case data follows the
//! interpreter's Unicode version). `_get_last_whitespace_index` uses
//! `string.whitespace`, which is ASCII only.
//!
//! As in NLTK, [`Punkt::span_tokenize`] only tokenizes the context around each
//! candidate sentence end (the word before it, the candidate and what follows
//! it), never the whole text. Tokens borrow the text; a token's type is
//! computed only for the tokens the second pass looks at, into reused buffers.

use std::borrow::Cow;
use std::fmt;

use rustc_hash::{FxHashMap, FxHashSet};

use super::chars::{lower, py_lower, py_space, py_upper, rx_digit, rx_space, rx_word};
use crate::ftfy::UnicodeVersion;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// `_ORTHO_BEG_UC`: seen at the beginning of a sentence, upper case.
pub const ORTHO_BEG_UC: i64 = 1 << 1;
/// `_ORTHO_MID_UC`: seen in the middle of a sentence, upper case.
pub const ORTHO_MID_UC: i64 = 1 << 2;
/// `_ORTHO_UNK_UC`: seen at an unknown position, upper case.
pub const ORTHO_UNK_UC: i64 = 1 << 3;
/// `_ORTHO_BEG_LC`: seen at the beginning of a sentence, lower case.
pub const ORTHO_BEG_LC: i64 = 1 << 4;
/// `_ORTHO_MID_LC`: seen in the middle of a sentence, lower case.
pub const ORTHO_MID_LC: i64 = 1 << 5;
/// `_ORTHO_UNK_LC`: seen at an unknown position, lower case.
pub const ORTHO_UNK_LC: i64 = 1 << 6;
const ORTHO_UC: i64 = ORTHO_BEG_UC + ORTHO_MID_UC + ORTHO_UNK_UC;
const ORTHO_LC: i64 = ORTHO_BEG_LC + ORTHO_MID_LC + ORTHO_UNK_LC;

/// The flags and the names `PunktParameters._debug_ortho_context` gives them,
/// in its order (`debug_decisions` puts the names in a set).
pub const ORTHO_NAMES: [(i64, &str); 6] = [
    (ORTHO_BEG_UC, "BEG-UC"),
    (ORTHO_MID_UC, "MID-UC"),
    (ORTHO_UNK_UC, "UNK-UC"),
    (ORTHO_BEG_LC, "BEG-LC"),
    (ORTHO_MID_LC, "MID-LC"),
    (ORTHO_UNK_LC, "UNK-LC"),
];

/// `PunktToken._get_type` of a numeric token.
const NUMBER: &str = "##number##";
/// The same when `$` matched before a final "\n" (only in caller tokens).
const NUMBER_NL: &str = "##number##\n";

const fn ascii_set(chars: &[u8]) -> u128 {
    let mut set = 0u128;
    let mut i = 0;
    while i < chars.len() {
        set |= 1 << chars[i];
        i += 1;
    }
    set
}

#[inline]
fn in_ascii_set(c: char, set: u128) -> bool {
    let cp = u32::from(c);
    cp < 0x80 && (set >> cp) & 1 != 0
}

/// The ASCII part of `_re_non_word_chars` (without the sentence-end characters).
const NON_WORD: u128 = ascii_set(b")\";}]*:@'({[");
/// What `_re_word_start` excludes.
const NOT_WORD_START: u128 = ascii_set(b"(\"`{[:;&#*@)}]-,");
/// `PunktSentenceTokenizer.PUNCTUATION`.
const PUNCTUATION: u128 = ascii_set(b";:,.!?");

/// The non-ASCII members of `_re_non_word_chars`: the curly quotes and the
/// guillemets.
#[inline]
fn is_non_word_quote(c: char) -> bool {
    matches!(
        c,
        '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}' | '\u{ab}' | '\u{bb}'
    )
}

/// The character class of `re_boundary_realignment`.
#[inline]
fn is_closing(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '}') || is_non_word_quote(c)
}

/// `string.whitespace`.
#[inline]
fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

/// The character at byte `i` of `s` (`i` is a character boundary).
#[inline]
fn char_at(s: &str, i: usize) -> Option<char> {
    let b = *s.as_bytes().get(i)?;
    if b < 0x80 {
        Some(char::from(b))
    } else {
        s.get(i..)?.chars().next()
    }
}

/// The end of the run of characters matching `pred` that starts at byte `i`.
#[inline]
fn skip_while(s: &str, mut i: usize, pred: impl Fn(char) -> bool) -> usize {
    while let Some(c) = char_at(s, i) {
        if !pred(c) {
            break;
        }
        i += c.len_utf8();
    }
    i
}

/// The one character of `s`, if it has exactly one.
#[inline]
fn single_char(s: &str) -> Option<char> {
    let mut it = s.chars();
    let c = it.next()?;
    it.next().is_none().then_some(c)
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// `PunktParameters`: abbreviation types, collocations, frequent sentence
/// starters and orthographic contexts.
#[derive(Clone, Default)]
pub struct Params {
    abbrev_types: FxHashSet<String>,
    /// `(a, b)` pairs, keyed by `a` so that a lookup needs no allocation.
    collocations: FxHashMap<String, FxHashSet<String>>,
    sent_starters: FxHashSet<String>,
    ortho_context: FxHashMap<String, i64>,
}

impl fmt::Debug for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Params")
            .field("abbrev_types", &self.abbrev_types.len())
            .field(
                "collocations",
                &self
                    .collocations
                    .values()
                    .map(FxHashSet::len)
                    .sum::<usize>(),
            )
            .field("sent_starters", &self.sent_starters.len())
            .field("ortho_context", &self.ortho_context.len())
            .finish()
    }
}

/// A `ValueError` that loading a `punkt_tab` model raises.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamsError {
    /// The file (`"ortho_context.tab"`).
    pub file: &'static str,
    /// The 1-based line number.
    pub line: usize,
    /// Python's message.
    pub message: String,
}

impl fmt::Display for ParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.file, self.line, self.message)
    }
}

impl std::error::Error for ParamsError {}

impl Params {
    /// Parameters from their four tables. Of repeated `ortho_context` keys
    /// the last value counts, as in a dict.
    pub fn new(
        abbrev_types: impl IntoIterator<Item = String>,
        collocations: impl IntoIterator<Item = (String, String)>,
        sent_starters: impl IntoIterator<Item = String>,
        ortho_context: impl IntoIterator<Item = (String, i64)>,
    ) -> Self {
        let mut colloc: FxHashMap<String, FxHashSet<String>> = FxHashMap::default();
        for (a, b) in collocations {
            colloc.entry(a).or_default().insert(b);
        }
        let mut ortho = FxHashMap::default();
        for (k, v) in ortho_context {
            ortho.insert(k, v);
        }
        Self {
            abbrev_types: abbrev_types.into_iter().collect(),
            collocations: colloc,
            sent_starters: sent_starters.into_iter().collect(),
            ortho_context: ortho,
        }
    }

    /// `load_punkt_params` on a `punkt_tab` model directory, given the text of
    /// its four files (decoded as UTF-8): `collocations.tab`,
    /// `sent_starters.txt`, `abbrev_types.txt` and `ortho_context.tab`.
    ///
    /// As `nltk.tabdata.PunktDecoder` reads them: universal newlines (`\r\n`
    /// and `\r` end lines too), one final `\n` removed from each line, fields
    /// split on tabs. A collocation line that does not have exactly two
    /// fields is a tuple no lookup can find, so it is dropped. An
    /// `ortho_context` line must have two fields and an `int()` second field;
    /// otherwise this is the `ValueError` NLTK raises. Values beyond `i64`
    /// keep their low 64 bits (two's complement, as Python's `&` sees them).
    pub fn from_punkt_tab(
        collocations: &str,
        sent_starters: &str,
        abbrev_types: &str,
        ortho_context: &str,
    ) -> Result<Self, ParamsError> {
        let mut ortho = Vec::new();
        for (i, line) in tab_lines(&universal_newlines(ortho_context)).enumerate() {
            let err = |message: String| ParamsError {
                file: "ortho_context.tab",
                line: i + 1,
                message,
            };
            let fields: Vec<&str> = line.split('\t').collect();
            match fields[..] {
                [k, v] => {
                    let n = py_int(v).ok_or_else(|| {
                        err(format!("invalid literal for int() with base 10: {v:?}"))
                    })?;
                    ortho.push((k.to_owned(), n));
                }
                [_] => {
                    return Err(err(
                        "not enough values to unpack (expected 2, got 1)".to_owned()
                    ));
                }
                _ => {
                    return Err(err(format!(
                        "too many values to unpack (expected 2, got {})",
                        fields.len()
                    )));
                }
            }
        }
        let collocations = universal_newlines(collocations);
        let colloc = tab_lines(&collocations).filter_map(|line| {
            let (a, b) = line.split_once('\t')?;
            (!b.contains('\t')).then(|| (a.to_owned(), b.to_owned()))
        });
        let set = |s: &str| -> Vec<String> {
            tab_lines(&universal_newlines(s))
                .map(str::to_owned)
                .collect()
        };
        Ok(Self::new(
            set(abbrev_types),
            colloc.collect::<Vec<_>>(),
            set(sent_starters),
            ortho,
        ))
    }

    /// Whether `s` is in `abbrev_types`.
    pub fn is_abbrev_type(&self, s: &str) -> bool {
        self.abbrev_types.contains(s)
    }

    /// Whether `(a, b)` is in `collocations`.
    pub fn is_collocation(&self, a: &str, b: &str) -> bool {
        self.collocations.get(a).is_some_and(|set| set.contains(b))
    }

    /// Whether `s` is in `sent_starters`.
    pub fn is_sent_starter(&self, s: &str) -> bool {
        self.sent_starters.contains(s)
    }

    /// `ortho_context[s]` (0 if absent, as the `defaultdict(int)` gives).
    pub fn ortho_context(&self, s: &str) -> i64 {
        self.ortho_context.get(s).copied().unwrap_or(0)
    }
}

/// Python's universal newlines on read: `\r\n` and `\r` become `\n`.
fn universal_newlines(s: &str) -> Cow<'_, str> {
    if s.contains('\r') {
        Cow::Owned(s.replace("\r\n", "\n").replace('\r', "\n"))
    } else {
        Cow::Borrowed(s)
    }
}

/// The lines of a text file as iterating over it and `rm_nl` give them.
fn tab_lines(s: &str) -> impl Iterator<Item = &str> {
    let body = s.strip_suffix('\n').unwrap_or(s);
    body.split('\n')
        .take(if s.is_empty() { 0 } else { usize::MAX })
}

/// `int(s)` for a str `s` (base 10), wrapped to 64 bits.
fn py_int(s: &str) -> Option<i64> {
    let s = s.trim_matches(py_space);
    let (neg, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let mut n: i64 = 0;
    let mut any = false;
    let mut after_underscore = false;
    for c in digits.chars() {
        if c == '_' {
            if !any || after_underscore {
                return None;
            }
            after_underscore = true;
            continue;
        }
        let d = decimal_value(c)?;
        n = n.wrapping_mul(10).wrapping_add(d);
        any = true;
        after_underscore = false;
    }
    if !any || after_underscore {
        return None;
    }
    Some(if neg { n.wrapping_neg() } else { n })
}

/// The value of a decimal digit (general category Nd). Each Unicode script's
/// digits are a run of ten code points from zero to nine.
fn decimal_value(c: char) -> Option<i64> {
    if c.is_ascii_digit() {
        return Some(i64::from(u32::from(c) - u32::from('0')));
    }
    if !rx_digit(c) {
        return None;
    }
    let mut cp = u32::from(c);
    let mut steps = 0i64;
    while let Some(prev) = cp.checked_sub(1).and_then(char::from_u32) {
        if !rx_digit(prev) {
            break;
        }
        steps += 1;
        cp -= 1;
    }
    Some(steps % 10)
}

// ---------------------------------------------------------------------------
// Language variables
// ---------------------------------------------------------------------------

/// How to find the bytes that can start a sentence-end character.
#[derive(Clone, Debug)]
enum Finder {
    None,
    One(u8),
    Two(u8, u8),
    Three(u8, u8, u8),
    Table(Box<[bool; 256]>),
}

impl Finder {
    fn find(&self, hay: &[u8], from: usize) -> Option<usize> {
        let h = hay.get(from..)?;
        let i = match *self {
            Self::None => None,
            Self::One(a) => memchr::memchr(a, h),
            Self::Two(a, b) => memchr::memchr2(a, b, h),
            Self::Three(a, b, c) => memchr::memchr3(a, b, c, h),
            Self::Table(ref t) => h.iter().position(|&b| t[usize::from(b)]),
        }?;
        Some(from + i)
    }
}

/// `PunktLanguageVars`: the sentence-end characters (`sent_end_chars`; NLTK's
/// default is `.`, `?` and `!`). Everything else in the class is fixed.
#[derive(Clone, Debug)]
pub struct LangVars {
    sent_end_chars: Vec<char>,
    end_ascii: u128,
    end_other: Vec<char>,
    non_word_ascii: u128,
    finder: Finder,
}

impl Default for LangVars {
    fn default() -> Self {
        Self::new(vec!['.', '?', '!'])
    }
}

impl LangVars {
    /// Language variables with these `sent_end_chars`, one character each.
    ///
    /// With none, NLTK fails to compile `period_context_re` (an
    /// `nltk.redos.error`) in every method that uses it (`span_tokenize`,
    /// `tokenize`, `debug_decisions`); here those find no candidate.
    pub fn new(sent_end_chars: Vec<char>) -> Self {
        let mut end_ascii = 0u128;
        let mut end_other = Vec::new();
        let mut leads = Vec::new();
        for &c in &sent_end_chars {
            let mut buf = [0u8; 4];
            let lead = c.encode_utf8(&mut buf).as_bytes()[0];
            if !leads.contains(&lead) {
                leads.push(lead);
            }
            if c.is_ascii() {
                end_ascii |= 1 << u32::from(c);
            } else if !end_other.contains(&c) {
                end_other.push(c);
            }
        }
        let non_word_ascii = NON_WORD | (end_ascii & !(1 << b'.'));
        let finder = match leads[..] {
            [] => Finder::None,
            [a] => Finder::One(a),
            [a, b] => Finder::Two(a, b),
            [a, b, c] => Finder::Three(a, b, c),
            _ => {
                let mut table = Box::new([false; 256]);
                for &b in &leads {
                    table[usize::from(b)] = true;
                }
                Finder::Table(table)
            }
        };
        Self {
            sent_end_chars,
            end_ascii,
            end_other,
            non_word_ascii,
            finder,
        }
    }

    /// `sent_end_chars`, as given.
    pub fn sent_end_chars(&self) -> &[char] {
        &self.sent_end_chars
    }

    #[inline]
    fn is_sent_end(&self, c: char) -> bool {
        if c.is_ascii() {
            in_ascii_set(c, self.end_ascii)
        } else {
            self.end_other.contains(&c)
        }
    }

    /// `_re_non_word_chars`: closing and opening punctuation, the curly quotes,
    /// the guillemets, and the sentence-end characters other than `.`.
    #[inline]
    fn is_non_word(&self, c: char) -> bool {
        if c.is_ascii() {
            in_ascii_set(c, self.non_word_ascii)
        } else {
            is_non_word_quote(c) || self.end_other.contains(&c)
        }
    }

    /// `tok.endswith(sent_end_chars)`.
    fn ends_with_sent_end(&self, tok: &str) -> bool {
        tok.chars().next_back().is_some_and(|c| self.is_sent_end(c))
    }

    /// `tok in sent_end_chars` (a tuple of one-character strings).
    fn is_sent_end_token(&self, tok: &str) -> bool {
        single_char(tok).is_some_and(|c| self.is_sent_end(c))
    }

    /// The end of `_re_multi_char_punct` (`\-{2,}|\.{2,}|(?:\.\s){2,}\.`)
    /// matched at byte `p`.
    fn multi_char_end(text: &str, at: usize) -> Option<usize> {
        let bytes = text.as_bytes();
        let run = |byte: u8| at + 2 + bytes[at + 2..].iter().take_while(|&&x| x == byte).count();
        match bytes.get(at)? {
            b'-' if bytes.get(at + 1) == Some(&b'-') => Some(run(b'-')),
            b'.' if bytes.get(at + 1) == Some(&b'.') => Some(run(b'.')),
            b'.' => {
                // Greedy `(?:\.\s){2,}` then `\.`: take every ". " pair, and
                // give the last one back when no dot follows them.
                let (mut pos, mut pairs, mut last) = (at, 0usize, at);
                while bytes.get(pos) == Some(&b'.') {
                    match char_at(text, pos + 1) {
                        Some(space) if rx_space(space) => {
                            last = pos;
                            pairs += 1;
                            pos += 1 + space.len_utf8();
                        }
                        _ => break,
                    }
                }
                if pairs >= 2 && bytes.get(pos) == Some(&b'.') {
                    Some(pos + 1)
                } else if pairs >= 3 {
                    Some(last + 1)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Whether `_re_multi_char_punct` matches at byte `p`.
    fn multi_char_at(text: &str, at: usize) -> bool {
        let bytes = text.as_bytes();
        match bytes.get(at) {
            Some(b'-') => bytes.get(at + 1) == Some(&b'-'),
            Some(b'.') => {
                if bytes.get(at + 1) == Some(&b'.') {
                    return true;
                }
                // ".\s.\s.": the position of the dot after a space at `pos`.
                let space_dot = |pos: usize| match char_at(text, pos) {
                    Some(space) if rx_space(space) => {
                        let dot = pos + space.len_utf8();
                        (bytes.get(dot) == Some(&b'.')).then_some(dot)
                    }
                    _ => None,
                };
                space_dot(at + 1)
                    .and_then(|dot| space_dot(dot + 1))
                    .is_some()
            }
            _ => false,
        }
    }

    /// The lookahead that ends a word in `_word_tokenize_fmt`:
    /// `\s|$|NonWord|MultiChar|,(?=$|\s|NonWord|MultiChar)`.
    fn word_ends_at(&self, text: &str, at: usize) -> bool {
        let Some(c) = char_at(text, at) else {
            return true;
        };
        if rx_space(c) || self.is_non_word(c) || Self::multi_char_at(text, at) {
            return true;
        }
        if c == ',' {
            let after = at + 1;
            return match char_at(text, after) {
                None => true,
                Some(next) => {
                    rx_space(next) || self.is_non_word(next) || Self::multi_char_at(text, after)
                }
            };
        }
        false
    }
}

/// `PunktLanguageVars.word_tokenize(line)`: `findall` of `_word_tokenize_fmt`.
#[derive(Debug)]
struct Words<'l, 's> {
    lang: &'l LangVars,
    s: &'s str,
    pos: usize,
}

impl<'s> Iterator for Words<'_, 's> {
    type Item = &'s str;

    fn next(&mut self) -> Option<&'s str> {
        let s = self.s;
        while let Some(c) = char_at(s, self.pos) {
            let start = self.pos;
            let after = start + c.len_utf8();
            if rx_space(c) {
                self.pos = after;
                continue;
            }
            let end = if let Some(e) = LangVars::multi_char_end(s, start) {
                e
            } else if in_ascii_set(c, NOT_WORD_START) {
                after
            } else {
                // `(?=WordStart)\S+?(?=End)`: the first end after one character.
                let mut e = after;
                while !self.lang.word_ends_at(s, e) {
                    e += char_at(s, e).map_or(1, char::len_utf8);
                }
                e
            };
            self.pos = end;
            return Some(&s[start..end]);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

/// A `PunktToken`'s text and the annotations the tokenizer reads.
#[derive(Clone, Copy, Debug)]
struct Tok<'t> {
    text: &'t str,
    /// `None` until a pass sets it.
    sentbreak: Option<bool>,
    abbr: bool,
    ellipsis: bool,
}

impl Tok<'_> {
    fn period_final(&self) -> bool {
        self.text.ends_with('.')
    }

    fn sentbreak(&self) -> bool {
        self.sentbreak == Some(true)
    }
}

/// `(body, true)` if `s` ends with "\n" (where `$` also matches), else `(s, false)`.
#[inline]
fn split_final_newline(s: &str) -> (&str, bool) {
    match s.strip_suffix('\n') {
        Some(body) => (body, true),
        None => (s, false),
    }
}

/// `_RE_NUMERIC` (`^-?[\.,]?\d[\d,\.-]*\.?$`) on a lowercased token: whether
/// it matches, and whether it stopped before a final "\n".
fn numeric(s: &str) -> Option<bool> {
    let (body, nl) = split_final_newline(s);
    let mut it = body.chars().peekable();
    if it.peek() == Some(&'-') {
        it.next();
    }
    if matches!(it.peek(), Some('.' | ',')) {
        it.next();
    }
    if !it.next().is_some_and(rx_digit) {
        return None;
    }
    it.all(|c| rx_digit(c) || matches!(c, ',' | '.' | '-'))
        .then_some(nl)
}

/// `is_ellipsis`: `\.\.+$` matched at the start.
fn is_ellipsis(tok: &str) -> bool {
    let (body, _) = split_final_newline(tok);
    body.len() >= 2 && body.bytes().all(|b| b == b'.')
}

/// `is_initial`: `[^\W\d]\.$` matched at the start.
fn is_initial(tok: &str) -> bool {
    let mut it = tok.chars();
    let (Some(c), Some('.')) = (it.next(), it.next()) else {
        return false;
    };
    rx_word(c) && !rx_digit(c) && matches!(it.as_str(), "" | "\n")
}

/// `type_no_period`: the type without a final period, if it has more than one
/// character.
#[inline]
fn type_no_period(ty: &str) -> &str {
    if ty.len() > 1 && ty.ends_with('.') {
        &ty[..ty.len() - 1]
    } else {
        ty
    }
}

/// `s.lower()`, into `buf` when it differs from `s`.
fn lower_into<'b>(s: &'b str, v: UnicodeVersion, buf: &'b mut String) -> &'b str {
    if s.is_ascii() {
        if !s.bytes().any(|b| b.is_ascii_uppercase()) {
            return s;
        }
        buf.clear();
        buf.push_str(s);
        buf.make_ascii_lowercase();
        return buf;
    }
    match lower(s, v) {
        Cow::Borrowed(t) => t,
        Cow::Owned(t) => {
            *buf = t;
            buf
        }
    }
}

/// Buffers for lowercased tokens, reused across tokens.
#[derive(Debug, Default)]
struct Scratch {
    first: String,
    second: String,
    abbrev: String,
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The exception an NLTK generator raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PunktError {
    /// `IndexError` with this message: `debug_decisions` found fewer than two
    /// tokens from the candidate on (`"list index out of range"`), or
    /// `sentences_from_tokens` asked for the first character of an empty token
    /// (`"string index out of range"`).
    IndexError(&'static str),
    /// `AssertionError` (no message): `sentences_from_text_legacy` did not find
    /// a token where it expected it in the text. This only happens around
    /// lines made of U+001C..U+001F, which `str.strip()` treats as blank but
    /// `regex`'s `\s` does not skip. (Under `python -O` NLTK skips the
    /// assertion and goes on with misplaced offsets; this port does not.)
    AssertionError,
}

impl PunktError {
    /// The Python exception type's name.
    pub fn python_type(self) -> &'static str {
        match self {
            Self::IndexError(_) => "IndexError",
            Self::AssertionError => "AssertionError",
        }
    }

    /// The exception's message.
    pub fn message(self) -> &'static str {
        match self {
            Self::IndexError(m) => m,
            Self::AssertionError => "",
        }
    }
}

impl fmt::Display for PunktError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.python_type(), self.message())
    }
}

impl std::error::Error for PunktError {}

/// A generator that raised: the items it yielded first, and the exception.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Raised<T> {
    /// What the generator yielded before raising.
    pub yielded: Vec<T>,
    /// What it raised.
    pub error: PunktError,
}

impl<T: fmt::Debug> fmt::Display for Raised<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} after {} items", self.error, self.yielded.len())
    }
}

impl<T: fmt::Debug> std::error::Error for Raised<T> {}

/// What `_ortho_heuristic` says about a token starting a sentence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OrthoHeuristic {
    /// `True`.
    Yes,
    /// `False`.
    No,
    /// `"unknown"`.
    Unknown,
}

/// The `REASON_*` a decision gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reason {
    /// `REASON_DEFAULT_DECISION`: no second-pass rule applied.
    DefaultDecision,
    /// `REASON_KNOWN_COLLOCATION`.
    KnownCollocation,
    /// `REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC`.
    AbbrWithOrthographicHeuristic,
    /// `REASON_ABBR_WITH_SENTENCE_STARTER`.
    AbbrWithSentenceStarter,
    /// `REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC`.
    InitialWithOrthographicHeuristic,
    /// `REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC`, whose text is the same as
    /// the initial one's in NLTK.
    NumberWithOrthographicHeuristic,
    /// `REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC`.
    InitialWithSpecialOrthographicHeuristic,
}

impl Reason {
    /// The reason's text in NLTK.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DefaultDecision => "default decision",
            Self::KnownCollocation => "known collocation (both words)",
            Self::AbbrWithOrthographicHeuristic => "abbreviation + orthographic heuristic",
            Self::AbbrWithSentenceStarter => "abbreviation + frequent sentence starter",
            Self::InitialWithOrthographicHeuristic | Self::NumberWithOrthographicHeuristic => {
                "initial + orthographic heuristic"
            }
            Self::InitialWithSpecialOrthographicHeuristic => {
                "initial + special orthographic heuristic"
            }
        }
    }
}

/// One dict of `debug_decisions`.
#[allow(clippy::struct_excessive_bools)] // NLTK's dict has these flags
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    /// `period_index`: where the candidate is, as a byte offset into the text.
    pub period_index: usize,
    /// `text`: the context the decision was made on.
    pub text: String,
    /// `type1`: the type of the token with the candidate.
    pub type1: String,
    /// `type2`: the type of the token after it.
    pub type2: String,
    /// `type1_in_abbrs`: the first pass found an abbreviation.
    pub type1_in_abbrs: bool,
    /// `type1_is_initial`.
    pub type1_is_initial: bool,
    /// `type2_is_sent_starter`.
    pub type2_is_sent_starter: bool,
    /// `type2_ortho_heuristic`.
    pub type2_ortho_heuristic: OrthoHeuristic,
    /// `type2_ortho_contexts`: the flags of `ORTHO_NAMES` set in the second
    /// type's orthographic context (NLTK gives their names, as a set).
    pub type2_ortho_contexts: i64,
    /// `collocation`.
    pub collocation: bool,
    /// `reason`.
    pub reason: Reason,
    /// `break_decision`: the first token's `sentbreak` (`None` if no pass set it).
    pub break_decision: Option<bool>,
}

// ---------------------------------------------------------------------------
// The tokenizer
// ---------------------------------------------------------------------------

/// `PunktSentenceTokenizer` with fixed parameters and language variables, for
/// one CPython Unicode version (case mappings and `str.isupper` /
/// `str.islower`).
#[derive(Clone, Debug)]
pub struct Punkt {
    params: Params,
    lang: LangVars,
    version: UnicodeVersion,
}

/// A match of `period_context_re`: the candidate's bytes, the end of
/// `after_tok`, and where `next_tok` starts (if that branch matched).
#[derive(Clone, Copy, Debug)]
struct Candidate {
    start: usize,
    end: usize,
    after_end: usize,
    next_tok: Option<usize>,
}

/// `period_context_re().finditer(text)`.
#[derive(Debug)]
struct Candidates<'p, 't> {
    lang: &'p LangVars,
    text: &'t str,
    pos: usize,
}

impl Iterator for Candidates<'_, '_> {
    type Item = Candidate;

    fn next(&mut self) -> Option<Candidate> {
        let text = self.text;
        loop {
            let Some(p) = self.lang.finder.find(text.as_bytes(), self.pos) else {
                self.pos = text.len();
                return None;
            };
            // `p` holds an ASCII byte or a lead byte: a character starts there.
            let c = char_at(text, p)?;
            let end = p + c.len_utf8();
            self.pos = end;
            if !self.lang.is_sent_end(c) {
                continue;
            }
            let Some(n) = char_at(text, end) else {
                continue;
            };
            if self.lang.is_non_word(n) {
                return Some(Candidate {
                    start: p,
                    end,
                    after_end: end + n.len_utf8(),
                    next_tok: None,
                });
            }
            if rx_space(n) {
                let q = skip_while(text, end, rx_space);
                if q < text.len() {
                    return Some(Candidate {
                        start: p,
                        end,
                        after_end: skip_while(text, q, |c| !rx_space(c)),
                        next_tok: Some(q),
                    });
                }
            }
        }
    }
}

/// `_match_potential_end_contexts(text)`: each candidate kept, and where its
/// context starts (the context runs to the candidate's `after_end`).
#[derive(Debug)]
struct Contexts<'p, 't> {
    candidates: Candidates<'p, 't>,
    /// `previous_slice`.
    prev_slice: (usize, usize),
    prev: Option<Candidate>,
}

impl Iterator for Contexts<'_, '_> {
    type Item = (Candidate, usize);

    fn next(&mut self) -> Option<(Candidate, usize)> {
        let bytes = self.candidates.text.as_bytes();
        loop {
            let Some(m) = self.candidates.next() else {
                return self.prev.take().map(|p| (p, self.prev_slice.0));
            };
            // `_get_last_whitespace_index`: 0 both for none and for index 0.
            let before = &bytes[self.prev_slice.1..m.start];
            let last = before.iter().rposition(|&b| is_ascii_ws(b)).unwrap_or(0);
            let start = if last != 0 {
                last + self.prev_slice.1 + 1
            } else {
                self.prev_slice.0
            };
            let out = match self.prev {
                Some(p) if self.prev_slice.1 <= start => Some((p, self.prev_slice.0)),
                _ => None,
            };
            self.prev = Some(m);
            self.prev_slice = (start, m.start);
            if out.is_some() {
                return out;
            }
        }
    }
}

/// `_annotate_tokens`: tokens after both passes. An `Err` (only possible for
/// an empty caller token) ends the stream.
struct Annotated<'p, 's, 't, I> {
    punkt: &'p Punkt,
    words: I,
    cur: Option<Tok<'t>>,
    sc: &'s mut Scratch,
}

impl<'t, I: Iterator<Item = &'t str>> Iterator for Annotated<'_, '_, 't, I> {
    type Item = Result<Tok<'t>, PunktError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut cur = self.cur.take()?;
        if let Some(w) = self.words.next() {
            let next = self.punkt.first_pass(w, &mut self.sc.abbrev);
            if let Err(e) = self.punkt.second_pass(&mut cur, &next, self.sc) {
                return Some(Err(e));
            }
            self.cur = Some(next);
        }
        Some(Ok(cur))
    }
}

impl Punkt {
    /// A tokenizer with these parameters and language variables, for CPython's
    /// Unicode version `version`.
    pub fn new(params: Params, lang: LangVars, version: UnicodeVersion) -> Self {
        Self {
            params,
            lang,
            version,
        }
    }

    /// The parameters.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The language variables.
    pub fn lang_vars(&self) -> &LangVars {
        &self.lang
    }

    /// The Unicode version.
    pub fn version(&self) -> UnicodeVersion {
        self.version
    }

    /// `span_tokenize(text, realign_boundaries)`: the sentences' `(start, end)`
    /// byte offsets. Without realignment these are NLTK's slices as they are:
    /// the last one ends at `len(text.rstrip())`, so it can be empty or have
    /// its start after its end.
    pub fn span_tokenize(&self, text: &str, realign: bool) -> Vec<(usize, usize)> {
        let mut sc = Scratch::default();
        let mut slices = Vec::new();
        let mut last_break = 0;
        for (m, ctx) in self.contexts(text) {
            if self.contains_sentbreak(&text[ctx..m.after_end], &mut sc) {
                slices.push((last_break, m.end));
                last_break = m.next_tok.unwrap_or(m.end);
            }
        }
        slices.push((last_break, text.trim_end_matches(py_space).len()));
        if realign {
            realign_boundaries(text, &slices)
        } else {
            slices
        }
    }

    /// `tokenize(text, realign_boundaries)`: the sentences (a span whose start
    /// is after its end gives `""`, as a Python slice does).
    pub fn tokenize<'t>(&self, text: &'t str, realign: bool) -> Vec<&'t str> {
        self.span_tokenize(text, realign)
            .into_iter()
            .map(|(s, e)| text.get(s..e).unwrap_or(""))
            .collect()
    }

    /// `text_contains_sentbreak(text)`: whether a token other than the last
    /// one ends a sentence.
    pub fn text_contains_sentbreak(&self, text: &str) -> bool {
        self.contains_sentbreak(text, &mut Scratch::default())
    }

    /// `sentences_from_tokens(tokens)`: the number of tokens in each sentence
    /// NLTK's generator yields.
    pub fn sentences_from_tokens<S: AsRef<str>>(
        &self,
        tokens: &[S],
    ) -> Result<Vec<usize>, Raised<usize>> {
        let mut sc = Scratch::default();
        let mut out = Vec::new();
        let mut n = 0;
        for tok in self.annotate(tokens.iter().map(AsRef::as_ref), &mut sc) {
            let tok = tok.map_err(|error| Raised {
                yielded: std::mem::take(&mut out),
                error,
            })?;
            n += 1;
            if tok.sentbreak() {
                out.push(n);
                n = 0;
            }
        }
        if n > 0 {
            out.push(n);
        }
        Ok(out)
    }

    /// `sentences_from_text_legacy(text)`: `_build_sentence_list` over the
    /// tokens of the whole text.
    pub fn sentences_from_text_legacy(&self, text: &str) -> Result<Vec<String>, Raised<String>> {
        let mut sc = Scratch::default();
        let mut out = Vec::new();
        let mut sentence = String::new();
        let mut pos = 0;
        for tok in self.annotate(self.words(text), &mut sc) {
            let tok = tok.map_err(|error| Raised {
                yielded: std::mem::take(&mut out),
                error,
            })?;
            let ws_end = skip_while(text, pos, rx_space);
            let ws = &text[pos..ws_end];
            pos = ws_end;
            let rest = &text[pos..];
            let piece = if rest.starts_with(tok.text) {
                tok.text
            } else if let Some(end) = match_spaced(rest, tok.text) {
                &rest[..end]
            } else {
                return Err(Raised {
                    yielded: out,
                    error: PunktError::AssertionError,
                });
            };
            pos += piece.len();
            if !sentence.is_empty() {
                sentence.push_str(ws);
            }
            sentence.push_str(piece);
            if tok.sentbreak() {
                out.push(std::mem::take(&mut sentence));
            }
        }
        if !sentence.is_empty() {
            out.push(sentence);
        }
        Ok(out)
    }

    /// `debug_decisions(text)`: one record per candidate that
    /// `_match_potential_end_contexts` keeps.
    pub fn debug_decisions(&self, text: &str) -> Result<Vec<Decision>, Raised<Decision>> {
        let mut sc = Scratch::default();
        let mut out = Vec::new();
        for (m, ctx) in self.contexts(text) {
            let context = &text[ctx..m.after_end];
            let mut toks = self
                .words(context)
                .map(|w| self.first_pass(w, &mut sc.abbrev))
                .collect::<Vec<_>>()
                .into_iter()
                .skip_while(|t| !self.lang.ends_with_sent_end(t.text));
            let (Some(mut t1), Some(t2)) = (toks.next(), toks.next()) else {
                return Err(Raised {
                    yielded: out,
                    error: PunktError::IndexError("list index out of range"),
                });
            };
            match self.decision(m.start, context, &mut t1, &t2, &mut sc) {
                Ok(d) => out.push(d),
                Err(error) => {
                    return Err(Raised {
                        yielded: out,
                        error,
                    });
                }
            }
        }
        Ok(out)
    }

    /// The dict `debug_decisions` yields, its values computed in NLTK's order
    /// (the second pass runs for `reason`, after `collocation`).
    fn decision(
        &self,
        period_index: usize,
        context: &str,
        t1: &mut Tok<'_>,
        t2: &Tok<'_>,
        sc: &mut Scratch,
    ) -> Result<Decision, PunktError> {
        let type1 = self.token_type(t1.text, &mut sc.first).to_owned();
        let type2 = self.token_type(t2.text, &mut sc.second).to_owned();
        let key1 = if t1.sentbreak() {
            type_no_period(&type1)
        } else {
            &type1
        };
        let key2 = if t2.sentbreak() {
            type_no_period(&type2)
        } else {
            &type2
        };
        let type2_is_sent_starter = self.params.is_sent_starter(key2);
        let type2_ortho_heuristic = self.ortho_heuristic(t2.text, key2)?;
        let type2_ortho_contexts = self.params.ortho_context(key2) & (ORTHO_UC | ORTHO_LC);
        let collocation = self.params.is_collocation(key1, key2);
        let type1_in_abbrs = t1.abbr;
        let reason = self
            .second_pass(t1, t2, sc)?
            .unwrap_or(Reason::DefaultDecision);
        Ok(Decision {
            period_index,
            text: context.to_owned(),
            type1_is_initial: is_initial(t1.text),
            type1,
            type2,
            type1_in_abbrs,
            type2_is_sent_starter,
            type2_ortho_heuristic,
            type2_ortho_contexts,
            collocation,
            reason,
            break_decision: t1.sentbreak,
        })
    }

    fn contexts<'p, 't>(&'p self, text: &'t str) -> Contexts<'p, 't> {
        Contexts {
            candidates: Candidates {
                lang: &self.lang,
                text,
                pos: 0,
            },
            prev_slice: (0, 0),
            prev: None,
        }
    }

    /// `_tokenize_words(text)`: the word tokens of each line (split on "\n")
    /// that `str.strip()` leaves non-empty.
    fn words<'a>(&'a self, text: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        text.split('\n')
            .filter(|line| !line.chars().all(py_space))
            .flat_map(move |line| Words {
                lang: &self.lang,
                s: line,
                pos: 0,
            })
    }

    fn annotate<'s, 't, I: Iterator<Item = &'t str>>(
        &self,
        mut words: I,
        sc: &'s mut Scratch,
    ) -> Annotated<'_, 's, 't, I> {
        let cur = words.next().map(|w| self.first_pass(w, &mut sc.abbrev));
        Annotated {
            punkt: self,
            words,
            cur,
            sc,
        }
    }

    fn contains_sentbreak(&self, text: &str, sc: &mut Scratch) -> bool {
        let mut found = false;
        for tok in self.annotate(self.words(text), sc) {
            // Word tokens are never empty, so the second pass cannot fail.
            let Ok(tok) = tok else {
                return false;
            };
            if found {
                return true;
            }
            if tok.sentbreak() {
                found = true;
            }
        }
        false
    }

    /// `PunktToken.type`: `tok.lower()` with numbers replaced by `##number##`.
    fn token_type<'b>(&self, tok: &'b str, buf: &'b mut String) -> &'b str {
        let low = lower_into(tok, self.version, buf);
        match numeric(low) {
            Some(false) => NUMBER,
            Some(true) => NUMBER_NL,
            None => low,
        }
    }

    /// `_first_pass_annotation`.
    fn first_pass<'t>(&self, tok: &'t str, buf: &mut String) -> Tok<'t> {
        let mut t = Tok {
            text: tok,
            sentbreak: None,
            abbr: false,
            ellipsis: false,
        };
        if self.lang.is_sent_end_token(tok) {
            t.sentbreak = Some(true);
        } else if is_ellipsis(tok) {
            t.ellipsis = true;
        } else if tok.ends_with('.') && !tok.ends_with("..") {
            if self.is_abbrev(&tok[..tok.len() - 1], buf) {
                t.abbr = true;
            } else {
                t.sentbreak = Some(true);
            }
        }
        t
    }

    /// `tok[:-1].lower()` or its part after the last `-` is an abbreviation type.
    fn is_abbrev(&self, stem: &str, buf: &mut String) -> bool {
        if self.params.abbrev_types.is_empty() {
            return false;
        }
        let low = lower_into(stem, self.version, buf);
        self.params.is_abbrev_type(low)
            || low
                .rfind('-')
                .is_some_and(|i| self.params.is_abbrev_type(&low[i + 1..]))
    }

    /// `_second_pass_annotation(t1, t2)`: updates `t1`, returns the reason.
    fn second_pass(
        &self,
        t1: &mut Tok<'_>,
        t2: &Tok<'_>,
        sc: &mut Scratch,
    ) -> Result<Option<Reason>, PunktError> {
        if !t1.period_final() {
            return Ok(None);
        }
        let Scratch { first, second, .. } = sc;
        let typ = type_no_period(self.token_type(t1.text, first));
        let ty2 = self.token_type(t2.text, second);
        let next_typ = if t2.sentbreak() {
            type_no_period(ty2)
        } else {
            ty2
        };
        let initial = is_initial(t1.text);

        if self.params.is_collocation(typ, next_typ) {
            t1.sentbreak = Some(false);
            t1.abbr = true;
            return Ok(Some(Reason::KnownCollocation));
        }

        if (t1.abbr || t1.ellipsis) && !initial {
            if self.ortho_heuristic(t2.text, next_typ)? == OrthoHeuristic::Yes {
                t1.sentbreak = Some(true);
                return Ok(Some(Reason::AbbrWithOrthographicHeuristic));
            }
            if self.first_upper(t2.text)? && self.params.is_sent_starter(next_typ) {
                t1.sentbreak = Some(true);
                return Ok(Some(Reason::AbbrWithSentenceStarter));
            }
        }

        if initial || typ == NUMBER {
            let h = self.ortho_heuristic(t2.text, next_typ)?;
            if h == OrthoHeuristic::No {
                t1.sentbreak = Some(false);
                t1.abbr = true;
                return Ok(Some(if initial {
                    Reason::InitialWithOrthographicHeuristic
                } else {
                    Reason::NumberWithOrthographicHeuristic
                }));
            }
            if h == OrthoHeuristic::Unknown
                && initial
                && self.first_upper(t2.text)?
                && self.params.ortho_context(next_typ) & ORTHO_LC == 0
            {
                t1.sentbreak = Some(false);
                t1.abbr = true;
                return Ok(Some(Reason::InitialWithSpecialOrthographicHeuristic));
            }
        }
        Ok(None)
    }

    /// `tok[0]`, which raises for an empty token.
    fn first_char(tok: &str) -> Result<char, PunktError> {
        tok.chars()
            .next()
            .ok_or(PunktError::IndexError("string index out of range"))
    }

    /// `first_upper`: `tok[0].isupper()`.
    fn first_upper(&self, tok: &str) -> Result<bool, PunktError> {
        Ok(py_upper(Self::first_char(tok)?, self.version))
    }

    /// `_ortho_heuristic` for a token with text `tok` and
    /// `type_no_sentperiod` `key`.
    fn ortho_heuristic(&self, tok: &str, key: &str) -> Result<OrthoHeuristic, PunktError> {
        if single_char(tok).is_some_and(|c| in_ascii_set(c, PUNCTUATION)) {
            return Ok(OrthoHeuristic::No);
        }
        let ortho = self.params.ortho_context(key);
        let c = Self::first_char(tok)?;
        if py_upper(c, self.version) && ortho & ORTHO_LC != 0 && ortho & ORTHO_MID_UC == 0 {
            return Ok(OrthoHeuristic::Yes);
        }
        if py_lower(c, self.version) && (ortho & ORTHO_UC != 0 || ortho & ORTHO_BEG_LC == 0) {
            return Ok(OrthoHeuristic::No);
        }
        Ok(OrthoHeuristic::Unknown)
    }
}

/// `_realign_boundaries(text, slices)`.
fn realign_boundaries(text: &str, slices: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(slices.len());
    let mut realign = 0;
    for (i, &(start, end)) in slices.iter().enumerate() {
        let start = start + realign;
        let Some(&(next_start, next_end)) = slices.get(i + 1) else {
            if start < end {
                out.push((start, end));
            }
            continue;
        };
        if let Some((keep, skip)) = boundary_realignment(text, next_start, next_end) {
            out.push((start, next_start + keep));
            realign = skip;
        } else {
            realign = 0;
            if start < end {
                out.push((start, end));
            }
        }
    }
    out
}

/// `re_boundary_realignment.match(text[start:end])`
/// (`["')\]}‘’“”«»]+?(?:\s+|(?=--)|$)`, MULTILINE): the length of the
/// closing punctuation (`len(m.group(0).rstrip())`) and `m.end()`, in bytes.
fn boundary_realignment(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    let sub = text.get(start..end)?;
    let quotes = sub
        .char_indices()
        .find(|&(_, c)| !is_closing(c))
        .map_or(sub.len(), |(i, _)| i);
    if quotes == 0 {
        return None;
    }
    let rest = &sub[quotes..];
    match rest.chars().next() {
        None => Some((quotes, quotes)),
        Some(c) if rx_space(c) => Some((quotes, skip_while(sub, quotes, rx_space))),
        Some(_) if rest.starts_with("--") => Some((quotes, quotes)),
        Some(_) => None,
    }
}

/// `re.compile(r"\s*".join(re.escape(c) for c in tok)).match(rest)`: the
/// match's end, trying longer whitespace runs first as the backtracking
/// engine does.
fn match_spaced(rest: &str, tok: &str) -> Option<usize> {
    fn go(
        rest: &str,
        chars: &[char],
        pos: usize,
        failed: &mut FxHashSet<(usize, usize)>,
    ) -> Option<usize> {
        let Some((&c, tail)) = chars.split_first() else {
            return Some(pos);
        };
        if failed.contains(&(chars.len(), pos)) {
            return None;
        }
        let mut ends = vec![pos];
        let mut q = pos;
        while let Some(s) = char_at(rest, q).filter(|&s| rx_space(s)) {
            q += s.len_utf8();
            ends.push(q);
        }
        for &e in ends.iter().rev() {
            if char_at(rest, e) == Some(c) {
                if let Some(r) = go(rest, tail, e + c.len_utf8(), failed) {
                    return Some(r);
                }
            }
        }
        failed.insert((chars.len(), pos));
        None
    }
    let chars: Vec<char> = tok.chars().collect();
    let (&first, tail) = chars.split_first()?;
    if char_at(rest, 0) != Some(first) {
        return None;
    }
    go(rest, tail, first.len_utf8(), &mut FxHashSet::default())
}

/// Turn byte offsets into `text` into code point offsets (Python indices), in
/// one pass over the text. The offsets may come in any order; each must be at
/// most `text.len()` and on a character boundary.
pub fn char_offsets(text: &str, offsets: &mut [usize]) {
    let bytes = text.as_bytes();
    let Some(&max) = offsets.iter().max() else {
        return;
    };
    if bytes[..max].is_ascii() {
        return;
    }
    let count = |a: usize, b: usize| {
        debug_assert!(text.is_char_boundary(b));
        bytes[a..b].iter().filter(|&&x| x & 0xc0 != 0x80).count()
    };
    if offsets.is_sorted() {
        let (mut byte, mut chars) = (0, 0);
        for o in offsets.iter_mut() {
            chars += count(byte, *o);
            byte = *o;
            *o = chars;
        }
    } else {
        let mut order: Vec<usize> = (0..offsets.len()).collect();
        order.sort_unstable_by_key(|&i| offsets[i]);
        let (mut byte, mut chars) = (0, 0);
        for i in order {
            let o = offsets[i];
            chars += count(byte, o);
            byte = o;
            offsets[i] = chars;
        }
    }
}

#[cfg(test)]
#[path = "testdata/punkt/replay.rs"]
mod replay;

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    fn default_punkt() -> Punkt {
        Punkt::new(Params::default(), LangVars::default(), V)
    }

    fn punkt_with(params: Params) -> Punkt {
        Punkt::new(params, LangVars::default(), V)
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    fn words<'a>(p: &'a Punkt, line: &'a str) -> Vec<&'a str> {
        Words {
            lang: &p.lang,
            s: line,
            pos: 0,
        }
        .collect()
    }

    #[test]
    fn regression_section() {
        // tokenize.doctest, "Regression Tests: PunktSentenceTokenizer".
        let p = default_punkt();
        assert_eq!(
            p.tokenize("See Section 3).  Or Section 2).  ", true),
            ["See Section 3).", "Or Section 2)."]
        );
        assert_eq!(
            p.tokenize("See Section 3.)  Or Section 2.)  ", true),
            ["See Section 3.)", "Or Section 2.)"]
        );
        assert_eq!(
            p.tokenize("See Section 3.)  Or Section 2.)  ", false),
            ["See Section 3.", ")  Or Section 2.", ")"]
        );
        assert_eq!(
            p.tokenize(
                ". This input starts with a dot. This used to cause issues.",
                true
            ),
            [
                ".",
                "This input starts with a dot.",
                "This used to cause issues."
            ]
        );
    }

    #[test]
    fn module_docstring_realign() {
        // punkt.py's module docstring (its default model gives the same here).
        let p = default_punkt();
        let text = "(How does it deal with this parenthesis?)  \"It should be part of the\n\
                    previous sentence.\" \"(And the same with this one.)\" ('And this one!')\n\
                    \"('(And (this)) '?)\" [(and this. )]";
        assert_eq!(
            p.tokenize(text, true),
            [
                "(How does it deal with this parenthesis?)",
                "\"It should be part of the\nprevious sentence.\"",
                "\"(And the same with this one.)\"",
                "('And this one!')",
                "\"('(And (this)) '?)\"",
                "[(and this. )]",
            ]
        );
        assert_eq!(
            p.tokenize(text, false),
            [
                "(How does it deal with this parenthesis?",
                ")  \"It should be part of the\nprevious sentence.",
                "\" \"(And the same with this one.",
                ")\" ('And this one!",
                "')\n\"('(And (this)) '?",
                ")\" [(and this.",
                ")]",
            ]
        );
    }

    #[test]
    fn match_potential_end_contexts_doctest() {
        // `_match_potential_end_contexts`' doctest: one context, at the last "!".
        let p = default_punkt();
        let text = "Very bad acting!!! I promise.";
        let got: Vec<_> = p
            .contexts(text)
            .map(|(m, ctx)| (m.start, m.end, &text[ctx..m.after_end]))
            .collect();
        assert_eq!(got, [(17, 18, "acting!!! I")]);
        let starts: Vec<_> = p.contexts("a").map(|(m, _)| m.start).collect();
        assert_eq!(starts, [] as [usize; 0]);
    }

    #[test]
    fn bengali_lang_vars() {
        // test_tokenize.py: test_punkt_tokenize_custom_lang_vars / no_custom_lang_vars.
        let text = "উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন। অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  উপস্থিত ছিলেন। এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও পরিকল্পনার নথি প্রকাশ করেছেন।";
        let bengali = Punkt::new(
            Params::default(),
            LangVars::new(vec!['.', '?', '!', '\u{964}']),
            V,
        );
        assert_eq!(
            bengali.tokenize(text, true),
            [
                "উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন।",
                "অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  উপস্থিত ছিলেন।",
                "এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও পরিকল্পনার নথি প্রকাশ করেছেন।",
            ]
        );
        assert_eq!(default_punkt().tokenize(text, true), [text]);
    }

    #[test]
    fn debug_decision_counts() {
        // test_tokenize.py: punkt_debug_decisions and its custom end character.
        let p = default_punkt();
        for (text, sents, splits) in [
            ("Subject: Some subject. Attachments: Some attachments", 2, 1),
            ("Subject: Some subject! Attachments: Some attachments", 2, 1),
            ("This is just a normal sentence, just like any other.", 1, 0),
        ] {
            assert_eq!(p.tokenize(text, true).len(), sents, "{text}");
            assert_eq!(p.debug_decisions(text).unwrap().len(), splits, "{text}");
        }
        let caret = Punkt::new(
            Params::default(),
            LangVars::new(vec!['.', '?', '!', '^']),
            V,
        );
        let text = "Subject: Some subject^ Attachments: Some attachments";
        assert_eq!(caret.tokenize(text, true).len(), 2);
        assert_eq!(caret.debug_decisions(text).unwrap().len(), 1);
    }

    #[test]
    fn sent_tokenize_cases_without_model() {
        // test_tokenize.py: test_sent_tokenize and the issue 2333 cases. They
        // use the English model; with empty parameters these give the same.
        let p = default_punkt();
        let cases: [(&str, &[&str]); 12] = [
            (
                "this is a test. . new sentence.",
                &["this is a test.", ".", "new sentence."],
            ),
            ("This. . . That", &["This.", ".", ".", "That"]),
            ("This... That", &["This... That"]),
            ("This.. . That", &["This.. .", "That"]),
            ("This. .. That", &["This.", ".. That"]),
            ("This. ,. That", &["This.", ",.", "That"]),
            ("This!!! That", &["This!!!", "That"]),
            ("This! That", &["This!", "That"]),
            ("Hello.\tThere", &["Hello.", "There"]),
            (
                "“First sentence.” Next one.",
                &["“First sentence.”", "Next one."],
            ),
            (
                "Il a dit «bonjour.» Puis il est parti.",
                &["Il a dit «bonjour.»", "Puis il est parti."],
            ),
            (
                "He said \"hi.\" Then left.",
                &["He said \"hi.\"", "Then left."],
            ),
        ];
        for (text, want) in cases {
            assert_eq!(p.tokenize(text, true), want, "{text:?}");
        }
    }

    #[test]
    fn word_tokenizer() {
        let p = default_punkt();
        assert_eq!(
            words(
                &p,
                "Mr.Smith, (x) a--b ... . . . x. . .y \u{1c}q J. S. Bach. pre-Dr. 3.5. -1. a,b a, b a,"
            ),
            [
                "Mr.Smith", ",", "(", "x", ")", "a", "--", "b", "...", ". . .", "x", ". . .", "y",
                "\u{1c}q", "J.", "S.", "Bach.", "pre-Dr.", "3.5.", "-", "1.", "a,b", "a", ",", "b",
                "a", ","
            ]
        );
        // `(?:\.\s){2,}\.` gives back the last pair when no dot follows.
        assert_eq!(words(&p, ". . . x"), [". . .", "x"]);
        assert_eq!(words(&p, ". . x"), [".", ".", "x"]);
        assert_eq!(words(&p, ".\u{3000}.\u{85}.y"), [".\u{3000}.\u{85}.", "y"]);
        // U+001C is `\S` for `regex`.
        assert_eq!(words(&p, "a.\u{1c}b"), ["a.\u{1c}b"]);
        assert_eq!(words(&p, "x.'abc"), ["x.", "'abc"]);
    }

    #[test]
    fn token_properties() {
        let p = default_punkt();
        let mut buf = String::new();
        assert_eq!(p.token_type("3.5.", &mut buf), NUMBER);
        assert_eq!(p.token_type("-.5", &mut buf), NUMBER);
        assert_eq!(p.token_type("12\n", &mut buf), NUMBER_NL);
        assert_eq!(p.token_type("--5", &mut buf), "--5");
        assert_eq!(p.token_type("\u{663}", &mut buf), NUMBER);
        assert_eq!(p.token_type("\u{b2}", &mut buf), "\u{b2}");
        assert_eq!(p.token_type("İ.", &mut buf), "i\u{307}.");
        assert_eq!(p.token_type("ΟΔΟΣ.", &mut buf), "οδος.");
        assert!(
            is_ellipsis("..") && is_ellipsis("...\n") && !is_ellipsis(".") && !is_ellipsis("a..")
        );
        assert!(is_initial("J.") && is_initial("é.") && !is_initial("1.") && !is_initial("JS."));
        assert!(is_initial("\u{301}.") && !is_initial("\u{b2}."));
        assert_eq!(type_no_period("."), ".");
        assert_eq!(type_no_period("a."), "a");
    }

    #[test]
    fn abbreviations_and_collocations() {
        let params = Params::new(
            strings(&["dr", "e.g"]),
            [("s".to_owned(), "bach".to_owned())],
            strings(&["the"]),
            [
                ("bach".to_owned(), ORTHO_MID_UC),
                ("the".to_owned(), ORTHO_BEG_UC | ORTHO_MID_LC),
            ],
        );
        let p = punkt_with(params);
        assert_eq!(
            p.tokenize("Ask Dr. Smith. He knows.", true),
            ["Ask Dr. Smith.", "He knows."]
        );
        assert_eq!(
            p.tokenize("See pre-Dr. Smith now.", true),
            ["See pre-Dr. Smith now."]
        );
        assert_eq!(
            p.tokenize("Johann S. Bach. The end.", true),
            ["Johann S. Bach.", "The end."]
        );
        // An abbreviation before a frequent sentence starter ends a sentence.
        assert_eq!(
            p.tokenize("Ask the dr. The end.", true),
            ["Ask the dr.", "The end."]
        );
        let d = p.debug_decisions("Ask the dr. The end.").unwrap();
        assert_eq!(d[0].reason, Reason::AbbrWithOrthographicHeuristic);
        assert_eq!(d[0].type2_ortho_contexts, ORTHO_BEG_UC | ORTHO_MID_LC);
        assert_eq!(d[0].break_decision, Some(true));
        let d = p.debug_decisions("Johann S. Bach. The end.").unwrap();
        assert_eq!(d[0].reason, Reason::KnownCollocation);
        assert!(d[0].collocation && d[0].type1_is_initial);
        assert_eq!(d[0].break_decision, Some(false));
    }

    #[test]
    fn edge_errors() {
        let p = default_punkt();
        let e = p.debug_decisions("a.\n\u{1c}").unwrap_err();
        assert_eq!(e.error, PunktError::IndexError("list index out of range"));
        assert_eq!(e.yielded, []);
        assert_eq!(p.sentences_from_tokens(&["Mr.", ""]), Ok(vec![1, 1]));
        let abbr = punkt_with(Params::new(strings(&["mr"]), [], [], []));
        assert_eq!(
            abbr.sentences_from_tokens(&["Hello.", "b", "Mr.", ""]),
            Err(Raised {
                yielded: vec![1],
                error: PunktError::IndexError("string index out of range"),
            })
        );
        assert_eq!(
            p.sentences_from_text_legacy("a.\n\u{1c}\nb"),
            Err(Raised {
                yielded: vec![],
                error: PunktError::AssertionError,
            })
        );
        assert_eq!(p.span_tokenize("Hello. \u{1c}", false), [(0, 6), (7, 6)]);
        assert_eq!(p.span_tokenize("Hello. \u{1c}", true), [(0, 6)]);
        assert_eq!(p.tokenize("Hello. \u{1c}", false), ["Hello.", ""]);
        assert_eq!(p.span_tokenize("", true), []);
        assert_eq!(p.span_tokenize("", false), [(0, 0)]);
        let none = Punkt::new(Params::default(), LangVars::new(vec![]), V);
        assert_eq!(none.span_tokenize("a. b", true), [(0, 4)]);
    }

    #[test]
    fn legacy_matches_tokens() {
        let p = default_punkt();
        assert_eq!(
            p.sentences_from_text_legacy("One.  Two!\n\nThree?")
                .unwrap(),
            ["One.", "Two!", "Three?"]
        );
        assert_eq!(match_spaced(". \n .x", ". ."), Some(5));
        assert_eq!(match_spaced("a b", "ab"), Some(3));
        assert_eq!(match_spaced("ba", "ab"), None);
    }

    #[test]
    fn text_contains_sentbreak() {
        let p = default_punkt();
        assert!(p.text_contains_sentbreak("One. Two"));
        assert!(!p.text_contains_sentbreak("One."));
        assert!(!p.text_contains_sentbreak("One two"));
        assert!(p.text_contains_sentbreak("acting!!! I"));
    }

    #[test]
    fn punkt_tab_parsing() {
        let p = Params::from_punkt_tab(
            "a\tb\r\nc\td\te\n\nf\tg",
            "the\nbut\n",
            "dr\r\nmr\rx",
            "the\t 1_0 \n\u{663}\t\u{663}\nbig\t18446744073709551619\n",
        )
        .unwrap();
        assert!(
            p.is_collocation("a", "b") && p.is_collocation("f", "g") && !p.is_collocation("c", "d")
        );
        assert!(p.is_sent_starter("the") && p.is_sent_starter("but") && !p.is_sent_starter(""));
        assert!(p.is_abbrev_type("dr") && p.is_abbrev_type("mr") && p.is_abbrev_type("x"));
        assert_eq!(p.ortho_context("the"), 10);
        assert_eq!(p.ortho_context("\u{663}"), 3);
        assert_eq!(p.ortho_context("big"), 3);
        assert_eq!(p.ortho_context("missing"), 0);
        let e = Params::from_punkt_tab("", "", "", "a\t1\nb\n").unwrap_err();
        assert_eq!((e.file, e.line), ("ortho_context.tab", 2));
        assert!(Params::from_punkt_tab("", "", "", "a\tx").is_err());
        assert!(Params::from_punkt_tab("", "", "", "a\t1_").is_err());
        assert!(Params::from_punkt_tab("", "", "", "a\t1\t2").is_err());
        assert!(Params::from_punkt_tab("", "", "", "").is_ok());
        let blank = Params::from_punkt_tab("", "\n", "", "").unwrap();
        assert!(blank.is_sent_starter(""));
        assert_eq!(py_int("-12"), Some(-12));
        assert_eq!(py_int("+\u{ff11}"), Some(1));
        assert_eq!(py_int("- 1"), None);
        assert_eq!(py_int("1__0"), None);
    }

    #[test]
    fn offsets() {
        let text = "aé b\u{1f600}c";
        let mut o = vec![0, 1, 3, 4, 5, 9, 10];
        char_offsets(text, &mut o);
        assert_eq!(o, [0, 1, 2, 3, 4, 5, 6]);
        let mut o = vec![10, 3, 0, 9, 3];
        char_offsets(text, &mut o);
        assert_eq!(o, [6, 2, 0, 5, 2]);
        let mut o = vec![2, 1];
        char_offsets("abc", &mut o);
        assert_eq!(o, [2, 1]);
    }

    #[test]
    fn send_sync() {
        fn check<T: Send + Sync>() {}
        check::<Punkt>();
    }
}
