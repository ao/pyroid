// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! What NeMo's heuristic filters share (`nemo_curator.stages.text.utils`): the word
//! splitters (`get_word_splitter`), `get_sentences`, `get_paragraphs`, and the Python `str`
//! operations the filters apply, with CPython's semantics.

use crate::filters::text::py_strip;
use crate::jieba::{Dictionary, ForceSplit};

/// Why NeMo's `score_document` raises for a document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreError {
    /// `ZeroDivisionError`: a ratio over zero words, lines or characters.
    ZeroDivision,
    /// `ValueError` from `max()` over nothing (no words, or no n-grams).
    EmptyMax,
}

impl ScoreError {
    /// The name of the Python exception NeMo raises.
    pub fn python_type(self) -> &'static str {
        match self {
            Self::ZeroDivision => "ZeroDivisionError",
            Self::EmptyMax => "ValueError",
        }
    }

    /// CPython 3.12's message for it.
    pub fn message(self) -> &'static str {
        match self {
            Self::ZeroDivision => "division by zero",
            Self::EmptyMax => "max() iterable argument is empty",
        }
    }
}

impl std::fmt::Display for ScoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.python_type(), self.message())
    }
}

impl std::error::Error for ScoreError {}

/// `a / b` for Python ints below 2**53: CPython's true division is then exactly
/// `float(a) / float(b)`.
#[inline]
#[allow(clippy::cast_precision_loss)] // counts of a document's characters, far below 2**53
pub(crate) fn ratio(a: usize, b: usize) -> f64 {
    a as f64 / b as f64
}

/// NeMo's `get_word_splitter(lang)`, for the languages pyroid ports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Splitter {
    /// `text.split()`: every language but `"zh"` and `"ja"`.
    Whitespace,
    /// `list(jieba.cut(text))` with jieba 0.42.1's default dictionary and HMM (`"zh"`).
    Jieba,
}

impl Splitter {
    /// The splitter for `lang`, already lowercased with Python's `str.lower()` as
    /// `get_word_splitter` does. `None` for `"ja"` (MeCab), which pyroid does not port.
    pub fn for_lang(lowered: &str) -> Option<Self> {
        match lowered {
            "zh" => Some(Self::Jieba),
            "ja" => None,
            _ => Some(Self::Whitespace),
        }
    }

    /// The words of `text`, in order (`self(text)` in NeMo).
    pub fn words(self, text: &str) -> Words<'_> {
        match self {
            Self::Whitespace => Words::Split(split(text)),
            Self::Jieba => {
                let mut out = Vec::new();
                Dictionary::default_ref()
                    .cut(text, true, &ForceSplit::default(), &mut out)
                    .expect("jieba's default dictionary has a positive total and frequencies");
                Words::List(out.into_iter())
            }
        }
    }
}

/// The words of one document (see [`Splitter::words`]).
#[derive(Debug)]
pub enum Words<'a> {
    /// `text.split()`.
    Split(Split<'a>),
    /// jieba's words.
    List(std::vec::IntoIter<&'a str>),
}

impl<'a> Iterator for Words<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        match self {
            Self::Split(s) => s.next(),
            Self::List(l) => l.next(),
        }
    }
}

/// Byte length of the `str.isspace()` character starting at byte `i` of `b` (valid
/// UTF-8, `i` any byte position), or 0. A continuation byte is never a whitespace start.
#[inline]
fn space_len(b: &[u8], i: usize) -> usize {
    let c = b[i];
    if c < 0x80 {
        return usize::from(matches!(c, b'\t'..=b'\r' | 0x1c..=b' '));
    }
    // U+0085, U+00A0, U+1680, U+2000..=U+200A, U+2028, U+2029, U+202F, U+205F, U+3000.
    match c {
        0xC2 if matches!(b[i + 1], 0x85 | 0xA0) => 2,
        0xE1 if b[i + 1] == 0x9A && b[i + 2] == 0x80 => 3,
        0xE2 if matches!(
            (b[i + 1], b[i + 2]),
            (0x80, 0x80..=0x8A | 0xA8 | 0xA9 | 0xAF) | (0x81, 0x9F)
        ) =>
        {
            3
        }
        0xE3 if b[i + 1] == 0x80 && b[i + 2] == 0x80 => 3,
        _ => 0,
    }
}

/// `s.split()`: the runs of non-whitespace (`str.isspace()`), in order.
pub fn split(s: &str) -> Split<'_> {
    Split { s, pos: 0 }
}

/// Iterator of [`split`].
#[derive(Clone, Debug)]
pub struct Split<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Iterator for Split<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let b = self.s.as_bytes();
        let mut i = self.pos;
        loop {
            if i >= b.len() {
                self.pos = i;
                return None;
            }
            match space_len(b, i) {
                0 => break,
                n => i += n,
            }
        }
        let start = i;
        while i < b.len() && space_len(b, i) == 0 {
            i += 1;
        }
        self.pos = i;
        // `start` and `i` are character boundaries: whitespace starts, or the ends.
        Some(&self.s[start..i])
    }
}

/// `get_sentences(text)`: the pieces of `text.split("\n")` that are not blank, unstripped.
pub fn sentences(text: &str) -> impl Iterator<Item = &str> {
    text.split('\n').filter(|x| !py_strip(x).is_empty())
}

/// `get_paragraphs(text)`: `text.split("\n\n")`.
pub fn paragraphs(text: &str) -> std::str::Split<'_, &'static str> {
    text.split("\n\n")
}

/// The byte that stands for characters a needle can never match.
pub(crate) const BARRIER: u8 = 0xFF;

/// `s.lower()` as NeMo's lowercase needles (ASCII, `…` and `»`) see it: ASCII letters
/// lowercased, U+212A KELVIN SIGN as `k`, U+0130 as `i` and a barrier (its `lower()` is
/// `i` + U+0307), `…` (U+2026) and `»` (U+00BB) kept as UTF-8, and every other non-ASCII
/// character as one [`BARRIER`] byte. These are the only characters whose `lower()`
/// contains ASCII, `…` or `»` (`test_nemo.py` checks this on every interpreter), so a
/// needle is in (or ends) `s.lower()` exactly when it is in (or ends) this projection.
pub(crate) fn project_lower(s: &str, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(s.len());
    if s.is_ascii() {
        out.extend(s.bytes().map(|b| b.to_ascii_lowercase()));
        return;
    }
    let mut buf = [0u8; 4];
    for c in s.chars() {
        match c {
            _ if c.is_ascii() => out.push((c as u8).to_ascii_lowercase()),
            '\u{212A}' => out.push(b'k'),
            '\u{130}' => out.extend_from_slice(&[b'i', BARRIER]),
            '\u{2026}' | '\u{BB}' => out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes()),
            _ => out.push(BARRIER),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::is_py_whitespace;

    #[test]
    fn space_len_is_isspace() {
        let mut buf = [0u8; 8];
        for c in (0..=0x10_FFFF).filter_map(char::from_u32) {
            let n = c.encode_utf8(&mut buf).len();
            let want = if is_py_whitespace(c) { n } else { 0 };
            assert_eq!(space_len(&buf[..n + 2], 0), want, "{c:?}");
            // No continuation byte starts a whitespace character.
            for i in 1..n {
                assert_eq!(space_len(&buf[..n + 2], i), 0, "{c:?} at {i}");
            }
        }
    }

    #[test]
    fn split_is_python_split() {
        let cases = [
            ("", vec![]),
            (" \t\n ", vec![]),
            ("a", vec!["a"]),
            (
                "  a  b\u{3000}c\u{85}d\u{a0}e\u{200b}f ",
                vec!["a", "b", "c", "d", "e\u{200b}f"],
            ),
            (
                "x\u{2028}y\u{1680}z\u{205f}\u{202f}w",
                vec!["x", "y", "z", "w"],
            ),
            ("\u{1c}\u{1d}\u{1e}\u{1f}é", vec!["é"]),
        ];
        for (s, want) in cases {
            assert_eq!(split(s).collect::<Vec<_>>(), want, "{s:?}");
        }
    }

    #[test]
    fn sentences_and_paragraphs() {
        let t = "a\n \n\nb \n\t\nc";
        assert_eq!(sentences(t).collect::<Vec<_>>(), ["a", "b ", "c"]);
        assert_eq!(sentences("").count(), 0);
        assert_eq!(paragraphs("a\n\n\nb").collect::<Vec<_>>(), ["a", "\nb"]);
        assert_eq!(paragraphs("").collect::<Vec<_>>(), [""]);
        assert_eq!(paragraphs("\n\n\n\n").collect::<Vec<_>>(), ["", "", ""]);
    }

    #[test]
    fn projection() {
        let mut out = Vec::new();
        project_lower("AbC \u{212A}\u{130}x…»é", &mut out);
        let mut want = b"abc ki\xFFx".to_vec();
        want.extend_from_slice("…»".as_bytes());
        want.push(BARRIER);
        assert_eq!(out, want);
    }

    #[test]
    fn splitters() {
        assert_eq!(Splitter::for_lang("zh"), Some(Splitter::Jieba));
        assert_eq!(Splitter::for_lang("ja"), None);
        assert_eq!(Splitter::for_lang("en"), Some(Splitter::Whitespace));
        assert_eq!(Splitter::for_lang(""), Some(Splitter::Whitespace));
        let words: Vec<&str> = Splitter::Jieba.words("我来到北京清华大学 ok").collect();
        assert_eq!(words, ["我", "来到", "北京", "清华大学", " ", "ok"]);
    }
}
