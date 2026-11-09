// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! What NeMo's heuristic filters share (`nemo_curator.stages.text.utils`): the word
//! splitters (`get_word_splitter`), `get_sentences`, `get_paragraphs`, and the Python `str`
//! operations the filters apply, with CPython's semantics.

use std::ops::ControlFlow;

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

/// The number of bytes of `b` for which `pred` holds. Counts 255 bytes at a time in `u8`
/// lanes, which the compiler vectorizes for a branch-free `pred`.
#[inline]
pub(crate) fn count_bytes(b: &[u8], pred: impl Fn(u8) -> bool) -> usize {
    b.chunks(255)
        .map(|chunk| {
            usize::from(
                chunk
                    .iter()
                    .fold(0u8, |acc, &c| acc.wrapping_add(u8::from(pred(c)))),
            )
        })
        .sum()
}

/// A UTF-8 continuation byte (`0b10xx_xxxx`).
#[inline]
pub(crate) fn is_continuation(c: u8) -> bool {
    (c & 0xC0) == 0x80
}

/// `len(s)`: the number of code points (bytes that are not continuation bytes).
#[inline]
pub(crate) fn char_len(s: &str) -> usize {
    let b = s.as_bytes();
    b.len() - count_bytes(b, is_continuation)
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
    /// `get_word_splitter(lang)`, which compares `lang.lower()` with `"zh"` and `"ja"`.
    /// `None` for `"ja"` (MeCab), which pyroid does not port.
    ///
    /// The only non-ASCII characters whose `lower()` holds ASCII are U+0130 (`"i̇"`) and
    /// U+212A (`"k"`), so `lang.lower() == "zh"` exactly when `lang` is `"zh"` in any
    /// ASCII case.
    pub fn for_lang(lang: &str) -> Option<Self> {
        if lang.eq_ignore_ascii_case("zh") {
            Some(Self::Jieba)
        } else if lang.eq_ignore_ascii_case("ja") {
            None
        } else {
            Some(Self::Whitespace)
        }
    }

    /// The words NeMo's filters see for a document: `splitter(text.strip())`.
    pub fn doc_words(self, text: &str) -> Words<'_> {
        match self {
            // `text.strip().split()` is `text.split()`.
            Self::Whitespace => Words::Split(split(text)),
            Self::Jieba => self.words(py_strip(text)),
        }
    }

    /// Each of [`Self::doc_words`] with its length in code points, in order.
    #[inline]
    pub fn for_each_word<'a>(self, text: &'a str, mut f: impl FnMut(&'a str, usize)) {
        match self {
            Self::Whitespace => for_each_word_span(text, |a, b, chars| {
                f(&text[a..b], chars);
                ControlFlow::Continue(())
            }),
            Self::Jieba => {
                for w in self.doc_words(text) {
                    f(w, char_len(w));
                }
            }
        }
    }

    /// The words of `text`, in order (`splitter(text)` in NeMo).
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

/// What a byte can be for [`split`]: part of a word, an ASCII whitespace character, or the
/// lead byte of a character that may be whitespace (U+0085 and U+00A0 start with C2,
/// U+1680 with E1, U+2000..=U+205F with E2, U+3000 with E3).
const WORD: u8 = 0;
const SPACE: u8 = 1;
const MAYBE: u8 = 2;

#[allow(clippy::cast_possible_truncation)] // c < 256
static SPACE_CLASS: [u8; 256] = {
    let mut t = [WORD; 256];
    let mut c = 0;
    while c < 256 {
        t[c] = match c as u8 {
            b'\t'..=b'\r' | 0x1c..=b' ' => SPACE,
            0xC2 | 0xE1 | 0xE2 | 0xE3 => MAYBE,
            _ => WORD,
        };
        c += 1;
    }
    t
};

/// Byte length of the whitespace character whose lead byte (C2, E1, E2 or E3) is at `i`,
/// or 0 if that character is not whitespace. `b` is valid UTF-8.
#[inline]
fn multi_space_len(b: &[u8], i: usize) -> usize {
    // U+0085, U+00A0, U+1680, U+2000..=U+200A, U+2028, U+2029, U+202F, U+205F, U+3000.
    match b[i] {
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

/// Byte length of the `str.isspace()` character starting at byte `i` of `b` (valid
/// UTF-8, `i` any byte position), or 0. A continuation byte is never a whitespace start.
#[cfg(test)]
fn space_len(b: &[u8], i: usize) -> usize {
    match SPACE_CLASS[usize::from(b[i])] {
        WORD => 0,
        SPACE => 1,
        _ => multi_space_len(b, i),
    }
}

// ---- 64-byte blocks as bit masks (SWAR: eight bytes per u64, no `unsafe`) ----------------

const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;
const LOW7: u64 = 0x7F7F_7F7F_7F7F_7F7F;

/// 0x80 in each byte of `x` whose value is below `k` (`k <= 0x80`), exactly: no carry
/// crosses a byte.
#[inline]
const fn bytes_below(x: u64, k: u8) -> u64 {
    !(((x & LOW7).wrapping_add(LO * (0x80 - k as u64))) | x) & HI
}

/// 0x80 in each byte of `x` equal to `v`, exactly.
#[inline]
const fn bytes_equal(x: u64, v: u8) -> u64 {
    let y = x ^ (LO * v as u64);
    !(((y & LOW7).wrapping_add(LOW7)) | y) & HI
}

/// The eight 0x80 flags of `m` as the bits of a byte (byte `j` to bit `j`).
#[inline]
fn flags_to_bits(m: u64) -> u64 {
    (m >> 7).wrapping_mul(0x0102_0408_1020_4080) >> 56
}

/// Bit masks of 64 bytes: ASCII whitespace, possible lead bytes of multi-byte whitespace
/// (C2, E1, E2, E3), and UTF-8 continuation bytes.
#[inline]
fn classify(block: &[u8; 64]) -> (u64, u64, u64) {
    let (mut space, mut maybe, mut cont) = (0u64, 0u64, 0u64);
    for (i, chunk) in block.chunks_exact(8).enumerate() {
        let x = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        // \t..=\r and \x1c..=' '.
        let sp = (bytes_below(x, 0x0E) & !bytes_below(x, 0x09))
            | (bytes_below(x, 0x21) & !bytes_below(x, 0x1C));
        space |= flags_to_bits(sp) << (8 * i);
        if x & HI != 0 {
            let mb = bytes_equal(x, 0xC2)
                | bytes_equal(x, 0xE1)
                | bytes_equal(x, 0xE2)
                | bytes_equal(x, 0xE3);
            maybe |= flags_to_bits(mb) << (8 * i);
            // 0b10xx_xxxx: bit 7 set, bit 6 (moved to bit 7 by the shift) clear.
            cont |= flags_to_bits(x & !(x << 1) & HI) << (8 * i);
        }
    }
    (space, maybe, cont)
}

/// One 64-byte block of a str (see [`for_each_block`]).
#[derive(Clone, Copy, Debug)]
struct Block {
    /// Bytes of whitespace characters (`str.isspace()`), multi-byte ones included.
    space: u64,
    /// UTF-8 continuation bytes.
    cont: u64,
    /// Bytes that exist (the last block may be short).
    valid: u64,
}

/// `f(base, block)` for each 64-byte block of `s` from byte `base`, until `f` breaks.
fn for_each_block(s: &str, mut f: impl FnMut(usize, Block) -> ControlFlow<()>) {
    let bytes = s.as_bytes();
    let mut carry = 0u64; // whitespace bytes that run into the next block
    let mut base = 0;
    while base < bytes.len() {
        let len = (bytes.len() - base).min(64);
        let (space, maybe, cont) = if let Some(block) = bytes[base..].first_chunk::<64>() {
            classify(block)
        } else {
            // The short last block, padded with spaces.
            let mut block = [b' '; 64];
            block[..len].copy_from_slice(&bytes[base..]);
            classify(&block)
        };
        let valid = if len == 64 {
            u64::MAX
        } else {
            (1u64 << len) - 1
        };
        let mut space = space | carry;
        carry = 0;
        let mut leads = maybe & valid;
        while leads != 0 {
            let at = leads.trailing_zeros() as usize;
            leads &= leads - 1;
            let width = multi_space_len(bytes, base + at);
            if width > 0 {
                let bits = ((1u128 << width) - 1) << at;
                #[allow(clippy::cast_possible_truncation)] // the two halves of 128 bits
                {
                    space |= bits as u64;
                    carry |= (bits >> 64) as u64;
                }
            }
        }
        if f(base, Block { space, cont, valid }).is_break() {
            return;
        }
        base += len;
    }
}

/// `len(s.split())`: the number of words, counted from the blocks' word starts.
pub(crate) fn count_words(s: &str) -> usize {
    let (mut n, mut prev) = (0usize, 0u64);
    for_each_block(s, |_, blk| {
        let word = !blk.space & blk.valid;
        n += (word & !((word << 1) | prev)).count_ones() as usize;
        prev = word >> 63;
        ControlFlow::Continue(())
    });
    n
}

/// `(len(s.split()), sum(len(w) for w in s.split()))`: every non-whitespace character is in
/// exactly one word.
pub(crate) fn count_words_and_chars(s: &str) -> (usize, usize) {
    let (mut n, mut chars, mut prev) = (0usize, 0usize, 0u64);
    for_each_block(s, |_, blk| {
        let word = !blk.space & blk.valid;
        n += (word & !((word << 1) | prev)).count_ones() as usize;
        chars += (word & !blk.cont).count_ones() as usize;
        prev = word >> 63;
        ControlFlow::Continue(())
    });
    (n, chars)
}

/// Each word of `s.split()` in order: `f(start, end, chars)` with its byte range and length
/// in code points, until `f` breaks.
pub(crate) fn for_each_word_span(
    s: &str,
    mut f: impl FnMut(usize, usize, usize) -> ControlFlow<()>,
) {
    // The open word's start and the continuation bytes before it.
    let mut open = (0usize, 0usize);
    let (mut cont_before, mut prev) = (0usize, 0u64);
    let mut stopped = false;
    for_each_block(s, |base, blk| {
        let word = !blk.space & blk.valid;
        let shifted = (word << 1) | prev;
        let starts = word & !shifted;
        // Bytes right after a word that are not word bytes (padding included).
        let ends = !word & shifted;
        let mut edges = starts | ends;
        while edges != 0 {
            let k = edges.trailing_zeros() as usize;
            let bit = 1u64 << k;
            edges &= edges - 1;
            let at = base + k;
            let conts = cont_before + (blk.cont & (bit - 1)).count_ones() as usize;
            if starts & bit != 0 {
                open = (at, conts);
            } else if f(open.0, at, at - open.0 - (conts - open.1)).is_break() {
                stopped = true;
                return ControlFlow::Break(());
            }
        }
        cont_before += (blk.cont & blk.valid).count_ones() as usize;
        prev = word >> 63;
        ControlFlow::Continue(())
    });
    // A word that runs to the end of a full last block.
    if !stopped && prev == 1 {
        let _ = f(open.0, s.len(), s.len() - open.0 - (cont_before - open.1));
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

impl Split<'_> {
    /// The next word's byte range and length in code points.
    #[inline]
    pub fn next_counted(&mut self) -> Option<(usize, usize, usize)> {
        let b = self.s.as_bytes();
        let mut i = self.pos;
        // Skip whitespace.
        loop {
            let &c = b.get(i)?;
            match SPACE_CLASS[usize::from(c)] {
                WORD => break,
                SPACE => i += 1,
                _ => match multi_space_len(b, i) {
                    0 => break,
                    n => i += n,
                },
            }
        }
        // `b[start]` starts a word: a word byte, or the lead of a character that is not
        // whitespace (its continuation bytes are word bytes).
        let start = i;
        let mut continuations = 0;
        i += 1;
        while let Some(&c) = b.get(i) {
            match SPACE_CLASS[usize::from(c)] {
                WORD => {}
                SPACE => break,
                _ if multi_space_len(b, i) > 0 => break,
                _ => {}
            }
            continuations += usize::from(is_continuation(c));
            i += 1;
        }
        self.pos = i;
        Some((start, i, i - start - continuations))
    }

    /// The next word's byte range.
    #[inline]
    pub fn next_span(&mut self) -> Option<(usize, usize)> {
        self.next_counted().map(|(a, b, _)| (a, b))
    }
}

impl<'a> Iterator for Split<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        // Both ends are character boundaries: whitespace starts, or the ends of `s`.
        self.next_span().map(|(a, b)| &self.s[a..b])
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
    fn swar_predicates_are_exact() {
        // Every byte value in every lane, next to bytes that could carry into it.
        for v in 0..=255u8 {
            for lane in 0..8 {
                for fill in [0x00u8, 0x7F, 0x80, 0xFF, v] {
                    let mut bytes = [fill; 8];
                    bytes[lane] = v;
                    let x = u64::from_le_bytes(bytes);
                    for k in [0x00, 0x09, 0x0E, 0x1C, 0x21, 0x80] {
                        let got = flags_to_bits(bytes_below(x, k));
                        let want = (0..8).fold(0, |m, j| m | (u64::from(bytes[j] < k) << j));
                        assert_eq!(got, want, "below {k:#x}: {bytes:?}");
                    }
                    for e in [0xC2, 0xE1, 0xE2, 0xE3, 0x00, 0x80] {
                        let got = flags_to_bits(bytes_equal(x, e));
                        let want = (0..8).fold(0, |m, j| m | (u64::from(bytes[j] == e) << j));
                        assert_eq!(got, want, "equal {e:#x}: {bytes:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn split_matches_a_char_split() {
        // Whitespace of every width next to non-whitespace characters with the same lead
        // bytes (C2, E1, E2, E3), and lengths in code points.
        let pieces = [
            " ",
            "\t",
            "\u{1c}",
            "\u{85}",
            "\u{a0}",
            "\u{a9}",
            "\u{1680}",
            "\u{1681}",
            "\u{2000}",
            "\u{200a}",
            "\u{200b}",
            "\u{2010}",
            "\u{2028}",
            "\u{202f}",
            "\u{205f}",
            "\u{2060}",
            "\u{3000}",
            "\u{3001}",
            "a",
            "é",
            "中",
            "\u{1f600}",
            "\u{0}",
        ];
        let mut state = 7u64;
        let mut next = |m: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            usize::try_from(state >> 33).unwrap() % m
        };
        for _ in 0..3000 {
            // Up to ~600 bytes: words and multi-byte whitespace across 64-byte blocks.
            let mut s = String::new();
            for _ in 0..next(160) {
                // Runs of one piece make long words and long whitespace.
                let piece = pieces[next(pieces.len())];
                for _ in 0..=next(3) * next(3) {
                    s.push_str(piece);
                }
            }
            let want: Vec<&str> = s
                .split(is_py_whitespace)
                .filter(|w| !w.is_empty())
                .collect();
            assert_eq!(split(&s).collect::<Vec<_>>(), want, "{s:?}");
            let mut words = split(&s);
            let mut counted = Vec::new();
            while let Some((a, b, chars)) = words.next_counted() {
                assert_eq!(chars, s[a..b].chars().count());
                counted.push(&s[a..b]);
            }
            assert_eq!(counted, want);
            assert_eq!(char_len(&s), s.chars().count());
            // The block functions.
            let mut spans = Vec::new();
            for_each_word_span(&s, |a, b, chars| {
                assert_eq!(chars, s[a..b].chars().count(), "{s:?}");
                spans.push(&s[a..b]);
                ControlFlow::Continue(())
            });
            assert_eq!(spans, want, "{s:?}");
            let chars: usize = want.iter().map(|w| w.chars().count()).sum();
            assert_eq!(count_words(&s), want.len(), "{s:?}");
            assert_eq!(count_words_and_chars(&s), (want.len(), chars), "{s:?}");
            // Stopping early.
            let mut first = None;
            for_each_word_span(&s, |a, b, _| {
                first = Some(&s[a..b]);
                ControlFlow::Break(())
            });
            assert_eq!(first, want.first().copied());
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
        assert_eq!(Splitter::for_lang("ZH"), Some(Splitter::Jieba));
        assert_eq!(Splitter::for_lang("jA"), None);
        assert_eq!(Splitter::for_lang("\u{212A}a"), Some(Splitter::Whitespace));
        assert_eq!(Splitter::for_lang("en"), Some(Splitter::Whitespace));
        assert_eq!(Splitter::for_lang(""), Some(Splitter::Whitespace));
        let words: Vec<&str> = Splitter::Jieba.words("我来到北京清华大学 ok").collect();
        assert_eq!(words, ["我", "来到", "北京", "清华大学", " ", "ok"]);
    }
}
