//! Python `str` whitespace semantics: `str.split()`, `str.strip()` and
//! `str.rstrip()` with no arguments.
//!
//! CPython treats a code point as whitespace when `str.isspace()` is true
//! (`Py_UNICODE_ISSPACE`: bidirectional class WS, B or S, or category Zs).
//! That set differs from Rust's [`char::is_whitespace`] (which has no
//! U+001C..U+001F) and is the same on every CPython from 3.12 to 3.15
//! (checked over all code points, see `python/tests/test_metrics_sacrebleu.py`).

/// `c.isspace()` in Python.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::pystr::is_space;
/// assert!(is_space('\u{1f}') && is_space('\u{3000}') && is_space('\u{85}'));
/// assert!(!is_space('\u{200b}') && !is_space('\u{180e}') && !is_space('a'));
/// ```
#[inline]
pub const fn is_space(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r'
            | '\u{1c}'..='\u{20}'
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
    )
}

/// Can this UTF-8 lead byte start a whitespace code point? (ASCII
/// whitespace, or the lead bytes of U+0085, U+00A0, U+1680, U+2000..U+205F
/// and U+3000.) A quick filter before decoding.
#[inline]
const fn may_start_space(b: u8) -> bool {
    matches!(b, b'\t'..=b'\r' | 0x1c..=0x20 | 0xc2 | 0xe1 | 0xe2 | 0xe3)
}

/// `s.split()` in Python: the maximal runs of non-whitespace code points.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::pystr::split;
/// let words: Vec<&str> = split(" a\u{a0}b\u{1c}c  d\n").collect();
/// assert_eq!(words, ["a", "b", "c", "d"]);
/// ```
pub fn split(s: &str) -> impl Iterator<Item = &str> {
    Split { rest: s }
}

struct Split<'a> {
    rest: &'a str,
}

/// Byte length of the whitespace code point at the start of `s`, if any.
#[inline]
fn space_len(s: &str) -> Option<usize> {
    let b = *s.as_bytes().first()?;
    if !may_start_space(b) {
        return None;
    }
    if b < 0x80 {
        return Some(1);
    }
    let c = s.chars().next()?;
    is_space(c).then(|| c.len_utf8())
}

impl<'a> Iterator for Split<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        // Skip leading whitespace.
        let mut start = 0;
        while let Some(n) = space_len(&self.rest[start..]) {
            start += n;
        }
        if start == self.rest.len() {
            self.rest = "";
            return None;
        }
        // Scan the word: only bytes that may start a whitespace code point
        // need a closer look.
        let bytes = self.rest.as_bytes();
        let mut end = start;
        while end < bytes.len() {
            if may_start_space(bytes[end]) && space_len(&self.rest[end..]).is_some() {
                break;
            }
            end += 1;
            // Step over continuation bytes so slicing stays on boundaries.
            while end < bytes.len() && (bytes[end] & 0xc0) == 0x80 {
                end += 1;
            }
        }
        let word = &self.rest[start..end];
        self.rest = &self.rest[end..];
        Some(word)
    }
}

/// `s.rstrip()` in Python.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::pystr::rstrip;
/// assert_eq!(rstrip(" a b \u{3000}\u{1f}\n"), " a b");
/// ```
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `s.strip()` in Python.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::pystr::strip;
/// assert_eq!(strip("\u{85} a b \u{2028}"), "a b");
/// ```
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whitespace set as CPython 3.12-3.15 report it (`str.isspace()`).
    const PY_SPACES: [u32; 29] = [
        0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001,
        0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029,
        0x202f, 0x205f, 0x3000,
    ];

    #[test]
    fn whitespace_set_is_pythons() {
        let ours: Vec<u32> = (0..=0x10_ffff_u32)
            .filter_map(char::from_u32)
            .filter(|&c| is_space(c))
            .map(u32::from)
            .collect();
        assert_eq!(ours, PY_SPACES);
    }

    #[test]
    fn lead_byte_filter_covers_every_space() {
        for c in PY_SPACES.iter().filter_map(|&c| char::from_u32(c)) {
            let mut buf = [0; 4];
            assert!(
                may_start_space(c.encode_utf8(&mut buf).as_bytes()[0]),
                "{c:?}"
            );
        }
    }

    #[test]
    fn split_matches_python() {
        let cases: [(&str, &[&str]); 9] = [
            ("", &[]),
            ("   ", &[]),
            ("a", &["a"]),
            (" a  b ", &["a", "b"]),
            ("a\u{200b}b", &["a\u{200b}b"]),
            ("é\u{a0}ü\u{2000}x", &["é", "ü", "x"]),
            ("\u{e2}\u{80}\u{80}", &["\u{e2}\u{80}\u{80}"]),
            ("日本\u{3000}語", &["日本", "語"]),
            ("a\u{1c}\u{1d}\u{1e}\u{1f}b\u{85}c", &["a", "b", "c"]),
        ];
        for (s, want) in cases {
            assert_eq!(split(s).collect::<Vec<_>>(), want, "{s:?}");
        }
    }

    #[test]
    fn split_every_code_point_as_separator() {
        for c in (0..=0x10_ffff_u32).filter_map(char::from_u32) {
            let s = format!("a{c}b");
            let n = split(&s).count();
            assert_eq!(n, if is_space(c) { 2 } else { 1 }, "{c:?}");
        }
    }

    #[test]
    fn strips() {
        assert_eq!(rstrip("x \t"), "x");
        assert_eq!(rstrip("\u{a0}"), "");
        assert_eq!(strip("\u{1680}x\u{200b}"), "x\u{200b}");
        assert_eq!(strip("\u{205f}"), "");
    }
}
