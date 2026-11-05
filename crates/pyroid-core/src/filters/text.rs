// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! The Python `str` operations the filters use, with CPython's semantics.

use memchr::memmem;

use super::unicode;
use crate::ftfy::UnicodeVersion;

/// `len(s)`: the number of code points.
#[inline]
pub(crate) fn py_len(s: &str) -> usize {
    // Every code point has exactly one non-continuation byte.
    s.len() - s.bytes().filter(|&b| (b & 0xC0) == 0x80).count()
}

/// `s.strip()`.
#[inline]
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(unicode::is_space)
}

/// `s.lstrip()`.
#[inline]
pub(crate) fn py_lstrip(s: &str) -> &str {
    s.trim_start_matches(unicode::is_space)
}

/// `s.rstrip()`.
#[inline]
pub(crate) fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(unicode::is_space)
}

/// `s.split()`: runs of non-whitespace.
pub(crate) fn py_split(s: &str) -> impl Iterator<Item = &str> {
    s.split(unicode::is_space).filter(|w| !w.is_empty())
}

/// `str.splitlines()` boundaries (`\r\n` counts as one).
#[inline]
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `s.splitlines()`: lines without their terminators; no trailing empty line.
pub(crate) fn py_splitlines(s: &str) -> SplitLines<'_> {
    SplitLines { rest: s }
}

/// Iterator returned by [`py_splitlines`].
#[derive(Debug)]
pub(crate) struct SplitLines<'a> {
    rest: &'a str,
}

impl<'a> Iterator for SplitLines<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        let s = self.rest;
        // Fast path: the ASCII breaks are all below 0x20; others start with 0xC2/0xE2.
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            if b < 0x20 || b == 0xC2 || b == 0xE2 {
                let c = s[i..].chars().next().unwrap_or('\0');
                if is_line_break(c) {
                    let mut end = i + c.len_utf8();
                    if c == '\r' && bytes.get(end) == Some(&b'\n') {
                        end += 1;
                    }
                    self.rest = &s[end..];
                    return Some(&s[..i]);
                }
                i += c.len_utf8();
            } else {
                i += 1;
            }
        }
        self.rest = "";
        Some(s)
    }
}

/// `re.split(r"\n{2,}", s)` (`min_run = 2`) or `re.split(r"\n+", s)`
/// (`min_run = 1`).
pub(crate) fn split_newline_runs(s: &str, min_run: usize) -> Vec<&str> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while let Some(off) = memchr::memchr(b'\n', &bytes[i..]) {
        let a = i + off;
        let mut b = a;
        while b < bytes.len() && bytes[b] == b'\n' {
            b += 1;
        }
        if b - a >= min_run {
            out.push(&s[start..a]);
            start = b;
        }
        i = b;
    }
    out.push(&s[start..]);
    out
}

/// A set of characters: ASCII bitmap plus a sorted list above ASCII.
#[derive(Debug, Clone)]
pub(crate) struct CharSet {
    ascii: u128,
    other: Vec<char>,
}

impl CharSet {
    pub(crate) fn new(chars: impl IntoIterator<Item = char>) -> Self {
        let mut ascii = 0u128;
        let mut other = Vec::new();
        for c in chars {
            if c.is_ascii() {
                ascii |= 1 << (c as u32);
            } else {
                other.push(c);
            }
        }
        other.sort_unstable();
        other.dedup();
        Self { ascii, other }
    }

    #[inline]
    pub(crate) fn contains(&self, c: char) -> bool {
        if c.is_ascii() {
            (self.ascii >> (c as u32)) & 1 == 1
        } else {
            self.other.binary_search(&c).is_ok()
        }
    }
}

/// `CITATION_REGEX.sub("", s)` for `\[\d*]|\[edit]|\[citation needed]`.
pub(crate) fn remove_citations(s: &str, v: UnicodeVersion) -> std::borrow::Cow<'_, str> {
    let Some(first) = memchr::memchr(b'[', s.as_bytes()) else {
        return std::borrow::Cow::Borrowed(s);
    };
    let mut out = String::new();
    let mut copied = 0;
    let mut i = first;
    while i < s.len() {
        let Some(off) = memchr::memchr(b'[', &s.as_bytes()[i..]) else {
            break;
        };
        let at = i + off;
        let after = &s[at + 1..];
        // Alternatives in order: `\[\d*]`, `\[edit]`, `\[citation needed]`.
        let digits = after
            .char_indices()
            .find(|&(_, c)| !unicode::is_digit(c, v))
            .map_or(after.len(), |(j, _)| j);
        let len = if after[digits..].starts_with(']') {
            Some(digits + 1)
        } else if after.starts_with("edit]") {
            Some(5)
        } else if after.starts_with("citation needed]") {
            Some(16)
        } else {
            None
        };
        match len {
            Some(n) => {
                out.push_str(&s[copied..at]);
                copied = at + 1 + n;
                i = copied;
            }
            None => i = at + 1,
        }
    }
    if copied == 0 {
        return std::borrow::Cow::Borrowed(s);
    }
    out.push_str(&s[copied..]);
    std::borrow::Cow::Owned(out)
}

/// `s.lower()` reduced to what an ASCII substring search can see: ASCII
/// letters lowercased, U+212A KELVIN SIGN as `k`, U+0130 as `i` + a barrier,
/// and every other non-ASCII character as one barrier byte (`0xFF`). On
/// CPython 3.11-3.15 these are exactly the characters whose `lower()`
/// contains ASCII (checked by `generate_unicode_tables.py`).
pub(crate) fn fold_lower_ascii(s: &str, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(s.len());
    if s.is_ascii() {
        out.extend(s.bytes().map(|b| b.to_ascii_lowercase()));
        return;
    }
    for c in s.chars() {
        match c {
            _ if c.is_ascii() => out.push((c as u8).to_ascii_lowercase()),
            '\u{212a}' => out.push(b'k'),
            '\u{130}' => out.extend_from_slice(&[b'i', 0xFF]),
            _ => out.push(0xFF),
        }
    }
}

/// `needle in haystack` on bytes.
#[inline]
pub(crate) fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    memmem::find(haystack, needle).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_matches_python() {
        let lines: Vec<&str> = py_splitlines("a\r\nb\rc\n\nd\u{2028}e\x0bf\n").collect();
        assert_eq!(lines, ["a", "b", "c", "", "d", "e", "f"]);
        assert_eq!(py_splitlines("").count(), 0);
        assert_eq!(py_splitlines("\n").collect::<Vec<_>>(), [""]);
        assert_eq!(py_splitlines("x\u{85}y").collect::<Vec<_>>(), ["x", "y"]);
        assert_eq!(py_splitlines("é\u{a0}").collect::<Vec<_>>(), ["é\u{a0}"]);
    }

    #[test]
    fn newline_runs() {
        assert_eq!(
            split_newline_runs("a\n\nb\nc\n\n\nd", 2),
            ["a", "b\nc", "d"]
        );
        assert_eq!(split_newline_runs("\na\n", 1), ["", "a", ""]);
        assert_eq!(split_newline_runs("", 1), [""]);
    }

    #[test]
    fn citations() {
        let v = UnicodeVersion::V15_1;
        assert_eq!(
            remove_citations("a[1] b[] c[edit] d[citation needed][x]", v),
            "a b c d[x]"
        );
        assert_eq!(remove_citations("[[12]]", v), "[]");
        assert_eq!(remove_citations("x[\u{663}]", v), "x"); // Arabic-Indic digit is \d
    }

    #[test]
    fn folding() {
        let mut b = Vec::new();
        fold_lower_ascii("JAVA\u{212a}\u{130}x é", &mut b);
        assert_eq!(b, b"javaki\xffx \xff");
    }

    #[test]
    fn lengths() {
        assert_eq!(py_len("aé😀"), 3);
        assert_eq!(py_strip("\u{3000} x \x1c"), "x");
    }
}
