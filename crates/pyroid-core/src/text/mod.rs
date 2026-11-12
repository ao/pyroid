//! Text kernels behind `pyroid.text` (the demonstration module for the
//! product-track conventions in `docs/dev/MODULES.md`).
//!
//! Kernels are plain functions over `&str`. Transforms return `Cow::Borrowed`
//! of the *whole* input when nothing changes, so the bindings can hand back
//! the caller's original object.

use std::borrow::Cow;

/// Python's `str.isspace()` for one character: exactly the characters that
/// `str.split()` (no arguments) splits on.
///
/// That is Unicode `White_Space` plus U+001C..U+001F (the ASCII file, group,
/// record and unit separators, which Python counts as whitespace and Rust's
/// `char::is_whitespace` does not).
pub fn is_py_whitespace(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r'
            | '\x1c'..=' '
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

/// True if `s` is already in `normalize_whitespace` form: no leading or
/// trailing whitespace, and words separated by exactly one ASCII space.
fn is_whitespace_normalized(s: &str) -> bool {
    let bytes = s.as_bytes();
    // Start of string behaves like "just saw a space", so leading space fails.
    let mut after_space = true;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii() {
            if b == b' ' {
                if after_space {
                    return false;
                }
                after_space = true;
            } else if matches!(b, b'\t'..=b'\r' | 0x1c..=0x1f) {
                return false;
            } else {
                after_space = false;
            }
            i += 1;
        } else {
            let c = s[i..].chars().next().expect("i is a char boundary");
            if is_py_whitespace(c) {
                return false;
            }
            after_space = false;
            i += c.len_utf8();
        }
    }
    !after_space || bytes.is_empty()
}

/// Collapse every run of whitespace to one ASCII space and strip both ends.
///
/// Identical to Python's `" ".join(s.split())`. Returns `Cow::Borrowed(s)`
/// when `s` is already normalized.
///
/// ```
/// use pyroid_core::text::normalize_whitespace;
/// assert_eq!(normalize_whitespace("  a \t\n b\u{3000}c "), "a b c");
/// assert!(matches!(normalize_whitespace("a b"), std::borrow::Cow::Borrowed("a b")));
/// ```
pub fn normalize_whitespace(s: &str) -> Cow<'_, str> {
    if is_whitespace_normalized(s) {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    for word in s.split(is_py_whitespace).filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    Cow::Owned(out)
}

/// True if `s` has at least `n` characters (Unicode code points, like
/// Python's `len(s) >= n`).
///
/// ```
/// use pyroid_core::text::min_chars;
/// assert!(min_chars("héllo", 5));
/// assert!(!min_chars("héllo", 6));
/// assert!(min_chars("", 0));
/// ```
pub fn min_chars(s: &str, n: usize) -> bool {
    let bytes = s.len();
    // A char is 1..=4 bytes, so the byte length bounds the char count.
    if bytes < n {
        return false;
    }
    if bytes / 4 >= n {
        return true;
    }
    s.chars().count() >= n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Python 3.11–3.15 `str.isspace()` true set, from `CPython`'s
    /// `_PyUnicode_IsWhitespace` (bidi class WS/B/S or category Zs).
    const PY_WHITESPACE: &[u32] = &[
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000,
        0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
        0x2029, 0x202f, 0x205f, 0x3000,
    ];

    #[test]
    fn whitespace_set_matches_python_exactly() {
        for c in (0..=0x10_ffff).filter_map(char::from_u32) {
            assert_eq!(
                is_py_whitespace(c),
                PY_WHITESPACE.contains(&u32::from(c)),
                "U+{:04X}",
                u32::from(c)
            );
        }
    }

    fn reference(s: &str) -> String {
        s.split(is_py_whitespace)
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn normalize_examples() {
        for (input, want) in [
            ("", ""),
            (" ", ""),
            ("a", "a"),
            ("a b", "a b"),
            (" a", "a"),
            ("a ", "a"),
            ("a  b", "a b"),
            ("a\tb", "a b"),
            ("a\u{a0}b", "a b"),
            ("a\u{1c}b", "a b"),
            ("\u{3000}日本\u{2028}語\u{85}", "日本 語"),
            ("a\u{200b}b", "a\u{200b}b"), // zero-width space is not whitespace
            ("é  ü", "é ü"),
        ] {
            assert_eq!(normalize_whitespace(input), want, "{input:?}");
            assert_eq!(reference(input), want, "{input:?}");
        }
    }

    #[test]
    fn normalize_borrows_iff_unchanged() {
        for s in ["", "a", "a b c", "日本 語", "x\u{200b}y"] {
            assert!(
                matches!(normalize_whitespace(s), Cow::Borrowed(b) if b == s),
                "{s:?}"
            );
        }
        for s in [" ", "a  b", "a\nb", "a ", "\u{a0}a"] {
            assert!(matches!(normalize_whitespace(s), Cow::Owned(_)), "{s:?}");
        }
    }

    #[test]
    fn normalize_every_char_between_words() {
        for c in (0..=0x10_ffff).filter_map(char::from_u32) {
            let s = format!("a{c}{c}b");
            assert_eq!(
                normalize_whitespace(&s),
                reference(&s),
                "U+{:04X}",
                u32::from(c)
            );
        }
    }

    #[test]
    fn min_chars_examples() {
        assert!(min_chars("", 0));
        assert!(!min_chars("", 1));
        assert!(min_chars("abc", 3));
        assert!(!min_chars("abc", 4));
        assert!(min_chars("😀😀", 2)); // 8 bytes, 2 chars
        assert!(!min_chars("😀😀", 3));
        assert!(min_chars("😀😀😀😀", 2)); // bytes / 4 shortcut
        assert!(!min_chars("ab", usize::MAX));
        let s = "é".repeat(100);
        assert!(min_chars(&s, 100));
        assert!(!min_chars(&s, 101));
    }
}
