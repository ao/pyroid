// Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
//! The few `str` operations tabulate relies on, with CPython's exact semantics.
//!
//! - [`is_space`]: `str.isspace()` for one char. CPython's whitespace (bidi class WS, B or S,
//!   or category Zs) is Rust's `White_Space` plus U+001C..U+001F; both sets have been stable
//!   since Unicode 6.3.
//! - [`strip`], [`rstrip`]: `str.strip()` / `str.rstrip()` without arguments.
//! - [`splitlines`]: `str.splitlines()` (no `keepends`).
//! - [`split_cr_lf`]: `re.split("[\r\n]", s)`.
//! - [`py_len`]: `len(s)` (code points).
//! - [`repeat`]: `s * n` (empty for `n <= 0`).

/// `c.isspace()` in CPython.
#[inline]
pub fn is_space(c: char) -> bool {
    matches!(c, '\u{1c}'..='\u{1f}') || c.is_whitespace()
}

/// `s.strip()`.
#[inline]
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `s.rstrip()`.
#[inline]
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `len(s)`: the number of code points.
#[inline]
pub fn py_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// Is `s` printable ASCII (U+0020..U+007E) only? Then every width function agrees with `len`.
#[inline]
pub fn is_printable_ascii(s: &str) -> bool {
    s.bytes().all(|b| (0x20..0x7f).contains(&b))
}

/// The line boundaries of `str.splitlines()`.
#[inline]
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\u{0b}'
            | '\u{0c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `s.splitlines()`: lines without their breaks; `"\r\n"` is one break; no trailing empty line.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if is_line_break(c) {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(j, '\n')) = it.peek() {
                    it.next();
                    end = j + 1;
                }
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `re.split("[\r\n]", s)`: always one more piece than there are `\r` and `\n`.
pub fn split_cr_lf(s: &str) -> Vec<&str> {
    s.split(['\r', '\n']).collect()
}

/// Does `s` hold `\r` or `\n` (`re.search(r"\r|\n|\r\n", s)`)?
#[inline]
pub fn has_cr_lf(s: &str) -> bool {
    s.bytes().any(|b| b == b'\n' || b == b'\r')
}

/// Does `s` hold an ESC, and does it hold `\r` or `\n`?
#[inline]
pub fn esc_and_newline(s: &str) -> (bool, bool) {
    let b = s.as_bytes();
    if memchr::memchr3(0x1b, b'\n', b'\r', b).is_none() {
        return (false, false);
    }
    (
        memchr::memchr(0x1b, b).is_some(),
        memchr::memchr2(b'\n', b'\r', b).is_some(),
    )
}

/// `s * n`.
#[inline]
pub fn repeat(s: &str, n: isize) -> String {
    match usize::try_from(n) {
        Ok(n) => s.repeat(n),
        Err(_) => String::new(),
    }
}

/// Append `n` spaces (none for `n <= 0`).
#[inline]
pub fn push_spaces(out: &mut String, n: isize) {
    if let Ok(n) = usize::try_from(n) {
        push_n_spaces(out, n);
    }
}

/// Append `n` spaces (copied from a constant, not pushed one char at a time).
#[inline]
pub fn push_n_spaces(out: &mut String, mut n: usize) {
    const SPACES: &str = "                                                                ";
    while n > SPACES.len() {
        out.push_str(SPACES);
        n -= SPACES.len();
    }
    out.push_str(&SPACES[..n]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("a"), vec!["a"]);
        assert_eq!(splitlines("a\n"), vec!["a"]);
        assert_eq!(splitlines("\n"), vec![""]);
        assert_eq!(splitlines("a\r\nb"), vec!["a", "b"]);
        assert_eq!(splitlines("a\n\rb"), vec!["a", "", "b"]);
        assert_eq!(
            splitlines("a\u{2028}b\x0cc\x1dd\u{85}"),
            vec!["a", "b", "c", "d"]
        );
        assert_eq!(splitlines("a\r"), vec!["a"]);
        assert_eq!(splitlines("a\r\r\n"), vec!["a", ""]);
    }

    #[test]
    fn split_cr_lf_matches_re_split() {
        assert_eq!(split_cr_lf(""), vec![""]);
        assert_eq!(split_cr_lf("a\r\nb"), vec!["a", "", "b"]);
        assert_eq!(split_cr_lf("a\n"), vec!["a", ""]);
    }

    #[test]
    fn whitespace_is_pythons() {
        for c in [
            '\t', '\n', '\x0b', '\x0c', '\r', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '\u{85}',
            '\u{a0}', '\u{1680}', '\u{2000}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
            '\u{205f}', '\u{3000}',
        ] {
            assert!(is_space(c), "{c:?}");
        }
        for c in ['\u{200b}', '\u{180e}', '\u{feff}', 'a', '\x00', '\x7f'] {
            assert!(!is_space(c), "{c:?}");
        }
        assert_eq!(strip("\x1c a \u{3000}"), "a");
        assert_eq!(rstrip(" a \x1f"), " a");
    }
}
