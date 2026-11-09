// Reproduces CPython 3.11-3.15 behaviour (PSF-2.0): str methods, int(), repr().
//! The Python `str`/`bytes`/`int` semantics warcio relies on: `bytes.decode` with the
//! UTF-8-else-latin-1 fallback, `strip`, `split_prefix`'s `str.upper()`, `str.lower()`
//! comparisons, `int()` of a str (any script's decimal digits, `_` separators,
//! `sys.get_int_max_str_digits()`), `int(bytes, 16)`, and `repr()` of str, bytes and lists.

use std::borrow::Cow;
use std::fmt::Write as _;

use crate::ftfy::UnicodeVersion;

#[rustfmt::skip]
#[path = "tables.rs"]
mod tables;

/// `c.isspace()` (the same set in every CPython version).
#[inline]
pub(crate) fn is_space(c: char) -> bool {
    crate::text::is_py_whitespace(c)
}

/// `s.rstrip()`.
pub(crate) fn rstrip(s: &str) -> &str {
    s.trim_end_matches(is_space)
}

/// `s.lstrip()`.
pub(crate) fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_space)
}

/// `s.strip()`.
pub(crate) fn strip(s: &str) -> &str {
    lstrip(rstrip(s))
}

/// `len(s)`: code points.
#[inline]
pub(crate) fn py_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// `StatusAndHeadersParser.decode_header(line)`: `line.decode('utf-8')`, else latin-1.
pub(crate) fn decode_header(b: &[u8]) -> Cow<'_, str> {
    match simdutf8::basic::from_utf8(b) {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(b.iter().map(|&x| char::from(x)).collect()),
    }
}

/// The byte offset of the `n`th code point of `s` (or `s.len()`), for `s[n:]`.
pub(crate) fn char_offset(s: &str, n: usize) -> usize {
    if s.is_ascii() {
        return n.min(s.len());
    }
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

/// `StatusAndHeadersParser.split_prefix(key, prefixes)`: the first prefix that
/// `key.upper()` starts with, and `key[len(prefix):]` (sliced in code points of the
/// original, which differs from the upper-cased length for 'ß', 'ﬆ' and the like).
pub(crate) fn split_prefix<'k>(
    key: &'k str,
    prefixes: &[&'static str],
) -> Option<(&'static str, &'k str)> {
    let longest = prefixes.iter().map(|p| p.len()).max().unwrap_or(0);
    // The upper-cased start of `key`, long enough to compare with every prefix.
    let mut upper = String::with_capacity(longest + 4);
    for c in key.chars() {
        if upper.len() >= longest {
            break;
        }
        if c.is_ascii() {
            upper.push(c.to_ascii_uppercase());
        } else {
            upper.extend(c.to_uppercase());
        }
    }
    let p = prefixes.iter().find(|p| upper.starts_with(**p))?;
    // Prefixes are ASCII: their length in code points is their byte length.
    Some((p, &key[char_offset(key, p.len())..]))
}

/// `s.lower() == target` for an ASCII lower-case `target`. Only ASCII letters and U+212A
/// (KELVIN SIGN, which lowercases to 'k') can lower-case to ASCII, so final-sigma rules
/// never matter here.
pub(crate) fn lower_eq(s: &str, target: &str) -> bool {
    if s.is_ascii() {
        return s.eq_ignore_ascii_case(target);
    }
    s.chars().flat_map(char::to_lowercase).eq(target.chars())
}

/// `int()`'s result: exact, as `i128` while the magnitude is below 10^36, else as decimal
/// digits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum PyInt {
    Small(i128),
    /// `|value| >= 10^36`: the sign and the ASCII digits (no leading zeros).
    Huge {
        neg: bool,
        digits: String,
    },
}

const SMALL_MAX_DIGITS: usize = 36;

impl PyInt {
    pub(crate) fn is_negative(&self) -> bool {
        match self {
            Self::Small(v) => *v < 0,
            Self::Huge { neg, .. } => *neg,
        }
    }

    /// `self - k` for a small non-negative `k`.
    pub(crate) fn sub_small(&self, k: u64) -> Self {
        match self {
            Self::Small(v) => Self::Small(v - i128::from(k)),
            // |v| >= 10^36 > k: the sign does not change.
            Self::Huge { neg: false, digits } => Self::Huge {
                neg: false,
                digits: dec_sub(digits, k),
            },
            Self::Huge { neg: true, digits } => Self::Huge {
                neg: true,
                digits: dec_add(digits, k),
            },
        }
    }

    /// The value, saturated to the `i128` range (huge values only ever meet small ones).
    pub(crate) fn saturating(&self) -> i128 {
        match self {
            Self::Small(v) => *v,
            Self::Huge { neg: false, .. } => i128::MAX / 4,
            Self::Huge { neg: true, .. } => i128::MIN / 4,
        }
    }

    /// `str(value)`.
    pub(crate) fn to_decimal(&self) -> String {
        match self {
            Self::Small(v) => v.to_string(),
            Self::Huge { neg, digits } => {
                if *neg {
                    format!("-{digits}")
                } else {
                    digits.clone()
                }
            }
        }
    }
}

/// `digits + k` (ASCII decimal digits).
fn dec_add(digits: &str, k: u64) -> String {
    let mut out: Vec<u8> = digits.bytes().rev().collect();
    let mut carry = u128::from(k);
    let mut i = 0;
    while carry > 0 {
        if i == out.len() {
            out.push(b'0');
        }
        let d = u128::from(out[i] - b'0') + carry;
        out[i] = b'0' + u8::try_from(d % 10).expect("digit");
        carry = d / 10;
        i += 1;
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII digits")
}

/// `digits - k` for `digits >= k` (ASCII decimal digits).
fn dec_sub(digits: &str, k: u64) -> String {
    let mut out: Vec<u8> = digits.bytes().rev().collect();
    let mut borrow = i128::from(k);
    let mut i = 0;
    while borrow > 0 {
        let d = i128::from(out[i] - b'0');
        let sub = borrow % 10;
        borrow /= 10;
        let mut v = d - sub;
        if v < 0 {
            v += 10;
            borrow += 1;
        }
        out[i] = b'0' + u8::try_from(v).expect("digit");
        i += 1;
    }
    while out.len() > 1 && out.last() == Some(&b'0') {
        out.pop();
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII digits")
}

/// The digit value of a `str.isdecimal()` character (decimal digits come in runs of ten
/// from zero).
fn decimal_value(c: char, v: UnicodeVersion) -> Option<u8> {
    if c.is_ascii_digit() {
        return Some(c as u8 - b'0');
    }
    if c.is_ascii() || !crate::filters::unicode::is_digit(c, v) {
        return None;
    }
    let mut start = u32::from(c);
    while start > 0
        && char::from_u32(start - 1).is_some_and(|p| crate::filters::unicode::is_digit(p, v))
    {
        start -= 1;
    }
    u8::try_from((u32::from(c) - start) % 10).ok()
}

/// `Py_ISSPACE`: the ASCII whitespace `int()` skips (not U+001C..U+001F).
#[inline]
fn is_ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// `int(s)` for a str, or `None` where it raises ValueError. CPython maps non-ASCII
/// whitespace to ' ' and non-ASCII decimal digits to ASCII first, then parses: ASCII
/// whitespace around, one sign, digits with single `_` between them; more than
/// `max_str_digits` digits (0 = no limit) raise.
pub(crate) fn py_int(s: &str, max_str_digits: usize, v: UnicodeVersion) -> Option<PyInt> {
    // _PyUnicode_TransformDecimalAndSpaceToASCII: '?' (an error) for anything else.
    let mut ascii: Cow<'_, [u8]> = Cow::Borrowed(s.as_bytes());
    if !s.is_ascii() {
        let mut t = Vec::with_capacity(s.len());
        for c in s.chars() {
            if (c as u32) < 127 {
                t.push(c as u8);
            } else if is_space(c) {
                t.push(b' ');
            } else {
                t.push(b'0' + decimal_value(c, v)?);
            }
        }
        ascii = Cow::Owned(t);
    }
    let b = &ascii[..];
    let mut i = 0;
    while i < b.len() && is_ascii_space(b[i]) {
        i += 1;
    }
    let neg = match b.get(i) {
        Some(b'+') => {
            i += 1;
            false
        }
        Some(b'-') => {
            i += 1;
            true
        }
        _ => false,
    };
    if b.get(i) == Some(&b'_') {
        return None;
    }
    let start = i;
    let mut digits = String::new();
    let mut prev = 0u8;
    while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
        if b[i] == b'_' {
            if prev == b'_' {
                return None;
            }
        } else if !(digits.is_empty() && b[i] == b'0') {
            digits.push(char::from(b[i]));
        }
        prev = b[i];
        i += 1;
    }
    if prev == b'_' || i == start {
        return None;
    }
    let count = b[start..i].iter().filter(|&&x| x != b'_').count();
    while i < b.len() && is_ascii_space(b[i]) {
        i += 1;
    }
    if i != b.len() {
        return None;
    }
    if max_str_digits > 0 && count > max_str_digits {
        return None;
    }
    if digits.len() < SMALL_MAX_DIGITS {
        let mag: i128 = if digits.is_empty() {
            0
        } else {
            digits.parse().expect("ASCII digits")
        };
        return Some(PyInt::Small(if neg { -mag } else { mag }));
    }
    Some(PyInt::Huge { neg, digits })
}

/// `int(b, 16)` for bytes, saturated to `i128` (callers compare with `2**31`), or `None`
/// where it raises ValueError: ASCII whitespace around, a sign, an optional `0x`/`0X`
/// prefix with an optional `_` after it, hex digits with single `_` between them.
pub(crate) fn py_int_hex(b: &[u8]) -> Option<i128> {
    let mut i = 0;
    while i < b.len() && is_ascii_space(b[i]) {
        i += 1;
    }
    let neg = match b.get(i) {
        Some(b'+') => {
            i += 1;
            false
        }
        Some(b'-') => {
            i += 1;
            true
        }
        _ => false,
    };
    if b.get(i) == Some(&b'0') && matches!(b.get(i + 1), Some(b'x' | b'X')) {
        i += 2;
        if b.get(i) == Some(&b'_') {
            i += 1;
        }
    }
    if b.get(i) == Some(&b'_') {
        return None;
    }
    let start = i;
    let mut v: i128 = 0;
    let mut prev = 0u8;
    while i < b.len() && (b[i].is_ascii_hexdigit() || b[i] == b'_') {
        if b[i] == b'_' {
            if prev == b'_' {
                return None;
            }
        } else {
            let d = char::from(b[i]).to_digit(16).expect("hex digit");
            v = v.saturating_mul(16).saturating_add(i128::from(d));
        }
        prev = b[i];
        i += 1;
    }
    if prev == b'_' || i == start {
        return None;
    }
    while i < b.len() && is_ascii_space(b[i]) {
        i += 1;
    }
    if i != b.len() {
        return None;
    }
    Some(if neg { -v } else { v })
}

/// `c.isprintable()`.
fn is_printable(c: char, v: UnicodeVersion) -> bool {
    let cp = u32::from(c);
    if cp < 0x80 {
        return (0x20..0x7f).contains(&cp);
    }
    let vi = match v {
        UnicodeVersion::V14_0 => 0,
        UnicodeVersion::V15_0 => 1,
        UnicodeVersion::V15_1 => 2,
        UnicodeVersion::V16_0 => 3,
        _ => 4,
    };
    let t = &tables::PRINTABLE;
    let after = t.partition_point(|&(lo, _, _)| lo <= cp);
    after > 0 && cp <= t[after - 1].1 && t[after - 1].2 <= vi
}

/// `repr(s)` appended to `out`.
pub(crate) fn push_str_repr(out: &mut String, s: &str, v: UnicodeVersion) {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if is_printable(c, v) => out.push(c),
            _ => {
                let cp = u32::from(c);
                if cp < 0x100 {
                    let _ = write!(out, "\\x{cp:02x}");
                } else if cp < 0x1_0000 {
                    let _ = write!(out, "\\u{cp:04x}");
                } else {
                    let _ = write!(out, "\\U{cp:08x}");
                }
            }
        }
    }
    out.push(quote);
}

/// `str(list_of_str)`: `"['a', 'b']"`.
pub(crate) fn list_repr(items: &[&str], v: UnicodeVersion) -> String {
    let mut out = String::from("[");
    for (i, s) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_str_repr(&mut out, s, v);
    }
    out.push(']');
    out
}

/// `repr(b)` (also `str(b)`, `'{}'.format(b)`) for bytes.
pub(crate) fn bytes_repr(b: &[u8]) -> String {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::with_capacity(b.len() + 3);
    out.push('b');
    out.push(char::from(quote));
    for &x in b {
        match x {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            _ if x == quote => {
                out.push('\\');
                out.push(char::from(x));
            }
            0x20..0x7f => out.push(char::from(x)),
            _ => {
                let _ = write!(out, "\\x{x:02x}");
            }
        }
    }
    out.push(char::from(quote));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    fn int(s: &str) -> Option<String> {
        py_int(s, 4300, V).map(|v| v.to_decimal())
    }

    #[test]
    fn int_like_cpython() {
        // Expected values from CPython 3.13.
        assert_eq!(int("\u{661}\u{662}").as_deref(), Some("12"));
        assert_eq!(int("\x1c5"), None);
        assert_eq!(int("5\x1c"), None);
        assert_eq!(int("\u{a0}5").as_deref(), Some("5"));
        assert_eq!(int("1_0").as_deref(), Some("10"));
        assert_eq!(int("_1"), None);
        assert_eq!(int("1__0"), None);
        assert_eq!(int("1_"), None);
        assert_eq!(int(" +5 ").as_deref(), Some("5"));
        assert_eq!(int("\x0b5\x0c").as_deref(), Some("5"));
        assert_eq!(int("\u{85}5").as_deref(), Some("5"));
        assert_eq!(int("-0").as_deref(), Some("0"));
        assert_eq!(int("007").as_deref(), Some("7"));
        assert_eq!(int(""), None);
        assert_eq!(int("+"), None);
        assert_eq!(int("5\0"), None);
        assert_eq!(int("\u{ff15}").as_deref(), Some("5"));
        let big = "9".repeat(50);
        assert_eq!(int(&big).as_deref(), Some(big.as_str()));
        assert_eq!(int(&"1".repeat(4301)), None);
        assert!(py_int(&"1".repeat(4301), 0, V).is_some());
        assert_eq!(
            int(&format!("{}_1", "1".repeat(4299))).map(|s| s.len()),
            Some(4300)
        );
    }

    #[test]
    fn huge_arithmetic() {
        let v = py_int(&format!("1{}", "0".repeat(40)), 0, V).unwrap();
        assert_eq!(v.sub_small(1).to_decimal(), "9".repeat(40));
        let n = py_int(&format!("-1{}", "0".repeat(40)), 0, V).unwrap();
        assert!(n.is_negative());
        assert_eq!(
            n.sub_small(5).to_decimal(),
            format!("-1{}5", "0".repeat(39))
        );
        assert_eq!(PyInt::Small(5).sub_small(7), PyInt::Small(-2));
    }

    #[test]
    fn int_hex_like_cpython() {
        assert_eq!(py_int_hex(b" 1a "), Some(26));
        assert_eq!(py_int_hex(b"0x1a"), Some(26));
        assert_eq!(py_int_hex(b"0x_1a"), Some(26));
        assert_eq!(py_int_hex(b"1_a"), Some(26));
        assert_eq!(py_int_hex(b"-5"), Some(-5));
        assert_eq!(py_int_hex(b"\x0b5"), Some(5));
        assert_eq!(py_int_hex(b"\x1c5"), None);
        assert_eq!(py_int_hex(b"+0X5"), Some(5));
        assert_eq!(py_int_hex(b"0_x5"), None);
        assert_eq!(py_int_hex(b"0x"), None);
        assert_eq!(py_int_hex(b""), None);
        assert_eq!(py_int_hex(b"\xc3\xa9"), None);
    }

    #[test]
    fn prefixes() {
        const P: &[&str] = &["GET", "POST", "PUT"];
        assert_eq!(split_prefix("get /x", P), Some(("GET", " /x")));
        // 'ﬆ'.upper() == 'ST': one code point of the key covers two of the prefix.
        assert_eq!(split_prefix("PO\u{fb06} /x", P), Some(("POST", "/x")));
        assert_eq!(split_prefix("po\u{17f}t /", P), Some(("POST", " /")));
        assert_eq!(split_prefix("HEAD", P), None);
        assert_eq!(split_prefix("", P), None);
    }

    #[test]
    fn lower_compare() {
        assert!(lower_eq("Content-Length", "content-length"));
        assert!(lower_eq("WARC-Bloc\u{212a}-Digest", "warc-block-digest"));
        assert!(!lower_eq("\u{130}", "i"));
        assert!(!lower_eq("uri ", "uri"));
    }

    #[test]
    fn reprs() {
        // str(['a\x00\x7f\x80\xa0\xadé͸\U0001f600', "it's", 'a"b', '\'"', '\\', '\t\n\r'])
        let items = [
            "a\x00\x7f\u{80}\u{a0}\u{ad}\u{e9}\u{378}\u{1f600}",
            "it's",
            "a\"b",
            "'\"",
            "\\",
            "\t\n\r",
        ];
        assert_eq!(
            list_repr(&items, V),
            r#"['a\x00\x7f\x80\xa0\xadé\u0378😀', "it's", 'a"b', '\'"', '\\', '\t\n\r']"#
        );
        assert_eq!(
            bytes_repr(b"a'b\x00\x80\t\n\r\\"),
            r#"b"a'b\x00\x80\t\n\r\\""#
        );
        assert_eq!(bytes_repr(b"\"'"), r#"b'"\''"#);
        assert_eq!(bytes_repr(b"'"), r#"b"'""#);
    }

    #[test]
    fn decoding_and_strip() {
        assert_eq!(decode_header(b"caf\xc3\xa9"), "caf\u{e9}");
        assert_eq!(decode_header(b"caf\xe9"), "caf\u{e9}");
        // A UTF-8-encoded surrogate is not UTF-8 for CPython either.
        assert_eq!(decode_header(b"\xed\xa0\x80"), "\u{ed}\u{a0}\u{80}");
        assert_eq!(rstrip("a \x1c\u{85}\r\n"), "a");
        assert_eq!(strip("\u{3000} a b \t"), "a b");
        assert_eq!(py_len("caf\u{e9}"), 4);
    }
}
