// Reproduces CPython 3.11-3.15 behaviour (PSF-2.0); character data from each version's unicodedata.
//! The Python `str` and number semantics dateutil's parser relies on, for the caller's CPython
//! (`unicodedata`) version: `isalpha`, `isdigit` and `isspace` per character (`_timelex`),
//! `str.lower()` (parserinfo lookups), `str.isdigit()`, and `int(str)` / `Decimal(str)` of the
//! tokens the parser converts.

use std::borrow::Cow;
use std::cmp::Ordering;

use crate::ftfy::UnicodeVersion;
use crate::json_repair::chars::{CharProps, UnicodeVersion as JrVersion};

/// An integer as the parser computes it. Python's are unbounded; values past `i64` make the
/// kernel hand the string to the Python reference ([`Fail::Reference`]).
pub type Int = i64;

/// Why the kernel stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fail {
    /// dateutil's `_parse` would raise `ValueError` or `IndexError`, which it turns into
    /// `return None, None` ("Unknown string format").
    Invalid,
    /// The result depends on Python behaviour the kernel does not reproduce; the caller runs
    /// dateutil's own algorithm (pyroid's Python copy of it) instead.
    Reference(Reason),
}

/// What [`Fail::Reference`] needs Python for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reason {
    /// An integer outside `i64` (Python's ints are unbounded).
    BigNumber,
    /// `Decimal` arithmetic (`% 1`, `* 60`) on a number with more than 26 significant digits,
    /// where the `decimal` context's precision would round or raise.
    LongDecimal,
    /// A value from a custom parserinfo table that the kernel cannot use as an index.
    InfoValue,
}

impl Reason {
    /// The reason as the fallback warning names it.
    pub fn describe(self) -> &'static str {
        match self {
            Self::BigNumber => "a number too large for the Rust port (past 64 bits)",
            Self::LongDecimal => "a fractional number with more than 26 significant digits",
            Self::InfoValue => "a parserinfo table value the Rust port does not handle",
        }
    }
}

/// The character predicates of one Unicode version.
#[derive(Clone, Copy, Debug)]
pub struct Props {
    pub(crate) v: UnicodeVersion,
    jr: CharProps,
}

impl Props {
    pub fn new(v: UnicodeVersion) -> Self {
        let jv = JrVersion::from_unidata_version(v.as_str()).expect("same version set");
        Self {
            v,
            jr: CharProps::new(jv),
        }
    }

    /// `c.isalpha()`.
    #[inline]
    pub fn is_alpha(self, c: char) -> bool {
        if c.is_ascii() {
            return c.is_ascii_alphabetic();
        }
        crate::filters::unicode::is_alpha(c, self.v)
    }

    /// `c.isdigit()` (decimal digits and the other `Numeric_Type=Digit` characters, e.g. `²`).
    #[inline]
    pub fn is_digit(self, c: char) -> bool {
        if c.is_ascii() {
            return c.is_ascii_digit();
        }
        self.jr.is_digit(c)
    }

    /// `c.isspace()` (the same set in every version).
    #[inline]
    pub fn is_space(c: char) -> bool {
        crate::text::is_py_whitespace(c)
    }

    /// `c.isdecimal()`.
    #[inline]
    pub fn is_decimal(self, c: char) -> bool {
        if c.is_ascii() {
            return c.is_ascii_digit();
        }
        crate::filters::unicode::is_digit(c, self.v)
    }

    /// The digit value of a `str.isdecimal()` character (Unicode decimal digits come in runs of
    /// ten consecutive code points from zero, in every version).
    pub fn decimal_value(self, c: char) -> Option<u8> {
        if c.is_ascii_digit() {
            return Some(c as u8 - b'0');
        }
        if !self.is_decimal(c) {
            return None;
        }
        let mut start = u32::from(c);
        while start > 0 && char::from_u32(start - 1).is_some_and(|p| self.is_decimal(p)) {
            start -= 1;
        }
        Some(u8::try_from((u32::from(c) - start) % 10).expect("< 10"))
    }

    /// `s.isdigit()`: non-empty and every character a digit.
    pub fn str_isdigit(self, s: &str) -> bool {
        if s.is_ascii() {
            return !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
        }
        !s.is_empty() && s.chars().all(|c| self.is_digit(c))
    }

    /// `s.lower()`.
    pub fn lower(self, s: &str) -> Cow<'_, str> {
        if s.is_ascii() {
            if s.bytes().any(|b| b.is_ascii_uppercase()) {
                return Cow::Owned(s.to_ascii_lowercase());
            }
            return Cow::Borrowed(s);
        }
        crate::filters::url::py_lower(s, self.v)
    }
}

/// `len(s)`: code points.
#[inline]
pub fn py_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// `s[a:b]` with code point indices (clamped as Python clamps; `b = None` is the end).
pub fn py_slice(s: &str, a: usize, b: Option<usize>) -> &str {
    if s.is_ascii() {
        let n = s.len();
        let a = a.min(n);
        let b = b.map_or(n, |b| b.min(n)).max(a);
        return &s[a..b];
    }
    let mut idx = s
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(s.len()));
    let start = idx.nth(a).unwrap_or(s.len());
    let end = match b {
        None => s.len(),
        Some(b) if b <= a => return "",
        Some(b) => s
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(s.len()))
            .nth(b)
            .unwrap_or(s.len()),
    };
    &s[start..end.max(start)]
}

/// `s.find(".")` as a code point index, `None` for -1.
pub fn py_find_dot(s: &str) -> Option<usize> {
    let byte = s.find('.')?;
    Some(py_len(&s[..byte]))
}

/// Accumulate decimal digit `d` into `acc` (`None` once past `i64`).
#[inline]
fn push_digit(acc: Option<Int>, d: u8) -> Option<Int> {
    acc?.checked_mul(10)?.checked_add(Int::from(d))
}

/// `int(s)` for a `str`, base 10: surrounding whitespace, one sign, Unicode decimal digits with
/// single underscores between them. `max_digits` is `sys.get_int_max_str_digits()` (0: no
/// limit).
#[allow(clippy::many_single_char_names)]
pub fn py_int(p: Props, s: &str, max_digits: usize) -> Result<Int, Fail> {
    let b = s.as_bytes();
    if !b.is_empty()
        && b.len() <= 18
        && (max_digits == 0 || b.len() <= max_digits)
        && b.iter().all(u8::is_ascii_digit)
    {
        return Ok(b.iter().fold(0, |acc, &d| acc * 10 + Int::from(d - b'0')));
    }
    let t = s.trim_matches(Props::is_space);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.is_empty() {
        return Err(Fail::Invalid);
    }
    let mut acc: Option<Int> = Some(0);
    let mut digits = 0usize;
    let mut prev_underscore = true; // a leading underscore is invalid
    for c in body.chars() {
        if c == '_' {
            if prev_underscore {
                return Err(Fail::Invalid);
            }
            prev_underscore = true;
            continue;
        }
        let d = p.decimal_value(c).ok_or(Fail::Invalid)?;
        acc = push_digit(acc, d);
        digits += 1;
        prev_underscore = false;
    }
    if prev_underscore {
        return Err(Fail::Invalid);
    }
    if max_digits > 0 && digits > max_digits {
        return Err(Fail::Invalid);
    }
    let v = acc.ok_or(Fail::Reference(Reason::BigNumber))?;
    Ok(if neg { -v } else { v })
}

/// A finite `Decimal(token)` for the tokens the parser converts: `Nd+` or `Nd+.Nd+` (the
/// tokenizer never makes another finite form; words such as `inf` are not finite). Digits are
/// kept as ASCII.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Num<'a> {
    /// Integer part, ASCII digits (at least one).
    pub int: Cow<'a, str>,
    /// Fraction digits (empty without a point).
    pub frac: Cow<'a, str>,
}

impl<'a> Num<'a> {
    /// `Decimal(s)` if it is finite, else `None` (dateutil's `_to_decimal` raises then).
    pub fn parse(p: Props, s: &'a str) -> Option<Self> {
        let (a, b) = match s.find('.') {
            Some(i) => (&s[..i], Some(&s[i + 1..])),
            None => (s, None),
        };
        let digits = |x: &'a str| -> Option<Cow<'a, str>> {
            if x.is_empty() {
                return None;
            }
            if x.bytes().all(|c| c.is_ascii_digit()) {
                return Some(Cow::Borrowed(x));
            }
            x.chars()
                .map(|c| p.decimal_value(c).map(|d| char::from(b'0' + d)))
                .collect::<Option<String>>()
                .map(Cow::Owned)
        };
        Some(Self {
            int: digits(a)?,
            frac: match b {
                Some(b) => digits(b)?,
                None => Cow::Borrowed(""),
            },
        })
    }

    /// `int(d)` (truncation; the values are never negative).
    pub fn trunc(&self) -> Result<Int, Fail> {
        let b = self.int.as_bytes();
        if b.len() <= 18 {
            // No overflow: at most 18 digits.
            return Ok(b.iter().fold(0, |a, &d| a * 10 + Int::from(d - b'0')));
        }
        let mut acc = Some(0);
        for &d in b {
            acc = push_digit(acc, d - b'0');
        }
        acc.ok_or(Fail::Reference(Reason::BigNumber))
    }

    /// `bool(d % 1)`.
    pub fn has_fraction(&self) -> bool {
        self.frac.bytes().any(|b| b != b'0')
    }

    /// `d` compared with the integer `n`.
    pub fn cmp_int(&self, n: Int) -> Ordering {
        // Past i64 the value is larger than any `n`.
        let ord = self.trunc().map_or(Ordering::Greater, |v| v.cmp(&n));
        if ord == Ordering::Equal && self.has_fraction() {
            Ordering::Greater
        } else {
            ord
        }
    }

    /// Significant digits of the coefficient (leading zeros dropped), as the `decimal` context
    /// precision sees them.
    fn sig_digits(&self) -> usize {
        let int = self.int.trim_start_matches('0');
        if int.is_empty() {
            self.frac.trim_start_matches('0').len().max(1)
        } else {
            int.len() + self.frac.len()
        }
    }

    /// The `decimal` precision `d % 1` and `60 * (d % 1)` need to be exact; `None` past 28,
    /// where the kernel hands the call to Python.
    pub fn precision_needed(&self) -> Result<u32, Fail> {
        let n = self.sig_digits();
        if n > 26 {
            return Err(Fail::Reference(Reason::LongDecimal));
        }
        Ok(u32::try_from(n + 2).expect("small"))
    }

    /// `int(60 * (d % 1))`.
    pub fn frac_times_60(&self) -> Result<Int, Fail> {
        self.precision_needed()?;
        let f = self.frac.trim_end_matches('0');
        if f.is_empty() {
            return Ok(0);
        }
        let mut num: u128 = 0;
        for b in f.bytes() {
            num = num * 10 + u128::from(b - b'0');
        }
        let den = 10u128.pow(u32::try_from(f.len()).expect("<= 26"));
        Ok(Int::try_from(num * 60 / den).expect("< 60"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_0;

    #[test]
    fn ints() {
        let p = Props::new(V);
        assert_eq!(py_int(p, "0012", 4300), Ok(12));
        assert_eq!(py_int(p, " +1_2 ", 4300), Ok(12));
        assert_eq!(py_int(p, "\u{0663}\u{0664}", 4300), Ok(34));
        assert_eq!(py_int(p, "", 4300), Err(Fail::Invalid));
        assert_eq!(py_int(p, "1__2", 4300), Err(Fail::Invalid));
        assert_eq!(py_int(p, "_1", 4300), Err(Fail::Invalid));
        assert_eq!(py_int(p, "1_", 4300), Err(Fail::Invalid));
        assert_eq!(py_int(p, "²", 4300), Err(Fail::Invalid));
        assert_eq!(py_int(p, ".5", 4300), Err(Fail::Invalid));
        assert_eq!(py_int(p, "12345", 4), Err(Fail::Invalid));
        assert_eq!(
            py_int(p, &"9".repeat(20), 4300),
            Err(Fail::Reference(Reason::BigNumber))
        );
    }

    #[test]
    fn nums() {
        let p = Props::new(V);
        let n = Num::parse(p, "10.5").unwrap();
        assert_eq!(n.trunc(), Ok(10));
        assert!(n.has_fraction());
        assert_eq!(n.frac_times_60(), Ok(30));
        assert_eq!(n.cmp_int(10), Ordering::Greater);
        assert_eq!(n.cmp_int(11), Ordering::Less);
        assert_eq!(Num::parse(p, "12.00").unwrap().cmp_int(12), Ordering::Equal);
        assert!(Num::parse(p, "inf").is_none());
        assert!(Num::parse(p, "1.").is_none());
        assert!(Num::parse(p, "²").is_none());
        assert_eq!(Num::parse(p, "\u{0661}.\u{0665}").unwrap().frac, "5");
        assert_eq!(Num::parse(p, "0.999").unwrap().frac_times_60(), Ok(59));
    }

    #[test]
    fn slices() {
        assert_eq!(py_slice("abcdef", 2, Some(4)), "cd");
        assert_eq!(py_slice("abc", 2, Some(9)), "c");
        assert_eq!(py_slice("abc", 5, None), "");
        assert_eq!(py_slice("é1é2é3", 1, Some(3)), "1é");
        assert_eq!(py_slice("é1é2é3", 4, None), "é3");
        assert_eq!(py_find_dot("é12.5"), Some(3));
    }
}
