//! Character classes as NLTK 3.10.4 sees them.
//!
//! NLTK compiles its tokenizer patterns with the third-party `regex` module
//! (`nltk.redos`), except the Treebank contraction patterns, which use the
//! standard `re`. Around the patterns it calls `str` methods. Each predicate
//! here reproduces one of those tests exactly:
//!
//! | function        | Python                                              |
//! |-----------------|-----------------------------------------------------|
//! | [`rx_word`]     | `regex` 2026.9.29's `\w` (its own Unicode tables)   |
//! | [`rx_digit`]    | `regex` 2026.9.29's `\d`                            |
//! | [`rx_space`]    | `regex`'s `\s`: `str.isspace()` minus U+001C..U+001F |
//! | [`py_space`]    | `str.isspace()` (the same on every CPython)         |
//! | [`py_upper`]    | `c.isupper()` for one character, per Unicode version |
//! | [`py_lower`]    | `c.islower()` for one character, per Unicode version |
//! | [`re_word`]     | `re`'s `\w`, per Unicode version                    |
//! | [`rx_ci`]       | `regex`'s `(?i)` match of one ASCII letter          |
//! | [`re_ci`]       | `re`'s `(?i)` match of one ASCII letter             |
//!
//! `generate_tables.py` dumps the tables from `regex` and from each CPython,
//! and checks the hard-coded facts (`rx_space`, the `(?i)` sets) on every
//! interpreter.

use std::borrow::Cow;

use crate::ftfy::UnicodeVersion;

#[rustfmt::skip]
#[allow(clippy::unreadable_literal, clippy::match_same_arms)] // generated
#[path = "tables.rs"]
mod tables;

#[inline]
fn in_ranges(ranges: &[(u32, u32)], cp: u32) -> bool {
    let after = ranges.partition_point(|&(lo, _)| lo <= cp);
    ranges
        .get(after.wrapping_sub(1))
        .is_some_and(|&(_, hi)| cp <= hi)
}

#[inline]
fn has(c: char, ascii: u128, ranges: &[(u32, u32)]) -> bool {
    let cp = u32::from(c);
    if cp < 0x80 {
        (ascii >> cp) & 1 != 0
    } else {
        in_ranges(ranges, cp)
    }
}

/// `regex`'s `\w`: letters, marks, decimal digits, connector punctuation and
/// the joiners (so U+0301 is a word character; `²` and `½` are not).
#[inline]
pub(crate) fn rx_word(c: char) -> bool {
    has(c, tables::RX_WORD_ASCII, &tables::RX_WORD)
}

/// `regex`'s `\d`: decimal digits (general category Nd) in `regex`'s Unicode version.
#[inline]
pub(crate) fn rx_digit(c: char) -> bool {
    has(c, tables::RX_DIGIT_ASCII, &tables::RX_DIGIT)
}

/// `regex`'s `\s`: Unicode White_Space, which is `str.isspace()` without the
/// four information separators U+001C..U+001F.
#[inline]
pub(crate) fn rx_space(c: char) -> bool {
    py_space(c) && !matches!(c, '\u{1c}'..='\u{1f}')
}

/// `c.isspace()` in Python.
#[inline]
pub(crate) fn py_space(c: char) -> bool {
    crate::text::is_py_whitespace(c)
}

/// `c.isupper()` for a one-character string (CPython's `Py_UNICODE_ISUPPER`).
#[inline]
pub(crate) fn py_upper(c: char, v: UnicodeVersion) -> bool {
    has(c, tables::UPPER_ASCII, tables::upper(v))
}

/// `c.islower()` for a one-character string (CPython's `Py_UNICODE_ISLOWER`).
#[inline]
pub(crate) fn py_lower(c: char, v: UnicodeVersion) -> bool {
    has(c, tables::LOWER_ASCII, tables::lower(v))
}

/// `re`'s `\w` (str pattern, default flags) in CPython's Unicode version `v`.
#[inline]
pub(crate) fn re_word(c: char, v: UnicodeVersion) -> bool {
    crate::filters::unicode::is_word(c, v)
}

/// `s.lower()` in CPython's Unicode version `v` (borrowed when unchanged).
pub fn lower(s: &str, v: UnicodeVersion) -> Cow<'_, str> {
    crate::filters::url::py_lower(s, v)
}

/// Whether `regex`'s `(?i)` pattern `letter` (lowercase ASCII) matches `c`:
/// the letter, its uppercase, and U+0130 for `i`, U+017F for `s`.
#[inline]
pub(crate) fn rx_ci(c: char, letter: u8) -> bool {
    debug_assert!(letter.is_ascii_lowercase());
    match u32::from(c) {
        cp if cp < 0x80 => u8::try_from(cp).is_ok_and(|b| b.to_ascii_lowercase() == letter),
        0x130 => letter == b'i',
        0x17f => letter == b's',
        _ => false,
    }
}

/// Whether `re`'s `(?i)` pattern `letter` (lowercase ASCII) matches `c`:
/// as [`rx_ci`], plus U+0131 (dotless i) for `i`.
#[inline]
pub(crate) fn re_ci(c: char, letter: u8) -> bool {
    rx_ci(c, letter) || (c == '\u{131}' && letter == b'i')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert!(rx_word('a') && rx_word('_') && rx_word('\u{301}') && rx_word('7'));
        assert!(!rx_word('\u{b2}') && !rx_word('-') && !rx_word(' '));
        assert!(rx_digit('7') && rx_digit('\u{663}') && !rx_digit('\u{b2}'));
        assert!(rx_space('\u{85}') && rx_space('\u{3000}') && !rx_space('\u{1c}'));
        assert!(py_space('\u{1c}') && !py_space('\u{200b}'));
        let v = UnicodeVersion::LATEST;
        assert!(py_upper('A', v) && py_upper('\u{24b6}', v) && !py_upper('a', v));
        assert!(py_lower('a', v) && py_lower('\u{aa}', v) && !py_lower('A', v));
        assert!(re_word('\u{b2}', v) && !re_word('\u{301}', v));
        assert!(rx_ci('I', b'i') && rx_ci('\u{130}', b'i') && !rx_ci('\u{131}', b'i'));
        assert!(re_ci('\u{131}', b'i') && re_ci('\u{17f}', b's') && !re_ci('k', b's'));
        assert_eq!(lower("ABC\u{130}", v), "abci\u{307}");
    }
}
