// Reproduces CPython 3.11-3.15 behaviour (PSF-2.0); character data from each version's
// unicodedata.
//! The Python `str` and `re` semantics sqlparse relies on, for the caller's Unicode version:
//! `upper`, `lower` (final sigma), `capitalize`, `strip`, `splitlines`, `len`, and the data
//! `re.IGNORECASE` compares with (`_sre.unicode_tolower`, `_sre.unicode_iscased`,
//! `re._casefix`).
//!
//! `\w`, `\d` and `isspace` come from [`crate::filters::unicode`] and [`crate::text`]; the rest
//! from `tables.rs` (`generate_tables.py`).

use std::borrow::Cow;

use super::tables::{self, Full};
use crate::ftfy::UnicodeVersion;

/// Whether `c` ends a line for `str.splitlines()`.
#[inline]
pub fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\x0b'
            | '\x0c'
            | '\r'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `s.splitlines()` (no keepends).
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let mut start = 0;
    let mut it = s.char_indices();
    while let Some((i, c)) = it.next() {
        if is_line_break(c) {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && b.get(end) == Some(&b'\n') {
                it.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

#[derive(Clone, Copy)]
enum Map {
    Upper,
    Lower,
    Title,
}

/// The delta tables of `v` against Unicode 17.0 (none for 17.0 itself).
fn full_delta(m: Map, v: UnicodeVersion) -> &'static [Full] {
    use UnicodeVersion as U;
    match (m, v) {
        (_, U::V17_0) => &[],
        (Map::Upper, U::V14_0) => &tables::UPPER_DELTA_14_0,
        (Map::Upper, U::V15_0) => &tables::UPPER_DELTA_15_0,
        (Map::Upper, U::V15_1) => &tables::UPPER_DELTA_15_1,
        (Map::Upper, U::V16_0) => &tables::UPPER_DELTA_16_0,
        (Map::Lower, U::V14_0) => &tables::LOWER_DELTA_14_0,
        (Map::Lower, U::V15_0) => &tables::LOWER_DELTA_15_0,
        (Map::Lower, U::V15_1) => &tables::LOWER_DELTA_15_1,
        (Map::Lower, U::V16_0) => &tables::LOWER_DELTA_16_0,
        (Map::Title, U::V14_0) => &tables::TITLE_DELTA_14_0,
        (Map::Title, U::V15_0) => &tables::TITLE_DELTA_15_0,
        (Map::Title, U::V15_1) => &tables::TITLE_DELTA_15_1,
        (Map::Title, U::V16_0) => &tables::TITLE_DELTA_16_0,
    }
}

fn lookup_full(c: char, m: Map, v: UnicodeVersion) -> Option<&'static Full> {
    let cp = u32::from(c);
    let delta = full_delta(m, v);
    if let Ok(i) = delta.binary_search_by_key(&cp, |e| e.0) {
        return Some(&delta[i]);
    }
    let base: &[Full] = match m {
        Map::Upper => &tables::UPPER,
        Map::Lower => &tables::LOWER,
        Map::Title => &tables::TITLE,
    };
    base.binary_search_by_key(&cp, |e| e.0)
        .ok()
        .map(|i| &base[i])
}

fn push_full(out: &mut String, e: &Full) {
    for cp in [e.1, e.2, e.3] {
        if cp != 0 {
            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
        }
    }
}

fn push_mapped(out: &mut String, c: char, m: Map, v: UnicodeVersion) {
    match lookup_full(c, m, v) {
        Some(e) => push_full(out, e),
        None => out.push(c),
    }
}

/// `s.upper()`.
pub fn upper(s: &str, v: UnicodeVersion) -> Cow<'_, str> {
    if s.is_ascii() {
        if !s.bytes().any(|b| b.is_ascii_lowercase()) {
            return Cow::Borrowed(s);
        }
        return Cow::Owned(s.to_ascii_uppercase());
    }
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        push_mapped(&mut out, c, Map::Upper, v);
    }
    if out == s {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(out)
    }
}

fn in_ranges(t: &[(u32, u32)], c: char) -> bool {
    let cp = u32::from(c);
    let after = t.partition_point(|&(lo, _)| lo <= cp);
    after > 0 && cp <= t[after - 1].1
}

fn cased(c: char, v: UnicodeVersion) -> bool {
    use UnicodeVersion as U;
    let t: &[(u32, u32)] = match v {
        U::V14_0 => tables::CASED_14_0,
        U::V15_0 => tables::CASED_15_0,
        U::V15_1 => tables::CASED_15_1,
        U::V16_0 => tables::CASED_16_0,
        U::V17_0 => tables::CASED_17_0,
    };
    in_ranges(t, c)
}

fn case_ignorable(c: char, v: UnicodeVersion) -> bool {
    use UnicodeVersion as U;
    let t: &[(u32, u32)] = match v {
        U::V14_0 => tables::CASE_IGNORABLE_14_0,
        U::V15_0 => tables::CASE_IGNORABLE_15_0,
        U::V15_1 => tables::CASE_IGNORABLE_15_1,
        U::V16_0 => tables::CASE_IGNORABLE_16_0,
        U::V17_0 => tables::CASE_IGNORABLE_17_0,
    };
    in_ranges(t, c)
}

/// `s.lower()`, with CPython's Final_Sigma rule for U+03A3.
pub fn lower(s: &str, v: UnicodeVersion) -> Cow<'_, str> {
    if s.is_ascii() {
        if !s.bytes().any(|b| b.is_ascii_uppercase()) {
            return Cow::Borrowed(s);
        }
        return Cow::Owned(s.to_ascii_lowercase());
    }
    let mut out = String::with_capacity(s.len());
    if s.contains('\u{3a3}') {
        let cs: Vec<char> = s.chars().collect();
        for (i, &c) in cs.iter().enumerate() {
            if c != '\u{3a3}' {
                push_mapped(&mut out, c, Map::Lower, v);
                continue;
            }
            // handle_capital_sigma:
            // \p{cased}\p{case-ignorable}* U+03A3 !(\p{case-ignorable}* \p{cased})
            let before = cs[..i].iter().rev().find(|&&p| !case_ignorable(p, v));
            let mut fin = before.is_some_and(|&p| cased(p, v));
            if fin {
                let after = cs[i + 1..].iter().find(|&&p| !case_ignorable(p, v));
                fin = after.is_none_or(|&p| !cased(p, v));
            }
            out.push(if fin { '\u{3c2}' } else { '\u{3c3}' });
        }
    } else {
        for c in s.chars() {
            push_mapped(&mut out, c, Map::Lower, v);
        }
    }
    if out == s {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(out)
    }
}

/// `s.capitalize()`: the first character's title-case mapping, then the rest lowered (the
/// final-sigma context of the rest still sees the original first character).
pub fn capitalize(s: &str, v: UnicodeVersion) -> Cow<'_, str> {
    let Some(first) = s.chars().next() else {
        return Cow::Borrowed(s);
    };
    let lowered = lower(s, v);
    // The first character's own lowercase mapping: everything after it is the rest's.
    let mut head = String::new();
    head.push(first);
    let first_lower = lower(&head, v);
    let rest = &lowered[first_lower.len()..];
    let mut out = String::with_capacity(s.len() + 4);
    push_mapped(&mut out, first, Map::Title, v);
    out.push_str(rest);
    if out == s {
        Cow::Borrowed(s)
    } else {
        Cow::Owned(out)
    }
}

/// Whether `s.upper() == target`, for an ASCII `target`, without building the upper-case
/// string in the common cases.
pub fn upper_eq(s: &str, target: &str, v: UnicodeVersion) -> bool {
    if s.is_ascii() {
        return s.len() == target.len()
            && s.bytes()
                .zip(target.bytes())
                .all(|(a, b)| a.to_ascii_uppercase() == b);
    }
    // No character upper-cases to nothing, so more characters than `target` can never match.
    if s.chars().count() > target.len() {
        return false;
    }
    upper(s, v) == target
}

/// `len(s)` (code points).
#[inline]
pub fn py_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// `s.strip()`.
#[inline]
pub fn strip(s: &str) -> &str {
    s.trim_matches(crate::text::is_py_whitespace)
}

/// `s.rstrip()`.
#[inline]
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(crate::text::is_py_whitespace)
}

/// `s.isspace()` of one character (`re`'s `\s`).
#[inline]
pub fn is_space(c: char) -> bool {
    crate::text::is_py_whitespace(c)
}

/// `re`'s `\w` (str pattern): `c.isalnum()` or `_`.
#[inline]
pub fn is_word(c: char, v: UnicodeVersion) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_';
    }
    crate::filters::unicode::is_word(c, v)
}

/// `re`'s `\d` (str pattern): `c.isdecimal()`.
#[inline]
pub fn is_digit(c: char, v: UnicodeVersion) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    crate::filters::unicode::is_digit(c, v)
}

/// `_sre.unicode_tolower(c)`: the simple lowercase mapping.
pub fn simple_lower(c: char, v: UnicodeVersion) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    let cp = u32::from(c);
    let delta: &[(u32, u32)] = match v {
        UnicodeVersion::V14_0 => &tables::SLOWER_DELTA_14_0,
        UnicodeVersion::V15_0 => &tables::SLOWER_DELTA_15_0,
        UnicodeVersion::V15_1 => &tables::SLOWER_DELTA_15_1,
        UnicodeVersion::V16_0 => &tables::SLOWER_DELTA_16_0,
        UnicodeVersion::V17_0 => &[],
    };
    let hit = match delta.binary_search_by_key(&cp, |e| e.0) {
        Ok(i) => Some(delta[i].1),
        Err(_) => tables::SLOWER
            .binary_search_by_key(&cp, |e| e.0)
            .ok()
            .map(|i| tables::SLOWER[i].1),
    };
    hit.and_then(char::from_u32).unwrap_or(c)
}

/// `_sre.unicode_iscased(c)`.
pub fn is_cased(c: char, v: UnicodeVersion) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    let t: &[(u32, u32)] = match v {
        UnicodeVersion::V14_0 => tables::ISCASED_14_0,
        UnicodeVersion::V15_0 => tables::ISCASED_15_0,
        UnicodeVersion::V15_1 => tables::ISCASED_15_1,
        UnicodeVersion::V16_0 => tables::ISCASED_16_0,
        UnicodeVersion::V17_0 => tables::ISCASED_17_0,
    };
    let cp = u32::from(c);
    let after = t.partition_point(|&(lo, _)| lo <= cp);
    after > 0 && cp <= t[after - 1].1
}

/// `re._casefix._EXTRA_CASES[lo]`: the other lowercase characters sharing `lo`'s uppercase.
pub fn extra_cases(lo: char) -> impl Iterator<Item = char> {
    let cp = u32::from(lo);
    let e = tables::EXTRA_CASES
        .binary_search_by_key(&cp, |e| e.0)
        .map_or((0, 0, 0), |i| tables::EXTRA_CASES[i]);
    [e.1, e.2]
        .into_iter()
        .filter(|&x| x != 0)
        .filter_map(char::from_u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::LATEST;

    #[test]
    fn case_mappings() {
        assert_eq!(upper("select", V), "SELECT");
        assert_eq!(upper("ſelect", V), "SELECT");
        assert_eq!(upper("stra\u{df}e", V), "STRASSE");
        assert_eq!(upper("\u{fb01}x", V), "FIx".to_uppercase());
        assert_eq!(capitalize("hELLO", V), "Hello");
        assert_eq!(capitalize("\u{1c6}x", V), "\u{1c5}x");
        assert_eq!(capitalize("", V), "");
        assert_eq!(capitalize("\u{130}A", V), "\u{130}a");
        assert_eq!(lower("ΑΣ", V), "ας");
        assert!(upper_eq("ſelect", "SELECT", V));
        assert!(upper_eq("create", "CREATE", V));
        assert!(!upper_eq("createx", "CREATE", V));
    }

    #[test]
    fn ignorecase_data() {
        assert_eq!(simple_lower('\u{130}', V), 'i');
        assert_eq!(simple_lower('\u{212a}', V), 'k');
        assert_eq!(simple_lower('A', V), 'a');
        assert!(is_cased('\u{17f}', V) && is_cased('a', V) && !is_cased('1', V));
        assert_eq!(extra_cases('i').collect::<Vec<_>>(), ['\u{131}']);
        assert_eq!(extra_cases('s').collect::<Vec<_>>(), ['\u{17f}']);
        assert_eq!(extra_cases('a').count(), 0);
    }
}
