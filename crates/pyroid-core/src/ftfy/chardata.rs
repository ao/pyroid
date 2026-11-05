// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! Character classes and single-byte codecs (ftfy's `chardata.py`).

use super::UnicodeVersion;
use super::tables::{
    CLASS_PAGES, CLASSES, DECODE, ENCODE_1A, ENCODE_HIGH, ENCODING_NAMES, PAGE_INDEX,
};
use super::unicode_tables as ut;

/// Number of encodings in ftfy's `CHARMAP_ENCODINGS`.
pub(crate) const N_ENCODINGS: usize = ENCODING_NAMES.len();
/// Bit mask with one bit per encoding in `CHARMAP_ENCODINGS`.
pub(crate) const ALL_ENCODINGS: u16 = (1 << N_ENCODINGS) - 1;
/// Index of `latin-1` in `CHARMAP_ENCODINGS`.
pub(crate) const LATIN_1: usize = 0;
/// Index of strict `windows-1252` in `DECODE` (after the charmap encodings).
const WINDOWS_1252: usize = N_ENCODINGS;

/// The per-character data the hot loops need.
#[derive(Clone, Copy)]
pub(crate) struct CharClass {
    /// Explicit-set part of the `BADNESS_RE` Shift-And mask.
    pub bad: u128,
    /// `UTF8_DETECTOR_RE` clue bits (`CLUE_*`).
    pub clue: u8,
    /// Bit i set: encodable in `CHARMAP_ENCODINGS[i]`.
    pub enc: u16,
}

#[inline]
pub(crate) fn class_of(cp: u32) -> CharClass {
    let id = if cp < 0x10000 {
        let page = PAGE_INDEX[(cp >> 8) as usize];
        CLASS_PAGES[page as usize][(cp & 0xFF) as usize]
    } else {
        0
    };
    let (bad, clue, enc) = CLASSES[id as usize];
    CharClass { bad, clue, enc }
}

/// Name of encoding `i` in `CHARMAP_ENCODINGS`.
#[inline]
pub(crate) fn encoding_name(i: usize) -> &'static str {
    ENCODING_NAMES[i]
}

/// Whether `name` is a sloppy codec (`encoding.startswith("sloppy")`).
#[inline]
pub(crate) fn is_sloppy(i: usize) -> bool {
    ENCODING_NAMES[i].starts_with("sloppy")
}

/// Encode one code point in encoding `i`, if possible.
#[inline]
pub(crate) fn encode_cp(i: usize, cp: u32) -> Option<u8> {
    if cp < 0x80 {
        if cp == 0x1A && !ENCODE_1A[i] {
            return None;
        }
        return Some(cp as u8);
    }
    let table = ENCODE_HIGH[i];
    table
        .binary_search_by_key(&cp, |&(c, _)| c)
        .ok()
        .map(|k| table[k].1)
}

/// `text.encode(CHARMAP_ENCODINGS[i])`, or `None` if a character is not encodable.
pub(crate) fn encode(i: usize, text: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for cp in super::text::code_points(text) {
        out.push(encode_cp(i, cp)?);
    }
    Some(out)
}

/// Strict `windows-1252` decoding; `None` on an undefined byte.
pub(crate) fn decode_windows_1252(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let cp = DECODE[WINDOWS_1252][b as usize];
        if cp == u32::MAX {
            return None;
        }
        super::text::push_cp(&mut out, cp);
    }
    Some(out)
}

#[inline]
fn in_ranges(table: &[(u32, u32)], cp: u32) -> bool {
    match table.binary_search_by(|&(lo, _)| lo.cmp(&cp)) {
        Ok(_) => true,
        Err(0) => false,
        Err(k) => cp <= table[k - 1].1,
    }
}

/// Python `re`'s `\s` for `str` patterns (the same in every supported version).
#[inline]
pub(crate) fn is_space(cp: u32) -> bool {
    if cp < 0x80 {
        return matches!(cp, 0x09..=0x0D | 0x1C..=0x20);
    }
    in_ranges(&ut::SPACE, cp)
}

/// Python `re`'s `\w` for `str` patterns, in the given Unicode version.
#[inline]
pub(crate) fn is_word(cp: u32, v: UnicodeVersion) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_alphanumeric() || cp == u32::from(b'_');
    }
    in_ranges(
        match v {
            UnicodeVersion::V14_0 => &ut::WORD_14_0_0,
            UnicodeVersion::V15_0 => &ut::WORD_15_0_0,
            UnicodeVersion::V15_1 => &ut::WORD_15_1_0,
            UnicodeVersion::V16_0 => &ut::WORD_16_0_0,
            UnicodeVersion::V17_0 => &ut::WORD_17_0_0,
        },
        cp,
    )
}

/// Python `re`'s `\d` for `str` patterns, in the given Unicode version.
#[inline]
pub(crate) fn is_digit(cp: u32, v: UnicodeVersion) -> bool {
    if cp < 0x80 {
        return (cp as u8).is_ascii_digit();
    }
    in_ranges(
        match v {
            UnicodeVersion::V14_0 => &ut::DIGIT_14_0_0,
            UnicodeVersion::V15_0 => &ut::DIGIT_15_0_0,
            UnicodeVersion::V15_1 => &ut::DIGIT_15_1_0,
            UnicodeVersion::V16_0 => &ut::DIGIT_16_0_0,
            UnicodeVersion::V17_0 => &ut::DIGIT_17_0_0,
        },
        cp,
    )
}

/// Whether `cp` is assigned (general category is not `Cn`) in version `v`.
#[inline]
pub(crate) fn is_assigned(cp: u32, v: UnicodeVersion) -> bool {
    in_ranges(
        match v {
            UnicodeVersion::V14_0 => &ut::ASSIGNED_14_0_0,
            UnicodeVersion::V15_0 => &ut::ASSIGNED_15_0_0,
            UnicodeVersion::V15_1 => &ut::ASSIGNED_15_1_0,
            UnicodeVersion::V16_0 => &ut::ASSIGNED_16_0_0,
            UnicodeVersion::V17_0 => &ut::ASSIGNED_17_0_0,
        },
        cp,
    )
}

/// ftfy's `possible_encoding(text, encoding)` for every charmap encoding at
/// once, plus whether the text is ASCII.
///
/// Returns `(is_ascii, mask)` where bit i of `mask` is set iff `text` could
/// be encoded in `CHARMAP_ENCODINGS[i]`.
pub(crate) fn possible_encodings(text: &[u8]) -> (bool, u16) {
    let mut mask = ALL_ENCODINGS;
    let mut ascii = true;
    let mut i = 0;
    while i < text.len() {
        let b = text[i];
        if b < 0x80 {
            if b == 0x1A {
                mask &= class_of(0x1A).enc;
            }
            i = super::text::skip_ascii(text, i + 1, 0x1A);
            continue;
        }
        ascii = false;
        let (cp, n) = super::text::decode_at(text, i);
        mask &= class_of(cp).enc;
        if mask == 0 {
            return (false, 0);
        }
        i += n;
    }
    (ascii, mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_roundtrip() {
        for (i, table) in DECODE.iter().enumerate().take(N_ENCODINGS) {
            for b in 0..=255u8 {
                let cp = table[b as usize];
                assert_ne!(cp, u32::MAX);
                assert_eq!(
                    encode_cp(i, cp),
                    Some(b),
                    "{} byte {b:#x}",
                    encoding_name(i)
                );
                assert_ne!(class_of(cp).enc & (1 << i), 0);
            }
        }
    }

    #[test]
    fn sloppy_1a() {
        let sloppy = (0..N_ENCODINGS).filter(|&i| is_sloppy(i)).count();
        assert_eq!(sloppy, 6);
        assert_eq!(encode_cp(1, 0xFFFD), Some(0x1A));
        assert_eq!(encode_cp(1, 0x1A), None);
        assert_eq!(encode_cp(LATIN_1, 0x1A), Some(0x1A));
    }

    #[test]
    fn possible() {
        assert_eq!(possible_encodings(b"abc"), (true, ALL_ENCODINGS));
        let (ascii, m) = possible_encodings("caf\u{e9}".as_bytes());
        assert!(!ascii);
        assert_ne!(m & 1, 0);
        assert_eq!(possible_encodings("\u{4e2d}".as_bytes()).1, 0);
    }

    #[test]
    fn properties() {
        assert!(is_space(0x1F) && is_space(0x3000) && !is_space(0x200B));
        assert!(is_word(0xE9, UnicodeVersion::V15_0) && !is_word(0x2019, UnicodeVersion::V15_0));
        assert!(is_digit(0x0660, UnicodeVersion::V14_0));
        assert!(
            is_assigned(0x41, UnicodeVersion::V14_0) && !is_assigned(0x378, UnicodeVersion::V17_0)
        );
    }
}
