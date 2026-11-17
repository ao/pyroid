// Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
//! Per-character tables: `NGram.normalize` and Python's `str.isupper` per Unicode version.
//!
//! Text is handled as code points (`u32`), so lone surrogates (U+D800..U+DFFF), which a
//! Python `str` can hold, are ordinary characters here as they are in langdetect.

use std::sync::LazyLock;

use super::tables::{NORMALIZE_RUNS, UPPER_TABLES};

/// `' '`.
pub const SPACE: u32 = 0x20;
/// One past the last code point.
pub const MAX_CP: u32 = 0x11_0000;

/// `NGram.normalize` for the Basic Multilingual Plane; it changes nothing above it.
static NORMALIZE_BMP: LazyLock<Box<[u16]>> = LazyLock::new(|| {
    let mut table: Vec<u16> = Vec::with_capacity(0x1_0000);
    for &(start, end, kind, value) in NORMALIZE_RUNS {
        if start >= 0x1_0000 {
            break;
        }
        for cp in start..=end.min(0xFFFF) {
            let out = if kind == 0 { cp } else { value };
            // Every run below 0x10000 maps into the BMP (checked by generate_tables.py).
            #[allow(clippy::cast_possible_truncation)]
            table.push(out as u16);
        }
    }
    debug_assert_eq!(table.len(), 0x1_0000);
    table.into_boxed_slice()
});

/// `NGram.normalize(ch)`.
#[inline]
pub fn normalize(cp: u32) -> u32 {
    if cp < 0x1_0000 {
        // cp < 0x10000 indexes the 65,536-entry table.
        u32::from(NORMALIZE_BMP[cp as usize])
    } else {
        cp
    }
}

/// `str.isupper()` of one character, for one Unicode database.
#[derive(Clone, Debug)]
pub struct Upper {
    bits: Box<[u64]>,
}

impl Upper {
    /// The table for the Unicode version `version` (`unicodedata.unidata_version`), if
    /// there is one.
    pub fn for_version(version: &str) -> Option<Self> {
        UPPER_TABLES
            .iter()
            .find(|(v, _)| *v == version)
            .map(|(_, ranges)| Self::from_ranges(ranges))
    }

    /// The Unicode versions with a built-in table.
    pub fn versions() -> Vec<&'static str> {
        UPPER_TABLES.iter().map(|(v, _)| *v).collect()
    }

    /// A table from inclusive ranges of uppercase code points.
    pub fn from_ranges(ranges: &[(u32, u32)]) -> Self {
        let mut bits = vec![0u64; (MAX_CP as usize).div_ceil(64)];
        for &(start, end) in ranges {
            for cp in start..=end.min(MAX_CP - 1) {
                bits[(cp / 64) as usize] |= 1 << (cp % 64);
            }
        }
        Self {
            bits: bits.into_boxed_slice(),
        }
    }

    /// A table from one flag per code point (`flags[cp]`), e.g. computed by the interpreter.
    pub fn from_flags(flags: &[u8]) -> Self {
        let mut bits = vec![0u64; (MAX_CP as usize).div_ceil(64)];
        for (cp, &flag) in flags.iter().enumerate().take(MAX_CP as usize) {
            if flag != 0 {
                bits[cp / 64] |= 1 << (cp % 64);
            }
        }
        Self {
            bits: bits.into_boxed_slice(),
        }
    }

    /// `chr(cp).isupper()`.
    #[inline]
    pub fn is_upper(&self, cp: u32) -> bool {
        self.bits
            .get((cp / 64) as usize)
            .is_some_and(|w| w & (1 << (cp % 64)) != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_examples() {
        // Basic Latin: letters stay, everything else is a space.
        assert_eq!(normalize(u32::from('a')), u32::from('a'));
        assert_eq!(normalize(u32::from('1')), SPACE);
        assert_eq!(normalize(u32::from('[')), SPACE);
        // Latin-1 exclusions, Romanian comma-below, Farsi yeh, kana, hangul.
        assert_eq!(normalize(0xAB), SPACE);
        assert_eq!(normalize(0xE9), 0xE9);
        assert_eq!(normalize(0x219), 0x15F);
        assert_eq!(normalize(0x6CC), 0x64A);
        assert_eq!(normalize(0x3044), 0x3042);
        assert_eq!(normalize(0x30AB), 0x30A2);
        assert_eq!(normalize(0xD55C), 0xAC00);
        assert_eq!(normalize(0x1EA1), 0x1EC3);
        assert_eq!(normalize(0x2014), SPACE);
        // Surrogates and astral code points are unchanged.
        assert_eq!(normalize(0xD800), 0xD800);
        assert_eq!(normalize(0x1F600), 0x1F600);
    }

    #[test]
    fn upper_examples() {
        for v in Upper::versions() {
            let u = Upper::for_version(v).unwrap();
            assert!(u.is_upper(u32::from('A')));
            assert!(!u.is_upper(u32::from('a')));
            assert!(u.is_upper(0x24B6)); // CIRCLED LATIN CAPITAL LETTER A (Other_Uppercase)
            assert!(!u.is_upper(0xD800));
            assert!(!u.is_upper(MAX_CP));
        }
        assert!(Upper::for_version("1.0.0").is_none());
    }
}
