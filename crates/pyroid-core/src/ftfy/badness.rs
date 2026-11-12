// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! The mojibake heuristic (ftfy's `badness.py`).
//!
//! `BADNESS_RE` is an alternation of 37 fixed-width sequences of character
//! classes (one has an optional `\s`, which the table generator expands into
//! two). Instead of a regex engine, every alternative is laid out as a run of
//! bit positions in one `u128`, and the text is scanned once with the
//! bit-parallel Shift-And algorithm: after each character, bit `p` of the
//! state is set iff some alternative matches up to its position `p` ending at
//! that character.

use super::UnicodeVersion;
use super::chardata::{class_of, is_space, is_word};
use super::tables::{
    BAD_ALTS, BAD_ANCHORED, BAD_ANY, BAD_ASCII, BAD_FINAL, BAD_NOTALPHA, BAD_SPACE, BAD_START,
    BAD_WORD,
};
use super::text::{decode_at, skip_ascii};

/// Every `BADNESS_RE` alternative has at least one non-ASCII element and is
/// at most this long, so a match lies within this many characters of a
/// non-ASCII character.
const MAX_ALT_LEN: usize = 8;

/// The Shift-And mask for one character, given the positions it may extend.
#[inline]
fn char_mask(cp: u32, wanted: u128, v: UnicodeVersion) -> u128 {
    let mut m = class_of(cp).bad;
    if cp != u32::from(b'\n') {
        m |= BAD_ANY;
    }
    if !u8::try_from(cp).is_ok_and(|b| b.is_ascii_alphabetic()) {
        m |= BAD_NOTALPHA;
    }
    // `\s` and `\w` cost a table search, so only test them when they matter.
    if wanted & BAD_SPACE != 0 && is_space(cp) {
        m |= BAD_SPACE;
    }
    if wanted & BAD_WORD != 0 && is_word(cp, v) {
        m |= BAD_WORD;
    }
    m
}

/// Complete masks for every two-byte character U+0080..U+07FF. `\s` and `\w`
/// agree across the supported versions there (checked by a unit test).
fn two_byte_masks() -> &'static [u128] {
    static TABLE: std::sync::OnceLock<Vec<u128>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        (0x80..0x800)
            .map(|cp| char_mask(cp, u128::MAX, UnicodeVersion::LATEST))
            .collect()
    })
}

/// ftfy's `is_bad`: whether `BADNESS_RE` matches anywhere in `text`.
pub(crate) fn is_bad(text: &[u8], v: UnicodeVersion) -> bool {
    let two = two_byte_masks();
    let mut state: u128 = 0;
    let mut start = BAD_START | BAD_ANCHORED;
    let mut i = 0;
    while i < text.len() {
        let b = text[i];
        if state == 0 && b < 0x80 && i > 0 {
            // Nothing is in progress: jump to just before the next non-ASCII
            // character, since a match must include one.
            let next = skip_ascii(text, i, 0x80);
            if next == text.len() {
                return false;
            }
            i = i.max(next.saturating_sub(MAX_ALT_LEN));
        }
        let b = text[i];
        let wanted = (state << 1) | start;
        if b < 0x80 {
            state = wanted & BAD_ASCII[b as usize];
            i += 1;
        } else if b < 0xE0 {
            let cp = (u32::from(b & 0x1F) << 6) | u32::from(text[i + 1] & 0x3F);
            state = wanted & two[(cp - 0x80) as usize];
            i += 2;
        } else {
            let (cp, n) = decode_at(text, i);
            state = wanted & char_mask(cp, wanted, v);
            i += n;
        }
        if state & BAD_FINAL != 0 {
            return true;
        }
        start = BAD_START;
    }
    false
}

/// ftfy's `badness`: the number of non-overlapping `BADNESS_RE` matches, as
/// counted by `len(BADNESS_RE.findall(text))`.
pub(crate) fn badness(text: &[u8], v: UnicodeVersion) -> usize {
    let mut masks = Vec::new();
    let mut i = 0;
    while i < text.len() {
        let (cp, n) = decode_at(text, i);
        masks.push(char_mask(cp, u128::MAX, v));
        i += n;
    }
    let mut count = 0;
    let mut pos = 0;
    while pos < masks.len() {
        // sre tries the alternatives in order and takes the first that matches.
        let hit = BAD_ALTS.iter().find_map(|&(first, len, anchored)| {
            let n = usize::from(len);
            if (anchored && pos != 0) || pos + n > masks.len() {
                return None;
            }
            (0..len)
                .all(|j| masks[pos + usize::from(j)] & (1u128 << (first + j)) != 0)
                .then_some(n)
        });
        match hit {
            Some(len) => {
                count += 1;
                pos += len;
            }
            None => pos += 1,
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_0;

    fn bad(s: &str) -> bool {
        is_bad(s.as_bytes(), V)
    }

    #[test]
    fn known_mojibake() {
        assert!(bad("Ã©"));
        assert!(bad("â€œ"));
        assert!(bad("\u{80}"));
        assert!(bad("Ã "));
        assert!(bad("xÃ "));
        assert!(bad("вЂ"));
        assert!(bad("ГўВЂВ\u{9d}a"));
    }

    #[test]
    fn clean_text() {
        assert!(!bad(""));
        assert!(!bad("hello world"));
        assert!(!bad("café"));
        assert!(!bad("naïve façade"));
        assert!(!bad("日本語のテキスト"));
        assert!(!bad("ГўВЂВ\na"));
    }

    #[test]
    fn anchored_only_at_start() {
        assert!(bad("Ã x"));
        assert!(!bad("1Â x"));
    }

    #[test]
    fn every_alternative_needs_non_ascii() {
        // Justifies the ASCII skipping in `is_bad`.
        for &(first, len, _) in &BAD_ALTS {
            assert!(usize::from(len) <= MAX_ALT_LEN);
            let ascii_only = (0..u32::from(len)).all(|j| {
                let bit = 1u128 << (u32::from(first) + j);
                BAD_ASCII.iter().any(|m| m & bit != 0)
            });
            assert!(
                !ascii_only,
                "alternative at bit {first} can match pure ASCII"
            );
        }
    }

    #[test]
    fn two_byte_masks_are_version_independent() {
        let versions = [
            UnicodeVersion::V14_0,
            UnicodeVersion::V15_0,
            UnicodeVersion::V15_1,
            UnicodeVersion::V16_0,
            UnicodeVersion::V17_0,
        ];
        let two = two_byte_masks();
        for cp in 0x80..0x800u32 {
            for v in versions {
                assert_eq!(
                    two[(cp - 0x80) as usize],
                    char_mask(cp, u128::MAX, v),
                    "{cp:#x} {v:?}"
                );
            }
        }
    }

    #[test]
    fn ascii_table_matches_char_mask() {
        for b in 0..0x80u32 {
            assert_eq!(BAD_ASCII[b as usize], char_mask(b, u128::MAX, V), "{b:#x}");
        }
    }

    #[test]
    fn counts() {
        assert_eq!(badness(b"", V), 0);
        // `Ã©Ã` matches `[A U][S E]\w` first, leaving `©` alone.
        assert_eq!(badness("Ã©Ã©".as_bytes(), V), 1);
        assert_eq!(badness("\u{80}\u{81}".as_bytes(), V), 2);
    }
}
