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
use super::text::decode_at;

/// The Shift-And mask for one character, given the positions it may extend.
#[inline]
fn char_mask(cp: u32, wanted: u128, v: UnicodeVersion) -> u128 {
    let mut m = class_of(cp).bad;
    if cp != u32::from(b'\n') {
        m |= BAD_ANY;
    }
    if !(cp < 0x80 && (cp as u8).is_ascii_alphabetic()) {
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

/// ftfy's `is_bad`: whether `BADNESS_RE` matches anywhere in `text`.
pub(crate) fn is_bad(text: &[u8], v: UnicodeVersion) -> bool {
    let mut state: u128 = 0;
    let mut start = BAD_START | BAD_ANCHORED;
    let mut i = 0;
    while i < text.len() {
        let b = text[i];
        let wanted = (state << 1) | start;
        if b < 0x80 {
            state = wanted & BAD_ASCII[b as usize];
            i += 1;
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
            let (first, len) = (u32::from(first), usize::from(len));
            if (anchored && pos != 0) || pos + len > masks.len() {
                return None;
            }
            (0..len)
                .all(|j| masks[pos + j] & (1u128 << (first + j as u32)) != 0)
                .then_some(len)
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
