//! One pass over a segment that tells which fixers could change it.
//!
//! Every fixer in ftfy only acts on specific characters. Scanning once for all
//! of them lets the pipeline skip the fixers that cannot fire, without
//! changing any result: a skipped fixer would have returned its input.

use super::chardata::{ALL_ENCODINGS, class_of};
use super::tables::{
    CLUE_FIRST2, CLUE_FIRST3, CLUE_FIRST4, ENCODE_1A, NFC_ACTIVE_ASTRAL, NFC_ACTIVE_BMP,
};
use super::text::decode_at;

pub(crate) const NON_ASCII: u32 = 1 << 0;
/// `&` (HTML entities).
pub(crate) const AMP: u32 = 1 << 1;
/// ESC (terminal escapes).
pub(crate) const ESC: u32 = 1 << 2;
/// CR, U+0085, U+2028, U+2029 (line breaks).
pub(crate) const LINE_BREAK: u32 = 1 << 3;
/// A character `remove_control_chars` deletes.
pub(crate) const CONTROL: u32 = 1 << 4;
/// U+0080..U+009F.
pub(crate) const C1: u32 = 1 << 5;
/// A Latin ligature.
pub(crate) const LIGATURE: u32 = 1 << 6;
/// U+3000 or a halfwidth/fullwidth form.
pub(crate) const WIDTH: u32 = 1 << 7;
/// A curly quote.
pub(crate) const CURLY: u32 = 1 << 8;
/// A lone surrogate.
pub(crate) const SURROGATE: u32 = 1 << 9;
/// A character that can make NFC change the text.
pub(crate) const NFC_ACTIVE: u32 = 1 << 10;
/// A character that can start a `UTF8_DETECTOR_RE` match.
pub(crate) const UTF8_CLUE: u32 = 1 << 11;

/// What [`scan`] found.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Features {
    pub bits: u32,
    /// `possible_encoding` for every charmap encoding (bit i = encoding i).
    pub encodings: u16,
}

impl Features {
    #[inline]
    pub(crate) fn has(self, bits: u32) -> bool {
        self.bits & bits != 0
    }

    /// Whether `fix_encoding` could change the text. If the text is ASCII,
    /// or no encoding is possible, no clue character starts a mojibake
    /// sequence and there is no C1 control, every branch of ftfy's
    /// `_fix_encoding_one_step_and_explain` returns the text unchanged.
    #[inline]
    pub(crate) fn encoding_candidate(self) -> bool {
        self.has(NON_ASCII) && (self.encodings != 0 || self.has(UTF8_CLUE | C1))
    }
}

/// Features of the ASCII bytes.
static ASCII: [u32; 128] = {
    let mut t = [0u32; 128];
    let mut b = 0;
    while b < 0x20 {
        t[b] = match b {
            0x09 | 0x0A | 0x0C => 0,
            0x0D => LINE_BREAK,
            0x1B => ESC | CONTROL,
            _ => CONTROL,
        };
        b += 1;
    }
    t[b'&' as usize] = AMP;
    t[0x7F] = CONTROL;
    t
};

#[inline]
fn nfc_active(cp: u32) -> bool {
    if cp < 0x10000 {
        NFC_ACTIVE_BMP[(cp >> 6) as usize] & (1 << (cp & 63)) != 0
    } else {
        match NFC_ACTIVE_ASTRAL.binary_search_by(|&(lo, _)| lo.cmp(&cp)) {
            Ok(_) => true,
            Err(0) => false,
            Err(k) => cp <= NFC_ACTIVE_ASTRAL[k - 1].1,
        }
    }
}

#[inline]
fn non_ascii_bits(cp: u32) -> u32 {
    let fixed = match cp {
        0x85 => C1 | LINE_BREAK,
        0x80..=0x9F => C1,
        0x0132 | 0x0133 | 0x0149 | 0x01C4..=0x01CC | 0x01F1..=0x01F3 | 0xFB00..=0xFB06 => LIGATURE,
        0x02BC | 0x2018..=0x201F => CURLY,
        0x2028 | 0x2029 => LINE_BREAK,
        0x206A..=0x206F | 0xFEFF | 0xFFF9..=0xFFFC => CONTROL,
        0x3000 | 0xFF01..=0xFFEF => WIDTH,
        0xD800..=0xDFFF => SURROGATE,
        _ => 0,
    };
    let clue = class_of(cp).clue & (CLUE_FIRST2 | CLUE_FIRST3 | CLUE_FIRST4) != 0;
    fixed
        | NON_ASCII
        | if clue { UTF8_CLUE } else { 0 }
        | if nfc_active(cp) { NFC_ACTIVE } else { 0 }
}

/// Scan `text` once.
pub(crate) fn scan(text: &[u8]) -> Features {
    let mut bits = 0;
    let mut encodings = ALL_ENCODINGS;
    let mut i = 0;
    while i < text.len() {
        let b = text[i];
        if b < 0x80 {
            bits |= ASCII[b as usize];
            if b == 0x1A {
                for (k, ok) in ENCODE_1A.iter().enumerate() {
                    if !ok {
                        encodings &= !(1 << k);
                    }
                }
            }
            i += 1;
            continue;
        }
        let (cp, n) = decode_at(text, i);
        bits |= non_ascii_bits(cp);
        encodings &= class_of(cp).enc;
        i += n;
    }
    Features { bits, encodings }
}

/// Whether `text` is ASCII with nothing any fixer acts on (no `&`, no CR, no
/// control characters other than TAB, LF and FF). Such text is returned
/// unchanged by every configuration.
pub(crate) fn is_quiet_ascii(text: &[u8]) -> bool {
    // A branch-free fold the compiler can vectorize, with an early exit
    // between chunks.
    text.chunks(4096).all(|chunk| {
        !chunk.iter().fold(false, |acc, &b| {
            let loud =
                b >= 0x7F || b == b'&' || (b < 0x20 && b != b'\t' && b != b'\n' && b != 0x0C);
            acc | loud
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features() {
        assert!(is_quiet_ascii(b"plain text\twith tabs\nand lines\x0c"));
        assert!(!is_quiet_ascii(b"a&b"));
        assert!(!is_quiet_ascii(b"a\rb"));
        assert!(!is_quiet_ascii("é".as_bytes()));
        let f = scan("x\u{2019}y".as_bytes());
        assert!(f.has(CURLY | NON_ASCII) && !f.has(NFC_ACTIVE));
        assert!(scan("e\u{301}".as_bytes()).has(NFC_ACTIVE));
        assert!(scan("\u{1161}".as_bytes()).has(NFC_ACTIVE));
        assert!(!scan("日本".as_bytes()).encoding_candidate());
        assert!(scan("Ã©".as_bytes()).encoding_candidate());
    }
}
