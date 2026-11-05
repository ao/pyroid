// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! Byte-level helpers for "surrogate-tolerant UTF-8".
//!
//! ftfy works on Python `str`, which may hold lone surrogates (U+D800..U+DFFF).
//! Internally every kernel works on bytes in the encoding Python's
//! `"surrogatepass"` error handler produces: ordinary UTF-8, except that each
//! surrogate code unit is encoded on its own as the three bytes
//! `ED A0..BF 80..BF`. A string with no surrogates is plain UTF-8, so `&str`
//! inputs need no conversion.
//!
//! All functions here assume well-formed input; [`validate`] checks it at the
//! public API boundary.

/// Decode the code point starting at `s[i]`; returns `(code point, byte length)`.
#[inline]
pub(crate) fn decode_at(s: &[u8], i: usize) -> (u32, usize) {
    let b0 = s[i];
    if b0 < 0x80 {
        (u32::from(b0), 1)
    } else if b0 < 0xE0 {
        ((u32::from(b0 & 0x1F) << 6) | u32::from(s[i + 1] & 0x3F), 2)
    } else if b0 < 0xF0 {
        (
            (u32::from(b0 & 0x0F) << 12)
                | (u32::from(s[i + 1] & 0x3F) << 6)
                | u32::from(s[i + 2] & 0x3F),
            3,
        )
    } else {
        (
            (u32::from(b0 & 0x07) << 18)
                | (u32::from(s[i + 1] & 0x3F) << 12)
                | (u32::from(s[i + 2] & 0x3F) << 6)
                | u32::from(s[i + 3] & 0x3F),
            4,
        )
    }
}

/// Decode the code point that ends just before `s[end]`.
#[inline]
pub(crate) fn decode_before(s: &[u8], end: usize) -> u32 {
    let mut start = end - 1;
    while s[start] & 0xC0 == 0x80 {
        start -= 1;
    }
    decode_at(s, start).0
}

/// Append code point `cp` (surrogates allowed) to `out`.
#[inline]
pub(crate) fn push_cp(out: &mut Vec<u8>, cp: u32) {
    // The `as u8` casts keep only the low bits on purpose.
    if cp < 0x80 {
        out.push(cp as u8);
    } else if cp < 0x800 {
        out.extend_from_slice(&[0xC0 | (cp >> 6) as u8, 0x80 | (cp & 0x3F) as u8]);
    } else if cp < 0x10000 {
        out.extend_from_slice(&[
            0xE0 | (cp >> 12) as u8,
            0x80 | ((cp >> 6) & 0x3F) as u8,
            0x80 | (cp & 0x3F) as u8,
        ]);
    } else {
        out.extend_from_slice(&[
            0xF0 | (cp >> 18) as u8,
            0x80 | ((cp >> 12) & 0x3F) as u8,
            0x80 | ((cp >> 6) & 0x3F) as u8,
            0x80 | (cp & 0x3F) as u8,
        ]);
    }
}

/// Iterator over the code points of a well-formed byte string.
pub(crate) struct CodePoints<'a> {
    s: &'a [u8],
    i: usize,
}

impl Iterator for CodePoints<'_> {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.i >= self.s.len() {
            return None;
        }
        let (cp, n) = decode_at(self.s, self.i);
        self.i += n;
        Some(cp)
    }
}

#[inline]
pub(crate) fn code_points(s: &[u8]) -> CodePoints<'_> {
    CodePoints { s, i: 0 }
}

/// Number of code points (what Python's `len()` returns).
#[inline]
pub(crate) fn cp_len(s: &[u8]) -> usize {
    s.iter().filter(|&&b| b & 0xC0 != 0x80).count()
}

/// Byte offset just after the first `n` code points of `s` (or `s.len()`).
pub(crate) fn cp_offset(s: &[u8], n: usize) -> usize {
    let mut seen = 0;
    for (i, &b) in s.iter().enumerate() {
        if b & 0xC0 != 0x80 {
            if seen == n {
                return i;
            }
            seen += 1;
        }
    }
    s.len()
}

/// Whether `s` contains an encoded surrogate.
#[inline]
pub(crate) fn has_surrogates(s: &[u8]) -> bool {
    s.windows(2).any(|w| w[0] == 0xED && w[1] >= 0xA0)
}

// ---------------------------------------------------------------------------
// Word-at-a-time (SWAR) skipping of ASCII runs
// ---------------------------------------------------------------------------

const LO: u64 = 0x0101_0101_0101_0101;
const HI: u64 = 0x8080_8080_8080_8080;

#[inline]
fn word(s: &[u8], i: usize) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&s[i..i + 8]);
    u64::from_le_bytes(w)
}

/// Nonzero if some byte of `x` is zero (may also flag bytes above a true hit).
#[inline]
const fn has_zero(x: u64) -> u64 {
    x.wrapping_sub(LO) & !x & HI
}

/// Nonzero if some byte of `x` is below `n` (`n <= 0x80`; bytes with the
/// high bit set are not reported). May also flag bytes above a true hit.
#[inline]
const fn has_less(x: u64, n: u8) -> u64 {
    x.wrapping_sub(LO * n as u64) & !x & HI
}

/// Offset of the first byte at or after `i` that is not plain printable
/// ASCII: a byte `>= 0x7F`, a control byte, or `&`. Scans 8 bytes at a time;
/// may stop early (on a false positive), never late.
#[inline]
pub(crate) fn skip_plain_ascii(s: &[u8], mut i: usize) -> usize {
    while i + 8 <= s.len() {
        let w = word(s, i);
        let loud =
            (w & HI) | has_less(w, 0x20) | has_zero(w ^ (LO * 0x26)) | has_zero(w ^ (LO * 0x7F));
        if loud != 0 {
            break;
        }
        i += 8;
    }
    i
}

/// Offset of the first non-ASCII byte at or after `i`, or of a byte in
/// `extra` (an ASCII byte, or 0x80 for none). Never skips past either.
#[inline]
pub(crate) fn skip_ascii(s: &[u8], mut i: usize, extra: u8) -> usize {
    while i + 8 <= s.len() {
        let w = word(s, i);
        if (w & HI) | has_zero(w ^ (LO * u64::from(extra))) != 0 {
            break;
        }
        i += 8;
    }
    while i < s.len() && s[i] < 0x80 && s[i] != extra {
        i += 1;
    }
    i
}

/// Check that `s` is UTF-8 in which surrogate code points are also allowed.
///
/// Returns the offset of the first invalid byte on failure.
pub fn validate(s: &[u8]) -> Result<(), usize> {
    let mut i = 0;
    while i < s.len() {
        match std::str::from_utf8(&s[i..]) {
            Ok(_) => return Ok(()),
            Err(e) => {
                let bad = i + e.valid_up_to();
                // The only extra sequence we accept: ED A0..BF 80..BF.
                if bad + 2 < s.len()
                    && s[bad] == 0xED
                    && (0xA0..=0xBF).contains(&s[bad + 1])
                    && (0x80..=0xBF).contains(&s[bad + 2])
                {
                    i = bad + 3;
                } else {
                    return Err(bad);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_code_points() {
        let cps = [0x41, 0xE9, 0x20AC, 0xD800, 0xDFFF, 0x1F600, 0x10FFFF, 0];
        let mut buf = Vec::new();
        for &cp in &cps {
            push_cp(&mut buf, cp);
        }
        assert_eq!(code_points(&buf).collect::<Vec<_>>(), cps);
        assert_eq!(cp_len(&buf), cps.len());
        assert!(validate(&buf).is_ok());
        assert!(has_surrogates(&buf));
        assert_eq!(decode_before(&buf, buf.len() - 1), 0x10FFFF);
    }

    #[test]
    fn validate_rejects_garbage() {
        assert_eq!(validate(b"ab\xff"), Err(2));
        assert_eq!(validate(b"\xed\xa0"), Err(0));
        assert_eq!(validate(b"\xc0\x80"), Err(0));
        assert!(validate("héllo".as_bytes()).is_ok());
    }

    #[test]
    fn swar_skips() {
        let s = b"0123456789abcdef&xyz\xc3\xa9";
        assert!(skip_plain_ascii(s, 0) <= 16);
        assert_eq!(
            s[skip_plain_ascii(s, 0)..].iter().position(|&b| b == b'&'),
            Some(16 - skip_plain_ascii(s, 0))
        );
        assert_eq!(skip_ascii(s, 0, 0x80), 20);
        assert_eq!(skip_ascii(s, 0, b'&'), 16);
        assert_eq!(skip_ascii(b"abc", 0, 0x80), 3);
        assert_eq!(skip_ascii(b"abcdefgh\x1aijklmnop", 0, 0x1A), 8);
    }

    #[test]
    fn offsets() {
        let s = "aé€😀b".as_bytes();
        assert_eq!(cp_offset(s, 0), 0);
        assert_eq!(cp_offset(s, 2), 3);
        assert_eq!(cp_offset(s, 4), 10);
        assert_eq!(cp_offset(s, 9), s.len());
    }
}
