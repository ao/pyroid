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
    fn offsets() {
        let s = "aé€😀b".as_bytes();
        assert_eq!(cp_offset(s, 0), 0);
        assert_eq!(cp_offset(s, 2), 3);
        assert_eq!(cp_offset(s, 4), 10);
        assert_eq!(cp_offset(s, 9), s.len());
    }
}
