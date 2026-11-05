// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! UTF-8 decoders: Python's strict `utf-8` and ftfy's `utf-8-variants`
//! (`bad_codecs/utf8_variants.py`).

use std::borrow::Cow;

/// Python's strict `bytes.decode("utf-8")`. The text bytes *are* the input
/// bytes, so this only validates.
#[inline]
pub(crate) fn decode_utf8(bytes: &[u8]) -> Option<&[u8]> {
    std::str::from_utf8(bytes).ok().map(str::as_bytes)
}

/// Whether Python's `$` matches at `pos`: at the end, or just before a
/// trailing newline.
#[inline]
fn dollar(b: &[u8], pos: usize) -> bool {
    pos == b.len() || (pos + 1 == b.len() && b[pos] == b'\n')
}

/// Whether ftfy's `SPECIAL_BYTES_RE` (NULL | CESU-8 | lone surrogate) matches
/// at `p`. Python's `$` also matches before a final `\n`, and that is kept.
#[inline]
fn special_at(b: &[u8], p: usize) -> bool {
    match b[p] {
        // NULL_EXPR: \xc0(\x80|$)
        0xC0 => (p + 1 < b.len() && b[p + 1] == 0x80) || dollar(b, p + 1),
        // SURROGATE_EXPR: \xed([\xa0-\xbf]|$)([\x80-\xbf]|$); it matches
        // whenever CESU8_EXPR does.
        0xED => {
            if dollar(b, p + 1) {
                return true;
            }
            if !(0xA0..=0xBF).contains(&b[p + 1]) {
                return false;
            }
            dollar(b, p + 2) || (0x80..=0xBF).contains(&b[p + 2])
        }
        _ => false,
    }
}

/// `CESU8_RE.match` at `p`, given at least six bytes remain.
#[inline]
fn cesu8_at(b: &[u8], p: usize) -> bool {
    b[p] == 0xED
        && (0xA0..=0xAF).contains(&b[p + 1])
        && (0x80..=0xBF).contains(&b[p + 2])
        && b[p + 3] == 0xED
        && (0xB0..=0xBF).contains(&b[p + 4])
        // `([\x80-\xbf]|$)`: `$` can only match here before a final "\n".
        && ((0x80..=0xBF).contains(&b[p + 5]) || dollar(b, p + 5))
}

/// `bytes.decode("utf-8-variants")`: UTF-8 that also accepts CESU-8
/// surrogate pairs and Java's `C0 80` for U+0000.
///
/// This follows ftfy's incremental decoder step by step, including two
/// consequences of Python's `$` matching before a trailing newline: a final
/// `C0 0A` decodes to U+0000, and a CESU-8 pair whose last byte is a final
/// `0A` decodes (using that byte) to an astral character.
pub(crate) fn decode_utf8_variants(b: &[u8]) -> Option<Cow<'_, [u8]>> {
    let mut out: Option<Vec<u8>> = None;
    let mut pos = 0;
    loop {
        let Some(p) = (pos..b.len()).find(|&p| special_at(b, p)) else {
            let rest = decode_utf8(&b[pos..])?;
            return Some(match out {
                None => Cow::Borrowed(rest),
                Some(mut v) => {
                    v.extend_from_slice(rest);
                    Cow::Owned(v)
                }
            });
        };
        let buf = out.get_or_insert_with(|| Vec::with_capacity(b.len()));
        buf.extend_from_slice(decode_utf8(&b[pos..p])?);
        if b[p] == 0xC0 {
            // `if len(input) > 1: return "\u0000", 2` -- whatever the next byte is.
            if b.len() - p < 2 {
                return None;
            }
            buf.push(0);
            pos = p + 2;
        } else {
            // Fewer than six bytes go to the strict decoder, which rejects
            // them; so does a non-CESU-8 surrogate.
            if b.len() - p < 6 || !cesu8_at(b, p) {
                return None;
            }
            let cp = ((u32::from(b[p + 1]) & 0x0F) << 16)
                + ((u32::from(b[p + 2]) & 0x3F) << 10)
                + ((u32::from(b[p + 4]) & 0x0F) << 6)
                + (u32::from(b[p + 5]) & 0x3F)
                + 0x10000;
            super::text::push_cp(buf, cp);
            pos = p + 6;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(b: &[u8]) -> Option<String> {
        decode_utf8_variants(b).map(|c| String::from_utf8(c.into_owned()).unwrap())
    }

    #[test]
    fn plain_utf8() {
        assert_eq!(dec("héllo".as_bytes()).as_deref(), Some("héllo"));
        assert_eq!(dec(b"\xff"), None);
        assert_eq!(dec(b"\xed\x9f\xbf").as_deref(), Some("\u{d7ff}"));
    }

    #[test]
    fn java_null() {
        assert_eq!(dec(b"a\xc0\x80b").as_deref(), Some("a\0b"));
        assert_eq!(dec(b"a\xc0"), None);
        assert_eq!(dec(b"a\xc0x"), None);
        // `$` before a final newline: the newline is consumed.
        assert_eq!(dec(b"a\xc0\n").as_deref(), Some("a\0"));
    }

    #[test]
    fn cesu8() {
        // U+1F600 as CESU-8: D83D DE00
        assert_eq!(
            dec(b"\xed\xa0\xbd\xed\xb8\x80").as_deref(),
            Some("\u{1f600}")
        );
        assert_eq!(dec(b"\xed\xa0\xbd"), None);
        assert_eq!(dec(b"\xed\xb8\x80\xed\xa0\xbd"), None);
        // `$` before a final newline: the 0x0A is used as the last byte.
        assert_eq!(
            dec(b"x\xed\xa0\xbd\xed\xb8\n").as_deref(),
            Some("x\u{1f60a}")
        );
    }
}
