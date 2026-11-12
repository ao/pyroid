//! Unicode → ASCII transliteration (`pyroid.translit`), from the anyascii 0.3.3
//! data (ISC licence; the `any_ascii` crate and the `anyascii` PyPI package
//! ship the same table). Nothing here is derived from Unidecode (GPL).
//!
//! - [`anyascii`]: byte-identical to `anyascii.anyascii` 0.3.3.
//! - [`unidecode`]: the same output, with Unidecode's choice of what to do
//!   with a character that has *no* transliteration ([`Missing`]); strict
//!   mode is [`first_missing`] followed by [`unidecode`].
//! - [`has_transliteration`]: whether anyascii has a mapping for a character.
//!   A character has one when it is ASCII, maps to a non-empty string, or is
//!   one of the 1,192 that anyascii's table maps to "" on purpose (combining
//!   marks, joiners, tatweel, silent vowel carriers, ...). Unassigned and
//!   private-use code points, lone surrogates, and characters of scripts
//!   anyascii does not cover have none.
//! - `*_wtf8`: the same over WTF-8 (UTF-8 that may also hold encoded lone
//!   surrogates, as Python's `"surrogatepass"` produces), for Python strs that
//!   hold lone surrogates. A surrogate never has a transliteration; anyascii
//!   drops it.
//!
//! Every function returns `Cow::Borrowed` of the whole input exactly when the
//! input is pure ASCII (the output then equals it), and an owned result
//! otherwise, even when that result equals the input (`Missing::Keep` on
//! characters without a transliteration): Unidecode, too, returns its input
//! object only for ASCII input.

mod tables;

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

pub use any_ascii::any_ascii_char as anyascii_char;

/// Bytes with the high bit set, in each lane of a `u64`.
const HIGH_BITS: u64 = 0x8080_8080_8080_8080;

/// Length of the ASCII prefix of `b` (eight bytes at a time).
pub fn ascii_prefix_len(b: &[u8]) -> usize {
    let mut chunks = b.chunks_exact(8);
    let mut done = 0;
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8]));
        let high = word & HIGH_BITS;
        if high != 0 {
            return done + (high.trailing_zeros() / 8) as usize;
        }
        done += 8;
    }
    done + chunks
        .remainder()
        .iter()
        .position(|&x| x >= 0x80)
        .unwrap_or(chunks.remainder().len())
}

/// Same result as Python's `str.isascii()`.
pub fn is_ascii(s: &str) -> bool {
    s.is_ascii()
}

/// Is `cp` one of the code points anyascii maps to "" on purpose?
fn deliberately_empty(cp: u32) -> bool {
    tables::EMPTY
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Does anyascii have a mapping for `c` (possibly the empty string)?
pub fn has_transliteration(c: char) -> bool {
    let cp = u32::from(c);
    c.is_ascii()
        || match BMP.entries.get(cp as usize) {
            Some(&entry) => entry != MISSING,
            None => !anyascii_char(c).is_empty() || deliberately_empty(cp),
        }
}

/// The BMP's mappings, flattened once from the crate's data (about 0.4 MB):
/// per code point, `offset << 8 | len` into `pool`, [`MISSING`] for code
/// points without a mapping, or [`LONG`] (ask the crate). Supplementary
/// planes ask the crate directly.
struct Bmp {
    entries: Box<[u32]>,
    /// Every distinct mapping once, then [`WORD`] bytes of padding.
    pool: Box<[u8]>,
}

const MISSING: u32 = u32::MAX;
const LONG: u32 = u32::MAX - 1;
/// Mappings up to this long are copied as one 8-byte word.
const WORD: usize = 8;

static BMP: LazyLock<Bmp> = LazyLock::new(|| {
    let mut entries = vec![MISSING; 0x1_0000];
    let mut pool: Vec<u8> = Vec::new();
    let mut seen: HashMap<&'static str, usize> = HashMap::new();
    for (cp, entry) in (0u32..).zip(entries.iter_mut()) {
        // Surrogates are not chars; ASCII never reaches the table.
        let Some(c) = char::from_u32(cp).filter(|c| !c.is_ascii()) else {
            continue;
        };
        let mapped = anyascii_char(c);
        if mapped.is_empty() {
            if deliberately_empty(cp) {
                *entry = 0;
            }
            continue;
        }
        let off = *seen.entry(mapped).or_insert_with(|| {
            pool.extend_from_slice(mapped.as_bytes());
            pool.len() - mapped.len()
        });
        *entry = match (u32::try_from(off), u8::try_from(mapped.len())) {
            (Ok(off), Ok(len)) if off < 0xFF_0000 => (off << 8) | u32::from(len),
            _ => LONG,
        };
    }
    pool.extend_from_slice(&[0; WORD]);
    Bmp {
        entries: entries.into_boxed_slice(),
        pool: pool.into_boxed_slice(),
    }
});

/// The code point at `b[i]` (a non-ASCII lead byte of valid UTF-8) and its
/// length in bytes.
#[inline]
fn decode(b: &[u8], i: usize) -> (u32, usize) {
    let lead = u32::from(b[i]);
    let cont = |k: usize| u32::from(b[i + k]) & 0x3F;
    if lead < 0xE0 {
        (((lead & 0x1F) << 6) | cont(1), 2)
    } else if lead < 0xF0 {
        (((lead & 0x0F) << 12) | (cont(1) << 6) | cont(2), 3)
    } else {
        (
            ((lead & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
            4,
        )
    }
}

/// What `missing` writes for the character `raw` (its UTF-8 bytes).
#[inline]
fn put_missing(out: &mut Vec<u8>, raw: &[u8], missing: Missing<'_>) {
    match missing {
        Missing::Drop => {}
        Missing::Replace(rep) => out.extend_from_slice(rep.as_bytes()),
        Missing::Keep => out.extend_from_slice(raw),
    }
}

/// The transliteration of `text`, whose bytes before `start` are ASCII.
///
/// A mapping of up to 8 bytes is appended as one word from the padded pool
/// and the output then truncated to its real length: one fixed-size copy per
/// character instead of a variable-length one (the crate forbids `unsafe`).
fn translit(text: &str, start: usize, missing: Missing<'_>) -> String {
    let bytes = text.as_bytes();
    let bmp = &*BMP;
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len() + bytes.len() / 2 + WORD);
    out.extend_from_slice(&bytes[..start]);
    let mut pos = start;
    while pos < bytes.len() {
        let lead = bytes[pos];
        if lead < 0x80 {
            // Single ASCII bytes between non-ASCII ones are common (spaces
            // in CJK or Cyrillic text): skip the run scan for them.
            if bytes.get(pos + 1).is_none_or(|&next| next >= 0x80) {
                out.push(lead);
                pos += 1;
            } else {
                let run = ascii_prefix_len(&bytes[pos..]);
                out.extend_from_slice(&bytes[pos..pos + run]);
                pos += run;
            }
            continue;
        }
        let (cp, len) = decode(bytes, pos);
        let raw = &bytes[pos..pos + len];
        match bmp.entries.get(cp as usize).copied().unwrap_or(LONG) {
            MISSING => put_missing(&mut out, raw, missing),
            LONG => {
                let ch = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
                let mapped = anyascii_char(ch);
                if mapped.is_empty() && !deliberately_empty(cp) {
                    put_missing(&mut out, raw, missing);
                } else {
                    out.extend_from_slice(mapped.as_bytes());
                }
            }
            entry => {
                let (off, n) = ((entry >> 8) as usize, (entry & 0xFF) as usize);
                if n <= WORD {
                    let word: &[u8; WORD] =
                        bmp.pool[off..off + WORD].try_into().unwrap_or(&[0; WORD]);
                    out.extend_from_slice(word);
                    out.truncate(out.len() - (WORD - n));
                } else {
                    out.extend_from_slice(&bmp.pool[off..off + n]);
                }
            }
        }
        pos += len;
    }
    // ASCII, whole mappings and whole input characters only: valid UTF-8.
    String::from_utf8(out).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

/// Transliterate `s` to ASCII, exactly as `anyascii.anyascii` 0.3.3 does.
pub fn anyascii(s: &str) -> Cow<'_, str> {
    let start = ascii_prefix_len(s.as_bytes());
    if start == s.len() {
        return Cow::Borrowed(s);
    }
    Cow::Owned(translit(s, start, Missing::Drop))
}

/// What [`unidecode`] does with a character that has no transliteration.
/// (Unidecode's `errors="strict"` is [`first_missing`]; its result for a
/// string without missing characters is the same in every mode.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Missing<'r> {
    /// Leave it out (`errors="ignore"`).
    Drop,
    /// Put this string in its place (`errors="replace"`, `replace_str`).
    Replace(&'r str),
    /// Keep the character itself (`errors="preserve"`).
    Keep,
}

/// One code unit of a WTF-8 string: a character or a lone surrogate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    Char(char),
    /// A lone surrogate (U+D800..=U+DFFF), or one byte that is not WTF-8.
    Surrogate(u16),
}

/// The first character without a transliteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingAt {
    /// Position in code points (Python's `str` index).
    pub index: usize,
    pub unit: Unit,
}

/// `s` transliterated as [`anyascii`] does; characters without a
/// transliteration are handled as `missing` says.
pub fn unidecode<'a>(s: &'a str, missing: Missing<'_>) -> Cow<'a, str> {
    let start = ascii_prefix_len(s.as_bytes());
    if start == s.len() {
        return Cow::Borrowed(s);
    }
    Cow::Owned(translit(s, start, missing))
}

/// The first character of `s` without a transliteration, if any.
pub fn first_missing(s: &str) -> Option<MissingAt> {
    first_missing_from(s, 0)
}

/// [`first_missing`], with `base` code points before `s`.
fn first_missing_from(s: &str, base: usize) -> Option<MissingAt> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut index = base;
    while i < b.len() {
        if b[i] < 0x80 {
            let run = ascii_prefix_len(&b[i..]);
            i += run;
            index += run;
        } else {
            let (cp, len) = decode(b, i);
            let c = char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER);
            if !has_transliteration(c) {
                return Some(MissingAt {
                    index,
                    unit: Unit::Char(c),
                });
            }
            i += len;
            index += 1;
        }
    }
    None
}

/// One piece of a WTF-8 string.
enum Piece<'a> {
    Str(&'a str),
    /// A lone surrogate (3 bytes), or one byte that is not WTF-8.
    Bad(u16, &'a [u8]),
}

/// Split WTF-8 into non-empty UTF-8 runs and lone surrogates. (Only strs
/// holding lone surrogates take this path, so it favours simplicity.)
fn pieces(mut b: &[u8]) -> Vec<Piece<'_>> {
    let mut out = Vec::new();
    while !b.is_empty() {
        let (head, rest) = match std::str::from_utf8(b) {
            Ok(s) => (s, &b[b.len()..]),
            Err(e) => {
                let (head, rest) = b.split_at(e.valid_up_to());
                // `head` is valid UTF-8 by construction of `valid_up_to`.
                (std::str::from_utf8(head).unwrap_or_default(), rest)
            }
        };
        if !head.is_empty() {
            out.push(Piece::Str(head));
        }
        let (unit, len) = match rest {
            [] => break,
            [0xED, hi @ 0xA0..=0xBF, lo @ 0x80..=0xBF, ..] => (
                0xD000 | (u16::from(hi & 0x3F) << 6) | u16::from(lo & 0x3F),
                3,
            ),
            [x, ..] => (u16::from(*x), 1),
        };
        let (raw, after) = rest.split_at(len);
        out.push(Piece::Bad(unit, raw));
        b = after;
    }
    out
}

/// [`anyascii`] over WTF-8: lone surrogates are dropped, as
/// `anyascii.anyascii` drops them.
pub fn anyascii_wtf8(b: &[u8]) -> Cow<'_, [u8]> {
    if b.is_ascii() {
        return Cow::Borrowed(b);
    }
    let mut out = String::with_capacity(b.len() + b.len() / 2);
    for piece in pieces(b) {
        if let Piece::Str(s) = piece {
            out.push_str(&anyascii(s));
        }
    }
    Cow::Owned(out.into_bytes())
}

/// [`Missing`] for WTF-8 output: the replacement may itself be WTF-8.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingWtf8<'r> {
    Drop,
    Replace(&'r [u8]),
    Keep,
}

/// [`unidecode`] over WTF-8. A lone surrogate has no transliteration; with
/// [`MissingWtf8::Keep`] its bytes are kept.
pub fn unidecode_wtf8<'a>(b: &'a [u8], missing: MissingWtf8<'_>) -> Cow<'a, [u8]> {
    if b.is_ascii() {
        return Cow::Borrowed(b);
    }
    let mut out: Vec<u8> = Vec::with_capacity(b.len() + b.len() / 2);
    for piece in pieces(b) {
        match piece {
            Piece::Str(s) => {
                for c in s.chars() {
                    let t = anyascii_char(c);
                    if c.is_ascii() || !t.is_empty() {
                        if c.is_ascii() {
                            out.push(c as u8);
                        } else {
                            out.extend_from_slice(t.as_bytes());
                        }
                    } else if !deliberately_empty(u32::from(c)) {
                        match missing {
                            MissingWtf8::Drop => {}
                            MissingWtf8::Replace(r) => out.extend_from_slice(r),
                            MissingWtf8::Keep => {
                                let mut buf = [0; 4];
                                out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    }
                }
            }
            Piece::Bad(_, raw) => match missing {
                MissingWtf8::Drop => {}
                MissingWtf8::Replace(r) => out.extend_from_slice(r),
                MissingWtf8::Keep => out.extend_from_slice(raw),
            },
        }
    }
    Cow::Owned(out)
}

/// [`first_missing`] over WTF-8 (a lone surrogate is always missing).
pub fn first_missing_wtf8(b: &[u8]) -> Option<MissingAt> {
    let mut index = 0;
    for piece in pieces(b) {
        match piece {
            Piece::Str(s) => {
                if let Some(m) = first_missing_from(s, index) {
                    return Some(m);
                }
                index += s.chars().count();
            }
            Piece::Bad(unit, _) => {
                return Some(MissingAt {
                    index,
                    unit: Unit::Surrogate(unit),
                });
            }
        }
    }
    None
}

/// The lone surrogates of a WTF-8 string, in order, with their positions in
/// code points.
pub fn surrogates_wtf8(b: &[u8]) -> Vec<(usize, u16)> {
    let mut out = Vec::new();
    let mut index = 0;
    for piece in pieces(b) {
        match piece {
            Piece::Str(s) => index += s.chars().count(),
            Piece::Bad(unit, _) => {
                out.push((index, unit));
                index += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests;
