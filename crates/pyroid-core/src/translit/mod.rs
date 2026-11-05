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
//! Every function returns `Cow::Borrowed` of the whole input when the output
//! equals it (always for pure-ASCII input).

mod tables;

use std::borrow::Cow;

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
    c.is_ascii() || !anyascii_char(c).is_empty() || deliberately_empty(u32::from(c))
}

/// Transliterate `s` to ASCII, exactly as `anyascii.anyascii` 0.3.3 does.
pub fn anyascii(s: &str) -> Cow<'_, str> {
    let start = ascii_prefix_len(s.as_bytes());
    if start == s.len() {
        return Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + s.len() / 2);
    out.push_str(&s[..start]);
    push_anyascii(&mut out, &s[start..]);
    Cow::Owned(out)
}

/// Append the transliteration of `s` to `out`.
fn push_anyascii(out: &mut String, text: &str) {
    let bytes = text.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        if bytes[pos] < 0x80 {
            let run = ascii_prefix_len(&bytes[pos..]);
            out.push_str(&text[pos..pos + run]);
            pos += run;
        } else {
            let Some(ch) = text[pos..].chars().next() else {
                break;
            };
            out.push_str(anyascii_char(ch));
            pos += ch.len_utf8();
        }
    }
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
    let mut out = String::with_capacity(s.len() + s.len() / 2);
    out.push_str(&s[..start]);
    push_unidecode(&mut out, &s[start..], missing);
    if missing == Missing::Keep && out == s {
        return Cow::Borrowed(s);
    }
    Cow::Owned(out)
}

/// Append the [`unidecode`] of `s` to `out`.
fn push_unidecode(out: &mut String, text: &str, missing: Missing<'_>) {
    let bytes = text.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        if bytes[pos] < 0x80 {
            let run = ascii_prefix_len(&bytes[pos..]);
            out.push_str(&text[pos..pos + run]);
            pos += run;
        } else {
            let Some(ch) = text[pos..].chars().next() else {
                break;
            };
            let ascii = anyascii_char(ch);
            if !ascii.is_empty() {
                out.push_str(ascii);
            } else if !deliberately_empty(u32::from(ch)) {
                match missing {
                    Missing::Drop => {}
                    Missing::Replace(rep) => out.push_str(rep),
                    Missing::Keep => out.push(ch),
                }
            }
            pos += ch.len_utf8();
        }
    }
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
            let n = ascii_prefix_len(&b[i..]);
            i += n;
            index += n;
        } else {
            let c = s[i..].chars().next()?;
            if !has_transliteration(c) {
                return Some(MissingAt {
                    index,
                    unit: Unit::Char(c),
                });
            }
            i += c.len_utf8();
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
            push_anyascii(&mut out, s);
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
    if out == b {
        return Cow::Borrowed(b);
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
