// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! datatrove's `simplify_text` (`datatrove/utils/text.py`), as a given CPython
//! version computes it.
//!
//! The steps run in datatrove's order, each only if its flag is on:
//!
//! 1. `lowercase`: `str.lower()`: full mappings (U+0130 becomes "i" U+0307)
//!    and CPython's Final_Sigma rule for U+03A3 (`handle_capital_sigma`).
//! 2. `norm_numbers`: `regex`'s `\p{Decimal_Number}+([.,،٫⎖⎗⎘]{1}\p{Decimal_Number}+)?` becomes "0".
//! 3. `norm_weekdays`, then `norm_monthnames`: the lowercase English names
//!    become "WEEKDAY" / "MONTH".
//! 4. `remove_punctuation`: each character of `PUNCTUATION` becomes " ".
//! 5. `norm_whitespace`: `re.sub(r"\s+", " ", text.strip())`.
//! 6. `norm_unicode_diacritics`: NFD, then drop general category Mn.
//! 7. Always: `.strip()`.
//!
//! The character data (case mappings, Final_Sigma classes, Mn, which
//! characters NFD touches, `regex`'s `\p{Decimal_Number}`, `PUNCTUATION`) is generated
//! from each CPython by `generate_tables.py` into `simplify_tables`. NFD
//! itself uses the `unicode-normalization` crate, which implements a newer
//! Unicode version than any CPython: decompositions and combining classes
//! never change once a character is assigned, so the only difference is that
//! characters the interpreter's version does not have are inert there (no
//! decomposition, combining class 0). The tables say, per version, which
//! characters NFD touches at all; every other character is copied as a
//! starter.
//!
//! Each step is one pass over the bytes with an ASCII fast path (numbers
//! share the whitespace pass when no step runs between them); a pass that
//! would change nothing writes nothing.

use std::borrow::Cow;

use unicode_normalization::char::{canonical_combining_class, decompose_canonical};

use super::config::{TextNormConfig, UnicodeVersion};
use super::simplify_tables as t;

/// datatrove's `simplify_text(text, config)`.
///
/// Returns `Cow::Borrowed(text)` (the whole input) when nothing changes.
pub fn simplify_text<'a>(
    text: &'a str,
    config: &TextNormConfig,
    unicode: UnicodeVersion,
) -> Cow<'a, str> {
    let mut buf = Vec::new();
    if !run(text.as_bytes(), *config, version_index(unicode), &mut buf) {
        let s = text.trim_matches(crate::text::is_py_whitespace);
        return if s.len() == text.len() {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(s.to_owned())
        };
    }
    strip_in_place(&mut buf);
    if buf == text.as_bytes() {
        return Cow::Borrowed(text);
    }
    Cow::Owned(String::from_utf8(buf).expect("every pass writes UTF-8"))
}

/// [`simplify_text`] into a caller-owned buffer (cleared first), so a loop
/// over many documents can reuse one allocation.
pub fn simplify_text_into(
    text: &str,
    config: &TextNormConfig,
    unicode: UnicodeVersion,
    out: &mut String,
) {
    let mut buf = std::mem::take(out).into_bytes();
    buf.clear();
    if run(text.as_bytes(), *config, version_index(unicode), &mut buf) {
        strip_in_place(&mut buf);
    } else {
        buf.clear();
        buf.extend_from_slice(text.trim_matches(crate::text::is_py_whitespace).as_bytes());
    }
    *out = String::from_utf8(buf).expect("every pass writes UTF-8");
}

/// Run every enabled pass. Returns false if none changed anything (the
/// result is then `text` itself, before the final strip); otherwise the
/// result is in `buf`.
fn run(text: &[u8], config: TextNormConfig, vi: usize, buf: &mut Vec<u8>) -> bool {
    let mut tmp = Vec::new();
    let mut in_buf = false;
    let mut step = |pass: &dyn Fn(&[u8], &mut Vec<u8>) -> bool| {
        let changed = if in_buf {
            tmp.clear();
            pass(buf, &mut tmp)
        } else {
            buf.clear();
            pass(text, buf)
        };
        if changed {
            if in_buf {
                std::mem::swap(buf, &mut tmp);
            }
            in_buf = true;
        }
    };
    if config.lowercase {
        step(&|s, d| lowercase(s, d, vi));
    }
    // Numbers fold into the whitespace pass when nothing runs in between.
    let fold_numbers = config.norm_numbers
        && config.norm_whitespace
        && !config.norm_weekdays
        && !config.norm_monthnames;
    if config.norm_numbers && !fold_numbers {
        step(&numbers);
    }
    if config.norm_weekdays {
        step(&|s, d| replace_words(s, d, &WEEKDAYS, b"WEEKDAY"));
    }
    if config.norm_monthnames {
        step(&|s, d| replace_words(s, d, &MONTHS, b"MONTH"));
    }
    match (config.remove_punctuation, config.norm_whitespace) {
        (true, false) => step(&translate_punctuation),
        (punct, true) => step(&|s, d| collapse(s, d, punct, fold_numbers)),
        (false, false) => {}
    }
    if config.norm_unicode_diacritics {
        step(&|s, d| strip_marks(s, d, vi));
    }
    in_buf
}

// ---------------------------------------------------------------------------
// Character data
// ---------------------------------------------------------------------------

fn version_index(v: UnicodeVersion) -> usize {
    match v {
        UnicodeVersion::V14_0 => 0,
        UnicodeVersion::V15_0 => 1,
        UnicodeVersion::V15_1 => 2,
        UnicodeVersion::V16_0 => 3,
        UnicodeVersion::V17_0 => 4,
    }
}

#[inline]
fn class_id(cp: u32) -> usize {
    let block = if cp < t::LIMIT {
        t::STAGE1[(cp >> t::SHIFT) as usize]
    } else if (t::TAGS_START..t::TAGS_END).contains(&cp) {
        t::STAGE1_TAGS[((cp - t::TAGS_START) >> t::SHIFT) as usize]
    } else {
        return 0;
    };
    let low = (cp & ((1 << t::SHIFT) - 1)) as usize;
    usize::from(t::STAGE2[(usize::from(block) << t::SHIFT) | low])
}

/// The flag byte of `cp` in version `vi` (version-independent bits included).
#[inline]
fn flags(cp: u32, vi: usize) -> u8 {
    t::CLASSES[class_id(cp)][vi]
}

/// `str.lower()` of one character other than U+03A3 that has a mapping.
fn push_lower(dst: &mut Vec<u8>, cp: u32, vi: usize) {
    let k = t::LOWER_SPECIAL.partition_point(|e| e.0 < cp);
    for &(c, mask, mapped) in &t::LOWER_SPECIAL[k..] {
        if c != cp {
            break;
        }
        if mask & (1 << vi) != 0 {
            dst.extend_from_slice(mapped.as_bytes());
            return;
        }
    }
    let r = t::LOWER_RUNS.partition_point(|e| e.1 < cp);
    let mapped = match t::LOWER_RUNS.get(r) {
        Some(&(first, _, delta, stride)) if first <= cp && (cp - first) % stride == 0 => {
            cp.wrapping_add_signed(delta)
        }
        _ => cp,
    };
    debug_assert_ne!(mapped, cp, "U+{cp:04X} has the LOWER flag but no mapping");
    push_char(
        dst,
        char::from_u32(mapped).expect("lowercase mappings are scalar values"),
    );
}

// ---------------------------------------------------------------------------
// UTF-8 helpers (the input of every pass is valid UTF-8)
// ---------------------------------------------------------------------------

/// The code point whose (non-ASCII) lead byte is `s[i]`, and its length.
#[inline]
fn decode(s: &[u8], i: usize) -> (u32, usize) {
    let b0 = u32::from(s[i]);
    let cont = |k: usize| u32::from(s[i + k] & 0x3F);
    if b0 < 0xE0 {
        (((b0 & 0x1F) << 6) | cont(1), 2)
    } else if b0 < 0xF0 {
        (((b0 & 0x0F) << 12) | (cont(1) << 6) | cont(2), 3)
    } else {
        (
            ((b0 & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
            4,
        )
    }
}

#[inline]
fn decode_any(s: &[u8], i: usize) -> (u32, usize) {
    if s[i] < 0x80 {
        (u32::from(s[i]), 1)
    } else {
        decode(s, i)
    }
}

/// The code point that ends just before `s[j]`, and where it starts.
fn decode_before(s: &[u8], j: usize) -> (u32, usize) {
    let mut k = j - 1;
    while s[k] & 0xC0 == 0x80 {
        k -= 1;
    }
    (decode_any(s, k).0, k)
}

/// End of the run of ASCII bytes starting at `i`.
#[inline]
fn ascii_end(s: &[u8], mut i: usize) -> usize {
    while let Some(chunk) = s.get(i..i + 8) {
        let w = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        if w & 0x8080_8080_8080_8080 != 0 {
            break;
        }
        i += 8;
    }
    while i < s.len() && s[i] < 0x80 {
        i += 1;
    }
    i
}

#[inline]
fn push_char(dst: &mut Vec<u8>, c: char) {
    let mut b = [0; 4];
    dst.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
}

// ---------------------------------------------------------------------------
// ASCII classes
// ---------------------------------------------------------------------------

/// `str.isspace()` for ASCII, as a bit set.
const WS_ASCII: u128 = (0x1f << 0x09) | (0x1f << 0x1c);

/// The version-independent flags (`WS`, `PUNCT`, `DIGIT`) of each ASCII byte.
static ASCII: [u8; 128] = {
    let mut a = [0u8; 128];
    let mut b = 0;
    while b < 128 {
        if WS_ASCII & (1 << b) != 0 {
            a[b] |= t::WS;
        }
        if t::PUNCT_ASCII & (1 << b) != 0 {
            a[b] |= t::PUNCT;
        }
        if matches!(b, 0x30..=0x39) {
            a[b] |= t::DIGIT;
        }
        b += 1;
    }
    a
};

#[inline]
fn ascii_flags(b: u8) -> u8 {
    ASCII[usize::from(b & 0x7F)]
}

#[inline]
fn is_ascii_space(b: u8) -> bool {
    matches!(b, b'\t'..=b'\r' | 0x1c..=b' ')
}

fn is_space_cp(cp: u32) -> bool {
    char::from_u32(cp).is_some_and(crate::text::is_py_whitespace)
}

/// `str.strip()` of a buffer.
fn strip_in_place(buf: &mut Vec<u8>) {
    let mut end = buf.len();
    while end > 0 {
        if buf[end - 1] < 0x80 {
            if !is_ascii_space(buf[end - 1]) {
                break;
            }
            end -= 1;
            continue;
        }
        let (cp, start) = decode_before(buf, end);
        if !is_space_cp(cp) {
            break;
        }
        end = start;
    }
    buf.truncate(end);
    let mut start = 0;
    while start < buf.len() {
        let (cp, n) = decode_any(buf, start);
        if !is_space_cp(cp) {
            break;
        }
        start += n;
    }
    buf.drain(..start);
}

// ---------------------------------------------------------------------------
// 1. lowercase
// ---------------------------------------------------------------------------

/// `str.lower()`.
fn lowercase(src: &[u8], dst: &mut Vec<u8>, vi: usize) -> bool {
    // Nothing is written up to the first character that changes.
    let mut i = 0;
    while i < src.len() {
        let b = src[i];
        if b < 0x80 {
            if b.is_ascii_uppercase() {
                break;
            }
            i += 1;
        } else {
            let (cp, n) = decode(src, i);
            if flags(cp, vi) & t::LOWER != 0 {
                break;
            }
            i += n;
        }
    }
    if i == src.len() {
        return false;
    }
    dst.reserve(src.len() + 8);
    dst.extend_from_slice(&src[..i]);
    while i < src.len() {
        if src[i] < 0x80 {
            let end = ascii_end(src, i);
            let at = dst.len();
            dst.extend_from_slice(&src[i..end]);
            dst[at..].make_ascii_lowercase();
            i = end;
            continue;
        }
        // A run of non-ASCII characters, copied up to the first one that
        // has a mapping.
        let start = i;
        let mut mapped = None;
        while i < src.len() && src[i] >= 0x80 {
            let (cp, n) = decode(src, i);
            if flags(cp, vi) & t::LOWER != 0 {
                mapped = Some((cp, n));
                break;
            }
            i += n;
        }
        dst.extend_from_slice(&src[start..i]);
        if let Some((cp, n)) = mapped {
            if cp == 0x3A3 {
                push_char(dst, if final_sigma(src, i, vi) { 'ς' } else { 'σ' });
            } else {
                push_lower(dst, cp, vi);
            }
            i += n;
        }
    }
    true
}

/// CPython's `handle_capital_sigma` for the U+03A3 at `src[i]`: final when
/// the nearest character before it that is not case-ignorable is cased, and
/// the nearest one after it is not (or there is none).
fn final_sigma(src: &[u8], i: usize, vi: usize) -> bool {
    let mut back = i;
    let mut after_cased = false;
    while back > 0 {
        let (cp, start) = decode_before(src, back);
        let f = flags(cp, vi);
        if f & t::CI == 0 {
            after_cased = f & t::CASED != 0;
            break;
        }
        back = start;
    }
    if !after_cased {
        return false;
    }
    let mut ahead = i + 2;
    while ahead < src.len() {
        let (cp, n) = decode_any(src, ahead);
        let f = flags(cp, vi);
        if f & t::CI == 0 {
            return f & t::CASED == 0;
        }
        ahead += n;
    }
    true
}

// ---------------------------------------------------------------------------
// 2. numbers
// ---------------------------------------------------------------------------

/// Length of the `\p{Decimal_Number}` character at `src[i]`, if there is one.
#[inline]
fn digit_at(src: &[u8], i: usize) -> Option<usize> {
    let b = *src.get(i)?;
    if b < 0x80 {
        return b.is_ascii_digit().then_some(1);
    }
    let (cp, n) = decode(src, i);
    (flags(cp, 0) & t::DIGIT != 0).then_some(n)
}

/// Length of the `[.,،٫⎖⎗⎘]` character at `src[i]`, if there is one.
#[inline]
fn separator_at(src: &[u8], i: usize) -> Option<usize> {
    let b = *src.get(i)?;
    if b < 0x80 {
        return matches!(b, b'.' | b',').then_some(1);
    }
    let (cp, n) = decode(src, i);
    matches!(cp, 0x060C | 0x066B | 0x2396..=0x2398).then_some(n)
}

/// End of the `NUMBERS_PATTERN` match that starts with the digit at `i`.
fn number_end(src: &[u8], mut i: usize) -> usize {
    while let Some(n) = digit_at(src, i) {
        i += n;
    }
    if let Some(s) = separator_at(src, i) {
        if let Some(n) = digit_at(src, i + s) {
            i += s + n;
            while let Some(n) = digit_at(src, i) {
                i += n;
            }
        }
    }
    i
}

/// Skip 8 bytes at a time while none is an ASCII digit or non-ASCII.
#[inline]
fn skip_plain(src: &[u8], mut i: usize) -> usize {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    while let Some(chunk) = src.get(i..i + 8) {
        let w = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        let v = w & !HI;
        // High bit of each byte: (v >= '0') && !(v > '9'), or a non-ASCII byte.
        let digit = (v + LO * (0x80 - 0x30)) & !(v + LO * (0x80 - 0x3A)) & HI;
        if (digit | (w & HI)) != 0 {
            break;
        }
        i += 8;
    }
    i
}

/// `NUMBERS_PATTERN.sub("0", text)`.
fn numbers(src: &[u8], dst: &mut Vec<u8>) -> bool {
    let mut copied = 0;
    let mut changed = false;
    let mut i = 0;
    while i < src.len() {
        let b = src[i];
        let n = if b < 0x80 {
            if !b.is_ascii_digit() {
                i = skip_plain(src, i + 1);
                continue;
            }
            1
        } else {
            let (cp, n) = decode(src, i);
            if flags(cp, 0) & t::DIGIT == 0 {
                i += n;
                continue;
            }
            n
        };
        let end = number_end(src, i + n);
        if &src[i..end] != b"0" {
            if !changed {
                dst.reserve(src.len());
                changed = true;
            }
            dst.extend_from_slice(&src[copied..i]);
            dst.push(b'0');
            copied = end;
        }
        i = end;
    }
    if changed {
        dst.extend_from_slice(&src[copied..]);
    }
    changed
}

// ---------------------------------------------------------------------------
// 3. weekdays and months
// ---------------------------------------------------------------------------

/// A literal alternation `w1|w2|...` of lowercase ASCII words (at least two
/// letters each), with the set of their two-letter prefixes for a quick
/// filter: bit `second - b'a'` of `second[first]`.
struct Words {
    list: &'static [&'static [u8]],
    second: [u32; 128],
}

const fn words(list: &'static [&'static [u8]]) -> Words {
    let mut second = [0u32; 128];
    let mut k = 0;
    while k < list.len() {
        let w = list[k];
        assert!(w.len() >= 2 && w[0] < 128 && w[1] >= b'a' && w[1] <= b'z');
        second[w[0] as usize] |= 1 << (w[1] - b'a');
        k += 1;
    }
    Words { list, second }
}

static WEEKDAYS: Words = words(&[
    b"monday",
    b"tuesday",
    b"wednesday",
    b"thursday",
    b"friday",
    b"saturday",
    b"sunday",
]);
static MONTHS: Words = words(&[
    b"january",
    b"february",
    b"march",
    b"april",
    b"may",
    b"june",
    b"july",
    b"august",
    b"september",
    b"october",
    b"november",
    b"december",
]);

/// `re.sub("w1|w2|...", repl, text)`: at each position the first alternative
/// that matches wins.
fn replace_words(src: &[u8], dst: &mut Vec<u8>, words: &Words, repl: &[u8]) -> bool {
    let mut copied = 0;
    let mut changed = false;
    let mut i = 0;
    'scan: while i + 1 < src.len() {
        // An ASCII byte is always a whole character in UTF-8.
        let (b, s) = (src[i], src[i + 1].wrapping_sub(b'a'));
        if b < 0x80 && s < 26 && words.second[usize::from(b)] & (1 << s) != 0 {
            for w in words.list {
                if w[0] == b && w[1] == src[i + 1] && src[i..].starts_with(w) {
                    if !changed {
                        dst.reserve(src.len());
                        changed = true;
                    }
                    dst.extend_from_slice(&src[copied..i]);
                    dst.extend_from_slice(repl);
                    i += w.len();
                    copied = i;
                    continue 'scan;
                }
            }
        }
        i += 1;
    }
    if changed {
        dst.extend_from_slice(&src[copied..]);
    }
    changed
}

// ---------------------------------------------------------------------------
// 4. punctuation, 5. whitespace
// ---------------------------------------------------------------------------

/// Length of the character at `src[i]` and whether it has any of `mask`
/// (`WS`, `PUNCT`, `DIGIT`).
#[inline]
fn common_at(src: &[u8], i: usize, mask: u8) -> (bool, usize) {
    let b = src[i];
    if b < 0x80 {
        return (ascii_flags(b) & mask != 0, 1);
    }
    let (cp, n) = decode(src, i);
    (flags(cp, 0) & mask != 0, n)
}

/// End of the run of characters from `i` that have none of `mask`.
#[inline]
fn run_without(src: &[u8], mut i: usize, mask: u8) -> usize {
    while i < src.len() {
        let b = src[i];
        if b < 0x80 {
            if ascii_flags(b) & mask != 0 {
                break;
            }
            i += 1;
        } else {
            let (cp, n) = decode(src, i);
            if flags(cp, 0) & mask != 0 {
                break;
            }
            i += n;
        }
    }
    i
}

/// `text.translate(PUNCTUATION_TRANS)`: each punctuation character becomes " ".
fn translate_punctuation(src: &[u8], dst: &mut Vec<u8>) -> bool {
    let mut i = run_without(src, 0, t::PUNCT);
    if i == src.len() {
        return false;
    }
    dst.reserve(src.len());
    dst.extend_from_slice(&src[..i]);
    while i < src.len() {
        let (_, n) = common_at(src, i, t::PUNCT);
        dst.push(b' ');
        let end = run_without(src, i + n, t::PUNCT);
        dst.extend_from_slice(&src[i + n..end]);
        i = end;
    }
    true
}

/// `re.sub(r"\s+", " ", text.strip())`, i.e. `" ".join(text.split())`; with
/// `punct`, of `text.translate(PUNCTUATION_TRANS)` (punctuation becomes a
/// space first, so it counts as whitespace here); with `numbers`, of
/// `NUMBERS_PATTERN.sub("0", text)` first (only valid when no pass runs in
/// between: weekdays and months off).
fn collapse(src: &[u8], dst: &mut Vec<u8>, punct: bool, numbers: bool) -> bool {
    let sep = if punct { t::WS | t::PUNCT } else { t::WS };
    let digit = if numbers { t::DIGIT } else { 0 };
    // Find the first place where the text may change: a digit, or a
    // separator other than one " " between two words.
    let mut i = 0;
    loop {
        i = run_without(src, i, sep | digit);
        if i == src.len() {
            return false;
        }
        if i == 0 || src[i] != b' ' || i + 1 == src.len() || common_at(src, i + 1, sep).0 {
            break;
        }
        i += 1;
    }
    // Compact a copy in place: the output never gets ahead of the input (a
    // space is only owed after at least one separator was dropped, and a
    // number is at least as long as its "0").
    dst.extend_from_slice(src);
    let buf = &mut dst[..];
    let (mut rd, mut wr) = (i, i);
    let mut pending = false;
    let mut any = i > 0;
    while rd < buf.len() {
        let b = buf[rd];
        let (fl, n) = if b < 0x80 {
            let fl = ascii_flags(b);
            if fl & digit == 0 {
                // Branch-free: write a space and the byte speculatively,
                // keep what is needed.
                let is_sep = fl & sep != 0;
                buf[wr] = b' ';
                wr += usize::from(!is_sep & pending);
                buf[wr] = b;
                wr += usize::from(!is_sep);
                pending = is_sep & any;
                any |= !is_sep;
                rd += 1;
                continue;
            }
            (fl, 1)
        } else {
            let (cp, n) = decode(buf, rd);
            (flags(cp, 0), n)
        };
        // A digit first: U+FF11 is both `\p{Decimal_Number}` and in `PUNCTUATION`, and
        // numbers run before punctuation.
        if fl & digit == 0 && fl & sep != 0 {
            pending = any;
            rd += n;
            continue;
        }
        if pending {
            buf[wr] = b' ';
            wr += 1;
            pending = false;
        }
        any = true;
        if fl & digit != 0 {
            buf[wr] = b'0';
            wr += 1;
            rd = number_end(buf, rd + n);
        } else {
            buf.copy_within(rd..rd + n, wr);
            wr += n;
            rd += n;
        }
    }
    dst.truncate(wr);
    true
}

// ---------------------------------------------------------------------------
// 6. diacritics
// ---------------------------------------------------------------------------

/// Emit the pending non-starters in canonical order.
fn flush(marks: &mut Vec<(u8, char)>, dst: &mut Vec<u8>) {
    if marks.len() > 1 {
        marks.sort_by_key(|m| m.0); // stable, as canonical ordering requires
    }
    for &(_, c) in &*marks {
        push_char(dst, c);
    }
    marks.clear();
}

/// End of the run from `i` of characters NFD leaves alone in this version
/// (no decomposition, combining class 0) that are not Mn.
#[inline]
fn inert_run(src: &[u8], mut i: usize, vi: usize) -> usize {
    while i < src.len() {
        if src[i] < 0x80 {
            i = ascii_end(src, i);
            continue;
        }
        let (cp, n) = decode(src, i);
        if flags(cp, vi) & (t::NFDX | t::MN) != 0 {
            break;
        }
        i += n;
    }
    i
}

/// `"".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")`.
///
/// Mn characters are dropped as they come: canonical ordering is a stable
/// sort by combining class, and dropping elements from a stable sort leaves
/// the others in the same order. An Mn character with combining class 0
/// still ends the run of non-starters before it.
fn strip_marks(src: &[u8], dst: &mut Vec<u8>, vi: usize) -> bool {
    if src.is_ascii() {
        return false;
    }
    let mut i = inert_run(src, 0, vi);
    if i == src.len() {
        return false;
    }
    dst.reserve(src.len());
    dst.extend_from_slice(&src[..i]);
    // Non-Mn characters with a non-zero combining class waiting for the next
    // starter (rare: a few Mc characters).
    let mut marks: Vec<(u8, char)> = Vec::new();
    while i < src.len() {
        let end = inert_run(src, i, vi);
        if end > i {
            if !marks.is_empty() {
                flush(&mut marks, dst);
            }
            dst.extend_from_slice(&src[i..end]);
            i = end;
            continue;
        }
        let (cp, n) = decode(src, i);
        let f = flags(cp, vi);
        if f & t::NFDX == 0 {
            // An Mn starter: it ends the pending run, then goes.
            if !marks.is_empty() {
                flush(&mut marks, dst);
            }
        } else {
            let c = char::from_u32(cp).expect("decoded from UTF-8");
            decompose_canonical(c, |d| {
                let ccc = canonical_combining_class(d);
                let df = if d == c { f } else { flags(u32::from(d), vi) };
                let mn = df & t::MN != 0;
                if ccc == 0 {
                    if !marks.is_empty() {
                        flush(&mut marks, dst);
                    }
                    if !mn {
                        push_char(dst, d);
                    }
                } else if !mn {
                    marks.push((ccc, d));
                }
            });
        }
        i += n;
    }
    flush(&mut marks, dst);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [UnicodeVersion; 5] = [
        UnicodeVersion::V14_0,
        UnicodeVersion::V15_0,
        UnicodeVersion::V15_1,
        UnicodeVersion::V16_0,
        UnicodeVersion::V17_0,
    ];

    fn only(f: impl FnOnce(&mut TextNormConfig)) -> TextNormConfig {
        let mut c = TextNormConfig {
            lowercase: false,
            norm_whitespace: false,
            remove_punctuation: false,
            norm_unicode_diacritics: false,
            norm_numbers: false,
            norm_weekdays: false,
            norm_monthnames: false,
        };
        f(&mut c);
        c
    }

    fn s(text: &str, config: TextNormConfig) -> String {
        simplify_text(text, &config, UnicodeVersion::V15_1).into_owned()
    }

    #[test]
    fn whitespace_flag_is_python_isspace() {
        for cp in 0..0x11_0000u32 {
            let ws = flags(cp, 0) & t::WS != 0;
            assert_eq!(ws, t::PY_WHITESPACE.contains(&cp), "U+{cp:04X}");
            if let Some(c) = char::from_u32(cp) {
                assert_eq!(ws, crate::text::is_py_whitespace(c), "U+{cp:04X}");
            }
        }
        for b in 0..0x80u8 {
            assert_eq!(is_ascii_space(b), flags(u32::from(b), 0) & t::WS != 0);
            assert_eq!(is_ascii_space(b), WS_ASCII & (1 << b) != 0);
            assert_eq!(
                ascii_flags(b),
                flags(u32::from(b), 0) & (t::WS | t::PUNCT | t::DIGIT)
            );
        }
    }

    #[test]
    fn crate_combining_classes_match_cpython() {
        // Only non-Mn characters with a non-zero class can show their class
        // in the output; check the crate agrees with CPython on every one.
        for &(cp, ccc) in &t::NON_MN_CCC {
            let c = char::from_u32(cp).unwrap();
            assert_eq!(canonical_combining_class(c), ccc, "U+{cp:04X}");
        }
    }

    #[test]
    fn lowercase_examples() {
        let lc = only(|c| c.lowercase = true);
        assert_eq!(
            s("HeLLo \u{130} \u{212a} \u{212b} \u{1e9e}", lc),
            "hello i\u{307} k \u{e5} \u{df}"
        );
        assert_eq!(
            s("ΣΑΣ ΣΑΣ'Α Σ ΑΣ\u{301} Α'Σ'", lc),
            "σας σασ'α σ ας\u{301} α'ς'"
        );
        assert_eq!(s("ΑΣ.Α ΑΣ1 1Σ", lc), "ασ.α ας1 1σ");
        assert!(matches!(
            simplify_text("already lower", &lc, UnicodeVersion::V14_0),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn number_examples() {
        let n = only(|c| c.norm_numbers = true);
        assert_eq!(
            s("1 12 1.5 1.2.3 12,000 3،4 ١٢٫٣ 5. .5 0 00", n),
            "0 0 0 0.0 0 0 0 0. .0 0 0"
        );
        assert!(matches!(
            simplify_text("0 a 0", &n, UnicodeVersion::V14_0),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn name_examples() {
        let w = only(|c| {
            c.norm_weekdays = true;
            c.norm_monthnames = true;
        });
        assert_eq!(
            s("mondays maymarch Monday junejuly", w),
            "WEEKDAYs MONTHMONTH Monday MONTHMONTH"
        );
    }

    #[test]
    fn punctuation_and_whitespace_examples() {
        let p = only(|c| c.remove_punctuation = true);
        assert_eq!(s("a,b\tc—d「e」\x01", p), "a b\tc d e");
        let w = only(|c| c.norm_whitespace = true);
        assert_eq!(s("  a \t b\u{3000}c ,d ", w), "a b c ,d");
        let pw = only(|c| {
            c.remove_punctuation = true;
            c.norm_whitespace = true;
        });
        assert_eq!(s(", a ,, b. c!", pw), "a b c");
        assert!(matches!(
            simplify_text("a b", &pw, UnicodeVersion::V14_0),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn diacritic_examples() {
        let d = only(|c| c.norm_unicode_diacritics = true);
        assert_eq!(
            s("café Ångström ǅ 한 ;", d),
            "cafe Angstrom ǅ \u{1112}\u{1161}\u{11ab} ;"
        );
        // Canonical ordering of the non-Mn marks that survive.
        assert_eq!(s("x\u{1d16d}\u{1d165}", d), "x\u{1d165}\u{1d16d}");
        // An Mn with combining class 0 (CGJ) still blocks reordering.
        assert_eq!(s("x\u{1d16d}\u{34f}\u{1d165}", d), "x\u{1d16d}\u{1d165}");
        // U+1171E is Mn up to Unicode 15.1, Mc from 16.0.
        let ahom = "\u{11700}\u{1171e}";
        for v in ALL {
            let got = simplify_text(ahom, &d, v);
            let expect = if v < UnicodeVersion::V16_0 {
                "\u{11700}"
            } else {
                ahom
            };
            assert_eq!(got, expect, "{v:?}");
        }
    }

    #[test]
    fn default_config_examples() {
        let cfg = TextNormConfig::default();
        assert_eq!(
            s("  Hello, World! It's 3.14 o'clock — café.  ", cfg),
            "hello world it s 0 o clock cafe"
        );
        assert_eq!(s("", cfg), "");
        assert!(matches!(
            simplify_text("abc def", &cfg, UnicodeVersion::V14_0),
            Cow::Borrowed(_)
        ));
        let mut out = String::from("junk");
        simplify_text_into(" A,B ", &cfg, UnicodeVersion::V17_0, &mut out);
        assert_eq!(out, "a b");
        simplify_text_into(" ab ", &cfg, UnicodeVersion::V17_0, &mut out);
        assert_eq!(out, "ab");
    }
}
