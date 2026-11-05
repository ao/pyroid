// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! The individual fixes (ftfy's `fixes.py`).
//!
//! Every text fixer takes surrogate-tolerant UTF-8 (see [`super::text`]) and
//! returns `None` when the text is unchanged, so the common case allocates
//! nothing. The byte transcoders (`restore_byte_a0`, `replace_lossy_sequences`)
//! work on the single-byte encodings `fix_encoding` produces.

use super::UnicodeVersion;
use super::chardata::is_digit;
use super::tables::{
    CONTROL_CHARS, HTML_ENTITIES, INVALID_CHARREFS, INVALID_CODEPOINTS, LIGATURES, SLOPPY_1252_C1,
    WIDTH_MAP,
};
use super::text::{decode_at, push_cp};

/// A replacement produced by a [`translate`] callback.
enum Rep {
    Cp(u32),
    Str(&'static str),
    Delete,
}

/// Rebuild `text`, replacing each code point for which `f(cp, byte_offset)`
/// returns a replacement. Returns `None` if `f` never fires.
#[inline]
fn translate(text: &[u8], mut f: impl FnMut(u32, usize) -> Option<Rep>) -> Option<Vec<u8>> {
    let mut out: Option<Vec<u8>> = None;
    let mut copied = 0;
    let mut i = 0;
    while i < text.len() {
        let b = text[i];
        let (cp, n) = if b < 0x80 {
            (u32::from(b), 1)
        } else {
            decode_at(text, i)
        };
        if let Some(rep) = f(cp, i) {
            let buf = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
            buf.extend_from_slice(&text[copied..i]);
            match rep {
                Rep::Cp(c) => push_cp(buf, c),
                Rep::Str(s) => buf.extend_from_slice(s.as_bytes()),
                Rep::Delete => {}
            }
            copied = i + n;
        }
        i += n;
    }
    let mut buf = out?;
    buf.extend_from_slice(&text[copied..]);
    Some(buf)
}

// ---------------------------------------------------------------------------
// unescape_html
// ---------------------------------------------------------------------------

/// Length of a `HTML_ENTITY_RE` (`&#?[0-9A-Za-z]{1,24};`) match at `i`.
#[inline]
fn entity_match(text: &[u8], i: usize) -> Option<usize> {
    let run = |from: usize| {
        text[from..]
            .iter()
            .take_while(|b| b.is_ascii_alphanumeric())
            .count()
    };
    let try_at = |from: usize| {
        let n = run(from);
        ((1..=24).contains(&n) && text.get(from + n) == Some(&b';')).then_some(from + n + 1 - i)
    };
    if text.get(i + 1) == Some(&b'#') {
        // `#?` is greedy; backtracking to "no #" cannot match, since `#` is
        // not alphanumeric.
        try_at(i + 2)
    } else {
        try_at(i + 1)
    }
}

/// What `html.unescape` does to a whole `&#...;` match, if it decodes it.
fn numeric_charref(entity: &[u8]) -> Option<Option<u32>> {
    let body = &entity[2..entity.len() - 1];
    let (digits, radix) = match body.first() {
        Some(b'x' | b'X') if body.len() > 1 && body[1..].iter().all(u8::is_ascii_hexdigit) => {
            (&body[1..], 16)
        }
        Some(_) if body.iter().all(u8::is_ascii_digit) => (body, 10),
        // html.unescape leaves a trailing part (and the `;`) behind: ftfy
        // then keeps the entity unchanged.
        _ => return None,
    };
    let mut num: u32 = 0;
    for &d in digits {
        let v = char::from(d).to_digit(radix).unwrap_or(0);
        num = num.saturating_mul(radix).saturating_add(v).min(0x0011_0000);
    }
    if let Ok(k) = INVALID_CHARREFS.binary_search_by_key(&num, |&(n, _)| n) {
        return Some(Some(INVALID_CHARREFS[k].1));
    }
    if (0xD800..=0xDFFF).contains(&num) || num > 0x10_FFFF {
        return Some(Some(0xFFFD));
    }
    if INVALID_CODEPOINTS.binary_search(&num).is_ok() {
        return Some(None); // decodes to the empty string
    }
    Some(Some(num))
}

/// ftfy's `unescape_html`.
pub(crate) fn unescape_html(text: &[u8]) -> Option<Vec<u8>> {
    let mut out: Option<Vec<u8>> = None;
    let mut copied = 0;
    let mut i = memchr(b'&', text, 0)?;
    loop {
        let next = match entity_match(text, i) {
            Some(n) => {
                let entity = &text[i..i + n];
                let replacement: Option<Vec<u8>> =
                    match HTML_ENTITIES.binary_search_by_key(&entity, |&(k, _)| k.as_bytes()) {
                        Ok(k) => Some(HTML_ENTITIES[k].1.as_bytes().to_vec()),
                        Err(_) if entity.starts_with(b"&#") => match numeric_charref(entity) {
                            // `if ";" in unescaped: return text`
                            Some(Some(0x3B)) | None => None,
                            Some(Some(cp)) => {
                                let mut v = Vec::with_capacity(4);
                                push_cp(&mut v, cp);
                                Some(v)
                            }
                            Some(None) => Some(Vec::new()),
                        },
                        Err(_) => None,
                    };
                if let Some(rep) = replacement {
                    let buf = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
                    buf.extend_from_slice(&text[copied..i]);
                    buf.extend_from_slice(&rep);
                    copied = i + n;
                }
                i + n
            }
            None => i + 1,
        };
        match memchr(b'&', text, next) {
            Some(j) => i = j,
            None => break,
        }
    }
    let mut buf = out?;
    buf.extend_from_slice(&text[copied..]);
    (buf != text).then_some(buf)
}

#[inline]
fn memchr(needle: u8, hay: &[u8], from: usize) -> Option<usize> {
    hay.get(from..)?
        .iter()
        .position(|&b| b == needle)
        .map(|k| k + from)
}

// ---------------------------------------------------------------------------
// simple character fixers
// ---------------------------------------------------------------------------

/// ftfy's `remove_terminal_escapes`: drop `\x1b\[(\d|;)*[a-zA-Z]`.
pub(crate) fn remove_terminal_escapes(text: &[u8], v: UnicodeVersion) -> Option<Vec<u8>> {
    let mut out: Option<Vec<u8>> = None;
    let mut copied = 0;
    let mut i = memchr(0x1B, text, 0)?;
    loop {
        let mut next = i + 1;
        if text.get(i + 1) == Some(&b'[') {
            let mut k = i + 2;
            while k < text.len() {
                let (cp, n) = decode_at(text, k);
                if cp == u32::from(b';') || is_digit(cp, v) {
                    k += n;
                } else {
                    break;
                }
            }
            if k < text.len() && text[k].is_ascii_alphabetic() {
                let buf = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
                buf.extend_from_slice(&text[copied..i]);
                copied = k + 1;
                next = k + 1;
            }
        }
        match memchr(0x1B, text, next) {
            Some(j) => i = j,
            None => break,
        }
    }
    let mut buf = out?;
    buf.extend_from_slice(&text[copied..]);
    Some(buf)
}

/// Rebuild `text` by visiting each occurrence of a lead byte found by
/// `find`. At a hit `p`, `f(text, p)` returns `Some((len, replacement))` to
/// replace `text[p..p + len]`, or `None` to keep it. Lead bytes must start a
/// code point (ASCII or a UTF-8 lead byte), so a hit is always a boundary.
#[inline]
fn replace_at_leads(
    text: &[u8],
    find: impl Fn(&[u8]) -> Option<usize>,
    f: impl Fn(&[u8], usize) -> Option<(usize, &'static [u8])>,
) -> Option<Vec<u8>> {
    let mut out: Option<Vec<u8>> = None;
    let mut copied = 0;
    let mut i = 0;
    while let Some(k) = find(&text[i..]) {
        let p = i + k;
        match f(text, p) {
            Some((len, rep)) => {
                let buf = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
                buf.extend_from_slice(&text[copied..p]);
                buf.extend_from_slice(rep);
                copied = p + len;
                i = p + len;
            }
            None => i = p + 1,
        }
    }
    let mut buf = out?;
    buf.extend_from_slice(&text[copied..]);
    Some(buf)
}

/// ftfy's `uncurl_quotes`: U+02BC and U+2018..U+201B become `'`,
/// U+201C..U+201F become `"`.
pub(crate) fn uncurl_quotes(text: &[u8]) -> Option<Vec<u8>> {
    replace_at_leads(
        text,
        |t| memchr::memchr2(0xE2, 0xCA, t),
        |t, p| match (t[p], t.get(p + 1), t.get(p + 2)) {
            (0xE2, Some(0x80), Some(0x98..=0x9B)) => Some((3, b"'")),
            (0xE2, Some(0x80), Some(0x9C..=0x9F)) => Some((3, b"\"")),
            (0xCA, Some(0xBC), _) => Some((2, b"'")),
            _ => None,
        },
    )
}

/// ftfy's `fix_latin_ligatures`.
pub(crate) fn fix_latin_ligatures(text: &[u8]) -> Option<Vec<u8>> {
    if !text.iter().any(|&b| matches!(b, 0xC4 | 0xC5 | 0xC7 | 0xEF)) {
        return None;
    }
    translate(text, |cp, _| {
        LIGATURES
            .binary_search_by_key(&cp, |&(c, _)| c)
            .ok()
            .map(|k| Rep::Str(LIGATURES[k].1))
    })
}

/// ftfy's `fix_character_width`.
pub(crate) fn fix_character_width(text: &[u8]) -> Option<Vec<u8>> {
    if !text.iter().any(|&b| b == 0xE3 || b == 0xEF) {
        return None;
    }
    translate(text, |cp, _| {
        if cp != 0x3000 && !(0xFF01..0xFFF0).contains(&cp) {
            return None;
        }
        WIDTH_MAP
            .binary_search_by_key(&cp, |&(c, _)| c)
            .ok()
            .map(|k| Rep::Str(WIDTH_MAP[k].1))
    })
}

/// ftfy's `fix_line_breaks`: CRLF, CR, U+2028, U+2029 and U+0085 become LF.
pub(crate) fn fix_line_breaks(text: &[u8]) -> Option<Vec<u8>> {
    replace_at_leads(
        text,
        |t| memchr::memchr3(b'\r', 0xC2, 0xE2, t),
        |t, p| match (t[p], t.get(p + 1), t.get(p + 2)) {
            // "\r\n" -> "\n": drop the CR and keep the LF.
            (b'\r', Some(b'\n'), _) => Some((1, b"")),
            (b'\r', _, _) => Some((1, b"\n")),
            (0xC2, Some(0x85), _) => Some((2, b"\n")),
            (0xE2, Some(0x80), Some(0xA8 | 0xA9)) => Some((3, b"\n")),
            _ => None,
        },
    )
}

/// ftfy's `fix_surrogates`: join surrogate pairs, replace the rest with U+FFFD.
pub(crate) fn fix_surrogates(text: &[u8]) -> Option<Vec<u8>> {
    if !super::text::has_surrogates(text) {
        return None;
    }
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let (cp, n) = decode_at(text, i);
        if (0xD800..=0xDBFF).contains(&cp) && i + n < text.len() {
            let (lo, m) = decode_at(text, i + n);
            if (0xDC00..=0xDFFF).contains(&lo) {
                push_cp(&mut out, 0x10000 + (cp - 0xD800) * 0x400 + (lo - 0xDC00));
                i += n + m;
                continue;
            }
        }
        if (0xD800..=0xDFFF).contains(&cp) {
            push_cp(&mut out, 0xFFFD);
        } else {
            out.extend_from_slice(&text[i..i + n]);
        }
        i += n;
    }
    Some(out)
}

/// ftfy's `remove_control_chars`.
pub(crate) fn remove_control_chars(text: &[u8]) -> Option<Vec<u8>> {
    let candidate = text
        .iter()
        .any(|&b| b < 0x20 && !matches!(b, 0x09 | 0x0A | 0x0C | 0x0D))
        || text.iter().any(|&b| b == 0x7F || b == 0xE2 || b == 0xEF);
    if !candidate {
        return None;
    }
    translate(text, |cp, _| {
        let hit = matches!(
            cp,
            0x00..=0x08 | 0x0B | 0x0E..=0x1F | 0x7F | 0x206A..=0x206F | 0xFEFF | 0xFFF9..=0xFFFC
        );
        debug_assert_eq!(hit, CONTROL_CHARS.binary_search(&cp).is_ok());
        hit.then_some(Rep::Delete)
    })
}

/// ftfy's `fix_c1_controls`: decode C1 controls as `sloppy-windows-1252`.
pub(crate) fn fix_c1_controls(text: &[u8]) -> Option<Vec<u8>> {
    if !has_c1(text) {
        return None;
    }
    translate(text, |cp, _| {
        (0x80..=0x9F)
            .contains(&cp)
            .then(|| Rep::Cp(SLOPPY_1252_C1[(cp - 0x80) as usize]))
    })
    .filter(|v| v != text)
}

/// `C1_CONTROL_RE.search(text)`.
#[inline]
pub(crate) fn has_c1(text: &[u8]) -> bool {
    text.windows(2)
        .any(|w| w[0] == 0xC2 && (0x80..=0x9F).contains(&w[1]))
}

// ---------------------------------------------------------------------------
// byte transcoders used inside fix_encoding
// ---------------------------------------------------------------------------

#[inline]
fn cont_no_85_a0(b: u8) -> bool {
    matches!(b, 0x80..=0x84 | 0x86..=0x9F | 0xA1..=0xBF)
}

#[inline]
fn cont(b: u8) -> bool {
    (0x80..=0xBF).contains(&b)
}

/// Length of an `ALTERED_UTF8_RE` match at `i` (alternatives in order).
#[inline]
fn altered_at(b: &[u8], i: usize) -> Option<usize> {
    let at = |k: usize| b.get(i + k).copied();
    let sp = |k: usize| at(k) == Some(b' ');
    let c = |k: usize, f: fn(u8) -> bool| at(k).is_some_and(f);
    let b0 = b[i];
    if matches!(b0, 0xC2 | 0xC3 | 0xC5 | 0xCE | 0xD0 | 0xD9) && sp(1) {
        return Some(2);
    }
    if matches!(b0, 0xE2 | 0xE3) && sp(1) && c(2, cont_no_85_a0) {
        return Some(3);
    }
    if matches!(b0, 0xE0..=0xE3) && c(1, cont_no_85_a0) && sp(2) {
        return Some(3);
    }
    let four = (sp(1) && c(2, cont) && c(3, cont))
        || (c(1, cont) && sp(2) && c(3, cont))
        || (c(1, cont) && c(2, cont) && sp(3));
    (b0 == 0xF0 && four).then_some(4)
}

/// `ALTERED_UTF8_RE.search(byts)`.
pub(crate) fn has_altered_utf8(b: &[u8]) -> bool {
    (0..b.len()).any(|i| altered_at(b, i).is_some())
}

/// ftfy's `restore_byte_a0`.
pub(crate) fn restore_byte_a0(byts: &[u8]) -> Vec<u8> {
    // A_GRAVE_WORD_RE: b"\xc3 (?! |quele|quela|quilo|s )" -> b"\xc3\xa0 "
    let mut first = Vec::with_capacity(byts.len() + 8);
    let mut i = 0;
    while i < byts.len() {
        if byts[i] == 0xC3 && byts.get(i + 1) == Some(&b' ') {
            let rest = &byts[i + 2..];
            let excluded = [&b" "[..], b"quele", b"quela", b"quilo", b"s "]
                .iter()
                .any(|p| rest.starts_with(p));
            if !excluded {
                first.extend_from_slice(b"\xc3\xa0 ");
                i += 2;
                continue;
            }
        }
        first.push(byts[i]);
        i += 1;
    }
    // ALTERED_UTF8_RE.sub(lambda m: m.group(0).replace(b" ", b"\xa0"), ...)
    let mut out = Vec::with_capacity(first.len());
    let mut i = 0;
    while i < first.len() {
        if let Some(n) = altered_at(&first, i) {
            out.extend(
                first[i..i + n]
                    .iter()
                    .map(|&b| if b == b' ' { 0xA0 } else { b }),
            );
            i += n;
        } else {
            out.push(first[i]);
            i += 1;
        }
    }
    out
}

type ByteClass = fn(u8) -> bool;

fn is_1a(x: u8) -> bool {
    x == 0x1A
}
fn is_q(x: u8) -> bool {
    x == b'?'
}
fn q_1a(x: u8) -> bool {
    x == 0x1A || x == b'?'
}
fn c_1a(x: u8) -> bool {
    x == 0x1A || cont(x)
}
fn qc_1a(x: u8) -> bool {
    x == 0x1A || x == b'?' || cont(x)
}
fn lead2(x: u8) -> bool {
    matches!(x, 0xC2..=0xDF)
}
fn lead_c2_c3(x: u8) -> bool {
    matches!(x, 0xC2..=0xC3)
}
fn is_ed(x: u8) -> bool {
    x == 0xED
}
fn hi_a0_af(x: u8) -> bool {
    matches!(x, 0xA0..=0xAF)
}
fn lo_b0_bf(x: u8) -> bool {
    matches!(x, 0xB0..=0xBF)
}
fn lead3(x: u8) -> bool {
    matches!(x, 0xE0..=0xEF)
}
fn lead4(x: u8) -> bool {
    matches!(x, 0xF0..=0xF4)
}

/// `LOSSY_UTF8_RE`, alternative by alternative (the final `\x1a` is separate).
static LOSSY_ALTS: [&[ByteClass]; 9] = [
    &[lead2, is_1a],
    &[lead_c2_c3, is_q],
    &[is_ed, hi_a0_af, q_1a, is_ed, lo_b0_bf, qc_1a],
    &[is_ed, hi_a0_af, qc_1a, is_ed, lo_b0_bf, q_1a],
    &[lead3, q_1a, c_1a],
    &[lead3, c_1a, q_1a],
    &[lead4, q_1a, c_1a, c_1a],
    &[lead4, c_1a, q_1a, c_1a],
    &[lead4, c_1a, c_1a, q_1a],
];

/// Length of a `LOSSY_UTF8_RE` match at `i` (alternatives in order).
#[inline]
fn lossy_at(b: &[u8], i: usize) -> Option<usize> {
    for alt in &LOSSY_ALTS {
        if b.len() - i >= alt.len() && alt.iter().enumerate().all(|(k, f)| f(b[i + k])) {
            return Some(alt.len());
        }
    }
    (b[i] == 0x1A).then_some(1)
}

/// ftfy's `replace_lossy_sequences`.
pub(crate) fn replace_lossy_sequences(byts: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(byts.len() + 8);
    let mut i = 0;
    while i < byts.len() {
        if let Some(n) = lossy_at(byts, i) {
            out.extend_from_slice("\u{fffd}".as_bytes());
            i += n;
        } else {
            out.push(byts[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_0;

    fn s(v: Option<Vec<u8>>, orig: &str) -> String {
        String::from_utf8(v.unwrap_or_else(|| orig.as_bytes().to_vec())).unwrap()
    }

    #[test]
    fn html() {
        let f = |t: &str| s(unescape_html(t.as_bytes()), t);
        // doctests from ftfy.fixes.unescape_html
        assert_eq!(f("&lt;tag&gt;"), "<tag>");
        assert_eq!(f("&Jscr;ohn &HilbertSpace;ancock"), "𝒥ohn ℋancock");
        assert_eq!(f("&checkmark;"), "✓");
        assert_eq!(f("P&eacute;rez"), "Pérez");
        assert_eq!(f("P&EACUTE;REZ"), "PÉREZ");
        assert_eq!(f("BUNDESSTRA&SZLIG;E"), "BUNDESSTRASSE");
        assert_eq!(f("&ntilde; &Ntilde; &NTILDE; &nTILDE;"), "ñ Ñ Ñ &nTILDE;");
        // numeric references
        assert_eq!(f("&#x1F600;&#65;&#x41;"), "😀AA");
        assert_eq!(f("&#59;"), "&#59;");
        assert_eq!(f("&#1;x"), "x");
        assert_eq!(f("&#128;"), "€");
        assert_eq!(f("&#xD800;"), "\u{fffd}");
        assert_eq!(f("&#12ab;"), "&#12ab;");
        assert_eq!(f("&amp;amp;"), "&amp;");
        assert!(unescape_html(b"no entities").is_none());
    }

    #[test]
    fn terminal_escapes() {
        let t = "\x1b[36;44mI'm blue, da ba dee da ba doo...\x1b[0m";
        assert_eq!(
            s(remove_terminal_escapes(t.as_bytes(), V), t),
            "I'm blue, da ba dee da ba doo..."
        );
        assert!(remove_terminal_escapes(b"\x1b[12", V).is_none());
    }

    #[test]
    fn character_fixers() {
        let t = "\u{201c}here\u{2019}s a test\u{201d}";
        assert_eq!(s(uncurl_quotes(t.as_bytes()), t), "\"here's a test\"");
        let t = "ﬂuﬃeﬆ";
        assert_eq!(s(fix_latin_ligatures(t.as_bytes()), t), "fluffiest");
        let t = "ＬＯＵＤ　ＮＯＩＳＥＳ";
        assert_eq!(s(fix_character_width(t.as_bytes()), t), "LOUD NOISES");
        let t = "Ｕﾀｰﾝ";
        assert_eq!(s(fix_character_width(t.as_bytes()), t), "Uターン");
        let t = "a\r\nb\rc\u{2028}d\u{2029}e\u{85}f\r\r\n";
        assert_eq!(s(fix_line_breaks(t.as_bytes()), t), "a\nb\nc\nd\ne\nf\n\n");
        let t = "a\x00b\x7f\u{feff}c\td";
        assert_eq!(s(remove_control_chars(t.as_bytes()), t), "abc\td");
        let t = "\u{80}\u{81}";
        assert_eq!(s(fix_c1_controls(t.as_bytes()), t), "€\u{81}");
    }

    #[test]
    fn surrogates() {
        let mut t = Vec::new();
        for cp in [0xD83D, 0xDCA9, 0x41, 0xDCA9, 0xD83D] {
            push_cp(&mut t, cp);
        }
        assert_eq!(
            String::from_utf8(fix_surrogates(&t).unwrap()).unwrap(),
            "💩A\u{fffd}\u{fffd}"
        );
    }

    #[test]
    fn transcoders() {
        assert_eq!(restore_byte_a0(b"voil\xc3 le"), b"voil\xc3\xa0 le");
        assert_eq!(restore_byte_a0(b"\xc3 s "), b"\xc3\xa0s ");
        assert_eq!(
            replace_lossy_sequences(b"\xe2\x80\x1a"),
            "\u{fffd}".as_bytes()
        );
        assert_eq!(replace_lossy_sequences(b"a\x1ab"), "a\u{fffd}b".as_bytes());
        assert!(has_altered_utf8(b"\xc3 "));
        assert!(!has_altered_utf8(b"\xc4 "));
    }
}
