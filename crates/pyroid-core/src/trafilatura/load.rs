// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! utils.load_html for a `str` input (decode_file returns it unchanged), with
//! repair_faulty_html, on the string's surrogatepass UTF-8 ("WTF-8").
//!
//! repair_faulty_html(s, s[:50].lower()):
//! 1. `INVALID_XML_CHARS.sub("", s)`: `[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]`;
//! 2. if "doctype" is in the lowered first 50 characters: `DOCTYPE_TAG.sub("", firstline,
//!    count=1)` on the first line (`partition("\n")`, which always puts the "\n" back), with
//!    `DOCTYPE_TAG = ^< ?! ?DOCTYPE[^>]*/[^<>]*>` (re.I);
//! 3. if one of the first 4 lines of `s[:4096]` contains "<html" and ends with "/>":
//!    `FAULTY_HTML.sub(r"\1>", s, count=1)` with `FAULTY_HTML = (<html.*?)\s*/>` (re.I).
//!
//! trafilatura 1.11.0 (`repair_faulty_html_for`) skips step 1, uses
//! `DOCTYPE_TAG = ^< ?! ?DOCTYPE.+?/ ?>` in step 2, and looks at the first 4 lines of the
//! whole string in step 3.
//!
//! Every literal in these patterns is ASCII without the letters i, k and s (the only ones
//! `re.IGNORECASE` equates with non-ASCII characters), and only ASCII lower-cases to ASCII
//! letters, so they are matched on bytes; Python's `\s`, line boundaries and code-point counts
//! are handled on the decoded code points, with lone surrogates as ordinary characters.

// index names follow the regex walk (i, j, k, n, p, s)
#![allow(clippy::many_single_char_names)]

use crate::html::lxml;
use crate::html::tree::Document;

/// The code point starting at byte `i` of WTF-8 `s` and its length (a sequence cut off at the
/// end counts as one byte).
fn cp_at(s: &[u8], i: usize) -> (u32, usize) {
    let c = u32::from(s[i]);
    let n = if c < 0x80 {
        1
    } else if c < 0xE0 {
        2
    } else if c < 0xF0 {
        3
    } else {
        4
    };
    if i + n > s.len() || n == 1 {
        return (c, 1);
    }
    let cont = |k: usize| u32::from(s[i + k]) & 0x3F;
    let cp = match n {
        2 => ((c & 0x1F) << 6) | cont(1),
        3 => ((c & 0x0F) << 12) | (cont(1) << 6) | cont(2),
        _ => ((c & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
    };
    (cp, n)
}

/// Byte offset after the first `n` code points.
fn prefix_bytes(s: &[u8], n: usize) -> usize {
    let mut seen = 0;
    let mut i = 0;
    while i < s.len() {
        if seen == n {
            return i;
        }
        i += cp_at(s, i).1;
        seen += 1;
    }
    s.len()
}

fn py_isspace(c: u32) -> bool {
    char::from_u32(c).is_some_and(super::pystr::is_space)
}

/// `"doctype" in s[:50].lower()`.
fn has_doctype(raw: &[u8]) -> bool {
    raw[..prefix_bytes(raw, 50)]
        .windows(7)
        .any(|w| w.eq_ignore_ascii_case(b"doctype"))
}

/// utils.is_dubious_html(s[:50].lower()): `"html" not in beginning`.
fn is_dubious_html(raw: &[u8]) -> bool {
    !raw[..prefix_bytes(raw, 50)]
        .windows(4)
        .any(|w| w.eq_ignore_ascii_case(b"html"))
}

/// `INVALID_XML_CHARS.sub("", s)`.
fn remove_invalid(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let b = s[i];
        if matches!(b, 0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F) {
            i += 1;
            continue;
        }
        if b == 0xEF && s.get(i + 1) == Some(&0xBF) && matches!(s.get(i + 2), Some(0xBE | 0xBF)) {
            i += 3;
            continue;
        }
        out.push(b);
        i += 1;
    }
    out
}

/// Length of the `^< ?! ?DOCTYPE[^>]*/[^<>]*>` match at the start of `line`, if any.
fn doctype_match(line: &[u8]) -> Option<usize> {
    let mut i = 0;
    if line.first() != Some(&b'<') {
        return None;
    }
    i += 1;
    if line.get(i) == Some(&b' ') {
        i += 1;
    }
    if line.get(i) != Some(&b'!') {
        return None;
    }
    i += 1;
    if line.get(i) == Some(&b' ') {
        i += 1;
    }
    if !line
        .get(i..i + 7)
        .is_some_and(|w| w.eq_ignore_ascii_case(b"doctype"))
    {
        return None;
    }
    i += 7;
    // [^>]* then '/': the '/' lies before the first '>'; [^<>]*> then needs no '<' between the
    // chosen '/' and that '>'. Backtracking tries the last '/' first, and an earlier one only
    // adds characters before the same '>'.
    let gt = i + line[i..].iter().position(|&b| b == b'>')?;
    let slash = i + line[i..gt].iter().rposition(|&b| b == b'/')?;
    if line[slash + 1..gt].contains(&b'<') {
        return None;
    }
    Some(gt + 1)
}

/// The first `max` lines of `s` as `str.splitlines()` cuts them.
fn first_lines(s: &[u8], max: usize) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let (mut start, mut i) = (0, 0);
    while i < s.len() && out.len() < max {
        let (c, n) = cp_at(s, i);
        if matches!(
            c,
            0x0A | 0x0B | 0x0C | 0x0D | 0x1C | 0x1D | 0x1E | 0x85 | 0x2028 | 0x2029
        ) {
            out.push(&s[start..i]);
            let mut end = i + n;
            if c == 0x0D && s.get(end) == Some(&b'\n') {
                end += 1;
            }
            start = end;
            i = end;
        } else {
            i += n;
        }
    }
    if out.len() < max && start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `FAULTY_HTML.sub(r"\1>", s, count=1)`; None when the pattern does not match. Linear: a
/// whitespace run is skipped once, and a failed start fails for every later start on its line.
fn faulty_html_sub(s: &[u8]) -> Option<Vec<u8>> {
    let mut from = 0;
    while let Some(p) = s[from..]
        .windows(5)
        .position(|w| w.eq_ignore_ascii_case(b"<html"))
    {
        let i = from + p;
        // (<html.*?)\s*/> : the lazy group grows one code point at a time, '.' excludes '\n'
        let mut j = i + 5;
        // end of the whitespace run that `j` is in (`\s*` from any position in it ends there)
        let mut ws_end = j;
        loop {
            if j >= ws_end {
                ws_end = j;
                while ws_end < s.len() {
                    let (c, n) = cp_at(s, ws_end);
                    if !py_isspace(c) {
                        break;
                    }
                    ws_end += n;
                }
            }
            let k = ws_end;
            if s.get(k..k + 2) == Some(b"/>") {
                let mut out = Vec::with_capacity(s.len());
                out.extend_from_slice(&s[..j]);
                out.push(b'>');
                out.extend_from_slice(&s[k + 2..]);
                return Some(out);
            }
            if j >= s.len() || s[j] == b'\n' {
                break;
            }
            j += cp_at(s, j).1;
        }
        // every end position a later start on this line could use was just tried
        from = j.max(i + 1);
    }
    None
}

/// utils.repair_faulty_html(s, s[:50].lower()) for the surrogatepass UTF-8 of a str `s`.
pub fn repair_faulty_html(raw: &[u8]) -> Vec<u8> {
    let doctype = has_doctype(raw);
    let mut s = remove_invalid(raw);
    if doctype {
        let nl = s.iter().position(|&b| b == b'\n');
        let (first, rest) = match nl {
            Some(n) => (&s[..n], &s[n + 1..]),
            None => (&s[..], &s[s.len()..]),
        };
        let cut = doctype_match(first).unwrap_or(0);
        let mut t = Vec::with_capacity(s.len() + 1);
        t.extend_from_slice(&first[cut..]);
        t.push(b'\n');
        t.extend_from_slice(rest);
        s = t;
    }
    let head = &s[..prefix_bytes(&s, 4096)];
    let trigger = first_lines(head, 4)
        .into_iter()
        .any(|l| l.windows(5).any(|w| w == b"<html") && l.ends_with(b"/>"));
    if trigger {
        if let Some(t) = faulty_html_sub(&s) {
            s = t;
        }
    }
    s
}

/// Length of trafilatura 1.11.0's `^< ?! ?DOCTYPE.+?/ ?>` match (re.I) at the start of
/// `line` (a first line: no "\n"), if any. The lazy `.+?` takes at least one code point and
/// stops at the first "/>" or "/ >".
fn doctype_match_v111(line: &[u8]) -> Option<usize> {
    let mut i = 0;
    if line.first() != Some(&b'<') {
        return None;
    }
    i += 1;
    if line.get(i) == Some(&b' ') {
        i += 1;
    }
    if line.get(i) != Some(&b'!') {
        return None;
    }
    i += 1;
    if line.get(i) == Some(&b' ') {
        i += 1;
    }
    if !line
        .get(i..i + 7)
        .is_some_and(|w| w.eq_ignore_ascii_case(b"doctype"))
    {
        return None;
    }
    i += 7;
    if i >= line.len() {
        return None;
    }
    let mut e = i + cp_at(line, i).1;
    while e < line.len() {
        if line[e] == b'/' {
            match (line.get(e + 1), line.get(e + 2)) {
                (Some(b' '), Some(b'>')) => return Some(e + 3),
                (Some(b'>'), _) => return Some(e + 2),
                _ => {}
            }
        }
        e += 1;
    }
    None
}

/// [`repair_faulty_html`] as the given release does it.
pub fn repair_faulty_html_for(raw: &[u8], version: super::Version) -> Vec<u8> {
    match version {
        super::Version::V2_3_1 => repair_faulty_html(raw),
        super::Version::V1_11_0 => repair_faulty_html_v111(raw),
    }
}

/// trafilatura 1.11.0's utils.repair_faulty_html(s, s[:50].lower()): the doctype fix (with
/// its own pattern), then the `<html .../>` fix when one of the first 4 lines of the whole
/// string calls for it; no control characters removed.
fn repair_faulty_html_v111(raw: &[u8]) -> Vec<u8> {
    let mut s = raw.to_vec();
    if has_doctype(raw) {
        let nl = s.iter().position(|&b| b == b'\n');
        let (first, rest) = match nl {
            Some(n) => (&s[..n], &s[n + 1..]),
            None => (&s[..], &s[s.len()..]),
        };
        let cut = doctype_match_v111(first).unwrap_or(0);
        let mut t = Vec::with_capacity(s.len() + 1);
        t.extend_from_slice(&first[cut..]);
        t.push(b'\n');
        t.extend_from_slice(rest);
        s = t;
    }
    let trigger = first_lines(&s, 4)
        .into_iter()
        .any(|l| l.windows(5).any(|w| w == b"<html") && l.ends_with(b"/>"));
    if trigger {
        if let Some(t) = faulty_html_sub(&s) {
            s = t;
        }
    }
    s
}

/// [`load_html`] as the given release does it (they differ in repair_faulty_html).
pub fn load_html_for(html: &[u8], version: super::Version) -> Option<Document> {
    match version {
        super::Version::V2_3_1 => load_html(html),
        super::Version::V1_11_0 => load_repaired(html, repair_faulty_html_v111(html)),
    }
}

/// lxml.html `_looks_like_full_html_unicode`: `^\s*<(?:html|!doctype)` (re.I).
pub fn looks_like_full_html(s: &str) -> bool {
    let t = s.trim_start_matches(super::pystr::is_space).as_bytes();
    t.get(..5).is_some_and(|w| w.eq_ignore_ascii_case(b"<html"))
        || t.get(..9)
            .is_some_and(|w| w.eq_ignore_ascii_case(b"<!doctype"))
}

/// utils.load_html for a `str` (given as surrogatepass UTF-8): the tree lxml returns, or None
/// where Trafilatura returns None (an empty or rejected document).
pub fn load_html(html: &[u8]) -> Option<Document> {
    load_repaired(html, repair_faulty_html(html))
}

/// load_html from the repair on: the parse of the str, the bytes fallback, the rejection test.
fn load_repaired(html: &[u8], repaired: Vec<u8>) -> Option<Document> {
    let check_flag = is_dubious_html(html);
    let bytes = || lxml::fromstring_bytes(&repaired).ok();
    let (tree, fallback) = match lxml::fromstring_wtf8(&repaired) {
        Ok(d) => (Some(d), false),
        Err(e) if e.is_value_error() => (bytes(), true),
        Err(_) => (None, false),
    };
    let len = |t: &Option<Document>| t.as_ref().map_or(0, |d| d.len(d.root));
    let tree = if (tree.is_none() || len(&tree) < 1) && !fallback {
        bytes()
    } else {
        tree
    };
    if tree.is_some() && check_flag && len(&tree) < 2 {
        return None;
    }
    tree
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doctype() {
        let r = repair_faulty_html(b"<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0//EN\">\n<html>");
        assert_eq!(r, b"\n<html>");
        assert_eq!(
            repair_faulty_html(b"<!doctype html><html>"),
            b"<!doctype html><html>\n"
        );
        assert_eq!(repair_faulty_html(b"< ! DocType a/b>x"), b"x\n");
        assert_eq!(
            repair_faulty_html(b"<!DOCTYPE a/<b>x"),
            b"<!DOCTYPE a/<b>x\n"
        );
    }

    #[test]
    fn faulty() {
        assert_eq!(
            repair_faulty_html(b"<html lang=\"en\" />\n<body>"),
            b"<html lang=\"en\">\n<body>"
        );
        // the line test is case-sensitive, the substitution is not
        assert_eq!(repair_faulty_html(b"<HTML/>"), b"<HTML/>");
        assert_eq!(repair_faulty_html(b"<html><HTML x/>"), b"<html><HTML x>");
        assert_eq!(repair_faulty_html(b"a\x01b\xef\xbf\xbec"), b"abc");
    }

    #[test]
    fn full_html() {
        assert!(looks_like_full_html(" \n<HTML>"));
        assert!(looks_like_full_html("\u{a0}<!DocType html>"));
        assert!(!looks_like_full_html("<head><html>"));
    }
}
