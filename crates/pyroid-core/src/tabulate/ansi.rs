// Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
//! tabulate's `_ansi_codes` regex, as a hand-written matcher with the same leftmost-first
//! results:
//!
//! ```text
//! ESC [ [\x30-\x3f]* [\x20-\x2f]* [\x40-\x7e]                          (CSI: colors etc.)
//! ESC ]8; (\w+=\w+:?)* ; ([^ESC]+) ESC \ ([^ESC]+) ESC ]8;; ESC \      (OSC 8 hyperlink)
//! ```
//!
//! `_strip_ansi` replaces each match with its link text (group 4; nothing for CSI). Python's
//! `\w` is Unicode-aware, so a non-ASCII char among a link's `key=value` parameters makes
//! the matcher answer [`Tri::Ask`].

use super::number::Tri;

const ESC: u8 = 0x1b;

/// The outcome of matching at one ESC.
enum At {
    /// No match here.
    No,
    /// A match ending at `end`; the replacement is `text` (a range of the input).
    Match { end: usize, text: (usize, usize) },
    /// Python's `\w` decides.
    Ask,
}

#[inline]
fn is_word_ascii(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Do the link parameters `p` (chars in `[\w=:]`) match `(\w+=\w+:?)*`?
fn params_ok(p: &[u8]) -> bool {
    // NFA states: 0 start (accepting), 1 in a key, 2 after '=', 3 in a value (accepting; a
    // word char may also start the next key), 4 after ':' (accepting).
    let mut states: u8 = 1 << 0;
    for &b in p {
        let mut next = 0u8;
        let word = is_word_ascii(b);
        for s in 0..5 {
            if states & (1 << s) == 0 {
                continue;
            }
            match (s, b) {
                (0 | 1 | 4, _) if word => next |= 1 << 1,
                (1, b'=') => next |= 1 << 2,
                (2, _) if word => next |= 1 << 3,
                (3, _) if word => next |= (1 << 3) | (1 << 1),
                (3, b':') => next |= 1 << 4,
                _ => {}
            }
        }
        states = next;
        if states == 0 {
            return false;
        }
    }
    states & ((1 << 0) | (1 << 3) | (1 << 4)) != 0
}

#[allow(clippy::many_single_char_names)] // positions in the input
fn match_at(s: &[u8], i: usize) -> At {
    debug_assert_eq!(s[i], ESC);
    let n = s.len();
    if i + 1 < n && s[i + 1] == b'[' {
        let mut j = i + 2;
        while j < n && (0x30..=0x3f).contains(&s[j]) {
            j += 1;
        }
        while j < n && (0x20..=0x2f).contains(&s[j]) {
            j += 1;
        }
        if j < n && (0x40..=0x7e).contains(&s[j]) {
            return At::Match {
                end: j + 1,
                text: (0, 0),
            };
        }
        return At::No;
    }
    if !s[i + 1..].starts_with(b"]8;") {
        return At::No;
    }
    let p0 = i + 4;
    let mut j = p0;
    while j < n && (is_word_ascii(s[j]) || s[j] == b'=' || s[j] == b':') {
        j += 1;
    }
    if j < n && s[j] >= 0x80 {
        return At::Ask;
    }
    if j >= n || s[j] != b';' || !params_ok(&s[p0..j]) {
        return At::No;
    }
    // URI: one or more non-ESC bytes, then ESC '\'.
    let uri = j + 1;
    let Some(k) = memchr::memchr(ESC, &s[uri..]).map(|k| uri + k) else {
        return At::No;
    };
    if k == uri || s.get(k + 1) != Some(&b'\\') {
        return At::No;
    }
    let text = k + 2;
    let Some(m) = memchr::memchr(ESC, &s[text..]).map(|m| text + m) else {
        return At::No;
    };
    if m == text || !s[m..].starts_with(b"\x1b]8;;\x1b\\") {
        return At::No;
    }
    At::Match {
        end: m + 7,
        text: (text, m),
    }
}

/// `_ansi_codes.sub(r"\4", s)`: `None` if nothing matched (the text is unchanged), `Err(())`
/// if Python has to decide.
#[allow(clippy::result_unit_err)]
pub fn strip(s: &str) -> Result<Option<String>, ()> {
    let b = s.as_bytes();
    let Some(first) = memchr::memchr(ESC, b) else {
        return Ok(None);
    };
    let mut out: Option<String> = None;
    let mut copied = 0;
    let mut i = first;
    loop {
        match match_at(b, i) {
            At::Ask => return Err(()),
            At::Match { end, text } => {
                let o = out.get_or_insert_with(|| String::with_capacity(s.len()));
                o.push_str(&s[copied..i]);
                o.push_str(&s[text.0..text.1]);
                copied = end;
                i = end;
            }
            At::No => i += 1,
        }
        match memchr::memchr(ESC, &b[i.min(b.len())..]) {
            Some(k) => i += k,
            None => break,
        }
    }
    if let Some(o) = out.as_mut() {
        o.push_str(&s[copied..]);
    }
    Ok(out)
}

/// `_ansi_codes.search(s) is not None`.
pub fn contains(s: &str) -> Tri {
    let b = s.as_bytes();
    let mut i = 0;
    while let Some(k) = memchr::memchr(ESC, &b[i..]) {
        i += k;
        match match_at(b, i) {
            At::Match { .. } => return Tri::Yes,
            At::Ask => return Tri::Ask,
            At::No => i += 1,
        }
    }
    Tri::No
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(s: &str) -> String {
        strip(s).unwrap().unwrap_or_else(|| s.to_owned())
    }

    #[test]
    fn strips_like_tabulate() {
        assert_eq!(st("\x1b[31mred\x1b[0m text"), "red text");
        assert_eq!(
            st("\x1b]8;;https://example.com\x1b\\This is a link\x1b]8;;\x1b\\"),
            "This is a link"
        );
        assert_eq!(st("\x1b]8;id=1:a=b;u\x1b\\t\x1b]8;;\x1b\\!"), "t!");
        assert_eq!(st("\x1b]8;a=bc=d;u\x1b\\t\x1b]8;;\x1b\\"), "t");
        assert_eq!(
            st("\x1b]8;a;u\x1b\\t\x1b]8;;\x1b\\"),
            "\x1b]8;a;u\x1b\\t\x1b]8;;\x1b\\"
        );
        assert_eq!(
            st("\x1b]8;;\x1b\\t\x1b]8;;\x1b\\"),
            "\x1b]8;;\x1b\\t\x1b]8;;\x1b\\"
        );
        assert_eq!(st("\x1b[1;2"), "\x1b[1;2");
        assert_eq!(st("a\x1b\x1b[0mb\x1b"), "a\x1bb\x1b");
        assert_eq!(st("\x1b[ 1m"), "\x1b[ 1m");
        assert_eq!(st("\x1b[1 m"), "");
        assert!(strip("plain").unwrap().is_none());
        assert!(strip("\x1b]8;é=1;u\x1b\\t\x1b]8;;\x1b\\").is_err());
    }

    #[test]
    fn params_language() {
        for ok in ["", "a=b", "a=b:", "a=b:c=d", "a=bc=d", "a_1=2:"] {
            assert!(params_ok(ok.as_bytes()), "{ok:?}");
        }
        for bad in ["a", "=b", "a=", "a==b", "a=b::", ":", "a:b", "a=b=c"] {
            assert!(!params_ok(bad.as_bytes()), "{bad:?}");
        }
    }

    #[test]
    fn search() {
        assert_eq!(contains("x\x1b[0m"), Tri::Yes);
        assert_eq!(contains("x\x1b"), Tri::No);
        assert_eq!(contains("plain"), Tri::No);
    }
}
