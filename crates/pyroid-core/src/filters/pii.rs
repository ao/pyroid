// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! datatrove's `PIIFormatter`: email and IPv4 address replacement.
//!
//! datatrove rewrites `doc.text` with two `re.sub` passes, in this order:
//!
//! 1. every match of [`EMAIL_PATTERN`] becomes the next email replacement;
//! 2. on that result, every match of [`IP_PATTERN`] that
//!    `ipaddress.ip_address` accepts and (by default) calls `is_global`
//!    becomes the next IP replacement; other matches stay as they are.
//!
//! "Next" cycles through the replacement tuple with a counter that belongs to
//! the formatter, so it carries over from one document to the next.
//!
//! The matchers here find exactly the matches CPython's `re` finds for these
//! two patterns: leftmost first, then the regex's backtracking priority. Both
//! patterns are ASCII except the email pattern's leading `\b`, whose word
//! characters are `re`'s Unicode `\w` for the interpreter's Unicode version.
//! They run in linear time (a failed attempt at one start proves the starts
//! up to the next separator fail too), where `re` can be quadratic.
//!
//! Whether an address is global is the interpreter's business (its
//! `ipaddress` tables changed in CPython patch releases), so this module only
//! finds candidates and checks that they parse ([`parse_ipv4`]); the caller
//! decides the rest, e.g. by asking Python once per distinct address.

use std::borrow::Cow;
use std::ops::Range;

use super::unicode::is_word;
use crate::ftfy::UnicodeVersion;

/// datatrove's email regex.
pub const EMAIL_PATTERN: &str = concat!(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:(?:[A-Za-z0-9](?:[",
    r"A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[",
    r"01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[A-Za-z0-9-]*[A-Za-z0-9]:)])",
);

/// datatrove's IPv4 regex (no anchors: it also matches inside longer numbers).
pub const IP_PATTERN: &str =
    r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)";

/// `PIIFormatter`'s default `email_replacement`.
pub const DEFAULT_EMAIL_REPLACEMENTS: [&str; 2] =
    ["email@example.com", "firstname.lastname@example.org"];

/// `PIIFormatter`'s default `ip_replacement`.
pub const DEFAULT_IP_REPLACEMENTS: [&str; 6] = [
    "22.214.171.124",
    "126.96.36.199",
    "188.8.131.52",
    "184.108.40.206",
    "220.127.116.11",
    "18.104.22.168",
];

/// The email pattern's local-part class `[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]`.
#[inline]
fn is_local(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+/=?^_`{|}~-".contains(&b)
}

/// `[A-Za-z0-9-]`.
#[inline]
fn is_label(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-'
}

/// End of the run of bytes satisfying `f` that starts at `i`.
#[inline]
fn run(b: &[u8], i: usize, f: impl Fn(u8) -> bool) -> usize {
    i + b[i..].iter().take_while(|&&c| f(c)).count()
}

/// Is `d` an octet the patterns accept: 1 to 3 ASCII digits worth at most 255
/// (`25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?` followed by a non-digit; leading
/// zeros allowed)?
#[inline]
fn octet(d: &[u8]) -> bool {
    (1..=3).contains(&d.len()) && d.iter().fold(0u32, |v, &c| v * 10 + u32::from(c - b'0')) <= 255
}

/// Every match of [`EMAIL_PATTERN`] in `text`, as `re.sub` / `re.finditer`
/// find them, appended to `out` as byte ranges.
pub fn email_matches(text: &str, version: UnicodeVersion, out: &mut Vec<Range<usize>>) {
    let b = text.as_bytes();
    // A match starts before an `@` (none: nothing to do).
    let Some(last_at) = memchr::memrchr(b'@', b) else {
        return;
    };
    let mut pos = 0;
    while let Some(off) = b
        .get(pos..)
        .and_then(|rest| rest.iter().position(|&c| is_local(c)))
    {
        let s = pos + off;
        if s >= last_at {
            break;
        }
        // `\b`: `\w` differs on the two sides of `s` (the local-part
        // character at `s` is ASCII, so `s` is a char boundary).
        let here = b[s].is_ascii_alphanumeric() || b[s] == b'_';
        let before = text[..s]
            .chars()
            .next_back()
            .is_some_and(|c| is_word(c, version));
        if here == before {
            pos = s + 1;
            continue;
        }
        match email_at(b, s) {
            Ok(end) => {
                out.push(s..end);
                pos = end;
            }
            // Every start up to `skip` reaches the same failure.
            Err(skip) => pos = skip,
        }
    }
}

/// The email match starting at `s` (where `\b` holds and a local-part
/// character is), or the position up to which no start can match.
fn email_at(b: &[u8], s: usize) -> Result<usize, usize> {
    // `[L]+(?:\.[L]+)*@`: each `[L]+` must take its whole run (a shorter one
    // is followed by an `L`, which is neither `.` nor `@`), so this is the
    // only way to reach the `@`.
    let mut i = run(b, s, is_local);
    while i + 1 < b.len() && b[i] == b'.' && is_local(b[i + 1]) {
        i = run(b, i + 1, is_local);
    }
    if b.get(i) != Some(&b'@') {
        return Err(i);
    }
    domain(b, i + 1).ok_or(i)
}

/// The end of the domain alternative matched at `p`, if any.
fn domain(b: &[u8], p: usize) -> Option<usize> {
    match b.get(p) {
        Some(c) if c.is_ascii_alphanumeric() => labels(b, p),
        Some(b'[') => bracketed(b, p + 1),
        _ => None,
    }
}

/// `(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?`
/// at `p`, with `re`'s backtracking order.
fn labels(b: &[u8], p: usize) -> Option<usize> {
    // A `label\.` group can only end at the end of its `[A-Za-z0-9-]` run
    // (anything shorter is followed by a run character, not a dot), so the
    // greedy `+` takes groups for as long as a run ends in an alphanumeric
    // followed by a dot.
    let mut groups = 0usize;
    let mut last_label_end = p;
    let mut cur = p;
    while b.get(cur).is_some_and(u8::is_ascii_alphanumeric) {
        let r = run(b, cur, is_label);
        if !(b[r - 1].is_ascii_alphanumeric() && b.get(r) == Some(&b'.')) {
            break;
        }
        groups += 1;
        last_label_end = r;
        cur = r + 1;
    }
    if groups == 0 {
        return None;
    }
    if b.get(cur).is_some_and(u8::is_ascii_alphanumeric) {
        // The final label: its run, up to the run's last alphanumeric.
        let r = run(b, cur, is_label);
        let last = b[cur..r].iter().rposition(u8::is_ascii_alphanumeric)?;
        return Some(cur + last + 1);
    }
    // No final label after the last group: give that group back, and its
    // label (which ends in an alphanumeric) is the final one.
    (groups >= 2).then_some(last_label_end)
}

/// `(?:octet\.){3}(?:octet|[A-Za-z0-9-]*[A-Za-z0-9]:)]` after the `[`.
fn bracketed(b: &[u8], mut j: usize) -> Option<usize> {
    for _ in 0..3 {
        let r = run(b, j, |c| c.is_ascii_digit());
        if !octet(&b[j..r]) || b.get(r) != Some(&b'.') {
            return None;
        }
        j = r + 1;
    }
    let r = run(b, j, |c| c.is_ascii_digit());
    if octet(&b[j..r]) && b.get(r) == Some(&b']') {
        return Some(r + 1);
    }
    let r = run(b, j, is_label);
    (r > j && b[r - 1].is_ascii_alphanumeric() && b.get(r..r + 2) == Some(b":]")).then_some(r + 2)
}

/// Every match of [`IP_PATTERN`] in `text`, as `re.sub` finds them, appended
/// to `out` as byte ranges.
pub fn ip_matches(text: &str, out: &mut Vec<Range<usize>>) {
    let b = text.as_bytes();
    // Every match contains a dot.
    if memchr::memchr(b'.', b).is_none() {
        return;
    }
    let mut pos = 0;
    while let Some(off) = b
        .get(pos..)
        .and_then(|rest| rest.iter().position(u8::is_ascii_digit))
    {
        let first = pos + off;
        let end = run(b, first, |c| c.is_ascii_digit());
        // The first octet runs to the end of its digit run (a dot must follow
        // it) and has at most 3 digits: only the last 3 starts can match.
        let found = (first.max(end.saturating_sub(3))..end).find_map(|s| ip_at(b, s).map(|e| s..e));
        match found {
            Some(m) => {
                pos = m.end;
                out.push(m);
            }
            None => pos = end,
        }
    }
}

/// The IP match starting at `s`, if any.
fn ip_at(b: &[u8], s: usize) -> Option<usize> {
    let mut j = s;
    for _ in 0..3 {
        let r = run(b, j, |c| c.is_ascii_digit());
        if !octet(&b[j..r]) || b.get(r) != Some(&b'.') {
            return None;
        }
        j = r + 1;
    }
    last_octet(b, j).map(|n| j + n)
}

/// The length of `25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?` at `j` when nothing
/// has to follow it: the first alternative that matches wins.
fn last_octet(b: &[u8], j: usize) -> Option<usize> {
    let at = |k: usize| b.get(j + k).copied();
    let digit = |k: usize| at(k).is_some_and(|c| c.is_ascii_digit());
    let c0 = at(0).filter(u8::is_ascii_digit)?;
    if c0 == b'2' && at(1) == Some(b'5') && at(2).is_some_and(|c| (b'0'..=b'5').contains(&c)) {
        return Some(3);
    }
    if c0 == b'2' && at(1).is_some_and(|c| (b'0'..=b'4').contains(&c)) && digit(2) {
        return Some(3);
    }
    if (c0 == b'0' || c0 == b'1') && digit(1) {
        return Some(if digit(2) { 3 } else { 2 });
    }
    Some(if digit(1) { 2 } else { 1 })
}

/// The address `ipaddress.ip_address(s)` gives for a dotted-quad `s`, or
/// `None` where it raises `ValueError` (not 4 decimal octets of at most 3
/// digits, an octet above 255, or a leading zero, which CPython rejects
/// since 3.9.5).
pub fn parse_ipv4(s: &str) -> Option<u32> {
    let mut value = 0u32;
    let mut parts = 0;
    for part in s.split('.') {
        let d = part.as_bytes();
        if parts == 4 || !d.iter().all(u8::is_ascii_digit) || !octet(d) {
            return None;
        }
        if d.len() > 1 && d[0] == b'0' {
            return None;
        }
        value = (value << 8) | d.iter().fold(0, |v, &c| v * 10 + u32::from(c - b'0'));
        parts += 1;
    }
    (parts == 4).then_some(value)
}

/// `regex.sub(get_replacement, text)` for the given matches: match `i` is
/// replaced when `valid(i)`, by `replacements[(start + k) % len]` for the
/// `k`-th valid match. Returns the text (borrowed when nothing was replaced)
/// and the number of replacements made.
///
/// # Errors
/// [`NoReplacements`] when a match must be replaced but `replacements` is
/// empty (datatrove's `IndexError`).
pub fn replace<'a, S: AsRef<str>>(
    text: &'a str,
    matches: &[Range<usize>],
    valid: impl Fn(usize) -> bool,
    replacements: &[S],
    start: usize,
) -> Result<(Cow<'a, str>, usize), NoReplacements> {
    let mut out = String::new();
    let mut last = 0;
    let mut used = 0;
    for (i, m) in matches.iter().enumerate() {
        if !valid(i) {
            continue;
        }
        if replacements.is_empty() {
            return Err(NoReplacements);
        }
        if used == 0 {
            out.reserve(text.len());
        }
        out.push_str(&text[last..m.start]);
        out.push_str(replacements[(start + used) % replacements.len()].as_ref());
        last = m.end;
        used += 1;
    }
    if used == 0 {
        return Ok((Cow::Borrowed(text), 0));
    }
    out.push_str(&text[last..]);
    Ok((Cow::Owned(out), used))
}

/// A match had to be replaced from an empty replacement tuple.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoReplacements;

impl std::fmt::Display for NoReplacements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("tuple index out of range")
    }
}

impl std::error::Error for NoReplacements {}

/// `PIIFormatter` settings (its constructor arguments).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiiConfig {
    pub remove_emails: bool,
    pub remove_ips: bool,
    pub only_remove_public_ips: bool,
    pub email_replacements: Vec<String>,
    pub ip_replacements: Vec<String>,
}

impl Default for PiiConfig {
    fn default() -> Self {
        Self {
            remove_emails: true,
            remove_ips: true,
            only_remove_public_ips: true,
            email_replacements: DEFAULT_EMAIL_REPLACEMENTS.map(str::to_owned).to_vec(),
            ip_replacements: DEFAULT_IP_REPLACEMENTS.map(str::to_owned).to_vec(),
        }
    }
}

/// A stateful `PIIFormatter`, one document at a time (the Python bindings
/// batch documents instead, and ask Python which addresses are global).
#[derive(Clone, Debug)]
pub struct PiiFormatter {
    pub config: PiiConfig,
    /// The email replacer's `_replace_i`.
    pub email_index: usize,
    /// The IP replacer's `_replace_i`.
    pub ip_index: usize,
}

impl PiiFormatter {
    pub fn new(config: PiiConfig) -> Self {
        Self {
            config,
            email_index: 0,
            ip_index: 0,
        }
    }

    /// `format(text)`. `is_global` answers `ipaddress.ip_address(a).is_global`
    /// for an address that parses (used when `only_remove_public_ips`).
    ///
    /// # Errors
    /// [`NoReplacements`], as [`replace`].
    pub fn format<'a>(
        &mut self,
        text: &'a str,
        version: UnicodeVersion,
        is_global: impl Fn(u32) -> bool,
    ) -> Result<Cow<'a, str>, NoReplacements> {
        let mut m = Vec::new();
        let mut cur = Cow::Borrowed(text);
        if self.config.remove_emails {
            email_matches(&cur, version, &mut m);
            let reps = &self.config.email_replacements;
            let (out, used) = replace(&cur, &m, |_| true, reps, self.email_index)?;
            if used > 0 {
                self.email_index = (self.email_index + used) % reps.len();
                cur = Cow::Owned(out.into_owned());
            }
        }
        if self.config.remove_ips {
            m.clear();
            ip_matches(&cur, &mut m);
            let public = self.config.only_remove_public_ips;
            let ok =
                |i: usize| parse_ipv4(&cur[m[i].clone()]).is_some_and(|a| !public || is_global(a));
            let reps = &self.config.ip_replacements;
            let (out, used) = replace(&cur, &m, ok, reps, self.ip_index)?;
            if used > 0 {
                self.ip_index = (self.ip_index + used) % reps.len();
                let out = out.into_owned();
                cur = Cow::Owned(out);
            }
        }
        Ok(cur)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    fn emails(t: &str) -> Vec<&str> {
        let mut m = Vec::new();
        email_matches(t, V, &mut m);
        m.into_iter().map(|r| &t[r]).collect()
    }

    fn ips(t: &str) -> Vec<&str> {
        let mut m = Vec::new();
        ip_matches(t, &mut m);
        m.into_iter().map(|r| &t[r]).collect()
    }

    #[test]
    fn email_basics() {
        assert_eq!(emails("write to a.b@c.d.com now"), ["a.b@c.d.com"]);
        assert_eq!(emails("x@y"), Vec::<&str>::new());
        assert_eq!(emails("a@b.c-"), ["a@b.c"]);
        assert_eq!(emails("a@b.c-d"), ["a@b.c-d"]);
        assert_eq!(emails("a@b.c.-x"), ["a@b.c"]);
        assert_eq!(emails("a@b.-x"), Vec::<&str>::new());
        assert_eq!(emails("a@b..c"), Vec::<&str>::new());
        assert_eq!(emails("a@b-.c"), Vec::<&str>::new());
        assert_eq!(
            emails("a@[1.2.3.4] b@[1.2.3.x9:]"),
            ["a@[1.2.3.4]", "b@[1.2.3.x9:]"]
        );
        assert_eq!(emails("a@[1.2.3.256]"), Vec::<&str>::new());
        assert_eq!(emails("a@[01.002.3.4]"), ["a@[01.002.3.4]"]);
        assert_eq!(emails("a..b@c.de"), ["b@c.de"]);
    }

    #[test]
    fn email_word_boundary() {
        // `\b` before a symbol needs a word character before it.
        assert_eq!(emails(" -x@y.z"), ["x@y.z"]);
        assert_eq!(emails("a-x@y.z"), ["a-x@y.z"]);
        assert_eq!(emails("é-x@y.z"), ["-x@y.z"]);
        // After a word character, a word-character start has no boundary.
        assert_eq!(emails("éx@y.z"), Vec::<&str>::new());
        assert_eq!(emails("_x@y.z"), ["_x@y.z"]);
        // The match after a match: `\b` sees the previous match's last char.
        assert_eq!(emails("a@b.cd-e@f.gh"), ["a@b.cd-e"]);
    }

    #[test]
    fn ip_basics() {
        assert_eq!(ips("1.2.3.4"), ["1.2.3.4"]);
        assert_eq!(ips("1234.5.6.7"), ["234.5.6.7"]);
        assert_eq!(ips("1.2.3.256"), ["1.2.3.25"]);
        assert_eq!(ips("1.2.3.4567"), ["1.2.3.45"]);
        assert_eq!(
            ips("1.2.3.199 1.2.3.249 1.2.3.26x"),
            ["1.2.3.199", "1.2.3.249", "1.2.3.26"]
        );
        assert_eq!(ips("01.2.3.4.5.6.7.8"), ["01.2.3.4", "5.6.7.8"]);
        assert_eq!(ips("1.2.3"), Vec::<&str>::new());
        assert_eq!(ips("999.1.1.1.1"), ["99.1.1.1"]);
    }

    #[test]
    fn parse() {
        assert_eq!(parse_ipv4("1.2.3.4"), Some(0x0102_0304));
        assert_eq!(parse_ipv4("255.255.255.255"), Some(u32::MAX));
        assert_eq!(parse_ipv4("01.2.3.4"), None);
        assert_eq!(parse_ipv4("0.0.0.0"), Some(0));
        assert_eq!(parse_ipv4("1.2.3.256"), None);
        assert_eq!(parse_ipv4("1.2.3"), None);
        assert_eq!(parse_ipv4("1.2.3.4.5"), None);
    }

    #[test]
    fn formatter_cycles_across_documents() {
        let mut f = PiiFormatter::new(PiiConfig::default());
        let all = |_| true;
        let out = f.format("a@b.cc and c@d.ee", V, all).unwrap();
        assert_eq!(out, "email@example.com and firstname.lastname@example.org");
        let out = f
            .format("x@y.zz 8.8.8.8 10.0.0.1 08.8.8.8", V, |a| a >> 24 != 10)
            .unwrap();
        assert_eq!(out, "email@example.com 22.214.171.124 10.0.0.1 08.8.8.8");
        assert_eq!((f.email_index, f.ip_index), (1, 1));
        assert!(matches!(
            f.format("no pii here", V, all).unwrap(),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn empty_replacements() {
        let m: Vec<Range<usize>> = std::iter::once(0..1).collect();
        assert_eq!(
            replace::<&str>("x", &m, |_| true, &[], 0),
            Err(NoReplacements)
        );
        assert_eq!(replace::<&str>("x", &m, |_| false, &[], 0).unwrap().1, 0);
    }
}
