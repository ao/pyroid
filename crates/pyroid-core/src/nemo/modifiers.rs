// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! The string modifiers' `modify_document` (`nemo_curator.stages.text.modifiers.string` and
//! `.modifiers.fasttext`), with the results of NeMo's Python code.
//!
//! Every function returns `Cow::Borrowed` of the whole input when NeMo's result equals it.

use std::borrow::Cow;

use memchr::memmem;

use crate::text::is_py_whitespace;

/// NeMo's `policy_substrings` (`stages/text/utils/constants.py`), already lowercase ASCII.
pub(crate) const POLICY_SUBSTRINGS: [&str; 14] = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "privacy overview",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "this website uses cookies to improve your experience while you navigate through the website. out of these \
     cookies, the cookies that are categorized as necessary are stored on your browser as they are essential for \
     the working of basic functionalities of the website. we also use third-party cookies that help us analyze and \
     understand how you use this website. these cookies will be stored in your browser only with your consent. \
     you also have the option to opt-out of these cookies. but opting out of some of these cookies may have an \
     effect on your browsing experience.",
    "necessary cookies are absolutely essential for the website to function properly. this category only includes \
     cookies that ensures basic functionalities and security features of the website. these cookies do not store \
     any personal information.",
    "any cookies that may not be particularly necessary for the website to function and is used specifically to \
     collect user personal data via analytics, ads, other embedded contents are termed as non-necessary cookies. \
     it is mandatory to procure user consent prior to running these cookies on your website.",
    "this site uses cookies, including for analytics, personalization, and advertising purposes. for more \
     information or to change your cookie settings, click here.",
    "if you continue to browse this site without changing your cookie settings, you agree to this use. acceptread \
     more",
];

/// The bytes `s.lower()` gives, as far as matching lowercase ASCII needles goes: ASCII letters lowercased,
/// U+212A KELVIN SIGN as `k`, U+0130 as `i` plus a non-ASCII byte (its lowercase is `i` + U+0307), and every
/// other non-ASCII character as one byte that no needle contains.
pub(crate) fn lower_ascii_projection(s: &str, out: &mut Vec<u8>) {
    out.clear();
    out.reserve(s.len());
    for c in s.chars() {
        match c {
            '\0'..='\x7f' => out.push(u8::try_from(u32::from(c)).unwrap_or(0).to_ascii_lowercase()),
            '\u{212a}' => out.push(b'k'),
            '\u{130}' => out.extend_from_slice(b"i\xff"),
            _ => out.push(0xff),
        }
    }
}

/// Whether `lower` (a [`lower_ascii_projection`]) contains `"lorem ipsum"` / a policy substring.
fn boilerplate_kind(lower: &[u8]) -> (bool, bool) {
    let lorem = memmem::find(lower, b"lorem ipsum").is_some();
    let policy = !lorem
        && POLICY_SUBSTRINGS
            .iter()
            .any(|p| memmem::find(lower, p.as_bytes()).is_some());
    (lorem, policy)
}

/// NeMo's `is_paragraph_indices_in_top_or_bottom_only`: the indices (ascending) are contiguous and touch the
/// first or the last paragraph, and are not all of them.
fn top_or_bottom_only(indices: &[usize], num_paragraphs: usize) -> bool {
    if indices.len() == num_paragraphs {
        return false;
    }
    let contiguous = indices.windows(2).all(|w| w[0] + 1 == w[1]);
    contiguous && (indices.first() == Some(&0) || indices.last() == Some(&(num_paragraphs - 1)))
}

/// `BoilerPlateStringModifier(remove_if_at_top_or_bottom).modify_document(text)`.
///
/// A paragraph (`text.split("\n\n")`) with "lorem ipsum" empties the document. Otherwise, with
/// `remove_if_at_top_or_bottom=False` a boilerplate paragraph (one with a policy string such as "terms of use")
/// empties it; with `True`, the boilerplate paragraphs are dropped when they form one block at the top or the
/// bottom, and the document is kept as it is otherwise.
#[must_use]
pub fn remove_boilerplate_strings(text: &str, remove_if_at_top_or_bottom: bool) -> Cow<'_, str> {
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut indices = Vec::new();
    let mut lower = Vec::new();
    for (idx, p) in paragraphs.iter().enumerate() {
        lower_ascii_projection(p, &mut lower);
        let (lorem, policy) = boilerplate_kind(&lower);
        if lorem {
            return Cow::Borrowed("");
        }
        if policy {
            if !remove_if_at_top_or_bottom {
                return Cow::Borrowed("");
            }
            indices.push(idx);
        }
    }
    if indices.is_empty() || !top_or_bottom_only(&indices, paragraphs.len()) {
        return Cow::Borrowed(text);
    }
    let kept: Vec<&str> = paragraphs
        .iter()
        .enumerate()
        .filter(|(i, _)| indices.binary_search(i).is_err())
        .map(|(_, p)| *p)
        .collect();
    Cow::Owned(kept.join("\n\n"))
}

/// `re.sub(open + "(.*?)" + close, r"\1", line)` for a line without `\n`.
fn sub_lazy_pair<'a>(line: &'a str, open: &[u8], close: &[u8]) -> Cow<'a, str> {
    let b = line.as_bytes();
    let mut out: Option<String> = None;
    let mut pos = 0;
    while let Some(i) = memmem::find(&b[pos..], open).map(|i| pos + i) {
        let Some(j) = memmem::find(&b[i + open.len()..], close).map(|j| i + open.len() + j) else {
            // No closing delimiter after this opening one, so none after any later one either.
            break;
        };
        let o = out.get_or_insert_with(|| String::with_capacity(line.len()));
        o.push_str(&line[pos..i]);
        o.push_str(&line[i + open.len()..j]);
        pos = j + close.len();
    }
    match out {
        None => Cow::Borrowed(line),
        Some(mut o) => {
            o.push_str(&line[pos..]);
            Cow::Owned(o)
        }
    }
}

/// `re.sub(r"\[.*?\]\((.*?)\)", r"\1", line)` for a line without `\n`.
fn sub_link(line: &str) -> Cow<'_, str> {
    let b = line.as_bytes();
    let mut out: Option<String> = None;
    let mut pos = 0;
    while let Some(i) = memchr::memchr(b'[', &b[pos..]).map(|i| pos + i) {
        // `.*?` is lazy: the first "](" after the "[", then the first ")" after that. Without a ")" after the
        // first "](", there is none after a later one, and later "["s find the same "](" or a later one.
        let Some(k) = memmem::find(&b[i + 1..], b"](").map(|k| i + 1 + k) else {
            break;
        };
        let Some(m) = memchr::memchr(b')', &b[k + 2..]).map(|m| k + 2 + m) else {
            break;
        };
        let o = out.get_or_insert_with(|| String::with_capacity(line.len()));
        o.push_str(&line[pos..i]);
        o.push_str(&line[k + 2..m]);
        pos = m + 1;
    }
    match out {
        None => Cow::Borrowed(line),
        Some(mut o) => {
            o.push_str(&line[pos..]);
            Cow::Owned(o)
        }
    }
}

/// One line through MarkdownRemover's four substitutions, in order.
fn remove_markdown_line(line: &str) -> Cow<'_, str> {
    let mut cur = Cow::Borrowed(line);
    for (open, close) in [(&b"**"[..], &b"**"[..]), (b"*", b"*"), (b"_", b"_")] {
        if let Cow::Owned(s) = sub_lazy_pair(&cur, open, close) {
            cur = Cow::Owned(s);
        }
    }
    if let Cow::Owned(s) = sub_link(&cur) {
        cur = Cow::Owned(s);
    }
    cur
}

/// `MarkdownRemover().modify_document(text)`: per line, `**a**`, `*a*` and `_a_` become `a` and `[a](b)` becomes
/// `b` (four `re.sub` calls with lazy groups, in that order).
#[must_use]
pub fn remove_markdown(text: &str) -> Cow<'_, str> {
    if memchr::memchr3(b'*', b'_', b'[', text.as_bytes()).is_none() {
        return Cow::Borrowed(text);
    }
    let mut out: Option<String> = None;
    let mut start = 0;
    for line in text.split('\n') {
        let new = remove_markdown_line(line);
        if let Some(o) = out.as_mut() {
            o.push('\n');
            o.push_str(&new);
        } else if let Cow::Owned(s) = new {
            // The first changed line: copy the lines before it, with their newlines.
            let mut o = String::with_capacity(text.len());
            o.push_str(&text[..start]);
            o.push_str(&s);
            out = Some(o);
        }
        start += line.len() + 1;
    }
    out.map_or(Cow::Borrowed(text), Cow::Owned)
}

/// Replace each run of `min_run` or more consecutive `unit`s by `keep` of them (`re.sub(r"(unit){3,}", ...)`).
fn collapse_runs<'a>(text: &'a str, unit: &str, min_run: usize, keep: usize) -> Cow<'a, str> {
    let b = text.as_bytes();
    let finder = memmem::Finder::new(unit);
    let unit_s = unit;
    let unit = unit.as_bytes();
    let mut out: Option<String> = None;
    let mut pos = 0;
    let mut search = 0;
    while let Some(i) = finder.find(&b[search..]).map(|i| search + i) {
        let mut end = i + unit.len();
        let mut count = 1;
        while b[end..].starts_with(unit) {
            end += unit.len();
            count += 1;
        }
        if count >= min_run {
            let o = out.get_or_insert_with(|| String::with_capacity(text.len()));
            o.push_str(&text[pos..i]);
            for _ in 0..keep {
                o.push_str(unit_s);
            }
            pos = end;
        }
        search = end;
    }
    match out {
        None => Cow::Borrowed(text),
        Some(mut o) => {
            o.push_str(&text[pos..]);
            Cow::Owned(o)
        }
    }
}

/// `NewlineNormalizer().modify_document(text)`: runs of 3 or more `\n` become `\n\n`, then runs of 3 or more
/// `\r\n` become `\r\n\r\n`.
#[must_use]
pub fn normalize_newlines(text: &str) -> Cow<'_, str> {
    match collapse_runs(text, "\n", 3, 2) {
        Cow::Borrowed(t) => collapse_runs(t, "\r\n", 3, 2),
        Cow::Owned(t) => Cow::Owned(collapse_runs(&t, "\r\n", 3, 2).into_owned()),
    }
}

/// `s.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_whitespace)
}

/// `QuotationRemover().modify_document(text)`: drop the enclosing `"` of a document that starts and ends with
/// one (and has more than 2 characters after `strip()`), unless it has a newline and its first line ends with
/// `"`.
#[must_use]
pub fn remove_quotation(text: &str) -> Cow<'_, str> {
    let b = text.as_bytes();
    let stripped = py_strip(text);
    if stripped.chars().nth(2).is_none() || b.first() != Some(&b'"') || b.last() != Some(&b'"') {
        return Cow::Borrowed(text);
    }
    let first_line = text.split('\n').next().unwrap_or_default();
    if !stripped.contains('\n') || !first_line.ends_with('"') {
        // A borrowed substring: the bindings copy it.
        return Cow::Borrowed(&text[1..text.len() - 1]);
    }
    Cow::Borrowed(text)
}

/// Whether `c` matches `ch` (an ASCII lowercase letter) under `re.IGNORECASE`: its ASCII case pair, and for `s`
/// also U+017F LATIN SMALL LETTER LONG S (one of `re`'s case equivalences).
#[inline]
fn ignorecase_eq(c: char, ch: u8) -> bool {
    u32::from(c) < 0x80 && u8::try_from(u32::from(c)).is_ok_and(|b| b.to_ascii_lowercase() == ch)
        || (ch == b's' && c == '\u{17f}')
}

/// Match the case-insensitive `lit` (lowercase ASCII, non-letters literal) at the start of `s`; the byte length.
fn match_ignorecase(s: &str, lit: &[u8]) -> Option<usize> {
    let mut chars = s.char_indices();
    for &l in lit {
        let (_, c) = chars.next()?;
        if !ignorecase_eq(c, l) {
            return None;
        }
    }
    Some(chars.next().map_or(s.len(), |(i, _)| i))
}

/// The byte length of the `https?://` or `www\.` prefix (case-insensitive) at the start of `s`.
fn url_prefix(s: &str) -> Option<usize> {
    if let Some(n) = match_ignorecase(s, b"http") {
        // `s?` is greedy. Backtracking to the empty choice cannot help when an `s` matched: `:` must follow
        // "http" then. (A failing `\S+` cannot be rescued either, for the same reason.)
        if let Some(m) = match_ignorecase(&s[n..], b"s") {
            if s[n + m..].starts_with("://") {
                return Some(n + m + 3);
            }
        }
        return s[n..].starts_with("://").then_some(n + 3);
    }
    match_ignorecase(s, b"www.")
}

/// The byte length of a match of `https?://\S+|www\.\S+` (`re.IGNORECASE`) at the start of `s`, if any.
fn url_at(s: &str) -> Option<usize> {
    let start = url_prefix(s)?;
    let rest = &s[start..];
    let len = rest.find(is_py_whitespace).unwrap_or(rest.len());
    (len > 0).then_some(start + len)
}

/// `UrlRemover().modify_document(text)`: `re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)`.
#[must_use]
pub fn remove_urls(text: &str) -> Cow<'_, str> {
    let mut out: Option<String> = None;
    let mut pos = 0;
    let mut i = 0;
    let b = text.as_bytes();
    while i < b.len() {
        // A match starts with h/H, w/W (ASCII); `ſ` can only appear after "http".
        let c = b[i];
        if matches!(c, b'h' | b'H' | b'w' | b'W') {
            if let Some(len) = url_at(&text[i..]) {
                let o = out.get_or_insert_with(|| String::with_capacity(text.len()));
                o.push_str(&text[pos..i]);
                i += len;
                pos = i;
                continue;
            }
        }
        i += 1;
        while i < b.len() && (b[i] & 0xC0) == 0x80 {
            i += 1;
        }
    }
    match out {
        None => Cow::Borrowed(text),
        Some(mut o) => {
            o.push_str(&text[pos..]);
            Cow::Owned(o)
        }
    }
}

/// `LineRemover(patterns).modify_document(text)`: the lines of `text.split("\n")` equal to a pattern are dropped.
#[must_use]
pub fn remove_lines<'a>(text: &'a str, patterns: &[String]) -> Cow<'a, str> {
    if !text.split('\n').any(|l| patterns.iter().any(|p| p == l)) {
        return Cow::Borrowed(text);
    }
    let kept: Vec<&str> = text
        .split('\n')
        .filter(|l| !patterns.iter().any(|p| p == l))
        .collect();
    Cow::Owned(kept.join("\n"))
}

/// One side of a [`slice_text`] call.
#[derive(Debug, Clone)]
pub enum SliceBound {
    /// A Python int index (code points; negative counts from the end).
    Index(i64),
    /// A substring: `find` for the left side, `rfind` for the right side.
    Substring(String),
    /// Neither: the start or the end of the text.
    Default,
}

/// The byte offset of Python index `idx` (as in a slice) into `text` of `len` code points.
fn slice_offset(text: &str, len: usize, idx: i64) -> usize {
    let len_i = i64::try_from(len).unwrap_or(i64::MAX);
    let cp = if idx < 0 {
        (idx + len_i).max(0)
    } else {
        idx.min(len_i)
    };
    let cp = usize::try_from(cp).unwrap_or(0);
    text.char_indices().nth(cp).map_or(text.len(), |(i, _)| i)
}

/// `Slicer(left, right, include_left, include_right, strip).modify_document(text)`.
#[must_use]
pub fn slice_text<'a>(
    text: &'a str,
    left: &SliceBound,
    right: &SliceBound,
    include_left: bool,
    include_right: bool,
    strip: bool,
) -> Cow<'a, str> {
    let len = text.chars().count();
    let lb = match left {
        SliceBound::Index(i) => slice_offset(text, len, *i),
        SliceBound::Substring(s) => match text.find(s.as_str()) {
            None => return Cow::Borrowed(""),
            Some(f) => {
                if include_left {
                    f
                } else {
                    f + s.len()
                }
            }
        },
        SliceBound::Default => 0,
    };
    let rb = match right {
        SliceBound::Index(i) => slice_offset(text, len, *i),
        SliceBound::Substring(s) => match text.rfind(s.as_str()) {
            None => return Cow::Borrowed(""),
            Some(f) => {
                if include_right {
                    f + s.len()
                } else {
                    f
                }
            }
        },
        SliceBound::Default => text.len(),
    };
    let result = if lb < rb { &text[lb..rb] } else { "" };
    Cow::Borrowed(if strip { py_strip(result) } else { result })
}

/// `FastTextLabelModifier(label).modify_document(text)`: `f"{label} {text}"` after replacing `\n` and
/// `__label__` by spaces.
#[must_use]
pub fn fasttext_label(text: &str, label: &str) -> String {
    let body = text.replace('\n', " ").replace("__label__", " ");
    let mut out = String::with_capacity(label.len() + 1 + body.len());
    out.push_str(label);
    out.push(' ');
    out.push_str(&body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boilerplate() {
        let doc = "Terms of Use\n\nreal text\n\nmore";
        assert_eq!(remove_boilerplate_strings(doc, true), "real text\n\nmore");
        assert_eq!(remove_boilerplate_strings(doc, false), "");
        // Not at the top or bottom: unchanged.
        let mid = "a\n\nprivacy policy\n\nb";
        assert!(matches!(remove_boilerplate_strings(mid, true), Cow::Borrowed(s) if s == mid));
        // Every paragraph is boilerplate: unchanged.
        assert_eq!(
            remove_boilerplate_strings("terms of use", true),
            "terms of use"
        );
        assert_eq!(remove_boilerplate_strings("x\n\nLorem Ipsum", true), "");
        // KELVIN SIGN lowercases to k.
        assert_eq!(
            remove_boilerplate_strings("a\n\ncoo\u{212a}ie policy", true),
            "a"
        );
        // U+0130 lowercases to i + U+0307, which breaks the match.
        assert_eq!(
            remove_boilerplate_strings("a\n\npr\u{130}vacy policy", true),
            "a\n\npr\u{130}vacy policy"
        );
    }

    #[test]
    fn markdown() {
        assert_eq!(remove_markdown("***a**"), "*a");
        assert_eq!(remove_markdown("**a*b**"), "a*b");
        assert_eq!(remove_markdown("x *y* _z_ [t](u)(v)"), "x y z u(v)");
        assert_eq!(remove_markdown("**\n**"), "\n");
        assert_eq!(remove_markdown("[a]b](c)"), "c");
        assert_eq!(remove_markdown("[a](b"), "[a](b");
        assert_eq!(remove_markdown("a_b_c\n*"), "abc\n*");
    }

    #[test]
    fn newlines() {
        assert_eq!(normalize_newlines("a\n\n\n\nb\n\nc"), "a\n\nb\n\nc");
        assert_eq!(normalize_newlines("a\r\n\r\n\r\nb"), "a\r\n\r\nb");
        assert_eq!(normalize_newlines("a\n\nb"), "a\n\nb");
    }

    #[test]
    fn quotation() {
        assert_eq!(remove_quotation("\"abc\""), "abc");
        assert_eq!(remove_quotation("\"a\""), "a");
        assert_eq!(remove_quotation(" \"\" "), " \"\" ");
        assert_eq!(remove_quotation("\"a\"\nb\""), "\"a\"\nb\"");
        assert_eq!(remove_quotation("\"a\nb\""), "a\nb");
    }

    #[test]
    fn urls() {
        assert_eq!(remove_urls("see HTTPS://x.y/z now"), "see  now");
        assert_eq!(remove_urls("http:// x www.a"), "http:// x ");
        assert_eq!(remove_urls("htt\u{17f}://k"), "htt\u{17f}://k");
        assert_eq!(remove_urls("http\u{17f}://k."), "");
        assert_eq!(remove_urls("httpss://a"), "httpss://a");
    }

    #[test]
    fn slicer() {
        let s = |l: SliceBound, r: SliceBound| {
            slice_text(" abc def ", &l, &r, true, true, true).into_owned()
        };
        assert_eq!(s(SliceBound::Index(0), SliceBound::Default), "abc def");
        assert_eq!(
            s(
                SliceBound::Substring("c".into()),
                SliceBound::Substring("d".into())
            ),
            "c d"
        );
        assert_eq!(
            s(SliceBound::Substring("q".into()), SliceBound::Default),
            ""
        );
        assert_eq!(s(SliceBound::Index(-3), SliceBound::Index(-1)), "ef");
        assert_eq!(s(SliceBound::Index(5), SliceBound::Index(2)), "");
    }

    #[test]
    fn label() {
        assert_eq!(
            fasttext_label("a\nb__label__c", "__label__hq"),
            "__label__hq a b c"
        );
    }
}
