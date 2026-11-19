// Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
//! `Detector.append` and `Detector.cleaning_text`.
//!
//! `append(text)` runs `URL_RE.sub(' ', text)`, then `MAIL_RE.sub(' ', ...)`, then
//! `NGram.normalize_vi`, and keeps the first `max_text_length` code points of the result,
//! collapsing runs of spaces. The two regexes are matched here by hand, with `re`'s
//! leftmost-first semantics, on code points:
//!
//! - `URL_RE = https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}`
//! - `MAIL_RE = [-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}`
//!
//! langdetect runs the substitutions over the whole text before it cuts it. Only a prefix
//! can reach the first `max_text_length` code points, so [`append`] processes a prefix long
//! enough that every match decision inside it reads only characters of the prefix (each
//! pattern looks ahead a bounded distance), and grows it if the substitutions shrank it too
//! much. The result is the same as processing the whole text.

use std::borrow::Cow;

use super::chars::SPACE;
use super::tables::{VI_BASE, VI_MARKS, VI_TABLE};

/// How far `URL_RE` reads from a match start: `https://` and 2,076 characters.
const URL_LOOKAHEAD: usize = 8 + 2076 + 1;
/// How far `MAIL_RE` reads from a match start: 64 + `@` + 255 + 255, plus one.
const MAIL_LOOKAHEAD: usize = 64 + 1 + 255 + 255 + 1;
/// `normalize_vi` reads one character ahead.
const VI_LOOKAHEAD: usize = 1;

#[inline]
fn is_url_char(c: u32) -> bool {
    matches!(c, 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A)
        || matches!(
            char::from_u32(c),
            Some('-' | '_' | '.' | '?' | '&' | '~' | ';' | '+' | '=' | '/' | '#')
        )
}

/// `[-_.0-9A-Za-z]`: the mail local part and the domain's second half.
#[inline]
fn is_mail_char(c: u32) -> bool {
    matches!(c, 0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A | 0x2D | 0x2E | 0x5F)
}

/// `[-_0-9A-Za-z]`: the domain's first half (no dot).
#[inline]
fn is_domain_char(c: u32) -> bool {
    c != 0x2E && is_mail_char(c)
}

fn run_len(text: &[u32], from: usize, max: usize, pred: impl Fn(u32) -> bool) -> usize {
    text[from.min(text.len())..]
        .iter()
        .take(max)
        .take_while(|&&c| pred(c))
        .count()
}

/// The end of a `URL_RE` match starting at `i`, if there is one.
fn url_match(text: &[u32], i: usize) -> Option<usize> {
    const HTTP: [u32; 4] = [0x68, 0x74, 0x74, 0x70];
    const SEP: [u32; 3] = [0x3A, 0x2F, 0x2F];
    if text.get(i..i + 4) != Some(&HTTP[..]) {
        return None;
    }
    // `s?` is greedy: try `https://` first, then `http://`.
    let head = if text.get(i + 4) == Some(&0x73) && text.get(i + 5..i + 8) == Some(&SEP[..]) {
        8
    } else if text.get(i + 4..i + 7) == Some(&SEP[..]) {
        7
    } else {
        return None;
    };
    let body = run_len(text, i + head, 2076, is_url_char);
    (body > 0).then_some(i + head + body)
}

/// `URL_RE.sub(' ', text)` for match starts before `stop`; returns the output and the
/// input position reached (`>= stop`, or the end).
fn sub_urls(text: &[u32], stop: usize) -> (Cow<'_, [u32]>, usize) {
    let mut out: Option<Vec<u32>> = None;
    let mut i = 0;
    let mut copied = 0; // text[..copied] is in `out` (or unchanged)
    while i < stop {
        match memchr_u32(0x68, &text[i..stop]) {
            None => {
                i = stop;
                break;
            }
            Some(off) => i += off,
        }
        if let Some(end) = url_match(text, i) {
            let o = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
            o.extend_from_slice(&text[copied..i]);
            o.push(SPACE);
            i = end;
            copied = end;
        } else {
            i += 1;
        }
    }
    let reached = i.max(stop.min(text.len()));
    match out {
        None => (Cow::Borrowed(&text[..reached]), reached),
        Some(mut o) => {
            o.extend_from_slice(&text[copied.min(reached)..reached]);
            (Cow::Owned(o), reached)
        }
    }
}

/// `MAIL_RE.sub(' ', text)` for match starts before `stop`; returns the output and the input
/// position reached.
fn sub_mails(text: &[u32], stop: usize) -> (Cow<'_, [u32]>, usize) {
    let mut out: Option<Vec<u32>> = None;
    let mut copied = 0;
    let mut i = 0;
    while i < stop {
        // A match needs an '@' after the local part: find the next one.
        let Some(off) = memchr_u32(0x40, &text[i..]) else {
            i = stop;
            break;
        };
        let at = i + off;
        // The local part: the run of mail characters right before '@', at most 64 of them.
        let mut start = at;
        while start > i && at - start < 64 && is_mail_char(text[start - 1]) {
            start -= 1;
        }
        if start >= stop || start == at {
            // No local part, or it starts at or past `stop`: no match starts before `stop`
            // up to this '@'.
            i = if start == at { at + 1 } else { stop };
            continue;
        }
        // `start` is the leftmost position whose run reaches '@' within 64 characters.
        // The domain: D1{1,255} then D2{1,255} (backtracking one D1 character if needed).
        let d = at + 1;
        let k1 = run_len(text, d, 255, is_domain_char);
        let end = if k1 == 0 {
            None
        } else {
            let k2 = run_len(text, d + k1, 255, is_mail_char);
            if k2 > 0 {
                Some(d + k1 + k2)
            } else if k1 >= 2 {
                Some(d + k1)
            } else {
                None
            }
        };
        match end {
            Some(end) => {
                let o = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
                o.extend_from_slice(&text[copied..start]);
                o.push(SPACE);
                i = end;
                copied = end;
            }
            None => i = at + 1,
        }
    }
    let reached = i.max(stop.min(text.len()));
    match out {
        None => (Cow::Borrowed(&text[..reached]), reached),
        Some(mut o) => {
            o.extend_from_slice(&text[copied.min(reached)..reached]);
            (Cow::Owned(o), reached)
        }
    }
}

#[inline]
fn vi_base(c: u32) -> Option<usize> {
    VI_BASE.iter().position(|&b| b == c)
}

#[inline]
fn vi_mark(c: u32) -> Option<usize> {
    VI_MARKS.iter().position(|&m| m == c)
}

/// `NGram.normalize_vi(text)` for pairs starting before `stop`.
fn normalize_vi(text: &[u32], stop: usize) -> (Cow<'_, [u32]>, usize) {
    let mut out: Option<Vec<u32>> = None;
    let mut copied = 0;
    let mut i = 0;
    while i < stop {
        // Every mark is in U+0300..U+0323: find the next character in that range.
        let Some(off) = text
            .get(i + 1..)
            .and_then(|rest| rest.iter().position(|&c| (0x300..=0x323).contains(&c)))
        else {
            i = stop;
            break;
        };
        let m = i + 1 + off; // a candidate mark at m, base at m - 1
        if m > stop {
            i = stop;
            break;
        }
        if let (Some(b), Some(k)) = (vi_base(text[m - 1]), vi_mark(text[m])) {
            let o = out.get_or_insert_with(|| Vec::with_capacity(text.len()));
            o.extend_from_slice(&text[copied..m - 1]);
            o.push(VI_TABLE[k][b]);
            copied = m + 1;
            i = m + 1;
        } else {
            i = m;
        }
    }
    let reached = i.max(stop.min(text.len()));
    match out {
        None => (Cow::Borrowed(&text[..reached]), reached),
        Some(mut o) => {
            o.extend_from_slice(&text[copied.min(reached)..reached]);
            (Cow::Owned(o), reached)
        }
    }
}

#[inline]
fn memchr_u32(needle: u32, hay: &[u32]) -> Option<usize> {
    hay.iter().position(|&c| c == needle)
}

/// The three substitutions of `append` on a prefix of `text`: their output for the whole
/// text is `out` followed by more. `complete` says the prefix is the whole text.
fn substitute(text: &[u32], complete: bool) -> Vec<u32> {
    let stop = |len: usize, look: usize| {
        if complete {
            len
        } else {
            len.saturating_sub(look)
        }
    };
    let (urls, _) = sub_urls(text, stop(text.len(), URL_LOOKAHEAD));
    let (mails, _) = sub_mails(&urls, stop(urls.len(), MAIL_LOOKAHEAD));
    let (vi, _) = normalize_vi(&mails, stop(mails.len(), VI_LOOKAHEAD));
    vi.into_owned()
}

/// `NGram.normalize_vi(text)`.
pub fn normalize_vi_all(text: &[u32]) -> Vec<u32> {
    normalize_vi(text, text.len()).0.into_owned()
}

/// `URL_RE.sub`, `MAIL_RE.sub` and `normalize_vi` over all of `text`.
pub fn substitute_all(text: &[u32]) -> Vec<u32> {
    substitute(text, true)
}

/// The first `max_len` code points of `chars`, with runs of spaces collapsed as `append`
/// does (`pre` starts as "not a space").
pub fn collapse(chars: &[u32], max_len: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(chars.len().min(max_len));
    let mut pre = 0u32;
    for &ch in chars.iter().take(max_len) {
        if ch != SPACE || pre != SPACE {
            out.push(ch);
        }
        pre = ch;
    }
    out
}

/// How many code points of a text [`append_prefix`] should be given first.
pub fn first_prefix(max_len: usize) -> usize {
    // Enough unless the substitutions shrink the prefix by more than this margin.
    max_len
        .saturating_add(URL_LOOKAHEAD + MAIL_LOOKAHEAD + VI_LOOKAHEAD)
        .saturating_add(1024)
}

/// [`append`] from a prefix of the text (`complete`: the prefix is all of it). `None` when
/// the prefix is too short to decide; call again with a longer one (twice as long).
pub fn append_prefix(prefix: &[u32], complete: bool, max_len: usize) -> Option<Vec<u32>> {
    let s = substitute(prefix, complete);
    (complete || s.len() >= max_len).then(|| collapse(&s, max_len))
}

/// What `Detector.append(text)` adds to `Detector.text`, for `max_text_length = max_len`.
pub fn append(text: &[u32], max_len: usize) -> Vec<u32> {
    let mut take = first_prefix(max_len);
    loop {
        let complete = take >= text.len();
        if let Some(out) = append_prefix(&text[..take.min(text.len())], complete, max_len) {
            return out;
        }
        take = take.saturating_mul(2);
    }
}

/// `Detector.cleaning_text`: when fewer than half the characters are `A`..`z`, drop them.
///
/// The non-Latin count in langdetect compares `unicode_block(ch)` (an int) with the string
/// `'Latin Extended Additional'`, which is never equal: every character from U+0300 counts.
/// Returns `None` when the text stays as it is.
pub fn clean(text: &[u32]) -> Option<Vec<u32>> {
    let mut latin = 0usize;
    let mut non_latin = 0usize;
    for &c in text {
        if (0x41..=0x7A).contains(&c) {
            latin += 1;
        } else if c >= 0x300 {
            non_latin += 1;
        }
    }
    if latin.saturating_mul(2) < non_latin && latin > 0 {
        Some(
            text.iter()
                .copied()
                .filter(|c| !(0x41..=0x7A).contains(c))
                .collect(),
        )
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cps(s: &str) -> Vec<u32> {
        s.chars().map(u32::from).collect()
    }

    fn string(v: &[u32]) -> String {
        v.iter().map(|&c| char::from_u32(c).unwrap()).collect()
    }

    fn app(s: &str, max: usize) -> String {
        string(&append(&cps(s), max))
    }

    #[test]
    fn urls_and_mails_become_spaces() {
        assert_eq!(app("see https://example.com/x?a=1 now", 10_000), "see now");
        assert_eq!(app("see http://e.c now", 10_000), "see now");
        assert_eq!(app("httpx://a", 10_000), "httpx://a");
        assert_eq!(app("https:/a", 10_000), "https:/a");
        assert_eq!(app("http://", 10_000), "http://");
        assert_eq!(app("mail me@example.org ok", 10_000), "mail ok");
        assert_eq!(app("a@b c", 10_000), "a@b c");
        assert_eq!(app("a@bc d", 10_000), " d");
        assert_eq!(app("", 10_000), "");
        assert_eq!(app("@@x.y", 10_000), "@@x.y");
    }

    #[test]
    fn spaces_collapse_and_text_is_cut() {
        assert_eq!(app("a   b", 10_000), "a b");
        assert_eq!(app("  a", 10_000), " a");
        assert_eq!(app("abcdef", 3), "abc");
        assert_eq!(app("abc", 0), "");
    }

    #[test]
    fn vietnamese_pairs() {
        // a + COMBINING GRAVE ACCENT -> U+00E0
        assert_eq!(append(&[0x61, 0x300], 10), vec![0xE0]);
        assert_eq!(append(&[0x61, 0x300, 0x300], 10), vec![0xE0, 0x300]);
        assert_eq!(append(&[0x62, 0x300], 10), vec![0x62, 0x300]);
    }

    #[test]
    fn prefix_processing_equals_whole_text() {
        // Many URLs (each shrinks to a space) force the prefix to grow.
        use std::fmt::Write as _;
        let mut s = String::new();
        for i in 0..400 {
            write!(s, "http://example.com/{i} word{i} x@y{i}.com ").unwrap();
        }
        let text = cps(&s);
        for max in [0, 1, 10, 100, 1000, 5000] {
            let whole = collapse(&substitute_all(&text), max);
            assert_eq!(append(&text, max), whole, "max {max}");
        }
    }

    #[test]
    fn cleaning() {
        assert_eq!(clean(&cps("abc")), None);
        // two Latin letters, five Cyrillic: Latin goes.
        assert_eq!(
            clean(&cps("ab привет")).map(|v| string(&v)),
            Some(" привет".into())
        );
    }
}
