// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! The heuristic filters' `score_document` (`nemo_curator.stages.text.filters.heuristic`,
//! `string.py`). Each function returns what NeMo's method returns for the same text, or
//! the [`ScoreError`] it raises; floats come from the same integer counts and the same
//! division, so they are bit-identical.

use std::sync::LazyLock;

use memchr::memmem::{self, Finder};

use super::tables::ALPHA;
use super::text::{ScoreError, Splitter, project_lower, ratio, sentences, split};
use crate::filters::text::{py_len, py_strip};

/// `ellipsis_marks`.
pub const ELLIPSIS_MARKS: [&str; 8] = [
    "...",
    "[...]",
    "\u{2026}",
    "(...)",
    "[\u{2026}]",
    "-\u{BB}",
    "read more..",
    "read more",
];

/// `bullet_list`.
pub const BULLETS: [char; 12] = [
    '\u{2022}', '\u{2023}', '\u{2043}', '\u{204C}', '\u{204D}', '\u{2219}', '\u{25CB}', '\u{25CF}',
    '\u{25D8}', '\u{25E6}', '\u{29BE}', '\u{29BF}',
];

/// `end_marks`.
pub const END_MARKS: [char; 5] = ['.', '?', '!', '"', '\''];

/// `common_english_words`.
pub const COMMON_ENGLISH_WORDS: [&str; 8] =
    ["the", "be", "to", "of", "and", "that", "have", "with"];

/// `policy_substrings` (NeMo lowercases each one).
pub const POLICY_SUBSTRINGS: [&str; 14] = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "privacy overview",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "this website uses cookies to improve your experience while you navigate through the \
     website. out of these cookies, the cookies that are categorized as necessary are stored on \
     your browser as they are essential for the working of basic functionalities of the \
     website. we also use third-party cookies that help us analyze and understand how you use \
     this website. these cookies will be stored in your browser only with your consent. you \
     also have the option to opt-out of these cookies. but opting out of some of these cookies \
     may have an effect on your browsing experience.",
    "necessary cookies are absolutely essential for the website to function properly. this \
     category only includes cookies that ensures basic functionalities and security features of \
     the website. these cookies do not store any personal information.",
    "any cookies that may not be particularly necessary for the website to function and is used \
     specifically to collect user personal data via analytics, ads, other embedded contents are \
     termed as non-necessary cookies. it is mandatory to procure user consent prior to running \
     these cookies on your website.",
    "this site uses cookies, including for analytics, personalization, and advertising \
     purposes. for more information or to change your cookie settings, click here.",
    "if you continue to browse this site without changing your cookie settings, you agree to \
     this use. acceptread more",
];

/// Every policy substring contains one of these, so a paragraph without them has none.
const POLICY_PREFILTER: [&str; 3] = ["cookie", "privacy", "terms of use"];

/// The boilerplate needles, built once.
struct Boilerplate {
    lorem: Finder<'static>,
    prefilter: Vec<Finder<'static>>,
    /// The policy substrings that do not contain a shorter one (that one finds them too).
    policy: Vec<Finder<'static>>,
}

static BOILERPLATE: LazyLock<Boilerplate> = LazyLock::new(|| {
    let minimal = POLICY_SUBSTRINGS.iter().filter(|p| {
        !POLICY_SUBSTRINGS
            .iter()
            .any(|q| q.len() < p.len() && p.contains(q))
    });
    Boilerplate {
        lorem: Finder::new("lorem ipsum"),
        prefilter: POLICY_PREFILTER.iter().map(|&p| Finder::new(p)).collect(),
        policy: minimal.map(|&p| Finder::new(p)).collect(),
    }
});

/// `x in white_space_list` for an ASCII byte (`"\b"` is U+0008).
#[inline]
fn is_listed_space(b: u8) -> bool {
    matches!(b, b'\t' | b'\n' | b'\r' | 0x08 | b' ')
}

/// A score over the code point count: `count / nchar if nchar > 0 else 1.0`.
#[inline]
fn per_char(text: &str, count: usize) -> f64 {
    match py_len(text) {
        0 => 1.0,
        n => ratio(count, n),
    }
}

/// `NonAlphaNumericFilter`: `(nchar - len(regex_alphanum.findall(text))) / nchar`, 1.0 if empty.
pub fn non_alpha_numeric(text: &str) -> f64 {
    // `[a-zA-Z0-9\n?!,.]` is ASCII: count bytes.
    let k = text
        .bytes()
        .filter(|&b| matches!(b, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'\n' | b'?' | b'!' | b',' | b'.'))
        .count();
    match py_len(text) {
        0 => 1.0,
        n => ratio(n - k, n),
    }
}

/// `len(regex_hash.findall(word))`: the runs of `#`.
fn hash_runs(word: &str) -> usize {
    let b = word.as_bytes();
    (0..b.len())
        .filter(|&i| b[i] == b'#' && (i == 0 || b[i - 1] != b'#'))
        .count()
}

/// `SymbolsToWordsFilter`: the share of words that are an ellipsis mark or mostly `#` runs.
///
/// A word that strips to `""` (jieba's whitespace words) divides by zero, as in NeMo.
pub fn symbols_to_words(text: &str, splitter: Splitter) -> Result<f64, ScoreError> {
    let (mut words, mut symbols) = (0usize, 0usize);
    for w in splitter.words(py_strip(text)) {
        words += 1;
        let word = py_strip(w);
        let len = py_len(word);
        if len == 0 {
            return Err(ScoreError::ZeroDivision);
        }
        if ELLIPSIS_MARKS.contains(&word) || ratio(hash_runs(word), len) > 0.5 {
            symbols += 1;
        }
    }
    if words == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(symbols, words))
}

/// `NumbersFilter`: `len(regex_digit.findall(text)) / nchar` (`[[:digit:]]` is ASCII 0-9).
pub fn numbers(text: &str) -> f64 {
    per_char(text, text.bytes().filter(u8::is_ascii_digit).count())
}

/// `regex_url`'s unit at byte `i`: one URL character, or `%` and two hex digits. Its length.
#[inline]
fn url_unit(b: &[u8], i: usize) -> usize {
    match b[i] {
        // `[A-Za-z0-9$\-_@.&+/:=?#~]` and `[!*\(\),]`.
        b'A'..=b'Z'
        | b'a'..=b'z'
        | b'0'..=b'9'
        | b'$'
        | b'-'
        | b'_'
        | b'@'
        | b'.'
        | b'&'
        | b'+'
        | b'/'
        | b':'
        | b'='
        | b'?'
        | b'#'
        | b'~'
        | b'!'
        | b'*'
        | b'('
        | b')'
        | b',' => 1,
        b'%' if i + 2 < b.len() && b[i + 1].is_ascii_hexdigit() && b[i + 2].is_ascii_hexdigit() => {
            3
        }
        _ => 0,
    }
}

/// The byte ranges of `regex_url.findall(text)`, in order, to `f` until it returns false.
///
/// `https?://(?:[A-Za-z0-9$\-_@.&+/:=?#~]|[!*\(\),]|(?:%[0-9A-Fa-f]{2}))+`: at most one unit
/// fits at any position, so the greedy loop takes the longest run; matches don't overlap.
fn for_each_url(text: &str, mut f: impl FnMut(&[u8]) -> bool) {
    let b = text.as_bytes();
    let mut from = 0;
    while let Some(p) = memmem::find(&b[from..], b"http") {
        let start = from + p;
        let mut i = start + 4;
        if b.get(i) == Some(&b's') {
            i += 1;
        }
        if b[i..].starts_with(b"://") {
            let units = i + 3;
            let mut end = units;
            while end < b.len() {
                match url_unit(b, end) {
                    0 => break,
                    n => end += n,
                }
            }
            if end > units {
                if !f(&b[start..end]) {
                    return;
                }
                from = end;
                continue;
            }
        }
        from = start + 1;
    }
}

/// `UrlsFilter` with NeMo's default `regex_url`: URL characters over all characters.
pub fn urls(text: &str) -> f64 {
    let mut url_chars = 0;
    // A URL is ASCII: bytes are characters.
    for_each_url(text, |u| {
        url_chars += u.len();
        true
    });
    per_char(text, url_chars)
}

/// `WhiteSpaceFilter`: characters in `white_space_list` over all characters.
pub fn white_space(text: &str) -> f64 {
    per_char(text, text.bytes().filter(|&b| is_listed_space(b)).count())
}

/// `ParenthesesFilter`: `{ } ⟨ ⟩ [ ] ( )` over all characters.
pub fn parentheses(text: &str) -> f64 {
    let b = text.as_bytes();
    let k = b
        .iter()
        .enumerate()
        .filter(|&(i, &c)| match c {
            b'{' | b'}' | b'[' | b']' | b'(' | b')' => true,
            // U+27E8 / U+27E9 are E2 9F A8 / E2 9F A9 (a lead byte has its continuations).
            0xE2 => b[i + 1] == 0x9F && matches!(b[i + 2], 0xA8 | 0xA9),
            _ => false,
        })
        .count();
    per_char(text, k)
}

/// `BoilerPlateStringFilter`: 1.0 if a paragraph contains `"lorem ipsum"`, else the share
/// of paragraphs (`text.split("\n\n")`) containing a policy substring, after
/// `paragraph.strip().lower()`.
pub fn boilerplate_string(text: &str) -> f64 {
    let bp = &*BOILERPLATE;
    let mut low = Vec::new();
    // `"\n"` is the only character projected to `\n`, so the projection splits into the
    // projected paragraphs. No needle starts or ends with whitespace or holds `"\n"`, so
    // `strip()` changes nothing and a match never crosses a paragraph break.
    project_lower(text, &mut low);
    if bp.lorem.find(&low).is_some() {
        return 1.0;
    }
    let (mut total, mut boilerplate) = (0usize, 0usize);
    let mut rest: &[u8] = &low;
    loop {
        let (p, next) = match memmem::find(rest, b"\n\n") {
            Some(i) => (&rest[..i], Some(&rest[i + 2..])),
            None => (rest, None),
        };
        total += 1;
        if bp.prefilter.iter().any(|f| f.find(p).is_some())
            && bp.policy.iter().any(|f| f.find(p).is_some())
        {
            boilerplate += 1;
        }
        match next {
            Some(n) => rest = n,
            None => break,
        }
    }
    ratio(boilerplate, total)
}

/// `WordCountFilter`: the number of words.
pub fn word_count(text: &str, splitter: Splitter) -> usize {
    splitter.words(py_strip(text)).count()
}

/// `PunctuationFilter`: the share of sentences whose stripped text does not end with one of
/// `end_marks`.
pub fn punctuation(text: &str) -> Result<f64, ScoreError> {
    let (mut total, mut without) = (0usize, 0usize);
    for s in sentences(text) {
        total += 1;
        without += usize::from(!py_strip(s).ends_with(END_MARKS));
    }
    if total == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(without, total))
}

/// `regex_alpha.search(c)`: `c` is in `[[:alpha:]]` (the `regex` module's Alphabetic).
#[inline]
pub fn is_regex_alpha(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    let cp = u32::from(c);
    ALPHA
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

/// `WordsWithoutAlphabetsFilter`: the share of words with at least one `[[:alpha:]]`
/// character.
pub fn words_without_alphabets(text: &str, splitter: Splitter) -> Result<f64, ScoreError> {
    let (mut total, mut alpha) = (0usize, 0usize);
    for w in splitter.words(py_strip(text)) {
        total += 1;
        alpha += usize::from(w.chars().any(is_regex_alpha));
    }
    if total == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(alpha, total))
}

/// `CommonEnglishWordsFilter`: the number of words of `text.split()` (always English) in
/// `common_english_words`; with `stop_at_false`, the count as soon as it reaches
/// `min_num_common_words` (checked after every word).
pub fn common_english_words(text: &str, min_num_common_words: i64, stop_at_false: bool) -> u64 {
    let mut count: u64 = 0;
    for w in split(py_strip(text)) {
        count += u64::from(COMMON_ENGLISH_WORDS.contains(&w));
        if stop_at_false && i128::from(count) >= i128::from(min_num_common_words) {
            return count;
        }
    }
    count
}

/// `MeanWordLengthFilter`: the mean length (code points) of the non-empty words.
pub fn mean_word_length(text: &str, splitter: Splitter) -> Result<f64, ScoreError> {
    let (mut n, mut sum) = (0usize, 0usize);
    for w in splitter.words(py_strip(text)) {
        let len = py_len(w);
        if len > 0 {
            n += 1;
            sum += len;
        }
    }
    if n == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(sum, n))
}

/// `LongWordFilter`: the length of the longest word.
pub fn long_word(text: &str, splitter: Splitter) -> Result<usize, ScoreError> {
    splitter
        .words(py_strip(text))
        .map(py_len)
        .max()
        .ok_or(ScoreError::EmptyMax)
}

/// The longest ellipsis mark, in characters (`"read more.."`).
const LONGEST_MARK: usize = 11;

/// `EllipsisFilter`: the share of sentences whose `strip().lower()` ends with an ellipsis
/// mark.
pub fn ellipsis(text: &str) -> Result<f64, ScoreError> {
    let (mut total, mut ending) = (0usize, 0usize);
    let mut tail = Vec::new();
    for s in sentences(text) {
        total += 1;
        let s = py_strip(s);
        // Only the last LONGEST_MARK characters can be part of a mark's match.
        let from = s
            .char_indices()
            .rev()
            .nth(LONGEST_MARK - 1)
            .map_or(0, |(i, _)| i);
        project_lower(&s[from..], &mut tail);
        ending += usize::from(ELLIPSIS_MARKS.iter().any(|m| tail.ends_with(m.as_bytes())));
    }
    if total == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(ending, total))
}

/// `BulletsFilter`: the share of sentences whose stripped text starts with a bullet.
pub fn bullets(text: &str) -> Result<f64, ScoreError> {
    let (mut total, mut bulleted) = (0usize, 0usize);
    for s in sentences(text) {
        total += 1;
        bulleted += usize::from(py_strip(s).starts_with(BULLETS));
    }
    if total == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(bulleted, total))
}

/// `PornographicUrlsFilter` with NeMo's default `regex_url`: 1 if a URL contains `"porn"`.
pub fn pornographic_urls(text: &str) -> u8 {
    let mut found = false;
    for_each_url(text, |u| {
        found = memmem::find(u, b"porn").is_some();
        !found
    });
    u8::from(found)
}

/// Where `SubstringFilter` looks for its substring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    /// `text.startswith(substring)`.
    Prefix,
    /// `text.endswith(substring)`.
    Suffix,
    /// `substring in text`.
    Any,
}

impl Position {
    /// `"prefix"`, `"suffix"` or `"any"`; `None` for anything else (NeMo raises ValueError).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "prefix" => Some(Self::Prefix),
            "suffix" => Some(Self::Suffix),
            "any" => Some(Self::Any),
            _ => None,
        }
    }
}

/// `SubstringFilter`: 1 if `substring` is at `position` of `text`, else 0.
pub fn substring(text: &str, substring: &str, position: Position) -> u8 {
    u8::from(match position {
        Position::Prefix => text.starts_with(substring),
        Position::Suffix => text.ends_with(substring),
        Position::Any => memmem::find(text.as_bytes(), substring.as_bytes()).is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WS: Splitter = Splitter::Whitespace;

    #[test]
    fn ratios_over_characters() {
        assert_eq!(non_alpha_numeric(""), 1.0);
        assert_eq!(non_alpha_numeric("ab!#"), 0.25);
        assert_eq!(non_alpha_numeric("é\n"), 0.5);
        assert_eq!(numbers("a1é2"), 0.5);
        assert_eq!(numbers("٣"), 0.0); // [[:digit:]] is ASCII only
        assert_eq!(numbers(""), 1.0);
        assert_eq!(white_space("a\x08 \u{a0}"), 0.5);
        assert_eq!(white_space(""), 1.0);
        assert_eq!(parentheses("(⟨x⟩]"), 0.8);
        assert_eq!(parentheses("⟪"), 0.0);
        assert_eq!(parentheses(""), 1.0);
        assert_eq!(non_alpha_numeric("abc"), 0.0);
        assert_eq!(numbers("1234567"), 1.0);
        assert_eq!(numbers("1x"), 1.0 / 2.0);
        assert_eq!(white_space("a b c"), 2.0 / 5.0);
    }

    fn url_list(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for_each_url(text, |u| {
            out.push(String::from_utf8(u.to_vec()).unwrap());
            true
        });
        out
    }

    #[test]
    fn url_regex() {
        assert_eq!(
            url_list("see http://a.b/c?d=1 and https://x"),
            ["http://a.b/c?d=1", "https://x"]
        );
        assert_eq!(url_list("http:// https://%2"), Vec::<String>::new());
        assert_eq!(url_list("http://a%2Fb%zz"), ["http://a%2Fb"]);
        assert_eq!(url_list("httphttp://x"), ["http://x"]);
        assert_eq!(url_list("https://ahttp://b"), ["https://ahttp://b"]);
        assert_eq!(
            url_list("HTTP://x httpss://y http://é"),
            Vec::<String>::new()
        );
        assert_eq!(url_list("http://a<b>(c)!,*"), ["http://a"]);
        assert_eq!(url_list("http://(c)!,*"), ["http://(c)!,*"]);
        assert_eq!(urls("http://ab"), 1.0);
        assert_eq!(urls("xhttp://ab"), 0.9);
        assert_eq!(urls(""), 1.0);
        assert_eq!(pornographic_urls("http://x.porn http://y"), 1);
        assert_eq!(pornographic_urls("porn http://x.PORN"), 0);
        assert_eq!(pornographic_urls("http://po rn"), 0);
    }

    #[test]
    fn symbols() {
        assert_eq!(symbols_to_words("# ## a#b ... …", WS), Ok(0.6));
        assert_eq!(symbols_to_words("#a# ab", WS), Ok(0.5));
        assert_eq!(symbols_to_words("#a", WS), Ok(0.0));
        assert_eq!(symbols_to_words("  ", WS), Err(ScoreError::ZeroDivision));
        // jieba yields whitespace words, which strip to "".
        assert_eq!(
            symbols_to_words("中文 中文", Splitter::Jieba),
            Err(ScoreError::ZeroDivision)
        );
    }

    #[test]
    fn boilerplate() {
        assert_eq!(boilerplate_string(""), 0.0);
        assert_eq!(boilerplate_string("Terms of Use\n\nok"), 0.5);
        assert_eq!(boilerplate_string("a\n\nb\n\nLOREM IPSUM"), 1.0);
        assert_eq!(boilerplate_string("lorem\nipsum"), 0.0);
        assert_eq!(boilerplate_string("privacy\n\npolicy"), 0.0);
        assert_eq!(boilerplate_string("PRİVACY POLICY"), 0.0);
        assert_eq!(boilerplate_string("cooKie policy \u{212A}"), 1.0);
        assert_eq!(
            boilerplate_string("coo\u{212A}ie policy\n\n\n\nx"),
            1.0 / 3.0
        );
        assert_eq!(boilerplate_string("We use cookies."), 1.0);
    }

    #[test]
    fn policy_prefilter_is_implied() {
        for p in POLICY_SUBSTRINGS {
            assert!(POLICY_PREFILTER.iter().any(|q| p.contains(q)), "{p}");
            assert_eq!(p, p.to_lowercase());
            assert!(p.is_ascii() && p.trim() == p && !p.contains('\n'));
        }
        assert_eq!(BOILERPLATE.policy.len(), 12);
    }

    #[test]
    fn sentence_ratios() {
        assert_eq!(punctuation("a.\nb\n \nc!  \n\"d\"\n'"), Ok(0.2));
        assert_eq!(punctuation(" \n\t"), Err(ScoreError::ZeroDivision));
        assert_eq!(
            ellipsis("a...\nb\nREAD MORE \nc[…]\nd-»\nx…y"),
            Ok(4.0 / 6.0)
        );
        assert_eq!(ellipsis("İ\nREAD MORİ\nread more.."), Ok(1.0 / 3.0));
        assert_eq!(ellipsis(""), Err(ScoreError::ZeroDivision));
        assert_eq!(bullets(" • a\n◦\nb •\n\n"), Ok(2.0 / 3.0));
        assert_eq!(bullets("\n"), Err(ScoreError::ZeroDivision));
    }

    #[test]
    fn word_scores() {
        assert_eq!(word_count("  a b\u{3000}c ", WS), 3);
        assert_eq!(word_count("", WS), 0);
        assert_eq!(words_without_alphabets("a 1 ß ٣ ǅ", WS), Ok(0.6));
        assert_eq!(
            words_without_alphabets("", WS),
            Err(ScoreError::ZeroDivision)
        );
        assert_eq!(mean_word_length("ab abcd é", WS), Ok(7.0 / 3.0));
        assert_eq!(mean_word_length(" ", WS), Err(ScoreError::ZeroDivision));
        assert_eq!(long_word("ab abcd é", WS), Ok(4));
        assert_eq!(long_word("", WS), Err(ScoreError::EmptyMax));
        assert_eq!(common_english_words("the to of", 2, true), 2);
        assert_eq!(common_english_words("the to of", 2, false), 3);
        assert_eq!(common_english_words("x y", 0, true), 0);
        assert_eq!(common_english_words("the y", -1, true), 1);
        assert_eq!(common_english_words("The tO", 2, true), 0);
        assert_eq!(common_english_words("", 0, true), 0);
    }

    #[test]
    fn alpha_table() {
        for c in (0..0x80u8).map(char::from) {
            assert_eq!(is_regex_alpha(c), c.is_ascii_alphabetic());
        }
        assert!(is_regex_alpha('é') && is_regex_alpha('中') && is_regex_alpha('ǅ'));
        assert!(!is_regex_alpha('٣') && !is_regex_alpha('\u{300}') && !is_regex_alpha('😀'));
        assert!(ALPHA.windows(2).all(|w| w[0].1 + 1 < w[1].0));
    }

    #[test]
    fn substrings() {
        assert_eq!(substring("abc", "a", Position::Prefix), 1);
        assert_eq!(substring("abc", "b", Position::Prefix), 0);
        assert_eq!(substring("abc", "c", Position::Suffix), 1);
        assert_eq!(substring("abc", "b", Position::Any), 1);
        assert_eq!(substring("", "", Position::Any), 1);
        assert_eq!(Position::parse("middle"), None);
    }
}
