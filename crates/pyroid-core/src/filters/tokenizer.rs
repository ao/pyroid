// Derived from spaCy 3.8.18 (MIT), © ExplosionAI GmbH, and datatrove 0.10.1
// (Apache-2.0), © HuggingFace; modified.
//! datatrove's English word tokenizer: spaCy 3.8.18 `spacy.blank("en")` plus
//! the `sentencizer` pipe, as `datatrove.utils.word_tokenizers.SpaCyTokenizer`
//! runs it.
//!
//! This is a port of `spacy/tokenizer.pyx` (`_tokenize_affixes`,
//! `_split_affixes`, `_attach_tokens`, `_apply_special_cases`) and of
//! `Sentencizer.predict`, over spaCy's own English rules (`tables.rs`). The
//! tokenizer's cache is not modelled: datatrove tokenizes inside
//! `memory_zone()`, where spaCy never writes the cache, so every span is
//! tokenized from the rules.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use rustc_hash::FxBuildHasher;

use super::regex::{Matcher, Nfa};
use super::tables;
use super::unicode;
use crate::ftfy::UnicodeVersion;

/// One token: a byte range of the tokenized text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Token {
    /// Start byte offset.
    pub start: u32,
    /// End byte offset (exclusive).
    pub end: u32,
    /// A whitespace token (spaCy keeps runs of whitespace other than one
    /// `' '` as tokens; datatrove's `strip_strings` drops them).
    pub space: bool,
}

impl Token {
    /// The token's text.
    #[inline]
    pub fn text<'a>(&self, text: &'a str) -> &'a str {
        &text[self.start as usize..self.end as usize]
    }
}

/// A special case whose rule string has an affix or a space: spaCy's
/// "faster heuristics" `PhraseMatcher` pass re-applies it after affix
/// splitting, keyed by its first token.
#[derive(Debug)]
struct Pattern {
    tokens: Vec<Box<str>>,
}

/// spaCy's English tokenizer + sentencizer, for one Unicode version.
///
/// Build-once state (rules, matcher patterns, the URL automaton); `Sync`, so
/// one instance serves every thread. Get one with [`SpacyEnglish::get`].
#[derive(Debug)]
pub struct SpacyEnglish {
    version: UnicodeVersion,
    specials: HashMap<&'static str, &'static [&'static str], FxBuildHasher>,
    patterns: HashMap<Box<str>, Vec<Pattern>, FxBuildHasher>,
    url: Nfa,
}

static INSTANCES: [LazyLock<SpacyEnglish>; 5] = [
    LazyLock::new(|| SpacyEnglish::new(UnicodeVersion::V14_0)),
    LazyLock::new(|| SpacyEnglish::new(UnicodeVersion::V15_0)),
    LazyLock::new(|| SpacyEnglish::new(UnicodeVersion::V15_1)),
    LazyLock::new(|| SpacyEnglish::new(UnicodeVersion::V16_0)),
    LazyLock::new(|| SpacyEnglish::new(UnicodeVersion::V17_0)),
];

/// spaCy refuses texts of 2**30 characters or more (`Errors.E025`).
pub const MAX_CHARS: usize = 1 << 30;

impl SpacyEnglish {
    /// The shared instance for `version` (built on first use).
    pub fn get(version: UnicodeVersion) -> &'static SpacyEnglish {
        let i = match version {
            UnicodeVersion::V14_0 => 0,
            UnicodeVersion::V15_0 => 1,
            UnicodeVersion::V15_1 => 2,
            UnicodeVersion::V16_0 => 3,
            UnicodeVersion::V17_0 => 4,
        };
        &INSTANCES[i]
    }

    fn new(version: UnicodeVersion) -> Self {
        let mut me = Self {
            version,
            specials: tables::RULES.iter().copied().collect(),
            patterns: HashMap::default(),
            url: Nfa::new(tables::URL),
        };
        // Tokenizer.add_special_case: rules with an affix, an infix or a space
        // go to the PhraseMatcher, as their tokenization without specials.
        let mut patterns: HashMap<Box<str>, Vec<Pattern>, FxBuildHasher> = HashMap::default();
        let mut toks = Vec::new();
        let mut infixes = Vec::new();
        for &(key, _) in tables::RULES {
            infixes.clear();
            me.find_infix(key, &mut infixes);
            if me.find_prefix(key) == 0
                && infixes.is_empty()
                && me.find_suffix_chars(key) == 0
                && !key.contains(' ')
            {
                continue;
            }
            toks.clear();
            me.tokenize_affixes(key, false, &mut toks);
            let tokens: Vec<Box<str>> = toks.iter().map(|t| t.text(key).into()).collect();
            if let Some(first) = tokens.first() {
                patterns
                    .entry(first.clone())
                    .or_default()
                    .push(Pattern { tokens });
            }
        }
        me.patterns = patterns;
        me
    }

    /// The Unicode version this instance reproduces.
    pub fn version(&self) -> UnicodeVersion {
        self.version
    }

    /// spaCy's tokens for `text` (`[t for t in nlp(text)]`), whitespace
    /// tokens included, appended to `out`.
    ///
    /// `text` must be shorter than [`MAX_CHARS`] characters (spaCy raises
    /// `ValueError` otherwise; the filters check this first).
    pub fn tokens(&self, text: &str, out: &mut Vec<Token>) {
        assert!(
            u32::try_from(text.len()).is_ok(),
            "text too long to tokenize"
        );
        let base = out.len();
        self.tokenize_affixes(text, true, out);
        self.apply_special_cases(text, out, base);
    }

    /// datatrove's `split_into_words(text, "en")`: the non-whitespace tokens.
    pub fn words<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut toks = Vec::new();
        self.tokens(text, &mut toks);
        toks.iter()
            .filter(|t| !t.space)
            .map(|t| t.text(text))
            .collect()
    }

    /// datatrove's `len(split_into_sentences(text, "en"))`: the sentencizer's
    /// sentence count, or 1 when there are no tokens.
    pub fn count_sentences(&self, text: &str) -> usize {
        let mut toks = Vec::new();
        self.tokens(text, &mut toks);
        self.count_sentences_in(text, &toks).max(1)
    }

    /// `Sentencizer.predict` over `toks`: the number of sentences (0 if
    /// there are no tokens).
    pub fn count_sentences_in(&self, text: &str, toks: &[Token]) -> usize {
        let mut ends = Vec::new();
        self.sentence_ends_in(text, toks, &mut ends);
        ends.len()
    }

    /// `Sentencizer.predict` over `toks`: each sentence's `end_char` (the end
    /// of its last token, whitespace tokens included), appended to `out`.
    pub fn sentence_ends_in(&self, text: &str, toks: &[Token], out: &mut Vec<usize>) {
        let Some(last) = toks.last() else {
            return;
        };
        let mut seen_period = false;
        for (i, t) in toks.iter().enumerate() {
            let s = t.text(text);
            let in_punct_chars = is_sent_punct(s);
            if seen_period && !in_punct_chars && !self.is_punct(s) {
                out.push(toks[i - 1].end as usize);
                seen_period = false;
            } else if in_punct_chars {
                seen_period = true;
            }
        }
        out.push(last.end as usize);
    }

    /// spaCy's `is_punct` lexeme attribute: every character is in a `P*`
    /// category.
    fn is_punct(&self, s: &str) -> bool {
        s.chars().all(|c| unicode::is_punct(c, self.version))
    }

    // ---------------------------------------------------------- the regexes

    /// `Tokenizer.find_prefix`: byte length of the prefix to split off.
    fn find_prefix(&self, s: &str) -> usize {
        let Some(c) = s.chars().next() else {
            return 0;
        };
        if !tables::PREFIX_FIRST.contains(c, self.version) {
            return 0;
        }
        Matcher { s, v: self.version }
            .match_at(tables::PREFIX, 0)
            .unwrap_or(0)
    }

    /// `suffix_search(s)`: the leftmost match of `(?:...$|...$)`.
    fn suffix_search(&self, s: &str) -> Option<(usize, usize)> {
        let v = self.version;
        let last = s.chars().next_back()?;
        // `$` also matches before a final "\n"; take the unfiltered path then.
        let slow = last == '\n';
        if !slow && !tables::SUFFIX_LAST.contains(last, v) {
            return None;
        }
        let tail_start = if slow {
            0
        } else {
            let mut p = s.len();
            for _ in 0..tables::SUFFIX_MAX_BOUNDED {
                match s[..p].chars().next_back() {
                    Some(c) => p -= c.len_utf8(),
                    None => break,
                }
            }
            p
        };
        let m = Matcher { s, v };
        for (pos, c) in s.char_indices() {
            let candidate = slow
                || if pos >= tail_start {
                    tables::SUFFIX_FIRST.contains(c, v)
                } else {
                    tables::SUFFIX_UNBOUNDED_FIRST.contains(c, v)
                };
            if candidate {
                if let Some(end) = m.match_at(tables::SUFFIX, pos) {
                    return Some((pos, end));
                }
            }
        }
        None
    }

    /// `Tokenizer.find_suffix`: the suffix length in characters
    /// (`match.end() - match.start()`).
    fn find_suffix_chars(&self, s: &str) -> usize {
        self.suffix_search(s)
            .map_or(0, |(a, b)| s[a..b].chars().count())
    }

    /// `Tokenizer.find_infix`: `(start, end)` of every infix match.
    fn find_infix(&self, s: &str, out: &mut Vec<(usize, usize)>) {
        let v = self.version;
        let m = Matcher { s, v };
        let mut pos = 0;
        while let Some(c) = s[pos..].chars().next() {
            if tables::INFIX_FIRST.contains(c, v) {
                if let Some(end) = m.match_at(tables::INFIX, pos) {
                    if end > pos {
                        out.push((pos, end));
                        pos = end;
                        continue;
                    }
                }
            }
            pos += c.len_utf8();
        }
    }

    /// `url_match(s)`.
    fn url_match(&self, s: &str) -> bool {
        // Both alternatives of URL_PATTERN (IP address, host name) contain a
        // literal '.', so without one there is no match.
        s.as_bytes().contains(&b'.') && self.url.is_match(s, self.version)
    }

    // ----------------------------------------------------- the tokenizer

    /// `Tokenizer._tokenize_affixes`.
    fn tokenize_affixes(&self, text: &str, with_specials: bool, out: &mut Vec<Token>) {
        let Some(c0) = text.chars().next() else {
            return;
        };
        let mut in_ws = unicode::is_space(c0);
        let mut start = 0;
        for (i, c) in text.char_indices() {
            if unicode::is_space(c) != in_ws {
                if start < i {
                    self.tokenize_span(text, start, i, in_ws, with_specials, out);
                }
                start = if c == ' ' { i + 1 } else { i };
                in_ws = !in_ws;
            }
        }
        if start < text.len() {
            self.tokenize_span(text, start, text.len(), in_ws, with_specials, out);
        }
    }

    #[allow(clippy::cast_possible_truncation)] // `tokens` checks the length
    fn push(out: &mut Vec<Token>, start: usize, end: usize, space: bool) {
        out.push(Token {
            start: start as u32,
            end: end as u32,
            space,
        });
    }

    fn push_pieces(out: &mut Vec<Token>, mut start: usize, pieces: &[&str]) {
        for piece in pieces {
            let space = piece.chars().all(unicode::is_space);
            Self::push(out, start, start + piece.len(), space);
            start += piece.len();
        }
    }

    /// `_try_specials_and_cache` + `Tokenizer._tokenize` for one span.
    fn tokenize_span(
        &self,
        text: &str,
        lo: usize,
        hi: usize,
        space: bool,
        with_specials: bool,
        out: &mut Vec<Token>,
    ) {
        let span = &text[lo..hi];
        if with_specials {
            if let Some(pieces) = self.specials.get(span) {
                Self::push_pieces(out, lo, pieces);
                return;
            }
        }
        // _split_affixes
        let mut prefixes: Vec<(usize, usize)> = Vec::new();
        let mut suffixes: Vec<(usize, usize)> = Vec::new();
        let (mut a, mut b) = (lo, hi);
        let mut last_size = 0;
        while a < b && b - a != last_size {
            let string = &text[a..b];
            if with_specials && self.specials.contains_key(string) {
                break;
            }
            last_size = b - a;
            let pre = self.find_prefix(string);
            if pre != 0 {
                let minus_pre = &string[pre..];
                if !minus_pre.is_empty() && with_specials && self.specials.contains_key(minus_pre) {
                    prefixes.push((a, a + pre));
                    a += pre;
                    break;
                }
            }
            let suf_chars = self.find_suffix_chars(&string[pre..]);
            // `string[-suf_len:]`: the last `suf_chars` characters of `string`.
            let suf = string
                .char_indices()
                .rev()
                .nth(suf_chars.wrapping_sub(1))
                .map_or(0, |(i, _)| string.len() - i);
            if suf_chars != 0 {
                let minus_suf = &string[..string.len() - suf];
                if !minus_suf.is_empty() && with_specials && self.specials.contains_key(minus_suf) {
                    suffixes.push((b - suf, b));
                    b -= suf;
                    break;
                }
            }
            // `pre_len + suf_len <= len(string)` always holds: the suffix was
            // found in `string[pre_len:]`.
            if pre != 0 {
                prefixes.push((a, a + pre));
                a += pre;
            }
            if suf_chars != 0 {
                suffixes.push((b - suf, b));
                b -= suf;
            }
        }
        // _attach_tokens
        for &(s, e) in &prefixes {
            Self::push(out, s, e, space);
        }
        if a < b {
            let core = &text[a..b];
            if let Some(pieces) = with_specials.then(|| self.specials.get(core)).flatten() {
                Self::push_pieces(out, a, pieces);
            } else if self.url_match(core) {
                Self::push(out, a, b, space);
            } else {
                let mut matches = Vec::new();
                self.find_infix(core, &mut matches);
                let mut start = 0;
                for (ms, me) in matches {
                    if ms == 0 {
                        continue;
                    }
                    if ms != start {
                        Self::push(out, a + start, a + ms, space);
                    }
                    if ms != me {
                        Self::push(out, a + ms, a + me, space);
                    }
                    start = me;
                }
                if start < core.len() {
                    Self::push(out, a + start, b, space);
                }
            }
        }
        for &(s, e) in suffixes.iter().rev() {
            Self::push(out, s, e, space);
        }
    }

    /// `Tokenizer._apply_special_cases` on `out[base..]`.
    fn apply_special_cases(&self, text: &str, out: &mut Vec<Token>, base: usize) {
        let toks = &out[base..];
        let mut matches: Vec<(usize, usize)> = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            let Some(pats) = self.patterns.get(t.text(text)) else {
                continue;
            };
            for p in pats {
                let end = i + p.tokens.len();
                if end <= toks.len()
                    && p.tokens[1..]
                        .iter()
                        .zip(&toks[i + 1..end])
                        .all(|(want, got)| **want == *got.text(text))
                {
                    matches.push((i, end));
                }
            }
        }
        if matches.is_empty() {
            return;
        }
        // _filter_special_spans: longest first, then leftmost; keep a span
        // unless its first or last token was covered by an earlier candidate.
        matches.sort_unstable_by(|x, y| (y.1 - y.0).cmp(&(x.1 - x.0)).then(x.0.cmp(&y.0)));
        let mut seen: HashSet<usize, FxBuildHasher> = HashSet::default();
        let mut kept = Vec::new();
        for (s, e) in matches {
            if !seen.contains(&s) && !seen.contains(&(e - 1)) {
                kept.push((s, e));
            }
            seen.extend(s..e);
        }
        kept.sort_unstable();
        // _retokenize_special_spans: re-split a span whose text is a rule.
        let mut new = Vec::with_capacity(toks.len());
        let mut k = kept.iter().peekable();
        let mut i = 0;
        while i < toks.len() {
            match k.peek() {
                Some(&&(s, e)) if s == i => {
                    k.next();
                    let (lo, hi) = (toks[s].start as usize, toks[e - 1].end as usize);
                    match self.specials.get(&text[lo..hi]) {
                        Some(pieces) => Self::push_pieces(&mut new, lo, pieces),
                        None => new.extend_from_slice(&toks[s..e]),
                    }
                    i = e;
                }
                _ => {
                    new.push(toks[i]);
                    i += 1;
                }
            }
        }
        out.truncate(base);
        out.extend(new);
    }
}

/// Is `s` one of the sentencizer's punctuation characters?
fn is_sent_punct(s: &str) -> bool {
    let mut it = s.chars();
    match (it.next(), it.next()) {
        (Some(c), None) => tables::SENT_PUNCT.binary_search(&c).is_ok(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected values recorded from datatrove 0.10.1 + spaCy 3.8.18
    /// (`split_into_words`, `len(split_into_sentences)`) on CPython 3.13.
    const CASES: &[(&str, &[&str], usize)] = &[
        ("", &[], 1),
        (" ", &[], 1),
        ("e.g .", &["e.g", "."], 1),
        ("(don't)", &["(", "do", "n't", ")"], 1),
        ("''hello''", &["''", "hello", "''"], 1),
        ("U.S.A.", &["U.S.A."], 1),
        ("10km", &["10", "km"], 1),
        ("$100", &["$", "100"], 1),
        (
            "http://www.example.com/path?x=1.",
            &["http://www.example.com/path?x=1", "."],
            1,
        ),
        ("user@example.com", &["user@example.com"], 1),
        ("a\u{a0}b", &["a", "b"], 1),
        ("  x  ", &["x"], 1),
        ("\n\nHello. World\n", &["Hello", ".", "World"], 2),
        (
            "Hello!!! World... Yes",
            &["Hello", "!", "!", "!", "World", "...", "Yes"],
            2,
        ),
        ("can't won't I'm", &["ca", "n't", "wo", "n't", "I", "'m"], 1),
        (":) :-( <3", &[":)", ":-(", "<3"], 1),
        ("dyn-o-mite", &["dyn", "-", "o", "-", "mite"], 1),
        ("3.14", &["3.14"], 1),
        ("4-5", &["4", "-", "5"], 1),
        ("a.b.c", &["a.b.c"], 1),
        (
            "The U.S. and e.g. etc.",
            &["The", "U.S.", "and", "e.g.", "etc", "."],
            1,
        ),
        (
            "\u{2018}quoted\u{2019} \u{201c}double\u{201d}",
            &[
                "\u{2018}", "quoted", "\u{2019}", "\u{201c}", "double", "\u{201d}",
            ],
            1,
        ),
        ("1990s", &["1990s"], 1),
        ("C++", &["C++"], 1),
        ("10a.m.", &["10", "a.m."], 1),
        ("(10a.m.)", &["(", "10", "a.m.", ")"], 1),
    ];

    #[test]
    fn recorded_cases() {
        for v in [
            UnicodeVersion::V14_0,
            UnicodeVersion::V15_0,
            UnicodeVersion::V15_1,
            UnicodeVersion::V16_0,
            UnicodeVersion::V17_0,
        ] {
            let tok = SpacyEnglish::get(v);
            for &(text, words, sents) in CASES {
                assert_eq!(tok.words(text), words, "words of {text:?}");
                assert_eq!(tok.count_sentences(text), sents, "sentences of {text:?}");
            }
        }
    }

    #[test]
    fn whitespace_tokens() {
        let tok = SpacyEnglish::get(UnicodeVersion::V15_1);
        let text = "  x  y\n";
        let mut toks = Vec::new();
        tok.tokens(text, &mut toks);
        let got: Vec<(&str, bool)> = toks.iter().map(|t| (t.text(text), t.space)).collect();
        assert_eq!(
            got,
            [
                ("  ", true),
                ("x", false),
                (" ", true),
                ("y", false),
                ("\n", true)
            ]
        );
    }

    #[test]
    fn long_url_like_token_is_linear() {
        // A backtracking URL matcher would recurse once per host label.
        let tok = SpacyEnglish::get(UnicodeVersion::V15_1);
        let text = "a.".repeat(200_000) + "b";
        assert_eq!(tok.words(&text).len(), 1);
    }
}
