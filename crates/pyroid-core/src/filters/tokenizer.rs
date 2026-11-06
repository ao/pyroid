// Derived from spaCy 3.8.18 (MIT), © ExplosionAI GmbH, and datatrove 0.10.1
// (Apache-2.0), © HuggingFace; modified.
//! datatrove's spaCy word tokenizers: spaCy 3.8.18 `spacy.blank(code)` plus
//! the `sentencizer` pipe, as `datatrove.utils.word_tokenizers.SpaCyTokenizer`
//! runs it, for every rule-based language datatrove uses (57 spaCy codes:
//! `en`, `de`, `fr`, ..., and `xx`, spaCy's multi-language rules, which
//! datatrove assigns to about 2,000 language codes).
//!
//! This is a port of `spacy/tokenizer.pyx` (`_tokenize_affixes`,
//! `_split_affixes`, `_attach_tokens`, `_apply_special_cases`) and of
//! `Sentencizer.predict`, over each language's own spaCy rules (`tables.rs`:
//! prefix / suffix / infix / URL / `token_match` patterns and tokenizer
//! exceptions). The tokenizer's cache is not modelled: datatrove tokenizes
//! inside `memory_zone()`, where spaCy never writes the cache, so every span
//! is tokenized from the rules.

use std::collections::{HashMap, HashSet};
use std::sync::{LazyLock, OnceLock};

use rustc_hash::FxBuildHasher;

use super::regex::{Alternatives, Class, Matcher, Nfa, Node, version_index};
use super::tables;
use super::text::CharSet;
use super::unicode;
use crate::ftfy::UnicodeVersion;

/// One language's generated tokenizer data (see `generate_tables.py`).
#[allow(clippy::struct_excessive_bools)] // generated data: one flag per pattern
#[derive(Debug)]
pub(crate) struct LangData {
    /// spaCy's language code (`spacy.blank(code)`).
    pub code: &'static str,
    pub prefix: &'static [Node],
    /// Characters a prefix can start with (a prefilter).
    pub prefix_first: &'static Class,
    /// Some prefix alternative can match empty (no prefilter then).
    pub prefix_nullable: bool,
    pub suffix: &'static [Node],
    pub suffix_first: &'static Class,
    pub suffix_nullable: bool,
    /// Characters a suffix can end with.
    pub suffix_last: &'static Class,
    /// First characters of the suffix alternatives of unbounded width.
    pub suffix_unbounded_first: &'static Class,
    /// Longest match, in characters, of the bounded suffix alternatives.
    pub suffix_max_bounded: usize,
    pub infix: &'static [Node],
    pub infix_first: &'static Class,
    /// Some infix alternative can match empty (Catalan, French, ...).
    pub infix_nullable: bool,
    pub url: Option<&'static [Node]>,
    pub token_match: Option<&'static [Node]>,
    /// The rules include `BASE_RULES`.
    pub base_rules: bool,
    /// The other rules: NUL between rules, U+0001 between the ORTH pieces of
    /// a rule (a one-piece rule is its key).
    pub rules: &'static str,
    /// How many rules in all (base included).
    pub n_rules: usize,
}

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

/// spaCy's tokenizer + sentencizer for one language and Unicode version.
///
/// Build-once state (rules, matcher patterns, the URL automaton); `Sync`, so
/// one instance serves every thread. Get one with [`SpacyTokenizer::get`].
#[derive(Debug)]
pub struct SpacyTokenizer {
    lang: &'static LangData,
    version: UnicodeVersion,
    specials: HashMap<Box<str>, Box<[&'static str]>, FxBuildHasher>,
    patterns: HashMap<Box<str>, Vec<Pattern>, FxBuildHasher>,
    /// First characters of the patterns' first tokens.
    pattern_first: CharSet,
    prefix: Alternatives,
    suffix: Alternatives,
    infix: Alternatives,
    url: Option<Nfa>,
    token_match: Option<Alternatives>,
    /// Slot of this (language, version) in the per-thread span cache.
    index: usize,
}

/// One slot per (language, Unicode version), built on first use.
static INSTANCES: LazyLock<Box<[OnceLock<SpacyTokenizer>]>> = LazyLock::new(|| {
    (0..tables::LANGS.len() * 5)
        .map(|_| OnceLock::new())
        .collect()
});

/// Index of English in `tables::LANGS`.
static ENGLISH: LazyLock<usize> = LazyLock::new(|| lang_index("en").expect("English tables"));

fn lang_index(code: &str) -> Option<usize> {
    tables::LANGS.binary_search_by(|l| l.code.cmp(code)).ok()
}

/// The spaCy codes pyroid has rules for (`spacy.blank(code)`), sorted.
pub fn spacy_languages() -> impl Iterator<Item = &'static str> {
    tables::LANGS.iter().map(|l| l.code)
}

/// Non-whitespace spans up to this many bytes are cached per thread.
const CACHE_MAX_SPAN: usize = 32;
/// Cached spans per thread and tokenizer (the cache is cleared when full).
const CACHE_CAP: usize = 1 << 16;

/// A cached span tokenization: token offsets within the span.
#[derive(Debug)]
struct Cached {
    toks: Box<[(u8, u8)]>,
    special: bool,
}

type SpanCache = HashMap<Box<str>, Cached, FxBuildHasher>;

thread_local! {
    static SPAN_CACHE: std::cell::RefCell<Vec<SpanCache>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// spaCy refuses texts of 2**30 characters or more (`Errors.E025`).
pub const MAX_CHARS: usize = 1 << 30;

/// Parse a `LangData::rules` string into `out`.
fn load_rules(blob: &'static str, out: &mut HashMap<Box<str>, Box<[&'static str]>, FxBuildHasher>) {
    if blob.is_empty() {
        return;
    }
    for rule in blob.split('\0') {
        let pieces: Box<[&'static str]> = rule.split('\u{1}').collect();
        let key: Box<str> = if pieces.len() == 1 {
            rule.into()
        } else {
            pieces.concat().into()
        };
        out.insert(key, pieces);
    }
}

impl SpacyTokenizer {
    /// The shared instance for spaCy language `code` (as `spacy.blank(code)`;
    /// see [`spacy_languages`]) and `version`, built on first use; `None` if
    /// pyroid has no rules for `code`.
    pub fn get(code: &str, version: UnicodeVersion) -> Option<&'static SpacyTokenizer> {
        lang_index(code).map(|i| Self::at(i, version))
    }

    /// The English tokenizer (`spacy.blank("en")`) for `version`.
    pub fn english(version: UnicodeVersion) -> &'static SpacyTokenizer {
        Self::at(*ENGLISH, version)
    }

    fn at(lang: usize, version: UnicodeVersion) -> &'static SpacyTokenizer {
        let index = lang * 5 + version_index(version);
        INSTANCES[index].get_or_init(|| Self::new(&tables::LANGS[lang], version, index))
    }

    fn new(lang: &'static LangData, version: UnicodeVersion, index: usize) -> Self {
        let mut specials = HashMap::with_capacity_and_hasher(lang.n_rules, FxBuildHasher);
        if lang.base_rules {
            load_rules(tables::BASE_RULES, &mut specials);
        }
        load_rules(lang.rules, &mut specials);
        debug_assert_eq!(specials.len(), lang.n_rules, "{}", lang.code);
        let mut me = Self {
            lang,
            version,
            specials,
            patterns: HashMap::default(),
            pattern_first: CharSet::new([]),
            prefix: Alternatives::new(lang.prefix, false),
            suffix: Alternatives::new(lang.suffix, false),
            infix: Alternatives::new(lang.infix, false),
            url: lang.url.map(Nfa::new),
            token_match: lang.token_match.map(|t| Alternatives::new(t, true)),
            index,
        };
        // Tokenizer.add_special_case: rules with an affix, an infix or a space
        // go to the PhraseMatcher, as their tokenization without specials.
        let mut patterns: HashMap<Box<str>, Vec<Pattern>, FxBuildHasher> = HashMap::default();
        let mut toks = Vec::new();
        let mut infixes = Vec::new();
        let mut keys: Vec<(&str, usize)> =
            me.specials.iter().map(|(k, p)| (&**k, p.len())).collect();
        // PhraseMatcher keys are per rule; the order of patterns sharing a
        // first token only matters for duplicates, so any fixed order works.
        keys.sort_unstable();
        for (key, n_pieces) in keys {
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
            // A one-token match is filtered after every longer one, so it can
            // only block itself; if the rule re-splits it the same way it is a
            // no-op and needs no pattern.
            if tokens.len() == 1 && n_pieces == 1 {
                continue;
            }
            if let Some(first) = tokens.first() {
                patterns
                    .entry(first.clone())
                    .or_default()
                    .push(Pattern { tokens });
            }
        }
        me.pattern_first = CharSet::new(patterns.keys().filter_map(|k| k.chars().next()));
        me.patterns = patterns;
        me
    }

    /// Can a pattern of the special-case matcher start with token `s`?
    #[inline]
    fn is_pattern_first(&self, s: &str) -> bool {
        s.chars()
            .next()
            .is_some_and(|c| self.pattern_first.contains(c))
            && self.patterns.contains_key(s)
    }

    /// spaCy's language code.
    pub fn language(&self) -> &'static str {
        self.lang.code
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
        if self.tokenize_affixes(text, true, out) {
            self.apply_special_cases(text, out, base);
        }
    }

    /// datatrove's `split_into_words(text, language)`: the non-whitespace
    /// tokens.
    pub fn words<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut toks = Vec::new();
        self.tokens(text, &mut toks);
        toks.iter()
            .filter(|t| !t.space)
            .map(|t| t.text(text))
            .collect()
    }

    /// datatrove's `len(split_into_sentences(text, language))`: the
    /// sentencizer's sentence count, or 1 when there are no tokens.
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
        if !self.lang.prefix_nullable && !self.lang.prefix_first.contains(c, self.version) {
            return 0;
        }
        self.prefix
            .match_at(&Matcher { s, v: self.version }, 0, false)
            .unwrap_or(0)
    }

    /// `suffix_search(s)`: the leftmost match of `(?:...$|...$)`.
    fn suffix_search(&self, s: &str) -> Option<(usize, usize)> {
        let v = self.version;
        let lang = self.lang;
        let m = Matcher { s, v };
        if lang.suffix_nullable {
            // Some alternative can match empty: try every position, the end
            // included, without prefilters.
            return s
                .char_indices()
                .map(|(i, _)| i)
                .chain([s.len()])
                .find_map(|pos| self.suffix.match_at(&m, pos, false).map(|end| (pos, end)));
        }
        let last = s.chars().next_back()?;
        // `$` also matches before a final "\n"; take the unfiltered path then.
        let slow = last == '\n';
        if !slow && !lang.suffix_last.contains(last, v) {
            return None;
        }
        let tail_start = if slow {
            0
        } else {
            let mut tail = s.len();
            for _ in 0..lang.suffix_max_bounded {
                match s[..tail].chars().next_back() {
                    Some(c) => tail -= c.len_utf8(),
                    None => break,
                }
            }
            tail
        };
        for (pos, c) in s.char_indices() {
            let candidate = slow
                || if pos >= tail_start {
                    lang.suffix_first.contains(c, v)
                } else {
                    lang.suffix_unbounded_first.contains(c, v)
                };
            if candidate {
                if let Some(end) = self.suffix.match_at(&m, pos, false) {
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

    /// `Tokenizer.find_infix`: `(start, end)` of every `infix_finditer`
    /// match, empty ones included (after an empty match the next one must
    /// not be empty at the same position, as in `re`).
    fn find_infix(&self, s: &str, out: &mut Vec<(usize, usize)>) {
        let v = self.version;
        let lang = self.lang;
        let m = Matcher { s, v };
        let mut pos = 0;
        let mut must_advance = false;
        'search: loop {
            let mut at = pos;
            loop {
                let c = s[at..].chars().next();
                let candidate =
                    lang.infix_nullable || c.is_some_and(|c| lang.infix_first.contains(c, v));
                if candidate {
                    if let Some(end) = self.infix.match_at(&m, at, must_advance && at == pos) {
                        out.push((at, end));
                        must_advance = end == at;
                        pos = end;
                        continue 'search;
                    }
                }
                match c {
                    Some(c) => at += c.len_utf8(),
                    None => break 'search,
                }
            }
        }
    }

    /// `url_match(s)`.
    fn url_match(&self, s: &str) -> bool {
        // Every language shares spaCy's URL_PATTERN, both of whose
        // alternatives (IP address, host name) contain a literal '.' (tested
        // below), so without one there is no match.
        self.url
            .as_ref()
            .is_some_and(|url| s.as_bytes().contains(&b'.') && url.is_match(s, self.version))
    }

    /// `token_match(s)` (only some languages have one).
    #[inline]
    fn token_match(&self, s: &str) -> bool {
        self.token_match
            .as_ref()
            .is_some_and(|t| t.is_match(s, self.version))
    }

    // ----------------------------------------------------- the tokenizer

    /// `Tokenizer._tokenize_affixes`.
    /// Returns true if a token can start a special-case pattern.
    fn tokenize_affixes(&self, text: &str, with_specials: bool, out: &mut Vec<Token>) -> bool {
        let Some(c0) = text.chars().next() else {
            return false;
        };
        let mut special = false;
        let mut in_ws = unicode::is_space(c0);
        let mut start = 0;
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let b = bytes[i];
            let (space_here, len) = if b < 0x80 {
                (matches!(b, b'\t'..=b'\r' | 0x1c..=b' '), 1)
            } else {
                let c = text[i..].chars().next().unwrap_or('\0');
                (unicode::is_space(c), c.len_utf8())
            };
            if space_here != in_ws {
                if start < i {
                    special |= self.span(text, start, i, in_ws, with_specials, out);
                }
                start = if b == b' ' { i + 1 } else { i };
                in_ws = !in_ws;
            }
            i += len;
        }
        if start < text.len() {
            special |= self.span(text, start, text.len(), in_ws, with_specials, out);
        }
        special
    }

    /// One span, through the per-thread cache when it is a short
    /// non-whitespace span (its tokens depend only on its text). Returns
    /// true if one of its tokens can start a special-case pattern.
    #[allow(clippy::cast_possible_truncation)] // cached spans are < 256 bytes
    fn span(
        &self,
        text: &str,
        lo: usize,
        hi: usize,
        space: bool,
        with_specials: bool,
        out: &mut Vec<Token>,
    ) -> bool {
        let before = out.len();
        let s = &text[lo..hi];
        if !with_specials || space || s.len() > CACHE_MAX_SPAN {
            self.tokenize_span(text, lo, hi, space, with_specials, out);
            return with_specials
                && out[before..]
                    .iter()
                    .any(|t| self.is_pattern_first(t.text(text)));
        }
        SPAN_CACHE.with_borrow_mut(|caches| {
            if caches.len() <= self.index {
                caches.resize_with(self.index + 1, SpanCache::default);
            }
            let cache = &mut caches[self.index];
            if let Some(hit) = cache.get(s) {
                out.extend(hit.toks.iter().map(|&(a, b)| Token {
                    start: (lo + usize::from(a)) as u32,
                    end: (lo + usize::from(b)) as u32,
                    space: false,
                }));
                return hit.special;
            }
            self.tokenize_span(text, lo, hi, false, true, out);
            let new = &out[before..];
            let special = new.iter().any(|t| self.is_pattern_first(t.text(text)));
            if cache.len() >= CACHE_CAP {
                cache.clear();
            }
            let toks = new
                .iter()
                .map(|t| ((t.start as usize - lo) as u8, (t.end as usize - lo) as u8))
                .collect();
            cache.insert(s.into(), Cached { toks, special });
            special
        })
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
        // The range last looked up in `specials` (and not found there).
        let mut checked = (lo, hi);
        // The range last tested by `token_match` (and not matched).
        let mut no_token_match = (usize::MAX, usize::MAX);
        while a < b && b - a != last_size {
            let string = &text[a..b];
            if self.token_match.is_some() {
                if self.token_match(string) {
                    break;
                }
                no_token_match = (a, b);
            }
            if with_specials && (a, b) != checked && self.specials.contains_key(string) {
                break;
            }
            checked = (a, b);
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
            let special = (with_specials && (a, b) != checked)
                .then(|| self.specials.get(core))
                .flatten();
            if let Some(pieces) = special {
                Self::push_pieces(out, a, pieces);
            } else if ((a, b) != no_token_match && self.token_match(core)) || self.url_match(core) {
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
            let s = t.text(text);
            if !s
                .chars()
                .next()
                .is_some_and(|c| self.pattern_first.contains(c))
            {
                continue;
            }
            let Some(pats) = self.patterns.get(s) else {
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
        matches.dedup();
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

    const VERSIONS: [UnicodeVersion; 5] = [
        UnicodeVersion::V14_0,
        UnicodeVersion::V15_0,
        UnicodeVersion::V15_1,
        UnicodeVersion::V16_0,
        UnicodeVersion::V17_0,
    ];

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
        for v in VERSIONS {
            let tok = SpacyTokenizer::english(v);
            for &(text, words, sents) in CASES {
                assert_eq!(tok.words(text), words, "words of {text:?}");
                assert_eq!(tok.count_sentences(text), sents, "sentences of {text:?}");
            }
        }
    }

    #[test]
    fn whitespace_tokens() {
        let tok = SpacyTokenizer::english(UnicodeVersion::V15_1);
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
        let tok = SpacyTokenizer::english(UnicodeVersion::V15_1);
        let text = "a.".repeat(200_000) + "b";
        assert_eq!(tok.words(&text).len(), 1);
    }

    #[test]
    fn every_language_builds_and_shares_the_url_pattern() {
        let url = tables::LANGS[0].url.expect("URL pattern");
        for lang in tables::LANGS {
            assert!(
                lang.url.is_some_and(|u| std::ptr::eq(u, url)),
                "{}: url_match differs (url_match's '.' shortcut assumes one pattern)",
                lang.code
            );
            for v in VERSIONS {
                let tok = SpacyTokenizer::get(lang.code, v).expect("listed language");
                assert_eq!(tok.language(), lang.code);
                assert_eq!(tok.specials.len(), lang.n_rules, "{}", lang.code);
                // Smoke: every tokenizer handles every script without panicking.
                let _ = tok.words("Grüße, 'l'homme' — 1990'da; 3.14% (x) dvs. ਪੰਜਾਬੀ!? 東京");
            }
        }
        assert!(SpacyTokenizer::get("zh", UnicodeVersion::V15_1).is_none());
    }

    #[test]
    fn long_tokens_with_unbounded_groups_do_not_recurse() {
        // Greek and Portuguese repeat multi-node groups without bound; a
        // backtracking matcher would use one stack frame per iteration.
        for code in ["el", "pt", "hu"] {
            let tok = SpacyTokenizer::get(code, UnicodeVersion::V15_1).expect(code);
            for text in [
                "a".to_string() + &"-a".repeat(100_000),
                "1".to_string() + &",1".repeat(100_000),
                ".a".repeat(100_000),
            ] {
                assert!(!tok.words(&text).is_empty(), "{code}");
            }
        }
    }
}
