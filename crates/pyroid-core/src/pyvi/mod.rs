// Derived from pyvi 0.1.1 (MIT), © 2016 Viet-Trung Tran, and spaCy 3.8.18
// (MIT), © ExplosionAI GmbH; modified.
//! Vietnamese word segmentation as spaCy 3.8.18's `VietnameseTokenizer` runs
//! pyvi 0.1.1 (`use_pyvi = true`, its default).
//!
//! - [`Segmenter::syllables`] is `pyvi_sylabelize_with_ws`: the `re.findall`
//!   of spaCy's pattern (whitespace runs, abbreviations, `==>` and friends,
//!   URLs, e-mail addresses, numbers with separators, other symbols, `\w+`),
//!   with `re`'s `\w`, `\d` and `\s` for the interpreter's Unicode version.
//! - [`Segmenter::tokenize`] is `pyvi_tokenize`: pyvi's CRF labels the
//!   non-space syllables (`word2features`: lower-cased words, `isupper`,
//!   `istitle`, `isdigit`, bigram and trigram hits in `words.txt`, scored by
//!   the CRFsuite model in `pyvi3.pkl` through [`crate::crfsuite`]), and a
//!   syllable labelled `I_W` joins the previous word unless either is a
//!   substring of `string.punctuation`, starts with a `str.isdigit()`
//!   character, or the rule on `istitle()` of their first characters forbids
//!   it.
//!
//! pyvi's model file and `words.txt` are bundled (`data/`, written by
//! `generate.py`). Case predicates and lowercasing follow CPython per Unicode
//! version (`unicode_tables.rs`, written by `generate_unicode.py`).
//!
//! [`vi`] is datatrove's Vietnamese word tokenizer for the filters and
//! MinHash.

use std::cell::RefCell;
use std::ops::Range;
use std::sync::LazyLock;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::crfsuite::{Model, Tagger};
use crate::filters::unicode::{is_digit as is_re_digit, is_word};
use crate::ftfy::UnicodeVersion;
use crate::text::is_py_whitespace;

#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
#[path = "unicode_tables.rs"]
mod tables;

#[cfg(test)]
mod replay;
pub mod vi;

/// pyvi 0.1.1's CRFsuite model (the model file inside `pyvi/models/pyvi3.pkl`).
static MODEL: &[u8] = include_bytes!("data/pyvi3.crfsuite");
/// pyvi 0.1.1's `pyvi/models/words.txt`.
static WORDS: &str = include_str!("data/words.txt");

// ---------------------------------------------------------------------------
// CPython's case predicates
// ---------------------------------------------------------------------------

#[inline]
fn in_ranges(ranges: &[(u32, u32)], cp: u32) -> bool {
    let after = ranges.partition_point(|&(lo, _)| lo <= cp);
    ranges
        .get(after.wrapping_sub(1))
        .is_some_and(|&(_, hi)| cp <= hi)
}

#[inline]
fn has(
    c: char,
    v: UnicodeVersion,
    ascii: u128,
    ranges: fn(UnicodeVersion) -> &'static [(u32, u32)],
) -> bool {
    let cp = u32::from(c);
    if cp < 0x80 {
        (ascii >> cp) & 1 != 0
    } else {
        in_ranges(ranges(v), cp)
    }
}

/// `Py_UNICODE_ISUPPER(c)`.
#[inline]
fn upper(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::UPPER_ASCII, tables::upper)
}

/// `Py_UNICODE_ISLOWER(c)`.
#[inline]
fn lower(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::LOWER_ASCII, tables::lower)
}

/// `Py_UNICODE_ISTITLE(c)`.
#[inline]
fn title(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::TITLE_ASCII, tables::title)
}

/// `Py_UNICODE_ISDIGIT(c)`: `c.isdigit()`.
#[inline]
pub(crate) fn char_isdigit(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::DIGIT_ASCII, tables::digit)
}

/// `c.istitle()` of one character (CPython's shortcut: titlecase or upper).
#[inline]
fn char_istitle(c: char, v: UnicodeVersion) -> bool {
    upper(c, v) || title(c, v)
}

/// `s.isupper()`.
pub(crate) fn py_isupper(s: &str, v: UnicodeVersion) -> bool {
    let mut it = s.chars();
    let Some(first) = it.next() else {
        return false;
    };
    if it.next().is_none() {
        return upper(first, v);
    }
    let mut cased = false;
    for c in s.chars() {
        if lower(c, v) || title(c, v) {
            return false;
        } else if !cased && upper(c, v) {
            cased = true;
        }
    }
    cased
}

/// `s.istitle()`.
pub(crate) fn py_istitle(s: &str, v: UnicodeVersion) -> bool {
    let mut it = s.chars();
    let Some(first) = it.next() else {
        return false;
    };
    if it.next().is_none() {
        return char_istitle(first, v);
    }
    let (mut cased, mut prev_cased) = (false, false);
    for c in s.chars() {
        if upper(c, v) || title(c, v) {
            if prev_cased {
                return false;
            }
            prev_cased = true;
            cased = true;
        } else if lower(c, v) {
            if !prev_cased {
                return false;
            }
            prev_cased = true;
            cased = true;
        } else {
            prev_cased = false;
        }
    }
    cased
}

/// `s.isdigit()`.
pub(crate) fn py_isdigit(s: &str, v: UnicodeVersion) -> bool {
    !s.is_empty() && s.chars().all(|c| char_isdigit(c, v))
}

// ---------------------------------------------------------------------------
// The syllable pattern
// ---------------------------------------------------------------------------

/// The character after `at` (a char boundary below `s.len()`).
#[inline]
fn char_at(s: &str, at: usize) -> char {
    s[at..].chars().next().expect("a character")
}

/// The end of the longest run of characters from `at` that satisfy `f`.
#[inline]
fn run(s: &str, at: usize, f: impl Fn(char) -> bool) -> usize {
    s[at..]
        .char_indices()
        .find(|&(_, c)| !f(c))
        .map_or(s.len(), |(i, _)| at + i)
}

/// `[a-zA-Z0-9_.+-]`: the local part of spaCy's e-mail pattern.
#[inline]
fn is_local(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '+' | '-')
}

/// `[a-zA-Z0-9-]`: a domain label of spaCy's e-mail pattern.
#[inline]
fn is_domain(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-'
}

/// `[a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+` at `at`, with `re`'s
/// backtracking: the end of the match.
fn email(s: &str, at: usize) -> Option<usize> {
    let b = s.as_bytes();
    let local = run(s, at, is_local);
    if local == at || b.get(local) != Some(&b'@') {
        return None;
    }
    // Greedy `(label\.)+`, then `label`; when the last label is missing, the
    // last repetition is given back and its label ends the match.
    let mut pos = local + 1;
    let mut dots = Vec::new();
    loop {
        let end = run(s, pos, is_domain);
        if end > pos && b.get(end) == Some(&b'.') {
            dots.push(end);
            pos = end + 1;
        } else {
            break;
        }
    }
    let last = run(s, pos, is_domain);
    match dots.len() {
        0 => None,
        _ if last > pos => Some(last),
        1 => None,
        n => Some(dots[n - 1]),
    }
}

/// `\d+([\.,_]\d+)+` at `at` (`at` starts a `\d`): the end of the match.
fn number(s: &str, at: usize, v: UnicodeVersion) -> Option<usize> {
    let digit = |c: char| is_re_digit(c, v);
    let b = s.as_bytes();
    let mut end = run(s, at, digit);
    let mut groups = 0;
    while matches!(b.get(end), Some(b'.' | b',' | b'_')) {
        let next = run(s, end + 1, digit);
        if next == end + 1 {
            break;
        }
        end = next;
        groups += 1;
    }
    (groups > 0).then_some(end)
}

/// The end of the `re.findall` match of spaCy's syllable pattern that starts
/// at `at` (every character starts one).
fn syllable_end(s: &str, at: usize, v: UnicodeVersion) -> usize {
    let c = char_at(s, at);
    let rest = &s[at..];
    // \s+
    if is_py_whitespace(c) {
        return run(s, at, is_py_whitespace);
    }
    // [A-ZĐ]+\.
    if c.is_ascii_uppercase() || c == 'Đ' {
        let end = run(s, at, |c| c.is_ascii_uppercase() || c == 'Đ');
        if s.as_bytes().get(end) == Some(&b'.') {
            return end + 1;
        }
    }
    for lit in [
        "Tp.", "Mr.", "Mrs.", "Ms.", "Dr.", "ThS.", "==>", "->", "...", ">>",
    ] {
        if rest.starts_with(lit) {
            return at + lit.len();
        }
    }
    let w = is_word(c, v);
    // \w+://[^\s]+
    if w {
        let end = run(s, at, |c| is_word(c, v));
        if s[end..].starts_with("://") {
            let after = end + 3;
            let url = run(s, after, |c| !is_py_whitespace(c));
            if url > after {
                return url;
            }
        }
    }
    if is_local(c) {
        if let Some(end) = email(s, at) {
            return end;
        }
    }
    if is_re_digit(c, v) {
        if let Some(end) = number(s, at, v) {
            return end;
        }
    }
    if w {
        run(s, at, |c| is_word(c, v))
    } else {
        // [^\w\s]
        at + c.len_utf8()
    }
}

// ---------------------------------------------------------------------------
// The CRF
// ---------------------------------------------------------------------------

/// pyvi's model and n-gram lists, with the attribute ids features need.
#[derive(Debug)]
pub struct Segmenter {
    model: Model,
    bigrams: FxHashSet<&'static str>,
    trigrams: FxHashSet<&'static str>,
    /// lower-cased word -> ids of `word.lower():w`, `-1:word.lower():w` and
    /// `+1:word.lower():w` (`NONE` if the model lacks one).
    lowers: FxHashMap<Box<[u8]>, [u32; 3]>,
    ids: FixedIds,
    i_w: u32,
}

const NONE: u32 = u32::MAX;
const LOWER_PREFIXES: [&str; 3] = ["word.lower():", "-1:word.lower():", "+1:word.lower():"];

/// The ids of the attributes whose names do not depend on the words.
#[derive(Debug, Clone, Copy)]
struct FixedIds {
    bias: u32,
    isupper: u32,
    istitle: u32,
    isdigit: u32,
    m1_istitle: u32,
    m1_isupper: u32,
    m1_bigram: u32,
    m2_trigram: u32,
    p1_istitle: u32,
    p1_isupper: u32,
    p1_bigram: u32,
    p2_trigram: u32,
}

/// One non-space syllable and what its features need.
#[derive(Debug, Clone)]
struct Word {
    start: usize,
    end: usize,
    /// Its `lower()`, in `Buffers::lowered`.
    low: Range<usize>,
    lower: [u32; 3],
    upper: bool,
    title: bool,
    digit: bool,
}

#[derive(Debug, Default)]
struct Buffers {
    segs: Vec<Range<usize>>,
    words: Vec<Word>,
    lowered: String,
    key: String,
    bigram: Vec<bool>,
    trigram: Vec<bool>,
    labels: Vec<u32>,
}

thread_local! {
    static BUFFERS: RefCell<Buffers> = RefCell::new(Buffers::default());
    static TAGGER: RefCell<Option<Tagger<'static>>> = const { RefCell::new(None) };
}

static SEGMENTER: LazyLock<Segmenter> =
    LazyLock::new(|| Segmenter::new().expect("the bundled pyvi model reads"));

/// `string.punctuation`.
const PUNCTUATION: &str = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##;

impl Segmenter {
    /// The shared segmenter (built on first use).
    pub fn get() -> &'static Segmenter {
        &SEGMENTER
    }

    fn new() -> Result<Self, crate::crfsuite::ModelError> {
        let model = Model::from_bytes(MODEL)?;
        // ViTokenizer: `for token in fin.read().split('\n')`, sorted into the
        // lists by `len(token.split(' '))`.
        let (mut bigrams, mut trigrams) = (FxHashSet::default(), FxHashSet::default());
        for line in WORDS.split('\n') {
            match line.split(' ').count() {
                2 => {
                    bigrams.insert(line);
                }
                3 => {
                    trigrams.insert(line);
                }
                _ => {}
            }
        }
        let mut lowers: FxHashMap<Box<[u8]>, [u32; 3]> = FxHashMap::default();
        for (attr, id) in model.attributes() {
            for (k, p) in LOWER_PREFIXES.iter().enumerate() {
                if let Some(rest) = attr.strip_prefix(p.as_bytes()) {
                    lowers.entry(rest.into()).or_insert([NONE; 3])[k] = id;
                }
            }
        }
        let id = |name: &str| model.attr_id(name.as_bytes()).unwrap_or(NONE);
        let ids = FixedIds {
            bias: id("bias"),
            isupper: id("word.isupper()"),
            istitle: id("word.istitle()"),
            isdigit: id("word.isdigit()"),
            m1_istitle: id("-1:word.istitle()"),
            m1_isupper: id("-1:word.isupper()"),
            m1_bigram: id("-1:word.bi_gram()"),
            m2_trigram: id("-2:word.tri_gram()"),
            p1_istitle: id("+1:word.istitle()"),
            p1_isupper: id("+1:word.isupper()"),
            p1_bigram: id("+1:word.bi_gram()"),
            p2_trigram: id("+2:word.tri_gram()"),
        };
        let i_w = model
            .label_id("I_W")
            .and_then(|i| u32::try_from(i).ok())
            .ok_or_else(|| crate::crfsuite::ModelError::Format("no I_W label".into()))?;
        Ok(Self {
            model,
            bigrams,
            trigrams,
            lowers,
            ids,
            i_w,
        })
    }

    /// The model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// `pyvi_sylabelize_with_ws(text)`: byte ranges of the syllables and the
    /// whitespace runs between them (a partition of `text`), appended to `out`.
    pub fn syllables(text: &str, version: UnicodeVersion, out: &mut Vec<Range<usize>>) {
        let mut at = 0;
        while at < text.len() {
            let end = syllable_end(text, at, version);
            out.push(at..end);
            at = end;
        }
    }

    /// The ids of the `word.lower()` attributes of a lowercased word.
    fn lower_ids(&self, low: &str) -> [u32; 3] {
        // CRFsuite cuts the attribute at its first NUL.
        let key = match memchr::memchr(0, low.as_bytes()) {
            Some(i) => &low.as_bytes()[..i],
            None => low.as_bytes(),
        };
        self.lowers.get(key).copied().unwrap_or([NONE; 3])
    }

    /// Fill `bigram`/`trigram` (`' '.join(...).lower() in bi_grams/tri_grams`
    /// for each pair and triple of consecutive words). The `lower()` of the
    /// joined words is their `lower()`s joined: a space is neither cased nor
    /// case-ignorable, so it bounds the Final_Sigma context as the ends of a
    /// string do.
    fn ngrams(&self, b: &mut Buffers) {
        let Buffers {
            words,
            lowered,
            key,
            bigram,
            trigram,
            ..
        } = b;
        bigram.clear();
        trigram.clear();
        let low = |w: &Word| &lowered[w.low.clone()];
        for k in 0..words.len().saturating_sub(1) {
            key.clear();
            key.push_str(low(&words[k]));
            key.push(' ');
            key.push_str(low(&words[k + 1]));
            bigram.push(self.bigrams.contains(key.as_str()));
            if k + 2 < words.len() {
                key.push(' ');
                key.push_str(low(&words[k + 2]));
                trigram.push(self.trigrams.contains(key.as_str()));
            }
        }
    }

    /// Score the words with `word2features` and return the labels.
    fn label(&self, tagger: &mut Tagger<'_>, b: &Buffers, out: &mut Vec<u32>) {
        let ids = self.ids;
        let n = b.words.len();
        tagger.clear();
        let add = |t: &mut Tagger<'_>, id: u32| {
            if id != NONE {
                t.add_one(id);
            }
        };
        for i in 0..n {
            let w = &b.words[i];
            tagger.push_item();
            add(tagger, ids.bias);
            add(tagger, w.lower[0]);
            if w.upper {
                add(tagger, ids.isupper);
            }
            if w.title {
                add(tagger, ids.istitle);
            }
            if w.digit {
                add(tagger, ids.isdigit);
            }
            if i > 0 {
                let p = &b.words[i - 1];
                add(tagger, p.lower[1]);
                if p.title {
                    add(tagger, ids.m1_istitle);
                }
                if p.upper {
                    add(tagger, ids.m1_isupper);
                }
                if b.bigram[i - 1] {
                    add(tagger, ids.m1_bigram);
                }
                if i > 1 && b.trigram[i - 2] {
                    add(tagger, ids.m2_trigram);
                }
            }
            if i + 1 < n {
                let q = &b.words[i + 1];
                add(tagger, q.lower[2]);
                if q.title {
                    add(tagger, ids.p1_istitle);
                }
                if q.upper {
                    add(tagger, ids.p1_isupper);
                }
                if b.bigram[i] {
                    add(tagger, ids.p1_bigram);
                }
                if i + 2 < n && b.trigram[i] {
                    add(tagger, ids.p2_trigram);
                }
            }
        }
        tagger.viterbi(out);
    }

    /// Syllables, words and n-gram hits of `text` (not whitespace-only).
    fn prepare(&self, text: &str, version: UnicodeVersion, b: &mut Buffers) {
        b.segs.clear();
        Self::syllables(text, version, &mut b.segs);
        b.words.clear();
        b.lowered.clear();
        for r in &b.segs {
            let s = &text[r.clone()];
            // Whitespace runs are syllables of their own.
            if s.chars().all(is_py_whitespace) {
                continue;
            }
            let start = b.lowered.len();
            if s.is_ascii() {
                b.lowered.extend(s.chars().map(|c| c.to_ascii_lowercase()));
            } else {
                b.lowered
                    .push_str(&crate::filters::url::py_lower(s, version));
            }
            let low = start..b.lowered.len();
            b.words.push(Word {
                start: r.start,
                end: r.end,
                lower: self.lower_ids(&b.lowered[low.clone()]),
                low,
                upper: py_isupper(s, version),
                title: py_istitle(s, version),
                digit: py_isdigit(s, version),
            });
        }
        self.ngrams(b);
    }

    /// The labels of the CRF on `b.words`.
    fn run_crf(&'static self, b: &Buffers, out: &mut Vec<u32>) {
        TAGGER.with_borrow_mut(|t| {
            let tagger = t.get_or_insert_with(|| Tagger::new(&self.model));
            self.label(tagger, b, out);
        });
    }

    /// `pyvi_tokenize(text)`: the words spaCy's tokenizer gets, as byte
    /// ranges of `text`, appended to `out` (none for an empty text; the whole
    /// text for a whitespace-only one).
    pub fn tokenize(
        &'static self,
        text: &str,
        version: UnicodeVersion,
        out: &mut Vec<Range<usize>>,
    ) {
        if text.is_empty() {
            return;
        }
        if text.chars().all(is_py_whitespace) {
            out.push(0..text.len());
            return;
        }
        BUFFERS.with_borrow_mut(|b| {
            self.prepare(text, version, b);
            let mut labels = std::mem::take(&mut b.labels);
            self.run_crf(b, &mut labels);
            self.merge(text, &b.words, &labels, version, out);
            b.labels = labels;
        });
    }

    /// The CRF's labels for the non-space syllables of `text`
    /// (`model.predict(...)[0]`), `true` for `I_W`; empty when the model
    /// does not run (an empty or whitespace-only text).
    pub fn labels(&'static self, text: &str, version: UnicodeVersion) -> Vec<bool> {
        if text.chars().all(is_py_whitespace) {
            return Vec::new();
        }
        BUFFERS.with_borrow_mut(|b| {
            self.prepare(text, version, b);
            let mut labels = Vec::new();
            self.run_crf(b, &mut labels);
            labels.iter().map(|&l| l == self.i_w).collect()
        })
    }

    /// spaCy's merge loop over the labelled words.
    fn merge(
        &self,
        text: &str,
        words: &[Word],
        labels: &[u32],
        version: UnicodeVersion,
        out: &mut Vec<Range<usize>>,
    ) {
        let Some(first) = words.first() else {
            return;
        };
        let mut cur = first.start..first.end;
        for i in 1..words.len() {
            let (w, p) = (
                &text[words[i].start..words[i].end],
                &text[words[i - 1].start..words[i - 1].end],
            );
            let (c, pc) = (char_at(w, 0), char_at(p, 0));
            if labels[i] == self.i_w
                && !PUNCTUATION.contains(w)
                && !PUNCTUATION.contains(p)
                && !char_isdigit(c, version)
                && !char_isdigit(pc, version)
                && (!char_istitle(c, version) || char_istitle(pc, version))
            {
                cur.end = words[i].end;
            } else {
                out.push(cur);
                cur = words[i].start..words[i].end;
            }
        }
        out.push(cur);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    fn syl(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        Segmenter::syllables(s, V, &mut out);
        out.into_iter().map(|r| &s[r]).collect()
    }

    fn tok(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        Segmenter::get().tokenize(s, V, &mut out);
        out.into_iter().map(|r| &s[r]).collect()
    }

    #[test]
    fn model_reads() {
        let m = Segmenter::get().model();
        assert_eq!(&*m.labels()[0], "B_W");
        assert_eq!(m.num_attrs(), 6335);
    }

    #[test]
    fn syllable_pattern() {
        assert_eq!(
            syl("TP. Hồ, Mrs. X ==> a->b ... >>"),
            [
                "TP.", " ", "Hồ", ",", " ", "Mrs.", " ", "X", " ", "==>", " ", "a", "->", "b", " ",
                "...", " ", ">>"
            ]
        );
        assert_eq!(syl("http://x.y/z w"), ["http://x.y/z", " ", "w"]);
        assert_eq!(
            syl("a@b.c. a@b.c.d x@y."),
            ["a@b.c", ".", " ", "a@b.c.d", " ", "x", "@", "y", "."]
        );
        assert_eq!(
            syl("1.000,5 1.2. 12"),
            ["1.000,5", " ", "1.2", ".", " ", "12"]
        );
        assert_eq!(syl(""), Vec::<&str>::new());
    }

    #[test]
    fn case_predicates() {
        assert!(py_isupper("ABC", V));
        assert!(!py_isupper("AbC", V));
        assert!(!py_isupper("123", V));
        assert!(py_istitle("Việt", V));
        assert!(!py_istitle("ViỆt", V));
        assert!(py_istitle("ǅ", V));
        assert!(py_isdigit("²3", V));
        assert!(!py_isdigit("", V));
    }

    #[test]
    fn datatrove_example() {
        // Recorded from datatrove 0.10.1 + spaCy 3.8.18 + pyvi 0.1.1 (word_tokenize).
        let t = "Hôm nay, Việt Nam  có 12.500 người. TP. Hồ Chí Minh đẹp lắm!\nMr. Smith đến từ Hà Nội.";
        assert_eq!(
            tok(t),
            [
                "Hôm nay",
                ",",
                "Việt Nam",
                "có",
                "12.500",
                "người",
                ".",
                "TP.",
                "Hồ Chí Minh",
                "đẹp",
                "lắm",
                "!",
                "Mr.",
                "Smith",
                "đến",
                "từ",
                "Hà Nội",
                "."
            ]
        );
    }
}
