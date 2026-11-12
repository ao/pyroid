// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! The repetition filters' `score_document`
//! (`nemo_curator.stages.text.filters.heuristic.repetition`).
//!
//! N-grams are windows over word ids: each distinct word gets a `u32`, and a window is
//! hashed as its slice of ids (FxHash), so a document's words are hashed once.

use std::cell::RefCell;
use std::hash::Hasher;
use std::ops::ControlFlow;

use memchr::memmem;
use rustc_hash::{FxHashSet, FxHasher};

use super::pyhash::{self, DistinctSet, HashKey};
use super::text::{
    ScoreError, Splitter, char_len, for_each_word_span, paragraphs, ratio, sentences,
};

/// Counts over `items`: how many, how many distinct, and their lengths (code points).
struct Repeats {
    total: usize,
    distinct: usize,
    chars: usize,
    distinct_chars: usize,
}

fn repeats<'a>(items: impl Iterator<Item = &'a str>) -> Repeats {
    let mut seen = FxHashSet::default();
    let mut r = Repeats {
        total: 0,
        distinct: 0,
        chars: 0,
        distinct_chars: 0,
    };
    for s in items {
        let len = char_len(s);
        r.total += 1;
        r.chars += len;
        if seen.insert(s) {
            r.distinct += 1;
            r.distinct_chars += len;
        }
    }
    r
}

/// `RepeatedLinesFilter`: `len(set(sentences)) / len(sentences)`.
pub fn repeated_lines(text: &str) -> Result<f64, ScoreError> {
    let r = repeats(sentences(text));
    if r.total == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(r.distinct, r.total))
}

/// `RepeatedParagraphsFilter`: `len(set(paragraphs)) / len(paragraphs)`.
pub fn repeated_paragraphs(text: &str) -> f64 {
    // `split` gives at least one paragraph.
    let r = repeats(paragraphs(text));
    ratio(r.distinct, r.total)
}

/// `RepeatedLinesByCharFilter`: `len("".join(set(sentences))) / len("".join(sentences))`.
pub fn repeated_lines_by_char(text: &str) -> Result<f64, ScoreError> {
    let r = repeats(sentences(text));
    if r.chars == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(r.distinct_chars, r.chars))
}

/// `RepeatedParagraphsByCharFilter`: the same over paragraphs; divides by zero when every
/// paragraph is empty (`""`, `"\n\n"`).
pub fn repeated_paragraphs_by_char(text: &str) -> Result<f64, ScoreError> {
    let r = repeats(paragraphs(text));
    if r.chars == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(r.distinct_chars, r.chars))
}

/// The words of the n-gram filters: byte ranges of a document, or a list of str.
trait WordList {
    fn count(&self) -> usize;
    fn word(&self, i: usize) -> &str;
}

/// Words as byte ranges of `text`.
struct Spans<'a> {
    text: &'a str,
    spans: &'a [(usize, usize)],
}

impl WordList for Spans<'_> {
    #[inline]
    fn count(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    fn word(&self, i: usize) -> &str {
        let (a, b) = self.spans[i];
        &self.text[a..b]
    }
}

impl WordList for [&str] {
    #[inline]
    fn count(&self) -> usize {
        self.len()
    }

    #[inline]
    fn word(&self, i: usize) -> &str {
        self[i]
    }
}

/// Per-thread buffers of the n-gram filters, reused from one document to the next.
#[derive(Debug, Default)]
struct Scratch {
    /// Each word's byte range in the document.
    spans: Vec<(usize, usize)>,
    /// Each word's hash (equal words, equal hashes).
    word_hash: Vec<u64>,
    /// Each window's hash.
    win_hash: Vec<u64>,
    /// The open-addressing table of distinct windows: 0 when empty, else index + 1 in `uniq`.
    slots: Vec<u32>,
    /// `(count, first start)` per distinct window, in order of first occurrence.
    uniq: Vec<(usize, usize)>,
    /// CPython's `hash()` of each word (the tie-break only).
    lanes: Vec<i64>,
    /// The distinct windows as CPython's `set` holds them (the tie-break only).
    set: DistinctSet,
    /// Scratch for [`pyhash::str_hash`].
    buf: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Run `f` with this thread's [`Scratch`] (fresh buffers if it is in use or gone).
fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    let mut f = Some(f);
    let done = SCRATCH
        .try_with(|s| {
            s.try_borrow_mut()
                .ok()
                .map(|mut s| (f.take().expect("called once"))(&mut s))
        })
        .ok()
        .flatten();
    match done {
        Some(r) => r,
        None => (f.take().expect("not called yet"))(&mut Scratch::default()),
    }
}

/// The document's words (NeMo's `splitter(text.strip())`) as byte ranges of `text`.
fn word_spans(text: &str, splitter: Splitter, spans: &mut Vec<(usize, usize)>) {
    spans.clear();
    match splitter {
        Splitter::Whitespace => for_each_word_span(text, |a, b, _| {
            spans.push((a, b));
            ControlFlow::Continue(())
        }),
        Splitter::Jieba => {
            // jieba's words are slices of the stripped text, so of `text`.
            let base = text.as_ptr() as usize;
            for w in splitter.doc_words(text) {
                let start = w.as_ptr() as usize - base;
                spans.push((start, start + w.len()));
            }
        }
    }
}

/// The multiplier of the rolling window hash (odd).
const ROLL: u64 = 0x9E37_79B9_7F4A_7C15;

/// `hash[i] = Σ_j word_hash[i + j] · ROLL^(n - 1 - j)` for every window of `n` words, rolled
/// from one window to the next.
fn window_hashes(word_hash: &[u64], n: usize, out: &mut Vec<u64>) {
    out.clear();
    if word_hash.len() < n || n == 0 {
        return;
    }
    let lead = (1..n).fold(1u64, |p, _| p.wrapping_mul(ROLL));
    let mut h = word_hash[..n]
        .iter()
        .fold(0u64, |h, &x| h.wrapping_mul(ROLL).wrapping_add(x));
    out.reserve(word_hash.len() - n + 1);
    out.push(h);
    for i in n..word_hash.len() {
        h = h
            .wrapping_sub(word_hash[i - n].wrapping_mul(lead))
            .wrapping_mul(ROLL)
            .wrapping_add(word_hash[i]);
        out.push(h);
    }
}

/// A word's hash: equal words, equal hashes.
#[inline]
fn word_hash(w: &str) -> u64 {
    let mut h = FxHasher::default();
    h.write(w.as_bytes());
    h.finish()
}

/// The table slot a window hash starts probing at.
#[inline]
fn slot_of(h: u64, mask: usize) -> usize {
    // The high half of a multiply mixes every bit of `h`; `mask` is below 2**32 in practice.
    #[allow(clippy::cast_possible_truncation)]
    let mixed = ((h ^ (h >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9) >> 32) as usize;
    mixed & mask
}

/// Windows `a` and `b` of `n` words are equal: the same words (hashes first, then bytes).
#[inline]
fn same_window<W: WordList + ?Sized>(
    words: &W,
    hash: &[u64],
    a: usize,
    b: usize,
    n: usize,
) -> bool {
    (0..n).all(|j| hash[a + j] == hash[b + j])
        && (0..n).all(|j| words.word(a + j) == words.word(b + j))
}

/// Count the distinct windows of `n` words (`n >= 1`, at most the number of words) into
/// `s.uniq`, in order of first occurrence, calling `each(i, repeat)` for every window `i` in
/// order (`repeat`: an equal window came before). Windows are told apart by their words,
/// never by their hashes alone.
fn tally_windows<W: WordList + ?Sized>(
    words: &W,
    n: usize,
    s: &mut Scratch,
    mut each: impl FnMut(usize, bool),
) {
    s.word_hash.clear();
    s.word_hash
        .extend((0..words.count()).map(|i| word_hash(words.word(i))));
    window_hashes(&s.word_hash, n, &mut s.win_hash);
    let windows = s.win_hash.len();
    let size = (windows * 2).next_power_of_two().max(16);
    s.slots.clear();
    s.slots.resize(size, 0);
    let mask = size - 1;
    s.uniq.clear();
    for window in 0..windows {
        let hash = s.win_hash[window];
        let mut slot = slot_of(hash, mask);
        let repeat = loop {
            let entry = s.slots[slot];
            if entry == 0 {
                s.slots[slot] = u32::try_from(s.uniq.len() + 1).expect("fewer than 2**32 windows");
                s.uniq.push((1, window));
                break false;
            }
            let seen = entry as usize - 1;
            let start = s.uniq[seen].1;
            if s.win_hash[start] == hash && same_window(words, &s.word_hash, start, window, n) {
                s.uniq[seen].0 += 1;
                break true;
            }
            slot = (slot + 1) & mask;
        };
        each(window, repeat);
    }
}

/// `RepeatingTopNGramsFilter`'s result for one document.
#[derive(Clone, Debug, PartialEq)]
pub enum TopNGram {
    /// NeMo's score: the most frequent n-gram is unique, there are fewer than `n` words, or
    /// the tie was resolved with the interpreter's hash key.
    Score(f64),
    /// Several n-grams share the top count. NeMo takes the first of them in the iteration
    /// order of `set(ngrams)`, which depends on the process's string hash seed; without the
    /// key the caller picks it in Python.
    Tied(Tie),
}

/// The tied n-grams of a [`TopNGram::Tied`] document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tie {
    /// How often each of them occurs.
    pub count: usize,
    /// Word index of each tied n-gram's first occurrence, in increasing order; `None` when
    /// every distinct n-gram is tied (NeMo then takes the first element of the set).
    pub starts: Option<Vec<usize>>,
}

/// NeMo's score for the n-gram `words` (joined with single spaces): the characters
/// `text.replace(joined, "")` removes, over `len(text)`.
pub fn top_ngram_score(text: &str, words: &[&str]) -> f64 {
    let joined = words.join(" ");
    let nchar = char_len(text);
    if nchar == 0 {
        return 1.0;
    }
    if joined.is_empty() {
        return 0.0;
    }
    // `str.replace` removes non-overlapping occurrences, left to right.
    let k = memmem::find_iter(text.as_bytes(), joined.as_bytes()).count();
    ratio(k * char_len(&joined), nchar)
}

/// The distinct n-grams `s.uniq` (as [`tally_windows`] leaves them) as CPython's `set(ngrams)`
/// holds them with `key`. `set()` inserts the n-grams in list order and a repeat changes
/// nothing, so inserting the first occurrences in order gives the same table.
///
/// Fills `s.set` (its items are indices into `s.uniq`); `s.word_hash` and `s.uniq` are
/// [`tally_windows`]' and `s.slots` is reused.
fn build_set<W: WordList + ?Sized>(key: HashKey, words: &W, n: usize, s: &mut Scratch) {
    // hash(word), once per distinct word (words with equal hashes are compared).
    let count = words.count();
    let size = (count * 2).next_power_of_two().max(16);
    s.slots.clear();
    s.slots.resize(size, 0);
    let mask = size - 1;
    s.lanes.clear();
    for word in 0..count {
        let hash = s.word_hash[word];
        let mut slot = slot_of(hash, mask);
        let lane = loop {
            let entry = s.slots[slot];
            if entry == 0 {
                s.slots[slot] = u32::try_from(word + 1).expect("fewer than 2**32 words");
                break pyhash::str_hash(key, words.word(word), &mut s.buf);
            }
            let seen = entry as usize - 1;
            if s.word_hash[seen] == hash && words.word(seen) == words.word(word) {
                break s.lanes[seen];
            }
            slot = (slot + 1) & mask;
        };
        s.lanes.push(lane);
    }
    s.set.clear();
    for &(_, start) in &s.uniq {
        // The n-grams are distinct: an equal hash is a collision, never the same tuple.
        s.set.add(pyhash::tuple_hash(
            s.lanes[start..start + n].iter().copied(),
        ));
    }
}

/// `list(set(ngrams))` for the n-grams of `words` (`n >= 1`), with CPython's hash `key`: the
/// start of each distinct n-gram's first occurrence, in the set's iteration order. The same
/// code resolves [`repeating_top_ngrams`]' ties; `pyroid.nemo` checks it against the
/// interpreter's own sets before it passes the key.
pub fn ngram_set_order(key: HashKey, words: &[&str], n: usize) -> Vec<usize> {
    if n == 0 || words.len() < n {
        return Vec::new();
    }
    with_scratch(|s| {
        tally_windows(words, n, s, |_, _| {});
        build_set(key, words, n, s);
        s.set.iter().map(|k| s.uniq[k as usize].1).collect()
    })
}

/// `RepeatingTopNGramsFilter(n)`: the share of characters the most frequent word n-gram
/// covers (`" ".join(ngram)` removed from `text`); 1.0 below `n` words.
///
/// When several n-grams share the top count, NeMo takes the first in `set(ngrams)` order:
/// with the interpreter's hash `key` that is resolved here, else the document comes back
/// [`TopNGram::Tied`]. `n < 1` gives no n-grams, and NeMo's `max()` raises ValueError.
pub fn repeating_top_ngrams(
    text: &str,
    n: i64,
    splitter: Splitter,
    key: Option<HashKey>,
) -> Result<TopNGram, ScoreError> {
    if n < 1 {
        return Err(ScoreError::EmptyMax);
    }
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    Ok(with_scratch(|s| {
        let mut spans = std::mem::take(&mut s.spans);
        word_spans(text, splitter, &mut spans);
        let words = Spans {
            text,
            spans: &spans,
        };
        let result = top_ngrams_of(text, &words, n, key, s);
        s.spans = spans;
        result
    }))
}

/// [`repeating_top_ngrams`] for the words of `text`.
fn top_ngrams_of(
    text: &str,
    words: &Spans<'_>,
    n: usize,
    key: Option<HashKey>,
    s: &mut Scratch,
) -> TopNGram {
    if words.count() < n {
        return TopNGram::Score(1.0);
    }
    tally_windows(words, n, s, |_, _| {});
    let score_at = |start: usize| {
        let g: Vec<&str> = (start..start + n).map(|j| words.word(j)).collect();
        TopNGram::Score(top_ngram_score(text, &g))
    };
    let top = s.uniq.iter().map(|&(c, _)| c).max().unwrap_or(0);
    let mut tied = s.uniq.iter().filter(|&&(c, _)| c == top);
    let first = tied.next().map_or(0, |&(_, s)| s);
    if tied.next().is_none() {
        return score_at(first);
    }
    if let Some(key) = key {
        // The first n-gram with the top count in set order.
        build_set(key, words, n, s);
        let start = s
            .set
            .iter()
            .map(|k| s.uniq[k as usize])
            .find(|&(count, _)| count == top)
            .map_or(first, |(_, start)| start);
        return score_at(start);
    }
    let uniq = &s.uniq;
    let starts: Vec<usize> = uniq
        .iter()
        .filter(|&&(c, _)| c == top)
        .map(|&(_, s)| s)
        .collect();
    let all = starts.len() == uniq.len();
    TopNGram::Tied(Tie {
        count: top,
        starts: (!all).then_some(starts),
    })
}

/// `RepeatingDuplicateNGramsFilter(n)`: the share of characters in n-grams that occurred
/// before (overlapping n-grams counted once, with the spaces between their words); 1.0
/// below `n` words. `n < 1` gives no n-grams: 0.0 (1.0 for `""`).
pub fn repeating_duplicate_ngrams(text: &str, n: i64, splitter: Splitter) -> f64 {
    let nchar = char_len(text);
    let score = |dup: usize| if nchar > 0 { ratio(dup, nchar) } else { 1.0 };
    if n < 1 {
        return score(0);
    }
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    with_scratch(|s| {
        let mut spans = std::mem::take(&mut s.spans);
        word_spans(text, splitter, &mut spans);
        let words = Spans {
            text,
            spans: &spans,
        };
        let result = if words.count() < n {
            1.0
        } else {
            let (mut dup, mut overlap) = (0usize, 0usize);
            tally_windows(&words, n, s, |i, repeat| {
                if repeat {
                    // The words not counted yet (each word is counted at most once), and the
                    // spaces before them.
                    let chars: usize = (i + overlap..i + n).map(|j| char_len(words.word(j))).sum();
                    dup += chars + (n - overlap).min(n - 1);
                    overlap = n;
                }
                overlap = overlap.saturating_sub(1);
            });
            score(dup)
        };
        s.spans = spans;
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemo::text::split;

    const WS: Splitter = Splitter::Whitespace;

    #[test]
    fn repeated() {
        assert_eq!(repeated_lines("a\nb\na\n \n"), Ok(2.0 / 3.0));
        assert_eq!(repeated_lines("a\n a"), Ok(1.0));
        assert_eq!(repeated_lines(" \n"), Err(ScoreError::ZeroDivision));
        assert_eq!(repeated_paragraphs("a\n\nb\n\na"), 2.0 / 3.0);
        assert_eq!(repeated_paragraphs(""), 1.0);
        assert_eq!(repeated_paragraphs("\n\n"), 0.5);
        assert_eq!(repeated_lines_by_char("ab\nc\nab"), Ok(3.0 / 5.0));
        assert_eq!(repeated_lines_by_char(""), Err(ScoreError::ZeroDivision));
        assert_eq!(repeated_paragraphs_by_char("é\n\nxy\n\né"), Ok(3.0 / 4.0));
        assert_eq!(
            repeated_paragraphs_by_char(""),
            Err(ScoreError::ZeroDivision)
        );
        assert_eq!(
            repeated_paragraphs_by_char("\n\n\n\n"),
            Err(ScoreError::ZeroDivision)
        );
        assert_eq!(repeated_paragraphs_by_char("\n\n\n"), Ok(1.0));
    }

    #[test]
    fn top_ngrams() {
        // "a b" twice, every other bigram once: unique top.
        let t = "a b a b c";
        assert_eq!(
            repeating_top_ngrams(t, 2, WS, None),
            Ok(TopNGram::Score(6.0 / 9.0))
        );
        // The joined n-gram is searched in the text: different spacing does not count,
        // occurrences inside other words do.
        assert_eq!(
            repeating_top_ngrams("x y\tx y xx yy", 2, WS, None),
            Ok(TopNGram::Score(9.0 / 13.0))
        );
        assert_eq!(
            repeating_top_ngrams("a b", 3, WS, None),
            Ok(TopNGram::Score(1.0))
        );
        assert_eq!(
            repeating_top_ngrams("", 1, WS, None),
            Ok(TopNGram::Score(1.0))
        );
        assert_eq!(
            repeating_top_ngrams("a", 0, WS, None),
            Err(ScoreError::EmptyMax)
        );
        assert_eq!(
            repeating_top_ngrams("", -3, WS, None),
            Err(ScoreError::EmptyMax)
        );
        assert_eq!(
            repeating_top_ngrams("a b c", 2, WS, None),
            Ok(TopNGram::Tied(Tie {
                count: 1,
                starts: None
            }))
        );
        assert_eq!(
            repeating_top_ngrams("a b a b c d c d e", 2, WS, None),
            Ok(TopNGram::Tied(Tie {
                count: 2,
                starts: Some(vec![0, 4])
            }))
        );
        assert_eq!(
            repeating_top_ngrams("x", 1, WS, None),
            Ok(TopNGram::Score(1.0))
        );
        assert_eq!(
            repeating_top_ngrams("x x y", 1, WS, None),
            Ok(TopNGram::Score(2.0 / 5.0))
        );
    }

    #[test]
    fn top_ngram_ties_with_seed_0() {
        // `PYTHONHASHSEED=0`: NeMo's RepeatingTopNGramsFilter(n).score_document(t) (the key is
        // then all zeros).
        let key = HashKey::from_secret(&[0; 16]);
        let w300: String = (0..300)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let cases: [(&str, [f64; 3]); 6] = [
            ("a b c", [0.2, 0.6, 1.0]),
            (
                "a b a b c d c d e",
                [
                    0.117_647_058_823_529_41,
                    0.352_941_176_470_588_26,
                    0.294_117_647_058_823_54,
                ],
            ),
            (
                "the cat the dog the cat the dog",
                [
                    0.387_096_774_193_548_4,
                    0.451_612_903_225_806_44,
                    0.709_677_419_354_838_7,
                ],
            ),
            (
                "x y z x y z q",
                [
                    0.153_846_153_846_153_85,
                    0.461_538_461_538_461_56,
                    0.769_230_769_230_769_3,
                ],
            ),
            (
                "中文 é 😀 中文 é 😀 a",
                [0.133_333_333_333_333_33, 0.4, 0.8],
            ),
            (
                &w300,
                [
                    0.002_879_769_618_430_525_4,
                    0.006_479_481_641_468_682_5,
                    0.007_919_366_450_683_946,
                ],
            ),
        ];
        for (text, want) in cases {
            for (n, &w) in (1..).zip(&want) {
                assert_eq!(
                    repeating_top_ngrams(text, n, WS, key),
                    Ok(TopNGram::Score(w)),
                    "{text:?} {n}"
                );
            }
        }
    }

    #[test]
    fn ngram_kernels_match_a_direct_implementation() {
        // NeMo's code, with a HashMap of word vectors, on documents from a small vocabulary
        // (many repeats, ties and probe chains).
        let vocab = [
            "a", "b", "ab", "é", "中文", "a b", "x", "#", "\u{3000}", "ab ",
        ];
        let mut state = 0x1234_5678_u64;
        let mut next = |m: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            usize::try_from(state >> 33).unwrap() % m
        };
        for _ in 0..400 {
            let len = next(200);
            let text: String = (0..len)
                .map(|_| vocab[next(vocab.len())])
                .collect::<Vec<_>>()
                .join(" ");
            let words: Vec<&str> = split(&text).collect();
            for n in 1..=6 {
                // RepeatingDuplicateNGramsFilter.
                let want = if words.len() < n {
                    1.0
                } else {
                    let mut counts: std::collections::HashMap<&[&str], usize> =
                        std::collections::HashMap::default();
                    let (mut dup, mut overlap) = (0usize, 0usize);
                    for i in 0..=words.len() - n {
                        let c = counts.entry(&words[i..i + n]).or_insert(0);
                        *c += 1;
                        if *c > 1 {
                            dup += words[i + overlap..i + n]
                                .iter()
                                .map(|w| w.chars().count())
                                .sum::<usize>()
                                + (n - overlap).min(n - 1);
                            overlap = n;
                        }
                        overlap = overlap.saturating_sub(1);
                    }
                    ratio(dup, text.chars().count())
                };
                let n64 = i64::try_from(n).unwrap();
                assert_eq!(
                    repeating_duplicate_ngrams(&text, n64, WS),
                    want,
                    "{text:?} {n}"
                );
                // RepeatingTopNGramsFilter: the counts and the tied first starts.
                if words.len() < n {
                    continue;
                }
                let mut counts: Vec<(&[&str], usize, usize)> = Vec::new();
                for i in 0..=words.len() - n {
                    match counts.iter_mut().find(|(g, _, _)| *g == &words[i..i + n]) {
                        Some(e) => e.1 += 1,
                        None => counts.push((&words[i..i + n], 1, i)),
                    }
                }
                let top = counts.iter().map(|e| e.1).max().unwrap();
                let starts: Vec<usize> =
                    counts.iter().filter(|e| e.1 == top).map(|e| e.2).collect();
                let got = repeating_top_ngrams(&text, n64, WS, None).unwrap();
                if let [s] = starts[..] {
                    assert_eq!(
                        got,
                        TopNGram::Score(top_ngram_score(&text, &words[s..s + n]))
                    );
                } else {
                    let all = starts.len() == counts.len();
                    assert_eq!(
                        got,
                        TopNGram::Tied(Tie {
                            count: top,
                            starts: (!all).then_some(starts)
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn duplicate_ngrams() {
        // Python: RepeatingDuplicateNGramsFilter(n).score_document(t).
        assert_eq!(repeating_duplicate_ngrams("a b a b", 2, WS), 3.0 / 7.0);
        assert_eq!(repeating_duplicate_ngrams("a b a b a b", 2, WS), 7.0 / 11.0);
        assert_eq!(
            repeating_duplicate_ngrams("ab cd ab cd ab", 2, WS),
            8.0 / 14.0
        );
        assert_eq!(repeating_duplicate_ngrams("x x x", 1, WS), 2.0 / 5.0);
        assert_eq!(repeating_duplicate_ngrams("a b c", 2, WS), 0.0);
        assert_eq!(repeating_duplicate_ngrams("a b", 3, WS), 1.0);
        assert_eq!(repeating_duplicate_ngrams("a", 0, WS), 0.0);
        assert_eq!(repeating_duplicate_ngrams("", 0, WS), 1.0);
        assert_eq!(repeating_duplicate_ngrams("", -1, WS), 1.0);
        assert_eq!(repeating_duplicate_ngrams("", 1, WS), 1.0);
    }
}
