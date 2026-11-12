// Derived from sacrebleu 2.6.0 (Apache-2.0), © Amazon.com, Inc. or its affiliates; modified.
//! BLEU's sufficient statistics, as sacrebleu 2.6.0's `BLEU` computes them.
//!
//! Per segment (`BLEU._compute_segment_statistics`), the row
//! `[hyp_len, ref_len, correct_1..correct_N, total_1..total_N]` where `N` is
//! `max_ngram_order`: `ref_len` is the closest reference length (ties go to
//! the shorter one; -1 for a segment without references), `total_n` the
//! hypothesis' n-gram count and `correct_n` its n-grams clipped by the
//! maximum count over the references. The score itself (`compute_bleu`) is
//! computed from the summed rows in Python.
//!
//! Each line is preprocessed as `_preprocess_segment` does after the
//! caller's optional lowercasing: `tokenizer(line.rstrip())`, then
//! `str.split()`.

use std::borrow::Borrow;
use std::hash::Hash;

use super::ngrams::{Grams, Vocab, windows};
use super::pystr;
use super::tokenizers::Tokenizer;
use crate::par::{self, PoolError, Threads};

/// The BLEU settings that change the statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BleuConfig {
    /// `BLEU(tokenize=...)`.
    pub tokenizer: Tokenizer,
    /// `BLEU(max_ngram_order=...)`, at least 1.
    pub max_order: usize,
}

impl BleuConfig {
    /// Length of one statistics row: `2 + 2 * max_order`.
    pub fn row_len(&self) -> usize {
        2 + 2 * self.max_order
    }
}

/// `usize` -> `i64` for counts (always far below `i64::MAX`).
pub(super) fn count(n: usize) -> i64 {
    i64::try_from(n).expect("count fits in i64")
}

/// The references of one segment (`BLEU._extract_reference_info`): their
/// vocabulary, merged n-gram counts and lengths.
#[derive(Clone, Debug)]
pub struct BleuRefs<V: Hash + Eq> {
    vocab: Vocab<V>,
    grams: Grams,
    lens: Vec<usize>,
}

impl<'t> BleuRefs<&'t str> {
    /// From the tokenized references (`None` ones already removed).
    fn new(refs: &'t [super::tokenizers::Tokens<'_>], max_order: usize) -> Self {
        let words: Vec<Vec<&'t str>> = refs
            .iter()
            .map(|r| {
                let mut w = Vec::new();
                r.push_to(&mut w);
                w
            })
            .collect();
        let mut vocab = Vocab::with_capacity(words.iter().map(Vec::len).sum());
        let ids: Vec<Vec<u32>> = words
            .iter()
            .map(|w| w.iter().map(|t| vocab.intern(*t)).collect())
            .collect();
        let max_id = vocab.unknown();
        let mut grams = Grams::new(max_order, max_id);
        for (i, r) in ids.iter().enumerate() {
            let one = Grams::count(r, max_order, max_id);
            if i == 0 {
                grams = one;
            } else {
                grams.merge_max(one);
            }
        }
        let lens = ids.iter().map(Vec::len).collect();
        Self { vocab, grams, lens }
    }

    fn into_owned(self) -> BleuRefs<Box<str>> {
        BleuRefs {
            vocab: self.vocab.to_owned(),
            grams: self.grams,
            lens: self.lens,
        }
    }
}

impl<V: Hash + Eq + Borrow<str>> BleuRefs<V> {
    /// Fills `row` for `hyp` (its tokens) except the matches; returns the
    /// hypothesis' ids, or `None` if sacrebleu would raise (`TypeError`: a
    /// segment without references, whose `ref_ngrams` is `None`, and a
    /// hypothesis with n-grams) or there is nothing to match.
    fn lengths(&self, hyp: &[&str], max_order: usize, row: &mut [i64]) -> (Vec<u32>, bool) {
        let ids: Vec<u32> = hyp.iter().map(|w| self.vocab.lookup(w)).collect();
        row[0] = count(ids.len());
        row[1] = closest_ref_len(ids.len(), &self.lens);
        let (correct, total) = row[2..].split_at_mut(max_order);
        correct.fill(0);
        for (n, t) in total.iter_mut().enumerate() {
            *t = i64::try_from(windows(ids.len(), n + 1)).expect("count fits in i64");
        }
        let raises = self.lens.is_empty() && !ids.is_empty();
        (ids, raises)
    }

    /// Fills `row` for `hyp` against counts that stay as they are (the cache).
    fn score(&self, hyp: &[&str], max_order: usize, row: &mut [i64]) -> bool {
        let (ids, raises) = self.lengths(hyp, max_order, row);
        if self.lens.is_empty() || ids.is_empty() {
            return raises;
        }
        let mut matches = vec![0; max_order];
        Grams::count(&ids, max_order, self.vocab.unknown())
            .overlap_by_order(&self.grams, &mut matches);
        for (c, m) in row[2..2 + max_order].iter_mut().zip(matches) {
            *c = i64::try_from(m).expect("count fits in i64");
        }
        false
    }

    /// [`Self::score`], using up the reference counts (one segment, no cache).
    fn score_once(mut self, hyp: &[&str], max_order: usize, row: &mut [i64]) -> bool {
        let (ids, raises) = self.lengths(hyp, max_order, row);
        if self.lens.is_empty() {
            return raises;
        }
        for (n, c) in row[2..2 + max_order].iter_mut().enumerate() {
            *c = i64::try_from(self.grams.take_matches(&ids, n + 1)).expect("count fits in i64");
        }
        false
    }
}

/// `BLEU._get_closest_ref_len`: the reference length closest to `hyp_len`,
/// the shorter one on ties, -1 without references.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::bleu::closest_ref_len;
/// assert_eq!(closest_ref_len(5, &[3, 7, 6]), 6);
/// assert_eq!(closest_ref_len(5, &[6, 4]), 4);
/// assert_eq!(closest_ref_len(5, &[]), -1);
/// ```
pub fn closest_ref_len(hyp_len: usize, ref_lens: &[usize]) -> i64 {
    let (mut closest_diff, mut closest_len) = (-1_i64, -1_i64);
    let hyp_len = count(hyp_len);
    for &r in ref_lens {
        let r = count(r);
        let diff = (hyp_len - r).abs();
        if closest_diff == -1 || diff < closest_diff {
            closest_diff = diff;
            closest_len = r;
        } else if diff == closest_diff && r < closest_len {
            closest_len = r;
        }
    }
    closest_len
}

/// The tokens BLEU counts for a line.
fn tokenize(tokenizer: Tokenizer, line: &str) -> super::tokenizers::Tokens<'_> {
    tokenizer.tokens(pystr::rstrip(line))
}

/// One segment's statistics row against its references (`None` ones
/// already removed); `true` if sacrebleu would raise (see [`BleuStats`]).
///
/// ```
/// use pyroid_core::metrics::sacrebleu::{BleuConfig, Tokenizer, bleu::segment_stats};
/// let cfg = BleuConfig { tokenizer: Tokenizer::Thirteen, max_order: 4 };
/// let mut row = [0; 10];
/// assert!(!segment_stats("The cat sat.", &["The cat sat down."], cfg, &mut row));
/// assert_eq!(row, [4, 5, 4, 2, 1, 0, 4, 3, 2, 1]);
/// ```
pub fn segment_stats(hyp: &str, refs: &[&str], cfg: BleuConfig, row: &mut [i64]) -> bool {
    let ref_toks: Vec<_> = refs.iter().map(|r| tokenize(cfg.tokenizer, r)).collect();
    let side = BleuRefs::new(&ref_toks, cfg.max_order);
    let hyp_toks = tokenize(cfg.tokenizer, hyp);
    let mut words = Vec::new();
    hyp_toks.push_to(&mut words);
    side.score_once(&words, cfg.max_order, row)
}

/// Statistics of a corpus: one row per segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BleuStats {
    /// `n_segments` rows of [`BleuConfig::row_len`] values.
    pub rows: Vec<i64>,
    /// Segments scored: `min(len(hypotheses), len of the shortest
    /// reference stream)` (sacrebleu zips them).
    pub n_segments: usize,
    /// `BLEU.num_refs` after caching these references: the number of
    /// non-`None` references if every segment has the same, else -1.
    pub num_refs: i64,
    /// Scored hypotheses ending in `" ."` (counted only when asked).
    pub tokenized_periods: usize,
    /// The first segment for which sacrebleu raises a `TypeError` (no
    /// references, and a non-empty hypothesis).
    pub type_error_at: Option<usize>,
}

impl BleuStats {
    /// The rows summed over the corpus (`sum_of_lists`).
    pub fn totals(&self, row_len: usize) -> Vec<i64> {
        sum_rows(&self.rows, row_len)
    }
}

/// Column sums of `rows` (`row_len` values each).
pub fn sum_rows(rows: &[i64], row_len: usize) -> Vec<i64> {
    let mut out = vec![0; row_len];
    for row in rows.chunks_exact(row_len) {
        for (o, v) in out.iter_mut().zip(row) {
            *o += v;
        }
    }
    out
}

/// `BLEU.num_refs` for these reference streams (`Metric._cache_references`).
pub(super) fn num_refs<S>(streams: &[Vec<Option<S>>], n_ref_segments: usize) -> i64 {
    let mut seen: Option<usize> = None;
    for i in 0..n_ref_segments {
        let k = streams.iter().filter(|s| s[i].is_some()).count();
        match seen {
            None => seen = Some(k),
            Some(prev) if prev != k => return -1,
            Some(_) => {}
        }
    }
    seen.map_or(-1, count)
}

/// Number of segments sacrebleu builds from `streams` (`zip(*references)`).
pub(super) fn ref_segments<S>(streams: &[Vec<Option<S>>]) -> usize {
    streams.iter().map(Vec::len).min().unwrap_or(0)
}

/// Hypotheses (of the first `n`) ending in `" ."`.
pub(super) fn tokenized_periods<S: AsRef<str>>(hyps: &[S], n: usize) -> usize {
    hyps[..n]
        .iter()
        .filter(|h| h.as_ref().ends_with(" ."))
        .count()
}

fn first_error(flags: &[bool]) -> Option<usize> {
    flags.iter().position(|&f| f)
}

/// BLEU statistics of `hyps` against reference `streams` (`streams[k][i]`
/// is the k-th reference of segment i; `None` is a missing one), as
/// `BLEU._extract_corpus_statistics` computes them. `check_tokenized`
/// counts hypotheses ending in `" ."` (`force=False`).
pub fn corpus_stats<S: AsRef<str> + Sync>(
    hyps: &[S],
    streams: &[Vec<Option<S>>],
    cfg: BleuConfig,
    check_tokenized: bool,
    threads: Threads,
) -> Result<BleuStats, PoolError> {
    let n_ref_segments = ref_segments(streams);
    let n = hyps.len().min(n_ref_segments);
    let row_len = cfg.row_len();
    let mut rows = vec![0; n * row_len];
    let flags = par::try_map_rows(&mut rows, row_len, threads, |i, row| {
        let refs: Vec<&str> = streams
            .iter()
            .filter_map(|s| s[i].as_ref().map(AsRef::as_ref))
            .collect();
        Ok::<_, PoolError>(segment_stats(hyps[i].as_ref(), &refs, cfg, row))
    })?;
    Ok(BleuStats {
        rows,
        n_segments: n,
        num_refs: num_refs(streams, n_ref_segments),
        tokenized_periods: if check_tokenized {
            tokenized_periods(hyps, n)
        } else {
            0
        },
        type_error_at: first_error(&flags),
    })
}

/// Pre-computed references (`BLEU(references=...)`): per segment, the
/// reference vocabulary, merged n-gram counts and lengths.
#[derive(Clone, Debug)]
pub struct BleuRefCache {
    segments: Vec<BleuRefs<Box<str>>>,
    max_order: usize,
    num_refs: i64,
}

impl BleuRefCache {
    /// Tokenizes and counts the references of every segment.
    pub fn new<S: AsRef<str> + Sync>(
        streams: &[Vec<Option<S>>],
        cfg: BleuConfig,
        threads: Threads,
    ) -> Result<Self, PoolError> {
        let n = ref_segments(streams);
        let segments = par::map(n, threads, |i| {
            let toks: Vec<_> = streams
                .iter()
                .filter_map(|s| s[i].as_ref())
                .map(|r| tokenize(cfg.tokenizer, r.as_ref()))
                .collect();
            BleuRefs::new(&toks, cfg.max_order).into_owned()
        })?;
        Ok(Self {
            segments,
            max_order: cfg.max_order,
            num_refs: num_refs(streams, n),
        })
    }

    /// Number of cached segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// No segments?
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// `BLEU.num_refs` for the cached references.
    pub fn num_refs(&self) -> i64 {
        self.num_refs
    }

    /// The `max_ngram_order` the references were counted with.
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// [`corpus_stats`] against the cached references; hypotheses are
    /// tokenized with `tokenizer`.
    pub fn corpus_stats<S: AsRef<str> + Sync>(
        &self,
        hyps: &[S],
        tokenizer: Tokenizer,
        check_tokenized: bool,
        threads: Threads,
    ) -> Result<BleuStats, PoolError> {
        let n = hyps.len().min(self.segments.len());
        let row_len = 2 + 2 * self.max_order;
        let mut rows = vec![0; n * row_len];
        let flags = par::try_map_rows(&mut rows, row_len, threads, |i, row| {
            let toks = tokenize(tokenizer, hyps[i].as_ref());
            let mut words = Vec::new();
            toks.push_to(&mut words);
            Ok::<_, PoolError>(self.segments[i].score(&words, self.max_order, row))
        })?;
        Ok(BleuStats {
            rows,
            n_segments: n,
            num_refs: self.num_refs,
            tokenized_periods: if check_tokenized {
                tokenized_periods(hyps, n)
            } else {
                0
            },
            type_error_at: first_error(&flags),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: BleuConfig = BleuConfig {
        tokenizer: Tokenizer::Thirteen,
        max_order: 4,
    };

    fn row(hyp: &str, refs: &[&str]) -> Vec<i64> {
        let mut r = vec![0; CFG.row_len()];
        assert!(!segment_stats(hyp, refs, CFG, &mut r));
        r
    }

    #[test]
    fn multi_reference_clips_by_the_max_count() {
        // "the" x3 in hyp; ref 1 has it twice, ref 2 once: clipped at 2.
        let r = row("the the the cat", &["the the dog", "the cat"]);
        assert_eq!(r[0], 4);
        assert_eq!(r[1], 3); // |4-3| = 1 = |4-2| - 1 -> 3 is closest
        assert_eq!(&r[2..6], [3, 2, 0, 0]); // 1-grams: the x2 + cat; 2-grams: "the the", "the cat"
        assert_eq!(&r[6..], [4, 3, 2, 1]);
    }

    #[test]
    fn segment_without_references() {
        let mut r = vec![0; CFG.row_len()];
        assert!(segment_stats("a b", &[], CFG, &mut r));
        assert!(!segment_stats("  ", &[], CFG, &mut r));
        assert_eq!(r, [0, -1, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn corpus_zips_and_counts_refs() {
        let hyps = vec!["a b .", "c d .", "e"];
        let streams = vec![
            vec![Some("a b"), Some("c d"), Some("e"), Some("unused")],
            vec![Some("a"), None, Some("x")],
        ];
        let stats = corpus_stats(&hyps, &streams, CFG, true, Threads::ONE).unwrap();
        assert_eq!(stats.n_segments, 3);
        assert_eq!(stats.num_refs, -1); // 2, 1, 2 references
        assert_eq!(stats.tokenized_periods, 2);
        assert_eq!(stats.type_error_at, None);
        let cache = BleuRefCache::new(&streams, CFG, Threads::ONE).unwrap();
        assert_eq!(cache.len(), 3);
        let cached = cache
            .corpus_stats(&hyps, CFG.tokenizer, true, Threads::ONE)
            .unwrap();
        assert_eq!(cached, stats);
        assert_eq!(
            stats.totals(CFG.row_len()),
            sum_rows(&stats.rows, CFG.row_len())
        );
    }

    #[test]
    fn long_orders_use_boxed_keys() {
        let cfg = BleuConfig {
            tokenizer: Tokenizer::None,
            max_order: 40,
        };
        let words: Vec<String> = (0..2_000).map(|i| format!("w{}", i % 900)).collect();
        let line = words.join(" ");
        let mut r = vec![0; cfg.row_len()];
        segment_stats(&line, &[&line], cfg, &mut r);
        assert_eq!(r[2], 2_000);
        assert_eq!(r[2 + 39], 2_000 - 39);
    }
}
