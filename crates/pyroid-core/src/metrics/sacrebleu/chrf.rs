// Derived from sacrebleu 2.6.0 (Apache-2.0), © Amazon.com, Inc. or its affiliates; modified.
//! chrF and chrF++ statistics, as sacrebleu 2.6.0's `CHRF` computes them.
//!
//! Per segment (`CHRF._compute_segment_statistics`), the row
//! `[n_hyp, n_ref, n_match] * (char_order + word_order)` of the best
//! reference: character n-grams `1..=char_order` (over code points, without
//! whitespace unless `whitespace=True`), then word n-grams
//! `1..=word_order` (words split as `_remove_punctuation` does). `n_hyp` is 0
//! for an order the reference has no n-gram of. With several references the
//! one with the highest f-score wins, the first on ties: [`f_score`] is
//! `CHRF._compute_f_score` with the same operations in the same order, so
//! the choice is the same as in Python. The corpus score is computed from
//! the summed rows in Python.

use std::borrow::Borrow;
use std::hash::Hash;

use super::bleu::{num_refs, ref_segments, sum_rows};
use super::ngrams::{Grams, Vocab, windows};
use super::pystr;
use crate::par::{self, PoolError, Threads};

/// The chrF settings that change the statistics or the best reference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChrfConfig {
    /// `CHRF(char_order=...)`.
    pub char_order: usize,
    /// `CHRF(word_order=...)`; `char_order + word_order` is at least 1.
    pub word_order: usize,
    /// `CHRF(whitespace=...)`: keep whitespace in character n-grams.
    pub whitespace: bool,
    /// `float(beta ** 2)` (Python's value).
    pub factor: f64,
    /// `float(1 + beta ** 2)` (Python's value).
    pub one_plus_factor: f64,
    /// `CHRF(eps_smoothing=...)`.
    pub eps_smoothing: bool,
}

impl ChrfConfig {
    /// `CHRF.order`: the number of `[hyp, ref, match]` triples.
    pub fn order(&self) -> usize {
        self.char_order + self.word_order
    }

    /// Length of one statistics row: `3 * order`.
    pub fn row_len(&self) -> usize {
        3 * self.order()
    }
}

/// `CHRF._compute_f_score` over a row of `[n_hyp, n_ref, n_match]` triples.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::{ChrfConfig, chrf::f_score};
/// let cfg = ChrfConfig {
///     char_order: 1, word_order: 0, whitespace: false,
///     factor: 4.0, one_plus_factor: 5.0, eps_smoothing: false,
/// };
/// // precision 1/2, recall 1/4: 5 * 0.5 * 0.25 / (4 * 0.5 + 0.25) * 100
/// assert_eq!(f_score(&[2, 4, 1], &cfg), 100.0 * (5.0 * 0.5 * 0.25 / 2.25));
/// ```
#[allow(clippy::cast_precision_loss)] // n-gram counts are far below 2^53: exact, as in Python
#[allow(clippy::similar_names)] // sacrebleu's names (avg_prec, avg_rec)
pub fn f_score(stats: &[i64], cfg: &ChrfConfig) -> f64 {
    let eps = 1e-16;
    let mut score = 0.0;
    let mut effective_order = 0_i64;
    let factor = cfg.factor;
    let (mut avg_prec, mut avg_rec) = (0.0, 0.0);

    for t in stats.chunks_exact(3).take(cfg.order()) {
        let (n_hyp, n_ref, n_match) = (t[0], t[1], t[2]);

        // chrF++.py style EPS smoothing (also used by Moses and NLTK)
        let prec = if n_hyp > 0 {
            n_match as f64 / n_hyp as f64
        } else {
            eps
        };
        let rec = if n_ref > 0 {
            n_match as f64 / n_ref as f64
        } else {
            eps
        };

        let denom = factor * prec + rec;
        score += if denom > 0.0 {
            cfg.one_plus_factor * prec * rec / denom
        } else {
            eps
        };

        // sacreBLEU <2.0.0 style effective order smoothing
        if n_hyp > 0 && n_ref > 0 {
            avg_prec += prec;
            avg_rec += rec;
            effective_order += 1;
        }
    }

    if cfg.eps_smoothing {
        return 100.0 * score / cfg.order() as f64;
    }

    if effective_order == 0 {
        avg_prec = 0.0;
        avg_rec = 0.0;
    } else {
        avg_prec /= effective_order as f64;
        avg_rec /= effective_order as f64;
    }

    // `if avg_prec + avg_rec:` (NaN is truthy: NaN == 0.0 is false)
    if avg_prec + avg_rec == 0.0 {
        0.0
    } else {
        let mut score = cfg.one_plus_factor * avg_prec * avg_rec;
        score /= (factor * avg_prec) + avg_rec;
        100.0 * score
    }
}

/// `string.punctuation`, `CHRF._PUNCTS`.
const fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
}

/// `CHRF._remove_punctuation`: the words of `sent`, with one leading or
/// trailing ASCII punctuation character split off.
fn remove_punctuation<'a>(sent: &'a str, out: &mut Vec<&'a str>) {
    for w in pystr::split(sent) {
        let mut chars = w.chars();
        let first = chars.next().expect("split() yields non-empty words");
        let Some(last) = chars.next_back() else {
            // len(w) == 1
            out.push(w);
            continue;
        };
        if is_punct(last) {
            let cut = w.len() - 1;
            out.extend([&w[..cut], &w[cut..]]);
        } else if is_punct(first) {
            out.extend([&w[..1], &w[1..]]);
        } else {
            out.push(w);
        }
    }
}

/// The code points chrF takes character n-grams of.
fn char_ids(line: &str, whitespace: bool, out: &mut Vec<u32>) {
    out.clear();
    if whitespace {
        out.extend(line.chars().map(u32::from));
    } else {
        // ''.join(line.split())
        out.extend(line.chars().filter(|&c| !pystr::is_space(c)).map(u32::from));
    }
}

/// Code points are at most this (for packing character n-grams).
const MAX_CHAR: u32 = 0x10_FFFF;

/// The n-gram counts of one line: every character order in one map, every
/// word order in another.
#[derive(Clone, Debug)]
struct Side {
    chars: Grams,
    words: Grams,
    n_chars: usize,
    n_words: usize,
}

impl Side {
    fn new(chars: &[u32], words: &[u32], max_word_id: u32, cfg: &ChrfConfig) -> Self {
        Self {
            chars: Grams::count(chars, cfg.char_order, MAX_CHAR),
            words: Grams::count(words, cfg.word_order, max_word_id),
            n_chars: chars.len(),
            n_words: words.len(),
        }
    }

    /// `_get_match_statistics` for every order, hypothesis `self`;
    /// `matches` is scratch space.
    fn stats(&self, reference: &Self, cfg: &ChrfConfig, matches: &mut Vec<u64>, out: &mut [i64]) {
        let (chars, words) = out.split_at_mut(3 * cfg.char_order);
        matches.resize(cfg.char_order, 0);
        self.chars.overlap_by_order(&reference.chars, matches);
        fill(chars, self.n_chars, reference.n_chars, matches);
        matches.resize(cfg.word_order, 0);
        self.words.overlap_by_order(&reference.words, matches);
        fill(words, self.n_words, reference.n_words, matches);
    }
}

fn as_i64(n: u64) -> i64 {
    i64::try_from(n).expect("count fits in i64")
}

/// One `[n_hyp, n_ref, n_match]` triple per order `n = 1, 2, ...` of `out`.
fn fill(out: &mut [i64], hyp_len: usize, ref_len: usize, matches: &[u64]) {
    for (n, (t, &m)) in out.chunks_exact_mut(3).zip(matches).enumerate() {
        let n_ref = windows(ref_len, n + 1);
        // Don't count hits if no reference exists for that n-gram
        t[0] = if n_ref > 0 {
            as_i64(windows(hyp_len, n + 1))
        } else {
            0
        };
        t[1] = as_i64(n_ref);
        t[2] = as_i64(m);
    }
}

/// Keeps `stats` in `best` if its f-score beats `best_f` (`>`: the first
/// reference wins ties).
fn keep_best(stats: &[i64], cfg: &ChrfConfig, best_f: &mut f64, best: &mut [i64]) -> bool {
    let f = f_score(stats, cfg);
    if f > *best_f {
        *best_f = f;
        best.copy_from_slice(stats);
        true
    } else {
        false
    }
}

/// The references of one segment, counted (for [`ChrfRefCache`]).
#[derive(Clone, Debug)]
struct ChrfRefs {
    vocab: Vocab<Box<str>>,
    sides: Vec<Side>,
}

impl ChrfRefs {
    fn new(refs: &[&str], cfg: &ChrfConfig) -> Self {
        let (vocab, word_ids) = ref_word_ids(refs, cfg);
        let max_id = vocab.unknown();
        let mut chars = Vec::new();
        let sides = refs
            .iter()
            .zip(&word_ids)
            .map(|(r, w)| {
                char_ids(r, cfg.whitespace, &mut chars);
                Side::new(&chars, w, max_id, cfg)
            })
            .collect();
        Self {
            vocab: vocab.to_owned(),
            sides,
        }
    }
}

/// The vocabulary and word ids of every reference (all of them interned
/// first, so the key layout is fixed before any n-gram is counted).
fn ref_word_ids<'a>(refs: &[&'a str], cfg: &ChrfConfig) -> (Vocab<&'a str>, Vec<Vec<u32>>) {
    if cfg.word_order == 0 {
        return (Vocab::with_capacity(0), vec![Vec::new(); refs.len()]);
    }
    let words: Vec<Vec<&str>> = refs
        .iter()
        .map(|r| {
            let mut w = Vec::new();
            remove_punctuation(r, &mut w);
            w
        })
        .collect();
    let mut vocab = Vocab::with_capacity(words.iter().map(Vec::len).sum());
    let ids = words
        .iter()
        .map(|w| w.iter().map(|t| vocab.intern(*t)).collect())
        .collect();
    (vocab, ids)
}

/// The hypothesis' code points and word ids against a reference vocabulary.
fn hyp_ids<V: Hash + Eq + Borrow<str>>(
    hyp: &str,
    vocab: &Vocab<V>,
    cfg: &ChrfConfig,
) -> (Vec<u32>, Vec<u32>) {
    let mut words = Vec::new();
    if cfg.word_order > 0 {
        remove_punctuation(hyp, &mut words);
    }
    let ids = words.iter().map(|w| vocab.lookup(w)).collect();
    let mut chars = Vec::new();
    char_ids(hyp, cfg.whitespace, &mut chars);
    (chars, ids)
}

/// One segment's best-reference row (`None` references already removed);
/// `false` if no reference was kept (none given), when sacrebleu's row is
/// `[]`.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::{ChrfConfig, chrf::segment_stats};
/// let cfg = ChrfConfig {
///     char_order: 2, word_order: 0, whitespace: false,
///     factor: 4.0, one_plus_factor: 5.0, eps_smoothing: false,
/// };
/// let mut row = [0; 6];
/// assert!(segment_stats("ab c", &["abd", "abc"], &cfg, &mut row));
/// assert_eq!(row, [3, 3, 3, 2, 2, 2]); // "abc" wins
/// ```
pub fn segment_stats(hyp: &str, refs: &[&str], cfg: &ChrfConfig, row: &mut [i64]) -> bool {
    let (vocab, word_ids) = ref_word_ids(refs, cfg);
    let (hyp_chars, hyp_words) = hyp_ids(hyp, &vocab, cfg);
    let max_id = vocab.unknown();
    let mut ref_chars = Vec::new();
    let mut stats = vec![0; cfg.row_len()];
    let mut matches = Vec::new();
    let mut best_f = -1.0;
    let mut found = false;
    for (r, w) in refs.iter().zip(&word_ids) {
        // Count the reference, then use its counts up against the
        // hypothesis' n-grams: no map for the hypothesis.
        char_ids(r, cfg.whitespace, &mut ref_chars);
        let (chars, words) = stats.split_at_mut(3 * cfg.char_order);
        let mut grams = Grams::count(&ref_chars, cfg.char_order, MAX_CHAR);
        matches.clear();
        matches.extend((1..=cfg.char_order).map(|n| grams.take_matches(&hyp_chars, n)));
        fill(chars, hyp_chars.len(), ref_chars.len(), &matches);
        if cfg.word_order > 0 {
            let mut grams = Grams::count(w, cfg.word_order, max_id);
            matches.clear();
            matches.extend((1..=cfg.word_order).map(|n| grams.take_matches(&hyp_words, n)));
            fill(words, hyp_words.len(), w.len(), &matches);
        }
        found |= keep_best(&stats, cfg, &mut best_f, row);
    }
    found
}

/// Statistics of a corpus: one best-reference row per segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChrfStats {
    /// `n_segments` rows of [`ChrfConfig::row_len`] values (zeros where
    /// `found` is false).
    pub rows: Vec<i64>,
    /// Per segment: was a best reference found? (sacrebleu's row is `[]`
    /// otherwise.)
    pub found: Vec<bool>,
    /// Segments scored (`zip` of hypotheses and reference segments).
    pub n_segments: usize,
    /// `CHRF.num_refs` (see [`super::BleuStats::num_refs`]).
    pub num_refs: i64,
}

impl ChrfStats {
    /// The rows summed over the corpus (`sum_of_lists`), if every segment
    /// has a row.
    pub fn totals(&self, row_len: usize) -> Option<Vec<i64>> {
        self.found
            .iter()
            .all(|&f| f)
            .then(|| sum_rows(&self.rows, row_len))
    }
}

/// chrF statistics of `hyps` against reference `streams` (see
/// [`super::bleu::corpus_stats`]).
pub fn corpus_stats<S: AsRef<str> + Sync>(
    hyps: &[S],
    streams: &[Vec<Option<S>>],
    cfg: &ChrfConfig,
    threads: Threads,
) -> Result<ChrfStats, PoolError> {
    let n_ref_segments = ref_segments(streams);
    let n = hyps.len().min(n_ref_segments);
    let row_len = cfg.row_len();
    let mut rows = vec![0; n * row_len];
    let found = par::try_map_rows(&mut rows, row_len, threads, |i, row| {
        let refs: Vec<&str> = streams
            .iter()
            .filter_map(|s| s[i].as_ref().map(AsRef::as_ref))
            .collect();
        Ok::<_, PoolError>(segment_stats(hyps[i].as_ref(), &refs, cfg, row))
    })?;
    Ok(ChrfStats {
        rows,
        found,
        n_segments: n,
        num_refs: num_refs(streams, n_ref_segments),
    })
}

/// Pre-computed references (`CHRF(references=...)`).
#[derive(Clone, Debug)]
pub struct ChrfRefCache {
    segments: Vec<ChrfRefs>,
    char_order: usize,
    word_order: usize,
    whitespace: bool,
    num_refs: i64,
}

impl ChrfRefCache {
    /// Counts the references of every segment. Only the orders and
    /// `whitespace` of `cfg` matter here.
    pub fn new<S: AsRef<str> + Sync>(
        streams: &[Vec<Option<S>>],
        cfg: &ChrfConfig,
        threads: Threads,
    ) -> Result<Self, PoolError> {
        let n = ref_segments(streams);
        let segments = par::map(n, threads, |i| {
            let refs: Vec<&str> = streams
                .iter()
                .filter_map(|s| s[i].as_ref().map(AsRef::as_ref))
                .collect();
            ChrfRefs::new(&refs, cfg)
        })?;
        Ok(Self {
            segments,
            char_order: cfg.char_order,
            word_order: cfg.word_order,
            whitespace: cfg.whitespace,
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

    /// `CHRF.num_refs` for the cached references.
    pub fn num_refs(&self) -> i64 {
        self.num_refs
    }

    /// `(char_order, word_order, whitespace)` the references were counted with.
    pub fn settings(&self) -> (usize, usize, bool) {
        (self.char_order, self.word_order, self.whitespace)
    }

    /// [`corpus_stats`] against the cached references. `cfg`'s orders must
    /// be the cache's; `beta`, `eps_smoothing` and the hypotheses'
    /// `whitespace` come from `cfg`.
    pub fn corpus_stats<S: AsRef<str> + Sync>(
        &self,
        hyps: &[S],
        cfg: &ChrfConfig,
        threads: Threads,
    ) -> Result<ChrfStats, PoolError> {
        assert_eq!(
            (cfg.char_order, cfg.word_order),
            (self.char_order, self.word_order),
            "chrF orders differ from the cached references'"
        );
        let n = hyps.len().min(self.segments.len());
        let row_len = cfg.row_len();
        let mut rows = vec![0; n * row_len];
        let found = par::try_map_rows(&mut rows, row_len, threads, |i, row| {
            let seg = &self.segments[i];
            let (chars, words) = hyp_ids(hyps[i].as_ref(), &seg.vocab, cfg);
            let h = Side::new(&chars, &words, seg.vocab.unknown(), cfg);
            let mut stats = vec![0; row_len];
            let mut matches = Vec::new();
            let mut best_f = -1.0;
            let mut found = false;
            for side in &seg.sides {
                h.stats(side, cfg, &mut matches, &mut stats);
                found |= keep_best(&stats, cfg, &mut best_f, row);
            }
            Ok::<_, PoolError>(found)
        })?;
        Ok(ChrfStats {
            rows,
            found,
            n_segments: n,
            num_refs: self.num_refs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHRF: ChrfConfig = ChrfConfig {
        char_order: 6,
        word_order: 0,
        whitespace: false,
        factor: 4.0,
        one_plus_factor: 5.0,
        eps_smoothing: false,
    };

    const CHRF_PP: ChrfConfig = ChrfConfig {
        word_order: 2,
        ..CHRF
    };

    fn words(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        remove_punctuation(s, &mut out);
        out
    }

    #[test]
    fn remove_punctuation_matches_sacrebleu() {
        assert_eq!(words("(hi) a, b. c"), ["(hi", ")", "a", ",", "b", ".", "c"]);
        assert_eq!(words("'x 'y' . ,,"), ["'", "x", "'y", "'", ".", ",", ","]);
        assert_eq!(words("é! ¿qué"), ["é", "!", "¿qué"]);
    }

    #[test]
    fn hypothesis_n_gram_counts_need_reference_n_grams() {
        let mut row = vec![0; CHRF.row_len()];
        assert!(segment_stats("abcdefg", &["abc"], &CHRF, &mut row));
        // orders 4..6: the reference has no n-grams, so n_hyp is 0 too.
        assert_eq!(row, [7, 3, 3, 6, 2, 2, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn best_reference_first_wins_ties() {
        let mut row = vec![0; CHRF_PP.row_len()];
        assert!(segment_stats(
            "a b",
            &["a c", "a d", "a b"],
            &CHRF_PP,
            &mut row
        ));
        let mut want = vec![0; CHRF_PP.row_len()];
        segment_stats("a b", &["a b"], &CHRF_PP, &mut want);
        assert_eq!(row, want);
        let mut tie = vec![0; CHRF_PP.row_len()];
        segment_stats("a b", &["a c", "a d"], &CHRF_PP, &mut tie);
        let mut first = vec![0; CHRF_PP.row_len()];
        segment_stats("a b", &["a c"], &CHRF_PP, &mut first);
        assert_eq!(tie, first);
        assert!(!segment_stats("a b", &[], &CHRF_PP, &mut first));
    }

    #[test]
    fn cache_matches_direct() {
        let hyps = vec!["The cat, sat.", "dog", "x y z"];
        let streams = vec![
            vec![Some("The cat sat."), Some("a dog"), None],
            vec![Some("A cat sat!"), None, None],
        ];
        for cfg in [
            CHRF,
            CHRF_PP,
            ChrfConfig {
                whitespace: true,
                ..CHRF_PP
            },
        ] {
            let direct = corpus_stats(&hyps, &streams, &cfg, Threads::ONE).unwrap();
            let cache = ChrfRefCache::new(&streams, &cfg, Threads::ONE).unwrap();
            assert_eq!(
                cache.corpus_stats(&hyps, &cfg, Threads::ONE).unwrap(),
                direct
            );
            assert_eq!(direct.found, [true, true, false]);
            assert_eq!(direct.num_refs, -1);
            assert_eq!(direct.totals(cfg.row_len()), None);
        }
    }

    #[test]
    fn f_score_eps_smoothing() {
        let cfg = ChrfConfig {
            eps_smoothing: true,
            char_order: 2,
            ..CHRF
        };
        let f = f_score(&[2, 2, 2, 0, 0, 0], &cfg);
        // order 1: prec = rec = 1 -> 5 * 1 * 1 / 5 = 1; order 2: eps
        let tiny = 5.0 * 1e-16 * 1e-16 / (4.0 * 1e-16 + 1e-16);
        assert_eq!(f, 100.0 * (1.0 + tiny) / 2.0);
    }
}
