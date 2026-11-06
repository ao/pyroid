// Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
//! rouge-score's scoring functions (`_score_ngrams`, `_score_lcs`,
//! `_summary_level_lcs`) over interned token ids.
//!
//! Tokens are interned per target/prediction pair to dense `u32` ids, so
//! counting is array indexing and n-grams of every order are interned as
//! `(id of the (n-1)-gram, id of the last token)` pairs. LCS lengths use the
//! bit-parallel algorithm of Hyyrö (2004): one `u64` word per 64 tokens of
//! one side, updated per token of the other. rougeLsum needs the LCS
//! *indices* exactly as rouge-score's `_backtrack_norec` picks them (its
//! tie-breaking decides which tokens are in the union), so it keeps every row
//! of that bit table (`m × n / 64` words instead of `m × n` integers) and
//! walks back through it as `_backtrack_norec` walks its table.

use std::borrow::Cow;

use rustc_hash::FxHashMap;

use super::{Score, porter};

/// `scoring.fmeasure(precision, recall)`, in Python's operation order.
pub fn fmeasure(precision: f64, recall: f64) -> f64 {
    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// `a / b` for Python ints (exact below 2^53, as Python's true division).
#[allow(clippy::cast_precision_loss)]
fn div(a: usize, b: usize) -> f64 {
    a as f64 / b as f64
}

fn float_score(precision: f64, recall: f64) -> Score {
    Score::Float {
        precision,
        recall,
        fmeasure: fmeasure(precision, recall),
    }
}

fn to_u32(n: usize) -> u32 {
    u32::try_from(n).expect("fewer than 2^32 distinct tokens per pair")
}

/// Token -> dense id, for the documents of one target/prediction pair.
/// With stemming, tokens longer than 3 bytes are stemmed once per distinct
/// token, and a stem shares its id with an equal (unstemmed) short token.
#[derive(Debug)]
pub(crate) struct Interner<'a> {
    stem: bool,
    ids: FxHashMap<Cow<'a, [u8]>, u32>,
    /// Raw token -> final id, for tokens that go through the stemmer.
    stemmed: FxHashMap<&'a [u8], u32>,
    /// The stemmer's output buffer.
    buf: Vec<u8>,
}

impl<'a> Interner<'a> {
    pub(crate) fn new(stem: bool) -> Self {
        Self {
            stem,
            ids: FxHashMap::default(),
            stemmed: FxHashMap::default(),
            buf: Vec::new(),
        }
    }

    /// Number of distinct (final) tokens so far.
    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn id(&mut self, token: &'a [u8]) -> u32 {
        if !self.stem || token.len() <= 3 {
            return self.final_id(Cow::Borrowed(token));
        }
        if let Some(&id) = self.stemmed.get(token) {
            return id;
        }
        porter::stem_into(token, &mut self.buf);
        let id = if let Some(&id) = self.ids.get(self.buf.as_slice()) {
            id
        } else {
            let id = to_u32(self.ids.len());
            let key = if self.buf == token {
                Cow::Borrowed(token)
            } else {
                Cow::Owned(self.buf.clone())
            };
            self.ids.insert(key, id);
            id
        };
        self.stemmed.insert(token, id);
        id
    }

    fn final_id(&mut self, token: Cow<'a, [u8]>) -> u32 {
        if let Some(&id) = self.ids.get(&*token) {
            return id;
        }
        let id = to_u32(self.ids.len());
        self.ids.insert(token, id);
        id
    }
}

/// Reusable buffers for scoring one pair.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    counts: Vec<u32>,
    pairs: FxHashMap<u64, u32>,
    masks: Masks,
    rows: Vec<u64>,
    row_of: Vec<u32>,
    marks: Vec<bool>,
    cnt_r: Vec<u32>,
    cnt_c: Vec<u32>,
}

/// The n-gram ids of both sides of a pair, of one order at a time.
#[derive(Debug)]
pub(crate) struct Ngrams<'s> {
    tokens: (&'s [u32], &'s [u32]),
    order: usize,
    ids: (Vec<u32>, Vec<u32>),
    vocab: usize,
}

impl<'s> Ngrams<'s> {
    /// Unigrams of `target` and `prediction` (ids below `vocab`).
    pub(crate) fn new(target: &'s [u32], prediction: &'s [u32], vocab: usize) -> Self {
        Self {
            tokens: (target, prediction),
            order: 1,
            ids: (target.to_vec(), prediction.to_vec()),
            vocab,
        }
    }

    /// Advance to n-grams of order `n` (at least the current order).
    fn advance_to(&mut self, n: usize, pairs: &mut FxHashMap<u64, u32>) {
        while self.order < n {
            self.order += 1;
            let k = self.order;
            pairs.clear();
            let mut next = |prev: &[u32], tokens: &[u32]| -> Vec<u32> {
                if tokens.len() < k {
                    return Vec::new();
                }
                (0..=tokens.len() - k)
                    .map(|i| {
                        let key = (u64::from(prev[i]) << 32) | u64::from(tokens[i + k - 1]);
                        let fresh = to_u32(pairs.len());
                        *pairs.entry(key).or_insert(fresh)
                    })
                    .collect()
            };
            let t = next(&self.ids.0, self.tokens.0);
            let p = next(&self.ids.1, self.tokens.1);
            self.ids = (t, p);
            self.vocab = pairs.len();
        }
    }

    /// `_score_ngrams(_create_ngrams(target, n), _create_ngrams(prediction, n))`.
    pub(crate) fn score(&mut self, n: usize, scratch: &mut Scratch) -> Score {
        debug_assert!(n >= self.order, "orders are scored in increasing order");
        self.advance_to(n, &mut scratch.pairs);
        let (t, p) = &self.ids;
        let counts = &mut scratch.counts;
        counts.clear();
        counts.resize(self.vocab, 0);
        for &x in t {
            counts[x as usize] += 1;
        }
        // sum(min(target[g], prediction[g])): each prediction n-gram takes one
        // target occurrence while any are left.
        let mut overlap = 0;
        for &x in p {
            let c = &mut counts[x as usize];
            if *c > 0 {
                *c -= 1;
                overlap += 1;
            }
        }
        float_score(div(overlap, p.len().max(1)), div(overlap, t.len().max(1)))
    }
}

/// `_score_ngrams(_create_ngrams(target, n), _create_ngrams(prediction, n))`
/// for one order `n >= 1`.
pub(crate) fn rouge_n(
    target: &[u32],
    prediction: &[u32],
    n: usize,
    vocab: usize,
    scratch: &mut Scratch,
) -> Score {
    Ngrams::new(target, prediction, vocab).score(n, scratch)
}

/// The positions of every symbol of one sequence, grouped by symbol
/// (counting sort): the match masks of the bit-parallel LCS, kept sparse.
#[derive(Debug, Default)]
struct Masks {
    /// Symbol id -> slot (`NONE`: absent). Sized to the vocabulary; only the
    /// entries of the current sequence are set, and [`Masks::clear`] resets them.
    slot: Vec<u32>,
    /// Positions of slot `k`: `pos[start[k]..start[k + 1]]`, increasing.
    start: Vec<u32>,
    pos: Vec<u32>,
    /// Symbols with a slot, to reset.
    symbols: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl Masks {
    fn build(&mut self, seq: &[u32], vocab: usize) {
        if self.slot.len() < vocab {
            self.slot.resize(vocab, NONE);
        }
        self.start.clear();
        self.symbols.clear();
        // Count per slot (slots in order of first appearance).
        for &s in seq {
            let slot = &mut self.slot[s as usize];
            if *slot == NONE {
                *slot = to_u32(self.symbols.len());
                self.symbols.push(s);
                self.start.push(0);
            }
            self.start[*slot as usize] += 1;
        }
        // Exclusive prefix sums, then fill (positions stay increasing).
        let mut acc = 0;
        for c in &mut self.start {
            let n = *c;
            *c = acc;
            acc += n;
        }
        self.start.push(acc);
        self.pos.clear();
        self.pos.resize(seq.len(), 0);
        let mut fill: Vec<u32> = self.start[..self.symbols.len()].to_vec();
        for (j, &s) in seq.iter().enumerate() {
            let k = self.slot[s as usize] as usize;
            self.pos[fill[k] as usize] = to_u32(j);
            fill[k] += 1;
        }
    }

    /// Positions of symbol `s` (empty if absent).
    fn positions(&self, s: u32) -> &[u32] {
        match self.slot.get(s as usize) {
            Some(&k) if k != NONE => {
                let k = k as usize;
                &self.pos[self.start[k] as usize..self.start[k + 1] as usize]
            }
            _ => &[],
        }
    }

    fn clear(&mut self) {
        for &s in &self.symbols {
            self.slot[s as usize] = NONE;
        }
        self.symbols.clear();
    }
}

/// One row of the bit-parallel LCS: `v = (v + u) | (v & !u)` with
/// `u = v & match_mask`, the mask given by its set positions (sorted).
/// Words without a match only change through an incoming carry.
#[allow(clippy::many_single_char_names)] // the algorithm's names: v, u, x, w, k
fn advance(v: &mut [u64], positions: &[u32]) {
    let mut carry = false;
    let mut k = 0; // first word not yet updated
    let mut idx = 0;
    while idx < positions.len() {
        let w = (positions[idx] >> 6) as usize;
        let mut bits = 0u64;
        while idx < positions.len() && (positions[idx] >> 6) as usize == w {
            bits |= 1 << (positions[idx] & 63);
            idx += 1;
        }
        while carry && k < w {
            carry = add_carry(&mut v[k]);
            k += 1;
        }
        let x = v[w];
        let u = x & bits;
        let (s1, c1) = x.overflowing_add(u);
        let (s2, c2) = s1.overflowing_add(u64::from(carry));
        v[w] = s2 | (x & !u);
        carry = c1 || c2;
        k = w + 1;
    }
    while carry && k < v.len() {
        carry = add_carry(&mut v[k]);
        k += 1;
    }
}

/// A word with no match and an incoming carry: `x = (x + 1) | x`; the carry
/// goes on iff `x` was all ones (which then stays all ones).
fn add_carry(x: &mut u64) -> bool {
    if *x == u64::MAX {
        true
    } else {
        *x |= *x + 1;
        false
    }
}

/// Zero bits among the first `n` bits of `v`: `table[i][n]` for the row `v`.
fn zeros(v: &[u64], n: usize) -> usize {
    let full = n / 64;
    let mut ones: usize = v[..full].iter().map(|w| w.count_ones() as usize).sum();
    let rest = n % 64;
    if rest > 0 {
        ones += (v[full] & ((1u64 << rest) - 1)).count_ones() as usize;
    }
    n - ones
}

/// Length of the longest common subsequence of `a` and `b`.
fn lcs_len(a: &[u32], b: &[u32], vocab: usize, scratch: &mut Scratch) -> usize {
    // Bits over the longer side, one update per token of the shorter.
    let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
    if a.is_empty() {
        return 0;
    }
    let masks = &mut scratch.masks;
    masks.build(b, vocab);
    let v = &mut scratch.rows;
    v.clear();
    v.resize(b.len().div_ceil(64), u64::MAX);
    for &s in a {
        let p = masks.positions(s);
        if !p.is_empty() {
            advance(v, p);
        }
    }
    masks.clear();
    zeros(v, b.len())
}

/// `_score_lcs(target, prediction)`.
pub(crate) fn rouge_l(
    target: &[u32],
    prediction: &[u32],
    vocab: usize,
    scratch: &mut Scratch,
) -> Score {
    if target.is_empty() || prediction.is_empty() {
        return Score::IntZero;
    }
    let lcs = lcs_len(target, prediction, vocab, scratch);
    float_score(div(lcs, prediction.len()), div(lcs, target.len()))
}

/// Mark `marks[i]` for every index `i` of `r` in the LCS of `r` and `c` that
/// `_backtrack_norec(_lcs_table(r, c), r, c)` returns. `masks` holds `c`.
#[allow(clippy::many_single_char_names, clippy::similar_names)] // rouge-score's names
fn mark_lcs(
    r: &[u32],
    c: &[u32],
    masks: &Masks,
    rows: &mut Vec<u64>,
    row_of: &mut Vec<u32>,
    marks: &mut [bool],
) {
    let (m, n) = (r.len(), c.len());
    let words = n.div_ceil(64);
    // Row i of the table (i = 0..=m) is stored row `row_of[i]`; rows without
    // a match equal the previous one and are not stored again. Bit j-1 of a
    // row is 0 iff table[i][j] - table[i][j-1] == 1.
    rows.clear();
    rows.resize(words, u64::MAX);
    row_of.clear();
    row_of.push(0);
    let mut stored = 0;
    for &s in r {
        let p = masks.positions(s);
        if !p.is_empty() {
            let last = stored * words;
            rows.extend_from_within(last..last + words);
            stored += 1;
            advance(&mut rows[stored * words..], p);
        }
        row_of.push(to_u32(stored));
    }
    if stored == 0 {
        return; // nothing in common: the LCS is empty
    }
    let row = |i: usize| -> &[u64] {
        let k = row_of[i] as usize * words;
        &rows[k..k + words]
    };
    let t = |i: usize, j: usize| zeros(row(i), j);
    // table[i][j] - table[i][j-1], for j >= 1.
    let inc = |i: usize, j: usize| usize::from((row(i)[(j - 1) / 64] >> ((j - 1) % 64)) & 1 == 0);
    let (mut i, mut j) = (m, n);
    let mut cur = t(i, j); // table[i][j]
    let mut up = t(i - 1, j); // table[i-1][j]
    while i > 0 && j > 0 {
        if r[i - 1] == c[j - 1] {
            marks[i - 1] = true;
            cur = up - inc(i - 1, j);
            i -= 1;
            j -= 1;
            if i > 0 {
                up = t(i - 1, j);
            }
        } else {
            let left = cur - inc(i, j); // table[i][j-1]
            if left > up {
                up -= inc(i - 1, j);
                cur = left;
                j -= 1;
            } else {
                cur = up;
                i -= 1;
                if i > 0 {
                    up = t(i - 1, j);
                }
            }
        }
    }
    debug_assert!(cur <= m.min(n));
}

/// `_summary_level_lcs(ref_sent, can_sent)`: union-LCS hits between each
/// reference sentence and all candidate sentences, each token counted at
/// most as often as it occurs on both sides.
pub(crate) fn rouge_lsum(
    refs: &[&[u32]],
    cands: &[&[u32]],
    vocab: usize,
    scratch: &mut Scratch,
) -> Score {
    let m: usize = refs.iter().map(|s| s.len()).sum();
    let n: usize = cands.iter().map(|s| s.len()).sum();
    if refs.is_empty() || cands.is_empty() || m == 0 || n == 0 {
        return Score::IntZero;
    }
    let Scratch {
        masks,
        rows,
        row_of,
        marks,
        cnt_r,
        cnt_c,
        ..
    } = scratch;
    // The union of the LCS indices of every (ref, cand) pair, per ref token.
    marks.clear();
    marks.resize(m, false);
    for c in cands.iter().filter(|c| !c.is_empty()) {
        masks.build(c, vocab);
        let mut off = 0;
        for r in refs {
            if !r.is_empty() {
                mark_lcs(r, c, masks, rows, row_of, &mut marks[off..off + r.len()]);
            }
            off += r.len();
        }
        masks.clear();
    }
    cnt_r.clear();
    cnt_r.resize(vocab, 0);
    cnt_c.clear();
    cnt_c.resize(vocab, 0);
    for &t in refs.iter().copied().flatten() {
        cnt_r[t as usize] += 1;
    }
    for &t in cands.iter().copied().flatten() {
        cnt_c[t as usize] += 1;
    }
    let mut hits = 0;
    let tokens = refs.iter().copied().flatten();
    for (&t, &marked) in tokens.zip(marks.iter()) {
        let t = t as usize;
        if marked && cnt_c[t] > 0 && cnt_r[t] > 0 {
            hits += 1;
            cnt_c[t] -= 1;
            cnt_r[t] -= 1;
        }
    }
    float_score(div(hits, n), div(hits, m))
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)] // rouge-score's names (t, r, c, i, j)
mod tests {
    use super::*;

    /// rouge-score's `_lcs_table`.
    fn lcs_table(r: &[u32], c: &[u32]) -> Vec<Vec<usize>> {
        let mut t = vec![vec![0; c.len() + 1]; r.len() + 1];
        for i in 1..=r.len() {
            for j in 1..=c.len() {
                t[i][j] = if r[i - 1] == c[j - 1] {
                    t[i - 1][j - 1] + 1
                } else {
                    t[i - 1][j].max(t[i][j - 1])
                };
            }
        }
        t
    }

    /// rouge-score's `_backtrack_norec`.
    fn backtrack(t: &[Vec<usize>], r: &[u32], c: &[u32]) -> Vec<usize> {
        let (mut i, mut j) = (r.len(), c.len());
        let mut lcs = Vec::new();
        while i > 0 && j > 0 {
            if r[i - 1] == c[j - 1] {
                lcs.insert(0, i - 1);
                i -= 1;
                j -= 1;
            } else if t[i][j - 1] > t[i - 1][j] {
                j -= 1;
            } else {
                i -= 1;
            }
        }
        lcs
    }

    /// rouge-score's `_summary_level_lcs`, verbatim.
    fn naive_lsum(refs: &[Vec<u32>], cands: &[Vec<u32>]) -> Score {
        let m: usize = refs.iter().map(Vec::len).sum();
        let n: usize = cands.iter().map(Vec::len).sum();
        if refs.is_empty() || cands.is_empty() || m == 0 || n == 0 {
            return Score::IntZero;
        }
        let mut cnt_r: FxHashMap<u32, i64> = FxHashMap::default();
        let mut cnt_c: FxHashMap<u32, i64> = FxHashMap::default();
        refs.iter()
            .flatten()
            .for_each(|&t| *cnt_r.entry(t).or_default() += 1);
        cands
            .iter()
            .flatten()
            .for_each(|&t| *cnt_c.entry(t).or_default() += 1);
        let mut hits = 0;
        for r in refs {
            let mut union: Vec<usize> = cands
                .iter()
                .flat_map(|c| backtrack(&lcs_table(r, c), r, c))
                .collect();
            union.sort_unstable();
            union.dedup();
            for i in union {
                let t = r[i];
                let (cc, cr) = (
                    cnt_c.get(&t).copied().unwrap_or(0),
                    cnt_r.get(&t).copied().unwrap_or(0),
                );
                if cc > 0 && cr > 0 {
                    hits += 1;
                    *cnt_c.get_mut(&t).unwrap() -= 1;
                    *cnt_r.get_mut(&t).unwrap() -= 1;
                }
            }
        }
        float_score(div(hits, n), div(hits, m))
    }

    fn naive_ngram(t: &[u32], p: &[u32], n: usize) -> Score {
        let grams = |x: &[u32]| {
            let mut c: FxHashMap<Vec<u32>, usize> = FxHashMap::default();
            if x.len() >= n {
                for w in x.windows(n) {
                    *c.entry(w.to_vec()).or_default() += 1;
                }
            }
            c
        };
        let (tg, pg) = (grams(t), grams(p));
        let overlap: usize = tg
            .iter()
            .map(|(g, &c)| c.min(pg.get(g).copied().unwrap_or(0)))
            .sum();
        let (tc, pc): (usize, usize) = (tg.values().sum(), pg.values().sum());
        float_score(div(overlap, pc.max(1)), div(overlap, tc.max(1)))
    }

    /// xorshift64*: deterministic pseudo-random test data without a dependency.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u32 {
            u32::try_from(self.next() % n).unwrap()
        }
        fn seq(&mut self, max_len: u64, alphabet: u64) -> Vec<u32> {
            let len = self.below(max_len + 1);
            (0..len).map(|_| self.below(alphabet)).collect()
        }
    }

    #[test]
    fn bit_rows_match_the_dp_table() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut scratch = Scratch::default();
        for round in 0..3_000 {
            let (max_len, alphabet) = match round % 4 {
                0 => (8, 2),
                1 => (40, 4),
                2 => (150, 6),
                _ => (300, 40),
            };
            let r = rng.seq(max_len, alphabet);
            let c = rng.seq(max_len, alphabet);
            let t = lcs_table(&r, &c);
            assert_eq!(
                lcs_len(&r, &c, 64, &mut scratch),
                t[r.len()][c.len()],
                "{r:?} {c:?}"
            );
            let mut marks = vec![false; r.len()];
            if !r.is_empty() && !c.is_empty() {
                scratch.masks.build(&c, 64);
                mark_lcs(
                    &r,
                    &c,
                    &scratch.masks,
                    &mut scratch.rows,
                    &mut scratch.row_of,
                    &mut marks,
                );
                scratch.masks.clear();
            }
            let got: Vec<usize> = (0..r.len()).filter(|&i| marks[i]).collect();
            assert_eq!(got, backtrack(&t, &r, &c), "{r:?} {c:?}");
        }
    }

    #[test]
    fn carries_across_many_words() {
        // Long runs of all-ones words: a match far below must carry through them.
        let mut scratch = Scratch::default();
        let mut c = vec![1u32; 1_000];
        c[0] = 0;
        c[999] = 0;
        let r = vec![0u32, 2, 2, 0, 1];
        let t = lcs_table(&r, &c);
        assert_eq!(lcs_len(&r, &c, 3, &mut scratch), t[r.len()][c.len()]);
        let mut marks = vec![false; r.len()];
        scratch.masks.build(&c, 3);
        mark_lcs(
            &r,
            &c,
            &scratch.masks,
            &mut scratch.rows,
            &mut scratch.row_of,
            &mut marks,
        );
        let got: Vec<usize> = (0..r.len()).filter(|&i| marks[i]).collect();
        assert_eq!(got, backtrack(&t, &r, &c));
    }

    #[test]
    fn lsum_matches_reference() {
        let mut rng = Rng(42);
        let mut scratch = Scratch::default();
        for round in 0..1_500 {
            let alphabet = [2, 3, 5, 12][round % 4];
            let sents = |rng: &mut Rng| -> Vec<Vec<u32>> {
                let k = rng.below(5);
                (0..k).map(|_| rng.seq(12, alphabet)).collect()
            };
            let refs = sents(&mut rng);
            let cands = sents(&mut rng);
            let r: Vec<&[u32]> = refs.iter().map(Vec::as_slice).collect();
            let c: Vec<&[u32]> = cands.iter().map(Vec::as_slice).collect();
            assert_eq!(
                rouge_lsum(&r, &c, 16, &mut scratch),
                naive_lsum(&refs, &cands),
                "{refs:?} {cands:?}"
            );
        }
    }

    #[test]
    fn ngrams_match_reference() {
        let mut rng = Rng(7);
        let mut scratch = Scratch::default();
        for round in 0..2_000 {
            let alphabet = [2, 3, 8][round % 3];
            let t = rng.seq(30, alphabet);
            let p = rng.seq(30, alphabet);
            let mut grams = Ngrams::new(&t, &p, 8);
            for n in 1..=9 {
                assert_eq!(
                    grams.score(n, &mut scratch),
                    naive_ngram(&t, &p, n),
                    "{n} {t:?} {p:?}"
                );
                assert_eq!(rouge_n(&t, &p, n, 8, &mut scratch), naive_ngram(&t, &p, n));
            }
        }
    }

    #[test]
    fn rouge_score_unit_cases() {
        let mut s = Scratch::default();
        // testLcsTable / testUnionLcs / testSummaryLevelLcs (ids for tokens 1..9).
        let refs: &[&[u32]] = &[&[1, 2, 3, 4, 5]];
        let cands: &[&[u32]] = &[&[1, 2, 6, 7, 8], &[1, 3, 8, 9, 5]];
        let Score::Float {
            precision,
            recall,
            fmeasure: f,
        } = rouge_lsum(refs, cands, 10, &mut s)
        else {
            panic!("float score expected");
        };
        assert_eq!((precision, recall), (0.4, 0.8));
        assert!((f - 0.5333).abs() < 1e-3);
        assert_eq!(rouge_lsum(&[&[1]], &[&[]], 2, &mut s), Score::IntZero);
        assert_eq!(rouge_lsum(&[], &[&[1]], 2, &mut s), Score::IntZero);
        assert_eq!(rouge_l(&[], &[1], 2, &mut s), Score::IntZero);
        assert_eq!(fmeasure(0.0, 0.0), 0.0);
        assert_eq!(
            fmeasure(1.0, 1.0 / 3.0),
            2.0 * 1.0 * (1.0 / 3.0) / (1.0 + 1.0 / 3.0)
        );
    }

    #[test]
    fn interner_shares_ids_between_stems_and_short_tokens() {
        let mut i = Interner::new(true);
        let a = i.id(b"cats"); // stems to "cat"
        let b = i.id(b"cat");
        let c = i.id(b"cats");
        assert_eq!((a, b, c), (0, 0, 0));
        assert_eq!(i.id(b"dog"), 1);
        assert_eq!(i.len(), 2);
        let mut i = Interner::new(false);
        assert_ne!(i.id(b"cats"), i.id(b"cat"));
    }
}
