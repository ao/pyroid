// Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
//! `Detector._extract_ngrams` and `Detector._detect_block`, with langdetect's floating-point
//! operations in langdetect's order.

use super::chars::{SPACE, Upper, normalize};
use super::mt::Random;
use super::profiles::Frozen;

/// `sum(prob)` of the running interpreter: CPython 3.12 switched float sums to Neumaier's
/// compensated summation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumMode {
    /// CPython 3.11: left to right.
    Plain,
    /// CPython 3.12 and later.
    Compensated,
}

/// `sum(xs)` (start value 0) as CPython computes it for a list of floats.
pub fn py_sum(xs: &[f64], mode: SumMode) -> f64 {
    let Some((&first, rest)) = xs.split_first() else {
        return 0.0;
    };
    // `0 + first`: the int start value joins the first float as 0.0 + first.
    let mut hi = 0.0 + first;
    match mode {
        SumMode::Plain => {
            for &x in rest {
                hi += x;
            }
            hi
        }
        SumMode::Compensated => {
            let mut lo = 0.0f64;
            for &x in rest {
                let t = hi + x;
                if hi.abs() >= x.abs() {
                    lo += (hi - t) + x;
                } else {
                    lo += (x - t) + hi;
                }
                hi = t;
            }
            if lo != 0.0 && lo.is_finite() {
                hi + lo
            } else {
                hi
            }
        }
    }
}

/// The rows of the n-grams `_extract_ngrams` finds in `text`, in order.
pub fn extract_ngrams(text: &[u32], map: &Frozen, upper: &Upper) -> Vec<u32> {
    extract_ngrams_ticked(text, map, upper, &mut || true).unwrap_or_default()
}

/// [`extract_ngrams`], calling `tick` every 65,536 characters; `None` if it returned `false`.
pub fn extract_ngrams_ticked(
    text: &[u32],
    map: &Frozen,
    upper: &Upper,
    tick: &mut dyn FnMut() -> bool,
) -> Option<Vec<u32>> {
    let mut out = Vec::with_capacity(text.len().saturating_mul(2).min(1 << 24));
    let mut grams = [SPACE, 0, 0];
    let mut len = 1usize;
    let mut capital = false;
    for (pos, &raw) in text.iter().enumerate() {
        if pos % 65_536 == 65_535 && !tick() {
            return None;
        }
        let ch = normalize(raw);
        let last = grams[len - 1];
        let mut add = true;
        if last == SPACE {
            grams[0] = SPACE;
            len = 1;
            capital = false;
            if ch == SPACE {
                add = false;
            }
        } else if len >= 3 {
            grams = [grams[1], grams[2], 0];
            len = 2;
        }
        if add {
            grams[len] = ch;
            len += 1;
            if upper.is_upper(ch) {
                if upper.is_upper(last) {
                    capital = true;
                }
            } else {
                capital = false;
            }
        }
        if capital {
            continue;
        }
        for n in 1..=3 {
            if len < n {
                break;
            }
            let w = &grams[len - n..len];
            if n == 1 && w[0] == SPACE {
                continue;
            }
            if let Some(row) = map.lookup_gram(w) {
                out.push(row);
            }
        }
    }
    Some(out)
}

/// The n-grams `_extract_ngrams` finds, as code points (for `Detector._extract_ngrams`).
pub fn extract_ngram_words(text: &[u32], map: &Frozen, upper: &Upper) -> Vec<Vec<u32>> {
    extract_ngrams(text, map, upper)
        .into_iter()
        .map(|r| map.word(r))
        .collect()
}

/// `Detector`'s class constants and the attributes `_detect_block` reads.
#[derive(Clone, Debug)]
pub struct Params {
    /// `self.alpha`.
    pub alpha: f64,
    /// `self.n_trial` (negative: no trials).
    pub n_trial: i64,
    /// `self.ALPHA_WIDTH`.
    pub alpha_width: f64,
    /// `self.ITERATION_LIMIT`.
    pub iteration_limit: f64,
    /// `self.CONV_THRESHOLD`.
    pub conv_threshold: f64,
    /// `self.BASE_FREQ`.
    pub base_freq: f64,
    /// `self.prior_map` (`None`: uniform over the language list).
    pub prior: Option<Vec<f64>>,
    /// `len(self.langlist)`.
    pub n_langs: usize,
    /// How `sum()` adds floats.
    pub sum_mode: SumMode,
}

impl Params {
    /// langdetect's defaults for a language list of `n_langs`.
    pub fn defaults(n_langs: usize, sum_mode: SumMode) -> Self {
        Self {
            alpha: 0.5,
            n_trial: 7,
            alpha_width: 0.05,
            iteration_limit: 1000.0,
            conv_threshold: 0.99999,
            base_freq: 10000.0,
            prior: None,
            n_langs,
            sum_mode,
        }
    }
}

/// Why `_detect_block` raised after it set `self.langprob`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// `IndexError: list index out of range` (a row or the prior is shorter than needed).
    Index,
    /// `ZeroDivisionError: float division by zero` (all probabilities became 0, or
    /// `BASE_FREQ` is 0).
    ZeroDivision,
    /// Stopped by the interrupt check.
    Interrupted,
}

/// The result of the trials: `langprob` (complete, or as far as it got before `failure`).
#[derive(Clone, Debug)]
pub struct Outcome {
    /// `self.langprob`.
    pub langprob: Vec<f64>,
    /// The exception the trials raised, if any.
    pub failure: Option<Failure>,
}

/// `_update_lang_prob(prob, word, alpha)`: `prob[i] *= weight + row[i]` for every `i`.
/// `saved` is scratch space.
#[inline]
fn update(
    prob: &mut [f64],
    row: super::profiles::RowRef<'_>,
    weight: f64,
    saved: &mut Vec<(usize, f64)>,
) -> Result<(), Failure> {
    let n = prob.len().min(row.len);
    if let Some(dense) = row.dense {
        // langdetect's own loop, over every language.
        for (x, &p) in prob[..n].iter_mut().zip(&dense[..n]) {
            *x *= weight + p;
        }
    } else {
        // `weight + 0.0` for the entries never set (`[0.0] * langsize`); the set entries from
        // their old values first.
        let w0 = weight + 0.0;
        let mut few = [(0usize, 0.0f64); super::profiles::DENSE_MIN];
        let mut count = 0usize;
        saved.clear();
        for (&i, &p) in row.lang.iter().zip(row.prob) {
            let i = i as usize;
            if i < n {
                let v = prob[i] * (weight + p);
                if count < few.len() {
                    few[count] = (i, v);
                    count += 1;
                } else {
                    saved.push((i, v));
                }
            }
        }
        for x in &mut prob[..n] {
            *x *= w0;
        }
        for &(i, v) in few[..count].iter().chain(saved.iter()) {
            prob[i] = v;
        }
    }
    if n < prob.len() {
        return Err(Failure::Index);
    }
    Ok(())
}

/// `if maxp < p: maxp = p` over `xs`, from `maxp = 0.0`: the largest value at least 0.0,
/// ignoring NaN; any order gives the same value, so four lanes run at once.
#[inline]
fn max_from_zero(xs: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let mut chunks = xs.chunks_exact(4);
    for c in &mut chunks {
        for k in 0..4 {
            if acc[k] < c[k] {
                acc[k] = c[k];
            }
        }
    }
    let mut m = 0.0f64;
    for &v in acc.iter().chain(chunks.remainder()) {
        if m < v {
            m = v;
        }
    }
    m
}

/// `_normalize_prob(prob)`: divide by the sum, return the largest value.
#[inline]
fn normalize_prob(prob: &mut [f64], mode: SumMode) -> Result<f64, Failure> {
    if prob.is_empty() {
        return Ok(0.0);
    }
    let sump = py_sum(prob, mode);
    if sump == 0.0 {
        return Err(Failure::ZeroDivision);
    }
    for x in prob.iter_mut() {
        *x /= sump;
    }
    Ok(max_from_zero(prob))
}

/// `_update_lang_prob` on the non-zero entries `live` of `prob` only (see [`trials`]).
#[inline]
fn update_live(
    prob: &mut [f64],
    live: &[u32],
    row: super::profiles::RowRef<'_>,
    weight: f64,
    saved: &mut Vec<(usize, f64)>,
) -> Result<(), Failure> {
    if row.len < prob.len() {
        return Err(Failure::Index);
    }
    let w0 = weight + 0.0;
    saved.clear();
    // The set entries from their old values (a zero stays zero), then the others.
    for (&i, &p) in row.lang.iter().zip(row.prob) {
        let i = i as usize;
        if i < prob.len() {
            saved.push((i, prob[i] * (weight + p)));
        }
    }
    for &i in live {
        prob[i as usize] *= w0;
    }
    for &(i, v) in saved.iter() {
        prob[i] = v;
    }
    Ok(())
}

/// `_normalize_prob` on the non-zero entries `live` of `prob`, which then drops the entries
/// that became zero.
#[inline]
fn normalize_live(prob: &mut [f64], live: &mut Vec<u32>, mode: SumMode) -> Result<f64, Failure> {
    if prob.is_empty() {
        return Ok(0.0);
    }
    let Some((&first, rest)) = live.split_first() else {
        return Err(Failure::ZeroDivision);
    };
    // Zeros add nothing to either sum (the running sum is never -0.0), so skipping them
    // gives the same bits.
    let mut hi = 0.0 + prob[first as usize];
    match mode {
        SumMode::Plain => {
            for &i in rest {
                hi += prob[i as usize];
            }
        }
        SumMode::Compensated => {
            let mut lo = 0.0f64;
            for &i in rest {
                let x = prob[i as usize];
                let t = hi + x;
                if hi.abs() >= x.abs() {
                    lo += (hi - t) + x;
                } else {
                    lo += (x - t) + hi;
                }
                hi = t;
            }
            if lo != 0.0 && lo.is_finite() {
                hi += lo;
            }
        }
    }
    let sump = hi;
    if sump == 0.0 {
        return Err(Failure::ZeroDivision);
    }
    let mut maxp = 0.0f64;
    for &i in live.iter() {
        let x = &mut prob[i as usize];
        let p = *x / sump;
        if maxp < p {
            maxp = p;
        }
        *x = p;
    }
    live.retain(|&i| prob[i as usize] != 0.0);
    Ok(maxp)
}

/// Can a trial skip the zero entries? Only while a zero entry provably stays zero and adds
/// nothing: every factor (`weight + p`) finite and non-negative, start values finite and
/// non-negative, and no product able to overflow between two normalisations.
fn skips_zeros(weight: f64, start: &[f64], map: &Frozen) -> bool {
    weight.is_finite()
        && weight >= 0.0
        && map.probs_nonneg_finite()
        && weight + map.max_prob() <= 1e50
        && start
            .iter()
            .all(|&v| v.is_finite() && (0.0..=1e8).contains(&v))
}

/// The trials of `_detect_block` over the n-gram rows `ngrams` (not empty), drawing from
/// `rng`. `tick` is called now and then; returning `false` stops with
/// [`Failure::Interrupted`].
pub fn trials(
    ngrams: &[u32],
    map: &Frozen,
    params: &Params,
    rng: &mut Random,
    tick: &mut dyn FnMut() -> bool,
) -> Outcome {
    let l = params.n_langs;
    let mut out = Outcome {
        langprob: vec![0.0; l],
        failure: None,
    };
    let n = ngrams.len() as u64;
    // `prob[j] / self.n_trial`: the int converted to a float.
    #[allow(clippy::cast_precision_loss)]
    let divisor = params.n_trial as f64;
    let mut steps = 0u32;
    let mut prob: Vec<f64> = Vec::with_capacity(l);
    let mut live: Vec<u32> = Vec::with_capacity(l);
    let mut saved: Vec<(usize, f64)> = Vec::with_capacity(8);
    for _ in 0..params.n_trial.max(0) {
        prob.clear();
        match &params.prior {
            Some(p) => prob.extend_from_slice(p),
            // `1.0 / len(self.langlist)` raises for an empty list.
            None if l == 0 => {
                out.failure = Some(Failure::ZeroDivision);
                return out;
            }
            #[allow(clippy::cast_precision_loss)]
            None => prob.resize(l, 1.0 / l as f64),
        }
        let alpha = params.alpha + rng.gauss(0.0, 1.0) * params.alpha_width;
        if params.base_freq == 0.0 {
            // `alpha / self.BASE_FREQ` in the first `_update_lang_prob` (every drawn n-gram
            // has a row) raises before any probability changes.
            out.failure = Some(Failure::ZeroDivision);
            return out;
        }
        let weight = alpha / params.base_freq;
        // Once few entries are non-zero, the loop runs over those only (when that is exact).
        let can_skip = skips_zeros(weight, &prob, map);
        let mut fast = false;
        let mut i = 0u64;
        loop {
            // n > 0, and randbelow(n) < n.
            #[allow(clippy::cast_possible_truncation)]
            let row = ngrams[rng.randbelow(n) as usize];
            let updated = if fast {
                update_live(&mut prob, &live, map.row(row), weight, &mut saved)
            } else {
                update(&mut prob, map.row(row), weight, &mut saved)
            };
            if let Err(f) = updated {
                out.failure = Some(f);
                return out;
            }
            if i % 5 == 0 {
                let normalized = if fast {
                    normalize_live(&mut prob, &mut live, params.sum_mode)
                } else {
                    normalize_prob(&mut prob, params.sum_mode)
                };
                match normalized {
                    Err(f) => {
                        out.failure = Some(f);
                        return out;
                    }
                    Ok(top) => {
                        // i counts steps of one trial (at most a few million).
                        #[allow(clippy::cast_precision_loss)]
                        let step = i as f64;
                        if top > params.conv_threshold || step >= params.iteration_limit {
                            break;
                        }
                        if can_skip && !fast && prob.iter().filter(|&&v| v != 0.0).count() <= 16 {
                            fast = true;
                            live.clear();
                            // prob.len() <= n_langs, far below 2^32.
                            #[allow(clippy::cast_possible_truncation)]
                            live.extend(
                                (0..prob.len())
                                    .filter(|&j| prob[j] != 0.0)
                                    .map(|j| j as u32),
                            );
                        }
                    }
                }
            }
            i += 1;
            steps = steps.wrapping_add(1);
            if steps % 4096 == 0 && !tick() {
                out.failure = Some(Failure::Interrupted);
                return out;
            }
        }
        // `self.langprob[j] += prob[j] / self.n_trial` for j < len(langprob): an IndexError
        // after the first len(prob) if prob is shorter.
        for (acc, p) in out.langprob.iter_mut().zip(&prob) {
            *acc += p / divisor;
        }
        if prob.len() < l {
            out.failure = Some(Failure::Index);
            return out;
        }
    }
    out
}

/// `_sort_probability(prob)`: `(index, p)` for `p > threshold`, highest first, ties in list
/// order.
pub fn sort_probability(prob: &[f64], threshold: f64) -> Vec<(usize, f64)> {
    let mut out: Vec<(usize, f64)> = prob
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, p)| p > threshold)
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_like_cpython() {
        // sum([0.1] * 10): 3.11 gives 0.9999999999999999, 3.12+ gives 1.0.
        let xs = [0.1; 10];
        assert_eq!(py_sum(&xs, SumMode::Plain), 0.999_999_999_999_999_9);
        assert_eq!(py_sum(&xs, SumMode::Compensated), 1.0);
        assert_eq!(py_sum(&[], SumMode::Compensated), 0.0);
        // sum([-0.0]) == 0.0 (the int start value), and an infinite sum stays infinite.
        assert!(py_sum(&[-0.0], SumMode::Compensated).is_sign_positive());
        assert_eq!(
            py_sum(&[f64::MAX, f64::MAX], SumMode::Compensated),
            f64::INFINITY
        );
    }

    #[test]
    fn sort_is_stable_descending() {
        let v = sort_probability(&[0.2, 0.5, 0.2, 0.05], 0.1);
        assert_eq!(v, vec![(1, 0.5), (0, 0.2), (2, 0.2)]);
    }
}
