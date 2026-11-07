// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! The `(b, r)` parameter searches of `MinHashLSH` and `MinHashLSHEnsemble`.
//!
//! Ports of `_false_positive_probability`, `_false_negative_probability` and
//! `_optimal_param` from datasketch 2.0.0's `lsh.py` (the `lsh_*` functions)
//! and `lshensemble.py` (the `ensemble_*` functions). The probabilities are
//! integrals computed with [`quad`], the port of `scipy.integrate.quad`, and
//! the integrands do the same IEEE operations in the same order as the
//! Python lambdas, so on the same libm every value, and with it every argmin,
//! is the one datasketch computes.
//!
//! Python's `s ** float(r)` is C `pow()` once CPython's special cases are out
//! of the way (numpy's `np.float64 ** float`, in the ensemble integrands,
//! calls `pow()` directly), and `f64::powf` calls the same libm `pow`. The
//! bases here are in `[0, 1]` and the exponents are the integers `r, b >= 1`,
//! so the only special cases CPython can take are `0.0 ** w` and `1.0 ** w`,
//! which give what `pow` gives, and an underflow to zero is not an error in
//! either. The exponents go through [`black_box`] so that LLVM never rewrites
//! `powf(2.0)` as `x * x`, which `pow` need not match in the last bit.

use std::hint::black_box;

use super::quad::quad;

/// `float(n)` for a Python int: round to nearest, as `as` does.
#[allow(clippy::cast_precision_loss)] // the same rounding as Python's float(int)
fn to_f64(n: usize) -> f64 {
    n as f64
}

/// `lsh._false_positive_probability(threshold, b, r)`: the integral of
/// `1 - (1 - s**r)**b` over `[0, threshold]`.
pub fn lsh_false_positive(threshold: f64, b: usize, r: usize) -> f64 {
    let (b, r) = (black_box(to_f64(b)), black_box(to_f64(r)));
    quad(|s| 1.0 - (1.0 - s.powf(r)).powf(b), 0.0, threshold).0
}

/// `lsh._false_negative_probability(threshold, b, r)`: the integral of
/// `1 - (1 - (1 - s**r)**b)` over `[threshold, 1]`.
pub fn lsh_false_negative(threshold: f64, b: usize, r: usize) -> f64 {
    let (b, r) = (black_box(to_f64(b)), black_box(to_f64(r)));
    quad(|s| 1.0 - (1.0 - (1.0 - s.powf(r)).powf(b)), threshold, 1.0).0
}

/// `lsh._optimal_param(threshold, num_perm, false_positive_weight,
/// false_negative_weight)`: the `(b, r)` with `b * r <= num_perm` that
/// minimizes `fp * fp_weight + fn * fn_weight`.
///
/// The loops are datasketch's: `b` from 1 to `num_perm`, `r` from 1 to
/// `int(num_perm / b)` (integer division here, which is the same for
/// `num_perm < 2**53`), the first strict minimum wins, and `(0, 0)` comes
/// back if no error is below infinity (a NaN threshold, for example).
#[allow(clippy::similar_names)] // datasketch's fp and fn
pub fn lsh_optimal_param(
    threshold: f64,
    num_perm: usize,
    fp_weight: f64,
    fn_weight: f64,
) -> (usize, usize) {
    let mut min_error = f64::INFINITY;
    let mut opt = (0, 0);
    for b in 1..=num_perm {
        let max_r = num_perm / b;
        for r in 1..=max_r {
            let fp = lsh_false_positive(threshold, b, r);
            let fn_ = lsh_false_negative(threshold, b, r);
            let error = fp * fp_weight + fn_ * fn_weight;
            if error < min_error {
                min_error = error;
                opt = (b, r);
            }
        }
    }
    opt
}

/// `lshensemble._false_positive_probability(threshold, b, r, xq)`: the
/// integral of `1 - (1 - (t / (1 + xq - t))**r)**b` over `[0, threshold]` if
/// `xq >= threshold`, else over `[0, xq]`. `xq` is the ratio `x / q` of the
/// indexed and query set sizes.
pub fn ensemble_false_positive(threshold: f64, b: usize, r: usize, xq: f64) -> f64 {
    let (b, r) = (black_box(to_f64(b)), black_box(to_f64(r)));
    let probability = |t: f64| 1.0 - (1.0 - (t / (1.0 + xq - t)).powf(r)).powf(b);
    if xq >= threshold {
        quad(probability, 0.0, threshold).0
    } else {
        quad(probability, 0.0, xq).0
    }
}

/// `lshensemble._false_negative_probability(threshold, b, r, xq)`: the
/// integral of `1 - (1 - (1 - (t / (1 + xq - t))**r)**b)` over
/// `[threshold, 1]` if `xq >= 1`, over `[threshold, xq]` if
/// `xq >= threshold`, and 0 otherwise.
pub fn ensemble_false_negative(threshold: f64, b: usize, r: usize, xq: f64) -> f64 {
    let (b, r) = (black_box(to_f64(b)), black_box(to_f64(r)));
    let probability = |t: f64| 1.0 - (1.0 - (1.0 - (t / (1.0 + xq - t)).powf(r)).powf(b));
    if xq >= 1.0 {
        quad(probability, threshold, 1.0).0
    } else if xq >= threshold {
        quad(probability, threshold, xq).0
    } else {
        0.0
    }
}

/// `lshensemble._optimal_param(threshold, num_perm, max_r, xq,
/// false_positive_weight, false_negative_weight)`: the `(b, r)` with
/// `r <= max_r` and `b * r <= num_perm` that minimizes
/// `fp * fp_weight + fn * fn_weight`.
///
/// Same order as datasketch (`b` outer, `r` inner, the first strict minimum
/// wins, `(0, 0)` if none). datasketch skips `b * r > num_perm` with
/// `continue`; stopping `r` at `num_perm / b` visits the same pairs.
///
/// `MinHashLSHEnsemble` calls it once per `xq` in `np.exp(np.linspace(-5, 5,
/// 10))` with `max_r = m`; those `xq` values come from numpy and should be
/// passed through unchanged.
#[allow(clippy::similar_names)] // datasketch's fp and fn
pub fn ensemble_optimal_param(
    threshold: f64,
    num_perm: usize,
    max_r: usize,
    xq: f64,
    fp_weight: f64,
    fn_weight: f64,
) -> (usize, usize) {
    let mut min_error = f64::INFINITY;
    let mut opt = (0, 0);
    for b in 1..=num_perm {
        for r in 1..=max_r.min(num_perm / b) {
            let fp = ensemble_false_positive(threshold, b, r, xq);
            let fn_ = ensemble_false_negative(threshold, b, r, xq);
            let error = fp * fp_weight + fn_ * fn_weight;
            if error < min_error {
                min_error = error;
                opt = (b, r);
            }
        }
    }
    opt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_at_the_ends_of_the_threshold_range() {
        // Empty intervals: quad returns 0 without evaluating the integrand.
        assert_eq!(lsh_false_positive(0.0, 4, 8).to_bits(), 0.0_f64.to_bits());
        assert_eq!(lsh_false_negative(1.0, 4, 8).to_bits(), 0.0_f64.to_bits());
        // b = r = 1: the integrands are s and 1 - s.
        assert_eq!(lsh_false_positive(0.5, 1, 1), 0.125);
        assert_eq!(lsh_false_negative(0.5, 1, 1), 0.125);
    }

    #[test]
    fn ensemble_branches() {
        // xq < threshold: fp integrates over [0, xq] whatever the threshold, fn is 0.
        let xq = 0.25;
        assert_eq!(ensemble_false_negative(0.5, 2, 3, xq), 0.0);
        assert_eq!(
            ensemble_false_positive(0.5, 2, 3, xq).to_bits(),
            ensemble_false_positive(0.75, 2, 3, xq).to_bits()
        );
        // threshold <= xq < 1: fn integrates over [threshold, xq]; xq >= 1:
        // over [threshold, 1].
        let (three, two) = (black_box(3.0), black_box(2.0));
        for (xq, hi) in [(0.75, 0.75), (2.0, 1.0)] {
            let q = |t: f64| 1.0 - (1.0 - (1.0 - (t / (1.0 + xq - t)).powf(three)).powf(two));
            let fn_ = ensemble_false_negative(0.5, 2, 3, xq);
            assert_eq!(fn_.to_bits(), quad(q, 0.5, hi).0.to_bits());
            assert!(fn_ > 0.0);
        }
    }

    #[test]
    fn small_searches_match_datasketch() {
        // Recorded from datasketch 2.0.0 (`lsh._optimal_param(t, n, 0.5, 0.5)`).
        assert_eq!(lsh_optimal_param(0.5, 2, 0.5, 0.5), (1, 1));
        assert_eq!(lsh_optimal_param(0.5, 3, 0.5, 0.5), (1, 1));
        assert_eq!(lsh_optimal_param(0.5, 8, 0.5, 0.5), (3, 2));
        assert_eq!(lsh_optimal_param(0.9, 128, 0.5, 0.5), (5, 25));
        assert_eq!(lsh_optimal_param(0.5, 128, 0.5, 0.5), (25, 5));
    }

    #[test]
    fn nan_threshold_returns_zero_zero() {
        // datasketch's `threshold > 1.0 or threshold < 0.0` check lets NaN
        // through; every error is then NaN and no pair is selected.
        assert_eq!(lsh_optimal_param(f64::NAN, 8, 0.5, 0.5), (0, 0));
    }

    #[test]
    fn ensemble_search_respects_max_r_and_num_perm() {
        for (n, m) in [(16, 4), (16, 16), (7, 3)] {
            let (b, r) = ensemble_optimal_param(0.5, n, m, 1.0, 0.5, 0.5);
            assert!((1..=m).contains(&r) && b * r <= n, "{n} {m}: {b} {r}");
        }
    }
}
