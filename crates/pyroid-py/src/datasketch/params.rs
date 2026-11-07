// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! The LSH and LSH Ensemble `(b, r)` searches (datasketch's `_optimal_param`
//! with `scipy.integrate.quad`, ported to Rust in `pyroid_core::datasketch`).

use pyo3::prelude::*;
use pyroid_core::datasketch::params;

/// `datasketch.lsh._false_positive_probability(threshold, b, r)`.
#[pyfunction]
fn lsh_false_positive(threshold: f64, b: usize, r: usize) -> f64 {
    params::lsh_false_positive(threshold, b, r)
}

/// `datasketch.lsh._false_negative_probability(threshold, b, r)`.
#[pyfunction]
fn lsh_false_negative(threshold: f64, b: usize, r: usize) -> f64 {
    params::lsh_false_negative(threshold, b, r)
}

/// `datasketch.lsh._optimal_param`; detached (it runs thousands of integrals).
#[pyfunction]
fn lsh_optimal_param(
    py: Python<'_>,
    threshold: f64,
    num_perm: usize,
    false_positive_weight: f64,
    false_negative_weight: f64,
) -> (usize, usize) {
    py.detach(|| {
        params::lsh_optimal_param(
            threshold,
            num_perm,
            false_positive_weight,
            false_negative_weight,
        )
    })
}

/// `datasketch.lshensemble._false_positive_probability(threshold, b, r, xq)`.
#[pyfunction]
fn ensemble_false_positive(threshold: f64, b: usize, r: usize, xq: f64) -> f64 {
    params::ensemble_false_positive(threshold, b, r, xq)
}

/// `datasketch.lshensemble._false_negative_probability(threshold, b, r, xq)`.
#[pyfunction]
fn ensemble_false_negative(threshold: f64, b: usize, r: usize, xq: f64) -> f64 {
    params::ensemble_false_negative(threshold, b, r, xq)
}

/// `datasketch.lshensemble._optimal_param`.
#[pyfunction]
fn ensemble_optimal_param(
    py: Python<'_>,
    threshold: f64,
    num_perm: usize,
    max_r: usize,
    xq: f64,
    false_positive_weight: f64,
    false_negative_weight: f64,
) -> (usize, usize) {
    py.detach(|| {
        params::ensemble_optimal_param(
            threshold,
            num_perm,
            max_r,
            xq,
            false_positive_weight,
            false_negative_weight,
        )
    })
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lsh_false_positive, m)?)?;
    m.add_function(wrap_pyfunction!(lsh_false_negative, m)?)?;
    m.add_function(wrap_pyfunction!(lsh_optimal_param, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_false_positive, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_false_negative, m)?)?;
    m.add_function(wrap_pyfunction!(ensemble_optimal_param, m)?)?;
    Ok(())
}
