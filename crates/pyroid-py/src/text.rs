//! `pyroid.text`: the demonstration product module. It shows the whole
//! convention in ~40 lines: a transform and a predicate over `TextInput`,
//! registered as a native submodule with one line in `lib.rs`.

use pyo3::prelude::*;

use crate::batch::{predicate, transform};
use crate::input::{Count, TextInput, Threads};

/// Collapse every run of whitespace to one space and strip both ends.
///
/// Same result as ``" ".join(s.split())`` for every document. Accepts a str,
/// a list/tuple of str, or an Arrow string array/column, and returns the same
/// kind; unchanged documents are returned as the original objects.
#[pyfunction]
#[pyo3(signature = (x, /, *, threads = None))]
fn normalize_whitespace(
    py: Python<'_>,
    x: TextInput,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, pyroid_core::text::normalize_whitespace)
}

/// True for each document with at least ``n`` characters (``len(s) >= n``).
///
/// Returns a bool, a list/tuple of bools, or an Arrow bool column (nulls stay
/// null), matching the input kind.
#[pyfunction]
#[pyo3(signature = (x, /, n, *, threads = None))]
fn min_chars(
    py: Python<'_>,
    x: TextInput,
    n: Count,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    predicate(py, x, threads, |s| pyroid_core::text::min_chars(s, n.0))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.text")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(normalize_whitespace, &m)?)?;
    m.add_function(wrap_pyfunction!(min_chars, &m)?)?;
    parent.add_submodule(&m)
}
