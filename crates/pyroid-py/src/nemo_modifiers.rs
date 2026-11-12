//! `pyroid.nemo`'s string modifiers (NeMo Curator 1.4.0's `DocumentModifier`s). Docs: `pyroid/nemo.pyi`.

use std::borrow::Cow;

use pyo3::prelude::*;
use pyroid_core::nemo::modifiers::{self as m, SliceBound};

use crate::batch::transform;
use crate::input::{TextInput, Threads};

/// ``BoilerPlateStringModifier(remove_if_at_top_or_bottom).modify_document`` for each document.
#[pyfunction]
#[pyo3(signature = (x, /, *, remove_if_at_top_or_bottom = true, threads = None))]
fn remove_boilerplate_strings(
    py: Python<'_>,
    x: TextInput,
    remove_if_at_top_or_bottom: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, |s| {
        m::remove_boilerplate_strings(s, remove_if_at_top_or_bottom)
    })
}

/// ``MarkdownRemover().modify_document`` for each document.
#[pyfunction]
#[pyo3(signature = (x, /, *, threads = None))]
fn remove_markdown(
    py: Python<'_>,
    x: TextInput,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, m::remove_markdown)
}

/// ``NewlineNormalizer().modify_document`` for each document.
#[pyfunction]
#[pyo3(signature = (x, /, *, threads = None))]
fn normalize_newlines(
    py: Python<'_>,
    x: TextInput,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, m::normalize_newlines)
}

/// ``QuotationRemover().modify_document`` for each document.
#[pyfunction]
#[pyo3(signature = (x, /, *, threads = None))]
fn remove_quotation(
    py: Python<'_>,
    x: TextInput,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, m::remove_quotation)
}

/// ``UrlRemover().modify_document`` for each document.
#[pyfunction]
#[pyo3(signature = (x, /, *, threads = None))]
fn remove_urls(
    py: Python<'_>,
    x: TextInput,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, m::remove_urls)
}

/// ``LineRemover(patterns).modify_document`` for each document.
#[pyfunction]
#[pyo3(signature = (x, /, patterns, *, threads = None))]
fn remove_lines(
    py: Python<'_>,
    x: TextInput,
    patterns: Vec<String>,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, |s| m::remove_lines(s, &patterns))
}

fn slice_bound(index: Option<i64>, sub: Option<String>) -> SliceBound {
    match (index, sub) {
        (Some(i), _) => SliceBound::Index(i),
        (None, Some(s)) => SliceBound::Substring(s),
        (None, None) => SliceBound::Default,
    }
}

/// ``Slicer(...).modify_document`` for each document; ``pyroid.nemo.slice_text`` resolves the bounds.
#[pyfunction]
#[pyo3(signature = (x, /, left_index, left_str, right_index, right_str, include_left, include_right, strip, *, threads = None))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn _slice_text(
    py: Python<'_>,
    x: TextInput,
    left_index: Option<i64>,
    left_str: Option<String>,
    right_index: Option<i64>,
    right_str: Option<String>,
    include_left: bool,
    include_right: bool,
    strip: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    let left = slice_bound(left_index, left_str);
    let right = slice_bound(right_index, right_str);
    transform(py, x, threads, |s| {
        m::slice_text(s, &left, &right, include_left, include_right, strip)
    })
}

/// ``FastTextLabelModifier(label).modify_document`` for each document (``label`` already formatted).
#[pyfunction]
#[pyo3(signature = (x, /, label, *, threads = None))]
fn _fasttext_label(
    py: Python<'_>,
    x: TextInput,
    label: String,
    threads: Option<Threads>,
) -> PyResult<Bound<'_, PyAny>> {
    transform(py, x, threads, |s| Cow::Owned(m::fasttext_label(s, &label)))
}

pub(crate) fn add(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(remove_boilerplate_strings, module)?)?;
    module.add_function(wrap_pyfunction!(remove_markdown, module)?)?;
    module.add_function(wrap_pyfunction!(normalize_newlines, module)?)?;
    module.add_function(wrap_pyfunction!(remove_quotation, module)?)?;
    module.add_function(wrap_pyfunction!(remove_urls, module)?)?;
    module.add_function(wrap_pyfunction!(remove_lines, module)?)?;
    module.add_function(wrap_pyfunction!(_slice_text, module)?)?;
    module.add_function(wrap_pyfunction!(_fasttext_label, module)?)?;
    Ok(())
}
