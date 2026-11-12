//! `pyroid._pyroid.metrics`: the native half of `pyroid.metrics`.
//!
//! One native submodule shared by the ROUGE ([`rouge`]) and BLEU/chrF
//! ([`sacrebleu`]) bindings; each adds its own functions and classes. The
//! public API (rouge-score's and sacrebleu's classes and functions) is the
//! Python package `python/pyroid/metrics/`.
//!
//! Shared here: `batch_texts(texts, name)`, the documents of a batch argument
//! as a list (`None` for Arrow nulls), for the paths that process them in
//! Python.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};

use crate::batch::to_list;
use crate::errors::input_error;
use crate::input::{TextInput, Threads, text_arg};

mod rouge;
mod sacrebleu;

/// A batch argument named `name`: a list/tuple of str or an Arrow string
/// array (not a single str, which has no pairs to score).
fn batch_arg(
    obj: &Bound<'_, PyAny>,
    threads_arg: Option<Threads>,
    name: &str,
) -> PyResult<TextInput> {
    if obj.is_instance_of::<PyString>() {
        return Err(input_error(
            obj.py(),
            format!(
                "{name} must be a list or tuple of str or an Arrow string array, not a single str"
            ),
        ));
    }
    text_arg(obj, threads_arg, name)
}

/// The documents of the batch argument `name` (a list/tuple of str or an
/// Arrow string array) as a list: the same str objects for a list or tuple,
/// new strs for Arrow, and `None` for each Arrow null.
#[pyfunction]
fn batch_texts<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyList>> {
    to_list(py, &batch_arg(texts, None, name)?)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.metrics")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(batch_texts, &m)?)?;
    rouge::add(&m)?;
    sacrebleu::add(&m)?;
    parent.add_submodule(&m)
}
