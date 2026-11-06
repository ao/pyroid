//! `pyroid._pyroid.metrics`: the native half of `pyroid.metrics`.
//!
//! One native submodule shared by the ROUGE ([`rouge`]) and BLEU/chrF
//! ([`sacrebleu`]) bindings; each adds its own functions and classes. The
//! public API (rouge-score's and sacrebleu's classes and functions) is the
//! Python package `python/pyroid/metrics/`.

use pyo3::prelude::*;

mod rouge;
mod sacrebleu;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.metrics")?;
    m.gil_used(false)?;
    rouge::add(&m)?;
    sacrebleu::add(&m)?;
    parent.add_submodule(&m)
}
