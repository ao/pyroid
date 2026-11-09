//! `pyroid.nemo`: NeMo Curator 1.4.0's heuristic filter scores, string modifiers and fastText stage helpers.

use pyo3::prelude::*;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.nemo")?;
    m.gil_used(false)?;
    // Filter scores and fastText helpers (added here).
    // String modifiers:
    crate::nemo_modifiers::add(&m)?;
    parent.add_submodule(&m)
}
