//! Bindings for `pyroid_core::metrics::rouge` (placeholder; see the metrics brief).

use pyo3::prelude::*;

/// Add this half's functions and classes to the shared `pyroid.metrics` native module.
#[allow(clippy::unnecessary_wraps)]
pub fn add(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
