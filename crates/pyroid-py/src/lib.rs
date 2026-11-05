//! PyO3 bindings: the `pyroid._pyroid` extension module.
//!
//! Keep this crate thin. Kernels live in `pyroid-core`; this crate only
//! converts arguments, releases the GIL around real work, and maps errors to
//! the Python exception hierarchy.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// Exception hierarchy. Every error pyroid raises derives from `PyroidError`.
// Add subclasses here, e.g.
//     create_exception!(pyroid, InputError, PyroidError, "Invalid input.");
// register them in `_pyroid` below, and re-export them from
// python/pyroid/__init__.py and python/pyroid/_pyroid.pyi. The module path
// "pyroid" makes `__module__ == "pyroid"`, so instances pickle by reference to
// the public name.
create_exception!(
    pyroid,
    PyroidError,
    PyException,
    "Base class for all pyroid errors."
);

/// Length of `text` in UTF-8 bytes.
///
/// Placeholder that proves the core -> bindings -> wheel pipeline; it will be
/// removed once real kernels land.
#[pyfunction]
#[pyo3(signature = (text, /))]
fn utf8_len(text: &str) -> usize {
    pyroid_core::utf8_len(text)
}

#[pymodule(gil_used = false)]
fn _pyroid(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("PyroidError", m.py().get_type::<PyroidError>())?;
    m.add_function(wrap_pyfunction!(utf8_len, m)?)?;
    Ok(())
}
