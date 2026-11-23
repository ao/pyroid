//! NumPy's C API, looked up with Python errors as errors.
//!
//! rust-numpy looks NumPy's C API up when it builds or type-checks its first
//! array in a process. That lookup imports numpy and runs numpy's Python
//! version check inside an `expect`, so any exception raised there becomes a
//! `PanicException`. One that happens in practice: a Ctrl-C that arrives
//! while a batch runs detached is handled by the first Python code that runs
//! after it, and for a batch that returns arrays, that code was the lookup.
//!
//! [`ready`] runs the Python part of the lookup through rust-numpy's own
//! fallible entry point, so its exceptions are ordinary ones (a
//! `KeyboardInterrupt` stays a `KeyboardInterrupt`). What is left (reading
//! the API capsule of the module just imported) runs no Python code, so no
//! signal handler can raise inside it. Call it before the first array of a
//! call, and before starting a batch whose result holds arrays.

use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

static READY: PyOnceLock<()> = PyOnceLock::new();

/// Make rust-numpy's C-API lookup infallible for the rest of the process.
///
/// `missing` is the message of the `ImportError` raised when numpy is not
/// installed (the original error is its `__cause__`); any other exception,
/// such as a `KeyboardInterrupt`, propagates as it is.
pub fn ready(py: Python<'_>, missing: &str) -> PyResult<()> {
    READY
        .get_or_try_init(py, || {
            // Imports numpy and numpy.lib, runs `NumpyVersion(numpy.__version__)`
            // and caches the module name: all of the lookup's Python code.
            numpy::get_array_module(py).map_err(|e| {
                if e.is_instance_of::<PyImportError>(py) {
                    let err = PyImportError::new_err(missing.to_owned());
                    err.set_cause(py, Some(e));
                    err
                } else {
                    e
                }
            })?;
            // Fills rust-numpy's C-API cache from the capsule of the module
            // imported above, with no Python code in between.
            numpy::npyffi::is_numpy_2(py);
            Ok(())
        })
        .copied()
}

/// `pyroid._pyroid._numpy_ready()`: for Python modules that import numpy at
/// import time (`pyroid.datasketch`), so every later native call finds the
/// lookup done.
#[pyfunction]
fn _numpy_ready(py: Python<'_>) -> PyResult<()> {
    ready(py, "pyroid needs numpy here: pip install numpy")
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_numpy_ready, m)?)
}
