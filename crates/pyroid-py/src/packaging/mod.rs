// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! `pyroid._pyroid.packaging`: the native classes behind `pyroid.packaging`.
//!
//! `pyroid.packaging` registers its exception classes and a few Python helpers with
//! `_configure` at import (packaging's own exception classes when packaging is installed), and
//! turns the `__class__` claim (see [`claims`]) on or off.

mod version;

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};

pub use version::PyVersion;

/// What `_configure` registered.
#[derive(Debug)]
struct Config {
    invalid_version: Py<PyAny>,
    /// `Version(x)` for a non-`str` `x`: packaging 26.3's own steps, in Python.
    nonstr_version: Py<PyAny>,
    /// packaging's `_Version` named tuple (or pyroid's copy), for `Version._version`.
    version_tuple: Py<PyAny>,
    /// `copyreg.__newobj__`, for `__reduce_ex__`.
    newobj: Py<PyAny>,
    /// packaging's `Version` class (`None` without packaging): the `__class__` claim.
    packaging_version: Option<Py<PyType>>,
    /// packaging's `_BaseVersion` (`None` without packaging): compared through `_key`.
    packaging_base_version: Option<Py<PyType>>,
}

static CONFIG: PyOnceLock<Config> = PyOnceLock::new();

/// Whether pyroid's objects report packaging's classes as `__class__` (off under the patch).
static CLAIMS: AtomicBool = AtomicBool::new(true);

fn config(py: Python<'_>) -> PyResult<&Config> {
    CONFIG.get(py).ok_or_else(|| {
        PyValueError::new_err("pyroid.packaging is not configured (import pyroid.packaging)")
    })
}

/// Is the `__class__` claim on?
fn claims() -> bool {
    CLAIMS.load(Ordering::Relaxed)
}

/// Register the classes and helpers (`pyroid.packaging` does it once at import).
#[pyfunction]
fn _configure(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<()> {
    let get = |k: &str| -> PyResult<Py<PyAny>> {
        config
            .get_item(k)?
            .map(Bound::unbind)
            .ok_or_else(|| PyValueError::new_err(format!("missing {k}")))
    };
    let opt_type = |k: &str| -> PyResult<Option<Py<PyType>>> {
        match config.get_item(k)? {
            Some(v) if !v.is_none() => Ok(Some(v.cast_into::<PyType>()?.unbind())),
            _ => Ok(None),
        }
    };
    let c = Config {
        invalid_version: get("InvalidVersion")?,
        nonstr_version: get("nonstr_version")?,
        version_tuple: get("_Version")?,
        newobj: py.import("copyreg")?.getattr("__newobj__")?.unbind(),
        packaging_version: opt_type("packaging_Version")?,
        packaging_base_version: opt_type("packaging_BaseVersion")?,
    };
    let _ = CONFIG.set(py, c);
    Ok(())
}

/// Turn the `__class__` claim on or off (`pyroid.patch("packaging")` turns it off).
#[pyfunction]
fn _set_claims(on: bool) {
    CLAIMS.store(on, Ordering::Relaxed);
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.packaging")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(_configure, &m)?)?;
    m.add_function(wrap_pyfunction!(_set_claims, &m)?)?;
    version::register(&m)?;
    parent.add_submodule(&m)
}
