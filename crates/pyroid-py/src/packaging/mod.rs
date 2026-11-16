// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! `pyroid._pyroid.packaging`: the native classes behind `pyroid.packaging`.
//!
//! `pyroid.packaging` registers its exception classes and a few Python helpers with
//! `_configure` at import (packaging's own exception classes when packaging is installed), and
//! turns the `__class__` claim (see [`claims`]) on or off.

mod batch;
mod cell;
mod markers;
mod specifiers;
mod version;

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyType};

/// What `_configure` registered.
#[derive(Debug)]
struct Config {
    invalid_version: Py<PyAny>,
    /// `Version(x)` for a non-`str` `x`: packaging 26.3's own steps, in Python.
    nonstr_version: Py<PyAny>,
    /// packaging's `_Version` named tuple (or pyroid's copy), for `Version._version`.
    version_tuple: Py<PyAny>,
    /// `_restore(cls, state)`: what pyroid's objects pickle through (`cls.__new__(cls)`, then
    /// `__setstate__(state)`).
    restore: Py<PyAny>,
    /// packaging's `Version` class (`None` without packaging): the `__class__` claim.
    packaging_version: Option<Py<PyType>>,
    /// packaging's `Version` class whatever its release (`None` without packaging): read
    /// through its slots.
    packaging_version_any: Option<Py<PyType>>,
    /// packaging's `_BaseVersion` (`None` without packaging): compared through `_key`.
    packaging_base_version: Option<Py<PyType>>,
    invalid_specifier: Py<PyAny>,
    /// pyroid's copy of `Specifier._regex` (for non-`str` arguments and lone surrogates).
    spec_regex: Py<PyAny>,
    /// `dict.fromkeys`.
    dict_fromkeys: Py<PyAny>,
    /// `_delegate(name, obj, *args)`: what pyroid hands to packaging's own code.
    delegate: Py<PyAny>,
    /// packaging's `Specifier` and `SpecifierSet` (the `__class__` claim).
    packaging_specifier: Option<Py<PyType>>,
    packaging_specifier_set: Option<Py<PyType>>,
    invalid_marker: Py<PyAny>,
    undefined_comparison: Py<PyAny>,
    undefined_environment_name: Py<PyAny>,
    invalid_requirement: Py<PyAny>,
    /// packaging's `_tokenizer.ParserSyntaxError` (or pyroid's copy): the `__cause__`.
    parser_syntax_error: Py<PyAny>,
    /// `ast.literal_eval`.
    literal_eval: Py<PyAny>,
    /// `_cached_default_environment` (a `functools.cache`; tests clear it).
    cached_default_environment: Py<PyAny>,
    /// The Python `canonicalize_name` (for non-`str` values, which fail as in packaging).
    canonicalize_name: Py<PyAny>,
    /// packaging's `_parser.Variable`, `Value`, `Op` (or pyroid's copies): `Marker._markers`.
    node_variable: Py<PyAny>,
    node_value: Py<PyAny>,
    node_op: Py<PyAny>,
    packaging_marker: Option<Py<PyType>>,
    /// `re.compile(r"[ \t]+")`: packaging's first tokenizer rule (its TypeError for a non-str).
    ws_regex: Py<PyAny>,
    /// The caller's `unicodedata.unidata_version` (for `\b` around marker tokens).
    unicode: pyroid_core::ftfy::UnicodeVersion,
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
        restore: get("restore")?,
        packaging_version: opt_type("packaging_Version")?,
        packaging_version_any: opt_type("packaging_Version_any")?,
        packaging_base_version: opt_type("packaging_BaseVersion")?,
        invalid_specifier: get("InvalidSpecifier")?,
        spec_regex: get("spec_regex")?,
        dict_fromkeys: py.get_type::<PyDict>().getattr("fromkeys")?.unbind(),
        delegate: get("delegate")?,
        packaging_specifier: opt_type("packaging_Specifier")?,
        packaging_specifier_set: opt_type("packaging_SpecifierSet")?,
        invalid_marker: get("InvalidMarker")?,
        undefined_comparison: get("UndefinedComparison")?,
        undefined_environment_name: get("UndefinedEnvironmentName")?,
        invalid_requirement: get("InvalidRequirement")?,
        parser_syntax_error: get("ParserSyntaxError")?,
        literal_eval: py.import("ast")?.getattr("literal_eval")?.unbind(),
        cached_default_environment: get("cached_default_environment")?,
        canonicalize_name: get("canonicalize_name")?,
        node_variable: get("Variable")?,
        node_value: get("Value")?,
        node_op: get("Op")?,
        packaging_marker: opt_type("packaging_Marker")?,
        ws_regex: py
            .import("re")?
            .call_method1("compile", (r"[ \t]+",))?
            .unbind(),
        unicode: {
            let v: String = py
                .import("unicodedata")?
                .getattr("unidata_version")?
                .extract()?;
            pyroid_core::ftfy::UnicodeVersion::from_unidata_version(&v)
                .unwrap_or(pyroid_core::ftfy::UnicodeVersion::V17_0)
        },
    };
    let _ = CONFIG.set(py, c);
    Ok(())
}

/// Turn the `__class__` claim on or off (`pyroid.patch("packaging")` turns it off).
#[pyfunction]
fn _set_claims(on: bool) {
    CLAIMS.store(on, Ordering::Relaxed);
}

/// Is the `__class__` claim on?
#[pyfunction]
fn _claims() -> bool {
    claims()
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.packaging")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(_configure, &m)?)?;
    m.add_function(wrap_pyfunction!(_set_claims, &m)?)?;
    m.add_function(wrap_pyfunction!(_claims, &m)?)?;
    version::register(&m)?;
    specifiers::register(&m)?;
    markers::register(&m)?;
    batch::register(&m)?;
    parent.add_submodule(&m)
}
