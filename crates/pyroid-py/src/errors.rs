//! The exception hierarchy. Every error pyroid defines derives from
//! `PyroidError`; standard Python exceptions (`ValueError`, `OverflowError`,
//! `UnicodeEncodeError`, ...) are used where they already say it best.
//!
//! - `PyroidError(Exception)`: base class.
//! - `InputError(PyroidError, TypeError)`: an argument has the wrong type
//!   (not a str, a list holding a non-str, an Arrow column that isn't
//!   strings, ...). It is a `TypeError`, so `except TypeError` still works.
//!
//! Classes report `__module__ == "pyroid"` so they pickle by their public
//! name. Product tracks add a subclass here with one `create_exception!` (or
//! `make_type` call for multiple bases) and one line in [`register`].

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

create_exception!(
    pyroid,
    PyroidError,
    PyException,
    "Base class for all pyroid errors."
);

static INPUT_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// Create `pyroid.<name>` with several base classes (`create_exception!`
/// supports only one).
fn make_type(
    py: Python<'_>,
    name: &str,
    doc: &str,
    bases: &[Bound<'_, PyType>],
) -> PyResult<Py<PyType>> {
    let ns = PyDict::new(py);
    ns.set_item("__module__", "pyroid")?;
    ns.set_item("__doc__", doc)?;
    let ty = py
        .get_type::<PyType>()
        .call1((name, PyTuple::new(py, bases)?, ns))?;
    Ok(ty.cast_into::<PyType>()?.unbind())
}

/// The `pyroid.InputError` class.
pub fn input_error_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    INPUT_ERROR
        .get_or_try_init(py, || {
            make_type(
                py,
                "InputError",
                "An argument has the wrong type or shape for this pyroid function.",
                &[py.get_type::<PyroidError>(), py.get_type::<PyTypeError>()],
            )
        })
        .map(|t| t.bind(py))
}

/// A `pyroid.InputError` with `msg`.
pub fn input_error(py: Python<'_>, msg: impl Into<String>) -> PyErr {
    match input_error_type(py) {
        Ok(ty) => PyErr::from_type(ty.clone(), msg.into()),
        Err(e) => e,
    }
}

/// Human-readable type of `obj` for error messages: `int`, `numpy.ndarray`.
pub fn type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .fully_qualified_name()
        .map_or_else(|_| "<unknown type>".to_owned(), |n| n.to_string())
}

/// Attach a note (PEP 678) to `err`; best effort.
pub fn with_note(py: Python<'_>, err: PyErr, note: impl AsRef<str>) -> PyErr {
    // A failing add_note (e.g. an exotic exception type) must not hide `err`.
    let _ = err.add_note(py, note.as_ref().to_owned());
    err
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("PyroidError", py.get_type::<PyroidError>())?;
    m.add("InputError", input_error_type(py)?)?;
    Ok(())
}
