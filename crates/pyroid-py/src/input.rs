//! Argument extraction shared by every text function: [`TextInput`] (the
//! documents) and [`Threads`] (the `threads=` keyword).

use std::num::NonZeroUsize;

use pyo3::exceptions::{PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyList, PyMemoryView, PyString, PyTuple};
use pyroid_core::par;

use crate::arrow::{self, ArrowStrings};
use crate::errors::{input_error, type_name, with_note};

const EXPECTED: &str = "expected str, list[str], tuple[str, ...] or an Arrow string array/column";

/// The documents a text function was called with, in the caller's container
/// kind. Extracting it never copies text:
/// - `str`, `list[str]`, `tuple[str, ...]` become [`PyBackedStr`]s, which
///   borrow each str's UTF-8 (CPython caches it on the str for non-ASCII) and
///   convert back into the *original* object;
/// - anything with `__arrow_c_stream__` / `__arrow_c_array__` (pyarrow,
///   polars, pandas, HF datasets, nanoarrow, arro3, duckdb, `pyroid.Column`)
///   is imported zero-copy through the Arrow PyCapsule interface.
///
/// Every variant is `Sync`, so `&TextInput` can be read inside
/// `Python::detach` from many threads. It is not `Send`: Arrow inputs must be
/// released on the thread that imported them, while attached.
pub enum TextInput {
    Str(PyBackedStr),
    List(Vec<PyBackedStr>),
    Tuple(Vec<PyBackedStr>),
    Arrow(ArrowStrings),
}

/// `PyBackedStr` for one str. Lone surrogates can't be encoded as UTF-8: the
/// `UnicodeEncodeError` is kept, with a note saying which item it was.
fn backed(s: &Bound<'_, PyString>, index: Option<usize>) -> PyResult<PyBackedStr> {
    PyBackedStr::try_from(s.clone()).map_err(|err| {
        let py = s.py();
        if !err.is_instance_of::<PyUnicodeEncodeError>(py) {
            return err;
        }
        let which = index.map_or_else(|| "the str".to_owned(), |i| format!("the str at index {i}"));
        with_note(
            py,
            err,
            format!(
                "pyroid: {which} contains a lone surrogate (U+D800..U+DFFF), which has no \
                 UTF-8 encoding. Remove or replace it first, e.g. \
                 s.encode('utf-8', 'replace').decode('utf-8')."
            ),
        )
    })
}

fn docs<'py>(
    items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Vec<PyBackedStr>> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.enumerate() {
        let Ok(s) = item.cast::<PyString>() else {
            return Err(input_error(
                item.py(),
                format!("expected str at index {i}, got {}", type_name(&item)),
            ));
        };
        out.push(backed(s, Some(i))?);
    }
    Ok(out)
}

impl<'a, 'py> FromPyObject<'a, 'py> for TextInput {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        if let Ok(s) = obj.cast::<PyString>() {
            return Ok(Self::Str(backed(&s, None)?));
        }
        if let Ok(list) = obj.cast::<PyList>() {
            return Ok(Self::List(docs(list.iter())?));
        }
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            return Ok(Self::Tuple(docs(tuple.iter())?));
        }
        if obj.is_instance_of::<PyBytes>()
            || obj.is_instance_of::<PyByteArray>()
            || obj.is_instance_of::<PyMemoryView>()
        {
            return Err(input_error(
                py,
                format!(
                    "{EXPECTED}, got {}; decode bytes to str first, e.g. data.decode('utf-8')",
                    type_name(&obj)
                ),
            ));
        }
        if let Some(strings) = arrow::import(&obj)? {
            return Ok(Self::Arrow(strings));
        }
        Err(input_error(
            py,
            format!(
                "{EXPECTED}, got {}; convert other iterables with list(...)",
                type_name(&obj)
            ),
        ))
    }
}

/// The `threads=` keyword: `None` (all of the shared pool) or a positive int
/// (at most that many tasks at once; `1` runs on the calling thread).
/// Functions take it as `Option<Threads>` with a `None` default.
#[derive(Clone, Copy, Debug)]
pub struct Threads(pub par::Threads);

impl<'a, 'py> FromPyObject<'a, 'py> for Threads {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let bad_type = || {
            input_error(
                py,
                format!(
                    "threads must be None or a positive int, got {}",
                    type_name(&obj)
                ),
            )
        };
        if obj.is_instance_of::<PyBool>() {
            return Err(bad_type());
        }
        let n: i64 = obj.extract().map_err(|_| bad_type())?;
        let n = usize::try_from(n)
            .ok()
            .and_then(NonZeroUsize::new)
            .ok_or_else(|| {
                PyValueError::new_err(format!("threads must be None or a positive int, got {n}"))
            })?;
        Ok(Self(par::Threads::Max(n)))
    }
}

/// The pool setting for an optional `threads=` argument.
pub fn threads(t: Option<Threads>) -> par::Threads {
    t.map_or(par::Threads::Auto, |t| t.0)
}

/// A non-negative int argument (e.g. `min_chars(x, n)`), with a clear error.
#[derive(Clone, Copy, Debug)]
pub struct Count(pub usize);

impl<'a, 'py> FromPyObject<'a, 'py> for Count {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        if obj.is_instance_of::<PyBool>() {
            return Err(input_error(py, "expected a non-negative int, got bool"));
        }
        let n: i64 = obj.extract().map_err(|_| {
            input_error(
                py,
                format!("expected a non-negative int, got {}", type_name(&obj)),
            )
        })?;
        usize::try_from(n)
            .map(Self)
            .map_err(|_| PyValueError::new_err(format!("expected a non-negative int, got {n}")))
    }
}
