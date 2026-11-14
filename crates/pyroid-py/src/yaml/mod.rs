// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! `pyroid._pyroid.yaml`: native entry points behind `pyroid.yaml`.
//!
//! The Python module (`python/pyroid/yaml/__init__.py`) owns PyYAML's signatures, the
//! choice between the native path and PyYAML itself, file objects and UTF-16 input. This
//! module loads (constructs Python objects from the core's node graphs, in
//! `SafeConstructor`'s order) and dumps (represents objects as nodes, then serializes and
//! emits them in the core). Errors are raised as the exception classes the Python module
//! registers with `_configure` (PyYAML's own when it is installed).

mod dump;
mod load;

use pyo3::exceptions::{PyRecursionError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyString, PyTuple};
use pyroid_core::yaml::error::{Arg, Mark, Msg, ReaderError, YamlError};

/// What `_configure` registered: PyYAML's classes (or pyroid's copies) and helpers.
#[derive(Debug)]
struct Classes {
    mark: Py<PyAny>,
    reader_error: Py<PyAny>,
    scanner_error: Py<PyAny>,
    parser_error: Py<PyAny>,
    composer_error: Py<PyAny>,
    constructor_error: Py<PyAny>,
    representer_error: Py<PyAny>,
    emitter_error: Py<PyAny>,
    /// `base64.encodebytes` (`represent_binary`).
    encodebytes: Py<PyAny>,
    /// `SafeConstructor.nan_value`: one shared float object, as in PyYAML.
    nan: Py<PyAny>,
    date: Py<PyAny>,
    datetime: Py<PyAny>,
    timedelta: Py<PyAny>,
    timezone: Py<PyAny>,
    utc: Py<PyAny>,
    /// `str.translate` table mapping U+10F800.. back to lone surrogates.
    surrogates: Py<PyAny>,
    /// PyYAML's constructor code for the slow paths: `{"int": f, "float": f, "bool": f,
    /// "timestamp": f, "binary": f}`, each taking the scalar's value.
    helpers: Py<PyAny>,
}

static CLASSES: PyOnceLock<Classes> = PyOnceLock::new();

fn classes(py: Python<'_>) -> PyResult<&Classes> {
    CLASSES
        .get(py)
        .ok_or_else(|| PyValueError::new_err("pyroid.yaml is not configured (import pyroid.yaml)"))
}

/// Register the classes and helpers (`pyroid.yaml` does it once at import).
#[pyfunction]
fn _configure(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<()> {
    let get = |k: &str| -> PyResult<Py<PyAny>> {
        config
            .get_item(k)?
            .map(Bound::unbind)
            .ok_or_else(|| PyValueError::new_err(format!("missing {k}")))
    };
    let c = Classes {
        mark: get("Mark")?,
        reader_error: get("ReaderError")?,
        scanner_error: get("ScannerError")?,
        parser_error: get("ParserError")?,
        composer_error: get("ComposerError")?,
        constructor_error: get("ConstructorError")?,
        representer_error: get("RepresenterError")?,
        emitter_error: get("EmitterError")?,
        encodebytes: get("encodebytes")?,
        nan: get("nan")?,
        date: get("date")?,
        datetime: get("datetime")?,
        timedelta: get("timedelta")?,
        timezone: get("timezone")?,
        utc: get("utc")?,
        surrogates: get("surrogates")?,
        helpers: get("helpers")?,
    };
    let _ = CLASSES.set(py, c);
    Ok(())
}

/// Where marks point: the stream's name and, for `str`/`bytes` input, its text (PyYAML's
/// `Mark.buffer` is the decoded stream plus `'\0'`; file objects have none).
struct Source<'a, 'py> {
    name: &'a Bound<'py, PyAny>,
    buffer: Option<&'a str>,
    /// The `str` built for `Mark.buffer`, made on the first error.
    cached: std::cell::OnceCell<Bound<'py, PyAny>>,
}

impl<'a, 'py> Source<'a, 'py> {
    fn new(name: &'a Bound<'py, PyAny>, buffer: Option<&'a str>) -> Self {
        Self {
            name,
            buffer,
            cached: std::cell::OnceCell::new(),
        }
    }

    fn mark(&self, py: Python<'py>, m: Mark) -> PyResult<Bound<'py, PyAny>> {
        let c = classes(py)?;
        let (buffer, pointer) = match self.buffer {
            Some(text) => {
                let b = if let Some(b) = self.cached.get() {
                    b.clone()
                } else {
                    let mut s = String::with_capacity(text.len() + 1);
                    s.push_str(text);
                    s.push('\0');
                    let b = PyString::new(py, &s).into_any();
                    let _ = self.cached.set(b.clone());
                    b
                };
                (b, m.index.into_pyobject(py)?.into_any())
            }
            None => (py.None().into_bound(py), py.None().into_bound(py)),
        };
        c.mark
            .bind(py)
            .call1((self.name, m.index, m.line, m.column, buffer, pointer))
    }

    fn opt_mark(&self, py: Python<'py>, m: Option<Mark>) -> PyResult<Bound<'py, PyAny>> {
        match m {
            Some(m) => self.mark(py, m),
            None => Ok(py.None().into_bound(py)),
        }
    }
}

/// `fmt % tuple(args)` (or `fmt`), as PyYAML builds its messages.
fn format_msg<'py>(py: Python<'py>, msg: &Msg) -> PyResult<Bound<'py, PyAny>> {
    let fmt = PyString::new(py, msg.fmt);
    if msg.args.is_empty() {
        return Ok(fmt.into_any());
    }
    let mut args = Vec::with_capacity(msg.args.len());
    for a in &msg.args {
        args.push(match a {
            Arg::Str(s) => PyString::new(py, s).into_any(),
            Arg::Int(n) => n.into_pyobject(py)?.into_any(),
            Arg::Utf8DecodeError(bytes) => {
                match PyBytes::new(py, bytes).call_method1("decode", ("utf-8",)) {
                    Ok(s) => s.str()?.into_any(), // not an error after all
                    Err(e) => e.value(py).str()?.into_any(),
                }
            }
        });
    }
    fmt.call_method1("__mod__", (PyTuple::new(py, args)?,))
}

fn opt_msg<'py>(py: Python<'py>, msg: Option<&Msg>) -> PyResult<Bound<'py, PyAny>> {
    match msg {
        Some(m) => format_msg(py, m),
        None => Ok(py.None().into_bound(py)),
    }
}

/// A `str` from a scalar that may carry lone surrogates in U+10F800.. (the scanner maps
/// `\uD800`-style escapes there; inputs holding such characters themselves go to PyYAML).
fn restore_surrogates<'py>(py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyAny>> {
    let out = PyString::new(py, s).into_any();
    if pyroid_core::yaml::reader::has_reserved(s) {
        let table = &classes(py)?.surrogates;
        return out.call_method1("translate", (table.bind(py),));
    }
    Ok(out)
}

fn reader_error(py: Python<'_>, name: &Bound<'_, PyAny>, e: &ReaderError) -> PyErr {
    let r = (|| -> PyResult<PyErr> {
        let cls = classes(py)?.reader_error.bind(py);
        let exc = cls.call1((name, e.position, e.character, e.encoding, e.reason))?;
        Ok(PyErr::from_value(exc))
    })();
    r.unwrap_or_else(|e| e)
}

/// The Python exception for a core error. `pending`: the reader error a file object hit
/// further on ([`YamlError::NeedMore`]).
fn to_pyerr<'py>(
    py: Python<'py>,
    err: YamlError,
    src: &Source<'_, 'py>,
    pending: Option<&Py<PyAny>>,
) -> PyErr {
    let r = (|| -> PyResult<PyErr> {
        Ok(match err {
            YamlError::Reader(e) => reader_error(py, src.name, &e),
            YamlError::Marked(m) => {
                let c = classes(py)?;
                let cls = match m.kind {
                    pyroid_core::yaml::error::ErrorKind::Scanner => &c.scanner_error,
                    pyroid_core::yaml::error::ErrorKind::Parser => &c.parser_error,
                    pyroid_core::yaml::error::ErrorKind::Composer => &c.composer_error,
                    pyroid_core::yaml::error::ErrorKind::Constructor => &c.constructor_error,
                };
                let exc = cls.bind(py).call1((
                    opt_msg(py, m.context.as_ref())?,
                    src.opt_mark(py, m.context_mark)?,
                    opt_msg(py, m.problem.as_ref())?,
                    src.opt_mark(py, m.problem_mark)?,
                ))?;
                PyErr::from_value(exc)
            }
            YamlError::Value(msg) => PyValueError::new_err(msg),
            YamlError::Chr(code) => match py.import("builtins")?.getattr("chr")?.call1((code,)) {
                Err(e) => e,
                Ok(_) => PyValueError::new_err("chr() arg not in range(0x110000)"),
            },
            YamlError::Type(msg) => pyo3::exceptions::PyTypeError::new_err(msg),
            YamlError::Recursion => PyRecursionError::new_err("maximum recursion depth exceeded"),
            YamlError::NeedMore => match pending {
                Some(p) => PyErr::from_value(p.bind(py).clone()),
                None => PyValueError::new_err("pyroid.yaml: the reader ran out of input"),
            },
            YamlError::Emitter(m) => {
                let exc = classes(py)?
                    .emitter_error
                    .bind(py)
                    .call1((format_msg(py, &m)?,))?;
                PyErr::from_value(exc)
            }
        })
    })();
    r.unwrap_or_else(|e| e)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "pyroid.yaml")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(_configure, &m)?)?;
    load::register(&m)?;
    dump::register(&m)?;
    parent.add_submodule(&m)
}
