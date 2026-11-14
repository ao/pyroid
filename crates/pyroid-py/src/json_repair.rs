// Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
//! `pyroid._pyroid.json_repair`: native entry points behind `pyroid.json_repair`.
//!
//! The typed shim (`python/pyroid/json_repair.py`) owns json_repair's
//! signatures, argument checks and delegation; this module runs the kernel and
//! converts results. Strings with lone surrogates (which `&str` cannot hold)
//! go through the kernel's char API via Python's `"surrogatepass"` UTF-8.

use std::borrow::Cow;
use std::sync::OnceLock;

use pyo3::exceptions::{PyAssertionError, PyRecursionError, PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use pyroid_core::json_repair::chars::{self, CharProps, UnicodeVersion};
use pyroid_core::json_repair::dumps::{DumpError, DumpOptions};
use pyroid_core::json_repair::value::{Int, ObjectIntoIter, Value};
use pyroid_core::json_repair::{
    self as kernel, LogEntry, MAX_DEPTH, ParseError, RepairError, RepairOptions, Repaired,
};
use pyroid_core::par;

use crate::batch::{BatchError, SINGLE_DOC_DETACH_BYTES, try_transform};
use crate::errors::{PyroidError, input_error, type_name, with_note};
use crate::input::{Threads, arg_note, text_arg};

/// The interpreter's Unicode tables (set once by the shim at import).
static PROPS: OnceLock<CharProps> = OnceLock::new();

fn props() -> CharProps {
    *PROPS.get_or_init(|| CharProps::new(*UnicodeVersion::ALL.last().expect("non-empty")))
}

/// The Unicode versions the kernel has tables for.
#[pyfunction]
fn supported_unicode_versions() -> Vec<&'static str> {
    UnicodeVersion::ALL.iter().map(|v| v.as_str()).collect()
}

/// Select the Unicode tables (`unicodedata.unidata_version`); once per process.
#[pyfunction]
fn _set_unicode_version(version: &str) -> PyResult<()> {
    let v = UnicodeVersion::from_unidata_version(version)
        .ok_or_else(|| PyValueError::new_err(format!("no tables for Unicode {version}")))?;
    let _ = PROPS.set(CharProps::new(v));
    Ok(())
}

/// The `pyroid.json_repair.NestingError` class (`PyroidError`, `ValueError`,
/// `RecursionError`: json_repair raises one or the other past its depth).
static NESTING_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

fn nesting_error_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    NESTING_ERROR
        .get_or_try_init(py, || {
            let ns = PyDict::new(py);
            ns.set_item("__module__", "pyroid.json_repair")?;
            ns.set_item(
                "__doc__",
                format!(
                    "The input nests deeper than pyroid.json_repair supports ({MAX_DEPTH} levels)."
                ),
            )?;
            let bases = PyTuple::new(
                py,
                [
                    py.get_type::<PyroidError>(),
                    py.get_type::<PyValueError>(),
                    py.get_type::<PyRecursionError>(),
                ],
            )?;
            let ty = py.get_type::<PyType>().call1(("NestingError", bases, ns))?;
            Ok(ty.cast_into::<PyType>()?.unbind())
        })
        .map(|t| t.bind(py))
}

/// `json.dumps` options, validated once per distinct set by the shim.
#[pyclass(frozen, module = "pyroid.json_repair", name = "_DumpConfig")]
#[derive(Debug)]
pub struct DumpConfig(DumpOptions);

#[pymethods]
impl DumpConfig {
    #[new]
    #[pyo3(signature = (*, ensure_ascii, allow_nan, sort_keys, indent, item_separator, key_separator))]
    fn new(
        py: Python<'_>,
        ensure_ascii: bool,
        allow_nan: bool,
        sort_keys: bool,
        indent: Option<String>,
        item_separator: String,
        key_separator: String,
    ) -> Self {
        Self(DumpOptions {
            ensure_ascii,
            allow_nan,
            sort_keys,
            indent,
            item_separator,
            key_separator,
            python_3_11: py.version_info() < (3, 12),
        })
    }

    fn __repr__(&self) -> String {
        format!("<pyroid.json_repair dump config {:?}>", self.0)
    }
}

/// What `repair` returns (1 = the object).
const MODE_STRING: u8 = 0;
const MODE_LOGGING: u8 = 2;

/// A Python str for kernel text (remapped surrogates become real ones).
fn py_str<'py>(py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyString>> {
    if !s.chars().any(chars::is_surrogate) {
        return Ok(PyString::new(py, s));
    }
    let bytes = chars::to_wtf8(s);
    PyString::from_encoded_object(
        PyBytes::new(py, &bytes).as_any(),
        Some(c"utf-8"),
        Some(c"surrogatepass"),
    )
}

fn py_int<'py>(py: Python<'py>, i: &Int) -> PyResult<Bound<'py, PyAny>> {
    match i {
        Int::Small(v) => Ok(v.into_pyobject(py)?.into_any()),
        Int::Big(s) => py.get_type::<PyInt>().call1((&**s,)),
    }
}

/// One container being filled while converting a value tree.
enum Frame<'py> {
    List {
        items: std::vec::IntoIter<Value>,
        out: Vec<Bound<'py, PyAny>>,
    },
    Dict {
        items: ObjectIntoIter,
        out: Bound<'py, PyDict>,
        key: Option<Bound<'py, PyString>>,
    },
}

fn scalar_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => py_int(py, i)?,
        Value::Float(f) => PyFloat::new(py, *f).into_any(),
        Value::Str(s) => py_str(py, s)?.into_any(),
        Value::Array(_) | Value::Object(_) => unreachable!("containers are frames"),
    })
}

/// The Python object for `value`, built iteratively (deep trees cannot
/// overflow the stack) and consuming the tree as it goes.
fn to_py(py: Python<'_>, value: Value) -> PyResult<Bound<'_, PyAny>> {
    let mut stack: Vec<Frame<'_>> = Vec::new();
    let mut pending = Some(value);
    loop {
        let mut finished = None;
        match pending.take() {
            Some(Value::Array(items)) => stack.push(Frame::List {
                out: Vec::with_capacity(items.len()),
                items: items.into_iter(),
            }),
            Some(Value::Object(obj)) => stack.push(Frame::Dict {
                items: obj.into_iter(),
                out: PyDict::new(py),
                key: None,
            }),
            Some(scalar) => finished = Some(scalar_to_py(py, &scalar)?),
            None => {}
        }
        if finished.is_none() {
            let top = stack.last_mut().expect("a frame is open");
            match top {
                Frame::List { items, out } => {
                    if let Some(v) = items.next() {
                        pending = Some(v);
                        continue;
                    }
                    finished = Some(PyList::new(py, std::mem::take(out))?.into_any());
                }
                Frame::Dict { items, out, key } => {
                    if let Some((k, v)) = items.next() {
                        *key = Some(py_str(py, &k)?);
                        pending = Some(v);
                        continue;
                    }
                    finished = Some(out.clone().into_any());
                }
            }
            stack.pop();
        }
        let obj = finished.expect("set above");
        match stack.last_mut() {
            None => return Ok(obj),
            Some(Frame::List { out, .. }) => out.push(obj),
            Some(Frame::Dict { out, key, .. }) => {
                out.set_item(key.take().expect("key before value"), obj)?;
            }
        }
    }
}

fn log_to_py(py: Python<'_>, log: Vec<LogEntry>) -> PyResult<Bound<'_, PyList>> {
    let items = log
        .into_iter()
        .map(|e| {
            let d = PyDict::new(py);
            d.set_item("text", py_str(py, &e.text)?)?;
            d.set_item("context", py_str(py, &e.context)?)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// The Python exception for a kernel error (`None`: delegate to json_repair).
fn repair_err(py: Python<'_>, e: RepairError) -> Option<PyErr> {
    Some(match e {
        RepairError::Parse(ParseError::Strict(msg)) => PyValueError::new_err(msg),
        RepairError::Parse(ParseError::Int(e)) => PyValueError::new_err(e.to_string()),
        RepairError::Parse(ParseError::Assertion) => PyAssertionError::new_err(()),
        RepairError::Parse(ParseError::Depth | ParseError::NeedStack) => {
            match nesting_error_type(py) {
                Ok(ty) => PyErr::from_type(
                    ty.clone(),
                    "Input nesting exceeds the supported parser recursion depth.",
                ),
                Err(e) => e,
            }
        }
        RepairError::Dump(DumpError::OutOfRangeFloat(msg)) => PyValueError::new_err(msg),
        RepairError::Unsupported | RepairError::Parse(ParseError::Unsupported) => return None,
    })
}

/// Kernel options from the shim's arguments.
#[allow(clippy::fn_params_excessive_bools)]
fn options(
    skip_json_loads: bool,
    logging: bool,
    stream_stable: bool,
    strict: bool,
    from_file: bool,
    max_int_digits: usize,
) -> RepairOptions {
    RepairOptions {
        skip_json_loads,
        logging,
        stream_stable,
        strict,
        try_valid_json_suffix: !from_file,
        max_int_digits,
        props: props(),
    }
}

/// `str.encode("utf-8", "surrogatepass")`, called unbound.
fn surrogatepass_bytes<'py>(text: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyBytes>> {
    let py = text.py();
    Ok(py
        .get_type::<PyString>()
        .call_method1("encode", (text, "utf-8", "surrogatepass"))?
        .cast_into::<PyBytes>()?)
}

/// The kernel text for a Python str: its UTF-8, or (with lone surrogates) the
/// remapped chars.
enum Text<'py> {
    Utf8(&'py str),
    Chars(Vec<char>),
    /// Holds a char reserved for remapped surrogates: delegate.
    Unsupported,
}

fn text_of<'a>(text: &'a Bound<'_, PyString>) -> PyResult<Text<'a>> {
    let py = text.py();
    match text.to_str() {
        Ok(s) => Ok(Text::Utf8(s)),
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            let bytes = surrogatepass_bytes(text)?;
            match chars::chars_from_wtf8(bytes.as_bytes()) {
                Some(Ok(c)) => Ok(Text::Chars(c)),
                Some(Err(chars::InputError::ReservedCodePoint)) => Ok(Text::Unsupported),
                None => Err(PyValueError::new_err("invalid surrogatepass UTF-8")),
            }
        }
        Err(e) => Err(e),
    }
}

fn run_kernel(text: &Text<'_>, opts: &RepairOptions) -> Result<Repaired, RepairError> {
    match text {
        Text::Utf8(s) => kernel::repair(s, opts),
        Text::Chars(c) => kernel::repair_chars(c, opts),
        Text::Unsupported => Err(RepairError::Unsupported),
    }
}

/// `repair_json(text, ...)` for the shim: `mode` 0 = the string result,
/// 1 = the object (`return_objects`), 2 = `(object, log)` (`logging`).
/// Returns `NotImplemented` when the call must be delegated to json_repair.
#[pyfunction]
#[pyo3(signature = (text, mode, skip_json_loads, stream_stable, strict, from_file, max_int_digits, dump))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn repair<'py>(
    text: &Bound<'py, PyString>,
    mode: u8,
    skip_json_loads: bool,
    stream_stable: bool,
    strict: bool,
    from_file: bool,
    max_int_digits: usize,
    dump: Option<&DumpConfig>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = text.py();
    let opts = options(
        skip_json_loads,
        mode == MODE_LOGGING,
        stream_stable,
        strict,
        from_file,
        max_int_digits,
    );
    let t = text_of(text)?;
    let len = match &t {
        Text::Utf8(s) => s.len(),
        Text::Chars(c) => c.len(),
        Text::Unsupported => 0,
    };
    let detach = len >= SINGLE_DOC_DETACH_BYTES;
    if mode == MODE_STRING {
        let dump = dump.map_or_else(DumpOptions::default, |d| d.0.clone());
        let work = || {
            run_kernel(&t, &opts).and_then(|r| {
                let out = kernel::dumps_result(&r, &dump);
                kernel::value::drop_deep(r.value);
                out
            })
        };
        let out = if detach { py.detach(work) } else { work() };
        return match out {
            Ok(s) => Ok(py_str(py, &s)?.into_any()),
            Err(e) => match repair_err(py, e) {
                Some(err) => Err(err),
                None => Ok(py.NotImplemented().into_bound(py)),
            },
        };
    }
    let out = if detach {
        py.detach(|| run_kernel(&t, &opts))
    } else {
        run_kernel(&t, &opts)
    };
    let r = match out {
        Ok(r) => r,
        Err(e) => {
            return match repair_err(py, e) {
                Some(err) => Err(err),
                None => Ok(py.NotImplemented().into_bound(py)),
            };
        }
    };
    let obj = to_py(py, r.value)?;
    if mode == MODE_LOGGING {
        let log = log_to_py(py, r.log.unwrap_or_default())?;
        return Ok(PyTuple::new(py, [obj, log.into_any()])?.into_any());
    }
    Ok(obj)
}

/// One document's batch result.
enum BatchOut {
    Same,
    New(String),
    Failed(RepairError),
}

/// `repair_json(doc)` for every document of a str, list or tuple on the shared
/// pool. Returns `(result, unsupported)`: the result in the input's kind (an
/// unchanged document is the original object; delegated ones are `None`) and
/// the indices the shim must delegate to json_repair. For Arrow input,
/// `delegate(doc)` gives those documents' results (`unsupported` is empty).
#[pyfunction]
#[pyo3(signature = (texts, /, skip_json_loads, stream_stable, strict, max_int_digits, dump, *, threads=None, delegate=None))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn repair_batch<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    skip_json_loads: bool,
    stream_stable: bool,
    strict: bool,
    max_int_digits: usize,
    dump: Option<&DumpConfig>,
    threads: Option<Threads>,
    delegate: Option<Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyAny>, Vec<usize>)> {
    let opts = options(
        skip_json_loads,
        false,
        stream_stable,
        strict,
        false,
        max_int_digits,
    );
    let dump = dump.map_or_else(DumpOptions::default, |d| d.0.clone());
    let one = |t: &Text<'_>, original: Option<&str>| -> BatchOut {
        match run_kernel(t, &opts).and_then(|r| {
            let out = kernel::dumps_result(&r, &dump);
            kernel::value::drop_deep(r.value);
            out
        }) {
            Ok(s) if original == Some(s.as_str()) => BatchOut::Same,
            Ok(s) => BatchOut::New(s),
            Err(e) => BatchOut::Failed(e),
        }
    };
    let finish = |outs: Vec<BatchOut>,
                  originals: &[Bound<'py, PyAny>]|
     -> PyResult<(Vec<Bound<'py, PyAny>>, Vec<usize>)> {
        let mut items = Vec::with_capacity(outs.len());
        let mut unsupported = Vec::new();
        for (i, (out, orig)) in outs.into_iter().zip(originals).enumerate() {
            items.push(match out {
                BatchOut::Same => orig.clone(),
                BatchOut::New(s) => py_str(py, &s)?.into_any(),
                BatchOut::Failed(e) => {
                    if let Some(err) = repair_err(py, e) {
                        return Err(with_note(
                            py,
                            err,
                            format!("pyroid: while repairing the document at index {i}"),
                        ));
                    }
                    unsupported.push(i);
                    py.None().into_bound(py)
                }
            });
        }
        Ok((items, unsupported))
    };

    if let Ok(s) = texts.cast::<PyString>() {
        let t = text_of(s)?;
        let original = match &t {
            Text::Utf8(u) => Some(*u),
            _ => None,
        };
        let out = one(&t, original);
        let (mut items, unsupported) = finish(vec![out], &[s.clone().into_any()])?;
        return Ok((items.pop().expect("one"), unsupported));
    }
    let is_list = texts.is_instance_of::<PyList>();
    if !is_list && !texts.is_instance_of::<PyTuple>() {
        // Arrow input: strings in, `pyroid.Column` out.
        return arrow_batch(py, texts, &opts, &dump, threads, delegate.as_ref())
            .map(|c| (c, Vec::new()));
    }
    let items: Vec<Bound<'py, PyAny>> = if is_list {
        texts.cast::<PyList>()?.iter().collect()
    } else {
        texts.cast::<PyTuple>()?.iter().collect()
    };
    let mut docs: Vec<Text<'_>> = Vec::with_capacity(items.len());
    let strs: Vec<Bound<'py, PyString>> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            item.cast::<PyString>().cloned().map_err(|_| {
                let err = input_error(
                    py,
                    format!("expected str at index {i}, got {}", type_name(item)),
                );
                arg_note(py, err, "texts")
            })
        })
        .collect::<PyResult<_>>()?;
    for s in &strs {
        docs.push(text_of(s)?);
    }
    let t = crate::input::threads(threads);
    let results = py
        .detach(|| {
            par::map(docs.len(), t, |i| {
                let original = match &docs[i] {
                    Text::Utf8(u) => Some(*u),
                    _ => None,
                };
                one(&docs[i], original)
            })
        })
        .map_err(|e| BatchError::from(e).into_pyerr(py))?;
    let (items, unsupported) = finish(results, &items)?;
    let result = if is_list {
        PyList::new(py, items)?.into_any()
    } else {
        PyTuple::new(py, items)?.into_any()
    };
    Ok((result, unsupported))
}

/// `repair_batch` for Arrow string input (a `pyroid.Column` of strings out).
/// A document the port does not handle goes to `delegate(doc)` (json_repair's
/// `repair_json` with the call's options), when given.
fn arrow_batch<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    opts: &RepairOptions,
    dump: &DumpOptions,
    threads_arg: Option<Threads>,
    delegate: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let input = text_arg(texts, threads_arg, "texts")?;
    try_transform(
        py,
        input,
        threads_arg,
        |s| match kernel::repair_json(s, opts, dump) {
            Ok(out) if out == s => Ok(Cow::Borrowed(s)),
            // The value holds a lone surrogate (stored remapped), which a
            // UTF-8 Arrow column cannot hold.
            Ok(out) if chars::contains_reserved(&out) => Err(ArrowFailure::Surrogate(out)),
            Ok(out) => Ok(Cow::Owned(out)),
            Err(e) => Err(ArrowFailure::Repair(e)),
        },
        |py, row, text, failure| match (failure, delegate) {
            (ArrowFailure::Repair(e), Some(delegate)) if is_unsupported(&e) => {
                delegated_row(py, delegate, row, text)
            }
            (failure, _) => Err(arrow_error(py, row, failure)),
        },
    )
}

fn is_unsupported(e: &RepairError) -> bool {
    matches!(
        e,
        RepairError::Unsupported | RepairError::Parse(ParseError::Unsupported)
    )
}

/// json_repair's result for one Arrow row, from the shim's `delegate`.
fn delegated_row(
    py: Python<'_>,
    delegate: &Bound<'_, PyAny>,
    row: usize,
    text: &str,
) -> PyResult<String> {
    let note = |e: PyErr| {
        with_note(
            py,
            e,
            format!("pyroid: while repairing row {row} of the Arrow input"),
        )
    };
    let out = delegate.call1((text,)).map_err(note)?;
    let out = out.cast_into::<PyString>().map_err(|e| note(e.into()))?;
    match out.to_str() {
        Ok(s) => Ok(s.to_owned()),
        Err(e) => Err(surrogate_note(py, row, e)),
    }
}

/// Why one row of an Arrow batch has no Arrow result.
enum ArrowFailure {
    Repair(RepairError),
    /// json_repair's result, which holds a lone surrogate.
    Surrogate(String),
}

fn surrogate_note(py: Python<'_>, row: usize, err: PyErr) -> PyErr {
    with_note(
        py,
        err,
        format!(
            "pyroid: row {row} of the Arrow input repairs to a str with a lone surrogate (an \
             escape such as \\ud800 kept as is), which an Arrow string column cannot hold; pass \
             a list to get the str, or ensure_ascii=True to keep the escape"
        ),
    )
}

fn arrow_error(py: Python<'_>, row: usize, failure: ArrowFailure) -> PyErr {
    match failure {
        ArrowFailure::Repair(e) => {
            let err = repair_err(py, e).unwrap_or_else(|| {
                PyroidError::new_err(
                    "pyroid.json_repair.repair_json_batch: a document holds a char in \
                     U+10F800..U+10FFFF, which the Rust port does not handle",
                )
            });
            with_note(
                py,
                err,
                format!("pyroid: while repairing row {row} of the Arrow input"),
            )
        }
        ArrowFailure::Surrogate(out) => {
            // The UnicodeEncodeError that encoding json_repair's str gives.
            let err = match py_str(py, &out).and_then(|s| s.to_str().map(|_| ())) {
                Err(e) => e,
                Ok(()) => PyValueError::new_err("unreachable: no lone surrogate"),
            };
            surrogate_note(py, row, err)
        }
    }
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "pyroid.json_repair")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(supported_unicode_versions, &m)?)?;
    m.add_function(wrap_pyfunction!(_set_unicode_version, &m)?)?;
    m.add_function(wrap_pyfunction!(repair, &m)?)?;
    m.add_function(wrap_pyfunction!(repair_batch, &m)?)?;
    m.add_class::<DumpConfig>()?;
    m.add("NestingError", nesting_error_type(py)?)?;
    m.add("MAX_DEPTH", MAX_DEPTH)?;
    parent.add_submodule(&m)
}
