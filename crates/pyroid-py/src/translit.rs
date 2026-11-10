//! `pyroid._pyroid.translit`: native entry points behind `pyroid.translit`
//! (Unicode → ASCII from the anyascii 0.3.3 data, ISC). API docs live in
//! `python/pyroid/translit.pyi`.
//!
//! Strs holding lone surrogates (which `&str` cannot hold) take a WTF-8 path
//! through Python's `"surrogatepass"` UTF-8, as `pyroid.ftfy` does: anyascii
//! drops lone surrogates, and Unidecode warns about each one and treats it as
//! a character without a transliteration.

use std::borrow::Cow;
use std::ffi::CString;

use pyo3::exceptions::{PyRuntimeWarning, PyTypeError, PyUnicodeEncodeError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyBytes, PyList, PyString, PyTuple, PyType};
use pyroid_core::par;
use pyroid_core::translit::{self as kernel, Missing, MissingAt, MissingWtf8, Unit};

use crate::batch::{BatchError, map_values, predicate, transform};
use crate::errors::{input_error, type_name, with_note};
use crate::input::{TextInput, Threads, arg_note, text_arg, threads};

/// `errors=`: Unidecode checks the value only when it meets a character
/// without a transliteration, so an invalid value is kept until then.
#[derive(Debug)]
enum Mode {
    Ignore,
    Replace,
    Strict,
    Preserve,
    /// `repr()` of the invalid value (a Python str: it may hold anything).
    Invalid(Py<PyString>),
}

#[derive(Debug)]
struct ErrorsArg(Mode);

impl<'a, 'py> FromPyObject<'a, 'py> for ErrorsArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let known = obj.cast::<PyString>().ok().and_then(|s| match s.to_str() {
            Ok("ignore") => Some(Mode::Ignore),
            Ok("replace") => Some(Mode::Replace),
            Ok("strict") => Some(Mode::Strict),
            Ok("preserve") => Some(Mode::Preserve),
            _ => None,
        });
        Ok(Self(match known {
            Some(m) => m,
            None => Mode::Invalid(obj.repr()?.unbind()),
        }))
    }
}

/// `replace_str=`: used only when a character has no transliteration (and
/// only then type-checked, as by Unidecode).
#[derive(Debug)]
enum Replace {
    Utf8(Cow<'static, str>),
    /// A str holding lone surrogates, as surrogatepass UTF-8.
    Wtf8(Vec<u8>),
    /// Not a str: the object, for the `TypeError`.
    NotStr(Py<PyAny>),
}

#[derive(Debug)]
struct ReplaceArg(Replace);

impl<'a, 'py> FromPyObject<'a, 'py> for ReplaceArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let Ok(s) = obj.cast::<PyString>() else {
            return Ok(Self(Replace::NotStr(obj.to_owned().unbind())));
        };
        Ok(Self(match s.to_str() {
            Ok(s) => Replace::Utf8(Cow::Owned(s.to_owned())),
            Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(obj.py()) => {
                Replace::Wtf8(surrogatepass_bytes(&s)?.as_bytes().to_vec())
            }
            Err(e) => return Err(e),
        }))
    }
}

/// `text` as UTF-8 extended with surrogates. Calls `str.encode` unbound, so a
/// subclass overriding `encode` does not interfere.
fn surrogatepass_bytes<'py>(text: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyBytes>> {
    let py = text.py();
    Ok(py
        .get_type::<PyString>()
        .call_method1(
            pyo3::intern!(py, "encode"),
            (text, "utf-8", "surrogatepass"),
        )?
        .cast_into::<PyBytes>()?)
}

fn str_from_wtf8<'py>(py: Python<'py>, b: &[u8]) -> PyResult<Bound<'py, PyString>> {
    match std::str::from_utf8(b) {
        Ok(s) => Ok(PyString::new(py, s)),
        Err(_) => PyString::from_encoded_object(
            PyBytes::new(py, b).as_any(),
            Some(c"utf-8"),
            Some(c"surrogatepass"),
        ),
    }
}

/// `repr()` of one code unit, as Unidecode's messages show it.
fn unit_repr(py: Python<'_>, unit: Unit) -> PyResult<String> {
    let s = match unit {
        Unit::Char(c) => PyString::new(py, c.encode_utf8(&mut [0; 4])),
        Unit::Surrogate(u) => {
            let b = [
                0xE0 | (u >> 12) as u8,
                0x80 | ((u >> 6) & 0x3F) as u8,
                0x80 | (u & 0x3F) as u8,
            ];
            str_from_wtf8(py, &b)?
        }
    };
    Ok(s.repr()?.to_string())
}

static UNIDECODE_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// A `pyroid.translit.UnidecodeError(message, index)`. The class is defined
/// in Python (it has Unidecode's constructor and `index` attribute).
fn unidecode_error(py: Python<'_>, message: &Bound<'_, PyAny>, index: Option<usize>) -> PyErr {
    let ty = UNIDECODE_ERROR.get_or_try_init(py, || {
        let ty = py
            .import("pyroid.translit")?
            .getattr("UnidecodeError")?
            .cast_into::<PyType>()?;
        Ok::<_, PyErr>(ty.unbind())
    });
    match ty.and_then(|t| t.bind(py).call1((message, index))) {
        Ok(err) => PyErr::from_value(err),
        Err(e) => e,
    }
}

/// The `TypeError` `str.join` gives for `obj` as item `index` (Unidecode's
/// error for a non-str `replace_str`). CPython builds it, so the type name is
/// its own C-level `tp_name`, which Python code cannot read under abi3.
fn join_type_error(py: Python<'_>, obj: &Bound<'_, PyAny>, index: usize) -> PyResult<PyErr> {
    let joined =
        PyString::new(py, "").call_method1(pyo3::intern!(py, "join"), (PyTuple::new(py, [obj])?,));
    let err = match joined {
        Err(e) if e.is_instance_of::<PyTypeError>(py) => e,
        Err(e) => return Ok(e),
        Ok(_) => return Ok(PyTypeError::new_err("replace_str must be a str")),
    };
    let msg = err.value(py).str()?.call_method1(
        pyo3::intern!(py, "replace"),
        ("sequence item 0:", format!("sequence item {index}:"), 1),
    )?;
    Ok(PyTypeError::new_err(msg.unbind()))
}

/// The exception Unidecode raises when it meets `missing` in `mode`.
fn missing_error(
    py: Python<'_>,
    mode: &Mode,
    replace: &Replace,
    missing: MissingAt,
) -> PyResult<PyErr> {
    Ok(match (mode, replace) {
        (Mode::Strict, _) => {
            let msg = format!(
                "no replacement found for character {} in position {}",
                unit_repr(py, missing.unit)?,
                missing.index
            );
            unidecode_error(py, PyString::new(py, &msg).as_any(), Some(missing.index))
        }
        (Mode::Invalid(repr), _) => {
            let msg =
                PyString::new(py, "invalid value for errors parameter ").add(repr.bind(py))?;
            unidecode_error(py, &msg, None)
        }
        (Mode::Replace, Replace::NotStr(obj)) => join_type_error(py, obj.bind(py), missing.index)?,
        _ => unreachable!("only strict, invalid errors and a non-str replace_str fail"),
    })
}

/// Is `obj` a str, list or tuple? Only those can hold lone surrogates that
/// `TextInput` rejects; any other `UnicodeEncodeError` (say, from an Arrow
/// producer) is the caller's to see.
fn may_hold_surrogates(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PyString>()
        || obj.is_instance_of::<PyList>()
        || obj.is_instance_of::<PyTuple>()
}

/// Does `mode` (with `replace`) raise at the first character without a
/// transliteration?
fn fails_on_missing(mode: &Mode, replace: &Replace) -> bool {
    match mode {
        Mode::Strict | Mode::Invalid(_) => true,
        Mode::Replace => matches!(replace, Replace::NotStr(_)),
        Mode::Ignore | Mode::Preserve => false,
    }
}

/// Where in a batch a failing document is, for the exception note.
fn doc_note(input: &TextInput, doc: usize) -> Option<String> {
    match input {
        TextInput::Str(_) => None,
        TextInput::List(_) | TextInput::Tuple(_) => {
            Some(format!("pyroid: in the str at index {doc}"))
        }
        TextInput::Arrow(_) => Some(format!("pyroid: in row {doc} of the Arrow input")),
    }
}

/// Transliterate `s` to ASCII, exactly as ``anyascii.anyascii`` 0.3.3 does.
///
/// Accepts a str, a list/tuple of str, or an Arrow string array/column, and
/// returns the same kind; unchanged (pure-ASCII) documents are returned as the
/// original objects. Lone surrogates are dropped, as anyascii drops them.
#[pyfunction]
#[pyo3(signature = (string, *, threads = None))]
fn anyascii<'py>(
    py: Python<'py>,
    string: &Bound<'py, PyAny>,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    match text_arg(string, threads, "string") {
        Ok(input) => transform(py, input, threads, kernel::anyascii),
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) && may_hold_surrogates(string) => {
            let strs = Strs::collect(string)?;
            let outs = strs.map_wtf8(py, threads, |b| {
                (owned(kernel::anyascii_wtf8(b)), Vec::new(), None)
            })?;
            strs.build(py, outs.into_iter().map(|r| r.0))
        }
        Err(e) => Err(e),
    }
}

/// Unidecode's ``unidecode(string, errors, replace_str)`` over anyascii's data.
///
/// The ``errors`` modes, ``replace_str``, the ``UnidecodeError`` (with
/// ``index``) and the lone-surrogate ``RuntimeWarning`` behave as in Unidecode
/// 1.4.0; the transliterations are anyascii's, so they differ from
/// Unidecode's (see ``pyroid/translit.pyi``).
#[pyfunction]
#[pyo3(
    signature = (
        string,
        errors = ErrorsArg(Mode::Ignore),
        replace_str = ReplaceArg(Replace::Utf8(Cow::Borrowed("?"))),
        *,
        threads = None,
    ),
    text_signature = "(string, errors='ignore', replace_str='?', *, threads=None)"
)]
fn unidecode<'py>(
    py: Python<'py>,
    string: &Bound<'py, PyAny>,
    errors: ErrorsArg,
    replace_str: ReplaceArg,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    let (mode, replace) = (errors.0, replace_str.0);
    let input = match text_arg(string, threads, "string") {
        Ok(input) => input,
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) && may_hold_surrogates(string) => {
            return unidecode_wtf8(py, string, &mode, &replace, threads);
        }
        Err(e) => return Err(e),
    };
    if let (Mode::Replace, Replace::Wtf8(rep)) = (&mode, &replace) {
        // The replacement holds a lone surrogate, so the output may too.
        if matches!(input, TextInput::Arrow(_)) {
            // The UnicodeEncodeError a str with a lone surrogate gives.
            let err = match str_from_wtf8(py, rep)?.to_str() {
                Err(e) => e,
                Ok(_) => PyTypeError::new_err("replace_str"),
            };
            return Err(with_note(
                py,
                err,
                "pyroid: replace_str holds a lone surrogate, which an Arrow string column \
                 cannot hold",
            ));
        }
        drop(input);
        return unidecode_wtf8(py, string, &mode, &replace, threads);
    }
    if fails_on_missing(&mode, &replace) {
        let found = map_values(py, &input, threads, kernel::first_missing)?;
        if let Some((doc, m)) = found
            .into_iter()
            .enumerate()
            .find_map(|(i, m)| m.flatten().map(|m| (i, m)))
        {
            let err = missing_error(py, &mode, &replace, m)?;
            return Err(match doc_note(&input, doc) {
                Some(note) => with_note(py, err, note),
                None => err,
            });
        }
        // Nothing is missing, so every mode gives the same output.
        return transform(py, input, threads, |s| kernel::unidecode(s, Missing::Drop));
    }
    match (&mode, &replace) {
        (Mode::Preserve, _) => {
            transform(py, input, threads, |s| kernel::unidecode(s, Missing::Keep))
        }
        (Mode::Replace, Replace::Utf8(r)) => {
            let r: &str = r;
            transform(py, input, threads, |s| {
                kernel::unidecode(s, Missing::Replace(r))
            })
        }
        _ => transform(py, input, threads, |s| kernel::unidecode(s, Missing::Drop)),
    }
}

/// The strs of a str, list or tuple argument, for the WTF-8 path.
struct Strs<'py> {
    kind: Kind,
    items: Vec<Bound<'py, PyString>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Str,
    List,
    Tuple,
}

/// Per-document result of the WTF-8 path: the output (`None`: unchanged, the
/// original object), the lone surrogates met (position, code unit), and the
/// first character without a transliteration (when the mode stops there).
type Wtf8Result = (Option<Vec<u8>>, Vec<(usize, u16)>, Option<MissingAt>);

impl<'py> Strs<'py> {
    /// `obj` must be a str, list or tuple (`TextInput` failed on a surrogate).
    fn collect(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        if let Ok(s) = obj.cast::<PyString>() {
            return Ok(Self {
                kind: Kind::Str,
                items: vec![s.clone()],
            });
        }
        let (kind, items): (Kind, Vec<Bound<'py, PyAny>>) = if let Ok(l) = obj.cast::<PyList>() {
            (Kind::List, l.iter().collect())
        } else {
            (Kind::Tuple, obj.cast::<PyTuple>()?.iter().collect())
        };
        let mut strs = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let Ok(s) = item.cast::<PyString>() else {
                let err = input_error(
                    py,
                    format!("expected str at index {i}, got {}", type_name(item)),
                );
                return Err(arg_note(py, err, "string"));
            };
            strs.push(s.clone());
        }
        Ok(Self { kind, items: strs })
    }

    /// Run `f` over the WTF-8 bytes of every item, detached, on the pool.
    fn map_wtf8<F>(
        &self,
        py: Python<'py>,
        threads_arg: Option<Threads>,
        f: F,
    ) -> PyResult<Vec<Wtf8Result>>
    where
        F: Fn(&[u8]) -> Wtf8Result + Sync,
    {
        let mut owned: Vec<Option<Bound<'py, PyBytes>>> = Vec::with_capacity(self.items.len());
        for s in &self.items {
            owned.push(match s.to_str() {
                Ok(_) => None,
                Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
                    Some(surrogatepass_bytes(s)?)
                }
                Err(e) => return Err(e),
            });
        }
        let bytes: Vec<&[u8]> = self
            .items
            .iter()
            .zip(&owned)
            .map(|(s, b)| match b {
                Some(b) => Ok(b.as_bytes()),
                None => s.to_str().map(str::as_bytes),
            })
            .collect::<PyResult<_>>()?;
        let t = threads(threads_arg);
        py.detach(|| par::map(bytes.len(), t, |i| f(bytes[i])))
            .map_err(|e| BatchError::from(e).into_pyerr(py))
    }

    /// The caller's container kind, from per-item outputs (`None`:
    /// unchanged, the original object).
    fn build(
        &self,
        py: Python<'py>,
        outs: impl Iterator<Item = Option<Vec<u8>>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut strs = Vec::with_capacity(self.items.len());
        for (item, out) in self.items.iter().zip(outs) {
            strs.push(match out {
                None => item.clone(),
                Some(v) => str_from_wtf8(py, &v)?,
            });
        }
        Ok(match self.kind {
            Kind::Str => strs.swap_remove(0).into_any(),
            Kind::List => PyList::new(py, strs)?.into_any(),
            Kind::Tuple => PyTuple::new(py, strs)?.into_any(),
        })
    }
}

/// The bytes of a changed result; `None` if unchanged.
fn owned(out: Cow<'_, [u8]>) -> Option<Vec<u8>> {
    match out {
        Cow::Borrowed(_) => None,
        Cow::Owned(v) => Some(v),
    }
}

/// Unidecode's `RuntimeWarning` for one lone surrogate.
fn warn_surrogate(py: Python<'_>, unit: u16) -> PyResult<()> {
    let msg = format!(
        "Surrogate character {} will be ignored. You might be using a narrow Python build.",
        unit_repr(py, Unit::Surrogate(unit))?
    );
    let msg = CString::new(msg).map_err(|e| PyTypeError::new_err(e.to_string()))?;
    PyErr::warn(py, &py.get_type::<PyRuntimeWarning>(), &msg, 1)
}

/// `unidecode` for a str/list/tuple holding lone surrogates, or with a
/// `replace_str` that holds one. Documents are processed in parallel; the
/// warnings and the first error then come out in document order, as from a
/// loop of Unidecode calls.
fn unidecode_wtf8<'py>(
    py: Python<'py>,
    string: &Bound<'py, PyAny>,
    mode: &Mode,
    replace: &Replace,
    threads_arg: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    let strs = Strs::collect(string)?;
    let rep: &[u8] = match replace {
        Replace::Utf8(r) => r.as_bytes(),
        Replace::Wtf8(b) => b,
        Replace::NotStr(_) => b"",
    };
    let missing = match mode {
        Mode::Preserve => MissingWtf8::Keep,
        Mode::Replace => MissingWtf8::Replace(rep),
        _ => MissingWtf8::Drop,
    };
    let fails = fails_on_missing(mode, replace);
    let results = strs.map_wtf8(py, threads_arg, |b| {
        let surrogates = kernel::surrogates_wtf8(b);
        let stop = if fails {
            kernel::first_missing_wtf8(b)
        } else {
            None
        };
        let out = if stop.is_some() {
            None
        } else {
            owned(kernel::unidecode_wtf8(b, missing))
        };
        (out, surrogates, stop)
    })?;
    let mut outs = Vec::with_capacity(results.len());
    for (doc, (out, surrogates, stop)) in results.into_iter().enumerate() {
        // Unidecode warns as it meets each surrogate; strict mode and an
        // invalid `errors` stop at the first missing character, while a
        // non-str `replace_str` fails only when the result is joined.
        let stops_early = matches!(mode, Mode::Strict | Mode::Invalid(_));
        for &(index, unit) in &surrogates {
            if stops_early && stop.is_some_and(|m| index > m.index) {
                break;
            }
            warn_surrogate(py, unit)?;
        }
        if let Some(m) = stop {
            let err = missing_error(py, mode, replace, m)?;
            return Err(if strs.kind == Kind::Str {
                err
            } else {
                with_note(py, err, format!("pyroid: in the str at index {doc}"))
            });
        }
        outs.push(out);
    }
    strs.build(py, outs.into_iter())
}

/// True for each document that is pure ASCII (``str.isascii()``).
///
/// A predicate: returns a bool, a list/tuple of bools, or an Arrow bool column
/// (nulls stay null), matching the input kind.
#[pyfunction]
#[pyo3(signature = (string, *, threads = None))]
fn is_ascii<'py>(
    py: Python<'py>,
    string: &Bound<'py, PyAny>,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    // A Python str records whether it is ASCII, so `str.isascii` answers in
    // O(1) without building the str's UTF-8 form (which TextInput would do,
    // and cache on the str, for every non-ASCII document).
    if let Ok(s) = string.cast::<PyString>() {
        return Ok(PyBool::new(py, str_isascii(s)?).to_owned().into_any());
    }
    let is_list = string.is_instance_of::<PyList>();
    if is_list || string.is_instance_of::<PyTuple>() {
        let items: Vec<Bound<'py, PyAny>> = if is_list {
            string.cast::<PyList>()?.iter().collect()
        } else {
            string.cast::<PyTuple>()?.iter().collect()
        };
        let mut mask = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let Ok(s) = item.cast::<PyString>() else {
                let err = input_error(
                    py,
                    format!("expected str at index {i}, got {}", type_name(item)),
                );
                return Err(arg_note(py, err, "string"));
            };
            mask.push(str_isascii(s)?);
        }
        return Ok(if is_list {
            PyList::new(py, mask)?.into_any()
        } else {
            PyTuple::new(py, mask)?.into_any()
        });
    }
    predicate(
        py,
        text_arg(string, threads, "string")?,
        threads,
        kernel::is_ascii,
    )
}

static STR_ISASCII: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `str.isascii(s)`, unbound (a subclass's override does not interfere).
fn str_isascii(s: &Bound<'_, PyString>) -> PyResult<bool> {
    let py = s.py();
    let f = STR_ISASCII.get_or_try_init(py, || {
        Ok::<_, PyErr>(py.get_type::<PyString>().getattr("isascii")?.unbind())
    })?;
    f.bind(py).call1((s,))?.is_truthy()
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.translit")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(anyascii, &m)?)?;
    m.add_function(wrap_pyfunction!(unidecode, &m)?)?;
    m.add_function(wrap_pyfunction!(is_ascii, &m)?)?;
    parent.add_submodule(&m)
}
