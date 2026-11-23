//! Argument extraction shared by every text function: [`TextInput`] (the
//! documents) and [`Threads`] (the `threads=` keyword).

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::exceptions::{PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyList, PyMemoryView, PyString, PyTuple};
use pyroid_core::par;

use crate::arrow::{self, ArrowStrings};
use crate::errors::{PyroidError, input_error, type_name, with_note};

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

fn not_a_str(item: &Bound<'_, PyAny>, i: usize) -> PyErr {
    input_error(
        item.py(),
        format!("expected str at index {i}, got {}", type_name(item)),
    )
}

fn docs<'py>(
    items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
) -> PyResult<Vec<PyBackedStr>> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.enumerate() {
        let Ok(s) = item.cast::<PyString>() else {
            return Err(not_a_str(&item, i));
        };
        out.push(backed(s, Some(i))?);
    }
    Ok(out)
}

/// A list/tuple shorter than this is always extracted on the calling thread.
const PAR_MIN_DOCS: usize = 2 * par::MIN_DOCS_PER_TASK;
/// Code points per pool task, at least: below this per task the batch is
/// extracted on the calling thread.
const PAR_MIN_CHARS_PER_TASK: usize = 1 << 19;
/// Code points per attach of a pool worker. Each attach creates and destroys
/// a thread state (`PyGILState_Ensure`/`Release`), so it must be large; a
/// worker still reaches a safe point (detaches) every few milliseconds, so a
/// stop-the-world pause another thread asks for (e.g. the GC) is not held up.
const PAR_CHARS_PER_ATTACH: usize = 1 << 20;

static IS_GIL_ENABLED: PyOnceLock<Option<Py<PyAny>>> = PyOnceLock::new();

/// Is the interpreter running with the GIL? (`sys._is_gil_enabled()`, 3.13+;
/// always true before.) Asked per call: an extension can re-enable it.
pub fn gil_enabled(py: Python<'_>) -> PyResult<bool> {
    let f = IS_GIL_ENABLED.get_or_try_init(py, || -> PyResult<_> {
        let sys = py.import(pyo3::intern!(py, "sys"))?;
        Ok(sys
            .getattr(pyo3::intern!(py, "_is_gil_enabled"))
            .ok()
            .map(Bound::unbind))
    })?;
    match f {
        Some(f) => f.bind(py).call0()?.is_truthy(),
        None => Ok(true),
    }
}

/// [`docs`] for a large batch on a free-threaded interpreter running without
/// the GIL: building the UTF-8 form of non-ASCII strs (CPython's
/// `PyUnicode_AsUTF8AndSize`, which caches it on the str) is the one part of
/// extraction that costs per byte, and it runs on the pool, on balanced ranges
/// of documents. Results and errors are exactly [`docs`]'s: the same objects in
/// the same order, and the error of the lowest failing index (a lone surrogate
/// before the first non-str item is reported first, as when serial), raised on
/// the calling thread (so `__context__` and notes are as when serial).
///
/// Falls back to [`docs`] when it would not pay: with the GIL (only one thread
/// could encode at a time), `threads=1`, or a small batch.
fn docs_par<'py>(
    py: Python<'py>,
    items: impl ExactSizeIterator<Item = Bound<'py, PyAny>>,
    threads: par::Threads,
) -> PyResult<Vec<PyBackedStr>> {
    if items.len() < PAR_MIN_DOCS || threads == par::Threads::ONE || gil_enabled(py)? {
        return docs(items);
    }
    // Pass 1 (attached, cheap): type-check up to the first non-str and take a
    // strong reference to every str, so a concurrent mutation of the list
    // can't free one while the workers read it. Lengths only weigh the work:
    // a str subclass counts as empty, so its `__len__` (user code) never runs.
    let mut strs: Vec<Py<PyString>> = Vec::with_capacity(items.len());
    let mut lens: Vec<usize> = Vec::with_capacity(items.len());
    let mut not_str = None;
    for (i, item) in items.enumerate() {
        let Ok(s) = item.cast::<PyString>() else {
            not_str = Some(not_a_str(&item, i));
            break;
        };
        lens.push(if s.is_exact_instance_of::<PyString>() {
            s.len()?
        } else {
            0
        });
        strs.push(s.clone().unbind());
    }
    let total: usize = lens.iter().sum();
    let tasks = if total < 2 * PAR_MIN_CHARS_PER_TASK {
        1
    } else {
        par::max_tasks(threads)
            .map_err(|e| PyroidError::new_err(e.to_string()))?
            .min(total / PAR_MIN_CHARS_PER_TASK)
    };
    let out = if tasks < 2 {
        strs.iter()
            .enumerate()
            .map(|(i, s)| backed(s.bind(py), Some(i)))
            .collect::<PyResult<Vec<_>>>()?
    } else {
        // Attach-sized ranges, dealt out in order to at most `tasks` tasks.
        let ranges = attach_ranges(&lens, total.div_ceil(tasks).min(PAR_CHARS_PER_ATTACH));
        let groups: Vec<&[Range<usize>]> = ranges.chunks(ranges.len().div_ceil(tasks)).collect();
        let first_err = AtomicUsize::new(usize::MAX);
        let parts = py
            .detach(|| {
                par::map_tasks(&groups, threads, |group| {
                    let mut part = Vec::new();
                    for r in *group {
                        if backed_range(&strs, r.clone(), &first_err, &mut part) {
                            break;
                        }
                    }
                    part
                })
            })
            .map_err(|e| PyroidError::new_err(e.to_string()))?;
        // Back on the calling thread (attached): every PyBackedStr is dropped
        // here, and the error of the lowest failing index is raised here.
        let failed = first_err.into_inner();
        if failed != usize::MAX {
            drop(parts);
            return Err(backed(strs[failed].bind(py), Some(failed))
                .expect_err("a str that failed to encode on the pool fails again"));
        }
        parts.into_iter().flatten().collect()
    };
    match not_str {
        Some(err) => Err(err),
        None => Ok(out),
    }
}

/// Contiguous index ranges of about `chunk` code points each (at least one
/// document each).
fn attach_ranges(lens: &[usize], chunk: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let (mut start, mut acc) = (0, 0);
    for (i, &n) in lens.iter().enumerate() {
        acc += n;
        if acc >= chunk {
            ranges.push(start..i + 1);
            (start, acc) = (i + 1, 0);
        }
    }
    if start < lens.len() {
        ranges.push(start..lens.len());
    }
    ranges
}

/// Appends the `PyBackedStr`s of `strs[range]` to `out`, on a pool worker.
/// At the first item that fails to encode (a lone surrogate) it lowers
/// `first_err` to that index, drops the error while still attached, and
/// returns `true`; items past an error another worker found are skipped
/// (only the lowest failing index matters, and the caller re-raises it).
///
/// The one place a pool worker attaches to the interpreter. `clippy.toml`
/// bans `Python::attach` because kernels must be pure Rust; this is not a
/// kernel but argument extraction, and it is deadlock-free because:
/// - only [`docs_par`] calls it, from `py.detach`, so the calling thread holds
///   no thread state (and on GIL builds no GIL) while it waits for the pool.
///   Waiting on the pool while attached would deadlock with a stop-the-world
///   request (it blocks the workers' attach), so nothing may do that;
/// - it runs only when `sys._is_gil_enabled()` was false; if an extension
///   re-enables the GIL meanwhile, the workers take turns at it, which is
///   slower but still deadlock-free (nobody waiting for them holds it);
/// - while attached it calls only `PyUnicode_AsUTF8AndSize` (via
///   `PyBackedStr`; CPython guards the UTF-8 cache with a per-object critical
///   section, so the same str twice in a batch is fine) and reference counting
///   on strs the caller keeps alive: no user code, nothing that waits for
///   another pool worker;
/// - it detaches after at most [`PAR_CHARS_PER_ATTACH`] code points, so a
///   stop-the-world request waits at most that long.
#[allow(clippy::disallowed_methods)]
fn backed_range(
    strs: &[Py<PyString>],
    range: Range<usize>,
    first_err: &AtomicUsize,
    out: &mut Vec<PyBackedStr>,
) -> bool {
    Python::attach(|py| {
        for i in range {
            if i > first_err.load(Ordering::Relaxed) {
                return true;
            }
            // The error (a lone surrogate) is dropped here, attached; the caller re-raises it.
            let Ok(s) = PyBackedStr::try_from(strs[i].bind(py).clone()) else {
                first_err.fetch_min(i, Ordering::Relaxed);
                return true;
            };
            out.push(s);
        }
        false
    })
}

/// `err` with the note PyO3 adds to the errors of converting argument `name`
/// (`while processing '<name>'`), for arguments a function converts itself.
pub fn arg_note(py: Python<'_>, err: PyErr, name: &str) -> PyErr {
    with_note(py, err, format!("while processing '{name}'"))
}

/// [`TextInput::extract_with`] for a `#[pyfunction]` argument `name` taken as
/// `&Bound<PyAny>` (so extraction can see `threads=`): errors get the note
/// PyO3 adds to argument errors (`while processing '<name>'`).
pub fn text_arg(
    obj: &Bound<'_, PyAny>,
    threads_arg: Option<Threads>,
    name: &str,
) -> PyResult<TextInput> {
    TextInput::extract_with(obj, threads_arg).map_err(|e| arg_note(obj.py(), e, name))
}

impl TextInput {
    /// Extract as [`FromPyObject`] does (which always runs on the calling
    /// thread), knowing the call's `threads=`: on a free-threaded interpreter
    /// running without the GIL, a large `list`/`tuple` builds its strs' UTF-8
    /// forms on the shared pool ([`docs_par`]). Same result, same errors.
    pub fn extract_with(obj: &Bound<'_, PyAny>, threads_arg: Option<Threads>) -> PyResult<Self> {
        let py = obj.py();
        if let Ok(list) = obj.cast::<PyList>() {
            return Ok(Self::List(docs_par(py, list.iter(), threads(threads_arg))?));
        }
        if let Ok(tuple) = obj.cast::<PyTuple>() {
            return Ok(Self::Tuple(docs_par(
                py,
                tuple.iter(),
                threads(threads_arg),
            )?));
        }
        obj.extract()
    }
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
        let hint = if obj.hasattr(pyo3::intern!(py, "__iter__"))? {
            "; convert other iterables with list(...)"
        } else {
            ""
        };
        Err(input_error(
            py,
            format!("{EXPECTED}, got {}{hint}", type_name(&obj)),
        ))
    }
}

/// A Python int argument, read without overflow: anything with `__index__`
/// (int, numpy integers), but not bool.
enum IntArg {
    /// Non-negative; values past `usize::MAX` saturate.
    Value(usize),
    /// Negative; the value as text, for the error message.
    Negative(String),
    /// Not an int.
    NotInt,
}

fn int_arg(obj: &Bound<'_, PyAny>) -> PyResult<IntArg> {
    let py = obj.py();
    if obj.is_instance_of::<PyBool>() {
        return Ok(IntArg::NotInt);
    }
    let Ok(index) = obj.call_method0(pyo3::intern!(py, "__index__")) else {
        return Ok(IntArg::NotInt);
    };
    if let Ok(n) = index.extract::<usize>() {
        return Ok(IntArg::Value(n));
    }
    Ok(if index.lt(0)? {
        IntArg::Negative(index.to_string())
    } else {
        IntArg::Value(usize::MAX)
    })
}

/// The `threads=` keyword: `None` (all of the shared pool) or a positive int
/// (at most that many tasks at once; `1` runs on the calling thread).
/// Functions take it as `Option<Threads>` with a `None` default.
#[derive(Clone, Copy, Debug)]
pub struct Threads(pub par::Threads);

impl<'a, 'py> FromPyObject<'a, 'py> for Threads {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        const WANT: &str = "threads must be None or a positive int";
        match int_arg(&obj)? {
            IntArg::Value(n) => NonZeroUsize::new(n)
                .map(|n| Self(par::Threads::Max(n)))
                .ok_or_else(|| PyValueError::new_err(format!("{WANT}, got 0"))),
            IntArg::Negative(n) => Err(PyValueError::new_err(format!("{WANT}, got {n}"))),
            IntArg::NotInt => Err(input_error(
                obj.py(),
                format!("{WANT}, got {}", type_name(&obj)),
            )),
        }
    }
}

/// The pool setting for an optional `threads=` argument.
pub fn threads(t: Option<Threads>) -> par::Threads {
    t.map_or(par::Threads::Auto, |t| t.0)
}

/// A non-negative int argument (e.g. `min_chars(x, n)`), with a clear error.
/// Values past `usize::MAX` saturate (no document is that long).
#[derive(Clone, Copy, Debug)]
pub struct Count(pub usize);

impl<'a, 'py> FromPyObject<'a, 'py> for Count {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        const WANT: &str = "expected a non-negative int";
        match int_arg(&obj)? {
            IntArg::Value(n) => Ok(Self(n)),
            IntArg::Negative(n) => Err(PyValueError::new_err(format!("{WANT}, got {n}"))),
            IntArg::NotInt => Err(input_error(
                obj.py(),
                format!("{WANT}, got {}", type_name(&obj)),
            )),
        }
    }
}

/// `threads=` for a function that extracts its text argument itself (with
/// [`text_arg`], so extraction can see `threads`). Extracting it never fails:
/// a bad value is kept and raised after the texts, so a bad text argument is
/// still reported first, as when PyO3 extracted both in signature order. The
/// error carries the note PyO3 would have added.
#[derive(Debug)]
pub struct ThreadsArg(Result<Threads, PyErr>);

impl<'a, 'py> FromPyObject<'a, 'py> for ThreadsArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(obj.extract::<Threads>().map_err(|e| {
            with_note(obj.py(), e, "while processing 'threads'")
        })))
    }
}

impl ThreadsArg {
    /// The setting to extract the texts with: the call's, or `1` if it is
    /// invalid (that error is raised by [`Self::resolve`] right after).
    pub fn for_texts(arg: Option<&Self>) -> Option<Threads> {
        match arg {
            None => None,
            Some(Self(Ok(t))) => Some(*t),
            Some(Self(Err(_))) => Some(Threads(par::Threads::ONE)),
        }
    }

    /// The call's `threads=`, or its extraction error.
    pub fn resolve(arg: Option<Self>) -> PyResult<Option<Threads>> {
        arg.map(|a| a.0).transpose()
    }
}

/// [`text_arg`] then [`ThreadsArg::resolve`]: the documents and `threads=` of
/// a call, with errors in signature order (texts first). `default` is the
/// setting used when the call passes none (e.g. an object's own `threads`).
pub fn texts_and_threads(
    texts: &Bound<'_, PyAny>,
    threads: Option<ThreadsArg>,
    default: Option<Threads>,
) -> PyResult<(TextInput, Option<Threads>)> {
    let t = ThreadsArg::for_texts(threads.as_ref()).or(default);
    let input = text_arg(texts, t, "texts")?;
    Ok((input, ThreadsArg::resolve(threads)?))
}
