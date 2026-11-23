//! The batch helpers every text function goes through.
//!
//! Pattern: extract while attached → `py.detach(..)` around the
//! rayon work → re-attach and build the output once, with exact-size
//! iterators. Kernels are pure Rust closures; they never call Python.
//!
//! - [`transform`]: `&str -> Cow<str>`; returns the caller's container kind
//!   (str → str, list → list, tuple → tuple, Arrow → `pyroid.Column` of
//!   strings). Unchanged documents come back as the *original* objects.
//! - [`predicate`]: `&str -> bool`; returns a mask in the caller's kind
//!   (bool, `list[bool]`, `tuple[bool, ...]`, `pyroid.Column` of bools).
//! - [`map_values`]: `&str -> R`; returns a `Vec<Option<R>>` (one per
//!   document, `None` for Arrow nulls) for tracks that build their own output.

use std::borrow::Cow;
use std::str::Utf8Error;

use pyo3::exceptions::PyUnicodeDecodeError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyList, PyString, PyTuple};
use pyroid_core::columnar::{BoolBuffers, BuildError, StringBuffers};
use pyroid_core::par::{self, PoolError};

use crate::arrow::{Chunk, Column, ColumnType, StrChunk, StrType, build_error};
use crate::errors::{PyroidError, with_note};
use crate::input::{TextInput, Threads, threads};

/// A single `str` shorter than this is processed without detaching: the
/// detach/re-attach round trip would cost more than it frees.
pub const SINGLE_DOC_DETACH_BYTES: usize = 4096;

/// What one document costs a kernel, which decides whether a single `str` is
/// processed detached (see [`detach_one`]). Batches always detach.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DocCost {
    /// A few µs per document (ftfy, translit, text): only long documents
    /// detach. With the GIL, detaching every call costs more than it frees:
    /// 8 Python threads calling `fix_text` / `anyascii` per doc ran 32-35%
    /// slower on 3.12 when every call detached.
    Light,
    /// ~100 µs per document or more (fastText, filters, MinHash): with the
    /// GIL every call detaches, so other Python threads run meanwhile (8
    /// threads calling per doc ran 1.4-2.9x faster on 3.12). Without the GIL
    /// nothing waits for it, so it behaves as `Light` (detaching every call
    /// measured up to 32% slower at 8 threads on 3.15t).
    Heavy,
}

/// Process one `str` of `len` bytes detached?
pub fn detach_one(py: Python<'_>, len: usize, cost: DocCost) -> bool {
    len >= SINGLE_DOC_DETACH_BYTES
        || (cost == DocCost::Heavy && crate::input::gil_enabled(py).unwrap_or(true))
}

/// Why a batch failed inside the detached region.
#[derive(Debug)]
pub enum BatchError {
    BadUtf8 {
        chunk: usize,
        row: usize,
        bytes: Vec<u8>,
        err: Utf8Error,
    },
    Pool(PoolError),
    Build(BuildError),
}

impl From<PoolError> for BatchError {
    fn from(e: PoolError) -> Self {
        Self::Pool(e)
    }
}

impl From<BuildError> for BatchError {
    fn from(e: BuildError) -> Self {
        Self::Build(e)
    }
}

impl BatchError {
    /// Convert to a Python exception (needs the interpreter).
    pub fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            Self::BadUtf8 {
                chunk,
                row,
                bytes,
                err,
            } => {
                let base = PyUnicodeDecodeError::new_utf8(py, &bytes, err)
                    .map_or_else(|e| e, |e| PyErr::from_value(e.into_any()));
                with_note(
                    py,
                    base,
                    format!(
                        "pyroid: row {row} of chunk {chunk} of the Arrow input is not valid UTF-8"
                    ),
                )
            }
            Self::Pool(e) => PyroidError::new_err(e.to_string()),
            Self::Build(e) => build_error(e),
        }
    }
}

/// `f` over every row of one Arrow chunk; `None` for nulls.
fn map_chunk<'c, R, F>(
    chunk: &'c StrChunk,
    index: usize,
    threads: par::Threads,
    f: &F,
) -> Result<Vec<Option<R>>, BatchError>
where
    R: Send,
    F: Fn(&'c str) -> R + Sync,
{
    par::try_map(chunk.len(), threads, |row| match chunk.get(row) {
        None => Ok(None),
        Some(Ok(s)) => Ok(Some(f(s))),
        Some(Err(err)) => Err(BatchError::BadUtf8 {
            chunk: index,
            row,
            bytes: chunk.bytes(row).unwrap_or_default().to_vec(),
            err,
        }),
    })
}

/// Is `out` the whole of `original` (same pointer and length)? A transform may
/// return `Cow::Borrowed` of a *substring*, which is a change.
fn is_whole(out: &str, original: &str) -> bool {
    out.as_ptr() == original.as_ptr() && out.len() == original.len()
}

/// The Python str for a transform result: the original object if unchanged.
fn result_str<'py>(py: Python<'py>, original: &PyBackedStr, out: &str) -> Bound<'py, PyString> {
    if is_whole(out, original) {
        // An owned result can't share the original's address, so this is
        // `Cow::Borrowed` of the whole input: hand back the original object.
        original.into_pyobject(py).unwrap_or_else(|e| match e {})
    } else {
        PyString::new(py, out)
    }
}

/// Apply a per-document transform; returns the caller's container kind.
pub fn transform<F>(
    py: Python<'_>,
    input: TextInput,
    threads_arg: Option<Threads>,
    f: F,
) -> PyResult<Bound<'_, PyAny>>
where
    F: for<'a> Fn(&'a str) -> Cow<'a, str> + Sync,
{
    let threads = threads(threads_arg);
    match &input {
        TextInput::Str(s) => {
            let out = if s.len() < SINGLE_DOC_DETACH_BYTES {
                f(s)
            } else {
                py.detach(|| f(s))
            };
            Ok(result_str(py, s, &out).into_any())
        }
        TextInput::List(docs) | TextInput::Tuple(docs) => {
            let outs = py
                .detach(|| par::map(docs.len(), threads, |i| f(&docs[i])))
                .map_err(|e| BatchError::from(e).into_pyerr(py))?;
            let items = docs.iter().zip(&outs).map(|(d, o)| result_str(py, d, o));
            Ok(match input {
                TextInput::List(_) => PyList::new(py, items)?.into_any(),
                _ => PyTuple::new(py, items)?.into_any(),
            })
        }
        TextInput::Arrow(strings) => {
            let large = strings.field.ty == StrType::LargeUtf8;
            let chunks = py
                .detach(|| {
                    strings
                        .chunks
                        .iter()
                        .enumerate()
                        .map(|(c, chunk)| {
                            let outs = map_chunk(chunk, c, threads, &f)?;
                            // A `string` chunk holds at most 2 GiB: split the
                            // output rather than fail (or change type).
                            Ok(if large {
                                vec![Chunk::LargeUtf8(StringBuffers::build(&outs, threads)?)]
                            } else {
                                StringBuffers::build_split(&outs, threads)?
                                    .into_iter()
                                    .map(Chunk::Utf8)
                                    .collect()
                            })
                        })
                        .collect::<Result<Vec<Vec<_>>, BatchError>>()
                        .map(|v| v.into_iter().flatten().collect::<Vec<_>>())
                })
                .map_err(|e| e.into_pyerr(py))?;
            let ty = if large {
                ColumnType::LargeUtf8
            } else {
                ColumnType::Utf8
            };
            Ok(Bound::new(py, Column::new(ty, &strings.field.name, chunks))?.into_any())
        }
    }
}

/// [`transform`] for a kernel that can fail on a document. Each failure is
/// handed, in input order and on the calling thread (attached), to
/// `on_err(py, doc, text, e)`, which returns the document's result instead (for
/// example from the incumbent library) or the exception to raise. `doc` is the
/// index in the list or tuple, or the row counted across the chunks of the
/// Arrow input (0 for a single `str`); `text` is the document.
pub fn try_transform<'py, F, E, G>(
    py: Python<'py>,
    input: TextInput,
    threads_arg: Option<Threads>,
    f: F,
    mut on_err: G,
) -> PyResult<Bound<'py, PyAny>>
where
    F: for<'a> Fn(&'a str) -> Result<Cow<'a, str>, E> + Sync,
    E: Send,
    G: FnMut(Python<'py>, usize, &str, E) -> PyResult<String>,
{
    let threads = threads(threads_arg);
    match &input {
        TextInput::Str(s) => {
            let out = if s.len() < SINGLE_DOC_DETACH_BYTES {
                f(s)
            } else {
                py.detach(|| f(s))
            };
            let out = match out {
                Ok(out) => out,
                Err(e) => Cow::Owned(on_err(py, 0, s, e)?),
            };
            Ok(result_str(py, s, &out).into_any())
        }
        TextInput::List(docs) | TextInput::Tuple(docs) => {
            let outs = py
                .detach(|| par::map(docs.len(), threads, |i| f(&docs[i])))
                .map_err(|e| BatchError::from(e).into_pyerr(py))?;
            let mut items = Vec::with_capacity(outs.len());
            for (i, (d, o)) in docs.iter().zip(outs).enumerate() {
                let o = match o {
                    Ok(o) => o,
                    Err(e) => Cow::Owned(on_err(py, i, d, e)?),
                };
                items.push(result_str(py, d, &o));
            }
            Ok(match input {
                TextInput::List(_) => PyList::new(py, items)?.into_any(),
                _ => PyTuple::new(py, items)?.into_any(),
            })
        }
        TextInput::Arrow(strings) => {
            let large = strings.field.ty == StrType::LargeUtf8;
            let mut chunks = Vec::with_capacity(strings.chunks.len());
            let mut base = 0;
            // Chunk by chunk: the kernel detached, the failures attached, the buffers detached.
            for (c, chunk) in strings.chunks.iter().enumerate() {
                let outs = py
                    .detach(|| map_chunk(chunk, c, threads, &f))
                    .map_err(|e| e.into_pyerr(py))?;
                let mut vals = Vec::with_capacity(outs.len());
                for (row, out) in outs.into_iter().enumerate() {
                    vals.push(match out {
                        None => None,
                        Some(Ok(s)) => Some(s),
                        Some(Err(e)) => {
                            // A failed row is valid UTF-8 (map_chunk checked it).
                            let text = chunk.get(row).and_then(Result::ok).unwrap_or_default();
                            Some(Cow::Owned(on_err(py, base + row, text, e)?))
                        }
                    });
                }
                base += chunk.len();
                let built = py
                    .detach(|| -> Result<Vec<Chunk>, BatchError> {
                        Ok(if large {
                            vec![Chunk::LargeUtf8(StringBuffers::build(&vals, threads)?)]
                        } else {
                            StringBuffers::build_split(&vals, threads)?
                                .into_iter()
                                .map(Chunk::Utf8)
                                .collect()
                        })
                    })
                    .map_err(|e| e.into_pyerr(py))?;
                chunks.extend(built);
            }
            let ty = if large {
                ColumnType::LargeUtf8
            } else {
                ColumnType::Utf8
            };
            Ok(Bound::new(py, Column::new(ty, &strings.field.name, chunks))?.into_any())
        }
    }
}

/// Apply a per-document predicate; returns a mask in the caller's kind.
pub fn predicate<F>(
    py: Python<'_>,
    input: TextInput,
    threads_arg: Option<Threads>,
    f: F,
) -> PyResult<Bound<'_, PyAny>>
where
    F: Fn(&str) -> bool + Sync,
{
    let threads = threads(threads_arg);
    match &input {
        TextInput::Str(s) => {
            let keep = if s.len() < SINGLE_DOC_DETACH_BYTES {
                f(s)
            } else {
                py.detach(|| f(s))
            };
            Ok(PyBool::new(py, keep).to_owned().into_any())
        }
        TextInput::List(docs) | TextInput::Tuple(docs) => {
            let mask = py
                .detach(|| par::map(docs.len(), threads, |i| f(&docs[i])))
                .map_err(|e| BatchError::from(e).into_pyerr(py))?;
            Ok(match input {
                TextInput::List(_) => PyList::new(py, mask)?.into_any(),
                _ => PyTuple::new(py, mask)?.into_any(),
            })
        }
        TextInput::Arrow(strings) => {
            let chunks = py
                .detach(|| {
                    strings
                        .chunks
                        .iter()
                        .enumerate()
                        .map(|(c, chunk)| {
                            let mask = map_chunk(chunk, c, threads, &|s: &str| f(s))?;
                            Ok(Chunk::Bool(BoolBuffers::build(&mask)))
                        })
                        .collect::<Result<Vec<_>, BatchError>>()
                })
                .map_err(|e| e.into_pyerr(py))?;
            Ok(Bound::new(
                py,
                Column::new(ColumnType::Bool, &strings.field.name, chunks),
            )?
            .into_any())
        }
    }
}

/// Every document of `input` as UTF-8, in order (`None` for an Arrow null).
pub fn rows(input: &TextInput) -> Result<Vec<Option<&str>>, BatchError> {
    Ok(match input {
        TextInput::Str(s) => vec![Some(&**s)],
        TextInput::List(d) | TextInput::Tuple(d) => d.iter().map(|s| Some(&**s)).collect(),
        TextInput::Arrow(strings) => {
            let mut out = Vec::with_capacity(strings.len());
            for (c, chunk) in strings.chunks.iter().enumerate() {
                for row in 0..chunk.len() {
                    out.push(match chunk.get(row) {
                        None => None,
                        Some(Ok(s)) => Some(s),
                        Some(Err(err)) => {
                            return Err(BatchError::BadUtf8 {
                                chunk: c,
                                row,
                                bytes: chunk.bytes(row).unwrap_or_default().to_vec(),
                                err,
                            });
                        }
                    });
                }
            }
            out
        }
    })
}

/// The documents of `input` as a Python list, for the paths that process
/// them one by one in Python: the original str objects for a str, list or
/// tuple; new strs for Arrow, and `None` for each Arrow null.
pub fn to_list<'py>(py: Python<'py>, input: &TextInput) -> PyResult<Bound<'py, PyList>> {
    match input {
        TextInput::Str(s) => PyList::new(py, [s]),
        TextInput::List(docs) | TextInput::Tuple(docs) => PyList::new(py, docs),
        TextInput::Arrow(_) => PyList::new(py, rows(input).map_err(|e| e.into_pyerr(py))?),
    }
}

/// Number of documents in `input` (a str is one; Arrow nulls count).
pub fn doc_count(input: &TextInput) -> usize {
    match input {
        TextInput::Str(_) => 1,
        TextInput::List(d) | TextInput::Tuple(d) => d.len(),
        TextInput::Arrow(strings) => strings.len(),
    }
}

/// `f(doc, row)` for every document, in parallel per `threads`, writing each
/// document's fixed-size result in place: `out` holds one `row_len`-value row
/// per document, in input order ([`doc_count`] rows). Returns `f`'s value
/// per document; Arrow nulls give `None` and leave their row untouched.
///
/// Pure Rust: call it while detached (e.g. inside `py.detach` holding a lock
/// on the destination).
pub fn fill_rows<T, R, F>(
    input: &TextInput,
    threads_arg: Option<Threads>,
    out: &mut [T],
    row_len: usize,
    f: F,
) -> Result<Vec<Option<R>>, BatchError>
where
    T: Send,
    R: Send,
    F: Fn(&str, &mut [T]) -> R + Sync,
{
    let threads = threads(threads_arg);
    match input {
        TextInput::Str(s) => Ok(vec![Some(f(s, out))]),
        TextInput::List(docs) | TextInput::Tuple(docs) => {
            par::try_map_rows(out, row_len, threads, |i, row| Ok(Some(f(&docs[i], row))))
        }
        TextInput::Arrow(strings) => {
            let mut res = Vec::with_capacity(strings.len());
            let mut rest = out;
            for (c, chunk) in strings.chunks.iter().enumerate() {
                let (rows, tail) = rest.split_at_mut(chunk.len() * row_len);
                rest = tail;
                res.extend(par::try_map_rows(
                    rows,
                    row_len,
                    threads,
                    |i, row| match chunk.get(i) {
                        None => Ok(None),
                        Some(Ok(s)) => Ok(Some(f(s, row))),
                        Some(Err(err)) => Err(BatchError::BadUtf8 {
                            chunk: c,
                            row: i,
                            bytes: chunk.bytes(i).unwrap_or_default().to_vec(),
                            err,
                        }),
                    },
                )?);
            }
            Ok(res)
        }
    }
}

/// Apply `f` to every document, detached and in parallel; one entry per
/// document in input order (`None` for Arrow nulls). For product functions
/// whose output is not a str or bool per document (scores, hashes, labels).
pub fn map_values<R, F>(
    py: Python<'_>,
    input: &TextInput,
    threads_arg: Option<Threads>,
    f: F,
) -> PyResult<Vec<Option<R>>>
where
    R: Send,
    F: Fn(&str) -> R + Sync,
{
    map_values_cost(py, input, threads_arg, DocCost::Light, f)
}

/// [`map_values`] for a kernel of the given [`DocCost`] (it only matters for
/// a single `str`).
pub fn map_values_cost<R, F>(
    py: Python<'_>,
    input: &TextInput,
    threads_arg: Option<Threads>,
    cost: DocCost,
    f: F,
) -> PyResult<Vec<Option<R>>>
where
    R: Send,
    F: Fn(&str) -> R + Sync,
{
    let threads = threads(threads_arg);
    let result = match input {
        TextInput::Str(s) => Ok(vec![Some(if detach_one(py, s.len(), cost) {
            py.detach(|| f(s))
        } else {
            f(s)
        })]),
        TextInput::List(docs) | TextInput::Tuple(docs) => py
            .detach(|| par::map(docs.len(), threads, |i| Some(f(&docs[i]))))
            .map_err(BatchError::from),
        TextInput::Arrow(strings) => py.detach(|| {
            let mut out = Vec::with_capacity(strings.len());
            for (c, chunk) in strings.chunks.iter().enumerate() {
                out.extend(map_chunk(chunk, c, threads, &|s: &str| f(s))?);
            }
            Ok(out)
        }),
    };
    result.map_err(|e| e.into_pyerr(py))
}
