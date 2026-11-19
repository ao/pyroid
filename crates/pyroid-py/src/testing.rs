//! Private hooks for pyroid's own test suite (`_pyroid._testing`). Not API.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyroid_core::columnar::{BoolBuffers, Offset, StringBuffers, bitmap};
use pyroid_core::par;

use crate::arrow::{Chunk, Column, ColumnType, build_error};
use crate::batch::{BatchError, map_values};
use crate::input::{TextInput, Threads};

/// Panic, either on the calling thread or inside a pool worker while detached
/// (proves panics become `PanicException`, never an abort).
#[pyfunction]
#[pyo3(signature = (*, in_worker))]
fn panic(py: Python<'_>, in_worker: bool) -> PyResult<()> {
    assert!(in_worker, "pyroid test panic on the calling thread");
    py.detach(|| {
        par::map(10_000, par::Threads::Auto, |i| {
            assert!(i != 7_777, "pyroid test panic in a worker thread");
        })
    })
    .map_err(|e| BatchError::from(e).into_pyerr(py))?;
    Ok(())
}

/// Size of the shared thread pool (builds it on first use).
#[pyfunction]
fn pool_size(py: Python<'_>) -> PyResult<usize> {
    par::pool_size().map_err(|e| BatchError::from(e).into_pyerr(py))
}

/// Build a `Column` from Python values, without pyarrow, in `chunks` pieces of
/// (nearly) equal length, or pieces of the lengths in `sizes`. `type` is
/// `"string"` (or `"large_string"` with `large=True`), `"large_string"`,
/// `"bool"` or `"large_binary"`; values are `str`, `bool` or `bytes` (or `None`).
#[pyfunction]
#[pyo3(signature = (values, *, large = false, chunks = 1, name = "", r#type = None, sizes = None))]
fn column(
    values: &Bound<'_, PyAny>,
    large: bool,
    chunks: usize,
    name: &str,
    r#type: Option<&str>,
    sizes: Option<Vec<usize>>,
) -> PyResult<Column> {
    let ty = match r#type.unwrap_or(if large { "large_string" } else { "string" }) {
        "string" => ColumnType::Utf8,
        "large_string" => ColumnType::LargeUtf8,
        "bool" => ColumnType::Bool,
        "large_binary" => ColumnType::LargeBinary,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown column type {other:?}"
            )));
        }
    };
    let n = values.len()?;
    let sizes = sizes.unwrap_or_else(|| {
        let per = n.div_ceil(chunks.max(1)).max(1);
        (0..n)
            .step_by(per)
            .map(|start| per.min(n - start))
            .collect()
    });
    if sizes.iter().sum::<usize>() != n {
        return Err(PyValueError::new_err("sizes must add up to len(values)"));
    }
    let mut out = Vec::new();
    let mut start = 0;
    for size in sizes {
        let part = values.get_item(PySlice::new(
            values.py(),
            start.try_into()?,
            (start + size).try_into()?,
            1,
        ))?;
        start += size;
        out.push(match ty {
            ColumnType::Utf8 => Chunk::Utf8(strings(&part.extract::<Vec<Option<String>>>()?)?),
            ColumnType::LargeUtf8 => {
                Chunk::LargeUtf8(strings(&part.extract::<Vec<Option<String>>>()?)?)
            }
            ColumnType::Bool => {
                Chunk::Bool(BoolBuffers::build(&part.extract::<Vec<Option<bool>>>()?))
            }
            ColumnType::LargeBinary => {
                Chunk::LargeBinary(binary(&part.extract::<Vec<Option<Vec<u8>>>>()?))
            }
        });
    }
    Ok(Column::new(ty, name, out))
}

fn strings<O: Offset>(values: &[Option<String>]) -> PyResult<StringBuffers<O>> {
    StringBuffers::build(values, par::Threads::ONE).map_err(build_error)
}

/// `large_binary` buffers (any bytes, so not through `StringBuffers::build`).
fn binary(values: &[Option<Vec<u8>>]) -> StringBuffers<i64> {
    let mut offsets = vec![0i64];
    let mut data = Vec::new();
    for v in values {
        data.extend_from_slice(v.as_deref().unwrap_or_default());
        offsets.push(i64::try_from(data.len()).expect("test data is small"));
    }
    let null_count = values.iter().filter(|v| v.is_none()).count();
    StringBuffers {
        len: values.len(),
        offsets,
        data,
        validity: (null_count > 0)
            .then(|| bitmap(values.iter().map(Option::is_some), values.len())),
        null_count,
    }
}

/// `(errno, strerror, winerror)` of the `OSError` for OS error `code` whose `io::Error` reads
/// `text`, as built on Windows (`windows=True`) or elsewhere: `errors::os_error_args`.
#[pyfunction]
#[pyo3(signature = (code, text, *, windows))]
fn os_error_args(code: i32, text: &str, windows: bool) -> (i32, String, Option<i32>) {
    crate::errors::os_error_args(code, text, windows)
}

/// `len(s)` per document via `batch::map_values` (`None` for Arrow nulls).
#[pyfunction]
#[pyo3(signature = (x, /, *, threads = None))]
fn char_lengths(
    py: Python<'_>,
    x: TextInput,
    threads: Option<Threads>,
) -> PyResult<Vec<Option<usize>>> {
    map_values(py, &x, threads, |s| s.chars().count())
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid._pyroid._testing")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(panic, &m)?)?;
    m.add_function(wrap_pyfunction!(pool_size, &m)?)?;
    m.add_function(wrap_pyfunction!(column, &m)?)?;
    m.add_function(wrap_pyfunction!(char_lengths, &m)?)?;
    m.add_function(wrap_pyfunction!(os_error_args, &m)?)?;
    parent.add_submodule(&m)
}
