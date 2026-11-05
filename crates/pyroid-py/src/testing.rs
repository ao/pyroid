//! Private hooks for pyroid's own test suite (`_pyroid._testing`). Not API.

use pyo3::prelude::*;
use pyroid_core::columnar::StringBuffers;
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

/// Build an Arrow string `Column` from Python values, without pyarrow, in
/// `chunks` pieces of (nearly) equal length.
#[pyfunction]
#[pyo3(signature = (values, *, large = false, chunks = 1, name = ""))]
fn column(values: Vec<Option<String>>, large: bool, chunks: usize, name: &str) -> PyResult<Column> {
    let per = values.len().div_ceil(chunks.max(1)).max(1);
    let mut out = Vec::new();
    for part in values.chunks(per) {
        out.push(if large {
            Chunk::LargeUtf8(StringBuffers::build(part, par::Threads::ONE).map_err(build_error)?)
        } else {
            Chunk::Utf8(StringBuffers::build(part, par::Threads::ONE).map_err(build_error)?)
        });
    }
    let ty = if large {
        ColumnType::LargeUtf8
    } else {
        ColumnType::Utf8
    };
    Ok(Column::new(ty, name, out))
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
    parent.add_submodule(&m)
}
