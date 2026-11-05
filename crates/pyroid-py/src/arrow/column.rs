//! `pyroid.Column`: the Arrow result of a pyroid function given Arrow input,
//! and the minimal C Data / C Stream *exporter* behind it.
//!
//! A `Column` owns Rust-allocated, immutable buffers (from
//! `pyroid_core::columnar`). Every exported struct keeps an `Arc` to the chunk
//! it points into, and every release callback only frees Rust memory: it never
//! touches Python. Consumers may therefore move and release exported structs
//! on any thread (polars and pyarrow do, from their own thread pools).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

use pyo3::exceptions::PyOverflowError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyList, PyString, PyTuple, PyType};
use pyroid_core::columnar::{BoolBuffers, BuildError, StringBuffers};

use super::ffi::{FLAG_NULLABLE, FfiArray, FfiSchema, FfiStream};
use super::import::{ARRAY_CAPSULE, SCHEMA_CAPSULE, STREAM_CAPSULE};
use crate::errors::PyroidError;

/// The Arrow type of a [`Column`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    Utf8,
    LargeUtf8,
    Bool,
}

impl ColumnType {
    fn format(self) -> &'static CStr {
        match self {
            Self::Utf8 => c"u",
            Self::LargeUtf8 => c"U",
            Self::Bool => c"b",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Utf8 => "string",
            Self::LargeUtf8 => "large_string",
            Self::Bool => "bool",
        }
    }
}

/// One chunk of a column.
#[derive(Debug)]
pub enum Chunk {
    Utf8(StringBuffers<i32>),
    LargeUtf8(StringBuffers<i64>),
    Bool(BoolBuffers),
}

impl Chunk {
    fn len(&self) -> usize {
        match self {
            Self::Utf8(b) => b.len,
            Self::LargeUtf8(b) => b.len,
            Self::Bool(b) => b.len,
        }
    }

    fn null_count(&self) -> usize {
        match self {
            Self::Utf8(b) => b.null_count,
            Self::LargeUtf8(b) => b.null_count,
            Self::Bool(b) => b.null_count,
        }
    }

    fn empty(ty: ColumnType) -> Self {
        match ty {
            ColumnType::Utf8 => Self::Utf8(StringBuffers {
                len: 0,
                offsets: vec![0],
                data: Vec::new(),
                validity: None,
                null_count: 0,
            }),
            ColumnType::LargeUtf8 => Self::LargeUtf8(StringBuffers {
                len: 0,
                offsets: vec![0],
                data: Vec::new(),
                validity: None,
                null_count: 0,
            }),
            ColumnType::Bool => Self::Bool(BoolBuffers::build(&[])),
        }
    }

    fn ty(&self) -> ColumnType {
        match self {
            Self::Utf8(_) => ColumnType::Utf8,
            Self::LargeUtf8(_) => ColumnType::LargeUtf8,
            Self::Bool(_) => ColumnType::Bool,
        }
    }
}

#[derive(Debug)]
struct ColumnData {
    ty: ColumnType,
    name: CString,
    chunks: Vec<Arc<Chunk>>,
}

/// A read-only Arrow column produced by pyroid (one or more chunks).
///
/// pyroid returns a `Column` when it is given Arrow input. It implements the
/// Arrow `PyCapsule` interface, so any Arrow library can take it without a copy:
/// `pyarrow.chunked_array(col)`, `pyarrow.array(col)`, `polars.Series(col)`,
/// or pass it back to pyroid. pyroid itself needs no Arrow library.
#[pyclass(module = "pyroid", name = "Column", frozen)]
#[derive(Debug)]
pub struct Column {
    inner: Arc<ColumnData>,
}

/// Map a buffer-building failure to the right Python exception.
pub fn build_error(e: BuildError) -> PyErr {
    match e {
        BuildError::Overflow { .. } => PyOverflowError::new_err(e.to_string()),
        BuildError::Pool(p) => PyroidError::new_err(p.to_string()),
    }
}

impl Column {
    /// A column of `ty` from `chunks` (which must all be of that type).
    pub fn new(ty: ColumnType, name: &str, chunks: Vec<Chunk>) -> Self {
        debug_assert!(chunks.iter().all(|c| c.ty() == ty));
        // Interior NULs can't be represented in an Arrow field name.
        let name = CString::new(name.replace('\0', "")).unwrap_or_default();
        Self {
            inner: Arc::new(ColumnData {
                ty,
                name,
                chunks: chunks
                    .into_iter()
                    .filter(|c| c.len() > 0)
                    .map(Arc::new)
                    .collect(),
            }),
        }
    }

    /// The whole column as one chunk (concatenating if needed).
    fn single_chunk(&self) -> PyResult<Arc<Chunk>> {
        let chunks = &self.inner.chunks;
        match chunks.as_slice() {
            [] => Ok(Arc::new(Chunk::empty(self.inner.ty))),
            [one] => Ok(Arc::clone(one)),
            many => {
                let joined = match self.inner.ty {
                    ColumnType::Utf8 => Chunk::Utf8(
                        StringBuffers::concat(&many.iter().map(|c| as_utf8(c)).collect::<Vec<_>>())
                            .map_err(build_error)?,
                    ),
                    ColumnType::LargeUtf8 => Chunk::LargeUtf8(
                        StringBuffers::concat(
                            &many.iter().map(|c| as_large(c)).collect::<Vec<_>>(),
                        )
                        .map_err(build_error)?,
                    ),
                    ColumnType::Bool => Chunk::Bool(BoolBuffers::concat(
                        &many.iter().map(|c| as_bool(c)).collect::<Vec<_>>(),
                    )),
                };
                Ok(Arc::new(joined))
            }
        }
    }
}

fn as_utf8(c: &Chunk) -> &StringBuffers<i32> {
    match c {
        Chunk::Utf8(b) => b,
        _ => unreachable!("column chunks share one type"),
    }
}

fn as_large(c: &Chunk) -> &StringBuffers<i64> {
    match c {
        Chunk::LargeUtf8(b) => b,
        _ => unreachable!("column chunks share one type"),
    }
}

fn as_bool(c: &Chunk) -> &BoolBuffers {
    match c {
        Chunk::Bool(b) => b,
        _ => unreachable!("column chunks share one type"),
    }
}

fn to_i64(n: usize) -> i64 {
    i64::try_from(n).expect("Rust allocations are at most isize::MAX")
}

// ---- export: ArrowSchema ---------------------------------------------------

struct SchemaPrivate {
    name: CString,
}

fn export_schema(ty: ColumnType, name: &CStr) -> FfiSchema {
    let private = Box::new(SchemaPrivate {
        name: name.to_owned(),
    });
    // The CString's heap buffer doesn't move when the Box becomes a raw pointer.
    let name_ptr = private.name.as_ptr();
    FfiSchema {
        format: ty.format().as_ptr(),
        name: name_ptr,
        metadata: ptr::null(),
        flags: FLAG_NULLABLE,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_schema),
        private_data: Box::into_raw(private).cast::<c_void>(),
    }
}

unsafe extern "C" fn release_schema(schema: *mut FfiSchema) {
    // SAFETY: the consumer passes a pointer to a live struct we exported (or
    // null, which we ignore).
    let Some(schema) = (unsafe { schema.as_mut() }) else {
        return;
    };
    if schema.release.is_none() || schema.private_data.is_null() {
        return;
    }
    // SAFETY: `private_data` came from `Box::into_raw(Box<SchemaPrivate>)` in
    // `export_schema` and is freed only here, once: we clear it right after.
    drop(unsafe { Box::from_raw(schema.private_data.cast::<SchemaPrivate>()) });
    schema.private_data = ptr::null_mut();
    schema.release = None;
}

// ---- export: ArrowArray ----------------------------------------------------

struct ArrayPrivate {
    /// Keeps the buffers alive.
    _chunk: Arc<Chunk>,
    /// The `buffers` C array the struct points to.
    buffers: [*const c_void; 3],
}

fn validity_ptr(v: Option<&Vec<u8>>) -> *const c_void {
    v.map_or(ptr::null(), |v| v.as_ptr().cast())
}

fn export_array(chunk: Arc<Chunk>) -> FfiArray {
    let (n_buffers, buffers) = match &*chunk {
        Chunk::Utf8(b) => (
            3,
            [
                validity_ptr(b.validity.as_ref()),
                b.offsets.as_ptr().cast(),
                b.data.as_ptr().cast(),
            ],
        ),
        Chunk::LargeUtf8(b) => (
            3,
            [
                validity_ptr(b.validity.as_ref()),
                b.offsets.as_ptr().cast(),
                b.data.as_ptr().cast(),
            ],
        ),
        Chunk::Bool(b) => (
            2,
            [
                validity_ptr(b.validity.as_ref()),
                b.values.as_ptr().cast(),
                ptr::null(),
            ],
        ),
    };
    let (length, null_count) = (to_i64(chunk.len()), to_i64(chunk.null_count()));
    let mut private = Box::new(ArrayPrivate {
        _chunk: chunk,
        buffers,
    });
    // Points into the Box's heap allocation, which `into_raw` doesn't move.
    let buffers_ptr = private.buffers.as_mut_ptr();
    FfiArray {
        length,
        null_count,
        offset: 0,
        n_buffers,
        n_children: 0,
        buffers: buffers_ptr,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_array),
        private_data: Box::into_raw(private).cast::<c_void>(),
    }
}

unsafe extern "C" fn release_array(array: *mut FfiArray) {
    // SAFETY: the consumer passes a pointer to a live struct we exported (or
    // null, which we ignore).
    let Some(array) = (unsafe { array.as_mut() }) else {
        return;
    };
    if array.release.is_none() || array.private_data.is_null() {
        return;
    }
    // SAFETY: `private_data` came from `Box::into_raw(Box<ArrayPrivate>)` in
    // `export_array` and is freed only here, once: we clear it right after.
    drop(unsafe { Box::from_raw(array.private_data.cast::<ArrayPrivate>()) });
    array.private_data = ptr::null_mut();
    array.buffers = ptr::null_mut();
    array.release = None;
}

// ---- export: ArrowArrayStream ----------------------------------------------

struct StreamPrivate {
    column: Arc<ColumnData>,
    next: usize,
}

/// `EINVAL`, the C Stream Interface's "invalid argument" code (22 on Linux,
/// macOS and Windows).
const EINVAL: c_int = 22;

/// Our stream's private data.
///
/// # Safety
/// `stream` must be null or a live stream exported by `export_stream`, and
/// the caller must not use the result after the stream is released.
unsafe fn stream_private<'a>(stream: *mut FfiStream) -> Option<&'a mut StreamPrivate> {
    // SAFETY: null or a live stream, per the caller's contract.
    let stream = unsafe { stream.as_mut() }?;
    // SAFETY: `private_data` is our `Box<StreamPrivate>` (or null after
    // release). The C Stream spec requires consumers to serialize calls on one
    // stream, so this is the only reference.
    unsafe { stream.private_data.cast::<StreamPrivate>().as_mut() }
}

unsafe extern "C" fn stream_get_schema(stream: *mut FfiStream, out: *mut FfiSchema) -> c_int {
    // SAFETY: the consumer passes our live stream.
    let Some(private) = (unsafe { stream_private(stream) }) else {
        return EINVAL;
    };
    if out.is_null() {
        return EINVAL;
    }
    let schema = export_schema(private.column.ty, &private.column.name);
    // SAFETY: `out` points to a writable ArrowSchema that the consumer owns
    // and has not initialized (or has released), so no drop is skipped.
    unsafe { ptr::write(out, schema) };
    0
}

unsafe extern "C" fn stream_get_next(stream: *mut FfiStream, out: *mut FfiArray) -> c_int {
    // SAFETY: the consumer passes our live stream.
    let Some(private) = (unsafe { stream_private(stream) }) else {
        return EINVAL;
    };
    if out.is_null() {
        return EINVAL;
    }
    let array = match private.column.chunks.get(private.next) {
        Some(chunk) => {
            private.next += 1;
            export_array(Arc::clone(chunk))
        }
        None => FfiArray::released(), // end of stream
    };
    // SAFETY: as in `stream_get_schema`.
    unsafe { ptr::write(out, array) };
    0
}

unsafe extern "C" fn stream_get_last_error(_stream: *mut FfiStream) -> *const c_char {
    ptr::null() // our callbacks only fail with EINVAL on misuse
}

unsafe extern "C" fn release_stream(stream: *mut FfiStream) {
    // SAFETY: the consumer passes a pointer to a live struct we exported (or
    // null, which we ignore).
    let Some(stream) = (unsafe { stream.as_mut() }) else {
        return;
    };
    if stream.release.is_none() || stream.private_data.is_null() {
        return;
    }
    // SAFETY: `private_data` came from `Box::into_raw(Box<StreamPrivate>)` in
    // `export_stream` and is freed only here, once: we clear it right after.
    drop(unsafe { Box::from_raw(stream.private_data.cast::<StreamPrivate>()) });
    stream.private_data = ptr::null_mut();
    stream.release = None;
}

fn export_stream(column: &Arc<ColumnData>) -> FfiStream {
    let private = Box::new(StreamPrivate {
        column: Arc::clone(column),
        next: 0,
    });
    FfiStream {
        get_schema: Some(stream_get_schema),
        get_next: Some(stream_get_next),
        get_last_error: Some(stream_get_last_error),
        release: Some(release_stream),
        private_data: Box::into_raw(private).cast::<c_void>(),
    }
}

// ---- capsules --------------------------------------------------------------

/// An exported struct on its way into a capsule.
#[repr(transparent)]
struct Exported<T>(T);

// SAFETY: exported structs point only at memory owned through their private
// data (Rust allocations behind `Arc`, never mutated), and their release
// callbacks only free that memory without touching Python. So they may be
// moved to, and released on, any thread — which `PyCapsule` requires because
// a capsule destructor can run anywhere.
unsafe impl Send for Exported<FfiSchema> {}
// SAFETY: as above.
unsafe impl Send for Exported<FfiArray> {}
// SAFETY: as above.
unsafe impl Send for Exported<FfiStream> {}

/// A capsule owning `value`. Its pointer is the address of the struct
/// (`Exported` is `repr(transparent)`), which is what consumers expect. If no
/// consumer moves the struct out, dropping the capsule releases it.
fn capsule<'py, T: 'static>(
    py: Python<'py>,
    value: T,
    name: &'static CStr,
) -> PyResult<Bound<'py, PyCapsule>>
where
    Exported<T>: Send,
{
    PyCapsule::new_with_value(py, Exported(value), name)
}

#[pymethods]
impl Column {
    /// Export as an Arrow C stream (`PyCapsule` interface). ``requested_schema``
    /// is accepted and ignored, as the spec allows.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        capsule(py, export_stream(&self.inner), STREAM_CAPSULE)
    }

    /// Export as one Arrow C array (`PyCapsule` interface), concatenating the
    /// chunks if there are several. ``requested_schema`` is ignored.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = requested_schema;
        let chunk = self.single_chunk()?;
        let schema = capsule(
            py,
            export_schema(self.inner.ty, &self.inner.name),
            SCHEMA_CAPSULE,
        )?;
        let array = capsule(py, export_array(chunk), ARRAY_CAPSULE)?;
        PyTuple::new(py, [schema, array])
    }

    /// `Column[str]` / `Column[bool]` in annotations (the stubs make `Column`
    /// generic in its value type).
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        py.import("types")?.getattr("GenericAlias")?.call1((cls, item))
    }

    fn __len__(&self) -> usize {
        self.inner.chunks.iter().map(|c| c.len()).sum()
    }

    /// Arrow type name: ``"string"``, ``"large_string"`` or ``"bool"``.
    #[getter(r#type)]
    fn type_name(&self) -> &'static str {
        self.inner.ty.name()
    }

    /// Field name, carried over from the input column (may be empty).
    #[getter]
    fn name(&self) -> String {
        self.inner.name.to_string_lossy().into_owned()
    }

    /// Number of nulls.
    #[getter]
    fn null_count(&self) -> usize {
        self.inner.chunks.iter().map(|c| c.null_count()).sum()
    }

    /// Number of chunks.
    #[getter]
    fn num_chunks(&self) -> usize {
        self.inner.chunks.len()
    }

    /// The values as a Python list (``None`` for nulls).
    fn to_pylist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for chunk in &self.inner.chunks {
            match &**chunk {
                Chunk::Utf8(b) => push_strings(py, &out, b)?,
                Chunk::LargeUtf8(b) => push_strings(py, &out, b)?,
                Chunk::Bool(b) => {
                    for i in 0..b.len {
                        out.append(b.get(i))?;
                    }
                }
            }
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!(
            "pyroid.Column<{}>(len={}, null_count={}, num_chunks={})",
            self.inner.ty.name(),
            self.__len__(),
            self.null_count(),
            self.num_chunks()
        )
    }
}

fn push_strings<O: pyroid_core::columnar::Offset>(
    py: Python<'_>,
    out: &Bound<'_, PyList>,
    b: &StringBuffers<O>,
) -> PyResult<()> {
    for i in 0..b.len {
        match b.get(i) {
            None => out.append(py.None())?,
            Some(bytes) => {
                // Built from `&str`s, so always valid UTF-8.
                let s = std::str::from_utf8(bytes)
                    .map_err(|e| PyroidError::new_err(format!("internal error: {e}")))?;
                out.append(PyString::new(py, s))?;
            }
        }
    }
    Ok(())
}
