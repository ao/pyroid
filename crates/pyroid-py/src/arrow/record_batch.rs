//! Arrow record batches: named columns of one chunk each, exported as a struct array
//! (`__arrow_c_array__`), a stream of that one batch (`__arrow_c_stream__`), or the struct
//! schema alone (`__arrow_c_schema__`).
//!
//! The children are [`Chunk`]s exported with `column.rs`'s exporters. A struct schema or
//! array owns its children (each a `Box` turned into a raw pointer); its release callback
//! drops them, which releases every child a consumer has not moved out (a moved-out child
//! is left released, so nothing is released twice). As for `Column`, the callbacks only free
//! Rust memory and never touch Python, so consumers may release on any thread.

use std::ffi::{CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};

use super::column::{Chunk, ColumnType, capsule, export_array, export_schema};
use super::ffi::{FfiArray, FfiSchema, FfiStream};
use super::import::{ARRAY_CAPSULE, SCHEMA_CAPSULE, STREAM_CAPSULE};

/// One column of a [`RecordBatch`].
#[derive(Debug)]
pub struct Field {
    /// The field name (no interior NULs).
    pub name: CString,
    /// The Arrow type.
    pub ty: ColumnType,
    /// The values (`num_rows` of them).
    pub chunk: Arc<Chunk>,
}

/// Columns of equal length, exported as one Arrow struct array.
#[derive(Debug)]
pub struct RecordBatch {
    /// Rows in every column.
    pub num_rows: usize,
    /// The columns, in order.
    pub fields: Vec<Field>,
}

impl RecordBatch {
    /// A batch of `fields` (all of the same length).
    pub fn new(fields: Vec<Field>) -> Self {
        let num_rows = fields.first().map_or(0, |f| f.chunk.len());
        debug_assert!(
            fields
                .iter()
                .all(|f| f.chunk.len() == num_rows && f.chunk.ty() == f.ty)
        );
        Self { num_rows, fields }
    }
}

fn to_i64(n: usize) -> i64 {
    i64::try_from(n).expect("Rust allocations are at most isize::MAX")
}

// ---- export: the struct ArrowSchema ------------------------------------------

struct StructSchemaPrivate {
    /// Each from `Box::into_raw`; freed by `release_struct_schema`.
    children: Vec<*mut FfiSchema>,
}

pub(super) fn export_struct_schema(batch: &RecordBatch) -> FfiSchema {
    let children = batch
        .fields
        .iter()
        .map(|f| Box::into_raw(Box::new(export_schema(f.ty, &f.name))))
        .collect::<Vec<_>>();
    let n_children = to_i64(children.len());
    let private = Box::into_raw(Box::new(StructSchemaPrivate { children }));
    // Taken from the raw pointer after `into_raw` (as `export_array` does for its buffers).
    // SAFETY: `private` is a valid, live allocation we just created.
    let children_ptr = unsafe { (*private).children.as_mut_ptr() };
    FfiSchema {
        format: c"+s".as_ptr(),
        name: c"".as_ptr(),
        metadata: ptr::null(),
        flags: 0,
        n_children,
        children: children_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_schema),
        private_data: private.cast::<c_void>(),
    }
}

unsafe extern "C" fn release_struct_schema(schema: *mut FfiSchema) {
    // SAFETY: the consumer passes a pointer to a live struct we exported (or
    // null, which we ignore).
    let Some(schema) = (unsafe { schema.as_mut() }) else {
        return;
    };
    if schema.release.is_none() || schema.private_data.is_null() {
        return;
    }
    // SAFETY: `private_data` came from `Box::into_raw(Box<StructSchemaPrivate>)` in
    // `export_struct_schema` and is freed only here, once: we clear it right after.
    let private = unsafe { Box::from_raw(schema.private_data.cast::<StructSchemaPrivate>()) };
    for child in private.children {
        // SAFETY: each child came from `Box::into_raw` in `export_struct_schema` and is
        // freed only here. Dropping it releases it unless a consumer moved it out.
        drop(unsafe { Box::from_raw(child) });
    }
    schema.private_data = ptr::null_mut();
    schema.children = ptr::null_mut();
    schema.release = None;
}

// ---- export: the struct ArrowArray -------------------------------------------

struct StructArrayPrivate {
    /// Each from `Box::into_raw`; freed by `release_struct_array`.
    children: Vec<*mut FfiArray>,
    /// The `buffers` C array: the (absent) validity bitmap.
    buffers: [*const c_void; 1],
}

pub(super) fn export_struct_array(batch: &RecordBatch) -> FfiArray {
    let children = batch
        .fields
        .iter()
        .map(|f| Box::into_raw(Box::new(export_array(Arc::clone(&f.chunk)))))
        .collect::<Vec<_>>();
    let n_children = to_i64(children.len());
    let private = Box::into_raw(Box::new(StructArrayPrivate {
        children,
        buffers: [ptr::null()],
    }));
    // SAFETY: `private` is a valid, live allocation we just created.
    let buffers_ptr = unsafe { &raw mut (*private).buffers }.cast::<*const c_void>();
    // SAFETY: as above.
    let children_ptr = unsafe { (*private).children.as_mut_ptr() };
    FfiArray {
        length: to_i64(batch.num_rows),
        null_count: 0,
        offset: 0,
        n_buffers: 1,
        n_children,
        buffers: buffers_ptr,
        children: children_ptr,
        dictionary: ptr::null_mut(),
        release: Some(release_struct_array),
        private_data: private.cast::<c_void>(),
    }
}

unsafe extern "C" fn release_struct_array(array: *mut FfiArray) {
    // SAFETY: the consumer passes a pointer to a live struct we exported (or
    // null, which we ignore).
    let Some(array) = (unsafe { array.as_mut() }) else {
        return;
    };
    if array.release.is_none() || array.private_data.is_null() {
        return;
    }
    // SAFETY: `private_data` came from `Box::into_raw(Box<StructArrayPrivate>)` in
    // `export_struct_array` and is freed only here, once: we clear it right after.
    let private = unsafe { Box::from_raw(array.private_data.cast::<StructArrayPrivate>()) };
    for child in private.children {
        // SAFETY: each child came from `Box::into_raw` in `export_struct_array` and is
        // freed only here. Dropping it releases it unless a consumer moved it out.
        drop(unsafe { Box::from_raw(child) });
    }
    array.private_data = ptr::null_mut();
    array.buffers = ptr::null_mut();
    array.children = ptr::null_mut();
    array.release = None;
}

// ---- export: a stream of one batch -------------------------------------------

struct BatchStreamPrivate {
    batch: Arc<RecordBatch>,
    done: bool,
}

/// `EINVAL`, the C Stream Interface's "invalid argument" code.
const EINVAL: c_int = 22;

/// Our stream's private data.
///
/// # Safety
/// `stream` must be null or a live stream exported by `export_batch_stream`, and the
/// caller must not use the result after the stream is released.
unsafe fn batch_stream_private<'a>(stream: *mut FfiStream) -> Option<&'a mut BatchStreamPrivate> {
    // SAFETY: null or a live stream, per the caller's contract.
    let stream = unsafe { stream.as_mut() }?;
    // SAFETY: `private_data` is our `Box<BatchStreamPrivate>` (or null after release). The
    // C Stream spec requires consumers to serialize calls on one stream.
    unsafe { stream.private_data.cast::<BatchStreamPrivate>().as_mut() }
}

unsafe extern "C" fn batch_stream_get_schema(stream: *mut FfiStream, out: *mut FfiSchema) -> c_int {
    // SAFETY: the consumer passes our live stream.
    let Some(private) = (unsafe { batch_stream_private(stream) }) else {
        return EINVAL;
    };
    if out.is_null() {
        return EINVAL;
    }
    let schema = export_struct_schema(&private.batch);
    // SAFETY: `out` points to a writable ArrowSchema that the consumer owns and has not
    // initialized (or has released), so no drop is skipped.
    unsafe { ptr::write(out, schema) };
    0
}

unsafe extern "C" fn batch_stream_get_next(stream: *mut FfiStream, out: *mut FfiArray) -> c_int {
    // SAFETY: the consumer passes our live stream.
    let Some(private) = (unsafe { batch_stream_private(stream) }) else {
        return EINVAL;
    };
    if out.is_null() {
        return EINVAL;
    }
    let array = if private.done {
        FfiArray::released() // end of stream
    } else {
        private.done = true;
        export_struct_array(&private.batch)
    };
    // SAFETY: as in `batch_stream_get_schema`.
    unsafe { ptr::write(out, array) };
    0
}

unsafe extern "C" fn batch_stream_get_last_error(_stream: *mut FfiStream) -> *const c_char {
    ptr::null() // our callbacks only fail with EINVAL on misuse
}

unsafe extern "C" fn release_batch_stream(stream: *mut FfiStream) {
    // SAFETY: the consumer passes a pointer to a live struct we exported (or
    // null, which we ignore).
    let Some(stream) = (unsafe { stream.as_mut() }) else {
        return;
    };
    if stream.release.is_none() || stream.private_data.is_null() {
        return;
    }
    // SAFETY: `private_data` came from `Box::into_raw(Box<BatchStreamPrivate>)` in
    // `export_batch_stream` and is freed only here, once: we clear it right after.
    drop(unsafe { Box::from_raw(stream.private_data.cast::<BatchStreamPrivate>()) });
    stream.private_data = ptr::null_mut();
    stream.release = None;
}

pub(super) fn export_batch_stream(batch: &Arc<RecordBatch>) -> FfiStream {
    let private = Box::new(BatchStreamPrivate {
        batch: Arc::clone(batch),
        done: false,
    });
    FfiStream {
        get_schema: Some(batch_stream_get_schema),
        get_next: Some(batch_stream_get_next),
        get_last_error: Some(batch_stream_get_last_error),
        release: Some(release_batch_stream),
        private_data: Box::into_raw(private).cast::<c_void>(),
    }
}

// ---- capsules ----------------------------------------------------------------

/// `__arrow_c_schema__`: the batch's struct schema.
pub fn schema_capsule<'py>(
    py: Python<'py>,
    batch: &RecordBatch,
) -> PyResult<Bound<'py, PyCapsule>> {
    capsule(py, export_struct_schema(batch), SCHEMA_CAPSULE)
}

/// `__arrow_c_array__`: `(schema, array)` capsules of the batch as a struct array.
pub fn array_capsules<'py>(py: Python<'py>, batch: &RecordBatch) -> PyResult<Bound<'py, PyTuple>> {
    let schema = capsule(py, export_struct_schema(batch), SCHEMA_CAPSULE)?;
    let array = capsule(py, export_struct_array(batch), ARRAY_CAPSULE)?;
    PyTuple::new(py, [schema, array])
}

/// `__arrow_c_stream__`: a stream that yields the batch once.
pub fn stream_capsule<'py>(
    py: Python<'py>,
    batch: &Arc<RecordBatch>,
) -> PyResult<Bound<'py, PyCapsule>> {
    capsule(py, export_batch_stream(batch), STREAM_CAPSULE)
}

#[cfg(test)]
mod tests;
