//! The record-batch exporter: struct schemas and arrays with their children, children a
//! consumer moves out before releasing the parent, the one-batch stream, and release
//! callbacks called twice or with null. No Python, so these run under Miri (its leak check
//! proves each release frees what the export allocated).

use std::ffi::{CStr, CString, c_char};
use std::ptr;
use std::slice;
use std::sync::Arc;

use pyroid_core::columnar::StringBuffers;
use pyroid_core::par::Threads;

use super::{
    EINVAL, Field, RecordBatch, batch_stream_get_last_error, batch_stream_get_next,
    batch_stream_get_schema, export_batch_stream, export_struct_array, export_struct_schema,
    release_struct_array, release_struct_schema,
};
use crate::arrow::column::{Chunk, ColumnType};
use crate::arrow::ffi::{FLAG_NULLABLE, FfiArray, FfiSchema, FfiStream, move_out};
use crate::arrow::import::{StrChunk, StrType};

fn large(rows: &[Option<&str>]) -> StringBuffers<i64> {
    StringBuffers::build(rows, Threads::ONE).unwrap()
}

const TEXT: [Option<&str>; 3] = [Some("<p>one</p>"), None, Some("tröis")];
const BYTES: [Option<&[u8]>; 3] = [Some(b"\xff\x00raw"), Some(b""), None];

fn binary(rows: &[Option<&[u8]>]) -> StringBuffers<i64> {
    let mut offsets = vec![0i64];
    let mut data = Vec::new();
    let mut valid = Vec::new();
    for r in rows {
        if let Some(b) = r {
            data.extend_from_slice(b);
        }
        valid.push(r.is_some());
        offsets.push(i64::try_from(data.len()).unwrap());
    }
    let null_count = valid.iter().filter(|v| !**v).count();
    StringBuffers {
        len: rows.len(),
        offsets,
        data,
        validity: Some(pyroid_core::columnar::bitmap(valid, rows.len())),
        null_count,
    }
}

fn batch() -> Arc<RecordBatch> {
    Arc::new(RecordBatch::new(vec![
        Field {
            name: CString::new("text").unwrap(),
            ty: ColumnType::LargeUtf8,
            chunk: Arc::new(Chunk::LargeUtf8(large(&TEXT))),
        },
        Field {
            name: CString::new("html").unwrap(),
            ty: ColumnType::LargeBinary,
            chunk: Arc::new(Chunk::LargeBinary(binary(&BYTES))),
        },
    ]))
}

fn c_str<'a>(p: *const c_char) -> &'a CStr {
    assert!(!p.is_null());
    // SAFETY: the exporter's strings are NUL-terminated and live until the struct is
    // released; callers don't keep the result past that.
    unsafe { CStr::from_ptr(p) }
}

fn schema_children(s: &FfiSchema) -> &[*mut FfiSchema] {
    let n = usize::try_from(s.n_children).unwrap();
    // SAFETY: an exported struct schema's `children` holds `n_children` pointers that live
    // until it is released.
    unsafe { slice::from_raw_parts(s.children.cast_const(), n) }
}

fn array_children(a: &FfiArray) -> &[*mut FfiArray] {
    let n = usize::try_from(a.n_children).unwrap();
    // SAFETY: as for `schema_children`.
    unsafe { slice::from_raw_parts(a.children.cast_const(), n) }
}

#[test]
fn struct_schema_fields_and_children() {
    let b = batch();
    let mut schema = export_struct_schema(&b);
    assert_eq!(c_str(schema.format), c"+s");
    assert_eq!(c_str(schema.name), c"");
    assert_eq!((schema.flags, schema.n_children), (0, 2));
    assert!(schema.dictionary.is_null() && schema.metadata.is_null());
    let want = [(c"text", c"U"), (c"html", c"Z")];
    for (&child, (name, format)) in schema_children(&schema).iter().zip(want) {
        // SAFETY: a live child of a live exported schema.
        let child = unsafe { &*child };
        assert_eq!((c_str(child.name), c_str(child.format)), (name, format));
        assert_eq!((child.flags, child.n_children), (FLAG_NULLABLE, 0));
        assert!(child.release.is_some());
    }
    // SAFETY: our live exported schema.
    unsafe { release_struct_schema(&raw mut schema) };
    assert!(schema.release.is_none() && schema.private_data.is_null());
    assert!(schema.children.is_null());
    // SAFETY: released: ignored.
    unsafe { release_struct_schema(&raw mut schema) };
    // SAFETY: null: ignored.
    unsafe { release_struct_schema(ptr::null_mut()) };
}

#[test]
fn struct_array_fields_children_and_round_trip() {
    let b = batch();
    let mut array = export_struct_array(&b);
    assert_eq!((array.length, array.null_count, array.offset), (3, 0, 0));
    assert_eq!((array.n_buffers, array.n_children), (1, 2));
    // SAFETY: an exported array's `buffers` holds `n_buffers` pointers.
    let bufs = unsafe { slice::from_raw_parts(array.buffers.cast_const(), 1) };
    assert!(bufs[0].is_null()); // no validity bitmap: no null rows
    let children = array_children(&array).to_vec();
    // SAFETY: a live child of a live exported array.
    let html = unsafe { &*children[1] };
    assert_eq!((html.length, html.null_count, html.n_buffers), (3, 1, 3));
    // SAFETY: as above; `large_binary` buffers are validity, i64 offsets and data.
    let hb = unsafe { slice::from_raw_parts(html.buffers.cast_const(), 3) };
    // SAFETY: four offsets for three rows.
    let offs = unsafe { slice::from_raw_parts(hb[1].cast::<i64>(), 4) };
    assert_eq!(offs, [0, 5, 5, 5]);
    // SAFETY: five data bytes.
    let data = unsafe { slice::from_raw_parts(hb[2].cast::<u8>(), 5) };
    assert_eq!(data, b"\xff\x00raw");

    // A consumer moves the text child out, releases the parent, then reads the child.
    // SAFETY: a live child struct, which nobody else touches.
    let text = unsafe { move_out(children[0]) };
    // SAFETY: our live exported array.
    unsafe { release_struct_array(&raw mut array) };
    assert!(array.release.is_none() && array.children.is_null() && array.buffers.is_null());
    let chunk = StrChunk::new(text, StrType::LargeUtf8).unwrap();
    for (i, row) in TEXT.iter().enumerate() {
        assert_eq!(chunk.get(i), row.map(Ok), "row {i}");
    }
    drop(chunk);
    assert_eq!(Arc::strong_count(&b), 1);
    for f in &b.fields {
        assert_eq!(Arc::strong_count(&f.chunk), 1, "{:?}", f.name);
    }
    // SAFETY: released / null: ignored.
    unsafe { release_struct_array(&raw mut array) };
    // SAFETY: as above.
    unsafe { release_struct_array(ptr::null_mut()) };
}

#[test]
fn an_empty_batch_exports() {
    let b = RecordBatch::new(vec![Field {
        name: CString::new("id").unwrap(),
        ty: ColumnType::LargeUtf8,
        chunk: Arc::new(Chunk::LargeUtf8(large(&[]))),
    }]);
    let array = export_struct_array(&b);
    assert_eq!((array.length, array.n_children), (0, 1));
    drop(array);
    let none = RecordBatch::new(Vec::new());
    let schema = export_struct_schema(&none);
    assert_eq!(schema.n_children, 0);
    drop(schema);
    drop(export_struct_array(&none));
}

fn get_schema(stream: &mut FfiStream) -> (i32, FfiSchema) {
    let mut out = FfiSchema::released();
    // SAFETY: `stream` is ours (live or released), and `out` a released struct.
    let code = unsafe { batch_stream_get_schema(stream, &raw mut out) };
    (code, out)
}

fn get_next(stream: &mut FfiStream) -> (i32, FfiArray) {
    let mut out = FfiArray::released();
    // SAFETY: as above.
    let code = unsafe { batch_stream_get_next(stream, &raw mut out) };
    (code, out)
}

#[test]
fn the_stream_yields_the_batch_once() {
    let b = batch();
    let mut stream = export_batch_stream(&b);
    let (code, schema) = get_schema(&mut stream);
    assert_eq!(code, 0);
    assert_eq!((c_str(schema.format), schema.n_children), (c"+s", 2));
    let (code, first) = get_next(&mut stream);
    assert_eq!((code, first.length, first.n_children), (0, 3, 2));
    let (code, end) = get_next(&mut stream);
    assert_eq!(code, 0);
    assert!(end.release.is_none()); // end of stream
    // SAFETY: our live stream.
    assert!(unsafe { batch_stream_get_last_error(&raw mut stream) }.is_null());
    drop(stream); // releases it; the batch lives on in `first`'s children
    drop(schema);
    assert_eq!(Arc::strong_count(&b), 1);
    drop(first);
    for f in &b.fields {
        assert_eq!(Arc::strong_count(&f.chunk), 1);
    }
    let mut released = FfiStream::released();
    assert_eq!(get_schema(&mut released).0, EINVAL);
    assert_eq!(get_next(&mut released).0, EINVAL);
    // SAFETY: null stream / null out-struct: rejected.
    let code = unsafe { batch_stream_get_next(ptr::null_mut(), ptr::null_mut()) };
    assert_eq!(code, EINVAL);
    let mut live = export_batch_stream(&b);
    // SAFETY: our live stream, a null out-struct: rejected.
    let code = unsafe { batch_stream_get_schema(&raw mut live, ptr::null_mut()) };
    assert_eq!(code, EINVAL);
}
