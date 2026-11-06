//! The C Data / C Stream exporter: the exported structs, their release
//! callbacks (twice, null, after a move, on another thread, children after
//! the stream), the stream callbacks, and export -> import round trips of
//! buffers built with `pyroid_core::columnar`. No Python (only plain Rust
//! methods of `Column`), so these run under Miri and the sanitizers; Miri's
//! leak check also proves every release frees what the export allocated.

use std::ffi::{CStr, c_void};
use std::ptr;
use std::slice;
use std::sync::Arc;

use pyroid_core::columnar::{BoolBuffers, StringBuffers, get_bit};
use pyroid_core::par::Threads;

use super::{
    Chunk, Column, ColumnType, EINVAL, Exported, export_array, export_schema, export_stream,
    release_array, release_schema, release_stream, stream_get_last_error, stream_get_next,
    stream_get_schema,
};
use crate::arrow::ffi::{FLAG_NULLABLE, FfiArray, FfiSchema, FfiStream};
use crate::arrow::import::{StrChunk, StrType};

fn utf8(rows: &[Option<&str>]) -> StringBuffers<i32> {
    StringBuffers::build(rows, Threads::ONE).unwrap()
}

fn large(rows: &[Option<&str>]) -> StringBuffers<i64> {
    StringBuffers::build(rows, Threads::ONE).unwrap()
}

fn sample() -> Vec<Option<String>> {
    let long = "a row longer than one 64-byte SIMD block: ".repeat(2) + "ünïcödé ✓";
    [
        Some("first"),
        None,
        Some(""),
        Some("héllo"),
        Some(long.as_str()),
        None,
        Some("日本語"),
    ]
    .iter()
    .map(|r| r.map(str::to_owned))
    .collect()
}

fn as_strs(rows: &[Option<String>]) -> Vec<Option<&str>> {
    rows.iter().map(Option::as_deref).collect()
}

/// The exported `buffers` C array.
fn buffers(array: &FfiArray) -> &[*const c_void] {
    let n = usize::try_from(array.n_buffers).unwrap();
    // SAFETY: an exported array's `buffers` holds `n_buffers` pointers, and
    // lives (in the private data) until the array is released.
    unsafe { slice::from_raw_parts(array.buffers.cast_const(), n) }
}

/// A C string the exporter owns (`format` / `name`).
fn c_str<'a>(p: *const std::ffi::c_char) -> &'a CStr {
    assert!(!p.is_null());
    // SAFETY: the exporter's strings are NUL-terminated and live until the
    // struct is released; callers don't keep the result past that.
    unsafe { CStr::from_ptr(p) }
}

/// Import an exported string array and check its rows against `rows`.
fn assert_round_trip(array: FfiArray, ty: StrType, rows: &[Option<&str>]) {
    let chunk = StrChunk::new(array, ty).unwrap();
    assert_eq!(chunk.len(), rows.len());
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(chunk.get(i), row.map(Ok), "row {i}");
        assert_eq!(chunk.bytes(i), row.map(str::as_bytes), "row {i}");
    }
}

/// Take the struct out of its `Send` wrapper (on the receiving thread).
fn unwrap<T>(e: Exported<T>) -> T {
    e.0
}

// ---- ArrowSchema ------------------------------------------------------------

#[test]
fn export_schema_fields() {
    for (ty, format) in [
        (ColumnType::Utf8, c"u"),
        (ColumnType::LargeUtf8, c"U"),
        (ColumnType::Bool, c"b"),
    ] {
        let schema = export_schema(ty, c"text");
        assert_eq!(c_str(schema.format), format);
        assert_eq!(c_str(schema.name), c"text");
        assert!(schema.metadata.is_null());
        assert_eq!((schema.flags, schema.n_children), (FLAG_NULLABLE, 0));
        assert!(schema.children.is_null() && schema.dictionary.is_null());
        assert!(schema.release.is_some() && !schema.private_data.is_null());
    }
}

#[test]
fn release_schema_once_twice_and_null() {
    let mut schema = export_schema(ColumnType::Utf8, c"");
    assert_eq!(c_str(schema.name), c"");
    // SAFETY: our live exported schema.
    unsafe { release_schema(&raw mut schema) };
    assert!(schema.release.is_none() && schema.private_data.is_null());
    // SAFETY: a released struct: the callback must ignore it.
    unsafe { release_schema(&raw mut schema) };
    // SAFETY: null is ignored.
    unsafe { release_schema(ptr::null_mut()) };
    drop(schema); // released: no callback
}

#[test]
fn a_moved_schema_stays_valid_and_releases_once() {
    let schema = export_schema(ColumnType::LargeUtf8, c"moved");
    // The spec allows consumers to move the struct (a bitwise copy) before
    // releasing it; nothing in it points at its own address.
    let mut moved = vec![FfiSchema::released(), schema];
    let mut boxed = Box::new(moved.pop().unwrap());
    assert_eq!(c_str(boxed.name), c"moved");
    // SAFETY: the live schema, at its new address.
    unsafe { release_schema(&raw mut *boxed) };
    assert!(boxed.release.is_none());
}

// ---- ArrowArray -------------------------------------------------------------

#[test]
fn export_array_utf8_and_large_utf8_fields() {
    let rows = sample();
    let rows = as_strs(&rows);
    let chunks = [Chunk::Utf8(utf8(&rows)), Chunk::LargeUtf8(large(&rows))];
    for chunk in chunks {
        let chunk = Arc::new(chunk);
        let mut array = export_array(Arc::clone(&chunk));
        assert_eq!(Arc::strong_count(&chunk), 2);
        assert_eq!((array.length, array.null_count, array.offset), (7, 2, 0));
        assert_eq!((array.n_buffers, array.n_children), (3, 0));
        assert!(array.children.is_null() && array.dictionary.is_null());
        let (validity, offsets, data) = match &*chunk {
            Chunk::Utf8(b) => (
                &b.validity,
                b.offsets.as_ptr().cast::<c_void>(),
                b.data.as_ptr(),
            ),
            Chunk::LargeUtf8(b) => (
                &b.validity,
                b.offsets.as_ptr().cast::<c_void>(),
                b.data.as_ptr(),
            ),
            Chunk::Bool(_) => unreachable!(),
        };
        let bufs = buffers(&array);
        assert_eq!(bufs[0], validity.as_ref().unwrap().as_ptr().cast());
        assert_eq!(bufs[1], offsets);
        assert_eq!(bufs[2], data.cast());

        // SAFETY: our live exported array.
        unsafe { release_array(&raw mut array) };
        assert!(array.release.is_none() && array.private_data.is_null());
        assert!(array.buffers.is_null());
        assert_eq!(Arc::strong_count(&chunk), 1); // the export's reference is gone
        // SAFETY: released: ignored.
        unsafe { release_array(&raw mut array) };
        // SAFETY: null: ignored.
        unsafe { release_array(ptr::null_mut()) };
        assert_eq!(Arc::strong_count(&chunk), 1);
    }
}

#[test]
fn export_array_without_nulls_has_a_null_bitmap() {
    let chunk = Arc::new(Chunk::Utf8(utf8(&[Some("a"), Some("b")])));
    let array = export_array(Arc::clone(&chunk));
    assert_eq!(array.null_count, 0);
    assert!(buffers(&array)[0].is_null());
    assert_round_trip(array, StrType::Utf8, &[Some("a"), Some("b")]);
    assert_eq!(Arc::strong_count(&chunk), 1);
}

#[test]
fn export_array_bool() {
    let values = [
        Some(true),
        None,
        Some(false),
        Some(true),
        None,
        Some(true),
        Some(false),
        Some(true),
        Some(true),
    ];
    let bools = BoolBuffers::build(&values);
    let chunk = Arc::new(Chunk::Bool(bools));
    let array = export_array(Arc::clone(&chunk));
    assert_eq!((array.length, array.null_count, array.n_buffers), (9, 2, 2));
    let bufs = buffers(&array);
    // SAFETY: the exported bitmaps hold ceil(9 / 8) bytes each.
    let validity = unsafe { slice::from_raw_parts(bufs[0].cast::<u8>(), 2) };
    // SAFETY: as above.
    let bits = unsafe { slice::from_raw_parts(bufs[1].cast::<u8>(), 2) };
    for (i, v) in values.iter().enumerate() {
        let got = get_bit(validity, i).then(|| get_bit(bits, i));
        assert_eq!(got, *v, "row {i}");
    }
    drop(array); // `FfiArray::drop` calls `release_array`
    assert_eq!(Arc::strong_count(&chunk), 1);

    // No nulls: no bitmap.
    let chunk = Arc::new(Chunk::Bool(BoolBuffers::build(&[Some(false)])));
    let array = export_array(chunk);
    assert!(buffers(&array)[0].is_null());
}

#[test]
fn export_import_round_trip() {
    let rows = sample();
    let rows = as_strs(&rows);
    let array = export_array(Arc::new(Chunk::Utf8(utf8(&rows))));
    assert_round_trip(array, StrType::Utf8, &rows);
    let array = export_array(Arc::new(Chunk::LargeUtf8(large(&rows))));
    assert_round_trip(array, StrType::LargeUtf8, &rows);
    // build_split's parts (one here) and an empty array.
    for part in StringBuffers::<i32>::build_split(&rows, Threads::ONE).unwrap() {
        assert_round_trip(
            export_array(Arc::new(Chunk::Utf8(part))),
            StrType::Utf8,
            &rows,
        );
    }
    let empty = export_array(Arc::new(Chunk::empty(ColumnType::Utf8)));
    assert_round_trip(empty, StrType::Utf8, &[]);
    let empty = export_array(Arc::new(Chunk::empty(ColumnType::LargeUtf8)));
    assert_round_trip(empty, StrType::LargeUtf8, &[]);
}

#[test]
fn a_moved_array_stays_valid_and_releases_once() {
    let rows = [Some("moved"), None, Some("row")];
    let chunk = Arc::new(Chunk::LargeUtf8(large(&rows)));
    let array = export_array(Arc::clone(&chunk));
    let mut arrays = vec![FfiArray::released(), FfiArray::released(), array];
    arrays.swap(0, 2); // a consumer moving structs around in its own storage
    let moved = arrays.swap_remove(0);
    drop(arrays);
    let boxed = Box::new(moved);
    assert_round_trip(*boxed, StrType::LargeUtf8, &rows);
    assert_eq!(Arc::strong_count(&chunk), 1);
}

#[test]
fn exported_structs_can_be_released_and_read_on_another_thread() {
    // What `Exported`'s `Send` impls promise: polars and pyarrow move and
    // release our structs on their own threads.
    let rows = sample();
    let rows = as_strs(&rows);
    let chunk = Arc::new(Chunk::Utf8(utf8(&rows)));
    let column = Column::new(ColumnType::Utf8, "t", vec![Chunk::Utf8(utf8(&rows))]);
    let to_drop = Exported(export_array(Arc::clone(&chunk)));
    let to_read = Exported(export_array(Arc::clone(&chunk)));
    let schema = Exported(export_schema(ColumnType::Utf8, c"t"));
    let stream = Exported(export_stream(&column.inner));
    std::thread::scope(|s| {
        s.spawn(move || drop(to_drop));
        s.spawn(|| assert_round_trip(unwrap(to_read), StrType::Utf8, &rows));
        s.spawn(move || assert_eq!(c_str(unwrap(schema).name), c"t"));
        s.spawn(move || {
            let mut stream = unwrap(stream);
            let mut array = FfiArray::released();
            // SAFETY: our live stream, and a released out-struct.
            let code = unsafe { stream_get_next(&raw mut stream, &raw mut array) };
            assert_eq!(code, 0);
            assert_eq!(array.length, 7);
        });
    });
    assert_eq!(Arc::strong_count(&chunk), 1);
    assert_eq!(Arc::strong_count(&column.inner), 1);
}

// ---- ArrowArrayStream -------------------------------------------------------

fn get_schema(stream: &mut FfiStream) -> (i32, FfiSchema) {
    let mut out = FfiSchema::released();
    // SAFETY: `stream` is ours (live or released), and `out` a released struct.
    let code = unsafe { stream_get_schema(stream, &raw mut out) };
    (code, out)
}

fn get_next(stream: &mut FfiStream) -> (i32, FfiArray) {
    let mut out = FfiArray::released();
    // SAFETY: as above.
    let code = unsafe { stream_get_next(stream, &raw mut out) };
    (code, out)
}

#[test]
fn stream_yields_schema_then_non_empty_chunks_then_end() {
    let first = [Some("a"), None, Some("bc")];
    let second = [Some("d")];
    let column = Column::new(
        ColumnType::LargeUtf8,
        "te\0xt", // the NUL is dropped
        vec![
            Chunk::LargeUtf8(large(&first)),
            Chunk::empty(ColumnType::LargeUtf8),
            Chunk::LargeUtf8(large(&second)),
        ],
    );
    assert_eq!(column.inner.chunks.len(), 2);
    let mut stream = export_stream(&column.inner);

    let (code, schema) = get_schema(&mut stream);
    assert_eq!(code, 0);
    assert_eq!(c_str(schema.format), c"U");
    assert_eq!(c_str(schema.name), c"text");
    // A second get_schema is a fresh, independently owned struct.
    let (code, again) = get_schema(&mut stream);
    assert_eq!(code, 0);
    assert_ne!(again.private_data, schema.private_data);
    drop(again);

    let (code, a) = get_next(&mut stream);
    assert_eq!(code, 0);
    let a = StrChunk::new(a, StrType::LargeUtf8).unwrap();
    let (code, b) = get_next(&mut stream);
    assert_eq!(code, 0);
    let b = StrChunk::new(b, StrType::LargeUtf8).unwrap();
    for _ in 0..2 {
        let (code, end) = get_next(&mut stream);
        assert_eq!(code, 0);
        assert!(end.release.is_none(), "end of stream is a released array");
    }
    // SAFETY: our live stream.
    assert!(unsafe { stream_get_last_error(&raw mut stream) }.is_null());

    // Release the stream first: its children must stay valid on their own.
    // SAFETY: our live stream.
    unsafe { release_stream(&raw mut stream) };
    assert!(stream.release.is_none() && stream.private_data.is_null());
    assert_eq!(Arc::strong_count(&column.inner), 1);
    assert_eq!(c_str(schema.name), c"text");
    assert_eq!(a.get(0), Some(Ok("a")));
    assert_eq!(a.get(1), None);
    assert_eq!(a.get(2), Some(Ok("bc")));
    assert_eq!(b.get(0), Some(Ok("d")));
    drop(column); // the chunks outlive the column, too
    assert_eq!(a.get(2), Some(Ok("bc")));
    drop((a, b, schema));
}

#[test]
fn stream_callbacks_reject_misuse() {
    let column = Column::new(ColumnType::Utf8, "", vec![Chunk::Utf8(utf8(&[Some("x")]))]);
    let mut stream = export_stream(&column.inner);
    // SAFETY: a null out-pointer is rejected before any write.
    let code = unsafe { stream_get_schema(&raw mut stream, ptr::null_mut()) };
    assert_eq!(code, EINVAL);
    // SAFETY: as above.
    let code = unsafe { stream_get_next(&raw mut stream, ptr::null_mut()) };
    assert_eq!(code, EINVAL);
    let (code, schema) = get_schema(&mut stream);
    assert_eq!((code, c_str(schema.name)), (0, c""));
    // A null stream.
    let mut out = FfiArray::released();
    // SAFETY: a null stream is rejected.
    let code = unsafe { stream_get_next(ptr::null_mut(), &raw mut out) };
    assert_eq!(code, EINVAL);
    let mut out_schema = FfiSchema::released();
    // SAFETY: as above.
    let code = unsafe { stream_get_schema(ptr::null_mut(), &raw mut out_schema) };
    assert_eq!(code, EINVAL);
    // SAFETY: null is fine for get_last_error.
    assert!(unsafe { stream_get_last_error(ptr::null_mut()) }.is_null());
    // SAFETY: null is ignored.
    unsafe { release_stream(ptr::null_mut()) };

    // After release: every callback refuses, and a second release is a no-op.
    // SAFETY: our live stream.
    unsafe { release_stream(&raw mut stream) };
    // SAFETY: released: ignored.
    unsafe { release_stream(&raw mut stream) };
    assert_eq!(get_schema(&mut stream).0, EINVAL);
    let (code, array) = get_next(&mut stream);
    assert_eq!(code, EINVAL);
    assert!(array.release.is_none());
    assert_eq!(Arc::strong_count(&column.inner), 1);
}

#[test]
fn a_moved_stream_keeps_its_position() {
    let column = Column::new(
        ColumnType::Utf8,
        "s",
        vec![
            Chunk::Utf8(utf8(&[Some("one")])),
            Chunk::Utf8(utf8(&[Some("two")])),
            Chunk::Utf8(utf8(&[Some("three")])),
        ],
    );
    let mut stream = export_stream(&column.inner);
    let (_, first) = get_next(&mut stream);
    let mut moved = Box::new(stream); // the consumer moves the struct
    let (_, second) = get_next(&mut moved);
    let mut moved_again = vec![*moved];
    let (_, third) = get_next(&mut moved_again[0]);
    let (_, end) = get_next(&mut moved_again[0]);
    assert!(end.release.is_none());
    drop(moved_again); // `FfiStream::drop` releases it
    assert_eq!(Arc::strong_count(&column.inner), 1);
    assert_round_trip(first, StrType::Utf8, &[Some("one")]);
    assert_round_trip(second, StrType::Utf8, &[Some("two")]);
    assert_round_trip(third, StrType::Utf8, &[Some("three")]);
}

#[test]
fn a_bool_stream() {
    let values = [Some(true), None, Some(false)];
    let column = Column::new(
        ColumnType::Bool,
        "flags",
        vec![Chunk::Bool(BoolBuffers::build(&values))],
    );
    let mut stream = export_stream(&column.inner);
    let (_, schema) = get_schema(&mut stream);
    assert_eq!(c_str(schema.format), c"b");
    let (code, array) = get_next(&mut stream);
    assert_eq!(
        (code, array.length, array.null_count, array.n_buffers),
        (0, 3, 1, 2)
    );
    let bufs = buffers(&array);
    // SAFETY: one byte holds the 3 validity bits.
    let validity = unsafe { bufs[0].cast::<u8>().read() };
    // SAFETY: one byte holds the 3 value bits.
    let bits = unsafe { bufs[1].cast::<u8>().read() };
    assert_eq!((validity, bits & validity), (0b101, 0b001));
}

#[test]
fn an_empty_column_streams_nothing() {
    for ty in [ColumnType::Utf8, ColumnType::LargeUtf8, ColumnType::Bool] {
        let column = Column::new(ty, "e", vec![Chunk::empty(ty)]);
        assert!(column.inner.chunks.is_empty());
        let mut stream = export_stream(&column.inner);
        let (code, end) = get_next(&mut stream);
        assert_eq!(code, 0);
        assert!(end.release.is_none());
    }
}

// ---- single_chunk (what __arrow_c_array__ exports) --------------------------

#[test]
fn single_chunk_concatenates_once_and_round_trips() {
    let rows = sample();
    let rows = as_strs(&rows);
    let (head, tail) = rows.split_at(3);
    let column = Column::new(
        ColumnType::Utf8,
        "joined",
        vec![Chunk::Utf8(utf8(head)), Chunk::Utf8(utf8(tail))],
    );
    let Ok(joined) = column.single_chunk() else {
        panic!("concat failed")
    };
    let Ok(again) = column.single_chunk() else {
        panic!("concat failed")
    };
    assert!(Arc::ptr_eq(&joined, &again), "the concatenation is cached");
    drop(again);
    assert_round_trip(export_array(Arc::clone(&joined)), StrType::Utf8, &rows);
    assert_eq!(Arc::strong_count(&joined), 2); // ours + the column's cache

    // One chunk is exported as is; no chunks as an empty array.
    let one = Column::new(
        ColumnType::LargeUtf8,
        "",
        vec![Chunk::LargeUtf8(large(&rows))],
    );
    let Ok(chunk) = one.single_chunk() else {
        panic!("single chunk")
    };
    assert!(Arc::ptr_eq(&chunk, &one.inner.chunks[0]));
    assert_round_trip(export_array(chunk), StrType::LargeUtf8, &rows);
    for ty in [ColumnType::Utf8, ColumnType::Bool] {
        let none = Column::new(ty, "", Vec::new());
        let Ok(empty) = none.single_chunk() else {
            panic!("empty")
        };
        let array = export_array(empty);
        assert_eq!(array.length, 0);
    }

    let bools = Column::new(
        ColumnType::Bool,
        "",
        vec![
            Chunk::Bool(BoolBuffers::build(&[Some(true), None])),
            Chunk::Bool(BoolBuffers::build(&[Some(false), Some(true)])),
        ],
    );
    let Ok(chunk) = bools.single_chunk() else {
        panic!("bool concat")
    };
    let array = export_array(chunk);
    assert_eq!((array.length, array.null_count), (4, 1));
}
