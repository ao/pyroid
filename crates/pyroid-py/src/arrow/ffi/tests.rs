//! The FFI structs' ownership rules: `Drop` releases a live struct exactly
//! once, `move_out` transfers ownership and leaves a released source, and the
//! `released()` values own nothing. Pure Rust (no Python), so these run under
//! Miri and the sanitizers.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{FfiArray, FfiSchema, FfiStream, Released, move_out};

/// A new release counter, and a reference to it for a struct's `private_data`.
fn counter() -> (Arc<AtomicUsize>, *mut c_void) {
    let count = Arc::new(AtomicUsize::new(0));
    let private = Arc::into_raw(Arc::clone(&count))
        .cast_mut()
        .cast::<c_void>();
    (count, private)
}

/// Count one release and drop the struct's reference to the counter.
///
/// # Safety
/// `private_data` must come from [`counter`] and not have been released.
unsafe fn count_release(private_data: *mut c_void) {
    // SAFETY: from `Arc::into_raw` in `counter`; the callbacks below clear
    // `private_data` and `release`, so this runs once per struct.
    let count = unsafe { Arc::from_raw(private_data.cast_const().cast::<AtomicUsize>()) };
    count.fetch_add(1, Ordering::SeqCst);
}

unsafe extern "C" fn release_schema(schema: *mut FfiSchema) {
    // SAFETY: `FfiSchema::drop` passes the live struct.
    let schema = unsafe { &mut *schema };
    // SAFETY: set by `live_schema`, and unreleased (release is still set).
    unsafe { count_release(schema.private_data) };
    schema.private_data = ptr::null_mut();
    schema.release = None;
}

unsafe extern "C" fn release_array(array: *mut FfiArray) {
    // SAFETY: `FfiArray::drop` passes the live struct.
    let array = unsafe { &mut *array };
    // SAFETY: set by `live_array`, and unreleased.
    unsafe { count_release(array.private_data) };
    array.private_data = ptr::null_mut();
    array.release = None;
}

unsafe extern "C" fn release_stream(stream: *mut FfiStream) {
    // SAFETY: `FfiStream::drop` passes the live struct.
    let stream = unsafe { &mut *stream };
    // SAFETY: set by `live_stream`, and unreleased.
    unsafe { count_release(stream.private_data) };
    stream.private_data = ptr::null_mut();
    stream.release = None;
}

fn live_schema() -> (FfiSchema, Arc<AtomicUsize>) {
    let (count, private) = counter();
    let schema = FfiSchema {
        format: c"u".as_ptr(),
        flags: 2,
        release: Some(release_schema),
        private_data: private,
        ..FfiSchema::released()
    };
    (schema, count)
}

fn live_array() -> (FfiArray, Arc<AtomicUsize>) {
    let (count, private) = counter();
    let array = FfiArray {
        length: 7,
        null_count: 1,
        offset: 3,
        release: Some(release_array),
        private_data: private,
        ..FfiArray::released()
    };
    (array, count)
}

fn live_stream() -> (FfiStream, Arc<AtomicUsize>) {
    let (count, private) = counter();
    let stream = FfiStream {
        release: Some(release_stream),
        private_data: private,
        ..FfiStream::released()
    };
    (stream, count)
}

#[test]
fn released_values_own_nothing() {
    let schema = FfiSchema::released();
    assert!(schema.is_released());
    assert!(schema.format.is_null() && schema.name.is_null() && schema.metadata.is_null());
    assert!(schema.children.is_null() && schema.dictionary.is_null());
    assert!(schema.private_data.is_null());
    assert_eq!((schema.flags, schema.n_children), (0, 0));

    let array = FfiArray::released();
    assert!(array.is_released());
    assert!(array.buffers.is_null() && array.children.is_null() && array.dictionary.is_null());
    assert!(array.private_data.is_null());
    assert_eq!(
        (
            array.length,
            array.null_count,
            array.offset,
            array.n_buffers,
            array.n_children
        ),
        (0, 0, 0, 0, 0)
    );

    let stream = FfiStream::released();
    assert!(stream.is_released());
    assert!(stream.get_schema.is_none() && stream.get_next.is_none());
    assert!(stream.get_last_error.is_none() && stream.private_data.is_null());

    // The trait's constructor is the same value.
    assert!(<FfiSchema as Released>::released().is_released());
    assert!(<FfiArray as Released>::released().is_released());
    assert!(<FfiStream as Released>::released().is_released());
    // Dropping released structs calls nothing (there is nothing to call).
}

#[test]
fn drop_releases_each_type_exactly_once() {
    let (schema, schema_count) = live_schema();
    let (array, array_count) = live_array();
    let (stream, stream_count) = live_stream();
    assert!(!schema.is_released() && !array.is_released() && !stream.is_released());
    drop(schema);
    drop(array);
    drop(stream);
    for count in [schema_count, array_count, stream_count] {
        assert_eq!(count.load(Ordering::SeqCst), 1);
        // Only the test's reference is left: the struct's was dropped.
        assert_eq!(Arc::strong_count(&count), 1);
    }
}

#[test]
fn move_out_transfers_ownership_of_an_array() {
    // As in a capsule: the producer's struct lives at a fixed heap address.
    let (array, count) = live_array();
    let mut slot = Box::new(array);
    // SAFETY: `slot` is a valid, initialized, unshared `FfiArray`.
    let moved = unsafe { move_out(&raw mut *slot) };
    assert!(slot.is_released());
    assert!(slot.private_data.is_null() && slot.buffers.is_null());
    assert_eq!((slot.length, slot.offset), (0, 0));
    assert!(!moved.is_released());
    assert_eq!((moved.length, moved.null_count, moved.offset), (7, 1, 3));
    drop(slot); // released: no callback
    assert_eq!(count.load(Ordering::SeqCst), 0);
    drop(moved);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn move_out_transfers_ownership_of_a_schema_and_a_stream() {
    let (schema, schema_count) = live_schema();
    let (stream, stream_count) = live_stream();
    let mut schema_slot = Box::new(schema);
    let mut stream_slot = Box::new(stream);
    // SAFETY: a valid, initialized, unshared `FfiSchema`.
    let schema = unsafe { move_out(&raw mut *schema_slot) };
    // SAFETY: a valid, initialized, unshared `FfiStream`.
    let stream = unsafe { move_out(&raw mut *stream_slot) };
    assert!(schema_slot.is_released() && schema_slot.format.is_null());
    assert!(stream_slot.is_released() && stream_slot.private_data.is_null());
    drop((schema_slot, stream_slot));
    assert_eq!(schema_count.load(Ordering::SeqCst), 0);
    assert_eq!(stream_count.load(Ordering::SeqCst), 0);
    assert_eq!(schema.flags, 2);
    drop((schema, stream));
    assert_eq!(schema_count.load(Ordering::SeqCst), 1);
    assert_eq!(stream_count.load(Ordering::SeqCst), 1);
}

#[test]
fn move_out_of_a_released_struct_yields_a_released_struct() {
    // What `take` sees when a capsule was already consumed.
    let mut array = FfiArray::released();
    // SAFETY: a valid, initialized, unshared `FfiArray`.
    let moved = unsafe { move_out(&raw mut array) };
    assert!(moved.is_released() && array.is_released());

    let mut schema = FfiSchema::released();
    // SAFETY: as above.
    assert!(unsafe { move_out(&raw mut schema) }.is_released());

    let mut stream = FfiStream::released();
    // SAFETY: as above.
    assert!(unsafe { move_out(&raw mut stream) }.is_released());
}

#[test]
fn a_second_move_out_finds_the_struct_released() {
    // Two consumers of one capsule: only the first gets the live struct.
    let (array, count) = live_array();
    let mut slot = Box::new(array);
    // SAFETY: a valid, initialized, unshared `FfiArray`.
    let first = unsafe { move_out(&raw mut *slot) };
    // SAFETY: as above (now released).
    let second = unsafe { move_out(&raw mut *slot) };
    assert!(!first.is_released());
    assert!(second.is_released());
    drop((second, slot));
    drop(first);
    assert_eq!(count.load(Ordering::SeqCst), 1);
}
