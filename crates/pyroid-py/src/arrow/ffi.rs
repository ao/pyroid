//! The Arrow C Data Interface and C Stream Interface structs, exactly as the
//! spec defines them (`#[repr(C)]`, same field order and types):
//! <https://arrow.apache.org/docs/format/CDataInterface.html>,
//! <https://arrow.apache.org/docs/format/CStreamInterface.html>.
//!
//! Ownership: whoever holds a struct with a non-null `release` owns it and
//! must call `release` exactly once. Each type's `Drop` does that, so a value
//! of these types is always an *owned* struct: either one we produced, or one
//! we moved out of a producer's capsule (and then marked released there).

use std::ffi::{c_char, c_int, c_void};
use std::ptr;

/// `ARROW_FLAG_NULLABLE`: the field may contain nulls.
pub const FLAG_NULLABLE: i64 = 2;

/// `struct ArrowSchema`.
#[repr(C)]
#[derive(Debug)]
pub struct FfiSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut FfiSchema,
    pub dictionary: *mut FfiSchema,
    pub release: Option<unsafe extern "C" fn(*mut FfiSchema)>,
    pub private_data: *mut c_void,
}

/// `struct ArrowArray`.
#[repr(C)]
#[derive(Debug)]
pub struct FfiArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut FfiArray,
    pub dictionary: *mut FfiArray,
    pub release: Option<unsafe extern "C" fn(*mut FfiArray)>,
    pub private_data: *mut c_void,
}

/// `struct ArrowArrayStream`.
#[repr(C)]
#[derive(Debug)]
pub struct FfiStream {
    pub get_schema: Option<unsafe extern "C" fn(*mut FfiStream, *mut FfiSchema) -> c_int>,
    pub get_next: Option<unsafe extern "C" fn(*mut FfiStream, *mut FfiArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(*mut FfiStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(*mut FfiStream)>,
    pub private_data: *mut c_void,
}

impl FfiSchema {
    /// A released (empty) struct, for producers to write into.
    pub const fn released() -> Self {
        Self {
            format: ptr::null(),
            name: ptr::null(),
            metadata: ptr::null(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

impl FfiArray {
    /// A released (empty) struct, for producers to write into.
    pub const fn released() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

impl FfiStream {
    /// A released (empty) struct.
    pub const fn released() -> Self {
        Self {
            get_schema: None,
            get_next: None,
            get_last_error: None,
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

impl Drop for FfiSchema {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            // SAFETY: a non-null `release` means this value owns a live struct
            // (see the module docs); the spec requires calling it exactly once,
            // with the struct, and the callback marks it released.
            unsafe { release(self) };
        }
    }
}

impl Drop for FfiArray {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            // SAFETY: as for `FfiSchema::drop`.
            unsafe { release(self) };
        }
    }
}

impl Drop for FfiStream {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            // SAFETY: as for `FfiSchema::drop`.
            unsafe { release(self) };
        }
    }
}

/// Move the struct out of `src` and mark `src` released, per the spec's
/// "Moving an array" rules (bitwise copy, then null the source's `release`).
///
/// # Safety
/// `src` must be non-null, aligned, and point to a valid, initialized `T`
/// that nobody else accesses concurrently.
pub unsafe fn move_out<T: Released>(src: *mut T) -> T {
    // SAFETY: the caller guarantees `src` is valid for reads of a `T`.
    let moved = unsafe { ptr::read(src) };
    // SAFETY: the caller guarantees `src` is valid for writes. Overwriting
    // without dropping is the point: ownership now lives in `moved`.
    unsafe { ptr::write(src, T::released()) };
    moved
}

/// Types that have a "released" (empty, non-owning) value.
pub trait Released {
    fn released() -> Self;
    /// Is this struct released (`release == NULL`), i.e. owns nothing?
    fn is_released(&self) -> bool;
}

impl Released for FfiSchema {
    fn released() -> Self {
        Self::released()
    }
    fn is_released(&self) -> bool {
        self.release.is_none()
    }
}

impl Released for FfiArray {
    fn released() -> Self {
        Self::released()
    }
    fn is_released(&self) -> bool {
        self.release.is_none()
    }
}

impl Released for FfiStream {
    fn released() -> Self {
        Self::released()
    }
    fn is_released(&self) -> bool {
        self.release.is_none()
    }
}

#[cfg(test)]
mod tests;
