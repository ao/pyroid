//! Lists and tuples of known length, filled in place. Here because all of
//! pyroid-py's `unsafe` lives in this module.
//!
//! `PyList::new(py, vec)` makes a result in two passes: the items go into the
//! `Vec`, then into the list, which is only allocated then (and zeroed on
//! free-threaded builds). Nothing checks for Ctrl-C in the second pass, and
//! where page faults are slow it can be most of the call. A [`Fill`]
//! allocates the list or tuple first and puts each item in its slot, so the
//! caller's Ctrl-C checks cover the whole call, and the result needs one
//! array of item pointers instead of two.
//!
//! Until its last slot is set, the object is not tracked by the cycle GC. Its
//! empty slots are NULL, so Python code that a Ctrl-C check can run (a signal
//! handler, or a finalizer during the collection `PyErr_CheckSignals` may
//! start) must not find it in `gc.get_objects()`. A fill that stops early
//! (an error, a Ctrl-C) frees what it made: the list and tuple deallocators
//! skip NULL slots.

use std::marker::PhantomData;

use pyo3::exceptions::PyOverflowError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// `PyList` or `PyTuple`: the types a [`Fill`] makes.
pub trait Slots {
    /// A new object of `len` empty slots (a new reference, or NULL with an
    /// exception set).
    ///
    /// # Safety
    /// The calling thread must be attached.
    unsafe fn alloc(len: ffi::Py_ssize_t) -> *mut ffi::PyObject;

    /// The first of the slots of `seq` (what `PyList_SET_ITEM` and
    /// `PyTuple_SET_ITEM` write to). They stay where they are while nothing
    /// else holds `seq`: only a list resize could move them.
    ///
    /// # Safety
    /// `seq` was made by this type's `alloc`.
    #[cfg(not(Py_LIMITED_API))]
    unsafe fn slots(seq: *mut ffi::PyObject) -> *mut *mut ffi::PyObject;

    /// Move the reference `item` into slot `i` of `seq`. The limited API has
    /// no access to the slots: this is `PyList_SetItem` / `PyTuple_SetItem`
    /// (the same store behind a call, and for a list also its lock on
    /// free-threaded builds). They fail only for an index out of range, the
    /// wrong type, or a tuple that someone else holds.
    ///
    /// # Safety
    /// `seq` was made by this type's `alloc` with more than `i` slots, slot `i`
    /// is empty, nothing else holds `seq`, and the thread is attached.
    #[cfg(Py_LIMITED_API)]
    unsafe fn set(seq: *mut ffi::PyObject, i: ffi::Py_ssize_t, item: *mut ffi::PyObject);
}

impl Slots for PyList {
    unsafe fn alloc(len: ffi::Py_ssize_t) -> *mut ffi::PyObject {
        // SAFETY: attached, per the caller.
        unsafe { ffi::PyList_New(len) }
    }

    #[cfg(not(Py_LIMITED_API))]
    unsafe fn slots(seq: *mut ffi::PyObject) -> *mut *mut ffi::PyObject {
        // SAFETY: a list, per the caller.
        unsafe { (*seq.cast::<ffi::PyListObject>()).ob_item }
    }

    #[cfg(Py_LIMITED_API)]
    #[inline]
    unsafe fn set(seq: *mut ffi::PyObject, i: ffi::Py_ssize_t, item: *mut ffi::PyObject) {
        // SAFETY: a list with slot `i`, per the caller; the slot takes `item`.
        let done = unsafe { ffi::PyList_SetItem(seq, i, item) };
        debug_assert_eq!(done, 0);
    }
}

impl Slots for PyTuple {
    unsafe fn alloc(len: ffi::Py_ssize_t) -> *mut ffi::PyObject {
        // SAFETY: attached, per the caller.
        unsafe { ffi::PyTuple_New(len) }
    }

    #[cfg(not(Py_LIMITED_API))]
    unsafe fn slots(seq: *mut ffi::PyObject) -> *mut *mut ffi::PyObject {
        // SAFETY: a tuple, per the caller; its slots follow the header (a raw
        // borrow, so the pointer covers all of them).
        unsafe { (&raw mut (*seq.cast::<ffi::PyTupleObject>()).ob_item).cast() }
    }

    #[cfg(Py_LIMITED_API)]
    #[inline]
    unsafe fn set(seq: *mut ffi::PyObject, i: ffi::Py_ssize_t, item: *mut ffi::PyObject) {
        // SAFETY: a tuple only we hold with slot `i`, per the caller.
        let done = unsafe { ffi::PyTuple_SetItem(seq, i, item) };
        debug_assert_eq!(done, 0);
    }
}

/// A new list or tuple of `len` slots, set in order by [`Fill::push`].
/// Dropping it before [`Fill::finish`] frees the object and its items.
#[derive(Debug)]
pub struct Fill<'py, T> {
    seq: Bound<'py, PyAny>,
    /// `T::slots(seq)`.
    #[cfg(not(Py_LIMITED_API))]
    slots: *mut *mut ffi::PyObject,
    /// The next empty slot.
    next: usize,
    len: usize,
    /// `new` untracked it (`PyTuple_New(0)` gives the untracked empty tuple).
    untracked: bool,
    kind: PhantomData<T>,
}

impl<'py, T: Slots> Fill<'py, T> {
    pub fn new(py: Python<'py>, len: usize) -> PyResult<Self> {
        let n = ffi::Py_ssize_t::try_from(len)
            .map_err(|_| PyOverflowError::new_err("too many items for one sequence"))?;
        // SAFETY: attached (`py`).
        let ptr = unsafe { T::alloc(n) };
        // SAFETY: `alloc` gives a new reference, or NULL with an exception set.
        let seq = unsafe { Bound::from_owned_ptr_or_err(py, ptr) }?;
        // SAFETY: a live object.
        let tracked = unsafe { ffi::PyObject_GC_IsTracked(seq.as_ptr()) } == 1;
        if tracked {
            // SAFETY: a tracked object only we hold; only `finish` tracks it again.
            unsafe { ffi::PyObject_GC_UnTrack(seq.as_ptr().cast()) };
        }
        #[cfg(not(Py_LIMITED_API))]
        let slots = {
            // SAFETY: `T::alloc` made `seq`.
            unsafe { T::slots(seq.as_ptr()) }
        };
        Ok(Self {
            #[cfg(not(Py_LIMITED_API))]
            slots,
            seq,
            next: 0,
            len,
            untracked: tracked,
            kind: PhantomData,
        })
    }

    /// Put `item` in the next empty slot. Panics if every slot is set.
    #[inline]
    pub fn push(&mut self, item: Bound<'py, PyAny>) {
        assert!(self.next < self.len, "more items than slots");
        // Slot `next` of the `len` that `T::alloc` made is still empty (slots
        // are set in order), and only we hold `seq` (it is untracked and not
        // handed out before `finish`), so no resize has moved the slots.
        #[cfg(not(Py_LIMITED_API))]
        {
            // SAFETY: `next < len`: inside the slots.
            let slot = unsafe { self.slots.add(self.next) };
            // SAFETY: an empty slot, as above; it takes the reference.
            unsafe { slot.write(item.into_ptr()) };
        }
        #[cfg(Py_LIMITED_API)]
        {
            // `next < len`, and `len` fits a `Py_ssize_t`.
            #[allow(clippy::cast_possible_wrap)]
            let i = self.next as ffi::Py_ssize_t;
            // SAFETY: an empty slot of `seq`, as above.
            unsafe { T::set(self.seq.as_ptr(), i, item.into_ptr()) };
        }
        self.next += 1;
    }

    /// The list or tuple, once every slot is set. Panics if one is empty.
    pub fn finish(self) -> Bound<'py, T> {
        assert_eq!(self.next, self.len, "fewer items than slots");
        if self.untracked {
            // SAFETY: `new` untracked it, and no slot is NULL any more.
            unsafe { ffi::PyObject_GC_Track(self.seq.as_ptr().cast()) };
        }
        // SAFETY: `T::alloc` made an object of type `T`.
        unsafe { self.seq.cast_into_unchecked() }
    }
}
