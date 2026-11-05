//! Importing Arrow string arrays through the Arrow `PyCapsule` interface
//! (<https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>).
//!
//! Supported: `utf8` (`u`), `large_utf8` (`U`) and `utf8_view` (`vu`), with
//! any offset and nulls, from `__arrow_c_stream__` (preferred) or
//! `__arrow_c_array__`. Everything else is rejected with an explanation.
//!
//! Soundness. The C Data Interface carries no buffer lengths, so a consumer
//! must trust the producer's guarantee that every buffer covers
//! `offset + length` rows (spec, "ArrowArray.buffers"); arrow-rs and pyarrow
//! extend the same trust. Everything else is checked once at import, before
//! any row is read: struct shape, null counts, non-null buffers, non-negative
//! and non-decreasing offsets, and in-bounds views (view buffers *do* carry
//! their sizes). UTF-8 is not trusted: each row is validated with
//! `simdutf8` (falling back to `str::from_utf8` for the error) when it is
//! read, inside the parallel kernel.

use std::ffi::{CStr, c_char, c_int};
use std::sync::{Mutex, PoisonError};
use std::{ptr, slice};

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyCapsule, PyTuple};

use super::ffi::{FfiArray, FfiSchema, FfiStream, Released, move_out};
use crate::errors::{input_error, type_name};

pub const SCHEMA_CAPSULE: &CStr = c"arrow_schema";
pub const ARRAY_CAPSULE: &CStr = c"arrow_array";
pub const STREAM_CAPSULE: &CStr = c"arrow_array_stream";

/// Length of one `utf8_view` view, and the longest string stored inline.
const VIEW_BYTES: usize = 16;
const VIEW_INLINE_MAX: usize = 12;

/// The Arrow string types pyroid reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrType {
    Utf8,
    LargeUtf8,
    Utf8View,
}

/// Type and field name of an imported column.
#[derive(Clone, Debug)]
pub struct Field {
    pub ty: StrType,
    pub name: String,
}

/// An imported Arrow string column: one or more validated chunks.
#[derive(Debug)]
pub struct ArrowStrings {
    pub field: Field,
    pub chunks: Vec<StrChunk>,
}

impl ArrowStrings {
    pub fn len(&self) -> usize {
        self.chunks.iter().map(StrChunk::len).sum()
    }
}

/// Read a C string owned by an unreleased Arrow struct.
///
/// # Safety
/// `p` must be null or a NUL-terminated string that outlives the returned
/// `String` construction (the spec guarantees this for `format`/`name` while
/// the schema is unreleased).
unsafe fn c_string(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: non-null and NUL-terminated per the caller's contract.
    let s = unsafe { CStr::from_ptr(p) };
    Some(s.to_string_lossy().into_owned())
}

/// A readable name for an Arrow format string, for error messages.
fn describe(format: &str, n_children: i64) -> String {
    let simple = match format {
        "n" => "null",
        "b" => "bool",
        "c" => "int8",
        "C" => "uint8",
        "s" => "int16",
        "S" => "uint16",
        "i" => "int32",
        "I" => "uint32",
        "l" => "int64",
        "L" => "uint64",
        "e" => "float16",
        "f" => "float32",
        "g" => "float64",
        "z" => "binary",
        "Z" => "large_binary",
        "vz" => "binary_view",
        "tdD" => "date32",
        "tdm" => "date64",
        "+l" => "list",
        "+L" => "large_list",
        "+vl" => "list_view",
        "+m" => "map",
        _ => "",
    };
    if !simple.is_empty() {
        return simple.to_owned();
    }
    if format == "+s" {
        return format!(
            "a struct with {n_children} field(s) (a table or record batch?); \
             pass one string column, e.g. table['text']"
        );
    }
    let family = match format.as_bytes().first() {
        Some(b'd') => "decimal",
        Some(b't') => "temporal",
        Some(b'w') => "fixed_size_binary",
        Some(b'+') => "nested",
        _ => "Arrow",
    };
    format!("{family} type (format {format:?})")
}

/// Check that `schema` describes a string column.
fn parse_schema(schema: &FfiSchema) -> Result<Field, String> {
    if schema.release.is_none() {
        return Err("the producer returned a released ArrowSchema".to_owned());
    }
    // SAFETY: the schema is unreleased, so `format` (mandatory) and `name`
    // (optional) are null or valid NUL-terminated strings (C Data spec).
    let format = unsafe { c_string(schema.format) }
        .ok_or_else(|| "the ArrowSchema has no format string".to_owned())?;
    // SAFETY: as above.
    let name = unsafe { c_string(schema.name) }.unwrap_or_default();
    if !schema.dictionary.is_null() {
        return Err(
            "got a dictionary-encoded column (e.g. pandas category or polars Categorical); \
             cast it to strings first, e.g. arr.cast(pa.string()) or s.cast(pl.String)"
                .to_owned(),
        );
    }
    let ty = match format.as_str() {
        "u" => StrType::Utf8,
        "U" => StrType::LargeUtf8,
        "vu" => StrType::Utf8View,
        other => {
            let what = describe(other, schema.n_children);
            let hint = if matches!(other, "z" | "Z" | "vz") {
                "; decode or cast the binary data to strings first"
            } else {
                ""
            };
            return Err(format!(
                "expected an Arrow string column (string, large_string or string_view), \
                 got {what}{hint}"
            ));
        }
    };
    Ok(Field { ty, name })
}

/// One imported Arrow string array, validated so that every row can be read.
#[derive(Debug)]
pub struct StrChunk {
    /// Owns the producer's buffers; dropping it calls the producer's release.
    _array: FfiArray,
    ty: StrType,
    len: usize,
    offset: usize,
    /// Validity bitmap, or null when there are no nulls.
    validity: *const u8,
    /// `utf8`/`large_utf8`: the offsets buffer.
    offsets: *const u8,
    /// `utf8`/`large_utf8`: the data buffer.
    data: *const u8,
    /// `utf8`/`large_utf8`: the last offset, i.e. the data bytes the producer
    /// guarantees (checked at import; reads are clamped to it).
    data_end: usize,
    /// `utf8_view`: the 16-byte views.
    views: *const u8,
    /// `utf8_view`: each variadic data buffer and its size in bytes.
    variadic: Variadic,
}

// SAFETY: every pointer in a `StrChunk` refers to producer memory that stays
// valid until `_array` is released, which only happens in `Drop`. The C Data
// Interface treats that memory as immutable while the struct is live; we rely
// on it, as every consumer does (arrow-rs, pyarrow). Concurrent reads of
// immutable memory are sound, so `&StrChunk` may be shared with worker
// threads. If a producer breaks the contract (e.g. an array over a `bytearray`
// that another thread writes during a batch), `raw` re-clamps every offset and
// view to the bounds checked at import, so reads stay in bounds; only the
// already-validated UTF-8 could be invalidated, which no consumer can prevent
// without copying. `StrChunk` is deliberately *not* `Send`: the producer's
// release callback may need the interpreter (e.g. pyarrow buffers backed by
// Python objects), so it must run on the importing, attached thread.
unsafe impl Sync for StrChunk {}

/// Read `N` bytes at `p + at` (no alignment needed: `[u8; N]` has alignment 1,
/// so offsets and sizes are read with `from_ne_bytes` instead of casting to
/// `*const i32`/`*const i64`, which the spec only *recommends* be aligned).
///
/// # Safety
/// `p + at .. p + at + N` must be readable.
unsafe fn read_bytes<const N: usize>(p: *const u8, at: usize) -> [u8; N] {
    // SAFETY: readable per the caller's contract; alignment 1.
    unsafe { p.wrapping_add(at).cast::<[u8; N]>().read() }
}

/// Read the `k`-th entry of a C array of buffer pointers.
///
/// # Safety
/// `buffers` must point to at least `k + 1` readable pointers.
unsafe fn buffer_at(buffers: *mut *const std::ffi::c_void, k: usize) -> *const u8 {
    // SAFETY: in bounds per the caller's contract; the C array is pointer-aligned.
    unsafe { buffers.wrapping_add(k).read() }.cast::<u8>()
}

/// Entry `k` of an `i32` (`utf8`) or `i64` (`large_utf8`) offsets buffer.
///
/// # Safety
/// `offsets` must be readable for `k + 1` entries of the type's width.
unsafe fn read_offset(ty: StrType, offsets: *const u8, k: usize) -> i64 {
    if ty == StrType::Utf8 {
        // SAFETY: entry k of 4-byte offsets is readable per the contract.
        i64::from(i32::from_ne_bytes(unsafe { read_bytes(offsets, k * 4) }))
    } else {
        // SAFETY: entry k of 8-byte offsets is readable per the contract.
        i64::from_ne_bytes(unsafe { read_bytes(offsets, k * 8) })
    }
}

fn view_i32(view: &[u8; VIEW_BYTES], at: usize) -> i32 {
    i32::from_ne_bytes([view[at], view[at + 1], view[at + 2], view[at + 3]])
}

/// Bit `i` of a validity bitmap.
///
/// # Safety
/// `validity` must be non-null and readable for at least `i + 1` bits.
unsafe fn bit(validity: *const u8, i: usize) -> bool {
    // SAFETY: byte i / 8 is readable per the caller's contract.
    let [byte] = unsafe { read_bytes::<1>(validity, i / 8) };
    byte & (1 << (i % 8)) != 0
}

/// Check `utf8`/`large_utf8` offsets `offset..=offset + len`: non-negative,
/// non-decreasing, addressable; and a non-null data buffer if any row is
/// non-empty.
///
/// # Safety
/// `offsets` must be null or readable for `offset + len + 1` entries (the
/// producer's guarantee for a live array).
unsafe fn check_offsets(
    ty: StrType,
    offsets: *const u8,
    data: *const u8,
    offset: usize,
    len: usize,
) -> Result<usize, String> {
    if offsets.is_null() {
        return Err("the offsets buffer is null".to_owned());
    }
    // SAFETY: entry `offset` exists per the caller's contract.
    let first = unsafe { read_offset(ty, offsets, offset) };
    if first < 0 {
        return Err(format!("negative string offset {first}"));
    }
    let mut prev = first;
    for row in 0..len {
        // SAFETY: offset + row + 1 <= offset + len, so the entry exists.
        let next = unsafe { read_offset(ty, offsets, offset + row + 1) };
        if next < prev {
            return Err(format!(
                "string offsets decrease at row {row} ({prev} > {next})"
            ));
        }
        prev = next;
    }
    let end = usize::try_from(prev)
        .ok()
        .filter(|&e| isize::try_from(e).is_ok())
        .ok_or("string offsets exceed the address space")?;
    if prev > first && data.is_null() {
        return Err("the data buffer is null".to_owned());
    }
    Ok(end)
}

/// Each variadic data buffer of a view array, with its size in bytes.
type Variadic = Vec<(*const u8, usize)>;

/// Read the variadic buffers of a `utf8_view` array and check that every
/// non-null long view points inside its buffer.
///
/// # Safety
/// `array` must be a live view array with a non-null `buffers` C array of
/// `n_buffers >= 3` entries, whose views buffer (if non-null) holds
/// `offset + len` views, and `is_valid(i)` must be safe for `i < len`.
unsafe fn check_views(
    array: &FfiArray,
    n_buffers: usize,
    offset: usize,
    len: usize,
    is_valid: impl Fn(usize) -> bool,
) -> Result<(*const u8, Variadic), String> {
    // SAFETY: index 1 < n_buffers, per the caller's contract. (So are the
    // indexes 2..n_buffers - 1 and n_buffers - 1 used below.)
    let views = unsafe { buffer_at(array.buffers, 1) };
    if views.is_null() {
        return Err("the views buffer is null".to_owned());
    }
    let n_variadic = n_buffers - 3;
    // SAFETY: see above.
    let sizes = unsafe { buffer_at(array.buffers, n_buffers - 1) };
    if n_variadic > 0 && sizes.is_null() {
        return Err("the variadic buffer sizes buffer is null".to_owned());
    }
    let mut variadic = Vec::with_capacity(n_variadic);
    for b in 0..n_variadic {
        // SAFETY: the sizes buffer holds one int64 per variadic buffer (spec,
        // "Binary view arrays"), and b < n_variadic.
        let size = i64::from_ne_bytes(unsafe { read_bytes(sizes, b * 8) });
        let size =
            usize::try_from(size).map_err(|_| format!("negative variadic buffer size {size}"))?;
        // SAFETY: see above.
        let p = unsafe { buffer_at(array.buffers, 2 + b) };
        if size > 0 && p.is_null() {
            return Err(format!("variadic data buffer {b} is null"));
        }
        variadic.push((p, size));
    }
    for i in (0..len).filter(|&i| is_valid(i)) {
        // SAFETY: the views buffer holds offset + len views.
        let view: [u8; VIEW_BYTES] = unsafe { read_bytes(views, (offset + i) * VIEW_BYTES) };
        let n = usize::try_from(view_i32(&view, 0))
            .map_err(|_| format!("negative string length at row {i}"))?;
        if n <= VIEW_INLINE_MAX {
            continue;
        }
        let in_bounds = usize::try_from(view_i32(&view, 8))
            .ok()
            .zip(usize::try_from(view_i32(&view, 12)).ok())
            .and_then(|(b, at)| Some((variadic.get(b)?.1, at)))
            .is_some_and(|(size, at)| at.checked_add(n).is_some_and(|e| e <= size));
        if !in_bounds {
            return Err(format!("string view at row {i} points outside its buffer"));
        }
    }
    Ok((views, variadic))
}

impl StrChunk {
    /// Validate `array` (of type `ty`) and take ownership of it.
    pub fn new(array: FfiArray, ty: StrType) -> Result<Self, String> {
        if array.release.is_none() {
            return Err("the producer returned a released ArrowArray".to_owned());
        }
        let len = usize::try_from(array.length)
            .map_err(|_| format!("negative ArrowArray.length {}", array.length))?;
        let offset = usize::try_from(array.offset)
            .map_err(|_| format!("negative ArrowArray.offset {}", array.offset))?;
        offset
            .checked_add(len)
            .and_then(|e| e.checked_add(1))
            .and_then(|e| e.checked_mul(VIEW_BYTES))
            .ok_or("ArrowArray offset + length overflows")?;
        if array.n_children != 0 || !array.dictionary.is_null() {
            return Err("a string ArrowArray must have no children and no dictionary".to_owned());
        }
        let n_buffers = usize::try_from(array.n_buffers).unwrap_or(0);
        let ok_buffers = match ty {
            StrType::Utf8 | StrType::LargeUtf8 => n_buffers == 3,
            StrType::Utf8View => n_buffers >= 3,
        };
        if !ok_buffers || array.buffers.is_null() {
            return Err(format!(
                "unexpected buffer layout for a string array ({} buffers)",
                array.n_buffers
            ));
        }
        // SAFETY: `buffers` is non-null with n_buffers >= 3 entries (checked).
        let mut validity = unsafe { buffer_at(array.buffers, 0) };
        match array.null_count {
            0 => validity = ptr::null(), // no nulls: any bitmap is irrelevant
            -1 => {}                     // unknown: use the bitmap if there is one
            n if n > 0 && !validity.is_null() => {}
            n if n > 0 => return Err(format!("null_count is {n} but there is no validity bitmap")),
            n => return Err(format!("invalid ArrowArray.null_count {n}")),
        }
        // SAFETY: a non-null bitmap covers offset + len bits, and i < len.
        let is_valid = |i: usize| validity.is_null() || unsafe { bit(validity, offset + i) };

        let (mut offsets, mut data, mut views) = (ptr::null(), ptr::null(), ptr::null());
        let mut data_end = 0;
        let mut variadic = Vec::new();
        if len > 0 {
            if ty == StrType::Utf8View {
                // SAFETY: a live view array with >= 3 buffers (checked above);
                // the producer guarantees the views buffer covers the rows.
                (views, variadic) =
                    unsafe { check_views(&array, n_buffers, offset, len, is_valid)? };
            } else {
                // SAFETY: n_buffers == 3 (checked above).
                offsets = unsafe { buffer_at(array.buffers, 1) };
                // SAFETY: as above.
                data = unsafe { buffer_at(array.buffers, 2) };
                // SAFETY: the producer guarantees a live array's offsets
                // buffer holds offset + len + 1 entries.
                data_end = unsafe { check_offsets(ty, offsets, data, offset, len)? };
            }
        }
        Ok(Self {
            _array: array,
            ty,
            len,
            offset,
            validity,
            offsets,
            data,
            data_end,
            views,
            variadic,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    fn is_valid(&self, i: usize) -> bool {
        assert!(i < self.len, "row {i} out of range for {} rows", self.len);
        // SAFETY: a non-null bitmap covers offset + len bits, and i < len.
        self.validity.is_null() || unsafe { bit(self.validity, self.offset + i) }
    }

    /// The raw bytes of row `i` (`i < len`). Callers must have checked
    /// `is_valid(i)`: views of null `utf8_view` rows are not validated.
    fn raw(&self, i: usize) -> &[u8] {
        assert!(i < self.len, "row {i} out of range for {} rows", self.len);
        let k = self.offset + i;
        if self.ty == StrType::Utf8View {
            // SAFETY: k < offset + len, so view k exists.
            let view: [u8; VIEW_BYTES] = unsafe { read_bytes(self.views, k * VIEW_BYTES) };
            // Each field is re-checked (not trusted from import): see the
            // `Sync` impl for why a misbehaving producer can't push us out of
            // bounds.
            let Ok(n) = usize::try_from(view_i32(&view, 0)) else {
                return &[];
            };
            if n <= VIEW_INLINE_MAX {
                // SAFETY: an inline string is bytes 4..4 + n of view k, inside
                // the views buffer, which lives as long as `self`.
                return unsafe {
                    slice::from_raw_parts(self.views.wrapping_add(k * VIEW_BYTES + 4), n)
                };
            }
            let (Ok(b), Ok(at)) = (
                usize::try_from(view_i32(&view, 8)),
                usize::try_from(view_i32(&view, 12)),
            ) else {
                return &[];
            };
            let Some(&(base, size)) = self.variadic.get(b) else {
                return &[];
            };
            if at.checked_add(n).is_none_or(|e| e > size) {
                return &[];
            }
            // SAFETY: at + n <= size of variadic buffer b (just checked), and
            // the producer guarantees that buffer's size.
            return unsafe { slice::from_raw_parts(base.wrapping_add(at), n) };
        }
        // SAFETY: k + 1 <= offset + len, so both entries exist.
        let start = unsafe { read_offset(self.ty, self.offsets, k) };
        // SAFETY: as above.
        let end = unsafe { read_offset(self.ty, self.offsets, k + 1) };
        // Validated at import: 0 <= start <= end <= data_end. Clamp anyway, so
        // a producer that mutates its buffers can't push us out of bounds.
        let end = usize::try_from(end).unwrap_or(0).min(self.data_end);
        let start = usize::try_from(start).unwrap_or(0).min(end);
        if start == end {
            return &[];
        }
        // SAFETY: the producer guarantees the data buffer covers `data_end`
        // bytes, 0 <= start < end <= data_end (clamped above), and `data` is
        // non-null when data_end > 0 (validated at import).
        unsafe { slice::from_raw_parts(self.data.wrapping_add(start), end - start) }
    }

    /// Row `i`: `None` if null, else its text (or why it isn't valid UTF-8).
    pub fn get(&self, i: usize) -> Option<Result<&str, std::str::Utf8Error>> {
        self.is_valid(i).then(|| {
            let raw = self.raw(i);
            // Fast SIMD check; on failure re-run std's (also a validator) for
            // the error position. Either way only validated bytes become &str.
            simdutf8::basic::from_utf8(raw).or_else(|_| std::str::from_utf8(raw))
        })
    }

    /// The raw bytes of row `i`, or `None` if it is null (for error messages).
    pub fn bytes(&self, i: usize) -> Option<&[u8]> {
        // `raw` is only sound on valid rows for `utf8_view` (null views are
        // not validated), so every caller goes through `is_valid` first.
        self.is_valid(i).then(|| self.raw(i))
    }
}

/// Serialises pyroid's moves out of capsules. A producer may return the same
/// cached capsule from every `__arrow_c_*__` call (a one-shot reader is a
/// natural case). Without the lock, two free-threaded callers could both read
/// the struct before either marks it released, and both would then release
/// it: a double free in the producer.
static TAKE_LOCK: Mutex<()> = Mutex::new(());

/// Move the struct out of a capsule named `name`, leaving it released. Fails
/// if the capsule was already consumed (by an earlier call or another thread).
fn take<T: Released>(
    producer: &Bound<'_, PyAny>,
    cap: &Bound<'_, PyAny>,
    name: &CStr,
) -> PyResult<T> {
    let py = cap.py();
    let bad = || {
        input_error(
            py,
            format!(
                "expected a PyCapsule named {name:?}, got {}",
                type_name(cap)
            ),
        )
    };
    let cap = cap.cast::<PyCapsule>().map_err(|_| bad())?;
    let p = cap.pointer_checked(Some(name)).map_err(|_| bad())?;
    let moved = {
        // `lock_py_attached` waits detached, so a waiter can't block a
        // stop-the-world pause; the section itself makes no Python calls.
        let _guard = TAKE_LOCK
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner);
        // SAFETY: a capsule with this name holds a pointer to a valid `T`
        // (PyCapsule interface spec), alive while we hold `cap`. The producer
        // may have handed the same capsule to other callers, so the
        // read-then-mark-released runs under TAKE_LOCK: of all pyroid callers
        // only the first moves out a live struct; later ones move out a
        // released one and fail below. (On GIL builds the GIL also serialises
        // us against other consumers; on free-threaded builds a *different*
        // library consuming the same capsule at the same moment is outside
        // what any consumer can guard against.)
        unsafe { move_out(p.as_ptr().cast::<T>()) }
    };
    if moved.is_released() {
        return Err(invalid(
            producer,
            &format!(
                "the {} capsule was already consumed (its struct is released); \
                 __arrow_c_array__/__arrow_c_stream__ must return a fresh capsule per call",
                name.to_string_lossy()
            ),
        ));
    }
    Ok(moved)
}

fn invalid(producer: &Bound<'_, PyAny>, msg: &str) -> PyErr {
    PyValueError::new_err(format!(
        "invalid Arrow data from {}: {msg}",
        type_name(producer)
    ))
}

fn import_array(obj: &Bound<'_, PyAny>) -> PyResult<ArrowStrings> {
    let py = obj.py();
    let pair = obj.call_method0(intern!(py, "__arrow_c_array__"))?;
    let pair = pair
        .cast::<PyTuple>()
        .ok()
        .filter(|t| t.len() == 2)
        .ok_or_else(|| {
            input_error(
                py,
                "__arrow_c_array__() must return a (schema, array) tuple of PyCapsules",
            )
        })?;
    let schema: FfiSchema = take(obj, &pair.get_item(0)?, SCHEMA_CAPSULE)?;
    let array: FfiArray = take(obj, &pair.get_item(1)?, ARRAY_CAPSULE)?;
    let field = parse_schema(&schema).map_err(|m| input_error(py, m))?;
    drop(schema);
    let chunk = StrChunk::new(array, field.ty).map_err(|m| invalid(obj, &m))?;
    Ok(ArrowStrings {
        field,
        chunks: vec![chunk],
    })
}

/// The message for a failed stream callback.
fn stream_error(obj: &Bound<'_, PyAny>, stream: &mut FfiStream, code: c_int) -> PyErr {
    let detail = stream.get_last_error.and_then(|f| {
        // SAFETY: `stream` is live; get_last_error returns null or a string
        // valid until the next call on the stream, and we copy it at once.
        let p = unsafe { f(stream) };
        // SAFETY: as above.
        unsafe { c_string(p) }
    });
    let detail = detail.unwrap_or_else(|| format!("error code {code}"));
    invalid(obj, &format!("the Arrow stream failed: {detail}"))
}

fn import_stream(obj: &Bound<'_, PyAny>) -> PyResult<ArrowStrings> {
    let py = obj.py();
    let cap = match obj.call_method0(intern!(py, "__arrow_c_stream__")) {
        Ok(cap) => cap,
        // Some producers export only certain types as streams; fall back to
        // the array protocol, and report the stream error if that fails too.
        Err(err) if obj.hasattr(intern!(py, "__arrow_c_array__"))? => {
            return import_array(obj).map_err(|_| err);
        }
        Err(err) => return Err(err),
    };
    let mut stream: FfiStream = take(obj, &cap, STREAM_CAPSULE)?;
    let (Some(_), Some(get_schema), Some(get_next)) =
        (stream.release, stream.get_schema, stream.get_next)
    else {
        return Err(invalid(
            obj,
            "the ArrowArrayStream is released or incomplete",
        ));
    };
    // Stream callbacks run attached: a producer may call back into Python
    // (e.g. a reader over a Python iterator).
    let mut schema = FfiSchema::released();
    // SAFETY: `stream` is a live, owned stream; `schema` is a writable,
    // released struct for the producer to fill in.
    let code = unsafe { get_schema(&raw mut stream, &raw mut schema) };
    if code != 0 {
        // On error the out-struct is not valid (spec): never release it.
        std::mem::forget(schema);
        return Err(stream_error(obj, &mut stream, code));
    }
    let field = parse_schema(&schema).map_err(|m| input_error(py, m))?;
    drop(schema);
    let mut chunks = Vec::new();
    loop {
        let mut array = FfiArray::released();
        // SAFETY: as for get_schema; at the end of the stream the producer
        // writes a released array.
        let code = unsafe { get_next(&raw mut stream, &raw mut array) };
        if code != 0 {
            std::mem::forget(array); // not valid on error (spec): never release it
            return Err(stream_error(obj, &mut stream, code));
        }
        if array.release.is_none() {
            break;
        }
        // Empty chunks are kept, so chunk numbers in error messages match the
        // producer's; `Column` drops empty output chunks.
        chunks.push(StrChunk::new(array, field.ty).map_err(|m| invalid(obj, &m))?);
    }
    Ok(ArrowStrings { field, chunks })
}

/// Import `obj` as Arrow strings if it implements the Arrow PyCapsule
/// interface; `Ok(None)` if it implements neither protocol.
pub fn import(obj: &Bound<'_, PyAny>) -> PyResult<Option<ArrowStrings>> {
    let py = obj.py();
    if obj.hasattr(intern!(py, "__arrow_c_stream__"))? {
        return import_stream(obj).map(Some);
    }
    if obj.hasattr(intern!(py, "__arrow_c_array__"))? {
        return import_array(obj).map(Some);
    }
    Ok(None)
}
