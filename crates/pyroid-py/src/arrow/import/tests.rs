//! The importer's pure parts on hand-built C Data structs: `StrChunk::new`
//! (every layout it accepts and every rejection), `get`/`bytes` on every row,
//! and `parse_schema`. A test producer owns the buffers and counts its release
//! calls, so each test also checks that the importer releases exactly once, on
//! success and on every error path. No Python, so these run under Miri and the
//! sanitizers.

use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use pyroid_core::columnar::bitmap;

use super::{StrChunk, StrType, VIEW_INLINE_MAX, parse_schema};
use crate::arrow::ffi::{FfiArray, FfiSchema};

// ---- a test producer --------------------------------------------------------

/// A producer-owned buffer that starts `skew` bytes into an 8-aligned
/// allocation, so `skew` 1 or 3 makes every 4- and 8-byte entry unaligned.
struct Buf {
    /// Owns the memory; never touched again after `ptr` is taken.
    _words: Vec<u64>,
    /// `skew` bytes into `_words`, with write permission (for the tests that
    /// play a producer which mutates its buffers).
    ptr: *mut u8,
}

impl Buf {
    fn new(bytes: &[u8], skew: usize) -> Self {
        let mut padded = vec![0u8; skew];
        padded.extend_from_slice(bytes);
        padded.resize(padded.len().div_ceil(8).max(1) * 8, 0);
        let mut words: Vec<u64> = padded
            .chunks_exact(8)
            .map(|w| u64::from_ne_bytes(w.try_into().unwrap()))
            .collect();
        // `as_mut_ptr` doesn't materialise a reference, and moving the Vec
        // below doesn't move its heap buffer, so `ptr` stays valid.
        let ptr = words.as_mut_ptr().cast::<u8>().wrapping_add(skew);
        Self { _words: words, ptr }
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "reads as a present buffer at call sites"
    )]
    fn aligned(bytes: &[u8]) -> Option<Self> {
        Some(Self::new(bytes, 0))
    }
}

/// What the test producer's release callback frees.
struct Private {
    _bufs: Vec<Option<Buf>>,
    /// The `buffers` C array.
    ptrs: Vec<*const c_void>,
    released: Arc<AtomicUsize>,
}

unsafe extern "C" fn release_array(array: *mut FfiArray) {
    // SAFETY: `FfiArray::drop` passes the live struct.
    let array = unsafe { &mut *array };
    // SAFETY: `private_data` is the `Box<Private>` from `array`; it is freed
    // only here, and `release` is cleared below, so never twice.
    let private = unsafe { Box::from_raw(array.private_data.cast::<Private>()) };
    private.released.fetch_add(1, Ordering::SeqCst);
    array.private_data = ptr::null_mut();
    array.buffers = ptr::null_mut();
    array.release = None;
}

/// A live `ArrowArray` over `bufs` (`None` = a null buffer pointer), and its
/// release count. Tests adjust the public fields to build malformed arrays.
fn array(len: i64, null_count: i64, bufs: Vec<Option<Buf>>) -> (FfiArray, Arc<AtomicUsize>) {
    let released = Arc::new(AtomicUsize::new(0));
    let ptrs: Vec<*const c_void> = bufs
        .iter()
        .map(|b| {
            b.as_ref()
                .map_or(ptr::null(), |b| b.ptr.cast_const().cast())
        })
        .collect();
    let n_buffers = i64::try_from(ptrs.len()).unwrap();
    let private = Box::new(Private {
        _bufs: bufs,
        ptrs,
        released: Arc::clone(&released),
    });
    // Points into `ptrs`' heap buffer, which doesn't move with the Box.
    let buffers = private.ptrs.as_ptr().cast_mut();
    let array = FfiArray {
        length: len,
        null_count,
        offset: 0,
        n_buffers,
        n_children: 0,
        buffers,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_array),
        private_data: Box::into_raw(private).cast::<c_void>(),
    };
    (array, released)
}

fn count(released: &AtomicUsize) -> usize {
    released.load(Ordering::SeqCst)
}

/// Rows as owned bytes (`None` = null).
type Rows = Vec<Option<Vec<u8>>>;

fn rows(values: &[Option<&str>]) -> Rows {
    values
        .iter()
        .map(|v| v.map(|s| s.as_bytes().to_vec()))
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn i64_bytes(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn offsets_bytes(ty: StrType, offsets: &[i64]) -> Vec<u8> {
    if ty == StrType::Utf8 {
        let narrow: Vec<i32> = offsets.iter().map(|&o| i32::try_from(o).unwrap()).collect();
        i32_bytes(&narrow)
    } else {
        i64_bytes(offsets)
    }
}

/// The validity bitmap of `rows`, or `None` if no row is null.
fn validity(rows: &Rows) -> Option<Vec<u8>> {
    rows.iter()
        .any(Option::is_none)
        .then(|| bitmap(rows.iter().map(Option::is_some), rows.len()))
}

fn null_count(rows: &Rows) -> i64 {
    i64::try_from(rows.iter().filter(|r| r.is_none()).count()).unwrap()
}

fn len(rows: &Rows) -> i64 {
    i64::try_from(rows.len()).unwrap()
}

/// A `utf8` or `large_utf8` array of `rows`, every buffer `skew` bytes off
/// alignment; the true null count.
fn string_array(ty: StrType, rows: &Rows, skew: usize) -> (FfiArray, Arc<AtomicUsize>) {
    let mut offsets = vec![0i64];
    let mut data = Vec::new();
    for row in rows {
        data.extend_from_slice(row.as_deref().unwrap_or_default());
        offsets.push(i64::try_from(data.len()).unwrap());
    }
    let bufs = vec![
        validity(rows).map(|v| Buf::new(&v, skew)),
        Some(Buf::new(&offsets_bytes(ty, &offsets), skew)),
        Some(Buf::new(&data, skew)),
    ];
    array(len(rows), null_count(rows), bufs)
}

/// A 16-byte view: `len`, then the inline bytes or (prefix, buffer, offset).
fn inline_view(bytes: &[u8]) -> [u8; 16] {
    assert!(bytes.len() <= VIEW_INLINE_MAX);
    let mut view = [0u8; 16];
    view[..4].copy_from_slice(&i32::try_from(bytes.len()).unwrap().to_ne_bytes());
    view[4..4 + bytes.len()].copy_from_slice(bytes);
    view
}

fn long_view(len: i32, prefix: &[u8], buffer: i32, offset: i32) -> [u8; 16] {
    let mut view = [0u8; 16];
    view[..4].copy_from_slice(&len.to_ne_bytes());
    let n = prefix.len().min(4);
    view[4..4 + n].copy_from_slice(&prefix[..n]);
    view[8..12].copy_from_slice(&buffer.to_ne_bytes());
    view[12..].copy_from_slice(&offset.to_ne_bytes());
    view
}

/// Garbage that fails every view check: null rows' views are undefined
/// (spec), so the importer must not validate them.
const GARBAGE_VIEW: [u8; 16] = [0xff; 16];

/// A `utf8_view` array of `rows`: short rows inline, long rows spread
/// round-robin over `n_data` variadic buffers (each with a 3-byte lead-in, so
/// view offsets aren't 0), null rows as `GARBAGE_VIEW`, every buffer `skew`
/// bytes off alignment.
fn view_array(rows: &Rows, n_data: usize, skew: usize) -> (FfiArray, Arc<AtomicUsize>) {
    let mut data = vec![b"pad".to_vec(); n_data];
    let mut views = Vec::new();
    let mut next = 0;
    for row in rows {
        let view = match row {
            None => GARBAGE_VIEW,
            Some(bytes) if bytes.len() <= VIEW_INLINE_MAX => inline_view(bytes),
            Some(bytes) => {
                let b = next % n_data;
                next += 1;
                let at = i32::try_from(data[b].len()).unwrap();
                data[b].extend_from_slice(bytes);
                long_view(
                    i32::try_from(bytes.len()).unwrap(),
                    bytes,
                    i32::try_from(b).unwrap(),
                    at,
                )
            }
        };
        views.extend_from_slice(&view);
    }
    let sizes: Vec<i64> = data
        .iter()
        .map(|d| i64::try_from(d.len()).unwrap())
        .collect();
    let mut bufs = vec![
        validity(rows).map(|v| Buf::new(&v, skew)),
        Some(Buf::new(&views, skew)),
    ];
    bufs.extend(data.iter().map(|d| Some(Buf::new(d, skew))));
    bufs.push(Some(Buf::new(&i64_bytes(&sizes), skew)));
    array(len(rows), null_count(rows), bufs)
}

/// Every row of `chunk` reads back as `expected`, through `get` and `bytes`.
fn assert_rows(chunk: &StrChunk, expected: &[Option<Vec<u8>>]) {
    assert_eq!(chunk.len(), expected.len());
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(chunk.bytes(i), row.as_deref(), "bytes({i})");
        assert_eq!(
            chunk.get(i),
            row.as_deref().map(std::str::from_utf8),
            "get({i})"
        );
    }
}

/// Import `array`, check every row, drop the chunk, and check it released the
/// array exactly once (and not before the drop).
fn assert_imports(
    array: FfiArray,
    ty: StrType,
    released: &AtomicUsize,
    expected: &[Option<Vec<u8>>],
) {
    let chunk = StrChunk::new(array, ty).unwrap();
    assert_rows(&chunk, expected);
    assert_eq!(count(released), 0);
    drop(chunk);
    assert_eq!(count(released), 1);
}

/// `StrChunk::new` fails with a message containing `needle`, having released
/// the array exactly once.
fn assert_rejects(array: FfiArray, ty: StrType, released: &AtomicUsize, needle: &str) {
    let err = StrChunk::new(array, ty).unwrap_err();
    assert!(err.contains(needle), "{err:?} does not contain {needle:?}");
    assert_eq!(count(released), 1, "released once after: {err}");
}

/// Mixed rows: empty, ASCII, multi-byte, exactly inline-sized, just over,
/// long (> 64 bytes, so simdutf8 takes its SIMD path) and nulls.
fn sample() -> Rows {
    let long = "lorem ipsum dolor sit amet, ".repeat(3) + "ünïcödé ✓ 𝄞";
    rows(&[
        Some(""),
        Some("a"),
        None,
        Some("héllo wörld"),
        Some("twelve bytes"),
        Some("thirteen byte"),
        None,
        Some(&long),
        Some("日本語のテキスト"),
        Some(""),
    ])
}

const STRING_TYPES: [StrType; 2] = [StrType::Utf8, StrType::LargeUtf8];

// ---- utf8 / large_utf8 ------------------------------------------------------

#[test]
fn string_arrays_round_trip() {
    for ty in STRING_TYPES {
        let rows = sample();
        let (a, released) = string_array(ty, &rows, 0);
        assert_imports(a, ty, &released, &rows);
    }
}

#[test]
fn string_arrays_without_nulls() {
    for ty in STRING_TYPES {
        let rows = rows(&[Some("x"), Some(""), Some("yz")]);
        let (a, released) = string_array(ty, &rows, 0);
        assert_eq!(a.null_count, 0);
        // SAFETY: `buffers` holds 3 entries (`string_array`).
        assert!(unsafe { a.buffers.read() }.is_null());
        assert_imports(a, ty, &released, &rows);
    }
}

#[test]
fn unaligned_buffers_are_read_bytewise() {
    // The spec only recommends aligned buffers; offsets are read with
    // `from_ne_bytes`, never through `*const i32`/`*const i64`.
    for skew in [1, 3, 5] {
        for ty in STRING_TYPES {
            let rows = sample();
            let (a, released) = string_array(ty, &rows, skew);
            assert_imports(a, ty, &released, &rows);
        }
    }
}

#[test]
fn null_count_minus_one_uses_the_bitmap_if_any() {
    for ty in STRING_TYPES {
        let rows = sample();
        let (mut a, released) = string_array(ty, &rows, 0);
        a.null_count = -1;
        assert_imports(a, ty, &released, &rows);

        // Unknown count and no bitmap: every row is valid.
        let rows = rows_without_nulls();
        let (mut a, released) = string_array(ty, &rows, 0);
        a.null_count = -1;
        assert_imports(a, ty, &released, &rows);
    }
}

fn rows_without_nulls() -> Rows {
    sample().into_iter().flatten().map(Some).collect()
}

#[test]
fn null_count_zero_ignores_a_bitmap() {
    // A producer may ship an all-set (or stale) bitmap with null_count 0: the
    // count wins, so "null" rows read as their (empty) bytes.
    for ty in STRING_TYPES {
        let rows = rows(&[Some("a"), None, Some("b")]);
        let (mut a, released) = string_array(ty, &rows, 0);
        a.null_count = 0;
        let expected = self::rows(&[Some("a"), Some(""), Some("b")]);
        assert_imports(a, ty, &released, &expected);
    }
}

#[test]
fn offset_slices_read_the_right_rows_and_bits() {
    for ty in STRING_TYPES {
        for skew in [0, 1] {
            let rows = sample();
            for (offset, length) in [(0, 10), (1, 9), (2, 3), (3, 7), (9, 1), (10, 0), (5, 0)] {
                let (mut a, released) = string_array(ty, &rows, skew);
                a.offset = offset;
                a.length = length;
                a.null_count = -1;
                let lo = usize::try_from(offset).unwrap();
                let hi = lo + usize::try_from(length).unwrap();
                assert_imports(a, ty, &released, &rows[lo..hi]);
            }
        }
    }
}

#[test]
fn offset_slices_with_their_own_null_count() {
    // Rows 3..8 of `sample` hold one null (row 6).
    for ty in STRING_TYPES {
        let rows = sample();
        let (mut a, released) = string_array(ty, &rows, 0);
        (a.offset, a.length, a.null_count) = (3, 5, 1);
        assert_imports(a, ty, &released, &rows[3..8]);
    }
}

#[test]
fn first_offset_need_not_be_zero() {
    // Offsets [4, 6, 6, 9] over "skipab" + "cde": rows "ab", "", "cde".
    for ty in STRING_TYPES {
        let bufs = vec![
            None,
            Buf::aligned(&offsets_bytes(ty, &[4, 6, 6, 9])),
            Buf::aligned(b"skipabcde"),
        ];
        let (a, released) = array(3, 0, bufs);
        let expected = rows(&[Some("ab"), Some(""), Some("cde")]);
        assert_imports(a, ty, &released, &expected);
    }
}

#[test]
fn offsets_outside_the_slice_are_not_checked() {
    // Entry 0 (before the slice) is larger than the rest: irrelevant.
    for ty in STRING_TYPES {
        let bufs = vec![
            None,
            Buf::aligned(&offsets_bytes(ty, &[9, 0, 1, 3])),
            Buf::aligned(b"abc"),
        ];
        let (mut a, released) = array(2, 0, bufs);
        a.offset = 1;
        let expected = rows(&[Some("a"), Some("bc")]);
        assert_imports(a, ty, &released, &expected);
    }
}

#[test]
fn a_null_data_buffer_is_fine_when_every_row_is_empty() {
    for ty in STRING_TYPES {
        let bufs = vec![None, Buf::aligned(&offsets_bytes(ty, &[2, 2, 2])), None];
        let (a, released) = array(2, 0, bufs);
        assert_imports(a, ty, &released, &rows(&[Some(""), Some("")]));
    }
}

#[test]
fn empty_arrays_read_no_buffers() {
    // length 0: no buffer is read, so null ones are accepted.
    for ty in [StrType::Utf8, StrType::LargeUtf8, StrType::Utf8View] {
        let (a, released) = array(0, 0, vec![None, None, None]);
        assert_imports(a, ty, &released, &[]);
    }
}

#[test]
fn invalid_utf8_rows_are_reported_not_trusted() {
    let mut long = "x".repeat(70).into_bytes();
    long[67] = 0xff; // past simdutf8's 64-byte SIMD block
    let mut long_ok = "ü".repeat(40).into_bytes(); // 80 bytes
    long_ok.push(b'!');
    let rows: Rows = vec![
        Some(vec![0xff]),
        Some(b"ok".to_vec()),
        Some(vec![b'a', 0xc3]),       // truncated sequence
        Some(vec![0xc0, 0x80]),       // overlong NUL
        Some(vec![0xed, 0xa0, 0x80]), // surrogate
        None,
        Some(long),
        Some(long_ok),
        Some(b"a\0b".to_vec()), // NUL is valid UTF-8
    ];
    for ty in STRING_TYPES {
        let (a, released) = string_array(ty, &rows, 1);
        let chunk = StrChunk::new(a, ty).unwrap();
        assert_rows(&chunk, &rows);
        let err = chunk.get(6).unwrap().unwrap_err();
        assert_eq!(err.valid_up_to(), 67);
        assert!(chunk.get(2).unwrap().unwrap_err().error_len().is_none());
        drop(chunk);
        assert_eq!(count(&released), 1);
    }
    let (a, released) = view_array(&rows, 2, 1);
    assert_imports(a, StrType::Utf8View, &released, &rows);
}

#[test]
#[should_panic(expected = "out of range")]
fn get_past_the_end_panics() {
    let rows = rows(&[Some("a")]);
    let (a, _released) = string_array(StrType::Utf8, &rows, 0);
    let chunk = StrChunk::new(a, StrType::Utf8).unwrap();
    let _ = chunk.get(1);
}

#[test]
#[should_panic(expected = "out of range")]
fn bytes_past_the_end_panics() {
    let rows = rows(&[Some("a")]);
    let (a, _released) = view_array(&rows, 1, 0);
    let chunk = StrChunk::new(a, StrType::Utf8View).unwrap();
    let _ = chunk.bytes(1);
}

// ---- utf8_view --------------------------------------------------------------

#[test]
fn view_arrays_round_trip() {
    for n_data in [1, 2, 3] {
        for skew in [0, 1, 3] {
            let rows = sample();
            let (a, released) = view_array(&rows, n_data, skew);
            assert_imports(a, StrType::Utf8View, &released, &rows);
        }
    }
}

#[test]
fn view_arrays_with_no_variadic_buffers() {
    // Only inline strings: n_buffers is 3 (validity, views, sizes).
    let rows = rows(&[Some("short"), None, Some(""), Some("twelve bytes")]);
    let (a, released) = view_array(&rows, 0, 0);
    assert_eq!(a.n_buffers, 3);
    assert_imports(a, StrType::Utf8View, &released, &rows);

    // ... and the sizes buffer may then be null.
    let views: Vec<u8> = [inline_view(b"abc"), inline_view(b"")].concat();
    let (a, released) = array(2, 0, vec![None, Buf::aligned(&views), None]);
    assert_imports(a, StrType::Utf8View, &released, &rows_of(&["abc", ""]));
}

fn rows_of(values: &[&str]) -> Rows {
    values.iter().map(|s| Some(s.as_bytes().to_vec())).collect()
}

#[test]
fn view_offset_slices() {
    let rows = sample();
    for (offset, length) in [(1, 9), (4, 4), (7, 1), (10, 0)] {
        let (mut a, released) = view_array(&rows, 2, 1);
        a.offset = offset;
        a.length = length;
        a.null_count = -1;
        let lo = usize::try_from(offset).unwrap();
        let hi = lo + usize::try_from(length).unwrap();
        assert_imports(a, StrType::Utf8View, &released, &rows[lo..hi]);
    }
}

#[test]
fn views_outside_the_slice_are_not_checked() {
    let views: Vec<u8> = [GARBAGE_VIEW, inline_view(b"in"), GARBAGE_VIEW].concat();
    let (mut a, released) = array(1, 0, vec![None, Buf::aligned(&views), None]);
    a.offset = 1;
    assert_imports(a, StrType::Utf8View, &released, &rows_of(&["in"]));
}

#[test]
fn unused_variadic_buffers_may_be_null_when_empty() {
    // Buffer 0 is null with size 0 (never referenced); buffer 1 holds the row.
    let text = "a long string, not inline";
    let views = long_view(i32::try_from(text.len()).unwrap(), b"a lo", 1, 0);
    let sizes = i64_bytes(&[0, i64::try_from(text.len()).unwrap()]);
    let bufs = vec![
        None,
        Buf::aligned(&views),
        None,
        Buf::aligned(text.as_bytes()),
        Buf::aligned(&sizes),
    ];
    let (a, released) = array(1, 0, bufs);
    assert_imports(a, StrType::Utf8View, &released, &rows_of(&[text]));
}

#[test]
fn a_long_view_may_end_exactly_at_its_buffer_end() {
    let data = b"0123456789abcdefXYZ";
    let views = long_view(16, b"0123", 0, 0);
    let tail = long_view(13, b"6789", 0, 6);
    let bufs = vec![
        None,
        Buf::aligned(&[views, tail].concat()),
        Buf::aligned(data),
        Buf::aligned(&i64_bytes(&[16])),
    ];
    let (a, released) = array(2, 0, bufs);
    let expected = rows_of(&["0123456789abcdef", "6789abcdefXYZ"]);
    // Row 1 ends at byte 19 > size 16: rejected.
    assert_rejects(a, StrType::Utf8View, &released, "row 1 points outside");

    let bufs = vec![
        None,
        Buf::aligned(&views),
        Buf::aligned(data),
        Buf::aligned(&i64_bytes(&[16])),
    ];
    let (a, released) = array(1, 0, bufs);
    assert_imports(a, StrType::Utf8View, &released, &expected[..1]);
}

// ---- rejections -------------------------------------------------------------

#[test]
fn rejects_a_released_array() {
    let err = StrChunk::new(FfiArray::released(), StrType::Utf8).unwrap_err();
    assert!(err.contains("released ArrowArray"), "{err}");
}

#[test]
fn rejects_negative_length_and_offset() {
    let rows = sample();
    for ty in [StrType::Utf8, StrType::LargeUtf8, StrType::Utf8View] {
        let (mut a, released) = string_array(StrType::LargeUtf8, &rows, 0);
        a.length = -1;
        assert_rejects(a, ty, &released, "negative ArrowArray.length -1");

        let (mut a, released) = string_array(StrType::LargeUtf8, &rows, 0);
        a.offset = i64::MIN;
        assert_rejects(a, ty, &released, "negative ArrowArray.offset");
    }
}

#[test]
fn rejects_offset_plus_length_overflow() {
    for (offset, length) in [(i64::MAX, i64::MAX), (i64::MAX, 0), (0, i64::MAX)] {
        let (mut a, released) = array(0, 0, vec![None, None, None]);
        (a.offset, a.length) = (offset, length);
        assert_rejects(a, StrType::Utf8, &released, "overflows");
    }
}

#[test]
fn rejects_children_and_dictionaries() {
    let rows = sample();
    let (mut a, released) = string_array(StrType::Utf8, &rows, 0);
    a.n_children = 1;
    assert_rejects(a, StrType::Utf8, &released, "no children");

    let (mut a, released) = string_array(StrType::Utf8, &rows, 0);
    a.n_children = -1;
    assert_rejects(a, StrType::Utf8, &released, "no children");

    // Never dereferenced: the importer only checks for null.
    let (mut a, released) = string_array(StrType::Utf8, &rows, 0);
    a.dictionary = ptr::dangling_mut();
    assert_rejects(a, StrType::Utf8, &released, "no dictionary");
}

#[test]
fn rejects_wrong_buffer_counts() {
    let three = || vec![None, None, None];
    for (ty, n) in [
        (StrType::Utf8, 2),
        (StrType::Utf8, 4),
        (StrType::LargeUtf8, 2),
        (StrType::LargeUtf8, 5),
        (StrType::Utf8View, 2),
        (StrType::Utf8View, 0),
    ] {
        let mut bufs = three();
        bufs.resize_with(n, || None);
        let (a, released) = array(1, 0, bufs);
        assert_rejects(a, ty, &released, &format!("({n} buffers)"));
    }
    // A negative count, with a valid C array behind it.
    let (mut a, released) = array(1, 0, three());
    a.n_buffers = -3;
    assert_rejects(a, StrType::Utf8, &released, "(-3 buffers)");
    // A null C array.
    let (mut a, released) = array(1, 0, three());
    a.buffers = ptr::null_mut();
    assert_rejects(a, StrType::Utf8, &released, "(3 buffers)");
}

#[test]
fn rejects_inconsistent_null_counts() {
    let rows = rows_without_nulls();
    for ty in STRING_TYPES {
        let (mut a, released) = string_array(ty, &rows, 0);
        a.null_count = 2;
        assert_rejects(
            a,
            ty,
            &released,
            "null_count is 2 but there is no validity bitmap",
        );

        let (mut a, released) = string_array(ty, &rows, 0);
        a.null_count = -2;
        assert_rejects(a, ty, &released, "invalid ArrowArray.null_count -2");

        let (mut a, released) = string_array(ty, &rows, 0);
        a.null_count = i64::MIN;
        assert_rejects(a, ty, &released, "invalid ArrowArray.null_count");
    }
    let (mut a, released) = view_array(&rows, 1, 0);
    a.null_count = 1;
    assert_rejects(a, StrType::Utf8View, &released, "no validity bitmap");
}

#[test]
fn rejects_null_string_buffers() {
    for ty in STRING_TYPES {
        let (a, released) = array(1, 0, vec![None, None, Buf::aligned(b"a")]);
        assert_rejects(a, ty, &released, "the offsets buffer is null");

        let offsets = offsets_bytes(ty, &[0, 1]);
        let (a, released) = array(1, 0, vec![None, Buf::aligned(&offsets), None]);
        assert_rejects(a, ty, &released, "the data buffer is null");
    }
}

#[test]
fn rejects_bad_string_offsets() {
    for ty in STRING_TYPES {
        for (offsets, needle) in [
            (vec![-1, 0, 1], "negative string offset -1"),
            (vec![0, 2, 1], "offsets decrease at row 1 (2 > 1)"),
            (vec![3, 2, 4], "offsets decrease at row 0 (3 > 2)"),
            (vec![0, -5, 1], "offsets decrease at row 0 (0 > -5)"),
        ] {
            for skew in [0, 1] {
                let bufs = vec![
                    None,
                    Some(Buf::new(&offsets_bytes(ty, &offsets), skew)),
                    Buf::aligned(b"abcd"),
                ];
                let (a, released) = array(2, 0, bufs);
                assert_rejects(a, ty, &released, needle);
            }
        }
    }
    // A bad offset inside a slice is still caught.
    let bufs = vec![
        None,
        Buf::aligned(&i64_bytes(&[0, 1, 0, 2])),
        Buf::aligned(b"ab"),
    ];
    let (mut a, released) = array(2, 0, bufs);
    a.offset = 1;
    assert_rejects(
        a,
        StrType::LargeUtf8,
        &released,
        "decrease at row 0 (1 > 0)",
    );
}

/// A one-row view array with the given views, variadic data and sizes.
fn view_case(
    views: &[[u8; 16]],
    data: &[Option<&[u8]>],
    sizes: Option<&[i64]>,
) -> (FfiArray, Arc<AtomicUsize>) {
    let mut bufs = vec![None, Buf::aligned(&views.concat())];
    bufs.extend(data.iter().map(|d| d.map(|d| Buf::new(d, 0))));
    bufs.push(sizes.map(|s| Buf::new(&i64_bytes(s), 1)));
    array(i64::try_from(views.len()).unwrap(), 0, bufs)
}

#[test]
fn rejects_bad_views() {
    let data: &[u8] = b"0123456789abcdefghij"; // 20 bytes
    let ok = long_view(15, b"0123", 0, 5);
    let cases: Vec<([u8; 16], &str)> = vec![
        (long_view(-1, b"", 0, 0), "negative string length at row 1"),
        (
            long_view(i32::MIN, b"", 0, 0),
            "negative string length at row 1",
        ),
        (long_view(13, b"", 1, 0), "row 1 points outside"), // no buffer 1
        (long_view(13, b"", -1, 0), "row 1 points outside"),
        (long_view(13, b"", i32::MAX, 0), "row 1 points outside"),
        (long_view(13, b"", 0, -1), "row 1 points outside"),
        (long_view(13, b"", 0, 8), "row 1 points outside"), // ends at 21 > 20
        (long_view(21, b"", 0, 0), "row 1 points outside"),
        (
            long_view(i32::MAX, b"", 0, i32::MAX),
            "row 1 points outside",
        ),
        (GARBAGE_VIEW, "negative string length at row 1"),
    ];
    for (bad, needle) in cases {
        let (a, released) = view_case(&[ok, bad], &[Some(data)], Some(&[20]));
        assert_rejects(a, StrType::Utf8View, &released, needle);
    }
}

#[test]
fn rejects_bad_variadic_buffers() {
    let data: &[u8] = b"0123456789abcdefghij";
    let view = long_view(13, b"0123", 0, 0);

    let (a, released) = view_case(&[view], &[None], Some(&[20]));
    assert_rejects(
        a,
        StrType::Utf8View,
        &released,
        "variadic data buffer 0 is null",
    );

    let (a, released) = view_case(&[view], &[Some(data)], None);
    assert_rejects(a, StrType::Utf8View, &released, "sizes buffer is null");

    for size in [-1, i64::MIN] {
        let (a, released) = view_case(&[view], &[Some(data), Some(data)], Some(&[20, size]));
        assert_rejects(
            a,
            StrType::Utf8View,
            &released,
            &format!("negative variadic buffer size {size}"),
        );
    }

    // A size smaller than the view needs.
    let (a, released) = view_case(&[view], &[Some(data)], Some(&[12]));
    assert_rejects(a, StrType::Utf8View, &released, "row 0 points outside");

    let (a, released) = view_case(&[view], &[Some(data)], Some(&[20]));
    // SAFETY: entry 1 of the 4-entry C array; `ptrs` is not otherwise borrowed.
    unsafe { a.buffers.wrapping_add(1).write(ptr::null()) };
    assert_rejects(a, StrType::Utf8View, &released, "the views buffer is null");
}

#[test]
fn null_rows_views_are_not_validated() {
    let data: &[u8] = b"0123456789abcdefghij";
    let views = [
        long_view(13, b"0123", 0, 2),
        GARBAGE_VIEW,
        long_view(99, b"", 7, 0),
    ];
    let bufs = vec![
        Buf::aligned(&bitmap([true, false, false], 3)),
        Buf::aligned(&views.concat()),
        Buf::aligned(data),
        Buf::aligned(&i64_bytes(&[20])),
    ];
    let expected: Rows = vec![Some(data[2..15].to_vec()), None, None];
    let (a, released) = array(3, 2, bufs);
    assert_imports(a, StrType::Utf8View, &released, &expected);
}

// ---- a producer that breaks the immutability contract ----------------------

/// Write `bytes` at `p + at`, as a misbehaving producer would.
///
/// # Safety
/// `p + at .. p + at + N` must be writable, and no reference to it be live.
unsafe fn poke<const N: usize>(p: *mut u8, at: usize, bytes: [u8; N]) {
    // SAFETY: per the caller's contract; `[u8; N]` has alignment 1.
    unsafe { p.wrapping_add(at).cast::<[u8; N]>().write(bytes) };
}

#[test]
fn mutated_offsets_stay_within_the_checked_bounds() {
    // `raw` re-clamps every read to what import validated (see the `Sync`
    // impl), so mutated offsets can't read outside the data buffer.
    let rows = rows_of(&["abc", "de", "f"]);
    let (a, released) = string_array(StrType::Utf8, &rows, 1);
    // SAFETY: entry 1 of the 3-entry C array.
    let offsets = unsafe { a.buffers.wrapping_add(1).read() }
        .cast::<u8>()
        .cast_mut();
    let chunk = StrChunk::new(a, StrType::Utf8).unwrap();
    // SAFETY: offsets entry 2 (bytes 8..12) of the 4-entry buffer; no row is borrowed.
    unsafe { poke(offsets, 8, 1000i32.to_ne_bytes()) };
    assert_eq!(chunk.bytes(1), Some(&b"def"[..])); // end 1000 clamped to 6
    assert_eq!(chunk.bytes(2), Some(&b""[..])); // start 1000 clamped to end 6
    // SAFETY: offsets entry 0 (bytes 0..4).
    unsafe { poke(offsets, 0, (-7i32).to_ne_bytes()) };
    assert_eq!(chunk.bytes(0), Some(&b"abc"[..])); // negative start -> 0
    // SAFETY: offsets entry 1 (bytes 4..8): end before start.
    unsafe { poke(offsets, 4, (-1i32).to_ne_bytes()) };
    assert_eq!(chunk.bytes(0), Some(&b""[..]));
    drop(chunk);
    assert_eq!(count(&released), 1);
}

#[test]
fn mutated_views_stay_within_the_checked_bounds() {
    let data: &[u8] = b"0123456789abcdefghij";
    let views = [long_view(13, b"0123", 0, 2), inline_view(b"inline")];
    let (a, released) = view_case(&views, &[Some(data)], Some(&[20]));
    // SAFETY: entry 1 of the 4-entry C array.
    let views_ptr = unsafe { a.buffers.wrapping_add(1).read() }
        .cast::<u8>()
        .cast_mut();
    let chunk = StrChunk::new(a, StrType::Utf8View).unwrap();
    assert_eq!(chunk.bytes(0), Some(&data[2..15]));
    for (at, bytes) in [
        (0, 30i32.to_ne_bytes()),    // longer than the buffer
        (0, (-3i32).to_ne_bytes()),  // negative length
        (8, 5i32.to_ne_bytes()),     // no such buffer
        (8, (-1i32).to_ne_bytes()),  // negative buffer index
        (12, (-1i32).to_ne_bytes()), // negative offset
    ] {
        // SAFETY: bytes at..at + 4 of view 0 (16 bytes); no row is borrowed.
        unsafe { poke(views_ptr, at, bytes) };
        assert_eq!(chunk.bytes(0), Some(&b""[..]), "after poking {at}");
        // SAFETY: as above; restore the original view.
        unsafe { poke(views_ptr, 0, long_view(13, b"0123", 0, 2)) };
    }
    // An inline length that grows past 12 turns it into a (now invalid) long view.
    // SAFETY: bytes 16..20 of the views buffer (view 1's length).
    unsafe { poke(views_ptr, 16, 13i32.to_ne_bytes()) };
    assert_eq!(chunk.bytes(1), Some(&b""[..]));
    drop(chunk);
    assert_eq!(count(&released), 1);
}

// ---- sharing across threads -------------------------------------------------

#[test]
fn a_chunk_is_readable_from_several_threads_at_once() {
    fn assert_sync<T: Sync>(_: &T) {}
    let rows = sample();
    let (a, utf8_released) = string_array(StrType::Utf8, &rows, 1);
    let utf8 = StrChunk::new(a, StrType::Utf8).unwrap();
    let (a, view_released) = view_array(&rows, 2, 1);
    let view = StrChunk::new(a, StrType::Utf8View).unwrap();
    assert_sync(&utf8);
    std::thread::scope(|s| {
        for _ in 0..3 {
            s.spawn(|| assert_rows(&utf8, &rows));
            s.spawn(|| assert_rows(&view, &rows));
        }
    });
    drop((utf8, view));
    assert_eq!(count(&utf8_released), 1);
    assert_eq!(count(&view_released), 1);
}

// ---- parse_schema -----------------------------------------------------------

struct SchemaPrivate {
    format: Option<CString>,
    name: Option<CString>,
    released: Arc<AtomicUsize>,
}

unsafe extern "C" fn release_schema(schema: *mut FfiSchema) {
    // SAFETY: `FfiSchema::drop` passes the live struct.
    let schema = unsafe { &mut *schema };
    // SAFETY: the `Box<SchemaPrivate>` from `schema`, freed only here.
    let private = unsafe { Box::from_raw(schema.private_data.cast::<SchemaPrivate>()) };
    private.released.fetch_add(1, Ordering::SeqCst);
    schema.private_data = ptr::null_mut();
    schema.release = None;
}

/// A live `ArrowSchema` with these format and name bytes (`None` = null).
fn schema(format: Option<&[u8]>, name: Option<&[u8]>) -> (FfiSchema, Arc<AtomicUsize>) {
    let released = Arc::new(AtomicUsize::new(0));
    let private = Box::new(SchemaPrivate {
        format: format.map(|f| CString::new(f).unwrap()),
        name: name.map(|n| CString::new(n).unwrap()),
        released: Arc::clone(&released),
    });
    // Taken after boxing: the strings' heap buffers don't move with the Box.
    let format = private.format.as_ref().map_or(ptr::null(), |f| f.as_ptr());
    let name = private.name.as_ref().map_or(ptr::null(), |n| n.as_ptr());
    let schema = FfiSchema {
        format,
        name,
        release: Some(release_schema),
        private_data: Box::into_raw(private).cast::<c_void>(),
        ..FfiSchema::released()
    };
    (schema, released)
}

#[test]
fn parse_schema_accepts_the_three_string_types() {
    for (format, ty) in [
        (&b"u"[..], StrType::Utf8),
        (b"U", StrType::LargeUtf8),
        (b"vu", StrType::Utf8View),
    ] {
        let (s, released) = schema(Some(format), Some(b"text"));
        let field = parse_schema(&s).unwrap();
        assert_eq!((field.ty, field.name.as_str()), (ty, "text"));
        assert_eq!(count(&released), 0); // parsing borrows, never releases
        drop(s);
        assert_eq!(count(&released), 1);
    }
}

#[test]
fn parse_schema_names() {
    let (s, _released) = schema(Some(b"u"), None);
    assert_eq!(parse_schema(&s).unwrap().name, "");
    let (s, _released) = schema(Some(b"u"), Some(b""));
    assert_eq!(parse_schema(&s).unwrap().name, "");
    let (s, _released) = schema(Some(b"u"), Some(b"\xffcol\xc3"));
    assert_eq!(parse_schema(&s).unwrap().name, "\u{fffd}col\u{fffd}");
    let (s, _released) = schema(Some(b"U"), Some("tëxt ✓".as_bytes()));
    assert_eq!(parse_schema(&s).unwrap().name, "tëxt ✓");
}

#[test]
fn parse_schema_rejects_other_types() {
    for (format, n_children, needle) in [
        (&b"z"[..], 0, "got binary; decode or cast"),
        (b"Z", 0, "got large_binary; decode or cast"),
        (b"vz", 0, "got binary_view; decode or cast"),
        (b"i", 0, "got int32"),
        (b"n", 0, "got null"),
        (b"tdD", 0, "got date32"),
        (b"+l", 1, "got list"),
        (b"+s", 2, "a struct with 2 field(s)"),
        (b"d:10,2", 0, "decimal type (format \"d:10,2\")"),
        (b"tss:UTC", 0, "temporal type"),
        (b"w:16", 0, "fixed_size_binary type"),
        (b"+w:3", 1, "nested type"),
        (b"", 0, "Arrow type (format \"\")"),
        (b"\xff", 0, "Arrow type (format \"\u{fffd}\")"),
        (b"uu", 0, "Arrow type (format \"uu\")"),
    ] {
        let (mut s, released) = schema(Some(format), Some(b"x"));
        s.n_children = n_children;
        let err = parse_schema(&s).unwrap_err();
        assert!(err.contains("expected an Arrow string column"), "{err}");
        assert!(err.contains(needle), "{err:?} does not contain {needle:?}");
        drop(s);
        assert_eq!(count(&released), 1);
    }
}

#[test]
fn parse_schema_rejects_malformed_schemas() {
    let (s, _released) = schema(None, Some(b"x"));
    assert!(parse_schema(&s).unwrap_err().contains("no format string"));

    let (mut s, released) = schema(Some(b"u"), Some(b"x"));
    s.dictionary = ptr::dangling_mut(); // only checked for null
    assert!(parse_schema(&s).unwrap_err().contains("dictionary-encoded"));
    drop(s);
    assert_eq!(count(&released), 1);

    let err = parse_schema(&FfiSchema::released()).unwrap_err();
    assert!(err.contains("released ArrowSchema"), "{err}");
}
