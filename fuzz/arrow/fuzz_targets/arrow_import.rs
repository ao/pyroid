//! pyroid-py's Arrow C Data importer, `StrChunk::new` and the row readers,
//! on producer structs built from fuzz bytes. The real source files are
//! compiled verbatim (see the `#[path]` modules); Python is linked, never
//! called. Input format and invariants: README.md.
//!
//! The C Data Interface carries no buffer lengths, and the importer trusts
//! the producer on them (import.rs, "Soundness"). So every buffer here is
//! exactly as long as a conformant producer guarantees, in its own `malloc`
//! allocation, and ASan flags any read past it:
//! - the `buffers` array: `n_buffers` pointers;
//! - validity: `ceil((offset + length) / 8)` bytes;
//! - utf8 / large_utf8 offsets: `offset + length + 1` entries;
//! - utf8 data: the last offset's worth of bytes when the offsets in the
//!   window `offset..=offset + length` are valid (non-negative,
//!   non-decreasing); when they are not, a 0-byte allocation: the importer
//!   must reject the array before reading any data;
//! - utf8_view views: `offset + length` views; variadic sizes: one int64 per
//!   variadic buffer; each variadic buffer: its declared size (0 bytes when
//!   the size is negative).
//!
//! Entries before `offset` (offsets, views) hold poison values that would
//! fail validation, so reading outside the window shows up as a mismatch.

#![no_main]

use std::ffi::c_void;
use std::ptr;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use libfuzzer_sys::fuzz_target;

#[allow(dead_code)]
#[path = "../../../crates/pyroid-py/src/errors.rs"]
mod errors;

// An inline module with a directory `#[path]`: its plain `mod` items load
// `ffi.rs` and `import.rs` from that directory as ordinary module files, so
// everything resolves as in pyroid-py: `super::ffi`, `crate::errors`, and
// their `#[cfg(test)] mod tests;` (`ffi/tests.rs`, not compiled here). A
// `#[path]` on each item would make them mod-rs files instead, and `mod
// tests;` would point at `arrow/tests.rs` (which breaks rustfmt).
#[allow(dead_code)]
#[path = "../../../crates/pyroid-py/src/arrow"]
mod arrow {
    pub mod ffi;
    pub mod import;
}

use arrow::ffi::FfiArray;
use arrow::import::{StrChunk, StrType};

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn free(p: *mut c_void);
}

/// `ARROW_FUZZ_VERBOSE=1`: print each outcome (to tally which checks a corpus
/// reaches: `-runs=0 corpus/arrow_import 2>&1 | sort | uniq -c`).
static VERBOSE: LazyLock<bool> = LazyLock::new(|| std::env::var_os("ARROW_FUZZ_VERBOSE").is_some());

// ------------------------------------------------------------------ limits

const MAX_ROWS: u16 = 1024;
const MAX_OFFSET: u8 = 64;
const MAX_VARIADIC_SIZE: u16 = 2048;
/// The largest utf8 data buffer built; inputs whose valid offsets need more
/// are skipped (a conformant producer would have to allocate it all).
const MAX_DATA: usize = 1 << 20;
const VIEW: usize = 16;

/// `length` / `offset` values for the header-only cases.
const SPECIAL: [i64; 10] = [
    -1,
    i64::MIN,
    i64::MAX,
    1 << 59,
    1 << 60,
    i64::MAX / 16,
    i64::MAX / 16 - 1,
    (1 << 59) - 1,
    -(1 << 40),
    0,
];

// ------------------------------------------------------------------ input

/// The fuzz bytes; reading past the end yields zeros.
struct Input<'a>(&'a [u8]);

impl Input<'_> {
    fn u8(&mut self) -> u8 {
        match self.0.split_first() {
            Some((&b, rest)) => {
                self.0 = rest;
                b
            }
            None => 0,
        }
    }

    fn array<const N: usize>(&mut self) -> [u8; N] {
        let mut b = [0; N];
        for x in &mut b {
            *x = self.u8();
        }
        b
    }

    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.array())
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.array())
    }

    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.array())
    }

    /// `n` bytes; once the input runs out, `'a'`s (valid UTF-8).
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let take = n.min(self.0.len());
        let mut v = self.0[..take].to_vec();
        self.0 = &self.0[take..];
        v.resize(n, b'a');
        v
    }
}

// ------------------------------------------------------------------ producer

static RELEASES: AtomicUsize = AtomicUsize::new(0);

/// `private_data`: every allocation the array owns.
struct Private {
    allocs: Vec<*mut c_void>,
}

impl Private {
    /// A `malloc` copy of `bytes` (a distinct allocation even when empty).
    fn buffer(&mut self, bytes: &[u8]) -> *mut u8 {
        // SAFETY: malloc has no preconditions (0 bytes gives a unique pointer).
        let p = unsafe { malloc(bytes.len()) }.cast::<u8>();
        assert!(!p.is_null(), "malloc({}) failed", bytes.len());
        // SAFETY: `p` is a fresh allocation of `bytes.len()` bytes.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len()) };
        self.allocs.push(p.cast());
        p
    }

    /// A `malloc` copy of a pointer array.
    fn pointers<T>(&mut self, ptrs: &[*mut T]) -> *mut *mut T {
        let n = size_of_val(ptrs);
        // SAFETY: as above.
        let p = unsafe { malloc(n) }.cast::<*mut T>();
        assert!(!p.is_null(), "malloc({n}) failed");
        // SAFETY: `p` is a fresh allocation of `n` bytes, and `malloc`
        // returns memory aligned for any pointer.
        unsafe { ptr::copy_nonoverlapping(ptrs.as_ptr(), p, ptrs.len()) };
        self.allocs.push(p.cast());
        p
    }
}

impl Drop for Private {
    fn drop(&mut self) {
        for &p in &self.allocs {
            // SAFETY: each pointer came from `malloc` and is freed once.
            unsafe { free(p) };
        }
    }
}

/// The release callback: frees everything, counts the call, marks the struct
/// released (spec, "Released structure").
unsafe extern "C" fn release(array: *mut FfiArray) {
    // SAFETY: the consumer passes the live struct it owns.
    let array = unsafe { &mut *array };
    assert!(
        array.release.is_some(),
        "release called on a released array"
    );
    // SAFETY: `private_data` is the `Box<Private>` of `build`; `release` is
    // cleared below, so this runs once per array.
    drop(unsafe { Box::from_raw(array.private_data.cast::<Private>()) });
    array.private_data = ptr::null_mut();
    array.release = None;
    RELEASES.fetch_add(1, SeqCst);
}

/// What the harness expects of the array, from the spec and its own bytes.
struct Expect {
    /// The importer must accept the array.
    ok: bool,
    /// Per row: `None` (null) or its bytes.
    rows: Vec<Option<Vec<u8>>>,
}

fn bit(bitmap: &[u8], i: usize) -> bool {
    bitmap[i / 8] & (1 << (i % 8)) != 0
}

/// Header fields and the shared part of the expectation.
struct Header {
    released: bool,
    buffers_null: bool,
    dictionary: bool,
    n_children: i64,
    n_buffers: Option<i64>,
    null_mask: u8,
    length: i64,
    offset: i64,
    null_count: i64,
    validity: Option<Vec<u8>>,
}

impl Header {
    fn read(inp: &mut Input<'_>) -> Option<Self> {
        let flags = inp.u8();
        let n_children = if flags & 0x08 != 0 {
            i64::from(inp.u8() as i8)
        } else {
            0
        };
        let n_buffers = (flags & 0x10 != 0).then(|| i64::from(inp.u8() as i8));
        let null_mask = if flags & 0x80 != 0 { inp.u8() } else { 0 };
        let (length, offset) = if flags & 0x20 != 0 {
            let l = SPECIAL[usize::from(inp.u8()) % SPECIAL.len()];
            let o = SPECIAL[usize::from(inp.u8()) % SPECIAL.len()];
            (l, o)
        } else {
            (
                i64::from(inp.u16() % (MAX_ROWS + 1)),
                i64::from(inp.u8() % (MAX_OFFSET + 1)),
            )
        };
        let mut h = Self {
            released: flags & 0x01 != 0,
            buffers_null: flags & 0x02 != 0,
            dictionary: flags & 0x04 != 0,
            n_children,
            n_buffers,
            null_mask,
            length,
            offset,
            null_count: 0,
            validity: None,
        };
        if flags & 0x20 != 0 {
            // Header-only: negative or overflowing lengths, which the
            // importer must reject before touching a buffer. (Huge valid
            // ones are skipped: no producer could back them.)
            return (!h.header_valid()).then_some(h);
        }
        let rows = h.rows().expect("small length and offset");
        if flags & 0x40 != 0 {
            h.validity = Some(inp.bytes(rows.div_ceil(8)));
        }
        let zeros = h.validity.as_ref().map_or(0, |v| {
            let off = usize::try_from(offset).expect("small");
            (off..rows).filter(|&i| !bit(v, i)).count()
        });
        h.null_count = match inp.u8() % 8 {
            0 | 1 => 0,
            2 => -1,
            3 | 4 => i64::try_from(zeros).expect("small"),
            5 => i64::from(inp.u8()) + 1,
            6 => [-2, i64::MIN, -(1 << 40)][usize::from(inp.u8()) % 3],
            _ => i64::MAX,
        };
        Some(h)
    }

    /// Non-negative `length` and `offset` whose `offset + length + 1` views
    /// are addressable.
    fn header_valid(&self) -> bool {
        let (Ok(len), Ok(off)) = (usize::try_from(self.length), usize::try_from(self.offset))
        else {
            return false;
        };
        off.checked_add(len)
            .and_then(|e| e.checked_add(1))
            .and_then(|e| e.checked_mul(VIEW))
            .is_some()
    }

    /// `offset + length`, if valid and small enough to build.
    fn rows(&self) -> Option<usize> {
        if !self.header_valid() {
            return None;
        }
        let rows = usize::try_from(self.offset + self.length).ok()?;
        (rows <= usize::from(MAX_ROWS) + usize::from(MAX_OFFSET)).then_some(rows)
    }

    fn is_null_slot(&self, k: usize) -> bool {
        k < 8 && self.null_mask & (1 << k) != 0
    }

    /// The importer's view of row `i`'s validity.
    fn valid(&self, validity: *const u8, i: usize) -> bool {
        if self.null_count == 0 || validity.is_null() {
            return true;
        }
        let off = usize::try_from(self.offset).expect("checked");
        bit(self.validity.as_deref().expect("non-null"), off + i)
    }

    /// The checks every type shares, before any buffer is read.
    fn ok(&self, conformant_n_buffers: bool, validity: *const u8) -> bool {
        !self.released
            && self.header_valid()
            && self.n_children == 0
            && !self.dictionary
            && conformant_n_buffers
            && !self.buffers_null
            && match self.null_count {
                0 | -1 => true,
                n => n > 0 && !validity.is_null(),
            }
    }
}

/// A conformant-sized array (see the module docs) and what to expect of it.
fn build(ty: StrType, inp: &mut Input<'_>) -> Option<(FfiArray, Expect)> {
    let h = Header::read(inp)?;
    let mut private = Box::new(Private { allocs: Vec::new() });
    let rows = h.rows();
    let len = usize::try_from(h.length).unwrap_or(0);

    let validity = match &h.validity {
        Some(v) if !h.is_null_slot(0) => private.buffer(v).cast_const(),
        _ => ptr::null(),
    };
    let mut slots: Vec<*const u8> = vec![validity];
    let mut expect = Expect {
        ok: false,
        rows: Vec::new(),
    };
    let conformant_n_buffers;
    if rows.is_none() {
        // Header-only: no data buffers at all.
        slots.extend([ptr::null(); 2]);
        conformant_n_buffers = true;
    } else if ty == StrType::Utf8View {
        conformant_n_buffers = view_buffers(&h, inp, &mut private, &mut slots, &mut expect);
    } else {
        conformant_n_buffers = offset_buffers(ty, &h, inp, &mut private, &mut slots, &mut expect)?;
    }
    expect.ok &= h.ok(conformant_n_buffers, validity);
    if len == 0 {
        expect.rows.clear();
    }
    assert!(!expect.ok || expect.rows.len() == len);

    let layout = i64::try_from(slots.len()).expect("small");
    let n_buffers = match h.n_buffers {
        // A view array's n_buffers says how many variadic buffers it has.
        Some(n) if ty != StrType::Utf8View || n < 3 => n,
        _ => layout,
    };
    let n = usize::try_from(n_buffers).unwrap_or(0);
    slots.resize(n, ptr::null());
    let buffers = if h.buffers_null {
        ptr::null_mut()
    } else {
        let ptrs: Vec<*mut c_void> = slots.iter().map(|p| p.cast_mut().cast()).collect();
        private.pointers(&ptrs).cast::<*const c_void>()
    };
    let children = if h.n_children > 0 {
        let n = usize::try_from(h.n_children).expect("positive");
        private.pointers(&vec![ptr::null_mut::<FfiArray>(); n])
    } else {
        ptr::null_mut()
    };
    let dictionary = if h.dictionary {
        // A zeroed (released) struct: never read by a string importer.
        private
            .buffer(&[0; size_of::<FfiArray>()])
            .cast::<FfiArray>()
    } else {
        ptr::null_mut()
    };
    let array = FfiArray {
        length: h.length,
        null_count: h.null_count,
        offset: h.offset,
        n_buffers,
        n_children: h.n_children,
        buffers,
        children,
        dictionary,
        release: (!h.released).then_some(release as unsafe extern "C" fn(*mut FfiArray)),
        // Freed by `release`, or by the harness if the array is released.
        private_data: Box::into_raw(private).cast(),
    };
    Some((array, expect))
}

/// Build the offsets and data buffers; `None` skips the input (valid
/// offsets that would need more than `MAX_DATA` data bytes).
fn offset_buffers(
    ty: StrType,
    h: &Header,
    inp: &mut Input<'_>,
    private: &mut Private,
    slots: &mut Vec<*const u8>,
    expect: &mut Expect,
) -> Option<bool> {
    let len = usize::try_from(h.length).expect("checked");
    let off = usize::try_from(h.offset).expect("checked");
    let large = ty == StrType::LargeUtf8;
    let mode = inp.u8() % 4;
    let mut window: Vec<i64> = Vec::with_capacity(len + 1);
    let mut at: i64 = if mode == 3 { i64::from(inp.u16()) } else { 0 };
    for _ in 0..=len {
        if mode == 2 {
            at = if large {
                inp.u64() as i64
            } else {
                i64::from(inp.u32() as i32)
            };
        }
        window.push(at);
        if mode != 2 {
            at += i64::from(inp.u8() % 17);
        }
    }
    if mode == 1 {
        let pos = usize::from(inp.u16()) % (len + 1);
        window[pos] = match inp.u8() % 4 {
            0 => -1,
            1 => {
                if large {
                    i64::MIN
                } else {
                    i64::from(i32::MIN)
                }
            }
            2 => window[pos] - 1 - i64::from(inp.u8()),
            _ => {
                if large {
                    i64::MAX
                } else {
                    i64::from(i32::MAX)
                }
            }
        };
    }
    // Poison before the window: negative, so a read there fails validation.
    let mut entries: Vec<i64> = (0..off).map(|i| -7 - i as i64).collect();
    entries.extend(&window);
    let bytes: Vec<u8> = if large {
        entries.iter().flat_map(|v| v.to_le_bytes()).collect()
    } else {
        entries
            .iter()
            .flat_map(|&v| i32::try_from(v).expect("i32 offsets").to_le_bytes())
            .collect()
    };
    let offsets = if h.is_null_slot(1) {
        ptr::null()
    } else {
        private.buffer(&bytes).cast_const()
    };

    let first = window[0];
    let last = window[len];
    let valid = first >= 0 && window.windows(2).all(|w| w[0] <= w[1]);
    let data_bytes = if valid {
        let n = usize::try_from(last).expect("non-negative");
        if n > MAX_DATA {
            return None;
        }
        inp.bytes(n)
    } else {
        Vec::new() // invalid offsets: the importer must not read data
    };
    let data = if h.is_null_slot(2) {
        ptr::null()
    } else {
        private.buffer(&data_bytes).cast_const()
    };
    slots.extend([offsets, data]);

    expect.ok = len == 0 || (!offsets.is_null() && valid && (last == first || !data.is_null()));
    if expect.ok {
        let validity = slots[0];
        expect.rows = (0..len)
            .map(|i| {
                h.valid(validity, i).then(|| {
                    let s = usize::try_from(window[i]).expect("valid");
                    let e = usize::try_from(window[i + 1]).expect("valid");
                    data_bytes[s..e].to_vec()
                })
            })
            .collect();
    }
    Some(h.n_buffers.is_none_or(|n| n == 3))
}

/// A view's fields.
fn view(n: i32, prefix: [u8; 4], buffer: i32, at: i32) -> [u8; VIEW] {
    let mut v = [0; VIEW];
    v[..4].copy_from_slice(&n.to_le_bytes());
    v[4..8].copy_from_slice(&prefix);
    v[8..12].copy_from_slice(&buffer.to_le_bytes());
    v[12..].copy_from_slice(&at.to_le_bytes());
    v
}

/// Build the views, variadic buffers and sizes; returns whether `n_buffers`
/// is the layout's.
fn view_buffers(
    h: &Header,
    inp: &mut Input<'_>,
    private: &mut Private,
    slots: &mut Vec<*const u8>,
    expect: &mut Expect,
) -> bool {
    let len = usize::try_from(h.length).expect("checked");
    let off = usize::try_from(h.offset).expect("checked");
    let n_var = usize::from(inp.u8() % 4);
    let mut sizes: Vec<i64> = Vec::with_capacity(n_var);
    let mut var_bufs: Vec<Vec<u8>> = Vec::with_capacity(n_var);
    for _ in 0..n_var {
        let s = inp.u16();
        let size = if s & 0x8000 != 0 {
            [-1, i64::MIN, -i64::from(s & 0x7fff)][usize::from(s % 3)]
        } else {
            i64::from(s % (MAX_VARIADIC_SIZE + 1))
        };
        var_bufs.push(inp.bytes(usize::try_from(size).unwrap_or(0)));
        sizes.push(size);
    }
    // Poison before the window: a negative length.
    let mut views: Vec<[u8; VIEW]> = (0..off).map(|_| view(-1, [0; 4], -1, -1)).collect();
    for _ in 0..len {
        let kind = inp.u8() % 6;
        let eligible: Vec<usize> = (0..n_var).filter(|&b| var_bufs[b].len() > 12).collect();
        let v = match kind {
            1 | 2 if !eligible.is_empty() => {
                let b = eligible[usize::from(inp.u8()) % eligible.len()];
                let size = var_bufs[b].len();
                let n = 13 + usize::from(inp.u16()) % (size - 12);
                let at = usize::from(inp.u16()) % (size - n + 1);
                let d = &var_bufs[b][at..at + n];
                let to_i32 = |x: usize| i32::try_from(x).expect("small");
                view(to_i32(n), [d[0], d[1], d[2], d[3]], to_i32(b), to_i32(at))
            }
            3 => {
                // Long, likely out of range.
                let n = 13 + i32::from(inp.u8());
                view(n, [0; 4], i32::from(inp.u8() as i8), inp.u32() as i32)
            }
            4 => inp.array(),
            5 => view([-1, i32::MIN][usize::from(inp.u8() & 1)], [0; 4], 0, 0),
            _ => {
                let n = usize::from(inp.u8()) % 13;
                let mut v = view(i32::try_from(n).expect("small"), [0; 4], 0, 0);
                v[4..4 + n].copy_from_slice(&inp.bytes(n));
                v
            }
        };
        views.push(v);
    }
    let flat: Vec<u8> = views.iter().flatten().copied().collect();
    let views_ptr = if h.is_null_slot(1) {
        ptr::null()
    } else {
        private.buffer(&flat).cast_const()
    };
    slots.push(views_ptr);
    let mut var_ptrs = Vec::with_capacity(n_var);
    for (b, d) in var_bufs.iter().enumerate() {
        let p = if h.is_null_slot(2 + b) {
            ptr::null()
        } else {
            private.buffer(d).cast_const()
        };
        var_ptrs.push(p);
    }
    slots.extend(&var_ptrs);
    let size_bytes: Vec<u8> = sizes.iter().flat_map(|s| s.to_le_bytes()).collect();
    let sizes_ptr = if h.is_null_slot(2 + n_var) {
        ptr::null()
    } else {
        private.buffer(&size_bytes).cast_const()
    };
    slots.push(sizes_ptr);

    let validity = slots[0];
    let mut ok = !views_ptr.is_null()
        && (n_var == 0 || !sizes_ptr.is_null())
        && sizes.iter().all(|&s| s >= 0)
        && (0..n_var).all(|b| sizes[b] == 0 || !var_ptrs[b].is_null());
    let mut rows = Vec::with_capacity(len);
    for i in 0..len {
        if !h.valid(validity, i) {
            rows.push(None);
            continue;
        }
        let v = &views[off + i];
        let field = |at: usize| i32::from_le_bytes([v[at], v[at + 1], v[at + 2], v[at + 3]]);
        let (n, b, at) = (field(0), field(8), field(12));
        let row = match usize::try_from(n) {
            Err(_) => None,
            Ok(n) if n <= 12 => Some(v[4..4 + n].to_vec()),
            Ok(n) => usize::try_from(b)
                .ok()
                .zip(usize::try_from(at).ok())
                .filter(|&(b, at)| b < n_var && at + n <= var_bufs[b].len())
                .map(|(b, at)| var_bufs[b][at..at + n].to_vec()),
        };
        ok &= row.is_some();
        rows.push(row);
    }
    expect.ok = len == 0 || ok;
    expect.rows = rows;
    // A view array's n_buffers *defines* how many variadic buffers it has, so
    // only values the importer must reject (below 3) replace the layout's.
    h.n_buffers.is_none_or(|n| n >= 3)
}

fuzz_target!(|data: &[u8]| {
    let mut inp = Input(data);
    let ty = [StrType::Utf8, StrType::LargeUtf8, StrType::Utf8View][usize::from(inp.u8() % 3)];
    let Some((array, expect)) = build(ty, &mut inp) else {
        return;
    };
    let released = array.release.is_none();
    let private = array.private_data.cast::<Private>();
    let len = usize::try_from(array.length).unwrap_or(0);
    RELEASES.store(0, SeqCst);
    let result = StrChunk::new(array, ty);
    if released {
        // SAFETY: from `Box::into_raw` in `build`; a released array's
        // callback never runs, so the harness still owns its buffers.
        drop(unsafe { Box::from_raw(private) });
    }
    let chunk = match result {
        Err(msg) => {
            if *VERBOSE {
                eprintln!("reject {ty:?}: {msg}");
            }
            assert!(!expect.ok, "rejected a valid {ty:?} array: {msg}");
            assert!(!msg.is_empty());
            assert_eq!(
                RELEASES.load(SeqCst),
                usize::from(!released),
                "release count on Err"
            );
            return;
        }
        Ok(chunk) => chunk,
    };
    if *VERBOSE {
        eprintln!("accept {ty:?}");
    }
    assert!(expect.ok, "accepted an invalid {ty:?} array");
    assert_eq!(RELEASES.load(SeqCst), 0, "released while still owned");
    assert_eq!(chunk.len(), len);
    for (i, want) in expect.rows.iter().enumerate() {
        let bytes = chunk.bytes(i);
        let text = chunk.get(i);
        match want {
            None => {
                assert!(bytes.is_none(), "row {i} should be null");
                assert!(text.is_none(), "row {i} should be null");
            }
            Some(want) => {
                assert_eq!(bytes, Some(&want[..]), "row {i} bytes");
                match text {
                    Some(Ok(s)) => assert_eq!(s.as_bytes(), &want[..], "row {i} text"),
                    Some(Err(_)) => assert!(std::str::from_utf8(want).is_err(), "row {i} is UTF-8"),
                    None => panic!("row {i} is null"),
                }
            }
        }
    }
    drop(chunk);
    assert_eq!(RELEASES.load(SeqCst), 1, "release count on drop");
});
