//! `pyroid.Column`: the Arrow result of a pyroid function given Arrow input,
//! and the minimal C Data / C Stream *exporter* behind it.
//!
//! A `Column` owns Rust-allocated, immutable buffers (from
//! `pyroid_core::columnar`). Every exported struct keeps an `Arc` to the chunk
//! it points into, and every release callback only frees Rust memory: it never
//! touches Python. Consumers may therefore move and release exported structs
//! on any thread (polars and pyarrow do, from their own thread pools).
//!
//! A column is a list of [`Span`]s, each a run of rows of one chunk. Slicing a
//! column (`col[a:b]`) makes shorter spans over the same chunks, without a
//! copy; they are exported with the Arrow `offset` field. A column is also a
//! read-only Python sequence: `len`, indexing, iteration (lazy, from the
//! buffers), `in`, `index` and `count`.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use pyo3::exceptions::{PyIndexError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::slice::PySliceIndices;
use pyo3::types::{PyBool, PyBytes, PyCapsule, PyList, PySlice, PyString, PyTuple, PyType};
use pyroid_core::columnar::{BoolBuffers, BuildError, Offset, StringBuffers, get_bit};

use super::ffi::{FLAG_NULLABLE, FfiArray, FfiSchema, FfiStream};
use super::fill::Fill;
use super::import::{ARRAY_CAPSULE, SCHEMA_CAPSULE, STREAM_CAPSULE};
use crate::errors::{PyroidError, type_name};

/// The Arrow type of a [`Column`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnType {
    Utf8,
    LargeUtf8,
    Bool,
    /// `large_binary` (`pyroid.warc`'s `html` column).
    LargeBinary,
}

impl ColumnType {
    pub(super) fn format(self) -> &'static CStr {
        match self {
            Self::Utf8 => c"u",
            Self::LargeUtf8 => c"U",
            Self::Bool => c"b",
            Self::LargeBinary => c"Z",
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Utf8 => "string",
            Self::LargeUtf8 => "large_string",
            Self::Bool => "bool",
            Self::LargeBinary => "large_binary",
        }
    }
}

/// One chunk of a column.
#[derive(Debug)]
pub enum Chunk {
    Utf8(StringBuffers<i32>),
    LargeUtf8(StringBuffers<i64>),
    Bool(BoolBuffers),
    /// The `large_utf8` layout holding any bytes.
    LargeBinary(StringBuffers<i64>),
}

impl Chunk {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Utf8(b) => b.len,
            Self::LargeUtf8(b) | Self::LargeBinary(b) => b.len,
            Self::Bool(b) => b.len,
        }
    }

    fn null_count(&self) -> usize {
        match self {
            Self::Utf8(b) => b.null_count,
            Self::LargeUtf8(b) | Self::LargeBinary(b) => b.null_count,
            Self::Bool(b) => b.null_count,
        }
    }

    fn validity(&self) -> Option<&[u8]> {
        match self {
            Self::Utf8(b) => b.validity.as_deref(),
            Self::LargeUtf8(b) | Self::LargeBinary(b) => b.validity.as_deref(),
            Self::Bool(b) => b.validity.as_deref(),
        }
    }

    fn empty(ty: ColumnType) -> Self {
        match ty {
            ColumnType::Utf8 => Self::Utf8(empty_strings()),
            ColumnType::LargeUtf8 => Self::LargeUtf8(empty_strings()),
            ColumnType::Bool => Self::Bool(BoolBuffers::build(&[])),
            ColumnType::LargeBinary => Self::LargeBinary(empty_strings()),
        }
    }

    pub(super) fn ty(&self) -> ColumnType {
        match self {
            Self::Utf8(_) => ColumnType::Utf8,
            Self::LargeUtf8(_) => ColumnType::LargeUtf8,
            Self::Bool(_) => ColumnType::Bool,
            Self::LargeBinary(_) => ColumnType::LargeBinary,
        }
    }
}

fn empty_strings<O: Offset>() -> StringBuffers<O> {
    StringBuffers {
        len: 0,
        offsets: vec![O::from_usize(0)],
        data: Vec::new(),
        validity: None,
        null_count: 0,
    }
}

/// Number of set bits among bits `start..start + len` of an LSB-first bitmap.
fn count_set(bitmap: &[u8], start: usize, len: usize) -> usize {
    let end = start + len;
    let mut i = start;
    let mut n = 0;
    while i < end && i % 8 != 0 {
        n += usize::from(get_bit(bitmap, i));
        i += 1;
    }
    while i + 8 <= end {
        n += bitmap[i / 8].count_ones() as usize;
        i += 8;
    }
    while i < end {
        n += usize::from(get_bit(bitmap, i));
        i += 1;
    }
    n
}

/// Rows `start..start + len` of one chunk: what a column is made of. A span
/// over a whole chunk has `start == 0`; slicing a column makes shorter spans
/// that share the chunk.
#[derive(Clone, Debug)]
pub(super) struct Span {
    chunk: Arc<Chunk>,
    start: usize,
    len: usize,
    null_count: usize,
}

impl Span {
    fn whole(chunk: Arc<Chunk>) -> Self {
        let (len, null_count) = (chunk.len(), chunk.null_count());
        Self {
            chunk,
            start: 0,
            len,
            null_count,
        }
    }

    /// Rows `from..to` of this span (`from <= to <= self.len`).
    fn sub(&self, from: usize, to: usize) -> Self {
        debug_assert!(from <= to && to <= self.len);
        let (start, len) = (self.start + from, to - from);
        let null_count = if len == self.len {
            self.null_count
        } else if self.null_count == 0 {
            0
        } else {
            len - self
                .chunk
                .validity()
                .map_or(len, |v| count_set(v, start, len))
        };
        Self {
            chunk: Arc::clone(&self.chunk),
            start,
            len,
            null_count,
        }
    }

    /// The chunk's row numbers this span covers.
    fn rows(&self) -> std::ops::Range<usize> {
        self.start..self.start + self.len
    }
}

#[derive(Debug)]
struct ColumnData {
    ty: ColumnType,
    name: CString,
    /// Non-empty spans, in order.
    spans: Vec<Span>,
    /// `ends[k]` is the number of rows in `spans[..=k]`.
    ends: Vec<usize>,
    /// The concatenation `__arrow_c_array__` exports, built on first use.
    joined: OnceLock<Arc<Chunk>>,
}

impl ColumnData {
    fn new(ty: ColumnType, name: CString, spans: impl IntoIterator<Item = Span>) -> Self {
        let spans: Vec<Span> = spans.into_iter().filter(|s| s.len > 0).collect();
        debug_assert!(spans.iter().all(|s| s.chunk.ty() == ty));
        let mut rows = 0;
        let ends = spans
            .iter()
            .map(|s| {
                rows += s.len;
                rows
            })
            .collect();
        Self {
            ty,
            name,
            spans,
            ends,
            joined: OnceLock::new(),
        }
    }

    fn len(&self) -> usize {
        self.ends.last().copied().unwrap_or(0)
    }

    /// First row of `spans[k]`.
    fn first_row(&self, k: usize) -> usize {
        if k == 0 { 0 } else { self.ends[k - 1] }
    }

    /// The index of the span holding row `i` (`i < len`). `guess` is tried
    /// first (the span of the previous row, say), so a walk over the rows
    /// searches only when it moves to the next chunk.
    fn span_index(&self, i: usize, guess: usize) -> usize {
        if guess < self.spans.len() && self.first_row(guess) <= i && i < self.ends[guess] {
            guess
        } else {
            self.ends.partition_point(|&end| end <= i)
        }
    }

    /// Span `k` and the number, in its chunk, of the column's row `i`.
    fn row_in(&self, k: usize, i: usize) -> (&Span, usize) {
        let span = &self.spans[k];
        (span, span.start + (i - self.first_row(k)))
    }

    /// The span holding row `i` (`i < len`) and the row's number in its chunk.
    fn locate(&self, i: usize) -> (&Span, usize) {
        self.row_in(self.span_index(i, 0), i)
    }

    /// Rows `from..to` (`from <= to <= len`), over the same chunks.
    fn slice(&self, from: usize, to: usize) -> Self {
        let mut spans = Vec::new();
        if from < to {
            let first = self.ends.partition_point(|&end| end <= from);
            for (k, span) in self.spans.iter().enumerate().skip(first) {
                let base = self.first_row(k);
                if base >= to {
                    break;
                }
                let end = self.ends[k];
                spans.push(span.sub(from.max(base) - base, to.min(end) - base));
            }
        }
        Self::new(self.ty, self.name.clone(), spans)
    }

    /// The rows `rows` yields (each `< len`), copied into new chunks. Calls
    /// `tick` every [`CHECK_ROWS`] rows and stops at its first error.
    fn gather<E>(
        &self,
        rows: impl IntoIterator<Item = usize>,
        mut tick: impl FnMut() -> Result<(), E>,
    ) -> Result<Self, E> {
        let rows = rows.into_iter();
        let mut out = Gather::new(self.ty, rows.size_hint().0);
        let mut k = 0;
        for (n, row) in rows.enumerate() {
            if n % CHECK_ROWS == CHECK_ROWS - 1 {
                tick()?;
            }
            // A slice's next row is mostly in the same chunk as the last.
            k = self.span_index(row, k);
            let (span, at) = self.row_in(k, row);
            out.push(&span.chunk, at);
        }
        let spans = out.finish().into_iter().map(|c| Span::whole(Arc::new(c)));
        Ok(Self::new(self.ty, self.name.clone(), spans))
    }

    /// The whole column as one span. Several spans are concatenated (calling
    /// `tick` as [`join`] does), and the concatenation is kept for the next call.
    fn single_span<E: From<BuildError>>(
        &self,
        mut tick: impl FnMut() -> Result<(), E>,
    ) -> Result<Span, E> {
        match self.spans.as_slice() {
            [] => Ok(Span::whole(Arc::new(Chunk::empty(self.ty)))),
            [one] => Ok(one.clone()),
            many => {
                if let Some(done) = self.joined.get() {
                    return Ok(Span::whole(Arc::clone(done)));
                }
                let joined = join(self.ty, many, &mut tick)?;
                Ok(Span::whole(Arc::clone(
                    self.joined.get_or_init(|| Arc::new(joined)),
                )))
            }
        }
    }
}

/// Rows read between two Ctrl-C checks in a loop that runs attached (a scan,
/// a copy, `to_pylist`), as `input.rs` does for long lists. Fewer under Miri,
/// so the unit tests reach a check without 65,536 interpreted rows.
const CHECK_ROWS: usize = if cfg!(miri) { 1 << 6 } else { 1 << 16 };

/// `rows` in runs of at most [`CHECK_ROWS`], for a Ctrl-C check before each.
fn blocks(rows: std::ops::Range<usize>) -> impl Iterator<Item = std::ops::Range<usize>> {
    let end = rows.end;
    rows.step_by(CHECK_ROWS)
        .map(move |start| start..end.min(start + CHECK_ROWS))
}

/// `py.check_signals()` every [`CHECK_ROWS`] calls.
fn tick(py: Python<'_>, n: &mut usize) -> PyResult<()> {
    *n += 1;
    if *n == CHECK_ROWS {
        *n = 0;
        py.check_signals()?;
    }
    Ok(())
}

impl Column {
    /// A column of `ty` from `chunks` (which must all be of that type).
    pub fn new(ty: ColumnType, name: &str, chunks: Vec<Chunk>) -> Self {
        Self::from_shared(ty, name, chunks.into_iter().map(Arc::new).collect())
    }

    /// A column of `ty` over chunks that something else also holds (no copy).
    pub fn from_shared(ty: ColumnType, name: &str, chunks: Vec<Arc<Chunk>>) -> Self {
        debug_assert!(chunks.iter().all(|c| c.ty() == ty));
        // Interior NULs can't be represented in an Arrow field name.
        let name = CString::new(name.replace('\0', "")).unwrap_or_default();
        Self::wrap(ColumnData::new(
            ty,
            name,
            chunks.into_iter().map(Span::whole),
        ))
    }

    fn wrap(data: ColumnData) -> Self {
        Self {
            inner: Arc::new(data),
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
        Chunk::LargeUtf8(b) | Chunk::LargeBinary(b) => b,
        _ => unreachable!("column chunks share one type"),
    }
}

fn as_bool(c: &Chunk) -> &BoolBuffers {
    match c {
        Chunk::Bool(b) => b,
        _ => unreachable!("column chunks share one type"),
    }
}

/// Bytes copied between two `tick` calls in [`join`] (16 MiB, a few ms). Fewer
/// under Miri, so the unit tests copy a row's bytes in several pieces.
const COPY_BYTES: usize = if cfg!(miri) { 1 << 4 } else { 1 << 24 };

/// Concatenate `spans` (of type `ty`) into one chunk, calling `tick` between
/// runs of [`CHECK_ROWS`] rows and of [`COPY_BYTES`] bytes; stops at its first
/// error.
fn join<E: From<BuildError>>(
    ty: ColumnType,
    spans: &[Span],
    tick: &mut impl FnMut() -> Result<(), E>,
) -> Result<Chunk, E> {
    Ok(match ty {
        ColumnType::Utf8 => Chunk::Utf8(join_strings(spans, as_utf8, tick)?),
        ColumnType::LargeUtf8 => Chunk::LargeUtf8(join_strings(spans, as_large, tick)?),
        ColumnType::LargeBinary => Chunk::LargeBinary(join_strings(spans, as_large, tick)?),
        ColumnType::Bool => {
            let len = spans.iter().map(|s| s.len).sum();
            let null_count = spans.iter().map(|s| s.null_count).sum();
            let mut values = Bits::with_capacity(len);
            let mut valid = Bits::with_capacity(if null_count > 0 { len } else { 0 });
            for s in spans {
                let b = as_bool(&s.chunk);
                for rows in blocks(s.rows()) {
                    tick()?;
                    values.extend_from_bitmap(&b.values, rows.start, rows.len());
                    if null_count > 0 {
                        valid.extend_validity(b.validity.as_deref(), rows);
                    }
                }
            }
            Chunk::Bool(BoolBuffers {
                len,
                values: values.finish(),
                validity: (null_count > 0).then(|| valid.finish()),
                null_count,
            })
        }
    })
}

fn join_strings<O: Offset, E: From<BuildError>>(
    spans: &[Span],
    get: fn(&Chunk) -> &StringBuffers<O>,
    tick: &mut impl FnMut() -> Result<(), E>,
) -> Result<StringBuffers<O>, E> {
    let bytes = |s: &Span| {
        let b = get(&s.chunk);
        b.offsets[s.start].to_usize()..b.offsets[s.start + s.len].to_usize()
    };
    let total: usize = spans.iter().map(|s| bytes(s).len()).sum();
    if total > O::MAX {
        return Err(BuildError::Overflow {
            bytes: total,
            max: O::MAX,
        }
        .into());
    }
    let len = spans.iter().map(|s| s.len).sum();
    let null_count = spans.iter().map(|s| s.null_count).sum();
    let mut offsets = Vec::with_capacity(len + 1);
    offsets.push(O::from_usize(0));
    let mut data = Vec::with_capacity(total);
    let mut valid = Bits::with_capacity(if null_count > 0 { len } else { 0 });
    for s in spans {
        let b = get(&s.chunk);
        let range = bytes(s);
        let (base, first) = (data.len(), range.start);
        for rows in blocks(s.rows()) {
            tick()?;
            offsets.extend(
                b.offsets[rows.start + 1..=rows.end]
                    .iter()
                    .map(|o| O::from_usize(base + (o.to_usize() - first))),
            );
            if null_count > 0 {
                valid.extend_validity(b.validity.as_deref(), rows);
            }
        }
        for from in range.clone().step_by(COPY_BYTES) {
            tick()?;
            data.extend_from_slice(&b.data[from..range.end.min(from + COPY_BYTES)]);
        }
    }
    Ok(StringBuffers {
        len,
        offsets,
        data,
        validity: (null_count > 0).then(|| valid.finish()),
        null_count,
    })
}

/// An LSB-first bitmap built one bit at a time.
#[derive(Debug, Default)]
struct Bits {
    /// The full bytes so far.
    bytes: Vec<u8>,
    /// The bits after them (`len % 8` of them).
    last: u8,
    len: usize,
}

impl Bits {
    fn with_capacity(bits: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(bits.div_ceil(8)),
            last: 0,
            len: 0,
        }
    }

    #[inline]
    fn push(&mut self, bit: bool) {
        self.last |= u8::from(bit) << (self.len % 8);
        self.len += 1;
        if self.len % 8 == 0 {
            self.bytes.push(std::mem::take(&mut self.last));
        }
    }

    /// Append bits `from..from + n` of the LSB-first bitmap `src`, a byte at
    /// a time where it can.
    fn extend_from_bitmap(&mut self, src: &[u8], from: usize, n: usize) {
        let end = from + n;
        let mut i = from;
        while i < end && self.len % 8 != 0 {
            self.push(get_bit(src, i));
            i += 1;
        }
        // Byte-aligned output: each byte is 8 source bits, from one source
        // byte or two (`i + 8 <= end`, so both exist).
        let shift = i % 8;
        while i + 8 <= end {
            let low = src[i / 8] >> shift;
            let high = if shift == 0 {
                0
            } else {
                src[i / 8 + 1] << (8 - shift)
            };
            self.bytes.push(low | high);
            self.len += 8;
            i += 8;
        }
        while i < end {
            self.push(get_bit(src, i));
            i += 1;
        }
    }

    /// Append `n` set bits.
    fn extend_ones(&mut self, n: usize) {
        let mut left = n;
        while left > 0 && self.len % 8 != 0 {
            self.push(true);
            left -= 1;
        }
        self.bytes.resize(self.bytes.len() + left / 8, u8::MAX);
        self.len += left / 8 * 8;
        for _ in 0..left % 8 {
            self.push(true);
        }
    }

    /// Append bits `rows` of a validity bitmap (`None`: every row is valid).
    fn extend_validity(&mut self, validity: Option<&[u8]>, rows: std::ops::Range<usize>) {
        match validity {
            Some(v) => self.extend_from_bitmap(v, rows.start, rows.len()),
            None => self.extend_ones(rows.len()),
        }
    }

    /// The bitmap: `len.div_ceil(8)` bytes.
    fn finish(mut self) -> Vec<u8> {
        if self.len % 8 != 0 {
            self.bytes.push(self.last);
        }
        self.bytes
    }
}

/// Builds a bool array row by row.
#[derive(Debug, Default)]
struct BoolsBuilder {
    values: Bits,
    valid: Bits,
    null_count: usize,
}

impl BoolsBuilder {
    fn with_capacity(rows: usize) -> Self {
        Self {
            values: Bits::with_capacity(rows),
            valid: Bits::with_capacity(rows),
            null_count: 0,
        }
    }

    fn push(&mut self, row: Option<bool>) {
        self.values.push(row == Some(true));
        self.valid.push(row.is_some());
        self.null_count += usize::from(row.is_none());
    }

    fn finish(self) -> BoolBuffers {
        BoolBuffers {
            len: self.values.len,
            values: self.values.finish(),
            validity: (self.null_count > 0).then(|| self.valid.finish()),
            null_count: self.null_count,
        }
    }
}

/// New chunks built row by row (a slice with a step).
enum Gather {
    Utf8(StringsBuilder<i32>),
    LargeUtf8(StringsBuilder<i64>),
    Bool(BoolsBuilder),
    LargeBinary(StringsBuilder<i64>),
}

impl Gather {
    /// A builder for about `rows` rows of type `ty`.
    fn new(ty: ColumnType, rows: usize) -> Self {
        match ty {
            ColumnType::Utf8 => Self::Utf8(StringsBuilder::with_capacity(rows)),
            ColumnType::LargeUtf8 => Self::LargeUtf8(StringsBuilder::with_capacity(rows)),
            ColumnType::Bool => Self::Bool(BoolsBuilder::with_capacity(rows)),
            ColumnType::LargeBinary => Self::LargeBinary(StringsBuilder::with_capacity(rows)),
        }
    }

    /// Append row `at` of `chunk` (which has this builder's type).
    fn push(&mut self, chunk: &Chunk, at: usize) {
        match (self, chunk) {
            (Self::Utf8(out), Chunk::Utf8(b)) => out.push(b.get(at)),
            (Self::LargeUtf8(out), Chunk::LargeUtf8(b))
            | (Self::LargeBinary(out), Chunk::LargeBinary(b)) => out.push(b.get(at)),
            (Self::Bool(out), Chunk::Bool(b)) => out.push(b.get(at)),
            _ => unreachable!("column chunks share one type"),
        }
    }

    fn finish(self) -> Vec<Chunk> {
        match self {
            Self::Utf8(b) => b.finish().into_iter().map(Chunk::Utf8).collect(),
            Self::LargeUtf8(b) => b.finish().into_iter().map(Chunk::LargeUtf8).collect(),
            Self::Bool(b) => vec![Chunk::Bool(b.finish())],
            Self::LargeBinary(b) => b.finish().into_iter().map(Chunk::LargeBinary).collect(),
        }
    }
}

/// Builds string arrays row by row, starting a new one where the current one
/// would pass `O::MAX` bytes (2 GiB for `utf8`), as the batch functions do.
struct StringsBuilder<O> {
    done: Vec<StringBuffers<O>>,
    offsets: Vec<O>,
    data: Vec<u8>,
    valid: Bits,
    null_count: usize,
}

impl<O: Offset> Default for StringsBuilder<O> {
    fn default() -> Self {
        Self {
            done: Vec::new(),
            offsets: vec![O::from_usize(0)],
            data: Vec::new(),
            valid: Bits::default(),
            null_count: 0,
        }
    }
}

impl<O: Offset> StringsBuilder<O> {
    fn with_capacity(rows: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(O::from_usize(0));
        Self {
            offsets,
            valid: Bits::with_capacity(rows),
            ..Self::default()
        }
    }

    /// Append a row. Rows come from arrays of the same offset type, so one row
    /// always fits an empty array.
    fn push(&mut self, row: Option<&[u8]>) {
        let n = row.map_or(0, <[u8]>::len);
        if self.data.len() + n > O::MAX {
            self.flush();
        }
        match row {
            Some(bytes) => self.data.extend_from_slice(bytes),
            None => self.null_count += 1,
        }
        self.valid.push(row.is_some());
        self.offsets.push(O::from_usize(self.data.len()));
    }

    fn flush(&mut self) {
        let len = self.valid.len;
        if len == 0 {
            return;
        }
        let valid = std::mem::take(&mut self.valid);
        let null_count = std::mem::take(&mut self.null_count);
        self.done.push(StringBuffers {
            len,
            offsets: std::mem::replace(&mut self.offsets, vec![O::from_usize(0)]),
            data: std::mem::take(&mut self.data),
            validity: (null_count > 0).then(|| valid.finish()),
            null_count,
        });
    }

    fn finish(mut self) -> Vec<StringBuffers<O>> {
        self.flush();
        self.done
    }
}

/// A read-only Arrow column produced by pyroid (one or more chunks).
///
/// pyroid returns a `Column` when it is given Arrow input. It implements the
/// Arrow `PyCapsule` interface, so any Arrow library can take it without a copy:
/// `pyarrow.chunked_array(col)`, `pyarrow.array(col)`, `polars.Series(col)`,
/// or pass it back to pyroid. pyroid itself needs no Arrow library.
///
/// It is also a read-only sequence of its values, which come back as
/// `to_pylist()` gives them (`str`, `bool` or `bytes`, and `None` for a null):
/// `len(col)`, `col[i]` (negative `i` counts from the end), `for value in col`,
/// `reversed(col)`, `value in col`, `index` and `count`. Iteration reads each
/// row when it gets to it, so it never builds a list of the whole column.
/// `col[a:b]` is a `Column` over the same buffers, without a copy; a slice
/// with a step (`col[::2]`) copies the rows it selects into a new `Column`.
// `sequence`: `len()` fills `sq_length`, so the C API's `PySequence_Size`
// works (numpy uses it). PyO3 adds `sq_item` for `__getitem__` either way.
#[pyclass(module = "pyroid", name = "Column", frozen, sequence)]
#[derive(Debug)]
pub struct Column {
    inner: Arc<ColumnData>,
}

/// Map a buffer-building failure to the right Python exception.
pub fn build_error(e: BuildError) -> PyErr {
    match e {
        BuildError::Overflow { .. } => PyOverflowError::new_err(e.to_string()),
        BuildError::Pool(p) => crate::interrupt::pool_error(&p),
    }
}

/// Why a copy that checks for Ctrl-C stopped, as a Python exception: the
/// check's own error, or a [`build_error`].
struct PyStop(PyErr);

impl From<BuildError> for PyStop {
    fn from(e: BuildError) -> Self {
        Self(build_error(e))
    }
}

fn to_i64(n: usize) -> i64 {
    i64::try_from(n).expect("Rust allocations are at most isize::MAX")
}

// ---- export: ArrowSchema ---------------------------------------------------

struct SchemaPrivate {
    name: CString,
}

pub(super) fn export_schema(ty: ColumnType, name: &CStr) -> FfiSchema {
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

/// Export a whole chunk.
pub(super) fn export_array(chunk: Arc<Chunk>) -> FfiArray {
    export_span(Span::whole(chunk))
}

/// Export a span: the chunk's buffers, with the span's rows picked out by
/// `offset` and `length` (C Data spec, "ArrowArray.offset").
fn export_span(span: Span) -> FfiArray {
    let Span {
        chunk,
        start,
        len,
        null_count,
    } = span;
    let (n_buffers, buffers) = match &*chunk {
        Chunk::Utf8(b) => (
            3,
            [
                validity_ptr(b.validity.as_ref()),
                b.offsets.as_ptr().cast(),
                b.data.as_ptr().cast(),
            ],
        ),
        Chunk::LargeUtf8(b) | Chunk::LargeBinary(b) => (
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
    debug_assert!(start + len <= chunk.len());
    let (length, null_count, offset) = (to_i64(len), to_i64(null_count), to_i64(start));
    let private = Box::into_raw(Box::new(ArrayPrivate {
        _chunk: chunk,
        buffers,
    }));
    // Derived from the raw pointer *after* `into_raw`, so it stays valid under
    // the aliasing model (a pointer taken from the Box first would not).
    // SAFETY: `private` is a valid, live allocation we just created.
    let buffers_ptr = unsafe { &raw mut (*private).buffers }.cast::<*const c_void>();
    FfiArray {
        length,
        null_count,
        offset,
        n_buffers,
        n_children: 0,
        buffers: buffers_ptr,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_array),
        private_data: private.cast::<c_void>(),
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
    let array = match private.column.spans.get(private.next) {
        Some(span) => {
            private.next += 1;
            export_span(span.clone())
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
pub(super) struct Exported<T>(T);

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
pub(super) fn capsule<'py, T: 'static>(
    py: Python<'py>,
    value: T,
    name: &'static CStr,
) -> PyResult<Bound<'py, PyCapsule>>
where
    Exported<T>: Send,
{
    PyCapsule::new_with_value(py, Exported(value), name)
}

// ---- values as Python objects ------------------------------------------------

/// Row `at` of `chunk` as `to_pylist` gives it: `str`, `bool` or `bytes`, or
/// `None` for a null.
fn row_value<'py>(py: Python<'py>, chunk: &Chunk, at: usize) -> PyResult<Bound<'py, PyAny>> {
    let none = || py.None().into_bound(py);
    Ok(match chunk {
        Chunk::Utf8(b) => b.get(at).map_or_else(|| Ok(none()), |s| str_value(py, s))?,
        Chunk::LargeUtf8(b) => b.get(at).map_or_else(|| Ok(none()), |s| str_value(py, s))?,
        Chunk::Bool(b) => b
            .get(at)
            .map_or_else(none, |v| PyBool::new(py, v).to_owned().into_any()),
        Chunk::LargeBinary(b) => b
            .get(at)
            .map_or_else(none, |v| PyBytes::new(py, v).into_any()),
    })
}

/// Rows `rows` of a string array, put in the next slots of `list`.
fn push_strings<'py, O: Offset>(
    py: Python<'py>,
    list: &mut Fill<'py, PyList>,
    b: &StringBuffers<O>,
    rows: std::ops::Range<usize>,
) -> PyResult<()> {
    for at in rows {
        list.push(match b.get(at) {
            None => py.None().into_bound(py),
            Some(bytes) => str_value(py, bytes)?,
        });
    }
    Ok(())
}

fn str_value<'py>(py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    // Built from `&str`s, so always valid UTF-8.
    let s = std::str::from_utf8(bytes)
        .map_err(|e| PyroidError::new_err(format!("internal error: {e}")))?;
    Ok(PyString::new(py, s).into_any())
}

/// What a value given to `in`, `index` or `count` is compared with.
enum Needle<'a> {
    /// `None`: the null rows (`None is None`; no value equals `None`).
    Null,
    /// A `str` on a string column or `bytes` on a binary one: equal rows
    /// have the same bytes.
    Bytes(&'a [u8]),
    /// A `bool` on a bool column.
    Bool(bool),
    /// A `str` that is not valid Unicode (lone surrogates): no row equals it.
    Nothing,
    /// Anything else: compared in Python, `row is value or row == value`, as
    /// `list` compares (so `1 in bools` holds, and `__eq__` overrides run).
    Python,
}

fn needle<'a>(ty: ColumnType, value: &'a Bound<'_, PyAny>) -> Needle<'a> {
    if value.is_none() {
        return Needle::Null;
    }
    // Exact types only: a subclass may override `__eq__`.
    match ty {
        ColumnType::Utf8 | ColumnType::LargeUtf8 if value.is_exact_instance_of::<PyString>() => {
            match value.cast::<PyString>().map(|s| s.to_str()) {
                Ok(Ok(s)) => Needle::Bytes(s.as_bytes()),
                _ => Needle::Nothing,
            }
        }
        ColumnType::LargeBinary if value.is_exact_instance_of::<PyBytes>() => value
            .cast::<PyBytes>()
            .map_or(Needle::Python, |b| Needle::Bytes(b.as_bytes())),
        ColumnType::Bool if value.is_exact_instance_of::<PyBool>() => value
            .cast::<PyBool>()
            .map_or(Needle::Python, |b| Needle::Bool(b.is_true())),
        _ => Needle::Python,
    }
}

impl ColumnData {
    /// Call `found(i)` for each row `i` in `from..to` equal to `value`, in
    /// order, until it returns `false`.
    fn scan(
        &self,
        value: &Bound<'_, PyAny>,
        from: usize,
        to: usize,
        mut found: impl FnMut(usize) -> bool,
    ) -> PyResult<()> {
        let py = value.py();
        let needle = needle(self.ty, value);
        if matches!(needle, Needle::Nothing) || from >= to {
            return Ok(());
        }
        let mut n = 0;
        let first = self.ends.partition_point(|&end| end <= from);
        for (k, span) in self.spans.iter().enumerate().skip(first) {
            let base = self.first_row(k);
            if base >= to {
                break;
            }
            let rows = from.max(base) - base..to.min(self.ends[k]) - base;
            if matches!(needle, Needle::Null) && span.null_count == 0 {
                continue;
            }
            for r in rows {
                tick(py, &mut n)?;
                let at = span.start + r;
                let hit = match (&needle, &*span.chunk) {
                    (Needle::Null, c) => c.validity().is_some_and(|v| !get_bit(v, at)),
                    (Needle::Bytes(s), Chunk::Utf8(b)) => b.get(at) == Some(*s),
                    (Needle::Bytes(s), Chunk::LargeUtf8(b) | Chunk::LargeBinary(b)) => {
                        b.get(at) == Some(*s)
                    }
                    (Needle::Bool(v), Chunk::Bool(b)) => b.get(at) == Some(*v),
                    _ => {
                        let row = row_value(py, &span.chunk, at)?;
                        row.is(value) || row.eq(value)?
                    }
                };
                if hit && !found(base + r) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

/// Row `i` of a column of `len` rows, negative counting from the end.
fn resolve(i: isize, len: usize) -> Option<usize> {
    let i = if i < 0 {
        len.checked_sub(i.unsigned_abs())?
    } else {
        usize::try_from(i).ok()?
    };
    (i < len).then_some(i)
}

/// An index as `list` takes it: an `int` or anything with `__index__`; one
/// past the `isize` range is an `IndexError`.
fn row_index(key: &Bound<'_, PyAny>) -> PyResult<isize> {
    let py = key.py();
    key.extract::<isize>().map_err(|e| {
        if e.is_instance_of::<PyOverflowError>(py) {
            PyIndexError::new_err(format!(
                "cannot fit '{}' into an index-sized integer",
                type_name(key)
            ))
        } else if e.is_instance_of::<PyTypeError>(py)
            && !key
                .get_type()
                .hasattr(intern!(py, "__index__"))
                .unwrap_or(false)
        {
            PyTypeError::new_err(format!(
                "Column indices must be integers or slices, not {}",
                type_name(key)
            ))
        } else {
            e
        }
    })
}

/// A `start`/`stop` argument of `index`, as `list.index` takes it: any
/// integer, clamped to the `isize` range.
struct SliceIndex(isize);

impl<'a, 'py> FromPyObject<'a, 'py> for SliceIndex {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        match obj.extract::<isize>() {
            Ok(i) => Ok(Self(i)),
            Err(e) if e.is_instance_of::<PyOverflowError>(py) => {
                let i = py
                    .import(intern!(py, "operator"))?
                    .call_method1(intern!(py, "index"), (obj,))?;
                Ok(Self(if i.lt(0)? { isize::MIN } else { isize::MAX }))
            }
            Err(e)
                if e.is_instance_of::<PyTypeError>(py)
                    && !obj
                        .get_type()
                        .hasattr(intern!(py, "__index__"))
                        .unwrap_or(false) =>
            {
                Err(PyTypeError::new_err(
                    "slice indices must be integers or have an __index__ method",
                ))
            }
            Err(e) => Err(e),
        }
    }
}

/// `start`/`stop` of `index(value, start, stop)` as rows: negative counts from
/// the end, then clamped to `0..=len`.
fn clamp(i: isize, len: usize) -> usize {
    if i < 0 {
        len.saturating_sub(i.unsigned_abs())
    } else {
        usize::try_from(i).map_or(len, |i| i.min(len))
    }
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
        let span = self
            .inner
            .single_span(|| py.check_signals().map_err(PyStop))
            .map_err(|PyStop(e)| e)?;
        let schema = capsule(
            py,
            export_schema(self.inner.ty, &self.inner.name),
            SCHEMA_CAPSULE,
        )?;
        let array = capsule(py, export_span(span), ARRAY_CAPSULE)?;
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
        py.import("types")?
            .getattr("GenericAlias")?
            .call1((cls, item))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// `col[i]`: the value of row `i` (``None`` for a null; negative `i`
    /// counts from the end). `col[a:b]`: a `Column` over the same buffers;
    /// `col[a:b:k]` with a step other than 1 copies the rows it selects.
    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        let len = self.inner.len();
        if let Ok(slice) = key.cast::<PySlice>() {
            let n = isize::try_from(len)
                .map_err(|_| PyOverflowError::new_err("column too long to slice"))?;
            let column = self.slice(py, slice.indices(n)?)?;
            return Ok(Bound::new(py, column)?.into_any());
        }
        let row = resolve(row_index(key)?, len)
            .ok_or_else(|| PyIndexError::new_err("Column index out of range"))?;
        let (span, at) = self.inner.locate(row);
        row_value(py, &span.chunk, at)
    }

    /// Iterate over the values, reading each row as it is reached.
    fn __iter__(&self) -> ColumnIterator {
        ColumnIterator::new(Arc::clone(&self.inner), false)
    }

    /// Iterate over the values from the last row to the first.
    fn __reversed__(&self) -> ColumnIterator {
        ColumnIterator::new(Arc::clone(&self.inner), true)
    }

    /// `value in col`, as for a list: some row is `value` or equals it.
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut hit = false;
        self.inner.scan(value, 0, self.inner.len(), |_| {
            hit = true;
            false
        })?;
        Ok(hit)
    }

    /// Return the first row in `start:stop` whose value equals `value` (as
    /// `list.index` compares). Raise `ValueError` if there is none.
    #[pyo3(
        signature = (value, start = SliceIndex(0), stop = SliceIndex(isize::MAX), /),
        text_signature = "($self, value, start=0, stop=sys.maxsize, /)"
    )]
    fn index(
        &self,
        value: &Bound<'_, PyAny>,
        start: SliceIndex,
        stop: SliceIndex,
    ) -> PyResult<usize> {
        let len = self.inner.len();
        let mut first = None;
        self.inner
            .scan(value, clamp(start.0, len), clamp(stop.0, len), |i| {
                first = Some(i);
                false
            })?;
        first.ok_or_else(|| {
            let shown = value
                .repr()
                .map_or_else(|_| "value".to_owned(), |r| r.to_string());
            PyValueError::new_err(format!("{shown} is not in Column"))
        })
    }

    /// Return the number of rows whose value equals `value`.
    #[pyo3(signature = (value, /))]
    fn count(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut n = 0;
        self.inner.scan(value, 0, self.inner.len(), |_| {
            n += 1;
            true
        })?;
        Ok(n)
    }

    /// Arrow type name: ``"string"``, ``"large_string"``, ``"bool"`` or
    /// ``"large_binary"``.
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
        self.inner.spans.iter().map(|s| s.null_count).sum()
    }

    /// Number of chunks.
    #[getter]
    fn num_chunks(&self) -> usize {
        self.inner.spans.len()
    }

    /// The values as a Python list (``None`` for nulls).
    fn to_pylist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        // Made at its final size and filled in place: appending one by one is
        // slow on free-threaded builds, and a list built from a collected
        // `Vec` would end in a pass with no Ctrl-C check.
        let mut list = Fill::<PyList>::new(py, self.inner.len())?;
        for span in &self.inner.spans {
            for rows in blocks(span.rows()) {
                py.check_signals()?;
                match &*span.chunk {
                    Chunk::Utf8(b) => push_strings(py, &mut list, b, rows)?,
                    Chunk::LargeUtf8(b) => push_strings(py, &mut list, b, rows)?,
                    Chunk::Bool(b) => {
                        for at in rows {
                            list.push(b.get(at).map_or_else(
                                || py.None().into_bound(py),
                                |v| PyBool::new(py, v).to_owned().into_any(),
                            ));
                        }
                    }
                    Chunk::LargeBinary(b) => {
                        for at in rows {
                            list.push(b.get(at).map_or_else(
                                || py.None().into_bound(py),
                                |v| PyBytes::new(py, v).into_any(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(list.finish())
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

impl Column {
    /// `col[slice]` for the slice's `indices(len(col))`.
    fn slice(&self, py: Python<'_>, ind: PySliceIndices) -> PyResult<Self> {
        let PySliceIndices {
            start,
            step,
            slicelength,
            ..
        } = ind;
        if slicelength == 0 {
            return Ok(Self::wrap(self.inner.slice(0, 0)));
        }
        // `indices` puts `start` on a row when the slice selects any.
        let first = usize::try_from(start).expect("slice start is a row");
        if step == 1 || slicelength == 1 {
            return Ok(Self::wrap(self.inner.slice(first, first + slicelength)));
        }
        let rows = (0..slicelength).map(|k| {
            let k = isize::try_from(k).expect("at most len(col) rows");
            usize::try_from(start + k * step).expect("slice rows are rows")
        });
        Ok(Self::wrap(self.inner.gather(rows, || py.check_signals())?))
    }
}

/// The iterator `iter(col)` and `reversed(col)` return. Threads may share
/// one: each call to `next` takes a row no other call takes.
#[pyclass(module = "pyroid", name = "ColumnIterator", frozen)]
#[derive(Debug)]
pub struct ColumnIterator {
    column: Arc<ColumnData>,
    /// Forward: the next row. Reversed: the rows left (the next is `pos - 1`).
    pos: AtomicUsize,
    /// The span of the last row taken, tried first for the next one. Threads
    /// sharing the iterator may overwrite each other's; it is only a guess.
    span: AtomicUsize,
    reversed: bool,
}

impl ColumnIterator {
    fn new(column: Arc<ColumnData>, reversed: bool) -> Self {
        let pos = if reversed { column.len() } else { 0 };
        Self {
            column,
            pos: AtomicUsize::new(pos),
            span: AtomicUsize::new(0),
            reversed,
        }
    }

    /// Take the next row, if any. Only the claim is atomic; the rows are
    /// immutable, so no ordering with other memory is needed.
    fn claim(&self) -> Option<usize> {
        let len = self.column.len();
        let mut pos = self.pos.load(Ordering::Relaxed);
        loop {
            // Reversed: the row is `pos - 1`, and `pos` drops to it.
            let (row, next) = if self.reversed {
                let row = pos.checked_sub(1)?;
                (row, row)
            } else {
                (pos < len).then_some((pos, pos + 1))?
            };
            match self
                .pos
                .compare_exchange_weak(pos, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return Some(row),
                Err(now) => pos = now,
            }
        }
    }
}

#[pymethods]
impl ColumnIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some(row) = self.claim() else {
            return Ok(None);
        };
        let guess = self.span.load(Ordering::Relaxed);
        let k = self.column.span_index(row, guess);
        if k != guess {
            self.span.store(k, Ordering::Relaxed);
        }
        let (span, at) = self.column.row_in(k, row);
        row_value(py, &span.chunk, at).map(Some)
    }

    /// Rows not yet taken (`operator.length_hint`).
    fn __length_hint__(&self) -> usize {
        let pos = self.pos.load(Ordering::Relaxed);
        if self.reversed {
            pos
        } else {
            self.column.len().saturating_sub(pos)
        }
    }
}

#[cfg(test)]
mod tests;
