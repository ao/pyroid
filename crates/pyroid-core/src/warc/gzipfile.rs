//! CPython's `gzip.GzipFile` read path, as datatrove reads `.warc.gz` files (fsspec wraps the
//! file in `gzip.GzipFile(fileobj=f)`).
//!
//! `GzipFile.read(n)` is `io.BufferedReader(_GzipReader(f)).read(n)`. This module reproduces
//! that stack call for call: the C `BufferedReader` (`Modules/_io/bufferedio.c`: the read-ahead
//! buffer, `MINUS_LAST_BLOCK` direct reads, `fill_buffer`, and the bytes a failing `read(n)`
//! loses), `_GzipReader.read` with its `_PaddedFile` (the same `f.read(k)` calls on the
//! underlying file, in the same sizes), the header parser, `_read_eof` (CRC and length checks,
//! zero padding), multi-member files, and the two zlib front ends: `zlib._ZlibDecompressor`
//! with `max_length` (3.12+) and `zlib.decompressobj` with `unconsumed_tail` (3.11). Inflate is
//! zlib-rs (through flate2); errors carry CPython's types and messages. When zlib-rs's fast
//! path finds the error it reports `repeated call with bad state` instead of zlib's message;
//! the member's deflate data, kept for that (up to 64 MiB per member), is then decoded again
//! through the slow path, which gives zlib's message.
//!
//! Every call on the underlying file goes through [`Source::read`], so a Python file object sees
//! the same `read(k)` calls it would see under CPython, and its exceptions surface at the same
//! `GzipFile.read(n)` call. After an exception the state is CPython's too (warcio keeps reading
//! after an `EOFError`).
//!
//! Sources must return at most `k` bytes per `read(k)` (`Source`'s contract); CPython would also
//! accept longer reads, with results that differ in places (a longer trailer read fails
//! `struct.unpack`).

use std::fmt::Write as _;
use std::ops::Range;

use flate2::{Crc, Decompress, FlushDecompress, Status};

use super::{Error, PyExc, Source};

/// The CPython-version dependent constants of the `GzipFile` read path.
///
/// | field | 3.11 | 3.12, 3.13 | 3.14 | 3.15+ |
/// |---|---|---|---|---|
/// | `buffer_size` (`io.DEFAULT_BUFFER_SIZE`) | 8192 | 8192 | 131072 | 131072 |
/// | `read_size` | 8192 (`io.DEFAULT_BUFFER_SIZE`) | 131072 (`gzip.READ_BUFFER_SIZE`) | 131072 | 131072 |
/// | `zlib_decompressor` | false | true | true | true |
/// | `check_header_crc` | false | false | false | true |
///
/// The Python layer can take them from the running interpreter: `io.DEFAULT_BUFFER_SIZE`,
/// `getattr(gzip, "READ_BUFFER_SIZE", io.DEFAULT_BUFFER_SIZE)`, `hasattr(zlib,
/// "_ZlibDecompressor")` and `sys.version_info >= (3, 15)`, or call [`GzipParams::for_python`].
///
/// 3.12 and 3.13 differ in one line (3.12 does not reset `buf` when the decompressor still holds
/// input); with zlib's inflate that branch always produces output, so both use the same values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GzipParams {
    /// `io.BufferedReader` buffer size (`GzipFile` uses the default, `io.DEFAULT_BUFFER_SIZE`).
    pub buffer_size: usize,
    /// Size of `_GzipReader`'s reads from the underlying file.
    pub read_size: usize,
    /// `zlib._ZlibDecompressor` (3.12+: the decompressor keeps unconsumed input, `needs_input`)
    /// instead of `zlib.decompressobj` (3.11: `unconsumed_tail` goes back to the `_PaddedFile`).
    pub zlib_decompressor: bool,
    /// Check the FHCRC header checksum (3.15+: `Corrupted gzip header. Checksums do not match`).
    pub check_header_crc: bool,
}

impl GzipParams {
    /// CPython 3.11.
    pub const PY311: Self = Self {
        buffer_size: 8192,
        read_size: 8192,
        zlib_decompressor: false,
        check_header_crc: false,
    };
    /// CPython 3.12 and 3.13.
    pub const PY312: Self = Self {
        buffer_size: 8192,
        read_size: 131_072,
        zlib_decompressor: true,
        check_header_crc: false,
    };
    /// CPython 3.14.
    pub const PY314: Self = Self {
        buffer_size: 131_072,
        read_size: 131_072,
        zlib_decompressor: true,
        check_header_crc: false,
    };
    /// CPython 3.15.
    pub const PY315: Self = Self {
        buffer_size: 131_072,
        read_size: 131_072,
        zlib_decompressor: true,
        check_header_crc: true,
    };

    /// The parameters of CPython `3.<minor>` (3.11 and later; later versions than 3.15 get 3.15's).
    pub fn for_python(minor: u32) -> Option<Self> {
        match minor {
            0..=10 => None,
            11 => Some(Self::PY311),
            12 | 13 => Some(Self::PY312),
            14 => Some(Self::PY314),
            _ => Some(Self::PY315),
        }
    }
}

const FHCRC: u8 = 2;
const FEXTRA: u8 = 4;
const FNAME: u8 = 8;
const FCOMMENT: u8 = 16;

/// `_BlocksOutputBuffer`-style growth for unlimited `decompress()` calls: the first output chunk.
const FIRST_CHUNK: usize = 64 * 1024;
/// The largest output chunk handed to one inflate call when growing a `Vec`.
const MAX_CHUNK: usize = 4 * 1024 * 1024;

fn eof_error() -> Error {
    Error::Py(
        PyExc::EofError,
        "Compressed file ended before the end-of-stream marker was reached".to_owned(),
    )
}

fn bad_gzip(msg: String) -> Error {
    Error::Py(PyExc::BadGzipFile, msg)
}

/// Python's `repr(bytes)`.
fn py_bytes_repr(b: &[u8]) -> String {
    let quote = if b.contains(&b'\'') && !b.contains(&b'"') {
        '"'
    } else {
        '\''
    };
    let mut s = String::with_capacity(b.len() * 4 + 3);
    s.push('b');
    s.push(quote);
    for &c in b {
        match c {
            b'\\' => s.push_str("\\\\"),
            b'\t' => s.push_str("\\t"),
            b'\n' => s.push_str("\\n"),
            b'\r' => s.push_str("\\r"),
            0x20..=0x7e => {
                if char::from(c) == quote {
                    s.push('\\');
                }
                s.push(char::from(c));
            }
            _ => {
                let _ = write!(s, "\\x{c:02x}");
            }
        }
    }
    s.push(quote);
    s
}

/// `GzipFile` opened for reading: `GzipFile(fileobj=f)` where `f` is the [`Source`] passed to
/// every call.
///
/// Use [`GzipFile::read`] as `GzipFile.read(n)`, or wrap it with [`GzipFile::source`] where a
/// [`Source`] is wanted (warcio's reader over the decompressed stream).
#[derive(Debug)]
pub struct GzipFile {
    buffer_size: usize,
    /// `BufferedReader.buffer` (allocated on the first fill).
    buf: Vec<u8>,
    pos: usize,
    /// `read_end`; `None` is CPython's -1 (no valid buffer).
    read_end: Option<usize>,
    /// `abs_pos`: the raw stream position `BufferedReader` believes in.
    abs_pos: u64,
    raw: GzipReader,
}

impl GzipFile {
    /// A `GzipFile` with the read path of the given CPython version family.
    ///
    /// # Panics
    /// If `params.buffer_size` or `params.read_size` is 0 (CPython refuses a 0 buffer size).
    pub fn new(params: GzipParams) -> Self {
        assert!(
            params.buffer_size > 0 && params.read_size > 0,
            "buffer sizes must be positive"
        );
        Self {
            buffer_size: params.buffer_size,
            buf: Vec::new(),
            pos: 0,
            read_end: None,
            abs_pos: 0,
            raw: GzipReader::new(params),
        }
    }

    /// This file as a [`Source`] over `raw` (borrowed for the call, as everywhere in `warc`).
    pub fn source<'a, S: Source + ?Sized>(&'a mut self, raw: &'a mut S) -> GzipSource<'a, S> {
        GzipSource { file: self, raw }
    }

    /// `GzipFile.mtime`: the MTIME field of the last member header read.
    pub fn mtime(&self) -> Option<u32> {
        self.raw.last_mtime
    }

    fn readahead(&self) -> usize {
        self.read_end.map_or(0, |end| end - self.pos)
    }

    /// `MINUS_LAST_BLOCK(self, size)`.
    fn minus_last_block(&self, size: usize) -> usize {
        if self.buffer_size.is_power_of_two() {
            size & !(self.buffer_size - 1)
        } else {
            self.buffer_size * (size / self.buffer_size)
        }
    }

    /// `GzipFile.read(n)`: append at most `n` bytes to `out`; nothing at the end of the data.
    ///
    /// On an error nothing is appended: the bytes this call had gathered are lost, as in
    /// CPython (`BufferedReader.read` discards its partial result).
    pub fn read<S: Source + ?Sized>(
        &mut self,
        src: &mut S,
        n: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let avail = self.readahead();
        if n <= avail {
            out.extend_from_slice(&self.buf[self.pos..self.pos + n]);
            self.pos += n;
            return Ok(());
        }
        let start = out.len();
        let r = self.read_generic(src, n, out);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    /// `_bufferedreader_read_generic`.
    fn read_generic<S: Source + ?Sized>(
        &mut self,
        src: &mut S,
        n: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let current = self.readahead();
        let mut remaining = n;
        if current > 0 {
            out.extend_from_slice(&self.buf[self.pos..self.pos + current]);
            remaining -= current;
            self.pos += current;
        }
        self.read_end = None;
        while remaining > 0 {
            let r = self.minus_last_block(remaining);
            if r == 0 {
                break;
            }
            let got = self.raw.read(
                src,
                &mut Out::Vec {
                    out: &mut *out,
                    limit: r,
                },
            )?;
            if got == 0 {
                return Ok(());
            }
            self.abs_pos += got as u64;
            remaining -= got;
        }
        self.pos = 0;
        self.read_end = Some(0);
        if self.buf.len() != self.buffer_size {
            self.buf.resize(self.buffer_size, 0);
        }
        while remaining > 0 {
            let start = self.read_end.unwrap_or(0);
            if start >= self.buffer_size {
                break;
            }
            let got = self
                .raw
                .read(src, &mut Out::Slice(&mut self.buf[start..]))?;
            if got == 0 {
                return Ok(());
            }
            self.abs_pos += got as u64;
            self.read_end = Some(start + got);
            let take = remaining.min(got);
            out.extend_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            remaining -= take;
        }
        Ok(())
    }

    /// `GzipFile.read()` / `read(-1)`: append everything up to the end of the data to `out`.
    ///
    /// On an error nothing is appended (CPython returns nothing either).
    pub fn read_all<S: Source + ?Sized>(
        &mut self,
        src: &mut S,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let start = out.len();
        let current = self.readahead();
        if current > 0 {
            out.extend_from_slice(&self.buf[self.pos..self.pos + current]);
            self.pos += current;
        }
        self.read_end = None;
        // `raw.readall()`: `while data := self.read(sys.maxsize)`. CPython's readall branch does
        // not advance `abs_pos`; neither does this.
        loop {
            match self.raw.read(
                src,
                &mut Out::Vec {
                    out: &mut *out,
                    limit: usize::MAX,
                },
            ) {
                Ok(0) => return Ok(()),
                Ok(_) => {}
                Err(e) => {
                    out.truncate(start);
                    return Err(e);
                }
            }
        }
    }

    /// `GzipFile.tell()`: `BufferedReader.seek(0, SEEK_CUR)` (never reads).
    pub fn tell(&mut self) -> u64 {
        let avail = self.readahead();
        if avail > 0 {
            self.abs_pos - avail as u64
        } else {
            // The raw seek path: `_GzipReader.seek(0, 1)` returns `_pos` and the buffer is reset.
            self.abs_pos = self.raw.pos;
            self.read_end = None;
            self.raw.pos
        }
    }
}

/// A [`GzipFile`] and its underlying file, as one [`Source`]: `read(n)` is `GzipFile.read(n)`,
/// and `read(usize::MAX)` is `GzipFile.read(-1)` (the `Source` convention).
#[derive(Debug)]
pub struct GzipSource<'a, S: Source + ?Sized> {
    pub file: &'a mut GzipFile,
    pub raw: &'a mut S,
}

impl<S: Source + ?Sized> Source for GzipSource<'_, S> {
    fn read(&mut self, n: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        if n == usize::MAX {
            self.file.read_all(self.raw, out)
        } else {
            self.file.read(self.raw, n, out)
        }
    }

    /// `GzipFile.tell()`: counts the bytes a failed `read` lost, as CPython does.
    fn tell(&mut self) -> Result<Option<i64>, Error> {
        Ok(Some(i64::try_from(self.file.tell()).unwrap_or(i64::MAX)))
    }
}

/// Where a `_GzipReader.read` call writes: `BufferedReader`'s own buffer, or the caller's result
/// (at most `limit` bytes appended; `usize::MAX` for `read(sys.maxsize)`).
enum Out<'a> {
    Slice(&'a mut [u8]),
    Vec { out: &'a mut Vec<u8>, limit: usize },
}

impl Out<'_> {
    fn size(&self) -> usize {
        match self {
            Out::Slice(s) => s.len(),
            Out::Vec { limit, .. } => *limit,
        }
    }
}

/// One inflate call over `input` into `out`: `(consumed, produced, stream_end)`, or zlib's
/// message (with CPython's fallback for a `Z_DATA_ERROR` without one).
fn inflate(
    z: &mut Decompress,
    input: &[u8],
    out: &mut [u8],
) -> Result<(usize, usize, bool), String> {
    let (in0, out0) = (z.total_in(), z.total_out());
    let r = z.decompress(input, out, FlushDecompress::Sync);
    let consumed = usize::try_from(z.total_in() - in0).unwrap_or(usize::MAX);
    let produced = usize::try_from(z.total_out() - out0).unwrap_or(usize::MAX);
    match r {
        Ok(status) => Ok((consumed, produced, status == Status::StreamEnd)),
        Err(e) => Err(e
            .message()
            .map_or("invalid input data", |m| m.trim_end_matches('\0'))
            .to_owned()),
    }
}

/// What zlib-rs reports when its fast path (`inflate_fast`) found the error: the dispatch loop
/// then sees the bad state and replaces the message, where zlib keeps it.
const BAD_STATE: &str = "repeated call with bad state";

/// The output space per call in [`slow_path_message`]: below zlib-rs's `INFLATE_FAST_MIN_LEFT`
/// (260), so the fast path never runs.
const SLOW_CHUNK: usize = 256;

/// The message zlib gives for the error in the raw deflate data `input` (decoded from its start
/// by zlib-rs's slow path, which keeps the message). `None` if no error shows.
fn slow_path_message(input: &[u8]) -> Option<String> {
    let mut z = Decompress::new(false);
    let mut out = [0u8; SLOW_CHUNK];
    let mut at = 0;
    loop {
        match inflate(&mut z, &input[at..], &mut out) {
            Ok((_, _, true) | (0, 0, false)) => return None,
            Ok((c, _, false)) => at += c,
            Err(m) => return (m != BAD_STATE).then_some(m),
        }
    }
}

/// CPython's `zlib.error` text for a decompression error.
fn zlib_message(msg: &str) -> String {
    format!("Error -3 while decompressing data: {msg:.200}")
}

/// `decompress(input, max_length)` into `out`: one inflate call with `max_length` of output space
/// (CPython splits large outputs over several inflate calls; zlib's result does not depend on
/// the split). Returns `(consumed, produced, stream_end)`; on an error `out` is as before.
fn inflate_out(
    z: &mut Decompress,
    input: &[u8],
    out: &mut Out,
) -> Result<(usize, usize, bool), String> {
    match out {
        Out::Slice(s) => inflate(z, input, s),
        Out::Vec { out, limit } => {
            let start = out.len();
            let mut consumed = 0;
            let mut chunk = FIRST_CHUNK;
            loop {
                let room = chunk.min(*limit - (out.len() - start));
                let w = out.len();
                out.resize(w + room, 0);
                match inflate(z, &input[consumed..], &mut out[w..]) {
                    Ok((c, p, end)) => {
                        out.truncate(w + p);
                        consumed += c;
                        let produced = out.len() - start;
                        if end || p < room || produced == *limit {
                            return Ok((consumed, produced, end));
                        }
                    }
                    Err(e) => {
                        out.truncate(start);
                        return Err(e);
                    }
                }
                chunk = (chunk * 2).min(MAX_CHUNK);
            }
        }
    }
}

/// The member's zlib decompressor: `_ZlibDecompressor` (3.12+) or `decompressobj` (3.11).
/// Input ranges index into the `_PaddedFile`'s data, where the bytes stay until consumed.
#[derive(Debug)]
struct Zlib {
    z: Decompress,
    eof: bool,
    /// `_ZlibDecompressor.needs_input`.
    needs_input: bool,
    /// Input the 3.12+ decompressor holds (`next_in`/`avail_in_real`).
    pending: Range<usize>,
    /// `unused_data` once the member's deflate stream ended, until prepended to the file.
    unused: Range<usize>,
    /// zlib's state after an error: every later call fails with the same message.
    error: Option<String>,
}

impl Zlib {
    fn new() -> Self {
        Self {
            z: Decompress::new(false),
            eof: false,
            needs_input: true,
            pending: 0..0,
            unused: 0..0,
            error: None,
        }
    }

    /// A fresh decompressor (`self._decompressor = self._decomp_factory(**self._decomp_args)`).
    fn reset(&mut self) {
        self.z.reset(false);
        self.eof = false;
        self.needs_input = true;
        self.pending = 0..0;
        self.unused = 0..0;
        self.error = None;
    }

    fn sticky(&self) -> Result<(), Error> {
        match &self.error {
            Some(msg) => Err(Error::Py(PyExc::ZlibError, msg.clone())),
            None => Ok(()),
        }
    }

    /// The first error of this member: zlib's message (recovered by a slow-path replay of the
    /// member's deflate data `data[start..end]` when zlib-rs replaced it), kept for later calls.
    fn fail(&mut self, msg: String, data: &[u8], start: Option<usize>, end: usize) -> Error {
        let msg = match start {
            Some(s) if msg == BAD_STATE => slow_path_message(&data[s..end]).unwrap_or(msg),
            _ => msg,
        };
        let msg = zlib_message(&msg);
        self.error = Some(msg.clone());
        Error::Py(PyExc::ZlibError, msg)
    }

    /// `_ZlibDecompressor.decompress(data, max_length)`, `data` being `new` (empty when the
    /// decompressor still holds input: `GzipFile` only reads when `needs_input`). `start` is
    /// where the member's deflate data begins in `data`, if still there.
    fn decompress_new(
        &mut self,
        data: &[u8],
        new: Range<usize>,
        out: &mut Out,
        start: Option<usize>,
    ) -> Result<usize, Error> {
        let input = if self.pending.is_empty() {
            new
        } else {
            self.pending.clone()
        };
        if let Err(e) = self.sticky() {
            // `decompress()` drops its input on an error; `needs_input` keeps its value.
            self.pending = 0..0;
            return Err(e);
        }
        match inflate_out(&mut self.z, &data[input.clone()], out) {
            Ok((consumed, produced, end)) => {
                let rest = input.start + consumed..input.end;
                if end {
                    self.eof = true;
                    self.needs_input = false;
                    self.pending = 0..0;
                    if !rest.is_empty() {
                        self.unused = rest;
                    }
                } else if rest.is_empty() {
                    self.pending = 0..0;
                    self.needs_input = true;
                } else {
                    self.needs_input = false;
                    self.pending = rest;
                }
                Ok(produced)
            }
            Err(msg) => {
                self.pending = 0..0;
                Err(self.fail(msg, data, start, input.end))
            }
        }
    }

    /// `decompressobj.decompress(data, max_length)`: returns the output size and the input
    /// `GzipFile` prepends back to the file (`unconsumed_tail`, else `unused_data`).
    fn decompress_old(
        &mut self,
        data: &[u8],
        input: Range<usize>,
        out: &mut Out,
        start: Option<usize>,
    ) -> Result<(usize, Range<usize>), Error> {
        self.sticky()?;
        match inflate_out(&mut self.z, &data[input.clone()], out) {
            Ok((consumed, produced, end)) => {
                if end {
                    self.eof = true;
                }
                Ok((produced, input.start + consumed..input.end))
            }
            Err(msg) => Err(self.fail(msg, data, start, input.end)),
        }
    }
}

/// How much of a member's compressed data `PaddedFile` keeps for a slow-path replay.
const REPLAY_LIMIT: usize = 64 * 1024 * 1024;

/// `gzip._PaddedFile` over the underlying file. Bytes read from the file are appended to `data`
/// and stay there while anything may still read them (the prepended buffer, the decompressor's
/// input, the current member's deflate data up to [`REPLAY_LIMIT`]); `data` is compacted
/// instead of grown, so nothing is copied per member.
#[derive(Debug)]
struct PaddedFile {
    data: Vec<u8>,
    /// `_buffer` is `data[b0..b0 + blen]`; it always ends at `data.len()` when set.
    b0: usize,
    blen: usize,
    /// `_read`; `None` once the buffer is used up.
    rd: Option<usize>,
    /// Where the current member's deflate data starts in `data`, while it is kept.
    pin: Option<usize>,
    capacity: usize,
}

impl PaddedFile {
    fn new(read_size: usize) -> Self {
        Self {
            data: Vec::new(),
            b0: 0,
            blen: 0,
            rd: Some(0),
            pin: None,
            capacity: read_size.max(8192).saturating_mul(8),
        }
    }

    /// The index in `data` of the next byte `read` returns.
    fn position(&self) -> usize {
        self.rd.map_or(self.data.len(), |r| self.b0 + r)
    }

    /// `_PaddedFile.read(size)`: the result as a range of `data`.
    fn read<S: Source + ?Sized>(
        &mut self,
        src: &mut S,
        size: usize,
    ) -> Result<Range<usize>, Error> {
        match self.rd {
            Some(r) if r + size <= self.blen => {
                self.rd = Some(r + size);
                Ok(self.b0 + r..self.b0 + r + size)
            }
            Some(r) => {
                // `self._buffer[read:] + self.file.read(size - self._length + read)`; `_read` is
                // None before the file read, so a failing read loses the buffer's rest.
                self.rd = None;
                let start = self.append(src, size - (self.blen - r), self.b0 + r)?;
                Ok(start..self.data.len())
            }
            None => {
                let start = self.append(src, size, self.data.len())?;
                Ok(start..self.data.len())
            }
        }
    }

    /// Read `want` bytes from the file onto the end of `data`, keeping `data[keep..]` (and the
    /// pinned member data); returns the new index of `keep`.
    fn append<S: Source + ?Sized>(
        &mut self,
        src: &mut S,
        want: usize,
        keep: usize,
    ) -> Result<usize, Error> {
        debug_assert!(self.rd.is_none() && keep <= self.data.len());
        if let Some(p) = self.pin {
            if self.data.len() + want - p > REPLAY_LIMIT {
                self.pin = None;
            }
        }
        let retain = self.pin.map_or(keep, |p| p.min(keep));
        if self.data.capacity() < self.capacity {
            self.data.reserve(self.capacity - self.data.len());
        }
        let mut keep = keep;
        if retain > 0 && self.data.len() + want > self.data.capacity() {
            self.data.copy_within(retain.., 0);
            self.data.truncate(self.data.len() - retain);
            keep -= retain;
            self.pin = self.pin.map(|p| p - retain);
        }
        let before = self.data.len();
        if let Err(e) = src.read(want, &mut self.data) {
            self.data.truncate(before);
            return Err(e);
        }
        Ok(keep)
    }

    /// `_PaddedFile.prepend(range)`: `range` is always a suffix of the last read's result.
    fn prepend(&mut self, range: Range<usize>) {
        match self.rd {
            None => {
                self.b0 = range.start;
                self.blen = range.len();
                self.rd = Some(0);
            }
            Some(r) => self.rd = Some(r.saturating_sub(range.len())),
        }
    }
}

/// `gzip._GzipReader` (with `_compression.DecompressReader`'s fields).
#[derive(Debug)]
struct GzipReader {
    read_size: usize,
    zlib_decompressor: bool,
    check_header_crc: bool,
    fp: PaddedFile,
    dec: Zlib,
    new_member: bool,
    last_mtime: Option<u32>,
    crc: Crc,
    stream_size: u64,
    /// `_pos`: bytes produced so far.
    pos: u64,
}

impl GzipReader {
    fn new(params: GzipParams) -> Self {
        Self {
            read_size: params.read_size,
            zlib_decompressor: params.zlib_decompressor,
            check_header_crc: params.check_header_crc,
            fp: PaddedFile::new(params.read_size),
            dec: Zlib::new(),
            new_member: true,
            last_mtime: None,
            crc: Crc::new(),
            stream_size: 0,
            pos: 0,
        }
    }

    /// `_GzipReader.read(size)` (`readinto`), `size` being `out.size()`: the number of bytes
    /// written at the start of the slice or appended to the `Vec`.
    fn read<S: Source + ?Sized>(&mut self, src: &mut S, out: &mut Out) -> Result<usize, Error> {
        if out.size() == 0 {
            return Ok(0);
        }
        let n = loop {
            if self.dec.eof {
                self.read_eof(src)?;
                self.new_member = true;
                self.dec.reset();
            }
            if self.new_member {
                self.crc.reset();
                self.stream_size = 0;
                if !self.read_gzip_header(src)? {
                    return Ok(0);
                }
                self.new_member = false;
                self.fp.pin = Some(self.fp.position());
            }
            let (n, buf_empty) = if self.zlib_decompressor {
                let (n, buf_empty) = if self.dec.needs_input {
                    let buf = self.fp.read(src, self.read_size)?;
                    let empty = buf.is_empty();
                    (
                        self.dec
                            .decompress_new(&self.fp.data, buf, out, self.fp.pin)?,
                        empty,
                    )
                } else {
                    (
                        self.dec
                            .decompress_new(&self.fp.data, 0..0, out, self.fp.pin)?,
                        true,
                    )
                };
                if !self.dec.unused.is_empty() {
                    let unused = std::mem::replace(&mut self.dec.unused, 0..0);
                    self.fp.prepend(unused);
                }
                (n, buf_empty)
            } else {
                let buf = self.fp.read(src, self.read_size)?;
                let empty = buf.is_empty();
                let (n, rest) = self
                    .dec
                    .decompress_old(&self.fp.data, buf, out, self.fp.pin)?;
                if !rest.is_empty() {
                    self.fp.prepend(rest);
                }
                (n, empty)
            };
            if n > 0 {
                break n;
            }
            if buf_empty {
                return Err(eof_error());
            }
        };
        match out {
            Out::Slice(s) => self.crc.update(&s[..n]),
            Out::Vec { out, .. } => self.crc.update(&out[out.len() - n..]),
        }
        self.stream_size += n as u64;
        self.pos += n as u64;
        Ok(n)
    }

    /// `_read_exact(fp, n)`, handing each piece to `sink`.
    fn read_exact<S: Source + ?Sized>(
        &mut self,
        src: &mut S,
        n: usize,
        sink: &mut dyn FnMut(&[u8]),
    ) -> Result<(), Error> {
        let r = self.fp.read(src, n)?;
        let mut got = r.len();
        sink(&self.fp.data[r]);
        while got < n {
            let r = self.fp.read(src, n - got)?;
            if r.is_empty() {
                return Err(eof_error());
            }
            got += r.len();
            sink(&self.fp.data[r]);
        }
        Ok(())
    }

    /// `_read_exact` into a small array.
    fn read_array<S: Source + ?Sized, const N: usize>(
        &mut self,
        src: &mut S,
    ) -> Result<[u8; N], Error> {
        let mut a = [0u8; N];
        let mut at = 0;
        self.read_exact(src, N, &mut |b| {
            let k = b.len().min(N - at);
            a[at..at + k].copy_from_slice(&b[..k]);
            at += k;
        })?;
        Ok(a)
    }

    /// `_read_until_null(fp, crc)`.
    fn read_until_null<S: Source + ?Sized>(
        &mut self,
        src: &mut S,
        crc: &mut Option<Crc>,
    ) -> Result<(), Error> {
        loop {
            let r = self.fp.read(src, 1)?;
            let s = &self.fp.data[r];
            if let Some(c) = crc.as_mut() {
                c.update(s);
            }
            if s.is_empty() || s == b"\0" {
                return Ok(());
            }
        }
    }

    /// `_read_gzip_header(fp)`: false at the end of the file.
    fn read_gzip_header<S: Source + ?Sized>(&mut self, src: &mut S) -> Result<bool, Error> {
        let r = self.fp.read(src, 2)?;
        let magic = &self.fp.data[r];
        if magic.is_empty() {
            return Ok(false);
        }
        if magic != b"\x1f\x8b" {
            return Err(bad_gzip(format!(
                "Not a gzipped file ({})",
                py_bytes_repr(magic)
            )));
        }
        let base: [u8; 8] = self.read_array(src)?;
        let (method, flag) = (base[0], base[1]);
        let mtime = u32::from_le_bytes([base[2], base[3], base[4], base[5]]);
        if method != 8 {
            return Err(bad_gzip("Unknown compression method".to_owned()));
        }
        let mut crc = (self.check_header_crc && flag & FHCRC != 0).then(|| {
            let mut c = Crc::new();
            c.update(b"\x1f\x8b");
            c.update(&base);
            c
        });
        if flag & FEXTRA != 0 {
            let len: [u8; 2] = self.read_array(src)?;
            if let Some(c) = crc.as_mut() {
                c.update(&len);
            }
            let mut c2 = crc.take();
            self.read_exact(src, usize::from(u16::from_le_bytes(len)), &mut |b| {
                if let Some(c) = c2.as_mut() {
                    c.update(b);
                }
            })?;
            crc = c2;
        }
        if flag & FNAME != 0 {
            self.read_until_null(src, &mut crc)?;
        }
        if flag & FCOMMENT != 0 {
            self.read_until_null(src, &mut crc)?;
        }
        if flag & FHCRC != 0 {
            let stored = u16::from_le_bytes(self.read_array(src)?);
            if let Some(c) = crc {
                let computed = c.sum() & 0xffff;
                if u32::from(stored) != computed {
                    return Err(bad_gzip(format!(
                        "Corrupted gzip header. Checksums do not match: {computed:04x} != {stored:04x}"
                    )));
                }
            }
        }
        self.last_mtime = Some(mtime);
        Ok(true)
    }

    /// `_read_eof`: the member trailer, then zero padding.
    fn read_eof<S: Source + ?Sized>(&mut self, src: &mut S) -> Result<(), Error> {
        let t: [u8; 8] = self.read_array(src)?;
        let crc32 = u32::from_le_bytes([t[0], t[1], t[2], t[3]]);
        let isize = u32::from_le_bytes([t[4], t[5], t[6], t[7]]);
        let crc = self.crc.sum();
        if crc32 != crc {
            return Err(bad_gzip(format!("CRC check failed {crc32:#x} != {crc:#x}")));
        }
        if u64::from(isize) != self.stream_size & 0xffff_ffff {
            return Err(bad_gzip("Incorrect length of data produced".to_owned()));
        }
        loop {
            let r = self.fp.read(src, 1)?;
            if self.fp.data[r.clone()] == *b"\0" {
                continue;
            }
            if !r.is_empty() {
                self.fp.prepend(r);
            }
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_repr() {
        assert_eq!(py_bytes_repr(b"PK"), "b'PK'");
        assert_eq!(py_bytes_repr(b"\x00\x01"), "b'\\x00\\x01'");
        assert_eq!(py_bytes_repr(b"'"), "b\"'\"");
        assert_eq!(py_bytes_repr(b"'\""), "b'\\'\"'");
        assert_eq!(py_bytes_repr(b"\\\n"), "b'\\\\\\n'");
        assert_eq!(py_bytes_repr(b"\x1f\x7f"), "b'\\x1f\\x7f'");
    }

    #[test]
    fn params() {
        assert_eq!(GzipParams::for_python(10), None);
        assert_eq!(GzipParams::for_python(13), Some(GzipParams::PY312));
        assert_eq!(GzipParams::for_python(16), Some(GzipParams::PY315));
    }

    #[test]
    fn source_adapter() {
        // One member: a final stored block holding "hello", then CRC-32 and length.
        let mut gz = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3, 1, 5, 0, 0xfa, 0xff];
        gz.extend_from_slice(b"hello");
        let mut crc = Crc::new();
        crc.update(b"hello");
        gz.extend_from_slice(&crc.sum().to_le_bytes());
        gz.extend_from_slice(&5u32.to_le_bytes());
        let mut raw = crate::warc::ReadSource(&gz[..]);
        let mut f = GzipFile::new(GzipParams::PY312);
        let mut out = Vec::new();
        f.source(&mut raw).read(2, &mut out).unwrap();
        assert_eq!(out, b"he");
        f.source(&mut raw).read(usize::MAX, &mut out).unwrap();
        assert_eq!(out, b"hello");
        assert_eq!((f.tell(), f.mtime()), (5, Some(0)));
        f.source(&mut raw).read(10, &mut out).unwrap();
        assert_eq!(out, b"hello");
    }
}
