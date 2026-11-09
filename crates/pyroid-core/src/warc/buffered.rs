// Ports warcio 1.8.2's bufferedreaders.py and limitreader.py (Apache-2.0).
//! `BufferedReader` (and so `DecompressingBufferedReader`), `ChunkedDataReader` and
//! `LimitReader`, statement by statement where the order of reads matters: the same
//! `read(n)`/`readline(n)` calls reach the inner stream at the same moments, with the same
//! sizes, so a source that fails part-way fails at the same point, and every quirk of the
//! originals is kept (the cumulative length in `readline`'s loop, the first-block
//! decompression fallback, "can't read past next member").
//!
//! The buffer is one reused `Vec` per reader instead of a `BytesIO` per block; a `buff` of
//! `None` and an exhausted one behave the same in warcio, so both are `pos == size`.

use super::zlib::{self, DecompressObj};
use super::{Error, Message, PyExc, Source};

/// warcio's `BUFF_SIZE`.
pub const BUFF_SIZE: usize = 16384;

/// A `Content-Encoding` warcio can decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecompType {
    /// `gzip` (and `x-gzip`): `zlib.decompressobj(16 + MAX_WBITS)`.
    Gzip,
    /// `deflate`: a zlib stream, retried as raw deflate if the first block fails.
    Deflate,
    /// `deflate_alt`: raw deflate.
    DeflateAlt,
    /// `br`: brotlicffi's `Decompressor`, when one is configured.
    Brotli,
}

impl DecompType {
    /// The key in `BufferedReader.DECOMPRESSORS`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gzip => "gzip",
            Self::Deflate => "deflate",
            Self::DeflateAlt => "deflate_alt",
            Self::Brotli => "br",
        }
    }
}

/// A decompressor object other than zlib's (brotlicffi's for `br`), as warcio uses it.
pub trait Decompressor: Send {
    /// `decompress(data)`: append all the output to `out`, or fail with the exception it
    /// raises ([`Error::Py`]; warcio writes `str(e)` to stderr).
    fn decompress(&mut self, data: &[u8], out: &mut Vec<u8>) -> Result<(), Error>;
    /// `unused_data` (`None` or empty is falsy; warcio sets brotlicffi's to `None`).
    fn unused_data(&self) -> &[u8] {
        &[]
    }
    /// Take `unused_data` for the next member.
    fn take_unused_data(&mut self) -> Vec<u8> {
        Vec::new()
    }
    /// `flush()` (its output is dropped; brotlicffi's returns `b''`).
    fn flush(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

/// Makes a new `br` decompressor (warcio's `brotli_decompressor`).
pub type DecompressorFactory = fn() -> Box<dyn Decompressor>;

pub(crate) enum Decomp {
    Zlib(DecompressObj),
    Other(Box<dyn Decompressor>),
}

impl Decomp {
    fn unused_len(&self) -> usize {
        match self {
            Self::Zlib(d) => d.unused_data().len(),
            Self::Other(d) => d.unused_data().len(),
        }
    }

    fn take_unused(&mut self) -> Vec<u8> {
        match self {
            Self::Zlib(d) => d.take_unused_data(),
            Self::Other(d) => d.take_unused_data(),
        }
    }

    fn flush(&mut self, zlib_raises: bool) -> Result<(), Error> {
        match self {
            Self::Zlib(d) => d.flush(zlib_raises),
            Self::Other(d) => d.flush(),
        }
    }
}

impl std::fmt::Debug for Decomp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Zlib(d) => d.fmt(f),
            Self::Other(_) => f.write_str("Decompressor"),
        }
    }
}

/// What a reader reads from: a source, or a record's raw stream.
pub(crate) trait Inner {
    /// `self.stream.read(n)` (`None` = `read()`; negative sizes are Python's).
    fn read(&mut self, n: Option<i128>, out: &mut Vec<u8>) -> Result<(), Error>;
    /// `self.stream.readline(n)`.
    fn readline(&mut self, n: Option<i128>, out: &mut Vec<u8>) -> Result<(), Error>;
    /// `sys.stderr.write(text)`.
    fn stderr(&mut self, text: String);
}

/// The Python file object under the iterator: reads are counted for `fh.tell()`.
pub(crate) struct SourceInner<'a> {
    pub src: &'a mut dyn Source,
    pub tell: &'a mut i64,
    pub msgs: &'a mut Vec<Message>,
}

impl SourceInner<'_> {
    fn read_src(&mut self, n: Option<i128>, out: &mut Vec<u8>) -> Result<(), Error> {
        // `read(-1)` / `read()` read everything: `usize::MAX` stands for them.
        let n = n.map_or(usize::MAX, |n| usize::try_from(n).unwrap_or(usize::MAX));
        let start = out.len();
        self.src.read(n, out)?;
        if out.len() - start > n {
            out.truncate(start + n);
        }
        *self.tell += i64::try_from(out.len() - start).expect("read size");
        Ok(())
    }
}

impl Inner for SourceInner<'_> {
    fn read(&mut self, n: Option<i128>, out: &mut Vec<u8>) -> Result<(), Error> {
        self.read_src(n, out)
    }

    fn readline(&mut self, _n: Option<i128>, _out: &mut Vec<u8>) -> Result<(), Error> {
        unreachable!("warcio never calls readline on the file object")
    }

    fn stderr(&mut self, text: String) {
        self.msgs.push(Message::Stderr(text));
    }
}

/// The `BufferedReader` state.
#[derive(Debug)]
pub(crate) struct BufCore {
    pub block_size: Option<i128>,
    pub decomp_type: Option<DecompType>,
    pub decomp: Option<Decomp>,
    brotli: Option<DecompressorFactory>,
    /// `buff`: `buf[pos..size]` is unread; `buf[size..]` is zeroed slack for decompression.
    pub buf: Vec<u8>,
    pub size: usize,
    pub pos: usize,
    pub starting_data: Option<Vec<u8>>,
    pub num_read: i64,
    pub num_block_read: i64,
    /// Raw input of the current block, when decompressing.
    raw: Vec<u8>,
    /// `self.stream` is None (after `close()`).
    pub stream_closed: bool,
    /// A `ChunkedDataReader`.
    pub chunked: Option<Chunked>,
    /// zlib's `flush()` raises on a broken stream (CPython 3.13 and later).
    pub zlib_flush_raises: bool,
}

impl BufCore {
    pub fn new(
        block_size: Option<i128>,
        decomp_type: Option<DecompType>,
        brotli: Option<DecompressorFactory>,
    ) -> Self {
        let mut c = Self {
            block_size,
            decomp_type: None,
            decomp: None,
            brotli,
            buf: Vec::new(),
            size: 0,
            pos: 0,
            starting_data: None,
            num_read: 0,
            num_block_read: 0,
            raw: Vec::new(),
            stream_closed: false,
            chunked: None,
            zlib_flush_raises: false,
        };
        c.init_decomp(decomp_type);
        c
    }

    /// `_init_decomp(decomp_type)`.
    pub fn init_decomp(&mut self, decomp_type: Option<DecompType>) {
        self.num_block_read = 0;
        self.decomp_type = decomp_type;
        self.decomp = decomp_type.and_then(|t| match t {
            DecompType::Gzip => Some(Decomp::Zlib(DecompressObj::new(zlib::WBITS_GZIP))),
            DecompType::Deflate => Some(Decomp::Zlib(DecompressObj::new(zlib::WBITS_ZLIB))),
            DecompType::DeflateAlt => Some(Decomp::Zlib(DecompressObj::new(zlib::WBITS_RAW))),
            // Only reachable with a factory: `content_stream` checks the supported names.
            DecompType::Brotli => self.brotli.map(|f| Decomp::Other(f())),
        });
    }

    /// `empty()` (`read_all_members` is never set by warcio's readers).
    #[inline]
    pub fn empty(&self) -> bool {
        self.pos >= self.size
    }

    /// `rem_length()`.
    #[inline]
    pub fn rem_length(&self) -> usize {
        (self.size - self.pos) + self.decomp.as_ref().map_or(0, Decomp::unused_len)
    }

    /// `tell()`.
    pub fn tell(&self) -> i64 {
        self.num_read
    }

    /// `read_next_member()`.
    pub fn read_next_member(&mut self) -> bool {
        let Some(d) = self.decomp.as_mut() else {
            return false;
        };
        if d.unused_len() == 0 {
            return false;
        }
        self.starting_data = Some(d.take_unused());
        self.init_decomp(self.decomp_type);
        true
    }

    /// `close_decompressor()`: `flush()` (whose output is dropped), then no decompressor.
    /// If `flush()` raises, the decompressor stays.
    pub fn close_decompressor(&mut self) -> Result<(), Error> {
        if let Some(d) = self.decomp.as_mut() {
            d.flush(self.zlib_flush_raises)?;
        }
        self.decomp = None;
        Ok(())
    }

    /// The part of `close()` after `self.stream.close()`.
    pub fn close(&mut self) -> Result<(), Error> {
        self.stream_closed = true;
        self.size = 0;
        self.pos = 0;
        self.close_decompressor()
    }

    fn set_none(&mut self) {
        self.size = 0;
        self.pos = 0;
    }

    /// `_fillbuff()`: `ChunkedDataReader`'s when de-chunking, else `BufferedReader`'s.
    pub fn fillbuff<I: Inner + ?Sized>(&mut self, inner: &mut I) -> Result<(), Error> {
        match self.chunked.take() {
            None => self.base_fillbuff(inner),
            Some(mut ch) => {
                let r = ch.fillbuff(self, inner);
                self.chunked = Some(ch);
                r
            }
        }
    }

    /// `BufferedReader._fillbuff()`.
    pub fn base_fillbuff<I: Inner + ?Sized>(&mut self, inner: &mut I) -> Result<(), Error> {
        if !self.empty() {
            return Ok(());
        }
        // can't read past next member
        if self.rem_length() > 0 {
            return Ok(());
        }
        let mut got_data = if let Some(d) = self.starting_data.take().filter(|d| !d.is_empty()) {
            self.raw = d;
            self.process_raw(inner)?
        } else {
            self.read_block(inner)?
        };
        // if raw data is not empty and decompressor set, but decompressed buff is empty,
        // keep reading -- decompressor likely needs more data to decompress
        while got_data && self.decomp.as_ref().is_some_and(|d| d.unused_len() == 0) && self.empty()
        {
            got_data = self.read_block(inner)?;
        }
        Ok(())
    }

    /// `data = self.stream.read(block_size); self._process_read(data)`; whether `data` was
    /// non-empty.
    fn read_block<I: Inner + ?Sized>(&mut self, inner: &mut I) -> Result<bool, Error> {
        if self.stream_closed {
            return Err(Error::Py(
                PyExc::AttributeError,
                "'NoneType' object has no attribute 'read'".into(),
            ));
        }
        if self.decomp.is_none() {
            // Pass-through: read straight into the buffer.
            self.buf.clear();
            if let Err(e) = inner.read(self.block_size, &mut self.buf) {
                self.buf.clear();
                self.set_none();
                return Err(e);
            }
            if self.buf.is_empty() {
                self.set_none();
                return Ok(false);
            }
            self.size = self.buf.len();
            self.pos = 0;
            self.count(self.size);
            return Ok(true);
        }
        let mut raw = std::mem::take(&mut self.raw);
        raw.clear();
        let r = inner.read(self.block_size, &mut raw);
        self.raw = raw;
        r?;
        self.process_raw(inner)
    }

    fn count(&mut self, n: usize) {
        let n = i64::try_from(n).expect("block size");
        self.num_read += n;
        self.num_block_read += n;
    }

    /// `_process_read(self.raw)`; whether the raw data was non-empty.
    fn process_raw<I: Inner + ?Sized>(&mut self, inner: &mut I) -> Result<bool, Error> {
        if self.raw.is_empty() {
            self.set_none();
            return Ok(false);
        }
        let mut end = 0usize;
        self.decompress(inner, &mut end)?;
        self.size = end;
        self.pos = 0;
        self.count(end);
        Ok(true)
    }

    /// `_decompress(self.raw)` into `buf[..*end]`.
    fn decompress<I: Inner + ?Sized>(
        &mut self,
        inner: &mut I,
        end: &mut usize,
    ) -> Result<(), Error> {
        let res = match self.decomp.as_mut() {
            None => {
                self.passthrough(end);
                return Ok(());
            }
            Some(Decomp::Zlib(d)) => d.decompress_into(&self.raw, &mut self.buf, end),
            Some(Decomp::Other(d)) => {
                self.buf.clear();
                let r = d.decompress(&self.raw, &mut self.buf);
                if r.is_ok() {
                    *end = self.buf.len();
                } else {
                    self.buf.clear();
                }
                r
            }
        };
        let Err(e) = res else {
            return Ok(());
        };
        if self.num_block_read == 0 {
            // if first read attempt, assume non-gzipped stream
            if self.decomp_type == Some(DecompType::Deflate) {
                self.init_decomp(Some(DecompType::DeflateAlt));
                return self.decompress(inner, end);
            }
            self.decomp = None;
            self.passthrough(end);
            return Ok(());
        }
        // otherwise (partly decompressed), something is wrong
        let msg = match &e {
            Error::Py(_, m) => m.clone(),
            Error::Source(_) => String::new(),
        };
        inner.stderr(msg + "\n");
        *end = 0;
        Ok(())
    }

    fn passthrough(&mut self, end: &mut usize) {
        self.buf.clear();
        self.buf.extend_from_slice(&self.raw);
        *end = self.buf.len();
    }

    /// `self.buff.read(length)`'s size: what is left, at most `length`. A length that does
    /// not fit `Py_ssize_t` makes `BytesIO` raise.
    #[inline]
    fn take_len(&self, length: Option<i128>) -> Result<usize, Error> {
        let avail = self.size - self.pos;
        Ok(match length {
            None => avail,
            Some(l) => avail.min(usize::try_from(ssize(l)?).unwrap_or(usize::MAX)),
        })
    }

    /// `read(length)`, appending to `out`.
    pub fn read<I: Inner + ?Sized>(
        &mut self,
        inner: &mut I,
        mut length: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        while length.is_none_or(|l| l > 0) {
            self.fillbuff(inner)?;
            if self.empty() {
                break;
            }
            let take = self.take_len(length)?;
            out.extend_from_slice(&self.buf[self.pos..self.pos + take]);
            self.pos += take;
            if let Some(l) = length.as_mut() {
                *l -= i128::try_from(take).expect("read size");
            }
        }
        Ok(())
    }

    /// `read(length)` without keeping the bytes; how many were read.
    pub fn skip<I: Inner + ?Sized>(
        &mut self,
        inner: &mut I,
        mut length: Option<i128>,
    ) -> Result<usize, Error> {
        let mut n = 0;
        while length.is_none_or(|l| l > 0) {
            self.fillbuff(inner)?;
            if self.empty() {
                break;
            }
            let take = self.take_len(length)?;
            self.pos += take;
            n += take;
            if let Some(l) = length.as_mut() {
                *l -= i128::try_from(take).expect("read size");
            }
        }
        Ok(n)
    }

    /// `self.buff.readline(length)` appended to `out`.
    #[inline]
    fn buff_readline(&mut self, length: Option<i128>, out: &mut Vec<u8>) -> Result<(), Error> {
        let avail = &self.buf[self.pos..self.size];
        let cap = match length {
            Some(l) if ssize(l)? >= 0 => avail.len().min(usize::try_from(l).unwrap_or(usize::MAX)),
            _ => avail.len(),
        };
        let n = memchr::memchr(b'\n', &avail[..cap]).map_or(cap, |i| i + 1);
        out.extend_from_slice(&avail[..n]);
        self.pos += n;
        Ok(())
    }

    /// `readline(length)`, appending the line to `out`.
    pub fn readline<I: Inner + ?Sized>(
        &mut self,
        inner: &mut I,
        mut length: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        if length == Some(0) {
            return Ok(());
        }
        self.fillbuff(inner)?;
        if self.empty() {
            return Ok(());
        }
        let start = out.len();
        // Never empty: the buffer has data and `length` is not 0.
        self.buff_readline(length, out)?;
        // we may be at a boundary
        while out.last() != Some(&b'\n') {
            if let Some(l) = length.as_mut().filter(|l| **l != 0) {
                // warcio subtracts the length of the whole line so far, each time
                *l -= i128::try_from(out.len() - start).expect("line length");
                if *l <= 0 {
                    break;
                }
            }
            self.fillbuff(inner)?;
            if self.empty() {
                break;
            }
            self.buff_readline(length, out)?;
        }
        Ok(())
    }
}

/// A Python int passed where `Py_ssize_t` is needed (`BytesIO.read(n)`).
fn ssize(l: i128) -> Result<i64, Error> {
    i64::try_from(l).map_err(|_| {
        Error::Py(
            PyExc::OverflowError,
            "cannot fit 'int' into an index-sized integer".into(),
        )
    })
}

/// `ChunkedDataReader`'s own state (`raise_exceptions` is never set by warcio).
#[derive(Debug, Default)]
pub(crate) struct Chunked {
    pub all_chunks_read: bool,
    pub not_chunked: bool,
}

/// `ChunkedDataException(msg, data)`.
struct ChunkedDataException {
    data: Vec<u8>,
}

impl Chunked {
    /// `ChunkedDataReader._fillbuff()`.
    pub fn fillbuff<I: Inner + ?Sized>(
        &mut self,
        core: &mut BufCore,
        inner: &mut I,
    ) -> Result<(), Error> {
        if self.not_chunked {
            return core.base_fillbuff(inner);
        }
        while core.empty() && !self.all_chunks_read && !self.not_chunked {
            if core.stream_closed {
                return Err(Error::Py(
                    PyExc::AttributeError,
                    "'NoneType' object has no attribute 'readline'".into(),
                ));
            }
            let mut length_header = Vec::new();
            inner.readline(Some(64), &mut length_header)?;
            match self.try_decode(core, inner, &length_header)? {
                Ok(()) => {}
                Err(ChunkedDataException { data }) => {
                    // Can't parse the data as chunked; treat it as non-chunk encoded from
                    // here on.
                    length_header.extend_from_slice(&data);
                    core.raw = length_header;
                    core.process_raw(inner)?;
                    self.not_chunked = true;
                    // parse as block as non-chunked
                    return core.base_fillbuff(inner);
                }
            }
        }
        Ok(())
    }

    /// `_try_decode(length_header)`; the inner result is a `ChunkedDataException`.
    fn try_decode<I: Inner + ?Sized>(
        &mut self,
        core: &mut BufCore,
        inner: &mut I,
        length_header: &[u8],
    ) -> Result<Result<(), ChunkedDataException>, Error> {
        let bad = || ChunkedDataException { data: Vec::new() };
        // ensure line ends with \r\n
        let Some(body) = length_header.strip_suffix(b"\r\n") else {
            return Ok(Err(bad()));
        };
        let size_field = body.split(|&b| b == b';').next().unwrap_or(b"");
        let Some(chunk_size) = super::pystr::py_int_hex(size_field) else {
            return Ok(Err(bad()));
        };
        // sanity check chunk size
        if chunk_size > 1 << 31 {
            return Ok(Err(bad()));
        }
        if chunk_size == 0 {
            // chunk_size 0 indicates end of file. read final bytes to compute digest.
            let mut final_data = Vec::new();
            inner.read(Some(2), &mut final_data)?;
            if final_data != b"\r\n" {
                return Ok(Err(bad()));
            }
            self.all_chunks_read = true;
            core.raw.clear();
            core.process_raw(inner)?;
            return Ok(Ok(()));
        }
        let mut chunk_size = chunk_size;
        let mut data: Vec<u8> = Vec::new();
        // read chunk
        while i128::try_from(data.len()).expect("len") < chunk_size {
            let before = data.len();
            inner.read(
                Some(chunk_size - i128::try_from(before).expect("len")),
                &mut data,
            )?;
            if data.len() == before {
                // assume the file was cut off
                chunk_size = i128::try_from(before).expect("len");
                self.all_chunks_read = true;
            }
        }
        // if we successfully read a block without running out, it should end in \r\n
        if !self.all_chunks_read {
            let mut clrf = Vec::new();
            inner.read(Some(2), &mut clrf)?;
            if clrf != b"\r\n" {
                return Ok(Err(ChunkedDataException { data }));
            }
        }
        // hand to base class for further processing
        core.raw = data;
        core.process_raw(inner)?;
        Ok(Ok(()))
    }
}

/// The `limit` of a `LimitReader` (`orig - limit` is `tell()`); values of 10^36 and more are
/// saturated (no stream reaches them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Limit {
    pub limit: i128,
    pub orig: i128,
}

impl Limit {
    /// `min(length, self.limit)` (or `self.limit`).
    pub fn clamp(&self, length: Option<i128>) -> i128 {
        match length {
            Some(n) => n.min(self.limit),
            None => self.limit,
        }
    }
}
