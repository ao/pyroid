// Ports warcio 1.8.2's LimitReader, DigestVerifyingReader and the readers that
// ArcWarcRecord.content_stream() returns (Apache-2.0).
//! The streams of a record: the iterator's `DecompressingBufferedReader` over the file
//! ([`TopReader`]), a record's `raw_stream` ([`RawStream`]: a `LimitReader`, a
//! `DigestVerifyingReader`, or the top reader itself when there is no length), and the
//! decoding readers of `content_stream()` ([`ContentStream`]).

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use super::buffered::{
    BUFF_SIZE, BufCore, Chunked, DecompType, DecompressorFactory, Inner, Limit, SourceInner,
};
use super::digest::{DigestChecker, Verifier};
use super::{Error, Message, Source};

/// A record's `raw_stream`, shared by the record, the iterator (`read_to_end`) and every
/// decoder `content_stream()` made over it.
pub type RawHandle = Arc<Mutex<RawStream>>;

pub(crate) fn lock(h: &RawHandle) -> MutexGuard<'_, RawStream> {
    h.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The iterator's reader over the file: `DecompressingBufferedReader(fh, block_size)`.
#[derive(Debug)]
pub(crate) struct TopReader {
    pub core: BufCore,
    /// `fh.tell()`: the initial position plus every byte the source returned.
    pub tell: i64,
}

impl TopReader {
    pub fn new(block_size: i64, tell: i64, zlib_flush_raises: bool) -> Self {
        let mut core = BufCore::new(Some(i128::from(block_size)), Some(DecompType::Gzip), None);
        core.zlib_flush_raises = zlib_flush_raises;
        Self { core, tell }
    }

    fn inner<'a>(
        tell: &'a mut i64,
        src: &'a mut dyn Source,
        msgs: &'a mut Vec<Message>,
    ) -> SourceInner<'a> {
        SourceInner { src, tell, msgs }
    }

    pub fn read(
        &mut self,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let start = out.len();
        let r = self
            .core
            .read(&mut Self::inner(&mut self.tell, src, msgs), n, out);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    pub fn readline(
        &mut self,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let start = out.len();
        let r = self
            .core
            .readline(&mut Self::inner(&mut self.tell, src, msgs), n, out);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    pub fn skip(
        &mut self,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: Option<i128>,
    ) -> Result<usize, Error> {
        self.core
            .skip(&mut Self::inner(&mut self.tell, src, msgs), n)
    }

    /// `close()`: whether the file object is to be closed first (`self.stream` was set),
    /// and what `close_decompressor()` raised.
    pub fn close(&mut self) -> (bool, Result<(), Error>) {
        let was_open = !self.core.stream_closed;
        (was_open, self.core.close())
    }
}

/// A record's `raw_stream`.
#[derive(Debug)]
pub struct RawStream {
    /// `None`: the stream is the iterator's reader itself (no length).
    pub(crate) limit: Option<Limit>,
    /// Set for a `DigestVerifyingReader`.
    pub(crate) verifier: Option<Verifier>,
    pub(crate) checker: DigestChecker,
    /// Scratch for digesting what `read_to_end` skips.
    scratch: Vec<u8>,
}

impl RawStream {
    pub(crate) fn new(
        limit: Option<Limit>,
        verifier: Option<Verifier>,
        checker: DigestChecker,
    ) -> Self {
        Self {
            limit,
            verifier,
            checker,
            scratch: Vec::new(),
        }
    }

    /// The record's `digest_checker`.
    pub fn digest_checker(&self) -> &DigestChecker {
        &self.checker
    }

    /// Whether this is a `LimitReader` (or `DigestVerifyingReader`) rather than the
    /// iterator's own reader.
    pub fn is_limited(&self) -> bool {
        self.limit.is_some()
    }

    /// Whether this is a `DigestVerifyingReader`.
    pub fn is_verifying(&self) -> bool {
        self.verifier.is_some()
    }

    /// `limit` of a `LimitReader` (saturated past `i64`).
    pub fn limit(&self) -> Option<i64> {
        self.limit.map(|l| {
            i64::try_from(l.limit.clamp(i128::from(i64::MIN), i128::from(i64::MAX)))
                .expect("clamped")
        })
    }

    /// `tell()`.
    pub(crate) fn tell(&self, top: &TopReader) -> i64 {
        match self.limit {
            None => top.core.tell(),
            Some(l) => i64::try_from(l.orig - l.limit).unwrap_or(i64::MAX),
        }
    }

    /// `_update(buff)`.
    fn update(&mut self, buff: &[u8], msgs: &mut Vec<Message>) -> Result<(), Error> {
        let Some(l) = self.limit.as_mut() else {
            return Ok(());
        };
        l.limit -= i128::try_from(buff.len()).expect("len");
        let limit = l.limit;
        if let Some(v) = self.verifier.as_mut() {
            v.update(buff, limit, &mut self.checker, msgs)?;
        }
        Ok(())
    }

    /// `read(n)`.
    pub(crate) fn read(
        &mut self,
        top: &mut TopReader,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let Some(l) = self.limit else {
            return top.read(src, msgs, n, out);
        };
        let length = l.clamp(n);
        if length == 0 {
            return Ok(());
        }
        let start = out.len();
        top.read(src, msgs, Some(length), out)?;
        let r = self.update(&out[start..], msgs);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    /// `readline(n)`.
    pub(crate) fn readline(
        &mut self,
        top: &mut TopReader,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let Some(l) = self.limit else {
            return top.readline(src, msgs, n, out);
        };
        let length = l.clamp(n);
        if length == 0 {
            return Ok(());
        }
        let start = out.len();
        top.readline(src, msgs, Some(length), out)?;
        let r = self.update(&out[start..], msgs);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    /// `read(n)` without keeping the bytes (digests still see them); how many were read.
    fn skip(
        &mut self,
        top: &mut TopReader,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: i128,
    ) -> Result<usize, Error> {
        if self.verifier.is_some() {
            let mut buf = std::mem::take(&mut self.scratch);
            buf.clear();
            let r = self.read(top, src, msgs, Some(n), &mut buf);
            let k = buf.len();
            self.scratch = buf;
            return r.map(|()| k);
        }
        let Some(l) = self.limit else {
            return top.skip(src, msgs, Some(n));
        };
        let length = l.clamp(Some(n));
        if length == 0 {
            return Ok(0);
        }
        let k = top.skip(src, msgs, Some(length))?;
        if let Some(l) = self.limit.as_mut() {
            l.limit -= i128::try_from(k).expect("len");
        }
        Ok(k)
    }

    /// `while self.record.raw_stream.read(BUFF_SIZE): pass` (`read_to_end`).
    pub(crate) fn read_all(
        &mut self,
        top: &mut TopReader,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
    ) -> Result<(), Error> {
        let n = i128::try_from(BUFF_SIZE).expect("BUFF_SIZE");
        while self.skip(top, src, msgs, n)? > 0 {}
        Ok(())
    }

    /// `begin_payload()` of a `DigestVerifyingReader`.
    pub(crate) fn begin_payload(&mut self, msgs: &mut Vec<Message>) -> Result<(), Error> {
        let limit = self.limit.map_or(0, |l| l.limit);
        match self.verifier.as_mut() {
            Some(v) => v.begin_payload(limit, &mut self.checker, msgs),
            None => Ok(()),
        }
    }
}

/// A decoder's view of the raw stream it wraps.
struct RawInner<'a> {
    raw: &'a mut RawStream,
    top: &'a mut TopReader,
    src: &'a mut dyn Source,
    msgs: &'a mut Vec<Message>,
}

impl Inner for RawInner<'_> {
    fn read(&mut self, n: Option<i128>, out: &mut Vec<u8>) -> Result<(), Error> {
        self.raw.read(self.top, self.src, self.msgs, n, out)
    }

    fn readline(&mut self, n: Option<i128>, out: &mut Vec<u8>) -> Result<(), Error> {
        self.raw.readline(self.top, self.src, self.msgs, n, out)
    }

    fn stderr(&mut self, text: String) {
        self.msgs.push(Message::Stderr(text));
    }
}

/// A reader `content_stream()` returns.
#[derive(Debug)]
pub enum ContentStream {
    /// The record's `raw_stream` itself.
    Raw(RawHandle),
    /// A `BufferedReader` or `ChunkedDataReader` over the raw stream.
    Decoder(Box<Decoder>),
}

/// `BufferedReader(raw_stream, decomp_type=...)` or `ChunkedDataReader(raw_stream, ...)`.
#[derive(Debug)]
pub struct Decoder {
    pub(crate) raw: RawHandle,
    pub(crate) core: BufCore,
}

impl Decoder {
    pub(crate) fn new(
        raw: RawHandle,
        chunked: bool,
        decomp: Option<DecompType>,
        brotli: Option<DecompressorFactory>,
        zlib_flush_raises: bool,
    ) -> Self {
        let mut core = BufCore::new(
            Some(i128::try_from(BUFF_SIZE).expect("BUFF_SIZE")),
            decomp,
            brotli,
        );
        core.zlib_flush_raises = zlib_flush_raises;
        if chunked {
            core.chunked = Some(Chunked::default());
        }
        Self { raw, core }
    }

    /// Whether this is a `ChunkedDataReader`.
    pub fn is_chunked(&self) -> bool {
        self.core.chunked.is_some()
    }

    /// The decompression in use (`decomp_type`), if any.
    pub fn decomp_type(&self) -> Option<DecompType> {
        self.core.decomp_type
    }

    /// `tell()` (`BufferedReader.tell`: the decoded bytes buffered so far).
    pub fn tell(&self) -> i64 {
        self.core.tell()
    }

    /// The raw stream it reads.
    pub fn raw(&self) -> &RawHandle {
        &self.raw
    }

    pub(crate) fn read(
        &mut self,
        top: &mut TopReader,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let raw = Arc::clone(&self.raw);
        let mut guard = lock(&raw);
        let mut inner = RawInner {
            raw: &mut guard,
            top,
            src,
            msgs,
        };
        let start = out.len();
        let r = self.core.read(&mut inner, n, out);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    pub(crate) fn readline(
        &mut self,
        top: &mut TopReader,
        src: &mut dyn Source,
        msgs: &mut Vec<Message>,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        let raw = Arc::clone(&self.raw);
        let mut guard = lock(&raw);
        let mut inner = RawInner {
            raw: &mut guard,
            top,
            src,
            msgs,
        };
        let start = out.len();
        let r = self.core.readline(&mut inner, n, out);
        if r.is_err() {
            out.truncate(start);
        }
        r
    }

    /// The part of `close()` after `self.stream.close()` (which the caller does first when
    /// `stream_open()`).
    pub(crate) fn close_own(&mut self) -> Result<(), Error> {
        self.core.close()
    }

    /// `self.stream` is still set.
    pub(crate) fn stream_open(&self) -> bool {
        !self.core.stream_closed
    }
}
