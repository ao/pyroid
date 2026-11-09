// Ports warcio 1.8.2's reading stack (Apache-2.0, Webrecorder, Ilya Kreymer et al.).
//! A port of warcio 1.8.2's WARC/ARC reader: `ArchiveIterator`, `ArcWarcRecordLoader`,
//! `StatusAndHeadersParser` and the ARC parsers, the buffered readers (`BufferedReader`,
//! `DecompressingBufferedReader`, `ChunkedDataReader`, `LimitReader`,
//! `DigestVerifyingReader`) and `ArcWarcRecord.content_stream`.
//!
//! The reader is a statement-by-statement port: it calls the [`Source`] with the same
//! `read(n)` sizes at the same moments as warcio does, raises the same exceptions with the
//! same messages (as [`Error::Py`]), and queues what warcio writes to `sys.stderr` or logs
//! as [`Message`]s, in order. Nothing here calls Python; the bindings turn errors into
//! exceptions and emit the messages.
//!
//! Memory stays bounded by the block size and what the caller reads: nothing is allocated
//! from a header value (`Content-Length`, chunk sizes).
mod archive;
mod buffered;
mod digest;
mod fast;
mod headers;
mod pystr;
mod record;
mod stream;
mod zlib;

pub use archive::{ArchiveReader, ReaderOptions};
pub use buffered::{BUFF_SIZE, DecompType, Decompressor, DecompressorFactory};
pub use digest::{DigestChecker, DigestKind};
pub use fast::{
    Doc, DocReader, KEY_ERROR_ARCHIVE_DATE, KEY_ERROR_URI, Rows, VarColumn, magic_mime_ok,
};
pub use headers::{StatusAndHeaders, make_warc_id};
pub use record::{Format, PyIntValue, Record};
pub use stream::{ContentStream, Decoder, RawHandle, RawStream};
pub use zlib::DecompressObj;

use std::any::Any;
use std::fmt;
use std::io;

/// What warcio or CPython would raise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyExc {
    /// `warcio.exceptions.ArchiveLoadFailed`.
    ArchiveLoadFailed,
    /// `EOFError`.
    EofError,
    /// `gzip.BadGzipFile`.
    BadGzipFile,
    /// `zlib.error`.
    ZlibError,
    /// `brotlicffi.error`.
    BrotliError,
    /// `AttributeError`.
    AttributeError,
    /// `KeyError` (the message is `str(e)`, i.e. the repr of the key).
    KeyError,
    /// `ValueError`.
    ValueError,
    /// `OSError`.
    OsError,
    /// `binascii.Error` (a `ValueError` subclass): base16/base64 digest values that do not
    /// decode, with `check_digests`.
    BinasciiError,
    /// `TypeError` (`get_record_offset()` before any record).
    TypeError,
    /// `OverflowError`: a read size past `sys.maxsize` (from a huge `Content-Length`).
    OverflowError,
    /// `warcio.statusandheaders.StatusAndHeadersParserException`: an HTTP status line that
    /// does not start with a known protocol or verb, with `verify_http=True`. Its
    /// `statusline` attribute is kept by the reader (`parser_exception_statusline`).
    StatusAndHeadersParserException,
}

/// An error from a reader call.
#[derive(Debug)]
pub enum Error {
    /// From the Source, re-raised unchanged by the bindings (e.g. a PyErr).
    Source(Box<dyn Any + Send>),
    /// An exception warcio/CPython raises, with its exact message (str(e)).
    Py(PyExc, String),
}

/// The payload of an [`Error::Source`] that is an `EOFError` (`isinstance(e, EOFError)`):
/// warcio catches those in a few places (the end of iteration, an empty revisit record), so
/// a source whose `read` raises one must wrap its own payload in this.
#[derive(Debug)]
pub struct EofPayload(pub Box<dyn Any + Send>);

impl Error {
    /// `isinstance(e, EOFError)`: an [`Error::Py`] with [`PyExc::EofError`], or a source
    /// error whose payload is an [`EofPayload`].
    pub fn is_eof_error(&self) -> bool {
        match self {
            Self::Py(kind, _) => *kind == PyExc::EofError,
            Self::Source(payload) => payload.is::<EofPayload>(),
        }
    }

    /// The exception and message of an [`Error::Py`].
    pub fn py(&self) -> Option<(PyExc, &str)> {
        match self {
            Self::Py(kind, msg) => Some((*kind, msg)),
            Self::Source(_) => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(_) => f.write_str("error from the source"),
            Self::Py(kind, msg) => write!(f, "{kind:?}: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// A raw byte stream: one call = one Python `stream.read(n)`.
pub trait Source {
    /// One call = one Python `stream.read(n)`: append at most `n` bytes to `out`; appending
    /// nothing means EOF. `n == usize::MAX` stands for `read(-1)` (a negative `block_size`).
    fn read(&mut self, n: usize, out: &mut Vec<u8>) -> Result<(), Error>;
}

/// A [`Source`] over any [`io::Read`] (a local file, a byte slice): `read(n)` returns what
/// one or more `io::Read::read` calls give, up to `n` bytes, like a Python buffered binary
/// file (it stops early only at end of file). I/O errors become [`Error::Source`] with the
/// [`io::Error`] as payload.
#[derive(Debug)]
pub struct ReadSource<R>(pub R);

impl<R: io::Read> Source for ReadSource<R> {
    fn read(&mut self, n: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        // Grow by at most STEP past what is filled, so a huge `n` allocates only as data comes.
        const STEP: usize = 1 << 16;
        let start = out.len();
        let mut filled = 0;
        while filled < n {
            let want = (n - filled).min(STEP);
            out.resize(start + filled + want, 0);
            match self.0.read(&mut out[start + filled..]) {
                Ok(0) => break,
                Ok(k) => filled += k,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    out.truncate(start + filled);
                    return Err(Error::Source(Box::new(e)));
                }
            }
        }
        out.truncate(start + filled);
        Ok(())
    }
}

/// What warcio writes to `sys.stderr` or logs, queued in order for the bindings to emit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// `sys.stderr.write(text)` (the text includes its trailing newline).
    Stderr(String),
    /// `logging.getLogger("warcio.recordloader").warning(text)`.
    LogWarning(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_source_fills_requests() {
        let mut src = ReadSource(&b"abcdefgh"[..]);
        let mut out = Vec::new();
        src.read(3, &mut out).unwrap();
        assert_eq!(out, b"abc");
        src.read(10, &mut out).unwrap();
        assert_eq!(out, b"abcdefgh");
        src.read(10, &mut out).unwrap();
        assert_eq!(out, b"abcdefgh");
    }

    #[test]
    fn eof_classification() {
        assert!(Error::Py(PyExc::EofError, String::new()).is_eof_error());
        assert!(!Error::Py(PyExc::ZlibError, String::new()).is_eof_error());
        assert!(Error::Source(Box::new(EofPayload(Box::new(1u8)))).is_eof_error());
        assert!(!Error::Source(Box::new(1u8)).is_eof_error());
    }
}
