// Ports warcio 1.8.2's ArcWarcRecord (Apache-2.0).
//! `ArcWarcRecord`: what the iterator yields.

use std::sync::Arc;

use super::archive::ReaderOptions;
use super::buffered::{DecompType, DecompressorFactory};
use super::digest::DigestChecker;
use super::headers::StatusAndHeaders;
use super::pystr::{self, PyInt};
use super::stream::{ContentStream, Decoder, RawHandle, lock};

/// `record.format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `'warc'` (also ARC records read with `arc2warc=True`).
    Warc,
    /// `'arc'`.
    Arc,
}

impl Format {
    /// The Python value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Warc => "warc",
            Self::Arc => "arc",
        }
    }
}

/// A Python `int`: `Big` holds the decimal digits of a value outside `i64`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PyIntValue {
    /// A value that fits `i64`.
    Int(i64),
    /// `str(value)` of a larger one; Python gets it with `int(text)`.
    Big(String),
}

impl PyIntValue {
    pub(crate) fn from_int(v: &PyInt) -> Self {
        match v {
            PyInt::Small(x) => {
                i64::try_from(*x).map_or_else(|_| Self::Big(x.to_string()), Self::Int)
            }
            PyInt::Huge { .. } => Self::Big(v.to_decimal()),
        }
    }

    /// `str(value)`.
    pub fn to_decimal(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Big(s) => s.clone(),
        }
    }
}

/// `ArcWarcRecord`.
#[derive(Debug)]
pub struct Record {
    /// `format`.
    pub format: Format,
    /// `rec_type` (`WARC-Type`; `'response'` or `'arc_header'` for ARC).
    pub rec_type: Option<String>,
    /// `rec_headers`.
    pub rec_headers: StatusAndHeaders,
    /// `http_headers`.
    pub http_headers: Option<StatusAndHeaders>,
    /// `content_type` (the record's own Content-Type).
    pub content_type: Option<String>,
    /// `length`: the record length after `int()` (0 when invalid), `None` without one.
    pub length: Option<PyIntValue>,
    /// `payload_length` (-1 unless the HTTP headers were parsed from a record with a length).
    pub payload_length: PyIntValue,
    /// `raw_stream` (its `digest_checker` is the record's).
    pub raw: RawHandle,
}

impl Record {
    /// `digest_checker` (a copy of its current state).
    pub fn digest_checker(&self) -> DigestChecker {
        lock(&self.raw).digest_checker().clone()
    }

    /// `content_stream()`: the record's raw stream, or a decoder over it.
    ///
    /// Without `opts.brotli`, `br` is not a supported encoding and the payload is returned
    /// as is, like warcio without brotlicffi.
    pub fn content_stream(&self, opts: &ReaderOptions) -> ContentStream {
        content_stream(
            &self.raw,
            self.http_headers.as_ref(),
            opts.brotli,
            opts.zlib_flush_raises,
        )
    }
}

/// `ArcWarcRecord.content_stream()` for a record's raw stream and HTTP headers.
pub(crate) fn content_stream(
    raw: &RawHandle,
    http_headers: Option<&StatusAndHeaders>,
    brotli: Option<DecompressorFactory>,
    zlib_flush_raises: bool,
) -> ContentStream {
    let Some(h) = http_headers.filter(|h| h.is_truthy()) else {
        return ContentStream::Raw(Arc::clone(raw));
    };
    let encoding = h
        .get_lower("content-encoding")
        .filter(|e| !e.is_empty())
        .and_then(|e| decomp_for(e, brotli.is_some()));
    if h.get_lower("transfer-encoding") == Some("chunked") {
        ContentStream::Decoder(Box::new(Decoder::new(
            Arc::clone(raw),
            true,
            encoding,
            brotli,
            zlib_flush_raises,
        )))
    } else if let Some(enc) = encoding {
        ContentStream::Decoder(Box::new(Decoder::new(
            Arc::clone(raw),
            false,
            Some(enc),
            brotli,
            zlib_flush_raises,
        )))
    } else {
        ContentStream::Raw(Arc::clone(raw))
    }
}

/// `encoding.lower()` (with 'x-gzip' as 'gzip') if it is a key of `DECOMPRESSORS`.
fn decomp_for(encoding: &str, brotli: bool) -> Option<DecompType> {
    let eq = |t| pystr::lower_eq(encoding, t);
    if eq("gzip") || eq("x-gzip") {
        Some(DecompType::Gzip)
    } else if eq("deflate") {
        Some(DecompType::Deflate)
    } else if eq("deflate_alt") {
        Some(DecompType::DeflateAlt)
    } else if brotli && eq("br") {
        Some(DecompType::Brotli)
    } else {
        None
    }
}
