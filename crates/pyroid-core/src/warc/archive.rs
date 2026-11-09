// Ports warcio 1.8.2's archiveiterator.py and ArcWarcRecordLoader (Apache-2.0).
//! `ArchiveIterator` (and `WARCIterator`/`ARCIterator`) with
//! `ArcWarcRecordLoader.parse_record_stream`: the generator `_iterate_records` as a state
//! machine, `_consume_blanklines`, `read_to_end`, offsets and lengths, the non-chunked gzip
//! error, and every warning warcio prints, queued as [`Message`]s.

use std::sync::Arc;
use std::sync::Mutex;

use super::buffered::{DecompressorFactory, Limit};
use super::digest::{DigestChecker, DigestKind, Verifier};
use super::headers::{self, FirstLine, LineSource, ParseError, Scratch, StatusAndHeaders};
use super::pystr::{self, PyInt};
use super::record::{Format, PyIntValue, Record};
use super::stream::{ContentStream, RawHandle, RawStream, TopReader, lock};
use super::{Error, Message, PyExc, Source};
use crate::ftfy::UnicodeVersion;

const WARC_TYPES: &[&str] = &["WARC/1.1", "WARC/1.0", "WARC/0.17", "WARC/0.18"];
const HTTP_TYPES: &[&str] = &["HTTP/1.0", "HTTP/1.1"];
const HTTP_VERBS: &[&str] = &[
    "GET", "HEAD", "POST", "PUT", "DELETE", "TRACE", "OPTIONS", "CONNECT", "PATCH",
];

const GZIP_ERR_MSG: &str = "
    ERROR: non-chunked gzip file detected, gzip block continues
    beyond single record.

    This file is probably not a multi-member gzip but a single gzip file.

    To allow seek, a gzipped {1} must have each record compressed into
    a single gzip member and concatenated together.

    This file is likely still valid and can be fixed by running:

    warcio recompress <path/to/file> <path/to/new_file>

";

/// The `ArchiveIterator` arguments, plus the facts about the Python that runs it.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)] // warcio's keyword arguments
pub struct ReaderOptions {
    /// `no_record_parse`.
    pub no_record_parse: bool,
    /// `verify_http`.
    pub verify_http: bool,
    /// `arc2warc`.
    pub arc2warc: bool,
    /// `ensure_http_headers`.
    pub ensure_http_headers: bool,
    /// `block_size` (`read(n)` size on the file; negative = `read(-1)`).
    pub block_size: i64,
    /// `check_digests`.
    pub check_digests: DigestKind,
    /// `Some(Warc)` for `WARCIterator`, `Some(Arc)` for `ARCIterator`.
    pub known_format: Option<Format>,
    /// brotlicffi's `Decompressor` for `br`, when it is importable (>= 1.2.1.0).
    pub brotli: Option<DecompressorFactory>,
    /// `unicodedata.unidata_version` of the running Python (digits for `int()`, `repr`).
    pub unicode: UnicodeVersion,
    /// `sys.get_int_max_str_digits()` (0 = no limit).
    pub int_max_str_digits: usize,
    /// `base64.b64decode('AA==AA') == b'\x00'` on the running Python (older releases stop
    /// at padding).
    pub base64_legacy_padding: bool,
    /// zlib's `decompressobj().flush()` raises on a broken stream (CPython 3.13 and later),
    /// so `close()` raises `zlib.error` after a corrupt gzip member.
    pub zlib_flush_raises: bool,
    /// `make_warc_id()` for arc2warc records (default: a random UUID).
    pub make_warc_id: fn() -> String,
}

impl Default for ReaderOptions {
    fn default() -> Self {
        Self {
            no_record_parse: false,
            verify_http: false,
            arc2warc: false,
            ensure_http_headers: false,
            block_size: 16384,
            check_digests: DigestKind::Off,
            known_format: None,
            brotli: None,
            unicode: UnicodeVersion::V15_1,
            int_max_str_digits: 4300,
            base64_legacy_padding: false,
            zlib_flush_raises: true,
            make_warc_id: headers::make_warc_id,
        }
    }
}

/// Where `_iterate_records` is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Gen {
    /// Not started, or looping.
    #[default]
    Run,
    /// Suspended at `yield self.record`.
    Yielded,
    /// Finished (returned or raised).
    Done,
}

/// Reused buffers.
#[derive(Debug, Default)]
struct Bufs {
    scratch: Scratch,
}

/// `ArchiveIterator(fileobj, ...)`.
///
/// Every call that reads takes the [`Source`]; it must be the same file object each time.
#[derive(Debug)]
pub struct ArchiveReader {
    opts: ReaderOptions,
    top: TopReader,
    msgs: Vec<Message>,
    known_format: Option<Format>,
    member_info: Option<(i64, i64)>,
    offset: i64,
    next_line: Option<Vec<u8>>,
    err_count: u64,
    /// `self.record`: its raw stream.
    record: Option<RawHandle>,
    /// `self.reader` was set to None by `close()`.
    closed: bool,
    state: Gen,
    raise_invalid_gzip: bool,
    empty_record: bool,
    parser_statusline: Option<String>,
    bufs: Bufs,
}

/// Lines from the iterator's reader (`self.reader.readline()`).
struct TopLines<'a> {
    top: &'a mut TopReader,
    src: &'a mut dyn Source,
    msgs: &'a mut Vec<Message>,
}

impl LineSource for TopLines<'_> {
    fn readline(&mut self, out: &mut Vec<u8>) -> Result<(), Error> {
        self.top.readline(self.src, self.msgs, None, out)
    }
}

/// Lines from a record's raw stream.
struct RawLines<'a> {
    raw: &'a mut RawStream,
    top: &'a mut TopReader,
    src: &'a mut dyn Source,
    msgs: &'a mut Vec<Message>,
}

impl LineSource for RawLines<'_> {
    fn readline(&mut self, out: &mut Vec<u8>) -> Result<(), Error> {
        self.raw.readline(self.top, self.src, self.msgs, None, out)
    }
}

fn eof_error() -> Error {
    Error::Py(PyExc::EofError, String::new())
}

/// What `_detect_type_load_headers` found.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Detected {
    Warc,
    Arc,
    Arc2Warc,
}

impl ArchiveReader {
    /// `ArchiveIterator(fh, **opts)`, with `fh.tell()` (0 when it raises: warcio then wraps
    /// the file in `UnseekableYetTellable`).
    pub fn new(opts: ReaderOptions, tell: i64) -> Self {
        Self {
            top: TopReader::new(opts.block_size, tell, opts.zlib_flush_raises),
            known_format: opts.known_format,
            opts,
            msgs: Vec::new(),
            member_info: None,
            offset: tell,
            next_line: None,
            err_count: 0,
            record: None,
            closed: false,
            state: Gen::Run,
            raise_invalid_gzip: false,
            empty_record: false,
            parser_statusline: None,
            bufs: Bufs::default(),
        }
    }

    /// The options.
    pub fn options(&self) -> &ReaderOptions {
        &self.opts
    }

    /// Take the queued messages (stderr writes and log warnings, in order).
    pub fn take_messages(&mut self) -> Vec<Message> {
        std::mem::take(&mut self.msgs)
    }

    /// Whether messages are queued.
    pub fn has_messages(&self) -> bool {
        !self.msgs.is_empty()
    }

    /// The `statusline` of the last `StatusAndHeadersParserException`.
    pub fn parser_exception_statusline(&self) -> Option<&str> {
        self.parser_statusline.as_deref()
    }

    /// `offset`.
    pub fn offset(&self) -> i64 {
        self.offset
    }

    /// `err_count`.
    pub fn err_count(&self) -> u64 {
        self.err_count
    }

    /// `member_info`.
    pub fn member_info(&self) -> Option<(i64, i64)> {
        self.member_info
    }

    /// `known_format`.
    pub fn known_format(&self) -> Option<Format> {
        self.known_format
    }

    /// `reader is None` (after `close()`).
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// `record is None`.
    pub fn has_record(&self) -> bool {
        self.record.is_some()
    }

    /// Whether the generator has finished (`next()` raises StopIteration).
    pub fn is_exhausted(&self) -> bool {
        self.state == Gen::Done
    }

    /// `next(iterator)`: the next record, or `None` at the end (StopIteration). After an
    /// error the iterator is exhausted, as a generator that raised.
    pub fn next_record(&mut self, src: &mut dyn Source) -> Result<Option<Record>, Error> {
        let r = self.step(src);
        if r.is_err() {
            self.state = Gen::Done;
        }
        r
    }

    fn step(&mut self, src: &mut dyn Source) -> Result<Option<Record>, Error> {
        match self.state {
            Gen::Done => return Ok(None),
            Gen::Yielded => {
                self.state = Gen::Run;
                if !self.after_record(src)? {
                    self.finish()?;
                    return Ok(None);
                }
            }
            Gen::Run => {}
        }
        loop {
            match self.next_record_inner(src) {
                Ok(rec) => {
                    if self.raise_invalid_gzip {
                        return Err(self.invalid_gzip_err());
                    }
                    self.state = Gen::Yielded;
                    return Ok(Some(rec));
                }
                Err(e) if e.is_eof_error() => self.empty_record = true,
                Err(e) => return Err(e),
            }
            if !self.after_record(src)? {
                self.finish()?;
                return Ok(None);
            }
        }
    }

    /// The loop body after the `try`: whether to go on.
    fn after_record(&mut self, src: &mut dyn Source) -> Result<bool, Error> {
        self.read_to_end(src)?;
        if self.top.core.decomp.is_some() {
            // if another gzip member, continue
            if self.top.core.read_next_member() {
                return Ok(true);
            }
            // if empty record, then we're done
            if self.empty_record {
                return Ok(false);
            }
            // otherwise, probably a gzip containing multiple non-chunked records
            self.raise_invalid_gzip = true;
            return Ok(true);
        }
        // non-gzip, so we're done
        Ok(!self.empty_record)
    }

    fn finish(&mut self) -> Result<(), Error> {
        self.state = Gen::Done;
        self.close()
    }

    fn invalid_gzip_err(&self) -> Error {
        let frmt = self.known_format.map_or("warc/arc", Format::as_str);
        Error::Py(
            PyExc::ArchiveLoadFailed,
            GZIP_ERR_MSG.replace("{1}", &frmt.to_uppercase()),
        )
    }

    /// `close()` (raises what zlib's `flush()` raises; the reader then stays).
    pub fn close(&mut self) -> Result<(), Error> {
        self.record = None;
        if !self.closed {
            self.top.core.close_decompressor()?;
            self.closed = true;
        }
        Ok(())
    }

    /// `_next_record(self.next_line)`.
    fn next_record_inner(&mut self, src: &mut dyn Source) -> Result<Record, Error> {
        let next_line = self.next_line.take();
        let mut rec_headers = StatusAndHeaders::default();
        let mut http = StatusAndHeaders::default();
        let r = self.parse_record_stream(src, next_line.as_deref(), &mut rec_headers, &mut http);
        // `self.next_line` is only reassigned by read_to_end.
        self.next_line = next_line;
        let rec = r?;
        self.record = Some(Arc::clone(&rec.raw));
        self.member_info = None;
        // Track known format for faster parsing of other records
        if !self.opts.arc2warc {
            self.known_format = Some(rec.format);
        }
        Ok(rec)
    }

    /// `_consume_blanklines()`: the next line (if any) and the bytes of blank lines.
    fn consume_blanklines(
        &mut self,
        src: &mut dyn Source,
    ) -> Result<(Option<Vec<u8>>, i64), Error> {
        let mut empty_size = 0i64;
        let mut first_line = true;
        loop {
            let mut line = Vec::new();
            self.top.readline(src, &mut self.msgs, None, &mut line)?;
            if line.is_empty() {
                return Ok((None, empty_size));
            }
            let stripped = line
                .iter()
                .rposition(|&b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c))
                .map_or(0, |i| i + 1);
            if stripped == 0 || first_line {
                empty_size += i64::try_from(line.len()).expect("line");
                if stripped != 0 {
                    // if first line is not blank, likely content-length was invalid
                    let err_offset = self.fh_tell(src)? - self.rem_length() - empty_size;
                    self.msgs.push(Message::Stderr(format!(
                        "    WARNING: Record not followed by newline, perhaps Content-Length is invalid\n    Offset: {}\n    Remainder: {}\n",
                        err_offset,
                        pystr::bytes_repr(&line)
                    )));
                    self.err_count += 1;
                }
                first_line = false;
                continue;
            }
            return Ok((Some(line), empty_size));
        }
    }

    /// `self.fh.tell()`.
    fn fh_tell(&self, src: &mut dyn Source) -> Result<i64, Error> {
        Ok(src.tell()?.unwrap_or(self.top.tell))
    }

    fn rem_length(&self) -> i64 {
        i64::try_from(self.top.core.rem_length()).expect("rem_length")
    }

    /// `read_to_end()`.
    pub fn read_to_end(&mut self, src: &mut dyn Source) -> Result<(), Error> {
        // no current record to read
        let Some(raw) = self.record.clone() else {
            return Ok(());
        };
        // already at end of this record, don't read until it is consumed
        if self.member_info.is_some() {
            return Ok(());
        }
        let curr_offset = self.offset;
        lock(&raw).read_all(&mut self.top, src, &mut self.msgs)?;
        let (next_line, empty_size) = self.consume_blanklines(src)?;
        self.next_line = next_line;
        self.offset = self.fh_tell(src)? - self.rem_length();
        if let Some(l) = &self.next_line {
            self.offset -= i64::try_from(l.len()).expect("line");
        }
        let mut length = self.offset - curr_offset;
        if self.top.core.decomp.is_none() {
            length -= empty_size;
        }
        self.member_info = Some((curr_offset, length));
        Ok(())
    }

    fn member_info_or_read(&mut self, src: &mut dyn Source) -> Result<(i64, i64), Error> {
        if self.member_info.is_none() {
            self.read_to_end(src)?;
        }
        self.member_info.ok_or_else(|| {
            Error::Py(
                PyExc::TypeError,
                "'NoneType' object is not subscriptable".into(),
            )
        })
    }

    /// `get_record_offset()`.
    pub fn get_record_offset(&mut self, src: &mut dyn Source) -> Result<i64, Error> {
        Ok(self.member_info_or_read(src)?.0)
    }

    /// `get_record_length()`.
    pub fn get_record_length(&mut self, src: &mut dyn Source) -> Result<i64, Error> {
        Ok(self.member_info_or_read(src)?.1)
    }

    /// `ArcWarcRecordLoader.parse_record_stream(self.reader, statusline, known_format, ...)`.
    #[allow(clippy::too_many_lines)] // one function in warcio, kept as one for review
    fn parse_record_stream(
        &mut self,
        src: &mut dyn Source,
        statusline: Option<&[u8]>,
        rec_headers: &mut StatusAndHeaders,
        http: &mut StatusAndHeaders,
    ) -> Result<Record, Error> {
        let detected = self.detect_type_load_headers(src, statusline, rec_headers)?;
        let (format, rec_type, uri, length, content_type, sub_len) = match detected {
            Detected::Arc => {
                let uri = rec_headers.get_lower("uri").map(str::to_owned);
                let rec_type = if uri.as_deref().is_some_and(|u| u.starts_with("filedesc://")) {
                    "arc_header"
                } else {
                    "response"
                };
                (
                    Format::Arc,
                    Some(rec_type.to_owned()),
                    uri,
                    rec_headers.get_lower("length").map(str::to_owned),
                    rec_headers.get_lower("content-type").map(str::to_owned),
                    rec_headers.total_len,
                )
            }
            Detected::Warc | Detected::Arc2Warc => {
                let rec_type = rec_headers.get_lower("warc-type").map(str::to_owned);
                let uri = self.ensure_target_uri_format(rec_headers);
                let sub_len = if detected == Detected::Warc {
                    0
                } else {
                    rec_headers.total_len
                };
                (
                    Format::Warc,
                    rec_type,
                    uri,
                    rec_headers.get_lower("content-length").map(str::to_owned),
                    rec_headers.get_lower("content-type").map(str::to_owned),
                    sub_len,
                )
            }
        };

        let length: Option<PyInt> = length.map(|l| {
            match pystr::py_int(&l, self.opts.int_max_str_digits, self.opts.unicode) {
                Some(v) => {
                    let v = v.sub_small(sub_len);
                    if v.is_negative() { PyInt::Small(0) } else { v }
                }
                // err condition
                None => PyInt::Small(0),
            }
        });

        let mut checker = DigestChecker {
            kind: self.opts.check_digests,
            ..DigestChecker::default()
        };
        // limit stream to the length for all valid records
        let mut raw = match &length {
            Some(l) => {
                let lim = l.saturating();
                let mut verifier = None;
                if self.opts.check_digests != DigestKind::Off {
                    let payload = rec_headers
                        .get_lower("warc-payload-digest")
                        .map(str::to_owned);
                    let block = rec_headers
                        .get_lower("warc-block-digest")
                        .map(str::to_owned);
                    let segment = rec_headers.get_lower("warc-segment-number");
                    if payload.as_deref().is_some_and(|p| !p.is_empty())
                        || block.as_deref().is_some_and(|b| !b.is_empty())
                    {
                        verifier = Some(Verifier::new(
                            rec_type.as_deref(),
                            payload,
                            block,
                            segment,
                            self.opts.base64_legacy_padding,
                            &mut checker,
                            &mut self.msgs,
                        )?);
                    }
                }
                RawStream::new(
                    Some(Limit {
                        limit: lim,
                        orig: lim,
                    }),
                    verifier,
                    checker,
                )
            }
            None => RawStream::new(None, None, checker),
        };

        let mut http_headers = None;
        let mut payload_length = PyIntValue::Int(-1);
        // load http headers if parsing
        if !self.opts.no_record_parse {
            let start = raw.tell(&self.top);
            let loaded = self.load_http_headers(
                src,
                rec_type.as_deref(),
                uri.as_deref(),
                &mut raw,
                length.as_ref(),
                http,
            )?;
            if loaded {
                http_headers = Some(std::mem::take(http));
            }
            if let Some(l) = &length
                && *l != PyInt::Small(0)
                && http_headers
                    .as_ref()
                    .is_some_and(StatusAndHeaders::is_truthy)
            {
                let used = raw.tell(&self.top) - start;
                let v = match l {
                    PyInt::Small(x) => PyInt::Small(x - i128::from(used)),
                    huge @ PyInt::Huge { .. } => huge.sub_small(u64::try_from(used).unwrap_or(0)),
                };
                payload_length = PyIntValue::from_int(&v);
            }
        }
        // generate validate http headers (eg. for replay)
        if !http_headers
            .as_ref()
            .is_some_and(StatusAndHeaders::is_truthy)
            && self.opts.ensure_http_headers
        {
            http_headers = Some(default_http_headers(
                length.as_ref(),
                content_type.as_deref(),
            ));
        }
        if raw.is_verifying() {
            raw.begin_payload(&mut self.msgs)?;
        }
        Ok(Record {
            format,
            rec_type,
            rec_headers: std::mem::take(rec_headers),
            http_headers,
            content_type,
            length: length.as_ref().map(PyIntValue::from_int),
            payload_length,
            raw: Arc::new(Mutex::new(raw)),
        })
    }

    /// `load_http_headers(rec_type, uri, stream, length)` into `out`; whether there are any.
    fn load_http_headers(
        &mut self,
        src: &mut dyn Source,
        rec_type: Option<&str>,
        uri: Option<&str>,
        raw: &mut RawStream,
        length: Option<&PyInt>,
        out: &mut StatusAndHeaders,
    ) -> Result<bool, Error> {
        // only if length == 0 don't parse
        if length == Some(&PyInt::Small(0)) {
            return Ok(false);
        }
        // only certain record types can have http headers
        let Some(rec_type @ ("response" | "request" | "revisit")) = rec_type else {
            return Ok(false);
        };
        // only http:/https: uris can have http headers
        let Some(uri) = uri else {
            return Err(Error::Py(
                PyExc::AttributeError,
                "'NoneType' object has no attribute 'startswith'".into(),
            ));
        };
        if !(uri.starts_with("http:") || uri.starts_with("https:")) {
            return Ok(false);
        }
        let statuslist = if rec_type == "request" {
            HTTP_VERBS
        } else {
            HTTP_TYPES
        };
        let mut lines = RawLines {
            raw,
            top: &mut self.top,
            src,
            msgs: &mut self.msgs,
        };
        let r = headers::parse(
            out,
            &mut lines,
            FirstLine::Read,
            statuslist,
            self.opts.verify_http,
            &mut self.bufs.scratch,
        );
        match r {
            Ok(()) => Ok(true),
            Err(ParseError::Eof) if rec_type == "revisit" => Ok(false),
            Err(ParseError::Io(e)) if rec_type == "revisit" && e.is_eof_error() => Ok(false),
            Err(ParseError::Eof) => Err(eof_error()),
            Err(ParseError::Io(e)) => Err(e),
            Err(ParseError::Status { msg, statusline }) => {
                self.parser_statusline = Some(statusline);
                Err(headers::status_error(msg))
            }
            Err(ParseError::Parts(_)) => unreachable!("only the ARC parser lists parts"),
        }
    }

    /// `_detect_type_load_headers(stream, statusline, known_format)`.
    fn detect_type_load_headers(
        &mut self,
        src: &mut dyn Source,
        statusline: Option<&[u8]>,
        out: &mut StatusAndHeaders,
    ) -> Result<Detected, Error> {
        let mut first = match statusline {
            Some(b) => FirstLine::Bytes(b),
            None => FirstLine::Read,
        };
        let warc_line;
        if self.known_format != Some(Format::Arc) {
            // try as warc first
            let mut lines = TopLines {
                top: &mut self.top,
                src,
                msgs: &mut self.msgs,
            };
            match headers::parse(
                out,
                &mut lines,
                first,
                WARC_TYPES,
                true,
                &mut self.bufs.scratch,
            ) {
                Ok(()) => return Ok(Detected::Warc),
                Err(ParseError::Status { statusline, .. }) => {
                    if self.known_format == Some(Format::Warc) {
                        return Err(Error::Py(
                            PyExc::ArchiveLoadFailed,
                            format!("Invalid WARC record, first line: {statusline}"),
                        ));
                    }
                    warc_line = statusline;
                    first = FirstLine::Str(&warc_line);
                }
                Err(ParseError::Eof) => return Err(eof_error()),
                Err(ParseError::Io(e)) => return Err(e),
                Err(ParseError::Parts(_)) => unreachable!("only the ARC parser lists parts"),
            }
        }
        // now try as arc
        let mut lines = TopLines {
            top: &mut self.top,
            src,
            msgs: &mut self.msgs,
        };
        let make_id = self.opts.make_warc_id;
        match headers::parse_arc(
            out,
            &mut lines,
            first,
            self.opts.arc2warc,
            &make_id,
            self.opts.unicode,
            &mut self.bufs.scratch,
        ) {
            Ok(()) => Ok(if self.opts.arc2warc {
                Detected::Arc2Warc
            } else {
                Detected::Arc
            }),
            Err(ParseError::Parts(parts)) => {
                let msg = if self.known_format == Some(Format::Arc) {
                    "Invalid ARC record, first line: "
                } else {
                    "Unknown archive format, first line: "
                };
                let parts: Vec<&str> = parts.iter().map(String::as_str).collect();
                Err(Error::Py(
                    PyExc::ArchiveLoadFailed,
                    format!("{msg}{}", pystr::list_repr(&parts, self.opts.unicode)),
                ))
            }
            Err(ParseError::Eof) => Err(eof_error()),
            Err(ParseError::Io(e)) => Err(e),
            Err(ParseError::Status { .. }) => unreachable!("the ARC parser lists parts"),
        }
    }

    /// `_ensure_target_uri_format(rec_headers)`.
    fn ensure_target_uri_format(&mut self, h: &mut StatusAndHeaders) -> Option<String> {
        let mut uri = h.get_lower("warc-target-uri")?.to_owned();
        if uri.starts_with('<') && uri.ends_with('>') {
            uri = uri[1..uri.len() - 1].to_owned();
            h.replace_header("WARC-Target-URI", &uri);
        }
        if uri.contains(' ') {
            self.msgs.push(Message::LogWarning(format!(
                "Replacing spaces in invalid WARC-Target-URI: {uri}"
            )));
            uri = uri.replace(' ', "%20");
            h.replace_header("WARC-Target-URI", &uri);
        }
        Some(uri)
    }

    // --- Streams -------------------------------------------------------------------------

    /// `raw_stream.read(n)` of a record (any record: an old one reads where the file is).
    pub fn raw_read(
        &mut self,
        raw: &RawHandle,
        src: &mut dyn Source,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        lock(raw).read(&mut self.top, src, &mut self.msgs, n, out)
    }

    /// `raw_stream.readline(n)`.
    pub fn raw_readline(
        &mut self,
        raw: &RawHandle,
        src: &mut dyn Source,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        lock(raw).readline(&mut self.top, src, &mut self.msgs, n, out)
    }

    /// `raw_stream.tell()`.
    pub fn raw_tell(&self, raw: &RawHandle) -> i64 {
        lock(raw).tell(&self.top)
    }

    /// `raw_stream.close()`: whether the file object is to be closed (first), and what the
    /// reader's `close_decompressor()` raised after that.
    pub fn raw_close(&mut self, _raw: &RawHandle) -> (bool, Result<(), Error>) {
        self.top.close()
    }

    /// `content_stream()` of a record.
    pub fn content_stream(&self, rec: &Record) -> ContentStream {
        rec.content_stream(&self.opts)
    }

    /// `content_stream().read(n)`.
    pub fn stream_read(
        &mut self,
        cs: &mut ContentStream,
        src: &mut dyn Source,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        match cs {
            ContentStream::Raw(raw) => self.raw_read(raw, src, n, out),
            ContentStream::Decoder(d) => d.read(&mut self.top, src, &mut self.msgs, n, out),
        }
    }

    /// `content_stream().readline(n)`.
    pub fn stream_readline(
        &mut self,
        cs: &mut ContentStream,
        src: &mut dyn Source,
        n: Option<i128>,
        out: &mut Vec<u8>,
    ) -> Result<(), Error> {
        match cs {
            ContentStream::Raw(raw) => self.raw_readline(raw, src, n, out),
            ContentStream::Decoder(d) => d.readline(&mut self.top, src, &mut self.msgs, n, out),
        }
    }

    /// `content_stream().tell()`.
    pub fn stream_tell(&self, cs: &ContentStream) -> i64 {
        match cs {
            ContentStream::Raw(raw) => self.raw_tell(raw),
            ContentStream::Decoder(d) => d.tell(),
        }
    }

    /// `content_stream().close()`: whether the file object is to be closed (first), and
    /// what was raised after that.
    pub fn stream_close(&mut self, cs: &mut ContentStream) -> (bool, Result<(), Error>) {
        match cs {
            ContentStream::Raw(raw) => self.raw_close(raw),
            ContentStream::Decoder(d) => {
                let mut close_file = false;
                if d.stream_open() {
                    let (cf, r) = self.top.close();
                    close_file = cf;
                    if r.is_err() {
                        return (close_file, r);
                    }
                }
                (close_file, d.close_own())
            }
        }
    }

    /// The iterator's reader's `tell()` (`num_read`, decompressed bytes buffered so far).
    pub fn reader_tell(&self) -> i64 {
        self.top.core.tell()
    }

    /// `fh.tell()` as the reader tracks it.
    pub fn source_tell(&self) -> i64 {
        self.top.tell
    }

    /// Whether the iterator's reader still decompresses (gzip input).
    pub fn is_decompressing(&self) -> bool {
        self.top.core.decomp.is_some()
    }
}

/// `default_http_headers(length, content_type)`.
fn default_http_headers(length: Option<&PyInt>, content_type: Option<&str>) -> StatusAndHeaders {
    let mut h = StatusAndHeaders::new("HTTP/1.0", "200 OK", 0);
    if let Some(ct) = content_type.filter(|c| !c.is_empty()) {
        h.add_header("Content-Type", ct);
    }
    if let Some(l) = length.filter(|l| !l.is_negative()) {
        h.add_header("Content-Length", &l.to_decimal());
    }
    h
}
