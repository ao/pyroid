// Follows datatrove 0.10.1's WarcReader.read_file / process_record (Apache-2.0).
//! The datatrove path: `for ri, record in enumerate(ArchiveIterator(f)): process_record(record)`
//! without Python objects for the records datatrove skips.
//!
//! [`DocReader`] keeps the records `process_record` keeps up to the points that need
//! Python (`magic.from_buffer` when `WARC-Identified-Payload-Type` is missing, `cchardet`
//! when the content is not UTF-8): it reads the content stream fully and says whether the
//! bytes are valid UTF-8; the caller decodes, detects and decides. Records of another type
//! or payload type are skipped by the iterator's own `read_to_end`, as in warcio.

use super::archive::{ArchiveReader, ReaderOptions};
use super::record::Record;
use super::{Error, Source};

/// `dict(record.rec_headers.headers)[key]` raised `KeyError(key)`: its `str()`.
pub const KEY_ERROR_URI: &str = "'uri'";
/// See [`KEY_ERROR_URI`].
pub const KEY_ERROR_ARCHIVE_DATE: &str = "'archive-date'";

/// A record `process_record` reads.
#[derive(Debug)]
pub struct Doc<'a> {
    /// `ri`: the record's index among all records of the file.
    pub index: usize,
    /// `record.rec_type == "conversion"` (else `"response"`).
    pub conversion: bool,
    /// `record.rec_headers["WARC-Record-ID"]` (`None` when missing).
    pub id: Option<&'a str>,
    /// `WARC-Target-URI`, else `dict(headers)["uri"]`; `Err` is the KeyError's `str()`.
    pub url: Result<&'a str, &'static str>,
    /// `WARC-Date`, else `dict(headers)["archive-date"]`; `Err` as for `url`.
    pub date: Result<&'a str, &'static str>,
    /// `WARC-Identified-Payload-Type`; `None` means datatrove asks `magic.from_buffer`.
    pub mime_type: Option<&'a str>,
    /// The HTTP `Content-Type` header, if any.
    pub content_type: Option<&'a str>,
    /// `record.content_stream().read()`.
    pub content: &'a [u8],
    /// The content is valid UTF-8 (`content.decode("UTF-8")` succeeds).
    pub utf8: bool,
    /// The record itself (headers, raw stream).
    pub record: &'a Record,
}

/// Whether datatrove keeps a mime type for a record type.
fn mime_ok(mime: &str, conversion: bool) -> bool {
    mime == "text/html" || mime == "application/xhtml+xml" || (conversion && mime == "text/plain")
}

/// `magic.from_buffer`'s answer passes datatrove's second check.
pub fn magic_mime_ok(mime: &str, conversion: bool) -> bool {
    mime_ok(mime, conversion)
}

/// The record loop of datatrove's `WarcReader.read_file` over an [`ArchiveReader`].
#[derive(Debug)]
pub struct DocReader {
    reader: ArchiveReader,
    next_index: usize,
    record: Option<Record>,
    content: Vec<u8>,
}

impl DocReader {
    /// `ArchiveIterator(f)` with datatrove's arguments (the defaults) over a file at `tell`.
    pub fn new(opts: ReaderOptions, tell: i64) -> Self {
        Self {
            reader: ArchiveReader::new(opts, tell),
            next_index: 0,
            record: None,
            content: Vec::new(),
        }
    }

    /// The iterator (for its messages, offsets and options).
    pub fn reader(&mut self) -> &mut ArchiveReader {
        &mut self.reader
    }

    /// The next record `process_record` gets past its record-type and payload-type checks,
    /// with its content read; `None` at the end of the file. After an error, `None`.
    pub fn next_doc(&mut self, src: &mut dyn Source) -> Result<Option<Doc<'_>>, Error> {
        self.record = None;
        loop {
            let Some(rec) = self.reader.next_record(src)? else {
                return Ok(None);
            };
            let index = self.next_index;
            self.next_index += 1;
            let conversion = match rec.rec_type.as_deref() {
                Some("response") => false,
                Some("conversion") => true,
                _ => continue,
            };
            if rec
                .rec_headers
                .get_lower("warc-identified-payload-type")
                .is_some_and(|m| !mime_ok(m, conversion))
            {
                continue;
            }
            self.content.clear();
            let mut cs = self.reader.content_stream(&rec);
            self.reader
                .stream_read(&mut cs, src, None, &mut self.content)?;
            let rec = self.record.insert(rec);
            return Ok(Some(make_doc(rec, index, conversion, &self.content)));
        }
    }
}

fn make_doc<'a>(rec: &'a Record, index: usize, conversion: bool, content: &'a [u8]) -> Doc<'a> {
    let h = &rec.rec_headers;
    let url = match h.get_lower("warc-target-uri").filter(|u| !u.is_empty()) {
        Some(u) => Ok(u),
        None => h.dict_get("uri").ok_or(KEY_ERROR_URI),
    };
    let date = match h.get_lower("warc-date").filter(|d| !d.is_empty()) {
        Some(d) => Ok(d),
        None => h.dict_get("archive-date").ok_or(KEY_ERROR_ARCHIVE_DATE),
    };
    Doc {
        index,
        conversion,
        id: h.get_lower("warc-record-id"),
        url,
        date,
        mime_type: h.get_lower("warc-identified-payload-type"),
        content_type: rec
            .http_headers
            .as_ref()
            .and_then(|hh| hh.get_lower("content-type")),
        content,
        utf8: simdutf8::basic::from_utf8(content).is_ok(),
        record: rec,
    }
}

/// A nullable Arrow `large_utf8`/`large_binary` column under construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VarColumn {
    /// `len + 1` offsets.
    pub offsets: Vec<i64>,
    /// The concatenated values.
    pub data: Vec<u8>,
    /// Per row: not null.
    pub valid: Vec<bool>,
}

impl Default for VarColumn {
    fn default() -> Self {
        Self {
            offsets: vec![0],
            data: Vec::new(),
            valid: Vec::new(),
        }
    }
}

impl VarColumn {
    /// Append a value (or a null).
    pub fn push(&mut self, v: Option<&[u8]>) {
        if let Some(b) = v {
            self.data.extend_from_slice(b);
        }
        self.valid.push(v.is_some());
        self.offsets
            .push(i64::try_from(self.data.len()).expect("column under 2^63 bytes"));
    }

    /// Rows.
    pub fn len(&self) -> usize {
        self.valid.len()
    }

    /// No rows.
    pub fn is_empty(&self) -> bool {
        self.valid.is_empty()
    }

    /// Row `i`.
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        let s = usize::try_from(self.offsets[i]).expect("offset");
        let e = usize::try_from(self.offsets[i + 1]).expect("offset");
        self.valid[i].then(|| &self.data[s..e])
    }
}

/// [`Doc`]s collected as columns, for Arrow batches.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Rows {
    /// `ri` of each row.
    pub index: Vec<u64>,
    /// `id` (`WARC-Record-ID`).
    pub id: VarColumn,
    /// `url` (null when it raised KeyError: see `key_error`).
    pub url: VarColumn,
    /// `date` (null when it raised KeyError).
    pub date: VarColumn,
    /// `WARC-Identified-Payload-Type` (null: datatrove asks magic).
    pub mime_type: VarColumn,
    /// The HTTP `Content-Type`.
    pub content_type: VarColumn,
    /// The content bytes.
    pub html: VarColumn,
    /// The content is valid UTF-8.
    pub utf8: Vec<bool>,
    /// The record is a `conversion` record.
    pub conversion: Vec<bool>,
    /// The KeyError datatrove raises for the row (url's first, then date's), if any.
    pub key_error: Vec<Option<&'static str>>,
}

impl Rows {
    /// Append a document.
    pub fn push(&mut self, d: &Doc<'_>) {
        self.index.push(d.index as u64);
        self.id.push(d.id.map(str::as_bytes));
        self.url.push(d.url.ok().map(str::as_bytes));
        self.date.push(d.date.ok().map(str::as_bytes));
        self.mime_type.push(d.mime_type.map(str::as_bytes));
        self.content_type.push(d.content_type.map(str::as_bytes));
        self.html.push(Some(d.content));
        self.utf8.push(d.utf8);
        self.conversion.push(d.conversion);
        self.key_error.push(d.url.err().or(d.date.err()));
    }

    /// Rows.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// No rows.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Bytes held in the variable-size columns.
    pub fn data_bytes(&self) -> usize {
        self.id.data.len()
            + self.url.data.len()
            + self.date.data.len()
            + self.mime_type.data.len()
            + self.content_type.data.len()
            + self.html.data.len()
    }
}
