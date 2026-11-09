//! The reader's contract with its callers, beyond what `warc_oracle.rs` compares: source
//! errors and `EOFError`s, the dead-generator rule, `close()`, offsets before any record,
//! and huge lengths.

use std::io::Write as _;

use pyroid_core::warc::{
    ArchiveReader, DocReader, EofPayload, Error, PyExc, PyIntValue, ReadSource, ReaderOptions,
    Source,
};

fn record(rec_type: &str, uri: &str, block: &[u8]) -> Vec<u8> {
    let mut v = format!(
        "WARC/1.0\r\nWARC-Type: {rec_type}\r\nWARC-Target-URI: {uri}\r\n\
         WARC-Record-ID: <urn:uuid:x>\r\nContent-Length: {}\r\n\r\n",
        block.len()
    )
    .into_bytes();
    v.extend_from_slice(block);
    v.extend_from_slice(b"\r\n\r\n");
    v
}

fn gz(b: &[u8]) -> Vec<u8> {
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    e.write_all(b).unwrap();
    e.finish().unwrap()
}

/// A source that serves `data` and then fails with `err` on read number `fail_at`.
struct Failing {
    data: Vec<u8>,
    pos: usize,
    reads: usize,
    fail_at: usize,
    eof: bool,
}

impl Source for Failing {
    fn read(&mut self, n: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        self.reads += 1;
        if self.reads == self.fail_at {
            let payload: Box<dyn std::any::Any + Send> = Box::new("boom");
            return Err(Error::Source(if self.eof {
                Box::new(EofPayload(payload))
            } else {
                payload
            }));
        }
        let end = self.data.len().min(self.pos + n);
        out.extend_from_slice(&self.data[self.pos..end]);
        self.pos = end;
        Ok(())
    }
}

fn three_records() -> Vec<u8> {
    let mut d = record("warcinfo", "urn:a", b"info");
    d.extend(record("resource", "urn:b", &[b'x'; 5000]));
    d.extend(record("resource", "urn:c", b"last"));
    d
}

#[test]
fn source_error_is_passed_through_and_ends_iteration() {
    let opts = ReaderOptions {
        block_size: 1000,
        ..ReaderOptions::default()
    };
    let mut src = Failing {
        data: three_records(),
        pos: 0,
        reads: 0,
        fail_at: 3,
        eof: false,
    };
    let mut a = ArchiveReader::new(opts, 0);
    assert!(a.next_record(&mut src).unwrap().is_some());
    let rec = a.next_record(&mut src).unwrap().unwrap();
    let mut out = Vec::new();
    let e = a.raw_read(&rec.raw, &mut src, None, &mut out).unwrap_err();
    match e {
        Error::Source(p) => assert_eq!(p.downcast_ref::<&str>(), Some(&"boom")),
        Error::Py(..) => panic!("{e:?}"),
    }
    assert!(out.is_empty(), "the failed read keeps nothing");
    // A later read goes on from where the reader is (warcio lost the bytes of that call).
    a.raw_read(&rec.raw, &mut src, None, &mut out).unwrap();
    assert!(!out.is_empty() && out.len() < 5000);
    // An error inside next() kills the generator.
    let mut src = Failing {
        data: three_records(),
        pos: 0,
        reads: 0,
        fail_at: 2,
        eof: false,
    };
    let mut a = ArchiveReader::new(
        ReaderOptions {
            block_size: 1000,
            ..ReaderOptions::default()
        },
        0,
    );
    assert!(a.next_record(&mut src).unwrap().is_some());
    // The second record's headers are in the first read; skipping its content is not.
    assert!(a.next_record(&mut src).unwrap().is_some());
    assert!(matches!(a.next_record(&mut src), Err(Error::Source(_))));
    assert!(a.is_exhausted());
    assert!(a.next_record(&mut src).unwrap().is_none());
}

#[test]
fn eof_error_from_the_source() {
    // While parsing the next record's headers (the first read ends right after its
    // "WARC/1.0\r\n"): the end of iteration (plain WARC).
    let data = three_records();
    let first = record("warcinfo", "urn:a", b"info").len() + "WARC/1.0\r\n".len();
    let mut src = Failing {
        data,
        pos: 0,
        reads: 0,
        fail_at: 2,
        eof: true,
    };
    let mut a = ArchiveReader::new(
        ReaderOptions {
            block_size: i64::try_from(first).unwrap(),
            ..ReaderOptions::default()
        },
        0,
    );
    assert!(a.next_record(&mut src).unwrap().is_some());
    assert!(a.next_record(&mut src).unwrap().is_none());
    assert!(a.is_closed() && a.is_exhausted());
    // While reading content: raised to the caller as an EOFError.
    let mut src = Failing {
        data: three_records(),
        pos: 0,
        reads: 0,
        fail_at: 2,
        eof: true,
    };
    let mut a = ArchiveReader::new(ReaderOptions::default(), 0);
    let mut out = Vec::new();
    let _ = a.next_record(&mut src).unwrap().unwrap();
    let rec = a.next_record(&mut src).unwrap().unwrap();
    // The whole file came in one read; the next read is the failing one, after the end.
    a.raw_read(&rec.raw, &mut src, None, &mut out).unwrap();
    assert_eq!(out.len(), 5000);
    let _ = a.next_record(&mut src).unwrap().unwrap();
    let e = a.next_record(&mut src).unwrap_err();
    assert!(e.is_eof_error(), "{e:?}");
}

#[test]
fn offsets_and_close() {
    let mut data = Vec::new();
    for r in [
        record("warcinfo", "urn:a", b"info"),
        record("resource", "urn:b", b"bb"),
    ] {
        data.extend(gz(&r));
    }
    let mut src = ReadSource(&data[..]);
    let mut a = ArchiveReader::new(ReaderOptions::default(), 0);
    let e = a.get_record_offset(&mut src).unwrap_err();
    assert_eq!(
        e.py(),
        Some((PyExc::TypeError, "'NoneType' object is not subscriptable"))
    );
    let _ = a.next_record(&mut src).unwrap().unwrap();
    assert_eq!(a.get_record_offset(&mut src).unwrap(), 0);
    let first_len = a.get_record_length(&mut src).unwrap();
    assert_eq!(
        first_len,
        i64::try_from(gz(&record("warcinfo", "urn:a", b"info")).len()).unwrap()
    );
    let _ = a.next_record(&mut src).unwrap().unwrap();
    assert_eq!(a.get_record_offset(&mut src).unwrap(), first_len);
    assert!(a.next_record(&mut src).unwrap().is_none());
    assert!(a.is_closed() && !a.has_record());
    // read_to_end is a no-op now; the offsets of the last record stay.
    a.read_to_end(&mut src).unwrap();
    assert_eq!(a.member_info().unwrap().0, first_len);
    assert_eq!(a.offset(), i64::try_from(data.len()).unwrap());
    a.close().unwrap();
}

#[test]
fn huge_and_odd_lengths() {
    let mut data = b"WARC/1.0\r\nWARC-Type: resource\r\nWARC-Target-URI: urn:x\r\n\
        Content-Length: 99999999999999999999999999999999999999999\r\n\r\nshort"
        .to_vec();
    data.extend_from_slice(b"\r\n\r\n");
    let mut src = ReadSource(&data[..]);
    let mut a = ArchiveReader::new(ReaderOptions::default(), 0);
    let rec = a.next_record(&mut src).unwrap().unwrap();
    assert_eq!(rec.length, Some(PyIntValue::Big("9".repeat(41))));
    let mut out = Vec::new();
    // A bounded read is fine; read() asks BytesIO for a size past sys.maxsize.
    a.raw_read(&rec.raw, &mut src, Some(3), &mut out).unwrap();
    assert_eq!(out, b"sho");
    let e = a.raw_read(&rec.raw, &mut src, None, &mut out).unwrap_err();
    assert_eq!(e.py().unwrap().0, PyExc::OverflowError);
    assert_eq!(out, b"sho");
}

#[test]
fn doc_reader_keeps_what_datatrove_keeps() {
    let mut data = Vec::new();
    let page = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<html>ok</html>";
    let mut resp = String::from_utf8(record("response", "http://e/", page)).unwrap();
    resp = resp.replacen(
        "Content-Length",
        "WARC-Identified-Payload-Type: text/html\r\nContent-Length",
        1,
    );
    data.extend(record("request", "http://e/", b"GET / HTTP/1.1\r\n\r\n"));
    data.extend(resp.as_bytes());
    data.extend(record("response", "http://e/", page));
    data.extend(
        resp.replace("text/html\r\nContent-Length", "image/png\r\nContent-Length")
            .as_bytes(),
    );
    let mut src = ReadSource(&data[..]);
    let mut r = DocReader::new(ReaderOptions::default(), 0);
    let d = r.next_doc(&mut src).unwrap().unwrap();
    assert_eq!((d.index, d.mime_type, d.utf8), (1, Some("text/html"), true));
    assert_eq!(d.content, b"<html>ok</html>");
    assert_eq!(d.content_type, Some("text/html"));
    assert_eq!((d.url, d.id), (Ok("http://e/"), Some("<urn:uuid:x>")));
    assert_eq!(d.date, Err("'archive-date'"));
    let d = r.next_doc(&mut src).unwrap().unwrap();
    assert_eq!((d.index, d.mime_type), (2, None));
    assert!(r.next_doc(&mut src).unwrap().is_none());
    let mut rows = pyroid_core::warc::Rows::default();
    assert!(rows.is_empty());
    let mut src = ReadSource(&data[..]);
    let mut r = DocReader::new(ReaderOptions::default(), 0);
    while let Some(d) = r.next_doc(&mut src).unwrap() {
        rows.push(&d);
    }
    assert_eq!(rows.len(), 2);
    assert_eq!(rows.html.get(1), Some(&b"<html>ok</html>"[..]));
    assert_eq!(rows.mime_type.get(1), None);
    assert_eq!(rows.key_error, [Some("'archive-date'"); 2]);
}
