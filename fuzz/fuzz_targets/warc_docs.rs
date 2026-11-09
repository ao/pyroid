//! The datatrove path (`pyroid_core::warc::DocReader`) on arbitrary input, against the
//! plain iterator.
//!
//! Input: `[flags] data...`; flag bit 0 gzips the data as one member per 256 bytes.
//!
//! Checked: no panic; `DocReader` yields exactly the records of `ArchiveReader` that
//! datatrove's `process_record` reads (type `response`/`conversion`, payload type absent or
//! one datatrove takes), in order, with the same index, the same `content_stream().read()`
//! bytes, `utf8 == from_utf8(content).is_ok()`, `id`/`url`/`date` from the headers with
//! datatrove's fallbacks, and the same final error; `Rows` holds what was pushed.

#![no_main]

use std::io::Write as _;

use libfuzzer_sys::fuzz_target;
use pyroid_core::warc::{ArchiveReader, DocReader, ReadSource, ReaderOptions, Rows};

fn gz(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for m in data.chunks(256) {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        e.write_all(m).unwrap();
        out.extend(e.finish().unwrap());
    }
    out
}

struct Expected {
    index: usize,
    content: Vec<u8>,
    id: Option<String>,
    url: Result<String, &'static str>,
    date: Result<String, &'static str>,
}

fn reference(data: &[u8]) -> (Vec<Expected>, Option<String>) {
    let mut src = ReadSource(data);
    let mut a = ArchiveReader::new(ReaderOptions::default(), 0);
    let mut out = Vec::new();
    let mut index = 0;
    loop {
        let rec = match a.next_record(&mut src) {
            Ok(Some(r)) => r,
            Ok(None) => return (out, None),
            Err(e) => return (out, Some(format!("{e:?}"))),
        };
        index += 1;
        let conversion = match rec.rec_type.as_deref() {
            Some("response") => false,
            Some("conversion") => true,
            _ => continue,
        };
        let h = &rec.rec_headers;
        if let Some(m) = h.get_header("WARC-Identified-Payload-Type")
            && !pyroid_core::warc::magic_mime_ok(m, conversion)
        {
            continue;
        }
        let mut content = Vec::new();
        let mut cs = a.content_stream(&rec);
        if let Err(e) = a.stream_read(&mut cs, &mut src, None, &mut content) {
            return (out, Some(format!("{e:?}")));
        }
        let pick = |name: &str, key: &str, err: &'static str| match h
            .get_header(name)
            .filter(|v| !v.is_empty())
        {
            Some(v) => Ok(v.to_owned()),
            None => h.dict_get(key).map(str::to_owned).ok_or(err),
        };
        out.push(Expected {
            index: index - 1,
            content,
            id: h.get_header("WARC-Record-ID").map(str::to_owned),
            url: pick("WARC-Target-URI", "uri", "'uri'"),
            date: pick("WARC-Date", "archive-date", "'archive-date'"),
        });
    }
}

fuzz_target!(|input: &[u8]| {
    let Some((&flags, data)) = input.split_first() else {
        return;
    };
    let data = if flags & 1 != 0 {
        gz(data)
    } else {
        data.to_vec()
    };
    let (want, want_err) = reference(&data);
    let mut src = ReadSource(&data[..]);
    let mut r = DocReader::new(ReaderOptions::default(), 0);
    let mut rows = Rows::default();
    let mut k = 0;
    let err = loop {
        match r.next_doc(&mut src) {
            Ok(None) => break None,
            Err(e) => break Some(format!("{e:?}")),
            Ok(Some(d)) => {
                let w = want.get(k).expect("an extra document");
                k += 1;
                assert_eq!(d.index, w.index);
                assert_eq!(d.content, &w.content[..]);
                assert_eq!(d.utf8, std::str::from_utf8(d.content).is_ok());
                assert_eq!(d.id, w.id.as_deref());
                assert_eq!(d.url, w.url.as_deref().map_err(|e| *e));
                assert_eq!(d.date, w.date.as_deref().map_err(|e| *e));
                rows.push(&d);
                assert_eq!(rows.html.get(rows.len() - 1), Some(d.content));
            }
        }
    };
    assert_eq!(k, want.len(), "documents missing");
    assert_eq!(err, want_err);
    assert_eq!(rows.len(), k);
});
