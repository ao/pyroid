//! The warc reader against warcio 1.8.2 (CPython 3.13.18) on warcio's own test files and
//! the synthetic ones in `tests/data/warc/synth/`: every case of `expected.json.gz`
//! (written by `tests/data/warc/gen.py`) runs the same `ArchiveIterator` options and read
//! plan, and must give the same records, headers, content, offsets, digest results,
//! messages (stderr writes and log warnings, in order) and final exception.

use std::fmt::Write as _;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::PathBuf;

use pyroid_core::warc::{
    ArchiveReader, DigestKind, Error, Format, Message, PyExc, PyIntValue, ReadSource,
    ReaderOptions, Record, StatusAndHeaders,
};
use serde_json::{Value, json};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/warc")
}

fn exc_name(k: PyExc) -> &'static str {
    match k {
        PyExc::ArchiveLoadFailed => "ArchiveLoadFailed",
        PyExc::EofError => "EOFError",
        PyExc::BadGzipFile => "BadGzipFile",
        PyExc::ZlibError | PyExc::BrotliError => "error",
        PyExc::AttributeError => "AttributeError",
        PyExc::KeyError => "KeyError",
        PyExc::ValueError => "ValueError",
        PyExc::OsError => "OSError",
        PyExc::BinasciiError => "Error",
        PyExc::TypeError => "TypeError",
        PyExc::OverflowError => "OverflowError",
        PyExc::StatusAndHeadersParserException => "StatusAndHeadersParserException",
    }
}

fn err_json(e: &Error) -> Value {
    match e.py() {
        Some((k, m)) => json!([exc_name(k), m]),
        None => json!(["<source>", ""]),
    }
}

fn sah(h: Option<&StatusAndHeaders>, skip_id: bool) -> Value {
    let Some(h) = h else {
        return Value::Null;
    };
    let headers: Vec<Value> = h
        .headers()
        .map(|(n, v)| {
            if skip_id && n == "WARC-Record-ID" {
                json!([n, "<id>"])
            } else {
                json!([n, v])
            }
        })
        .collect();
    json!({"protocol": h.protocol, "statusline": h.statusline, "headers": headers, "total_len": h.total_len})
}

fn int_str(v: &PyIntValue) -> String {
    v.to_decimal()
}

fn summary(b: &[u8]) -> Value {
    if b.len() <= 64 {
        let mut s = String::new();
        for x in b {
            let _ = write!(s, "{x:02x}");
        }
        json!({"hex": s})
    } else {
        use sha1::Digest as _;
        let d = sha1::Sha1::digest(b);
        let mut s = String::new();
        for x in d {
            let _ = write!(s, "{x:02x}");
        }
        json!({"len": b.len(), "sha1": s})
    }
}

const CHUNKS: [i128; 7] = [1, 2, 3, 5, 8, 13, 16384];

/// One plan's reads on a record; `Ok(None)` when the plan reads nothing.
fn run_plan(
    a: &mut ArchiveReader,
    src: &mut ReadSource<std::fs::File>,
    rec: &Record,
    plan: &str,
) -> Result<Option<Vec<u8>>, Error> {
    let mut out = Vec::new();
    match plan {
        "none" | "offsets" => return Ok(None),
        "raw" => a.raw_read(&rec.raw, src, None, &mut out)?,
        "content" | "stale" => {
            let mut cs = a.content_stream(rec);
            a.stream_read(&mut cs, src, None, &mut out)?;
        }
        "chunks" => {
            let mut cs = a.content_stream(rec);
            let mut i = 0;
            loop {
                let before = out.len();
                a.stream_read(&mut cs, src, Some(CHUNKS[i % CHUNKS.len()]), &mut out)?;
                i += 1;
                if out.len() == before {
                    break;
                }
            }
        }
        "lines" | "lines7" => {
            let n = if plan == "lines7" { Some(7) } else { None };
            let mut cs = a.content_stream(rec);
            loop {
                let before = out.len();
                a.stream_readline(&mut cs, src, n, &mut out)?;
                if out.len() == before {
                    break;
                }
            }
        }
        _ => panic!("plan {plan}"),
    }
    Ok(Some(out))
}

fn drain(a: &mut ArchiveReader, at: usize, events: &mut Vec<Value>) {
    for m in a.take_messages() {
        events.push(match m {
            Message::Stderr(s) => json!(["stderr", at, s]),
            Message::LogWarning(s) => json!(["log", at, s]),
        });
    }
}

fn options(case: &Value) -> ReaderOptions {
    let o = &case["opts"];
    let mut opts = ReaderOptions::default();
    if let Some(b) = o["block_size"].as_i64() {
        opts.block_size = b;
    }
    opts.check_digests = match &o["check_digests"] {
        Value::Bool(true) => DigestKind::Record,
        Value::String(s) if s == "log" => DigestKind::Log,
        Value::String(s) if s == "raise" => DigestKind::Raise,
        _ => DigestKind::Off,
    };
    opts.ensure_http_headers = o["ensure_http_headers"].as_bool().unwrap_or(false);
    opts.no_record_parse = o["no_record_parse"].as_bool().unwrap_or(false);
    opts.verify_http = o["verify_http"].as_bool().unwrap_or(false);
    opts.arc2warc = o["arc2warc"].as_bool().unwrap_or(false);
    opts.known_format = match case["cls"].as_str() {
        Some("WARCIterator") => Some(Format::Warc),
        Some("ARCIterator") => Some(Format::Arc),
        _ => None,
    };
    opts
}

fn run_case(case: &Value) -> Value {
    let path = data_dir().join(case["file"].as_str().unwrap());
    let offset = case["offset"].as_u64().unwrap_or(0);
    let mut file = std::fs::File::open(&path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut src = ReadSource(file);
    let opts = options(case);
    let skip_id = opts.arc2warc;
    let plan = case["plan"].as_str().unwrap();
    let mut a = ArchiveReader::new(opts, i64::try_from(offset).unwrap());
    let mut records = Vec::new();
    let mut events = Vec::new();
    let mut error = Value::Null;
    let mut count = 0usize;
    let mut prev: Option<Record> = None;
    loop {
        let rec = match a.next_record(&mut src) {
            Ok(Some(r)) => r,
            Ok(None) => {
                drain(&mut a, count, &mut events);
                break;
            }
            Err(e) => {
                drain(&mut a, count, &mut events);
                error = err_json(&e);
                break;
            }
        };
        drain(&mut a, count, &mut events);
        count += 1;
        let mut r = json!({
            "format": rec.format.as_str(),
            "rec_type": rec.rec_type,
            "rec_headers": sah(Some(&rec.rec_headers), skip_id),
            "http_headers": sah(rec.http_headers.as_ref(), skip_id),
            "content_type": rec.content_type,
            "length": rec.length.as_ref().map(int_str),
            "payload_length": int_str(&rec.payload_length),
        });
        let early = (|| -> Result<(), Error> {
            if plan == "offsets" {
                let off = a.get_record_offset(&mut src)?;
                let len = a.get_record_length(&mut src)?;
                r["early"] = json!([off, len]);
            }
            if let Some(p) = prev.as_ref().filter(|_| plan == "stale") {
                // reads of the previous record's streams after the iterator moved on
                let mut raw = Vec::new();
                a.raw_read(&p.raw, &mut src, Some(50), &mut raw)?;
                let mut cs = a.content_stream(p);
                let mut content = Vec::new();
                a.stream_read(&mut cs, &mut src, Some(20), &mut content)?;
                r["stale"] = json!([summary(&raw), summary(&content), a.raw_tell(&p.raw)]);
            }
            Ok(())
        })();
        if let Err(e) = early {
            drain(&mut a, count, &mut events);
            error = err_json(&e);
            break;
        }
        let res = (|| -> Result<(), Error> {
            if let Some(data) = run_plan(&mut a, &mut src, &rec, plan)? {
                r["content"] = summary(&data);
            }
            if plan != "none" {
                let off = a.get_record_offset(&mut src)?;
                let len = a.get_record_length(&mut src)?;
                r["offset"] = json!([off, len]);
            }
            Ok(())
        })();
        let dc = rec.digest_checker();
        r["passed"] = json!(dc.passed);
        r["problems"] = json!(dc.problems);
        records.push(r);
        prev = Some(rec);
        drain(&mut a, count, &mut events);
        if let Err(e) = res {
            error = err_json(&e);
            break;
        }
    }
    json!({"records": records, "events": events, "error": error, "err_count": a.err_count(), "final_offset": a.offset()})
}

/// Where `got` and `want` first differ.
fn first_diff(path: &str, got: &Value, want: &Value) -> String {
    match (got, want) {
        (Value::Object(g), Value::Object(w)) => {
            for (k, wv) in w {
                let gv = g.get(k).unwrap_or(&Value::Null);
                if gv != wv {
                    return first_diff(&format!("{path}.{k}"), gv, wv);
                }
            }
            format!("{path}: extra keys in {got}")
        }
        (Value::Array(g), Value::Array(w)) => {
            for (i, wv) in w.iter().enumerate() {
                let gv = g.get(i).unwrap_or(&Value::Null);
                if gv != wv {
                    return first_diff(&format!("{path}[{i}]"), gv, wv);
                }
            }
            format!("{path}: {} items, want {}", g.len(), w.len())
        }
        _ => format!("{path}: got {got}, want {want}"),
    }
}

fn strip_ids(v: &mut Value) {
    for rec in v["records"].as_array_mut().into_iter().flatten() {
        for key in ["rec_headers", "http_headers"] {
            if !rec[key].is_object() {
                continue;
            }
            for h in rec[key]["headers"].as_array_mut().into_iter().flatten() {
                if h[0] == "WARC-Record-ID" {
                    h[1] = json!("<id>");
                }
            }
        }
    }
}

#[test]
fn matches_warcio() {
    let mut gz = Vec::new();
    std::fs::File::open(data_dir().join("expected.json.gz"))
        .unwrap()
        .read_to_end(&mut gz)
        .unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_string(&mut text)
        .unwrap();
    let expected: Value = serde_json::from_str(&text).unwrap();
    let cases = expected["cases"].as_array().unwrap();
    assert!(cases.len() > 800);
    let mut failures = Vec::new();
    let (mut n_records, mut n_events, mut n_errors) = (0, 0, 0);
    for case in cases {
        let got = run_case(case);
        n_records += got["records"].as_array().map_or(0, Vec::len);
        n_events += got["events"].as_array().map_or(0, Vec::len);
        n_errors += usize::from(!got["error"].is_null());
        let mut want = json!({
            "records": case["records"], "events": case["events"], "error": case["error"],
            "err_count": case["err_count"], "final_offset": case["final_offset"],
        });
        if case["opts"]["arc2warc"] == json!(true) {
            strip_ids(&mut want);
        }
        if got != want {
            failures.push(format!(
                "{} opts={} plan={} cls={}: {}",
                case["file"],
                case["opts"],
                case["plan"],
                case["cls"],
                first_diff("", &got, &want)
            ));
        }
    }
    // The fixtures exercise records, messages and errors (counts from warcio).
    assert!(
        n_records > 3000 && n_events > 700 && n_errors > 300,
        "{n_records} {n_events} {n_errors}"
    );
    if !failures.is_empty() {
        let n = failures.len();
        let shown: Vec<String> = failures.into_iter().take(40).collect();
        panic!(
            "{n} of {} cases differ:\n{}",
            cases.len(),
            shown.join("\n\n")
        );
    }
}

fn sha1_hex(b: &[u8]) -> String {
    use sha1::Digest as _;
    sha1::Sha1::digest(b)
        .iter()
        .fold(String::new(), |mut s, x| {
            let _ = write!(s, "{x:02x}");
            s
        })
}

/// `DocReader` against datatrove 0.10.1's `process_record` loop over `open(path, "rb")`
/// (with a `magic` that rejects everything, as libmagic is not in the dev image). Documents
/// whose bytes are not UTF-8 go through cchardet in datatrove, so only their metadata is
/// compared.
#[test]
fn matches_datatrove() {
    let mut gz = Vec::new();
    std::fs::File::open(data_dir().join("expected.json.gz"))
        .unwrap()
        .read_to_end(&mut gz)
        .unwrap();
    let mut text = String::new();
    flate2::read::GzDecoder::new(&gz[..])
        .read_to_string(&mut text)
        .unwrap();
    let expected: Value = serde_json::from_str(&text).unwrap();
    let mut kept = 0;
    for entry in expected["datatrove"].as_array().unwrap() {
        let file = entry["file"].as_str().unwrap();
        let want_docs = entry["docs"].as_array().unwrap();
        let mut src = ReadSource(std::fs::File::open(data_dir().join(file)).unwrap());
        let mut reader = pyroid_core::warc::DocReader::new(ReaderOptions::default(), 0);
        let mut next = 0;
        let mut error = Value::Null;
        loop {
            match reader.next_doc(&mut src) {
                Ok(None) => break,
                Err(e) => {
                    error = err_json(&e);
                    break;
                }
                Ok(Some(d)) => {
                    if d.mime_type.is_none() {
                        continue; // magic.from_buffer
                    }
                    let meta = |url: &str, date: &str| json!([d.index, d.id, url, date]);
                    if !d.utf8 {
                        // cchardet decides; compare what does not depend on it
                        if let Some(w) = want_docs.get(next).filter(|w| w[0] == json!(d.index)) {
                            let (Ok(url), Ok(date)) = (d.url, d.date) else {
                                panic!("{file}: {d:?}")
                            };
                            assert_eq!(
                                meta(url, date),
                                json!(w.as_array().unwrap()[..4]),
                                "{file}"
                            );
                            next += 1;
                        }
                        continue;
                    }
                    let url = match d.url {
                        Ok(u) => u,
                        Err(k) => {
                            error = json!(["KeyError", k]);
                            break;
                        }
                    };
                    let date = match d.date {
                        Ok(v) => v,
                        Err(k) => {
                            error = json!(["KeyError", k]);
                            break;
                        }
                    };
                    let mut got = meta(url, date);
                    got.as_array_mut().unwrap().push(json!(sha1_hex(d.content)));
                    assert_eq!(Some(&got), want_docs.get(next), "{file}");
                    next += 1;
                    kept += 1;
                }
            }
        }
        assert_eq!(next, want_docs.len(), "{file}: documents left");
        assert_eq!(error, entry["error"], "{file}");
    }
    assert!(kept >= 10, "{kept}");
}
