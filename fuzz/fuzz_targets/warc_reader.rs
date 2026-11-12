//! The warc reader (`pyroid_core::warc::ArchiveReader`) on arbitrary input.
//!
//! Input: `[options] [block size] [plan: 4 bytes] data...`.
//! - options bits: 0 no_record_parse, 1 verify_http, 2 arc2warc, 3 ensure_http_headers,
//!   4-5 check_digests (off, record, log, raise), 6 gzip the data in members (split at the
//!   bytes of the plan, one member byte flipped when bit 7 is set too), 7 WARCIterator.
//! - block size: `1 + b * 61` (small blocks reach every buffer boundary).
//! - plan: read sizes (`1 + p % 61`) cycled for the chunked run.
//!
//! Checked:
//! - no panic; heap stays under 64 MiB + 4 KiB per input byte, so nothing is allocated from
//!   a header value (`Content-Length: 99999999999`, chunk sizes, a `block_size`);
//! - determinism: the same input gives the same records, contents, messages and error;
//! - reading each record's content stream in the plan's chunk sizes gives the same bytes,
//!   messages and outcome as one `read()` (warcio fills its buffers at the same moments
//!   either way), unless the whole read raised OverflowError (a length past `sys.maxsize`,
//!   which only a bounded read avoids);

#![no_main]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

use libfuzzer_sys::fuzz_target;
use pyroid_core::warc::{
    ArchiveReader, DigestKind, Error, Format, Message, PyExc, ReadSource, ReaderOptions,
};

/// Counts live heap bytes and their peak.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator unchanged; only counts sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        // SAFETY: the caller's contract for `alloc` is passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` came from `alloc` with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            let live = LIVE.fetch_add(new_size - layout.size(), Ordering::Relaxed) + new_size
                - layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        } else {
            LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
        }
        // SAFETY: the caller's contract for `realloc` is passed on.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    records: Vec<(String, Option<String>, usize, Vec<u8>, Option<(i64, i64)>)>,
    messages: Vec<Message>,
    error: Option<(Option<PyExc>, String)>,
    err_count: u64,
}

fn err(e: &Error) -> (Option<PyExc>, String) {
    match e.py() {
        Some((k, m)) => (Some(k), m.to_owned()),
        None => (None, String::new()),
    }
}

fn run(opts: &ReaderOptions, data: &[u8], plan: Option<&[u8]>) -> Outcome {
    let mut src = ReadSource(data);
    let mut a = ArchiveReader::new(opts.clone(), 0);
    let mut out = Outcome {
        records: Vec::new(),
        messages: Vec::new(),
        error: None,
        err_count: 0,
    };
    let mut i = 0usize;
    loop {
        let rec = match a.next_record(&mut src) {
            Ok(Some(r)) => r,
            Ok(None) => break,
            Err(e) => {
                out.error = Some(err(&e));
                break;
            }
        };
        out.messages.extend(a.take_messages());
        let mut content = Vec::new();
        let mut cs = a.content_stream(&rec);
        let r = match plan {
            None => a.stream_read(&mut cs, &mut src, None, &mut content),
            Some(p) => loop {
                let n = 1 + i128::from(p[i % p.len()] % 61);
                i += 1;
                let before = content.len();
                if let Err(e) = a.stream_read(&mut cs, &mut src, Some(n), &mut content) {
                    break Err(e);
                }
                if content.len() == before {
                    break Ok(());
                }
            },
        };
        let offsets = r.as_ref().ok().and_then(|()| {
            let o = a.get_record_offset(&mut src).ok()?;
            let l = a.get_record_length(&mut src).ok()?;
            Some((o, l))
        });
        out.messages.extend(a.take_messages());
        out.records.push((
            rec.format.as_str().to_owned(),
            rec.rec_type.clone(),
            rec.rec_headers.len(),
            content,
            offsets,
        ));
        if let Err(e) = r {
            // The bytes of a failed read() are lost; chunks before the failing one are not.
            out.records.last_mut().unwrap().3.clear();
            out.error = Some(err(&e));
            break;
        }
    }
    out.messages.extend(a.take_messages());
    out.err_count = a.err_count();
    out
}

fn gzip_members(data: &[u8], cuts: &[u8], flip: bool) -> Vec<u8> {
    let mut out = Vec::new();
    let mut rest = data;
    for (k, &c) in cuts.iter().enumerate() {
        let n = (usize::from(c) * 7).min(rest.len());
        let (m, r) = rest.split_at(n);
        rest = r;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        e.write_all(m).unwrap();
        let mut g = e.finish().unwrap();
        if flip && k == 1 && g.len() > 12 {
            let i = 10 + usize::from(c) % (g.len() - 10);
            g[i] ^= 0x55;
        }
        out.extend(g);
    }
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    e.write_all(rest).unwrap();
    out.extend(e.finish().unwrap());
    out
}

fuzz_target!(|input: &[u8]| {
    let [flags, bs, p0, p1, p2, p3, data @ ..] = input else {
        return;
    };
    let plan = [*p0, *p1, *p2, *p3];
    let opts = ReaderOptions {
        no_record_parse: flags & 1 != 0,
        verify_http: flags & 2 != 0,
        arc2warc: flags & 4 != 0,
        ensure_http_headers: flags & 8 != 0,
        check_digests: match (flags >> 4) & 3 {
            0 => DigestKind::Off,
            1 => DigestKind::Record,
            2 => DigestKind::Log,
            _ => DigestKind::Raise,
        },
        known_format: (flags & 0xc0 == 0x80).then_some(Format::Warc),
        block_size: 1 + i64::from(*bs) * 61,
        make_warc_id: || "<urn:uuid:fuzz>".to_owned(),
        ..ReaderOptions::default()
    };
    let data: Vec<u8> = if flags & 0x40 != 0 {
        gzip_members(data, &plan, flags & 0x80 != 0)
    } else {
        data.to_vec()
    };
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    let whole = run(&opts, &data, None);
    let again = run(&opts, &data, None);
    assert_eq!(whole, again, "not deterministic");
    let chunked = run(&opts, &data, Some(&plan));
    let overflow = whole
        .error
        .as_ref()
        .is_some_and(|(k, _)| *k == Some(PyExc::OverflowError));
    if !overflow {
        assert_eq!(whole, chunked, "chunked reads differ from read()");
    }
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(base);
    let bound = (64 << 20) + 4096 * data.len();
    assert!(peak <= bound, "heap peak {peak} > {bound}");
});
