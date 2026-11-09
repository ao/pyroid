//! Time the warc reader on a gzipped WARC held in memory, against raw inflate. Ignored by
//! default:
//!
//! ```sh
//! PYROID_WARC_BENCH=/target/data/cc2639-00000-100m.warc.gz \
//!     cargo test -p pyroid-core --profile release --test warc_bench -- --ignored --nocapture
//! ```
//!
//! Runs (best of `PYROID_WARC_RUNS`, default 3):
//! - `inflate`: zlib-rs through flate2, member by member, into a reused buffer (the floor);
//! - `iterate`: `ArchiveIterator(open(p, 'rb'))`, every record, nothing read (warcio's own
//!   per-member decompression, `read_to_end` skipping the content);
//! - `docs`: datatrove's `process_record` loop (`DocReader`) on the same stream;
//! - `plain-*`: the same over a decompressed stream (flate2's `MultiGzDecoder`, 16 KiB reads,
//!   standing in for `gzip.GzipFile`), minus nothing: `plain-src` is the source alone.

#![allow(clippy::print_stderr, clippy::cast_precision_loss)] // a benchmark report

use std::io::Read as _;
use std::time::Instant;

use pyroid_core::warc::{ArchiveReader, DocReader, ReadSource, ReaderOptions, Source};

fn inflate(data: &[u8]) -> (usize, usize) {
    let mut out = vec![0u8; 1 << 18];
    let (mut pos, mut total, mut members) = (0usize, 0usize, 0usize);
    'outer: while pos < data.len() {
        let mut d = flate2::Decompress::new_gzip(15);
        loop {
            let (bi, bo) = (d.total_in(), d.total_out());
            let st = d.decompress(&data[pos..], &mut out, flate2::FlushDecompress::None);
            pos += usize::try_from(d.total_in() - bi).unwrap();
            total += usize::try_from(d.total_out() - bo).unwrap();
            match st {
                Ok(flate2::Status::StreamEnd) => {
                    members += 1;
                    break;
                }
                Ok(_) if d.total_in() == bi && d.total_out() == bo => break 'outer,
                Ok(_) => {}
                Err(_) => break 'outer,
            }
        }
    }
    (members, total)
}

fn iterate(src: &mut dyn Source) -> usize {
    let mut a = ArchiveReader::new(ReaderOptions::default(), 0);
    let mut n = 0;
    while let Ok(Some(_)) = a.next_record(src) {
        n += 1;
    }
    n
}

fn docs(src: &mut dyn Source) -> (usize, usize, usize) {
    let mut r = DocReader::new(ReaderOptions::default(), 0);
    let (mut n, mut bytes, mut utf8) = (0, 0, 0);
    while let Ok(Some(d)) = r.next_doc(src) {
        n += 1;
        bytes += d.content.len();
        utf8 += usize::from(d.utf8);
    }
    (n, bytes, utf8)
}

/// `read(16384)` until EOF.
fn drain(src: &mut dyn Source) -> usize {
    let mut buf = Vec::with_capacity(1 << 14);
    let mut n = 0;
    loop {
        buf.clear();
        if src.read(1 << 14, &mut buf).is_err() || buf.is_empty() {
            return n;
        }
        n += buf.len();
    }
}

fn best<T: std::fmt::Debug>(label: &str, runs: usize, mut f: impl FnMut() -> T) {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..runs {
        let t = Instant::now();
        last = Some(f());
        times.push(t.elapsed().as_secs_f64());
    }
    let b = times.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!(
        "{label:>14}: best {b:.3} s of {times:.3?} -> {:?}",
        last.unwrap()
    );
}

#[test]
#[ignore = "benchmark; needs PYROID_WARC_BENCH"]
fn reader_time() {
    let Ok(path) = std::env::var("PYROID_WARC_BENCH") else {
        return;
    };
    let runs = std::env::var("PYROID_WARC_RUNS").map_or(3, |s| s.parse().unwrap());
    let mut data = Vec::new();
    std::fs::File::open(&path)
        .unwrap()
        .read_to_end(&mut data)
        .unwrap();
    eprintln!("{path}: {} bytes", data.len());
    best("inflate", runs, || inflate(&data));
    best("iterate", runs, || iterate(&mut ReadSource(&data[..])));
    best("docs", runs, || docs(&mut ReadSource(&data[..])));
    let gz = || ReadSource(flate2::read::MultiGzDecoder::new(&data[..]));
    best("plain-src", runs, || drain(&mut gz()));
    best("plain-iterate", runs, || iterate(&mut gz()));
    best("plain-docs", runs, || docs(&mut gz()));
}
