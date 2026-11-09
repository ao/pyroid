//! Single-thread time per document of `trafilatura::extract` from the raw string, against which
//! `.scratch/traf/time_ref.py` times `trafilatura.extract(html_str, ...)`.
//!
//! Not a correctness test, and ignored by default. Run it in the dev container with
//! optimizations:
//!
//! ```text
//! PYROID_HTMLX_DATA=/target/cc PYROID_TRAF_SPLIT=eval PYROID_TRAF_MODE=datatrove \
//!     PYROID_TRAF_VERSION=1.11.0 \
//!     cargo test -p pyroid-core --release --test trafilatura_bench -- --ignored --nocapture
//! ```
//!
//! Each round runs every document in order with a fresh dedup cache; a document's time is its
//! minimum over the rounds (`PYROID_TRAF_ROUNDS`, default 5), split into `load_html` and the
//! extraction.

#![allow(clippy::print_stderr, clippy::cast_precision_loss)] // a benchmark report

use std::hint::black_box;
use std::io::Read;
use std::time::Instant;

use pyroid_core::trafilatura::{self, Focus, LruCache, Options, Version};

fn read_bin(path: &str) -> Vec<Vec<u8>> {
    let mut b = Vec::new();
    std::fs::File::open(path)
        .unwrap()
        .read_to_end(&mut b)
        .unwrap();
    let at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    let mut i = 4;
    for _ in 0..at(0) {
        let k = at(i);
        out.push(b[i + 4..i + 4 + k].to_vec());
        i += 4 + k;
    }
    out
}

#[test]
#[ignore = "benchmark; set PYROID_HTMLX_DATA and run with --ignored"]
fn extract_time_per_document() {
    let data = std::env::var("PYROID_HTMLX_DATA").expect("set PYROID_HTMLX_DATA");
    let split = std::env::var("PYROID_TRAF_SPLIT").unwrap_or_else(|_| "eval".into());
    let mode = std::env::var("PYROID_TRAF_MODE").unwrap_or_else(|_| "datatrove".into());
    let rounds: usize = std::env::var("PYROID_TRAF_ROUNDS").map_or(5, |v| v.parse().unwrap());
    let (opts, dedup) = match mode.as_str() {
        "datatrove" => (Options::datatrove(), true),
        "default" => (Options::default(), false),
        "fast" => (
            Options {
                fast: true,
                focus: Focus::Precision,
                comments: false,
                ..Options::default()
            },
            false,
        ),
        m => panic!("PYROID_TRAF_MODE {m}"),
    };
    let version = match std::env::var("PYROID_TRAF_VERSION").as_deref() {
        Err(_) | Ok("2.3.1") => Version::V2_3_1,
        Ok("1.11.0") => Version::V1_11_0,
        Ok(x) => panic!("PYROID_TRAF_VERSION {x}"),
    };
    let opts = Options { version, ..opts };
    let docs = read_bin(&format!("{data}/{split}/raw.bin"));
    let n = docs.len();
    let (mut load, mut total) = (vec![f64::INFINITY; n], vec![f64::INFINITY; n]);
    for _ in 0..rounds {
        let mut cache = LruCache::default();
        for (k, html) in docs.iter().enumerate() {
            let t0 = Instant::now();
            let tree = trafilatura::load_html_for(black_box(html), opts.version);
            let t1 = Instant::now();
            let out = match tree {
                Some(t) => {
                    trafilatura::extract_tree(t, &opts, dedup.then_some(&mut cache)).unwrap()
                }
                None => None,
            };
            let t2 = Instant::now();
            black_box(out);
            load[k] = load[k].min((t1 - t0).as_secs_f64());
            total[k] = total[k].min((t2 - t0).as_secs_f64());
        }
    }
    let ms = |v: &[f64]| 1000.0 * v.iter().sum::<f64>() / n as f64;
    eprintln!(
        "{split} {mode} ({version:?}), {n} docs, min of {rounds}: load_html {:.3} + extraction {:.3} = {:.3} ms/doc",
        ms(&load),
        ms(&total) - ms(&load),
        ms(&total)
    );
}
