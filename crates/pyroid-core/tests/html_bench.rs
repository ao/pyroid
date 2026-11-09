//! Time `lxml::fromstring_wtf8` (lxml.html.fromstring on a str: input handling, the parse,
//! the fragment handling, dropping the tree) on the strings of a `.bin` file, as
//! `html_data/lxml_sets.py bench` times lxml on the same file. Ignored by default:
//!
//! ```sh
//! PYROID_HTMLX_BENCH=/target/cc/dev/repaired.bin \
//!     cargo test -p pyroid-core --release --test html_bench -- --ignored --nocapture
//! ```
//!
//! Prints per run the total and the mean per document, then the best run and the sum over
//! documents of each one's fastest time, and the share of the drop.

#![allow(clippy::print_stderr, clippy::cast_precision_loss)] // a benchmark report

use std::io::Read;
use std::time::Instant;

use pyroid_core::html::lxml::fromstring_wtf8;

fn read_bin(path: &str) -> std::io::Result<Vec<Vec<u8>>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut u = [0u8; 4];
    f.read_exact(&mut u)?;
    let n = u32::from_le_bytes(u) as usize;
    let mut docs = Vec::with_capacity(n);
    for _ in 0..n {
        f.read_exact(&mut u)?;
        let mut b = vec![0u8; u32::from_le_bytes(u) as usize];
        f.read_exact(&mut b)?;
        docs.push(b);
    }
    Ok(docs)
}

#[test]
#[ignore = "benchmark; needs PYROID_HTMLX_BENCH"]
fn parse_time() {
    let Ok(path) = std::env::var("PYROID_HTMLX_BENCH") else {
        return;
    };
    let runs: usize = std::env::var("PYROID_HTMLX_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let docs = read_bin(&path).expect("bin file");
    let bytes: usize = docs.iter().map(Vec::len).sum();
    let n = docs.len() as f64;
    let mut per_doc = vec![f64::INFINITY; docs.len()];
    let mut best = f64::INFINITY;
    let mut nodes = 0usize;
    for r in 0..runs {
        let t0 = Instant::now();
        for (i, s) in docs.iter().enumerate() {
            let t = Instant::now();
            let d = fromstring_wtf8(s);
            nodes += d.map_or(0, |d| d.nodes.len());
            per_doc[i] = per_doc[i].min(t.elapsed().as_secs_f64());
        }
        let total = t0.elapsed().as_secs_f64();
        best = best.min(total);
        eprintln!("run {r}: {total:.3} s total, {:.3} ms/doc", 1e3 * total / n);
    }
    let (mut tp, mut td) = (0.0, 0.0);
    for s in &docs {
        let t = Instant::now();
        let d = fromstring_wtf8(s);
        tp += t.elapsed().as_secs_f64();
        let t = Instant::now();
        drop(d);
        td += t.elapsed().as_secs_f64();
    }
    eprintln!(
        "html: {} docs, {:.1} MB; best run {best:.3} s = {:.3} ms/doc ({:.1} MB/s); sum of \
         per-doc minima {:.3} ms/doc; one pass: parse {:.3} + drop {:.3} ms/doc [{nodes}]",
        docs.len(),
        bytes as f64 / 1e6,
        1e3 * best / n,
        bytes as f64 / 1e6 / best,
        1e3 * per_doc.iter().sum::<f64>() / n,
        1e3 * tp / n,
        1e3 * td / n,
    );
}
