//! Single-thread throughput of `ftfy::fix_text` on a recorded corpus.
//!
//! Not a correctness test, and ignored by default. Run it in the dev container
//! with optimizations, pointing it at a corpus recorded by
//! `tests/ftfy_data/oracle.py corpus`:
//!
//! ```text
//! PYROID_FTFY_BENCH=/target/oracle/corpus-fw100k-15.0.0.jsonl \
//!     cargo test -p pyroid-core --profile profiling --test ftfy_bench -- --ignored --nocapture
//! ```

#![allow(clippy::print_stderr, clippy::cast_precision_loss)] // a benchmark report

use std::hint::black_box;
use std::time::Instant;

use pyroid_core::ftfy::{TextFixerConfig, fix_text};
use serde_json::Value;

#[test]
#[ignore = "benchmark; set PYROID_FTFY_BENCH and run with --ignored"]
fn fix_text_throughput() {
    let path = std::env::var("PYROID_FTFY_BENCH").expect("set PYROID_FTFY_BENCH");
    let limit: usize =
        std::env::var("PYROID_FTFY_BENCH_DOCS").map_or(usize::MAX, |v| v.parse().unwrap());
    let docs: Vec<String> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .take(limit)
        .filter_map(|l| {
            let row: Value = serde_json::from_str(l).unwrap();
            row["in"].as_str().map(str::to_owned)
        })
        .collect();
    let bytes: usize = docs.iter().map(String::len).sum();
    let cfg = TextFixerConfig {
        explain: false,
        ..TextFixerConfig::default()
    };

    // Warm up, then take the best of several passes.
    let mut changed = 0;
    for d in &docs {
        changed += usize::from(matches!(fix_text(d, &cfg), std::borrow::Cow::Owned(_)));
    }
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for d in &docs {
            black_box(fix_text(black_box(d), &cfg));
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    eprintln!(
        "{} docs, {:.1} MB, {changed} changed: best {:.3} s = {:.2} us/doc, {:.0} MB/s",
        docs.len(),
        bytes as f64 / 1e6,
        best,
        best * 1e6 / docs.len() as f64,
        bytes as f64 / 1e6 / best
    );
}
