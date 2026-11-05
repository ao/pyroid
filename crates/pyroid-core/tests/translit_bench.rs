//! Throughput of the translit kernels alone (no Python), on corpora exported
//! as NUL-separated UTF-8 files. Ignored by default:
//!
//! `PYROID_TRANSLIT_CORPUS=/target/corpus cargo test --release -p pyroid-core --test translit_bench -- --ignored --nocapture`

#![allow(clippy::print_stderr, clippy::cast_precision_loss)] // a benchmark report

use std::time::Instant;

use pyroid_core::translit::{Missing, anyascii, unidecode};

#[test]
#[ignore = "benchmark; needs PYROID_TRANSLIT_CORPUS"]
fn kernel_throughput() {
    let Ok(dir) = std::env::var("PYROID_TRANSLIT_CORPUS") else {
        return;
    };
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("corpus dir")
        .map(|e| e.expect("entry").path())
        .collect();
    paths.sort();
    for path in paths {
        let data = std::fs::read_to_string(&path).expect("UTF-8 corpus");
        let docs: Vec<&str> = data.split('\0').collect();
        let chars = data.chars().filter(|c| !c.is_ascii()).count();
        let mut best = [f64::MAX; 2];
        for _ in 0..5 {
            let t = Instant::now();
            let n: usize = docs.iter().map(|d| anyascii(d).len()).sum();
            best[0] = best[0].min(t.elapsed().as_secs_f64());
            let t = Instant::now();
            let m: usize = docs.iter().map(|d| unidecode(d, Missing::Drop).len()).sum();
            best[1] = best[1].min(t.elapsed().as_secs_f64());
            assert_eq!(n, m);
        }
        let (mb, nd, nc) = (data.len() as f64 / 1e6, docs.len() as f64, chars as f64);
        eprintln!(
            "{:>24} {:>6} docs {:>6.1} MB {:>8} non-ASCII chars | anyascii {:>7.2} us/doc {:>6.0} MB/s {:>5.2} ns/char | unidecode {:>7.2} us/doc",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            docs.len(),
            mb,
            chars,
            best[0] / nd * 1e6,
            mb / best[0],
            best[0] / nc.max(1.0) * 1e9,
            best[1] / nd * 1e6,
        );
    }
}
