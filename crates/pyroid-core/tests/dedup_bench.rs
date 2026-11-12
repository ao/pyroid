//! Indicative micro-benchmarks for the MinHash kernel (run in Docker only):
//! `cargo test --release -p pyroid-core --test dedup_bench -- --ignored --nocapture`

#![allow(clippy::print_stderr)] // a benchmark report

use std::hint::black_box;
use std::time::Instant;

use pyroid_core::dedup::{HashConfig, HashFn, MinHasher, MinhashConfig, Precision, UnicodeVersion};

fn hasher(hash_fc: HashFn) -> MinHasher {
    let config = MinhashConfig {
        hash_config: HashConfig {
            precision: Precision::P64,
            hash_fc,
        },
        ..MinhashConfig::default()
    };
    MinHasher::new(config, UnicodeVersion::LATEST).unwrap()
}

#[test]
#[ignore = "benchmark"]
fn permutations_ns_per_element() {
    let h = hasher(HashFn::Xxhash);
    let mut x = 0x9e37_79b9_7f4a_7c15_u64;
    let shingles: Vec<u64> = (0..500)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        })
        .collect();
    let mut out = vec![0; h.num_hashes()];
    let iters = 4000;
    let t = Instant::now();
    for _ in 0..iters {
        black_box(h.signature_from_shingles(black_box(&shingles), &mut out));
    }
    let ns = t.elapsed().as_secs_f64() * 1e9 / f64::from(iters) / (500.0 * 112.0);
    eprintln!("permutations: {ns:.3} ns per shingle x hash");
}

#[test]
#[ignore = "benchmark"]
fn shingle_hash_ns() {
    for (name, fc) in [("sha1", HashFn::Sha1), ("xxhash", HashFn::Xxhash)] {
        let h = hasher(fc);
        let words: Vec<String> = (0..500).map(|i| format!("word{}", i % 97)).collect();
        let mut sh = Vec::new();
        let iters = 2000;
        let t = Instant::now();
        for _ in 0..iters {
            h.shingles_from_tokens(black_box(&words), &mut sh);
            black_box(&sh);
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / f64::from(iters) / 496.0;
        eprintln!("{name}: {ns:.1} ns per 5-gram shingle");
    }
}

#[test]
#[ignore = "benchmark"]
fn sha1_fast_path_vs_digest() {
    use sha1::{Digest, Sha1};
    let msg = b"the quiet river runs past mill";
    let iters = 400_000;
    for round in 0..3 {
        let t = Instant::now();
        for i in 0..iters {
            let mut m = *msg;
            m[0] = u8::try_from(i & 0x7f).unwrap();
            black_box(pyroid_core::dedup::minhash::sha1_prefix64(black_box(&m)));
        }
        let fast = t.elapsed().as_secs_f64() * 1e9 / f64::from(iters);
        let t = Instant::now();
        for i in 0..iters {
            let mut m = *msg;
            m[0] = u8::try_from(i & 0x7f).unwrap();
            black_box(Sha1::digest(black_box(&m)));
        }
        let slow = t.elapsed().as_secs_f64() * 1e9 / f64::from(iters);
        eprintln!(
            "round {round}: block compress {fast:.1} ns, Digest {slow:.1} ns per 30-byte message"
        );
    }
}
