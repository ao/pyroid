//! Kernel timings for `pyroid_core::filters` (no Python): run with
//! `PYROID_FILTERS_BENCH=texts.jsonl cargo test --profile profiling -p pyroid-core --test filters_bench -- --nocapture`
//! where each line of `texts.jsonl` is a JSON string. `PYROID_FILTERS_BENCH_LANG`
//! (a datatrove language code, default `eng`) picks the tokenizer. Indicative only.

#![allow(clippy::print_stderr)]

use std::time::Instant;

use pyroid_core::filters::{
    C4Quality, Doc, FineWebQuality, GopherQuality, GopherRepetition, Pipeline, Step, WordTokenizer,
    tokenizer_for,
};
use pyroid_core::ftfy::UnicodeVersion;

fn best_of<F: FnMut()>(n: usize, mut f: F) -> f64 {
    (0..n)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn bench() {
    let Ok(path) = std::env::var("PYROID_FILTERS_BENCH") else {
        return;
    };
    let texts: Vec<String> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let n = texts.len() as f64;
    let lang = std::env::var("PYROID_FILTERS_BENCH_LANG").unwrap_or_else(|_| "eng".to_owned());
    let wt: &'static dyn WordTokenizer =
        tokenizer_for(&lang, UnicodeVersion::V15_1).expect("a language pyroid tokenizes");
    eprintln!("language {lang}, {} docs", texts.len());
    let report = |name: &str, secs: f64| eprintln!("{name:<28} {:>8.1} us/doc", secs / n * 1e6);

    report(
        "tokenize (words)",
        best_of(3, || {
            let mut out = Vec::new();
            for t in &texts {
                out.clear();
                wt.words(t, &mut out).unwrap();
            }
        }),
    );
    let gq = GopherQuality::default();
    let gr = GopherRepetition::default();
    let c4 = C4Quality::default();
    let fw = FineWebQuality::default();
    report(
        "gopher_quality",
        best_of(3, || {
            for t in &texts {
                gq.filter(&mut Doc::new(t, wt)).unwrap();
            }
        }),
    );
    report(
        "gopher_repetition",
        best_of(3, || {
            for t in &texts {
                gr.filter(&mut Doc::new(t, wt)).unwrap();
            }
        }),
    );
    report(
        "c4_quality",
        best_of(3, || {
            for t in &texts {
                c4.filter(t, wt).unwrap();
            }
        }),
    );
    report(
        "fineweb_quality",
        best_of(3, || {
            for t in &texts {
                fw.filter(&mut Doc::new(t, wt)).unwrap();
            }
        }),
    );
    let chain = Pipeline {
        steps: vec![
            (Step::GopherRepetition(gr.clone()), wt),
            (Step::GopherQuality(gq.clone()), wt),
            (
                Step::C4Quality(C4Quality {
                    filter_no_terminal_punct: false,
                    ..C4Quality::default()
                }),
                wt,
            ),
            (Step::FineWebQuality(fw.clone()), wt),
        ],
    };
    report(
        "fineweb chain (4 filters)",
        best_of(3, || {
            for t in &texts {
                chain.run(t).unwrap();
            }
        }),
    );
}
