//! fastText `.bin` / `.ftz`: `Model::from_bytes` on arbitrary bytes, then
//! every inference entry point on whatever loads.
//!
//! Input: a model file. Its last byte `n` also picks an extra input text, the
//! `n % 48` bytes before it (decoded lossily); the loader ignores trailing
//! bytes, so the fuzzer can append text to a seed model without breaking it.
//! A second text joins the model's first words and label with that tail.
//!
//! Invariants, beyond "no panic, abort, hang or out-of-bounds access":
//! - heap: loading peaks at most 1 MiB + 300 x the file size (the bound
//!   `crates/pyroid-core/tests/fasttext_load.rs` asserts); inference at most
//!   that plus 8 KiB per byte of query text;
//! - `predict`: the Python layer's preconditions (no `'\n'`; `k` an int32,
//!   any `threshold`); errors in fastText's order; at most `k` (and at most
//!   `nlabels`) distinct, in-range labels; probabilities NaN (fastText's
//!   semantics for non-finite or overflowing weights) or in `[0, bound]`
//!   (bound `1 + 2e-5`, or `(1 + 1e-5)^depth` for hierarchical softmax,
//!   whose path products can each add `1e-5`), at least `threshold`, best
//!   first; softmax / one-vs-all / ns with threshold 0 return `min(k,
//!   nlabels)` labels; `predict_into` agrees bit for bit;
//! - vectors have `dim` floats; `line_ids` rows are inside the input matrix.

#![no_main]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, Ordering::Relaxed};

use libfuzzer_sys::fuzz_target;
use pyroid_core::fasttext::{LossKind, Model, PredictError, Prediction, tokenize};

// ------------------------------------------------------------------ heap accounting

/// Forwards to `System` and tracks live bytes and their peak (the target is
/// single-threaded, so global counters are exact).
struct CountingAlloc;

static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);

/// Note an attempt to grow the heap by `delta` bytes; it counts towards the
/// peak even if the allocator refused it.
fn track(delta: i64, granted: bool) {
    let now = LIVE.load(Relaxed).saturating_add(delta);
    if granted {
        LIVE.store(now, Relaxed);
    }
    PEAK.fetch_max(now, Relaxed);
}

#[allow(clippy::cast_possible_wrap)]
fn signed(n: usize) -> i64 {
    n as i64
}

// SAFETY: every method forwards to `System` with the caller's arguments; the
// bookkeeping only touches atomics.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        let p = unsafe { System.alloc(layout) };
        track(signed(layout.size()), !p.is_null());
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        let p = unsafe { System.alloc_zeroed(layout) };
        track(signed(layout.size()), !p.is_null());
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        track(-signed(layout.size()), true);
        // SAFETY: `ptr` came from `System` (through this allocator) with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        track(signed(new_size) - signed(layout.size()), !p.is_null());
        p
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Run `f`; return its result and the peak growth of the live heap meanwhile.
fn peak_heap<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let out = f();
    (out, u64::try_from(PEAK.load(Relaxed) - base).unwrap_or(0))
}

/// The heap budget for a file of `len` bytes and `text` bytes of queries.
fn budget(len: usize, text: usize) -> u64 {
    (1 << 20) + 300 * len as u64 + 8192 * text as u64
}

// ------------------------------------------------------------------ inputs

/// The longest dictionary entry prefix put into a query text.
const WORD_BYTES: usize = 32;

/// `(k, threshold)` pairs: what Python can pass (`k` is an int32).
const CASES: [(i64, f32); 7] = [
    (1, 0.0),
    (-1, 0.0),
    (2, 0.3),
    (0, 0.0),
    (-2, 0.0),
    (i32::MAX as i64, f32::NAN),
    (-1, -1.0),
];

fn tail_text(data: &[u8]) -> String {
    let Some((&n, rest)) = data.split_last() else {
        return String::new();
    };
    let n = usize::from(n) % 48;
    String::from_utf8_lossy(&rest[rest.len().saturating_sub(n)..]).into_owned()
}

fn prefix(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(WORD_BYTES)]).into_owned()
}

/// The query texts: the tail, and the model's first words and label with it.
fn texts(m: &Model, tail: &str) -> Vec<String> {
    let mut vocab = String::new();
    for i in 0..m.nwords().min(6) {
        vocab.push_str(&prefix(m.word(i)));
        vocab.push(' ');
    }
    if m.nlabels() > 0 {
        vocab.push_str(&prefix(m.label(0)));
        vocab.push(' ');
    }
    vocab.push_str(tail);
    vec![tail.to_owned(), vocab]
}

// ------------------------------------------------------------------ checks

fn bits(r: &Result<Vec<Prediction>, PredictError>) -> Result<Vec<(u32, usize)>, PredictError> {
    r.as_ref()
        .map(|ps| ps.iter().map(|p| (p.prob.to_bits(), p.label)).collect())
        .map_err(|&e| e)
}

/// The largest probability fastText can report.
fn max_prob(m: &Model) -> f64 {
    if m.args().loss == LossKind::HierarchicalSoftmax {
        // Each tree level adds at most ln(1 + 1e-5) (plus float rounding).
        #[allow(clippy::cast_precision_loss)]
        let depth = m.nlabels() as f64;
        (depth * 1.1e-5).exp() + 1e-6
    } else {
        1.0 + 2e-5
    }
}

fn check_predictions(m: &Model, ps: &[Prediction], k: i64, threshold: f32) {
    assert!(k == -1 || k >= 1, "Ok with k = {k}");
    let nlabels = m.nlabels();
    assert!(
        ps.len() <= nlabels,
        "{} predictions, {nlabels} labels",
        ps.len()
    );
    if let Ok(k) = usize::try_from(k) {
        assert!(ps.len() <= k, "{} predictions for k = {k}", ps.len());
    }
    let mut seen = vec![false; nlabels];
    for p in ps {
        assert!(p.label < nlabels, "label {} of {nlabels}", p.label);
        assert!(!seen[p.label], "label {} twice: {ps:?}", p.label);
        seen[p.label] = true;
        let _ = m.label(p.label);
    }
    if ps.iter().any(|p| p.prob.is_nan()) {
        return; // NaN scores: fastText's order is whatever its heap makes of them
    }
    let bound = max_prob(m);
    for p in ps {
        assert!(
            p.prob >= 0.0 && f64::from(p.prob) <= bound,
            "probability {} outside [0, {bound}]: {ps:?}",
            p.prob
        );
        assert!(
            !(p.prob < threshold),
            "probability {} below threshold {threshold}",
            p.prob
        );
    }
    assert!(
        ps.windows(2).all(|w| w[0].prob >= w[1].prob),
        "not best first: {ps:?}"
    );
    let flat = m.args().loss != LossKind::HierarchicalSoftmax;
    if flat && !(threshold > 0.0) {
        // Every label scores >= 0 >= threshold (or NaN), so the heap fills.
        let want = usize::try_from(k).map_or(nlabels, |k| k.min(nlabels));
        assert_eq!(ps.len(), want, "k = {k}, threshold {threshold}: {ps:?}");
    }
}

fn check_predict(m: &Model, text: &str, out: &mut Vec<Prediction>, has_tokens: bool) {
    for (k, threshold) in CASES {
        let got = m.predict(text, k, threshold);
        let into = m
            .predict_into(text, k, threshold, out)
            .map(|()| out.clone());
        assert_eq!(bits(&got), bits(&into), "predict_into differs");
        match got {
            Ok(ps) if !has_tokens || !m.is_supervised() => {
                assert!(ps.is_empty(), "{ps:?} without a head or tokens");
                assert!(!has_tokens, "an unsupervised model predicted");
            }
            Ok(ps) => check_predictions(m, &ps, k, threshold),
            Err(e) => {
                let _ = e.to_string();
                assert!(has_tokens, "{e:?} on a line without tokens");
                match e {
                    PredictError::Newline => panic!("Newline error without a newline"),
                    PredictError::NotSupervised => assert!(!m.is_supervised()),
                    PredictError::BadK => assert!(m.is_supervised() && (k == 0 || k < -1)),
                    PredictError::NaN => assert!(m.is_supervised() && (k == -1 || k >= 1)),
                }
            }
        }
    }
}

fn check_word(m: &Model, w: &str) {
    if let Some(v) = m.word_vector(w) {
        assert_eq!(v.len(), m.dim());
    }
    let (strings, ids) = m.subwords(w);
    // Only a pruned (hence quantized) model drops n-gram ids.
    if m.is_quantized() {
        assert!(ids.len() <= strings.len(), "{strings:?} {ids:?}");
    } else {
        assert_eq!(ids.len(), strings.len(), "{strings:?} {ids:?}");
    }
    let _ = (m.word_id(w), m.label_id(w), m.subword_id(w));
}

/// One line (no `'\n'`).
fn check_line(m: &Model, text: &str, out: &mut Vec<Prediction>) {
    let ids = m.line_ids(text);
    if m.dim() > 0 {
        // A row id past the matrix would be silently skipped by `average`.
        let rows = m.input_matrix().map(|(rows, _, _)| rows);
        for &id in &ids {
            let i = usize::try_from(id).unwrap_or_else(|_| panic!("negative row {id}"));
            match rows {
                Some(rows) => assert!(i < rows, "row {i} of {rows}"),
                None => assert!(m.input_vector(i).is_some(), "row {i} out of range"),
            }
        }
    }
    check_predict(m, text, out, !ids.is_empty());
    match m.sentence_vector(text) {
        Ok(v) => assert_eq!(v.len(), m.dim()),
        Err(e) => panic!("sentence_vector: {e:?}"),
    }
    let (words, labels) = m.line(text);
    for w in words.iter().chain(&labels) {
        assert!(!w.is_empty());
    }
    for w in text.split(' ').take(8).chain(words.into_iter().take(8)) {
        check_word(m, w);
    }
}

fn check_text(m: &Model, text: &str, out: &mut Vec<Prediction>) {
    for t in tokenize(text) {
        assert!(!t.is_empty());
    }
    if text.contains('\n') {
        // The Python layer raises before predicting; so does the core.
        assert_eq!(m.predict(text, 1, 0.0), Err(PredictError::Newline));
        assert_eq!(m.sentence_vector(text), Err(PredictError::Newline));
        check_line(m, &text.replace('\n', " "), out);
    } else {
        check_line(m, text, out);
    }
}

fn exercise(m: &Model, texts: &[String]) {
    let mut out = Vec::new();
    for t in texts {
        check_text(m, t, &mut out);
    }
    for i in 0..m.nwords().min(4) {
        let _ = m.word_count(i);
    }
    for i in 0..m.nlabels().min(4) {
        let _ = m.label_count(i);
    }
    for (rows, cols, data) in [m.input_matrix(), m.output_matrix()].into_iter().flatten() {
        assert_eq!(rows.checked_mul(cols), Some(data.len()));
    }
    let _ = (m.args(), m.version(), m.ntokens(), m.is_output_quantized());
}

fuzz_target!(|data: &[u8]| {
    let (loaded, peak) = peak_heap(|| Model::from_bytes(data));
    assert!(
        peak <= budget(data.len(), 0),
        "loading {} bytes peaked at {peak} heap bytes",
        data.len()
    );
    let model = match loaded {
        Ok(m) => m,
        Err(e) => {
            let _ = e.to_string();
            return;
        }
    };
    let texts = texts(&model, &tail_text(data));
    let text_bytes = texts.iter().map(String::len).sum();
    let ((), peak) = peak_heap(|| exercise(&model, &texts));
    assert!(
        peak <= budget(data.len(), text_bytes),
        "inference on a {}-byte model (dim {}) peaked at {peak} heap bytes",
        data.len(),
        model.dim()
    );
});
