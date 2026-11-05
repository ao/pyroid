//! Robustness tests for the fastText loader (`pyroid_core::fasttext`).
//!
//! A small little-endian writer (a port of the session's `ftwriter.py`)
//! builds valid synthetic models of every kind; the tests then truncate,
//! corrupt and mutate them. Every load, and every inference call on a model
//! that does load, must return (`Ok` or `Err`) without panicking, quickly,
//! and without allocating far more than the file's size. A counting global
//! allocator measures the last part, per thread.
//!
//! The heavier fuzz runs are `#[ignore]`d: `cargo test --test fasttext_load
//! -- --ignored --nocapture` runs them and prints their counts.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::too_many_lines
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::io::{self, Read};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Once;
use std::time::{Duration, Instant};

use pyroid_core::fasttext::{LoadError, MAGIC, Model, ModelKind, PredictError, Prediction, hash};

// ===================================================================== heap accounting

struct CountingAlloc;

thread_local! {
    /// Bytes this thread has allocated minus bytes it has freed.
    static LIVE: Cell<i64> = const { Cell::new(0) };
    /// The highest `LIVE` since the last `peak_heap` started.
    static PEAK: Cell<i64> = const { Cell::new(0) };
    /// Suppress the panic hook's message (the case runner reports panics itself).
    static QUIET: Cell<bool> = const { Cell::new(false) };
}

/// Note an attempt to grow this thread's heap by `delta` bytes (which counts
/// towards the peak); `granted` says whether the allocator agreed.
fn track(delta: i64, granted: bool) {
    let _ = LIVE.try_with(|live| {
        let now = live.get().saturating_add(delta);
        if granted {
            live.set(now);
        }
        let _ = PEAK.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

// SAFETY: every method forwards to `System` with the caller's arguments; the
// bookkeeping only touches const-initialized thread-locals without
// destructors, which neither allocate nor unwind.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        let p = unsafe { System.alloc(layout) };
        track(layout.size() as i64, !p.is_null());
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::alloc_zeroed`'s contract.
        let p = unsafe { System.alloc_zeroed(layout) };
        track(layout.size() as i64, !p.is_null());
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        track(-(layout.size() as i64), true);
        // SAFETY: `ptr` came from `System` (through this allocator) with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        track(new_size as i64 - layout.size() as i64, !p.is_null());
        p
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// Run `f`; return its result and the peak growth of this thread's live heap
/// while it ran. Attempted allocations count even if the allocator refuses.
fn peak_heap<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let base = LIVE.with(Cell::get);
    PEAK.with(|p| p.set(base));
    let out = f();
    let peak = PEAK.with(Cell::get);
    (out, u64::try_from(peak - base).unwrap_or(0))
}

/// The most heap a load of `len` bytes may use: a fixed allowance (the 64 KiB
/// read buffer, the 64 KiB float chunk, small tables) plus a constant factor
/// of the file size. The factor covers the subword-id table, which holds up
/// to `maxn` (at most 32) ids per byte of dictionary text, doubled by `Vec`
/// growth. A missing size check shows up as gigabytes, far above this.
fn heap_budget(len: usize) -> u64 {
    (1 << 20) + 300 * len as u64
}

// ===================================================================== deterministic RNG

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn byte(&mut self) -> u8 {
        self.next() as u8
    }

    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        let unit = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
        lo + (hi - lo) * unit
    }
}

// ===================================================================== model writer

const HS: i32 = 1;
const NS: i32 = 2;
const SOFTMAX: i32 = 3;
const OVA: i32 = 4;
const CBOW: i32 = 1;
const SUPERVISED: i32 = 3;

/// A synthetic model, as `ftwriter.write_model` takes it.
#[derive(Clone, Debug)]
struct Spec {
    words: Vec<Vec<u8>>,
    labels: Vec<Vec<u8>>,
    /// Empty: `1000 - 3i` for words, `500 - 7i` for labels (sorted, as fastText stores them).
    word_counts: Vec<i64>,
    label_counts: Vec<i64>,
    dim: i32,
    loss: i32,
    model: i32,
    minn: i32,
    maxn: i32,
    word_ngrams: i32,
    bucket: i32,
    version: i32,
    quant: bool,
    qnorm: bool,
    qout: bool,
    /// Sub-quantizer width of a quantized input matrix (the output uses 2).
    dsub: i32,
    /// The pruning index (`None`: `pruneidx_size = -1`).
    prune: Option<Vec<(i32, i32)>>,
    seed: u64,
}

fn strs(xs: &[&str]) -> Vec<Vec<u8>> {
    xs.iter().map(|s| s.as_bytes().to_vec()).collect()
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            words: strs(&["</s>", "the", "cat", "sat", "on", "mat", "café", "日本"]),
            labels: strs(&[
                "__label__a",
                "__label__b",
                "__label__c",
                "__label__d",
                "__label__e",
            ]),
            word_counts: Vec::new(),
            label_counts: Vec::new(),
            dim: 4,
            loss: SOFTMAX,
            model: SUPERVISED,
            minn: 2,
            maxn: 4,
            word_ngrams: 1,
            bucket: 11,
            version: 12,
            quant: false,
            qnorm: false,
            qout: false,
            dsub: 2,
            prune: None,
            seed: 0,
        }
    }
}

/// A named fixed-width field of a built model, for targeted corruption.
#[derive(Clone, Debug)]
struct Field {
    name: String,
    offset: usize,
    width: usize,
}

#[derive(Clone, Debug)]
struct Built {
    bytes: Vec<u8>,
    fields: Vec<Field>,
}

impl Built {
    fn field(&self, name: &str) -> &Field {
        self.fields
            .iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("no field {name}"))
    }

    /// The bytes with field `name` set to `value` (little-endian, truncated
    /// to the field's width).
    fn patched(&self, name: &str, value: i64) -> Vec<u8> {
        let f = self.field(name);
        let mut b = self.bytes.clone();
        b[f.offset..f.offset + f.width].copy_from_slice(&value.to_le_bytes()[..f.width]);
        b
    }

    /// Patch several fields.
    fn patched_all(&self, patches: &[(&str, i64)]) -> Vec<u8> {
        let mut out = Built {
            bytes: self.bytes.clone(),
            fields: self.fields.clone(),
        };
        for &(name, value) in patches {
            out.bytes = out.patched(name, value);
        }
        out.bytes
    }
}

#[derive(Default)]
struct Writer {
    buf: Vec<u8>,
    fields: Vec<Field>,
}

impl Writer {
    fn field(&mut self, name: impl Into<String>, bytes: &[u8]) {
        self.fields.push(Field {
            name: name.into(),
            offset: self.buf.len(),
            width: bytes.len(),
        });
        self.buf.extend_from_slice(bytes);
    }

    fn i32(&mut self, name: impl Into<String>, v: i32) {
        self.field(name, &v.to_le_bytes());
    }

    fn i64(&mut self, name: impl Into<String>, v: i64) {
        self.field(name, &v.to_le_bytes());
    }

    fn u8(&mut self, name: impl Into<String>, v: u8) {
        self.field(name, &[v]);
    }

    fn raw(&mut self, b: &[u8]) {
        self.buf.extend_from_slice(b);
    }

    fn f32(&mut self, x: f32) {
        self.raw(&x.to_le_bytes());
    }

    fn dense(&mut self, p: &str, rows: usize, cols: i32, rng: &mut Rng) {
        self.i64(format!("{p}.m"), rows as i64);
        self.i64(format!("{p}.n"), i64::from(cols));
        for _ in 0..rows * cols.max(0) as usize {
            let x = rng.uniform(-1.0, 1.0);
            self.f32(x);
        }
    }

    /// A product quantizer for `dim` columns in sub-vectors of `dsub`.
    fn pq(&mut self, p: &str, dim: i32, dsub: i32, rng: &mut Rng, lo: f32, hi: f32) -> usize {
        let nsubq = (dim as u32).div_ceil(dsub as u32) as i32;
        let last = if dim % dsub == 0 { dsub } else { dim % dsub };
        self.i32(format!("{p}.dim"), dim);
        self.i32(format!("{p}.nsubq"), nsubq);
        self.i32(format!("{p}.dsub"), dsub);
        self.i32(format!("{p}.lastdsub"), last);
        for _ in 0..dim * 256 {
            let x = rng.uniform(lo, hi);
            self.f32(x);
        }
        nsubq as usize
    }

    fn quant(&mut self, p: &str, rows: usize, dim: i32, dsub: i32, qnorm: bool, rng: &mut Rng) {
        let nsubq = (dim as u32).div_ceil(dsub as u32) as usize;
        self.u8(format!("{p}.qnorm"), u8::from(qnorm));
        self.i64(format!("{p}.m"), rows as i64);
        self.i64(format!("{p}.n"), i64::from(dim));
        self.i32(format!("{p}.codesize"), (rows * nsubq) as i32);
        for _ in 0..rows * nsubq {
            let c = rng.byte();
            self.raw(&[c]);
        }
        self.pq(&format!("{p}.pq"), dim, dsub, rng, -1.0, 1.0);
        if qnorm {
            for _ in 0..rows {
                let c = rng.byte();
                self.raw(&[c]);
            }
            self.pq(&format!("{p}.npq"), 1, 1, rng, 0.1, 2.0);
        }
    }
}

impl Spec {
    fn build(&self) -> Built {
        let mut rng = Rng::new(self.seed);
        let mut w = Writer::default();
        w.i32("magic", MAGIC);
        w.i32("version", self.version);
        for (name, v) in [
            ("args.dim", self.dim),
            ("args.ws", 5),
            ("args.epoch", 5),
            ("args.minCount", 1),
            ("args.neg", 5),
            ("args.wordNgrams", self.word_ngrams),
            ("args.loss", self.loss),
            ("args.model", self.model),
            ("args.bucket", self.bucket),
            ("args.minn", self.minn),
            ("args.maxn", self.maxn),
            ("args.lrUpdateRate", 100),
        ] {
            w.i32(name, v);
        }
        w.field("args.t", &1e-4f64.to_le_bytes());
        let (nw, nl) = (self.words.len(), self.labels.len());
        w.i32("dict.size", (nw + nl) as i32);
        w.i32("dict.nwords", nw as i32);
        w.i32("dict.nlabels", nl as i32);
        w.i64("dict.ntokens", 12_345);
        w.i64(
            "dict.pruneidx_size",
            self.prune.as_ref().map_or(-1, |p| p.len() as i64),
        );
        let entries = self.words.iter().map(|e| (e, 0u8)).enumerate().chain(
            self.labels
                .iter()
                .map(|e| (e, 1u8))
                .enumerate()
                .map(|(i, x)| (nw + i, x)),
        );
        for (i, (entry, kind)) in entries {
            let count = if kind == 0 {
                self.word_counts
                    .get(i)
                    .copied()
                    .unwrap_or(1000 - 3 * i as i64)
            } else {
                self.label_counts
                    .get(i - nw)
                    .copied()
                    .unwrap_or(500 - 7 * (i - nw) as i64)
            };
            w.raw(entry);
            w.u8(format!("entry{i}.nul"), 0);
            w.i64(format!("entry{i}.count"), count);
            w.u8(format!("entry{i}.type"), kind);
        }
        if let Some(prune) = &self.prune {
            for (k, &(key, value)) in prune.iter().enumerate() {
                w.i32(format!("prune{k}.key"), key);
                w.i32(format!("prune{k}.value"), value);
            }
        }
        let rows_in = nw
            + match &self.prune {
                None => self.bucket.max(0) as usize,
                Some(p) => (p.iter().map(|x| x.1).max().unwrap_or(-1) + 1) as usize,
            };
        w.u8("quant_input", u8::from(self.quant));
        if self.quant {
            w.quant("in", rows_in, self.dim, self.dsub, self.qnorm, &mut rng);
        } else {
            w.dense("in", rows_in, self.dim, &mut rng);
        }
        let rows_out = if self.model == SUPERVISED { nl } else { nw };
        w.u8("qout", u8::from(self.qout));
        if self.quant && self.qout {
            w.quant("out", rows_out, self.dim, 2, self.qnorm, &mut rng);
        } else {
            w.dense("out", rows_out, self.dim, &mut rng);
        }
        Built {
            bytes: w.buf,
            fields: w.fields,
        }
    }
}

/// A pruning index over the 11 n-gram buckets: 6 rows survive, and key 3 is
/// written twice (the later pair wins, as with fastText's `std::unordered_map`).
fn prune_index() -> Vec<(i32, i32)> {
    vec![(0, 0), (3, 1), (6, 2), (9, 3), (12, 4), (3, 5)]
}

/// One small model of each kind the loader distinguishes.
fn kinds() -> Vec<(&'static str, Spec)> {
    let d = Spec::default;
    vec![
        ("softmax", d()),
        (
            "hs",
            Spec {
                loss: HS,
                word_ngrams: 2,
                seed: 1,
                ..d()
            },
        ),
        (
            "ova",
            Spec {
                loss: OVA,
                seed: 2,
                ..d()
            },
        ),
        (
            "quant-qnorm-qout-pruned",
            Spec {
                quant: true,
                qnorm: true,
                qout: true,
                dim: 6,
                dsub: 4,
                word_ngrams: 2,
                prune: Some(prune_index()),
                seed: 3,
                ..d()
            },
        ),
        (
            // nsubq = 4: rows * nsubq overflows u64 for rows >= 2^62.
            "quant-hs",
            Spec {
                quant: true,
                loss: HS,
                dim: 8,
                dsub: 2,
                seed: 4,
                ..d()
            },
        ),
        (
            "cbow",
            Spec {
                model: CBOW,
                loss: NS,
                labels: Vec::new(),
                minn: 3,
                maxn: 6,
                bucket: 13,
                seed: 5,
                ..d()
            },
        ),
    ]
}

// ===================================================================== exercising a model

const TEXTS: &[&str] = &[
    "",
    " ",
    "the cat sat on the mat",
    "café 日本 zebra qq",
    "</s>",
    "the </s> cat",
    "__label__a the __label__zz",
    "__label__x",
    "\t\u{0}x\u{b}y\u{c}\r",
    "🙂🙂 éé ü",
    "a\u{0}b",
    "  the  ",
    "<>",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Depth {
    Light,
    Full,
}

/// Invariants every successful prediction must keep, whatever the weights.
fn check_predictions(m: &Model, ps: &[Prediction], k: i64) {
    assert!(ps.len() <= m.nlabels(), "{} predictions", ps.len());
    if k > 0 {
        assert!(ps.len() as u64 <= k as u64);
    }
    for p in ps {
        assert!(
            p.label < m.nlabels(),
            "label {} of {}",
            p.label,
            m.nlabels()
        );
        let _ = m.label(p.label);
    }
    if ps.iter().all(|p| p.prob.is_finite()) {
        assert!(
            ps.windows(2).all(|w| w[0].prob >= w[1].prob),
            "not best-first: {ps:?}"
        );
    }
}

/// Call every inference entry point; the only requirement is not panicking
/// (plus the prediction invariants).
fn exercise(m: &Model, depth: Depth) {
    let long = "abcé".repeat(100);
    let mut texts: Vec<&str> = TEXTS.to_vec();
    let ks: &[(i64, f32)] = if depth == Depth::Full {
        texts.push(&long);
        &[
            (1, 0.0),
            (-1, 0.0),
            (2, 0.3),
            (-1, 0.99),
            (i64::MAX, 0.0),
            (0, 0.0),
            (-2, 0.0),
            (1, f32::NAN),
            (-1, f32::INFINITY),
            (-1, -1.0),
        ]
    } else {
        &[(1, 0.0), (-1, 0.0), (0, 0.0)]
    };
    for &t in &texts {
        for &(k, threshold) in ks {
            match m.predict(t, k, threshold) {
                Ok(ps) => check_predictions(m, &ps, k),
                Err(e) => {
                    let _ = e.message();
                }
            }
        }
        let _ = m.sentence_vector(t);
        let _ = m.line_ids(t);
        let (words, labels) = m.line(t);
        for w in t.split(' ').chain(words).chain(labels) {
            let _ = m.word_vector(w);
            let _ = m.subwords(w);
            let _ = m.word_id(w);
            let _ = m.label_id(w);
            let _ = m.subword_id(w);
        }
    }
    let _ = m.predict("a\nb", 1, 0.0);
    let _ = m.sentence_vector("a\nb");
    for i in 0..m.nwords().min(64) {
        let w = String::from_utf8_lossy(m.word(i)).into_owned();
        let _ = m.word_count(i);
        let _ = m.word_vector(&w);
        let _ = m.subwords(&w);
        if depth == Depth::Full {
            let _ = m.predict(&w, -1, 0.0);
        }
    }
    for i in 0..m.nlabels().min(64) {
        let l = String::from_utf8_lossy(m.label(i)).into_owned();
        let _ = m.label_count(i);
        let _ = m.label_id(&l);
        let _ = m.word_vector(&l);
        let _ = m.subwords(&l);
    }
    for i in [0, 1, 7, 100, 1000, 1 << 20, 1 << 40, usize::MAX] {
        if let Some(v) = m.input_vector(i) {
            assert_eq!(v.len(), m.dim());
        }
    }
    for (rows, cols, data) in [m.input_matrix(), m.output_matrix()].into_iter().flatten() {
        assert_eq!(rows.checked_mul(cols), Some(data.len()));
    }
    let _ = (m.args(), m.version(), m.dim(), m.ntokens());
    let _ = (m.is_supervised(), m.is_quantized(), m.is_output_quantized());
}

// ===================================================================== running cases

#[derive(Debug)]
enum Outcome {
    Loaded,
    Rejected(LoadError),
    Panicked(String),
}

impl Outcome {
    fn class(&self) -> &'static str {
        match self {
            Self::Loaded => "ok",
            Self::Rejected(LoadError::Truncated { .. }) => "truncated",
            Self::Rejected(LoadError::Invalid(_)) => "invalid",
            Self::Rejected(LoadError::WrongFormat { .. }) => "wrong-format",
            Self::Rejected(LoadError::Io(_)) => "io",
            Self::Panicked(_) => "PANIC",
        }
    }
}

struct Case {
    outcome: Outcome,
    /// Peak heap growth of the load alone.
    peak: u64,
    elapsed: Duration,
}

fn quiet_panics() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if !QUIET.with(Cell::get) {
                default(info);
            }
        }));
    });
}

fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic".to_owned())
}

/// Load `bytes` and, if that succeeds, exercise the model; panics are caught.
fn run(bytes: &[u8], depth: Depth) -> Case {
    quiet_panics();
    QUIET.with(|q| q.set(true));
    let start = Instant::now();
    let (loaded, peak) = peak_heap(|| catch_unwind(|| Model::from_bytes(bytes)));
    let outcome = match loaded {
        Err(p) => Outcome::Panicked(format!("load: {}", panic_message(&*p))),
        Ok(Err(e)) => Outcome::Rejected(e),
        Ok(Ok(model)) => match catch_unwind(AssertUnwindSafe(|| exercise(&model, depth))) {
            Ok(()) => Outcome::Loaded,
            Err(p) => Outcome::Panicked(format!("inference: {}", panic_message(&*p))),
        },
    };
    let elapsed = start.elapsed();
    QUIET.with(|q| q.set(false));
    Case {
        outcome,
        peak,
        elapsed,
    }
}

/// Outcome counts and failures of a batch of cases.
#[derive(Default)]
struct Tally {
    classes: BTreeMap<&'static str, usize>,
    failures: Vec<String>,
    cases: usize,
    max_elapsed: Duration,
    /// Largest load heap peak, and the file size it came from.
    max_peak: (u64, usize),
}

impl Tally {
    fn add(&mut self, what: &dyn Fn() -> String, bytes: &[u8], case: &Case) {
        self.cases += 1;
        *self.classes.entry(case.outcome.class()).or_default() += 1;
        self.max_elapsed = self.max_elapsed.max(case.elapsed);
        self.max_peak = self.max_peak.max((case.peak, bytes.len()));
        let mut fail = |why: String| {
            if self.failures.len() < 20 {
                self.failures.push(format!("{}: {why}", what()));
            }
        };
        if let Outcome::Panicked(msg) = &case.outcome {
            fail(format!("panicked: {msg}"));
        }
        if case.peak > heap_budget(bytes.len()) {
            fail(format!(
                "load of {} bytes peaked at {} heap bytes",
                bytes.len(),
                case.peak
            ));
        }
        if case.elapsed > Duration::from_secs(2) {
            fail(format!("took {:?}", case.elapsed));
        }
    }

    fn fail(&mut self, what: String) {
        if self.failures.len() < 20 {
            self.failures.push(what);
        }
    }

    fn merge(&mut self, other: Self) {
        for (class, n) in other.classes {
            *self.classes.entry(class).or_default() += n;
        }
        for f in other.failures {
            self.fail(f);
        }
        self.cases += other.cases;
        self.max_elapsed = self.max_elapsed.max(other.max_elapsed);
        self.max_peak = self.max_peak.max(other.max_peak);
    }

    fn ok(&self) -> usize {
        self.classes.get("ok").copied().unwrap_or(0)
    }

    #[allow(clippy::print_stderr)] // the counts the test report quotes (`--nocapture`)
    fn finish(self, name: &str) {
        eprintln!(
            "[{name}] {} cases: {:?}; slowest {:?}; largest load heap peak {} bytes (file of {})",
            self.cases, self.classes, self.max_elapsed, self.max_peak.0, self.max_peak.1
        );
        assert!(
            self.failures.is_empty(),
            "[{name}] failures:\n{}",
            self.failures.join("\n")
        );
    }
}

fn load(bytes: &[u8]) -> Model {
    Model::from_bytes(bytes).unwrap_or_else(|e| panic!("model does not load: {e}"))
}

fn load_spec(spec: &Spec) -> Model {
    load(&spec.build().bytes)
}

// ===================================================================== 1. truncation

#[test]
fn every_prefix_of_a_valid_model_is_truncated() {
    let mut tally = Tally::default();
    for (name, spec) in kinds() {
        let built = spec.build();
        let full = run(&built.bytes, Depth::Full);
        assert!(
            matches!(full.outcome, Outcome::Loaded),
            "{name}: {:?}",
            full.outcome
        );
        for len in 0..built.bytes.len() {
            let prefix = &built.bytes[..len];
            let case = run(prefix, Depth::Light);
            if !matches!(case.outcome, Outcome::Rejected(LoadError::Truncated { .. })) {
                tally.fail(format!("{name}: prefix {len}: {:?}", case.outcome));
            }
            tally.add(&|| format!("{name} prefix {len}"), prefix, &case);
        }
    }
    tally.finish("truncation");
}

#[test]
fn trailing_bytes_after_a_valid_model_are_ignored() {
    let mut rng = Rng::new(99);
    for (name, spec) in kinds() {
        let built = spec.build();
        let clean = load(&built.bytes);
        let garbage: Vec<u8> = (0..1000).map(|_| rng.byte()).collect();
        let tails: [&[u8]; 4] = [b"\0", &[0xFF; 8], &garbage, &built.bytes];
        for tail in tails {
            let mut bytes = built.bytes.clone();
            bytes.extend_from_slice(tail);
            let m = load(&bytes);
            for t in TEXTS {
                assert_eq!(m.predict(t, -1, 0.0), clean.predict(t, -1, 0.0), "{name}");
                assert_eq!(m.line_ids(t), clean.line_ids(t), "{name}");
            }
            assert_eq!(m.input_vector(3), clean.input_vector(3), "{name}");
        }
    }
}

// ===================================================================== 2. field corruption

fn i32_values(orig: i64) -> Vec<i64> {
    let mut v: Vec<i64> = vec![
        i32::MIN.into(),
        (i32::MIN + 1).into(),
        -1_000_000,
        -256,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        11,
        12,
        13,
        32,
        33,
        255,
        256,
        65_536,
        1 << 24,
        1 << 30,
        (i32::MAX - 1).into(),
        i32::MAX.into(),
    ];
    v.extend([orig - 1, orig + 1, orig * 2]);
    v.retain(|&x| x != orig && i32::try_from(x).is_ok());
    v.sort_unstable();
    v.dedup();
    v
}

fn i64_values(orig: i64) -> Vec<i64> {
    let mut v = vec![
        i64::MIN,
        i64::MIN + 1,
        -(1 << 40),
        -2,
        -1,
        0,
        1,
        2,
        3,
        5,
        1 << 31,
        1 << 32,
        1 << 40,
        1 << 61,
        1 << 62,
        (1 << 62) + 1,
        i64::MAX / 3 + 1,
        999_999_999_999_999,
        1_000_000_000_000_000,
        i64::MAX - 1,
        i64::MAX,
    ];
    v.extend([
        orig.wrapping_sub(1),
        orig.wrapping_add(1),
        orig.wrapping_mul(2),
    ]);
    v.retain(|&x| x != orig);
    v.sort_unstable();
    v.dedup();
    v
}

fn read_field(bytes: &[u8], f: &Field) -> i64 {
    let mut b = [0u8; 8];
    b[..f.width].copy_from_slice(&bytes[f.offset..f.offset + f.width]);
    match f.width {
        1 => i64::from(b[0]),
        4 => i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        _ => i64::from_le_bytes(b),
    }
}

/// The last dot-separated part of a field name (`entry3.count` -> `count`).
fn last_part(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

/// Values to try in a field.
fn values_for(f: &Field, orig: i64) -> Vec<i64> {
    match f.width {
        1 if last_part(&f.name) == "nul" => vec![i64::from(b'x'), 1, 0x20, 0x80, 0xFF],
        1 => (0..=255).filter(|&v| v != orig).collect(),
        4 => i32_values(orig),
        _ if f.name == "args.t" => [f64::NAN, f64::INFINITY, -1.0, 0.0, 1e308]
            .iter()
            .map(|x| x.to_bits() as i64)
            .collect(),
        _ => i64_values(orig),
    }
}

fn is_invalid(o: &Outcome) -> bool {
    matches!(o, Outcome::Rejected(LoadError::Invalid(_)))
}

/// What a corrupted field must produce, where that is certain.
#[derive(Clone, Copy, Debug)]
enum Expect {
    Loaded,
    Invalid,
    WrongMagic,
    TooNew,
}

impl Expect {
    fn holds(self, o: &Outcome) -> bool {
        match self {
            Self::Loaded => matches!(o, Outcome::Loaded),
            Self::Invalid => is_invalid(o),
            Self::WrongMagic => matches!(
                o,
                Outcome::Rejected(LoadError::WrongFormat { version: None, .. })
            ),
            Self::TooNew => matches!(
                o,
                Outcome::Rejected(LoadError::WrongFormat {
                    version: Some(_),
                    ..
                })
            ),
        }
    }
}

/// The certain outcome of setting field `name` of `spec`'s model to `v`;
/// `None`: anything but a panic is acceptable.
fn expected(spec: &Spec, name: &str, v: i64) -> Option<Expect> {
    match name {
        "magic" => return Some(Expect::WrongMagic),
        "version" if v > 12 => return Some(Expect::TooNew),
        _ => {}
    }
    let entry: Option<usize> = name
        .strip_prefix("entry")
        .and_then(|rest| rest.split('.').next())
        .and_then(|i| i.parse().ok());
    let invalid = match name {
        "args.loss" => !(1..=4).contains(&v),
        "args.model" => !(1..=3).contains(&v) || (v == 3 && spec.labels.is_empty()),
        // Older (even negative) versions load (11 drops char n-grams); the
        // rest is unused at inference.
        "version" | "args.ws" | "args.epoch" | "args.minCount" | "args.neg"
        | "args.lrUpdateRate" | "args.t" | "dict.ntokens" | "args.minn" => false,
        "args.dim" | "dict.size" | "dict.nwords" | "dict.nlabels" => true,
        "args.maxn" => !(0..=32).contains(&v),
        "args.wordNgrams" => v > 32,
        // Every base model hashes n-grams, so it needs buckets, and a
        // larger bucket needs more input rows than an unpruned model has.
        "args.bucket" => v <= 0 || (v > i64::from(spec.bucket) && spec.prune.is_none()),
        // A dense model with a pruning index is fastText's "Invalid model file".
        "dict.pruneidx_size" if spec.prune.is_none() && !spec.quant && v <= 0 => v == 0,
        "qout" if !spec.quant => false,
        _ => match (entry, last_part(name)) {
            (Some(_), "type") => true,
            // Label counts of 1e15 or more break the Huffman tree (fastText
            // recurses forever); word counts are unused.
            (Some(i), "count") => {
                i >= spec.words.len() && spec.loss == HS && v >= 1_000_000_000_000_000
            }
            _ if name.starts_with("prune") && last_part(name) == "key" => false,
            _ => return None,
        },
    };
    Some(if invalid {
        Expect::Invalid
    } else {
        Expect::Loaded
    })
}

#[test]
fn corrupting_any_single_field_never_panics() {
    let mut tally = Tally::default();
    for (kind, spec) in kinds() {
        let built = spec.build();
        for f in &built.fields {
            let orig = read_field(&built.bytes, f);
            // Bool bytes: every non-zero value must behave like 1.
            let is_bool = matches!(last_part(&f.name), "qnorm" | "qout" | "quant_input");
            let as_one = is_bool.then(|| {
                run(&built.patched(&f.name, 1), Depth::Light)
                    .outcome
                    .class()
            });
            for v in values_for(f, orig) {
                let bytes = built.patched(&f.name, v);
                let case = run(&bytes, Depth::Light);
                let what = || format!("{kind}: {} = {v} (was {orig})", f.name);
                if let Some(want) = expected(&spec, &f.name, v) {
                    if !want.holds(&case.outcome) {
                        tally.fail(format!("{}: want {want:?}, got {:?}", what(), case.outcome));
                    }
                }
                if let Some(one) = as_one {
                    if v > 1 && case.outcome.class() != one {
                        tally.fail(format!(
                            "{}: {} but 1 gives {one}",
                            what(),
                            case.outcome.class()
                        ));
                    }
                }
                tally.add(&what, &bytes, &case);
            }
        }
    }
    tally.finish("field corruption");
}

#[test]
fn malformed_dictionaries_are_invalid() {
    let base = Spec::default();
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "nwords + nlabels != size",
            base.build().patched("dict.nwords", 7),
        ),
        (
            "negative nlabels",
            base.build()
                .patched_all(&[("dict.nlabels", -1), ("dict.nwords", 14)]),
        ),
        (
            "a label before the words",
            base.build()
                .patched_all(&[("entry0.type", 1), ("entry8.type", 0)]),
        ),
        ("entry type 2", base.build().patched("entry3.type", 2)),
        (
            "label typed as word",
            base.build().patched("entry9.type", 0),
        ),
        (
            "supervised without labels",
            Spec {
                labels: Vec::new(),
                ..base.clone()
            }
            .build()
            .bytes,
        ),
        (
            "negative pruned row",
            Spec {
                quant: true,
                prune: Some(vec![(0, 0), (3, -1)]),
                ..base.clone()
            }
            .build()
            .bytes,
        ),
        (
            "pruning index on a dense model",
            Spec {
                prune: Some(prune_index()),
                ..base.clone()
            }
            .build()
            .bytes,
        ),
        (
            "hs label count 1e15",
            Spec {
                loss: HS,
                label_counts: vec![1_000_000_000_000_000, 4, 3, 2, 1],
                ..base.clone()
            }
            .build()
            .bytes,
        ),
        (
            "hs label count i64::MAX",
            Spec {
                loss: HS,
                label_counts: vec![5, 4, 3, 2, i64::MAX],
                ..base.clone()
            }
            .build()
            .bytes,
        ),
        (
            "n-grams without buckets",
            Spec {
                bucket: 0,
                ..base.clone()
            }
            .build()
            .bytes,
        ),
        (
            "word n-grams without buckets",
            Spec {
                bucket: 0,
                maxn: 0,
                word_ngrams: 2,
                ..base.clone()
            }
            .build()
            .bytes,
        ),
        (
            "input rows < nwords + bucket",
            base.build().patched("args.bucket", 12),
        ),
        (
            "output rows < nlabels",
            Spec {
                quant: true,
                qout: true,
                ..base.clone()
            }
            .build()
            .patched("out.m", 4),
        ),
        ("dim != input columns", base.build().patched("args.dim", 5)),
        ("negative dim", base.build().patched("args.dim", -4)),
    ];
    for (what, bytes) in cases {
        let case = run(&bytes, Depth::Full);
        assert!(is_invalid(&case.outcome), "{what}: {:?}", case.outcome);
    }
    // Words that look like labels, duplicate entries and a large-but-finite
    // hs count are fine.
    let lookalike = load_spec(&Spec {
        words: base.labels.clone(),
        labels: base.words.clone(),
        ..base.clone()
    });
    assert_eq!(lookalike.word_id("__label__a"), 0);
    // Entry types decide, not prefixes: here `</s>` is a label.
    assert_eq!(
        lookalike.line_ids("__label__a")[0],
        0,
        "a known word, not skipped"
    );
    assert_eq!(
        lookalike.line("__label__a"),
        (vec!["__label__a"], vec!["</s>"])
    );
    let dup = Spec {
        words: strs(&["the", "the", "cat"]),
        ..base.clone()
    };
    let m = load_spec(&dup);
    assert_eq!(
        m.word_id("the"),
        1,
        "the later duplicate wins, as in fastText"
    );
    let ok = Spec {
        loss: HS,
        label_counts: vec![999_999_999_999_999, 4, 3, 2, 1],
        ..base
    };
    assert!(matches!(
        run(&ok.build().bytes, Depth::Full).outcome,
        Outcome::Loaded
    ));
}

// ===================================================================== 3. huge declarations

/// Files that declare enormous sizes must fail at once, with a small heap.
#[test]
fn huge_declared_sizes_fail_fast() {
    let dense = Spec::default().build();
    let quant = Spec {
        quant: true,
        qnorm: true,
        qout: true,
        prune: Some(prune_index()),
        ..Spec::default()
    }
    .build();
    let quant4 = Spec {
        quant: true,
        dim: 8,
        dsub: 2,
        ..Spec::default()
    }
    .build();
    let header_only = |b: &Built, upto: &str| -> Vec<u8> {
        let f = b.field(upto);
        b.bytes[..f.offset + f.width].to_vec()
    };
    let mut no_nul = header_only(&dense, "dict.pruneidx_size");
    no_nul.extend(std::iter::repeat_n(b'a', 100_000));
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "dictionary of i32::MAX entries",
            dense.patched_all(&[
                ("dict.size", i64::from(i32::MAX)),
                ("dict.nwords", i64::from(i32::MAX) - 5),
            ]),
        ),
        ("an entry without its NUL", no_nul),
        (
            "pruneidx_size i64::MAX",
            quant.patched("dict.pruneidx_size", i64::MAX),
        ),
        (
            "pruneidx_size 2^40",
            quant.patched("dict.pruneidx_size", 1 << 40),
        ),
        ("dense 2^40 x 4", dense.patched("in.m", 1 << 40)),
        ("dense 2^62 x 4 (overflows)", dense.patched("in.m", 1 << 62)),
        (
            "dense i64::MAX x i64::MAX",
            dense.patched_all(&[("in.m", i64::MAX), ("in.n", i64::MAX)]),
        ),
        ("dense output 2^40 rows", dense.patched("out.m", 1 << 40)),
        (
            "codesize i32::MAX",
            quant.patched("in.codesize", i64::from(i32::MAX)),
        ),
        (
            "product quantizer of i32::MAX columns",
            quant.patched_all(&[
                ("in.pq.dim", i64::from(i32::MAX)),
                ("in.pq.nsubq", 1),
                ("in.pq.dsub", i64::from(i32::MAX)),
                ("in.pq.lastdsub", i64::from(i32::MAX)),
            ]),
        ),
        ("norm codes for 2^40 rows", quant.patched("in.m", 1 << 40)),
        (
            "quantized 2^62 rows x 4 codes (overflows u64)",
            quant4.patched("in.m", 1 << 62),
        ),
        ("quantized i64::MAX rows", quant4.patched("in.m", i64::MAX)),
    ];
    for (what, bytes) in &cases {
        let case = run(bytes, Depth::Full);
        assert!(
            matches!(case.outcome, Outcome::Rejected(_)),
            "{what}: {:?}",
            case.outcome
        );
        assert!(
            case.peak <= (1 << 20) + 4 * bytes.len() as u64,
            "{what}: {} bytes peaked at {} heap bytes",
            bytes.len(),
            case.peak
        );
        assert!(
            case.elapsed < Duration::from_secs(1),
            "{what}: {:?}",
            case.elapsed
        );
    }
}

/// `dim = 0`: a 0-column matrix may declare 2^63 rows in 16 bytes. That is
/// harmless (every vector is empty) and must stay so.
#[test]
fn zero_dim_models_with_enormous_row_counts_are_harmless() {
    let built = Spec {
        dim: 0,
        ..Spec::default()
    }
    .build();
    for patches in [
        &[("in.m", i64::MAX)][..],
        &[("out.m", i64::MAX)],
        &[("in.m", i64::MAX), ("args.bucket", i64::from(i32::MAX))],
        &[
            ("in.m", i64::MAX),
            ("args.bucket", i64::from(i32::MAX)),
            ("args.wordNgrams", 3),
        ],
    ] {
        let bytes = built.patched_all(patches);
        let case = run(&bytes, Depth::Full);
        assert!(
            matches!(case.outcome, Outcome::Loaded),
            "{patches:?}: {:?}",
            case.outcome
        );
        let m = load(&bytes);
        assert_eq!(m.dim(), 0);
        let huge_input = patches.iter().any(|p| p.0 == "in.m");
        assert_eq!(m.input_vector(1 << 62).is_some(), huge_input);
        let ps = m.predict("the cat zebra", -1, 0.0).expect("predicts");
        assert_eq!(ps.len(), 5);
    }
}

/// Character n-grams cost up to `maxn` hash steps and ids per byte of a
/// word, and word n-grams `wordNgrams` ids per token: a header with a huge
/// (or negative, which fastText compares as a huge `size_t`) value would make
/// one long dictionary entry, or one long token, quadratic. The loader caps
/// both at 32.
#[test]
fn ngram_lengths_beyond_the_cap_are_rejected() {
    let long_word = vec![b'a'; 3000];
    let spec = Spec {
        model: CBOW,
        loss: NS,
        labels: Vec::new(),
        words: vec![b"</s>".to_vec(), long_word],
        dim: 2,
        minn: 1,
        bucket: 7,
        ..Spec::default()
    };
    for maxn in [i32::MAX, 1 << 20, 33, -1, i32::MIN] {
        let bytes = Spec {
            maxn,
            ..spec.clone()
        }
        .build()
        .bytes;
        let case = run(&bytes, Depth::Light);
        assert!(is_invalid(&case.outcome), "maxn {maxn}: {:?}", case.outcome);
        assert!(
            case.peak <= heap_budget(bytes.len()),
            "maxn {maxn}: peak {}",
            case.peak
        );
    }
    for word_ngrams in [33, 1 << 30, i32::MAX] {
        let bytes = Spec {
            word_ngrams,
            ..spec.clone()
        }
        .build()
        .bytes;
        assert!(
            is_invalid(&run(&bytes, Depth::Light).outcome),
            "wordNgrams {word_ngrams}"
        );
    }
    // At the cap, work stays linear in the text.
    let token = "é".repeat(10_000);
    let tokens = "x ".repeat(10_000);
    for maxn in [0, 6, 32] {
        let bytes = Spec {
            maxn,
            word_ngrams: 32,
            ..spec.clone()
        }
        .build()
        .bytes;
        let case = run(&bytes, Depth::Light);
        assert!(
            matches!(case.outcome, Outcome::Loaded),
            "maxn {maxn}: {:?}",
            case.outcome
        );
        assert!(
            case.peak <= heap_budget(bytes.len()),
            "maxn {maxn}: peak {}",
            case.peak
        );
        let m = load(&bytes);
        let (ids, peak) = peak_heap(|| m.line_ids(&token));
        assert!(ids.len() <= 32 * 10_002, "{}", ids.len());
        assert!(
            peak <= 64 * 32 * 10_002,
            "maxn {maxn}: line_ids peaked at {peak}"
        );
        let (ids, _) = peak_heap(|| m.line_ids(&tokens));
        assert!(
            ids.len() <= 33 * 10_001 * (maxn as usize + 1),
            "{}",
            ids.len()
        );
        let (v, _) = peak_heap(|| m.word_vector(&token));
        assert_eq!(v.map(|v| v.len()), Some(2));
    }
}

/// fastText's `word2int` table rebuilt from the public API (same hash, size
/// and linear probing): the longest run of occupied slots, which bounds
/// every lookup's probes.
fn longest_probe_run(m: &Model) -> usize {
    let entries: Vec<&[u8]> = (0..m.nwords())
        .map(|i| m.word(i))
        .chain((0..m.nlabels()).map(|i| m.label(i)))
        .collect();
    let len = (entries.len() as f64 / 0.7).ceil() as usize;
    let mut table = vec![usize::MAX; len];
    for (i, e) in entries.iter().enumerate() {
        let mut slot = hash(e) as usize % len;
        while table[slot] != usize::MAX && entries[table[slot]] != *e {
            slot = (slot + 1) % len;
        }
        table[slot] = i;
    }
    let (mut best, mut run) = (0, 0);
    // Twice round, for the run that wraps.
    for k in 0..2 * len {
        if table[k % len] == usize::MAX {
            run = 0;
        } else {
            run += 1;
            best = best.max(run.min(len));
        }
    }
    best
}

/// `n` distinct words for a table of `ceil(n / 0.7)` slots whose home slots
/// are all 0 (`same`) or exactly `0..n`.
fn crafted_words(n: usize, same: bool) -> Vec<Vec<u8>> {
    let len = (n as f64 / 0.7).ceil() as usize;
    let mut by_slot: Vec<Option<Vec<u8>>> = vec![None; n];
    let mut pile = Vec::new();
    let mut i = 0u64;
    while if same {
        pile.len() < n
    } else {
        by_slot.iter().any(Option::is_none)
    } {
        let w = format!("w{i:x}").into_bytes();
        i += 1;
        let slot = hash(&w) as usize % len;
        if same && slot == 0 {
            pile.push(w);
        } else if !same && slot < n && by_slot[slot].is_none() {
            by_slot[slot] = Some(w);
        }
    }
    if same {
        pile
    } else {
        by_slot.into_iter().flatten().collect()
    }
}

fn words_only(words: Vec<Vec<u8>>) -> Spec {
    Spec {
        model: CBOW,
        loss: NS,
        words,
        labels: Vec::new(),
        dim: 1,
        minn: 0,
        maxn: 0,
        bucket: 0,
        ..Spec::default()
    }
}

/// Linear probing scans a whole run of occupied slots: dictionaries crafted
/// to collide are refused (fastText would load them, quadratically).
#[test]
fn hash_flooded_dictionaries_are_rejected() {
    for same in [true, false] {
        let bytes = words_only(crafted_words(1100, same)).build().bytes;
        let case = run(&bytes, Depth::Light);
        assert!(is_invalid(&case.outcome), "same {same}: {:?}", case.outcome);
        assert!(case.elapsed < Duration::from_secs(1));
        // Up to the cap they load, and lookups still work.
        let words = crafted_words(1000, same);
        let m = load_spec(&words_only(words.clone()));
        assert_eq!(longest_probe_run(&m), 1000);
        for (i, w) in words.iter().enumerate().step_by(97) {
            assert_eq!(m.word_id(std::str::from_utf8(w).expect("ascii")), i as i32);
        }
        assert_eq!(m.word_id("absent"), -1);
    }
    // An ordinary large dictionary is far from the cap.
    let mut rng = Rng::new(7);
    let words: Vec<Vec<u8>> = (0..20_000)
        .map(|i| {
            let n = 2 + rng.below(10);
            let mut w: Vec<u8> = (0..n).map(|_| b'a' + rng.below(26) as u8).collect();
            w.extend_from_slice(format!("{i}").as_bytes());
            w
        })
        .collect();
    let m = load_spec(&words_only(words));
    assert!(longest_probe_run(&m) < 200, "{}", longest_probe_run(&m));
}

// ===================================================================== reading paths

/// A reader that hands out at most `chunk` bytes per call, then fails at `fail_at`.
struct Trickle<'a> {
    data: &'a [u8],
    pos: usize,
    chunk: usize,
    fail_at: usize,
}

impl Read for Trickle<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.fail_at {
            return Err(io::Error::other("disk on fire"));
        }
        let n = buf
            .len()
            .min(self.chunk)
            .min(self.data.len() - self.pos)
            .min(self.fail_at - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

#[test]
fn read_paths_handle_short_sources_errors_and_small_reads() {
    for (name, spec) in kinds() {
        let bytes = spec.build().bytes;
        let len = bytes.len() as u64;
        let clean = load(&bytes);
        let trickle = |chunk, fail_at| Trickle {
            data: &bytes,
            pos: 0,
            chunk,
            fail_at,
        };
        let m = Model::read(trickle(1, usize::MAX), len).expect(name);
        assert_eq!(
            m.predict("the cat", -1, 0.0),
            clean.predict("the cat", -1, 0.0)
        );
        let m = Model::read(trickle(7, usize::MAX), len).expect(name);
        assert_eq!(m.line_ids("café zebra"), clean.line_ids("café zebra"));
        // The source is shorter than its declared length (a file that shrank).
        assert!(matches!(
            Model::read(&bytes[..bytes.len() - 1], len),
            Err(LoadError::Truncated { .. })
        ));
        assert!(matches!(
            Model::read(&bytes[..], len - 1),
            Err(LoadError::Truncated { .. })
        ));
        assert!(matches!(
            Model::read(&bytes[..], len + 1000),
            Ok(_) | Err(LoadError::Truncated { .. })
        ));
        for fail_at in [0, 3, 100, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                matches!(Model::read(trickle(5, fail_at), len), Err(LoadError::Io(_))),
                "{name}: fail at {fail_at}"
            );
        }
    }
}

/// `Model::read` trusts `len`, so its size checks pass for a source that
/// claims to be endless; an absurd matrix must then still fail through
/// `try_reserve_exact` rather than abort.
#[test]
fn an_overstated_length_cannot_force_an_absurd_allocation() {
    let built = Spec::default().build();
    let n = built.field("in.n");
    // 2^58 x 4 floats: 2^62 bytes, beyond any address space.
    let mut head = built.patched("in.m", 1 << 58);
    head.truncate(n.offset + n.width);
    let source = head.as_slice().chain(io::repeat(0));
    let res = Model::read(source, u64::MAX);
    assert!(matches!(res, Err(LoadError::Invalid(_))), "{res:?}");
}

#[test]
fn load_from_disk() {
    let dir = std::env::temp_dir().join(format!("pyroid-fasttext-load-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let bytes = Spec::default().build().bytes;
    let path = dir.join("m.bin");
    std::fs::write(&path, &bytes).expect("write");
    let m = Model::load(&path).expect("loads");
    assert_eq!(m.nlabels(), 5);
    std::fs::write(&path, &bytes[..bytes.len() / 2]).expect("write");
    assert!(matches!(
        Model::load(&path),
        Err(LoadError::Truncated { .. })
    ));
    assert!(matches!(
        Model::load(dir.join("missing.bin")),
        Err(LoadError::Io(_))
    ));
    assert!(Model::load(&dir).is_err(), "a directory");
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

// ===================================================================== 3. mutation fuzz

fn interesting_value(rng: &mut Rng, width: usize) -> i64 {
    let pool = if width == 4 {
        i32_values(0)
    } else {
        i64_values(0)
    };
    pool[rng.below(pool.len())]
}

/// One random structural mutation of `b` (`others`: models to splice with).
fn mutate(rng: &mut Rng, base: &Built, others: &[Built]) -> Vec<u8> {
    let mut b = base.bytes.clone();
    let meta_end = base.field("quant_input").offset + 1;
    let rounds = [1, 1, 1, 2, 3][rng.below(5)];
    for _ in 0..rounds {
        if b.is_empty() {
            b.push(rng.byte());
        }
        let len = b.len();
        match rng.below(10) {
            0 => {
                for _ in 0..=rng.below(4) {
                    let i = rng.below(len);
                    b[i] ^= 1 << rng.below(8);
                }
            }
            1 => {
                let i = rng.below(meta_end.min(len));
                b[i] = rng.byte();
            }
            2 | 3 => {
                let f = &base.fields[rng.below(base.fields.len())];
                if f.offset + f.width <= len {
                    let v = if f.width == 1 {
                        i64::from(rng.byte())
                    } else {
                        interesting_value(rng, f.width)
                    };
                    b[f.offset..f.offset + f.width].copy_from_slice(&v.to_le_bytes()[..f.width]);
                }
            }
            4 => {
                let at = rng.below(len + 1);
                let n = 1 + rng.below(16);
                let ins: Vec<u8> = (0..n).map(|_| rng.byte()).collect();
                b.splice(at..at, ins);
            }
            5 => {
                let at = rng.below(len);
                let n = (1 + rng.below(64)).min(len - at);
                b.drain(at..at + n);
            }
            6 => {
                let src = rng.below(len);
                let n = (1 + rng.below(64)).min(len - src);
                let chunk = b[src..src + n].to_vec();
                let at = rng.below(len + 1);
                b.splice(at..at, chunk);
            }
            7 => {
                let other = &others[rng.below(others.len())].bytes;
                let cut = rng.below(len + 1);
                let from = rng.below(other.len() + 1);
                b.truncate(cut);
                b.extend_from_slice(&other[from..]);
            }
            8 => {
                // A field of one model at the same offset of another.
                let other = &others[rng.below(others.len())];
                let f = &other.fields[rng.below(other.fields.len())];
                let w = f.width.min(len.saturating_sub(f.offset));
                b[f.offset.min(len)..f.offset.min(len) + w]
                    .copy_from_slice(&other.bytes[f.offset..f.offset + w]);
            }
            _ => {
                // An unaligned interesting value in the header.
                let width = [4, 8][rng.below(2)];
                if len >= width {
                    let at = rng.below((meta_end + 8).min(len - width + 1));
                    let v = interesting_value(rng, width);
                    b[at..at + width].copy_from_slice(&v.to_le_bytes()[..width]);
                }
            }
        }
    }
    b
}

fn fuzz(cases_per_kind: usize, depth: Depth, name: &str) {
    let built: Vec<(&str, Built)> = kinds().into_iter().map(|(n, s)| (n, s.build())).collect();
    let all: Vec<Built> = built.iter().map(|(_, b)| b.clone()).collect();
    let mut tally = Tally::default();
    // One thread per model kind (the heap accounting is per thread).
    std::thread::scope(|s| {
        let workers: Vec<_> = built
            .iter()
            .enumerate()
            .map(|(k, (kind, base))| {
                let all = &all;
                s.spawn(move || {
                    let mut tally = Tally::default();
                    let mut rng = Rng::new(0xF00D + k as u64);
                    for i in 0..cases_per_kind {
                        let bytes = mutate(&mut rng, base, all);
                        let case = run(&bytes, depth);
                        tally.add(
                            &|| format!("{kind} case {i}: {}", hex(&bytes)),
                            &bytes,
                            &case,
                        );
                    }
                    tally
                })
            })
            .collect();
        for w in workers {
            tally.merge(w.join().expect("fuzz worker"));
        }
    });
    assert!(tally.ok() > 0, "nothing loaded");
    tally.finish(name);
}

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    for x in b.iter().take(400) {
        let _ = write!(s, "{x:02x}");
    }
    if b.len() > 400 {
        let _ = write!(s, "... ({} bytes)", b.len());
    }
    s
}

#[test]
fn mutation_fuzz() {
    fuzz(1000, Depth::Light, "mutation fuzz");
}

#[test]
#[ignore = "heavy: 300k cases, then the first 30k again with the full exercise"]
fn mutation_fuzz_heavy() {
    fuzz(50_000, Depth::Light, "mutation fuzz (heavy)");
    fuzz(5_000, Depth::Full, "mutation fuzz (heavy, full exercise)");
}

// ===================================================================== 4. semantics

#[test]
fn label_id_of_a_word_is_its_id_minus_nwords() {
    let m = load_spec(&Spec::default());
    assert_eq!(m.nwords(), 8);
    assert_eq!(m.label_id("__label__a"), 0);
    assert_eq!(m.label_id("__label__e"), 4);
    assert_eq!(m.word_id("the"), 1);
    assert_eq!(m.label_id("the"), 1 - 8, "fastText's getLabelId quirk");
    assert_eq!(m.label_id("</s>"), -8);
    assert_eq!(m.label_id("zebra"), -1);
    assert_eq!(m.label_id("__label__zz"), -1);
    assert_eq!(m.word_id("__label__b"), 9);
}

#[test]
fn version_11_supervised_models_have_no_char_ngrams() {
    let v11 = load_spec(&Spec {
        version: 11,
        ..Spec::default()
    });
    assert_eq!(v11.version(), 11);
    assert_eq!(v11.args().maxn, 0);
    assert_eq!(v11.args().minn, 2);
    // An unknown word contributes nothing; only </s> remains.
    assert_eq!(v11.line_ids("zebra"), vec![0]);
    assert_eq!(v11.line_ids("cat"), vec![2, 0]);
    let v12 = load_spec(&Spec::default());
    assert_eq!(v12.args().maxn, 4);
    assert!(v12.line_ids("zebra").len() > 1);
    let cbow11 = load_spec(&Spec {
        version: 11,
        model: CBOW,
        labels: Vec::new(),
        ..Spec::default()
    });
    assert_eq!(cbow11.args().maxn, 4, "only supervised models are patched");
    assert_eq!(cbow11.args().model, ModelKind::Cbow);
}

#[test]
fn predict_error_order() {
    let m = load_spec(&Spec::default());
    for t in ["a\nb", "\n", "the cat\n", "\n\n"] {
        assert_eq!(m.predict(t, 1, 0.0), Err(PredictError::Newline), "{t:?}");
        assert_eq!(m.predict(t, 0, 0.0), Err(PredictError::Newline), "{t:?}");
        assert_eq!(m.sentence_vector(t), Err(PredictError::Newline), "{t:?}");
    }
    for k in [0, -2, -100, i64::MIN] {
        assert_eq!(
            m.predict("the cat", k, 0.0),
            Err(PredictError::BadK),
            "k {k}"
        );
        // `</s>` is a known word here, so even an empty line has a token.
        assert_eq!(m.predict("", k, 0.0), Err(PredictError::BadK), "k {k}");
    }
    assert_eq!(m.predict("the cat", -1, 0.0).map(|p| p.len()), Ok(5));
    assert_eq!(m.predict("the cat", 2, 0.0).map(|p| p.len()), Ok(2));
    assert_eq!(m.predict("the cat", i64::MAX, 0.0).map(|p| p.len()), Ok(5));
    assert_eq!(m.predict("the cat", -1, 1.5), Ok(Vec::new()));

    // Without `</s>` in the dictionary and without n-grams, a line of
    // unknown words and labels has no tokens: no predictions and no checks.
    let bare = load_spec(&Spec {
        words: strs(&["the", "cat"]),
        maxn: 0,
        ..Spec::default()
    });
    for t in ["", "zebra qq", "__label__a __label__zz", " \t"] {
        for k in [0, -2, 1, -1] {
            assert_eq!(bare.predict(t, k, 0.0), Ok(Vec::new()), "{t:?} k {k}");
        }
    }
    assert_eq!(bare.predict("the", 0, 0.0), Err(PredictError::BadK));

    let cbow = load_spec(&Spec {
        model: CBOW,
        loss: NS,
        labels: Vec::new(),
        ..Spec::default()
    });
    assert_eq!(
        cbow.predict("the cat", 1, 0.0),
        Err(PredictError::NotSupervised)
    );
    assert_eq!(
        cbow.predict("the cat", 0, 0.0),
        Err(PredictError::NotSupervised)
    );
    assert_eq!(cbow.predict("x\n", 1, 0.0), Err(PredictError::Newline));
    let cbow_bare = load_spec(&Spec {
        model: CBOW,
        loss: NS,
        labels: Vec::new(),
        words: strs(&["the"]),
        maxn: 0,
        ..Spec::default()
    });
    assert_eq!(cbow_bare.predict("zebra", 1, 0.0), Ok(Vec::new()));
    assert_eq!(
        cbow_bare.predict("the", 1, 0.0),
        Err(PredictError::NotSupervised)
    );
}

#[test]
fn a_literal_eos_ends_the_line() {
    for spec in [
        Spec::default(),
        Spec {
            word_ngrams: 3,
            ..Spec::default()
        },
    ] {
        let m = load_spec(&spec);
        assert_eq!(m.line_ids("the </s> cat sat"), m.line_ids("the"));
        assert_eq!(m.line_ids("the\t</s>\u{0}cat"), m.line_ids("the"));
        assert_eq!(m.line("the </s> cat").0, vec!["the", "</s>"]);
        assert_eq!(m.line("the").0, vec!["the", "</s>"]);
        assert_eq!(m.line("</s>").0, vec!["</s>"]);
        assert_eq!(
            m.predict("the </s> cat", -1, 0.0),
            m.predict("the", -1, 0.0)
        );
        // `the</s>` is one (unknown) token, not an end of line.
        assert_ne!(m.line_ids("the</s> cat"), m.line_ids("the"));
    }
}

#[test]
fn label_tokens_are_skipped() {
    for spec in [
        Spec::default(),
        Spec {
            word_ngrams: 2,
            ..Spec::default()
        },
    ] {
        let m = load_spec(&spec);
        assert_eq!(
            m.line_ids("__label__a the __label__zz cat"),
            m.line_ids("the cat")
        );
        let (words, labels) = m.line("__label__a the __label__zz cat");
        assert_eq!(words, vec!["the", "cat", "</s>"]);
        assert_eq!(labels, vec!["__label__a"], "unknown labels are dropped");
    }
}

#[test]
fn pruned_models_can_have_more_subword_strings_than_ids() {
    let spec = Spec {
        quant: true,
        prune: Some(prune_index()),
        ..Spec::default()
    };
    let m = load_spec(&spec);
    let nwords = m.nwords() as i32;
    let mut longer = 0;
    for w in ["zebra", "cat", "café", "日本語", "qqqqqq"] {
        let (strings, ids) = m.subwords(w);
        assert!(strings.len() >= ids.len(), "{w}");
        longer += usize::from(strings.len() > ids.len());
        for &id in &ids {
            assert!(
                (0..nwords + 6).contains(&id),
                "{w}: id {id} outside the pruned rows"
            );
        }
        assert!(m.word_vector(w).is_some());
    }
    assert!(longer > 0, "no n-gram was pruned");
    let (strings, ids) = m.subwords("cat");
    assert_eq!((strings[0].as_str(), ids[0]), ("cat", 2));
}

#[test]
fn softmax_probabilities_sum_to_one() {
    let base = Spec {
        word_ngrams: 2,
        ..Spec::default()
    };
    for (spec, tolerance) in [
        (base.clone(), 1e-4),
        (
            Spec {
                quant: true,
                qout: true,
                qnorm: true,
                ..base.clone()
            },
            1e-4,
        ),
        (
            Spec {
                labels: (0..40)
                    .map(|i| format!("__label__{i}").into_bytes())
                    .collect(),
                dim: 10,
                dsub: 3,
                quant: true,
                qout: true,
                ..base.clone()
            },
            1e-4,
        ),
        // Hierarchical softmax: the leaves' path products sum to 1 too.
        (
            Spec {
                loss: HS,
                ..base.clone()
            },
            1e-3,
        ),
    ] {
        let m = load_spec(&spec);
        let n = m.nlabels() as f64;
        for t in TEXTS.iter().chain(&["zebra", "the cat sat"]) {
            let ps = m.predict(t, -1, 0.0).expect("predicts");
            assert_eq!(ps.len(), m.nlabels(), "{t:?}");
            let sum: f64 = ps.iter().map(|p| f64::from(p.prob)).sum();
            let want = if spec.loss == HS { 1.0 } else { 1.0 + n * 1e-5 };
            assert!(
                (sum - want).abs() < tolerance,
                "{t:?}: sum {sum}, want {want}"
            );
        }
    }
}

/// Non-finite weights load (fastText checks nothing); dense NaN scores are
/// `Err(NaN)`, everything else still returns in-range labels.
#[test]
fn non_finite_weights_never_panic() {
    let set = |built: &Built, after: &str, index: usize, x: f32| -> Vec<u8> {
        let f = built.field(after);
        let at = f.offset + f.width + 4 * index;
        let mut b = built.bytes.clone();
        b[at..at + 4].copy_from_slice(&x.to_le_bytes());
        b
    };
    for (name, spec) in kinds() {
        let built = spec.build();
        let dense_out = !(spec.quant && spec.qout);
        // The first float after the output (or input) shape / quantizer header.
        let targets = [
            if spec.quant { "in.pq.lastdsub" } else { "in.n" },
            if dense_out {
                "out.n"
            } else {
                "out.pq.lastdsub"
            },
        ];
        for target in targets {
            for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let bytes = set(&built, target, 0, x);
                let case = run(&bytes, Depth::Full);
                assert!(
                    matches!(case.outcome, Outcome::Loaded),
                    "{name} {target} {x}: {:?}",
                    case.outcome
                );
            }
        }
        if spec.model == SUPERVISED && dense_out && spec.loss != HS {
            // Every label's score is a dot product with a NaN row 0.
            let m = load(&set(&built, "out.n", 0, f32::NAN));
            assert_eq!(
                m.predict("the cat", 1, 0.0),
                Err(PredictError::NaN),
                "{name}"
            );
        }
    }
}

// ===================================================================== real models

/// Print a real model's longest `word2int` run (the loader caps it at 1024).
#[allow(clippy::print_stderr)] // a measurement the test report quotes (`--nocapture`)
fn note_probe_run(name: &str, m: &Model) {
    let run = longest_probe_run(m);
    eprintln!(
        "[{name}] {} entries, longest word2int run {run}",
        m.nwords() + m.nlabels()
    );
    assert!(run <= 1024);
}

const LID_FTZ: &str = "/data/models/lid.176.ftz";
const LID_BIN: &str = "/data/models/lid.176.bin";

/// Inside the dev container only (the models live in /data).
#[test]
fn real_model_loads_and_rejects_its_prefixes() {
    let path = Path::new(LID_FTZ);
    if !path.exists() {
        return;
    }
    let bytes = std::fs::read(path).expect("read");
    let m = Model::load(path).expect("lid.176.ftz loads");
    assert!(m.is_quantized());
    assert_eq!(m.nlabels(), 176);
    let p = m
        .predict("Bonjour tout le monde", 1, 0.0)
        .expect("predicts");
    assert_eq!(m.label(p[0].label), b"__label__fr");
    exercise(&m, Depth::Full);
    note_probe_run("lid.176.ftz", &m);
    // Long inputs cost linear time (debug build).
    let start = Instant::now();
    for text in ["x".repeat(200_000), "bonjour à tous ".repeat(20_000)] {
        assert_eq!(m.predict(&text, 3, 0.0).map(|p| p.len()), Ok(3));
        assert!(m.sentence_vector(&text).is_ok());
    }
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
    let mut tally = Tally::default();
    let n = bytes.len();
    let cuts = (0..64)
        .map(|i| i * 7)
        .chain((1..32).map(|i| n * i / 32))
        .chain([n - 1]);
    for cut in cuts {
        let case = run(&bytes[..cut], Depth::Light);
        if !matches!(case.outcome, Outcome::Rejected(LoadError::Truncated { .. })) {
            tally.fail(format!("prefix {cut}: {:?}", case.outcome));
        }
        tally.add(
            &|| format!("lid.176.ftz prefix {cut}"),
            &bytes[..cut],
            &case,
        );
    }
    tally.finish("lid.176.ftz prefixes");
}

/// Header and dictionary-header corruption of the real models.
#[test]
#[ignore = "heavy: hundreds of loads of lid.176.ftz, plus lid.176.bin"]
fn real_models_header_corruption() {
    for path in [LID_FTZ, LID_BIN] {
        if !Path::new(path).exists() {
            continue;
        }
        let bytes = std::fs::read(path).expect("read");
        let m = Model::from_bytes(&bytes).expect("loads");
        exercise(&m, Depth::Full);
        note_probe_run(path, &m);
        drop(m);
        let mut tally = Tally::default();
        let mut fields: Vec<Field> = [
            "magic",
            "version",
            "dim",
            "ws",
            "epoch",
            "minCount",
            "neg",
            "wordNgrams",
            "loss",
            "model",
            "bucket",
            "minn",
            "maxn",
            "lrUpdateRate",
        ]
        .iter()
        .enumerate()
        .map(|(i, name)| Field {
            name: (*name).to_owned(),
            offset: 4 * i,
            width: 4,
        })
        .collect();
        fields.extend(
            [
                ("dict.size", 64, 4),
                ("dict.nwords", 68, 4),
                ("dict.nlabels", 72, 4),
                ("dict.pruneidx_size", 84, 8),
            ]
            .map(|(name, offset, width)| Field {
                name: name.to_owned(),
                offset,
                width,
            }),
        );
        let big = bytes.len() > 10_000_000;
        for f in &fields {
            let orig = read_field(&bytes, f);
            let mut values = values_for(f, orig);
            if big {
                values
                    .retain(|v| [-1, 0, 1, 33, 1 << 30, i64::from(i32::MAX), i64::MAX].contains(v));
            }
            for v in values {
                let mut b = bytes.clone();
                b[f.offset..f.offset + f.width].copy_from_slice(&v.to_le_bytes()[..f.width]);
                let case = run(&b, Depth::Light);
                tally.add(&|| format!("{path}: {} = {v}", f.name), &b, &case);
            }
        }
        tally.finish(path);
    }
}
