//! `pyroid._pyroid.jieba`: the native half of `pyroid.jieba` (jieba 0.42.1's
//! segmentation, `pyroid_core::jieba`).
//!
//! `_Native` is one `jieba.Tokenizer`'s `FREQ` and `total` behind a lock;
//! each call segments a snapshot of it, so `add_word` from another thread
//! never changes a batch half-way. The Python package (`pyroid/jieba/`) adds
//! jieba's API on top: argument handling, generators, `load_userdict`'s line
//! parsing, custom dictionary files, tags, and the delegated submodules.
//!
//! `Force_Split_Words` is process-wide, as in jieba (`finalseg` keeps one set
//! for every `Tokenizer`).

use std::hash::BuildHasher;
use std::ops::Range;
use std::sync::{Arc, LazyLock, PoisonError, RwLock};

use pyo3::exceptions::{PyOverflowError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pyroid_core::jieba::{CutError, DictError, Dictionary, ForceSplit, SuggestError};
use pyroid_core::par::{self, PoolError};
use rustc_hash::{FxBuildHasher, FxHashMap};

use crate::batch::{DocCost, map_values_cost};
use crate::errors::{PyroidError, with_note};
use crate::input::{TextInput, Threads};

/// `finalseg.Force_Split_Words` (`None` while empty).
static FORCE_SPLIT: RwLock<Option<Arc<ForceSplit>>> = RwLock::new(None);
static EMPTY: LazyLock<ForceSplit> = LazyLock::new(ForceSplit::default);

fn force_split() -> Option<Arc<ForceSplit>> {
    FORCE_SPLIT
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// CPython 3.14 reworded `math.log`'s and float division's errors.
fn new_messages(py: Python<'_>) -> bool {
    py.version_info() >= (3, 14)
}

fn cut_error(py: Python<'_>, e: CutError) -> PyErr {
    match e {
        CutError::MathDomain => PyValueError::new_err(if new_messages(py) {
            "expected a positive input"
        } else {
            "math domain error"
        }),
    }
}

fn suggest_error(py: Python<'_>, e: SuggestError) -> PyErr {
    match e {
        SuggestError::Cut(e) => cut_error(py, e),
        SuggestError::ZeroDivision => PyZeroDivisionError::new_err(if new_messages(py) {
            "division by zero"
        } else {
            "float division by zero"
        }),
        SuggestError::Overflow => {
            PyOverflowError::new_err("pyroid.jieba: frequency does not fit in a 64-bit integer")
        }
    }
}

fn dict_error(e: DictError) -> PyErr {
    match e {
        DictError::Overflow => PyOverflowError::new_err(
            "pyroid.jieba: the total frequency does not fit in a 64-bit integer",
        ),
    }
}

/// One output word: a byte range of its document, or (full mode only) a new
/// string.
enum Piece {
    Span(usize, usize),
    Owned(Box<str>),
}

/// The byte range of `w` (a slice of `doc`) in `doc`.
fn span(doc: &str, w: &str) -> Piece {
    let start = w.as_ptr() as usize - doc.as_ptr() as usize;
    Piece::Span(start, start + w.len())
}

/// The text of `piece`, a word of `doc`.
fn piece_text<'d>(doc: &'d str, piece: &'d Piece) -> &'d str {
    match piece {
        Piece::Span(start, end) => &doc[*start..*end],
        Piece::Owned(word) => word,
    }
}

/// One output item: a word, or (`tokenize`) a word and its offsets.
trait Word: Sync {
    type Object<'py>: IntoPyObject<'py>;
    fn piece(&self) -> &Piece;
    /// The Python object, given the word's str.
    fn object<'py>(&self, s: Bound<'py, PyString>) -> Self::Object<'py>;
}

impl Word for Piece {
    type Object<'py> = Bound<'py, PyString>;
    fn piece(&self) -> &Piece {
        self
    }
    fn object<'py>(&self, s: Bound<'py, PyString>) -> Self::Object<'py> {
        s
    }
}

impl Word for (Piece, usize, usize) {
    type Object<'py> = (Bound<'py, PyString>, usize, usize);
    fn piece(&self) -> &Piece {
        &self.0
    }
    fn object<'py>(&self, s: Bound<'py, PyString>) -> Self::Object<'py> {
        (s, self.1, self.2)
    }
}

/// What a segmentation call returns per document.
type DocResult<T> = Option<Result<Vec<T>, CutError>>;

/// Document `i` of a batch and its words, unless it is an Arrow null.
fn doc_words<'a, 'd, W>(
    texts: &[Option<&'d str>],
    results: &'a [DocResult<W>],
    i: usize,
) -> Option<(&'d str, &'a [W])> {
    match (texts[i], &results[i]) {
        (Some(doc), Some(Ok(words))) => Some((doc, words)),
        _ => None,
    }
}

/// Words per interning task, at least: a batch with fewer than twice this is
/// interned in one go, without detaching.
const INTERN_MIN_WORDS_PER_TASK: usize = 1 << 13;

/// The words of documents `docs`, interned: their distinct texts, and every
/// word in order as an index into `texts`. `by_shard[s]` lists the indices of
/// the texts of shard `s` (empty with one shard).
struct Local<'d> {
    docs: Range<usize>,
    texts: Vec<&'d str>,
    ids: Vec<u32>,
    by_shard: Vec<Vec<u32>>,
}

/// The shard (of `shards`) a text's batch-wide index is decided in.
fn shard_of(text: &str, shards: usize) -> usize {
    // FxHashMap places entries by the low and top bits; the middle ones are free.
    let h = (FxBuildHasher.hash_one(text) >> 32) % shards as u64;
    usize::try_from(h).expect("below the shard count")
}

fn index(n: usize) -> u32 {
    u32::try_from(n).expect("fewer than 2**32 distinct words")
}

/// Intern the words of documents `docs` (pure Rust).
fn intern_local<'d, W: Word>(
    texts: &[Option<&'d str>],
    results: &'d [DocResult<W>],
    docs: Range<usize>,
    shards: usize,
) -> Local<'d> {
    let mut seen: FxHashMap<&'d str, u32> = FxHashMap::default();
    let mut distinct: Vec<&'d str> = Vec::new();
    let mut by_shard = vec![Vec::new(); if shards > 1 { shards } else { 0 }];
    let mut ids = Vec::new();
    for i in docs.clone() {
        let Some((doc, words)) = doc_words(texts, results, i) else {
            continue;
        };
        ids.reserve(words.len());
        for w in words {
            let text = piece_text(doc, w.piece());
            let id = *seen.entry(text).or_insert_with(|| {
                let id = index(distinct.len());
                distinct.push(text);
                if shards > 1 {
                    by_shard[shard_of(text, shards)].push(id);
                }
                id
            });
            ids.push(id);
        }
    }
    Local {
        docs,
        texts: distinct,
        ids,
        by_shard,
    }
}

/// Shard `s` of a batch's distinct texts: each text of that shard once, and
/// for each range its texts' indices into `texts` (in `by_shard[s]` order).
struct Shard<'d> {
    texts: Vec<&'d str>,
    of_range: Vec<Vec<u32>>,
}

fn merge_shard<'d>(locals: &[Local<'d>], s: usize) -> Shard<'d> {
    let mut seen: FxHashMap<&'d str, u32> = FxHashMap::default();
    let mut texts = Vec::new();
    let mut of_range = Vec::with_capacity(locals.len());
    for local in locals {
        let mut ids = Vec::with_capacity(local.by_shard[s].len());
        for &j in &local.by_shard[s] {
            let text = local.texts[j as usize];
            ids.push(*seen.entry(text).or_insert_with(|| {
                texts.push(text);
                index(texts.len() - 1)
            }));
        }
        of_range.push(ids);
    }
    Shard { texts, of_range }
}

/// The interned words of a batch: its distinct texts, and for each range of
/// documents its [`Local`] and, with several ranges, the batch-wide index of
/// each of the range's texts.
struct Interned<'d> {
    texts: Vec<&'d str>,
    parts: Vec<(Local<'d>, Option<Vec<u32>>)>,
}

/// Each text of `locals` once, in parallel: every range's distinct texts go
/// to the shard their hash picks, and each shard is merged on its own.
fn merge(locals: Vec<Local<'_>>, threads: par::Threads) -> Result<Interned<'_>, PoolError> {
    let ids: Vec<usize> = (0..locals.len()).collect();
    let shards = par::map_tasks(&ids, threads, |&s| merge_shard(&locals, s))?;
    let mut offsets = Vec::with_capacity(shards.len());
    let mut n = 0;
    for shard in &shards {
        offsets.push(index(n));
        n += shard.texts.len();
    }
    let remaps = par::map_tasks(&ids, threads, |&r| {
        let mut remap = vec![0; locals[r].texts.len()];
        for (s, shard) in shards.iter().enumerate() {
            for (&j, &k) in locals[r].by_shard[s].iter().zip(&shard.of_range[r]) {
                remap[j as usize] = offsets[s] + k;
            }
        }
        remap
    })?;
    let texts = shards.into_iter().flat_map(|s| s.texts).collect();
    let parts = locals
        .into_iter()
        .zip(remaps.into_iter().map(Some))
        .collect();
    Ok(Interned { texts, parts })
}

/// Up to `n` contiguous ranges of `weights` with about equal sums.
fn balanced(weights: &[usize], n: usize) -> Vec<Range<usize>> {
    let share = weights.iter().sum::<usize>().div_ceil(n);
    let mut out = Vec::with_capacity(n);
    let (mut start, mut acc) = (0, 0);
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if out.len() + 1 < n && acc >= share * (out.len() + 1) {
            out.push(start..i + 1);
            start = i + 1;
        }
    }
    if start < weights.len() || out.is_empty() {
        out.push(start..weights.len());
    }
    out
}

/// The interned words of a batch: a small one on the calling thread, still
/// attached (with the GIL, detaching again would cost each call a second
/// hand-off); a larger one detached, on the pool, in contiguous ranges of
/// about equal word counts, then merged.
fn intern_batch<'d, W: Word>(
    py: Python<'_>,
    texts: &[Option<&'d str>],
    results: &'d [DocResult<W>],
    threads: Option<Threads>,
) -> PyResult<Interned<'d>> {
    let threads = crate::input::threads(threads);
    let weights: Vec<usize> = (0..results.len())
        .map(|i| doc_words(texts, results, i).map_or(0, |(_, w)| w.len()))
        .collect();
    let total: usize = weights.iter().sum();
    let one = |mut local: Local<'d>| Interned {
        texts: std::mem::take(&mut local.texts),
        parts: vec![(local, None)],
    };
    if total < 2 * INTERN_MIN_WORDS_PER_TASK {
        return Ok(one(intern_local(texts, results, 0..results.len(), 1)));
    }
    let tasks = if threads == par::Threads::ONE {
        1
    } else {
        par::max_tasks(threads)
            .map_err(|e| PyroidError::new_err(e.to_string()))?
            .min(total / INTERN_MIN_WORDS_PER_TASK)
    };
    let ranges = balanced(&weights, tasks);
    let shards = ranges.len();
    py.detach(|| {
        let mut locals = par::map_tasks(&ranges, threads, |r| {
            intern_local(texts, results, r.clone(), shards)
        })?;
        if locals.len() == 1 {
            return Ok(one(locals.remove(0)));
        }
        merge(locals, threads)
    })
    .map_err(|e: PoolError| PyroidError::new_err(e.to_string()))
}

/// Each document of `x` (`None` for an Arrow null), in input order.
fn docs(x: &TextInput) -> Vec<Option<&str>> {
    match x {
        TextInput::Str(s) => vec![Some(&**s)],
        TextInput::List(d) | TextInput::Tuple(d) => d.iter().map(|s| Some(&**s)).collect(),
        TextInput::Arrow(strings) => strings
            .chunks
            .iter()
            .flat_map(|c| (0..c.len()).map(move |i| c.get(i).and_then(Result::ok)))
            .collect(),
    }
}

/// Build the result: a list of words for a str, else a list with one list per
/// document (`None` for Arrow nulls). The lowest failing document raises
/// (for a batch, with a note giving its index).
///
/// A batch repeats words, and finding a text again costs less than decoding
/// its UTF-8 into a new str, so each distinct text of a batch becomes one str
/// (the lists then share str objects, which only `is` can tell). Finding the
/// distinct texts is pure Rust, done detached and on the pool for a large
/// batch; the strs and lists are then made on the calling thread, which owns
/// them (on free-threaded builds, freeing objects is cheapest there). A single
/// str repeats too little for that to pay: each word is decoded.
fn build<'py, W: Word>(
    py: Python<'py>,
    x: &TextInput,
    results: &[DocResult<W>],
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    let failed = results.iter().enumerate().find_map(|(i, r)| match r {
        Some(Err(e)) => Some((i, *e)),
        _ => None,
    });
    if let Some((i, e)) = failed {
        let err = cut_error(py, e);
        return Err(match x {
            TextInput::Str(_) => err,
            _ => with_note(
                py,
                err,
                format!("while segmenting the document at index {i}"),
            ),
        });
    }
    if let (TextInput::Str(s), [Some(Ok(words))]) = (x, results) {
        let items = words
            .iter()
            .map(|w| w.object(PyString::new(py, piece_text(s, w.piece()))));
        return Ok(PyList::new(py, items)?.into_any());
    }
    let texts = docs(x);
    let interned = intern_batch(py, &texts, results, threads)?;
    let strs: Vec<Bound<'py, PyString>> = interned
        .texts
        .iter()
        .map(|t| PyString::new(py, t))
        .collect();
    let mut out: Vec<Option<Bound<'py, PyList>>> = Vec::with_capacity(results.len());
    for (local, remap) in &interned.parts {
        let mut ids: &[u32] = &local.ids;
        for i in local.docs.clone() {
            let list = match doc_words(&texts, results, i) {
                Some((_, words)) => {
                    let (mine, rest) = ids.split_at(words.len());
                    ids = rest;
                    let items = words.iter().zip(mine).map(|(w, &id)| {
                        let id = remap.as_ref().map_or(id, |m| m[id as usize]);
                        w.object(strs[id as usize].clone())
                    });
                    Some(PyList::new(py, items)?)
                }
                None => None,
            };
            out.push(list);
        }
    }
    Ok(PyList::new(py, out)?.into_any())
}

/// One `jieba.Tokenizer`'s dictionary (`FREQ` and `total`).
#[pyclass(frozen, module = "pyroid.jieba", name = "_Native")]
#[derive(Debug)]
pub struct Native {
    dict: RwLock<Arc<Dictionary>>,
}

impl Native {
    fn from_dict(d: Dictionary) -> Self {
        Self {
            dict: RwLock::new(Arc::new(d)),
        }
    }

    fn snapshot(&self) -> Arc<Dictionary> {
        self.dict
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Run `f` on every document with this dictionary and the current
    /// `Force_Split_Words`, detached and in parallel.
    fn run<R: Send>(
        &self,
        py: Python<'_>,
        x: &TextInput,
        threads: Option<Threads>,
        f: impl Fn(&Dictionary, &ForceSplit, &str) -> Result<Vec<R>, CutError> + Sync,
    ) -> PyResult<Vec<DocResult<R>>> {
        let dict = self.snapshot();
        let fs = force_split();
        let fs: &ForceSplit = fs.as_deref().unwrap_or(&EMPTY);
        map_values_cost(py, x, threads, DocCost::Light, |s| f(&dict, fs, s))
    }
}

/// `bytes.strip()`: Python strips b" \t\n\r\x0b\x0c" (`trim_ascii` keeps `\x0b`).
fn py_bytes_strip(b: &[u8]) -> &[u8] {
    let ws = |c: &u8| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c');
    let start = b.iter().position(|c| !ws(c)).unwrap_or(b.len());
    let end = b.iter().rposition(|c| !ws(c)).map_or(start, |i| i + 1);
    &b[start..end]
}

/// Parse a dictionary file the way `gen_pfdict` does, when every line is
/// `word freq[ ...]` with an ASCII-digit `freq`: `(entries, total)`. `None`
/// for anything else (the Python layer then parses it, with jieba's errors).
fn parse_simple(data: &[u8]) -> Option<(Vec<(String, i64)>, i64)> {
    let body = data.strip_suffix(b"\n").unwrap_or(data);
    if body.is_empty() && data.is_empty() {
        return Some((Vec::new(), 0));
    }
    let mut entries = Vec::new();
    let mut total: i64 = 0;
    for raw in body.split(|&b| b == b'\n') {
        let line = std::str::from_utf8(py_bytes_strip(raw)).ok()?;
        let mut parts = line.split(' ');
        let word = parts.next()?;
        let freq = parts.next()?;
        if freq.is_empty() || !freq.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let freq: i64 = freq.parse().ok()?;
        total = total.checked_add(freq)?;
        entries.push((word.to_owned(), freq));
    }
    Some((entries, total))
}

#[pymethods]
impl Native {
    /// jieba's default dictionary (`dict.txt` of jieba 0.42.1).
    #[new]
    fn new() -> Self {
        Self::from_dict(Dictionary::default_ref().clone())
    }

    /// A dictionary from `(word, freq)` entries in file order and the file's
    /// total, as `gen_pfdict` builds it.
    #[staticmethod]
    fn from_entries(
        py: Python<'_>,
        words: Vec<String>,
        freqs: Vec<i64>,
        total: i64,
    ) -> PyResult<Self> {
        if words.len() != freqs.len() {
            return Err(PyValueError::new_err("words and freqs differ in length"));
        }
        let entries: Vec<(String, i64)> = words.into_iter().zip(freqs).collect();
        py.detach(|| Dictionary::from_entries(entries, total))
            .map(Self::from_dict)
            .map_err(dict_error)
    }

    /// A dictionary from a file's bytes when every line is plain
    /// `word freq[ tag]`, else `None`.
    #[staticmethod]
    fn from_simple_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Option<Self>> {
        let data = data.to_vec();
        py.detach(|| {
            parse_simple(&data)
                .map(|(entries, total)| Dictionary::from_entries(entries, total))
                .transpose()
        })
        .map(|d| d.map(Self::from_dict))
        .map_err(dict_error)
    }

    /// `jieba.cut(sentence, cut_all, HMM)` as a list; a list of lists for a batch.
    #[pyo3(signature = (sentence, /, cut_all = false, hmm = true, *, threads = None))]
    fn cut<'py>(
        &self,
        py: Python<'py>,
        sentence: TextInput,
        cut_all: bool,
        hmm: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = self.run(py, &sentence, threads, |d, fs, s| {
            if cut_all {
                let mut out = Vec::new();
                d.cut_all(s, &mut out);
                Ok(out
                    .into_iter()
                    .map(|w| match w {
                        std::borrow::Cow::Borrowed(w) => span(s, w),
                        std::borrow::Cow::Owned(w) => Piece::Owned(w.into()),
                    })
                    .collect())
            } else {
                let mut out = Vec::new();
                d.cut(s, hmm, fs, &mut out)?;
                Ok(out.into_iter().map(|w| span(s, w)).collect())
            }
        })?;
        build(py, &sentence, &results, threads)
    }

    /// `jieba.cut_for_search(sentence, HMM)` as a list (a list of lists for a batch).
    #[pyo3(signature = (sentence, /, hmm = true, *, threads = None))]
    fn cut_for_search<'py>(
        &self,
        py: Python<'py>,
        sentence: TextInput,
        hmm: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = self.run(py, &sentence, threads, |d, fs, s| {
            let mut out = Vec::new();
            d.cut_for_search(s, hmm, fs, &mut out)?;
            Ok(out.into_iter().map(|w| span(s, w)).collect())
        })?;
        build(py, &sentence, &results, threads)
    }

    /// `jieba.tokenize(unicode_sentence, mode, HMM)` as a list of `(word, start, end)`.
    #[pyo3(signature = (unicode_sentence, /, search = false, hmm = true, *, threads = None))]
    fn tokenize<'py>(
        &self,
        py: Python<'py>,
        unicode_sentence: TextInput,
        search: bool,
        hmm: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = self.run(py, &unicode_sentence, threads, |d, fs, s| {
            let mut out = Vec::new();
            d.tokenize(s, search, hmm, fs, &mut out)?;
            Ok(out
                .into_iter()
                .map(|(w, a, b)| (span(s, w), a, b))
                .collect())
        })?;
        build(py, &unicode_sentence, &results, threads)
    }

    /// `FREQ[word] = freq; total += freq` and the missing prefixes with 0.
    fn add_word(&self, word: &str, freq: i64) -> PyResult<()> {
        let mut g = self.dict.write().unwrap_or_else(PoisonError::into_inner);
        Arc::make_mut(&mut g)
            .add_word(word, freq)
            .map_err(dict_error)
    }

    /// `add_word` for each `(word, freq)`, in order, under one lock.
    fn add_words(&self, words: Vec<String>, freqs: Vec<i64>) -> PyResult<()> {
        if words.len() != freqs.len() {
            return Err(PyValueError::new_err("words and freqs differ in length"));
        }
        let mut g = self.dict.write().unwrap_or_else(PoisonError::into_inner);
        let d = Arc::make_mut(&mut g);
        for (w, f) in words.iter().zip(freqs) {
            d.add_word(w, f).map_err(dict_error)?;
        }
        Ok(())
    }

    /// `FREQ.get(word)`.
    fn get(&self, word: &str) -> Option<i64> {
        self.snapshot().get(word)
    }

    /// `len(FREQ)`.
    fn __len__(&self) -> usize {
        self.snapshot().len()
    }

    /// `Tokenizer.total`.
    #[getter]
    fn total(&self) -> i64 {
        self.snapshot().total()
    }

    /// Every `(key, value)` of `FREQ`: the dictionary's in UTF-8 byte order,
    /// then the added keys.
    fn entries(&self, py: Python<'_>) -> Vec<(String, i64)> {
        let d = self.snapshot();
        py.detach(|| d.entries())
    }

    /// `get_DAG(sentence)`.
    fn get_dag<'py>(&self, py: Python<'py>, sentence: &str) -> PyResult<Bound<'py, PyDict>> {
        let dag = self.snapshot().dag(sentence);
        let out = PyDict::new(py);
        for (k, ends) in dag.into_iter().enumerate() {
            out.set_item(k, ends)?;
        }
        Ok(out)
    }

    /// `suggest_freq(word)` for a str (without tuning).
    fn suggest_freq(&self, py: Python<'_>, word: &str) -> PyResult<i64> {
        self.snapshot()
            .suggest_freq(word)
            .map_err(|e| suggest_error(py, e))
    }

    /// `suggest_freq(segments)` for a tuple of segments (without tuning).
    fn suggest_freq_split(&self, py: Python<'_>, segments: Vec<String>) -> PyResult<i64> {
        let segs: Vec<&str> = segments.iter().map(String::as_str).collect();
        self.snapshot()
            .suggest_freq_split(&segs)
            .map_err(|e| suggest_error(py, e))
    }

    /// An independent copy (later changes to either do not affect the other).
    fn copy(&self) -> Self {
        Self::from_dict((*self.snapshot()).clone())
    }

    /// Whether this is jieba's default dictionary, unmodified.
    fn is_pristine_default(&self) -> bool {
        self.snapshot().is_pristine_default()
    }
}

/// `finalseg.add_force_split(word)`.
#[pyfunction]
fn add_force_split(word: &str) {
    let mut g = FORCE_SPLIT.write().unwrap_or_else(PoisonError::into_inner);
    Arc::make_mut(g.get_or_insert_with(Default::default)).insert(word.into());
}

/// The words in `finalseg.Force_Split_Words`, sorted.
#[pyfunction]
fn force_split_words() -> Vec<String> {
    let mut v: Vec<String> = force_split()
        .map(|s| s.iter().map(std::string::ToString::to_string).collect())
        .unwrap_or_default();
    v.sort_unstable();
    v
}

/// Empty `finalseg.Force_Split_Words` (`Force_Split_Words.clear()`).
#[pyfunction]
fn clear_force_split() {
    *FORCE_SPLIT.write().unwrap_or_else(PoisonError::into_inner) = None;
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.jieba")?;
    m.gil_used(false)?;
    m.add_class::<Native>()?;
    m.add_function(wrap_pyfunction!(add_force_split, &m)?)?;
    m.add_function(wrap_pyfunction!(force_split_words, &m)?)?;
    m.add_function(wrap_pyfunction!(clear_force_split, &m)?)?;
    m.add("DEFAULT_TOTAL", pyroid_core::jieba::DEFAULT_TOTAL)?;
    m.add("DEFAULT_FREQ_LEN", pyroid_core::jieba::DEFAULT_FREQ_LEN)?;
    parent.add_submodule(&m)
}
