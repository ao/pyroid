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

use std::sync::{Arc, LazyLock, PoisonError, RwLock};

use pyo3::exceptions::{PyOverflowError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use pyroid_core::jieba::{CutError, DictError, Dictionary, ForceSplit, SuggestError};
use rustc_hash::FxHashMap;

use crate::batch::{DocCost, map_values_cost};
use crate::errors::with_note;
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

/// The Python strs already made in this call, by text: jieba's output repeats
/// words, and a cache hit is cheaper than decoding the UTF-8 again. (The
/// lists then share str objects, which only `is` can tell.)
type StrCache<'py, 'd> = FxHashMap<&'d str, Bound<'py, PyString>>;

/// Entries the cache keeps before it starts over.
const STR_CACHE_MAX: usize = 1 << 16;

fn piece_str<'py, 'd>(
    py: Python<'py>,
    doc: &'d str,
    piece: &Piece,
    cache: &mut StrCache<'py, 'd>,
) -> Bound<'py, PyString> {
    match piece {
        Piece::Span(start, end) => {
            let word = &doc[*start..*end];
            if let Some(hit) = cache.get(word) {
                return hit.clone();
            }
            if cache.len() >= STR_CACHE_MAX {
                cache.clear();
            }
            let new = PyString::new(py, word);
            cache.insert(word, new.clone());
            new
        }
        Piece::Owned(word) => PyString::new(py, word),
    }
}

/// What a segmentation call returns per document.
type DocResult<T> = Option<Result<Vec<T>, CutError>>;

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

/// Build the result: a list for a str, else a list with one list per
/// document (`None` for Arrow nulls). The first failing document raises.
fn build<'py, 'd, T>(
    py: Python<'py>,
    x: &'d TextInput,
    results: Vec<DocResult<T>>,
    item: impl Fn(Python<'py>, &'d str, &T, &mut StrCache<'py, 'd>) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let texts = docs(x);
    let mut cache = StrCache::default();
    let mut one = |doc: &'d str,
                   r: Result<Vec<T>, CutError>,
                   i: Option<usize>|
     -> PyResult<Bound<'py, PyAny>> {
        let words = r.map_err(|e| {
            let err = cut_error(py, e);
            match i {
                Some(i) => with_note(
                    py,
                    err,
                    format!("while segmenting the document at index {i}"),
                ),
                None => err,
            }
        })?;
        let items = words
            .iter()
            .map(|w| item(py, doc, w, &mut cache))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, items)?.into_any())
    };
    if let TextInput::Str(s) = x {
        let r = results.into_iter().next().flatten().expect("one result");
        return one(s, r, None);
    }
    let mut out = Vec::with_capacity(results.len());
    for (i, (r, doc)) in results.into_iter().zip(texts).enumerate() {
        out.push(match (r, doc) {
            (Some(r), Some(doc)) => one(doc, r, Some(i))?,
            _ => py.None().into_bound(py),
        });
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

    /// `jieba.cut(x, cut_all, HMM)` as a list; a list of lists for a batch.
    #[pyo3(signature = (x, /, cut_all = false, hmm = true, *, threads = None))]
    fn cut<'py>(
        &self,
        py: Python<'py>,
        x: TextInput,
        cut_all: bool,
        hmm: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = self.run(py, &x, threads, |d, fs, s| {
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
        build(py, &x, results, |py, doc, p, c| {
            Ok(piece_str(py, doc, p, c).into_any())
        })
    }

    /// `jieba.cut_for_search(x, HMM)` as a list (a list of lists for a batch).
    #[pyo3(signature = (x, /, hmm = true, *, threads = None))]
    fn cut_for_search<'py>(
        &self,
        py: Python<'py>,
        x: TextInput,
        hmm: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = self.run(py, &x, threads, |d, fs, s| {
            let mut out = Vec::new();
            d.cut_for_search(s, hmm, fs, &mut out)?;
            Ok(out.into_iter().map(|w| span(s, w)).collect())
        })?;
        build(py, &x, results, |py, doc, p, c| {
            Ok(piece_str(py, doc, p, c).into_any())
        })
    }

    /// `jieba.tokenize(x, mode, HMM)` as a list of `(word, start, end)`.
    #[pyo3(signature = (x, /, search = false, hmm = true, *, threads = None))]
    fn tokenize<'py>(
        &self,
        py: Python<'py>,
        x: TextInput,
        search: bool,
        hmm: bool,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let results = self.run(py, &x, threads, |d, fs, s| {
            let mut out = Vec::new();
            d.tokenize(s, search, hmm, fs, &mut out)?;
            Ok(out
                .into_iter()
                .map(|(w, a, b)| (span(s, w), a, b))
                .collect())
        })?;
        build(py, &x, results, |py, doc, (p, a, b), c| {
            Ok(PyTuple::new(
                py,
                [
                    piece_str(py, doc, p, c).into_any(),
                    a.into_pyobject(py)?.into_any(),
                    b.into_pyobject(py)?.into_any(),
                ],
            )?
            .into_any())
        })
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
