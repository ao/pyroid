// Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
//! `pyroid._pyroid.langdetect`: the native half of `pyroid.langdetect`
//! (`pyroid_core::langdetect`).
//!
//! `_Profiles` is one factory's `word_lang_prob_map` behind a lock; a detection reads a
//! frozen snapshot of it, so `add_profile` from another thread never changes a detection
//! half-way. `append_text` and `detect_block` are `Detector.append` and
//! `Detector._detect_block` on a detector's state; `detect_one` is the module-level
//! `detect`/`detect_langs` in one call; `detect_many` the batch functions. The Python
//! package (`pyroid/langdetect/`) keeps langdetect's classes and attributes on top.

use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, TryLockError};
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use pyroid_core::columnar::StringBuffers;
use pyroid_core::langdetect::chars::Upper;
use pyroid_core::langdetect::detect::{self, Failure, Outcome, Params, SumMode};
use pyroid_core::langdetect::mt::{self, EntropyKeys, Random, SeededState};
use pyroid_core::langdetect::profiles::{Builder, Frozen, Key, OutOfRange};
use pyroid_core::langdetect::{self as core, text};
use pyroid_core::par::{self, PoolError};

use crate::arrow::{self, ArrowStrings, Chunk, Column, ColumnType, StrType, build_error};
use crate::batch::BatchError;
use crate::input::{Threads, threads};

// ---------------------------------------------------------------------------
// Interpreter-dependent tables
// ---------------------------------------------------------------------------

static UPPER: OnceLock<Upper> = OnceLock::new();

/// `str.isupper` of the running interpreter, and how its `sum()` adds floats.
fn tables(py: Python<'_>) -> PyResult<(&'static Upper, SumMode)> {
    let sum = if py.version_info() >= (3, 12) {
        SumMode::Compensated
    } else {
        SumMode::Plain
    };
    if let Some(u) = UPPER.get() {
        return Ok((u, sum));
    }
    let version: String = py
        .import(intern!(py, "unicodedata"))?
        .getattr(intern!(py, "unidata_version"))?
        .extract()?;
    let upper = if let Some(u) = Upper::for_version(&version) {
        u
    } else {
        // A Unicode version newer than the built-in tables: ask the interpreter, once.
        let b = py.import(intern!(py, "builtins"))?;
        let map = b.getattr(intern!(py, "map"))?;
        let chars = map.call1((
            b.getattr(intern!(py, "chr"))?,
            b.getattr("range")?.call1((0x11_0000,))?,
        ))?;
        let flags = map.call1((b.getattr(intern!(py, "str"))?.getattr("isupper")?, chars))?;
        let bytes = b.getattr(intern!(py, "bytes"))?.call1((flags,))?;
        Upper::from_flags(bytes.cast::<PyBytes>()?.as_bytes())
    };
    let _ = UPPER.set(upper);
    Ok((UPPER.get().unwrap_or_else(|| unreachable!()), sum))
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

#[inline]
fn is_surrogate(c: u32) -> bool {
    (0xD800..=0xDFFF).contains(&c)
}

/// A str's text: its UTF-8 (no lone surrogates), or its code points.
enum Source<'a> {
    Utf8(&'a str),
    Cps(Cow<'a, [u32]>),
}

impl Source<'_> {
    /// The first `n` code points, and whether that is all of them.
    fn prefix(&self, n: usize) -> (Cow<'_, [u32]>, bool) {
        match self {
            Source::Utf8(s) => {
                let mut out = Vec::with_capacity(n.min(s.len()));
                let mut chars = s.chars();
                for c in chars.by_ref().take(n) {
                    out.push(u32::from(c));
                }
                let done = chars.next().is_none();
                (Cow::Owned(out), done)
            }
            Source::Cps(v) => (Cow::Borrowed(&v[..n.min(v.len())]), n >= v.len()),
        }
    }

    fn all(&self) -> Cow<'_, [u32]> {
        match self {
            Source::Utf8(s) => Cow::Owned(s.chars().map(u32::from).collect()),
            Source::Cps(v) => Cow::Borrowed(v),
        }
    }
}

/// All code points of a str whose UTF-8 cannot be taken (lone surrogates).
fn surrogate_code_points(s: &Bound<'_, PyString>) -> PyResult<Vec<u32>> {
    let py = s.py();
    let b = s.call_method1(intern!(py, "encode"), ("utf-32-le", "surrogatepass"))?;
    let b = b.cast::<PyBytes>()?.as_bytes();
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn source<'a>(s: &'a Bound<'_, PyString>) -> PyResult<Source<'a>> {
    match s.to_str() {
        Ok(t) => Ok(Source::Utf8(t)),
        Err(_) => Ok(Source::Cps(Cow::Owned(surrogate_code_points(s)?))),
    }
}

fn code_points(s: &Bound<'_, PyString>) -> PyResult<Vec<u32>> {
    Ok(source(s)?.all().into_owned())
}

/// A Python str from code points (lone surrogates included).
fn to_py_str<'py>(py: Python<'py>, cps: &[u32]) -> PyResult<Bound<'py, PyString>> {
    if cps.iter().any(|&c| is_surrogate(c)) {
        let mut b = Vec::with_capacity(cps.len() * 4);
        for c in cps {
            b.extend_from_slice(&c.to_le_bytes());
        }
        let s = PyBytes::new(py, &b)
            .call_method1(intern!(py, "decode"), ("utf-32-le", "surrogatepass"))?;
        Ok(s.cast_into::<PyString>()?)
    } else {
        let s: String = cps.iter().filter_map(|&c| char::from_u32(c)).collect();
        Ok(PyString::new(py, &s))
    }
}

/// `Detector.append(text)`'s addition to `Detector.text`, for an int `max_text_length`.
fn appended(text: &Bound<'_, PyString>, max_len: usize) -> PyResult<Vec<u32>> {
    let src = source(text)?;
    let mut take = text::first_prefix(max_len);
    loop {
        let (prefix, complete) = src.prefix(take);
        if let Some(out) = text::append_prefix(&prefix, complete, max_len) {
            return Ok(out);
        }
        take = take.saturating_mul(2);
    }
}

/// `Detector.append`: what `append(text)` adds to `Detector.text` (`max_text_length` already
/// clamped to `0..=sys.maxsize`).
#[pyfunction]
fn append_text<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
    max_len: usize,
) -> PyResult<Bound<'py, PyString>> {
    to_py_str(py, &appended(text, max_len)?)
}

/// `URL_RE.sub`, `MAIL_RE.sub` and `NGram.normalize_vi` over all of `text`.
#[pyfunction]
fn substitute<'py>(py: Python<'py>, text: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyString>> {
    to_py_str(py, &text::substitute_all(&code_points(text)?))
}

/// The first `n` code points of `text`, runs of spaces collapsed as `append` does.
#[pyfunction]
fn collapse<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
    n: usize,
) -> PyResult<Bound<'py, PyString>> {
    to_py_str(py, &text::collapse(&code_points(text)?, n))
}

/// `Detector.cleaning_text` on `text`: the new text, or `None` if it stays.
#[pyfunction]
fn clean_text<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyString>>> {
    text::clean(&code_points(text)?)
        .map(|v| to_py_str(py, &v))
        .transpose()
}

/// `NGram.normalize(ch)`.
#[pyfunction]
fn normalize<'py>(py: Python<'py>, ch: &Bound<'py, PyString>) -> PyResult<Bound<'py, PyString>> {
    let cps = code_points(ch)?;
    let [c] = cps[..] else {
        return Err(PyTypeError::new_err(format!(
            "ord() expected a character, but string of length {} found",
            cps.len()
        )));
    };
    to_py_str(py, &[core::chars::normalize(c)])
}

/// `NGram.normalize_vi(text)`.
#[pyfunction]
fn normalize_vi<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyString>> {
    to_py_str(py, &text::normalize_vi_all(&code_points(text)?))
}

/// The Unicode versions with a built-in `str.isupper` table.
#[pyfunction]
fn unicode_versions() -> Vec<&'static str> {
    Upper::versions()
}

// ---------------------------------------------------------------------------
// Locks
// ---------------------------------------------------------------------------

/// Lock `m` without ever blocking while attached: a thread waiting for a lock while attached
/// would hold up a stop-the-world pause (free-threaded) or the GIL, and the holder may need
/// either to finish.
fn lock<'a, T: Send>(py: Python<'_>, m: &'a Mutex<T>) -> MutexGuard<'a, T> {
    loop {
        match m.try_lock() {
            Ok(g) => return g,
            Err(TryLockError::Poisoned(p)) => return p.into_inner(),
            Err(TryLockError::WouldBlock) => {
                py.detach(|| drop(m.lock().unwrap_or_else(PoisonError::into_inner)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Inner {
    builder: Builder,
    frozen: Option<Arc<Frozen>>,
    /// Keys that are not str: the objects (by number) and a dict object -> row.
    foreign: Vec<Py<PyAny>>,
    foreign_rows: Option<Py<PyDict>>,
}

/// One factory's `word_lang_prob_map`.
#[pyclass(module = "pyroid.langdetect", name = "_Profiles", frozen)]
#[derive(Debug)]
pub struct Profiles {
    inner: Mutex<Inner>,
}

static DEFAULTDICT: PyOnceLock<Py<PyType>> = PyOnceLock::new();

fn defaultdict(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    DEFAULTDICT
        .get_or_try_init(py, || {
            Ok(py
                .import(intern!(py, "collections"))?
                .getattr(intern!(py, "defaultdict"))?
                .cast_into::<PyType>()?
                .unbind())
        })
        .map(|t| t.bind(py))
}

/// `1.0 * v / n` as Python computes it, for an int or float `v` and `n`.
fn probability(v: &Bound<'_, PyAny>, n: &Bound<'_, PyAny>) -> PyResult<f64> {
    if let (Ok(vf), Ok(nf)) = (v.extract::<f64>(), n.extract::<f64>()) {
        if nf != 0.0 {
            return Ok((1.0 * vf) / nf);
        }
    }
    // Zero, or a value too large for a float: Python's own operators raise the error.
    let py = v.py();
    let one = PyFloat::new(py, 1.0);
    one.mul(v)?.div(n)?.extract::<f64>()
}

impl Profiles {
    fn snapshot(&self, py: Python<'_>) -> Arc<Frozen> {
        let mut g = lock(py, &self.inner);
        if let Some(f) = &g.frozen {
            return Arc::clone(f);
        }
        let f = Arc::new(g.builder.freeze());
        g.frozen = Some(Arc::clone(&f));
        f
    }

    fn foreign_row(&self, py: Python<'_>, word: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        let dict = lock(py, &self.inner)
            .foreign_rows
            .as_ref()
            .map(|d| d.clone_ref(py));
        match dict {
            None => {
                // `word in {}` still hashes the key (an unhashable key raises).
                word.hash()?;
                Ok(None)
            }
            Some(d) => d.bind(py).get_item(word)?.map(|r| r.extract()).transpose(),
        }
    }

    fn row_of(&self, py: Python<'_>, word: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
        if let Ok(s) = word.cast::<PyString>() {
            let cps = code_points(s)?;
            Ok(lock(py, &self.inner).builder.find(&cps))
        } else {
            self.foreign_row(py, word)
        }
    }
}

fn set_error(_: OutOfRange) -> PyErr {
    PyIndexError::new_err("list assignment index out of range")
}

#[pymethods]
impl Profiles {
    #[new]
    fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                builder: Builder::default(),
                frozen: None,
                foreign: Vec::new(),
                foreign_rows: None,
            }),
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        lock(py, &self.inner).builder.len()
    }

    /// `add_profile`'s loop for the shapes `load_profile` makes (a dict of str to int or
    /// float, a list `n_words` of ints or floats, int `index` and `langsize`). Returns
    /// `False`, changing nothing, for anything else; raises what langdetect raises, at the
    /// same word, leaving the same rows behind.
    fn _add(
        &self,
        py: Python<'_>,
        freq: &Bound<'_, PyAny>,
        n_words: &Bound<'_, PyAny>,
        index: &Bound<'_, PyAny>,
        langsize: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        let ty = freq.get_type();
        if !(ty.is(py.get_type::<PyDict>()) || ty.is(defaultdict(py)?)) {
            return Ok(false);
        }
        let Ok(n_words) = n_words.cast_exact::<PyList>() else {
            return Ok(false);
        };
        if !index.is_exact_instance_of::<PyInt>() || !langsize.is_exact_instance_of::<PyInt>() {
            return Ok(false);
        }
        let (Ok(index), Ok(langsize)) = (index.extract::<i64>(), langsize.extract::<i64>()) else {
            return Ok(false);
        };
        let row_len = usize::try_from(langsize.max(0)).unwrap_or(usize::MAX);
        let freq = freq.cast::<PyDict>()?;
        let items = freq.items();
        // Read everything first; the rows change only once all types are known to be the
        // fast path's.
        let mut n_cache: [Option<Bound<'_, PyAny>>; 3] = [None, None, None];
        let mut work: Vec<(Vec<u32>, Option<PyResult<f64>>)> = Vec::with_capacity(items.len());
        for item in items.iter() {
            let (k, v) = item.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
            let Ok(k) = k.cast::<PyString>() else {
                return Ok(false);
            };
            if !(v.is_exact_instance_of::<PyInt>() || v.is_exact_instance_of::<PyFloat>()) {
                return Ok(false);
            }
            let cps = code_points(k)?;
            if !(1..=3).contains(&cps.len()) {
                work.push((cps, None));
                continue;
            }
            let slot = cps.len() - 1;
            let n = if let Some(n) = &n_cache[slot] {
                n.clone()
            } else {
                match n_words.get_item(slot) {
                    Ok(n) => {
                        if !(n.is_exact_instance_of::<PyInt>()
                            || n.is_exact_instance_of::<PyFloat>())
                        {
                            return Ok(false);
                        }
                        n_cache[slot] = Some(n.clone());
                        n
                    }
                    Err(e) => {
                        // Raised at this word, after its row is made.
                        work.push((cps, Some(Err(e))));
                        break;
                    }
                }
            };
            let prob = probability(&v, &n);
            let failed = prob.is_err();
            work.push((cps, Some(prob)));
            if failed {
                break;
            }
        }
        let mut g = lock(py, &self.inner);
        g.frozen = None;
        for (cps, prob) in work {
            let row = g.builder.ensure(&cps, row_len);
            match prob {
                None => {}
                Some(Err(e)) => return Err(e),
                Some(Ok(p)) => g.builder.set(row, index, p).map_err(set_error)?,
            }
        }
        Ok(true)
    }

    /// `word in word_lang_prob_map`.
    fn _contains(&self, py: Python<'_>, word: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self.row_of(py, word)?.is_some())
    }

    /// `word_lang_prob_map[word] = [0.0] * length` for a new `word`.
    fn _create(&self, py: Python<'_>, word: &Bound<'_, PyAny>, length: usize) -> PyResult<()> {
        if let Ok(s) = word.cast::<PyString>() {
            let cps = code_points(s)?;
            let mut g = lock(py, &self.inner);
            g.frozen = None;
            g.builder.ensure(&cps, length);
            return Ok(());
        }
        word.hash()?;
        let dict = {
            let mut g = lock(py, &self.inner);
            g.foreign_rows
                .get_or_insert_with(|| PyDict::new(py).unbind())
                .clone_ref(py)
        };
        if dict.bind(py).contains(word)? {
            return Ok(());
        }
        let row = {
            let mut g = lock(py, &self.inner);
            g.frozen = None;
            let id = g.foreign.len();
            g.foreign.push(word.clone().unbind());
            g.builder.push_foreign(length, id)
        };
        dict.bind(py).set_item(word, row)
    }

    /// The length of `word`'s row.
    fn _row_len(&self, py: Python<'_>, word: &Bound<'_, PyAny>) -> PyResult<usize> {
        let row = self
            .row_of(py, word)?
            .ok_or_else(|| PyKeyError::new_err(word.clone().unbind()))?;
        Ok(lock(py, &self.inner).builder.row_len(row))
    }

    /// `word_lang_prob_map[word][index] = value` (`0 <= index < length`).
    fn _set(
        &self,
        py: Python<'_>,
        word: &Bound<'_, PyAny>,
        index: i64,
        value: f64,
    ) -> PyResult<()> {
        let row = self
            .row_of(py, word)?
            .ok_or_else(|| PyKeyError::new_err(word.clone().unbind()))?;
        let mut g = lock(py, &self.inner);
        g.frozen = None;
        g.builder.set(row, index, value).map_err(set_error)
    }

    /// `word_lang_prob_map[word]` as a new list.
    fn _row<'py>(&self, py: Python<'py>, word: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let row = self
            .row_of(py, word)?
            .ok_or_else(|| PyKeyError::new_err(word.clone().unbind()))?;
        let dense = lock(py, &self.inner).builder.dense(row);
        PyList::new(py, dense)
    }

    /// The keys, in insertion order.
    fn _keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let (keys, foreign): (Vec<Key>, Vec<Py<PyAny>>) = {
            let g = lock(py, &self.inner);
            (
                g.builder.keys().to_vec(),
                g.foreign.iter().map(|o| o.clone_ref(py)).collect(),
            )
        };
        let out = PyList::empty(py);
        for k in keys {
            match k {
                Key::Packed(p) => out.append(to_py_str(py, &core::profiles::unpack(p))?)?,
                Key::Long(cps) => out.append(to_py_str(py, &cps)?)?,
                Key::Foreign(id) => out.append(foreign[id].bind(py))?,
            }
        }
        Ok(out)
    }

    /// A store for langdetect's own `word_lang_prob_map` (a dict of str to lists of floats),
    /// for `pyroid.patch("langdetect")`; `None` if a row holds anything but floats.
    #[staticmethod]
    fn _from_map(map: &Bound<'_, PyDict>) -> PyResult<Option<Self>> {
        let mut builder = Builder::default();
        for item in map.items().iter() {
            let (key, row) = item.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
            let Ok(key) = key.cast::<PyString>() else {
                continue; // never looked up
            };
            let cps = code_points(key)?;
            if !(1..=3).contains(&cps.len()) {
                continue;
            }
            let Ok(row) = row.cast_exact::<PyList>() else {
                return Ok(None);
            };
            let row = row.to_tuple();
            let row_id = builder.ensure(&cps, row.len());
            for (index, value) in row.iter().enumerate() {
                let Ok(value) = value.cast_exact::<PyFloat>() else {
                    return Ok(None);
                };
                let value = value.value();
                if value.to_bits() != 0 {
                    // index < row.len() fits in i64.
                    #[allow(clippy::cast_possible_wrap)]
                    builder
                        .set(row_id, index as i64, value)
                        .map_err(set_error)?;
                }
            }
        }
        Ok(Some(Self {
            inner: Mutex::new(Inner {
                builder,
                frozen: None,
                foreign: Vec::new(),
                foreign_rows: None,
            }),
        }))
    }
}

// ---------------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------------

/// A `random.Random` state from `getstate()`, for seeds other than ints.
#[pyclass(module = "pyroid.langdetect", name = "_SeedState", frozen)]
#[derive(Debug)]
pub struct SeedState {
    words: Box<[u32; 624]>,
    index: usize,
    gauss_next: Option<f64>,
}

#[pymethods]
impl SeedState {
    /// From `random.Random(seed).getstate()`.
    #[new]
    fn new(state: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let (_version, internal, gauss_next): (Bound<'_, PyAny>, Vec<u64>, Option<f64>) =
            state.extract()?;
        if internal.len() != 625 {
            return Err(PyValueError::new_err("state vector is the wrong size"));
        }
        let mut words = Box::new([0u32; 624]);
        for (w, &v) in words.iter_mut().zip(&internal) {
            *w = u32::try_from(v).map_err(|_| PyValueError::new_err("bad state word"))?;
        }
        let index = usize::try_from(internal[624]).unwrap_or(usize::MAX);
        if index > 624 {
            return Err(PyValueError::new_err("invalid state"));
        }
        Ok(Self {
            words,
            index,
            gauss_next,
        })
    }
}

/// How a detection seeds its generator.
enum Seed {
    /// `seed=None`: from the operating system.
    Entropy,
    /// An int seed (its absolute value).
    Int(u128),
    /// Any other seed, as `random.Random(seed)` left it.
    State(Py<SeedState>),
}

fn seed_from(obj: &Bound<'_, PyAny>) -> PyResult<Seed> {
    if obj.is_none() {
        return Ok(Seed::Entropy);
    }
    if obj.is_exact_instance_of::<PyInt>() {
        let v: i128 = obj.extract()?;
        return Ok(Seed::Int(v.unsigned_abs()));
    }
    if let Ok(s) = obj.cast::<SeedState>() {
        return Ok(Seed::State(s.clone().unbind()));
    }
    Err(PyTypeError::new_err(
        "seed must be None, an int or a _SeedState",
    ))
}

thread_local! {
    /// The seeded states of the last int seeds this thread used.
    static INT_SEEDS: RefCell<Vec<(u128, Box<SeededState>)>> = const { RefCell::new(Vec::new()) };
}

fn int_rng(n: u128) -> Random {
    INT_SEEDS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some((_, s)) = cache.iter().find(|(k, _)| *k == n) {
            return Random::from_seeded(s);
        }
        let s = Box::new(mt::init_by_array(&mt::int_key(n)));
        let r = Random::from_seeded(&s);
        if cache.len() >= 4 {
            cache.remove(0);
        }
        cache.push((n, s));
        r
    })
}

fn state_rng(s: &SeedState) -> Random {
    Random::from_state(&s.words, s.index, s.gauss_next).unwrap_or_else(|| unreachable!())
}

/// `os.urandom(n)`.
fn urandom(py: Python<'_>, n: usize) -> PyResult<Vec<u8>> {
    let b = py
        .import(intern!(py, "os"))?
        .call_method1(intern!(py, "urandom"), (n,))?;
    Ok(b.cast::<PyBytes>()?.as_bytes().to_vec())
}

/// The generator for one call on the calling thread.
fn rng_for(py: Python<'_>, seed: &Seed) -> PyResult<Random> {
    Ok(match seed {
        Seed::Int(n) => int_rng(*n),
        Seed::State(s) => state_rng(s.get()),
        Seed::Entropy => {
            // As `random.seed(None)`: 624 words from os.urandom.
            let bytes = urandom(py, 624 * 4)?;
            let key: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Random::from_seeded(&mt::init_by_array(&key))
        }
    })
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Outcome codes returned to the Python side.
const OK: u8 = 0;
const NO_FEATURES: u8 = 1;
const INDEX_ERROR: u8 = 2;
const ZERO_DIVISION: u8 = 3;
/// Stopped by a signal handler that raised (the error is raised instead).
const INTERRUPTED: u8 = 5;

/// The parameters of `_detect_block`: `(alpha, n_trial, ALPHA_WIDTH, ITERATION_LIMIT,
/// CONV_THRESHOLD, BASE_FREQ)`.
#[derive(FromPyObject)]
struct DetectorParams(f64, i64, f64, f64, f64, f64);

fn params(
    p: &DetectorParams,
    prior: Option<Vec<f64>>,
    n_langs: usize,
    sum_mode: SumMode,
) -> Params {
    Params {
        alpha: p.0,
        n_trial: p.1,
        alpha_width: p.2,
        iteration_limit: p.3,
        conv_threshold: p.4,
        base_freq: p.5,
        prior,
        n_langs,
        sum_mode,
    }
}

fn failure_code(f: Failure) -> u8 {
    match f {
        Failure::Index => INDEX_ERROR,
        Failure::ZeroDivision => ZERO_DIVISION,
        Failure::Interrupted => INTERRUPTED,
    }
}

/// `Detector._detect_block` on `text` (`Detector.text`): `(cleaned text or None, code,
/// langprob or None)`. Code 0: done; 1: no features (`langprob` stays `None`); 2:
/// `IndexError`; 3: `ZeroDivisionError` (`langprob` as far as the trials got).
/// What [`detect_block`] returns.
type Block<'py> = (Option<Bound<'py, PyString>>, u8, Option<Bound<'py, PyList>>);

#[pyfunction]
#[pyo3(signature = (profiles, text, n_langs, seed, params, prior))]
fn detect_block<'py>(
    py: Python<'py>,
    profiles: &Profiles,
    text: &Bound<'py, PyString>,
    n_langs: usize,
    seed: &Bound<'py, PyAny>,
    params: DetectorParams,
    prior: Option<Vec<f64>>,
) -> PyResult<Block<'py>> {
    let (upper, sum_mode) = tables(py)?;
    let map = profiles.snapshot(py);
    let cps = code_points(text)?;
    let seed = seed_from(seed)?;
    let p = self::params(&params, prior, n_langs, sum_mode);
    let mut rng = rng_for(py, &seed)?;
    let mut pending: Option<PyErr> = None;
    let mut tick = || match py.check_signals() {
        Ok(()) => true,
        Err(e) => {
            pending = Some(e);
            false
        }
    };
    let (cleaned, result) = core::detect_block(&cps, &map, upper, &p, &mut rng, &mut tick);
    if let Some(e) = pending {
        return Err(e);
    }
    let cleaned = cleaned.map(|c| to_py_str(py, &c)).transpose()?;
    Ok(match result {
        Ok(out) => (cleaned, OK, Some(PyList::new(py, out.langprob)?)),
        Err(core::Error::NoFeatures) => (cleaned, NO_FEATURES, None),
        Err(core::Error::Trials(Outcome { langprob, failure })) => (
            cleaned,
            failure.map_or(OK, failure_code),
            Some(PyList::new(py, langprob)?),
        ),
    })
}

/// `Detector._extract_ngrams` on `text`: the n-grams as strs.
#[pyfunction]
fn extract_ngrams<'py>(
    py: Python<'py>,
    profiles: &Profiles,
    text: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyList>> {
    let (upper, _) = tables(py)?;
    let map = profiles.snapshot(py);
    let cps = code_points(text)?;
    let words = detect::extract_ngram_words(&cps, &map, upper);
    let items = words
        .iter()
        .map(|w| to_py_str(py, w))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// One document's result, before it becomes Python objects.
enum DocResult {
    /// `(language index, probability)`, highest first.
    Langs(Vec<(usize, f64)>),
    Failed(u8),
}

/// What every document of a call shares.
struct Ctx<'a> {
    map: &'a Frozen,
    upper: &'a Upper,
    params: &'a Params,
    /// `Detector.PROB_THRESHOLD`.
    threshold: f64,
}

fn run_doc(
    text: &[u32],
    complete: bool,
    ctx: &Ctx<'_>,
    rng: &mut Random,
    tick: &mut dyn FnMut() -> bool,
) -> Option<DocResult> {
    let appended = text::append_prefix(text, complete, core::MAX_TEXT_LENGTH)?;
    let (_, result) = core::detect_block(&appended, ctx.map, ctx.upper, ctx.params, rng, tick);
    Some(match result {
        Ok(out) => DocResult::Langs(detect::sort_probability(&out.langprob, ctx.threshold)),
        Err(core::Error::NoFeatures) => DocResult::Failed(NO_FEATURES),
        Err(core::Error::Trials(o)) => DocResult::Failed(o.failure.map_or(OK, failure_code)),
    })
}

/// [`run_doc`] on a whole document, from its UTF-8 or code points.
fn run_source(
    src: &Source<'_>,
    ctx: &Ctx<'_>,
    rng: &mut Random,
    tick: &mut dyn FnMut() -> bool,
) -> DocResult {
    let mut take = text::first_prefix(core::MAX_TEXT_LENGTH);
    loop {
        let (prefix, complete) = src.prefix(take);
        if let Some(r) = run_doc(&prefix, complete, ctx, rng, tick) {
            return r;
        }
        take = take.saturating_mul(2);
    }
}

/// The module-level `detect(text)` (`probs=False`: the language, or `'unknown'`) or
/// `detect_langs(text)` (`probs=True`: `[(lang, prob), ...]`) for `factory`'s data, as a
/// fresh `Detector` would give it. Returns `(code, result)`; see [`detect_block`].
#[pyfunction]
#[pyo3(signature = (profiles, langlist, text, seed, params, threshold, probs, unknown))]
#[allow(clippy::too_many_arguments)]
fn detect_one<'py>(
    py: Python<'py>,
    profiles: &Profiles,
    langlist: &Bound<'py, PyList>,
    text: &Bound<'py, PyString>,
    seed: &Bound<'py, PyAny>,
    params: DetectorParams,
    threshold: f64,
    probs: bool,
    unknown: &Bound<'py, PyAny>,
) -> PyResult<(u8, Bound<'py, PyAny>)> {
    let (upper, sum_mode) = tables(py)?;
    let map = profiles.snapshot(py);
    let seed = seed_from(seed)?;
    let p = self::params(&params, None, langlist.len(), sum_mode);
    let mut rng = rng_for(py, &seed)?;
    let src = source(text)?;
    let mut pending: Option<PyErr> = None;
    let mut tick = || match py.check_signals() {
        Ok(()) => true,
        Err(e) => {
            pending = Some(e);
            false
        }
    };
    let ctx = Ctx {
        map: &map,
        upper,
        params: &p,
        threshold,
    };
    let r = run_source(&src, &ctx, &mut rng, &mut tick);
    if let Some(e) = pending {
        return Err(e);
    }
    match r {
        DocResult::Failed(code) => Ok((code, py.None().into_bound(py))),
        DocResult::Langs(langs) => Ok((OK, doc_value(py, langlist, &langs, probs, unknown)?)),
    }
}

fn doc_value<'py>(
    py: Python<'py>,
    langlist: &Bound<'py, PyList>,
    langs: &[(usize, f64)],
    probs: bool,
    unknown: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if probs {
        let items = langs
            .iter()
            .map(|&(i, p)| {
                PyTuple::new(py, [langlist.get_item(i)?, PyFloat::new(py, p).into_any()])
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, items)?.into_any())
    } else {
        match langs.first() {
            Some(&(i, _)) => langlist.get_item(i),
            None => Ok(unknown.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

/// How long one chunk of a batch should take: Ctrl-C is checked between chunks.
const CHUNK_TIME: Duration = Duration::from_millis(50);

/// A document of a list or tuple, readable from pool threads.
enum Doc {
    Utf8(PyBackedStr),
    Cps(Vec<u32>),
    /// Not a str: the Python side raises langdetect's error for it.
    NotStr,
}

impl Doc {
    fn source(&self) -> Option<Source<'_>> {
        match self {
            Doc::Utf8(s) => Some(Source::Utf8(s)),
            Doc::Cps(v) => Some(Source::Cps(Cow::Borrowed(v))),
            Doc::NotStr => None,
        }
    }
}

fn doc(item: &Bound<'_, PyAny>) -> PyResult<Doc> {
    let Ok(s) = item.cast::<PyString>() else {
        return Ok(Doc::NotStr);
    };
    Ok(match PyBackedStr::try_from(s.clone()) {
        Ok(b) => Doc::Utf8(b),
        Err(_) => Doc::Cps(surrogate_code_points(s)?),
    })
}

/// Per-document generators for a batch.
enum BatchSeed {
    Int(u128),
    State(Py<SeedState>),
    /// One 32-byte key per document of the current chunk.
    Entropy(Vec<[u8; 32]>),
}

/// What the Python side gets back from [`detect_many`].
type Many<'py> = (Bound<'py, PyAny>, i64, u8);

/// The batch functions: `detect` / `detect_langs` of every document, as separate calls give
/// them. Returns `(results, index, code)`: `index` is -1, or the first document (in order)
/// whose detection raised (code as in [`detect_block`]; a non-str item has code 4) when
/// `errors_none` is false. With `errors_none`, such documents give `None`.
#[pyfunction]
#[pyo3(signature = (profiles, langlist, texts, seed, params, threshold, probs, unknown, language, errors_none, threads=None))]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn detect_many<'py>(
    py: Python<'py>,
    profiles: &Profiles,
    langlist: &Bound<'py, PyList>,
    texts: &Bound<'py, PyAny>,
    seed: &Bound<'py, PyAny>,
    params: DetectorParams,
    threshold: f64,
    probs: bool,
    unknown: &Bound<'py, PyAny>,
    language: &Bound<'py, PyAny>,
    errors_none: bool,
    threads: Option<Threads>,
) -> PyResult<Many<'py>> {
    let (upper, sum_mode) = tables(py)?;
    let map = profiles.snapshot(py);
    let n_langs = langlist.len();
    let p = self::params(&params, None, n_langs, sum_mode);
    let seed = seed_from(seed)?;
    let threads = self::threads(threads);
    let ctx = Ctx {
        map: &map,
        upper,
        params: &p,
        threshold,
    };
    let out = Out {
        langlist,
        probs,
        unknown,
        language,
        errors_none,
    };
    if !(texts.is_instance_of::<PyList>() || texts.is_instance_of::<PyTuple>()) {
        if let Some(strings) = arrow::import(texts)? {
            return many_arrow(py, &strings, &ctx, &seed, threads, &out);
        }
        return Err(PyTypeError::new_err(format!(
            "expected a list or tuple of str, or an Arrow string array, got {}",
            crate::errors::type_name(texts)
        )));
    }
    let is_tuple = texts.is_instance_of::<PyTuple>();
    let items = if let Ok(l) = texts.cast::<PyList>() {
        l.to_tuple()
    } else {
        texts.cast::<PyTuple>()?.clone()
    };
    let n = items.len();
    let mut values: Vec<Bound<'py, PyAny>> = Vec::with_capacity(n);
    let mut start = 0usize;
    let mut chunk = 256usize;
    while start < n {
        let end = (start + chunk).min(n);
        let docs = (start..end)
            .map(|i| doc(&items.get_item(i)?))
            .collect::<PyResult<Vec<Doc>>>()?;
        let bseed = batch_seed(py, &seed, docs.len())?;
        let t0 = Instant::now();
        let results = py
            .detach(|| {
                par_docs(docs.len(), threads, |i| {
                    let Some(src) = docs[i].source() else {
                        return Ok::<_, PoolError>(None); // not a str
                    };
                    let mut rng = batch_rng(&bseed, i);
                    Ok(Some(run_source(&src, &ctx, &mut rng, &mut || true)))
                })
            })
            .map_err(|e| BatchError::from(e).into_pyerr(py))?;
        for (i, r) in results.into_iter().enumerate() {
            let code = match r {
                None => 4,
                Some(DocResult::Failed(c)) => c,
                Some(DocResult::Langs(langs)) => {
                    values.push(out.object(py, &langs)?);
                    continue;
                }
            };
            if errors_none && code == NO_FEATURES {
                values.push(py.None().into_bound(py));
            } else {
                let index = i64::try_from(start + i).unwrap_or(i64::MAX);
                return Ok((py.None().into_bound(py), index, code));
            }
        }
        py.check_signals()?;
        chunk = next_chunk(chunk, t0.elapsed(), docs.len());
        start = end;
    }
    let list = PyList::new(py, values)?;
    let res = if is_tuple {
        list.to_tuple().into_any()
    } else {
        list.into_any()
    };
    Ok((res, -1, OK))
}

/// `f(0..len)` in order on the shared pool. A detection takes 30 µs to a few ms, so tasks are
/// a share of the documents per thread rather than the pool's 64-document minimum (which would
/// leave most threads idle on a few hundred long texts). At most `max_tasks(threads)` tasks.
fn par_docs<R, E, F>(len: usize, threads: par::Threads, f: F) -> Result<Vec<R>, E>
where
    R: Send,
    E: Send + From<PoolError>,
    F: Fn(usize) -> Result<R, E> + Sync + Send,
{
    let max = par::max_tasks(threads)?;
    if max <= 1 || len <= 1 {
        return (0..len).map(f).collect();
    }
    let per = len.div_ceil(max).max(1);
    let ranges: Vec<std::ops::Range<usize>> = (0..len)
        .step_by(per)
        .map(|a| a..(a + per).min(len))
        .collect();
    let parts = par::map_tasks(&ranges, threads, |r| {
        r.clone().map(&f).collect::<Result<Vec<R>, E>>()
    })?;
    let mut out = Vec::with_capacity(len);
    for part in parts {
        out.extend(part?);
    }
    Ok(out)
}

fn next_chunk(chunk: usize, took: Duration, done: usize) -> usize {
    #[allow(clippy::cast_precision_loss)]
    let per_doc = took.as_secs_f64() / done.max(1) as f64;
    if per_doc <= 0.0 {
        return (chunk * 2).min(65_536);
    }
    // Docs per CHUNK_TIME, within reason.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let want = (CHUNK_TIME.as_secs_f64() / per_doc) as usize;
    want.clamp(64, 65_536).min(chunk.saturating_mul(4))
}

fn batch_seed(py: Python<'_>, seed: &Seed, docs: usize) -> PyResult<BatchSeed> {
    Ok(match seed {
        Seed::Int(n) => BatchSeed::Int(*n),
        Seed::State(s) => BatchSeed::State(s.clone_ref(py)),
        Seed::Entropy => {
            let bytes = urandom(py, 32)?;
            let mut k = [0u8; 32];
            k.copy_from_slice(&bytes);
            let mut stream = EntropyKeys::new(k);
            let keys = (0..docs)
                .map(|_| {
                    let st = stream.next_state();
                    let mut key = [0u8; 32];
                    for (i, w) in st.iter().take(8).enumerate() {
                        key[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                    }
                    key
                })
                .collect();
            BatchSeed::Entropy(keys)
        }
    })
}

fn batch_rng(seed: &BatchSeed, i: usize) -> Random {
    match seed {
        BatchSeed::Int(n) => int_rng(*n),
        BatchSeed::State(s) => state_rng(s.get()),
        BatchSeed::Entropy(keys) => Random::from_seeded(&EntropyKeys::new(keys[i]).next_state()),
    }
}

/// How a batch turns a document's result into Python objects.
struct Out<'a, 'py> {
    langlist: &'a Bound<'py, PyList>,
    /// `detect_langs` (a list of `Language`) rather than `detect` (the language).
    probs: bool,
    /// `Detector.UNKNOWN_LANG`.
    unknown: &'a Bound<'py, PyAny>,
    /// The `Language` class.
    language: &'a Bound<'py, PyAny>,
    errors_none: bool,
}

impl<'py> Out<'_, 'py> {
    fn object(&self, py: Python<'py>, langs: &[(usize, f64)]) -> PyResult<Bound<'py, PyAny>> {
        if self.probs {
            let items = langs
                .iter()
                .map(|&(i, p)| self.language.call1((self.langlist.get_item(i)?, p)))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyList::new(py, items)?.into_any())
        } else {
            match langs.first() {
                Some(&(i, _)) => self.langlist.get_item(i),
                None => Ok(self.unknown.clone()),
            }
        }
    }
}

fn many_arrow<'py>(
    py: Python<'py>,
    strings: &ArrowStrings,
    ctx: &Ctx<'_>,
    seed: &Seed,
    threads: par::Threads,
    out: &Out<'_, 'py>,
) -> PyResult<Many<'py>> {
    // Language names as str, for a string column of detect()'s results.
    let names: Option<Vec<String>> = if out.probs {
        None
    } else {
        out.langlist
            .iter()
            .map(|l| l.extract::<String>().ok())
            .collect()
    };
    let unknown_name: Option<String> = out.unknown.extract().ok();
    let mut objects: Vec<Bound<'py, PyAny>> = Vec::new();
    let mut labels: Vec<Option<String>> = Vec::new();
    let mut base = 0usize;
    for (c, chunk) in strings.chunks.iter().enumerate() {
        let len = chunk.len();
        let mut start = 0usize;
        let mut size = 256usize;
        while start < len {
            let end = (start + size).min(len);
            let bseed = batch_seed(py, seed, end - start)?;
            let t0 = Instant::now();
            let results = py
                .detach(|| {
                    par_docs(end - start, threads, |i| match chunk.get(start + i) {
                        None => Ok(None),
                        Some(Err(err)) => Err(BatchError::BadUtf8 {
                            chunk: c,
                            row: start + i,
                            bytes: chunk.bytes(start + i).unwrap_or_default().to_vec(),
                            err,
                        }),
                        Some(Ok(s)) => {
                            let mut rng = batch_rng(&bseed, i);
                            Ok(Some(run_source(
                                &Source::Utf8(s),
                                ctx,
                                &mut rng,
                                &mut || true,
                            )))
                        }
                    })
                })
                .map_err(|e| e.into_pyerr(py))?;
            for (i, r) in results.into_iter().enumerate() {
                let value = match r {
                    None => None,
                    Some(DocResult::Langs(langs)) => Some(langs),
                    Some(DocResult::Failed(code)) => {
                        if out.errors_none && code == NO_FEATURES {
                            None
                        } else {
                            let index = i64::try_from(base + start + i).unwrap_or(i64::MAX);
                            return Ok((py.None().into_bound(py), index, code));
                        }
                    }
                };
                match (&names, value) {
                    (Some(names), v) => labels.push(v.map(|langs| match langs.first() {
                        Some(&(i, _)) => names[i].clone(),
                        None => unknown_name.clone().unwrap_or_default(),
                    })),
                    (None, None) => objects.push(py.None().into_bound(py)),
                    (None, Some(langs)) => objects.push(out.object(py, &langs)?),
                }
            }
            py.check_signals()?;
            size = next_chunk(size, t0.elapsed(), end - start);
            start = end;
        }
        base += len;
    }
    if names.is_some() {
        let large = strings.field.ty == StrType::LargeUtf8;
        let chunk = if large {
            Chunk::LargeUtf8(StringBuffers::build(&labels, par::Threads::ONE).map_err(build_error)?)
        } else {
            Chunk::Utf8(StringBuffers::build(&labels, par::Threads::ONE).map_err(build_error)?)
        };
        let ty = if large {
            ColumnType::LargeUtf8
        } else {
            ColumnType::Utf8
        };
        let col = Column::new(ty, &strings.field.name, vec![chunk]);
        return Ok((Bound::new(py, col)?.into_any(), -1, OK));
    }
    Ok((PyList::new(py, objects)?.into_any(), -1, OK))
}

// ---------------------------------------------------------------------------
// Testing hooks
// ---------------------------------------------------------------------------

/// Draws from a generator seeded like `random.Random(seed)`: `ops` is a list of
/// `("random", n)`, `("gauss", n)` or `("below", k)` (one `_randbelow(k)`); returns the values.
#[pyfunction]
fn _draws<'py>(
    py: Python<'py>,
    seed: &Bound<'py, PyAny>,
    ops: Vec<(String, u64)>,
) -> PyResult<Bound<'py, PyList>> {
    let mut rng = rng_for(py, &seed_from(seed)?)?;
    let out = PyList::empty(py);
    for (op, n) in ops {
        match op.as_str() {
            "random" => {
                for _ in 0..n {
                    out.append(rng.random())?;
                }
            }
            "gauss" => {
                for _ in 0..n {
                    out.append(rng.gauss(0.0, 1.0))?;
                }
            }
            "below" if n > 0 => out.append(rng.randbelow(n))?,
            _ => return Err(PyValueError::new_err(format!("unknown op {op:?}"))),
        }
    }
    Ok(out)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.langdetect")?;
    m.gil_used(false)?;
    m.add_class::<Profiles>()?;
    m.add_class::<SeedState>()?;
    m.add_function(wrap_pyfunction!(append_text, &m)?)?;
    m.add_function(wrap_pyfunction!(substitute, &m)?)?;
    m.add_function(wrap_pyfunction!(collapse, &m)?)?;
    m.add_function(wrap_pyfunction!(clean_text, &m)?)?;
    m.add_function(wrap_pyfunction!(normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(normalize_vi, &m)?)?;
    m.add_function(wrap_pyfunction!(unicode_versions, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_block, &m)?)?;
    m.add_function(wrap_pyfunction!(extract_ngrams, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_one, &m)?)?;
    m.add_function(wrap_pyfunction!(detect_many, &m)?)?;
    m.add_function(wrap_pyfunction!(_draws, &m)?)?;
    m.add("MAX_TEXT_LENGTH", core::MAX_TEXT_LENGTH)?;
    parent.add_submodule(&m)
}
