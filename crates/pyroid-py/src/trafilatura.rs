// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! `pyroid._pyroid.trafilatura`: native entry points behind `pyroid.trafilatura`.
//!
//! The Python module (`python/pyroid/trafilatura.py`) owns Trafilatura's signatures, argument
//! checks, settings and delegation; this module runs the port and converts results. A str is
//! handed to the core as its UTF-8, or as its `"surrogatepass"` UTF-8 when it holds lone
//! surrogates (what the core expects); bytes must be valid UTF-8 (the shim delegates the rest).
//!
//! The dedup cache (`LRUCache`) sits behind a `Mutex` that is only ever locked while detached
//! from the interpreter, so a thread waiting for it never holds the GIL. An extraction holds it
//! for the whole document (a batch: the whole batch), which is what keeps a cached run equal to
//! Trafilatura's sequential one.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyList, PyString, PyTuple, PyType};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::html::tree::Document;
use pyroid_core::par::{self, PoolError};
use pyroid_core::trafilatura::{self as core, Focus, LruCache, Options, PyError, Version};

use crate::batch::BatchError;
use crate::errors::{input_error, type_name, with_note};
use crate::input::Threads;

/// The Unicode versions the port has tables for (`unicodedata.unidata_version` spelling).
#[pyfunction]
fn supported_unicode_versions() -> Vec<&'static str> {
    [
        UnicodeVersion::V14_0,
        UnicodeVersion::V15_0,
        UnicodeVersion::V15_1,
        UnicodeVersion::V16_0,
        UnicodeVersion::V17_0,
    ]
    .iter()
    .map(|v| v.as_str())
    .collect()
}

/// Whether `data` is valid UTF-8 (what trafilatura's `bytes.decode("utf-8")` accepts).
#[pyfunction]
fn is_utf8(data: &Bound<'_, PyBytes>) -> bool {
    simdutf8::basic::from_utf8(data.as_bytes()).is_ok()
}

/// Check a `threads=` argument (the batch functions that may not reach the native batch).
#[pyfunction]
#[pyo3(signature = (*, threads = None))]
fn check_threads(threads: Option<Threads>) -> bool {
    threads.is_some()
}

/// The Trafilatura releases the port reproduces.
#[pyfunction]
fn supported_versions() -> Vec<&'static str> {
    vec!["2.3.1"]
}

fn version_of(v: &str) -> Option<Version> {
    match v {
        "2.3.1" => Some(Version::V2_3_1),
        _ => None,
    }
}

// ------------------------------------------------------------------------------------ options

/// The extraction options of one call, built (and cached) by the shim from Trafilatura's
/// arguments and settings.
#[pyclass(frozen, module = "pyroid.trafilatura", name = "_Options")]
#[derive(Debug)]
pub struct Opts(Options);

#[pymethods]
impl Opts {
    #[new]
    #[pyo3(signature = (
        *, version, unicode, focus, comments, fast, tables, min_extracted_size, min_output_size,
        min_output_comm_size, min_duplcheck_size, max_repetitions, max_tree_size
    ))]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn new(
        version: &str,
        unicode: &str,
        focus: &str,
        comments: bool,
        fast: bool,
        tables: bool,
        min_extracted_size: usize,
        min_output_size: usize,
        min_output_comm_size: usize,
        min_duplcheck_size: usize,
        max_repetitions: i64,
        max_tree_size: Option<usize>,
    ) -> PyResult<Self> {
        let version = version_of(version)
            .ok_or_else(|| PyValueError::new_err(format!("no trafilatura mode {version:?}")))?;
        let unicode = UnicodeVersion::from_unidata_version(unicode)
            .ok_or_else(|| PyValueError::new_err(format!("no tables for Unicode {unicode}")))?;
        let focus = match focus {
            "balanced" => Focus::Balanced,
            "precision" => Focus::Precision,
            "recall" => Focus::Recall,
            _ => return Err(PyValueError::new_err(format!("unknown focus {focus:?}"))),
        };
        Ok(Self(Options {
            version,
            focus,
            comments,
            fast,
            tables,
            unicode,
            min_extracted_size,
            min_output_size,
            min_output_comm_size,
            min_duplcheck_size,
            max_repetitions,
            max_tree_size,
        }))
    }

    fn __repr__(&self) -> String {
        format!("<pyroid.trafilatura options {:?}>", self.0)
    }
}

// -------------------------------------------------------------------------------------- cache

/// Least Recently Used (LRU) cache of string counts: `trafilatura.deduplication.LRUCache`.
#[pyclass(frozen, module = "pyroid.trafilatura", name = "LRUCache")]
#[derive(Debug)]
pub struct Lru {
    inner: Mutex<LruCache>,
}

fn guard(m: &Mutex<LruCache>) -> MutexGuard<'_, LruCache> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Lru {
    /// Run `f` on the cache, locked while detached.
    fn with<R: Send>(&self, py: Python<'_>, f: impl FnOnce(&mut LruCache) -> R + Send) -> R {
        py.detach(|| f(&mut guard(&self.inner)))
    }
}

/// A cache's entries, oldest first, as `(key, count)` pairs.
type Items = Vec<(String, i64)>;

fn items_of(c: &LruCache) -> Items {
    c.items().map(|(k, v)| (k.to_owned(), v)).collect()
}

fn set_items(c: &mut LruCache, items: &Items) {
    c.clear();
    for (k, v) in items {
        c.put(k, *v);
    }
}

/// A key as the core holds it (lone surrogates cannot be keys).
fn key_of(key: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = key.py();
    let s = key
        .cast::<PyString>()
        .map_err(|_| input_error(py, format!("LRUCache keys are str, got {}", type_name(key))))?;
    Ok(s.to_str()?.to_owned())
}

#[pymethods]
impl Lru {
    #[new]
    #[pyo3(signature = (maxsize = None), text_signature = "(maxsize=128)")]
    fn new(maxsize: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let n = match maxsize {
            None => 128,
            Some(m) => {
                if !m.is_instance_of::<PyInt>() {
                    return Err(input_error(
                        m.py(),
                        format!("maxsize must be an int, got {}", type_name(m)),
                    ));
                }
                if m.lt(1)? {
                    return Err(PyValueError::new_err("maxsize must be at least 1"));
                }
                // Past usize::MAX no cache can fill up: saturate.
                usize::try_from(m.extract::<u64>().unwrap_or(u64::MAX)).unwrap_or(usize::MAX)
            }
        };
        let n = NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN);
        Ok(Self {
            inner: Mutex::new(LruCache::new(n)),
        })
    }

    /// The most entries the cache keeps.
    #[getter]
    fn maxsize(&self, py: Python<'_>) -> usize {
        self.with(py, |c| c.maxsize())
    }

    /// Retrieve a value from the cache, or -1 if the key is absent.
    fn get(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<i64> {
        let k = key_of(key)?;
        Ok(self.with(py, |c| c.get(&k)))
    }

    /// Store a given key in the cache, evicting the oldest entry if full.
    fn put(&self, py: Python<'_>, key: &Bound<'_, PyAny>, value: i64) -> PyResult<()> {
        let k = key_of(key)?;
        self.with(py, |c| c.put(&k, value));
        Ok(())
    }

    /// Increment the stored count, return the previous count or -1.
    fn increment(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<i64> {
        let k = key_of(key)?;
        Ok(self.with(py, |c| c.increment(&k)))
    }

    /// Delete all cache content.
    fn clear(&self, py: Python<'_>) {
        self.with(py, LruCache::clear);
    }

    /// Always true, even when empty (as in Trafilatura).
    #[allow(clippy::unused_self)]
    fn __bool__(&self) -> bool {
        true
    }

    /// The entries, oldest first, as `(key, count)` pairs (a copy).
    fn _items(&self, py: Python<'_>) -> Items {
        self.with(py, |c| items_of(c))
    }

    /// Replace the entries with `items` (oldest first).
    fn _set_items(&self, py: Python<'_>, items: Items) {
        self.with(py, move |c| set_items(c, &items));
    }

    /// A read-only snapshot of the entries, oldest first (`types.MappingProxyType` over an
    /// `OrderedDict`): Trafilatura's `.cache`, but a copy.
    #[getter]
    fn cache<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items = self.with(py, |c| items_of(c));
        let od = py
            .import("collections")?
            .getattr("OrderedDict")?
            .call1((items,))?;
        py.import("types")?
            .getattr("MappingProxyType")?
            .call1((od,))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, (usize,), Items) {
        let (maxsize, items) = slf.get().with(slf.py(), |c| (c.maxsize(), items_of(c)));
        (slf.get_type(), (maxsize,), items)
    }

    fn __setstate__(&self, py: Python<'_>, items: Items) {
        self.with(py, move |c| set_items(c, &items));
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let (maxsize, len) = self.with(py, |c| (c.maxsize(), c.len()));
        format!("<pyroid.trafilatura.LRUCache maxsize={maxsize}, {len} entries>")
    }
}

// ------------------------------------------------------------------------------------- inputs

/// One document as the core reads it.
enum Input<'py> {
    /// A str with its UTF-8.
    Str(Bound<'py, PyString>),
    /// A str holding lone surrogates: its `"surrogatepass"` UTF-8; or valid UTF-8 bytes.
    Bytes(Bound<'py, PyBytes>),
}

impl Input<'_> {
    fn as_bytes(&self) -> &[u8] {
        match self {
            // to_str succeeded when the input was built
            Input::Str(s) => s.to_str().map_or(&[][..], str::as_bytes),
            Input::Bytes(b) => b.as_bytes(),
        }
    }
}

/// The input for a str.
fn str_input<'py>(s: &Bound<'py, PyString>) -> PyResult<Input<'py>> {
    let py = s.py();
    match s.to_str() {
        Ok(_) => Ok(Input::Str(s.clone())),
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            let b = py
                .get_type::<PyString>()
                .call_method1("encode", (s, "utf-8", "surrogatepass"))?
                .cast_into::<PyBytes>()?;
            Ok(Input::Bytes(b))
        }
        Err(e) => Err(e),
    }
}

/// The input for a str or bytes argument; `None` for bytes that are not UTF-8 (delegated).
fn input<'py>(html: &Bound<'py, PyAny>) -> PyResult<Option<Input<'py>>> {
    if let Ok(s) = html.cast::<PyString>() {
        return str_input(s).map(Some);
    }
    if let Ok(b) = html.cast::<PyBytes>() {
        if simdutf8::basic::from_utf8(b.as_bytes()).is_err() {
            return Ok(None);
        }
        return Ok(Some(Input::Bytes(b.clone())));
    }
    Err(input_error(
        html.py(),
        format!("expected str or bytes, got {}", type_name(html)),
    ))
}

/// The exception Trafilatura raises: CPython's ValueError for `int()` past
/// `sys.get_int_max_str_digits()` (4300; the shim delegates other limits), in `html.unescape`
/// of a numeric character reference.
fn py_error(e: PyError) -> PyErr {
    match e {
        PyError::IntMaxStrDigits { digits } => PyValueError::new_err(format!(
            "Exceeds the limit (4300 digits) for integer string conversion: value has {digits} \
             digits; use sys.set_int_max_str_digits() to increase the limit"
        )),
    }
}

/// Run `f` detached, with the cache (if any) locked.
fn detached<R: Send>(
    py: Python<'_>,
    cache: Option<&Lru>,
    f: impl FnOnce(Option<&mut LruCache>) -> R + Send,
) -> R {
    py.detach(|| match cache {
        None => f(None),
        Some(c) => f(Some(&mut guard(&c.inner))),
    })
}

// ------------------------------------------------------------------------------ single calls

/// `trafilatura.extract(html, ...)` with output "txt": the text, `None`, or `NotImplemented`
/// when `html` is bytes that are not UTF-8 (the shim delegates those).
#[pyfunction]
#[pyo3(signature = (html, opts, cache = None))]
fn extract<'py>(
    html: &Bound<'py, PyAny>,
    opts: &Opts,
    cache: Option<&Bound<'py, Lru>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = html.py();
    let Some(inp) = input(html)? else {
        return Ok(py.NotImplemented().into_bound(py));
    };
    let bytes = inp.as_bytes();
    let o = &opts.0;
    let r = detached(py, cache.map(Bound::get), |c| core::extract(bytes, o, c));
    match r.map_err(py_error)? {
        None => Ok(py.None().into_bound(py)),
        Some(s) => Ok(PyString::new(py, &s).into_any()),
    }
}

/// `trafilatura.bare_extraction(html, ...)` with output "python": `(text, comments, body,
/// commentsbody)` (the XML of the two trees), `None`, or `NotImplemented` as for [`extract`].
#[pyfunction]
#[pyo3(signature = (html, opts, cache = None))]
fn bare_extraction<'py>(
    html: &Bound<'py, PyAny>,
    opts: &Opts,
    cache: Option<&Bound<'py, Lru>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = html.py();
    let Some(inp) = input(html)? else {
        return Ok(py.NotImplemented().into_bound(py));
    };
    let bytes = inp.as_bytes();
    let o = &opts.0;
    let r = detached(py, cache.map(Bound::get), |c| {
        core::bare_extraction(bytes, o, c)
    });
    match r.map_err(py_error)? {
        None => Ok(py.None().into_bound(py)),
        Some(x) => {
            let comments = x.comments.map(|c| PyString::new(py, &c));
            Ok(PyTuple::new(
                py,
                [
                    PyString::new(py, &x.text).into_any(),
                    comments.map_or_else(|| py.None().into_bound(py), Bound::into_any),
                    PyString::new(py, &x.body).into_any(),
                    PyString::new(py, &x.commentsbody).into_any(),
                ],
            )?
            .into_any())
        }
    }
}

// ------------------------------------------------------------------------------------ batches

type Out = Result<Option<String>, PyError>;

/// `f(0), ..., f(n - 1)` in order on at most `max_tasks(t)` workers that each take the next
/// index (pages vary a lot in cost, so a static split would leave workers idle).
fn dynamic_map<R, F>(n: usize, threads: par::Threads, f: F) -> Result<Vec<R>, PoolError>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    if n <= 1 || threads == par::Threads::ONE {
        return Ok((0..n).map(f).collect());
    }
    let tasks = par::max_tasks(threads)?.min(n);
    if tasks <= 1 {
        return Ok((0..n).map(f).collect());
    }
    let next = AtomicUsize::new(0);
    let workers: Vec<usize> = (0..tasks).collect();
    let parts = par::map_tasks(&workers, threads, |_| {
        let mut out = Vec::new();
        loop {
            let idx = next.fetch_add(1, Ordering::Relaxed);
            if idx >= n {
                break;
            }
            out.push((idx, f(idx)));
        }
        out
    })?;
    let mut slots: Vec<Option<R>> = std::iter::repeat_with(|| None).take(n).collect();
    for (i, r) in parts.into_iter().flatten() {
        slots[i] = Some(r);
    }
    Ok(slots
        .into_iter()
        .map(|r| r.expect("every index ran once"))
        .collect())
}

/// Documents per round of a cached batch, per worker.
const ROUND_PER_WORKER: usize = 8;

/// A batch through one cache, in document order: equal to calling [`core::extract`] in a loop.
///
/// The pages are parsed ahead in parallel: in each round one worker extracts the pages parsed
/// in the round before, in order, while the others parse the next round's (the extracting
/// worker joins them when it is done). With `stop_on_error`, the batch ends at the first error
/// (the last output).
fn cached_batch(
    docs: &[&[u8]],
    opts: &Options,
    cache: &mut LruCache,
    threads: par::Threads,
    stop_on_error: bool,
) -> Result<Vec<Out>, PoolError> {
    let n = docs.len();
    let tasks = if n <= 1 || threads == par::Threads::ONE {
        1
    } else {
        par::max_tasks(threads)?.min(n)
    };
    let mut out = Vec::with_capacity(n);
    if tasks <= 1 {
        for d in docs {
            let r = core::extract(d, opts, Some(&mut *cache));
            let stop = stop_on_error && r.is_err();
            out.push(r);
            if stop {
                break;
            }
        }
        return Ok(out);
    }
    let round = (ROUND_PER_WORKER * tasks).min(n);
    let mut cur: Vec<Option<Document>> = dynamic_map(round, threads, |i| core::load_html(docs[i]))?;
    let mut start = 0;
    let cache = Mutex::new(cache);
    let workers: Vec<usize> = (0..tasks).collect();
    loop {
        let end = start + cur.len();
        let next_len = round.min(n - end);
        let todo = Mutex::new(std::mem::take(&mut cur));
        let next: Vec<Mutex<Option<Document>>> = (0..next_len).map(|_| Mutex::new(None)).collect();
        let claimed = AtomicUsize::new(0);
        let stopped = AtomicBool::new(false);
        let mut parts = par::map_tasks(&workers, threads, |&w| {
            let mut res = Vec::new();
            if w == 0 {
                let pages =
                    std::mem::take(&mut *todo.lock().unwrap_or_else(PoisonError::into_inner));
                let mut lru = cache.lock().unwrap_or_else(PoisonError::into_inner);
                for page in pages {
                    let r = page.map_or(Ok(None), |doc| {
                        core::extract_tree(doc, opts, Some(&mut **lru))
                    });
                    let stop = stop_on_error && r.is_err();
                    res.push(r);
                    if stop {
                        stopped.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
            loop {
                if stopped.load(Ordering::Relaxed) {
                    break;
                }
                let idx = claimed.fetch_add(1, Ordering::Relaxed);
                if idx >= next_len {
                    break;
                }
                let doc = core::load_html(docs[end + idx]);
                *next[idx].lock().unwrap_or_else(PoisonError::into_inner) = doc;
            }
            res
        })?;
        out.append(&mut parts[0]);
        if stopped.load(Ordering::Relaxed) || next_len == 0 {
            return Ok(out);
        }
        cur = next
            .into_iter()
            .map(|m| m.into_inner().unwrap_or_else(PoisonError::into_inner))
            .collect();
        start = end;
    }
}

/// `extract(doc)` for every str of a list or tuple, on the shared pool, in order.
///
/// With a `cache`, the documents go through it in order (the result equals a loop). An error
/// (Trafilatura's ValueError) is raised with a note naming the document (`raise_errors`, which
/// also stops a cached batch there, as a loop would), or returned in its place as an exception
/// object.
#[pyfunction]
#[pyo3(signature = (htmls, opts, cache = None, raise_errors = true, *, threads = None))]
fn extract_batch<'py>(
    htmls: &Bound<'py, PyAny>,
    opts: &Opts,
    cache: Option<&Bound<'py, Lru>>,
    raise_errors: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyList>> {
    let py = htmls.py();
    let items: Vec<Bound<'py, PyAny>> = if let Ok(l) = htmls.cast::<PyList>() {
        l.iter().collect()
    } else if let Ok(t) = htmls.cast::<PyTuple>() {
        t.iter().collect()
    } else {
        return Err(input_error(
            py,
            format!("expected a list or tuple of str, got {}", type_name(htmls)),
        ));
    };
    let inputs = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let s = item.cast::<PyString>().map_err(|_| {
                input_error(
                    py,
                    format!("expected str at index {i}, got {}", type_name(item)),
                )
            })?;
            str_input(s)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let docs: Vec<&[u8]> = inputs.iter().map(Input::as_bytes).collect();
    let o = &opts.0;
    let t = crate::input::threads(threads);
    let results = detached(py, cache.map(Bound::get), |c| match c {
        None => dynamic_map(docs.len(), t, |i| core::extract(docs[i], o, None)),
        Some(c) => cached_batch(&docs, o, c, t, raise_errors),
    })
    .map_err(|e| BatchError::from(e).into_pyerr(py))?;
    let mut list = Vec::with_capacity(results.len());
    for (i, r) in results.into_iter().enumerate() {
        list.push(match r {
            Ok(None) => py.None().into_bound(py),
            Ok(Some(s)) => PyString::new(py, &s).into_any(),
            Err(e) if raise_errors => {
                return Err(with_note(
                    py,
                    py_error(e),
                    format!("pyroid: while extracting the document at index {i}"),
                ));
            }
            Err(e) => py_error(e).into_value(py).into_bound(py).into_any(),
        });
    }
    PyList::new(py, list)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "pyroid.trafilatura")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(supported_unicode_versions, &m)?)?;
    m.add_function(wrap_pyfunction!(supported_versions, &m)?)?;
    m.add_function(wrap_pyfunction!(is_utf8, &m)?)?;
    m.add_function(wrap_pyfunction!(check_threads, &m)?)?;
    m.add_function(wrap_pyfunction!(extract, &m)?)?;
    m.add_function(wrap_pyfunction!(bare_extraction, &m)?)?;
    m.add_function(wrap_pyfunction!(extract_batch, &m)?)?;
    m.add_class::<Opts>()?;
    m.add_class::<Lru>()?;
    m.add("LRU_SIZE", core::LRU_SIZE)?;
    parent.add_submodule(&m)
}
