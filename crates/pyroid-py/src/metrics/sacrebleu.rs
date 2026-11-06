// Derived from sacrebleu 2.6.0 (Apache-2.0), © Amazon.com, Inc. or its affiliates; modified.
//! Bindings for `pyroid_core::metrics::sacrebleu`: the native half of
//! `pyroid.metrics.sacrebleu`.
//!
//! The Python module (`python/pyroid/metrics/sacrebleu.py`) owns sacrebleu's
//! classes, argument checks, logging and the score formulas; these functions
//! only compute the integer statistics:
//!
//! - `sb_tokenize(name, line)`: one native tokenizer's `__call__`.
//! - `bleu_corpus_stats` / `chrf_corpus_stats`: statistics of hypotheses
//!   against reference streams (`refs[k][i]` is the k-th reference of
//!   segment i, `None` for a missing one).
//! - `_BleuRefCache` / `_ChrfRefCache`: pre-computed references
//!   (`BLEU(references=...)`), scored with `bleu_cache_stats` /
//!   `chrf_cache_stats`.
//!
//! Every str is extracted on the calling thread (zero-copy `PyBackedStr`),
//! then the work runs detached on the shared pool.

use pyo3::exceptions::{PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyList, PyString, PyTuple};
use pyroid_core::metrics::sacrebleu::{
    BleuConfig, BleuRefCache, BleuStats, ChrfConfig, ChrfRefCache, ChrfStats, Tokenizer,
};
use pyroid_core::par::PoolError;

use crate::batch::SINGLE_DOC_DETACH_BYTES;
use crate::errors::{PyroidError, input_error, type_name, with_note};
use crate::input::{Threads, threads};

fn pool_error(e: PoolError) -> PyErr {
    PyroidError::new_err(e.to_string())
}

fn tokenizer(name: &str) -> PyResult<Tokenizer> {
    Tokenizer::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("not a native tokenizer: {name:?}")))
}

/// The items of a list or tuple (what the Python side always passes).
fn items<'py>(obj: &Bound<'py, PyAny>, what: &str) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = obj.cast::<PyList>() {
        return Ok(list.iter().collect());
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        return Ok(tuple.iter().collect());
    }
    Err(input_error(
        obj.py(),
        format!("{what} must be a list or tuple, got {}", type_name(obj)),
    ))
}

/// `PyBackedStr` of a str item, or `None` for `None` when `none_ok`.
fn backed(
    item: &Bound<'_, PyAny>,
    none_ok: bool,
    what: impl Fn() -> String,
) -> PyResult<Option<PyBackedStr>> {
    let py = item.py();
    if none_ok && item.is_none() {
        return Ok(None);
    }
    let Ok(s) = item.cast::<PyString>() else {
        let want = if none_ok { "a str or None" } else { "a str" };
        return Err(input_error(
            py,
            format!("{} must be {want}, got {}", what(), type_name(item)),
        ));
    };
    PyBackedStr::try_from(s.clone()).map(Some).map_err(|err| {
        if !err.is_instance_of::<PyUnicodeEncodeError>(py) {
            return err;
        }
        with_note(
            py,
            err,
            format!(
                "pyroid: {} contains a lone surrogate (U+D800..U+DFFF), which has no UTF-8 \
                 encoding. Remove or replace it first, e.g. s.encode('utf-8', 'replace').decode('utf-8').",
                what()
            ),
        )
    })
}

/// The first `n` hypotheses.
fn hypotheses(hyps: &[Bound<'_, PyAny>], n: usize) -> PyResult<Vec<PyBackedStr>> {
    hyps[..n]
        .iter()
        .enumerate()
        .map(|(i, h)| backed(h, false, || format!("hypotheses[{i}]")).map(|s| s.expect("not None")))
        .collect()
}

/// Reference streams, each cut to the shortest one's length (`zip(*refs)`).
fn streams(refs: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<Option<PyBackedStr>>>> {
    let streams = items(refs, "references")?
        .iter()
        .enumerate()
        .map(|(k, s)| items(s, &format!("references[{k}]")))
        .collect::<PyResult<Vec<_>>>()?;
    let n = streams.iter().map(Vec::len).min().unwrap_or(0);
    streams
        .iter()
        .enumerate()
        .map(|(k, s)| {
            s[..n]
                .iter()
                .enumerate()
                .map(|(i, r)| backed(r, true, || format!("references[{k}][{i}]")))
                .collect()
        })
        .collect()
}

fn total_bytes(hyps: &[PyBackedStr], streams: &[Vec<Option<PyBackedStr>>]) -> usize {
    let refs: usize = streams.iter().flatten().flatten().map(|s| s.len()).sum();
    refs + hyps.iter().map(|s| s.len()).sum::<usize>()
}

/// Run `f` detached unless the work is one small segment (then the
/// round trip would cost more than it frees; it never touches the pool).
fn run<R: Send>(py: Python<'_>, n: usize, bytes: usize, f: impl FnOnce() -> R + Send) -> R {
    if n <= 1 && bytes < SINGLE_DOC_DETACH_BYTES {
        f()
    } else {
        py.detach(f)
    }
}

fn int_list<'py>(py: Python<'py>, values: &[i64]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, values)
}

/// `rows` as a list of lists (`row_len` values each; `[]` where `found` is false).
fn row_lists<'py>(
    py: Python<'py>,
    rows: &[i64],
    row_len: usize,
    found: Option<&[bool]>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for (i, row) in rows.chunks_exact(row_len.max(1)).enumerate() {
        if found.is_some_and(|f| !f[i]) {
            out.append(PyList::empty(py))?;
        } else {
            out.append(int_list(py, row)?)?;
        }
    }
    Ok(out)
}

/// `(totals, n_segments, num_refs, tokenized_periods, type_error_at, rows)`.
fn bleu_result<'py>(
    py: Python<'py>,
    stats: &BleuStats,
    row_len: usize,
    want_rows: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let totals = int_list(py, &stats.totals(row_len))?.into_any();
    let rows = if want_rows {
        row_lists(py, &stats.rows, row_len, None)?.into_any()
    } else {
        py.None().into_bound(py)
    };
    (
        totals,
        stats.n_segments,
        stats.num_refs,
        stats.tokenized_periods,
        stats.type_error_at,
        rows,
    )
        .into_pyobject(py)
}

/// BLEU statistics of `hyps` against reference streams `refs`.
///
/// Returns `(totals, n_segments, num_refs, tokenized_periods,
/// type_error_at, rows)`; `rows` (per-segment lists) only if `rows`.
#[pyfunction]
#[pyo3(signature = (hyps, refs, tokenizer_name, max_order, check_tokenized, rows, *, threads = None))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn bleu_corpus_stats<'py>(
    py: Python<'py>,
    hyps: &Bound<'py, PyAny>,
    refs: &Bound<'py, PyAny>,
    tokenizer_name: &str,
    max_order: usize,
    check_tokenized: bool,
    rows: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyTuple>> {
    let cfg = BleuConfig {
        tokenizer: tokenizer(tokenizer_name)?,
        max_order: max_order.max(1),
    };
    let hyp_items = items(hyps, "hypotheses")?;
    let streams = streams(refs)?;
    let n = hyp_items
        .len()
        .min(streams.iter().map(Vec::len).min().unwrap_or(0));
    let hyps = hypotheses(&hyp_items, n)?;
    let t = self::threads(threads);
    let bytes = total_bytes(&hyps, &streams);
    let stats = run(py, n, bytes, || {
        pyroid_core::metrics::sacrebleu::bleu::corpus_stats(
            &hyps,
            &streams,
            cfg,
            check_tokenized,
            t,
        )
    })
    .map_err(pool_error)?;
    bleu_result(py, &stats, cfg.row_len(), rows)
}

/// Pre-computed BLEU references (`BLEU(references=...)`).
#[pyclass(frozen, module = "pyroid.metrics.sacrebleu", name = "_BleuRefCache")]
#[derive(Debug)]
struct PyBleuRefCache(BleuRefCache);

#[pymethods]
impl PyBleuRefCache {
    #[new]
    #[pyo3(signature = (refs, tokenizer_name, max_order, *, threads = None))]
    fn new(
        py: Python<'_>,
        refs: &Bound<'_, PyAny>,
        tokenizer_name: &str,
        max_order: usize,
        threads: Option<Threads>,
    ) -> PyResult<Self> {
        let cfg = BleuConfig {
            tokenizer: tokenizer(tokenizer_name)?,
            max_order: max_order.max(1),
        };
        let streams = streams(refs)?;
        let t = self::threads(threads);
        let n = streams.iter().map(Vec::len).min().unwrap_or(0);
        let bytes = total_bytes(&[], &streams);
        run(py, n, bytes, || BleuRefCache::new(&streams, cfg, t))
            .map(Self)
            .map_err(pool_error)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// `BLEU.num_refs` for the cached references.
    #[getter]
    fn num_refs(&self) -> i64 {
        self.0.num_refs()
    }

    /// The `max_ngram_order` the references were counted with.
    #[getter]
    fn max_order(&self) -> usize {
        self.0.max_order()
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyroid BLEU reference cache: {} segments, max_order={}>",
            self.0.len(),
            self.0.max_order()
        )
    }
}

/// [`bleu_corpus_stats`] against a `_BleuRefCache`.
#[pyfunction]
#[pyo3(signature = (cache, hyps, tokenizer_name, check_tokenized, rows, *, threads = None))]
fn bleu_cache_stats<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyBleuRefCache>,
    hyps: &Bound<'py, PyAny>,
    tokenizer_name: &str,
    check_tokenized: bool,
    rows: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyTuple>> {
    let tok = tokenizer(tokenizer_name)?;
    let cache = &cache.get().0;
    let hyp_items = items(hyps, "hypotheses")?;
    let n = hyp_items.len().min(cache.len());
    let hyps = hypotheses(&hyp_items, n)?;
    let t = self::threads(threads);
    let bytes = total_bytes(&hyps, &[]);
    let stats = run(py, n, bytes, || {
        cache.corpus_stats(&hyps, tok, check_tokenized, t)
    })
    .map_err(pool_error)?;
    bleu_result(py, &stats, 2 + 2 * cache.max_order(), rows)
}

/// `(totals or None, n_segments, num_refs, rows or None)`: totals are
/// `None` when some segment has no row (sacrebleu's `[]`); then `rows` are
/// always returned.
fn chrf_result<'py>(
    py: Python<'py>,
    stats: &ChrfStats,
    row_len: usize,
    want_rows: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let totals = stats.totals(row_len);
    let totals_obj = match &totals {
        Some(t) => int_list(py, t)?.into_any(),
        None => py.None().into_bound(py),
    };
    let rows = if want_rows || totals.is_none() {
        row_lists(py, &stats.rows, row_len, Some(&stats.found))?.into_any()
    } else {
        py.None().into_bound(py)
    };
    (totals_obj, stats.n_segments, stats.num_refs, rows).into_pyobject(py)
}

#[allow(clippy::fn_params_excessive_bools)]
fn chrf_config(
    char_order: usize,
    word_order: usize,
    whitespace: bool,
    factor: f64,
    one_plus_factor: f64,
    eps_smoothing: bool,
) -> PyResult<ChrfConfig> {
    if char_order + word_order == 0 {
        return Err(PyValueError::new_err(
            "char_order + word_order must be at least 1",
        ));
    }
    Ok(ChrfConfig {
        char_order,
        word_order,
        whitespace,
        factor,
        one_plus_factor,
        eps_smoothing,
    })
}

/// chrF statistics of `hyps` against reference streams `refs`.
///
/// `factor` and `one_plus_factor` are Python's `float(beta ** 2)` and
/// `float(1 + beta ** 2)`. Returns `(totals, n_segments, num_refs, rows)`
/// (see `chrf_result`).
#[pyfunction]
#[pyo3(signature = (
    hyps, refs, char_order, word_order, whitespace, factor, one_plus_factor, eps_smoothing, rows,
    *, threads = None
))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn chrf_corpus_stats<'py>(
    py: Python<'py>,
    hyps: &Bound<'py, PyAny>,
    refs: &Bound<'py, PyAny>,
    char_order: usize,
    word_order: usize,
    whitespace: bool,
    factor: f64,
    one_plus_factor: f64,
    eps_smoothing: bool,
    rows: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyTuple>> {
    let cfg = chrf_config(
        char_order,
        word_order,
        whitespace,
        factor,
        one_plus_factor,
        eps_smoothing,
    )?;
    let hyp_items = items(hyps, "hypotheses")?;
    let streams = streams(refs)?;
    let n = hyp_items
        .len()
        .min(streams.iter().map(Vec::len).min().unwrap_or(0));
    let hyps = hypotheses(&hyp_items, n)?;
    let t = self::threads(threads);
    let bytes = total_bytes(&hyps, &streams);
    let stats = run(py, n, bytes, || {
        pyroid_core::metrics::sacrebleu::chrf::corpus_stats(&hyps, &streams, &cfg, t)
    })
    .map_err(pool_error)?;
    chrf_result(py, &stats, cfg.row_len(), rows)
}

/// Pre-computed chrF references (`CHRF(references=...)`).
#[pyclass(frozen, module = "pyroid.metrics.sacrebleu", name = "_ChrfRefCache")]
#[derive(Debug)]
struct PyChrfRefCache(ChrfRefCache);

#[pymethods]
impl PyChrfRefCache {
    #[new]
    #[pyo3(signature = (refs, char_order, word_order, whitespace, *, threads = None))]
    fn new(
        py: Python<'_>,
        refs: &Bound<'_, PyAny>,
        char_order: usize,
        word_order: usize,
        whitespace: bool,
        threads: Option<Threads>,
    ) -> PyResult<Self> {
        let cfg = chrf_config(char_order, word_order, whitespace, 0.0, 1.0, false)?;
        let streams = streams(refs)?;
        let t = self::threads(threads);
        let n = streams.iter().map(Vec::len).min().unwrap_or(0);
        let bytes = total_bytes(&[], &streams);
        run(py, n, bytes, || ChrfRefCache::new(&streams, &cfg, t))
            .map(Self)
            .map_err(pool_error)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// `CHRF.num_refs` for the cached references.
    #[getter]
    fn num_refs(&self) -> i64 {
        self.0.num_refs()
    }

    /// `(char_order, word_order, whitespace)` the references were counted with.
    #[getter]
    fn settings(&self) -> (usize, usize, bool) {
        self.0.settings()
    }

    fn __repr__(&self) -> String {
        let (c, w, ws) = self.0.settings();
        format!(
            "<pyroid chrF reference cache: {} segments, char_order={c}, word_order={w}, \
             whitespace={ws}>",
            self.0.len()
        )
    }
}

/// [`chrf_corpus_stats`] against a `_ChrfRefCache` (whose orders must be
/// `char_order`/`word_order`).
#[pyfunction]
#[pyo3(signature = (
    cache, hyps, char_order, word_order, whitespace, factor, one_plus_factor, eps_smoothing, rows,
    *, threads = None
))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn chrf_cache_stats<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyChrfRefCache>,
    hyps: &Bound<'py, PyAny>,
    char_order: usize,
    word_order: usize,
    whitespace: bool,
    factor: f64,
    one_plus_factor: f64,
    eps_smoothing: bool,
    rows: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyTuple>> {
    let cfg = chrf_config(
        char_order,
        word_order,
        whitespace,
        factor,
        one_plus_factor,
        eps_smoothing,
    )?;
    let cache = &cache.get().0;
    let (c, w, _) = cache.settings();
    if (c, w) != (char_order, word_order) {
        return Err(PyValueError::new_err(
            "chrF orders differ from the cached references' orders",
        ));
    }
    let hyp_items = items(hyps, "hypotheses")?;
    let n = hyp_items.len().min(cache.len());
    let hyps = hypotheses(&hyp_items, n)?;
    let t = self::threads(threads);
    let bytes = total_bytes(&hyps, &[]);
    let stats = run(py, n, bytes, || cache.corpus_stats(&hyps, &cfg, t)).map_err(pool_error)?;
    chrf_result(py, &stats, cfg.row_len(), rows)
}

/// `tokenizer(line)` for a native sacrebleu tokenizer (`13a`, `intl`,
/// `zh`, `char`, `none`).
#[pyfunction]
fn sb_tokenize(py: Python<'_>, tokenizer_name: &str, line: PyBackedStr) -> PyResult<String> {
    let tok = tokenizer(tokenizer_name)?;
    Ok(run(py, 1, line.len(), || tok.tokenize(&line).into_owned()))
}

/// `(sacrebleu, regex, Unicode)` versions the native tables were generated from.
#[pyfunction]
fn sb_table_versions() -> (&'static str, &'static str, &'static str) {
    use pyroid_core::metrics::sacrebleu::{
        REGEX_UNICODE_VERSION, REGEX_VERSION, SACREBLEU_VERSION,
    };
    (SACREBLEU_VERSION, REGEX_VERSION, REGEX_UNICODE_VERSION)
}

/// Add this half's functions and classes to the shared `pyroid.metrics` native module.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sb_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(sb_table_versions, m)?)?;
    m.add_function(wrap_pyfunction!(bleu_corpus_stats, m)?)?;
    m.add_function(wrap_pyfunction!(bleu_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(chrf_corpus_stats, m)?)?;
    m.add_function(wrap_pyfunction!(chrf_cache_stats, m)?)?;
    m.add_class::<PyBleuRefCache>()?;
    m.add_class::<PyChrfRefCache>()?;
    Ok(())
}
