// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! `pyroid._pyroid.dedup`: native entry points behind `pyroid.dedup`.
//!
//! The typed Python module (`python/pyroid/dedup.py`) owns the config
//! classes and argument handling; this module runs the kernels. Mutable
//! state (`_MinHashState`, `_ExactState`) sits behind a `Mutex` that is only
//! ever locked while detached from the interpreter, so a thread waiting for
//! it never holds the GIL (no lock-order deadlock on GIL builds).

use std::sync::{Arc, Mutex, PoisonError};

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyList, PyString, PyTuple};
use pyroid_core::columnar::BoolBuffers;
use pyroid_core::dedup::exact::normalized_hash;
use pyroid_core::dedup::{
    CapacityError, Clusters, ExactSeen, HashConfig, HashFn, MinHasher, MinhashConfig, Precision,
    SignatureStore, TextNormConfig, TokenizeError, UnicodeVersion, simplify_text,
};
use pyroid_core::filters;
use pyroid_core::par;

use crate::arrow::{Chunk, Column, ColumnType};
use crate::batch::{
    BatchError, DocCost, doc_count, fill_rows, map_values, map_values_cost, transform,
};
use crate::errors::{PyroidError, input_error, type_name, with_note};
use crate::input::{TextInput, Threads, ThreadsArg, texts_and_threads, threads};

fn unicode_version(v: &str) -> PyResult<UnicodeVersion> {
    UnicodeVersion::from_unidata_version(v)
        .ok_or_else(|| PyValueError::new_err(format!("unsupported unicode_version {v:?}")))
}

/// A validated `TextNormConfig` plus the interpreter's Unicode version.
#[pyclass(frozen, module = "pyroid.dedup", name = "_NativeNorm")]
#[derive(Debug)]
pub struct NativeNorm {
    norm: TextNormConfig,
    unicode: UnicodeVersion,
}

#[pymethods]
impl NativeNorm {
    #[new]
    #[pyo3(signature = (
        *, lowercase, norm_whitespace, remove_punctuation, norm_unicode_diacritics,
        norm_numbers, norm_weekdays, norm_monthnames, unicode_version
    ))]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    fn new(
        lowercase: bool,
        norm_whitespace: bool,
        remove_punctuation: bool,
        norm_unicode_diacritics: bool,
        norm_numbers: bool,
        norm_weekdays: bool,
        norm_monthnames: bool,
        unicode_version: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            norm: TextNormConfig {
                lowercase,
                norm_whitespace,
                remove_punctuation,
                norm_unicode_diacritics,
                norm_numbers,
                norm_weekdays,
                norm_monthnames,
            },
            unicode: self::unicode_version(unicode_version)?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyroid.dedup native norm {:?} {:?}>",
            self.norm, self.unicode
        )
    }
}

/// A validated `MinhashConfig` with its permutation parameters.
#[pyclass(frozen, module = "pyroid.dedup", name = "_NativeConfig")]
#[derive(Debug)]
pub struct NativeConfig {
    hasher: MinHasher,
}

#[pymethods]
impl NativeConfig {
    /// `language`: the datatrove code whose tokenizer splits the words
    /// (`MinhashDedupSignature(language=...)`); `None` is datatrove's
    /// default, English. It must be one pyroid tokenizes natively (the shim
    /// checks, and tokenizes the others in Python).
    #[new]
    #[pyo3(signature = (
        *, n_grams, num_buckets, hashes_per_bucket, seed, hash_fc, precision, norm,
        language = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        n_grams: usize,
        num_buckets: usize,
        hashes_per_bucket: usize,
        seed: u32,
        hash_fc: &str,
        precision: u32,
        norm: &NativeNorm,
        language: Option<&str>,
    ) -> PyResult<Self> {
        let hash_fc = HashFn::from_name(hash_fc)
            .ok_or_else(|| PyValueError::new_err(format!("unknown hash_fc {hash_fc:?}")))?;
        let precision = Precision::from_bits(precision)
            .ok_or_else(|| PyValueError::new_err("precision must be 32 or 64"))?;
        let config = MinhashConfig {
            n_grams,
            num_buckets,
            hashes_per_bucket,
            seed,
            norm_config: norm.norm,
            hash_config: HashConfig { precision, hash_fc },
        };
        let mut hasher = MinHasher::new(config, norm.unicode)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if let Some(language) = language {
            crate::segmenters::prepare(py, language)?;
            let tok =
                filters::tokenizer_for(language, norm.unicode).ok_or_else(
                    || match filters::backend(language) {
                        None => PyValueError::new_err(format!(
                            "Language '{language}' doesn't have a tokenizer assigned. Pass in a \
                         WordTokenizer directly or update tokenizer_assignment.csv"
                        )),
                        Some(b) => PyNotImplementedError::new_err(format!(
                            "pyroid has no native word tokenizer for language {language:?} \
                         (datatrove's {}({:?})){}",
                            b.class,
                            b.arg,
                            crate::segmenters::hint(language)
                        )),
                    },
                )?;
            hasher = hasher.with_tokenizer(tok);
        }
        Ok(Self { hasher })
    }

    /// The permutation parameters ``(a, b)``, as datatrove draws them.
    fn parameters(&self) -> (Vec<u64>, Vec<u64>) {
        let (a, b) = self.hasher.parameters();
        (a.to_vec(), b.to_vec())
    }

    #[getter]
    fn num_hashes(&self) -> usize {
        self.hasher.num_hashes()
    }

    fn __repr__(&self) -> String {
        format!("<pyroid.dedup native config {:?}>", self.hasher.config())
    }
}

/// One document's stage-1 result: `None` when it has no shingles.
type Signature = Option<Box<[u64]>>;

/// Stage 1 with datatrove's (spaCy) tokenizer.
fn text_signature(hasher: &MinHasher, text: &str) -> Result<Signature, TokenizeError> {
    let mut out = vec![0; hasher.num_hashes()];
    Ok(hasher
        .signature(text, &mut out)?
        .then(|| out.into_boxed_slice()))
}

fn token_signature(hasher: &MinHasher, tokens: &[PyBackedStr]) -> Signature {
    let mut out = vec![0; hasher.num_hashes()];
    hasher
        .signature_from_tokens(tokens, &mut out)
        .then(|| out.into_boxed_slice())
}

/// Signatures of every document (`None` for Arrow nulls), detached.
fn signatures_of(
    py: Python<'_>,
    hasher: &MinHasher,
    input: &TextInput,
    threads_arg: Option<Threads>,
) -> PyResult<Vec<Signature>> {
    map_values_cost(py, input, threads_arg, DocCost::Heavy, |s| {
        text_signature(hasher, s)
    })?
    .into_iter()
    .enumerate()
    .map(|(i, r)| match r {
        None => Ok(None),
        Some(Ok(sig)) => Ok(sig),
        // spaCy's E025 (2**30 characters or more), raised as datatrove raises it.
        Some(Err(e)) => Err(with_note(
            py,
            PyValueError::new_err(e.to_string()),
            format!("pyroid: while tokenizing document {i}"),
        )),
    })
    .collect()
}

/// Why adding documents to a MinHash index failed (converted once re-attached).
#[derive(Debug)]
enum AddError {
    Batch(BatchError),
    /// spaCy's E025 on document `.0`.
    Tokenize(usize, TokenizeError),
    Capacity(CapacityError),
    Pool(par::PoolError),
}

impl From<CapacityError> for AddError {
    fn from(e: CapacityError) -> Self {
        Self::Capacity(e)
    }
}

impl AddError {
    fn into_pyerr(self, py: Python<'_>) -> PyErr {
        match self {
            Self::Batch(e) => e.into_pyerr(py),
            // spaCy's E025 (2**30 characters or more), raised as datatrove raises it.
            Self::Tokenize(i, e) => with_note(
                py,
                PyValueError::new_err(e.to_string()),
                format!("pyroid: while tokenizing document {i}"),
            ),
            Self::Capacity(e) => PyValueError::new_err(e.to_string()),
            Self::Pool(e) => PyroidError::new_err(e.to_string()),
        }
    }
}

/// Stage 1 of every document of `input`, written straight into `store` (one
/// flat row per document: no per-document signature allocations, no copy).
/// Pure Rust: runs detached. On error `store` is unchanged.
fn append_text_signatures(
    store: &mut SignatureStore,
    hasher: &MinHasher,
    input: &TextInput,
    threads_arg: Option<Threads>,
) -> Result<(), AddError> {
    let h = hasher.num_hashes();
    store.append_with(doc_count(input), |rows| {
        fill_rows(input, threads_arg, rows, h, |s, row| {
            hasher.signature(s, row)
        })
        .map_err(AddError::Batch)?
        .into_iter()
        .enumerate()
        .map(|(i, r)| match r {
            None => Ok(false), // Arrow null: no signature
            Some(Ok(has)) => Ok(has),
            Some(Err(e)) => Err(AddError::Tokenize(i, e)), // the first failing document
        })
        .collect()
    })
}

/// `list[list[str]]` (pre-tokenized documents) for the test hooks.
fn token_lists(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<PyBackedStr>>> {
    let py = obj.py();
    let Ok(outer) = obj.cast::<PyList>() else {
        return Err(input_error(
            py,
            format!("expected list[list[str]], got {}", type_name(obj)),
        ));
    };
    outer
        .iter()
        .enumerate()
        .map(|(i, doc)| {
            let Ok(doc) = doc.cast::<PyList>() else {
                return Err(input_error(
                    py,
                    format!(
                        "expected a list of str at index {i}, got {}",
                        type_name(&doc)
                    ),
                ));
            };
            doc.iter()
                .map(|t| {
                    t.cast_into::<PyString>()
                        .map_err(|e| input_error(py, format!("expected str tokens: {e}")))
                        .and_then(PyBackedStr::try_from)
                })
                .collect()
        })
        .collect()
}

fn token_signatures(
    py: Python<'_>,
    hasher: &MinHasher,
    docs: &[Vec<PyBackedStr>],
    threads_arg: Option<Threads>,
) -> PyResult<Vec<Signature>> {
    let t = threads(threads_arg);
    py.detach(|| par::map(docs.len(), t, |i| token_signature(hasher, &docs[i])))
        .map_err(|e| BatchError::from(e).into_pyerr(py))
}

fn signatures_to_py<'py>(py: Python<'py>, sigs: &[Signature]) -> PyResult<Bound<'py, PyList>> {
    let items = sigs
        .iter()
        .map(|s| match s {
            None => Ok(py.None().into_bound(py)),
            Some(v) => Ok(PyList::new(py, v.iter())?.into_any()),
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// datatrove's MinHash signature of each document (flat, bucket by bucket).
#[pyfunction]
#[pyo3(signature = (texts, /, config, *, threads = None))]
fn _minhash_signatures<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    config: &NativeConfig,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let (texts, threads) = texts_and_threads(texts, threads, None)?;
    let sigs = signatures_of(py, &config.hasher, &texts, threads)?;
    if let TextInput::Str(_) = texts {
        return signatures_to_py(py, &sigs)?.get_item(0);
    }
    Ok(signatures_to_py(py, &sigs)?.into_any())
}

/// Signatures from pre-tokenized documents (datatrove's
/// ``word_tokenize(simplify_text(text))``); a parity-test hook.
#[pyfunction]
#[pyo3(signature = (tokens, /, config, *, threads = None))]
fn _signatures_from_tokens<'py>(
    py: Python<'py>,
    tokens: &Bound<'py, PyAny>,
    config: &NativeConfig,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyList>> {
    let docs = token_lists(tokens)?;
    let sigs = token_signatures(py, &config.hasher, &docs, threads)?;
    signatures_to_py(py, &sigs)
}

/// datatrove's ``simplify_text`` over a str, list/tuple of str or Arrow column.
#[pyfunction]
#[pyo3(signature = (texts, /, norm, *, threads = None))]
fn _simplify_text<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    norm: &NativeNorm,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let (texts, threads) = texts_and_threads(texts, threads, None)?;
    transform(py, texts, threads, |s| {
        simplify_text(s, &norm.norm, norm.unicode)
    })
}

/// A mask in the caller's kind: bool, list, tuple, or an Arrow bool `Column`
/// with the input's chunking and nulls.
fn mask_like<'py>(
    py: Python<'py>,
    input: &TextInput,
    values: &[Option<bool>],
) -> PyResult<Bound<'py, PyAny>> {
    let plain = || values.iter().map(|v| v.unwrap_or(true));
    Ok(match input {
        TextInput::Str(_) => PyBool::new(py, values.first().copied().flatten().unwrap_or(true))
            .to_owned()
            .into_any(),
        TextInput::List(_) => PyList::new(py, plain())?.into_any(),
        TextInput::Tuple(_) => PyTuple::new(py, plain())?.into_any(),
        TextInput::Arrow(strings) => {
            let mut chunks = Vec::with_capacity(strings.chunks.len());
            let mut start = 0;
            for c in &strings.chunks {
                let end = start + c.len();
                chunks.push(Chunk::Bool(BoolBuffers::build(&values[start..end])));
                start = end;
            }
            Bound::new(
                py,
                Column::new(ColumnType::Bool, &strings.field.name, chunks),
            )?
            .into_any()
        }
    })
}

/// Which documents of `input` are Arrow nulls.
fn null_rows(input: &TextInput) -> Vec<bool> {
    match input {
        TextInput::Arrow(strings) => strings
            .chunks
            .iter()
            .flat_map(|c| (0..c.len()).map(move |i| c.get(i).is_none()))
            .collect(),
        TextInput::Str(_) => vec![false],
        TextInput::List(d) | TextInput::Tuple(d) => vec![false; d.len()],
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // A panic mid-update can leave a partial batch behind, never unsound state.
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Debug)]
struct MinHashInner {
    store: SignatureStore,
    clusters: Option<Arc<Clusters>>,
}

impl MinHashInner {
    fn append(&mut self, sigs: Vec<Signature>) -> PyResult<(usize, usize)> {
        let start = self.store.len();
        self.store
            .reserve(sigs.len())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        for s in &sigs {
            self.store
                .push(s.as_deref())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        self.clusters = None;
        Ok((start, self.store.len()))
    }

    fn clusters(&mut self, t: par::Threads) -> Result<Arc<Clusters>, par::PoolError> {
        if let Some(c) = &self.clusters {
            return Ok(Arc::clone(c));
        }
        let c = Arc::new(self.store.cluster(t)?);
        self.clusters = Some(Arc::clone(&c));
        Ok(c)
    }
}

/// The in-memory MinHash index behind `pyroid.dedup.MinHashDedup`.
#[pyclass(frozen, module = "pyroid.dedup", name = "_MinHashState")]
#[derive(Debug)]
pub struct MinHashState {
    config: Py<NativeConfig>,
    threads: Option<Threads>,
    inner: Mutex<MinHashInner>,
}

impl MinHashState {
    fn hasher(&self) -> &MinHasher {
        &self.config.get().hasher
    }

    fn append(&self, py: Python<'_>, sigs: Vec<Signature>) -> PyResult<(usize, usize)> {
        py.detach(|| lock(&self.inner).append(sigs))
    }

    fn clusters(&self, py: Python<'_>) -> PyResult<Arc<Clusters>> {
        let t = threads(self.threads);
        py.detach(|| lock(&self.inner).clusters(t))
            .map_err(|e| PyroidError::new_err(e.to_string()))
    }
}

#[pymethods]
impl MinHashState {
    #[new]
    #[pyo3(signature = (config, /, *, threads = None))]
    fn new(config: Py<NativeConfig>, threads: Option<Threads>) -> Self {
        let store = SignatureStore::new(config.get().hasher.config());
        Self {
            config,
            threads,
            inner: Mutex::new(MinHashInner {
                store,
                clusters: None,
            }),
        }
    }

    /// Add documents; returns ``(start, stop)``, the indices they were given.
    #[pyo3(signature = (texts, /, *, threads = None))]
    fn add(
        &self,
        py: Python<'_>,
        texts: &Bound<'_, PyAny>,
        threads: Option<ThreadsArg>,
    ) -> PyResult<(usize, usize)> {
        let (texts, threads) = texts_and_threads(texts, threads, self.threads)?;
        // The lock is held (detached) while the signatures are computed, so
        // they can be written into the store in place.
        let t = threads.or(self.threads);
        py.detach(|| {
            let mut inner = lock(&self.inner);
            let start = inner.store.len();
            append_text_signatures(&mut inner.store, self.hasher(), &texts, t)?;
            inner.clusters = None;
            Ok((start, inner.store.len()))
        })
        .map_err(|e: AddError| e.into_pyerr(py))
    }

    /// Add pre-tokenized documents (the Python-tokenizer path, and a
    /// parity-test hook).
    #[pyo3(signature = (tokens, /, *, threads = None))]
    fn _add_tokens(
        &self,
        py: Python<'_>,
        tokens: &Bound<'_, PyAny>,
        threads: Option<Threads>,
    ) -> PyResult<(usize, usize)> {
        let docs = token_lists(tokens)?;
        let sigs = token_signatures(py, self.hasher(), &docs, threads.or(self.threads))?;
        drop(docs);
        self.append(py, sigs)
    }

    /// [`Self::fit`] for pre-tokenized documents: replace the index with them
    /// and return their keep mask as a list (the shim packs it).
    #[pyo3(signature = (tokens, /, *, threads = None))]
    fn _fit_tokens<'py>(
        &self,
        py: Python<'py>,
        tokens: &Bound<'py, PyAny>,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyList>> {
        let t = threads.or(self.threads);
        let docs = token_lists(tokens)?;
        let sigs = token_signatures(py, self.hasher(), &docs, t)?;
        drop(docs);
        let tt = self::threads(t);
        let keep = py.detach(|| {
            // A new store, filled outside the lock; the old index stays as
            // it was if this fails.
            let mut fresh = MinHashInner {
                store: SignatureStore::new(self.hasher().config()),
                clusters: None,
            };
            fresh.append(sigs)?;
            let mut inner = lock(&self.inner);
            inner.store = fresh.store;
            inner.clusters = None;
            inner
                .clusters(tt)
                .map_err(|e| PyroidError::new_err(e.to_string()))
        })?;
        PyList::new(py, (0..keep.len()).map(|i| keep.is_kept(i)))
    }

    /// Replace the index with ``texts``; return their keep mask in the
    /// caller's kind (Arrow nulls stay null).
    #[pyo3(signature = (texts, /, *, threads = None))]
    fn fit<'py>(
        &self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        threads: Option<ThreadsArg>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (texts, threads) = texts_and_threads(texts, threads, self.threads)?;
        let t = threads.or(self.threads);
        let nulls = null_rows(&texts);
        let tt = self::threads(t);
        let keep = py
            .detach(|| {
                // A new store, filled in place outside the lock; the old index
                // stays as it was if this fails.
                let mut fresh = SignatureStore::new(self.hasher().config());
                append_text_signatures(&mut fresh, self.hasher(), &texts, t)?;
                let mut inner = lock(&self.inner);
                inner.store = fresh;
                inner.clusters = None;
                inner.clusters(tt).map_err(AddError::Pool)
            })
            .map_err(|e| e.into_pyerr(py))?;
        let values: Vec<Option<bool>> = (0..keep.len())
            .map(|i| (!nulls[i]).then(|| keep.is_kept(i)))
            .collect();
        mask_like(py, &texts, &values)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| lock(&self.inner).store.len())
    }

    fn reset(&self, py: Python<'_>) {
        py.detach(|| {
            let mut inner = lock(&self.inner);
            inner.store.clear();
            inner.clusters = None;
        });
    }

    fn keep_mask<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let c = self.clusters(py)?;
        PyList::new(py, (0..c.len()).map(|i| c.is_kept(i)))
    }

    fn duplicates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let c = self.clusters(py)?;
        PyList::new(py, c.duplicates())
    }

    #[pyo3(name = "clusters")]
    fn clusters_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let c = self.clusters(py)?;
        let items = c
            .clusters()
            .into_iter()
            .map(|members| PyList::new(py, members))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    fn cluster_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let c = self.clusters(py)?;
        PyList::new(py, c.cluster_ids())
    }

    fn cluster_sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let c = self.clusters(py)?;
        PyList::new(py, c.cluster_sizes())
    }

    fn representatives<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let c = self.clusters(py)?;
        PyList::new(py, (0..c.len()).map(|i| c.representative(i)))
    }

    /// Stage-2 duplicate pairs of one bucket, in datatrove's order (test hook).
    fn _bucket_pairs<'py>(&self, py: Python<'py>, bucket: usize) -> PyResult<Bound<'py, PyList>> {
        let pairs = py.detach(|| {
            let inner = lock(&self.inner);
            (bucket < inner.store.num_buckets()).then(|| inner.store.bucket_pairs(bucket))
        });
        let pairs = pairs.ok_or_else(|| PyIndexError::new_err("bucket out of range"))?;
        PyList::new(py, pairs)
    }

    /// The signature of document ``i`` (``None`` if it has no shingles).
    fn signature<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        let sig = py.detach(|| {
            let inner = lock(&self.inner);
            (i < inner.store.len()).then(|| inner.store.get(i).map(<[u64]>::to_vec))
        });
        match sig.ok_or_else(|| PyIndexError::new_err("document index out of range"))? {
            None => Ok(py.None().into_bound(py)),
            Some(v) => Ok(PyList::new(py, v)?.into_any()),
        }
    }
}

fn exact_hashes(
    py: Python<'_>,
    input: &TextInput,
    norm: Option<&NativeNorm>,
    threads_arg: Option<Threads>,
) -> PyResult<Vec<Option<u128>>> {
    let cfg = norm.map(|n| (n.norm, n.unicode));
    map_values(py, input, threads_arg, |s| match &cfg {
        None => normalized_hash(s, None, UnicodeVersion::LATEST),
        Some((norm, v)) => normalized_hash(s, Some(norm), *v),
    })
}

/// Keep-first mask of exact (optionally normalized) duplicates.
#[pyfunction]
#[pyo3(signature = (texts, /, norm, *, threads = None))]
fn _exact_dedup<'py>(
    py: Python<'py>,
    texts: &Bound<'py, PyAny>,
    norm: Option<&NativeNorm>,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let (texts, threads) = texts_and_threads(texts, threads, None)?;
    let hashes = exact_hashes(py, &texts, norm, threads)?;
    let keep = py.detach(|| ExactSeen::new().keep_first(&hashes));
    mask_like(py, &texts, &keep)
}

/// The streaming exact-dedup state behind `pyroid.dedup.ExactDedup`.
#[pyclass(frozen, module = "pyroid.dedup", name = "_ExactState")]
#[derive(Debug)]
pub struct ExactState {
    norm: Option<Py<NativeNorm>>,
    seen: Mutex<ExactSeen>,
}

#[pymethods]
impl ExactState {
    #[new]
    #[pyo3(signature = (norm, /))]
    fn new(norm: Option<Py<NativeNorm>>) -> Self {
        Self {
            norm,
            seen: Mutex::new(ExactSeen::new()),
        }
    }

    /// Keep mask for one more batch: ``True`` for each first copy so far.
    #[pyo3(signature = (texts, /, *, threads = None))]
    fn add<'py>(
        &self,
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        threads: Option<ThreadsArg>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (texts, threads) = texts_and_threads(texts, threads, None)?;
        let norm = self.norm.as_ref().map(Py::get);
        let hashes = exact_hashes(py, &texts, norm, threads)?;
        let keep = py.detach(|| lock(&self.seen).keep_first(&hashes));
        mask_like(py, &texts, &keep)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        py.detach(|| lock(&self.seen).len())
    }

    fn reset(&self, py: Python<'_>) {
        py.detach(|| lock(&self.seen).clear());
    }
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.dedup")?;
    m.gil_used(false)?;
    m.add_class::<NativeNorm>()?;
    m.add_class::<NativeConfig>()?;
    m.add_class::<MinHashState>()?;
    m.add_class::<ExactState>()?;
    m.add_function(wrap_pyfunction!(_minhash_signatures, &m)?)?;
    m.add_function(wrap_pyfunction!(_signatures_from_tokens, &m)?)?;
    m.add_function(wrap_pyfunction!(_simplify_text, &m)?)?;
    m.add_function(wrap_pyfunction!(_exact_dedup, &m)?)?;
    register_more(&m)?;
    parent.add_submodule(&m)
}

// ===========================================================================
// More datatrove 0.10.1 steps: one inline module per step, each registering
// its classes and functions on `pyroid._pyroid.dedup`.
// ===========================================================================

fn register_more(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sentence::register(m)?;
    files::register(m)?;
    Ok(())
}

// ---- SentenceDedup ----
/// datatrove's sentence dedup (`SentenceDedupSignature` -> `SentenceFindDedups`
/// -> `SentenceDedupFilter`): the native side of `pyroid.dedup.SentenceDedup`
/// and of the datatrove drop-ins. `_SentenceState` holds one task's
/// signatures; `_SentenceDups` one task's duplicate windows (from the state,
/// or parsed from `.c4_dup` files) and runs stage 3.
mod sentence {
    use std::borrow::Cow;
    use std::fs::File;
    use std::io::{self, Read};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use pyo3::exceptions::{
        PyAssertionError, PyNotImplementedError, PyOverflowError, PyValueError, PyZeroDivisionError,
    };
    use pyo3::prelude::*;
    use pyo3::pybacked::{PyBackedBytes, PyBackedStr};
    use pyo3::types::{PyBytes, PyList, PyNone, PyString, PyTuple};
    use pyroid_core::columnar::StringBuffers;
    use pyroid_core::dedup::sentence::{
        self as core, Duplicates, FileError, Filtered, SentDedupConfig, SentenceDeduper,
        SentenceError, Signature,
    };
    use pyroid_core::dedup::{HashConfig, HashFn, Precision};
    use pyroid_core::filters::FilterError;
    use pyroid_core::par::{self, PoolError};

    use super::{NativeNorm, lock, mask_like};
    use crate::arrow::{Chunk, Column, ColumnType, StrType, build_error};
    use crate::batch::{BatchError, doc_count};
    use crate::errors::with_note;
    use crate::input::{TextInput, Threads, ThreadsArg, texts_and_threads, threads};

    /// A sentence-dedup kernel error (or a batch error), raised once attached.
    #[derive(Debug)]
    enum RunError {
        Batch(BatchError),
        Doc { index: usize, err: SentenceError },
    }

    impl From<PoolError> for RunError {
        fn from(e: PoolError) -> Self {
            Self::Batch(BatchError::from(e))
        }
    }

    fn sentence_err(err: &SentenceError) -> PyErr {
        let msg = err.to_string();
        match err {
            SentenceError::Tokenize(FilterError::ZeroDivision) => PyZeroDivisionError::new_err(msg),
            SentenceError::Tokenize(_) | SentenceError::DuplicateLineIndex => {
                PyValueError::new_err(msg)
            }
            SentenceError::NoTokenizer => PyNotImplementedError::new_err(
                "pyroid has no native word tokenizer for this language \
                 (pyroid.filters.languages() lists the native ones)",
            ),
            SentenceError::SentenceOverflow(_) | SentenceError::DocOverflow(_) => {
                PyOverflowError::new_err(msg)
            }
        }
    }

    impl RunError {
        fn into_pyerr(self, py: Python<'_>) -> PyErr {
            match self {
                Self::Batch(e) => e.into_pyerr(py),
                Self::Doc { index, err } => with_note(
                    py,
                    sentence_err(&err),
                    format!("pyroid.dedup: raised by document {index} of the batch"),
                ),
            }
        }
    }

    fn file_err(py: Python<'_>, e: FileError) -> PyErr {
        let msg = e.to_string();
        match e {
            FileError::Io { file, err } => with_note(
                py,
                PyErr::from(err),
                format!("pyroid.dedup: while reading file {file}"),
            ),
            FileError::Truncated { .. } => PyValueError::new_err(msg),
            FileError::Unsorted { .. } => PyAssertionError::new_err(msg),
        }
    }

    /// `f(index, text)` for every document of `input` (`index` counts Arrow
    /// nulls), in parallel; `None` for Arrow nulls. Pure Rust: call it
    /// detached.
    fn map_indexed<'a, R, F>(
        input: &'a TextInput,
        t: par::Threads,
        f: F,
    ) -> Result<Vec<Option<R>>, RunError>
    where
        R: Send,
        F: Fn(usize, &'a str) -> Result<R, SentenceError> + Sync + Send,
    {
        let doc = |index: usize, s: &'a str| {
            f(index, s)
                .map(Some)
                .map_err(|err| RunError::Doc { index, err })
        };
        match input {
            TextInput::Str(s) => Ok(vec![doc(0, s)?]),
            TextInput::List(d) | TextInput::Tuple(d) => par::try_map(d.len(), t, |i| doc(i, &d[i])),
            TextInput::Arrow(strings) => {
                let mut out = Vec::with_capacity(strings.len());
                let mut base = 0;
                for (c, chunk) in strings.chunks.iter().enumerate() {
                    out.extend(par::try_map(chunk.len(), t, |row| match chunk.get(row) {
                        None => Ok(None),
                        Some(Ok(s)) => doc(base + row, s),
                        Some(Err(err)) => Err(RunError::Batch(BatchError::BadUtf8 {
                            chunk: c,
                            row,
                            bytes: chunk.bytes(row).unwrap_or_default().to_vec(),
                            err,
                        })),
                    })?);
                    base += chunk.len();
                }
                Ok(out)
            }
        }
    }

    /// A validated `SentDedupConfig` with the interpreter's Unicode version
    /// and the language's tokenizer.
    #[pyclass(frozen, module = "pyroid.dedup", name = "_SentenceConfig")]
    #[derive(Debug)]
    pub struct SentenceConfig {
        kernel: SentenceDeduper,
    }

    #[pymethods]
    impl SentenceConfig {
        #[new]
        #[pyo3(signature = (
            *, n_sentences, split_sentences, only_dedup_in_index, min_doc_words,
            min_num_sentences, min_words_to_remove_span, hash_fc, precision, norm, language
        ))]
        #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
        fn new(
            py: Python<'_>,
            n_sentences: usize,
            split_sentences: bool,
            only_dedup_in_index: bool,
            min_doc_words: usize,
            min_num_sentences: usize,
            min_words_to_remove_span: usize,
            hash_fc: &str,
            precision: u32,
            norm: &NativeNorm,
            language: &str,
        ) -> PyResult<Self> {
            let hash_fc = HashFn::from_name(hash_fc)
                .ok_or_else(|| PyValueError::new_err(format!("unknown hash_fc {hash_fc:?}")))?;
            let precision = Precision::from_bits(precision)
                .ok_or_else(|| PyValueError::new_err("precision must be 32 or 64"))?;
            let config = SentDedupConfig {
                n_sentences,
                split_sentences,
                only_dedup_in_index,
                min_doc_words,
                min_num_sentences,
                min_words_to_remove_span,
                norm_config: norm.norm,
                hash_config: HashConfig { precision, hash_fc },
            };
            crate::segmenters::prepare(py, language)?;
            let kernel = SentenceDeduper::new(config, norm.unicode, language)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self { kernel })
        }

        /// Does pyroid have a word tokenizer for the language?
        #[getter]
        fn has_tokenizer(&self) -> bool {
            self.kernel.has_tokenizer()
        }

        /// Does stage 1 (``stage3=False``) or stage 3 need the tokenizer?
        fn needs_tokenizer(&self, stage3: bool) -> bool {
            self.kernel.config().needs_tokenizer(stage3)
        }

        fn __repr__(&self) -> String {
            format!(
                "<pyroid.dedup native sentence config {:?}>",
                self.kernel.config()
            )
        }
    }

    fn precision_of(config: &SentenceConfig) -> Precision {
        config.kernel.config().hash_config.precision
    }

    #[derive(Debug, Default)]
    struct StateInner {
        /// Signatures of every document added (sorted lazily).
        sigs: Vec<Signature>,
        docs: usize,
        dups: Option<Arc<Duplicates>>,
    }

    /// One task's stage-1 signatures (`pyroid.dedup.SentenceDedup`'s store).
    #[pyclass(frozen, module = "pyroid.dedup", name = "_SentenceState")]
    #[derive(Debug)]
    pub struct SentenceState {
        config: Py<SentenceConfig>,
        threads: Option<Threads>,
        inner: Mutex<StateInner>,
    }

    #[pymethods]
    impl SentenceState {
        #[new]
        #[pyo3(signature = (config, /, *, threads = None))]
        fn new(config: Py<SentenceConfig>, threads: Option<Threads>) -> Self {
            Self {
                config,
                threads,
                inner: Mutex::new(StateInner::default()),
            }
        }

        /// Stage 1 for a batch; returns ``(start, stop)``, the indices the
        /// documents were given (Arrow nulls get one and no signatures). On
        /// error nothing is added.
        #[pyo3(signature = (texts, /, *, threads = None))]
        fn add(
            &self,
            py: Python<'_>,
            texts: &Bound<'_, PyAny>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<(usize, usize)> {
            let (texts, threads) = texts_and_threads(texts, threads, self.threads)?;
            let t = self::threads(threads.or(self.threads));
            let kernel = &self.config.get().kernel;
            py.detach(|| {
                let hashes = map_indexed(&texts, t, |_, s| {
                    let mut v = Vec::new();
                    kernel.hashes(s, &mut v).map(|()| v)
                })?;
                // Numbered under the lock, so concurrent batches get disjoint ranges.
                let mut inner = lock(&self.inner);
                let start = inner.docs;
                let mut recs = to_records(&hashes, start)?;
                inner.sigs.append(&mut recs);
                inner.docs += hashes.len();
                inner.dups = None;
                Ok((start, inner.docs))
            })
            .map_err(|e: RunError| e.into_pyerr(py))
        }

        /// Replace the state with ``texts`` (stage 1 and 2), then run stage
        /// 3 on the same documents: ``(text, keep)`` in the caller's kind.
        #[pyo3(signature = (texts, /, *, threads = None))]
        fn fit<'py>(
            &self,
            py: Python<'py>,
            texts: &Bound<'py, PyAny>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<Bound<'py, PyTuple>> {
            let (texts, threads) = texts_and_threads(texts, threads, self.threads)?;
            let t = self::threads(threads.or(self.threads));
            let config = self.config.get();
            let kernel = &config.kernel;
            let dups = py
                .detach(|| {
                    let hashes = map_indexed(&texts, t, |_, s| {
                        let mut v = Vec::new();
                        kernel.hashes(s, &mut v).map(|()| v)
                    })?;
                    let mut sigs = to_records(&hashes, 0)?;
                    let dups = Arc::new(Duplicates::within(&mut sigs));
                    *lock(&self.inner) = StateInner {
                        sigs,
                        docs: hashes.len(),
                        dups: Some(Arc::clone(&dups)),
                    };
                    Ok(dups)
                })
                .map_err(|e: RunError| e.into_pyerr(py))?;
            let out = run_filter(py, &texts, config, &dups, 0, None, false, t)?;
            PyTuple::new(py, [out.get_item(0)?, out.get_item(1)?])
        }

        fn __len__(&self, py: Python<'_>) -> usize {
            py.detach(|| lock(&self.inner).docs)
        }

        /// Number of signatures stored.
        #[getter]
        fn num_signatures(&self, py: Python<'_>) -> usize {
            py.detach(|| lock(&self.inner).sigs.len())
        }

        fn reset(&self, py: Python<'_>) {
            py.detach(|| *lock(&self.inner) = StateInner::default());
        }

        /// Stage 2 within this task (one datatrove task, any number of
        /// finder workers).
        fn duplicates(&self, py: Python<'_>) -> SentenceDups {
            SentenceDups(py.detach(|| {
                let mut inner = lock(&self.inner);
                if let Some(d) = &inner.dups {
                    return Arc::clone(d);
                }
                let d = Arc::new(Duplicates::within(&mut inner.sigs));
                inner.dups = Some(Arc::clone(&d));
                d
            }))
        }

        /// The bytes of datatrove's ``{bucket:04d}/{rank:05d}.c4_sig`` files,
        /// for the buckets datatrove creates.
        fn signature_files<'py>(
            &self,
            py: Python<'py>,
            finder_workers: usize,
        ) -> PyResult<Bound<'py, PyList>> {
            if finder_workers == 0 {
                return Err(PyValueError::new_err("finder_workers must be >= 1"));
            }
            let p = precision_of(self.config.get());
            let files = py.detach(|| {
                let mut inner = lock(&self.inner);
                core::signature_files(&mut inner.sigs, p, finder_workers)
            });
            PyList::new(py, files.iter().map(|f| PyBytes::new(py, f)))
        }

        /// ``(hash, doc, window)`` of every signature, sorted (a test hook).
        fn signatures(&self, py: Python<'_>) -> Vec<(u64, u32, u16)> {
            py.detach(|| {
                let mut inner = lock(&self.inner);
                inner.sigs.sort_unstable();
                inner.sigs.iter().map(|s| (s.hash, s.doc, s.sent)).collect()
            })
        }
    }

    /// The Python str for a stage-3 text: the original object if unchanged.
    #[allow(clippy::ptr_arg)] // `Borrowed` (unchanged) vs `Owned` is the signal
    fn text_obj<'py>(
        py: Python<'py>,
        original: &PyBackedStr,
        out: &Cow<'_, str>,
    ) -> Bound<'py, PyAny> {
        match out {
            Cow::Borrowed(_) => original
                .into_pyobject(py)
                .unwrap_or_else(|e| match e {})
                .into_any(),
            Cow::Owned(s) => PyString::new(py, s).into_any(),
        }
    }

    /// Stage-3 texts in the caller's kind (Arrow nulls stay null).
    fn texts_like<'py>(
        py: Python<'py>,
        input: &TextInput,
        res: &[Option<Filtered<'_>>],
        t: par::Threads,
    ) -> PyResult<Bound<'py, PyAny>> {
        let texts = |docs: &[PyBackedStr]| {
            docs.iter()
                .zip(res)
                .map(|(d, r)| match r {
                    Some(f) => text_obj(py, d, &f.text),
                    None => PyNone::get(py).to_owned().into_any(),
                })
                .collect::<Vec<_>>()
        };
        Ok(match input {
            TextInput::Str(s) => texts(std::slice::from_ref(s)).remove(0),
            TextInput::List(d) => PyList::new(py, texts(d))?.into_any(),
            TextInput::Tuple(d) => PyTuple::new(py, texts(d))?.into_any(),
            TextInput::Arrow(strings) => {
                let large = strings.field.ty == StrType::LargeUtf8;
                let mut chunks = Vec::with_capacity(strings.chunks.len());
                let mut start = 0;
                for c in &strings.chunks {
                    let end = start + c.len();
                    let vals: Vec<Option<&str>> = res[start..end]
                        .iter()
                        .map(|r| r.as_ref().map(|f| &*f.text))
                        .collect();
                    if large {
                        chunks.push(Chunk::LargeUtf8(
                            StringBuffers::build(&vals, t).map_err(build_error)?,
                        ));
                    } else {
                        let built = StringBuffers::build_split(&vals, t).map_err(build_error)?;
                        chunks.extend(built.into_iter().map(Chunk::Utf8));
                    }
                    start = end;
                }
                let ty = if large {
                    ColumnType::LargeUtf8
                } else {
                    ColumnType::Utf8
                };
                Bound::new(py, Column::new(ty, &strings.field.name, chunks))?.into_any()
            }
        })
    }

    /// One reader per source: a local file (read in Rust) or bytes.
    enum Source {
        File(File),
        Bytes { data: PyBackedBytes, pos: usize },
    }

    impl Read for Source {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            match self {
                Self::File(f) => f.read(buf),
                Self::Bytes { data, pos } => {
                    let rest = &data[(*pos).min(data.len())..];
                    let n = rest.len().min(buf.len());
                    buf[..n].copy_from_slice(&rest[..n]);
                    *pos += n;
                    Ok(n)
                }
            }
        }
    }

    enum SourceSpec {
        Path(PathBuf),
        Bytes(PyBackedBytes),
    }

    /// `bytes` (contents) or a path (`str` / `os.PathLike`) per source.
    fn source_specs(objs: &Bound<'_, PyList>) -> PyResult<Vec<SourceSpec>> {
        objs.iter()
            .map(|o| {
                if let Ok(b) = o.extract::<PyBackedBytes>() {
                    Ok(SourceSpec::Bytes(b))
                } else {
                    o.extract::<PathBuf>().map(SourceSpec::Path)
                }
            })
            .collect()
    }

    fn open_all(specs: Vec<SourceSpec>, first: usize) -> Result<Vec<Source>, FileError> {
        specs
            .into_iter()
            .enumerate()
            .map(|(i, s)| match s {
                SourceSpec::Path(p) => {
                    File::open(p)
                        .map(Source::File)
                        .map_err(|err| FileError::Io {
                            file: first + i,
                            err,
                        })
                }
                SourceSpec::Bytes(data) => Ok(Source::Bytes { data, pos: 0 }),
            })
            .collect()
    }

    /// One task's duplicate windows; runs stage 3.
    #[pyclass(frozen, module = "pyroid.dedup", name = "_SentenceDups")]
    #[derive(Debug)]
    pub struct SentenceDups(Arc<Duplicates>);

    #[pymethods]
    impl SentenceDups {
        /// From the contents of datatrove's ``.c4_dup`` files of one rank.
        #[staticmethod]
        fn parse(py: Python<'_>, files: Vec<PyBackedBytes>) -> PyResult<Self> {
            let d = py
                .detach(|| {
                    let slices: Vec<&[u8]> = files.iter().map(|b| &**b).collect();
                    Duplicates::parse(&slices)
                })
                .map_err(|e| file_err(py, e))?;
            Ok(Self(Arc::new(d)))
        }

        /// Every ``(doc, window)`` pair, sorted.
        fn pairs(&self, py: Python<'_>) -> Vec<(u32, u16)> {
            py.detach(|| self.0.pairs().collect())
        }

        /// The documents in ``start <= doc < stop`` that have duplicates.
        fn docs_in(&self, start: usize, stop: usize) -> Vec<u32> {
            let docs = self.0.docs();
            let lo = docs.partition_point(|&d| (d as usize) < start);
            let hi = docs.partition_point(|&d| (d as usize) < stop);
            docs[lo..hi.max(lo)].to_vec()
        }

        fn __len__(&self) -> usize {
            self.0.len()
        }

        /// Stage 3 over ``texts`` (documents ``start``, ``start + 1``, ... of
        /// the task, or ``ids``). Returns ``(text, keep, formatted, stats)``:
        /// text and keep in the caller's kind; with ``details``, the
        /// exclusion-writer text (``None`` without duplicates) and
        /// ``(spans, kept)`` per document, else ``None`` twice.
        #[pyo3(signature = (texts, /, config, *, start = 0, ids = None, details = false, threads = None))]
        #[allow(clippy::too_many_arguments)]
        fn filter<'py>(
            &self,
            py: Python<'py>,
            texts: &Bound<'py, PyAny>,
            config: &SentenceConfig,
            start: usize,
            ids: Option<Vec<usize>>,
            details: bool,
            threads: Option<ThreadsArg>,
        ) -> PyResult<Bound<'py, PyTuple>> {
            let (texts, threads) = texts_and_threads(texts, threads, None)?;
            let t = self::threads(threads);
            run_filter(
                py,
                &texts,
                config,
                &self.0,
                start,
                ids.as_deref(),
                details,
                t,
            )
        }
    }

    /// One document's `(hash, window)` pairs per document as `.c4_sig`
    /// records, the first numbered `start`.
    fn to_records(
        hashes: &[Option<Vec<(u64, usize)>>],
        start: usize,
    ) -> Result<Vec<Signature>, RunError> {
        let mut recs = Vec::new();
        for (i, h) in hashes.iter().enumerate() {
            if let Some(h) = h {
                core::records(start + i, h, &mut recs)
                    .map_err(|err| RunError::Doc { index: i, err })?;
            }
        }
        Ok(recs)
    }

    /// Stage 3 over `texts`: `(text, keep, formatted, stats)` (see
    /// `_SentenceDups.filter`).
    #[allow(clippy::too_many_arguments)]
    fn run_filter<'py>(
        py: Python<'py>,
        texts: &TextInput,
        config: &SentenceConfig,
        dups: &Duplicates,
        start: usize,
        ids: Option<&[usize]>,
        details: bool,
        t: par::Threads,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if let Some(ids) = ids {
            if ids.len() != doc_count(texts) {
                return Err(PyValueError::new_err(format!(
                    "{} ids for {} documents",
                    ids.len(),
                    doc_count(texts)
                )));
            }
        }
        let kernel = &config.kernel;
        let res = py
            .detach(|| {
                map_indexed(texts, t, |i, s| {
                    let doc = ids.map_or(start + i, |ids| ids[i]);
                    kernel.filter(s, dups.of_doc(doc))
                })
            })
            .map_err(|e| e.into_pyerr(py))?;
        let text = texts_like(py, texts, &res, t)?;
        let keep: Vec<Option<bool>> = res.iter().map(|r| r.as_ref().map(|f| f.keep)).collect();
        let keep = mask_like(py, texts, &keep)?;
        let none = || PyNone::get(py).to_owned().into_any();
        let (formatted, stats) = if details {
            let formatted = PyList::new(
                py,
                res.iter().map(|r| {
                    r.as_ref()
                        .and_then(|f| f.formatted.as_deref())
                        .map_or_else(none, |s| PyString::new(py, s).into_any())
                }),
            )?;
            let stats = PyList::new(
                py,
                res.iter()
                    .map(|r| r.as_ref().map_or((0, 0), |f| (f.spans, f.kept))),
            )?;
            (formatted.into_any(), stats.into_any())
        } else {
            (none(), none())
        };
        PyTuple::new(py, [text, keep, formatted, stats])
    }

    /// datatrove's ``SentenceFindDedups`` for one finder task: the bucket's
    /// ``.c4_sig`` files and the index files (local paths or bytes), in
    /// datatrove's order. Returns each signature file's ``.c4_dup`` bytes.
    #[pyfunction]
    #[pyo3(signature = (sigs, index, /, *, precision, only_dedup_in_index))]
    fn _sentence_find_dups<'py>(
        py: Python<'py>,
        sigs: &Bound<'py, PyList>,
        index: &Bound<'py, PyList>,
        precision: u32,
        only_dedup_in_index: bool,
    ) -> PyResult<Bound<'py, PyList>> {
        let p = Precision::from_bits(precision)
            .ok_or_else(|| PyValueError::new_err("precision must be 32 or 64"))?;
        let (sigs, index) = (source_specs(sigs)?, source_specs(index)?);
        let n = sigs.len();
        let out = py
            .detach(|| {
                let sigs = open_all(sigs, 0)?;
                let index = open_all(index, n)?;
                core::find_duplicates(sigs, index, p, only_dedup_in_index)
            })
            .map_err(|e| file_err(py, e))?;
        PyList::new(py, out.iter().map(|b| PyBytes::new(py, b)))
    }

    /// datatrove's ``SentenceDedupBuildIndex``: the distinct hashes of the
    /// ``.c4_sig`` files (local paths or bytes), as index-file bytes.
    #[pyfunction]
    #[pyo3(signature = (sigs, /, *, precision))]
    fn _sentence_build_index<'py>(
        py: Python<'py>,
        sigs: &Bound<'py, PyList>,
        precision: u32,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let p = Precision::from_bits(precision)
            .ok_or_else(|| PyValueError::new_err("precision must be 32 or 64"))?;
        let sigs = source_specs(sigs)?;
        let out = py
            .detach(|| core::build_index(open_all(sigs, 0)?, p))
            .map_err(|e| file_err(py, e))?;
        Ok(PyBytes::new(py, &out))
    }

    /// datatrove's ``get_hashes`` per document, as ``(hash, window)`` pairs
    /// (no range checks); a parity-test hook.
    #[pyfunction]
    #[pyo3(signature = (texts, /, config, *, threads = None))]
    fn _sentence_hashes<'py>(
        py: Python<'py>,
        texts: &Bound<'py, PyAny>,
        config: &SentenceConfig,
        threads: Option<ThreadsArg>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (texts, threads) = texts_and_threads(texts, threads, None)?;
        let t = self::threads(threads);
        let kernel = &config.kernel;
        let hashes = py
            .detach(|| {
                map_indexed(&texts, t, |_, s| {
                    let mut v = Vec::new();
                    kernel.hashes(s, &mut v).map(|()| v)
                })
            })
            .map_err(|e| e.into_pyerr(py))?;
        let mut items = hashes
            .into_iter()
            .map(|h| match h {
                Some(v) => Ok(PyList::new(py, v)?.into_any()),
                None => Ok(PyNone::get(py).to_owned().into_any()),
            })
            .collect::<PyResult<Vec<_>>>()?;
        if let TextInput::Str(_) = texts {
            return Ok(items.remove(0));
        }
        Ok(PyList::new(py, items)?.into_any())
    }

    pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SentenceConfig>()?;
        m.add_class::<SentenceState>()?;
        m.add_class::<SentenceDups>()?;
        m.add_function(wrap_pyfunction!(_sentence_find_dups, m)?)?;
        m.add_function(wrap_pyfunction!(_sentence_build_index, m)?)?;
        m.add_function(wrap_pyfunction!(_sentence_hashes, m)?)?;
        Ok(())
    }
}
// ---- end SentenceDedup ----

// ---- MinHash files ----
/// datatrove's MinHash stages 1-3 on disk, with its intermediate files byte
/// for byte (`pyroid_core::dedup::files`): `_SigFileWriter` writes one
/// rank's `.minhash.sig` files, `_minhash_hash_range` and `_minhash_buckets`
/// run one bucket worker of stage 2, `_minhash_cluster` runs stage 3. The
/// `pyroid dedup` CLI drives them.
mod files {
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    use pyo3::exceptions::{PyOSError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use pyroid_core::dedup::TokenizeError;
    use pyroid_core::dedup::files::{
        self as core, BucketOptions, ClusterOptions, MinhashFileError, SigFileWriter,
    };

    use super::{NativeConfig, lock};
    use crate::batch::{BatchError, doc_count, fill_rows};
    use crate::errors::{PyroidError, with_note};
    use crate::input::{Threads, ThreadsArg, texts_and_threads, threads};

    const MIB: usize = 1 << 20;

    /// `OSError(errno, strerror, path)` (Python picks the subclass, e.g.
    /// `FileNotFoundError`); without an errno, the `io::Error`'s own class
    /// with the path in the message.
    fn io_error(py: Python<'_>, path: &Path, err: io::Error) -> PyErr {
        if let Some(code) = err.raw_os_error() {
            let built = (|| -> PyResult<PyErr> {
                let strerror = py.import("os")?.call_method1("strerror", (code,))?;
                let exc = py
                    .get_type::<PyOSError>()
                    .call1((code, strerror, path.as_os_str()))?;
                Ok(PyErr::from_value(exc))
            })();
            if let Ok(e) = built {
                return e;
            }
        }
        PyErr::from(io::Error::new(
            err.kind(),
            format!("{}: {err}", path.display()),
        ))
    }

    fn files_err(py: Python<'_>, e: MinhashFileError) -> PyErr {
        match e {
            MinhashFileError::Io { path, err } => io_error(py, &path, err),
            MinhashFileError::Malformed { .. } | MinhashFileError::Invalid(_) => {
                PyValueError::new_err(e.to_string())
            }
            MinhashFileError::Pool(e) => PyroidError::new_err(e.to_string()),
        }
    }

    /// Why `_SigFileWriter.add` failed (converted once re-attached).
    #[derive(Debug)]
    enum AddError {
        Batch(BatchError),
        /// spaCy's E025 on document `.0` of the batch.
        Tokenize(usize, TokenizeError),
        Files(MinhashFileError),
    }

    impl AddError {
        fn into_pyerr(self, py: Python<'_>) -> PyErr {
            match self {
                Self::Batch(e) => e.into_pyerr(py),
                Self::Tokenize(i, e) => with_note(
                    py,
                    PyValueError::new_err(e.to_string()),
                    format!("pyroid: while tokenizing document {i}"),
                ),
                Self::Files(e) => files_err(py, e),
            }
        }
    }

    #[derive(Debug)]
    struct WriterInner {
        /// `None` once finished or aborted.
        writer: Option<SigFileWriter>,
        documents: u64,
        closed: &'static str,
    }

    impl WriterInner {
        fn open(&mut self) -> Result<&mut SigFileWriter, MinhashFileError> {
            let closed = self.closed;
            self.writer
                .as_mut()
                .ok_or_else(|| MinhashFileError::Invalid(closed.to_owned()))
        }
    }

    /// Stage 1 for one rank: datatrove's ``MinhashDedupSignature`` writing
    /// ``folder/bucket_XXX/{rank:05d}.minhash.sig``.
    #[pyclass(frozen, module = "pyroid.dedup", name = "_SigFileWriter")]
    #[derive(Debug)]
    pub struct SigWriter {
        config: Py<NativeConfig>,
        threads: Option<Threads>,
        inner: Mutex<WriterInner>,
    }

    #[pymethods]
    impl SigWriter {
        #[new]
        #[pyo3(signature = (folder, rank, config, /, *, memory_budget = 256 * MIB, threads = None))]
        fn new(
            py: Python<'_>,
            folder: PathBuf,
            rank: u32,
            config: Py<NativeConfig>,
            memory_budget: usize,
            threads: Option<Threads>,
        ) -> PyResult<Self> {
            let cfg = *config.get().hasher.config();
            let writer = py
                .detach(|| SigFileWriter::new(folder, rank, &cfg, memory_budget))
                .map_err(|e| files_err(py, e))?;
            Ok(Self {
                config,
                threads,
                inner: Mutex::new(WriterInner {
                    writer: Some(writer),
                    documents: 0,
                    closed: "",
                }),
            })
        }

        /// Add documents (the rank's next ones, in order); returns
        /// ``(start, stop)``, the document indices they were given.
        #[pyo3(signature = (texts, /, *, threads = None))]
        fn add(
            &self,
            py: Python<'_>,
            texts: &Bound<'_, PyAny>,
            threads: Option<ThreadsArg>,
        ) -> PyResult<(u64, u64)> {
            let (texts, call_threads) = texts_and_threads(texts, threads, self.threads)?;
            let t = call_threads.or(self.threads);
            let hasher = &self.config.get().hasher;
            let h = hasher.num_hashes();
            py.detach(|| {
                let mut inner = lock(&self.inner);
                let writer = inner.open().map_err(AddError::Files)?;
                // The batch's signatures, computed in place, then appended.
                let mut rows = vec![0u64; doc_count(&texts) * h];
                let present = fill_rows(&texts, t, &mut rows, h, |s, row| hasher.signature(s, row))
                    .map_err(AddError::Batch)?
                    .into_iter()
                    .enumerate()
                    .map(|(i, r)| match r {
                        None => Ok(false), // Arrow null: no signature
                        Some(Ok(has)) => Ok(has),
                        Some(Err(e)) => Err(AddError::Tokenize(i, e)),
                    })
                    .collect::<Result<Vec<bool>, AddError>>()?;
                let range = writer
                    .append(&rows, &present, self::threads(t))
                    .map_err(AddError::Files)?;
                inner.documents = range.end;
                Ok((range.start, range.end))
            })
            .map_err(|e: AddError| e.into_pyerr(py))
        }

        /// Write every bucket's file; returns the number of documents added.
        fn finish(&self, py: Python<'_>) -> PyResult<u64> {
            let t = threads(self.threads);
            py.detach(|| {
                let mut inner = lock(&self.inner);
                let res = inner.open()?.finish(t);
                inner.writer = None;
                inner.closed = if res.is_ok() {
                    "finish() was already called"
                } else {
                    "finish() failed earlier"
                };
                res
            })
            .map_err(|e| files_err(py, e))
        }

        /// Remove the temporary files and write nothing; does nothing once
        /// finished or aborted.
        fn abort(&self, py: Python<'_>) {
            py.detach(|| {
                let mut inner = lock(&self.inner);
                if let Some(mut w) = inner.writer.take() {
                    w.abort();
                    inner.closed = "the writer was aborted";
                }
            });
        }

        /// Documents added so far.
        #[getter]
        fn documents(&self, py: Python<'_>) -> u64 {
            py.detach(|| lock(&self.inner).documents)
        }
    }

    /// datatrove's ``get_worker_hash_range``: ``(hash_min, hash_max)`` of
    /// one bucket worker, from the bucket's first signature file.
    #[pyfunction]
    #[pyo3(signature = (first_file, config, bucket_worker, workers_per_bucket, /))]
    fn _minhash_hash_range(
        py: Python<'_>,
        first_file: Option<PathBuf>,
        config: &NativeConfig,
        bucket_worker: usize,
        workers_per_bucket: usize,
    ) -> PyResult<(u64, u64)> {
        let cfg = config.hasher.config();
        py.detach(|| {
            core::worker_hash_range(
                first_file.as_deref(),
                cfg,
                bucket_worker,
                workers_per_bucket,
            )
        })
        .map_err(|e| files_err(py, e))
    }

    /// Stage 2 for one bucket worker: the pairs of ``files`` (one bucket's
    /// signature files, in datatrove's reader order) written to ``output``;
    /// returns the number of pairs.
    #[pyfunction]
    #[pyo3(signature = (
        files, output, config, /, *, hash_min = 0, hash_max = None,
        memory_budget = 64 * MIB, max_open = 512
    ))]
    #[allow(clippy::too_many_arguments)]
    fn _minhash_buckets(
        py: Python<'_>,
        files: Vec<PathBuf>,
        output: PathBuf,
        config: &NativeConfig,
        hash_min: u64,
        hash_max: Option<u64>,
        memory_budget: usize,
        max_open: usize,
    ) -> PyResult<u64> {
        let cfg = config.hasher.config();
        let defaults = BucketOptions::new(cfg);
        let opts = BucketOptions {
            hash_min,
            hash_max: hash_max.unwrap_or(defaults.hash_max),
            memory_budget,
            max_open,
        };
        py.detach(|| core::minhash_buckets(&files, &output, cfg, &opts))
            .map_err(|e| files_err(py, e))
    }

    /// Stage 3: the union-find over the pairs of ``dups`` (in the order
    /// given), then the ``.remove`` (and ``.clusters`` / ``.sizes``) files in
    /// ``output_folder``.
    #[pyfunction]
    #[pyo3(signature = (
        dups, output_folder, /, *, save_cluster_id = false, save_cluster_size = false,
        ignore_index_matches = false
    ))]
    #[allow(clippy::fn_params_excessive_bools)]
    fn _minhash_cluster(
        py: Python<'_>,
        dups: Vec<PathBuf>,
        output_folder: PathBuf,
        save_cluster_id: bool,
        save_cluster_size: bool,
        ignore_index_matches: bool,
    ) -> PyResult<Bound<'_, PyDict>> {
        let opts = ClusterOptions {
            save_cluster_id,
            save_cluster_size,
            ignore_index_matches,
        };
        let s = py
            .detach(|| core::minhash_cluster(&dups, &output_folder, opts))
            .map_err(|e| files_err(py, e))?;
        let out = PyDict::new(py);
        out.set_item("nodes", s.nodes)?;
        out.set_item("removed", s.removed)?;
        out.set_item("clusters", s.clusters)?;
        out.set_item("files", s.files)?;
        Ok(out)
    }

    pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<SigWriter>()?;
        m.add_function(wrap_pyfunction!(_minhash_hash_range, m)?)?;
        m.add_function(wrap_pyfunction!(_minhash_buckets, m)?)?;
        m.add_function(wrap_pyfunction!(_minhash_cluster, m)?)?;
        Ok(())
    }
}
// ---- end MinHash files ----
