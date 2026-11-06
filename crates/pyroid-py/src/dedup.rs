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
            let tok =
                filters::tokenizer_for(language, norm.unicode).ok_or_else(
                    || match filters::backend(language) {
                        None => PyValueError::new_err(format!(
                            "Language '{language}' doesn't have a tokenizer assigned. Pass in a \
                         WordTokenizer directly or update tokenizer_assignment.csv"
                        )),
                        Some(b) => PyNotImplementedError::new_err(format!(
                            "pyroid has no native word tokenizer for language {language:?} \
                         (datatrove's {}({:?}))",
                            b.class, b.arg
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
    Ok(())
}

// ---- SentenceDedup ----
mod sentence {
    use pyo3::prelude::*;

    #[allow(clippy::unnecessary_wraps)] // placeholder
    pub(super) fn register(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        Ok(())
    }
}
// ---- end SentenceDedup ----
