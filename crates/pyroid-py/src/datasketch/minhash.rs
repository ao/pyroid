// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! MinHash entry points: permutation parameters, signature updates,
//! `MinHash.bulk` and `minhash_batch`.
//!
//! The Python class (`python/pyroid/datasketch/minhash.py`) decides when a
//! call can run natively. These functions return `None` instead of raising
//! when their input is not in the canonical form they handle (a dtype other
//! than the scheme's, a non-`bytes` token, ...): the caller then runs
//! datasketch's own numpy code, so results and errors stay datasketch's.

use std::cell::RefCell;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyList, PyString, PyTuple};
use pyroid_core::datasketch::minhash::{Permutations, Scheme, Sha1Bits, Signature};
use pyroid_core::datasketch::{rng, shingle};
use pyroid_core::par;

use crate::batch::{BatchError, fill_rows};
use crate::errors::{PyroidError, input_error, type_name, with_note};
use crate::input::{Threads, ThreadsArg, gil_enabled, texts_and_threads, threads};

/// Folding more than this many (token, permutation) pairs detaches from the
/// interpreter when the GIL is enabled, so other Python threads run meanwhile.
const DETACH_WORK: usize = 1 << 16;

/// Token bytes extracted per detached round of `bulk` (bounds the copy).
const BULK_CHUNK_BYTES: usize = 32 << 20;

/// Token bytes per pipelined round of `bulk` (copied while the pool hashes the
/// previous round).
const PIPELINE_ROUND_BYTES: usize = 4 << 20;

fn scheme_arg(name: &str) -> PyResult<Scheme> {
    Scheme::from_name(name).ok_or_else(|| PyValueError::new_err(format!("unknown scheme {name:?}")))
}

fn bits_arg(bits: u8) -> PyResult<Sha1Bits> {
    match bits {
        32 => Ok(Sha1Bits::B32),
        64 => Ok(Sha1Bits::B64),
        _ => Err(PyValueError::new_err(format!(
            "hash bits must be 32 or 64, got {bits}"
        ))),
    }
}

/// A signature value type: `u64` (legacy, affine64) or `u32` (affine32).
trait Value: numpy::Element + Copy + Send + Sync + 'static {
    fn signature(row: &mut [Self]) -> Signature<'_>;
}

impl Value for u64 {
    fn signature(row: &mut [Self]) -> Signature<'_> {
        Signature::U64(row)
    }
}

impl Value for u32 {
    fn signature(row: &mut [Self]) -> Signature<'_> {
        Signature::U32(row)
    }
}

/// Is `obj` exactly a `numpy.ndarray` (not a subclass such as a masked array,
/// whose own semantics datasketch's numpy code keeps)?
fn exact_ndarray(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_exact_instance_of::<numpy::PyUntypedArray>()
}

/// A new 1-D array that owns its data (as numpy's own results do).
fn owned1<'py, T: numpy::Element + Copy>(py: Python<'py>, v: &[T]) -> Bound<'py, PyAny> {
    PyArray1::from_slice(py, v).into_any()
}

/// The rows of a `(2, n)` array of `T` as `(a, b)`; `None` unless it is
/// exactly that (any strides: datasketch's legacy permutations are a transpose).
fn rows2<T: numpy::Element + Copy>(obj: &Bound<'_, PyAny>) -> Option<(Vec<T>, Vec<T>)> {
    if !exact_ndarray(obj) {
        return None;
    }
    let arr = obj.cast::<PyArray2<T>>().ok()?;
    let ro = arr.try_readonly().ok()?;
    let v = ro.as_array();
    if v.nrows() != 2 {
        return None;
    }
    Some((v.row(0).to_vec(), v.row(1).to_vec()))
}

/// `MinHash.permutations` in canonical form for `scheme`, or `None`.
fn read_permutations(scheme: Scheme, obj: &Bound<'_, PyAny>) -> Option<Permutations> {
    Some(match scheme {
        Scheme::Legacy => {
            let (a, b) = rows2::<u64>(obj)?;
            Permutations::Legacy { a, b }
        }
        Scheme::Affine64 => {
            let (a, b) = rows2::<u64>(obj)?;
            Permutations::Affine64 { a, b }
        }
        Scheme::Affine32 => {
            let (a, b) = rows2::<u32>(obj)?;
            Permutations::Affine32 { a, b }
        }
    })
}

/// A contiguous 1-D `T` array of length `n` as a Vec, or `None`.
fn read_values<T: numpy::Element + Copy>(obj: &Bound<'_, PyAny>, n: usize) -> Option<Vec<T>> {
    if !exact_ndarray(obj) {
        return None;
    }
    let arr = obj.cast::<PyArray1<T>>().ok()?;
    let ro = arr.try_readonly().ok()?;
    let s = ro.as_slice().ok()?;
    (s.len() == n).then(|| s.to_vec())
}

/// `np.random.RandomState(seed)` permutations as datasketch draws them.
///
/// Legacy: the `(num_perm, 2)` array of `(a, b)` pairs (datasketch's
/// `np.array([...]).T` is its transpose). Affine: the `(2, num_perm)` array.
#[pyfunction]
fn permutations<'py>(
    py: Python<'py>,
    scheme: &str,
    seed: u32,
    num_perm: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let scheme = scheme_arg(scheme)?;
    let p = rng::permutations(scheme, seed, num_perm);
    let shape_err = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    Ok(match p {
        Permutations::Legacy { a, b } => {
            let pairs: Vec<u64> = a.iter().zip(&b).flat_map(|(&x, &y)| [x, y]).collect();
            let arr = Array2::from_shape_vec((num_perm, 2), pairs).map_err(shape_err)?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        Permutations::Affine32 { a, b } => {
            let arr = Array2::from_shape_vec((2, num_perm), [a, b].concat()).map_err(shape_err)?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
        Permutations::Affine64 { a, b } => {
            let arr = Array2::from_shape_vec((2, num_perm), [a, b].concat()).map_err(shape_err)?;
            PyArray2::from_owned_array(py, arr).into_any()
        }
    })
}

/// Fold `hashes` into a copy of `hashvalues`; `None` if not canonical.
fn fold<'py, T: Value>(
    py: Python<'py>,
    perm: &Permutations,
    hashvalues: &Bound<'py, PyAny>,
    hashes: &[u64],
) -> Option<Bound<'py, PyAny>> {
    let mut hv = read_values::<T>(hashvalues, perm.len())?;
    let run = |hv: &mut Vec<T>| perm.update_hashes(T::signature(hv), hashes).is_ok();
    let ok = if hashes.len() * perm.len() > DETACH_WORK && gil_enabled(py).unwrap_or(true) {
        py.detach(|| run(&mut hv))
    } else {
        run(&mut hv)
    };
    ok.then(|| owned1(py, &hv))
}

fn fold_any<'py>(
    py: Python<'py>,
    scheme: Scheme,
    permutations: &Bound<'py, PyAny>,
    hashvalues: &Bound<'py, PyAny>,
    hashes: &[u64],
) -> Option<Bound<'py, PyAny>> {
    let perm = read_permutations(scheme, permutations)?;
    match scheme {
        Scheme::Legacy | Scheme::Affine64 => fold::<u64>(py, &perm, hashvalues, hashes),
        Scheme::Affine32 => fold::<u32>(py, &perm, hashvalues, hashes),
    }
}

/// The SHA1 hashes of `tokens` (a list or tuple of `bytes`), or `None`.
fn token_hashes(tokens: &Bound<'_, PyAny>, bits: Sha1Bits) -> Option<Vec<u64>> {
    let hash = |t: Bound<'_, PyAny>| {
        t.cast_exact::<PyBytes>()
            .ok()
            .map(|b| bits.hash(b.as_bytes()))
    };
    if let Ok(list) = tokens.cast_exact::<PyList>() {
        list.iter().map(hash).collect()
    } else if let Ok(tuple) = tokens.cast_exact::<PyTuple>() {
        tuple.iter().map(hash).collect()
    } else {
        None
    }
}

/// `MinHash.update(token)` with a SHA1 hash: the new hash values, or `None`.
#[pyfunction]
fn update_one<'py>(
    py: Python<'py>,
    scheme: &str,
    hashvalues: &Bound<'py, PyAny>,
    permutations: &Bound<'py, PyAny>,
    token: &Bound<'py, PyAny>,
    bits: u8,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let (scheme, bits) = (scheme_arg(scheme)?, bits_arg(bits)?);
    let Ok(token) = token.cast_exact::<PyBytes>() else {
        return Ok(None);
    };
    let h = bits.hash(token.as_bytes());
    Ok(fold_any(py, scheme, permutations, hashvalues, &[h]))
}

/// `MinHash.update_batch(tokens)` with a SHA1 hash: the new hash values, or
/// `None`. `tokens` is a non-empty list or tuple.
#[pyfunction]
fn update_tokens<'py>(
    py: Python<'py>,
    scheme: &str,
    hashvalues: &Bound<'py, PyAny>,
    permutations: &Bound<'py, PyAny>,
    tokens: &Bound<'py, PyAny>,
    bits: u8,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let (scheme, bits) = (scheme_arg(scheme)?, bits_arg(bits)?);
    let Some(hashes) = token_hashes(tokens, bits) else {
        return Ok(None);
    };
    Ok(fold_any(py, scheme, permutations, hashvalues, &hashes))
}

/// Fold hash values returned by a user's hash function: the new hash
/// values, or `None` unless every value is an `int` the scheme takes as is
/// (`0 <= h < 2**64`; `< 2**32` for affine32).
#[pyfunction]
fn update_hashes<'py>(
    py: Python<'py>,
    scheme: &str,
    hashvalues: &Bound<'py, PyAny>,
    permutations: &Bound<'py, PyAny>,
    hashes: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let scheme = scheme_arg(scheme)?;
    let mut hs = Vec::with_capacity(hashes.len());
    for h in hashes.iter() {
        if !h.is_exact_instance_of::<PyInt>() {
            return Ok(None);
        }
        let Ok(v) = h.extract::<u64>() else {
            return Ok(None);
        };
        if scheme == Scheme::Affine32 && v > u64::from(u32::MAX) {
            return Ok(None);
        }
        hs.push(v);
    }
    Ok(fold_any(py, scheme, permutations, hashvalues, &hs))
}

/// How `bulk` treats tokens that are not `bytes`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tokens {
    /// `MinHash.bulk`: the document goes back to the caller (datasketch's path).
    BytesOnly,
    /// `minhash_batch`: `str` tokens are hashed as UTF-8, anything else raises.
    BytesOrStr,
}

/// Token bytes of a run of documents, copied so they can be read detached.
#[derive(Default)]
struct Arena {
    bytes: Vec<u8>,
    /// End offset of every token in `bytes`.
    token_ends: Vec<usize>,
    /// Per document of the round: `Some(token range in token_ends)`, or
    /// `None` for a document the caller handles.
    docs: Vec<Option<(usize, usize)>>,
}

impl Arena {
    fn clear(&mut self) {
        self.bytes.clear();
        self.token_ends.clear();
        self.docs.clear();
    }

    fn tokens(&self, doc: usize) -> Option<impl Iterator<Item = &[u8]>> {
        let (start, end) = self.docs[doc]?;
        Some((start..end).map(move |t| {
            let s = if t == 0 { 0 } else { self.token_ends[t - 1] };
            &self.bytes[s..self.token_ends[t]]
        }))
    }
}

fn surrogate_note(py: Python<'_>, err: PyErr, doc: usize, token: usize) -> PyErr {
    if !err.is_instance_of::<PyUnicodeEncodeError>(py) {
        return err;
    }
    with_note(
        py,
        err,
        format!(
            "pyroid: token {token} of document {doc} contains a lone surrogate \
             (U+D800..U+DFFF), which has no UTF-8 encoding"
        ),
    )
}

/// Copy document `i`'s tokens into `arena`. `Ok(false)`: the caller handles it.
fn extract_doc(
    arena: &mut Arena,
    doc: &Bound<'_, PyAny>,
    i: usize,
    mode: Tokens,
) -> PyResult<bool> {
    let py = doc.py();
    // MinHash.bulk: only exact lists and tuples (a subclass may iterate differently).
    let strict = mode == Tokens::BytesOnly;
    let list = if strict {
        doc.cast_exact::<PyList>().ok()
    } else {
        doc.cast::<PyList>().ok()
    };
    let tuple = if strict {
        doc.cast_exact::<PyTuple>().ok()
    } else {
        doc.cast::<PyTuple>().ok()
    };
    let items: Vec<Bound<'_, PyAny>> = if let Some(list) = list {
        list.iter().collect()
    } else if let Some(tuple) = tuple {
        tuple.iter().collect()
    } else if mode == Tokens::BytesOnly {
        return Ok(false);
    } else {
        return Err(input_error(
            py,
            format!(
                "expected a list or tuple of tokens at index {i}, got {}",
                type_name(doc)
            ),
        ));
    };
    let (bytes_len, tokens_len) = (arena.bytes.len(), arena.token_ends.len());
    for (j, item) in items.iter().enumerate() {
        let b = if strict {
            item.cast_exact::<PyBytes>().ok()
        } else {
            item.cast::<PyBytes>().ok()
        };
        if let Some(b) = b {
            arena.bytes.extend_from_slice(b.as_bytes());
        } else if mode == Tokens::BytesOnly {
            arena.bytes.truncate(bytes_len);
            arena.token_ends.truncate(tokens_len);
            return Ok(false);
        } else if let Ok(s) = item.cast::<PyString>() {
            let s = s.to_str().map_err(|e| surrogate_note(py, e, i, j))?;
            arena.bytes.extend_from_slice(s.as_bytes());
        } else {
            return Err(input_error(
                py,
                format!(
                    "expected bytes or str at token {j} of document {i}, got {}",
                    type_name(item)
                ),
            ));
        }
        arena.token_ends.push(arena.bytes.len());
    }
    Ok(true)
}

/// Copy documents `start..` into `arena` until it holds about `budget`
/// bytes; returns the end index. Documents the caller handles go to
/// `fallback`.
fn extract_round(
    arena: &mut Arena,
    docs: &Bound<'_, PyList>,
    start: usize,
    budget: usize,
    mode: Tokens,
    fallback: &mut Vec<usize>,
) -> PyResult<usize> {
    arena.clear();
    let n = docs.len();
    let mut end = start;
    while end < n && (end == start || arena.bytes.len() < budget) {
        let doc = docs.get_item(end)?;
        let first = arena.token_ends.len();
        if extract_doc(arena, &doc, end, mode)? {
            arena.docs.push(Some((first, arena.token_ends.len())));
        } else {
            arena.docs.push(None);
            fallback.push(end);
        }
        end += 1;
    }
    Ok(end)
}

/// Hash one round's documents into their rows (pure Rust; runs detached).
fn hash_round<T: Value>(
    perm: &Permutations,
    arena: &Arena,
    rows: &mut [T],
    bits: Sha1Bits,
    threads: par::Threads,
) -> Result<(), BatchError> {
    par::try_map_rows(rows, perm.len(), threads, |i, row| {
        if let Some(tokens) = arena.tokens(i) {
            // The permutations and rows have the scheme's width, and
            // affine32 runs with 32-bit hashes (checked by the caller).
            let _ = perm.update_tokens(T::signature(row), bits, tokens);
        }
        Ok::<(), BatchError>(())
    })?;
    Ok(())
}

/// Signatures of every document of `docs`, starting from `initial`.
///
/// Copying tokens out of the Python objects needs the interpreter and runs
/// on the calling thread. With more than one thread allowed, rounds are
/// pipelined: a helper thread waits for the pool to hash round `i` while the
/// calling thread copies round `i + 1`; the calling thread only waits for the
/// pool while detached.
fn bulk_rows<T: Value>(
    py: Python<'_>,
    perm: &Permutations,
    initial: &[T],
    docs: &Bound<'_, PyList>,
    bits: Sha1Bits,
    mode: Tokens,
    threads: par::Threads,
) -> PyResult<(Vec<T>, Vec<usize>)> {
    let k = initial.len();
    let n = docs.len();
    let mut out: Vec<T> = Vec::with_capacity(n * k);
    for _ in 0..n {
        out.extend_from_slice(initial);
    }
    let mut fallback = Vec::new();
    if threads == par::Threads::ONE {
        let mut arena = Arena::default();
        let mut start = 0;
        while start < n {
            let end = extract_round(
                &mut arena,
                docs,
                start,
                BULK_CHUNK_BYTES,
                mode,
                &mut fallback,
            )?;
            let rows = &mut out[start * k..end * k];
            py.detach(|| hash_round(perm, &arena, rows, bits, threads))
                .map_err(|e| e.into_pyerr(py))?;
            start = end;
        }
        return Ok((out, fallback));
    }
    std::thread::scope(|s| -> PyResult<()> {
        let mut rest: &mut [T] = &mut out;
        let mut spare = Arena::default();
        let mut pending: Option<
            std::thread::ScopedJoinHandle<'_, (Arena, Result<(), BatchError>)>,
        > = None;
        let mut start = 0;
        let finish = |pending: &mut Option<std::thread::ScopedJoinHandle<'_, _>>| -> PyResult<Option<Arena>> {
            let Some(h) = pending.take() else {
                return Ok(None);
            };
            let joined: std::thread::Result<(Arena, Result<(), BatchError>)> = py.detach(|| h.join());
            let (arena, res) = joined.unwrap_or_else(|p| std::panic::resume_unwind(p));
            res.map_err(|e| e.into_pyerr(py))?;
            Ok(Some(arena))
        };
        while start < n {
            let mut arena = std::mem::take(&mut spare);
            let end = match extract_round(
                &mut arena,
                docs,
                start,
                PIPELINE_ROUND_BYTES,
                mode,
                &mut fallback,
            ) {
                Ok(end) => end,
                Err(e) => {
                    finish(&mut pending)?;
                    return Err(e);
                }
            };
            if let Some(done) = finish(&mut pending)? {
                spare = done;
            }
            let (rows, tail) = std::mem::take(&mut rest).split_at_mut((end - start) * k);
            rest = tail;
            pending = Some(s.spawn(move || {
                let res = hash_round(perm, &arena, rows, bits, threads);
                (arena, res)
            }));
            start = end;
        }
        finish(&mut pending)?;
        Ok(())
    })?;
    Ok((out, fallback))
}

fn matrix<T: Value>(
    py: Python<'_>,
    rows: usize,
    k: usize,
    v: Vec<T>,
) -> PyResult<Bound<'_, PyAny>> {
    // numpy allocates it, so the result owns its data like numpy's own arrays.
    debug_assert_eq!(v.len(), rows * k);
    let arr = PyArray2::<T>::zeros(py, [rows, k], false);
    {
        let internal =
            |e: &dyn std::fmt::Display| PyroidError::new_err(format!("pyroid.datasketch: {e}"));
        let mut rw = arr.try_readwrite().map_err(|e| internal(&e))?;
        rw.as_slice_mut()
            .map_err(|e| internal(&e))?
            .copy_from_slice(&v);
    }
    Ok(arr.into_any())
}

#[allow(clippy::too_many_arguments)]
fn bulk_any<'py>(
    py: Python<'py>,
    scheme: Scheme,
    initial: &Bound<'py, PyAny>,
    permutations: &Bound<'py, PyAny>,
    docs: &Bound<'py, PyList>,
    bits: Sha1Bits,
    mode: Tokens,
    threads: par::Threads,
) -> PyResult<Option<(Bound<'py, PyAny>, Vec<usize>)>> {
    let Some(perm) = read_permutations(scheme, permutations) else {
        return Ok(None);
    };
    if scheme == Scheme::Affine32 && bits == Sha1Bits::B64 {
        return Ok(None);
    }
    let k = perm.len();
    Ok(match scheme {
        Scheme::Legacy | Scheme::Affine64 => {
            let Some(init) = read_values::<u64>(initial, k) else {
                return Ok(None);
            };
            let (rows, fb) = bulk_rows(py, &perm, &init, docs, bits, mode, threads)?;
            Some((matrix(py, docs.len(), k, rows)?, fb))
        }
        Scheme::Affine32 => {
            let Some(init) = read_values::<u32>(initial, k) else {
                return Ok(None);
            };
            let (rows, fb) = bulk_rows(py, &perm, &init, docs, bits, mode, threads)?;
            Some((matrix(py, docs.len(), k, rows)?, fb))
        }
    })
}

/// `MinHash.bulk`: one row of hash values per document, starting from
/// `initial`, plus the indices of documents the caller must compute itself
/// (not a list/tuple of `bytes`). `None` if the state is not canonical.
#[pyfunction]
#[pyo3(signature = (scheme, initial, permutations, docs, bits, *, threads = None))]
fn bulk<'py>(
    py: Python<'py>,
    scheme: &str,
    initial: &Bound<'py, PyAny>,
    permutations: &Bound<'py, PyAny>,
    docs: &Bound<'py, PyList>,
    bits: u8,
    threads: Option<Threads>,
) -> PyResult<Option<(Bound<'py, PyAny>, Vec<usize>)>> {
    let (scheme, bits) = (scheme_arg(scheme)?, bits_arg(bits)?);
    bulk_any(
        py,
        scheme,
        initial,
        permutations,
        docs,
        bits,
        Tokens::BytesOnly,
        self::threads(threads),
    )
}

/// `minhash_batch` over token lists (`bytes` or `str` tokens).
#[pyfunction]
#[pyo3(signature = (scheme, initial, permutations, docs, bits, *, threads = None))]
fn batch_tokens<'py>(
    py: Python<'py>,
    scheme: &str,
    initial: &Bound<'py, PyAny>,
    permutations: &Bound<'py, PyAny>,
    docs: &Bound<'py, PyList>,
    bits: u8,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyAny>> {
    let (scheme, bits) = (scheme_arg(scheme)?, bits_arg(bits)?);
    let res = bulk_any(
        py,
        scheme,
        initial,
        permutations,
        docs,
        bits,
        Tokens::BytesOrStr,
        self::threads(threads),
    )?;
    res.map(|(m, _)| m).ok_or_else(|| {
        PyroidError::new_err("pyroid.datasketch: internal permutation state is not canonical")
    })
}

thread_local! {
    static SHINGLES: RefCell<(shingle::Scratch, Vec<u64>)> = RefCell::default();
}

fn text_rows<T: Value>(
    perm: &Permutations,
    input: &crate::input::TextInput,
    threads_arg: Option<Threads>,
    ngram: usize,
    bits: Sha1Bits,
    out: &mut [T],
) -> Result<(), BatchError> {
    let k = perm.len();
    fill_rows(input, threads_arg, out, k, |text, row| {
        SHINGLES.with_borrow_mut(|(scratch, hashes)| {
            hashes.clear();
            shingle::for_each_word_ngram(text, ngram, scratch, |s| hashes.push(bits.hash(s)));
            let _ = perm.update_hashes(T::signature(row), hashes);
        });
    })?;
    Ok(())
}

/// `minhash_batch` over texts: word `ngram`-gram shingles, one row per text
/// (a 1-D array for a single `str`). Arrow nulls give empty signatures.
#[pyfunction]
#[pyo3(signature = (scheme, permutations, texts, ngram, *, threads = None))]
fn batch_texts<'py>(
    py: Python<'py>,
    scheme: &str,
    permutations: &Bound<'py, PyAny>,
    texts: &Bound<'py, PyAny>,
    ngram: usize,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let scheme = scheme_arg(scheme)?;
    if ngram == 0 {
        return Err(PyValueError::new_err("ngram must be a positive int, got 0"));
    }
    let Some(perm) = read_permutations(scheme, permutations) else {
        return Err(PyroidError::new_err(
            "pyroid.datasketch: internal permutation state is not canonical",
        ));
    };
    let (input, threads_arg) = texts_and_threads(texts, threads, None)?;
    let n = crate::batch::doc_count(&input);
    let k = perm.len();
    let single = matches!(input, crate::input::TextInput::Str(_));
    let bits = if scheme == Scheme::Affine64 {
        Sha1Bits::B64
    } else {
        Sha1Bits::B32
    };
    let max = scheme.max_value();
    let arr = match scheme {
        Scheme::Legacy | Scheme::Affine64 => {
            let mut out = vec![max; n * k];
            py.detach(|| text_rows(&perm, &input, threads_arg, ngram, bits, &mut out))
                .map_err(|e| e.into_pyerr(py))?;
            if single {
                owned1(py, &out)
            } else {
                matrix(py, n, k, out)?
            }
        }
        Scheme::Affine32 => {
            #[allow(clippy::cast_possible_truncation)]
            let mut out = vec![max as u32; n * k];
            py.detach(|| text_rows(&perm, &input, threads_arg, ngram, bits, &mut out))
                .map_err(|e| e.into_pyerr(py))?;
            if single {
                owned1(py, &out)
            } else {
                matrix(py, n, k, out)?
            }
        }
    };
    Ok(arr)
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(permutations, m)?)?;
    m.add_function(wrap_pyfunction!(update_one, m)?)?;
    m.add_function(wrap_pyfunction!(update_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(update_hashes, m)?)?;
    m.add_function(wrap_pyfunction!(bulk, m)?)?;
    m.add_function(wrap_pyfunction!(batch_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(batch_texts, m)?)?;
    m.add_function(wrap_pyfunction!(lean_pack, m)?)?;
    m.add_function(wrap_pyfunction!(lean_unpack, m)?)?;
    Ok(())
}

/// A `struct` byte order: native (with `@`'s alignment) or a standard one.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ByteOrder {
    /// `@`: native order, native alignment.
    NativeAligned,
    /// `=`, `<`, `>`, `!`: standard sizes, no alignment.
    Standard { big: bool },
}

fn byte_order(s: &str) -> Option<ByteOrder> {
    Some(match s {
        "@" => ByteOrder::NativeAligned,
        "=" => ByteOrder::Standard {
            big: cfg!(target_endian = "big"),
        },
        "<" => ByteOrder::Standard { big: false },
        ">" | "!" => ByteOrder::Standard { big: true },
        _ => return None,
    })
}

impl ByteOrder {
    fn big(self) -> bool {
        match self {
            Self::NativeAligned => cfg!(target_endian = "big"),
            Self::Standard { big } => big,
        }
    }

    fn put(self, out: &mut Vec<u8>, le: &[u8]) {
        if self.big() {
            out.extend(le.iter().rev());
        } else {
            out.extend_from_slice(le);
        }
    }

    /// Offset of the first hash value: after `q`, `i` (and `B` for the
    /// affine formats), aligned to the value size for `@`.
    fn values_offset(self, affine: bool, value_size: usize) -> usize {
        let header: usize = if affine { 13 } else { 12 };
        match self {
            Self::NativeAligned => header.next_multiple_of(value_size),
            Self::Standard { .. } => header,
        }
    }
}

/// A contiguous 1-D `T` array's values, or `None`.
fn contiguous<T: numpy::Element + Copy>(obj: &Bound<'_, PyAny>) -> Option<Vec<T>> {
    if !exact_ndarray(obj) {
        return None;
    }
    let arr = obj.cast::<PyArray1<T>>().ok()?;
    let ro = arr.try_readonly().ok()?;
    ro.as_slice().ok().map(<[T]>::to_vec)
}

/// `LeanMinHash.serialize`'s bytes (datasketch's `struct.pack_into` layout),
/// or `None` if `struct` would raise or the state is not canonical (the caller
/// then runs the `struct` code itself).
#[pyfunction]
fn lean_pack<'py>(
    py: Python<'py>,
    scheme: &str,
    seed: i64,
    hashvalues: &Bound<'py, PyAny>,
    byteorder: &str,
) -> Option<Bound<'py, PyBytes>> {
    let (Some(scheme), Some(order)) = (Scheme::from_name(scheme), byte_order(byteorder)) else {
        return None;
    };
    // Values as little-endian bytes of the serialized width.
    let (values, size): (Vec<u8>, usize) = match scheme {
        Scheme::Legacy => {
            let s = contiguous::<u64>(hashvalues)?;
            let mut v = Vec::with_capacity(4 * s.len());
            for x in s {
                // struct.error ('I' format requires 0 <= number <= 4294967295): `struct` reports it.
                v.extend_from_slice(&u32::try_from(x).ok()?.to_le_bytes());
            }
            (v, 4)
        }
        Scheme::Affine32 => {
            let s = contiguous::<u32>(hashvalues)?;
            (s.iter().flat_map(|x| x.to_le_bytes()).collect(), 4)
        }
        Scheme::Affine64 => {
            let s = contiguous::<u64>(hashvalues)?;
            (s.iter().flat_map(|x| x.to_le_bytes()).collect(), 8)
        }
    };
    let n = values.len() / size;
    let count = i32::try_from(n).ok()?;
    let affine = scheme != Scheme::Legacy;
    let offset = order.values_offset(affine, size);
    let mut out = Vec::with_capacity(offset + values.len());
    order.put(&mut out, &seed.to_le_bytes());
    order.put(
        &mut out,
        &(if affine { -count } else { count }).to_le_bytes(),
    );
    if affine {
        out.push(if scheme == Scheme::Affine32 { 1 } else { 2 });
    }
    out.resize(offset, 0);
    for v in values.chunks_exact(size) {
        order.put(&mut out, v);
    }
    Some(PyBytes::new(py, &out))
}

/// `LeanMinHash.deserialize`: `(seed, scheme, hashvalues)` read from `buf`,
/// or `None` where datasketch's `struct` code would raise (too short, unknown
/// scheme code): the caller then runs it to raise the same error.
#[pyfunction]
fn lean_unpack<'py>(
    py: Python<'py>,
    buf: &Bound<'py, PyAny>,
    byteorder: &str,
) -> Option<(i64, &'static str, Bound<'py, PyAny>)> {
    let order = byte_order(byteorder)?;
    let owned;
    let data: &[u8] = if let Ok(b) = buf.cast::<PyBytes>() {
        b.as_bytes()
    } else if let Ok(b) = buf.cast::<pyo3::types::PyByteArray>() {
        owned = b.to_vec();
        &owned
    } else {
        return None;
    };
    let get = |at: usize, len: usize| -> Option<Vec<u8>> {
        let mut v = data.get(at..at + len)?.to_vec();
        if order.big() {
            v.reverse();
        }
        Some(v)
    };
    let (Some(seed), Some(count)) = (get(0, 8), get(8, 4)) else {
        return None;
    };
    let seed = i64::from_le_bytes(seed.try_into().unwrap_or_default());
    let count = i32::from_le_bytes(count.try_into().unwrap_or_default());
    let (scheme, n) = if count >= 0 {
        (Scheme::Legacy, count.unsigned_abs() as usize)
    } else {
        match data.get(12) {
            Some(1) => (Scheme::Affine32, count.unsigned_abs() as usize),
            Some(2) => (Scheme::Affine64, count.unsigned_abs() as usize),
            _ => return None,
        }
    };
    let size = if scheme == Scheme::Affine64 { 8 } else { 4 };
    let offset = order.values_offset(scheme != Scheme::Legacy, size);
    let end = n.checked_mul(size).and_then(|b| b.checked_add(offset))?;
    let raw = data.get(offset..end)?;
    let big = order.big();
    let arr = match scheme {
        Scheme::Legacy | Scheme::Affine32 => {
            let vals = raw.chunks_exact(4).map(|c| {
                let b = [c[0], c[1], c[2], c[3]];
                if big {
                    u32::from_be_bytes(b)
                } else {
                    u32::from_le_bytes(b)
                }
            });
            if scheme == Scheme::Legacy {
                PyArray1::from_iter(py, vals.map(u64::from)).into_any()
            } else {
                PyArray1::from_iter(py, vals).into_any()
            }
        }
        Scheme::Affine64 => {
            let vals = raw.chunks_exact(8).map(|c| {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                if big {
                    u64::from_be_bytes(b)
                } else {
                    u64::from_le_bytes(b)
                }
            });
            PyArray1::from_iter(py, vals).into_any()
        }
    };
    Some((seed, scheme.name(), arr))
}
