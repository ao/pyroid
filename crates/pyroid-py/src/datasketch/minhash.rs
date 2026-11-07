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

/// The rows of a `(2, n)` array of `T` as `(a, b)`; `None` unless it is
/// exactly that (any strides: datasketch's legacy permutations are a transpose).
fn rows2<T: numpy::Element + Copy>(obj: &Bound<'_, PyAny>) -> Option<(Vec<T>, Vec<T>)> {
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
    ok.then(|| PyArray1::from_vec(py, hv).into_any())
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
    let hash = |t: Bound<'_, PyAny>| t.cast::<PyBytes>().ok().map(|b| bits.hash(b.as_bytes()));
    if let Ok(list) = tokens.cast::<PyList>() {
        list.iter().map(hash).collect()
    } else if let Ok(tuple) = tokens.cast::<PyTuple>() {
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
    let Ok(token) = token.cast::<PyBytes>() else {
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
    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = doc.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = doc.cast::<PyTuple>() {
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
        if let Ok(b) = item.cast::<PyBytes>() {
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

/// Signatures of every document of `docs`, starting from `initial`.
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
    let mut arena = Arena::default();
    let mut start = 0;
    while start < n {
        arena.clear();
        let mut end = start;
        while end < n && (end == start || arena.bytes.len() < BULK_CHUNK_BYTES) {
            let doc = docs.get_item(end)?;
            let first = arena.token_ends.len();
            if extract_doc(&mut arena, &doc, end, mode)? {
                arena.docs.push(Some((first, arena.token_ends.len())));
            } else {
                arena.docs.push(None);
                fallback.push(end);
            }
            end += 1;
        }
        let rows = &mut out[start * k..end * k];
        let arena = &arena;
        let result = py.detach(|| {
            par::try_map_rows(rows, k, threads, |i, row| {
                if let Some(tokens) = arena.tokens(i) {
                    // The permutations and rows have the scheme's width, and
                    // affine32 runs with 32-bit hashes (checked by the caller).
                    let _ = perm.update_tokens(T::signature(row), bits, tokens);
                }
                Ok::<(), BatchError>(())
            })
        });
        result.map_err(|e| e.into_pyerr(py))?;
        start = end;
    }
    Ok((out, fallback))
}

fn matrix<T: Value>(
    py: Python<'_>,
    rows: usize,
    k: usize,
    v: Vec<T>,
) -> PyResult<Bound<'_, PyAny>> {
    let arr =
        Array2::from_shape_vec((rows, k), v).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray2::from_owned_array(py, arr).into_any())
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
                PyArray1::from_vec(py, out).into_any()
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
                PyArray1::from_vec(py, out).into_any()
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
    Ok(())
}
