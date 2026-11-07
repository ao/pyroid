// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! `_LSHNative` and `_ForestNative`: the indexes behind `pyroid.datasketch`'s
//! `MinHashLSH` and `MinHashLSHForest` (`python/pyroid/datasketch/lsh.py`,
//! `lshforest.py`), which keep datasketch's attributes and checks.
//!
//! Each index is a core [`LshIndex`] / [`ForestIndex`] over `u32` ids, the
//! key objects (`objs[id]`) and a `dict` from key to id (so keys compare with
//! Python equality, as datasketch's dicts and sets do), all behind one `Mutex`:
//! - it is taken with `lock_py_attached`: a thread that has to wait for it
//!   waits detached, so it never holds up the interpreter (no lock-order
//!   deadlock with the GIL or a stop-the-world pause);
//! - Python code can run while it is held (a key's `__hash__`/`__eq__`); a
//!   re-entrant call from such code raises `RuntimeError` instead of
//!   deadlocking;
//! - batch work runs detached on the shared pool, on data copied out of
//!   Python first.
//!
//! Band keys are computed here when `hashvalues` is a C-contiguous
//! `numpy.ndarray` of an integer or float dtype in native byte order (what
//! `MinHash` and `WeightedMinHash` hold). Anything else goes through the
//! Python callback `slow(minhash, n)`: datasketch's own expression
//! `[self._H(minhash.hashvalues[s:e]) for s, e in self.hashranges[:n]]`, so
//! its results and errors are datasketch's.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard, PoisonError};

use numpy::{
    PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::sync::{MutexExt, PyOnceLock};
use pyo3::types::{PyBytes, PyDict, PyList, PySet, PyType};
use pyroid_core::datasketch::forest::ForestIndex;
use pyroid_core::datasketch::lsh::{
    BandKey, Bands, CapacityError, LshIndex, RemoveError, band_keys, band_keys_many, swapped_rows,
};
use pyroid_core::par;

use crate::errors::{PyroidError, type_name, with_note};
use crate::input::{Threads, threads};

thread_local! {
    /// The index locks this thread holds (their addresses).
    static HELD: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// A held index lock.
struct Locked<'a, T> {
    guard: MutexGuard<'a, T>,
    addr: usize,
}

impl<T> Deref for Locked<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for Locked<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for Locked<'_, T> {
    fn drop(&mut self) {
        HELD.with_borrow_mut(|held| {
            if let Some(i) = held.iter().rposition(|&a| a == self.addr) {
                held.swap_remove(i);
            }
        });
    }
}

/// Take `m`, waiting detached if another thread holds it.
fn lock<'a, T>(py: Python<'_>, m: &'a Mutex<T>) -> PyResult<Locked<'a, T>> {
    let addr = std::ptr::from_ref(m).addr();
    if HELD.with_borrow(|held| held.contains(&addr)) {
        return Err(PyRuntimeError::new_err(
            "pyroid.datasketch: the index was called back into while it was busy (from a \
             key's __hash__ or __eq__, or a hashvalues attribute)",
        ));
    }
    // A panic mid-update can leave a partial update behind, never unsound state.
    let guard = m
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    HELD.with_borrow_mut(|held| held.push(addr));
    Ok(Locked { guard, addr })
}

fn capacity_error(e: CapacityError) -> PyErr {
    PyroidError::new_err(format!("pyroid.datasketch: {e}"))
}

fn pool_error(e: par::PoolError) -> PyErr {
    PyroidError::new_err(e.to_string())
}

fn corrupt() -> PyErr {
    PyRuntimeError::new_err(
        "pyroid.datasketch: inconsistent index state (an earlier call panicked)",
    )
}

/// Key objects by id, and the `dict` from key to id.
#[derive(Debug)]
struct KeyMap {
    objs: Vec<Option<Py<PyAny>>>,
    ids: Py<PyDict>,
}

impl KeyMap {
    fn new(py: Python<'_>) -> Self {
        Self {
            objs: Vec::new(),
            ids: PyDict::new(py).unbind(),
        }
    }

    /// The id of `key` (hashing it: `TypeError` if it is unhashable).
    fn lookup(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Option<u32>> {
        match self.ids.bind(py).get_item(key)? {
            None => Ok(None),
            Some(id) => id.extract().map(Some),
        }
    }

    fn obj<'py>(&self, py: Python<'py>, id: u32) -> PyResult<Bound<'py, PyAny>> {
        match self.objs.get(id as usize) {
            Some(Some(obj)) => Ok(obj.bind(py).clone()),
            _ => Err(corrupt()),
        }
    }

    fn bind(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>, id: u32) -> PyResult<()> {
        self.ids.bind(py).set_item(key, id)?;
        let i = id as usize;
        if self.objs.len() <= i {
            self.objs.resize_with(i + 1, || None);
        }
        self.objs[i] = Some(key.clone().unbind());
        Ok(())
    }

    fn forget(&mut self, py: Python<'_>, id: u32) -> PyResult<()> {
        if let Some(obj) = self.objs.get_mut(id as usize).and_then(Option::take) {
            self.ids.bind(py).del_item(obj.bind(py))?;
        }
        Ok(())
    }

    fn traverse(&self, visit: &pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        for obj in self.objs.iter().flatten() {
            visit.call(obj)?;
        }
        visit.call(&self.ids)
    }
}

/// What both cores provide for id bookkeeping.
trait Core {
    fn alloc(&mut self) -> Result<u32, CapacityError>;
    fn release(&mut self, id: u32) -> bool;
    fn contains(&self, id: u32) -> bool;
    fn ids_in_order(&self) -> Vec<u32>;
}

impl Core for LshIndex {
    fn alloc(&mut self) -> Result<u32, CapacityError> {
        LshIndex::alloc(self)
    }
    fn release(&mut self, id: u32) -> bool {
        LshIndex::release(self, id)
    }
    fn contains(&self, id: u32) -> bool {
        LshIndex::contains(self, id)
    }
    fn ids_in_order(&self) -> Vec<u32> {
        self.ids().collect()
    }
}

impl Core for ForestIndex {
    fn alloc(&mut self) -> Result<u32, CapacityError> {
        ForestIndex::alloc(self)
    }
    fn release(&mut self, id: u32) -> bool {
        ForestIndex::release(self, id)
    }
    fn contains(&self, id: u32) -> bool {
        ForestIndex::contains(self, id)
    }
    fn ids_in_order(&self) -> Vec<u32> {
        self.ids().collect()
    }
}

#[derive(Debug)]
struct State<C> {
    core: C,
    keys: KeyMap,
}

impl<C: Core> State<C> {
    /// `key in keys`.
    fn has(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self
            .keys
            .lookup(py, key)?
            .is_some_and(|id| self.core.contains(id)))
    }

    /// The id of `key`, allocating one for a new key.
    fn resolve(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<u32> {
        self.resolve_new(py, key).map(|(id, _)| id)
    }

    /// [`Self::resolve`], and whether the id was allocated now.
    fn resolve_new(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<(u32, bool)> {
        if let Some(id) = self.keys.lookup(py, key)? {
            return Ok((id, false));
        }
        let id = self.core.alloc().map_err(capacity_error)?;
        if let Err(e) = self.keys.bind(py, key, id) {
            self.core.release(id);
            return Err(e);
        }
        Ok((id, true))
    }

    /// Drop the key object of `id` if the core freed it.
    fn after_release(&mut self, py: Python<'_>, id: u32, freed: bool) -> PyResult<()> {
        if freed {
            self.keys.forget(py, id)
        } else {
            Ok(())
        }
    }

    /// The keys with an entry, in insertion order.
    fn key_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let objs = self
            .core
            .ids_in_order()
            .into_iter()
            .map(|id| self.keys.obj(py, id))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, objs)
    }

    fn objs<'py>(&self, py: Python<'py>, ids: &[u32]) -> PyResult<Vec<Bound<'py, PyAny>>> {
        ids.iter().map(|&id| self.keys.obj(py, id)).collect()
    }
}

fn new_state<C>(py: Python<'_>, core: C) -> Mutex<State<C>> {
    Mutex::new(State {
        core,
        keys: KeyMap::new(py),
    })
}

/// Rebuild the key map of a state read with `from_bytes` (ids `0..n`).
fn restore_keys(py: Python<'_>, objs: &Bound<'_, PyList>, n: usize) -> PyResult<KeyMap> {
    if objs.len() != n {
        return Err(PyValueError::new_err(format!(
            "invalid pickled index state: {} keys for {n} ids",
            objs.len()
        )));
    }
    let mut keys = KeyMap::new(py);
    for (i, obj) in objs.iter().enumerate() {
        let id = u32::try_from(i).map_err(|_| PyValueError::new_err("too many keys"))?;
        keys.bind(py, &obj, id)?;
    }
    if keys.ids.bind(py).len() != n {
        return Err(PyValueError::new_err(
            "invalid pickled index state: two keys compare equal",
        ));
    }
    Ok(keys)
}

/// The values of a `hashvalues` array.
#[derive(Debug)]
enum Values {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// `f(values)` for the values' element type.
macro_rules! with_values {
    ($values:expr, $v:ident => $body:expr) => {
        match $values {
            Values::U8($v) => $body,
            Values::U16($v) => $body,
            Values::U32($v) => $body,
            Values::U64($v) => $body,
            Values::I8($v) => $body,
            Values::I16($v) => $body,
            Values::I32($v) => $body,
            Values::I64($v) => $body,
            Values::F32($v) => $body,
            Values::F64($v) => $body,
        }
    };
}

/// A copied `hashvalues` array (or a batch of them, axis 0).
#[derive(Debug)]
struct Signature {
    values: Values,
    shape: Vec<usize>,
}

fn product(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1_usize, |acc, &d| acc.checked_mul(d))
}

impl Signature {
    /// Values per row (axis 0 indexes the rows).
    fn row_len(&self) -> usize {
        product(&self.shape[1..]).unwrap_or(0)
    }

    fn itemsize(&self) -> usize {
        with_values!(&self.values, v => item_size(v))
    }

    fn band_keys(&self, bands: Bands) -> Vec<BandKey> {
        let row_len = self.row_len();
        with_values!(&self.values, v => band_keys(v, row_len, bands))
    }

    /// `_H(hashvalues[start:end])`.
    fn rows(&self, start: usize, end: usize) -> Vec<u8> {
        let row_len = self.row_len();
        with_values!(&self.values, v => swapped_rows(v, row_len, start, end))
    }

    /// `core.query` of this signature's band keys, computed band by band.
    fn query(&self, core: &LshIndex, bands: Bands) -> Vec<u32> {
        let row_len = self.row_len();
        with_values!(&self.values, v => core.query_values(v, row_len, bands))
    }

    /// [`Self::query`] of each item of a batch (axis 0), in parallel.
    fn query_many(
        &self,
        core: &LshIndex,
        bands: Bands,
        t: par::Threads,
    ) -> Result<Vec<Vec<u32>>, par::PoolError> {
        let items = self.shape[0];
        let item_len = product(&self.shape[1..]).unwrap_or(0);
        let row_len = product(&self.shape[2..]).unwrap_or(0);
        with_values!(&self.values, v => {
            core.query_values_many(v, items, item_len, row_len, bands, t)
        })
    }

    /// The band keys of each item of a batch (axis 0), in parallel.
    fn band_keys_many(
        &self,
        bands: Bands,
        t: par::Threads,
    ) -> Result<Vec<Vec<BandKey>>, par::PoolError> {
        let items = self.shape[0];
        let item_len = product(&self.shape[1..]).unwrap_or(0);
        let row_len = product(&self.shape[2..]).unwrap_or(0);
        with_values!(&self.values, v => band_keys_many(v, items, item_len, row_len, bands, t))
    }
}

fn item_size<T>(_: &[T]) -> usize {
    std::mem::size_of::<T>()
}

static NDARRAY: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// `numpy.ndarray`, or `None` if numpy has not been imported (then nothing
/// is an ndarray).
fn ndarray_type(py: Python<'_>) -> PyResult<Option<&Bound<'_, PyType>>> {
    if let Some(ty) = NDARRAY.get(py) {
        return Ok(Some(ty.bind(py)));
    }
    let modules = py
        .import(intern!(py, "sys"))?
        .getattr(intern!(py, "modules"))?;
    let Some(numpy) = modules.cast::<PyDict>()?.get_item(intern!(py, "numpy"))? else {
        return Ok(None);
    };
    let ty = numpy
        .getattr(intern!(py, "ndarray"))?
        .cast_into::<PyType>()?;
    Ok(Some(NDARRAY.get_or_init(py, || ty.unbind()).bind(py)))
}

fn values_of(arr: &Bound<'_, PyUntypedArray>) -> Option<Values> {
    macro_rules! typed {
        ($t:ty, $variant:ident) => {
            arr.cast::<PyArrayDyn<$t>>()
                .ok()?
                .to_vec()
                .ok()
                .map(Values::$variant)
        };
    }
    let dtype = arr.dtype();
    match (dtype.kind(), dtype.itemsize()) {
        (b'u', 1) => typed!(u8, U8),
        (b'u', 2) => typed!(u16, U16),
        (b'u', 4) => typed!(u32, U32),
        (b'u', 8) => typed!(u64, U64),
        (b'i', 1) => typed!(i8, I8),
        (b'i', 2) => typed!(i16, I16),
        (b'i', 4) => typed!(i32, I32),
        (b'i', 8) => typed!(i64, I64),
        (b'f', 4) => typed!(f32, F32),
        (b'f', 8) => typed!(f64, F64),
        _ => None,
    }
}

/// `hv` copied, when it is an exact `numpy.ndarray`, C-contiguous, of at
/// least `min_ndim` dimensions and a supported dtype in native byte order.
fn signature(
    py: Python<'_>,
    hv: &Bound<'_, PyAny>,
    min_ndim: usize,
) -> PyResult<Option<Signature>> {
    let Some(ndarray) = ndarray_type(py)? else {
        return Ok(None);
    };
    if !hv.get_type().is(ndarray) {
        return Ok(None);
    }
    let arr = hv.cast::<PyUntypedArray>()?;
    if arr.ndim() < min_ndim.max(1) || !arr.is_c_contiguous() || product(arr.shape()).is_none() {
        return Ok(None);
    }
    Ok(values_of(arr).map(|values| Signature {
        values,
        shape: arr.shape().to_vec(),
    }))
}

/// The band keys a `slow` callback returned (a list of bytes).
fn py_band_keys(hs: &Bound<'_, PyAny>) -> PyResult<Vec<BandKey>> {
    hs.try_iter()?
        .map(|h| {
            let h = h?;
            let bytes = h.cast::<PyBytes>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "pyroid.datasketch: band keys must be bytes, got {}",
                    type_name(&h)
                ))
            })?;
            Ok(BandKey::from(bytes.as_bytes()))
        })
        .collect()
}

/// One item's band keys: computed here later, or already from Python.
#[derive(Debug)]
enum Hs {
    Native(Signature),
    Python(Vec<BandKey>),
}

fn dup_error() -> PyErr {
    PyValueError::new_err("The given key already exists")
}

fn item_note(py: Python<'_>, err: PyErr, j: usize) -> PyErr {
    with_note(py, err, format!("pyroid: item {j} of the batch"))
}

fn bytes_list<'py>(py: Python<'py>, hs: &[BandKey]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, hs.iter().map(|h| PyBytes::new(py, h)))
}

/// The native `MinHashLSH` index: `keys` and one band table per band.
#[pyclass(frozen, module = "pyroid.datasketch.lsh", name = "_LSHNative")]
#[derive(Debug)]
pub struct LshNative {
    num_tables: usize,
    /// Rows per band; `None`: `r` is not a non-negative int, so every band
    /// key comes from the Python callback.
    r: Option<usize>,
    state: Mutex<State<LshIndex>>,
}

impl LshNative {
    fn bands(&self, n: usize) -> Option<Bands> {
        self.r.map(|r| Bands { b: n, r })
    }

    /// `minhash`'s band keys for the first `n` bands, unevaluated.
    fn prepare(
        &self,
        py: Python<'_>,
        minhash: &Bound<'_, PyAny>,
        n: usize,
        slow: &Bound<'_, PyAny>,
    ) -> PyResult<Hs> {
        if self.r.is_some() {
            let hv = minhash.getattr(intern!(py, "hashvalues"))?;
            if let Some(sig) = signature(py, &hv, 1)? {
                return Ok(Hs::Native(sig));
            }
        }
        py_band_keys(&slow.call1((minhash, n))?).map(Hs::Python)
    }

    fn band_keys_of(
        &self,
        py: Python<'_>,
        minhash: &Bound<'_, PyAny>,
        n: usize,
        slow: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<BandKey>> {
        Ok(match self.prepare(py, minhash, n, slow)? {
            Hs::Python(hs) => hs,
            Hs::Native(sig) => sig.band_keys(self.bands(n).expect("native band keys need r")),
        })
    }

    /// Band keys of prepared items, computed in parallel (detached).
    fn compute(
        &self,
        py: Python<'_>,
        items: &[Hs],
        t: par::Threads,
    ) -> PyResult<Vec<Vec<BandKey>>> {
        let bands = self.bands(self.num_tables);
        py.detach(|| {
            par::map(items.len(), t, |j| match &items[j] {
                Hs::Python(hs) => hs.clone(),
                Hs::Native(sig) => sig.band_keys(bands.expect("native band keys need r")),
            })
        })
        .map_err(pool_error)
    }

    fn has(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        lock(py, &self.state)?.has(py, key)
    }

    /// The duplicate check of one batch item: against the index and the
    /// batch so far (`pending`).
    fn check_new(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        pending: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        if pending.contains(key)? || self.has(py, key)? {
            return Err(dup_error());
        }
        pending.set_item(key, py.None())
    }

    /// Insert prepared batch items: all of them, or none if a key is a
    /// duplicate by now or cannot be stored.
    fn commit(
        &self,
        py: Python<'_>,
        keys: &[Bound<'_, PyAny>],
        hss: Vec<Vec<BandKey>>,
        check_duplication: bool,
        t: par::Threads,
    ) -> PyResult<()> {
        let mut st = lock(py, &self.state)?;
        let mut ids = Vec::with_capacity(keys.len());
        for key in keys {
            let id = st.keys.lookup(py, key)?;
            if check_duplication && id.is_some_and(|id| st.core.contains(id)) {
                return Err(dup_error());
            }
            ids.push(id);
        }
        let mut fresh = Vec::new();
        let mut resolved = Vec::with_capacity(keys.len());
        for (key, id) in keys.iter().zip(&ids) {
            // A key new to the index may repeat within the batch: resolve finds
            // the id its first occurrence got.
            let id = match id {
                Some(id) => Ok(*id),
                None => st.resolve_new(py, key).map(|(id, allocated)| {
                    if allocated {
                        fresh.push(id);
                    }
                    id
                }),
            };
            match id {
                Ok(id) => resolved.push(id),
                Err(e) => {
                    for id in fresh {
                        st.core.release(id);
                        st.keys.forget(py, id)?;
                    }
                    return Err(e);
                }
            }
        }
        let items: Vec<(u32, Vec<BandKey>)> = resolved.into_iter().zip(hss).collect();
        let core = &mut st.core;
        py.detach(|| core.insert_many(&items, t))
            .map_err(pool_error)
    }

    /// Run `queries` (pure Rust, detached, under the lock); one list of keys each.
    fn lists<'py, F>(&self, py: Python<'py>, queries: F) -> PyResult<Bound<'py, PyList>>
    where
        F: FnOnce(&LshIndex) -> Result<Vec<Vec<u32>>, par::PoolError> + Send,
    {
        let st = lock(py, &self.state)?;
        let core = &st.core;
        let found = py.detach(|| queries(core)).map_err(pool_error)?;
        let lists = found
            .iter()
            .map(|ids| PyList::new(py, st.objs(py, ids)?))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, lists)
    }

    fn check_table(&self, table: usize) -> PyResult<()> {
        if table < self.num_tables {
            Ok(())
        } else {
            Err(PyIndexError::new_err("band table index out of range"))
        }
    }
}

/// A batch of signatures as one array: rows along axis 0.
fn signature_array(py: Python<'_>, values: &Bound<'_, PyAny>) -> PyResult<Signature> {
    signature(py, values, 2)?.ok_or_else(|| {
        PyTypeError::new_err(format!(
            "pyroid.datasketch: expected a C-contiguous numpy array of signatures (2-D, an \
             integer or float dtype in native byte order), got {}",
            type_name(values)
        ))
    })
}

#[pymethods]
impl LshNative {
    #[new]
    #[pyo3(signature = (num_tables, r))]
    fn new(py: Python<'_>, num_tables: usize, r: Option<usize>) -> Self {
        Self {
            num_tables,
            r,
            state: new_state(py, LshIndex::new(num_tables)),
        }
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(lock(py, &self.state)?.core.len())
    }

    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        // Skipped while another thread holds the lock: GC then sees the
        // references as external, which only delays collection.
        match self.state.try_lock() {
            Ok(st) => st.keys.traverse(&visit),
            Err(_) => Ok(()),
        }
    }

    /// datasketch's `is_empty()`: some band table has no bucket.
    fn is_empty(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(lock(py, &self.state)?.core.has_empty_table())
    }

    /// `key in lsh.keys`.
    fn contains(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.has(py, key)
    }

    /// `_insert` after its length, scheme and key checks (and `pickle.dumps`).
    #[pyo3(signature = (key, minhash, check_duplication, slow))]
    fn insert(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        minhash: &Bound<'_, PyAny>,
        check_duplication: bool,
        slow: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if check_duplication && self.has(py, key)? {
            return Err(dup_error());
        }
        let hs = self.band_keys_of(py, minhash, self.num_tables, slow)?;
        let mut st = lock(py, &self.state)?;
        let id = st.resolve(py, key)?;
        st.core.insert(id, &hs);
        Ok(())
    }

    /// Insert `keys[j]` with `minhashes[j]`, as `insert` one at a time, or
    /// nothing if an item fails. `prepare(key, minhash)` runs `_insert`'s
    /// checks and returns the key to store.
    #[pyo3(signature = (keys, minhashes, prepare, slow, check_duplication, threads))]
    #[allow(clippy::too_many_arguments)]
    fn insert_many(
        &self,
        py: Python<'_>,
        keys: Vec<Bound<'_, PyAny>>,
        minhashes: Vec<Bound<'_, PyAny>>,
        prepare: &Bound<'_, PyAny>,
        slow: &Bound<'_, PyAny>,
        check_duplication: bool,
        threads: Option<Threads>,
    ) -> PyResult<()> {
        if keys.len() != minhashes.len() {
            return Err(PyValueError::new_err("keys and minhashes differ in length"));
        }
        let pending = PyDict::new(py);
        let mut stored = Vec::with_capacity(keys.len());
        let mut items = Vec::with_capacity(keys.len());
        for (j, (key, minhash)) in keys.iter().zip(&minhashes).enumerate() {
            let item = || -> PyResult<(Bound<'_, PyAny>, Hs)> {
                let key = prepare.call1((key, minhash))?;
                if check_duplication {
                    self.check_new(py, &key, &pending)?;
                }
                let hs = self.prepare(py, minhash, self.num_tables, slow)?;
                if !check_duplication {
                    key.hash()?;
                }
                Ok((key, hs))
            };
            let (key, hs) = item().map_err(|e| item_note(py, e, j))?;
            stored.push(key);
            items.push(hs);
        }
        let t = self::threads(threads);
        let hss = self.compute(py, &items, t)?;
        drop(items);
        self.commit(py, &stored, hss, check_duplication, t)
    }

    /// [`Self::insert_many`] with the signatures as the rows of one array.
    /// `prepare(key)` (optional) returns the key to store.
    #[pyo3(signature = (keys, values, prepare, check_duplication, threads))]
    fn insert_array(
        &self,
        py: Python<'_>,
        keys: Vec<Bound<'_, PyAny>>,
        values: &Bound<'_, PyAny>,
        prepare: Option<&Bound<'_, PyAny>>,
        check_duplication: bool,
        threads: Option<Threads>,
    ) -> PyResult<()> {
        let sig = signature_array(py, values)?;
        if keys.len() != sig.shape[0] {
            return Err(PyValueError::new_err("keys and minhashes differ in length"));
        }
        let pending = PyDict::new(py);
        let mut stored = Vec::with_capacity(keys.len());
        for (j, key) in keys.iter().enumerate() {
            let item = || -> PyResult<Bound<'_, PyAny>> {
                let key = match prepare {
                    Some(f) => f.call1((key,))?,
                    None => key.clone(),
                };
                if check_duplication {
                    self.check_new(py, &key, &pending)?;
                } else {
                    key.hash()?;
                }
                Ok(key)
            };
            stored.push(item().map_err(|e| item_note(py, e, j))?);
        }
        let t = self::threads(threads);
        let hss = match self.bands(self.num_tables) {
            Some(bands) => py
                .detach(|| sig.band_keys_many(bands, t))
                .map_err(pool_error)?,
            None => return Err(PyValueError::new_err("this index has no integer band size")),
        };
        self.commit(py, &stored, hss, check_duplication, t)
    }

    /// `_remove` after `pickle.dumps`.
    fn remove(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut st = lock(py, &self.state)?;
        let Some(id) = st.keys.lookup(py, key)?.filter(|&id| st.core.contains(id)) else {
            return Err(PyValueError::new_err("The given key does not exist"));
        };
        match st.core.remove(id) {
            Ok(freed) => st.after_release(py, id, freed),
            Err(RemoveError::NotFound) => {
                Err(PyValueError::new_err("The given key does not exist"))
            }
            Err(RemoveError::Missing { .. }) => Err(PyKeyError::new_err(key.clone().unbind())),
        }
    }

    /// `[self._H(minhash.hashvalues[s:e]) for s, e in self.hashranges[:n]]`.
    fn band_keys<'py>(
        &self,
        py: Python<'py>,
        minhash: &Bound<'py, PyAny>,
        n: usize,
        slow: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        bytes_list(py, &self.band_keys_of(py, minhash, n, slow)?)
    }

    /// The keys in the buckets of `minhash` in the first `n` bands, each once.
    fn query<'py>(
        &self,
        py: Python<'py>,
        minhash: &Bound<'py, PyAny>,
        n: usize,
        slow: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let hs = self.prepare(py, minhash, n, slow)?;
        let st = lock(py, &self.state)?;
        let ids = match &hs {
            Hs::Python(hs) => st.core.query(hs),
            Hs::Native(sig) => sig.query(&st.core, self.bands(n).expect("native band keys need r")),
        };
        PyList::new(py, st.objs(py, &ids)?)
    }

    /// `query` of every item; `prepare(minhash)` runs `query`'s checks.
    #[pyo3(signature = (minhashes, prepare, slow, threads))]
    fn query_many<'py>(
        &self,
        py: Python<'py>,
        minhashes: Vec<Bound<'py, PyAny>>,
        prepare: &Bound<'py, PyAny>,
        slow: &Bound<'py, PyAny>,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyList>> {
        let mut items = Vec::with_capacity(minhashes.len());
        for (j, minhash) in minhashes.iter().enumerate() {
            let item = || -> PyResult<Hs> {
                prepare.call1((minhash,))?;
                self.prepare(py, minhash, self.num_tables, slow)
            };
            items.push(item().map_err(|e| item_note(py, e, j))?);
        }
        let t = self::threads(threads);
        let bands = self.bands(self.num_tables);
        let items = &items;
        self.lists(py, |core| {
            par::map(items.len(), t, |j| match &items[j] {
                Hs::Python(hs) => core.query(hs),
                Hs::Native(sig) => sig.query(core, bands.expect("native band keys need r")),
            })
        })
    }

    /// [`Self::query_many`] with the signatures as the rows of one array.
    #[pyo3(signature = (values, threads))]
    fn query_array<'py>(
        &self,
        py: Python<'py>,
        values: &Bound<'py, PyAny>,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyList>> {
        let sig = signature_array(py, values)?;
        let t = self::threads(threads);
        let Some(bands) = self.bands(self.num_tables) else {
            return Err(PyValueError::new_err("this index has no integer band size"));
        };
        let sig = &sig;
        self.lists(py, |core| sig.query_many(core, bands, t))
    }

    /// `_merge` after its checks: every key of `other` (in order) with its
    /// band keys; `check_overlap` raises first if a key is in both.
    fn merge(
        &self,
        py: Python<'_>,
        other: &Bound<'_, LshNative>,
        check_overlap: bool,
    ) -> PyResult<()> {
        let snapshot = {
            let o = lock(py, &other.get().state)?;
            o.core
                .ids()
                .map(|id| {
                    let hs = o.core.keys_of(id).ok_or_else(corrupt)?.to_vec();
                    Ok((o.keys.obj(py, id)?, hs))
                })
                .collect::<PyResult<Vec<_>>>()?
        };
        let mut st = lock(py, &self.state)?;
        if check_overlap {
            for (key, _) in &snapshot {
                if st.has(py, key)? {
                    return Err(PyValueError::new_err(
                        "The keys are overlapping, duplicate key exists.",
                    ));
                }
            }
        }
        for (key, hs) in &snapshot {
            let id = st.resolve(py, key)?;
            st.core.insert(id, hs);
        }
        Ok(())
    }

    /// The keys, in insertion order (`list(lsh.keys)`).
    fn keys_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        lock(py, &self.state)?.key_list(py)
    }

    /// `lsh.keys.get(key)` as a new list, `None` if absent.
    fn keys_get<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let st = lock(py, &self.state)?;
        match st.keys.lookup(py, key)?.and_then(|id| st.core.keys_of(id)) {
            None => Ok(None),
            Some(hs) => bytes_list(py, hs).map(Some),
        }
    }

    /// `lsh.keys.itemcounts()`.
    fn keys_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let st = lock(py, &self.state)?;
        let out = PyDict::new(py);
        for id in st.core.ids() {
            let n = st.core.keys_of(id).map_or(0, <[BandKey]>::len);
            out.set_item(st.keys.obj(py, id)?, n)?;
        }
        Ok(out)
    }

    /// The number of buckets in band `table`.
    fn table_len(&self, py: Python<'_>, table: usize) -> PyResult<usize> {
        self.check_table(table)?;
        Ok(lock(py, &self.state)?.core.table_len(table))
    }

    /// The band keys of band `table`.
    fn table_keys<'py>(&self, py: Python<'py>, table: usize) -> PyResult<Bound<'py, PyList>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        PyList::new(py, st.core.buckets(table).map(|(h, _)| PyBytes::new(py, h)))
    }

    /// Bucket `h` of band `table` as a new set, `None` if absent.
    fn table_get<'py>(
        &self,
        py: Python<'py>,
        table: usize,
        h: &[u8],
    ) -> PyResult<Option<Bound<'py, PySet>>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        match st.core.bucket(table, h) {
            None => Ok(None),
            Some(ids) => PySet::new(py, st.objs(py, ids)?).map(Some),
        }
    }

    fn table_contains(&self, py: Python<'_>, table: usize, h: &[u8]) -> PyResult<bool> {
        self.check_table(table)?;
        Ok(lock(py, &self.state)?.core.bucket(table, h).is_some())
    }

    /// `hashtables[table].itemcounts()`.
    fn table_counts<'py>(&self, py: Python<'py>, table: usize) -> PyResult<Bound<'py, PyDict>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        let out = PyDict::new(py);
        for (h, ids) in st.core.buckets(table) {
            out.set_item(PyBytes::new(py, h), ids.len())?;
        }
        Ok(out)
    }

    fn __getstate__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyList>)> {
        let st = lock(py, &self.state)?;
        let core = &st.core;
        let (blob, ids) = py.detach(|| core.to_bytes());
        Ok((
            PyBytes::new(py, &blob),
            PyList::new(py, st.objs(py, &ids)?)?,
        ))
    }

    fn __setstate__(
        &self,
        py: Python<'_>,
        state: (Bound<'_, PyBytes>, Bound<'_, PyList>),
    ) -> PyResult<()> {
        let (blob, objs) = state;
        let core = LshIndex::from_bytes(blob.as_bytes())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if core.num_tables() != self.num_tables {
            return Err(PyValueError::new_err(
                "invalid pickled index state: wrong number of band tables",
            ));
        }
        let keys = restore_keys(py, &objs, core.id_count())?;
        *lock(py, &self.state)? = State { core, keys };
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(
        Bound<'py, PyType>,
        (usize, Option<usize>),
        (Bound<'py, PyBytes>, Bound<'py, PyList>),
    )> {
        let me = slf.get();
        Ok((
            slf.get_type(),
            (me.num_tables, me.r),
            me.__getstate__(slf.py())?,
        ))
    }
}

/// The native `MinHashLSHForest` index.
#[pyclass(frozen, module = "pyroid.datasketch.lshforest", name = "_ForestNative")]
#[derive(Debug)]
pub struct ForestNative {
    l: usize,
    /// `k`: the band size and the maximum prefix depth.
    depth: usize,
    state: Mutex<State<ForestIndex>>,
}

/// `minhash.hashvalues`, copied if this module reads it.
fn hashvalues_of(py: Python<'_>, minhash: &Bound<'_, PyAny>) -> PyResult<Option<Signature>> {
    signature(py, &minhash.getattr(intern!(py, "hashvalues"))?, 1)
}

impl ForestNative {
    fn check_table(&self, table: usize) -> PyResult<()> {
        if table < self.l {
            Ok(())
        } else {
            Err(PyIndexError::new_err("table index out of range"))
        }
    }
}

#[pymethods]
impl ForestNative {
    #[new]
    #[pyo3(signature = (l, depth))]
    fn new(py: Python<'_>, l: usize, depth: usize) -> Self {
        Self {
            l,
            depth,
            state: new_state(py, ForestIndex::new(l)),
        }
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(lock(py, &self.state)?.core.len())
    }

    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        match self.state.try_lock() {
            Ok(st) => st.keys.traverse(&visit),
            Err(_) => Ok(()),
        }
    }

    /// `add` after its length and scheme checks.
    fn add(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        minhash: &Bound<'_, PyAny>,
        slow: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if lock(py, &self.state)?.has(py, key)? {
            return Err(PyValueError::new_err(
                "The given key has already been added",
            ));
        }
        let bands = Bands {
            b: self.l,
            r: self.depth,
        };
        let hs = match hashvalues_of(py, minhash)? {
            Some(sig) => sig.band_keys(bands),
            None => py_band_keys(&slow.call1((minhash,))?)?,
        };
        let mut st = lock(py, &self.state)?;
        let id = st.resolve(py, key)?;
        st.core.add(id, hs);
        Ok(())
    }

    fn index(&self, py: Python<'_>) -> PyResult<()> {
        let mut st = lock(py, &self.state)?;
        let core = &mut st.core;
        py.detach(|| core.index());
        Ok(())
    }

    fn is_empty(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(lock(py, &self.state)?.core.has_empty_table())
    }

    fn contains(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        lock(py, &self.state)?.has(py, key)
    }

    /// `query(minhash, want)` after its checks; `None` when `hashvalues` is
    /// not an array this module reads (the caller runs datasketch's loop).
    fn query<'py>(
        &self,
        py: Python<'py>,
        minhash: &Bound<'py, PyAny>,
        want: usize,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let Some(sig) = hashvalues_of(py, minhash)? else {
            return Ok(None);
        };
        let full: Vec<Vec<u8>> = (0..self.l)
            .map(|i| sig.rows(i * self.depth, (i + 1) * self.depth))
            .collect();
        let row_bytes = sig.row_len() * sig.itemsize();
        let st = lock(py, &self.state)?;
        let core = &st.core;
        let depth = self.depth;
        let ids = py.detach(|| core.query(&full, depth, row_bytes, want));
        PyList::new(py, st.objs(py, &ids)?).map(Some)
    }

    /// `[self._H(minhash.hashvalues[start:start + r]) for start, _ in
    /// self.hashranges]`, or `None` (see [`Self::query`]).
    fn prefixes<'py>(
        &self,
        py: Python<'py>,
        minhash: &Bound<'py, PyAny>,
        r: usize,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let Some(sig) = hashvalues_of(py, minhash)? else {
            return Ok(None);
        };
        let hps = (0..self.l).map(|i| {
            let start = i * self.depth;
            PyBytes::new(py, &sig.rows(start, start.saturating_add(r)))
        });
        PyList::new(py, hps).map(Some)
    }

    /// The keys `_query` yields for table `table` and prefix `hp`.
    fn prefix_keys<'py>(
        &self,
        py: Python<'py>,
        table: usize,
        hp: &[u8],
        prefix_size: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        let ids: Vec<u32> = st.core.prefix_members(table, hp, prefix_size).collect();
        PyList::new(py, st.objs(py, &ids)?)
    }

    fn keys_list<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        lock(py, &self.state)?.key_list(py)
    }

    /// `forest.keys[key]` as a new list, `None` if absent.
    fn keys_get<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let st = lock(py, &self.state)?;
        match st.keys.lookup(py, key)?.and_then(|id| st.core.keys_of(id)) {
            None => Ok(None),
            Some(hs) => bytes_list(py, hs).map(Some),
        }
    }

    /// `forest.keys[key] = hs`.
    fn keys_set(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        hs: Vec<PyBackedBytes>,
    ) -> PyResult<()> {
        let hs = hs.iter().map(|h| BandKey::from(&**h)).collect();
        let mut st = lock(py, &self.state)?;
        let id = st.resolve(py, key)?;
        st.core.set_keys(id, hs);
        Ok(())
    }

    /// `del forest.keys[key]`; false if absent.
    fn keys_del(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut st = lock(py, &self.state)?;
        let Some(id) = st.keys.lookup(py, key)? else {
            return Ok(false);
        };
        match st.core.del_keys(id) {
            None => Ok(false),
            Some(freed) => st.after_release(py, id, freed).map(|()| true),
        }
    }

    fn table_len(&self, py: Python<'_>, table: usize) -> PyResult<usize> {
        self.check_table(table)?;
        Ok(lock(py, &self.state)?.core.table_len(table))
    }

    fn table_keys<'py>(&self, py: Python<'py>, table: usize) -> PyResult<Bound<'py, PyList>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        PyList::new(py, st.core.buckets(table).map(|(h, _)| PyBytes::new(py, h)))
    }

    /// Bucket `h` of table `table` as a new list, `None` if absent.
    fn table_get<'py>(
        &self,
        py: Python<'py>,
        table: usize,
        h: &[u8],
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        match st.core.bucket(table, h) {
            None => Ok(None),
            Some(ids) => PyList::new(py, st.objs(py, ids)?).map(Some),
        }
    }

    fn table_contains(&self, py: Python<'_>, table: usize, h: &[u8]) -> PyResult<bool> {
        self.check_table(table)?;
        Ok(lock(py, &self.state)?.core.bucket(table, h).is_some())
    }

    /// `sorted_hashtables[table]` as a new list.
    fn sorted_list<'py>(&self, py: Python<'py>, table: usize) -> PyResult<Bound<'py, PyList>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        bytes_list(py, st.core.sorted(table))
    }

    fn sorted_len(&self, py: Python<'_>, table: usize) -> PyResult<usize> {
        self.check_table(table)?;
        Ok(lock(py, &self.state)?.core.sorted(table).len())
    }

    /// `sorted_hashtables[table][i]` for `0 <= i < len`.
    fn sorted_item<'py>(
        &self,
        py: Python<'py>,
        table: usize,
        i: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.check_table(table)?;
        let st = lock(py, &self.state)?;
        match st.core.sorted(table).get(i) {
            Some(h) => Ok(PyBytes::new(py, h)),
            None => Err(PyIndexError::new_err("list index out of range")),
        }
    }

    fn __getstate__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyList>)> {
        let st = lock(py, &self.state)?;
        let core = &st.core;
        let (blob, ids) = py.detach(|| core.to_bytes());
        Ok((
            PyBytes::new(py, &blob),
            PyList::new(py, st.objs(py, &ids)?)?,
        ))
    }

    fn __setstate__(
        &self,
        py: Python<'_>,
        state: (Bound<'_, PyBytes>, Bound<'_, PyList>),
    ) -> PyResult<()> {
        let (blob, objs) = state;
        let core = ForestIndex::from_bytes(blob.as_bytes())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if core.num_tables() != self.l {
            return Err(PyValueError::new_err(
                "invalid pickled index state: wrong number of tables",
            ));
        }
        let n = core.id_count();
        let keys = restore_keys(py, &objs, n)?;
        *lock(py, &self.state)? = State { core, keys };
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(
        Bound<'py, PyType>,
        (usize, usize),
        (Bound<'py, PyBytes>, Bound<'py, PyList>),
    )> {
        let me = slf.get();
        Ok((slf.get_type(), (me.l, me.depth), me.__getstate__(slf.py())?))
    }
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LshNative>()?;
    m.add_class::<ForestNative>()?;
    Ok(())
}
