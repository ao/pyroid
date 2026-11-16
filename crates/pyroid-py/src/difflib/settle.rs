//! What keeps `difflib.SequenceMatcher` instances working when `pyroid.unpatch("difflib")`
//! runs while other threads use them.
//!
//! Under `pyroid.patch("difflib")`, difflib's own class has pyroid's methods, and a matcher
//! they set up keeps `b2j`, `bjunk`, `bpopular`, `fullbcount` and `matching_blocks` in
//! pyroid's native state, which only pyroid's descriptors on the class can read. Before
//! `unpatch` takes the descriptors off, such a matcher gets CPython's plain attributes instead
//! ([`settle`]). To find these matchers without walking the heap, each thread keeps weak
//! references to the ones its calls gave native state, and lists the ones its native
//! `__init__`, `set_seq1` or `set_seq2` is changing at the moment ([`Gate`]). Switching the
//! patch off ([`_sm_patch`]) makes every call that starts later run CPython's method, then
//! settles every recorded matcher that no call is changing; a call that was changing one
//! settles it when it ends. Nothing walks the heap, and nothing waits for another thread.
//!
//! pyroid.difflib's own `SequenceMatcher` (and its subclasses) keep pyroid's methods for good:
//! their instances are neither recorded nor settled.

use std::cell::OnceCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyString, PyType, PyWeakrefMethods, PyWeakrefReference};

use super::sm::{BSide, cached_list, discard, instance_dict};

/// Whether `pyroid.patch("difflib")` is on.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// Bumped by every switch on or off. Never 0 while the patch is on.
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// pyroid.difflib's own `SequenceMatcher`.
static OWN: PyOnceLock<Py<PyType>> = PyOnceLock::new();

/// A matcher given native state, weakly, and the generation it happened in.
type Made = (u64, Py<PyWeakrefReference>);

/// One thread's records. Its lock is held only briefly, never while Python code runs, and
/// the generation is read under it: a switch-off (which bumps the generation, then takes
/// each cell's records under its lock) either finds a record or the thread that adds it
/// sees the new generation.
struct Cell {
    /// The addresses of the matchers this thread's native calls are changing, innermost last.
    busy: Vec<usize>,
    made: Vec<Made>,
    /// When `made` is this long, its dead references go.
    prune_at: usize,
}

const PRUNE_MIN: usize = 64;

type Shared = Arc<Mutex<Cell>>;

/// Every thread's [`Cell`]. A thread that has ended leaves its cell here until a new thread
/// takes over its records.
static CELLS: Mutex<Vec<Shared>> = Mutex::new(Vec::new());

thread_local! {
    static MINE: OnceCell<Shared> = const { OnceCell::new() };
}

/// `m` locked. A thread that has to wait does so detached, so it cannot hold up the
/// interpreter (or, with the GIL, the thread holding the lock); and it never holds the lock
/// while detached, so every holder is attached and a stop-the-world pause (as before a
/// fork) waits for it to let go.
fn lock<'a, T: Send>(py: Python<'_>, m: &'a Mutex<T>) -> MutexGuard<'a, T> {
    loop {
        match m.try_lock() {
            Ok(guard) => return guard,
            Err(TryLockError::Poisoned(e)) => return e.into_inner(),
            Err(TryLockError::WouldBlock) => {
                py.detach(|| drop(m.lock().unwrap_or_else(PoisonError::into_inner)));
            }
        }
    }
}

/// `f` on this thread's [`Cell`]. `f` must not run Python code or drop Python objects.
fn with_cell<R>(py: Python<'_>, f: impl FnOnce(&mut Cell) -> R) -> R {
    let (result, orphans) = MINE.with(|mine| {
        let mut orphans = Vec::new();
        let shared = mine.get_or_init(|| {
            let shared = Arc::new(Mutex::new(Cell {
                busy: Vec::new(),
                made: Vec::new(),
                prune_at: PRUNE_MIN,
            }));
            let mut all = lock(py, &CELLS);
            // The records of threads that have ended move to this one (a cell a switch-off
            // is reading stays for the next thread: no waiting while CELLS is held).
            all.retain(|c| {
                if Arc::strong_count(c) > 1 {
                    return true;
                }
                match c.try_lock() {
                    Ok(mut dead) => orphans.append(&mut dead.made),
                    Err(TryLockError::Poisoned(e)) => orphans.append(&mut e.into_inner().made),
                    Err(TryLockError::WouldBlock) => return true,
                }
                false
            });
            all.push(Arc::clone(&shared));
            shared
        });
        (f(&mut lock(py, shared)), orphans)
    });
    if !orphans.is_empty() {
        keep(py, orphans);
    }
    result
}

fn remove_last(busy: &mut Vec<usize>, addr: usize) {
    if let Some(i) = busy.iter().rposition(|&a| a == addr) {
        busy.remove(i);
    }
}

/// Put the live records of `made` (taken out of a cell) into this thread's cell; those of an
/// earlier generation (a switch-off may have missed them meanwhile) are settled instead.
fn keep(py: Python<'_>, made: Vec<Made>) {
    // Upgraded with no lock held: dropping the strong references may free a matcher, and the
    // __del__ of a subclass may run difflib code on this thread.
    let mut live = Vec::with_capacity(made.len());
    let mut strong = Vec::with_capacity(made.len());
    for (generation, weak) in made {
        if let Some(obj) = weak.bind(py).upgrade() {
            strong.push(obj);
            live.push((generation, weak));
        }
    }
    let stale = with_cell(py, |c| {
        let now = GENERATION.load(SeqCst);
        let (current, stale): (Vec<_>, Vec<_>) = live.into_iter().partition(|(g, _)| *g == now);
        c.made.extend(current);
        c.prune_at = PRUNE_MIN.max(2 * c.made.len());
        stale
    });
    for (_, weak) in stale {
        if let Some(obj) = weak.bind(py).upgrade() {
            if let Err(e) = settle(&obj) {
                e.write_unraisable(py, Some(&obj));
            }
        }
    }
    drop(strong);
}

/// A native call that may change a matcher's state (`__init__`, `set_seq1`, `set_seq2`).
pub(super) enum Gate<'py> {
    /// The matcher is of pyroid.difflib's own class: nothing to record.
    Own,
    /// A matcher of difflib's class (or a subclass) under the patch.
    Patched {
        obj: Bound<'py, PyAny>,
        generation: u64,
        /// The call gave the matcher native state (recorded when it ends).
        made: bool,
    },
}

impl Gate<'_> {
    /// The generation a native index made in this call records (0 for pyroid's own class).
    pub(super) fn generation(&self) -> u64 {
        match self {
            Gate::Own => 0,
            Gate::Patched { generation, .. } => *generation,
        }
    }

    /// The call gave the matcher native state. `known`: the index it replaces was made in
    /// this generation, so the matcher is recorded already.
    pub(super) fn made(&mut self, known: bool) {
        if let Gate::Patched { made, .. } = self {
            *made |= !known;
        }
    }
}

/// The [`Gate`] for a call on `obj`; `None` if `obj` is of difflib's class (or a subclass)
/// and the patch is off, when the caller runs CPython's method.
pub(super) fn enter<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Gate<'py>>> {
    let py = obj.py();
    if let Some(own) = OWN.get(py) {
        if obj.get_type().is_subclass(own.bind(py).as_any())? {
            return Ok(Some(Gate::Own));
        }
    }
    let addr = obj.as_ptr() as usize;
    // Listed as busy before the check: a switch-off either comes first (and this call runs
    // CPython's method) or finds the matcher busy (and this call settles it at the end).
    let (generation, active) = with_cell(py, |c| {
        c.busy.push(addr);
        (GENERATION.load(SeqCst), ACTIVE.load(SeqCst))
    });
    if !active {
        with_cell(py, |c| remove_last(&mut c.busy, addr));
        return Ok(None);
    }
    Ok(Some(Gate::Patched {
        obj: obj.clone(),
        generation,
        made: false,
    }))
}

impl Drop for Gate<'_> {
    fn drop(&mut self) {
        let Gate::Patched {
            obj,
            generation,
            made,
        } = self
        else {
            return;
        };
        let py = obj.py();
        let addr = obj.as_ptr() as usize;
        let weak = if *made {
            PyWeakrefReference::new(obj).ok().map(Bound::unbind)
        } else {
            None
        };
        let unrecorded = *made && weak.is_none();
        let (ended, full) = with_cell(py, |c| {
            remove_last(&mut c.busy, addr);
            let ended = GENERATION.load(SeqCst) != *generation;
            let mut full = Vec::new();
            if let (false, Some(weak)) = (ended, weak) {
                c.made.push((*generation, weak));
                if c.made.len() >= c.prune_at {
                    full = std::mem::take(&mut c.made);
                }
            }
            (ended, full)
        });
        if !full.is_empty() {
            keep(py, full);
        }
        if (ended || unrecorded) && !std::thread::panicking() {
            // The patch went off during the call (or the matcher could not be recorded).
            if let Err(e) = settle(obj) {
                e.write_unraisable(py, Some(&*obj));
            }
        }
    }
}

/// `_sm_forked()`, in the child of a fork: the other threads are gone, and so are the calls
/// they were making (their records of made matchers stay, for a later switch-off).
#[pyfunction]
pub fn _sm_forked(py: Python<'_>) {
    let mine = MINE.with(|m| m.get().cloned());
    for cell in lock(py, &CELLS).iter() {
        if mine.as_ref().is_some_and(|m| Arc::ptr_eq(m, cell)) {
            continue;
        }
        if let Ok(mut c) = cell.try_lock() {
            c.busy.clear();
        }
    }
}

/// `_sm_own(cls)`: pyroid.difflib's own `SequenceMatcher` (set once, at import).
#[pyfunction]
pub fn _sm_own(py: Python<'_>, cls: Bound<'_, PyType>) {
    let _ = OWN.set(py, cls.unbind());
}

/// `_sm_patch(on)`: `pyroid.patch("difflib")` was switched on, or is being switched off.
/// Switching off gives every recorded matcher CPython's state ([`settle`]) and returns how
/// many it settled.
#[pyfunction]
pub fn _sm_patch(py: Python<'_>, on: bool) -> PyResult<usize> {
    if on {
        GENERATION.fetch_add(1, SeqCst);
        ACTIVE.store(true, SeqCst);
        return Ok(0);
    }
    ACTIVE.store(false, SeqCst);
    GENERATION.fetch_add(1, SeqCst);
    let cells: Vec<Shared> = lock(py, &CELLS).clone();
    let mut busy = Vec::new();
    let mut made = Vec::new();
    for cell in &cells {
        let mut c = lock(py, cell);
        busy.extend_from_slice(&c.busy);
        made.append(&mut c.made);
        c.prune_at = PRUNE_MIN;
    }
    drop(cells);
    let mut settled = 0;
    let mut first_error = None;
    for (_, weak) in made {
        let Some(obj) = weak.bind(py).upgrade() else {
            continue;
        };
        if busy.contains(&(obj.as_ptr() as usize)) {
            continue; // the call changing it settles it when it ends
        }
        match settle(&obj) {
            Ok(true) => settled += 1,
            Ok(false) => {}
            Err(e) => {
                first_error.get_or_insert(e);
            }
        }
    }
    first_error.map_or(Ok(settled), Err)
}

/// The `_pyroid_` entries of the instance dict `d`.
fn native_keys<'py>(d: &Bound<'py, PyDict>) -> Vec<Bound<'py, PyString>> {
    d.keys()
        .iter()
        .filter_map(|k| k.cast_into::<PyString>().ok())
        .filter(|k| k.to_str().is_ok_and(|s| s.starts_with("_pyroid_")))
        .collect()
}

/// Give `obj` CPython's attributes in place of pyroid's native state: `b2j`, `bjunk`,
/// `bpopular`, `fullbcount` and `matching_blocks` as plain instance attributes (what
/// pickling gives), and no `_pyroid_` entries. False if it had no native state.
///
/// Safe while other threads use `obj` with CPython's methods: an attribute is only added
/// where the instance has none (a `set_seq2` running meanwhile writes its own), and only from
/// an index that is still `b`'s.
pub(super) fn settle(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    if native_keys(&d).is_empty() {
        return Ok(false);
    }
    let setdefault = |name: &Bound<'_, PyString>, value: Bound<'_, PyAny>| -> PyResult<()> {
        d.call_method1(pyo3::intern!(py, "setdefault"), (name, value))?;
        Ok(())
    };
    let b = obj.getattr(pyo3::intern!(py, "b"))?;
    let side = d
        .get_item(pyo3::intern!(py, "_pyroid_b"))?
        .and_then(|side| side.cast_into::<BSide>().ok())
        .filter(|side| side.get().b().bind(py).is(&b));
    if let Some(side) = &side {
        let s = side.get();
        let b2j = pyo3::intern!(py, "b2j");
        if !d.contains(b2j)? {
            setdefault(b2j, s.b2j(py)?.into_any())?;
        }
        let bjunk = pyo3::intern!(py, "bjunk");
        if !d.contains(bjunk)? {
            setdefault(bjunk, s.bjunk(py)?.into_any())?;
        }
        let bpopular = pyo3::intern!(py, "bpopular");
        if !d.contains(bpopular)? {
            setdefault(bpopular, s.bpopular(py)?.into_any())?;
        }
    }
    let fullbcount = pyo3::intern!(py, "fullbcount");
    if !d.contains(fullbcount)? {
        let value = match &side {
            Some(side) if d.contains(pyo3::intern!(py, "_pyroid_qr"))? => {
                side.get().fullbcount(py)?.into_any()
            }
            _ => py.None().into_bound(py),
        };
        setdefault(fullbcount, value)?;
    }
    let matching_blocks = pyo3::intern!(py, "matching_blocks");
    if !d.contains(matching_blocks)? {
        let value = match cached_list(obj, &d)? {
            Some(list) => list.into_any(),
            None => py.None().into_bound(py),
        };
        setdefault(matching_blocks, value)?;
    }
    // Read again: making the blocks may have cached `a`'s encoding.
    for key in native_keys(&d) {
        discard(&d, &key)?;
    }
    Ok(true)
}
