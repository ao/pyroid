//! State that is set once and read lock-free, but that Python code may replace later.
//!
//! packaging's objects are plain Python objects whose slots its own code (and some callers,
//! hatchling for one) assign after construction. pyroid's classes are frozen (shared across
//! threads without a borrow flag); a [`StateCell`] keeps the first state in a `OnceLock` and a
//! replacement behind a mutex, so the common case never locks.

use std::ops::Deref;
use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use pyo3::prelude::*;
use pyo3::types::PyBool;

/// A value given at construction (read without locking) that Python code may replace later.
///
/// Constructing one is cheap (no `OnceLock::set`, measured at about 35 ns on the hot path); a
/// replacement, or the first value of an object made empty by `__new__`, goes behind a mutex.
#[derive(Debug)]
pub struct StateCell<T> {
    first: Option<T>,
    replaced: OnceLock<Mutex<Arc<T>>>,
}

/// A borrowed or shared current value of a [`StateCell`].
pub enum StateRef<'a, T> {
    First(&'a T),
    Replaced(Arc<T>),
}

impl<T> Deref for StateRef<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            StateRef::First(v) => v,
            StateRef::Replaced(a) => a,
        }
    }
}

impl<T> StateCell<T> {
    /// An empty cell.
    pub fn new() -> Self {
        Self {
            first: None,
            replaced: OnceLock::new(),
        }
    }

    /// A cell holding `v`.
    pub fn with(v: T) -> Self {
        Self {
            first: Some(v),
            replaced: OnceLock::new(),
        }
    }

    /// The current value, if any.
    pub fn get(&self) -> Option<StateRef<'_, T>> {
        if let Some(m) = self.replaced.get() {
            let g = m.lock().unwrap_or_else(PoisonError::into_inner);
            return Some(StateRef::Replaced(Arc::clone(&g)));
        }
        self.first.as_ref().map(StateRef::First)
    }

    /// Is a value set?
    pub fn is_set(&self) -> bool {
        self.first.is_some() || self.replaced.get().is_some()
    }

    /// Set or replace the value.
    pub fn set(&self, v: T) {
        let mut v = Some(Arc::new(v));
        let m = self
            .replaced
            .get_or_init(|| Mutex::new(v.take().unwrap_or_else(|| unreachable!())));
        if let Some(v) = v {
            let mut g = m.lock().unwrap_or_else(PoisonError::into_inner);
            *g = v;
        }
    }
}

impl<T> Default for StateCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// A cached hash: CPython hashes are never -1, so -1 means "not computed yet".
#[derive(Debug)]
pub struct HashCache(AtomicI64);

impl HashCache {
    /// Nothing cached.
    pub fn new() -> Self {
        Self(AtomicI64::new(-1))
    }

    /// The cached hash, if any.
    pub fn get(&self) -> Option<isize> {
        let h = self.0.load(Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        (h != -1).then_some(h as isize)
    }

    /// Cache `h`.
    pub fn set(&self, h: isize) {
        self.0.store(h as i64, Ordering::Relaxed);
    }
}

impl Default for HashCache {
    fn default() -> Self {
        Self::new()
    }
}

/// A `prereleases` value: `None`, `True`, `False`, or (rarely) any other object, kept as is
/// because packaging tests it both with `is False` and by truth value.
#[derive(Debug)]
pub struct Prerel {
    kind: AtomicU8,
    other: Mutex<Option<Py<PyAny>>>,
}

/// The kind of a [`Prerel`] value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrerelKind {
    None,
    False,
    True,
    /// Another object; its truth value.
    Other(bool),
}

const K_NONE: u8 = 0;
const K_FALSE: u8 = 1;
const K_TRUE: u8 = 2;
const K_OTHER_FALSY: u8 = 3;
const K_OTHER_TRUTHY: u8 = 4;

impl Prerel {
    /// From a Python value.
    pub fn new(value: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let p = Self {
            kind: AtomicU8::new(K_NONE),
            other: Mutex::new(None),
        };
        if let Some(v) = value {
            p.set(v)?;
        }
        Ok(p)
    }

    /// Store `value`.
    pub fn set(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let k = if value.is_none() {
            K_NONE
        } else if value.is(PyBool::new(value.py(), false)) {
            K_FALSE
        } else if value.is(PyBool::new(value.py(), true)) {
            K_TRUE
        } else {
            let truthy = value.is_truthy()?;
            let mut g = self.other.lock().unwrap_or_else(PoisonError::into_inner);
            *g = Some(value.clone().unbind());
            if truthy {
                K_OTHER_TRUTHY
            } else {
                K_OTHER_FALSY
            }
        };
        self.kind.store(k, Ordering::Release);
        Ok(())
    }

    /// The kind of the stored value.
    pub fn kind(&self) -> PrerelKind {
        match self.kind.load(Ordering::Acquire) {
            K_FALSE => PrerelKind::False,
            K_TRUE => PrerelKind::True,
            K_OTHER_FALSY => PrerelKind::Other(false),
            K_OTHER_TRUTHY => PrerelKind::Other(true),
            _ => PrerelKind::None,
        }
    }

    /// The stored value as a Python object.
    pub fn get<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        match self.kind() {
            PrerelKind::None => py.None().into_bound(py),
            PrerelKind::False => PyBool::new(py, false).to_owned().into_any(),
            PrerelKind::True => PyBool::new(py, true).to_owned().into_any(),
            PrerelKind::Other(_) => {
                let g = self.other.lock().unwrap_or_else(PoisonError::into_inner);
                g.as_ref()
                    .map_or_else(|| py.None().into_bound(py), |o| o.bind(py).clone())
            }
        }
    }
}

/// An optional argument whose default is "not passed", so that an explicit `None` is a value
/// (packaging's `Specifier(None)` and `SpecifierSet(None)` raise `TypeError`).
#[derive(Debug)]
pub struct Passed(pub Option<Py<PyAny>>);

impl<'a, 'py> FromPyObject<'a, 'py> for Passed {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Passed(Some(obj.to_owned().unbind())))
    }
}
