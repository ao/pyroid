//! State that is set once and read lock-free, but that Python code may replace later.
//!
//! packaging's objects are plain Python objects whose slots its own code (and some callers,
//! hatchling for one) assign after construction. pyroid's classes are frozen (shared across
//! threads without a borrow flag); a [`StateCell`] keeps the first state in a `OnceLock` and a
//! replacement behind a mutex, so the common case never locks.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use pyo3::prelude::*;
use pyo3::types::PyBool;

/// A value set once (lock-free reads) that may be replaced later (behind a mutex).
#[derive(Debug)]
pub struct StateCell<T> {
    first: OnceLock<T>,
    replaced_flag: AtomicBool,
    replaced: Mutex<Option<Arc<T>>>,
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

impl<T> Default for StateCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> StateCell<T> {
    /// An empty cell.
    pub fn new() -> Self {
        Self { first: OnceLock::new(), replaced_flag: AtomicBool::new(false), replaced: Mutex::new(None) }
    }

    /// A cell holding `v`.
    pub fn with(v: T) -> Self {
        let c = Self::new();
        let _ = c.first.set(v);
        c
    }

    /// The current value, if any.
    pub fn get(&self) -> Option<StateRef<'_, T>> {
        if self.replaced_flag.load(Ordering::Acquire) {
            let g = self.replaced.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(a) = g.as_ref() {
                return Some(StateRef::Replaced(Arc::clone(a)));
            }
        }
        self.first.get().map(StateRef::First)
    }

    /// Is a value set?
    pub fn is_set(&self) -> bool {
        self.first.get().is_some()
    }

    /// Set the value (the first time) or replace it.
    pub fn set(&self, v: T) {
        if let Err(v) = self.first.set(v) {
            let mut g = self.replaced.lock().unwrap_or_else(PoisonError::into_inner);
            *g = Some(Arc::new(v));
            self.replaced_flag.store(true, Ordering::Release);
        }
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
        let p = Self { kind: AtomicU8::new(K_NONE), other: Mutex::new(None) };
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
            if truthy { K_OTHER_TRUTHY } else { K_OTHER_FALSY }
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
                g.as_ref().map_or_else(|| py.None().into_bound(py), |o| o.bind(py).clone())
            }
        }
    }
}
