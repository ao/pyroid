//! Ctrl-C during a batch.
//!
//! CPython runs a signal's Python handler (for SIGINT, the one that raises
//! `KeyboardInterrupt`) when the main thread next runs Python code. A batch
//! runs detached, so Ctrl-C used to take effect only once the batch had
//! finished. [`detach_batch`] runs a batch as a `par::interruptible` call:
//! every `par::POLL_INTERVAL` the calling thread attaches for a moment and
//! runs the pending handlers. If one raises, the batch stops at its next item
//! and the call raises that exception ([`pool_error`]).

use std::cell::RefCell;

use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::intern;
use pyo3::prelude::*;
use pyroid_core::par;

use crate::errors::PyroidError;

thread_local! {
    /// The exception a signal handler raised during this thread's batch.
    static PENDING: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

/// `par`'s interrupt check. `par` calls it only on the thread that called in
/// from Python, while it is detached, never on a pool worker.
fn check(first: bool) -> par::Poll {
    // The calling thread, between items or while it waits for the pool; no
    // pool thread ever gets here.
    #[allow(clippy::disallowed_methods)]
    Python::attach(|py| {
        let handles = if first { handles_signals(py) } else { Ok(true) };
        match handles.and_then(|h| py.check_signals().map(|()| h)) {
            Ok(true) => par::Poll::Continue,
            Ok(false) => par::Poll::Never,
            Err(e) => {
                PENDING.with(|p| *p.borrow_mut() = Some(e));
                par::Poll::Stop
            }
        }
    })
}

/// Does this thread run signal handlers? CPython runs them only on the main
/// thread (of the main interpreter); elsewhere checking can only cost time.
fn handles_signals(py: Python<'_>) -> PyResult<bool> {
    let threading = py.import(intern!(py, "threading"))?;
    let main = threading.call_method0(intern!(py, "main_thread"))?;
    let current = threading.call_method0(intern!(py, "current_thread"))?;
    Ok(main.is(&current))
}

/// `py.detach(f)` for a batch: its `par` maps stop once a signal handler
/// raises, and return `PoolError::interrupted()`, which [`pool_error`] turns
/// into that exception. `f` must not update shared state in a `par` map (an
/// interrupt would stop it halfway); such work goes in `par::uninterruptible`.
pub fn detach_batch<T, F>(py: Python<'_>, f: F) -> T
where
    F: Send + FnOnce() -> T,
    T: Send,
{
    PENDING.with(|p| p.borrow_mut().take());
    py.detach(|| par::interruptible(f))
}

/// The Python exception for a `PoolError`: what the signal handler raised for
/// an interrupted batch, else `PyroidError` (the pool could not start).
pub fn pool_error(e: &par::PoolError) -> PyErr {
    if e.is_interrupted() {
        PENDING
            .with(|p| p.borrow_mut().take())
            .unwrap_or_else(|| PyKeyboardInterrupt::new_err(()))
    } else {
        PyroidError::new_err(e.to_string())
    }
}

pub fn register() {
    par::set_interrupt_check(check);
}
