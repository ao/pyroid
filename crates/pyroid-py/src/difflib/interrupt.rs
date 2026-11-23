//! Ctrl-C during long `pyroid.difflib` calls.
//!
//! CPython runs a signal's Python handler (for SIGINT, the one that raises
//! `KeyboardInterrupt`) when the main thread next runs Python code, so a
//! native call that runs for a minute used to act on Ctrl-C only when it
//! ended. [`run`] turns on `pyroid_core::difflib`'s checks for the calling
//! thread: about every 100 ms of work its loops attach for a moment and run
//! the pending handlers; if one raises, the work stops early, its partial
//! result is dropped and the call raises that exception. Nothing the
//! instance holds is changed by a call that stops, so it stays usable.
//!
//! The batches (`ratio_batch`, `close_matches_batch`, `get_close_matches` over
//! long lists) run under `crate::interrupt::detach_batch` instead, which stops
//! their `par` maps between items.

use std::cell::RefCell;

use pyo3::exceptions::PyKeyboardInterrupt;
use pyo3::intern;
use pyo3::prelude::*;
use pyroid_core::difflib::stop::Poll;
use pyroid_core::difflib::{finish_checks, start_checks};

thread_local! {
    /// The exception a signal handler raised during this thread's call.
    static PENDING: RefCell<Option<PyErr>> = const { RefCell::new(None) };
}

/// The core's check: run the pending signal handlers.
fn check(first: bool) -> Poll {
    // Only on the thread that called in from Python, inside `run` (detached
    // or not); `start_checks` is never called on a pool thread.
    #[allow(clippy::disallowed_methods)]
    Python::attach(|py| {
        // The handlers first: is_main_thread runs Python code, which may run
        // them too, and what they raise must not be lost.
        let found = py.check_signals().and_then(|()| {
            // CPython runs signal handlers on the main thread only.
            Ok(!first || is_main_thread(py)?)
        });
        match found {
            Ok(true) => Poll::Continue,
            Ok(false) => Poll::Never,
            Err(e) => {
                PENDING.with(|p| *p.borrow_mut() = Some(e));
                Poll::Stop
            }
        }
    })
}

fn is_main_thread(py: Python<'_>) -> PyResult<bool> {
    let threading = py.import(intern!(py, "threading"))?;
    let main = threading
        .call_method0(intern!(py, "main_thread"))?
        .getattr(intern!(py, "ident"))?;
    main.eq(threading.call_method0(intern!(py, "get_ident"))?)
}

/// Turns the checks off if `f` panics.
struct Checks;

impl Drop for Checks {
    fn drop(&mut self) {
        finish_checks();
    }
}

/// `f()` with Ctrl-C checks, detached from the interpreter if `detach`.
/// If a signal handler raises while `f` runs, `f`'s result is dropped and
/// this returns that exception.
pub fn run<T, F>(py: Python<'_>, detach: bool, f: F) -> PyResult<T>
where
    F: Send + FnOnce() -> T,
    T: Send,
{
    PENDING.with(|p| p.borrow_mut().take());
    let checks = Checks;
    start_checks(check);
    let out = if detach { py.detach(f) } else { f() };
    std::mem::forget(checks);
    if finish_checks() {
        return Err(PENDING
            .with(|p| p.borrow_mut().take())
            .unwrap_or_else(|| PyKeyboardInterrupt::new_err(())));
    }
    Ok(out)
}
