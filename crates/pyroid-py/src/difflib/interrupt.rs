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
//! Batches run in parts ([`in_parts`]) of about 50 ms each, with the checks
//! on for the items the calling thread runs itself and signals checked
//! between parts. An item on a pool thread is not stopped halfway.
//!
//! The crate-wide `interrupt` module (feat/fixes3) is not on main yet; when
//! it is, the batches can move to its `detach_batch`.

use std::cell::RefCell;
use std::ops::Range;
use std::time::{Duration, Instant};

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

/// About how long one part of a batch should take.
const PART: Duration = Duration::from_millis(50);

/// `part(range)` for consecutive ranges covering `0..n`, results in order,
/// with signals checked between parts. The first range has `first` items;
/// later ones are sized from how long the last took (about [`PART`] each,
/// at least `min` items). `part` runs its range with [`run`].
pub fn in_parts<R>(
    py: Python<'_>,
    n: usize,
    first: usize,
    min: usize,
    mut part: impl FnMut(Range<usize>) -> PyResult<Vec<R>>,
) -> PyResult<Vec<R>> {
    let min = min.max(1);
    let mut out = Vec::with_capacity(n);
    let (mut start, mut len) = (0, first.max(min));
    while start < n {
        let end = start + len.min(n - start);
        let t = Instant::now();
        out.extend(part(start..end)?);
        let took = t.elapsed().as_secs_f64();
        py.check_signals()?;
        // Aim for PART, changing the size at most 4x at a time.
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let next = (len as f64 * (PART.as_secs_f64() / took.max(1e-6)).clamp(0.25, 4.0)) as usize;
        (start, len) = (end, next.max(min));
    }
    Ok(out)
}
