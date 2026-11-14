//! Batch parallelism: one lazily built, fork-aware rayon pool shared by every
//! kernel, plus order-preserving maps over `0..len`.
//!
//! Rules that keep this deadlock-free (see A1 §5 C4):
//! - Closures passed here are pure Rust. They never call into Python; the
//!   bindings run them while detached from the interpreter. One documented
//!   exception: argument extraction on free-threaded builds
//!   (`pyroid-py/src/input.rs`, `backed_range`) attaches pool workers briefly.
//! - So the bindings must never wait for the pool while attached: a
//!   stop-the-world request would then block the workers' attach while the
//!   waiting thread never reaches a safe point (a deadlock, not just latency).
//! - Work too small for two tasks of [`MIN_DOCS_PER_TASK`] documents, or with
//!   `threads = 1`, runs on the calling thread and never touches the pool.
//! - The pool is built on first use, not at import, and rebuilt in a forked
//!   child (worker threads do not survive `fork()`).
//!
//! Interrupts (Ctrl-C): inside [`interruptible`], the thread that called in
//! from Python asks the bindings' check (see [`set_interrupt_check`]) every
//! [`POLL_INTERVAL`] whether to stop: between items when it does the work
//! itself, and while it waits when the pool does. Once the check says stop,
//! the remaining items are skipped (pool workers see a flag before each item)
//! and the map returns [`PoolError::interrupted`]. Work that must not stop
//! halfway, such as filling an index's tables, runs in [`uninterruptible`].

use std::cell::Cell;
use std::fmt;
use std::num::NonZeroUsize;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Minimum number of documents per rayon task.
///
/// Rayon fork/join costs tens of microseconds, and per-document work is
/// microseconds, so tasks smaller than this cost more than they save.
pub const MIN_DOCS_PER_TASK: usize = 64;

/// Environment variable that sets the pool size. It is read once, when the
/// pool is first built. Unset means rayon's default (`RAYON_NUM_THREADS`, else
/// the number of available CPUs).
pub const NUM_THREADS_ENV: &str = "PYROID_NUM_THREADS";

/// How many threads a batch call may use (the Python `threads=` argument).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Threads {
    /// `threads=None`: use the whole shared pool.
    #[default]
    Auto,
    /// `threads=n`: run at most `n` tasks at once on the shared pool (so at
    /// most `min(n, pool size)` threads). `1` runs on the calling thread.
    Max(NonZeroUsize),
}

impl Threads {
    /// Single-threaded: run on the calling thread.
    pub const ONE: Self = Self::Max(NonZeroUsize::MIN);
}

/// The shared pool could not be started, or the call was interrupted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolError(Kind);

#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    Start(String),
    Interrupted,
}

impl PoolError {
    fn start(msg: String) -> Self {
        Self(Kind::Start(msg))
    }

    /// The error of a map stopped by the interrupt check.
    #[must_use]
    pub const fn interrupted() -> Self {
        Self(Kind::Interrupted)
    }

    /// Did the interrupt check stop the call (rather than the pool fail)?
    #[must_use]
    pub const fn is_interrupted(&self) -> bool {
        matches!(self.0, Kind::Interrupted)
    }
}

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Kind::Start(msg) => f.write_str(msg),
            Kind::Interrupted => f.write_str("interrupted"),
        }
    }
}

impl std::error::Error for PoolError {}

struct Pool {
    pid: u32,
    pool: Arc<ThreadPool>,
}

static POOL: Mutex<Option<Pool>> = Mutex::new(None);

/// Parse the value of [`NUM_THREADS_ENV`]; `None` means "unset".
fn parse_num_threads(value: Option<&str>) -> Result<usize, PoolError> {
    match value {
        None => Ok(0), // 0 = rayon's default
        Some(v) => v
            .trim()
            .parse::<NonZeroUsize>()
            .map(NonZeroUsize::get)
            .map_err(|_| {
                PoolError::start(format!(
                    "{NUM_THREADS_ENV} must be a positive integer, got {v:?}"
                ))
            }),
    }
}

fn build_pool() -> Result<ThreadPool, PoolError> {
    let value = match std::env::var(NUM_THREADS_ENV) {
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(v)) => {
            return Err(PoolError::start(format!(
                "{NUM_THREADS_ENV} must be a positive integer, got {v:?}"
            )));
        }
    };
    let n = parse_num_threads(value.as_deref())?;
    ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("pyroid-{i}"))
        .build()
        .map_err(|e| PoolError::start(format!("could not start pyroid's thread pool: {e}")))
}

/// The shared pool, built on first use in this process.
fn pool() -> Result<Arc<ThreadPool>, PoolError> {
    let pid = std::process::id();
    // The lock is held only to read or install the pool; a panic cannot leave
    // the `Option` half-written, so a poisoned lock is still usable.
    let mut guard = POOL.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(p) = guard.as_ref().filter(|p| p.pid == pid) {
        return Ok(Arc::clone(&p.pool));
    }
    // First use, or first use in a child after fork(): the parent's workers
    // don't exist here, so its pool would never run a job. Leak it rather than
    // drop it: dropping would signal threads (and possibly locks) that the
    // child never inherited.
    if let Some(stale) = guard.take() {
        std::mem::forget(stale);
    }
    let pool = Arc::new(build_pool()?);
    *guard = Some(Pool {
        pid,
        pool: Arc::clone(&pool),
    });
    Ok(pool)
}

/// Number of threads in the shared pool (builds it if needed).
pub fn pool_size() -> Result<usize, PoolError> {
    Ok(pool()?.current_num_threads())
}

/// How many tasks a batch may run at once: the pool size, capped by
/// `threads` (builds the pool if needed).
pub fn max_tasks(threads: Threads) -> Result<usize, PoolError> {
    let size = pool_size()?;
    Ok(match threads {
        Threads::Auto => size,
        Threads::Max(n) => n.get().min(size),
    })
}

/// How often the calling thread of an [`interruptible`] call asks the
/// bindings' interrupt check whether to stop. Short enough that Ctrl-C feels
/// immediate; long enough that, with the GIL, the wait to attach (up to the
/// interpreter's 5 ms switch interval while another thread runs Python) costs
/// a serial batch at most a few percent.
pub const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// What the bindings' interrupt check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poll {
    /// Nothing to act on: carry on.
    Continue,
    /// Stop the call (a signal handler raised).
    Stop,
    /// This thread never runs signal handlers: don't ask again in this call.
    Never,
}

/// The interrupt check: `check(first)`, where `first` is true for the first
/// check of a call. Set once, by the bindings.
static INTERRUPT_CHECK: OnceLock<fn(bool) -> Poll> = OnceLock::new();

/// Install the interrupt check that [`interruptible`] calls use. Later calls
/// are ignored.
pub fn set_interrupt_check(check: fn(bool) -> Poll) {
    // Only the first check is kept: there is one set of bindings per process.
    let _ = INTERRUPT_CHECK.set(check);
}

/// Interrupt state of the thread that called in, for the current call.
#[derive(Clone, Copy)]
struct Call {
    /// Nesting depth of [`interruptible`]; 0 outside a call.
    depth: u32,
    /// Nesting depth of [`uninterruptible`].
    paused: u32,
    /// When the next check is due.
    next: Option<Instant>,
    /// Has this call checked yet?
    checked: bool,
    /// The check said [`Poll::Never`].
    never: bool,
    /// The check said [`Poll::Stop`]: every later map of the call stops too.
    stopped: bool,
}

impl Call {
    const IDLE: Self = Self {
        depth: 0,
        paused: 0,
        next: None,
        checked: false,
        never: false,
        stopped: false,
    };

    /// Do this call's maps stop for interrupts (they check, or have stopped)?
    fn active(self) -> bool {
        self.depth > 0
            && self.paused == 0
            && (self.stopped || (!self.never && INTERRUPT_CHECK.get().is_some()))
    }

    /// Does this call still ask the check?
    fn polling(self) -> bool {
        self.active() && !self.stopped
    }
}

thread_local! {
    static CALL: Cell<Call> = const { Cell::new(Call::IDLE) };
}

/// Restores the thread's [`Call`] on drop (also when unwinding).
struct Restore(Call);

impl Drop for Restore {
    fn drop(&mut self) {
        let now = CALL.get();
        let outer = self.0;
        CALL.set(if outer.depth == 0 {
            Call::IDLE
        } else {
            // Keep what the inner part learned; restore the nesting.
            Call {
                depth: outer.depth,
                paused: outer.paused,
                ..now
            }
        });
    }
}

/// Run `f` as a call that can be interrupted: maps on this thread check the
/// interrupt check every [`POLL_INTERVAL`], and return
/// [`PoolError::interrupted`] once it says stop. For the thread that called in
/// from Python, detached, around a batch's work.
pub fn interruptible<T>(f: impl FnOnce() -> T) -> T {
    let outer = CALL.get();
    CALL.set(if outer.depth == 0 {
        Call {
            depth: 1,
            next: Some(Instant::now() + POLL_INTERVAL),
            ..Call::IDLE
        }
    } else {
        Call {
            depth: outer.depth + 1,
            ..outer
        }
    });
    let _restore = Restore(outer);
    f()
}

/// Run `f` with no interrupt checks: for work that must not stop halfway
/// (state updated in place, such as an index's tables).
pub fn uninterruptible<T>(f: impl FnOnce() -> T) -> T {
    let outer = CALL.get();
    CALL.set(Call {
        paused: outer.paused + 1,
        ..outer
    });
    let _restore = Restore(outer);
    f()
}

/// Has the current call been stopped? Asks the check if one is due at `now`.
fn stop_due(now: Instant) -> bool {
    let call = CALL.get();
    if !call.active() {
        return false;
    }
    if call.stopped {
        return true;
    }
    if call.next.is_some_and(|t| now < t) {
        return false;
    }
    let Some(check) = INTERRUPT_CHECK.get() else {
        return false;
    };
    let poll = check(!call.checked);
    let mut call = CALL.get();
    call.checked = true;
    call.next = Some(Instant::now() + POLL_INTERVAL);
    match poll {
        Poll::Continue => {}
        Poll::Stop => call.stopped = true,
        Poll::Never => call.never = true,
    }
    CALL.set(call);
    call.stopped
}

/// Was the current call stopped?
fn stopped() -> bool {
    let call = CALL.get();
    call.active() && call.stopped
}

/// Interrupt checks for a loop on the calling thread: [`Ticker::tick`] once
/// per item reads the clock only every `stride` items, with `stride` adapted
/// so that reads are about a millisecond apart.
struct Ticker {
    on: bool,
    left: u32,
    stride: u32,
    last: Option<Instant>,
}

impl Ticker {
    fn new() -> Self {
        Self {
            on: CALL.get().active(),
            left: 1,
            stride: 1,
            last: None,
        }
    }

    /// Should the loop stop before its next item?
    #[inline]
    fn tick(&mut self) -> bool {
        if !self.on {
            return false;
        }
        self.left -= 1;
        self.left == 0 && self.slow()
    }

    #[cold]
    fn slow(&mut self) -> bool {
        let now = Instant::now();
        if let Some(last) = self.last {
            let gap = now - last;
            if gap < Duration::from_micros(500) {
                self.stride = (self.stride * 2).min(1 << 16);
            } else if gap > Duration::from_millis(5) {
                self.stride = (self.stride / 2).max(1);
            }
        }
        self.last = Some(now);
        self.left = self.stride;
        stop_due(now)
    }
}

/// Run `job` on `pool`. In an interruptible call the calling thread waits for
/// it in slices of [`POLL_INTERVAL`] and checks in between; once the check
/// says stop it raises the flag `job`'s items read, then waits for `job` to
/// end (it borrows the caller's data), and the caller returns
/// [`PoolError::interrupted`]. Elsewhere this is `pool.install(|| job(..))`.
fn run_on<R, J>(pool: &ThreadPool, job: J) -> R
where
    R: Send,
    J: FnOnce(&AtomicBool) -> R + Send,
{
    let cancel = AtomicBool::new(false);
    if !CALL.get().polling() {
        return pool.install(|| job(&cancel));
    }
    // From here on the calling thread only waits and checks; the job runs on
    // the pool as `install` would run it.
    let done: Mutex<Option<std::thread::Result<R>>> = Mutex::new(None);
    let ready = Condvar::new();
    pool.in_place_scope(|s| {
        s.spawn(|_| {
            // Catch a panic so the waiter below is always woken; it is
            // re-raised on the calling thread.
            let r = panic::catch_unwind(AssertUnwindSafe(|| job(&cancel)));
            *done.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
            ready.notify_one();
        });
        let mut slot = done.lock().unwrap_or_else(PoisonError::into_inner);
        while slot.is_none() {
            let call = CALL.get();
            if cancel.load(Ordering::Relaxed) || !call.polling() {
                slot = ready.wait(slot).unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            let wait = call.next.map_or(POLL_INTERVAL, |t| {
                t.saturating_duration_since(Instant::now())
            });
            slot = ready
                .wait_timeout(slot, wait)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
            if slot.is_none() {
                // Don't hold the lock while the check attaches to Python.
                drop(slot);
                if stop_due(Instant::now()) {
                    cancel.store(true, Ordering::Relaxed);
                }
                slot = done.lock().unwrap_or_else(PoisonError::into_inner);
            }
        }
    });
    let r = done
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .expect("the job stored its result before the scope ended");
    r.unwrap_or_else(|payload| panic::resume_unwind(payload))
}

/// Lets a long task stop early when its call is interrupted: check
/// [`Stop::requested`] between the task's items.
#[derive(Clone, Copy, Debug)]
pub enum Stop<'a> {
    /// The task runs on the calling thread, which checks itself.
    Here,
    /// The task runs on the pool; the calling thread raises the flag.
    Flag(&'a AtomicBool),
}

impl Stop<'_> {
    /// Should the task stop before its next item? Once it is true, the map
    /// returns [`PoolError::interrupted`] whatever the tasks return.
    #[must_use]
    pub fn requested(self) -> bool {
        match self {
            Self::Here => stop_due(Instant::now()),
            Self::Flag(flag) => flag.load(Ordering::Relaxed),
        }
    }
}

/// `f(task)` for every element of `tasks`, each one rayon job on the shared
/// pool, results in order. For a few coarse tasks the caller has already
/// sized (e.g. [`max_tasks`] ranges of balanced work); with one task, or
/// `threads = 1`, they run on the calling thread. Make at most
/// `max_tasks(threads)` tasks: rayon cannot cap how many run at once. An
/// interrupt stops the map between tasks (see [`map_tasks_stoppable`] to stop
/// within one).
pub fn map_tasks<T, R, F>(tasks: &[T], threads: Threads, f: F) -> Result<Vec<R>, PoolError>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    map_tasks_stoppable(tasks, threads, |t, _| f(t))
}

/// [`map_tasks`] whose tasks get a [`Stop`] to check between their own items.
pub fn map_tasks_stoppable<T, R, F>(
    tasks: &[T],
    threads: Threads,
    f: F,
) -> Result<Vec<R>, PoolError>
where
    T: Sync,
    R: Send,
    F: Fn(&T, Stop<'_>) -> R + Sync + Send,
{
    if stopped() {
        return Err(PoolError::interrupted());
    }
    if tasks.len() <= 1 || threads == Threads::ONE {
        let mut ticker = Ticker::new();
        let out = tasks
            .iter()
            .map(|t| {
                if ticker.tick() {
                    Err(PoolError::interrupted())
                } else {
                    Ok(f(t, Stop::Here))
                }
            })
            .collect::<Result<Vec<R>, PoolError>>()?;
        return if stopped() {
            Err(PoolError::interrupted())
        } else {
            Ok(out)
        };
    }
    let pool = pool()?;
    let out = run_on(&pool, |cancel| {
        tasks
            .par_iter()
            .with_max_len(1)
            .map(|t| (!cancel.load(Ordering::Relaxed)).then(|| f(t, Stop::Flag(cancel))))
            .collect::<Option<Vec<R>>>()
    });
    out.filter(|_| !stopped())
        .ok_or_else(PoolError::interrupted)
}

/// `None`: run serially on the calling thread. `Some(m)`: run on the pool with
/// tasks of at least `m` items. Fewer than two tasks' worth of items stays on the
/// calling thread: one task on the pool would only add the round trip.
fn plan(len: usize, threads: Threads) -> Option<usize> {
    let max_tasks = match threads {
        Threads::Auto => usize::MAX,
        Threads::Max(n) => n.get(),
    };
    if max_tasks == 1 || len < 2 * MIN_DOCS_PER_TASK {
        return None;
    }
    // Tasks of >= ceil(len / n) items means at most n tasks.
    Some(MIN_DOCS_PER_TASK.max(len.div_ceil(max_tasks)))
}

/// `f(0), f(1), ..., f(len - 1)` in order, in parallel per `threads`.
///
/// Stops early on the first error (which error is reported is unspecified if
/// several items fail). Panics in `f` propagate to the caller.
pub fn try_map<R, E, F>(len: usize, threads: Threads, f: F) -> Result<Vec<R>, E>
where
    R: Send,
    E: Send + From<PoolError>,
    F: Fn(usize) -> Result<R, E> + Sync + Send,
{
    if stopped() {
        return Err(PoolError::interrupted().into());
    }
    match plan(len, threads) {
        None => {
            let mut ticker = Ticker::new();
            (0..len)
                .map(|i| {
                    if ticker.tick() {
                        Err(PoolError::interrupted().into())
                    } else {
                        f(i)
                    }
                })
                .collect()
        }
        Some(min_len) => {
            let pool = pool()?;
            let out = run_on(&pool, |cancel| {
                (0..len)
                    .into_par_iter()
                    .with_min_len(min_len)
                    .map(|i| {
                        if cancel.load(Ordering::Relaxed) {
                            Err(PoolError::interrupted().into())
                        } else {
                            f(i)
                        }
                    })
                    .collect()
            });
            if stopped() {
                return Err(PoolError::interrupted().into());
            }
            out
        }
    }
}

/// Infallible [`try_map`].
pub fn map<R, F>(len: usize, threads: Threads, f: F) -> Result<Vec<R>, PoolError>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    try_map(len, threads, |i| Ok::<R, PoolError>(f(i)))
}

/// `f(i, row_i)` for every `row_len`-element row of `out`, in order, in
/// parallel per `threads`: each document writes its result in place (no
/// per-document allocation, no copy afterwards). Errors as in [`try_map`].
///
/// # Panics
/// If `row_len` is 0 or does not divide `out.len()`.
pub fn try_map_rows<T, R, E, F>(
    out: &mut [T],
    row_len: usize,
    threads: Threads,
    f: F,
) -> Result<Vec<R>, E>
where
    T: Send,
    R: Send,
    E: Send + From<PoolError>,
    F: Fn(usize, &mut [T]) -> Result<R, E> + Sync + Send,
{
    assert!(
        row_len > 0 && out.len() % row_len == 0,
        "rows of {row_len} over {} values",
        out.len()
    );
    if stopped() {
        return Err(PoolError::interrupted().into());
    }
    match plan(out.len() / row_len, threads) {
        None => {
            let mut ticker = Ticker::new();
            out.chunks_mut(row_len)
                .enumerate()
                .map(|(i, row)| {
                    if ticker.tick() {
                        Err(PoolError::interrupted().into())
                    } else {
                        f(i, row)
                    }
                })
                .collect()
        }
        Some(min_len) => {
            let pool = pool()?;
            let res = run_on(&pool, |cancel| {
                out.par_chunks_mut(row_len)
                    .with_min_len(min_len)
                    .enumerate()
                    .map(|(i, row)| {
                        if cancel.load(Ordering::Relaxed) {
                            Err(PoolError::interrupted().into())
                        } else {
                            f(i, row)
                        }
                    })
                    .collect()
            });
            if stopped() {
                return Err(PoolError::interrupted().into());
            }
            res
        }
    }
}

/// Run `f` on every element of `items` (consuming them), in parallel per
/// `threads`. `weight` is the number of documents the items cover; it decides
/// whether the pool is worth using. Each item is one task.
pub fn for_each<T, F>(items: Vec<T>, weight: usize, threads: Threads, f: F) -> Result<(), PoolError>
where
    T: Send,
    F: Fn(T) + Sync + Send,
{
    if stopped() {
        return Err(PoolError::interrupted());
    }
    if plan(weight, threads).is_none() || items.len() <= 1 {
        let mut ticker = Ticker::new();
        for item in items {
            if ticker.tick() {
                return Err(PoolError::interrupted());
            }
            f(item);
        }
        return Ok(());
    }
    let max_tasks = match threads {
        Threads::Auto => usize::MAX,
        Threads::Max(n) => n.get(),
    };
    let min_len = items.len().div_ceil(max_tasks).max(1);
    let pool = pool()?;
    run_on(&pool, |cancel| {
        items
            .into_par_iter()
            .with_min_len(min_len)
            .for_each(|item| {
                if !cancel.load(Ordering::Relaxed) {
                    f(item);
                }
            });
    });
    if stopped() {
        Err(PoolError::interrupted())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn n(v: usize) -> Threads {
        Threads::Max(NonZeroUsize::new(v).unwrap())
    }

    #[test]
    fn parse_env() {
        assert_eq!(parse_num_threads(None), Ok(0));
        assert_eq!(parse_num_threads(Some("4")), Ok(4));
        assert_eq!(parse_num_threads(Some(" 2 ")), Ok(2));
        for bad in ["0", "-1", "abc", "", "1.5"] {
            let err = parse_num_threads(Some(bad)).unwrap_err();
            assert!(err.to_string().contains(NUM_THREADS_ENV), "{err}");
        }
    }

    #[test]
    fn plan_serial_and_parallel() {
        assert_eq!(plan(10, Threads::Auto), None);
        assert_eq!(plan(MIN_DOCS_PER_TASK, Threads::Auto), None);
        assert_eq!(plan(10_000, Threads::ONE), None);
        assert_eq!(plan(10_000, Threads::Auto), Some(MIN_DOCS_PER_TASK));
        assert_eq!(plan(10_000, n(4)), Some(2_500));
        assert_eq!(plan(10_001, n(4)), Some(2_501));
        // One task's worth (65..=127 items) stays on the calling thread.
        assert_eq!(plan(100, n(4)), None);
        assert_eq!(plan(2 * MIN_DOCS_PER_TASK - 1, Threads::Auto), None);
        assert_eq!(
            plan(2 * MIN_DOCS_PER_TASK, Threads::Auto),
            Some(MIN_DOCS_PER_TASK)
        );
        assert_eq!(plan(2 * MIN_DOCS_PER_TASK, n(2)), Some(MIN_DOCS_PER_TASK));
    }

    #[test]
    fn map_preserves_order_for_every_thread_setting() {
        let want: Vec<usize> = (0..10_000).map(|i| i * 3).collect();
        for t in [Threads::Auto, Threads::ONE, n(2), n(3), n(64)] {
            assert_eq!(map(10_000, t, |i| i * 3).unwrap(), want, "{t:?}");
        }
        assert_eq!(map(0, Threads::Auto, |i| i).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn try_map_reports_errors() {
        let r: Result<Vec<usize>, PoolError> = try_map(5_000, Threads::Auto, |i| {
            if i == 4_321 {
                Err(PoolError::start(format!("bad {i}")))
            } else {
                Ok(i)
            }
        });
        assert_eq!(r.unwrap_err().to_string(), "bad 4321");
    }

    #[test]
    fn threads_max_caps_concurrency() {
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        map(4_096, n(2), |_| {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_micros(20));
            active.fetch_sub(1, Ordering::SeqCst);
        })
        .unwrap();
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    #[test]
    fn worker_panic_propagates_and_pool_survives() {
        let r = std::panic::catch_unwind(|| {
            map(10_000, Threads::Auto, |i| assert!(i != 7_777, "boom")).unwrap();
        });
        assert!(r.is_err());
        assert_eq!(map(1_000, Threads::Auto, |i| i).unwrap().len(), 1_000);
    }

    #[test]
    fn try_map_rows_writes_in_place_in_order() {
        for t in [Threads::Auto, Threads::ONE, n(3)] {
            let mut out = vec![0_u64; 3 * 5_000];
            let r: Result<Vec<usize>, PoolError> = try_map_rows(&mut out, 3, t, |i, row| {
                row.fill(i as u64);
                Ok(i)
            });
            assert_eq!(r.unwrap(), (0..5_000).collect::<Vec<_>>(), "{t:?}");
            assert!(
                out.chunks(3)
                    .enumerate()
                    .all(|(i, row)| row == [i as u64; 3])
            );
        }
        let mut out = vec![0_u8; 4 * 1_000];
        let r: Result<Vec<()>, PoolError> = try_map_rows(&mut out, 4, Threads::Auto, |i, _| {
            if i == 777 {
                Err(PoolError::start("row 777".into()))
            } else {
                Ok(())
            }
        });
        assert_eq!(r.unwrap_err().to_string(), "row 777");
    }

    #[test]
    fn for_each_visits_everything() {
        let sum = AtomicUsize::new(0);
        let items: Vec<usize> = (1..=100).collect();
        for_each(items, 10_000, Threads::Auto, |x| {
            sum.fetch_add(x, Ordering::Relaxed);
        })
        .unwrap();
        assert_eq!(sum.load(Ordering::Relaxed), 5_050);
    }

    #[test]
    fn map_tasks_in_order() {
        let tasks: Vec<std::ops::Range<usize>> = (0..7).map(|i| i * 10..i * 10 + 10).collect();
        for t in [Threads::Auto, Threads::ONE, n(3)] {
            let sums = map_tasks(&tasks, t, |r| r.clone().sum::<usize>()).unwrap();
            assert_eq!(
                sums,
                (0..7)
                    .map(|i| (i * 10..i * 10 + 10).sum())
                    .collect::<Vec<usize>>()
            );
        }
        assert!(
            max_tasks(n(1)).unwrap() == 1
                && max_tasks(Threads::Auto).unwrap() == pool_size().unwrap()
        );
    }

    // The interrupt check is process-wide (set once); each test thread says
    // through `MODE` what it should answer, and counts the checks.
    thread_local! {
        static MODE: Cell<Poll> = const { Cell::new(Poll::Continue) };
        static STOP_AT: Cell<usize> = const { Cell::new(usize::MAX) };
        static CHECKS: Cell<usize> = const { Cell::new(0) };
    }

    fn test_check(first: bool) -> Poll {
        let n = CHECKS.get() + 1;
        CHECKS.set(n);
        if first && MODE.get() == Poll::Never {
            return Poll::Never;
        }
        if n >= STOP_AT.get() {
            Poll::Stop
        } else {
            Poll::Continue
        }
    }

    fn arm(mode: Poll, stop_at: usize) {
        set_interrupt_check(test_check);
        MODE.set(mode);
        STOP_AT.set(stop_at);
        CHECKS.set(0);
    }

    /// Items of ~0.2 ms: 20,000 of them take 4 s serially.
    fn slow_item(done: &AtomicUsize, i: usize) -> usize {
        std::thread::sleep(Duration::from_micros(200));
        done.fetch_add(1, Ordering::Relaxed);
        i
    }

    #[test]
    fn interrupt_stops_a_serial_map() {
        arm(Poll::Continue, 2);
        let done = AtomicUsize::new(0);
        let t0 = Instant::now();
        let r = interruptible(|| map(20_000, Threads::ONE, |i| slow_item(&done, i)));
        assert!(r.unwrap_err().is_interrupted());
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        assert!(done.load(Ordering::Relaxed) < 20_000);
        assert_eq!(CHECKS.get(), 2);
        // The call is over: the next one runs to the end.
        assert_eq!(map(1_000, Threads::ONE, |i| i).unwrap().len(), 1_000);
    }

    #[test]
    fn interrupt_stops_the_pool() {
        arm(Poll::Continue, 2);
        let done = AtomicUsize::new(0);
        let t0 = Instant::now();
        // 20,000 items over 2 threads: 2 s uninterrupted.
        let r = interruptible(|| map(20_000, n(2), |i| slow_item(&done, i)));
        assert!(r.unwrap_err().is_interrupted());
        assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
        assert!(done.load(Ordering::Relaxed) < 20_000);
        // Each call starts afresh, on the whole pool too.
        arm(Poll::Continue, 2);
        let t0 = Instant::now();
        let r: Result<Vec<usize>, PoolError> =
            interruptible(|| try_map(2_000_000, Threads::Auto, |i| Ok(slow_item(&done, i))));
        assert!(r.unwrap_err().is_interrupted());
        assert!(t0.elapsed() < Duration::from_secs(1), "{:?}", t0.elapsed());
    }

    #[test]
    fn a_stopped_call_stops_every_later_map() {
        arm(Poll::Continue, 1);
        let done = AtomicUsize::new(0);
        interruptible(|| {
            assert!(
                map(20_000, Threads::ONE, |i| slow_item(&done, i))
                    .unwrap_err()
                    .is_interrupted()
            );
            assert!(map(10, Threads::ONE, |i| i).unwrap_err().is_interrupted());
            assert!(
                for_each(vec![1, 2, 3], 3, Threads::ONE, |_| ())
                    .unwrap_err()
                    .is_interrupted()
            );
            // ... except work that must finish.
            assert_eq!(
                uninterruptible(|| map(10, Threads::Auto, |i| i))
                    .unwrap()
                    .len(),
                10
            );
        });
        assert_eq!(map(10, Threads::ONE, |i| i).unwrap().len(), 10);
    }

    #[test]
    fn never_means_one_check_per_call() {
        arm(Poll::Never, 1);
        let done = AtomicUsize::new(0);
        let r = interruptible(|| map(2_000, Threads::ONE, |i| slow_item(&done, i)));
        assert_eq!(r.unwrap().len(), 2_000);
        assert_eq!(CHECKS.get(), 1);
    }

    #[test]
    fn uninterruptible_work_is_never_stopped() {
        arm(Poll::Continue, 1);
        let done = AtomicUsize::new(0);
        let r = interruptible(|| uninterruptible(|| map(1_000, n(2), |i| slow_item(&done, i))));
        assert_eq!(r.unwrap().len(), 1_000);
        assert_eq!(CHECKS.get(), 0);
    }

    #[test]
    fn short_calls_never_check() {
        arm(Poll::Continue, 1);
        let r = interruptible(|| map(100_000, Threads::Auto, |i| i * 2));
        assert_eq!(r.unwrap().len(), 100_000);
        assert_eq!(CHECKS.get(), 0);
    }

    #[test]
    fn a_panic_in_an_interruptible_pool_map_propagates() {
        arm(Poll::Continue, usize::MAX);
        let r = std::panic::catch_unwind(|| {
            interruptible(|| map(10_000, Threads::Auto, |i| assert!(i != 7_777, "boom")))
        });
        assert!(r.is_err());
        assert_eq!(CALL.get().depth, 0);
        assert_eq!(map(1_000, Threads::Auto, |i| i).unwrap().len(), 1_000);
    }

    #[test]
    fn pool_size_is_positive() {
        assert!(pool_size().unwrap() >= 1);
    }
}
