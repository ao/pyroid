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
//! - Work below [`MIN_DOCS_PER_TASK`] documents, or with `threads = 1`, runs
//!   on the calling thread and never touches the pool.
//! - The pool is built on first use, not at import, and rebuilt in a forked
//!   child (worker threads do not survive `fork()`).

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, PoisonError};

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

/// The shared pool could not be started.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolError(String);

impl fmt::Display for PoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
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
                PoolError(format!(
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
            return Err(PoolError(format!(
                "{NUM_THREADS_ENV} must be a positive integer, got {v:?}"
            )));
        }
    };
    let n = parse_num_threads(value.as_deref())?;
    ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(|i| format!("pyroid-{i}"))
        .build()
        .map_err(|e| PoolError(format!("could not start pyroid's thread pool: {e}")))
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

/// `f(task)` for every element of `tasks`, each one rayon job on the shared
/// pool, results in order. For a few coarse tasks the caller has already
/// sized (e.g. [`max_tasks`] ranges of balanced work); with one task, or
/// `threads = 1`, they run on the calling thread. Make at most
/// `max_tasks(threads)` tasks: rayon cannot cap how many run at once.
pub fn map_tasks<T, R, F>(tasks: &[T], threads: Threads, f: F) -> Result<Vec<R>, PoolError>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    if tasks.len() <= 1 || threads == Threads::ONE {
        return Ok(tasks.iter().map(f).collect());
    }
    let pool = pool()?;
    Ok(pool.install(|| tasks.par_iter().with_max_len(1).map(f).collect()))
}

/// `None`: run serially on the calling thread. `Some(m)`: run on the pool with
/// tasks of at least `m` items.
fn plan(len: usize, threads: Threads) -> Option<usize> {
    let max_tasks = match threads {
        Threads::Auto => usize::MAX,
        Threads::Max(n) => n.get(),
    };
    if max_tasks == 1 || len <= MIN_DOCS_PER_TASK {
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
    match plan(len, threads) {
        None => (0..len).map(f).collect(),
        Some(min_len) => {
            let pool = pool()?;
            pool.install(|| {
                (0..len)
                    .into_par_iter()
                    .with_min_len(min_len)
                    .map(f)
                    .collect()
            })
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
    match plan(out.len() / row_len, threads) {
        None => out
            .chunks_mut(row_len)
            .enumerate()
            .map(|(i, row)| f(i, row))
            .collect(),
        Some(min_len) => {
            let pool = pool()?;
            pool.install(|| {
                out.par_chunks_mut(row_len)
                    .with_min_len(min_len)
                    .enumerate()
                    .map(|(i, row)| f(i, row))
                    .collect()
            })
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
    if plan(weight, threads).is_none() || items.len() <= 1 {
        items.into_iter().for_each(f);
        return Ok(());
    }
    let max_tasks = match threads {
        Threads::Auto => usize::MAX,
        Threads::Max(n) => n.get(),
    };
    let min_len = items.len().div_ceil(max_tasks).max(1);
    let pool = pool()?;
    pool.install(|| items.into_par_iter().with_min_len(min_len).for_each(f));
    Ok(())
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
        assert_eq!(plan(100, n(4)), Some(MIN_DOCS_PER_TASK));
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
                Err(PoolError(format!("bad {i}")))
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
                Err(PoolError("row 777".into()))
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

    #[test]
    fn pool_size_is_positive() {
        assert!(pool_size().unwrap() >= 1);
    }
}
