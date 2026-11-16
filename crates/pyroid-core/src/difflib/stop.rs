//! Ctrl-C during a long call.
//!
//! The bindings turn checks on for the calling thread ([`super::start_checks`])
//! around a call that may run long, such as `get_opcodes()` on long
//! sequences or `ndiff` of large files. The matcher's and `Differ`'s loops
//! then [`Ticker::tick`] as they go; about every [`POLL_INTERVAL`] of work
//! the ticker asks the bindings' check (which runs the pending signal
//! handlers), and once that says stop every later tick says stop too, so
//! the loops end early. The caller then discards the partial result
//! ([`super::finish_checks`] says it stopped) and raises.
//!
//! Without checks a tick is one subtraction and one comparison.

use std::time::{Duration, Instant};

/// How often a long call asks the check: short enough that Ctrl-C feels
/// immediate, long enough that attaching to the interpreter costs nothing
/// measurable.
pub const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// What the bindings' check found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Poll {
    /// Nothing to act on: carry on.
    Continue,
    /// Stop the call (a signal handler raised).
    Stop,
    /// This thread never runs signal handlers: don't ask again in this call.
    Never,
}

/// The bindings' check: `check(first)`, `first` true for the call's first ask.
pub type Check = fn(bool) -> Poll;

/// Work units between clock reads, to start with (then adapted so that
/// reads are about a millisecond apart).
const FIRST_STRIDE: i64 = 1 << 14;

/// Interrupt checks for one call on one thread.
#[derive(Debug)]
pub struct Ticker {
    check: Option<Check>,
    /// Work units left before the next clock read.
    budget: i64,
    stride: i64,
    last: Option<Instant>,
    due: Option<Instant>,
    asked: bool,
    stopped: bool,
}

impl Default for Ticker {
    fn default() -> Self {
        Self::OFF
    }
}

impl Ticker {
    /// No checks.
    pub const OFF: Self = Self {
        check: None,
        budget: i64::MAX,
        stride: FIRST_STRIDE,
        last: None,
        due: None,
        asked: false,
        stopped: false,
    };

    /// Checks with `check` from now on.
    pub fn start(&mut self, check: Check) {
        *self = Self {
            check: Some(check),
            budget: FIRST_STRIDE,
            ..Self::OFF
        };
    }

    /// No more checks; true if the call was stopped.
    pub fn finish(&mut self) -> bool {
        let stopped = self.stopped;
        *self = Self::OFF;
        stopped
    }

    /// Account for `cost` units of work: true if the call should stop now.
    #[inline]
    pub fn tick(&mut self, cost: usize) -> bool {
        #[allow(clippy::cast_possible_wrap)] // costs are far below 2**63
        {
            self.budget = self.budget.saturating_sub(cost as i64);
        }
        self.budget < 0 && self.slow()
    }

    /// Has the call been stopped?
    #[inline]
    pub fn stopped(&self) -> bool {
        self.stopped
    }

    #[cold]
    fn slow(&mut self) -> bool {
        if self.stopped {
            self.budget = -1;
            return true;
        }
        let Some(check) = self.check else {
            self.budget = i64::MAX;
            return false;
        };
        let now = Instant::now();
        if let Some(last) = self.last {
            let gap = now - last;
            if gap < Duration::from_micros(500) {
                self.stride = self.stride.saturating_mul(2).min(1 << 40);
            } else if gap > Duration::from_millis(5) {
                self.stride = (self.stride / 2).max(1);
            }
        }
        self.last = Some(now);
        self.budget = self.stride;
        let due = *self.due.get_or_insert(now + POLL_INTERVAL);
        if now < due {
            return false;
        }
        let first = !self.asked;
        self.asked = true;
        match check(first) {
            Poll::Continue => self.due = Some(Instant::now() + POLL_INTERVAL),
            Poll::Never => {
                self.check = None;
                self.budget = i64::MAX;
            }
            Poll::Stop => {
                self.stopped = true;
                self.budget = -1;
            }
        }
        self.stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ASKED: AtomicUsize = AtomicUsize::new(0);

    fn stop_on_second(_first: bool) -> Poll {
        if ASKED.fetch_add(1, Ordering::Relaxed) == 0 {
            Poll::Continue
        } else {
            Poll::Stop
        }
    }

    fn never(_first: bool) -> Poll {
        ASKED.fetch_add(1, Ordering::Relaxed);
        Poll::Never
    }

    fn spin(t: &mut Ticker, limit: Duration) -> bool {
        let start = Instant::now();
        while start.elapsed() < limit {
            if t.tick(1) {
                return true;
            }
        }
        false
    }

    #[test]
    fn off_never_stops() {
        let mut t = Ticker::OFF;
        for _ in 0..1_000_000 {
            assert!(!t.tick(1 << 20));
        }
        assert!(!t.finish());
    }

    #[test]
    fn stops_once_the_check_says_so() {
        // One test owns ASKED at a time.
        ASKED.store(0, Ordering::Relaxed);
        let mut t = Ticker::default();
        t.start(stop_on_second);
        assert!(spin(&mut t, Duration::from_secs(10)));
        assert!(t.stopped() && t.tick(1) && t.tick(0));
        assert!(t.finish());
        assert!(!t.tick(1) && !t.stopped());
        assert_eq!(ASKED.load(Ordering::Relaxed), 2);

        ASKED.store(0, Ordering::Relaxed);
        t.start(never);
        assert!(!spin(&mut t, 3 * POLL_INTERVAL));
        assert_eq!(ASKED.load(Ordering::Relaxed), 1);
        assert!(!t.finish());
    }
}
