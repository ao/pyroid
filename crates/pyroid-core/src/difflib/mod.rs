//! CPython's `difflib`, ported exactly (`pyroid.difflib`).
//!
//! - [`matcher`]: `SequenceMatcher`'s algorithm over dense element ids:
//!   `b2j`, junk and popular elements, `find_longest_match`,
//!   `get_matching_blocks`, `get_opcodes`, `get_grouped_opcodes`, the ratios.
//! - [`intern`]: element ids for code points and other keys.
//! - [`differ`]: `Differ.compare` / `ndiff`, in CPython 3.11-3.13's and 3.14+'s versions.
//! - [`mdiff`]: `_mdiff`, the side-by-side rows of `HtmlDiff`.
//! - [`format`]: `unified_diff` and `context_diff`.
//! - [`html`]: `HtmlDiff.make_table`'s rows.
//!
//! [`StrMatcher`] is `SequenceMatcher` over the characters of two strings,
//! for `get_close_matches` and the batch functions.

pub mod differ;
pub mod format;
pub mod html;
pub mod intern;
pub mod matcher;
pub mod mdiff;
pub mod stop;

#[cfg(test)]
mod tests;

use std::cell::RefCell;

use intern::CharInterner;
pub use matcher::Work;
use matcher::{BIndex, calculate_ratio, real_quick_ratio};

/// `SequenceMatcher(isjunk, autojunk=autojunk)` after `set_seq2(b)`, where
/// `b` is a string given as code points.
#[derive(Debug, Clone)]
pub struct StrMatcher {
    chars: CharInterner,
    index: BIndex,
}

impl StrMatcher {
    /// `isjunk(c)` is called once per distinct character of `b`, in order.
    pub fn new(b: &[u32], isjunk: impl Fn(u32) -> bool, autojunk: bool) -> Self {
        let (chars, ids) = CharInterner::build(b.iter().copied());
        let junk: Vec<bool> = chars.chars().iter().map(|&c| isjunk(c)).collect();
        let index = BIndex::new(ids, junk.len(), &junk, autojunk);
        Self { chars, index }
    }

    /// `set_seq2(b)` on this matcher, reusing its memory (`w` lends buffers).
    pub fn reset(
        &mut self,
        b: impl IntoIterator<Item = u32>,
        isjunk: impl Fn(u32) -> bool,
        autojunk: bool,
        w: &mut Work,
    ) {
        let mut ids = std::mem::take(&mut w.ids);
        self.chars.rebuild(b, &mut ids);
        let mut junk = std::mem::take(&mut w.junk);
        junk.clear();
        junk.extend(self.chars.chars().iter().map(|&c| isjunk(c)));
        self.index.rebuild(&mut ids, junk.len(), &junk, autojunk);
        w.ids = ids;
        w.junk = junk;
    }

    /// An empty matcher, for [`Self::reset`].
    pub fn empty() -> Self {
        Self::new(&[], |_| false, true)
    }

    /// Length of `b`.
    pub fn len(&self) -> usize {
        self.index.b().len()
    }

    /// True if `b` is empty.
    pub fn is_empty(&self) -> bool {
        self.index.b().is_empty()
    }

    /// `a` as ids of `b`'s characters.
    pub fn encode(&self, a: &[u32]) -> Vec<u32> {
        self.chars.encode(a.iter().copied())
    }

    /// `a`'s code points as ids of `b`'s characters.
    pub fn encode_iter(&self, a: impl Iterator<Item = u32>) -> impl Iterator<Item = u32> {
        a.map(|c| self.chars.get(c))
    }

    /// The index behind it.
    pub fn index(&self) -> &BIndex {
        &self.index
    }

    /// `set_seq1(a); ratio()`.
    pub fn ratio(&self, a: &[u32]) -> f64 {
        self.ratio_with(a.iter().copied(), &mut Work::default())
    }

    /// `set_seq1(a); quick_ratio()`.
    pub fn quick_ratio(&self, a: &[u32]) -> f64 {
        let ids = self.encode(a);
        calculate_ratio(self.index.quick_matches(&ids), a.len() + self.len())
    }

    /// `set_seq1(a); ratio()` for `a`'s code points, with reusable buffers.
    pub fn ratio_with(&self, a: impl Iterator<Item = u32>, w: &mut Work) -> f64 {
        let mut ids = std::mem::take(&mut w.ids);
        ids.clear();
        ids.extend(a.map(|c| self.chars.get(c)));
        let r = calculate_ratio(self.index.matched(&ids, w), ids.len() + self.len());
        w.ids = ids;
        r
    }

    /// `real_quick_ratio() >= cutoff and quick_ratio() >= cutoff and
    /// ratio() >= cutoff` (`get_close_matches`' test): the ratio if it passes.
    pub fn close_match(&self, a: &[u32], cutoff: f64) -> Option<f64> {
        self.close_match_with(a.len(), a.iter().copied(), cutoff, &mut Work::default())
    }

    /// [`Self::close_match`] for `a`'s `la` code points, with reusable buffers.
    pub fn close_match_with(
        &self,
        la: usize,
        a: impl Iterator<Item = u32>,
        cutoff: f64,
        w: &mut Work,
    ) -> Option<f64> {
        let lb = self.len();
        if w.ticker().tick(1 + la) || real_quick_ratio(la, lb) < cutoff {
            return None;
        }
        let mut ids = std::mem::take(&mut w.ids);
        ids.clear();
        ids.extend(a.map(|c| self.chars.get(c)));
        let mut r = None;
        if calculate_ratio(self.index.quick_matches_with(&ids, w), la + lb) >= cutoff {
            let ratio = calculate_ratio(self.index.matched(&ids, w), la + lb);
            r = (ratio >= cutoff).then_some(ratio);
        }
        w.ids = ids;
        r
    }
}

thread_local! {
    static WORK: RefCell<Work> = RefCell::new(Work::default());
}

/// `f` with this thread's reusable [`Work`] buffers.
pub fn with_work<R>(f: impl FnOnce(&mut Work) -> R) -> R {
    WORK.with(|w| match w.try_borrow_mut() {
        Ok(mut w) => f(&mut w),
        Err(_) => f(&mut Work::default()), // re-entered: fresh buffers
    })
}

/// `heapq.nlargest(n, [(score, x), ...])` as `get_close_matches` uses it:
/// the indices of the best `n` candidates, by score, then by the candidate
/// (`cmp_text(x, y)` compares candidates `x` and `y` as Python compares the
/// strings; larger first), then by position (earlier first).
pub fn nlargest(
    scored: &[(f64, usize)],
    n: usize,
    cmp_text: impl Fn(usize, usize) -> std::cmp::Ordering,
) -> Vec<usize> {
    let mut order: Vec<(f64, usize)> = scored.to_vec();
    // Scores are ratios: never NaN, and 0.0 is never negative.
    order.sort_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then_with(|| cmp_text(y.1, x.1))
            .then_with(|| x.1.cmp(&y.1))
    });
    order.truncate(n);
    order.into_iter().map(|(_, i)| i).collect()
}

/// Turn on Ctrl-C checks for this thread's [`with_work`] buffers: from now
/// until [`finish_checks`], the loops that use them ask `check` about every
/// [`stop::POLL_INTERVAL`], and stop early once it says so. For the thread
/// that called in from Python, never a pool worker.
pub fn start_checks(check: stop::Check) {
    WORK.with(|w| {
        if let Ok(mut w) = w.try_borrow_mut() {
            w.ticker().start(check);
        }
    });
}

/// Turn the checks of [`start_checks`] off: true if they stopped the work
/// (whose results are then incomplete).
pub fn finish_checks() -> bool {
    WORK.with(|w| w.try_borrow_mut().is_ok_and(|mut w| w.ticker().finish()))
}
