//! CPython's `difflib`, ported exactly (`pyroid.difflib`).
//!
//! - [`matcher`]: `SequenceMatcher`'s algorithm over dense element ids:
//!   `b2j`, junk and popular elements, `find_longest_match`,
//!   `get_matching_blocks`, `get_opcodes`, `get_grouped_opcodes`, the ratios.
//! - [`intern`]: element ids for code points and other keys.
//! - [`differ`]: `Differ.compare` / `ndiff`, in CPython 3.11-3.13's and 3.14+'s versions.
//! - [`mdiff`]: `_mdiff`, the side-by-side rows of `HtmlDiff`.
//! - [`format`]: `unified_diff` and `context_diff`.
//!
//! [`StrMatcher`] is `SequenceMatcher` over the characters of two strings,
//! for `get_close_matches` and the batch functions.

pub mod differ;
pub mod format;
pub mod intern;
pub mod matcher;
pub mod mdiff;

#[cfg(test)]
mod tests;

use intern::CharInterner;
use matcher::{BIndex, calculate_ratio, matches, real_quick_ratio};

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

    /// The index behind it.
    pub fn index(&self) -> &BIndex {
        &self.index
    }

    /// `set_seq1(a); ratio()`.
    pub fn ratio(&self, a: &[u32]) -> f64 {
        let ids = self.encode(a);
        calculate_ratio(
            matches(&self.index.matching_blocks(&ids)),
            a.len() + self.len(),
        )
    }

    /// `set_seq1(a); quick_ratio()`.
    pub fn quick_ratio(&self, a: &[u32]) -> f64 {
        let ids = self.encode(a);
        calculate_ratio(self.index.quick_matches(&ids), a.len() + self.len())
    }

    /// `real_quick_ratio() >= cutoff and quick_ratio() >= cutoff and
    /// ratio() >= cutoff` (`get_close_matches`' test): the ratio if it passes.
    pub fn close_match(&self, a: &[u32], cutoff: f64) -> Option<f64> {
        let (la, lb) = (a.len(), self.len());
        if real_quick_ratio(la, lb) < cutoff {
            return None;
        }
        let ids = self.encode(a);
        if calculate_ratio(self.index.quick_matches(&ids), la + lb) < cutoff {
            return None;
        }
        let r = calculate_ratio(matches(&self.index.matching_blocks(&ids)), la + lb);
        (r >= cutoff).then_some(r)
    }
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
