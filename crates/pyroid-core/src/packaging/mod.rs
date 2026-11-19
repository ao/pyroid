// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! packaging 26.3's versions, specifiers, markers and requirements (`pyroid.packaging`).
//!
//! - [`version`]: PEP 440 parsing, ordering, hashing and printing.
//! - [`specifiers`]: specifier parsing and the range engine behind `contains` and `filter`.
//! - [`markers`]: the PEP 508 tokenizer and parser (markers and requirements).
//! - [`num`]: the arbitrary-size integers PEP 440 numbers are.

pub mod markers;
pub mod num;
pub mod specifiers;
pub mod version;

use std::cmp::Ordering;

/// The PEP 440 order of `keys[range]`, as indices into `keys` (stable; `reverse` is newest first
/// and still keeps equal versions in input order, as `sorted(..., reverse=True)` does).
#[must_use]
pub fn sort_run(keys: &[version::Version], range: (usize, usize), reverse: bool) -> Vec<usize> {
    let mut idx: Vec<usize> = (range.0..range.1).collect();
    idx.sort_by(|a, b| order(keys, *a, *b, reverse));
    idx
}

/// Merge two runs from [`sort_run`]; on ties `left` (earlier input) comes first, which keeps
/// the sort stable.
#[must_use]
pub fn merge_runs(
    keys: &[version::Version],
    left: &[usize],
    right: &[usize],
    reverse: bool,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if order(keys, right[j], left[i], reverse) == Ordering::Less {
            out.push(right[j]);
            j += 1;
        } else {
            out.push(left[i]);
            i += 1;
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

fn order(keys: &[version::Version], a: usize, b: usize, reverse: bool) -> Ordering {
    let o = keys[a].cmp_key(&keys[b]);
    if reverse { o.reverse() } else { o }
}

#[cfg(test)]
mod tests;
