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

use crate::par::{self, PoolError, Threads};

/// The order that sorts `keys` by PEP 440 (stable; `reverse` is newest first and still keeps
/// equal versions in input order, as `sorted(..., reverse=True)` does): runs sorted on the
/// shared pool, then merged.
///
/// # Errors
/// The pool could not start.
pub fn sort_indices(keys: &[version::Version], reverse: bool, threads: Threads) -> Result<Vec<usize>, PoolError> {
    let cmp = |a: &usize, b: &usize| {
        let o = keys[*a].cmp_key(&keys[*b]);
        if reverse { o.reverse() } else { o }
    };
    let n = keys.len();
    let tasks = if n < 8192 { 1 } else { par::max_tasks(threads)?.min(n / 4096).max(1) };
    if tasks == 1 {
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(cmp);
        return Ok(idx);
    }
    let bounds: Vec<(usize, usize)> = (0..tasks).map(|t| (t * n / tasks, (t + 1) * n / tasks)).collect();
    let runs = par::map_tasks(&bounds, threads, |&(a, b)| {
        let mut idx: Vec<usize> = (a..b).collect();
        idx.sort_by(cmp);
        idx
    })?;
    // Merge neighbouring runs pairwise (on the pool), log2(tasks) rounds; on ties the left run
    // (earlier input) wins, which keeps the sort stable.
    let mut runs = runs;
    while runs.len() > 1 {
        let pairs: Vec<(usize, usize)> = (0..runs.len()).step_by(2).map(|i| (i, i + 1)).collect();
        let merged = par::map_tasks(&pairs, threads, |&(l, r)| {
            let Some(right) = runs.get(r) else { return runs[l].clone() };
            let left = &runs[l];
            let mut out = Vec::with_capacity(left.len() + right.len());
            let (mut i, mut j) = (0, 0);
            while i < left.len() && j < right.len() {
                if cmp(&right[j], &left[i]) == Ordering::Less {
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
        })?;
        runs = merged;
    }
    let out = runs.pop().unwrap_or_default();
    Ok(out)
}

#[cfg(test)]
mod tests;
