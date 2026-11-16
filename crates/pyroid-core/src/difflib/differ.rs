//! `difflib.Differ.compare` (and so `ndiff`), as CPython 3.11-3.13 and
//! 3.14+ write it.
//!
//! Lines are given twice: as ids (equal lines, equal ids; a line of `a` that
//! is not in `b` is `NONE`) for the line-level matcher, and as code points
//! for the intraline one. The output is a list of [`DiffOp`]s the bindings
//! turn into strings (`"- " + line`, ...), so the caller's line objects are
//! never re-encoded.

use super::StrMatcher;
use super::matcher::{BIndex, Tag, calculate_ratio, matches, opcodes, real_quick_ratio};
use crate::text::is_py_whitespace;

/// Which `Differ._fancy_replace` to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DifferVersion {
    /// CPython 3.11 to 3.13: the best pair of the whole block, recursively.
    Classic,
    /// CPython 3.14 and later: a window of 10 lines around the diagonal, no recursion.
    Windowed,
}

/// The `charjunk` the intraline matcher uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharJunk {
    /// `charjunk=None`.
    Nothing,
    /// `IS_CHARACTER_JUNK`: a space or a tab.
    SpaceTab,
}

impl CharJunk {
    fn is_junk(self, c: u32) -> bool {
        match self {
            Self::Nothing => false,
            Self::SpaceTab => c == u32::from(b' ') || c == u32::from(b'\t'),
        }
    }
}

/// Which input a line comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

/// One line of `compare`'s output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DiffOp {
    /// `prefix + line`, with `prefix` one of `"- "`, `"+ "`, `"  "`.
    Line(u8, Side, usize),
    /// `"? " + tags + "\n"`.
    Tags(String),
}

impl DiffOp {
    /// The first character of the output line.
    pub fn first(&self) -> u8 {
        match self {
            Self::Line(tag, _, _) => *tag,
            Self::Tags(_) => b'?',
        }
    }
}

/// The inputs of one `compare` call.
#[derive(Debug)]
pub struct Lines<'a> {
    /// `a`'s lines as ids of `b`'s lines (`NONE` if absent).
    pub a: &'a [u32],
    /// `b`'s lines as ids (first-occurrence order).
    pub b: &'a [u32],
    /// Distinct lines of `b`.
    pub b_keys: usize,
    /// `linejunk`'s verdict for each distinct line of `b` (empty: `None`).
    pub b_junk: &'a [bool],
    pub a_chars: &'a [&'a [u32]],
    pub b_chars: &'a [&'a [u32]],
}

/// The intraline `SequenceMatcher(charjunk)` after `set_seq2(b[j])`.
fn cruncher(b: &[u32], junk: CharJunk) -> StrMatcher {
    StrMatcher::new(b, |c| junk.is_junk(c), true)
}

/// `real_quick_ratio() > best and quick_ratio() > best and ratio() > best`:
/// the ratio if all three hold.
fn better(m: &StrMatcher, a: &[u32], best: f64) -> Option<f64> {
    let (la, lb) = (a.len(), m.len());
    if real_quick_ratio(la, lb) <= best {
        return None;
    }
    let ids = m.encode(a);
    if calculate_ratio(m.index().quick_matches(&ids), la + lb) <= best {
        return None;
    }
    let r = calculate_ratio(matches(&m.index().matching_blocks(&ids)), la + lb);
    (r > best).then_some(r)
}

/// `Differ(linejunk, charjunk).compare(a, b)`.
pub fn compare(lines: &Lines<'_>, junk: CharJunk, version: DifferVersion) -> Vec<DiffOp> {
    let index = BIndex::new(lines.b.to_vec(), lines.b_keys, lines.b_junk, true);
    let blocks = index.matching_blocks(lines.a);
    let mut d = Diff {
        lines,
        junk,
        version,
        out: Vec::with_capacity(lines.a.len().max(lines.b.len())),
    };
    for op in opcodes(&blocks) {
        match op.tag {
            Tag::Replace => d.fancy_replace(op.i1, op.i2, op.j1, op.j2),
            Tag::Delete => d.dump(b'-', Side::A, op.i1, op.i2),
            Tag::Insert => d.dump(b'+', Side::B, op.j1, op.j2),
            Tag::Equal => d.dump(b' ', Side::A, op.i1, op.i2),
        }
    }
    d.out
}

struct Diff<'l, 'a> {
    lines: &'l Lines<'a>,
    junk: CharJunk,
    version: DifferVersion,
    out: Vec<DiffOp>,
}

/// Pending work of the 3.11-3.13 recursion, run from an explicit stack.
enum Task {
    Helper(usize, usize, usize, usize),
    Pair(usize, usize, bool),
}

impl Diff<'_, '_> {
    fn dump(&mut self, tag: u8, side: Side, lo: usize, hi: usize) {
        self.out
            .extend((lo..hi).map(|i| DiffOp::Line(tag, side, i)));
    }

    fn plain_replace(&mut self, alo: usize, ahi: usize, blo: usize, bhi: usize) {
        if bhi - blo < ahi - alo {
            self.dump(b'+', Side::B, blo, bhi);
            self.dump(b'-', Side::A, alo, ahi);
        } else {
            self.dump(b'-', Side::A, alo, ahi);
            self.dump(b'+', Side::B, blo, bhi);
        }
    }

    fn fancy_replace(&mut self, alo: usize, ahi: usize, blo: usize, bhi: usize) {
        match self.version {
            DifferVersion::Classic => self.classic(alo, ahi, blo, bhi),
            DifferVersion::Windowed => self.windowed(alo, ahi, blo, bhi),
        }
    }

    /// 3.11-3.13 `_fancy_replace` and `_fancy_helper`, with the recursion as
    /// a stack of tasks (CPython's order: before, the pair, after).
    fn classic(&mut self, alo: usize, ahi: usize, blo: usize, bhi: usize) {
        let mut stack = vec![Task::Helper(alo, ahi, blo, bhi)];
        while let Some(task) = stack.pop() {
            match task {
                Task::Pair(i, j, identical) => self.pair(i, j, identical),
                Task::Helper(alo, ahi, blo, bhi) => {
                    if alo < ahi && blo < bhi {
                        if let Some((i, j, identical)) = self.classic_best(alo, ahi, blo, bhi) {
                            stack.push(Task::Helper(i + 1, ahi, j + 1, bhi));
                            stack.push(Task::Pair(i, j, identical));
                            stack.push(Task::Helper(alo, i, blo, j));
                        } else {
                            self.plain_replace(alo, ahi, blo, bhi);
                        }
                    } else if alo < ahi {
                        self.dump(b'-', Side::A, alo, ahi);
                    } else if blo < bhi {
                        self.dump(b'+', Side::B, blo, bhi);
                    }
                }
            }
        }
    }

    /// The synch pair of a 3.11-3.13 replace block, `None` for a plain replace.
    fn classic_best(
        &self,
        alo: usize,
        ahi: usize,
        blo: usize,
        bhi: usize,
    ) -> Option<(usize, usize, bool)> {
        let l = self.lines;
        let (mut best_ratio, cutoff) = (0.74, 0.75);
        let mut best = (0, 0);
        let mut eq: Option<(usize, usize)> = None;
        for j in blo..bhi {
            let cruncher = cruncher(l.b_chars[j], self.junk);
            for i in alo..ahi {
                if l.a[i] == l.b[j] {
                    if eq.is_none() {
                        eq = Some((i, j));
                    }
                    continue;
                }
                if let Some(r) = better(&cruncher, l.a_chars[i], best_ratio) {
                    best_ratio = r;
                    best = (i, j);
                }
            }
        }
        if best_ratio < cutoff {
            return eq.map(|(i, j)| (i, j, true));
        }
        Some((best.0, best.1, false))
    }

    /// 3.14+ `_fancy_replace`.
    #[allow(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        clippy::needless_range_loop
    )] // line indices are far below 2**63; CPython's loop
    fn windowed(&mut self, alo: usize, ahi: usize, blo: usize, bhi: usize) {
        const WINDOW: i64 = 10;
        let l = self.lines;
        let cutoff = 0.74999;
        let (mut dump_i, mut dump_j) = (alo, blo);
        for j in blo..bhi {
            let aequiv = (alo + (j - blo)) as i64;
            let lo = (aequiv - WINDOW).max(dump_i as i64) as usize;
            let hi = (aequiv + WINDOW + 1).min(ahi as i64) as usize;
            if lo >= hi {
                break;
            }
            let cruncher = cruncher(l.b_chars[j], self.junk);
            let mut best_ratio = cutoff;
            let mut best_i = None;
            for i in lo..hi {
                if let Some(r) = better(&cruncher, l.a_chars[i], best_ratio) {
                    best_ratio = r;
                    best_i = Some(i);
                }
            }
            let Some(best_i) = best_i else { continue };
            self.windowed_helper(dump_i, best_i, dump_j, j);
            self.pair(best_i, j, l.a[best_i] == l.b[j]);
            (dump_i, dump_j) = (best_i + 1, j + 1);
        }
        self.windowed_helper(dump_i, ahi, dump_j, bhi);
    }

    /// 3.14+ `_fancy_helper`.
    fn windowed_helper(&mut self, alo: usize, ahi: usize, blo: usize, bhi: usize) {
        if alo < ahi {
            if blo < bhi {
                self.plain_replace(alo, ahi, blo, bhi);
            } else {
                self.dump(b'-', Side::A, alo, ahi);
            }
        } else if blo < bhi {
            self.dump(b'+', Side::B, blo, bhi);
        }
    }

    /// The synch pair: `"  " + a[i]` if identical, else `_qformat`'s quad.
    fn pair(&mut self, i: usize, j: usize, identical: bool) {
        if identical {
            self.out.push(DiffOp::Line(b' ', Side::A, i));
            return;
        }
        let (aelt, belt) = (self.lines.a_chars[i], self.lines.b_chars[j]);
        let cruncher = cruncher(belt, self.junk);
        let blocks = cruncher.index().matching_blocks(&cruncher.encode(aelt));
        let (mut atags, mut btags) = (Vec::new(), Vec::new());
        for op in opcodes(&blocks) {
            let (la, lb) = (op.i2 - op.i1, op.j2 - op.j1);
            match op.tag {
                Tag::Replace => {
                    atags.extend(std::iter::repeat_n(b'^', la));
                    btags.extend(std::iter::repeat_n(b'^', lb));
                }
                Tag::Delete => atags.extend(std::iter::repeat_n(b'-', la)),
                Tag::Insert => btags.extend(std::iter::repeat_n(b'+', lb)),
                Tag::Equal => {
                    atags.extend(std::iter::repeat_n(b' ', la));
                    btags.extend(std::iter::repeat_n(b' ', lb));
                }
            }
        }
        let atags = keep_original_ws(aelt, &atags);
        let btags = keep_original_ws(belt, &btags);
        self.out.push(DiffOp::Line(b'-', Side::A, i));
        if !atags.is_empty() {
            self.out.push(DiffOp::Tags(atags));
        }
        self.out.push(DiffOp::Line(b'+', Side::B, j));
        if !btags.is_empty() {
            self.out.push(DiffOp::Tags(btags));
        }
    }
}

/// `_keep_original_ws(s, tags).rstrip()`: a space tag over a whitespace
/// character becomes that character (tabs stay aligned).
fn keep_original_ws(s: &[u32], tags: &[u8]) -> String {
    let mut out = String::with_capacity(tags.len());
    for (&c, &t) in s.iter().zip(tags) {
        match char::from_u32(c) {
            Some(ch) if t == b' ' && is_py_whitespace(ch) => out.push(ch),
            _ => out.push(char::from(t)),
        }
    }
    let end = out.trim_end_matches(is_py_whitespace).len();
    out.truncate(end);
    out
}
