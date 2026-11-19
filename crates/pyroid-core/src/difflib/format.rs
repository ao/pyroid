//! `unified_diff` and `context_diff` as a list of output lines whose text
//! the bindings assemble from the caller's own strings.

use super::differ::Side;
use super::matcher::{GroupedOpcode, Tag, py_slice};

/// One output line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Out {
    /// The two file header lines (`---`/`+++` or `***`/`---`).
    FileHeaders,
    /// A control line, without `lineterm` (a hunk header, a range line, the
    /// `***************` separator).
    Control(String),
    /// `prefix + line`.
    Line(Prefix, Side, usize),
}

/// The prefix of a content line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prefix {
    /// Unified `' '`, context `'  '`.
    Context,
    /// Unified `'-'`, context `'- '`.
    Removed,
    /// Unified `'+'`, context `'+ '`.
    Added,
    /// Context `'! '`.
    Changed,
}

/// `_format_range_unified(start, stop)`.
pub fn format_range_unified(start: i64, stop: i64) -> String {
    let mut beginning = start + 1;
    let length = stop - start;
    if length == 1 {
        return beginning.to_string();
    }
    if length == 0 {
        beginning -= 1;
    }
    format!("{beginning},{length}")
}

/// `_format_range_context(start, stop)`.
pub fn format_range_context(start: i64, stop: i64) -> String {
    let mut beginning = start + 1;
    let length = stop - start;
    if length == 0 {
        beginning -= 1;
    }
    if length <= 1 {
        return beginning.to_string();
    }
    format!("{beginning},{}", beginning + length - 1)
}

/// `unified_diff(a, b, n=...)` over `get_grouped_opcodes(n)`'s groups, for
/// sequences of `la` and `lb` lines.
pub fn unified(groups: &[Vec<GroupedOpcode>], la: usize, lb: usize) -> Vec<Out> {
    let mut out = Vec::new();
    for (g, group) in groups.iter().enumerate() {
        if g == 0 {
            out.push(Out::FileHeaders);
        }
        let (first, last) = (group[0], group[group.len() - 1]);
        out.push(Out::Control(format!(
            "@@ -{} +{} @@",
            format_range_unified(first.i1, last.i2),
            format_range_unified(first.j1, last.j2)
        )));
        for op in group {
            if op.tag == Tag::Equal {
                for i in py_slice(la, op.i1, op.i2) {
                    out.push(Out::Line(Prefix::Context, Side::A, i));
                }
                continue;
            }
            if matches!(op.tag, Tag::Replace | Tag::Delete) {
                for i in py_slice(la, op.i1, op.i2) {
                    out.push(Out::Line(Prefix::Removed, Side::A, i));
                }
            }
            if matches!(op.tag, Tag::Replace | Tag::Insert) {
                for j in py_slice(lb, op.j1, op.j2) {
                    out.push(Out::Line(Prefix::Added, Side::B, j));
                }
            }
        }
    }
    out
}

/// `context_diff(a, b, n=...)` over `get_grouped_opcodes(n)`'s groups.
pub fn context(groups: &[Vec<GroupedOpcode>], la: usize, lb: usize) -> Vec<Out> {
    let prefix = |tag: Tag| match tag {
        Tag::Insert => Prefix::Added,
        Tag::Delete => Prefix::Removed,
        Tag::Replace => Prefix::Changed,
        Tag::Equal => Prefix::Context,
    };
    let mut out = Vec::new();
    for (g, group) in groups.iter().enumerate() {
        if g == 0 {
            out.push(Out::FileHeaders);
        }
        let (first, last) = (group[0], group[group.len() - 1]);
        out.push(Out::Control("***************".to_owned()));
        out.push(Out::Control(format!(
            "*** {} ****",
            format_range_context(first.i1, last.i2)
        )));
        if group
            .iter()
            .any(|op| matches!(op.tag, Tag::Replace | Tag::Delete))
        {
            for op in group.iter().filter(|op| op.tag != Tag::Insert) {
                for i in py_slice(la, op.i1, op.i2) {
                    out.push(Out::Line(prefix(op.tag), Side::A, i));
                }
            }
        }
        out.push(Out::Control(format!(
            "--- {} ----",
            format_range_context(first.j1, last.j2)
        )));
        if group
            .iter()
            .any(|op| matches!(op.tag, Tag::Replace | Tag::Insert))
        {
            for op in group.iter().filter(|op| op.tag != Tag::Delete) {
                for j in py_slice(lb, op.j1, op.j2) {
                    out.push(Out::Line(prefix(op.tag), Side::B, j));
                }
            }
        }
    }
    out
}
