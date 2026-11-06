// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! Lookahead scans with memoization.
//!
//! json_repair rescans the rest of the input from many nearby start points
//! (`skip_to_character`, `scroll_whitespaces`), which is quadratic on long
//! strings. The answers are pure functions of the buffer, so each scan
//! remembers the span it proved free of targets: a later query that starts
//! inside that span is answered without scanning.
//!
//! Escapes: `skip_to_character` counts backslashes only from its start
//! position. From `p`, let `q` be the first non-backslash at or after `p`; the
//! answer is `q` if `s[q]` is a target and `q - p` is even, else the first
//! target after `q` preceded by an even run of backslashes. Only that second
//! part is memoized (it no longer depends on `p`).

use super::super::chars::CharProps;
use super::Parser;

/// The chars `skip_to_character` is ever asked for, as a bit set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Targets(u16);

const ALPHABET: [char; 10] = [
    '"', '\'', '\u{201c}', '\u{201d}', ']', ':', '}', '`', '{', '[',
];

#[inline]
fn bit(c: char) -> u16 {
    match c {
        '"' => 1,
        '\'' => 1 << 1,
        '\u{201c}' => 1 << 2,
        '\u{201d}' => 1 << 3,
        ']' => 1 << 4,
        ':' => 1 << 5,
        '}' => 1 << 6,
        '`' => 1 << 7,
        '{' => 1 << 8,
        '[' => 1 << 9,
        _ => 0,
    }
}

impl Targets {
    /// `{c}`. `c` must be one of the chars json_repair searches for.
    pub(crate) fn one(c: char) -> Self {
        let b = bit(c);
        debug_assert!(b != 0, "unsupported skip_to_character target {c:?}");
        Self(b)
    }

    /// `{a, b}`.
    pub(crate) fn two(a: char, b: char) -> Self {
        Self(Self::one(a).0 | Self::one(b).0)
    }

    /// `STRING_DELIMITERS + extra`.
    pub(crate) fn delims_and(extra: &[char]) -> Self {
        let mut m = bit('"') | bit('\'') | bit('\u{201c}') | bit('\u{201d}');
        for &c in extra {
            m |= Self::one(c).0;
        }
        Self(m)
    }

    #[inline]
    pub(crate) fn contains(self, c: char) -> bool {
        bit(c) & self.0 != 0
    }

    #[allow(dead_code)]
    pub(crate) fn chars(self) -> impl Iterator<Item = char> {
        ALPHABET.into_iter().filter(move |&c| bit(c) & self.0 != 0)
    }
}

/// A proven span: no answer in `[from, to)`; the answer at `from` is `to`
/// (`None`: no answer up to the end).
#[derive(Clone, Copy, Debug)]
struct Span {
    from: usize,
    to: Option<usize>,
}

impl Span {
    #[inline]
    #[allow(clippy::option_option)]
    fn answer(self, x: usize) -> Option<Option<usize>> {
        match self.to {
            Some(t) if self.from <= x && x <= t => Some(Some(t)),
            None if self.from <= x => Some(None),
            _ => None,
        }
    }
}

/// Per-parse memo. Cleared whenever the buffer changes.
#[derive(Debug, Default)]
pub(crate) struct Memo {
    unescaped: Vec<(Targets, Span)>,
    non_space: Option<Span>,
    /// Per quote char: `(position, run length)` for every unescaped quote, where
    /// the run counts the consecutive unescaped quotes from there up to the
    /// next unescaped `]` (or the end). Positions are as when built; `inserts`
    /// maps them to the current buffer.
    quote_runs: Vec<(char, Vec<(u32, u32)>)>,
    /// Positions of the `{`s inserted since `quote_runs` were built (ascending,
    /// current coordinates; a char at current `p` was at `p - #inserts < p`).
    inserts: Vec<usize>,
    /// `_initial_container_has_non_comma_trailing_content()` at index 0.
    pub(crate) initial_container: Option<bool>,
}

impl Memo {
    pub(crate) fn invalidate(&mut self) {
        self.unescaped.clear();
        self.quote_runs.clear();
        self.inserts.clear();
        self.non_space = None;
        self.initial_container = None;
    }
}

impl Memo {
    /// The buffer got char `c` inserted at `p`, after a non-backslash (so no
    /// backslash run changed): shift every remembered span past `p`.
    pub(crate) fn shift_insert(&mut self, p: usize, c: char, props: CharProps) {
        let shift = |x: usize| if x >= p { x + 1 } else { x };
        for (targets, span) in &mut self.unescaped {
            let covers = span.from < p && span.to.is_none_or(|t| t >= p);
            span.from = shift(span.from);
            span.to = span.to.map(shift);
            if covers && targets.contains(c) {
                // The new char is the first unescaped target after `from`.
                span.to = Some(p);
            }
        }
        if let Some(span) = &mut self.non_space {
            let covers = span.from < p && span.to.is_none_or(|t| t >= p);
            span.from = shift(span.from);
            span.to = span.to.map(shift);
            if covers && !props.is_space(c) {
                span.to = Some(p);
            }
        }
        if c == ']' || is_quote_target(c) {
            self.quote_runs.clear();
            self.inserts.clear();
        } else if !self.quote_runs.is_empty() {
            if self.inserts.last().is_some_and(|&last| last > p) {
                // Not in ascending order: rebuild on the next query.
                self.quote_runs.clear();
                self.inserts.clear();
            } else {
                self.inserts.push(p);
            }
        }
        self.initial_container = None;
    }
}

/// A char some `quote_runs` table may count.
fn is_quote_target(c: char) -> bool {
    matches!(c, '"' | '\'' | '\u{201c}' | '\u{201d}')
}

impl Parser<'_> {
    /// First index `>= from` that is not a space (or `len`).
    pub(crate) fn next_non_space(&mut self, from: usize) -> usize {
        let n = self.s.len();
        if from >= n {
            return from;
        }
        if !self.props.is_space(self.s.at(from)) {
            return from;
        }
        if let Some(span) = self.scan.non_space {
            if let Some(ans) = span.answer(from) {
                return ans.unwrap_or(n);
            }
        }
        let mut i = from + 1;
        while i < n && self.props.is_space(self.s.at(i)) {
            i += 1;
        }
        self.scan.non_space = Some(Span {
            from,
            to: (i < n).then_some(i),
        });
        i
    }

    /// The number of consecutive `{quote, ']'}` stops that are `quote`, from the
    /// unescaped `quote` at absolute `p` (json_repair reaches them by repeated
    /// `skip_to_character([quote, "]"])` from just past each one). `None` if
    /// `p` is not such a quote.
    pub(crate) fn quote_run_len(&mut self, quote: char, p: usize) -> Option<u32> {
        if self.scan.quote_runs.iter().all(|(q, _)| *q != quote) {
            let runs = self.build_quote_runs(quote);
            self.scan.quote_runs.push((quote, runs));
        }
        let (_, runs) = self.scan.quote_runs.iter().find(|(q, _)| *q == quote)?;
        // Back to build coordinates: `p` is not an inserted `{` (it is a quote).
        let before = self.scan.inserts.partition_point(|&x| x < p);
        let p = u32::try_from(p - before).ok()?;
        runs.binary_search_by_key(&p, |&(pos, _)| pos)
            .ok()
            .map(|k| runs[k].1)
    }

    fn build_quote_runs(&self, quote: char) -> Vec<(u32, u32)> {
        // Stops: `quote` or `]` preceded by an even run of backslashes.
        let mut stops: Vec<(u32, bool)> = Vec::new();
        let mut backslashes = 0usize;
        for (i, c) in self.s.iter().enumerate() {
            if c == '\\' {
                backslashes += 1;
                continue;
            }
            if (c == quote || c == ']') && backslashes % 2 == 0 {
                stops.push((u32::try_from(i).unwrap_or(u32::MAX), c == quote));
            }
            backslashes = 0;
        }
        let mut runs = Vec::with_capacity(stops.len());
        let mut run = 0u32;
        for &(pos, is_quote) in stops.iter().rev() {
            run = if is_quote { run + 1 } else { 0 };
            if is_quote {
                runs.push((pos, run));
            }
        }
        runs.reverse();
        runs
    }

    /// `skip_to_character` from absolute `p`: the absolute index of the hit.
    pub(crate) fn next_unescaped(&mut self, targets: Targets, p: usize) -> Option<usize> {
        let n = self.s.len();
        // q: the first non-backslash at or after p.
        let mut q = p;
        while q < n && self.s.at(q) == '\\' {
            q += 1;
        }
        if q >= n {
            return None;
        }
        if targets.contains(self.s.at(q)) && (q - p) % 2 == 0 {
            return Some(q);
        }
        self.next_target_after_plain(targets, q + 1)
    }

    /// The first target at or after `x` preceded by an even run of backslashes,
    /// where `s[x - 1]` is not a backslash (or `x == 0`).
    fn next_target_after_plain(&mut self, targets: Targets, x: usize) -> Option<usize> {
        if let Some((_, span)) = self.scan.unescaped.iter().find(|(t, _)| *t == targets) {
            if let Some(ans) = span.answer(x) {
                return ans;
            }
        }
        let n = self.s.len();
        let mut i = x;
        let mut backslashes = 0usize;
        let found = loop {
            if i >= n {
                break None;
            }
            let ch = self.s.at(i);
            if ch == '\\' {
                backslashes += 1;
            } else {
                if targets.contains(ch) && backslashes % 2 == 0 {
                    break Some(i);
                }
                backslashes = 0;
            }
            i += 1;
        };
        let span = Span { from: x, to: found };
        if let Some(slot) = self.scan.unescaped.iter_mut().find(|(t, _)| *t == targets) {
            slot.1 = span;
        } else {
            self.scan.unescaped.push((targets, span));
        }
        found
    }
}
