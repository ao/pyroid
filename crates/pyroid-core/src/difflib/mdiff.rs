//! `difflib._mdiff`, the side-by-side line pairs `HtmlDiff` formats, from
//! `ndiff`'s output.

use std::collections::VecDeque;

use super::differ::{DiffOp, Side};

/// The text of one side of a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Text {
    /// The input line itself (CPython's `line[2:]` of the ndiff line).
    Line(Side, usize),
    /// A line with `\0+`, `\0-`, `\0^` and `\1` markers, as code points.
    Marked(Vec<u32>),
}

/// One side of a row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cell {
    /// `None`.
    Nothing,
    /// `('', '\n')`, the filler opposite an added or deleted line.
    Blank,
    /// `(line number, text)`.
    Line(usize, Text),
}

/// One `_mdiff` item: `(from, to, flag)`; `flag` is `None` on the context
/// separator `(None, None, None)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub from: Cell,
    pub to: Cell,
    pub flag: Option<bool>,
}

/// `_mdiff(fromlines, tolines, context)` from `ndiff(fromlines, tolines)`'s
/// output `ops`, whose lines are `a_chars` and `b_chars` (code points).
/// `context` is `None` or `numlines >= 0`.
pub fn mdiff(
    ops: &[DiffOp],
    a_chars: &[&[u32]],
    b_chars: &[&[u32]],
    context: Option<usize>,
) -> Vec<Row> {
    let mut lines = LineIter {
        ops,
        a_chars,
        b_chars,
        next: 0,
        window: VecDeque::new(),
        num: [0, 0],
    };
    let pairs = pair_lines(lines.run());
    match context {
        None => pairs,
        Some(numlines) => with_context(pairs, numlines + 1),
    }
}

/// `_line_iterator`'s state.
struct LineIter<'a> {
    ops: &'a [DiffOp],
    a_chars: &'a [&'a [u32]],
    b_chars: &'a [&'a [u32]],
    next: usize,
    /// Indices into `ops` of the look-ahead lines.
    window: VecDeque<usize>,
    /// `num_lines`.
    num: [usize; 2],
}

/// A `_make_line` format key.
#[derive(Clone, Copy)]
enum Key {
    Plain,
    Intraline,
    Whole(u8),
}

impl LineIter<'_> {
    fn chars(&self, side: Side, i: usize) -> &[u32] {
        match side {
            Side::A => self.a_chars[i],
            Side::B => self.b_chars[i],
        }
    }

    /// `_make_line(lines, key, side)`; `copy` is `_make_line(lines[:], ...)`.
    #[allow(clippy::many_single_char_names)] // CPython's names
    fn make(&mut self, key: Key, side: usize, copy: bool) -> Cell {
        self.num[side] += 1;
        let num = self.num[side];
        let first = if copy {
            self.window[0]
        } else {
            self.window.pop_front().expect("a line to format")
        };
        let DiffOp::Line(tag, op_side, i) = self.ops[first] else {
            unreachable!("ndiff output starts a line pair with a '?' line")
        };
        match key {
            Key::Plain => Cell::Line(num, Text::Line(op_side, i)),
            Key::Whole(k) => {
                let line = self.chars(op_side, i);
                let mut text = Vec::with_capacity(line.len() + 3);
                text.extend([0, u32::from(k)]);
                if line.is_empty() {
                    text.push(u32::from(b' '));
                } else {
                    text.extend_from_slice(line);
                }
                text.push(1);
                Cell::Line(num, Text::Marked(text))
            }
            Key::Intraline => {
                let markers = self.window.pop_front().expect("a '?' line");
                let DiffOp::Tags(tags) = &self.ops[markers] else {
                    unreachable!("an intraline pair without its '?' line")
                };
                // text = "- " + line, markers = "? " + tags + "\n", as code points.
                let line = self.chars(op_side, i);
                let mut text: Vec<u32> = Vec::with_capacity(line.len() + 8);
                text.extend([u32::from(tag), u32::from(b' ')]);
                text.extend_from_slice(line);
                let marks: Vec<u32> = [u32::from(b'?'), u32::from(b' ')]
                    .into_iter()
                    .chain(tags.chars().map(u32::from))
                    .chain([u32::from(b'\n')])
                    .collect();
                // change_re = (\++|\-+|\^+): maximal runs of one marker.
                let mut spans = Vec::new();
                let mut p = 0;
                while p < marks.len() {
                    let c = marks[p];
                    if c == u32::from(b'+') || c == u32::from(b'-') || c == u32::from(b'^') {
                        let start = p;
                        while p < marks.len() && marks[p] == c {
                            p += 1;
                        }
                        spans.push((c, start, p));
                    } else {
                        p += 1;
                    }
                }
                for &(k, begin, end) in spans.iter().rev() {
                    // text[0:begin] + '\0' + key + text[begin:end] + '\1' + text[end:]
                    let n = text.len();
                    let (b, e) = (begin.min(n), end.min(n).max(begin.min(n)));
                    let mut new = Vec::with_capacity(n + 3);
                    new.extend_from_slice(&text[..b]);
                    new.extend([0, k]);
                    new.extend_from_slice(&text[b..e]);
                    new.push(1);
                    new.extend_from_slice(&text[end.min(n)..]);
                    text = new;
                }
                text.drain(..2.min(text.len()));
                Cell::Line(num, Text::Marked(text))
            }
        }
    }

    /// The first characters of the next four lines, `X` past the end.
    fn peek(&mut self) -> [u8; 4] {
        while self.window.len() < 4 && self.next < self.ops.len() {
            self.window.push_back(self.next);
            self.next += 1;
        }
        let mut s = [b'X'; 4];
        for (k, &i) in self.window.iter().enumerate() {
            s[k] = self.ops[i].first();
        }
        s
    }

    /// `_line_iterator()`, collected.
    fn run(&mut self) -> Vec<(Cell, Cell, bool)> {
        let mut out = Vec::new();
        let mut pending = 0i64;
        let mut to_yield: i64;
        loop {
            let s = self.peek();
            let starts = |p: &[u8]| s.starts_with(p);
            let (from, to);
            if starts(b"X") {
                to_yield = pending;
                (from, to) = (Cell::Nothing, Cell::Nothing);
            } else if starts(b"-?+?") {
                let f = self.make(Key::Intraline, 0, false);
                let t = self.make(Key::Intraline, 1, false);
                out.push((f, t, true));
                continue;
            } else if starts(b"--++") {
                pending -= 1;
                let f = self.make(Key::Whole(b'-'), 0, false);
                out.push((f, Cell::Nothing, true));
                continue;
            } else if starts(b"--?+") || starts(b"--+") || starts(b"- ") {
                from = self.make(Key::Whole(b'-'), 0, false);
                to = Cell::Nothing;
                (to_yield, pending) = (pending - 1, 0);
            } else if starts(b"-+?") {
                let f = self.make(Key::Plain, 0, false);
                let t = self.make(Key::Intraline, 1, false);
                out.push((f, t, true));
                continue;
            } else if starts(b"-?+") {
                let f = self.make(Key::Intraline, 0, false);
                let t = self.make(Key::Plain, 1, false);
                out.push((f, t, true));
                continue;
            } else if starts(b"-") {
                pending -= 1;
                let f = self.make(Key::Whole(b'-'), 0, false);
                out.push((f, Cell::Nothing, true));
                continue;
            } else if starts(b"+--") {
                pending += 1;
                let t = self.make(Key::Whole(b'+'), 1, false);
                out.push((Cell::Nothing, t, true));
                continue;
            } else if starts(b"+ ") || starts(b"+-") {
                from = Cell::Nothing;
                to = self.make(Key::Whole(b'+'), 1, false);
                (to_yield, pending) = (pending + 1, 0);
            } else if starts(b"+") {
                pending += 1;
                let t = self.make(Key::Whole(b'+'), 1, false);
                out.push((Cell::Nothing, t, true));
                continue;
            } else if starts(b" ") {
                let f = self.make(Key::Plain, 0, true);
                let t = self.make(Key::Plain, 1, false);
                out.push((f, t, false));
                continue;
            } else {
                // A '?' line first: ndiff never produces it (CPython would loop forever).
                unreachable!("ndiff output with a stray '?' line");
            }
            while to_yield < 0 {
                to_yield += 1;
                out.push((Cell::Nothing, Cell::Blank, true));
            }
            while to_yield > 0 {
                to_yield -= 1;
                out.push((Cell::Blank, Cell::Nothing, true));
            }
            if starts(b"X") {
                return out;
            }
            out.push((from, to, true));
        }
    }
}

/// `_line_pair_iterator()`: one from line and one to line per row.
fn pair_lines(lines: Vec<(Cell, Cell, bool)>) -> Vec<Row> {
    let mut out = Vec::with_capacity(lines.len());
    let mut froms: VecDeque<(Cell, bool)> = VecDeque::new();
    let mut tos: VecDeque<(Cell, bool)> = VecDeque::new();
    let mut it = lines.into_iter();
    loop {
        while froms.is_empty() || tos.is_empty() {
            let Some((from, to, diff)) = it.next() else {
                return out;
            };
            if from != Cell::Nothing {
                froms.push_back((from, diff));
            }
            if to != Cell::Nothing {
                tos.push_back((to, diff));
            }
        }
        let (from, fd) = froms.pop_front().expect("non-empty");
        let (to, td) = tos.pop_front().expect("non-empty");
        out.push(Row {
            from,
            to,
            flag: Some(fd || td),
        });
    }
}

/// The `context` branch of `_mdiff` (`context = numlines + 1`).
fn with_context(pairs: Vec<Row>, context: usize) -> Vec<Row> {
    let mut out = Vec::new();
    let mut it = pairs.into_iter();
    loop {
        let mut index = 0usize;
        let mut ring: Vec<Option<Row>> = vec![None; context];
        loop {
            let Some(row) = it.next() else { return out };
            let diff = row.flag == Some(true);
            ring[index % context] = Some(row);
            index += 1;
            if diff {
                break;
            }
        }
        let mut to_write;
        if index > context {
            out.push(Row {
                from: Cell::Nothing,
                to: Cell::Nothing,
                flag: None,
            });
            to_write = context;
        } else {
            to_write = index;
            index = 0;
        }
        while to_write > 0 {
            let i = index % context;
            index += 1;
            out.push(ring[i].clone().expect("a stored row"));
            to_write -= 1;
        }
        to_write = context - 1;
        while to_write > 0 {
            let Some(row) = it.next() else { return out };
            if row.flag == Some(true) {
                to_write = context - 1;
            } else {
                to_write -= 1;
            }
            out.push(row);
        }
    }
}
