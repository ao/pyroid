//! `HtmlDiff.make_table`'s rows: tab expansion, `_mdiff`, line wrapping,
//! `_format_line` and `_convert_flags`, as CPython writes them (the templates
//! around the rows stay Python's).

use super::differ::Side;
use super::mdiff::{Cell, Row, Text};
use crate::text::is_py_whitespace;

/// `_tab_newline_replace`'s `expand_tabs`: tabs to `\t` runs up to the next
/// stop (spaces stay spaces), then trailing newlines removed.
pub fn expand_tabs(line: &[u32], tabsize: i64) -> Vec<u32> {
    const NUL: u32 = 0;
    let (space, tab) = (u32::from(b' '), u32::from(b'\t'));
    let mut out = Vec::with_capacity(line.len());
    let mut col: i64 = 0;
    // line.replace(' ', '\0').expandtabs(tabsize).replace(' ', '\t').replace('\0', ' ')
    for &c in line {
        let c = if c == space { NUL } else { c };
        if c == tab {
            if tabsize > 0 {
                let incr = tabsize - col.rem_euclid(tabsize);
                col += incr;
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // 1..=tabsize
                out.extend(std::iter::repeat_n(tab, incr as usize));
            }
        } else {
            col = if c == u32::from(b'\n') || c == u32::from(b'\r') {
                0
            } else {
                col + 1
            };
            out.push(if c == NUL { space } else { c });
        }
    }
    while out.last() == Some(&u32::from(b'\n')) {
        out.pop();
    }
    out
}

/// A cell's line number: a number, or `''`/`'>'` (blank, wrapped).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Num {
    Line(usize),
    Blank,
    Wrapped,
}

/// One side of a row, before `_format_line`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HCell {
    pub num: Num,
    pub text: Vec<u32>,
}

/// One row of `_collect_lines`: a pair of cells, or a context separator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HRow {
    Pair(HCell, HCell, bool),
    Separator,
}

fn cell(c: &Cell, lines: &dyn Fn(Side, usize) -> Vec<u32>) -> Option<HCell> {
    match c {
        Cell::Nothing => None,
        Cell::Blank => Some(HCell {
            num: Num::Blank,
            text: vec![u32::from(b'\n')],
        }),
        Cell::Line(n, Text::Line(side, i)) => Some(HCell {
            num: Num::Line(*n),
            text: lines(*side, *i),
        }),
        Cell::Line(n, Text::Marked(t)) => Some(HCell {
            num: Num::Line(*n),
            text: t.clone(),
        }),
    }
}

/// `_split_line(data_list, line_num, text)`, iteratively.
fn split_line(out: &mut Vec<HCell>, mut c: HCell, max: usize) {
    loop {
        if c.num == Num::Blank {
            out.push(c);
            return;
        }
        let size = c.text.len();
        let nuls = c.text.iter().filter(|&&x| x == 0).count();
        if size <= max || size.saturating_sub(nuls * 3) <= max {
            out.push(c);
            return;
        }
        let text = &c.text;
        let (mut i, mut n, mut mark) = (0usize, 0usize, None);
        while n < max && i < size {
            if text[i] == 0 {
                i += 1;
                mark = text.get(i).copied();
                i += 1;
            } else if text[i] == 1 {
                i += 1;
                mark = None;
            } else {
                i += 1;
                n += 1;
            }
        }
        let i = i.min(size);
        let mut line1 = text[..i].to_vec();
        let mut line2 = text[i..].to_vec();
        if let Some(m) = mark {
            line1.push(1);
            line2.splice(0..0, [0, m]);
        }
        out.push(HCell {
            num: c.num,
            text: line1,
        });
        c = HCell {
            num: Num::Wrapped,
            text: line2,
        };
    }
}

/// `_collect_lines(diffs)`'s input: `_mdiff`'s rows, wrapped at `wrapcolumn`
/// (`_line_wrapper`) if it is not 0. `lines(side, i)` is an input line's text.
pub fn rows(
    mdiff: &[Row],
    lines: &dyn Fn(Side, usize) -> Vec<u32>,
    wrapcolumn: usize,
) -> Vec<HRow> {
    let mut out = Vec::with_capacity(mdiff.len());
    for row in mdiff {
        let (Some(flag), Some(from), Some(to)) =
            (row.flag, cell(&row.from, lines), cell(&row.to, lines))
        else {
            // A context separator (None, None, None): _format_line(*None) raises TypeError.
            out.push(HRow::Separator);
            continue;
        };
        if wrapcolumn == 0 {
            out.push(HRow::Pair(from, to, flag));
            continue;
        }
        let (mut from_cells, mut to_cells) = (Vec::new(), Vec::new());
        split_line(&mut from_cells, from, wrapcolumn);
        split_line(&mut to_cells, to, wrapcolumn);
        let filler = || HCell {
            num: Num::Blank,
            text: vec![u32::from(b' ')],
        };
        let k = from_cells.len().max(to_cells.len());
        let mut from_cells = from_cells.into_iter();
        let mut to_cells = to_cells.into_iter();
        for _ in 0..k {
            let f = from_cells.next().unwrap_or_else(filler);
            let t = to_cells.next().unwrap_or_else(filler);
            out.push(HRow::Pair(f, t, flag));
        }
    }
    out
}

/// Where rendered text goes: a `String`, or code points when a line has a
/// lone surrogate.
pub trait Sink {
    fn push_str(&mut self, s: &str);
    fn push_cp(&mut self, c: u32);
}

impl Sink for String {
    fn push_str(&mut self, s: &str) {
        self.push_str(s);
    }
    fn push_cp(&mut self, c: u32) {
        self.push(char::from_u32(c).unwrap_or(char::REPLACEMENT_CHARACTER));
    }
}

impl Sink for Vec<u32> {
    fn push_str(&mut self, s: &str) {
        self.extend(s.chars().map(u32::from));
    }
    fn push_cp(&mut self, c: u32) {
        self.push(c);
    }
}

/// True if a cell holds a lone surrogate (render into code points then).
pub fn has_surrogates(rows: &[HRow]) -> bool {
    rows.iter().any(|r| match r {
        HRow::Pair(f, t, _) => [f, t]
            .iter()
            .any(|c| c.text.iter().any(|&x| char::from_u32(x).is_none())),
        HRow::Separator => false,
    })
}

/// `_format_line(side, flag, linenum, text)`, then `make_table`'s final
/// replacements (`\0+`, `\0-`, `\0^`, `\1`, `\t` to markup); `changed_class`:
/// CPython 3.15's `class="diff_changed"` on changed lines.
fn format_line(
    out: &mut impl Sink,
    buf: &mut Vec<u32>,
    prefix: &str,
    c: &HCell,
    flag: bool,
    changed_class: bool,
) {
    let num = match c.num {
        Num::Line(n) => n.to_string(),
        Num::Blank => String::new(),
        Num::Wrapped => ">".to_owned(),
    };
    out.push_str("<td class=\"diff_header\"");
    if matches!(c.num, Num::Line(_)) {
        out.push_str(&format!(" id=\"{prefix}{num}\""));
    }
    out.push_str(&format!(">{num}</td><td"));
    out.push_str(if changed_class && flag {
        " class=\"diff_changed\" "
    } else {
        " "
    });
    out.push_str("nowrap=\"nowrap\">");
    // text.replace("&", "&amp;").replace(">", "&gt;").replace("<", "&lt;")
    //     .replace(" ", "&nbsp;").rstrip()
    buf.clear();
    for &ch in &c.text {
        match ch {
            0x26 => buf.extend("&amp;".chars().map(u32::from)),
            0x3E => buf.extend("&gt;".chars().map(u32::from)),
            0x3C => buf.extend("&lt;".chars().map(u32::from)),
            0x20 => buf.extend("&nbsp;".chars().map(u32::from)),
            _ => buf.push(ch),
        }
    }
    while buf
        .last()
        .is_some_and(|&x| char::from_u32(x).is_some_and(is_py_whitespace))
    {
        buf.pop();
    }
    let mut i = 0;
    while i < buf.len() {
        let ch = buf[i];
        let span = match (ch, buf.get(i + 1)) {
            (0, Some(0x2B)) => Some("<span class=\"diff_add\">"),
            (0, Some(0x2D)) => Some("<span class=\"diff_sub\">"),
            (0, Some(0x5E)) => Some("<span class=\"diff_chg\">"),
            _ => None,
        };
        if let Some(span) = span {
            out.push_str(span);
            i += 2;
            continue;
        }
        match ch {
            1 => out.push_str("</span>"),
            9 => out.push_str("&nbsp;"),
            _ => out.push_cp(ch),
        }
        i += 1;
    }
    out.push_str("</td>");
}

/// `make_table`'s `data_rows` after its final replacements: `_collect_lines`,
/// `_convert_flags` and the row format, with the prefixes `_make_prefix`
/// chose. (No replacement can span the rows' ends: they start with spaces
/// and end with a newline.)
pub fn render(
    out: &mut impl Sink,
    rows: &[HRow],
    prefixes: [&str; 2],
    context: bool,
    numlines: usize,
    changed_class: bool,
) {
    let toprefix = prefixes[1];
    let n = rows.len();
    let mut next_id = vec![String::new(); n.max(1)];
    let mut next_href = vec![String::new(); n.max(1)];
    let (mut num_chg, mut in_change, mut last) = (0usize, false, 0usize);
    for (i, row) in rows.iter().enumerate() {
        if matches!(row, HRow::Pair(_, _, true)) {
            if !in_change {
                in_change = true;
                last = i;
                next_id[i.saturating_sub(numlines)] =
                    format!(" id=\"difflib_chg_{toprefix}_{num_chg}\"");
                num_chg += 1;
                next_href[last] = format!("<a href=\"#difflib_chg_{toprefix}_{num_chg}\">n</a>");
            }
        } else {
            in_change = false;
        }
    }
    let first_changed = matches!(rows.first(), Some(HRow::Pair(_, _, true)));
    if !first_changed {
        next_href[0] = format!("<a href=\"#difflib_chg_{toprefix}_0\">f</a>");
    }
    next_href[last] = format!("<a href=\"#difflib_chg_{toprefix}_top\">t</a>");
    let start = |out: &mut dyn FnMut(&str), i: usize| {
        out("            <tr><td class=\"diff_next\"");
        out(&next_id[i]);
        out(">");
        out(&next_href[i]);
        out("</td>");
    };
    let middle = |out: &mut dyn FnMut(&str), i: usize| {
        out("<td class=\"diff_next\">");
        out(&next_href[i]);
        out("</td>");
    };
    if rows.is_empty() {
        let text = if context {
            "<td></td><td>&nbsp;No Differences Found&nbsp;</td>"
        } else {
            "<td></td><td>&nbsp;Empty File&nbsp;</td>"
        };
        let mut w = |s: &str| out.push_str(s);
        start(&mut w, 0);
        w(text);
        middle(&mut w, 0);
        w(text);
        w("</tr>\n");
        return;
    }
    let mut buf = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        match r {
            HRow::Separator => {
                if i > 0 {
                    out.push_str("        </tbody>        \n        <tbody>\n");
                }
            }
            HRow::Pair(f, t, flag) => {
                start(&mut |s: &str| out.push_str(s), i);
                format_line(out, &mut buf, prefixes[0], f, *flag, changed_class);
                middle(&mut |s: &str| out.push_str(s), i);
                format_line(out, &mut buf, prefixes[1], t, *flag, changed_class);
                out.push_str("</tr>\n");
            }
        }
    }
}
