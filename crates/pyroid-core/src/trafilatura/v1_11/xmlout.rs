// Derived from trafilatura 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
//! trafilatura 1.11.0's xml.xmltotxt (include_formatting=False) and utils.sanitize (spaces not
//! preserved).

use super::super::pystr::{self, IntError, PyError, py_int, remove_control_characters};
use crate::ftfy::UnicodeVersion;
use crate::html::tree::{Document, Name, NodeId, names as t};

const NEWLINE_ELEMS: &[Name] = &[
    t::CODE,
    t::GRAPHIC,
    t::HEAD,
    t::LB,
    t::LIST,
    t::P,
    t::QUOTE,
    t::ROW,
    t::TABLE,
];
const SPECIAL_FORMATTING: &[Name] = &[t::DEL, t::HEAD, t::HI, t::REF];
/// The longest string `"|" * n` builds here: CPython would allocate it (or raise
/// MemoryError); past this the port reports MemoryError instead.
const MAX_REPEAT: i128 = 1 << 26;
/// The same bound on the whole output.
const MAX_OUT: usize = 1 << 28;
/// sys.maxsize on 64-bit CPython.
const PY_SSIZE_T_MAX: i128 = i64::MAX as i128;

fn is_tag(d: &Document, e: NodeId, tag: Name) -> bool {
    d.is_element(e) && d.tag(e) == tag
}

/// xml.replace_element_text without formatting.
fn replace_element_text(d: &Document, e: NodeId) -> String {
    let mut s = d.text(e).unwrap_or("").to_owned();
    if is_tag(d, e, t::REF) && !s.is_empty() {
        let link = format!("[{s}]");
        s = match d.attr(e, t::TARGET).filter(|x| !x.is_empty()) {
            Some(target) => format!("{link}({target})"),
            None => link,
        };
    }
    if is_tag(d, e, t::CELL) && !s.is_empty() && d.len(e) > 0 {
        if d.first_child(e).is_some_and(|c| is_tag(d, c, t::P)) {
            s.push(' ');
        }
    } else if is_tag(d, e, t::ITEM) && !s.is_empty() {
        s = format!("- {s}\n");
    }
    s
}

fn has_cell_ancestor(d: &Document, e: NodeId) -> bool {
    d.ancestors(e).any(|a| is_tag(d, a, t::CELL))
}

fn push(out: &mut String, s: &str) -> Result<(), PyError> {
    if out.len() + s.len() > MAX_OUT {
        return Err(PyError::Memory);
    }
    out.push_str(s);
    Ok(())
}

/// `ch * n` for a one- or four-character `ch`, as CPython raises for it.
fn repeat(out: &mut String, ch: &str, n: i128) -> Result<(), PyError> {
    if n <= 0 {
        return Ok(());
    }
    if n > PY_SSIZE_T_MAX {
        return Err(PyError::Overflow {
            message: "cannot fit 'int' into an index-sized integer",
        });
    }
    let unit = i128::try_from(ch.len()).unwrap_or(1);
    if unit > 1 && n > PY_SSIZE_T_MAX / unit {
        return Err(PyError::Overflow {
            message: "repeated string is too long",
        });
    }
    if n * unit > MAX_REPEAT {
        return Err(PyError::Memory);
    }
    let n = usize::try_from(n).unwrap_or(0);
    push(out, &ch.repeat(n))
}

/// The end of a row without text or tail: the missing cells' bars, and the separator under a
/// header row.
fn row_end(d: &Document, row: NodeId, uv: UnicodeVersion, out: &mut String) -> Result<(), PyError> {
    let max_span: i128 = match d.get(row, "span") {
        None => 1,
        Some(span) => match py_int(span, uv) {
            Ok(value) => value,
            Err(IntError::Invalid) => {
                return Err(PyError::IntLiteral {
                    literal: span.to_owned(),
                });
            }
            Err(IntError::TooManyDigits(digits)) => {
                return Err(PyError::IntMaxStrDigits { digits });
            }
        },
    };
    let cells = d
        .iter_vec(row, &crate::html::tree::Matcher::tags(&[t::CELL]), false)
        .len();
    repeat(
        out,
        "|",
        max_span.saturating_sub(i128::try_from(cells).unwrap_or(i128::MAX)),
    )?;
    push(out, "\n")?;
    if d.children(row)
        .any(|c| is_tag(d, c, t::CELL) && d.attr(c, t::ROLE) == Some("head"))
    {
        push(out, "\n")?;
        repeat(out, "---|", max_span)?;
        push(out, "\n")?;
    }
    Ok(())
}

/// xml.process_element, iteratively (the trees are at most a few hundred levels deep, below
/// CPython's recursion limit).
fn process_element(
    d: &Document,
    root: NodeId,
    v: UnicodeVersion,
    out: &mut String,
) -> Result<(), PyError> {
    // (node, children done)
    let mut stack: Vec<(NodeId, bool)> = vec![(root, false)];
    while let Some((e, done)) = stack.pop() {
        if !done {
            if d.text(e).is_some() {
                push(out, &replace_element_text(d, e))?;
            }
            stack.push((e, true));
            let kids: Vec<NodeId> = d.children(e).collect();
            for &k in kids.iter().rev() {
                stack.push((k, false));
            }
            continue;
        }
        let g = if d.is_element(e) { d.tag(e) } else { t::NONE };
        if d.text(e).is_none() && d.tail(e).is_none() {
            if g == t::GRAPHIC {
                let text = format!(
                    "{} {}",
                    d.attr(e, t::TITLE).unwrap_or(""),
                    d.get(e, "alt").unwrap_or("")
                );
                for s in [
                    "![",
                    pystr::strip(&text),
                    "]",
                    "(",
                    d.attr(e, t::SRC).unwrap_or(""),
                    ")",
                ] {
                    push(out, s)?;
                }
            } else if NEWLINE_ELEMS.contains(&g) {
                if g == t::ROW {
                    row_end(d, e, v, out)?;
                } else {
                    push(out, "\n")?;
                }
            } else if g != t::CELL {
                continue;
            }
        }
        if NEWLINE_ELEMS.contains(&g) && !has_cell_ancestor(d, e) {
            push(out, "\n\n")?;
        } else if g == t::CELL {
            push(out, " | ")?;
        } else if d.is_element(e) && d.tag_name(e) == "comments" {
            push(out, "\n\n")?;
        } else if !SPECIAL_FORMATTING.contains(&g) {
            push(out, " ")?;
        }
        if let Some(tail) = d.tail(e) {
            push(out, tail)?;
        }
    }
    Ok(())
}

/// utils.sanitize(text): line_processing on every line (entity spaces, control characters
/// out, whitespace collapsed), empty lines dropped.
fn sanitize(text: &str, v: UnicodeVersion) -> String {
    let mut out = String::with_capacity(text.len());
    for line in pystr::splitlines(text) {
        let l = line
            .replace("&#13;", "\r")
            .replace("&#10;", "\n")
            .replace("&nbsp;", "\u{a0}");
        let l = pystr::trim(&remove_control_characters(&l, v));
        if l.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&l);
    }
    if out.contains('\u{2424}') {
        out.replace('\u{2424}', "")
    } else {
        out
    }
}

/// xml.xmltotxt(body, include_formatting=False).
pub fn xmltotxt(d: &Document, body: NodeId, v: UnicodeVersion) -> Result<String, PyError> {
    let mut out = String::new();
    process_element(d, body, v, &mut out)?;
    Ok(pystr::html_unescape(&sanitize(&out, v))?.into_owned())
}
