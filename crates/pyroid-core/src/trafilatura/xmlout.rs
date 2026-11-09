// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi, and libxml2 2.14.6 (MIT),
// © Daniel Veillard and contributors; modified.
//! xml.py `xmltotxt` for include_formatting=False, utils.sanitize, and lxml's
//! `etree.tostring(node, method="xml")` (libxml2 xmlsave.c) for readability's summary and the
//! bare_extraction trees.

use super::main_ext::{INLINE_CARRIED, INLINE_CONSUMING, INLINE_FORMATTABLE};
use super::matchers::escape_brackets;
use super::pystr::{self, PyError, is_printable, is_space, text_chars_test};
use crate::ftfy::UnicodeVersion;
use crate::html::tree::{Document, Kind, Matcher, NIL, Name, NodeId, names as t};

const NEWLINE_ELEMS: &[Name] = &[
    t::GRAPHIC,
    t::HEAD,
    t::LB,
    t::LIST,
    t::P,
    t::QUOTE,
    t::ROW,
    t::TABLE,
];
const SPECIAL_FORMATTING: &[Name] = &[t::CODE, t::DEL, t::HEAD, t::HI, t::REF, t::ITEM, t::CELL];
/// Inline nesting deeper than this is not followed (HTML trees are at most 256 deep).
const MAX_INLINE_DEPTH: usize = 4096;

/// The `returnlist` of process_element: only the concatenation and the last entry matter.
struct Out {
    s: String,
    /// last character of the most recently appended entry (None: nothing yet, or it was "")
    last: Option<char>,
}

impl Out {
    fn push(&mut self, x: &str) {
        self.s.push_str(x);
        self.last = x.chars().next_back();
    }
    /// `_last_char(returnlist) in SEPARATORS` (" ", "\n", "|", "")
    fn last_is_sep(&self) -> bool {
        matches!(self.last, None | Some(' ' | '\n' | '|'))
    }
    /// xml._append_spaced.
    fn append_spaced(&mut self, text: &str) {
        let text = if text.starts_with(' ') && self.last_is_sep() {
            &text[1..]
        } else {
            text
        };
        if !text.is_empty() {
            self.push(text);
        }
    }
}

fn escape_cell(text: String, in_cell: bool) -> String {
    if in_cell {
        text.replace('|', "\\|").replace('\n', " ")
    } else {
        text
    }
}

#[inline]
fn tag_of(d: &Document, e: NodeId) -> Name {
    if d.is_element(e) { d.tag(e) } else { t::NONE }
}

fn item_needs_marker(d: &Document, item: NodeId) -> bool {
    if text_chars_test(d.text(item)) {
        return true;
    }
    for c in d.children(item) {
        if tag_of(d, c) == t::LIST {
            return false;
        }
        let s = d.itertext_join(c, "") + d.tail(c).unwrap_or("");
        if text_chars_test(Some(&s))
            || !d
                .iter_vec(c, &Matcher::tags(&[t::GRAPHIC]), true)
                .is_empty()
        {
            return true;
        }
    }
    false
}

fn list_marker(d: &Document, item: NodeId) -> String {
    let n = d
        .ancestors(item)
        .filter(|&a| tag_of(d, a) == t::LIST)
        .count();
    format!("{}- ", "  ".repeat(n.saturating_sub(1)))
}

fn closes_container(d: &Document, e: NodeId, tag: Name) -> bool {
    if tag_of(d, e) == tag {
        return d.len(e) == 0;
    }
    match d.next(e) {
        None => true,
        Some(n) => tag == t::ITEM && tag_of(d, n) == t::ITEM,
    }
}

fn opens_item(d: &Document, mut e: NodeId) -> bool {
    while tag_of(d, e) != t::ITEM {
        let Some(p) = d.parent(e) else { return false };
        if d.prev(e).is_some() || text_chars_test(d.text(p)) {
            return false;
        }
        e = p;
    }
    true
}

/// xml._md_link.
fn md_link(text: &str, url: Option<&str>, image: bool) -> String {
    let esc = escape_brackets(text);
    let prefix = if image { "!" } else { "" };
    match url {
        None => format!("{prefix}[{esc}]"),
        Some(u) => {
            let safe = if u.contains([' ', '<', '>', '(', ')']) {
                format!(
                    "<{}>",
                    u.replace('\\', "\\\\")
                        .replace('<', "\\<")
                        .replace('>', "\\>")
                )
            } else {
                u.to_owned()
            };
            format!("{prefix}[{esc}]({safe})")
        }
    }
}

fn image_markup(d: &Document, e: NodeId) -> String {
    let alt = format!(
        "{} {}",
        d.attr(e, t::TITLE).unwrap_or(""),
        d.attr(e, t::ALT).unwrap_or("")
    );
    md_link(
        pystr::strip(&alt),
        Some(d.attr(e, t::SRC).unwrap_or("")),
        true,
    )
}

fn consumes_inline_children(d: &Document, e: NodeId) -> bool {
    INLINE_CONSUMING.contains(&tag_of(d, e)) && d.len(e) > 0
}

fn collect_inline_text(d: &Document, e: NodeId, depth: usize) -> String {
    let mut parts = String::new();
    if let Some(s) = d.text(e) {
        parts.push_str(s);
    }
    for c in d.children(e) {
        let g = tag_of(d, c);
        if g == t::GRAPHIC {
            parts.push_str(&image_markup(d, c));
        } else if g == t::LB {
            parts.push('\n');
        } else if INLINE_FORMATTABLE.contains(&g) {
            if depth < MAX_INLINE_DEPTH {
                parts.push_str(&replace_element_text(d, c, depth + 1));
            }
        } else if let Some(s) = d.text(c).filter(|_| d.is_element(c)) {
            parts.push_str(s);
        }
        if let Some(s) = d.tail(c) {
            parts.push_str(s);
        }
    }
    parts
}

/// xml.replace_element_text with include_formatting=False.
fn replace_element_text(d: &Document, e: NodeId, depth: usize) -> String {
    let mut elem_text = if consumes_inline_children(d, e) {
        collect_inline_text(d, e, depth)
    } else {
        d.text(e).unwrap_or("").to_owned()
    };
    let g = tag_of(d, e);
    if g == t::REF {
        let stripped = pystr::strip(&elem_text).to_owned();
        if !stripped.is_empty() {
            let target = d.attr(e, t::TARGET).filter(|s| !s.is_empty());
            let link = md_link(&stripped, target, false);
            elem_text = elem_text.replacen(&stripped, &link, 1);
        }
    }
    if g == t::CELL {
        elem_text = pystr::strip(&elem_text).to_owned();
        if !elem_text.is_empty() && d.len(e) > 0 {
            elem_text.push(' ');
        }
    }
    elem_text
}

fn append_tail(d: &Document, e: NodeId, out: &mut Out, in_cell: bool, in_item: bool) {
    let Some(tail) = d.tail(e).filter(|s| !s.is_empty()) else {
        return;
    };
    let g = tag_of(d, e);
    if in_cell || in_item || g == t::LIST {
        let mut core = pystr::strip(tail).to_owned();
        let first_space = tail.chars().next().is_some_and(is_space);
        if !core.is_empty() && (first_space || !INLINE_FORMATTABLE.contains(&g)) {
            core = format!(" {core}");
        }
        let last_space = tail.chars().next_back().is_some_and(is_space);
        if last_space
            && d.next(e)
                .is_some_and(|n| INLINE_CARRIED.contains(&tag_of(d, n)))
        {
            core.push(' ');
        }
        out.append_spaced(&escape_cell(core, in_cell));
        return;
    }
    let tail = if NEWLINE_ELEMS.contains(&g) {
        pystr::lstrip(tail)
    } else {
        tail
    };
    out.push(tail);
}

/// One pending step of process_element, which recurses into the children between the part
/// before them (Enter) and the part after (Exit).
enum Step {
    Enter(NodeId, bool, bool),
    Exit {
        e: NodeId,
        in_cell: bool,
        in_item: bool,
        renders_inline: bool,
    },
}

/// xml.process_element, iteratively.
// the branches mirror process_element's elif chain
#[allow(clippy::if_same_then_else)]
fn process_element(d: &Document, root: NodeId, out: &mut Out) {
    let mut stack = vec![Step::Enter(root, false, false)];
    while let Some(step) = stack.pop() {
        match step {
            Step::Enter(e, in_cell, in_item) => {
                let g = tag_of(d, e);
                let in_cell = in_cell || g == t::CELL;
                let in_item = in_item || g == t::ITEM;
                if g == t::CELL && d.prev(e).is_none() {
                    out.push("| ");
                }
                let text = d.text(e).filter(|_| d.is_element(e));
                if NEWLINE_ELEMS.contains(&g) && !in_cell && !in_item && !out.last_is_sep() {
                    out.push("\n");
                } else if [t::HEAD, t::P, t::QUOTE, t::TABLE].contains(&g)
                    && !out.last.is_some_and(is_space)
                    && !out.last_is_sep()
                    && (text.is_some_and(|s| !s.is_empty())
                        || d.len(e) == 0
                        || d.first_child(e).is_some_and(|c| tag_of(d, c) != t::LB))
                {
                    out.push(" ");
                }
                let consumes = consumes_inline_children(d, e);
                let renders_inline = text.is_some_and(|s| !s.is_empty()) || consumes;
                if g == t::ITEM && !in_cell && item_needs_marker(d, e) {
                    out.push(&list_marker(d, e));
                }
                if renders_inline && (g != t::ITEM || consumes || text_chars_test(text)) {
                    out.push(&escape_cell(replace_element_text(d, e, 0), in_cell));
                }
                if g == t::LIST && in_item && !matches!(out.last, None | Some('\n')) {
                    out.push("\n");
                }
                stack.push(Step::Exit {
                    e,
                    in_cell,
                    in_item,
                    renders_inline,
                });
                if !consumes {
                    let mut kids = Vec::new();
                    let mut c = d.n(e).first;
                    while c != NIL {
                        kids.push(c);
                        c = d.n(c).next;
                    }
                    for &k in kids.iter().rev() {
                        stack.push(Step::Enter(k, in_cell, in_item));
                    }
                }
            }
            Step::Exit {
                e,
                in_cell,
                in_item,
                renders_inline,
            } => {
                let g = tag_of(d, e);
                if !renders_inline {
                    if g == t::GRAPHIC {
                        out.push(&escape_cell(image_markup(d, e), in_cell));
                        if let Some(tl) = d.tail(e).filter(|s| !s.is_empty()) {
                            out.push(&escape_cell(format!(" {}", pystr::strip(tl)), in_cell));
                        }
                    } else if NEWLINE_ELEMS.contains(&g) {
                        if g == t::ROW {
                            let cells: Vec<NodeId> =
                                d.children(e).filter(|&c| tag_of(d, c) == t::CELL).collect();
                            if cells.iter().any(|&c| d.attr(c, t::ROLE) == Some("head")) {
                                out.push(&format!("\n|{}\n", "---|".repeat(cells.len())));
                            }
                        } else if !(in_cell || (g == t::LB && in_item && opens_item(d, e))) {
                            out.push("\n");
                        }
                    } else if g != t::CELL && g != t::ITEM {
                        if in_cell {
                            append_tail(d, e, out, in_cell, in_item);
                        }
                        continue;
                    }
                }
                let last_in_item = in_item && closes_container(d, e, t::ITEM);
                let last_in_cell = in_cell && closes_container(d, e, t::CELL);
                if NEWLINE_ELEMS.contains(&g) && !in_cell && !in_item {
                    out.push("\n");
                } else if g == t::CELL {
                    out.push(" | ");
                } else if (g == t::HEAD || g == t::ITEM) && in_cell && !last_in_cell {
                    out.append_spaced(" ");
                } else if !SPECIAL_FORMATTING.contains(&g) && !last_in_item && !last_in_cell {
                    out.append_spaced(" ");
                }
                if g != t::GRAPHIC {
                    append_tail(d, e, out, in_cell, in_item);
                }
                if last_in_item && !in_cell {
                    out.push("\n");
                }
            }
        }
    }
}

/// utils.line_processing(line, preserve_space=True): entity spaces, remove_control_characters.
fn line_processing_preserve(line: &str, v: UnicodeVersion) -> String {
    let l = line
        .replace("&#13;", "\r")
        .replace("&#10;", "\n")
        .replace("&nbsp;", "\u{a0}");
    if l.chars().all(|c| is_printable(c, v)) {
        return l;
    }
    l.chars()
        .filter(|&c| is_printable(c, v) || is_space(c))
        .collect()
}

/// utils.sanitize(text, preserve_space=True).
pub fn sanitize_preserve(text: &str, v: UnicodeVersion) -> String {
    let mut out = String::with_capacity(text.len());
    let mut first = true;
    for line in pystr::splitlines(text) {
        let l = line_processing_preserve(line, v);
        if l.is_empty() {
            continue;
        }
        if !first {
            out.push('\n');
        }
        first = false;
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
    let mut out = Out {
        s: String::new(),
        last: None,
    };
    process_element(d, body, &mut out);
    Ok(pystr::html_unescape(&sanitize_preserve(&out.s, v))?.into_owned())
}

/// libxml2 xmlEscapeText for text content and attribute values (UTF-8 output, so non-ASCII is
/// written as is): `&`, `<`, `>` always; `\r` as `&#13;`; in attributes also `"`, `\t`, `\n`;
/// other C0 controls become `&#xFFFD;`.
fn xml_escape(s: &str, attr: bool, out: &mut String) {
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '\r' => out.push_str("&#13;"),
            '"' if attr => out.push_str("&quot;"),
            '\t' if attr => out.push_str("&#9;"),
            '\n' if attr => out.push_str("&#10;"),
            '\t' | '\n' => out.push(c),
            '\u{1}'..='\u{1f}' => out.push_str("&#xFFFD;"),
            _ => out.push(c),
        }
    }
}

fn open_tag(d: &Document, id: NodeId, out: &mut String) {
    out.push('<');
    out.push_str(d.tag_name(id));
    for a in &d.n(id).attrs {
        out.push(' ');
        out.push_str(d.name(a.name));
        out.push_str("=\"");
        xml_escape(&a.value, true, out);
        out.push('"');
    }
}

/// lxml `etree.tostring(node, encoding=str, method="xml")` (with_tail=True, no pretty
/// printing): libxml2 xmlNodeDumpOutput for a node of a non-XHTML document, then the tail.
pub fn tostring_xml(d: &Document, root: NodeId) -> String {
    let mut out = String::new();
    // (node, closing)
    let mut stack: Vec<(NodeId, bool)> = vec![(root, false)];
    while let Some((id, closing)) = stack.pop() {
        let n = d.n(id);
        if closing {
            out.push_str("</");
            out.push_str(d.tag_name(id));
            out.push('>');
        } else {
            match n.kind {
                Kind::Element => {
                    open_tag(d, id, &mut out);
                    // libxml2 writes `<x/>` when the node has no children at all, text included
                    if n.text.is_none() && n.first == NIL {
                        out.push_str("/>");
                    } else {
                        out.push('>');
                        if let Some(s) = &n.text {
                            xml_escape(s, false, &mut out);
                        }
                        stack.push((id, true));
                        let mut kids = Vec::new();
                        let mut c = n.first;
                        while c != NIL {
                            kids.push(c);
                            c = d.n(c).next;
                        }
                        for &k in kids.iter().rev() {
                            stack.push((k, false));
                        }
                        continue;
                    }
                }
                Kind::Comment => {
                    out.push_str("<!--");
                    out.push_str(n.text.as_deref().unwrap_or(""));
                    out.push_str("-->");
                }
                Kind::Pi | Kind::Entity => {}
            }
        }
        // the node is finished: its tail (the root's too, with_tail=True)
        if let Some(s) = &d.n(id).tail {
            xml_escape(s, false, &mut out);
        }
    }
    out
}
