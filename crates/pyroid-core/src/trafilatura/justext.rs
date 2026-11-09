// Derived from jusText 3.0.2 (BSD-2-Clause), © Jan Pomikalek; modified.
//! jusText as trafilatura.external.try_justext runs it: ParagraphMaker over `lxml.sax.saxify`,
//! classify_paragraphs(..., 50, 150, 0.1, 0.2, 0.25, no_headings=True) and
//! revise_paragraph_classification(..., 150), with every stoplist merged
//! (external.jt_stoplist_init).

use std::sync::LazyLock;

use rustc_hash::FxHashSet;

use super::matchers::normalize_whitespace;
use super::pystr::{self, py_len};
use crate::ftfy::UnicodeVersion;
use crate::html::tree::{Document, Kind, NIL, Name, NodeId, TagSet, names as t};

/// The merged stoplist, lower-cased (`generate_tables.py` writes it, one word per line).
static STOPLIST: LazyLock<FxHashSet<&'static str>> = LazyLock::new(|| {
    include_str!("stoplist.txt")
        .split('\n')
        .filter(|w| !w.is_empty())
        .collect()
});

static PARAGRAPH_TAGS: LazyLock<TagSet> = LazyLock::new(|| {
    TagSet::of(&[
        t::BODY,
        t::BLOCKQUOTE,
        t::CAPTION,
        t::CENTER,
        t::COL,
        t::COLGROUP,
        t::DD,
        t::DIV,
        t::DL,
        t::DT,
        t::FIELDSET,
        t::FORM,
        t::LEGEND,
        t::OPTGROUP,
        t::OPTION,
        t::P,
        t::PRE,
        t::TABLE,
        t::TD,
        t::TEXTAREA,
        t::TFOOT,
        t::TH,
        t::THEAD,
        t::TR,
        t::UL,
        t::LI,
        t::H1,
        t::H2,
        t::H3,
        t::H4,
        t::H5,
        t::H6,
    ])
});

struct Paragraph {
    dom_path: String,
    text_nodes: Vec<String>,
    chars_count_in_links: usize,
    /// `Paragraph.text`, computed when the paragraph is complete.
    text: String,
}

#[derive(Default)]
struct ParagraphMaker {
    path: Vec<Name>,
    paragraphs: Vec<Paragraph>,
    paragraph: Option<Paragraph>,
    link: bool,
    br: bool,
}

impl ParagraphMaker {
    fn dom(&self, d: &Document) -> String {
        let mut s = String::new();
        for (i, &n) in self.path.iter().enumerate() {
            if i > 0 {
                s.push('.');
            }
            s.push_str(d.name(n));
        }
        s
    }

    fn start_new(&mut self, d: &Document) {
        if let Some(mut p) = self.paragraph.take() {
            // `if self.paragraph and self.paragraph.contains_text()`: Paragraph defines __len__
            p.text = normalize_whitespace(pystr::strip(&p.text_nodes.concat()));
            if !p.text.is_empty() && !p.text_nodes.is_empty() {
                self.paragraphs.push(p);
            }
        }
        self.paragraph = Some(Paragraph {
            dom_path: self.dom(d),
            text_nodes: Vec::new(),
            chars_count_in_links: 0,
            text: String::new(),
        });
    }

    fn start(&mut self, d: &Document, name: Name) {
        self.path.push(name);
        if PARAGRAPH_TAGS.contains(name) || (name == t::BR && self.br) {
            self.start_new(d);
        } else {
            self.br = name == t::BR;
            if self.br {
                self.append_text(" ");
            } else if name == t::A {
                self.link = true;
            }
        }
    }

    fn end(&mut self, d: &Document, name: Name) {
        self.path.pop();
        if PARAGRAPH_TAGS.contains(name) {
            self.start_new(d);
        }
        if name == t::A {
            self.link = false;
        }
    }

    fn append_text(&mut self, s: &str) -> usize {
        let text = normalize_whitespace(s);
        let n = py_len(&text);
        if let Some(p) = self.paragraph.as_mut() {
            p.text_nodes.push(text);
        }
        n
    }

    fn characters(&mut self, s: &str) {
        if s.is_empty() || pystr::str_isspace(s) {
            return;
        }
        let n = self.append_text(s);
        if self.link {
            if let Some(p) = self.paragraph.as_mut() {
                p.chars_count_in_links += n;
            }
        }
        self.br = false;
    }
}

/// `lxml.sax.saxify(root, handler)`: element events, texts and tails (the root's too).
fn saxify(d: &Document, root: NodeId, pm: &mut ParagraphMaker) {
    // (node, children started)
    let mut stack: Vec<(NodeId, bool)> = vec![(root, false)];
    while let Some((id, started)) = stack.pop() {
        let n = d.n(id);
        if n.kind != Kind::Element {
            if let Some(s) = n.tail.as_deref().filter(|s| !s.is_empty()) {
                pm.characters(s);
            }
            continue;
        }
        if started {
            pm.end(d, n.tag);
            if let Some(s) = n.tail.as_deref().filter(|s| !s.is_empty()) {
                pm.characters(s);
            }
            continue;
        }
        pm.start(d, n.tag);
        if let Some(s) = n.text.as_deref().filter(|s| !s.is_empty()) {
            pm.characters(s);
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
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Short,
    NearGood,
    Good,
    Bad,
}

/// classify_paragraphs then revise_paragraph_classification (headings off).
#[allow(clippy::if_same_then_else, clippy::cast_precision_loss)]
fn classify(ps: &[Paragraph], v: UnicodeVersion) -> Vec<Class> {
    let (length_low, length_high, stop_low, stop_high, max_ld) =
        (50usize, 150usize, 0.1, 0.2, 0.25);
    let stoplist = &*STOPLIST;
    let mut ct = Vec::with_capacity(ps.len());
    for p in ps {
        let length = py_len(&p.text);
        let words: Vec<&str> = pystr::split_ws(&p.text).collect();
        let sd = if words.is_empty() {
            0.0
        } else {
            words
                .iter()
                .filter(|w| stoplist.contains(&*pystr::lower(w, v)))
                .count() as f64
                / words.len() as f64
        };
        let ld = if length == 0 {
            0.0
        } else {
            p.chars_count_in_links as f64 / length as f64
        };
        let c = if ld > max_ld {
            Class::Bad
        } else if p.text.contains('\u{a9}') || p.text.contains("&copy") {
            Class::Bad
        } else if p.dom_path.contains("select") {
            Class::Bad
        } else if length < length_low {
            if p.chars_count_in_links > 0 {
                Class::Bad
            } else {
                Class::Short
            }
        } else if sd >= stop_high {
            if length > length_high {
                Class::Good
            } else {
                Class::NearGood
            }
        } else if sd >= stop_low {
            Class::NearGood
        } else {
            Class::Bad
        };
        ct.push(c);
    }
    let neighbour = |ct: &[Class], i: usize, ignore_neargood: bool, forward: bool| -> Class {
        let mut j = i;
        loop {
            if forward {
                j += 1;
                if j >= ct.len() {
                    return Class::Bad;
                }
            } else {
                if j == 0 {
                    return Class::Bad;
                }
                j -= 1;
            }
            let c = ct[j];
            if c == Class::Good || c == Class::Bad || (c == Class::NearGood && !ignore_neargood) {
                return c;
            }
        }
    };
    let mut new_classes: Vec<(usize, Class)> = Vec::new();
    for i in 0..ct.len() {
        if ct[i] != Class::Short {
            continue;
        }
        let prev = neighbour(&ct, i, true, false);
        let next = neighbour(&ct, i, true, true);
        let c = if prev == Class::Good && next == Class::Good {
            Class::Good
        } else if prev == Class::Bad && next == Class::Bad {
            Class::Bad
        } else if (prev == Class::Bad && neighbour(&ct, i, false, false) == Class::NearGood)
            || (next == Class::Bad && neighbour(&ct, i, false, true) == Class::NearGood)
        {
            Class::Good
        } else {
            Class::Bad
        };
        new_classes.push((i, c));
    }
    for (i, c) in new_classes {
        ct[i] = c;
    }
    for i in 0..ct.len() {
        if ct[i] != Class::NearGood {
            continue;
        }
        let prev = neighbour(&ct, i, true, false);
        let next = neighbour(&ct, i, true, true);
        ct[i] = if prev == Class::Bad && next == Class::Bad {
            Class::Bad
        } else {
            Class::Good
        };
    }
    ct
}

/// The texts of the paragraphs jusText keeps, in order.
pub fn good_paragraphs(d: &Document, root: NodeId, v: UnicodeVersion) -> Vec<String> {
    let mut pm = ParagraphMaker::default();
    pm.start_new(d);
    saxify(d, root, &mut pm);
    pm.start_new(d);
    let classes = classify(&pm.paragraphs, v);
    pm.paragraphs
        .into_iter()
        .zip(classes)
        .filter(|(_, c)| *c == Class::Good)
        .map(|(p, _)| p.text)
        .collect()
}
