// Derived from jusText 3.0.2 (BSD-2-Clause), © Jan Pomikalek; modified.
//! jusText as trafilatura.external.try_justext runs it: ParagraphMaker over `lxml.sax.saxify`,
//! classify_paragraphs with the release's thresholds ([`JtParams`]; 2.3.1: 50, 150, 0.1, 0.2,
//! 0.25; 1.11.0: 50, 200, 0.1, 0.2, 0.2; both no_headings=True) and
//! revise_paragraph_classification, with every stoplist merged (2.3.1's
//! external.jt_stoplist_init; 1.11.0 loads the same words from its pickle).

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

/// The classify_paragraphs thresholds a Trafilatura release passes (no_headings=True, so
/// revise_paragraph_classification's max_heading_distance plays no part).
#[derive(Clone, Copy, Debug)]
pub struct JtParams {
    pub length_low: usize,
    pub length_high: usize,
    pub stopwords_low: f64,
    pub stopwords_high: f64,
    pub max_link_density: f64,
}

/// trafilatura 2.3.1: classify_paragraphs(..., 50, 150, 0.1, 0.2, 0.25, True).
pub const JT_2_3_1: JtParams = JtParams {
    length_low: 50,
    length_high: 150,
    stopwords_low: 0.1,
    stopwords_high: 0.2,
    max_link_density: 0.25,
};

/// classify_paragraphs then revise_paragraph_classification (headings off).
#[allow(clippy::if_same_then_else, clippy::cast_precision_loss)]
fn classify(ps: &[Paragraph], v: UnicodeVersion, jp: &JtParams) -> Vec<Class> {
    let (length_low, length_high, stop_low, stop_high, max_ld) = (
        jp.length_low,
        jp.length_high,
        jp.stopwords_low,
        jp.stopwords_high,
        jp.max_link_density,
    );
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
    revise(&mut ct);
    ct
}

/// revise_paragraph_classification with no_headings=True: the short paragraphs, then the
/// near-good ones, take a class from their neighbours.
///
/// jusText's _get_neighbour walks from a paragraph to the nearest one of the classes it stops
/// on, which is quadratic over a run of short paragraphs (a page of many small blocks). The
/// same answers come from one pass each way (see `neighbours`).
fn revise(ct: &mut [Class]) {
    let good_bad = |c: Class| c == Class::Good || c == Class::Bad;
    let not_short = |c: Class| c != Class::Short;
    // classify short: the loop reads the classes from before it (new_classes is applied after)
    let prev_gb = neighbours(ct, good_bad, false);
    let next_gb = neighbours(ct, good_bad, true);
    let prev_ns = neighbours(ct, not_short, false);
    let next_ns = neighbours(ct, not_short, true);
    let mut new_classes: Vec<(usize, Class)> = Vec::new();
    for i in 0..ct.len() {
        if ct[i] != Class::Short {
            continue;
        }
        let (prev, next) = (prev_gb[i], next_gb[i]);
        let c = if prev == Class::Good && next == Class::Good {
            Class::Good
        } else if prev == Class::Bad && next == Class::Bad {
            Class::Bad
        } else if (prev == Class::Bad && prev_ns[i] == Class::NearGood)
            || (next == Class::Bad && next_ns[i] == Class::NearGood)
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
    // revise neargood: this loop updates the classes as it goes. The paragraphs after the current
    // one are not updated yet, so the next neighbours come from the classes the loop starts with;
    // the previous neighbour is the nearest good or bad class before it, updated ones included.
    let next_gb = neighbours(ct, good_bad, true);
    let mut prev = Class::Bad;
    for (c, &next) in ct.iter_mut().zip(&next_gb) {
        if *c == Class::NearGood {
            *c = if prev == Class::Bad && next == Class::Bad {
                Class::Bad
            } else {
                Class::Good
            };
        }
        if good_bad(*c) {
            prev = *c;
        }
    }
}

/// jusText's _get_neighbour for every paragraph: the class of the nearest paragraph before it
/// (after it with `forward`) whose class `stops` on, or bad when there is none.
fn neighbours(ct: &[Class], stops: impl Fn(Class) -> bool, forward: bool) -> Vec<Class> {
    let mut out = vec![Class::Bad; ct.len()];
    let mut seen = Class::Bad;
    let mut visit = |i: usize| {
        out[i] = seen;
        if stops(ct[i]) {
            seen = ct[i];
        }
    };
    if forward {
        (0..ct.len()).rev().for_each(&mut visit);
    } else {
        (0..ct.len()).for_each(&mut visit);
    }
    out
}

/// The texts of the paragraphs jusText keeps, in order.
pub fn good_paragraphs(
    d: &Document,
    root: NodeId,
    v: UnicodeVersion,
    jp: &JtParams,
) -> Vec<String> {
    let mut pm = ParagraphMaker::default();
    pm.start_new(d);
    saxify(d, root, &mut pm);
    pm.start_new(d);
    let classes = classify(&pm.paragraphs, v, jp);
    pm.paragraphs
        .into_iter()
        .zip(classes)
        .filter(|(_, c)| *c == Class::Good)
        .map(|(p, _)| p.text)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// revise_paragraph_classification as jusText writes it, _get_neighbour walking to the
    /// nearest paragraph of a class it stops on (this module's code before `neighbours`).
    fn revise_walking(ct: &mut [Class]) {
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
                if c == Class::Good || c == Class::Bad || (c == Class::NearGood && !ignore_neargood)
                {
                    return c;
                }
            }
        };
        let mut new_classes: Vec<(usize, Class)> = Vec::new();
        for i in 0..ct.len() {
            if ct[i] != Class::Short {
                continue;
            }
            let prev = neighbour(ct, i, true, false);
            let next = neighbour(ct, i, true, true);
            let c = if prev == Class::Good && next == Class::Good {
                Class::Good
            } else if prev == Class::Bad && next == Class::Bad {
                Class::Bad
            } else if (prev == Class::Bad && neighbour(ct, i, false, false) == Class::NearGood)
                || (next == Class::Bad && neighbour(ct, i, false, true) == Class::NearGood)
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
            let prev = neighbour(ct, i, true, false);
            let next = neighbour(ct, i, true, true);
            ct[i] = if prev == Class::Bad && next == Class::Bad {
                Class::Bad
            } else {
                Class::Good
            };
        }
    }

    const ALL: [Class; 4] = [Class::Short, Class::NearGood, Class::Good, Class::Bad];

    fn check(ct: &[Class]) {
        let (mut a, mut b) = (ct.to_vec(), ct.to_vec());
        revise(&mut a);
        revise_walking(&mut b);
        assert_eq!(a, b, "classes {ct:?}");
    }

    #[test]
    fn revise_matches_walking_exhaustive() {
        // every sequence of up to 8 classes
        for n in 0..=8u32 {
            for code in 0..4usize.pow(n) {
                let ct: Vec<Class> = (0..n).map(|k| ALL[(code >> (2 * k)) & 3]).collect();
                check(&ct);
            }
        }
    }

    #[test]
    fn revise_matches_walking_random() {
        // xorshift64; long runs of one class are likely with the skewed draws
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for _ in 0..3000 {
            let n = usize::try_from(next() % 300).unwrap();
            let skew = next() % 4;
            let ct: Vec<Class> = (0..n)
                .map(|_| {
                    let r = next();
                    if r % 8 < 5 {
                        ALL[usize::try_from(skew).unwrap()]
                    } else {
                        ALL[usize::try_from((r >> 8) % 4).unwrap()]
                    }
                })
                .collect();
            check(&ct);
        }
    }

    #[test]
    fn revise_is_linear() {
        // A million short, then near-good, paragraphs between two good ones: the walking version
        // reads about 10^12 classes for the first and 5 * 10^11 for the second, this one a few
        // million. The deadline is far from both.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for fill in [Class::Short, Class::NearGood] {
                let mut ct = vec![fill; 1_000_002];
                ct[0] = Class::Good;
                ct[1_000_001] = Class::Good;
                revise(&mut ct);
                let _ = tx.send(ct.iter().all(|&c| c == Class::Good));
            }
        });
        for _ in 0..2 {
            let all_good = rx
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("revise_paragraph_classification is not linear in the paragraphs");
            assert!(all_good);
        }
    }
}
