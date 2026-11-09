// Derived from readability-lxml as vendored in trafilatura 2.3.1 (Apache-2.0); modified.
//! readability_lxml.Document(doc, min_text_length=25).summary() and external.try_readability:
//! the summary is serialized with lxml's XML serializer and parsed again with lxml's HTML
//! parser, as `fromstring_bytes(doc.summary())` does.

use std::collections::HashMap;

use super::pystr::{self, py_len, trim};
use super::{Ctx, matchers, xmlout, xpaths};
use crate::html::tree::{Matcher, Name, NodeId, names as t};

const MIN_TEXT_LENGTH: usize = 25;

/// Tags whose names start with one of readability's DIV_TO_P_PREFIXES.
fn div_to_p_blocked(name: &str) -> bool {
    [
        "address",
        "article",
        "aside",
        "audio",
        "blockquote",
        "dl",
        "div",
        "img",
        "ol",
        "p",
        "table",
        "ul",
    ]
    .iter()
    .any(|p| name.starts_with(p))
}

fn tag_score(name: &str) -> f64 {
    match name {
        "div" | "article" => 5.0,
        "pre" | "td" | "blockquote" => 3.0,
        "address" | "ol" | "ul" | "dl" | "dd" | "dt" | "li" | "form" | "aside" => -3.0,
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "th" | "header" | "footer" | "nav" => -5.0,
        _ => 0.0,
    }
}

/// XPath `normalize-space()` is not empty: a character other than XML whitespace.
fn xpath_nonblank(s: &str) -> bool {
    s.chars().any(|c| !matches!(c, ' ' | '\t' | '\r' | '\n'))
}

/// The readability candidates, in dict insertion order.
#[derive(Default)]
pub(super) struct Candidates {
    pub(super) order: Vec<NodeId>,
    pub(super) score: HashMap<NodeId, f64>,
}

/// An `AssertionError` (drop_tree on an element without parent) or another exception:
/// try_readability's `except Exception`.
pub(super) struct Failed;

impl Ctx<'_> {
    pub(super) fn text_length(&self, e: NodeId) -> usize {
        py_len(&trim(&self.d.text_content(e)))
    }

    /// lxml.html drop_tree.
    pub(super) fn drop_tree(&mut self, e: NodeId) -> Result<(), Failed> {
        if self.d.parent(e).is_none() {
            return Err(Failed);
        }
        self.delete_element(e, true);
        if self.raised { Err(Failed) } else { Ok(()) }
    }

    #[allow(clippy::cast_precision_loss)]
    pub(super) fn get_link_density(&self, e: NodeId) -> f64 {
        let total = match self.text_length(e) {
            0 => 1,
            n => n,
        };
        let links: usize = self
            .findall_desc(e, t::A)
            .into_iter()
            .map(|a| self.text_length(a))
            .sum();
        links as f64 / total as f64
    }

    pub(super) fn class_weight(&self, e: NodeId) -> f64 {
        let mut w = 0.0;
        for a in [t::CLASS, t::ID] {
            let Some(v) = self.d.attr(e, a).filter(|s| !s.is_empty()) else {
                continue;
            };
            if matchers::NEGATIVE.search_icase(v) {
                w -= 25.0;
            }
            if matchers::POSITIVE.search_icase(v) {
                w += 25.0;
            }
        }
        w
    }

    fn score_node(&self, e: NodeId) -> f64 {
        self.class_weight(e) + tag_score(&pystr::lower(self.d.tag_name(e), self.v))
    }

    fn remove_unlikely_candidates(&mut self, doc: NodeId) -> Result<(), Failed> {
        let mut verdicts: HashMap<String, bool> = HashMap::new();
        for e in xpaths::select(&self.d, doc, |_, _| true) {
            let attrs: Vec<&str> = [self.d.attr(e, t::CLASS), self.d.attr(e, t::ID)]
                .into_iter()
                .flatten()
                .filter(|s| !s.is_empty())
                .collect();
            let attrs = attrs.join(" ");
            let g = self.d.tag(e);
            if py_len(&attrs) < 2 || g == t::BODY || g == t::HTML {
                continue;
            }
            let v = if let Some(&v) = verdicts.get(&attrs) {
                v
            } else {
                let low = pystr::lower(&attrs, self.v);
                let v = matchers::UNLIKELY.search(&low) && !matchers::OK_MAYBE.search(&low);
                verdicts.insert(attrs, v);
                v
            };
            if v {
                self.drop_tree(e)?;
            }
        }
        Ok(())
    }

    /// `fragment_fromstring("<tag/>")`: a new element that stays a child of the fragment's body.
    pub(super) fn fragment_element(&mut self, tag: Name) -> NodeId {
        let holder = self.d.element(t::BODY);
        self.d.sub_element(holder, tag)
    }

    fn transform_misused_divs_into_paragraphs(&mut self, doc: NodeId) -> Result<(), Failed> {
        for e in self.findall_desc(doc, t::DIV) {
            let blocked = self
                .d
                .iter_vec(e, &Matcher::AnyElement, false)
                .into_iter()
                .any(|d| div_to_p_blocked(self.d.tag_name(d)));
            if blocked {
                continue;
            }
            let loose_text = self.d.text(e).is_some_and(xpath_nonblank)
                || self
                    .d
                    .children(e)
                    .any(|c| self.d.tail(c).is_some_and(xpath_nonblank));
            if self.find_desc(e, t::A).is_none() || loose_text {
                self.d.set_tag(e, t::P);
            }
        }
        for e in self.findall_desc(doc, t::DIV) {
            if self.d.text(e).is_some_and(|s| !pystr::strip(s).is_empty()) {
                let p = self.fragment_element(t::P);
                let tx = self.d.text(e).map(str::to_owned);
                self.set_text(p, tx);
                self.set_text(e, None);
                self.insert(e, 0, p);
            }
            let kids = self.d.children_vec(e);
            for (pos, &child) in kids.iter().enumerate().rev() {
                if self
                    .d
                    .tail(child)
                    .is_some_and(|s| !pystr::strip(s).is_empty())
                {
                    let p = self.fragment_element(t::P);
                    let tl = self.d.tail(child).map(str::to_owned);
                    self.set_text(p, tl);
                    self.set_tail(child, None);
                    self.insert(e, isize::try_from(pos + 1).unwrap_or(isize::MAX), p);
                }
                if self.d.is_element(child) && self.d.tag(child) == t::BR {
                    self.drop_tree(child)?;
                }
            }
            if self.raised {
                return Err(Failed);
            }
        }
        Ok(())
    }

    pub(super) fn score_paragraphs(&mut self, doc: NodeId) -> Candidates {
        let mut c = Candidates::default();
        for e in self
            .d
            .iter_vec(doc, &Matcher::tags(&[t::P, t::PRE, t::TD]), true)
        {
            let Some(parent) = self.d.parent(e) else {
                continue;
            };
            let grand = self.d.parent(parent);
            let text = trim(&self.d.text_content(e));
            let len = py_len(&text);
            if len < MIN_TEXT_LENGTH {
                continue;
            }
            for n in [Some(parent), grand].into_iter().flatten() {
                if let std::collections::hash_map::Entry::Vacant(v) = c.score.entry(n) {
                    v.insert(self.score_node(n));
                    c.order.push(n);
                }
            }
            #[allow(clippy::cast_precision_loss)]
            let score = 1.0 + text.split(',').count() as f64 + (len as f64 / 100.0).min(3.0);
            if let Some(s) = c.score.get_mut(&parent) {
                *s += score;
            }
            if let Some(g) = grand {
                if let Some(s) = c.score.get_mut(&g) {
                    *s += score / 2.0;
                }
            }
        }
        for i in 0..c.order.len() {
            let n = c.order[i];
            let ld = self.get_link_density(n);
            if let Some(s) = c.score.get_mut(&n) {
                *s *= 1.0 - ld;
            }
        }
        c
    }

    pub(super) fn get_article(&mut self, c: &Candidates, best: NodeId, best_score: f64) -> NodeId {
        let threshold = (best_score * 0.2).max(10.0);
        let output = self.fragment_element(t::DIV);
        let siblings = match self.d.parent(best) {
            Some(p) => self.d.children_vec(p),
            None => vec![best],
        };
        for s in siblings {
            let mut append = false;
            if s == best || c.score.get(&s).is_some_and(|&x| x >= threshold) {
                append = true;
            } else if self.d.is_element(s) && self.d.tag(s) == t::P {
                let ld = self.get_link_density(s);
                let content = self.d.text(s).unwrap_or("");
                let n = py_len(content);
                if (n > 80 && ld < 0.25) || (n <= 80 && ld == 0.0 && matchers::dot_space(content)) {
                    append = true;
                }
            }
            if append {
                self.append(output, s);
            }
        }
        output
    }

    // the branches mirror readability's elif chain (one per removal reason)
    #[allow(clippy::if_same_then_else, clippy::cast_precision_loss)]
    fn readability_sanitize(&mut self, node: NodeId, c: &Candidates) -> Result<(), Failed> {
        for h in self.d.iter_vec(
            node,
            &Matcher::tags(&[t::H1, t::H2, t::H3, t::H4, t::H5, t::H6]),
            true,
        ) {
            if self.class_weight(h) < 0.0 || self.get_link_density(h) > 0.33 {
                self.drop_tree(h)?;
            }
        }
        for e in self
            .d
            .iter_vec(node, &Matcher::tags(&[t::FORM, t::TEXTAREA]), true)
        {
            self.drop_tree(e)?;
        }
        for e in self.d.iter_vec(node, &Matcher::tags(&[t::IFRAME]), true) {
            if self
                .d
                .attr(e, t::SRC)
                .is_some_and(|s| matchers::VIDEO.search_icase(s))
            {
                self.set_text(e, Some("VIDEO".into()));
            } else {
                self.drop_tree(e)?;
            }
        }
        let tags = [
            t::TABLE,
            t::UL,
            t::DIV,
            t::ASIDE,
            t::HEADER,
            t::FOOTER,
            t::SECTION,
        ];
        let mut list = self.d.iter_vec(node, &Matcher::tags(&tags), true);
        list.reverse();
        for e in list {
            let weight = self.class_weight(e);
            let score = c.score.get(&e).copied().unwrap_or(0.0);
            if weight + score < 0.0 {
                self.drop_tree(e)?;
                continue;
            }
            if self.d.text_content(e).matches(',').count() >= 10 {
                continue;
            }
            let mut counts: HashMap<Name, i64> = HashMap::new();
            for d in self.d.iter_vec(
                e,
                &Matcher::tags(&[t::P, t::IMG, t::LI, t::A, t::EMBED, t::INPUT]),
                false,
            ) {
                *counts.entry(self.d.tag(d)).or_default() += 1;
            }
            let cnt = |g: Name| counts.get(&g).copied().unwrap_or(0);
            let hidden = self
                .findall_desc(e, t::INPUT)
                .into_iter()
                .filter(|&i| self.d.attr(i, t::TYPE) == Some("hidden"))
                .count();
            let hidden = i64::try_from(hidden).unwrap_or(i64::MAX);
            let (p, img, li, input, embed) = (
                cnt(t::P),
                cnt(t::IMG),
                cnt(t::LI) - 100,
                cnt(t::INPUT) - hidden,
                cnt(t::EMBED),
            );
            let content_length = self.text_length(e);
            let link_density = self.get_link_density(e);
            let to_remove = if p != 0 && (img as f64) > 1.0 + p as f64 * 1.3 {
                true
            } else if li > p && self.d.tag(e) != t::OL && self.d.tag(e) != t::UL {
                true
            } else if (input as f64) > p as f64 / 3.0 {
                true
            } else if content_length < MIN_TEXT_LENGTH && img == 0 {
                true
            } else if content_length < MIN_TEXT_LENGTH && img > 2 {
                true
            } else if link_density > (if weight >= 25.0 { 0.5 } else { 0.2 }) {
                true
            } else if (embed == 1 && content_length < 75) || embed > 1 {
                true
            } else if content_length == 0 {
                // the nearest non-empty sibling on each side
                let mut sum = 0usize;
                let mut n = self.d.next(e);
                while let Some(s) = n {
                    let l = self.text_length(s);
                    if l > 0 {
                        sum += l;
                        break;
                    }
                    n = self.d.next(s);
                }
                let mut n = self.d.prev(e);
                while let Some(s) = n {
                    let l = self.text_length(s);
                    if l > 0 {
                        sum += l;
                        break;
                    }
                    n = self.d.prev(s);
                }
                sum <= 1000
            } else {
                false
            };
            if to_remove {
                self.drop_tree(e)?;
            }
        }
        Ok(())
    }

    /// readability_lxml.Document(doc).summary(), up to the serialization: the article element.
    fn readability_summary(&mut self, doc: NodeId) -> Result<NodeId, Failed> {
        for e in self.d.iter_vec(
            doc,
            &Matcher::tags(&[t::SCRIPT, t::STYLE, t::FENCEDFRAME]),
            true,
        ) {
            self.drop_tree(e)?;
        }
        self.remove_unlikely_candidates(doc)?;
        self.transform_misused_divs_into_paragraphs(doc)?;
        let c = self.score_paragraphs(doc);
        // max(..., key=score): the first candidate with the highest score
        let mut best: Option<(NodeId, f64)> = None;
        for &n in &c.order {
            let s = c.score.get(&n).copied().unwrap_or(0.0);
            if best.is_none_or(|(_, b)| s > b) {
                best = Some((n, s));
            }
        }
        let article = match best {
            Some((b, s)) => self.get_article(&c, b, s),
            None => self
                .d
                .children(doc)
                .find(|&k| self.d.is_element(k) && self.d.tag(k) == t::BODY)
                .unwrap_or(doc),
        };
        self.readability_sanitize(article, &c)?;
        if self.raised {
            Err(Failed)
        } else {
            Ok(article)
        }
    }

    /// external.try_readability: the re-parsed summary, or `HtmlElement()` (no text) when
    /// readability or the parser fails.
    pub fn try_readability(&mut self, raw: NodeId) -> NodeId {
        let outer = self.raised;
        let r = self.readability_summary(raw);
        self.raised = outer;
        if let Ok(article) = r {
            let summary = xmlout::tostring_xml(&self.d, article);
            if let Ok(doc) = crate::html::lxml::fromstring_bytes(summary.as_bytes()) {
                return self.d.import(&doc, doc.root);
            }
        }
        let name = self.d.intern("HtmlElement");
        self.d.element(name)
    }
}
