// Derived from readability-lxml as vendored in trafilatura 1.11.0 (Apache-2.0); modified.
//! trafilatura 1.11.0's readability_lxml.py: `Document(doc, min_text_length=25,
//! retry_length=250).summary()` with its ruthless retry, and external.try_readability. The
//! scoring helpers are the 2.3.1 ones (unchanged between the releases).

use std::collections::HashMap;
use std::sync::LazyLock;

use super::super::matchers::{self, LitSet};
use super::super::pystr::{icase_fold, py_len};
use super::super::readability::{Candidates, Failed};
use super::super::xmlout;
use super::V11;
use crate::html::tree::{Iter, Matcher, Name, NodeId, names as t};

const MIN_TEXT_LENGTH: usize = 25;
const RETRY_LENGTH: usize = 250;

/// `okMaybeItsACandidateRe` (1.11.0: no "content").
static OK_MAYBE: LazyLock<LitSet> =
    LazyLock::new(|| LitSet::new(&["and", "article", "body", "column", "main", "shadow"]));

/// `divToPElementsRe` (`<(?:a|blockquote|dl|div|img|ol|p|pre|table|ul)`, re.I) finds a start
/// tag in the serialized children: an element below whose name starts with one of these.
fn div_to_p_blocked(name: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "a",
        "blockquote",
        "dl",
        "div",
        "img",
        "ol",
        "p",
        "table",
        "ul",
    ];
    PREFIXES.iter().any(|p| {
        let mut cs = name.chars();
        p.chars()
            .all(|pc| cs.next().is_some_and(|c| icase_fold(c) == pc))
    })
}

impl V11<'_, '_> {
    fn remove_unlikely_candidates(&mut self, doc: NodeId) -> Result<(), Failed> {
        for e in super::super::xpaths::select(&self.d, doc, |_, _| true) {
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
            if matchers::UNLIKELY.search_icase(&attrs) && !OK_MAYBE.search_icase(&attrs) {
                self.drop_tree(e)?;
            }
        }
        Ok(())
    }

    fn transform_misused_divs_into_paragraphs(&mut self, doc: NodeId) -> Result<(), Failed> {
        for e in self.findall_desc(doc, t::DIV) {
            let blocked = self
                .d
                .iter_vec(e, &Matcher::AnyElement, false)
                .into_iter()
                .any(|x| div_to_p_blocked(self.d.tag_name(x)));
            if !blocked {
                self.d.set_tag(e, t::P);
            }
        }
        for e in self.findall_desc(doc, t::DIV) {
            if self
                .d
                .text(e)
                .is_some_and(|s| !super::super::pystr::strip(s).is_empty())
            {
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
                    .is_some_and(|s| !super::super::pystr::strip(s).is_empty())
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

    /// Document.sanitize: the serialized article (`self.doc` becomes `node`).
    // the branches mirror readability's elif chain (one per removal reason)
    #[allow(clippy::if_same_then_else)]
    fn sanitize(&mut self, node: NodeId, c: &Candidates) -> Result<String, Failed> {
        // live iterations, as lxml's: a dropped element's descendants are still visited, and
        // the walk ends when it leaves them
        let mut it = Iter::new(
            &self.d,
            node,
            Matcher::tags(&[t::H1, t::H2, t::H3, t::H4, t::H5, t::H6]),
            true,
        );
        while let Some(h) = it.next(&self.d) {
            if self.class_weight(h) < 0.0 || self.get_link_density(h) > 0.33 {
                self.drop_tree(h)?;
            }
        }
        let mut it = Iter::new(&self.d, node, Matcher::tags(&[t::FORM, t::TEXTAREA]), true);
        while let Some(e) = it.next(&self.d) {
            self.drop_tree(e)?;
        }
        let mut it = Iter::new(&self.d, node, Matcher::tags(&[t::IFRAME]), true);
        while let Some(e) = it.next(&self.d) {
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
        // `node.xpath("//table|//ul|//div|//aside|//header|//footer|//section")`: the whole
        // document, reversed
        let tags = [
            t::TABLE,
            t::UL,
            t::DIV,
            t::ASIDE,
            t::HEADER,
            t::FOOTER,
            t::SECTION,
        ];
        let mut list = self.doc_iter(node, &Matcher::tags(&tags));
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
            #[allow(clippy::cast_precision_loss)]
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
                // the nearest non-empty sibling on each side (the `allowed` set it fills only
                // holds elements already visited)
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
        if self.raised {
            return Err(Failed);
        }
        Ok(xmlout::tostring_xml(&self.d, node))
    }

    /// Document.summary(): ruthless first, once more without removing unlikely candidates if
    /// nothing scores or the result is short.
    fn summary(&mut self, doc: NodeId) -> Result<String, Failed> {
        let mut it = Iter::new(&self.d, doc, Matcher::tags(&[t::SCRIPT, t::STYLE]), true);
        while let Some(e) = it.next(&self.d) {
            self.drop_tree(e)?;
        }
        let mut doc = doc;
        let mut ruthless = true;
        loop {
            if ruthless {
                self.remove_unlikely_candidates(doc)?;
            }
            self.transform_misused_divs_into_paragraphs(doc)?;
            let c = self.score_paragraphs(doc);
            // sorted(..., reverse=True)[0]: the first candidate with the highest score
            let mut best: Option<(NodeId, f64)> = None;
            for &n in &c.order {
                let s = c.score.get(&n).copied().unwrap_or(0.0);
                if best.is_none_or(|(_, b)| s > b) {
                    best = Some((n, s));
                }
            }
            let article = match best {
                Some((b, s)) => self.get_article(&c, b, s),
                None if ruthless => {
                    ruthless = false;
                    continue;
                }
                None => self
                    .d
                    .children(doc)
                    .find(|&k| self.d.is_element(k) && self.d.tag(k) == t::BODY)
                    .unwrap_or(doc),
            };
            if self.raised {
                return Err(Failed);
            }
            let cleaned = self.sanitize(article, &c)?;
            doc = article;
            if ruthless && py_len(&cleaned) < RETRY_LENGTH {
                ruthless = false;
                continue;
            }
            return Ok(cleaned);
        }
    }

    /// external.try_readability: the re-parsed summary, `Element('div')` where readability
    /// raises, None where the parser fails (`fromstring_bytes` returns None).
    pub fn try_readability(&mut self, raw: NodeId) -> Option<NodeId> {
        let r = self.try_catch(|cx| V11::new(cx).summary(raw));
        match r {
            Ok(Ok(summary)) => {
                let doc = crate::html::lxml::fromstring_bytes(summary.as_bytes()).ok()?;
                Some(self.d.import(&doc, doc.root))
            }
            _ => Some(self.d.element(t::DIV)),
        }
    }
}
