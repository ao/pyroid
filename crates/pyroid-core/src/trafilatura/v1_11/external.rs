// Derived from trafilatura 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
//! trafilatura 1.11.0's external.py: the comparison with readability and the jusText rescue.

use super::super::justext::{self, JtParams};
use super::super::pystr::{py_len, trim};
use super::super::xpaths::{p_text_rel, select_first};
use super::{V11, xpaths};
use crate::html::tree::{Matcher, Name, NodeId, TagSet, names as t};

/// xml.TEI_VALID_TAGS (1.11.0).
const TEI_VALID_TAGS: &[Name] = &[
    t::AB,
    t::BODY,
    t::CELL,
    t::CODE,
    t::DEL,
    t::DIV,
    t::GRAPHIC,
    t::HEAD,
    t::HI,
    t::ITEM,
    t::LB,
    t::LIST,
    t::P,
    t::QUOTE,
    t::REF,
    t::ROW,
    t::TABLE,
];

/// external.custom_justext: classify_paragraphs(..., 50, 200, 0.1, 0.2, 0.2, no_headings=True)
/// and revise_paragraph_classification(..., 200), on the jt-stopwords pickle (the stoplists
/// of jusText merged, as for 2.3.1).
const JT: JtParams = JtParams {
    length_low: 50,
    length_high: 200,
    stopwords_low: 0.1,
    stopwords_high: 0.2,
    max_link_density: 0.2,
};

impl V11<'_, '_> {
    /// external.compare_extraction: (body, text, length).
    // the branches mirror the elif chain of the decision
    #[allow(clippy::if_same_then_else)]
    pub fn compare_extraction(
        &mut self,
        tree: NodeId,
        backup: NodeId,
        body: NodeId,
        text: String,
        len_text: usize,
    ) -> (NodeId, String, usize) {
        let (mut body, mut text, mut len_text) = (body, text, len_text);
        let min = self.o.min_extracted_size;
        if self.focus == super::Focus::Recall && len_text > min * 10 {
            return (body, text, len_text);
        }
        let mut backup = self.prune_unwanted_nodes(backup, xpaths::PAYWALL_DISCARD_XPATH, false);
        if self.focus == super::Focus::Precision {
            backup = self.prune_unwanted_nodes(backup, self.p.overall_discard, false);
        }
        let Some(algo) = self.try_readability(backup) else {
            // `tostring(None, ...)`: TypeError
            self.raise();
            return (body, text, len_text);
        };
        // tostring(method="text") with the tail
        let mut raw = self.d.text_content(algo);
        raw.push_str(self.d.tail(algo).unwrap_or(""));
        let algo_text = trim(&raw);
        let len_algo = py_len(&algo_text);
        let use_readability = if len_algo == 0 || len_algo == len_text {
            false
        } else if len_text == 0 && len_algo > 0 {
            true
        } else if len_text > 2 * len_algo {
            false
        } else if len_algo > 2 * len_text && !algo_text.starts_with('{') {
            true
        } else if !p_text_rel(&self.d, body) && len_algo > min * 2 {
            true
        } else if self.findall_desc(body, t::TABLE).len() > self.findall_desc(body, t::P).len()
            && len_algo > min * 2
        {
            true
        } else {
            self.focus == super::Focus::Recall
                && self.find_desc(body, t::HEAD).is_none()
                && select_first(&self.d, algo, |d, e| {
                    [t::H2, t::H3, t::H4].contains(&d.tag(e))
                })
                .is_some()
                && len_algo > len_text
        };
        if use_readability {
            (body, text, len_text) = (algo, algo_text, len_algo);
        }
        let mut jt_result = false;
        if xpaths::sanitized_any(&self.d, body) || len_text < min {
            let (b2, t2, l2, ok) = self.justext_rescue(tree, body);
            jt_result = ok;
            if self.raised {
                return (body, text, len_text);
            }
            if jt_result && len_text <= 4 * l2 {
                (body, text, len_text) = (b2, t2, l2);
            }
        }
        if use_readability && !jt_result {
            (body, text, len_text) = self.sanitize_tree(body);
        }
        (body, text, len_text)
    }

    /// external.justext_rescue(tree, options, postbody, 0, ''): (body, text, length, used).
    fn justext_rescue(&mut self, tree: NodeId, postbody: NodeId) -> (NodeId, String, usize, bool) {
        let tree = self.prune_unwanted_nodes(tree, xpaths::PAYWALL_DISCARD_XPATH, false);
        let tree = self.prune_unwanted_nodes(tree, xpaths::REMOVE_COMMENTS_XPATH, false);
        // try_justext
        let result = self.d.element(t::BODY);
        for s in justext::good_paragraphs(&self.d, tree, self.v, &JT) {
            let p = self.d.element(t::P);
            self.set_text(p, Some(s));
            self.append(result, p);
        }
        let temp = trim(&self.d.itertext_join(result, " "));
        let n = py_len(&temp);
        if n > 0 {
            (result, temp, n, true)
        } else {
            (postbody, String::new(), 0, false)
        }
    }

    /// external.sanitize_tree: clean and convert the readability output, keep TEI tags only.
    fn sanitize_tree(&mut self, tree: NodeId) -> (NodeId, String, usize) {
        let cleaned = self.tree_cleaning(tree);
        // `tree.findall(SANITIZED_XPATH)`: ElementPath reads the union as one long tag name and
        // finds nothing
        self.d.strip_tags(cleaned, &Matcher::tags(&[t::A]));
        self.d.strip_tags(cleaned, &Matcher::tags(&[t::SPAN]));
        let cleaned = self.convert_tags(cleaned);
        for e in self
            .d
            .iter_vec(cleaned, &Matcher::tags(&[t::TD, t::TH, t::TR]), true)
        {
            let g = self.d.tag(e);
            if g == t::TR {
                self.d.set_tag(e, t::ROW);
            } else {
                if g == t::TH {
                    self.set_attr(e, "role", "head");
                }
                self.d.set_tag(e, t::CELL);
            }
        }
        let valid = TagSet::of(TEI_VALID_TAGS);
        let mut strip = TagSet::default();
        for e in self.d.iter_vec(cleaned, &Matcher::AnyElement, true) {
            let g = self.d.tag(e);
            if !valid.contains(g) {
                strip.insert(g);
            }
        }
        self.d.strip_tags(cleaned, &Matcher::Tags(strip));
        let text = trim(&self.d.itertext_join(cleaned, " "));
        let n = py_len(&text);
        (cleaned, text, n)
    }
}
