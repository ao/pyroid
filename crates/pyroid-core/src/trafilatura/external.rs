// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! external.py: compare_extraction (readability and jusText as fallbacks), justext_rescue and
//! sanitize_tree.

use std::collections::HashSet;

use super::htmlproc::tagset_of;
use super::pystr::{py_len, trim};
use super::{Ctx, Focus, xpaths};
use crate::html::tree::{Matcher, Name, NodeId, TagSet, names as t};

/// xml.TEI_VALID_TAGS.
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

impl Ctx<'_> {
    /// external._prefer_readability.
    #[allow(clippy::cast_precision_loss)]
    fn prefer_readability(
        &self,
        body: NodeId,
        algo: NodeId,
        algo_text: &str,
        len_text: usize,
        len_algo: usize,
    ) -> bool {
        if len_algo == 0 || len_algo == len_text {
            return false;
        }
        if len_text > 2 * len_algo {
            return false;
        }
        let recall = self.focus == Focus::Recall;
        len_text == 0
            || (len_algo > 2 * len_text && !algo_text.starts_with('{'))
            || (len_algo > self.o.min_extracted_size * 2
                && (!xpaths::p_text_rel(&self.d, body)
                    || self.findall_desc(body, t::TABLE).len()
                        > self.findall_desc(body, t::P).len()))
            || (recall && (len_algo as f64) > 1.5 * len_text as f64 && !algo_text.starts_with('{'))
            || (recall
                && self.find_desc(body, t::HEAD).is_none()
                && !self
                    .d
                    .iter_vec(algo, &Matcher::tags(&[t::H2, t::H3, t::H4]), true)
                    .is_empty()
                && len_algo > len_text)
    }

    /// external.compare_extraction: `cleaned_backup` (cleaned, not converted) feeds jusText,
    /// `raw` (a copy of the uncleaned tree) feeds readability.
    pub fn compare_extraction(
        &mut self,
        cleaned_backup: NodeId,
        raw: NodeId,
        body: NodeId,
        text: String,
    ) -> (NodeId, String) {
        let mut len_text = py_len(&text);
        if self.focus == Focus::Recall && len_text > self.o.min_extracted_size * 10 {
            return (body, text);
        }
        let (mut body, mut text) = (body, text);
        if self.focus == Focus::Precision {
            self.prune_unwanted_nodes(raw, self.p.overall_discard, false);
        }
        let algo = self.try_readability(raw);
        // tostring(method="text"): the text content and the root's tail
        let algo_text = trim(&(self.d.text_content(algo) + self.d.tail(algo).unwrap_or("")));
        let len_algo = py_len(&algo_text);
        let mut use_readability =
            self.prefer_readability(body, algo, &algo_text, len_text, len_algo);
        if use_readability {
            body = algo;
            text = algo_text;
            len_text = len_algo;
        }
        let unclean = xpaths::sanitized_any(&self.d, body);
        if unclean || len_text < self.o.min_extracted_size {
            let (body2, text2) = self.justext_rescue(cleaned_backup);
            let len2 = py_len(&text2);
            // unclean: shorter is fine (boilerplate dropped); with images off the graphic guard
            // always passes. Merely short: it must add text.
            let accept = if unclean {
                len_text <= self.p.justext_override_ratio * len2
            } else {
                len_text < len2
            };
            if !text2.is_empty() && accept {
                body = body2;
                text = text2;
                use_readability = false;
            }
        }
        if use_readability {
            return self.sanitize_tree(body);
        }
        (body, text)
    }

    /// external.sanitize_tree (links and images off).
    fn sanitize_tree(&mut self, tree: NodeId) -> (NodeId, String) {
        let cleaned = self.tree_cleaning(tree);
        self.d.strip_tags(cleaned, &Matcher::tags(&[t::SPAN, t::A]));
        let cleaned = self.convert_tags(cleaned);
        for g in self
            .d
            .iter_vec(cleaned, &Matcher::tags(&[t::GRAPHIC]), true)
        {
            match self.handle_image(g) {
                None => self.delete_element(g, true),
                Some(img) => {
                    let attrs = self.d.n(img).attrs.clone();
                    self.d.nm(g).attrs = attrs;
                }
            }
        }
        let mut seen: HashSet<Option<NodeId>> = HashSet::new();
        for tr in self.d.iter_vec(cleaned, &Matcher::tags(&[t::TR]), true) {
            let parent = self.d.parent(tr);
            if !seen.contains(&parent)
                && self
                    .d
                    .children(tr)
                    .any(|c| self.d.is_element(c) && self.d.tag(c) == t::TH)
            {
                seen.insert(parent);
                for c in self.d.children_vec(tr) {
                    if self.d.is_element(c) && self.d.tag(c) == t::TH {
                        self.d.set(c, "role", "head");
                    }
                }
            }
        }
        for e in self
            .d
            .iter_vec(cleaned, &Matcher::tags(&[t::TD, t::TH, t::TR]), true)
        {
            let g = if self.d.tag(e) == t::TR {
                t::ROW
            } else {
                t::CELL
            };
            self.d.set_tag(e, g);
        }
        let valid = TagSet::of(TEI_VALID_TAGS);
        let strip: Vec<Name> = self
            .d
            .iter_vec(cleaned, &Matcher::AnyElement, true)
            .into_iter()
            .map(|e| self.d.tag(e))
            .filter(|&g| !valid.contains(g))
            .collect();
        self.d.strip_tags(cleaned, &Matcher::Tags(tagset_of(strip)));
        let text = trim(&self.d.itertext_join(cleaned, " "));
        (cleaned, text)
    }

    /// baseline.basic_cleaning: BASIC_CLEAN_XPATH, deleting with the tails kept.
    pub fn basic_cleaning(&mut self, tree: NodeId) {
        for e in xpaths::basic_clean(&self.d, tree) {
            self.delete_element(e, true);
        }
    }

    /// external.justext_rescue: basic_cleaning, jusText's paragraphs, the good ones as <p>.
    pub fn justext_rescue(&mut self, tree: NodeId) -> (NodeId, String) {
        self.basic_cleaning(tree);
        let body = self.d.element(t::BODY);
        // try_justext: make_paragraphs and the classification run in `try`; they assign nothing
        let good =
            super::justext::good_paragraphs(&self.d, tree, self.v, &super::justext::JT_2_3_1);
        for text in good {
            let e = self.d.sub_element(body, t::P);
            self.set_text(e, Some(text));
        }
        let text = trim(&self.d.itertext_join(body, " "));
        (body, text)
    }
}
