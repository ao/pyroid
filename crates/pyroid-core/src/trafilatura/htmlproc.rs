// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! htmlprocessing.py, xml.delete_element and deduplication.duplicate_test, for the options this
//! port supports (no links, images or formatting).

use super::pystr::{self, py_len, trim, trim_or_none};
use super::xpaths::{self, XList};
use super::{Ctx, Focus, matchers};
use crate::html::tree::{Iter, Matcher, Name, NodeId, TagSet, names as t};

/// REND_TAG_MAPPING keys, in dict order.
const REND_TAGS: &[Name] = &[
    t::EM,
    t::I,
    t::B,
    t::STRONG,
    t::U,
    t::KBD,
    t::SAMP,
    t::TT,
    t::VAR,
    t::SUB,
    t::SUP,
];

/// CONVERSIONS keys.
const CONVERSION_TAGS: &[Name] = &[
    t::DL,
    t::OL,
    t::UL,
    t::H1,
    t::H2,
    t::H3,
    t::H4,
    t::H5,
    t::H6,
    t::BR,
    t::HR,
    t::BLOCKQUOTE,
    t::PRE,
    t::Q,
    t::DEL,
    t::S,
    t::STRIKE,
    t::DETAILS,
];

const CODE_INDICATORS: &[&str] = &["{", "(\"", "('", "\n    "];

/// `TagSet` of tags gathered from the tree (`{str(c.tag) for c in ...}`).
pub fn tagset_of(tags: impl IntoIterator<Item = Name>) -> TagSet {
    let mut s = TagSet::default();
    for g in tags {
        s.insert(g);
    }
    s
}

impl Ctx<'_> {
    /// xml.delete_element: remove the element, joining its tail to the previous sibling or the
    /// parent's text.
    pub fn delete_element(&mut self, e: NodeId, keep_tail: bool) {
        let Some(parent) = self.d.parent(e) else {
            return;
        };
        if keep_tail {
            if let Some(tail) = self.d.tail(e).filter(|s| !s.is_empty()).map(str::to_owned) {
                match self.d.prev(e) {
                    None => {
                        let v = format!("{}{tail}", self.d.text(parent).unwrap_or(""));
                        self.set_text(parent, Some(v));
                    }
                    Some(p) => {
                        let v = format!("{}{tail}", self.d.tail(p).unwrap_or(""));
                        self.set_tail(p, Some(v));
                    }
                }
            }
        }
        self.d.unlink(e);
    }

    /// htmlprocessing.prune_unwanted_nodes.
    pub fn prune_unwanted_nodes(&mut self, tree: NodeId, nodelist: XList, with_backup: bool) {
        if !with_backup {
            for expr in nodelist {
                for sub in expr(&self.d, tree) {
                    self.delete_element(sub, true);
                }
            }
            return;
        }
        let mut matches: Vec<NodeId> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for expr in nodelist {
            for e in expr(&self.d, tree) {
                if seen.insert(e) {
                    matches.push(e);
                }
            }
        }
        let outer: Vec<NodeId> = matches
            .iter()
            .copied()
            .filter(|&e| !self.d.ancestors(e).any(|a| seen.contains(&a)))
            .collect();
        let old_len = self.text_content_len(tree);
        let removed: usize = outer.iter().map(|&e| self.text_content_len(e)).sum();
        // old_len - removed > old_len / 7, compared exactly
        if 7 * (old_len as i128 - removed as i128) > old_len as i128 {
            for e in outer {
                self.delete_element(e, true);
            }
        }
    }

    /// htmlprocessing.recover_math.
    fn recover_math(&mut self, tree: NodeId) {
        for m in self.d.iter_vec(tree, &Matcher::tags(&[t::MATH]), true) {
            let ann = xpaths::tex_annotations(&self.d, m);
            let latex = match ann.first() {
                Some(&a) => trim(self.d.text(a).unwrap_or("")),
                None => trim(self.d.get(m, "alttext").unwrap_or("")),
            };
            if latex.is_empty() {
                continue;
            }
            let (o, c) = if self.d.get(m, "display") == Some("block") {
                ("\\[", "\\]")
            } else {
                ("\\(", "\\)")
            };
            let tail = format!("{o}{latex}{c}{}", self.d.tail(m).unwrap_or(""));
            self.set_tail(m, Some(tail));
        }
    }

    /// htmlprocessing._handle_forms.
    fn handle_forms(&mut self, tree: NodeId) {
        let forms = self.d.iter_vec(tree, &Matcher::tags(&[t::FORM]), true);
        if forms.is_empty() {
            return;
        }
        let total = self.text_content_len(tree);
        for form in forms {
            if total > 0 && 2 * self.text_content_len(form) > total {
                self.d.set_tag(form, t::DIV);
            } else {
                self.delete_element(form, true);
            }
        }
    }

    /// htmlprocessing.tree_cleaning; returns the tree to use (a copy in recall mode when the
    /// deletions would remove every paragraph).
    pub fn tree_cleaning(&mut self, tree: NodeId) -> NodeId {
        let mut tree = tree;
        self.recover_math(tree);
        let mut cleaning: Vec<Name> = self
            .p
            .manually_cleaned
            .iter()
            .copied()
            .filter(|&g| g != t::FORM)
            .collect();
        let clean_forms = self.p.manually_cleaned.contains(&t::FORM);
        if self.o.tables {
            // figures holding a table and ARIA layout tables become divs
            let mut it = Iter::new(&self.d, tree, Matcher::tags(&[t::FIGURE, t::TABLE]), true);
            while let Some(e) = it.next(&self.d) {
                let to_div = if self.d.tag(e) == t::FIGURE {
                    self.find_desc(e, t::TABLE).is_some()
                } else {
                    matches!(self.d.attr(e, t::ROLE), Some("presentation" | "none"))
                };
                if to_div {
                    self.d.set_tag(e, t::DIV);
                }
            }
        } else {
            cleaning.extend([t::TABLE, t::TD, t::TH, t::TR]);
        }
        self.d
            .strip_tags(tree, &Matcher::tags(self.p.manually_stripped));
        let cleaning_set = TagSet::of(&cleaning);
        let mut tcopy = None;
        if self.focus == Focus::Recall && self.find_desc(tree, t::P).is_some() {
            // copy only if no paragraph can survive the deletions
            let ps = self.d.iter_vec(tree, &Matcher::tags(&[t::P]), true);
            let survives = ps.into_iter().any(|p| {
                self.d
                    .first_ancestor_in(p, &cleaning_set)
                    .is_none_or(|a| a == tree)
            });
            if !survives {
                tcopy = Some(self.d.deep_copy(tree));
            }
        }
        for &g in &cleaning {
            let mut it = Iter::new(&self.d, tree, Matcher::tags(&[g]), true);
            while let Some(e) = it.next(&self.d) {
                self.delete_element(e, true);
            }
        }
        if let Some(c) = tcopy {
            if self.find_desc(tree, t::P).is_none() {
                tree = c;
            }
        }
        if clean_forms {
            self.handle_forms(tree);
        }
        self.prune_html(tree);
        tree
    }

    /// htmlprocessing.prune_html.
    fn prune_html(&mut self, tree: NodeId) {
        let tails = self.focus != Focus::Precision;
        let cands = self
            .d
            .iter_vec(tree, &Matcher::tags(self.p.cut_empty_elems), false);
        let dels: Vec<NodeId> = cands
            .into_iter()
            .filter(|&e| self.d.text(e).is_none() && self.d.len(e) == 0)
            .collect();
        for e in dels {
            self.delete_element(e, tails);
        }
    }

    /// htmlprocessing.collect_link_info: (total length, count, short ones) of non-empty texts.
    fn collect_link_info(&self, links: &[NodeId]) -> (usize, usize, usize) {
        let mut total = 0;
        let mut n = 0;
        let mut short = 0;
        for &l in links {
            let s = trim(&self.d.text_content(l));
            if s.is_empty() {
                continue;
            }
            let len = py_len(&s);
            total += len;
            n += 1;
            if len < 10 {
                short += 1;
            }
        }
        (total, n, short)
    }

    /// htmlprocessing.is_paragraph_listing.
    fn is_paragraph_listing(&self, links: &[NodeId]) -> bool {
        links.iter().all(|&l| {
            matches!(self.d.parent(l), Some(p) if self.d.tag(p) == t::P && self.findall_desc(p, t::REF).len() <= 1)
        })
    }

    /// htmlprocessing.link_density_test: (boilerplate, short with links).
    #[allow(clippy::cast_precision_loss)]
    fn link_density_test(&self, e: NodeId, favor_precision: bool) -> (bool, bool) {
        let links = self.findall_desc(e, t::REF);
        if links.is_empty() || self.find_desc(e, t::GRAPHIC).is_some() {
            return (false, false);
        }
        let text = trim(&self.d.text_content(e));
        let elemlen = py_len(&text);
        if links.len() == 1 {
            let len_threshold = if favor_precision { 10 } else { 100 };
            let ll = py_len(&trim(&self.d.text_content(links[0])));
            if ll > len_threshold && (ll as f64) > (elemlen as f64) * 0.9 {
                return (true, false);
            }
        }
        let limitlen = if self.d.tag(e) == t::P {
            if self.d.next(e).is_none() { 60 } else { 30 }
        } else if self.d.next(e).is_none() {
            300
        } else {
            100
        };
        if elemlen < limitlen {
            let (linklen, elemnum, shortelems) = self.collect_link_info(&links);
            if elemnum == 0 {
                return (true, false);
            }
            let r = (linklen as f64) > (elemlen as f64) * 0.8
                || (elemnum > 1 && (shortelems as f64) / (elemnum as f64) > 0.8);
            return (r, true);
        }
        if links.len() > 4 {
            let (linklen, elemnum, _) = self.collect_link_info(&links);
            if (linklen as f64) > (elemlen as f64) * self.p.link_farm_ratio
                && linklen < 100 * elemnum
                && !self.is_paragraph_listing(&links)
            {
                return (true, false);
            }
        }
        (false, false)
    }

    /// htmlprocessing.link_density_test_tables.
    #[allow(clippy::cast_precision_loss)]
    pub fn link_density_test_tables(&self, e: NodeId) -> bool {
        let links = self.findall_desc(e, t::REF);
        if links.is_empty() {
            return false;
        }
        let elemlen = py_len(&trim(&self.d.text_content(e)));
        if elemlen < 200 {
            return false;
        }
        let (linklen, _, _) = self.collect_link_info(&links);
        if elemlen < 1000 {
            (linklen as f64) > 0.8 * elemlen as f64
        } else {
            (linklen as f64) > 0.5 * elemlen as f64
        }
    }

    /// htmlprocessing.delete_by_link_density.
    pub fn delete_by_link_density(
        &mut self,
        subtree: NodeId,
        tag: Name,
        backtracking: bool,
        favor_precision: bool,
    ) {
        let len_threshold = if favor_precision { 200 } else { 100 };
        let depth_threshold = if favor_precision { 1 } else { 3 };
        let mut deletions: Vec<NodeId> = Vec::new();
        // the loop body does not change the tree: a snapshot walk is the live one
        for e in self.d.iter_vec(subtree, &Matcher::tags(&[tag]), true) {
            let (result, short_with_links) = self.link_density_test(e, favor_precision);
            if result
                || (backtracking
                    && short_with_links
                    && self.d.len(e) >= depth_threshold
                    && py_len(&trim(&self.d.text_content(e))) < len_threshold)
            {
                if tag == t::P {
                    if let Some(p) = self.d.parent(e) {
                        let g = self.d.tag(p);
                        if g == t::ITEM || g == t::TD || g == t::TH {
                            continue;
                        }
                    }
                }
                deletions.push(e);
            }
        }
        let mut seen = std::collections::HashSet::new();
        for e in deletions {
            if seen.insert(e) {
                self.delete_element(e, true);
            }
        }
    }

    /// xml.separates_inline.
    pub fn separates_inline(&self, e: NodeId) -> bool {
        self.d.tag(e) == t::LB
            && self
                .d
                .next(e)
                .is_some_and(|n| super::main_ext::INLINE_CARRIED.contains(&self.d.tag(n)))
    }

    /// htmlprocessing.handle_textnode.
    pub fn handle_textnode(
        &mut self,
        e: NodeId,
        comments_fix: bool,
        preserve_spaces: bool,
    ) -> Option<NodeId> {
        if self.d.tag(e) == t::GRAPHIC && self.image_src(e).is_some() {
            return Some(e);
        }
        let truthy = |s: Option<&str>| s.is_some_and(|s| !s.is_empty());
        if self.d.tag(e) == t::DONE
            || (self.d.len(e) == 0
                && !truthy(self.d.text(e))
                && !truthy(self.d.tail(e))
                && (comments_fix || !self.separates_inline(e)))
        {
            return None;
        }
        if !comments_fix && self.d.tag(e) == t::LB {
            if !preserve_spaces {
                let v = trim_or_none(self.d.tail(e));
                self.set_tail(e, v);
            }
            return Some(e);
        }
        if !truthy(self.d.text(e)) && self.d.len(e) == 0 {
            let tail = self.d.tail(e).map(str::to_owned);
            self.set_text(e, tail);
            self.set_tail(e, Some(String::new()));
            if comments_fix && self.d.tag(e) == t::LB {
                self.d.set_tag(e, t::P);
            }
        }
        if !preserve_spaces {
            let v = trim_or_none(self.d.text(e));
            self.set_text(e, v);
            if truthy(self.d.tail(e)) {
                let v = trim_or_none(self.d.tail(e));
                self.set_tail(e, v);
            }
        }
        if (!truthy(self.d.text(e)) && self.textfilter(e))
            || (self.dedup() && self.duplicate_test(e))
        {
            return None;
        }
        Some(e)
    }

    /// htmlprocessing.process_node.
    pub fn process_node(&mut self, e: NodeId) -> Option<NodeId> {
        let truthy = |s: Option<&str>| s.is_some_and(|s| !s.is_empty());
        if self.d.tag(e) == t::DONE
            || (self.d.len(e) == 0 && !truthy(self.d.text(e)) && !truthy(self.d.tail(e)))
        {
            return None;
        }
        let (tx, tl) = (trim_or_none(self.d.text(e)), trim_or_none(self.d.tail(e)));
        self.set_text(e, tx);
        self.set_tail(e, tl);
        if self.d.tag(e) != t::LB
            && self.d.text(e).is_none()
            && self.d.tail(e).is_some()
            && self.d.len(e) == 0
        {
            let tail = self.d.tail(e).map(str::to_owned);
            self.set_text(e, tail);
            self.set_tail(e, None);
        }
        if (self.d.text(e).is_some() || self.d.tail(e).is_some())
            && (self.textfilter(e) || (self.dedup() && self.duplicate_test(e)))
        {
            return None;
        }
        Some(e)
    }

    /// utils.textfilter.
    pub fn textfilter(&self, e: NodeId) -> bool {
        let testtext = match self.d.text(e) {
            None => self.d.tail(e),
            Some(s) => Some(s),
        };
        match testtext {
            None | Some("") => true,
            Some(s) => {
                pystr::str_isspace(s)
                    || pystr::splitlines(s)
                        .iter()
                        .any(|l| matchers::re_filter_match(l, self.v))
            }
        }
    }

    /// deduplication.duplicate_test.
    pub fn duplicate_test(&mut self, e: NodeId) -> bool {
        let teststring = trim(&self.d.itertext_join(e, " "));
        let previous = self.lru_increment(&teststring);
        py_len(&teststring) > self.o.min_duplcheck_size && previous > self.o.max_repetitions
    }

    /// htmlprocessing.convert_lists.
    fn convert_lists(&mut self, e: NodeId) {
        let name = self.d.tag_name(e).to_owned();
        self.set_attr(e, "rend", &name);
        self.d.set_tag(e, t::LIST);
        let mut i = 1;
        let mut it = Iter::new(&self.d, e, Matcher::tags(&[t::DD, t::DT, t::LI]), true);
        while let Some(s) = it.next(&self.d) {
            let g = self.d.tag(s);
            if g == t::DD || g == t::DT {
                let v = format!("{}-{i}", self.d.tag_name(s));
                self.set_attr(s, "rend", &v);
                if g == t::DD {
                    i += 1;
                }
            }
            self.d.set_tag(s, t::ITEM);
        }
    }

    /// htmlprocessing.convert_quotes.
    fn convert_quotes(&mut self, e: NodeId) {
        let is_pre = self.d.tag(e) == t::PRE;
        let hljs = if is_pre {
            xpaths::hljs_spans(&self.d, e)
        } else {
            Vec::new()
        };
        for &s in &hljs {
            self.d.attrib_clear(s);
        }
        let is_code = is_pre
            && ((self.d.len(e) == 1
                && self
                    .d
                    .first_child(e)
                    .is_some_and(|c| self.d.tag(c) == t::SPAN))
                || !hljs.is_empty()
                || CODE_INDICATORS
                    .iter()
                    .any(|i| self.d.text(e).unwrap_or("").contains(i)));
        self.d.set_tag(e, if is_code { t::CODE } else { t::QUOTE });
    }

    /// htmlprocessing.convert_headings.
    fn convert_headings(&mut self, e: NodeId) {
        self.d.attrib_clear(e);
        let name = self.d.tag_name(e).to_owned();
        self.set_attr(e, "rend", &name);
        self.d.set_tag(e, t::HEAD);
    }

    /// htmlprocessing.convert_details.
    fn convert_details(&mut self, e: NodeId) {
        self.d.set_tag(e, t::DIV);
        let mut it = Iter::new(&self.d, e, Matcher::tags(&[t::SUMMARY]), true);
        while let Some(s) = it.next(&self.d) {
            self.d.set_tag(s, t::HEAD);
        }
    }

    /// htmlprocessing.convert_tags (links, images and formatting off).
    pub fn convert_tags(&mut self, tree: NodeId) -> NodeId {
        let containers = if self.o.tables {
            TagSet::of(&[t::DIV, t::LI, t::P, t::TABLE])
        } else {
            TagSet::of(&[t::DIV, t::LI, t::P])
        };
        {
            let mut it = Iter::new(&self.d, tree, Matcher::tags(&[t::A]), true);
            while let Some(e) = it.next(&self.d) {
                if self
                    .d
                    .first_ancestor_in(e, &containers)
                    .is_some_and(|a| a != tree)
                {
                    self.d.set_tag(e, t::REF);
                }
            }
        }
        self.d.strip_tags(tree, &Matcher::tags(&[t::A]));
        {
            let mut it = Iter::new(&self.d, tree, Matcher::tags(&[t::STRONG]), true);
            while let Some(e) = it.next(&self.d) {
                if self
                    .d
                    .attr(e, t::CLASS)
                    .unwrap_or("")
                    .contains("schema-faq-question")
                {
                    self.d.attrib_clear(e);
                    self.set_attr(e, "rend", "h3");
                    self.d.set_tag(e, t::HEAD);
                }
            }
        }
        {
            let mut it = Iter::new(&self.d, tree, Matcher::tags(&[t::SUB, t::SUP]), true);
            while let Some(e) = it.next(&self.d) {
                if self.d.text(e).is_none_or(str::is_empty) && self.d.len(e) == 0 {
                    self.delete_element(e, true);
                }
            }
        }
        self.d.strip_tags(tree, &Matcher::tags(REND_TAGS));
        let mut it = Iter::new(&self.d, tree, Matcher::tags(CONVERSION_TAGS), true);
        while let Some(e) = it.next(&self.d) {
            let g = self.d.tag(e);
            if g == t::DL || g == t::OL || g == t::UL {
                self.convert_lists(e);
            } else if [t::H1, t::H2, t::H3, t::H4, t::H5, t::H6].contains(&g) {
                self.convert_headings(e);
            } else if g == t::BR || g == t::HR {
                self.d.set_tag(e, t::LB);
            } else if g == t::BLOCKQUOTE || g == t::PRE || g == t::Q {
                self.convert_quotes(e);
            } else if g == t::DEL || g == t::S || g == t::STRIKE {
                self.d.set_tag(e, t::DEL);
                self.set_attr(e, "rend", "overstrike");
            } else if g == t::DETAILS {
                self.convert_details(e);
            }
        }
        tree
    }
}
