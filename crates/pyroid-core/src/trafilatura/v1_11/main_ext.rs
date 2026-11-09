// Derived from trafilatura 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
//! trafilatura 1.11.0's main_extractor.py.

use std::collections::HashSet;

use super::super::pystr::{self, py_len, text_chars_test};
use super::super::xpaths::select;
use super::{V11, concat, xpaths};
use crate::html::tree::{Iter, Matcher, Name, NodeId, TagSet, names as t};

const P_FORMATTING: &[Name] = &[t::HI, t::REF];
const TABLE_ELEMS: &[Name] = &[t::TD, t::TH];
const TABLE_ALL: &[Name] = &[t::TD, t::TH, t::HI];
const FORMATTING: &[Name] = &[t::HI, t::REF, t::SPAN];
const CODES_QUOTES: &[Name] = &[t::CODE, t::QUOTE];
const NOT_AT_THE_END: &[Name] = &[t::HEAD, t::REF];
/// utils.FORMATTING_PROTECTED.
const FORMATTING_PROTECTED: &[Name] = &[
    t::CELL,
    t::HEAD,
    t::HI,
    t::ITEM,
    t::P,
    t::QUOTE,
    t::REF,
    t::TD,
];

#[inline]
fn truthy(s: Option<&str>) -> bool {
    s.is_some_and(|s| !s.is_empty())
}

/// lxml.etree's check of an attribute name (`_getNsTag`, then `_attributeValidOrRaise`): a
/// `{uri}` prefix is a namespace (kept here as part of the name), the rest an XML name
/// without colon.
fn attr_name_valid(k: &str) -> bool {
    match k.strip_prefix('{') {
        Some(r) => r
            .find('}')
            .is_some_and(|i| !r[i + 1..].is_empty() && super::super::xml_tag_valid(&r[i + 1..])),
        None => super::super::xml_tag_valid(k),
    }
}

impl V11<'_, '_> {
    fn is_text_element(&self, e: Option<NodeId>) -> bool {
        e.is_some_and(|e| text_chars_test(Some(&self.d.itertext_join(e, ""))))
    }

    fn handle_titles(&mut self, e: NodeId) -> Option<NodeId> {
        let title = if self.d.len(e) == 0 {
            self.process_node(e)
        } else {
            let title = self.d.deep_copy(e);
            for child in self.d.children_vec(e) {
                if let Some(pc) = self.handle_textnode(child, false, false) {
                    self.append(title, pc);
                }
                if self.d.is_element(child) {
                    self.d.set_tag(child, t::DONE);
                }
            }
            Some(title)
        };
        if self.is_text_element(title) {
            title
        } else {
            None
        }
    }

    fn handle_formatting(&mut self, e: NodeId) -> Option<NodeId> {
        let formatting = self.process_node(e);
        if self.d.len(e) == 0 && formatting.is_none() {
            return None;
        }
        let parent = self.d.parent(e).or_else(|| self.d.prev(e));
        if parent.is_none_or(|p| !FORMATTING_PROTECTED.contains(&self.d.tag(p))) {
            let p = self.d.element(t::P);
            match formatting {
                Some(f) => self.insert(p, 0, f),
                // `insert(0, None)`: TypeError
                None => self.raise(),
            }
            Some(p)
        } else {
            formatting
        }
    }

    /// main_extractor.add_sub_element: tag, text, tail and every attribute of the source.
    fn add_sub_element(&mut self, new_child: NodeId, subelem: NodeId, processed: NodeId) {
        let tag = self.d.tag(processed);
        let sub = self.sub_element_checked(new_child, tag);
        let (tx, tl) = (
            self.d.text(processed).map(str::to_owned),
            self.d.tail(processed).map(str::to_owned),
        );
        self.set_text(sub, tx);
        self.set_tail(sub, tl);
        let attrs: Vec<(String, String)> = self
            .d
            .attrs(subelem)
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        for (k, v) in attrs {
            if !attr_name_valid(&k) {
                self.raise();
            }
            self.set_attr(sub, &k, &v);
        }
    }

    fn process_nested_elements(&mut self, child: NodeId, new_child: NodeId) {
        let tx = self.d.text(child).map(str::to_owned);
        self.set_text(new_child, tx);
        let mut it = Iter::new(&self.d, child, Matcher::AnyElement, false);
        while let Some(sub) = it.next(&self.d) {
            if self.d.tag(sub) == t::LIST {
                if let Some(pl) = self.handle_lists(sub) {
                    self.append(new_child, pl);
                }
            } else if let Some(pc) = self.handle_textnode(sub, false, false) {
                self.add_sub_element(new_child, sub, pc);
            }
            self.d.set_tag(sub, t::DONE);
        }
    }

    fn define_newelem(&mut self, processed: Option<NodeId>, orig: NodeId) {
        let Some(p) = processed else { return };
        let tag = self.d.tag(p);
        let child = self.sub_element_checked(orig, tag);
        let (tx, tl) = (
            self.d.text(p).map(str::to_owned),
            self.d.tail(p).map(str::to_owned),
        );
        self.set_text(child, tx);
        self.set_tail(child, tl);
    }

    fn update_elem_rendition(&mut self, e: NodeId, new_elem: NodeId) {
        if let Some(r) = self.d.attr(e, t::REND).map(str::to_owned) {
            self.set_attr(new_elem, "rend", &r);
        }
    }

    pub fn handle_lists(&mut self, e: NodeId) -> Option<NodeId> {
        let tag = self.d.tag(e);
        let processed = self.d.element(tag);
        if self.d.text(e).is_some_and(|s| !pystr::strip(s).is_empty()) {
            let item = self.d.sub_element(processed, t::ITEM);
            let tx = self.d.text(e).map(str::to_owned);
            self.set_text(item, tx);
        }
        let mut it = Iter::new(&self.d, e, Matcher::tags(&[t::ITEM]), false);
        while let Some(child) = it.next(&self.d) {
            let new_child = self.d.element(t::ITEM);
            if self.d.len(child) == 0 {
                if let Some(pc) = self.process_node(child) {
                    let mut text = self.d.text(pc).map(str::to_owned);
                    if let Some(tail) = self
                        .d
                        .tail(pc)
                        .filter(|s| !pystr::strip(s).is_empty())
                        .map(str::to_owned)
                    {
                        text = Some(concat(self, text.as_deref(), Some(&format!(" {tail}"))));
                    }
                    self.set_text(new_child, text);
                    self.append(processed, new_child);
                }
            } else {
                self.process_nested_elements(child, new_child);
                if let Some(ct) = self
                    .d
                    .tail(child)
                    .filter(|s| !pystr::strip(s).is_empty())
                    .map(str::to_owned)
                {
                    let kids: Vec<NodeId> = self
                        .d
                        .children(new_child)
                        .filter(|&c| !(self.d.is_element(c) && self.d.tag(c) == t::DONE))
                        .collect();
                    if let Some(&last) = kids.last() {
                        let v = match self.d.tail(last) {
                            Some(lt) if !pystr::strip(lt).is_empty() => format!("{lt} {ct}"),
                            _ => ct,
                        };
                        self.set_tail(last, Some(v));
                    }
                }
            }
            if truthy(self.d.text(new_child)) || self.d.len(new_child) > 0 {
                self.update_elem_rendition(child, new_child);
                self.append(processed, new_child);
            }
            self.d.set_tag(child, t::DONE);
        }
        self.d.set_tag(e, t::DONE);
        if self.is_text_element(Some(processed)) {
            self.update_elem_rendition(e, processed);
            return Some(processed);
        }
        None
    }

    fn is_code_block_element(&self, e: NodeId) -> bool {
        if truthy(self.d.attr(e, t::LANG)) || self.d.tag(e) == t::CODE {
            return true;
        }
        if let Some(p) = self.d.parent(e) {
            if self.d.attr(p, t::CLASS).unwrap_or("").contains("highlight") {
                return true;
            }
        }
        self.d.len(e) == 1
            && self
                .d
                .children(e)
                .any(|c| self.d.is_element(c) && self.d.tag(c) == t::CODE)
    }

    fn handle_code_blocks(&mut self, e: NodeId) -> NodeId {
        let processed = self.d.deep_copy(e);
        for c in self.d.iter_vec(e, &Matcher::AnyElement, true) {
            self.d.set_tag(c, t::DONE);
        }
        self.d.set_tag(processed, t::CODE);
        processed
    }

    fn handle_quotes(&mut self, e: NodeId) -> Option<NodeId> {
        if self.is_code_block_element(e) {
            return Some(self.handle_code_blocks(e));
        }
        let tag = self.d.tag(e);
        let processed = self.d.element(tag);
        let mut it = Iter::new(&self.d, e, Matcher::AnyElement, true);
        while let Some(child) = it.next(&self.d) {
            let pc = self.process_node(child);
            self.define_newelem(pc, processed);
            self.d.set_tag(child, t::DONE);
        }
        if self.is_text_element(Some(processed)) {
            self.d.strip_tags(processed, &Matcher::tags(&[t::QUOTE]));
            return Some(processed);
        }
        None
    }

    fn handle_other_elements(&mut self, e: NodeId, potential: &TagSet) -> Option<NodeId> {
        if self.d.tag(e) == t::DIV && self.d.attr(e, t::CLASS).unwrap_or("").contains("w3-code") {
            return Some(self.handle_code_blocks(e));
        }
        if !potential.contains(self.d.tag(e)) {
            return None;
        }
        if self.d.tag(e) == t::DIV {
            let p = self.handle_textnode(e, false, true)?;
            if text_chars_test(self.d.text(p)) {
                self.d.attrib_clear(p);
                if self.d.tag(p) == t::DIV {
                    self.d.set_tag(p, t::P);
                }
                return Some(p);
            }
        }
        None
    }

    fn handle_paragraphs(&mut self, e: NodeId, potential: &TagSet) -> Option<NodeId> {
        self.d.attrib_clear(e);
        if self.d.len(e) == 0 {
            return self.process_node(e);
        }
        let tag = self.d.tag(e);
        let processed = self.d.element(tag);
        let mut it = Iter::new(&self.d, e, Matcher::AnyElement, true);
        while let Some(child) = it.next(&self.d) {
            let g = self.d.tag(child);
            if !potential.contains(g) && g != t::DONE {
                continue;
            }
            if let Some(pc) = self.handle_textnode(child, false, true) {
                if self.d.tag(pc) == t::P {
                    let v = if truthy(self.d.text(processed)) {
                        let (a, b) = (
                            self.d.text(processed).map(|s| format!("{s} ")),
                            self.d.text(pc).map(str::to_owned),
                        );
                        Some(concat(self, a.as_deref(), b.as_deref()))
                    } else {
                        self.d.text(pc).map(str::to_owned)
                    };
                    self.set_text(processed, v);
                    self.d.set_tag(child, t::DONE);
                    continue;
                }
                let tag = self.d.tag(child);
                let newsub = self.d.element(tag);
                if P_FORMATTING.contains(&self.d.tag(pc)) {
                    if self.d.len(pc) > 0 {
                        // lxml's child iterator: the next child is fetched before the body
                        let mut cur = self.d.first_child(pc);
                        while let Some(item) = cur {
                            cur = self.d.next(item);
                            if text_chars_test(
                                self.d.text(item).filter(|_| self.d.is_element(item)),
                            ) {
                                let v = format!(" {}", self.d.text(item).unwrap_or(""));
                                self.set_text(item, Some(v));
                            }
                            let tag = self.d.tag(item);
                            if self.d.is_element(item) {
                                self.d.strip_tags(pc, &Matcher::tags(&[tag]));
                            }
                        }
                    }
                    if self.d.tag(child) == t::HI {
                        match self.d.attr(child, t::REND).map(str::to_owned) {
                            Some(r) => self.set_attr(newsub, "rend", &r),
                            // `set("rend", None)`: TypeError
                            None => self.raise(),
                        }
                    } else if self.d.tag(child) == t::REF {
                        if let Some(tg) = self.d.attr(child, t::TARGET).map(str::to_owned) {
                            self.set_attr(newsub, "target", &tg);
                        }
                    }
                }
                let (tx, tl) = (
                    self.d.text(pc).map(str::to_owned),
                    self.d.tail(pc).map(str::to_owned),
                );
                self.set_text(newsub, tx);
                self.set_tail(newsub, tl);
                self.append(processed, newsub);
            }
            self.d.set_tag(child, t::DONE);
        }
        if let Some(last) = self.d.last_child(processed) {
            if self.d.tag(last) == t::LB && self.d.tail(last).is_none() {
                self.d.unlink(last);
            }
            return Some(processed);
        }
        if truthy(self.d.text(processed)) {
            return Some(processed);
        }
        None
    }

    fn handle_table(&mut self, table: NodeId, potential: &TagSet) -> Option<NodeId> {
        let newtable = self.d.element(t::TABLE);
        self.d
            .strip_tags(table, &Matcher::tags(&[t::THEAD, t::TBODY, t::TFOOT]));
        let mut max_cols: i128 = 0;
        for tr in self.d.iter_vec(table, &Matcher::tags(&[t::TR]), true) {
            let mut sum: i128 = 0;
            for td in self.d.iter_vec(tr, &Matcher::tags(TABLE_ELEMS), true) {
                let v = match self
                    .d
                    .attr(td, t::COLSPAN)
                    .map(|s| pystr::py_int(s, self.v))
                {
                    None => 1,
                    Some(Ok(n)) => n,
                    Some(Err(_)) => {
                        // int(): ValueError
                        self.raise();
                        return None;
                    }
                };
                sum = sum.saturating_add(v);
            }
            max_cols = max_cols.max(sum);
        }
        let span = (max_cols > 1).then(|| max_cols.to_string());
        let make_row = |me: &mut Self| {
            let r = me.d.element(t::ROW);
            if let Some(s) = &span {
                me.d.set(r, "span", s);
            }
            r
        };
        let (mut seen_header_row, mut seen_header) = (false, false);
        let mut newrow = make_row(self);
        let mut tablepot = potential.clone();
        tablepot.insert(t::DIV);
        let mut it = Iter::new(&self.d, table, Matcher::AnyNode, false);
        while let Some(sub) = it.next(&self.d) {
            let g = if self.d.is_element(sub) {
                self.d.tag(sub)
            } else {
                t::NONE
            };
            if g == t::TR {
                if self.d.len(newrow) > 0 {
                    self.append(newtable, newrow);
                    newrow = make_row(self);
                    seen_header_row = seen_header_row || seen_header;
                }
            } else if TABLE_ELEMS.contains(&g) {
                let is_header = g == t::TH && !seen_header_row;
                seen_header = seen_header || is_header;
                let new_child = self.d.element(t::CELL);
                if is_header {
                    self.d.set(new_child, "role", "head");
                }
                if self.d.len(sub) == 0 {
                    if let Some(pc) = self.process_node(sub) {
                        let (tx, tl) = (
                            self.d.text(pc).map(str::to_owned),
                            self.d.tail(pc).map(str::to_owned),
                        );
                        self.set_text(new_child, tx);
                        self.set_tail(new_child, tl);
                    }
                } else {
                    let (tx, tl) = (
                        self.d.text(sub).map(str::to_owned),
                        self.d.tail(sub).map(str::to_owned),
                    );
                    self.set_text(new_child, tx);
                    self.set_tail(new_child, tl);
                    self.d.set_tag(sub, t::DONE);
                    let mut cit = Iter::new(&self.d, sub, Matcher::AnyNode, false);
                    while let Some(child) = cit.next(&self.d) {
                        if !self.d.is_element(child) {
                            continue;
                        }
                        let cg = self.d.tag(child);
                        let processed = if TABLE_ALL.contains(&cg) {
                            if TABLE_ELEMS.contains(&cg) {
                                self.d.set_tag(child, t::CELL);
                            }
                            self.handle_textnode(child, true, true)
                        } else if cg == t::LIST && self.focus == super::Focus::Recall {
                            if let Some(pl) = self.handle_lists(child) {
                                self.append(new_child, pl);
                            }
                            None
                        } else {
                            self.handle_textelem(child, &tablepot)
                        };
                        self.define_newelem(processed, new_child);
                        self.d.set_tag(child, t::DONE);
                    }
                }
                if truthy(self.d.text(new_child)) || self.d.len(new_child) > 0 {
                    self.append(newrow, new_child);
                }
            } else if g == t::TABLE {
                break;
            }
            if self.d.is_element(sub) {
                self.d.set_tag(sub, t::DONE);
            }
        }
        // the last row loses its span attribute
        self.d.nm(newrow).attrs.retain(|a| a.name != t::SPAN);
        if self.d.len(newrow) > 0 {
            self.append(newtable, newrow);
        }
        (self.d.len(newtable) > 0).then_some(newtable)
    }

    pub fn handle_textelem(&mut self, e: NodeId, potential: &TagSet) -> Option<NodeId> {
        let g = self.d.tag(e);
        if g == t::LIST {
            self.handle_lists(e)
        } else if CODES_QUOTES.contains(&g) {
            self.handle_quotes(e)
        } else if g == t::HEAD {
            self.handle_titles(e)
        } else if g == t::P {
            self.handle_paragraphs(e, potential)
        } else if g == t::LB {
            if text_chars_test(self.d.tail(e)) {
                if let Some(te) = self.process_node(e) {
                    let p = self.d.element(t::P);
                    let tl = self.d.tail(te).map(str::to_owned);
                    self.set_text(p, tl);
                    return Some(p);
                }
            }
            None
        } else if FORMATTING.contains(&g) {
            self.handle_formatting(e)
        } else if g == t::TABLE && potential.contains(t::TABLE) {
            self.handle_table(e, potential)
        } else {
            // `graphic` needs include_images, which this port does not support
            self.handle_other_elements(e, potential)
        }
    }

    fn recover_wild_text(&mut self, tree: NodeId, result_body: NodeId, potential: &mut TagSet) {
        let recall = self.focus == super::Focus::Recall;
        if recall {
            potential.insert(t::DIV);
            potential.insert(t::LB);
        }
        let (search_tree, _) = self.prune_unwanted_sections(tree, potential);
        if potential.contains(t::REF) {
            self.d.strip_tags(search_tree, &Matcher::tags(&[t::SPAN]));
        } else {
            self.d
                .strip_tags(search_tree, &Matcher::tags(&[t::A, t::REF, t::SPAN]));
        }
        let subelems = xpaths::wild_text_search(&self.d, search_tree, recall);
        // extend() consumes the generator lazily: each result is appended before the next
        for e in subelems {
            if let Some(p) = self.handle_textelem(e, potential) {
                self.append(result_body, p);
            }
        }
    }

    /// main_extractor.prune_unwanted_sections (1.11.0): the pruned tree, and whether it is the
    /// backup copy prune_unwanted_nodes can return (a document of its own).
    fn prune_unwanted_sections(&mut self, tree: NodeId, potential: &TagSet) -> (NodeId, bool) {
        let favor_precision = self.focus == super::Focus::Precision;
        let input = tree;
        let mut tree = self.prune_unwanted_nodes(tree, self.p.overall_discard, true);
        let is_copy = tree != input;
        tree = self.prune_unwanted_nodes(tree, xpaths::PAYWALL_DISCARD_XPATH, false);
        if !potential.contains(t::GRAPHIC) {
            tree = self.prune_unwanted_nodes(tree, self.p.discard_image_elements, false);
        }
        if self.focus != super::Focus::Recall {
            tree = self.prune_unwanted_nodes(tree, self.p.teaser_discard, false);
            if favor_precision {
                tree = self.prune_unwanted_nodes(tree, self.p.precision_discard, false);
            }
        }
        self.delete_by_link_density(tree, t::DIV, true, favor_precision);
        self.delete_by_link_density(tree, t::LIST, false, favor_precision);
        self.delete_by_link_density(tree, t::P, false, favor_precision);
        if favor_precision {
            while let Some(last) = self.d.last_child(tree) {
                if !(self.d.is_element(last) && self.d.tag(last) == t::HEAD) {
                    break;
                }
                self.d.unlink(last);
            }
            self.delete_by_link_density(tree, t::HEAD, false, false);
            self.delete_by_link_density(tree, t::QUOTE, false, false);
        }
        (tree, is_copy)
    }

    /// main_extractor._extract.
    fn extract_main(&mut self, tree: NodeId) -> (NodeId, String, TagSet) {
        let mut potential = TagSet::of(self.p.tag_catalog);
        if self.o.tables {
            for g in [t::TABLE, t::TD, t::TH, t::TR] {
                potential.insert(g);
            }
        }
        let result_body = self.d.element(t::BODY);
        for expr in self.p.body_xpath {
            let Some(subtree) = expr(&self.d, tree) else {
                continue;
            };
            let (subtree, is_copy) = self.prune_unwanted_sections(subtree, &potential);
            if potential.contains(t::TABLE) || self.focus == super::Focus::Precision {
                // a live iteration: removing a table leaves the walk inside it
                let mut it = Iter::new(&self.d, subtree, Matcher::tags(&[t::TABLE]), true);
                while let Some(e) = it.next(&self.d) {
                    if self.link_density_test_tables(e) {
                        self.d.unlink(e);
                    }
                }
            }
            if self.d.len(subtree) == 0 {
                continue;
            }
            // `subtree.xpath('//p//text()')`: the whole document, the copy's or the page's (a
            // subtree removed by the link-density pass still belongs to the page, and so do
            // its top-level elements after the first)
            let roots = if is_copy {
                vec![subtree]
            } else {
                self.doc_roots(tree)
            };
            let (mut any_p, mut plen) = (false, 0);
            for r in roots {
                let (a, n) = super::super::xpaths::p_text(&self.d, r);
                any_p |= a;
                plen += n;
            }
            let factor = if self.focus == super::Focus::Precision {
                1
            } else {
                3
            };
            if !any_p || plen < self.o.min_extracted_size * factor {
                potential.insert(t::DIV);
            }
            if !potential.contains(t::REF) {
                self.d.strip_tags(subtree, &Matcher::tags(&[t::REF]));
            }
            if !potential.contains(t::SPAN) {
                self.d.strip_tags(subtree, &Matcher::tags(&[t::SPAN]));
            }
            let mut subelems = select(&self.d, subtree, |_, _| true);
            let tags: HashSet<Name> = subelems.iter().map(|&e| self.d.tag(e)).collect();
            if tags.len() == 1 && tags.contains(&t::LB) {
                subelems = vec![subtree];
            }
            // a list comprehension: every element is handled before any is appended
            let mut processed = Vec::new();
            for e in subelems {
                if let Some(p) = self.handle_textelem(e, &potential) {
                    processed.push(p);
                }
            }
            for p in processed {
                self.append(result_body, p);
            }
            while let Some(last) = self.d.last_child(result_body) {
                if !NOT_AT_THE_END.contains(&self.d.tag(last)) {
                    break;
                }
                self.d.unlink(last);
            }
            if self.d.len(result_body) > 1 {
                break;
            }
        }
        let temp_text = pystr::strip(&self.d.itertext_join(result_body, " ")).to_owned();
        (result_body, temp_text, potential)
    }

    /// main_extractor.extract_content: (body, text, length).
    pub fn extract_content(&mut self, cleaned: NodeId) -> (NodeId, String, usize) {
        let backup = self.d.deep_copy(cleaned);
        let (result_body, mut temp_text, mut potential) = self.extract_main(cleaned);
        if self.d.len(result_body) == 0 || py_len(&temp_text) < self.o.min_extracted_size {
            self.recover_wild_text(backup, result_body, &mut potential);
            pystr::strip(&self.d.itertext_join(result_body, " ")).clone_into(&mut temp_text);
        }
        self.d
            .strip_elements(result_body, &Matcher::tags(&[t::DONE]));
        self.d.strip_tags(result_body, &Matcher::tags(&[t::DIV]));
        let n = py_len(&temp_text);
        (result_body, temp_text, n)
    }

    /// main_extractor.extract_comments: (comments body, text); the comments leave `tree`.
    pub fn extract_comments(&mut self, tree: NodeId) -> (NodeId, String) {
        let comments_body = self.d.element(t::BODY);
        let potential = TagSet::of(self.p.tag_catalog);
        for expr in self.p.comments_xpath {
            let Some(subtree) = expr(&self.d, tree) else {
                continue;
            };
            let subtree = self.prune_unwanted_nodes(subtree, self.p.comments_discard, false);
            self.d
                .strip_tags(subtree, &Matcher::tags(&[t::A, t::REF, t::SPAN]));
            for e in select(&self.d, subtree, |_, _| true) {
                if potential.contains(self.d.tag(e)) {
                    if let Some(p) = self.handle_textnode(e, true, false) {
                        self.d.attrib_clear(p);
                        self.append(comments_body, p);
                    }
                }
            }
            if self.d.len(comments_body) > 0 {
                self.d.unlink(subtree);
                break;
            }
        }
        let text = pystr::strip(&self.d.itertext_join(comments_body, " ")).to_owned();
        (comments_body, text)
    }
}
