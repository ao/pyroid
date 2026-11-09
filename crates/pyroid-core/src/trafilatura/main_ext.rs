// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! main_extractor.py: BODY_XPATH candidates, the handle_* functions, wild-text recovery and the
//! comments extractor.

use std::collections::{HashMap, HashSet};

use super::htmlproc::tagset_of;
use super::pystr::{self, py_len, text_chars_test, trim};
use super::{Ctx, Focus, matchers, xpaths};
use crate::html::tree::{Iter, Matcher, Name, NodeId, TagSet, names as t};

pub const INLINE_CONSUMING: &[Name] = &[t::HI, t::REF, t::DEL];
pub const INLINE_FORMATTABLE: &[Name] = &[t::HI, t::REF, t::DEL, t::CODE];
pub const INLINE_CARRIED: &[Name] = &[t::HI, t::REF, t::DEL, t::CODE, t::GRAPHIC];
const TABLE_ELEMS: &[Name] = &[t::TD, t::TH];
const KEEP_ATTRS: &[&str] = &["rend", "role", "target", "src", "alt", "title"];
const CODES_QUOTES: &[Name] = &[t::CODE, t::QUOTE];
const REBUILT_BLOCKS: &[Name] = &[t::CODE, t::QUOTE, t::LIST, t::TABLE];
const NOT_AT_THE_END: &[Name] = &[t::HEAD, t::REF];
pub const FORMATTING_PROTECTED: &[Name] = &[
    t::CELL,
    t::HEAD,
    t::HI,
    t::ITEM,
    t::P,
    t::QUOTE,
    t::REF,
    t::TD,
];
pub const SPACING_PROTECTED: &[Name] = &[t::CODE, t::PRE];
const MAX_SPAN: usize = 100;

#[inline]
fn truthy(s: Option<&str>) -> bool {
    s.is_some_and(|s| !s.is_empty())
}

impl Ctx<'_> {
    /// _QUOTE_TAGS: TAG_CATALOG plus ref and graphic.
    fn quote_tags(&self) -> TagSet {
        let mut s = TagSet::of(self.p.tag_catalog);
        s.insert(t::REF);
        s.insert(t::GRAPHIC);
        s
    }

    /// utils.image_src.
    pub fn image_src(&self, e: NodeId) -> Option<String> {
        let is_image = |s: &str| py_len(s) <= 8192 && matchers::image_extension(s, self.v);
        for a in [t::DATA_SRC, t::SRC] {
            let s = self.d.attr(e, a).unwrap_or("");
            if is_image(s) {
                return Some(s.to_owned());
            }
        }
        self.d
            .attrs(e)
            .find(|(k, v)| k.starts_with("data-src") && is_image(v))
            .map(|(_, v)| v.to_owned())
    }

    /// main_extractor._elem_text: `trim("".join(e.itertext()))`.
    pub fn elem_text(&self, e: NodeId) -> String {
        trim(&self.d.itertext_join(e, ""))
    }

    fn is_text_element(&self, e: Option<NodeId>) -> bool {
        e.is_some_and(|e| text_chars_test(Some(&self.d.itertext_join(e, ""))))
    }

    fn wraps_inline(&self, e: NodeId) -> bool {
        self.d.len(e) > 0
            && (self.d.tag(e) == t::REF
                || self
                    .d
                    .children(e)
                    .any(|c| INLINE_CARRIED.contains(&self.d.tag(c))))
    }

    fn handle_titles(&mut self, e: NodeId) -> Option<NodeId> {
        let title = if self.d.len(e) == 0 {
            self.process_node(e)
        } else {
            let title = self.d.deep_copy(e);
            self.d.strip_elements(title, &Matcher::tags(&[t::DONE]));
            let mut it = Iter::new(&self.d, e, Matcher::AnyElement, false);
            while let Some(c) = it.next(&self.d) {
                self.d.set_tag(c, t::DONE);
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
        let formatting = self.process_node(e)?;
        if let Some(p) = self.d.parent(e) {
            if FORMATTING_PROTECTED.contains(&self.d.tag(p)) {
                return Some(formatting);
            }
        }
        let wrapper = self.d.element(t::P);
        self.append(wrapper, formatting);
        Some(wrapper)
    }

    fn process_nested_elements(&mut self, child: NodeId, new_child_elem: NodeId) {
        let tx = self.d.text(child).map(str::to_owned);
        self.set_text(new_child_elem, tx);
        let quote_tags = self.quote_tags();
        let mut it = Iter::new(&self.d, child, Matcher::AnyElement, false);
        while let Some(sub) = it.next(&self.d) {
            let g = self.d.tag(sub);
            if g == t::LIST {
                let r = self.handle_lists(sub);
                self.append_block(new_child_elem, r, sub);
            } else if g == t::P && self.d.len(sub) > 0 {
                let r = self.handle_paragraphs(sub, &quote_tags);
                self.append_block(new_child_elem, r, sub);
            } else if g == t::GRAPHIC {
                let r = self.handle_image(sub);
                self.append_block(new_child_elem, r, sub);
            } else if INLINE_CARRIED.contains(&g) {
                self.define_newelem(Some(sub), new_child_elem, true);
            } else {
                if self.d.len(sub) > 0 && text_chars_test(self.d.tail(sub)) {
                    if let Some(last) = self.d.last_child(sub) {
                        let v = format!(
                            "{}{}",
                            self.d.tail(last).unwrap_or(""),
                            self.d.tail(sub).unwrap_or("")
                        );
                        self.set_tail(last, Some(v));
                        self.set_tail(sub, None);
                    }
                }
                let r = self.handle_textnode(sub, false, false);
                self.define_newelem(r, new_child_elem, false);
            }
            self.d.set_tag(sub, t::DONE);
        }
    }

    /// main_extractor._append_block.
    fn append_block(&mut self, parent: NodeId, processed: Option<NodeId>, source: NodeId) {
        if let Some(p) = processed {
            let tl = self.d.tail(source).map(str::to_owned);
            self.set_tail(p, tl);
            self.append(parent, p);
        } else if truthy(self.d.tail(source)) && text_chars_test(self.d.tail(source)) {
            let last = self.d.last_child(parent);
            let before = match last {
                None => self.d.text(parent),
                Some(l) => self.d.tail(l),
            }
            .unwrap_or("")
            .to_owned();
            let mut tail = self.d.tail(source).unwrap_or("").to_owned();
            let before_ends_space = before.chars().next_back().is_some_and(pystr::is_space);
            let tail_starts_space = tail.chars().next().is_some_and(pystr::is_space);
            if self.d.tag(source) != t::GRAPHIC
                && (!before.is_empty() || last.is_some())
                && !before_ends_space
                && !tail_starts_space
            {
                tail = format!(" {tail}");
            }
            let v = Some(before + &tail);
            match last {
                None => self.set_text(parent, v),
                Some(l) => self.set_tail(l, v),
            }
        }
    }

    fn update_elem_rendition(&mut self, e: NodeId, new_elem: NodeId) {
        if let Some(r) = self
            .d
            .attr(e, t::REND)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
        {
            self.set_attr(new_elem, "rend", &r);
        }
    }

    fn copy_attrs(&mut self, source: NodeId, target: NodeId) {
        let attrs: Vec<(String, String)> = self
            .d
            .attrs(source)
            .filter(|(k, _)| KEEP_ATTRS.contains(k))
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        for (k, v) in attrs {
            self.set_attr(target, &k, &v);
        }
    }

    /// main_extractor.define_newelem (recursion bounded by the tree depth).
    fn define_newelem(&mut self, processed: Option<NodeId>, orig: NodeId, keep_children: bool) {
        let Some(p) = processed else { return };
        let child = self.sub_element_checked(orig, self.d.tag(p));
        let (tx, tl) = (
            self.d.text(p).map(str::to_owned),
            self.d.tail(p).map(str::to_owned),
        );
        self.set_text(child, tx);
        self.set_tail(child, tl);
        self.copy_attrs(p, child);
        if keep_children {
            for sub in self.d.children_vec(p) {
                let g = self.d.tag(sub);
                if self.d.is_element(sub) && (INLINE_CARRIED.contains(&g) || g == t::LB) {
                    self.define_newelem(Some(sub), child, true);
                    for c in self.d.iter_vec(sub, &Matcher::AnyElement, true) {
                        self.d.set_tag(c, t::DONE);
                    }
                }
            }
        }
    }

    fn handle_lists(&mut self, e: NodeId) -> Option<NodeId> {
        let processed = self.d.element(self.d.tag(e));
        if text_chars_test(self.d.text(e)) {
            let item = self.d.sub_element(processed, t::ITEM);
            let tx = self.d.text(e).map(str::to_owned);
            self.set_text(item, tx);
        }
        let mut it = Iter::new(&self.d, e, Matcher::tags(&[t::ITEM]), false);
        while let Some(child) = it.next(&self.d) {
            // items of a nested list are processed by the recursive call and renamed "done",
            // but lxml fetched the first of them before that
            if self.d.tag(child) == t::DONE {
                continue;
            }
            let new_child = self.d.element(t::ITEM);
            if self.d.len(child) == 0 {
                if let Some(pc) = self.process_node(child) {
                    let parts: Vec<&str> = [self.d.text(pc), self.d.tail(pc)]
                        .into_iter()
                        .flatten()
                        .filter(|s| !s.is_empty())
                        .collect();
                    let v = parts.join(" ");
                    self.set_text(new_child, Some(v));
                }
            } else {
                self.process_nested_elements(child, new_child);
                if text_chars_test(self.d.tail(child)) && self.d.len(new_child) > 0 {
                    if let Some(last) = self.d.last_child(new_child) {
                        let ct = self.d.tail(child).unwrap_or("").to_owned();
                        let v = if text_chars_test(self.d.tail(last)) {
                            format!("{} {ct}", self.d.tail(last).unwrap_or(""))
                        } else {
                            ct
                        };
                        self.set_tail(last, Some(v));
                    }
                }
            }
            if self.is_text_element(Some(new_child))
                || self.find_desc_or_self(new_child, t::GRAPHIC)
            {
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

    /// `next(e.iter(tag), None) is not None` (the element itself included).
    fn find_desc_or_self(&self, e: NodeId, tag: Name) -> bool {
        self.d.tag(e) == tag && self.d.is_element(e) || self.find_desc(e, tag).is_some()
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
        match self
            .d
            .children(e)
            .find(|&c| self.d.is_element(c) && self.d.tag(c) == t::CODE)
        {
            Some(c) => {
                self.d.len(e) == 1
                    && pystr::strip(self.d.text(e).unwrap_or("")).is_empty()
                    && pystr::strip(self.d.tail(c).unwrap_or("")).is_empty()
            }
            None => false,
        }
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
        let processed = self.d.element(self.d.tag(e));
        self.process_nested_elements(e, processed);
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
        if self.d.tag(e) != t::DIV || !potential.contains(t::DIV) {
            return None;
        }
        let p = self.handle_textnode(e, false, true)?;
        if !text_chars_test(self.d.text(p)) {
            return None;
        }
        self.d.attrib_clear(p);
        self.d.set_tag(p, t::P);
        Some(p)
    }

    fn handle_paragraphs(&mut self, e: NodeId, potential: &TagSet) -> Option<NodeId> {
        if self.d.len(e) == 0 {
            return self.process_node(e);
        }
        let processed = self.d.element(self.d.tag(e));
        let strip: Vec<Name> = self
            .d
            .iter_vec(e, &Matcher::AnyElement, false)
            .into_iter()
            .map(|c| self.d.tag(c))
            .filter(|&g| !potential.contains(g) && g != t::DONE)
            .collect();
        self.d.strip_tags(e, &Matcher::Tags(tagset_of(strip)));
        let mut it = Iter::new(&self.d, e, Matcher::AnyElement, true);
        while let Some(child) = it.next(&self.d) {
            if let Some(pc) = self.handle_textnode(child, false, true) {
                if self.d.tag(pc) == t::P {
                    let parts: Vec<String> = [self.d.text(processed), self.d.text(pc)]
                        .into_iter()
                        .flatten()
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                        .collect();
                    self.set_text(processed, Some(parts.join(" ")));
                    self.d.set_tag(child, t::DONE);
                    continue;
                }
                let mut newsub = self.d.element(self.d.tag(child));
                if INLINE_CONSUMING.contains(&self.d.tag(pc)) {
                    if self.wraps_inline(pc) {
                        self.define_newelem(Some(pc), processed, true);
                        self.d.set_tag(child, t::DONE);
                        continue;
                    }
                    for item in self.d.children_vec(pc) {
                        if self.d.tag(item) == t::LB && truthy(self.d.tail(item)) {
                            let v = format!(" {}", pystr::lstrip(self.d.tail(item).unwrap_or("")));
                            self.set_tail(item, Some(v));
                        } else if text_chars_test(self.d.text(item)) {
                            let v = format!(" {}", self.d.text(item).unwrap_or(""));
                            self.set_text(item, Some(v));
                        }
                    }
                    let tags = tagset_of(
                        self.d
                            .children(pc)
                            .map(|c| self.d.tag(c))
                            .collect::<Vec<_>>(),
                    );
                    self.d.strip_tags(pc, &Matcher::Tags(tags));
                    self.copy_attrs(child, newsub);
                }
                let (tx, tl) = (
                    self.d.text(pc).map(str::to_owned),
                    self.d.tail(pc).map(str::to_owned),
                );
                self.set_text(newsub, tx);
                self.set_tail(newsub, tl);
                if self.d.tag(pc) == t::GRAPHIC {
                    if let Some(img) = self.handle_image(pc) {
                        newsub = img;
                    }
                }
                self.append(processed, newsub);
            }
            self.d.set_tag(child, t::DONE);
        }
        if let Some(last) = self.d.last_child(processed) {
            if self.d.tag(last) == t::LB && self.d.tail(last).is_none() {
                self.delete_element(last, true);
            }
            return Some(processed);
        }
        if text_chars_test(self.d.text(processed)) {
            return Some(processed);
        }
        None
    }

    fn define_cell_type(&mut self, is_header: bool) -> NodeId {
        let c = self.d.element(t::CELL);
        if is_header {
            self.d.set(c, "role", "head");
        }
        c
    }

    /// main_extractor._span: `min(int(v), 100) if v.isdecimal() else 1` (`int()` raises past
    /// 4300 digits).
    fn span(&mut self, cell: NodeId, attr: Name) -> usize {
        let v = self.d.attr(cell, attr).unwrap_or("1");
        if v.is_empty() || !v.chars().all(|c| pystr::is_decimal(c, self.v)) {
            return 1;
        }
        if v.chars().count() > 4300 {
            self.raise();
            return 1;
        }
        let mut n: usize = 0;
        for c in v.chars() {
            n = n
                .saturating_mul(10)
                .saturating_add(pystr::decimal_value(c, self.v).unwrap_or(0) as usize);
        }
        n.min(MAX_SPAN)
    }

    fn flush_rowspan_phantoms(&mut self, rowspan: &mut HashMap<usize, usize>, newrow: NodeId) {
        loop {
            let col = self.d.len(newrow);
            let Some(v) = rowspan.get_mut(&col) else {
                break;
            };
            *v -= 1;
            if *v == 0 {
                rowspan.remove(&col);
            }
            let c = self.define_cell_type(false);
            self.append(newrow, c);
        }
    }

    fn finalize_row(
        &mut self,
        newtable: NodeId,
        newrow: NodeId,
        rowspan: &mut HashMap<usize, usize>,
        max_cols: usize,
    ) {
        self.flush_rowspan_phantoms(rowspan, newrow);
        while self.d.len(newrow) < max_cols {
            let c = self.define_cell_type(false);
            self.append(newrow, c);
        }
        if self
            .d
            .children(newrow)
            .any(|c| truthy(self.d.text(c)) || self.d.len(c) > 0)
        {
            self.append(newtable, newrow);
        }
    }

    fn fill_cell(
        &mut self,
        new_child: NodeId,
        cell: NodeId,
        nested: &HashSet<NodeId>,
        ptags_with_div: &TagSet,
    ) {
        if self.d.len(cell) == 0 {
            if let Some(pc) = self.process_node(cell) {
                let (tx, tl) = (
                    self.d.text(pc).map(str::to_owned),
                    self.d.tail(pc).map(str::to_owned),
                );
                self.set_text(new_child, tx);
                self.set_tail(new_child, tl);
            }
            return;
        }
        let (tx, tl) = (
            self.d.text(cell).map(str::to_owned),
            self.d.tail(cell).map(str::to_owned),
        );
        self.set_text(new_child, tx);
        self.set_tail(new_child, tl);
        self.d.set_tag(cell, t::DONE);
        let tables = TagSet::of(&[t::TABLE]);
        let mut it = Iter::new(&self.d, cell, Matcher::AnyNode, false);
        while let Some(child) = it.next(&self.d) {
            if !self.d.is_element(child) || self.d.tag(child) == t::DONE {
                continue;
            }
            if nested.contains(&child) {
                if self.d.tag(child) == t::TABLE {
                    // next(child.iterancestors("table")): the enclosing table always exists here
                    let outer_nested = self
                        .d
                        .first_ancestor_in(child, &tables)
                        .is_some_and(|a| nested.contains(&a));
                    if !outer_nested {
                        self.append_block(new_child, None, child);
                        self.set_tail(child, None);
                    }
                }
                continue;
            }
            if self.separates_inline(child) && !text_chars_test(self.d.tail(child)) {
                let lb = self.d.element(t::LB);
                self.append(new_child, lb);
                self.d.set_tag(child, t::DONE);
                continue;
            }
            let g = self.d.tag(child);
            let processed = if TABLE_ELEMS.contains(&g) {
                self.d.set_tag(child, t::CELL);
                self.handle_textnode(child, true, true)
            } else if INLINE_CONSUMING.contains(&g) {
                let r = self.handle_textnode(child, true, true);
                if r.is_none() && self.d.len(child) > 0 {
                    Some(child)
                } else {
                    r
                }
            } else if g == t::LIST {
                let pl = self.handle_lists(child);
                let keep = if self.focus == Focus::Recall {
                    pl
                } else {
                    None
                };
                self.append_block(new_child, keep, child);
                continue;
            } else {
                self.handle_textelem(child, ptags_with_div)
            };
            self.define_newelem(processed, new_child, true);
            self.d.set_tag(child, t::DONE);
        }
    }

    fn handle_table(&mut self, table: NodeId, potential: &TagSet) -> Option<NodeId> {
        let newtable = self.d.element(t::TABLE);
        let mut ptags_with_div = potential.clone();
        ptags_with_div.insert(t::DIV);
        self.d
            .strip_tags(table, &Matcher::tags(&[t::THEAD, t::TBODY, t::TFOOT]));
        let mut nested: HashSet<NodeId> = HashSet::new();
        for nt in self.d.iter_vec(table, &Matcher::tags(&[t::TABLE]), false) {
            nested.extend(self.d.iter_vec(nt, &Matcher::AnyNode, true));
        }
        let mut rows: Vec<Vec<NodeId>> = vec![Vec::new()];
        let mut captions: Vec<String> = Vec::new();
        let mut max_cols = 0usize;
        for e in self.d.children_vec(table) {
            if !self.d.is_element(e) {
                continue;
            }
            let g = self.d.tag(e);
            if g == t::TR {
                let cells: Vec<NodeId> = self
                    .d
                    .children(e)
                    .filter(|&c| self.d.is_element(c) && TABLE_ELEMS.contains(&self.d.tag(c)))
                    .collect();
                let cols: usize = cells.iter().map(|&c| self.span(c, t::COLSPAN)).sum();
                max_cols = max_cols.max(cols);
                rows.push(cells);
            } else if TABLE_ELEMS.contains(&g) {
                if let Some(r) = rows.last_mut() {
                    r.push(e);
                }
                continue;
            } else if g == t::CAPTION {
                captions.push(pystr::strip(&self.d.itertext_join(e, " ")).to_owned());
                for sub in self.d.iter_vec(e, &Matcher::AnyElement, false) {
                    if self.d.tag(sub) == t::GRAPHIC {
                        self.set_tail(sub, None);
                    } else {
                        self.d.set_tag(sub, t::DONE);
                    }
                }
            } else if g == t::TABLE {
                continue;
            }
            self.d.set_tag(e, t::DONE);
        }
        let max_cols = max_cols.min(MAX_SPAN);
        for cap in captions.into_iter().filter(|c| !c.is_empty()) {
            let cell = self.define_cell_type(true);
            self.set_text(cell, Some(cap));
            let row = self.d.element(t::ROW);
            self.append(row, cell);
            self.finalize_row(newtable, row, &mut HashMap::new(), max_cols);
        }
        let mut header_row_emitted = false;
        let mut rowspan: HashMap<usize, usize> = HashMap::new();
        for cells in rows {
            let newrow = self.d.element(t::ROW);
            let mut row_has_th = false;
            for cell in cells {
                let is_header = self.d.tag(cell) == t::TH && !header_row_emitted;
                row_has_th = row_has_th || is_header;
                self.flush_rowspan_phantoms(&mut rowspan, newrow);
                let new_child = self.define_cell_type(is_header);
                let colspan = self.span(cell, t::COLSPAN);
                let rs = self.span(cell, t::ROWSPAN);
                if rs > 1 {
                    let l = self.d.len(newrow);
                    for c in l..l + colspan {
                        rowspan.insert(c, rs - 1);
                    }
                }
                self.fill_cell(new_child, cell, &nested, &ptags_with_div);
                self.append(newrow, new_child);
                for _ in 1..colspan {
                    let c = self.define_cell_type(is_header);
                    self.append(newrow, c);
                }
                self.d.set_tag(cell, t::DONE);
            }
            self.finalize_row(newtable, newrow, &mut rowspan, max_cols);
            header_row_emitted = header_row_emitted || row_has_th;
        }
        (self.d.len(newtable) > 0).then_some(newtable)
    }

    /// main_extractor.handle_image (options.url is None).
    pub fn handle_image(&mut self, e: NodeId) -> Option<NodeId> {
        let mut link = self.image_src(e)?;
        if !link.starts_with("http") {
            if let Some(rest) = link.strip_prefix("//") {
                link = format!("http://{rest}");
            }
        }
        let p = self.d.element(self.d.tag(e));
        self.set_attr(p, "src", &link);
        if let Some(a) = self
            .d
            .attr(e, t::ALT)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
        {
            self.set_attr(p, "alt", &a);
        }
        if let Some(a) = self
            .d
            .attr(e, t::TITLE)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
        {
            self.set_attr(p, "title", &a);
        }
        let tl = self.d.tail(e).map(str::to_owned);
        self.set_tail(p, tl);
        Some(p)
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
        } else if INLINE_CONSUMING.contains(&g) {
            self.handle_formatting(e)
        } else if g == t::TABLE && potential.contains(t::TABLE) {
            self.handle_table(e, potential)
        } else if g == t::GRAPHIC && potential.contains(t::GRAPHIC) {
            self.handle_image(e)
        } else {
            self.handle_other_elements(e, potential)
        }
    }

    fn recover_wild_text(
        &mut self,
        tree: NodeId,
        result_body: NodeId,
        potential: &TagSet,
        consumed: &HashSet<NodeId>,
    ) {
        let mut potential = potential.clone();
        let recall = self.focus == Focus::Recall;
        if recall {
            potential.insert(t::DIV);
            potential.insert(t::LB);
        }
        let search_tree = self.prune_unwanted_sections(tree, &potential, self.o.fast);
        let subelems = xpaths::wild_text_search(&self.d, search_tree, recall);
        let elem_texts: Vec<String> = self
            .d
            .children_vec(result_body)
            .into_iter()
            .map(|c| self.elem_text(c))
            .collect();
        let mut existing = elem_texts
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        let mut existing_len = py_len(&existing);
        let mut existing_elems: HashSet<String> = elem_texts.into_iter().collect();
        let mut handled: HashSet<NodeId> = HashSet::new();
        for sub in subelems {
            if (!consumed.is_empty() && consumed.contains(&sub))
                || self.d.ancestors(sub).any(|a| handled.contains(&a))
            {
                continue;
            }
            let Some(processed) = self.handle_textelem(sub, &potential) else {
                continue;
            };
            if processed == sub {
                handled.insert(sub);
            }
            let text = self.elem_text(processed);
            let images: HashSet<Option<String>> = self
                .d
                .iter_vec(processed, &Matcher::tags(&[t::GRAPHIC]), true)
                .into_iter()
                .map(|g| self.d.attr(g, t::SRC).map(str::to_owned))
                .collect();
            if text.is_empty() && !images.is_empty() {
                let body_images: HashSet<Option<String>> = self
                    .d
                    .iter_vec(result_body, &Matcher::tags(&[t::GRAPHIC]), true)
                    .into_iter()
                    .map(|g| self.d.attr(g, t::SRC).map(str::to_owned))
                    .collect();
                if images.is_subset(&body_images) {
                    continue;
                }
            }
            let under_cap = existing_len <= self.p.dedupe_scan_cap;
            let fragment = if text_chars_test(self.d.tail(processed)) {
                trim(&format!("{text}{}", self.d.tail(processed).unwrap_or("")))
            } else {
                String::new()
            };
            if !text.is_empty()
                && (existing_elems.contains(&text)
                    || (under_cap
                        && ((py_len(&text) > self.p.min_duplicate_length
                            && existing.contains(&text))
                            || (!fragment.is_empty() && existing.contains(&fragment)))))
            {
                continue;
            }
            self.append(result_body, processed);
            if under_cap {
                let add = if fragment.is_empty() {
                    &text
                } else {
                    &fragment
                };
                existing.push('\n');
                existing.push_str(add);
                existing_len += 1 + py_len(add);
            }
            existing_elems.insert(text);
        }
    }

    /// main_extractor.prune_unwanted_sections.
    pub fn prune_unwanted_sections(
        &mut self,
        tree: NodeId,
        potential: &TagSet,
        keep_teasers: bool,
    ) -> NodeId {
        let favor_precision = self.focus == Focus::Precision;
        self.prune_unwanted_nodes(tree, self.p.overall_discard, true);
        if !potential.contains(t::GRAPHIC) {
            self.prune_unwanted_nodes(tree, self.p.discard_image_elements, false);
        }
        if self.focus != Focus::Recall {
            if !keep_teasers {
                self.prune_unwanted_nodes(tree, self.p.teaser_discard, false);
            }
            if favor_precision {
                self.prune_unwanted_nodes(tree, self.p.precision_discard, false);
            }
        }
        for _ in 0..2 {
            self.delete_by_link_density(tree, t::DIV, true, favor_precision);
            self.delete_by_link_density(tree, t::LIST, false, favor_precision);
            self.delete_by_link_density(tree, t::P, false, favor_precision);
        }
        if potential.contains(t::TABLE) || favor_precision {
            let tables: Vec<NodeId> = self
                .d
                .iter_vec(tree, &Matcher::tags(&[t::TABLE]), true)
                .into_iter()
                .filter(|&e| self.link_density_test_tables(e))
                .collect();
            for e in tables {
                self.delete_element(e, false);
            }
        }
        if favor_precision {
            while let Some(last) = self.d.last_child(tree) {
                if self.d.tag(last) != t::HEAD {
                    break;
                }
                self.delete_element(last, false);
            }
            self.delete_by_link_density(tree, t::HEAD, false, true);
            self.delete_by_link_density(tree, t::QUOTE, false, true);
        }
        let mut strip = vec![t::SPAN];
        if !potential.contains(t::REF) {
            strip.push(t::REF);
        }
        self.d.strip_tags(tree, &Matcher::tags(&strip));
        tree
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
            let subtree = self.prune_unwanted_sections(subtree, &potential, false);
            if self.d.len(subtree) == 0 {
                continue;
            }
            // subtree.xpath("//p//text()"): the document of `tree`, even if the subtree was detached
            let (any_p, plen) = xpaths::p_text(&self.d, tree);
            let factor = if self.focus == Focus::Precision { 1 } else { 3 };
            if !any_p || plen < self.o.min_extracted_size * factor {
                potential.insert(t::DIV);
            }
            let mut subelems = xpaths::select(&self.d, subtree, |_, _| true);
            let tags: HashSet<Name> = subelems.iter().map(|&e| self.d.tag(e)).collect();
            let inline_lb = |g: &Name| INLINE_FORMATTABLE.contains(g) || *g == t::LB;
            if (tags.len() == 1 && tags.contains(&t::LB))
                || (self.d.tag(subtree) == t::DIV
                    && potential.contains(t::DIV)
                    && tags.iter().all(inline_lb))
            {
                subelems = vec![subtree];
            }
            let mut lead = None;
            let start = self.d.len(result_body);
            if subelems != [subtree] && text_chars_test(self.d.text(subtree)) {
                let l = self.d.element(t::P);
                let tx = self.d.text(subtree).map(str::to_owned);
                self.set_text(l, tx);
                lead = self.process_node(l);
            }
            for &e in &subelems {
                if self.d.top(e) == result_body {
                    continue;
                }
                let (g, tail) = (self.d.tag(e), self.d.tail(e).map(str::to_owned));
                let processed = self.handle_textelem(e, &potential);
                if let Some(p) = processed {
                    self.append(result_body, p);
                    if self.d.top(e) != result_body {
                        self.d.set_tag(e, t::DONE);
                    }
                }
                if REBUILT_BLOCKS.contains(&g)
                    && text_chars_test(tail.as_deref())
                    && processed.is_none_or(|p| self.d.tail(p).is_none())
                {
                    let te = self.d.element(t::P);
                    self.set_text(te, tail);
                    if self.process_node(te).is_some() {
                        self.append(result_body, te);
                    }
                }
            }
            while let Some(last) = self.d.last_child(result_body) {
                if !NOT_AT_THE_END.contains(&self.d.tag(last)) {
                    break;
                }
                self.delete_element(last, false);
            }
            if self
                .d
                .children(result_body)
                .filter(|&c| self.d.tag(c) != t::GRAPHIC)
                .count()
                > 1
            {
                if let Some(l) = lead {
                    self.insert(result_body, isize::try_from(start).unwrap_or(isize::MAX), l);
                }
                break;
            }
        }
        let temp_text = pystr::strip(&self.d.itertext_join(result_body, " ")).to_owned();
        (result_body, temp_text, potential)
    }

    /// main_extractor.extract_content.
    pub fn extract_content(&mut self, cleaned: NodeId) -> (NodeId, String) {
        let backup = self.d.deep_copy(cleaned);
        let order = self.d.iter_vec(cleaned, &Matcher::AnyNode, true);
        let (result_body, mut temp_text, potential) = self.extract_main(cleaned);
        if self.d.len(result_body) == 0 || py_len(&temp_text) < self.o.min_extracted_size {
            // copies of what the main pass consumed (zip with the backup, both in source order)
            let twins = self.d.iter_vec(backup, &Matcher::AnyNode, true);
            let consumed: HashSet<NodeId> = order
                .iter()
                .zip(&twins)
                .filter(|&(&e, _)| self.d.tag(e) == t::DONE && self.d.is_element(e))
                .map(|(_, &w)| w)
                .collect();
            self.recover_wild_text(backup, result_body, &potential, &consumed);
            pystr::strip(&self.d.itertext_join(result_body, " ")).clone_into(&mut temp_text);
        }
        let mut previous: Option<String> = None;
        for el in self.d.children_vec(result_body) {
            let current = self.elem_text(el);
            if !current.is_empty()
                && Some(&current) == previous.as_ref()
                && py_len(&current) > self.p.min_duplicate_length
            {
                self.delete_element(el, false);
            } else {
                previous = Some(current);
            }
        }
        self.d
            .strip_elements(result_body, &Matcher::tags(&[t::DONE]));
        self.d.strip_tags(result_body, &Matcher::tags(&[t::DIV]));
        for lb in self.d.iter_vec(result_body, &Matcher::tags(&[t::LB]), true) {
            if truthy(self.d.tail(lb))
                && !text_chars_test(self.d.tail(lb))
                && !self
                    .d
                    .ancestors(lb)
                    .any(|a| SPACING_PROTECTED.contains(&self.d.tag(a)))
            {
                self.set_tail(lb, None);
            }
        }
        (result_body, temp_text)
    }

    /// main_extractor.extract_comments: (comments body, its text); the comments are removed
    /// from `tree`.
    pub fn extract_comments(&mut self, tree: NodeId) -> (NodeId, String) {
        let comments_body = self.d.element(t::BODY);
        let potential = TagSet::of(self.p.tag_catalog);
        for expr in self.p.comments_xpath {
            let Some(subtree) = expr(&self.d, tree) else {
                continue;
            };
            self.prune_unwanted_nodes(subtree, self.p.comments_discard, false);
            self.d
                .strip_tags(subtree, &Matcher::tags(&[t::REF, t::SPAN]));
            for e in xpaths::select(&self.d, subtree, |_, _| true) {
                if potential.contains(self.d.tag(e)) {
                    if let Some(processed) = self.handle_textnode(e, true, false) {
                        self.d.attrib_clear(processed);
                        self.append(comments_body, processed);
                    }
                }
            }
            if self.d.len(comments_body) > 0 {
                self.delete_element(subtree, false);
                break;
            }
        }
        let text = pystr::strip(&self.d.itertext_join(comments_body, " ")).to_owned();
        (comments_body, text)
    }
}
