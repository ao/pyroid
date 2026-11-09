// Derived from trafilatura 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
//! trafilatura 1.11.0, as `datatrove[processing]==0.10.1` installs it: core.bare_extraction,
//! htmlprocessing.py and the parts of main_extractor.py, external.py, readability_lxml.py,
//! baseline.py and xml.py that differ from 2.3.1.
//!
//! [`V11`] wraps the extraction state; its methods shadow the 2.3.1 methods of the same name
//! (through `Deref` to [`Ctx`]), and the helpers that are the same in both releases
//! (delete_element, textfilter, duplicate_test, the assignment checks) are shared. The lists
//! and XPaths come from the release's [`super::profile::Profile`]; the control flow (no
//! cascade, the readability retry, the baseline tiers, exceptions that escape) is this code.
//!
//! Not reproduced: a `"|" * n` for a row span past 2**26 characters (or an output past 2**28
//! bytes) is reported as MemoryError, where CPython may still build it; an attribute named
//! `{uri}local` copied into a list item stays a plain name (the bare_extraction XML would show
//! lxml's namespace prefix); colspans summing to over 4300 digits (`str()` raises) saturate;
//! characters U+F0000-U+F07FF in JSON-LD strings count as the lone surrogates they stand for.

mod baseline;
mod external;
mod main_ext;
mod readability;
mod xmlout;
pub(crate) mod xpaths;

use std::ops::{Deref, DerefMut};

pub use xmlout::xmltotxt;

use super::pystr::{PyError, py_len, trim};
use super::xpaths::XList;
use super::{Bodies, Ctx, Focus};
use crate::html::tree::{Iter, Matcher, Name, NodeId, TagSet, names as t};

/// The extraction state, running trafilatura 1.11.0's code.
pub(crate) struct V11<'c, 'a> {
    cx: &'c mut Ctx<'a>,
}

impl<'a> Deref for V11<'_, 'a> {
    type Target = Ctx<'a>;
    fn deref(&self) -> &Ctx<'a> {
        self.cx
    }
}

impl DerefMut for V11<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.cx
    }
}

/// htmlprocessing.REND_TAG_MAPPING keys.
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
/// htmlprocessing.CONVERSIONS keys.
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

/// `a + " " + b` style concatenation where Python raises TypeError on None.
pub(crate) fn concat(cx: &mut Ctx<'_>, a: Option<&str>, b: Option<&str>) -> String {
    if let (Some(a), Some(b)) = (a, b) {
        format!("{a}{b}")
    } else {
        cx.raise();
        String::new()
    }
}

impl<'c, 'a> V11<'c, 'a> {
    pub fn new(cx: &'c mut Ctx<'a>) -> Self {
        V11 { cx }
    }

    /// core.bare_extraction from the loaded tree on, up to the output conversion: None where
    /// it returns None (a TypeError or ValueError caught), Err for the exception that escapes.
    pub fn bare(&mut self, tree: NodeId) -> Result<Option<Bodies>, PyError> {
        let tree_backup = self.d.deep_copy(tree);
        // tree_cleaning works on the loaded tree itself
        let cleaned = self.tree_cleaning(tree);
        let cleaned_backup = self.d.deep_copy(cleaned);
        let mut cleaned = self.convert_tags(cleaned);
        let (commentsbody, len_comments) = if self.o.comments {
            let (cb, tc) = self.extract_comments(cleaned);
            (Some(cb), py_len(&tc))
        } else {
            (None, 0)
        };
        if self.focus == Focus::Precision {
            cleaned = self.prune_unwanted_nodes(cleaned, xpaths::REMOVE_COMMENTS_XPATH, false);
        }
        let (mut postbody, temp_text, mut len_text) = self.extract_content(cleaned);
        if !self.o.fast && !self.raised {
            let raw = self.d.deep_copy(tree_backup);
            (postbody, _, len_text) =
                self.compare_extraction(cleaned_backup, raw, postbody, temp_text, len_text);
        }
        if len_text < self.o.min_extracted_size && !self.raised {
            let c = self.d.deep_copy(tree_backup);
            (postbody, _, len_text) = self.baseline(c)?;
        }
        if self.raised {
            return Ok(None);
        }
        if let Some(max) = self.o.max_tree_size.filter(|&m| m > 0) {
            if self.d.len(postbody) > max {
                self.d.strip_tags(postbody, &Matcher::tags(&[t::HI]));
            }
            if self.d.len(postbody) > max {
                return Ok(None);
            }
        }
        if len_text < self.o.min_output_size && len_comments < self.o.min_output_comm_size {
            return Ok(None);
        }
        if self.dedup() && self.duplicate_test(postbody) {
            return Ok(None);
        }
        Ok(Some(Bodies {
            body: postbody,
            commentsbody,
        }))
    }

    /// htmlprocessing.tree_cleaning (on `tree` itself; a copy in recall mode when the deletions
    /// would remove every paragraph).
    fn tree_cleaning(&mut self, tree: NodeId) -> NodeId {
        let mut cleaning: Vec<Name> = self.p.manually_cleaned.to_vec();
        if self.o.tables {
            // `.//figure[descendant::table]` -> div
            let figures = super::xpaths::select(&self.d, tree, |d, e| {
                d.tag(e) == t::FIGURE
                    && super::xpaths::select_first(d, e, |d, c| d.tag(c) == t::TABLE).is_some()
            });
            for f in figures {
                self.d.set_tag(f, t::DIV);
            }
        } else {
            cleaning.extend([t::TABLE, t::TD, t::TH, t::TR]);
        }
        let stripped = Matcher::tags(self.p.manually_stripped);
        self.d.strip_tags(tree, &stripped);
        let mut tree = tree;
        let tcopy = if self.focus == Focus::Recall && self.find_desc(tree, t::P).is_some() {
            Some(self.d.deep_copy(tree))
        } else {
            None
        };
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
        self.prune_html(tree);
        tree
    }

    /// The top-level elements of the document holding `x`, in document order: lxml's `//`
    /// searches all of them, and libxml2's HTML parser puts what follows `</html>` in a second
    /// root element. (Copies and removed subtrees have no top-level siblings here, as lxml's
    /// copies have none.)
    pub(crate) fn doc_roots(&self, x: NodeId) -> Vec<NodeId> {
        let mut first = self.d.top(x);
        while let Some(p) = self.d.prev(first) {
            first = p;
        }
        let mut out = vec![first];
        while let Some(n) = self.d.next(out[out.len() - 1]) {
            out.push(n);
        }
        out
    }

    /// `//*` matching `m` in the document of `x`, in document order.
    pub(crate) fn doc_iter(&self, x: NodeId, m: &Matcher) -> Vec<NodeId> {
        let mut out = Vec::new();
        for r in self.doc_roots(x) {
            out.extend(self.d.iter_vec(r, m, true));
        }
        out
    }

    /// htmlprocessing.prune_html: `//processing-instruction()|//*[not(node())]` over the whole
    /// document, deleting the CUT_EMPTY_ELEMS among them.
    fn prune_html(&mut self, tree: NodeId) {
        let cut = TagSet::of(self.p.cut_empty_elems);
        let empties: Vec<NodeId> = self
            .doc_iter(tree, &Matcher::AnyElement)
            .into_iter()
            .filter(|&e| {
                self.d.text(e).is_none() && self.d.len(e) == 0 && cut.contains(self.d.tag(e))
            })
            .collect();
        for e in empties {
            self.delete_element(e, true);
        }
    }

    /// htmlprocessing.prune_unwanted_nodes (1.11.0): each tail goes to the previous sibling's
    /// tail, or the parent's tail, after a space; with a backup, the tree or (if more than
    /// six sevenths of the text went) a copy taken before.
    pub fn prune_unwanted_nodes(
        &mut self,
        tree: NodeId,
        nodelist: XList,
        with_backup: bool,
    ) -> NodeId {
        let backup = with_backup.then(|| (self.text_content_len(tree), self.d.deep_copy(tree)));
        for expr in nodelist {
            for sub in expr(&self.d, tree) {
                if let Some(tail) = self.d.tail(sub).map(str::to_owned) {
                    let prev = self.d.prev(sub).or_else(|| self.d.parent(sub));
                    if let Some(p) = prev {
                        let v = format!("{} {tail}", self.d.tail(p).unwrap_or(""));
                        self.set_tail(p, Some(v));
                    }
                }
                self.d.unlink(sub);
            }
        }
        match backup {
            Some((old_len, copy)) if 7 * self.text_content_len(tree) <= old_len => copy,
            _ => tree,
        }
    }

    /// htmlprocessing.collect_link_info: (total length, count, short ones).
    fn collect_link_info(&self, links: &[NodeId], favor_precision: bool) -> (usize, usize, usize) {
        let threshold = if favor_precision { 50 } else { 10 };
        let (mut total, mut n, mut short) = (0, 0, 0);
        for &l in links {
            let s = trim(&self.d.text_content(l));
            if s.is_empty() {
                continue;
            }
            let len = py_len(&s);
            total += len;
            n += 1;
            if len < threshold {
                short += 1;
            }
        }
        (total, n, short)
    }

    /// htmlprocessing.link_density_test: (boilerplate, whether the link texts list is not
    /// empty).
    fn link_density_test(&self, e: NodeId, text: &str, favor_precision: bool) -> (bool, bool) {
        let links = self.findall_desc(e, t::REF);
        if links.is_empty() {
            return (false, false);
        }
        let last = self.d.next(e).is_none();
        let limitlen = if self.d.tag(e) == t::P {
            if favor_precision {
                200
            } else if last {
                60
            } else {
                30
            }
        } else if last {
            300
        } else {
            100
        };
        let elemlen = py_len(text);
        if elemlen < limitlen {
            let (linklen, elemnum, shortelems) = self.collect_link_info(&links, favor_precision);
            if elemnum == 0 {
                return (true, false);
            }
            let r = (linklen as f64) > 0.8 * (elemlen as f64)
                || (elemnum > 1 && (shortelems as f64) / (elemnum as f64) > 0.8);
            return (r, true);
        }
        (false, false)
    }

    /// htmlprocessing.link_density_test_tables.
    pub fn link_density_test_tables(&self, e: NodeId) -> bool {
        let links = self.findall_desc(e, t::REF);
        if links.is_empty() {
            return false;
        }
        let elemlen = py_len(&trim(&self.d.text_content(e)));
        if elemlen <= 250 {
            return false;
        }
        let (linklen, elemnum, _) = self.collect_link_info(&links, false);
        if elemnum == 0 {
            return true;
        }
        if elemlen < 1000 {
            (linklen as f64) > 0.8 * elemlen as f64
        } else {
            (linklen as f64) > 0.5 * elemlen as f64
        }
    }

    /// htmlprocessing.delete_by_link_density; with backtracking, the children of short
    /// link-rich elements go.
    pub fn delete_by_link_density(
        &mut self,
        subtree: NodeId,
        tag: Name,
        backtracking: bool,
        favor_precision: bool,
    ) {
        let threshold = if favor_precision { 200 } else { 100 };
        let mut deletions: Vec<NodeId> = Vec::new();
        for e in self.d.iter_vec(subtree, &Matcher::tags(&[tag]), true) {
            let text = trim(&self.d.text_content(e));
            let (result, templist) = self.link_density_test(e, &text, favor_precision);
            if result {
                deletions.push(e);
            } else if backtracking && templist {
                let n = py_len(&text);
                if 0 < n && n < threshold && self.d.len(e) >= 3 {
                    deletions.extend(self.d.children(e));
                }
            }
        }
        let mut seen = std::collections::HashSet::new();
        for e in deletions {
            if seen.insert(e) {
                self.delete_element(e, true);
            }
        }
    }

    /// htmlprocessing.handle_textnode (1.11.0).
    pub fn handle_textnode(
        &mut self,
        e: NodeId,
        comments_fix: bool,
        preserve_spaces: bool,
    ) -> Option<NodeId> {
        let truthy = |s: Option<&str>| s.is_some_and(|s| !s.is_empty());
        if self.d.tag(e) == t::DONE
            || (self.d.len(e) == 0 && !truthy(self.d.text(e)) && !truthy(self.d.tail(e)))
        {
            return None;
        }
        if !comments_fix && self.d.tag(e) == t::LB {
            if !preserve_spaces {
                let v = self.d.tail(e).map(trim);
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
            let v = self.d.text(e).map(trim);
            self.set_text(e, v);
            if truthy(self.d.tail(e)) {
                let v = self.d.tail(e).map(trim);
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

    /// htmlprocessing.process_node (1.11.0).
    pub fn process_node(&mut self, e: NodeId) -> Option<NodeId> {
        let truthy = |s: Option<&str>| s.is_some_and(|s| !s.is_empty());
        if self.d.tag(e) == t::DONE
            || (self.d.len(e) == 0 && !truthy(self.d.text(e)) && !truthy(self.d.tail(e)))
        {
            return None;
        }
        let (tx, tl) = (self.d.text(e).map(trim), self.d.tail(e).map(trim));
        self.set_text(e, tx);
        self.set_tail(e, tl);
        if self.d.tag(e) != t::LB && !truthy(self.d.text(e)) && truthy(self.d.tail(e)) {
            let tail = self.d.tail(e).map(str::to_owned);
            self.set_text(e, tail);
            self.set_tail(e, None);
        }
        if (truthy(self.d.text(e)) || truthy(self.d.tail(e)))
            && (self.textfilter(e) || (self.dedup() && self.duplicate_test(e)))
        {
            return None;
        }
        Some(e)
    }

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

    /// htmlprocessing.convert_quotes (1.11.0): a pre is code when it wraps a single span or
    /// holds highlight.js spans.
    fn convert_quotes(&mut self, e: NodeId) {
        let mut code = false;
        if self.d.tag(e) == t::PRE {
            if self.d.len(e) == 1
                && self
                    .d
                    .first_child(e)
                    .is_some_and(|c| self.d.is_element(c) && self.d.tag(c) == t::SPAN)
            {
                code = true;
            }
            let hljs = super::xpaths::hljs_spans(&self.d, e);
            if !hljs.is_empty() {
                code = true;
                for s in hljs {
                    self.d.attrib_clear(s);
                }
            }
        }
        self.d.set_tag(e, if code { t::CODE } else { t::QUOTE });
    }

    /// htmlprocessing.convert_tags (links, images and formatting off).
    pub fn convert_tags(&mut self, tree: NodeId) -> NodeId {
        // `.//div//a|.//ul//a` (and `|.//table//a`): links below a container strictly inside
        // the tree become ref
        let containers = if self.o.tables {
            TagSet::of(&[t::DIV, t::UL, t::TABLE])
        } else {
            TagSet::of(&[t::DIV, t::UL])
        };
        let links: Vec<NodeId> = super::xpaths::select(&self.d, tree, |d, e| {
            d.tag(e) == t::A
                && d.ancestors(e)
                    .take_while(|&a| a != tree)
                    .any(|a| d.is_element(a) && containers.contains(d.tag(a)))
        });
        for a in links {
            self.d.set_tag(a, t::REF);
        }
        self.d.strip_tags(tree, &Matcher::tags(&[t::A]));
        self.d.strip_tags(tree, &Matcher::tags(REND_TAGS));
        let mut it = Iter::new(&self.d, tree, Matcher::tags(CONVERSION_TAGS), true);
        while let Some(e) = it.next(&self.d) {
            let g = self.d.tag(e);
            if g == t::DL || g == t::OL || g == t::UL {
                self.convert_lists(e);
            } else if [t::H1, t::H2, t::H3, t::H4, t::H5, t::H6].contains(&g) {
                self.d.attrib_clear(e);
                let name = self.d.tag_name(e).to_owned();
                self.set_attr(e, "rend", &name);
                self.d.set_tag(e, t::HEAD);
            } else if g == t::BR || g == t::HR {
                self.d.set_tag(e, t::LB);
            } else if g == t::BLOCKQUOTE || g == t::PRE || g == t::Q {
                self.convert_quotes(e);
            } else if g == t::DEL || g == t::S || g == t::STRIKE {
                self.d.set_tag(e, t::DEL);
                self.set_attr(e, "rend", "overstrike");
            } else if g == t::DETAILS {
                self.d.set_tag(e, t::DIV);
                let mut sub = Iter::new(&self.d, e, Matcher::tags(&[t::SUMMARY]), true);
                while let Some(s) = sub.next(&self.d) {
                    self.d.set_tag(s, t::HEAD);
                }
            }
        }
        tree
    }

    /// baseline.basic_cleaning (1.11.0's BASIC_CLEAN_XPATH).
    fn basic_cleaning(&mut self, tree: NodeId) {
        for e in xpaths::basic_clean(&self.d, tree) {
            self.delete_element(e, true);
        }
    }
}
