// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! An exact port of Trafilatura's text extraction (`trafilatura.extract` with `txt` output and
//! `trafilatura.bare_extraction`), for Trafilatura 2.3.1 with lxml 6.1.3 (libxml2 2.14.6) and
//! jusText 3.0.2: the same text, byte for byte, for the options it supports.
//!
//! ```
//! use pyroid_core::trafilatura::{extract, LruCache, Options};
//!
//! let html = "<html><body><article><p>Trafilatura keeps the main text of a page and drops \
//!             the navigation, the footer and the other boilerplate around it, \
//!             which is what this paragraph is.</p></article></body></html>";
//! // trafilatura.extract(html)
//! let text = extract(html.as_bytes(), &Options::default(), None).unwrap();
//! assert!(text.unwrap().starts_with("Trafilatura keeps"));
//!
//! // datatrove's call: extract(html, favor_precision=True, include_comments=False,
//! // deduplicate=True), with the cache Trafilatura keeps per process
//! let mut cache = LruCache::default();
//! let _ = extract(html.as_bytes(), &Options::datatrove(), Some(&mut cache)).unwrap();
//! ```
//!
//! Supported: `fast`, `favor_precision` / `favor_recall` ([`Focus`]), `include_comments`,
//! `include_tables`, `deduplicate` (pass the cache), the size settings of settings.cfg, and the
//! caller's Unicode version (CPython 3.11-3.15). Not supported (the Python layer hands these to
//! Trafilatura): `include_formatting`, `include_links`, `include_images`, `url`,
//! `target_language`, `with_metadata`, `prune_xpath`, output formats other than `txt` / python.
//!
//! Input is the HTML as a Python `str`, encoded with `"surrogatepass"` (UTF-8, lone surrogates
//! as three bytes each). The module keeps no state: the dedup cache is the caller's.

// Python compares lengths as floats (`len(a) < 0.2 * len(b)`): exact below 2**52, which every
// string here is. Functions follow their Python counterparts statement by statement.
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]

mod baseline;
mod dedup;
mod external;
mod htmlproc;
mod justext;
mod load;
mod main_ext;
mod matchers;
mod profile;
mod pyjson;
mod pystr;
mod readability;
#[rustfmt::skip]
mod tables;
mod xmlout;
mod xpaths;

pub use dedup::{LRU_SIZE, LruCache};
pub use load::{load_html, repair_faulty_html};
pub use pystr::PyError;

use crate::ftfy::UnicodeVersion;
use crate::html::tree::{Document, Matcher, NodeId, names as t};
use profile::Profile;

/// The Trafilatura release whose behaviour to reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Version {
    /// trafilatura 2.3.1 (with lxml 6.1.3 / libxml2 2.14.6 and jusText 3.0.2).
    #[default]
    V2_3_1,
}

/// `favor_precision` / `favor_recall` (settings.Extractor.focus).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Focus {
    /// Neither flag (the default).
    #[default]
    Balanced,
    /// `favor_precision=True`.
    Precision,
    /// `favor_recall=True` (it wins when both flags are set, as in Trafilatura).
    Recall,
}

/// The extraction options this port supports, with Trafilatura's defaults
/// (`trafilatura.extract(html)`) from [`Options::default`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::struct_excessive_bools)] // mirrors settings.Extractor
pub struct Options {
    pub version: Version,
    pub focus: Focus,
    /// `include_comments` (default true).
    pub comments: bool,
    /// `fast`: no readability / jusText comparison (default false).
    pub fast: bool,
    /// `include_tables` (default true).
    pub tables: bool,
    /// The `unicodedata` version of the caller's CPython.
    pub unicode: UnicodeVersion,
    /// settings.cfg `MIN_EXTRACTED_SIZE` (250).
    pub min_extracted_size: usize,
    /// settings.cfg `MIN_OUTPUT_SIZE` (1).
    pub min_output_size: usize,
    /// settings.cfg `MIN_OUTPUT_COMM_SIZE` (1).
    pub min_output_comm_size: usize,
    /// settings.cfg `MIN_DUPLCHECK_SIZE` (100).
    pub min_duplcheck_size: usize,
    /// settings.cfg `MAX_REPETITIONS` (2).
    pub max_repetitions: i64,
    /// settings.cfg `MAX_TREE_SIZE` (empty: no limit).
    pub max_tree_size: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            version: Version::V2_3_1,
            focus: Focus::Balanced,
            comments: true,
            fast: false,
            tables: true,
            unicode: UnicodeVersion::LATEST,
            min_extracted_size: 250,
            min_output_size: 1,
            min_output_comm_size: 1,
            min_duplcheck_size: 100,
            max_repetitions: 2,
            max_tree_size: None,
        }
    }
}

impl Options {
    /// datatrove 0.10.1's `Trafilatura(favour_precision=True)` step:
    /// `extract(html, favor_precision=True, include_comments=False, deduplicate=True)` (pass
    /// the cache to [`extract`]).
    pub fn datatrove() -> Self {
        Options {
            focus: Focus::Precision,
            comments: false,
            ..Options::default()
        }
    }
}

/// What `trafilatura.bare_extraction` returns (output_format "python") for the fields this
/// port computes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Extraction {
    /// `document.text`: `xmltotxt(body)` (not NFC-normalized, unlike `extract`).
    pub text: String,
    /// `document.raw_text` (the same string as `text` for the python format).
    pub raw_text: String,
    /// `document.comments` (`None` unless `include_comments`).
    pub comments: Option<String>,
    /// `lxml.etree.tostring(document.body, encoding="unicode")`.
    pub body: String,
    /// `lxml.etree.tostring(document.commentsbody, encoding="unicode")`.
    pub commentsbody: String,
}

/// `trafilatura.extract(html, ...)` with `output_format="txt"`: the text, or `None` where
/// Trafilatura returns None. `Err` where Trafilatura raises (a `ValueError` from `int()` of
/// more than 4300 digits in `html.unescape` of the output).
pub fn extract(
    html: &[u8],
    opts: &Options,
    dedup: Option<&mut LruCache>,
) -> Result<Option<String>, PyError> {
    let Some(doc) = load_html(html) else {
        return Ok(None);
    };
    extract_tree(doc, opts, dedup)
}

/// [`extract`] on a tree from [`load_html`] (`trafilatura.extract(tree)`).
pub fn extract_tree(
    doc: Document,
    opts: &Options,
    dedup: Option<&mut LruCache>,
) -> Result<Option<String>, PyError> {
    if (doc.root as usize) >= doc.nodes.len() {
        return Ok(None);
    }
    let mut cx = Ctx::new(doc, opts, dedup);
    let root = cx.d.root;
    let Some(r) = cx.bare(root) else {
        return Ok(None);
    };
    // determine_returnstring: f"{xmltotxt(body)}\n{xmltotxt(commentsbody)}".strip(), NFC
    let body = xmlout::xmltotxt(&cx.d, r.body, cx.v)?;
    let comments = xmlout::xmltotxt(&cx.d, r.commentsbody, cx.v)?;
    let s = format!("{body}\n{comments}");
    Ok(Some(pystr::nfc(pystr::strip(&s), cx.v).into_owned()))
}

/// `trafilatura.bare_extraction(html, ...)` (output_format "python"), or `None`.
pub fn bare_extraction(
    html: &[u8],
    opts: &Options,
    dedup: Option<&mut LruCache>,
) -> Result<Option<Extraction>, PyError> {
    let Some(doc) = load_html(html) else {
        return Ok(None);
    };
    bare_extraction_tree(doc, opts, dedup)
}

/// [`bare_extraction`] on a tree from [`load_html`].
pub fn bare_extraction_tree(
    doc: Document,
    opts: &Options,
    dedup: Option<&mut LruCache>,
) -> Result<Option<Extraction>, PyError> {
    if (doc.root as usize) >= doc.nodes.len() {
        return Ok(None);
    }
    let mut cx = Ctx::new(doc, opts, dedup);
    let root = cx.d.root;
    let Some(r) = cx.bare(root) else {
        return Ok(None);
    };
    let text = xmlout::xmltotxt(&cx.d, r.body, cx.v)?;
    let (comments, commentsbody) = if opts.comments {
        (
            Some(xmlout::xmltotxt(&cx.d, r.commentsbody, cx.v)?),
            r.commentsbody,
        )
    } else {
        // Document() default: an empty <body/>
        (None, cx.d.element(t::BODY))
    };
    Ok(Some(Extraction {
        raw_text: text.clone(),
        text,
        comments,
        body: xmlout::tostring_xml(&cx.d, r.body),
        commentsbody: xmlout::tostring_xml(&cx.d, commentsbody),
    }))
}

/// The bodies trafilatura_sequence returns.
struct Bodies {
    body: NodeId,
    commentsbody: NodeId,
}

/// Extraction state for one document: the arena holding every tree involved, the options, the
/// cache, and the Python exception state.
pub(crate) struct Ctx<'a> {
    pub d: Document,
    pub o: &'a Options,
    pub p: &'static Profile,
    /// `options.focus`, which the recall retry overrides.
    pub focus: Focus,
    pub v: UnicodeVersion,
    lru: Option<&'a mut LruCache>,
    /// The input holds strings lxml would refuse to assign (so assignments are checked).
    strict: bool,
    /// A Python exception is propagating: the code since the raise is discarded at the next
    /// `try` (or by bare_extraction), and touches no shared state (the cache) meanwhile.
    pub raised: bool,
}

impl<'a> Ctx<'a> {
    fn new(d: Document, o: &'a Options, lru: Option<&'a mut LruCache>) -> Self {
        let strict = !d.nodes.iter().all(|n| {
            n.text.as_deref().is_none_or(pystr::xml_valid)
                && n.tail.as_deref().is_none_or(pystr::xml_valid)
                && n.attrs.iter().all(|a| pystr::xml_valid(&a.value))
        });
        Ctx {
            p: profile::get(o.version),
            focus: o.focus,
            v: o.unicode,
            d,
            o,
            lru,
            strict,
            raised: false,
        }
    }

    /// `options.dedup`.
    pub fn dedup(&self) -> bool {
        self.lru.is_some()
    }

    /// `LRU.increment(key)`, or -1 (and no change) while an exception propagates.
    pub fn lru_increment(&mut self, key: &str) -> i64 {
        if self.raised {
            return -1;
        }
        self.lru.as_mut().map_or(-1, |c| c.increment(key))
    }

    /// Raise a Python exception (see [`Ctx::raised`]).
    pub fn raise(&mut self) {
        self.raised = true;
    }

    /// Run `f` inside `try: ... except Exception:`: whether it raised (the state is reset).
    pub fn try_catch<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> Result<R, ()> {
        let outer = self.raised;
        self.raised = false;
        let r = f(self);
        let raised = self.raised;
        self.raised = outer;
        if raised { Err(()) } else { Ok(r) }
    }

    fn check(&mut self, s: Option<&str>) {
        if self.strict && s.is_some_and(|s| !pystr::xml_valid(s)) {
            self.raise();
        }
    }

    /// `e.text = v` (lxml validates the string).
    pub fn set_text(&mut self, e: NodeId, v: Option<String>) {
        self.check(v.as_deref());
        self.d.set_text(e, v);
    }

    /// `e.tail = v`.
    pub fn set_tail(&mut self, e: NodeId, v: Option<String>) {
        self.check(v.as_deref());
        self.d.set_tail(e, v);
    }

    /// `e.set(name, value)`.
    pub fn set_attr(&mut self, e: NodeId, name: &str, value: &str) {
        self.check(Some(value));
        self.d.set(e, name, value);
    }

    /// `parent.append(child)`: lxml raises ValueError for a cycle.
    pub fn append(&mut self, parent: NodeId, child: NodeId) {
        if self.d.is_ancestor_or_self(child, parent) {
            self.raise();
            return;
        }
        self.d.append(parent, child);
    }

    /// `parent.insert(index, child)`.
    pub fn insert(&mut self, parent: NodeId, index: isize, child: NodeId) {
        if self.d.is_ancestor_or_self(child, parent) {
            self.raise();
            return;
        }
        self.d.insert(parent, index, child);
    }

    /// `etree.SubElement(parent, tag)` with a tag taken from the tree: lxml.etree raises
    /// ValueError for a name that is not an XML name, or holds a colon.
    pub fn sub_element_checked(&mut self, parent: NodeId, tag: crate::html::tree::Name) -> NodeId {
        if !xml_tag_valid(self.d.name(tag)) {
            self.raise();
        }
        self.d.sub_element(parent, tag)
    }

    /// Code-point length of `text_content()`.
    pub fn text_content_len(&self, id: NodeId) -> usize {
        let mut n = 0;
        self.d.itertext(id, &mut |s| n += pystr::py_len(s));
        n
    }

    /// ElementPath `.//tag`: descendants (not self), document order.
    pub fn findall_desc(&self, e: NodeId, tag: crate::html::tree::Name) -> Vec<NodeId> {
        self.d.iter_vec(e, &Matcher::tags(&[tag]), false)
    }

    /// ElementPath `find(".//tag")`.
    pub fn find_desc(&self, e: NodeId, tag: crate::html::tree::Name) -> Option<NodeId> {
        xpaths::select_first(&self.d, e, |d, c| d.tag(c) == tag)
    }

    /// core.bare_extraction from the loaded tree on, up to the output conversion: None where
    /// Trafilatura returns None.
    fn bare(&mut self, tree: NodeId) -> Option<Bodies> {
        let (postbody, temp_text, commentsbody, temp_comments) = self.trafilatura_sequence(tree);
        if self.raised {
            return None;
        }
        if let Some(max) = self.o.max_tree_size.filter(|&m| m > 0) {
            if self.d.len(postbody) > max {
                self.d.strip_tags(postbody, &Matcher::tags(&[t::HI]));
            }
            if self.d.len(postbody) > max {
                return None;
            }
        }
        let (len_text, len_comments) = (pystr::py_len(&temp_text), pystr::py_len(&temp_comments));
        if len_text < self.o.min_output_size && len_comments < self.o.min_output_comm_size {
            return None;
        }
        if self.dedup() && self.duplicate_test(postbody) {
            return None;
        }
        Some(Bodies {
            body: postbody,
            commentsbody,
        })
    }

    /// core._forum_thread_page: `.//script[@type="application/ld+json"]` holding a
    /// DiscussionForumPosting.
    fn forum_thread_page(&self, tree: NodeId) -> bool {
        self.findall_desc(tree, t::SCRIPT).into_iter().any(|s| {
            self.d.attr(s, t::TYPE) == Some("application/ld+json")
                && self
                    .d
                    .text(s)
                    .is_some_and(|x| !x.is_empty() && matchers::forum_posting(x))
        })
    }

    /// core._prepare_tree: `tree_cleaning(copy(tree))`, a pre-conversion backup, convert_tags.
    fn prepare_tree(&mut self, tree: NodeId) -> (NodeId, NodeId) {
        let c = self.d.deep_copy(tree);
        let cleaned = self.tree_cleaning(c);
        let backup = self.d.deep_copy(cleaned);
        let cleaned = self.convert_tags(cleaned);
        (cleaned, backup)
    }

    /// core._extract_and_compare.
    fn extract_and_compare(
        &mut self,
        cleaned: NodeId,
        backup: NodeId,
        tree: NodeId,
    ) -> (NodeId, String) {
        let (body, text) = self.extract_content(cleaned);
        if self.o.fast {
            return (body, text);
        }
        let raw = self.d.deep_copy(tree);
        self.compare_extraction(backup, raw, body, text)
    }

    /// core.trafilatura_sequence.
    fn trafilatura_sequence(&mut self, tree: NodeId) -> (NodeId, String, NodeId, String) {
        let precision = self.focus == Focus::Precision;
        let is_forum = self.forum_thread_page(tree);
        self.prune_unwanted_nodes(tree, self.p.raw_tree_prune, false);
        if !self.o.comments && (precision || !is_forum) {
            self.prune_unwanted_nodes(tree, self.p.remove_comments_and_lists, false);
        }
        let (mut cleaned, backup) = self.prepare_tree(tree);
        let mut commentsbody = self.d.element(t::BODY);
        let mut temp_comments = String::new();
        let mut forum_posts: Option<NodeId> = None;
        if self.o.comments {
            let (cb, tc) = self.extract_comments(cleaned);
            commentsbody = cb;
            temp_comments = tc;
            if !temp_comments.is_empty() && is_forum {
                forum_posts = Some(commentsbody);
                commentsbody = self.d.element(t::BODY);
                temp_comments = String::new();
                let c = self.d.deep_copy(backup);
                cleaned = self.convert_tags(c);
            }
        }
        if precision && !is_forum {
            self.prune_unwanted_nodes(cleaned, self.p.remove_comments, false);
        }
        let (mut postbody, mut temp_text) = self.extract_and_compare(cleaned, backup, tree);
        if self.raised {
            return (postbody, temp_text, commentsbody, temp_comments);
        }

        // 3. baseline rescue on the original tree, kept only if it adds text
        if pystr::py_len(&temp_text) < self.o.min_extracted_size && !precision {
            let (b_body, b_text, b_len) = self.baseline(tree);
            if self.raised {
                return (postbody, temp_text, commentsbody, temp_comments);
            }
            if b_len > pystr::py_len(&temp_text) {
                postbody = b_body;
                temp_text = b_text;
                forum_posts = None;
            }
        }

        // 4. recall escalation
        let len_text = pystr::py_len(&temp_text);
        if self.focus == Focus::Balanced
            && 0 < len_text
            && len_text < self.p.escalation_max_length
            && (len_text as f64)
                < self.p.escalation_page_share * (pystr::py_len(&self.html2txt(tree)) as f64)
            && !self.raised
        {
            let esc_tree = if is_forum || !self.o.comments {
                tree
            } else {
                let c = self.d.deep_copy(tree);
                let list = if temp_comments.is_empty() {
                    self.p.remove_comments
                } else {
                    self.p.remove_comments_and_lists
                };
                self.prune_unwanted_nodes(c, list, false);
                c
            };
            if self.raised {
                return (postbody, temp_text, commentsbody, temp_comments);
            }
            self.focus = Focus::Recall;
            let r = self.try_catch(|cx| cx.recall_retry(esc_tree)).ok();
            self.focus = Focus::Balanced;
            let j = if self.o.fast {
                None
            } else {
                self.try_catch(|cx| {
                    let c = cx.d.deep_copy(esc_tree);
                    cx.justext_rescue(c)
                })
                .ok()
            };
            let r_len = r.as_ref().map_or(0, |(_, s)| pystr::py_len(s));
            let j_len = j.as_ref().map_or(0, |(_, s)| pystr::py_len(s));
            if j_len > r_len && (j_len as f64) > self.p.escalation_justext_ratio * (len_text as f64)
            {
                if let Some((b, s)) = j {
                    postbody = b;
                    temp_text = s;
                    forum_posts = None;
                }
            } else if r_len >= self.o.min_extracted_size
                && (r_len as f64) > self.p.escalation_accept_ratio * (len_text as f64)
            {
                if let Some((b, s)) = r {
                    postbody = b;
                    temp_text = s;
                    forum_posts = None;
                }
            }
        }
        if self.raised {
            return (postbody, temp_text, commentsbody, temp_comments);
        }

        if let Some(posts) = forum_posts {
            // salvage the captured posts missing from the body
            let existing: Vec<String> = self
                .d
                .children_vec(postbody)
                .into_iter()
                .map(|el| self.elem_text(el))
                .filter(|s| !s.is_empty())
                .collect();
            let existing = existing.join("\n");
            let salvaged: Vec<NodeId> = self
                .d
                .children_vec(posts)
                .into_iter()
                .filter(|&el| {
                    let s = self.elem_text(el);
                    !s.is_empty() && !existing.contains(&s)
                })
                .collect();
            if !salvaged.is_empty() {
                for el in salvaged {
                    self.append(postbody, el);
                }
                pystr::strip(&self.d.itertext_join(postbody, " ")).clone_into(&mut temp_text);
            }
        }
        (postbody, temp_text, commentsbody, temp_comments)
    }

    /// core._recall_retry: stages 1-2 again, in recall mode (the caller set the focus).
    fn recall_retry(&mut self, esc_tree: NodeId) -> (NodeId, String) {
        let (cleaned, backup) = self.prepare_tree(esc_tree);
        self.extract_and_compare(cleaned, backup, esc_tree)
    }
}

/// lxml `_pyXmlNameIsValid`: libxml2's xmlValidateNameValue (XML 1.0 fifth edition Name) and
/// no colon.
fn xml_tag_valid(name: &str) -> bool {
    fn start(c: char) -> bool {
        matches!(c, 'a'..='z' | 'A'..='Z' | '_' | ':' | '\u{C0}'..='\u{D6}' | '\u{D8}'..='\u{F6}'
            | '\u{F8}'..='\u{2FF}' | '\u{370}'..='\u{37D}' | '\u{37F}'..='\u{1FFF}' | '\u{200C}'..='\u{200D}'
            | '\u{2070}'..='\u{218F}' | '\u{2C00}'..='\u{2FEF}' | '\u{3001}'..='\u{D7FF}'
            | '\u{F900}'..='\u{FDCF}' | '\u{FDF0}'..='\u{FFFD}' | '\u{10000}'..='\u{EFFFF}')
    }
    fn rest(c: char) -> bool {
        start(c)
            || matches!(c, '0'..='9' | '-' | '.' | '\u{B7}' | '\u{300}'..='\u{36F}' | '\u{203F}'..='\u{2040}')
    }
    let mut cs = name.chars();
    cs.next().is_some_and(start) && cs.all(rest) && !name.contains(':')
}
