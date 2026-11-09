// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! baseline.py: `baseline()` (the rescue on the uncleaned tree) and `html2txt()` (the page
//! measure of the recall escalation), on elements (both copy their input).

use super::pyjson::{self, Value};
use super::pystr::{self, py_len, remove_control_characters, trim};
use super::{Ctx, matchers, xpaths};
use crate::html::tree::{Matcher, Name, NodeId, TagSet, names as t};

/// baseline._BLOCK_ELEMS.
const BLOCK_ELEMS: &[Name] = &[
    t::ADDRESS,
    t::ARTICLE,
    t::ASIDE,
    t::BLOCKQUOTE,
    t::BR,
    t::DD,
    t::DIV,
    t::DL,
    t::DT,
    t::FIGCAPTION,
    t::FIGURE,
    t::FOOTER,
    t::FORM,
    t::H1,
    t::H2,
    t::H3,
    t::H4,
    t::H5,
    t::H6,
    t::HEADER,
    t::HR,
    t::LI,
    t::MAIN,
    t::NAV,
    t::OL,
    t::P,
    t::PRE,
    t::SECTION,
    t::SUMMARY,
    t::TABLE,
    t::TD,
    t::TH,
    t::TR,
    t::UL,
];

/// `_walk_json` nesting at which CPython raises RecursionError (an estimate: the limit counts
/// the caller's frames too).
const WALK_JSON_MAX_DEPTH: usize = 980;

/// (body, text, length) as baseline returns them.
type Result3 = (NodeId, String, usize);

impl Ctx<'_> {
    /// baseline._spaced_text on `element` itself (callers pass a copy): block boundaries get a
    /// space, then `" ".join(text_content().split())`.
    fn spaced_text(&mut self, element: NodeId) -> String {
        for e in self.d.iter_vec(element, &Matcher::tags(BLOCK_ELEMS), true) {
            let text = match self.d.text(e).filter(|s| !s.is_empty()) {
                Some(s) => format!(" {}", remove_control_characters(s, self.v)),
                None => " ".to_owned(),
            };
            self.set_text(e, Some(text));
            let tail = match self.d.tail(e).filter(|s| !s.is_empty()) {
                Some(s) => format!(" {}", remove_control_characters(s, self.v)),
                None => " ".to_owned(),
            };
            self.set_tail(e, Some(tail));
        }
        pystr::split_ws(&self.d.text_content(element))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// baseline.block_text: `_spaced_text(copy(element))`.
    fn block_text(&mut self, e: NodeId) -> String {
        let c = self.d.deep_copy(e);
        self.spaced_text(c)
    }

    /// baseline.html2txt(tree) for an element (clean=True).
    pub fn html2txt(&mut self, tree: NodeId) -> String {
        let tree = self.d.deep_copy(tree);
        let body = self.find_desc(tree, t::BODY).unwrap_or(tree);
        self.basic_cleaning(body);
        self.spaced_text(body)
    }

    /// baseline._render_text: unescape, drop control characters, then render the markup (a
    /// fragment parse) or trim.
    fn render_text(&mut self, raw: &str) -> String {
        let Ok(raw) = pystr::html_unescape(raw).map(std::borrow::Cow::into_owned) else {
            self.raise();
            return String::new();
        };
        let raw = remove_control_characters(&raw, self.v).into_owned();
        if matchers::html_markup(&raw) {
            if let Ok(Some(s)) = self.try_catch(|cx| cx.fragment_block_text(&raw)) {
                return s;
            }
        }
        trim(&raw)
    }

    /// `block_text(fragment_fromstring(raw, create_parent="div"))`; None where lxml raises.
    fn fragment_block_text(&mut self, raw: &str) -> Option<String> {
        let html = if super::load::looks_like_full_html(raw) {
            raw.to_owned()
        } else {
            format!("<html><body>{raw}</body></html>")
        };
        let doc = crate::html::lxml::fromstring_str(&html).ok()?;
        let root = self.d.import(&doc, doc.root);
        // assert _nons(doc.tag) == 'html' and exactly one body
        if self.d.tag(root) != t::HTML {
            return None;
        }
        let bodies: Vec<NodeId> = self
            .d
            .children(root)
            .filter(|&c| self.d.is_element(c) && self.d.tag(c) == t::BODY)
            .collect();
        let [body] = bodies[..] else { return None };
        let div = self.d.element(t::DIV);
        if let Some(text) = self
            .d
            .text(body)
            .filter(|s| !pystr::strip(s).is_empty())
            .map(str::to_owned)
        {
            self.set_text(div, Some(text));
        }
        for c in self.d.children_vec(body) {
            self.append(div, c);
        }
        Some(self.block_text(div))
    }

    /// baseline._attempt(texts, dedupe) with _build_body: the body if its text is longer than
    /// _MIN_CONTENT_LENGTH.
    fn attempt(&mut self, texts: Vec<String>, dedupe: bool) -> Option<Result3> {
        let postbody = self.d.element(t::BODY);
        let mut temp_text = String::new();
        let mut temp_len = 0usize;
        for text in texts {
            let text = remove_control_characters(&text, self.v).into_owned();
            let len = py_len(&text);
            if !text.is_empty()
                && (!dedupe
                    || len <= self.p.min_duplicate_length
                    || temp_len > self.p.dedupe_scan_cap
                    || !temp_text.contains(&text))
            {
                let p = self.d.sub_element(postbody, t::P);
                self.set_text(p, Some(text.clone()));
                if temp_text.is_empty() {
                    temp_text = text;
                    temp_len = len;
                } else {
                    temp_text.push('\n');
                    temp_text.push_str(&text);
                    temp_len += 1 + len;
                }
            }
        }
        (temp_len > self.p.baseline_min_content).then_some((postbody, temp_text, temp_len))
    }

    /// `_attempt(map(_render_text, texts), dedupe=True)`; None also when a rendering raised.
    fn attempt_rendered(&mut self, texts: Vec<String>) -> Option<Result3> {
        let mut rendered = Vec::with_capacity(texts.len());
        for raw in texts {
            let r = self.render_text(&raw);
            if self.raised {
                return None;
            }
            rendered.push(r);
        }
        self.attempt(rendered, true)
    }

    /// baseline._collect_json_content: (full-text bodies, teaser descriptions).
    fn collect_json_content(&mut self, tree: NodeId) -> (Vec<String>, Vec<String>) {
        let mut bodies = Vec::new();
        let mut teasers = Vec::new();
        for e in self.findall_desc(tree, t::SCRIPT) {
            if self.d.attr(e, t::TYPE) != Some("application/ld+json") {
                continue;
            }
            let Some(text) = self
                .d
                .text(e)
                .filter(|s| !s.is_empty() && matchers::JSON_HOOKS.search(s))
            else {
                continue;
            };
            if let Ok(v) = pyjson::loads(text, false) {
                walk_json(&v, &mut bodies, &mut teasers);
            }
        }
        self.discourse_texts(tree, &mut bodies);
        (bodies, teasers)
    }

    /// baseline._discourse_texts.
    fn discourse_texts(&mut self, tree: NodeId, texts: &mut Vec<String>) {
        let Some(node) = xpaths::select_first(&self.d, tree, |d, e| {
            d.tag(e) == t::DIV && d.attr(e, t::ID) == Some("data-preloaded")
        }) else {
            return;
        };
        let Ok(preloaded) = pyjson::loads(self.d.get(node, "data-preloaded").unwrap_or(""), true)
        else {
            return;
        };
        let Value::Dict(m) = &preloaded else { return };
        for (key, value) in m {
            if !key.starts_with("topic_") {
                continue;
            }
            // `json.loads(value)["post_stream"]["posts"]` in `try`
            let Some(s) = value.as_str() else { continue };
            let Ok(topic) = pyjson::loads(s, true) else {
                continue;
            };
            let Some(posts) = topic.get("post_stream").and_then(|p| p.get("posts")) else {
                continue;
            };
            match posts {
                Value::List(items) => {
                    for p in items {
                        if let Some(c) = p.get("cooked").and_then(Value::as_str) {
                            texts.push(c.to_owned());
                        }
                    }
                }
                // iterating a str or a dict yields strings, never a dict
                Value::Str(_) | Value::Dict(_) => {}
                // iterating anything else raises TypeError, outside the `try`
                Value::Null | Value::Bool(_) | Value::Number(_) => {
                    self.raise();
                    return;
                }
            }
        }
    }

    /// baseline.baseline(tree): (body, text, length).
    pub fn baseline(&mut self, tree: NodeId) -> Result3 {
        let empty = |cx: &mut Self| (cx.d.element(t::BODY), String::new(), 0);
        let tree = self.d.deep_copy(tree);
        let (json_bodies, json_teasers) = self.collect_json_content(tree);
        if self.raised {
            return empty(self);
        }
        if let Some(r) = self.attempt_rendered(json_bodies) {
            return r;
        }
        if self.raised {
            return empty(self);
        }
        self.basic_cleaning(tree);
        // article tags not nested in another: a dominant one makes much smaller ones noise
        let articles: Vec<NodeId> = xpaths::select(&self.d, tree, |d, e| d.tag(e) == t::ARTICLE)
            .into_iter()
            .filter(|&e| {
                !self
                    .d
                    .ancestors(e)
                    .any(|a| self.d.is_element(a) && self.d.tag(a) == t::ARTICLE)
            })
            .collect();
        let mut article_texts = Vec::new();
        for e in articles {
            let text = self.block_text(e);
            if self.raised {
                return empty(self);
            }
            if py_len(&text) > self.p.baseline_min_content {
                article_texts.push(text);
            }
        }
        if !article_texts.is_empty() {
            let max = article_texts.iter().map(|s| py_len(s)).max().unwrap_or(0);
            let kept: Vec<String> = article_texts
                .into_iter()
                .filter(|s| 5 * py_len(s) >= max)
                .collect();
            if let Some(r) = self.attempt(kept, false) {
                return r;
            }
        }
        // text paragraphs, skipping those inside an element already scraped
        let scraped = TagSet::of(&[t::BLOCKQUOTE, t::CODE, t::P, t::PRE, t::Q, t::QUOTE]);
        let elems: Vec<NodeId> = self
            .d
            .iter_vec(tree, &Matcher::Tags(scraped.clone()), true)
            .into_iter()
            .filter(|&e| {
                !self
                    .d
                    .ancestors(e)
                    .any(|a| self.d.is_element(a) && scraped.contains(self.d.tag(a)))
            })
            .collect();
        let mut texts = Vec::with_capacity(elems.len());
        for e in elems {
            let text = self.block_text(e);
            if self.raised {
                return empty(self);
            }
            texts.push(text);
        }
        if let Some(r) = self.attempt(texts, true) {
            return r;
        }
        // teaser tier, then the whole body's text
        let teaser = self.attempt_rendered(json_teasers);
        if self.raised {
            return empty(self);
        }
        let postbody = self.d.element(t::BODY);
        if let Some(body_elem) = self.find_desc(tree, t::BODY) {
            let p = self.d.sub_element(postbody, t::P);
            let mut lines: Vec<String> = Vec::new();
            self.d.itertext(body_elem, &mut |s| {
                let x = trim(s);
                if !x.is_empty() {
                    lines.push(x);
                }
            });
            let text = remove_control_characters(&lines.join("\n"), self.v).into_owned();
            self.set_text(p, Some(text.clone()));
            let n = py_len(&text);
            if teaser.as_ref().is_none_or(|t| n >= t.2) {
                return (postbody, text, n);
            }
        }
        teaser.unwrap_or((postbody, String::new(), 0))
    }
}

/// baseline._walk_json, depth-first like the recursive original: each dict item's own texts,
/// then its `@graph`, then its `mainEntity`, then the next item.
fn walk_json(node: &Value, bodies: &mut Vec<String>, teasers: &mut Vec<String>) {
    // (items of one as_list() call, next index, depth)
    let mut stack: Vec<(Vec<&Value>, usize, usize)> = vec![(as_list(Some(node)), 0, 1)];
    while let Some(top) = stack.last_mut() {
        let (items, idx, depth) = (&top.0, top.1, top.2);
        let Some(&item) = items.get(idx) else {
            stack.pop();
            continue;
        };
        top.1 += 1;
        if !matches!(item, Value::Dict(_)) {
            continue;
        }
        for key in ["articleBody", "reviewBody"] {
            if let Some(s) = item
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                bodies.push(s.to_owned());
            }
        }
        for key in ["recipeInstructions", "step"] {
            for step in as_list(item.get(key)) {
                match step {
                    Value::Str(s) => bodies.push(s.clone()),
                    Value::Dict(_) => {
                        let mut subs = vec![step];
                        subs.extend(as_list(step.get("itemListElement")));
                        for sub in subs {
                            if let Some(s) = sub.get("text").and_then(Value::as_str) {
                                bodies.push(s.to_owned());
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if let Some(a @ Value::Dict(_)) = item.get("acceptedAnswer") {
            if let Some(s) = a.get("text").and_then(Value::as_str) {
                bodies.push(s.to_owned());
            }
        }
        let is_teaser = item
            .get("@type")
            .is_some_and(|t| t.str_contains("Product") || t.str_contains("VideoObject"));
        if is_teaser {
            if let Some(s) = item.get("description").and_then(Value::as_str) {
                teasers.push(s.to_owned());
            }
        }
        if depth + 1 > WALK_JSON_MAX_DEPTH {
            // RecursionError, caught by the caller: the texts collected so far stay
            return;
        }
        // the recursive calls, @graph first: pushed in reverse
        stack.push((as_list(item.get("mainEntity")), 0, depth + 1));
        stack.push((as_list(item.get("@graph")), 0, depth + 1));
    }
}

/// utils.as_list.
fn as_list(v: Option<&Value>) -> Vec<&Value> {
    match v {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::List(items)) => items.iter().collect(),
        Some(other) => vec![other],
    }
}
