// Derived from trafilatura 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
//! trafilatura 1.11.0's baseline.py: `baseline(deepcopy(tree_backup))`.

use super::super::pyjson::{self, Value};
use super::super::pystr::{self, PyError, py_len, trim};
use super::super::xpaths::select;
use super::V11;
use crate::html::tree::{Matcher, NodeId, names as t};

/// (body, text, length) as baseline returns them.
type Result3 = (NodeId, String, usize);

/// What `json.loads(script.text).get("articleBody")` gives (in `try: ... except Exception:`).
enum Body {
    /// Falsy (or the lookup raised): skipped.
    None,
    Str(String),
    /// Another truthy value: the code after it raises TypeError.
    Other,
}

fn article_body(text: &str) -> Body {
    let Ok(Value::Dict(m)) = pyjson::loads(text, true) else {
        return Body::None;
    };
    match m.get("articleBody") {
        None | Some(Value::Null | Value::Bool(false) | Value::Number(false)) => Body::None,
        Some(Value::Str(s)) if s.is_empty() => Body::None,
        Some(Value::Str(s)) => Body::Str(s.clone()),
        Some(Value::List(l)) if l.is_empty() => Body::None,
        Some(Value::Dict(d)) if d.is_empty() => Body::None,
        Some(_) => Body::Other,
    }
}

impl V11<'_, '_> {
    /// `SubElement(postbody, 'p').text = text`: lxml checks the string whatever its origin.
    fn add_p(&mut self, body: NodeId, text: &str) {
        let p = self.d.sub_element(body, t::P);
        if !pystr::xml_valid(text) || pyjson::has_surrogate(text) {
            self.raise();
        }
        self.d.set_text(p, Some(text.to_owned()));
    }

    /// `temp_text += " " + text if temp_text else text`.
    fn join_text(temp: &mut String, text: &str) {
        if !temp.is_empty() {
            temp.push(' ');
        }
        temp.push_str(text);
    }

    /// baseline.baseline on a copy of the uncleaned tree: (body, text, length), or the
    /// AttributeError it lets escape.
    pub fn baseline(&mut self, tree: NodeId) -> Result<Result3, PyError> {
        let postbody = self.d.element(t::BODY);
        // JSON-LD articleBody
        let mut temp_text = String::new();
        let scripts: Vec<NodeId> = select(&self.d, tree, |d, e| {
            d.tag(e) == t::SCRIPT && d.attr(e, t::TYPE) == Some("application/ld+json")
        });
        for e in scripts {
            let Some(raw) = self
                .d
                .text(e)
                .filter(|s| !s.is_empty() && s.contains("articleBody"))
            else {
                continue;
            };
            let text = match article_body(raw) {
                Body::None => continue,
                Body::Other => {
                    // `"<p>" in value` / `trim(value)` then `str + None`: TypeError
                    self.raise();
                    return Ok((postbody, temp_text, 0));
                }
                Body::Str(s) if s.contains("<p>") => {
                    let wtf8 = pyjson::to_wtf8(&s);
                    let Some(doc) =
                        super::super::load::load_html_for(&wtf8, super::super::Version::V1_11_0)
                    else {
                        return Err(PyError::NoneAttribute {
                            attr: "text_content",
                        });
                    };
                    trim(&doc.text_content(doc.root))
                }
                Body::Str(s) => trim(&s),
            };
            self.add_p(postbody, &text);
            Self::join_text(&mut temp_text, &text);
            if self.raised {
                return Ok((postbody, temp_text, 0));
            }
        }
        if py_len(&temp_text) > 100 {
            let n = py_len(&temp_text);
            return Ok((postbody, temp_text, n));
        }
        self.basic_cleaning(tree);
        // article elements
        let mut temp_text = String::new();
        for e in select(&self.d, tree, |d, e| d.tag(e) == t::ARTICLE) {
            let text = trim(&self.d.text_content(e));
            if py_len(&text) > 100 {
                let p = self.d.sub_element(postbody, t::P);
                self.set_text(p, Some(text.clone()));
                Self::join_text(&mut temp_text, &text);
            }
        }
        // (the articleBody paragraphs count too)
        if self.d.len(postbody) > 0 {
            let n = py_len(&temp_text);
            return Ok((postbody, temp_text, n));
        }
        // text paragraphs, each distinct text once
        let mut seen = std::collections::HashSet::new();
        let mut temp_text = String::new();
        let tags = [t::BLOCKQUOTE, t::CODE, t::P, t::PRE, t::Q, t::QUOTE];
        for e in self.d.iter_vec(tree, &Matcher::tags(&tags), true) {
            let entry = trim(&self.d.text_content(e));
            if seen.insert(entry.clone()) {
                let p = self.d.sub_element(postbody, t::P);
                self.set_text(p, Some(entry.clone()));
                Self::join_text(&mut temp_text, &entry);
            }
        }
        if py_len(&temp_text) > 100 {
            let n = py_len(&temp_text);
            return Ok((postbody, temp_text, n));
        }
        // the whole body, one trimmed text node per line
        let postbody = self.d.element(t::BODY);
        let p = self.d.sub_element(postbody, t::P);
        if let Some(body) = self.find_desc(tree, t::BODY) {
            let mut lines: Vec<String> = Vec::new();
            self.d.itertext(body, &mut |s| lines.push(trim(s)));
            let text = lines.join("\n");
            self.set_text(p, Some(text.clone()));
            let n = py_len(&text);
            return Ok((postbody, text, n));
        }
        // html2txt(tree, clean=False): no body, so ""
        self.set_text(p, Some(String::new()));
        Ok((postbody, String::new(), 0))
    }
}
