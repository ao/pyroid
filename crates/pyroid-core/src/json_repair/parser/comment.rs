// Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
//! `parse_comment.py`.

use super::super::value::Value;
use super::{Ctx, Parser, R};

impl Parser<'_> {
    /// `skip_comment()`: consume one comment without parsing a following value.
    pub(crate) fn skip_comment(&mut self) {
        let ch = self.cur();
        let next = self.get_char_at(1);
        if ch == Some('#') || (ch == Some('/') && next == Some('/')) {
            let hash = ch == Some('#');
            let (in_array, in_value, in_key) = if hash {
                (
                    self.ctx_has(Ctx::Array),
                    self.ctx_has(Ctx::ObjectValue),
                    self.ctx_has(Ctx::ObjectKey),
                )
            } else {
                (false, false, false)
            };
            let start = self.index;
            while let Some(c) = self.cur() {
                let stop = matches!(c, '\n' | '\r')
                    || (in_array && c == ']')
                    || (in_value && c == '}')
                    || (in_key && c == ':');
                if stop {
                    break;
                }
                self.index += 1;
            }
            if self.logger.is_some() {
                let comment: String = self.slice(start, self.index).iter().collect();
                self.log(|| format!("Found line comment: {comment}, ignoring"));
            }
        } else if ch == Some('/') && next == Some('*') {
            let start = self.index;
            self.index += 2;
            // `comment` starts as "/*"; it ends once it ends with "*/".
            let mut prev = '*';
            loop {
                let Some(c) = self.cur() else {
                    self.log(|| {
                        "Reached end-of-string while parsing block comment; unclosed block comment."
                            .into()
                    });
                    break;
                };
                self.index += 1;
                if prev == '*' && c == '/' {
                    break;
                }
                prev = c;
            }
            if self.logger.is_some() {
                let comment: String = self.slice(start, self.index).iter().collect();
                self.log(|| format!("Found block comment: {comment}, ignoring"));
            }
        } else if ch == Some('/') {
            // Make progress on standalone slashes that reach the repair fallback.
            self.index += 1;
        }
    }

    /// `parse_comment()`: skip comments; resume value parsing only at the top level.
    pub(crate) fn parse_comment(&mut self) -> R<Value> {
        loop {
            self.skip_comment();
            if !self.ctx_empty() {
                return Ok(Value::empty_str());
            }
            self.skip_whitespaces();
            if !matches!(self.cur(), Some('#' | '/')) {
                self.enter()?;
                let r = self.parse_json();
                self.leave();
                return r;
            }
        }
    }
}
