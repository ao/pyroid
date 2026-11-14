// Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
//! `parse_array.py` (without schema support: `schema=` is delegated).

use super::super::value::Value;
use super::scan::Targets;
use super::{Ctx, Parser, R, is_delim};

impl Parser<'_> {
    /// `parse_array(closing_delimiter=...)`: the index is just past the opener.
    pub(crate) fn parse_array(&mut self, closing: char) -> R<Vec<Value>> {
        self.enter()?;
        let r = self.parse_array_inner(closing);
        self.leave();
        r
    }

    fn parse_array_inner(&mut self, closing: char) -> R<Vec<Value>> {
        let array_is_object_value = self.ctx_current() == Some(Ctx::ObjectValue);
        let mut closed_before_parent_member = false;
        let mut arr: Vec<Value> = Vec::new();
        // `all(not isinstance(item, (dict, list)) for item in arr)`, kept up to date.
        let mut containers_in_arr = 0usize;
        self.ctx_set(Ctx::Array);
        let ch = loop {
            self.skip_whitespace_and_comments();
            let ch = self.cur();
            let Some(c) = ch else { break ch };
            if c == closing || c == '}' {
                break ch;
            }
            if c == ',' {
                self.index += 1;
                continue;
            }
            let value = if is_delim(c) {
                // A string followed by ':' is often a missing object start.
                let i = self.skip_to_character(Targets::one(c), 1);
                let i = self.scroll_whitespaces(i + 1);
                if self.get_char_at(i) == Some(':') {
                    if array_is_object_value && !arr.is_empty() && containers_in_arr == 0 {
                        self.log(|| {
                            "Closed unclosed array before object member after scalar items".into()
                        });
                        closed_before_parent_member = true;
                        break ch;
                    }
                    self.parse_object()?
                } else {
                    self.parse_string()?
                }
            } else {
                self.parse_json()?
            };
            let next = self.cur();
            if value.is_strictly_empty() && next != Some(closing) && next != Some(',') {
                self.index += 1;
            } else if value.is_str("...") && self.get_char_at(-1) == Some('.') {
                self.log(|| "While parsing an array, found a stray '...'; ignoring it".into());
            } else {
                if value.is_container() {
                    containers_in_arr += 1;
                }
                arr.push(value);
            }
        };
        if ch != Some(closing) {
            self.log(|| {
                format!("While parsing an array we missed the closing {closing}, ignoring it")
            });
        }
        if !closed_before_parent_member {
            self.index += 1;
        }
        self.ctx_reset();
        Ok(arr)
    }
}
