// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! `parse_object.py` (without schema support: `schema=` is delegated).

use super::super::value::{Object, Value, new_object};
use super::{Ctx, ParseError, Parser, R, is_delim, opt_delim};

/// `_strip_comments_for_empty_object_classification(body)`.
fn strip_comments(body: &[char]) -> Vec<char> {
    let n = body.len();
    let mut stripped = Vec::with_capacity(n);
    let mut in_quote: Option<char> = None;
    let mut backslashes = 0usize;
    let mut i = 0;
    while i < n {
        let ch = body[i];
        let next = body.get(i + 1).copied();
        if ch == '\\' {
            backslashes += 1;
            stripped.push(ch);
            i += 1;
            continue;
        }
        if let Some(q) = in_quote {
            stripped.push(ch);
            if ch == q && backslashes % 2 == 0 {
                in_quote = None;
            }
            backslashes = 0;
            i += 1;
            continue;
        }
        if is_delim(ch) && backslashes % 2 == 0 {
            in_quote = Some(ch);
            stripped.push(ch);
            backslashes = 0;
            i += 1;
            continue;
        }
        backslashes = 0;
        if ch == '#' || (ch == '/' && next == Some('/')) {
            i += if ch == '/' { 2 } else { 1 };
            while i < n && body[i] != '\n' && body[i] != '\r' {
                i += 1;
            }
            continue;
        }
        if ch == '/' && next == Some('*') {
            i += 2;
            while i + 1 < n && !(body[i] == '*' && body[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(n);
            continue;
        }
        stripped.push(ch);
        i += 1;
    }
    stripped
}

/// `s.startswith(prefix)` for char slices.
fn starts_with(s: &[char], prefix: &[char]) -> bool {
    s.len() >= prefix.len() && s[..prefix.len()] == *prefix
}

/// `needle in s` for char slices.
fn contains(s: &[char], needle: &[char]) -> bool {
    s.windows(needle.len()).any(|w| w == needle)
}

/// `s.replace(from, to)` for a 2-char `from` and 1-char `to`.
fn replace2(s: &[char], from: [char; 2], to: char) -> Vec<char> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if i + 1 < s.len() && s[i] == from[0] && s[i + 1] == from[1] {
            out.push(to);
            i += 2;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// The row statistics of the array `_merge_object_array_continuation` extends.
#[derive(Clone, Debug)]
pub(crate) struct ListStats {
    key: String,
    /// Length of the array these stats describe.
    len: usize,
    /// Number of list items, the first one's length, whether all have that length.
    lists: usize,
    first: usize,
    all_same: bool,
    /// Number of non-list items at the end.
    trailing: usize,
}

impl ListStats {
    fn compute(key: String, items: &[Value]) -> Self {
        let mut st = Self {
            key,
            len: 0,
            lists: 0,
            first: 0,
            all_same: true,
            trailing: 0,
        };
        for v in items {
            st.push(v);
        }
        st
    }

    fn push(&mut self, v: &Value) {
        self.len += 1;
        if let Value::Array(a) = v {
            if self.lists == 0 {
                self.first = a.len();
            } else if a.len() != self.first {
                self.all_same = false;
            }
            self.lists += 1;
            self.trailing = 0;
        } else {
            self.trailing += 1;
        }
    }

    /// After removing the trailing scalars only.
    fn truncate(&mut self, len: usize) {
        self.trailing -= self.len - len;
        self.len = len;
    }

    /// `expected_len`: the common length of the list items, if any.
    fn expected_len(&self) -> Option<usize> {
        (self.lists > 0 && self.all_same).then_some(self.first)
    }
}

/// `_classify_empty_object_repair` outcomes (no schema: never `schema_set_object`).
enum EmptyRepair {
    Keep,
    Object(Vec<char>),
    Array,
}

impl Parser<'_> {
    /// `parse_object()`: the index is just past the `{`.
    pub(crate) fn parse_object(&mut self) -> R<Value> {
        self.enter()?;
        let r = self.parse_object_inner();
        self.leave();
        r
    }

    fn parse_object_inner(&mut self) -> R<Value> {
        self.merge_stats.push(None);
        let mut obj = new_object();
        let start_index = self.index;
        let parsing_object_value = self.ctx_current() == Some(Ctx::ObjectValue);
        loop {
            self.skip_whitespace_and_comments();
            let ch = self.cur();
            if ch.is_none() || ch == Some('}') {
                break;
            }
            if ch == Some(',') {
                self.index += 1;
                continue;
            }
            if ch == Some(':') {
                self.log(|| "While parsing an object we found a : before a key, ignoring".into());
                self.index += 1;
            }
            let (key, rollback_index) = self.parse_object_key(&mut obj)?;
            if self.ctx_has(Ctx::Array) && obj.contains_key(&key) {
                if self.strict {
                    self.log(|| {
                        "Duplicate key found in strict mode while parsing object, raising an error"
                            .into()
                    });
                    return Err(ParseError::Strict(
                        "Duplicate key found in strict mode while parsing object.",
                    ));
                }
                if !parsing_object_value {
                    if self.should_split_duplicate_object(rollback_index) {
                        self.log(|| {
                            "While parsing an object we found a duplicate key, closing the object here and rolling back the index".into()
                        });
                        self.split_object_on_duplicate_key(rollback_index);
                        break;
                    }
                    self.log(|| {
                        "While parsing an object we found a duplicate key with a normal comma separator, keeping duplicate-key overwrite behavior".into()
                    });
                }
            }
            self.skip_whitespace_and_comments();
            let ch = self.cur();
            if ch.is_none() || ch == Some('}') {
                continue;
            }
            if ch != Some(':') {
                if self.strict {
                    self.log(|| {
                        "Missing ':' after key in strict mode while parsing object, raising an error"
                            .into()
                    });
                    return Err(ParseError::Strict(
                        "Missing ':' after key in strict mode while parsing object.",
                    ));
                }
                self.log(|| "While parsing an object we missed a : after a key".into());
            }
            self.index += 1;
            let value = self.parse_object_value()?;
            if self.strict && value.is_str("") && !opt_delim(self.get_char_at(-1)) {
                self.log(|| {
                    "Parsed value is empty in strict mode while parsing object, raising an error"
                        .into()
                });
                return Err(ParseError::Strict(
                    "Parsed value is empty in strict mode while parsing object.",
                ));
            }
            if let Some(Some(st)) = self.merge_stats.last() {
                if st.key == key {
                    *self.merge_stats.last_mut().expect("checked") = None;
                }
            }
            if let Some(old) = obj.insert(key, value) {
                super::super::value::drop_deep(old);
            }
            if matches!(self.cur(), Some(',' | '\'' | '"')) {
                self.index += 1;
            }
            if self.cur() == Some(']') && self.ctx_has(Ctx::Array) {
                self.log(|| {
                    "While parsing an object we found a closing array bracket, closing the object here and rolling back the index".into()
                });
                self.index -= 1;
                break;
            }
        }
        self.index += 1;
        self.merge_stats.pop();

        if let Some(v) = self.repair_empty_object_result(&obj, start_index)? {
            return Ok(v);
        }
        self.complete_object_parse(obj)
    }

    /// `_parse_object_key`: the key and the index where it started.
    fn parse_object_key(&mut self, obj: &mut Object) -> R<(String, isize)> {
        let mut key = String::new();
        let mut rollback_index = self.index;
        self.ctx_set(Ctx::ObjectKey);
        while self.cur().is_some() {
            rollback_index = self.index;
            if self.cur() == Some('[')
                && key.is_empty()
                && self.merge_object_array_continuation(obj)?
            {
                continue;
            }
            let Value::Str(raw_key) = self.parse_string()? else {
                return Err(ParseError::Assertion);
            };
            key = raw_key;
            if key.is_empty() {
                self.skip_whitespace_and_comments();
            }
            if !key.is_empty() || matches!(self.cur(), Some(':' | '}')) {
                if key.is_empty() && self.strict {
                    self.log(|| {
                        "Empty key found in strict mode while parsing object, raising an error"
                            .into()
                    });
                    return Err(ParseError::Strict(
                        "Empty key found in strict mode while parsing object.",
                    ));
                }
                break;
            }
        }
        self.ctx_reset();
        Ok((key, rollback_index))
    }

    /// `_merge_object_array_continuation`.
    fn merge_object_array_continuation(&mut self, obj: &mut Object) -> R<bool> {
        // `not prev_key` is also true for an empty-string key.
        let Some(prev_key) = obj.keys().last().filter(|k| !k.is_empty()).cloned() else {
            return Ok(false);
        };
        if !matches!(obj.get(&prev_key), Some(Value::Array(_))) || self.strict {
            return Ok(false);
        }
        self.index += 1;
        let mut new_array = self.parse_array(']')?;
        let Some(Value::Array(prev_value)) = obj.get_mut(&prev_key) else {
            unreachable!("checked above")
        };
        // json_repair recomputes the row lengths and pops/re-pushes the trailing
        // scalars on every continuation (quadratic over many continuations);
        // the stats are kept up to date instead.
        let cached = self.merge_stats.last_mut().and_then(Option::take);
        let mut stats = match cached {
            Some(st) if st.key == prev_key && st.len == prev_value.len() => st,
            _ => ListStats::compute(prev_key, prev_value),
        };
        match stats.expected_len() {
            Some(expected_len) if expected_len > 0 => {
                let t = stats.trailing;
                if t > 0 && t % expected_len == 0 {
                    self.log(|| {
                        "While parsing an object we found row values without an inner array, grouping them into rows".into()
                    });
                    let tail: Vec<Value> = prev_value.split_off(prev_value.len() - t);
                    stats.truncate(prev_value.len());
                    let mut rest = tail.into_iter();
                    loop {
                        let row: Vec<Value> = rest.by_ref().take(expected_len).collect();
                        if row.is_empty() {
                            break;
                        }
                        let row = Value::Array(row);
                        stats.push(&row);
                        prev_value.push(row);
                    }
                }
                // Otherwise json_repair pops the trailing scalars and pushes them
                // back unchanged.
                if !new_array.is_empty() {
                    if new_array.iter().all(|v| matches!(v, Value::Array(_))) {
                        self.log(|| {
                            "While parsing an object we found additional rows, appending them without flattening".into()
                        });
                        for v in &new_array {
                            stats.push(v);
                        }
                        prev_value.extend(new_array);
                    } else {
                        let v = Value::Array(new_array);
                        stats.push(&v);
                        prev_value.push(v);
                    }
                }
            }
            _ => {
                let items = if new_array.len() == 1 && matches!(new_array[0], Value::Array(_)) {
                    let Some(Value::Array(inner)) = new_array.pop() else {
                        unreachable!("checked above")
                    };
                    inner
                } else {
                    new_array
                };
                for v in &items {
                    stats.push(v);
                }
                prev_value.extend(items);
            }
        }
        if let Some(slot) = self.merge_stats.last_mut() {
            *slot = Some(stats);
        }
        self.skip_whitespace_and_comments();
        if self.cur() == Some(',') {
            self.index += 1;
        }
        self.skip_whitespace_and_comments();
        Ok(true)
    }

    /// `_should_split_duplicate_object`.
    fn should_split_duplicate_object(&mut self, rollback_index: isize) -> bool {
        let mut lookback_idx = rollback_index - self.index - 1;
        let mut prev = self.get_char_at(lookback_idx);
        while self.opt_space(prev) {
            lookback_idx -= 1;
            prev = self.get_char_at(lookback_idx);
        }
        let key_start_char = self.get_char_at(rollback_index - self.index);
        let next = self.scroll_whitespaces(0);
        let next_non_whitespace = self.get_char_at(next);
        !(opt_delim(key_start_char) && prev == Some(',') && next_non_whitespace == Some(':'))
    }

    /// `_split_object_on_duplicate_key`: insert a `{` where the duplicate key
    /// starts and roll back to just before it.
    fn split_object_on_duplicate_key(&mut self, rollback_index: isize) {
        self.index = rollback_index - 1;
        let at = (self.index + 1).clamp(0, self.len()) as usize;
        self.splice(at, at, &['{']);
    }

    /// `_parse_object_value`.
    fn parse_object_value(&mut self) -> R<Value> {
        self.ctx_set(Ctx::ObjectValue);
        self.skip_whitespace_and_comments();
        let ch = self.cur();
        let r = if let Some(c @ (',' | '}')) = ch {
            self.log(|| format!("While parsing an object value we found a stray {c}, ignoring it"));
            Ok(Value::empty_str())
        } else {
            self.parse_json()
        };
        self.ctx_reset();
        r
    }

    /// `_repair_empty_object_result`: `Some(value)` when an empty object is
    /// reparsed as something else.
    fn repair_empty_object_result(&mut self, obj: &Object, start_index: isize) -> R<Option<Value>> {
        if !obj.is_empty() || self.index - start_index <= 2 {
            return Ok(None);
        }
        let body = self.slice(start_index, self.index - 1);
        if self.get_char_at(-1) == Some('}')
            && strip_comments(body).iter().all(|&c| self.is_space(c))
        {
            return Ok(None);
        }
        if self.strict {
            self.log(|| {
                "Parsed object is empty but contains extra characters in strict mode, raising an error"
                    .into()
            });
            return Err(ParseError::Strict(
                "Parsed object is empty but contains extra characters in strict mode.",
            ));
        }
        match self.classify_empty_object_repair(start_index) {
            EmptyRepair::Object(normalized) => {
                let end_index = self.index + 1;
                let a = (start_index - 1).clamp(0, self.len()) as usize;
                let b = end_index.clamp(a as isize, self.len()) as usize;
                self.splice(a, b, &normalized);
                self.index = start_index;
                self.ctx_set(Ctx::ObjectKey);
                let repaired = self.parse_object()?;
                self.ctx_reset();
                self.deferred.push(Ctx::ObjectKey);
                Ok(Some(repaired))
            }
            EmptyRepair::Array => {
                self.log(|| {
                    "Parsed object is empty, we will try to parse this as an array instead".into()
                });
                self.index = start_index;
                self.ctx_set(Ctx::ObjectKey);
                let repaired = self.parse_array(']')?;
                self.ctx_reset();
                self.deferred.push(Ctx::ObjectKey);
                Ok(Some(Value::Array(repaired)))
            }
            EmptyRepair::Keep => Ok(None),
        }
    }

    /// `_classify_empty_object_repair`.
    fn classify_empty_object_repair(&mut self, start_index: isize) -> EmptyRepair {
        let attempted: Vec<char> = self.slice(start_index - 1, self.index + 1).to_vec();
        let mut body: &[char] = attempted.get(1..).unwrap_or(&[]);
        if body.last() == Some(&'}') {
            body = &body[..body.len() - 1];
        }
        let lstrip = |s: &[char]| -> usize { s.iter().take_while(|&&c| self.is_space(c)).count() };
        body = &body[lstrip(body)..];
        if body.is_empty() {
            return EmptyRepair::Keep;
        }
        if (starts_with(body, &['\\', '"']) && contains(body, &['\\', '"', ':']))
            || (starts_with(body, &['\\', '\'']) && contains(body, &['\\', '\'', ':']))
        {
            let normalized = replace2(&replace2(&attempted, ['\\', '"'], '"'), ['\\', '\''], '\'');
            self.log(|| {
                "Parsed object is empty but the input starts like an escaped object key, normalizing and reparsing it as an object".into()
            });
            return EmptyRepair::Object(normalized);
        }
        let stripped = strip_comments(body);
        let body = &stripped[lstrip(&stripped)..];
        if body.is_empty() {
            return EmptyRepair::Keep;
        }
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        for &ch in body {
            if ch == '\\' {
                backslashes += 1;
                continue;
            }
            if let Some(q) = in_quote {
                if ch == q && backslashes % 2 == 0 {
                    in_quote = None;
                }
            } else if is_delim(ch) && backslashes % 2 == 0 {
                in_quote = Some(ch);
            } else if ch == ':' && backslashes % 2 == 0 {
                self.log(|| {
                    "Parsed object is empty but the input still contains an object-style separator, keeping object repair".into()
                });
                return EmptyRepair::Keep;
            }
            backslashes = 0;
        }
        EmptyRepair::Array
    }

    /// `_complete_object_parse`.
    fn complete_object_parse(&mut self, mut obj: Object) -> R<Value> {
        if !self.ctx_empty() {
            if self.cur() == Some('}')
                && !matches!(self.ctx_current(), Some(Ctx::ObjectKey | Ctx::ObjectValue))
            {
                self.log(|| {
                    "Found an extra closing brace that shouldn't be there, skipping it".into()
                });
                self.index += 1;
            }
            return Ok(Value::Object(obj));
        }
        self.skip_whitespaces();
        if self.cur() == Some(',') {
            self.index += 1;
            self.skip_whitespaces();
            if opt_delim(self.cur()) && !self.strict {
                self.log(|| {
                    "Found a comma and string delimiter after object closing brace, checking for additional key-value pairs".into()
                });
                if let Value::Object(additional) = self.parse_object()? {
                    for (k, v) in additional {
                        if let Some(old) = obj.insert(k, v) {
                            super::super::value::drop_deep(old);
                        }
                    }
                }
            }
        }
        Ok(Value::Object(obj))
    }
}
