// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! `parse_string.py` and `parse_string_helpers/`.

use super::super::chars::surrogate_char;
use super::super::value::Value;
use super::scan::Targets;
use super::{Ctx, ParseError, Parser, R, is_delim, opt_delim};

/// `„`, which opens a low smart quote span (closed by `”` or `"`).
const LOW_QUOTE: char = '\u{201e}';
const LEFT_QUOTE: char = '\u{201c}';
const RIGHT_QUOTE: char = '\u{201d}';

/// `INLINE_CONTAINER_CLOSING_DELIMITERS`.
#[inline]
fn closing_of(c: char) -> Option<char> {
    match c {
        '[' => Some(']'),
        '{' => Some('}'),
        '(' => Some(')'),
        _ => None,
    }
}

#[inline]
fn is_opener(c: char) -> bool {
    matches!(c, '[' | '{' | '(')
}

/// `_matching_string_delimiter`.
#[inline]
fn matching_delim(c: char) -> char {
    if c == LEFT_QUOTE { RIGHT_QUOTE } else { c }
}

/// `classify_object_value_comma` results.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CommaClass {
    Container,
    Member,
    Str,
    StrNoFutureDelimiter,
}

/// `StringParseState`.
#[allow(clippy::struct_excessive_bools)]
struct StrState {
    missing_quotes: bool,
    doubled_quotes: bool,
    lstring_delimiter: char,
    /// `rstring_delimiter[0]`.
    rstring_delimiter: char,
    /// Low smart quote spans pushed onto `rstring_delimiter` (as `\0` sentinels).
    low_spans: usize,
    acc: Vec<char>,
    /// Indices of the whitespace chars in `acc` (ascending).
    acc_spaces: Vec<usize>,
    unmatched_delimiter: bool,
    pending_inline_container: bool,
    inline_container_stack: Vec<char>,
    object_value_has_no_future_delimiter: bool,
    /// `lookahead_cache`: targets -> (start, match).
    lookahead_cache: Vec<(Targets, isize, Option<isize>)>,
    object_value_unmatched_opening_braces: usize,
    regex_character_class_start: Option<usize>,
}

impl StrState {
    fn new() -> Self {
        Self {
            missing_quotes: false,
            doubled_quotes: false,
            lstring_delimiter: '"',
            rstring_delimiter: '"',
            low_spans: 0,
            acc: Vec::new(),
            acc_spaces: Vec::new(),
            unmatched_delimiter: false,
            pending_inline_container: false,
            inline_container_stack: Vec::new(),
            object_value_has_no_future_delimiter: false,
            lookahead_cache: Vec::new(),
            object_value_unmatched_opening_braces: 0,
            regex_character_class_start: None,
        }
    }

    /// `_active_rstring_delimiter` (`\0` inside a low smart quote span).
    #[inline]
    fn active_rdelim(&self) -> char {
        if self.low_spans > 0 {
            '\0'
        } else {
            self.rstring_delimiter
        }
    }

    #[inline]
    fn in_low_span(&self) -> bool {
        self.low_spans > 0
    }

    #[inline]
    fn last(&self) -> Option<char> {
        self.acc.last().copied()
    }

    /// `not acc or acc[-1] != c`.
    #[inline]
    fn last_is_not(&self, c: char) -> bool {
        self.last() != Some(c)
    }

    fn pop(&mut self) {
        if self.acc.pop().is_some() && self.acc_spaces.last() == Some(&self.acc.len()) {
            self.acc_spaces.pop();
        }
    }

    fn truncate(&mut self, len: usize) {
        self.acc.truncate(len);
        while self.acc_spaces.last().is_some_and(|&i| i >= len) {
            self.acc_spaces.pop();
        }
    }
}

impl Parser<'_> {
    /// `_append_string_content(state, content)`.
    fn acc_append(&mut self, st: &mut StrState, content: &[char]) {
        for &c in content {
            self.acc_push(st, c);
        }
    }

    #[inline]
    fn acc_push(&self, st: &mut StrState, c: char) {
        let i = st.acc.len();
        match c {
            '{' => st.object_value_unmatched_opening_braces += 1,
            '}' if st.object_value_unmatched_opening_braces > 0 => {
                st.object_value_unmatched_opening_braces -= 1;
            }
            '[' => st.regex_character_class_start = Some(i + 1),
            ']' => st.regex_character_class_start = None,
            _ => {}
        }
        if self.is_space(c) {
            st.acc_spaces.push(i);
        }
        st.acc.push(c);
    }

    /// `_append_literal_char`: append the current char, advance, return the next.
    fn append_literal_char(&mut self, st: &mut StrState, c: char) -> Option<char> {
        self.acc_push(st, c);
        self.index += 1;
        self.cur()
    }

    /// `_replace_trailing_backslash(state, content)`.
    fn replace_trailing_backslash(&self, st: &mut StrState, content: &[char]) {
        st.pop();
        for &c in content {
            self.acc_push(st, c);
        }
    }

    /// `rstrip()` of the accumulator.
    fn acc_rstrip(&self, st: &mut StrState) {
        let mut n = st.acc.len();
        while n > 0 && self.is_space(st.acc[n - 1]) {
            n -= 1;
        }
        st.truncate(n);
    }

    /// `_quote_belongs_to_regex_character_class`.
    fn quote_belongs_to_regex_character_class(&mut self, st: &StrState) -> bool {
        let Some(start) = st.regex_character_class_start else {
            return false;
        };
        // `any(c.isspace() for c in acc[start:])`
        if st.acc_spaces.last().is_some_and(|&i| i >= start) {
            return false;
        }
        let closing = self.skip_to_character(Targets::one(']'), 1);
        self.get_char_at(closing) == Some(']')
    }

    /// `_cached_skip_to_character`: json_repair's own per-string lookahead cache
    /// (its answers can differ from a fresh scan, so it is reproduced as is).
    fn cached_skip_to_character(
        &mut self,
        st: &mut StrState,
        targets: Targets,
        idx: isize,
    ) -> isize {
        let start_index = self.index + idx;
        if let Some(&(_, cached_start, cached_match)) =
            st.lookahead_cache.iter().find(|(t, _, _)| *t == targets)
        {
            match cached_match {
                None if start_index >= cached_start => return self.len() - self.index,
                Some(m) if cached_start <= start_index && start_index <= m => {
                    return m - self.index;
                }
                _ => {}
            }
        }
        let match_offset = self.skip_to_character(targets, idx);
        let set = |st: &mut StrState, entry: Option<isize>| {
            if let Some(slot) = st
                .lookahead_cache
                .iter_mut()
                .find(|(t, _, _)| *t == targets)
            {
                *slot = (targets, start_index, entry);
            } else {
                st.lookahead_cache.push((targets, start_index, entry));
            }
        };
        if self.get_char_at(match_offset).is_none() {
            set(st, None);
            return match_offset;
        }
        let match_index = self.index + match_offset;
        if match_index == 0 || self.char_at(match_index - 1) != Some('\\') {
            set(st, Some(match_index));
        }
        match_offset
    }

    /// `parse_string()`.
    pub(crate) fn parse_string(&mut self) -> R<Value> {
        let mut st = match self.prepare_string_entry()? {
            Entry::Direct(v) => return Ok(v),
            Entry::Scan(st) => st,
        };
        let ch = self.scan_string_body(&mut st);
        Ok(self.finalize_string_result(&mut st, ch))
    }

    /// `_try_parse_simple_quoted_string`.
    fn try_parse_simple_quoted_string(&mut self) -> Option<String> {
        if self.cur() != Some('"') {
            return None;
        }
        let start = (self.index + 1).max(0) as usize;
        let end = self.find_plain('"', start)?;
        if self
            .s
            .slice(start, end)
            .iter()
            .any(|&c| matches!(c, '\\' | '\n' | '\r'))
        {
            return None;
        }
        let next_index = self.next_non_space(end + 1);
        let next_char = self.s.get(next_index);
        let ok = match self.ctx_current() {
            Some(Ctx::ObjectKey) => next_char == Some(':'),
            Some(Ctx::ObjectValue) => matches!(next_char, None | Some(',' | '}')),
            Some(Ctx::Array) => matches!(next_char, None | Some(',' | ']')),
            None => next_char.is_none(),
        };
        if !ok {
            return None;
        }
        let value: String = self.s.slice(start, end).iter().collect();
        self.index = end as isize + 1;
        Some(value)
    }

    /// `parse_boolean_or_null`: an unquoted complete literal at the index.
    fn parse_boolean_or_null(&mut self) -> Option<Value> {
        let lower = |c: Option<char>| c.map(|c| c.to_ascii_lowercase());
        let ch = lower(self.cur());
        let starting_index = self.index;
        for (literal, value) in [
            ("true", Value::Bool(true)),
            ("false", Value::Bool(false)),
            ("null", Value::Null),
            ("none", Value::Null),
        ] {
            if ch != literal.chars().next() {
                continue;
            }
            self.index = starting_index;
            let mut complete = true;
            for expected in literal.chars() {
                if lower(self.cur()) != Some(expected) {
                    complete = false;
                    break;
                }
                self.index += 1;
            }
            if complete {
                let next = self.cur();
                if next.is_none()
                    || self.opt_space(next)
                    || matches!(next, Some(',' | ')' | ']' | '}'))
                {
                    if literal == "none" {
                        self.log(|| "Converted unquoted Python None literal to JSON null".into());
                    }
                    return Some(value);
                }
            }
        }
        self.index = starting_index;
        None
    }

    /// `parse_json_llm_block`: a value fenced in ```` ```json ... ``` ````.
    fn parse_json_llm_block(&mut self) -> R<Option<Value>> {
        let fence: [char; 7] = ['`', '`', '`', 'j', 's', 'o', 'n'];
        if *self.slice(self.index, self.index + 7) != fence {
            return Ok(None);
        }
        let i = self.skip_to_character(Targets::one('`'), 7);
        if *self.slice(self.index + i, self.index + i + 3) != ['`', '`', '`'] {
            return Ok(None);
        }
        self.index += 7;
        self.enter()?;
        let r = self.parse_json();
        self.leave();
        r.map(Some)
    }

    /// `_prepare_string_entry`.
    #[allow(clippy::too_many_lines)]
    fn prepare_string_entry(&mut self) -> R<Entry> {
        let mut ch = self.cur();
        if matches!(ch, Some('#' | '/')) {
            return Ok(Entry::Direct(self.parse_comment()?));
        }
        if ch.is_some_and(|c| !is_delim(c) && !self.props.is_alnum(c)) {
            // Skip to the next delimiter or alphanumeric (memoized: unclosed
            // nesting reruns this skip over the same text at every level).
            self.index = self.next_delim_or_alnum(self.index as usize) as isize;
            ch = self.cur();
        }
        let Some(c) = ch else {
            return Ok(Entry::Direct(Value::empty_str()));
        };
        if let Some(v) = self.try_parse_simple_quoted_string() {
            return Ok(Entry::Direct(Value::Str(v)));
        }
        let mut st = StrState::new();
        if c == '\'' {
            st.lstring_delimiter = '\'';
            st.rstring_delimiter = '\'';
        } else if c == LEFT_QUOTE {
            st.lstring_delimiter = LEFT_QUOTE;
            st.rstring_delimiter = RIGHT_QUOTE;
        } else if self.props.is_alnum(c) {
            if matches!(c.to_ascii_lowercase(), 't' | 'f' | 'n')
                && self.ctx_current() != Some(Ctx::ObjectKey)
            {
                if let Some(v) = self.parse_boolean_or_null() {
                    return Ok(Entry::Direct(v));
                }
            }
            self.log(|| "While parsing a string, we found a literal instead of a quote".into());
            st.missing_quotes = true;
        }
        if !st.missing_quotes {
            self.index += 1;
        }
        if self.cur() == Some('`') {
            match self.parse_json_llm_block()? {
                // `ret_val is not False`: a parsed `false` counts as no block.
                Some(Value::Bool(false)) | None => {}
                Some(v) => return Ok(Entry::Direct(v)),
            }
            self.log(|| {
                "While parsing a string, we found code fences but they did not enclose valid JSON, continuing parsing the string".into()
            });
        }
        let l = st.lstring_delimiter;
        let r = st.rstring_delimiter;
        if self.cur() == Some(l) {
            let next = self.get_char_at(1);
            if (self.ctx_current() == Some(Ctx::ObjectKey) && next == Some(':'))
                || (self.ctx_current() == Some(Ctx::ObjectValue) && matches!(next, Some(',' | '}')))
                || (self.ctx_current() == Some(Ctx::Array) && matches!(next, Some(',' | ']')))
            {
                self.index += 1;
                return Ok(Entry::Direct(Value::empty_str()));
            }
            if next == Some(l) {
                self.log(|| {
                    "While parsing a string, we found a doubled quote and then a quote again, ignoring it".into()
                });
                if self.strict {
                    return Err(ParseError::Strict(
                        "Found doubled quotes followed by another quote.",
                    ));
                }
                return Ok(Entry::Direct(Value::empty_str()));
            }
            let i = self.skip_to_character(Targets::one(r), 1);
            if self.get_char_at(i + 1) == Some(r) {
                self.log(|| {
                    "While parsing a string, we found a valid starting doubled quote".into()
                });
                st.doubled_quotes = true;
                self.index += 1;
            } else {
                if self.get_char_at(i) == Some(r) {
                    let outer_quote_idx = self.skip_to_character(Targets::one(r), i + 1);
                    let after_outer_quote_idx = self.scroll_whitespaces(outer_quote_idx + 1);
                    let after_outer_quote = self.get_char_at(after_outer_quote_idx);
                    if self.ctx_current() == Some(Ctx::ObjectValue)
                        && !self.only_whitespace_until(i)
                        && self.get_char_at(outer_quote_idx) == Some(r)
                        && matches!(after_outer_quote, None | Some(',' | '}'))
                    {
                        self.log(|| {
                            "While parsing a string, we found a leading quote that starts a quoted span, keeping it".into()
                        });
                        self.acc_push(&mut st, l);
                        st.unmatched_delimiter = true;
                        self.index += 1;
                        return Ok(Entry::Scan(st));
                    }
                }
                let i = self.scroll_whitespaces(1);
                let next_c = self.get_char_at(i);
                if opt_delim(next_c) || matches!(next_c, Some('{' | '[')) {
                    self.log(|| {
                        "While parsing a string, we found a doubled quote but also another quote afterwards, ignoring it".into()
                    });
                    if self.strict {
                        return Err(ParseError::Strict(
                            "Found doubled quotes followed by another quote while parsing a string.",
                        ));
                    }
                    self.index += 1;
                    return Ok(Entry::Direct(Value::empty_str()));
                }
                if !matches!(next_c, Some(',' | ']' | '}')) {
                    self.log(|| {
                        "While parsing a string, we found a doubled quote but it was a mistake, removing one quote".into()
                    });
                    self.index += 1;
                }
            }
        }
        Ok(Entry::Scan(st))
    }

    /// `_normalize_escape_sequence`: `Some(next char)` if handled (the
    /// accumulator ends with a backslash and `ch` is the char after it).
    #[allow(clippy::option_option)]
    fn normalize_escape_sequence(&mut self, st: &mut StrState, ch: char) -> Option<Option<char>> {
        self.log(|| "Found a stray escape sequence, normalizing it".into());
        let active = st.active_rdelim();
        if st.in_low_span() && ch == '"' {
            self.replace_trailing_backslash(st, &[ch]);
            st.low_spans -= 1;
            self.index += 1;
            return Some(self.cur());
        }
        if ch == '\\' {
            let run_start = self.index - 1;
            let mut run_end = self.index + 1;
            while run_end < self.len() && self.char_at(run_end) == Some('\\') {
                run_end += 1;
            }
            let run_length = run_end - run_start;
            let next_char = self.get_char_at(run_end - self.index);
            let after = self.scroll_whitespaces(run_end - self.index + 1);
            let after_quote = self.get_char_at(after);
            if run_length % 2 == 0
                && (next_char != Some(active)
                    || matches!(after_quote, None | Some(',' | ':' | '}' | ']')))
            {
                let half = vec!['\\'; (run_length / 2) as usize];
                self.replace_trailing_backslash(st, &half);
                self.index = run_end;
                return Some(self.cur());
            }
        }
        if ch == active || matches!(ch, 't' | 'n' | 'r' | 'b' | '\\') {
            let c = match ch {
                't' => '\t',
                'n' => '\n',
                'r' => '\r',
                'b' => '\u{8}',
                other => other,
            };
            self.replace_trailing_backslash(st, &[c]);
            self.index += 1;
            let mut next = self.cur();
            while let Some(n) = next {
                if !(st.last() == Some('\\') && (n == active || n == '\\')) {
                    break;
                }
                self.replace_trailing_backslash(st, &[n]);
                self.index += 1;
                next = self.cur();
            }
            return Some(next);
        }
        if ch == 'u' || ch == 'x' {
            let num: isize = if ch == 'u' { 4 } else { 2 };
            let digits = self.slice(self.index + 1, self.index + 1 + num);
            if digits.len() == num as usize && digits.iter().all(char::is_ascii_hexdigit) {
                let cp = digits
                    .iter()
                    .fold(0u32, |a, d| a * 16 + d.to_digit(16).expect("hex"));
                self.log(|| "Found a unicode escape sequence, normalizing it".into());
                let c = char::from_u32(cp).unwrap_or_else(|| surrogate_char(cp));
                self.replace_trailing_backslash(st, &[c]);
                self.index += 1 + num;
                return Some(self.cur());
            }
        } else if ch == LOW_QUOTE || (is_delim(ch) && ch != active) {
            self.log(|| {
                "Found a delimiter that was escaped but shouldn't be escaped, removing the escape"
                    .into()
            });
            self.replace_trailing_backslash(st, &[ch]);
            self.index += 1;
            return Some(self.cur());
        }
        None
    }

    /// `_brace_before_code_fence_belongs_to_string`.
    fn brace_before_code_fence_belongs_to_string(
        &mut self,
        st: &mut StrState,
        fence_idx: isize,
    ) -> bool {
        let mut quote_search_idx = fence_idx + 3;
        let next_content_idx = self.scroll_comment_prefixed_member_start(quote_search_idx);
        let mut keep_post_fence_container = false;
        if self.get_char_at(next_content_idx).is_some_and(is_opener) {
            if let Some(container_end_idx) = self.skip_inline_container(next_content_idx) {
                if self.post_fence_container_starts_next_member(container_end_idx) {
                    return false;
                }
                keep_post_fence_container = true;
                quote_search_idx = container_end_idx;
            }
        }
        let outer = st.rstring_delimiter;
        let mut quote_idx = self.skip_to_character(Targets::one(outer), quote_search_idx);
        while self.get_char_at(quote_idx) == Some(outer) {
            let after_quote_idx = self.scroll_whitespaces(quote_idx + 1);
            let after_quote = self.get_char_at(after_quote_idx);
            if matches!(after_quote, None | Some(',' | '}' | ']')) {
                if keep_post_fence_container {
                    st.pending_inline_container = true;
                }
                return true;
            }
            quote_idx = self.skip_to_character(Targets::one(outer), quote_idx + 1);
        }
        false
    }

    /// The end of a bare key (`[alnum_-]*`) starting at offset `key_idx`.
    fn bare_key_end(&self, mut key_idx: isize) -> isize {
        while let Some(c) = self.get_char_at(key_idx) {
            if !(self.props.is_alnum(c) || c == '_' || c == '-') {
                break;
            }
            key_idx += 1;
        }
        key_idx
    }

    /// `_bare_key_is_followed_by_colon`.
    fn bare_key_is_followed_by_colon(&mut self, key_idx: isize) -> bool {
        match self.get_char_at(key_idx) {
            Some(c) if self.props.is_alnum(c) || c == '_' => {}
            _ => return false,
        }
        let end = self.bare_key_end(key_idx);
        let end = self.scroll_whitespaces(end);
        self.get_char_at(end) == Some(':')
    }

    /// `_post_fence_container_starts_next_member`.
    fn post_fence_container_starts_next_member(&mut self, container_end_idx: isize) -> bool {
        let after_container_idx = self.scroll_whitespaces(container_end_idx);
        let after_container = self.get_char_at(after_container_idx);
        if matches!(after_container, None | Some('}')) {
            return true;
        }
        if after_container != Some(',') {
            return false;
        }
        let next_member_idx = self.scroll_comment_prefixed_member_start(after_container_idx + 1);
        matches!(self.get_char_at(next_member_idx), None | Some('}'))
            || self.object_member_starts_at(next_member_idx)
    }

    /// `_starts_nested_inline_container`.
    fn starts_nested_inline_container(&mut self, idx: isize) -> bool {
        let opening = self.get_char_at(idx);
        let mut prev_idx = idx - 1;
        while prev_idx >= 0 {
            let Some(prev_char) = self.get_char_at(prev_idx) else {
                return true;
            };
            if !self.is_space(prev_char) {
                if is_opener(prev_char) {
                    return true;
                }
                if prev_char != ',' && prev_char != ':' {
                    return false;
                }
                let next_idx = self.scroll_whitespaces(idx + 1);
                let next_char = self.get_char_at(next_idx);
                if matches!(opening, Some('[' | '(')) {
                    return match next_char {
                        Some(c) => {
                            matches!(c, ']' | ')' | '-' | 't' | 'f' | 'n')
                                || is_delim(c)
                                || is_opener(c)
                                || self.props.is_digit(c)
                        }
                        None => false,
                    };
                }
                if opening != Some('{') {
                    return false;
                }
                if next_char == Some('}') || opt_delim(next_char) {
                    return true;
                }
                return prev_char == ':' && self.bare_key_is_followed_by_colon(next_idx);
            }
            prev_idx -= 1;
        }
        true
    }

    /// `_skip_inline_container`: the offset just past the balanced container at
    /// offset `idx`, or `None`.
    fn skip_inline_container(&mut self, idx: isize) -> Option<isize> {
        let Some(closing) = self.get_char_at(idx).and_then(closing_of) else {
            return Some(idx);
        };
        let mut stack = vec![closing];
        let mut i = idx + 1;
        while let Some(&top) = stack.last() {
            let ch = self.get_char_at(i)?;
            if is_delim(ch) {
                let end_delimiter = matching_delim(ch);
                i = self.skip_to_character(Targets::one(end_delimiter), i + 1);
                if self.get_char_at(i) != Some(end_delimiter) {
                    return None;
                }
            } else if let Some(cl) =
                closing_of(ch).filter(|_| self.starts_nested_inline_container(i))
            {
                stack.push(cl);
            } else if ch == top {
                stack.pop();
                if stack.is_empty() {
                    return Some(i + 1);
                }
            }
            i += 1;
        }
        None
    }

    /// `_scroll_comment_prefixed_member_start`.
    fn scroll_comment_prefixed_member_start(&mut self, idx: isize) -> isize {
        let mut idx = self.scroll_whitespaces(idx);
        loop {
            let ch = self.get_char_at(idx);
            if ch == Some('#') {
                while self
                    .get_char_at(idx)
                    .is_some_and(|c| c != '\n' && c != '\r')
                {
                    idx += 1;
                }
                idx = self.scroll_whitespaces(idx);
                continue;
            }
            if ch == Some('/') {
                let next = self.get_char_at(idx + 1);
                if next == Some('/') {
                    idx += 2;
                    while self
                        .get_char_at(idx)
                        .is_some_and(|c| c != '\n' && c != '\r')
                    {
                        idx += 1;
                    }
                    idx = self.scroll_whitespaces(idx);
                    continue;
                }
                if next == Some('*') {
                    idx += 2;
                    loop {
                        let Some(c) = self.get_char_at(idx) else {
                            return idx;
                        };
                        if c == '*' && self.get_char_at(idx + 1) == Some('/') {
                            idx += 2;
                            break;
                        }
                        idx += 1;
                    }
                    idx = self.scroll_whitespaces(idx);
                    continue;
                }
            }
            return idx;
        }
    }

    /// `_quoted_object_member_follows`.
    fn quoted_object_member_follows(&mut self, quote_idx: isize) -> bool {
        let comma_idx = self.scroll_whitespaces(quote_idx + 1);
        if self.get_char_at(comma_idx) != Some(',') {
            return false;
        }
        let next_member_idx = self.scroll_comment_prefixed_member_start(comma_idx + 1);
        self.object_member_starts_at(next_member_idx)
    }

    /// `_object_member_starts_at`.
    fn object_member_starts_at(&mut self, next_member_idx: isize) -> bool {
        let Some(next_member) = self.get_char_at(next_member_idx) else {
            return false;
        };
        if next_member == '}' {
            return false;
        }
        if is_delim(next_member) {
            let key_end_delimiter = matching_delim(next_member);
            let key_end_idx =
                self.skip_to_character(Targets::one(key_end_delimiter), next_member_idx + 1);
            if self.get_char_at(key_end_idx) != Some(key_end_delimiter) {
                return false;
            }
            let after_key_idx = self.scroll_whitespaces(key_end_idx + 1);
            return self.get_char_at(after_key_idx) == Some(':');
        }
        if self.props.is_alnum(next_member) || next_member == '_' {
            return self.bare_key_is_followed_by_colon(next_member_idx);
        }
        false
    }

    /// `_only_whitespace_until(end)`: offsets `1..end` are all spaces (or past the end).
    fn only_whitespace_until(&mut self, end: isize) -> bool {
        if end <= 1 {
            return true;
        }
        let from = self.index + 1;
        if from < 0 {
            return (1..end).all(|j| {
                let c = self.get_char_at(j);
                c.is_none() || self.opt_space(c)
            });
        }
        let nns = self.next_non_space(from as usize) as isize;
        nns >= self.index + end || nns >= self.len()
    }

    /// `_handle_right_delimiter_candidate`: (handled, next char, break).
    #[allow(clippy::too_many_lines)]
    fn handle_right_delimiter_candidate(
        &mut self,
        st: &mut StrState,
        ch: char,
    ) -> (bool, Option<char>, bool) {
        let outer = st.rstring_delimiter;
        let l = st.lstring_delimiter;
        if st.doubled_quotes && self.get_char_at(1) == Some(outer) {
            self.log(|| "While parsing a string, we found a doubled quote, ignoring it".into());
            self.index += 1;
            return (true, Some(ch), false);
        }
        if st.missing_quotes && self.ctx_current() == Some(Ctx::ObjectValue) {
            let mut i = 1;
            let mut next_c = self.get_char_at(i);
            while next_c.is_some_and(|c| c != outer && c != l) {
                i += 1;
                next_c = self.get_char_at(i);
            }
            if next_c.is_some() {
                i += 1;
                i = self.scroll_whitespaces(i);
                if self.get_char_at(i) == Some(':') {
                    self.index -= 1;
                    let next_char = self.cur();
                    self.log(|| {
                        "In a string with missing quotes and object value context, I found a delimeter but it turns out it was the beginning on the next key. Stopping here.".into()
                    });
                    return (false, next_char, true);
                }
            }
            return (false, Some(ch), false);
        }
        if st.unmatched_delimiter {
            st.unmatched_delimiter = false;
            let next_char = self.append_literal_char(st, ch);
            return (true, next_char, false);
        }
        let in_key = self.ctx_has(Ctx::ObjectKey);
        let in_value = self.ctx_has(Ctx::ObjectValue);
        let in_array = self.ctx_has(Ctx::Array);
        let current_is_value = self.ctx_current() == Some(Ctx::ObjectValue);
        let mut i = 1;
        let mut next_c = self.get_char_at(i);
        let mut check_comma_in_object_value = true;
        while let Some(c) = next_c {
            if c == outer || c == l {
                break;
            }
            if check_comma_in_object_value && self.props.is_alpha(c) {
                check_comma_in_object_value = false;
            }
            if (in_key && matches!(c, ':' | '}'))
                || (in_value && c == '}')
                || (in_array && matches!(c, ']' | ','))
                || (check_comma_in_object_value && current_is_value && c == ',')
            {
                break;
            }
            i += 1;
            next_c = self.get_char_at(i);
        }
        if next_c == Some(',') && current_is_value {
            i += 1;
            i = self.skip_to_character(Targets::one(outer), i);
            i += 1;
            i = self.scroll_whitespaces(i);
            let next_c = self.get_char_at(i);
            if matches!(next_c, Some('}' | ',')) {
                self.log(|| {
                    "While parsing a string, we found a misplaced quote that would have closed the string but has a different meaning here, ignoring it".into()
                });
                let next_char = self.append_literal_char(st, ch);
                return (true, next_char, false);
            }
        } else if next_c == Some(outer) && self.get_char_at(i - 1) != Some('\\') {
            if self.only_whitespace_until(i)
                && !(current_is_value && self.quoted_object_member_follows(i))
            {
                return (false, Some(ch), true);
            }
            match self.ctx_current() {
                Some(Ctx::ObjectValue) => {
                    if self.quoted_object_member_follows(i) {
                        self.log(|| {
                            "While parsing a string, we found a misplaced quote that would have closed the string but has a different meaning here, ignoring it".into()
                        });
                        let next_char = self.append_literal_char(st, ch);
                        return (true, next_char, false);
                    }
                    i = self.skip_to_character(Targets::one(outer), i + 1);
                    i += 1;
                    let mut next_c = self.get_char_at(i);
                    while let Some(c) = next_c {
                        if c == ':' {
                            break;
                        }
                        if matches!(c, ',' | ']' | '}')
                            || (c == outer && self.get_char_at(i - 1) != Some('\\'))
                        {
                            break;
                        }
                        i += 1;
                        next_c = self.get_char_at(i);
                    }
                    if next_c != Some(':') {
                        self.log(|| {
                            "While parsing a string, we found a misplaced quote that would have closed the string but has a different meaning here, ignoring it".into()
                        });
                        st.unmatched_delimiter = !st.unmatched_delimiter;
                        let next_char = self.append_literal_char(st, ch);
                        return (true, next_char, false);
                    }
                }
                Some(Ctx::Array) => {
                    // json_repair walks the quotes pairwise to the next `]`: an
                    // even run of `outer` quotes keeps the string open. The walk
                    // is memoized (one pass per buffer), as it reruns per quote.
                    let p0 = self.index + i;
                    let even_delimiters = match self.quote_run_len(outer, p0 as usize) {
                        Some(k) => k % 2 == 0,
                        None => self.quote_run_even_slow(outer, i),
                    };
                    if even_delimiters {
                        self.log(|| {
                            "While parsing a string in Array context, we detected a quoted section that would have closed the string but has a different meaning here, ignoring it".into()
                        });
                        st.unmatched_delimiter = !st.unmatched_delimiter;
                        let next_char = self.append_literal_char(st, ch);
                        return (true, next_char, false);
                    }
                    return (false, Some(ch), true);
                }
                Some(Ctx::ObjectKey) => {
                    self.log(|| {
                        "While parsing a string in Object Key context, we detected a quoted section that would have closed the string but has a different meaning here, ignoring it".into()
                    });
                    let next_char = self.append_literal_char(st, ch);
                    return (true, next_char, false);
                }
                None => {}
            }
        }
        (false, Some(ch), false)
    }

    /// The literal pairwise walk of `_handle_right_delimiter_candidate`'s array
    /// branch, from the quote at offset `i`.
    fn quote_run_even_slow(&mut self, outer: char, mut i: isize) -> bool {
        let pair = Targets::two(outer, ']');
        let mut next_c = self.get_char_at(i);
        let mut even_delimiters = next_c == Some(outer);
        while next_c == Some(outer) {
            i = self.skip_to_character(pair, i + 1);
            next_c = self.get_char_at(i);
            if next_c != Some(outer) {
                even_delimiters = false;
                break;
            }
            i = self.skip_to_character(pair, i + 1);
            next_c = self.get_char_at(i);
        }
        even_delimiters
    }

    /// `classify_object_value_comma` (with json_repair's per-string cache).
    fn classify_object_value_comma(&mut self, st: &mut StrState) -> CommaClass {
        let next_idx = self.scroll_whitespaces(1);
        let Some(next_c) = self.get_char_at(next_idx) else {
            return CommaClass::Member;
        };
        if next_c == '}' {
            return CommaClass::Member;
        }
        if is_delim(next_c) {
            let key_end_idx = self.skip_to_character(Targets::one(next_c), next_idx + 1);
            if self.get_char_at(key_end_idx).is_none() {
                return CommaClass::Str;
            }
            let key_end_idx = self.scroll_whitespaces(key_end_idx + 1);
            return if self.get_char_at(key_end_idx) == Some(':') {
                CommaClass::Member
            } else {
                CommaClass::Str
            };
        }
        if next_c == '`' {
            let end = self.bare_key_end(next_idx + 1);
            let end = self.scroll_whitespaces(end);
            return if self.get_char_at(end) == Some(':') {
                CommaClass::Member
            } else {
                CommaClass::Str
            };
        }
        if self.props.is_alnum(next_c) || next_c == '_' {
            let end = self.bare_key_end(next_idx);
            let end = self.scroll_whitespaces(end);
            if self.get_char_at(end) == Some(':')
                && self.bare_member_has_recoverable_value(st, end + 1)
            {
                return CommaClass::Member;
            }
        }
        if next_c == '{' || next_c == '[' {
            return CommaClass::Container;
        }
        let next_special_idx =
            self.cached_skip_to_character(st, Targets::delims_and(&['{', '[']), next_idx);
        let Some(next_special) = self.get_char_at(next_special_idx) else {
            return CommaClass::StrNoFutureDelimiter;
        };
        if next_special == '{' || next_special == '[' {
            return CommaClass::Str;
        }
        let key_end_idx =
            self.cached_skip_to_character(st, Targets::one(next_special), next_special_idx + 1);
        if self.get_char_at(key_end_idx).is_none() {
            return CommaClass::Str;
        }
        let key_end_idx = self.scroll_whitespaces(key_end_idx + 1);
        if self.get_char_at(key_end_idx) == Some(':') {
            CommaClass::Member
        } else {
            CommaClass::Str
        }
    }

    /// `_bare_member_has_recoverable_value`.
    fn bare_member_has_recoverable_value(&mut self, st: &mut StrState, value_idx: isize) -> bool {
        let value_start_idx = self.scroll_whitespaces(value_idx);
        let value_start = self.get_char_at(value_start_idx);
        if let Some(v) = value_start {
            if is_delim(v) || matches!(v, '{' | '[' | '-') || self.props.is_digit(v) {
                return true;
            }
        }
        for literal in ["true", "false", "null"] {
            let all = literal
                .chars()
                .enumerate()
                .all(|(off, c)| self.get_char_at(value_start_idx + off as isize) == Some(c));
            if all {
                let value_end = self.get_char_at(value_start_idx + literal.len() as isize);
                if value_end.is_none()
                    || self.opt_space(value_end)
                    || matches!(value_end, Some(',' | '}' | ']'))
                {
                    return true;
                }
            }
        }
        let value_end_idx =
            self.cached_skip_to_character(st, Targets::delims_and(&['}']), value_start_idx);
        self.get_char_at(value_end_idx) == Some('}')
    }

    /// `update_inline_container_stack`: whether to keep `ch` as string content.
    fn update_inline_container_stack(st: &mut StrState, ch: char) -> bool {
        if ch == '{' || ch == '[' {
            if st.pending_inline_container {
                st.inline_container_stack.push(ch);
                st.pending_inline_container = false;
                return false;
            }
            if !st.inline_container_stack.is_empty() {
                st.inline_container_stack.push(ch);
            }
        }
        if let Some(&top) = st.inline_container_stack.last() {
            if (ch == '}' && top == '{') || (ch == ']' && top == '[') {
                st.inline_container_stack.pop();
                return true;
            }
        }
        false
    }

    /// `_scan_string_body`: the char that stopped the scan.
    #[allow(clippy::too_many_lines)]
    fn scan_string_body(&mut self, st: &mut StrState) -> Option<char> {
        let outer = st.rstring_delimiter;
        let mut ch = self.cur();
        while let Some(c) = ch {
            if c == outer && !st.in_low_span() {
                break;
            }
            if st.missing_quotes {
                if self.ctx_current() == Some(Ctx::ObjectKey) && (c == ':' || self.is_space(c)) {
                    self.log(|| {
                        "While parsing a string missing the left delimiter in object key context, we found a :, stopping here".into()
                    });
                    break;
                }
                if self.ctx_current() == Some(Ctx::Array) && (c == ']' || c == ',') {
                    self.log(|| {
                        "While parsing a string missing the left delimiter in array context, we found a ] or ,, stopping here".into()
                    });
                    break;
                }
            }
            if c == LOW_QUOTE && st.last_is_not('\\') {
                st.low_spans += 1;
                ch = self.append_literal_char(st, c);
                continue;
            }
            if st.in_low_span() && c == RIGHT_QUOTE {
                st.low_spans -= 1;
                ch = self.append_literal_char(st, c);
                continue;
            }
            if is_opener(c)
                && st.last_is_not('\\')
                && (st.pending_inline_container
                    || (self.ctx_current() == Some(Ctx::ObjectValue)
                        && c == '{'
                        && self.get_char_at(-1) != Some('\\')
                        && {
                            let k = self.scroll_whitespaces(1);
                            self.bare_key_is_followed_by_colon(k)
                        }))
            {
                if let Some(container_end_idx) = self.skip_inline_container(0) {
                    self.log(|| {
                        "While parsing a string in object value context, we found a balanced inline container that belongs to the string, keeping it".into()
                    });
                    st.pending_inline_container = false;
                    st.inline_container_stack.clear();
                    let content: Vec<char> = self
                        .slice(self.index, self.index + container_end_idx)
                        .to_vec();
                    self.acc_append(st, &content);
                    self.index += container_end_idx;
                    ch = self.cur();
                    continue;
                }
            }
            if !self.stream_stable
                && self.ctx_current() == Some(Ctx::ObjectValue)
                && c == ','
                && !st.pending_inline_container
                && st.inline_container_stack.is_empty()
            {
                let class = if st.object_value_has_no_future_delimiter {
                    CommaClass::Str
                } else {
                    self.classify_object_value_comma(st)
                };
                if class == CommaClass::Member {
                    self.log(|| {
                        "While parsing a string missing the right delimiter in object value context, we found a comma that starts the next object member. Stopping here".into()
                    });
                    break;
                }
                if class == CommaClass::StrNoFutureDelimiter {
                    st.object_value_has_no_future_delimiter = true;
                }
                st.pending_inline_container = class == CommaClass::Container;
                self.log(|| {
                    "While parsing a string in object value context, we found a comma that belongs to the string, keeping it".into()
                });
                ch = self.append_literal_char(st, c);
                continue;
            }
            if Self::update_inline_container_stack(st, c) {
                ch = self.append_literal_char(st, c);
                continue;
            }
            if !self.stream_stable
                && self.ctx_current() == Some(Ctx::ObjectValue)
                && c == '}'
                && st.last_is_not(outer)
            {
                if st.object_value_unmatched_opening_braces > 0 {
                    ch = self.append_literal_char(st, c);
                    continue;
                }
                let mut rstring_delimiter_missing = true;
                self.skip_whitespaces();
                if self.get_char_at(1) == Some('\\') {
                    rstring_delimiter_missing = false;
                }
                let mut i = self.cached_skip_to_character(st, Targets::one(outer), 1);
                let mut next_c = self.get_char_at(i);
                if next_c.is_some() {
                    i += 1;
                    i = self.scroll_whitespaces(i);
                    next_c = self.get_char_at(i);
                    if matches!(next_c, None | Some(',' | '}')) {
                        rstring_delimiter_missing = false;
                    } else {
                        i = self.skip_to_character(Targets::one(st.lstring_delimiter), i);
                        next_c = self.get_char_at(i);
                        if next_c.is_none() {
                            rstring_delimiter_missing = false;
                        } else {
                            i = self.scroll_whitespaces(i + 1);
                            next_c = self.get_char_at(i);
                            if next_c.is_some_and(|n| n != ':') {
                                rstring_delimiter_missing = false;
                            }
                        }
                    }
                } else {
                    let i = self.skip_to_character(Targets::one(':'), 1);
                    if self.get_char_at(i).is_some() {
                        break;
                    }
                    let i = self.scroll_whitespaces(1);
                    let j = self.skip_to_character(Targets::one('}'), i);
                    if j - i > 1 {
                        rstring_delimiter_missing = false;
                    }
                }
                if rstring_delimiter_missing {
                    self.log(|| {
                        "While parsing a string missing the left delimiter in object value context, we found a , or } and we couldn't determine that a right delimiter was present. Stopping here".into()
                    });
                    break;
                }
            }
            if !self.stream_stable && c == ']' && self.ctx_has(Ctx::Array) && st.last_is_not(outer)
            {
                let i = self.skip_to_character(Targets::one(outer), 0);
                if self.get_char_at(i).is_none() {
                    break;
                }
            }
            if self.ctx_current() == Some(Ctx::ObjectValue) && c == '}' {
                let i = self.scroll_whitespaces(1);
                let next_c = self.get_char_at(i);
                if next_c == Some('`')
                    && self.get_char_at(i + 1) == Some('`')
                    && self.get_char_at(i + 2) == Some('`')
                {
                    if self.brace_before_code_fence_belongs_to_string(st, i) {
                        self.log(|| {
                            "While parsing a string in object value context, we found a literal fenced snippet after }, keeping it in the string".into()
                        });
                        ch = self.append_literal_char(st, c);
                        continue;
                    }
                    self.log(|| {
                        "While parsing a string in object value context, we found a } that closes the object before code fences, stopping here".into()
                    });
                    break;
                }
                if next_c.is_none() {
                    self.log(|| {
                        "While parsing a string in object value context, we found a } that closes the object, stopping here".into()
                    });
                    break;
                }
            }
            self.acc_push(st, c);
            self.index += 1;
            ch = self.cur();
            let Some(c) = ch else {
                if self.stream_stable && st.last() == Some('\\') {
                    st.pop();
                }
                break;
            };
            if st.last() == Some('\\') {
                if let Some(next) = self.normalize_escape_sequence(st, c) {
                    ch = next;
                    continue;
                }
            }
            if c == ':' && !st.missing_quotes && self.ctx_current() == Some(Ctx::ObjectKey) {
                let i = self.skip_to_character(Targets::one(st.lstring_delimiter), 1);
                if self.get_char_at(i).is_some() {
                    let i = self.skip_to_character(Targets::one(outer), i + 1);
                    if self.get_char_at(i).is_some() {
                        let i = self.scroll_whitespaces(i + 1);
                        let next = self.get_char_at(i);
                        if let Some(n @ (',' | '}')) = next {
                            self.log(|| {
                                format!("While parsing a string missing the right delimiter in object key context, we found a {n} stopping here")
                            });
                            break;
                        }
                    }
                } else {
                    self.log(|| {
                        "While parsing a string missing the right delimiter in object key context, we found a :, stopping here".into()
                    });
                    break;
                }
            }
            if st.in_low_span() && c == '"' {
                st.low_spans -= 1;
                ch = self.append_literal_char(st, c);
                continue;
            }
            if c == outer
                && self.ctx_current() == Some(Ctx::ObjectValue)
                && self.quote_belongs_to_regex_character_class(st)
            {
                self.log(|| {
                    "While parsing a string, we found a bare quote inside a regex character class, keeping it".into()
                });
                ch = self.append_literal_char(st, c);
                continue;
            }
            if c == outer && st.last().is_some_and(|l| l != '\\') {
                let (handled, next, should_break) = self.handle_right_delimiter_candidate(st, c);
                ch = next;
                // Not handled: the loop condition sees the closing quote and stops.
                let _ = handled;
                if should_break {
                    break;
                }
            }
        }
        ch
    }

    /// `_finalize_string_result`.
    fn finalize_string_result(&mut self, st: &mut StrState, ch: Option<char>) -> Value {
        let outer = st.rstring_delimiter;
        if st.missing_quotes && self.ctx_current() == Some(Ctx::ObjectKey) && self.opt_space(ch) {
            self.log(|| {
                "While parsing a string, handling an extreme corner case in which the LLM added a comment instead of valid string, invalidate the string and return an empty value".into()
            });
            self.skip_whitespaces();
            if !matches!(self.cur(), Some(':' | ',')) {
                return Value::empty_str();
            }
        }
        if ch == Some(outer) {
            self.index += 1;
        } else if !self.stream_stable {
            self.log(|| "While parsing a string, we missed the closing quote, ignoring".into());
            self.acc_rstrip(st);
        }
        if !self.stream_stable && (st.missing_quotes || st.last() == Some('\n')) {
            self.acc_rstrip(st);
        }
        Value::Str(std::mem::take(&mut st.acc).into_iter().collect())
    }

    /// `json_str.find(c, from)` (plain, no escapes): the absolute index.
    fn find_plain(&self, c: char, from: usize) -> Option<usize> {
        (from..self.s.len()).find(|&i| self.s.at(i) == c)
    }
}

/// What `_prepare_string_entry` decided.
enum Entry {
    /// A value to return right away.
    Direct(Value),
    /// Scan the string body with this state.
    Scan(StrState),
}
