// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! `parser_parenthesized.py` and `JSONParser.parse_parenthesized`.

use super::super::value::Value;
use super::tuple::TupleIndex;
use super::{Parser, R, is_delim};

/// Chars a tuple scan runs directly before it switches to a [`TupleIndex`].
const DIRECT_BUDGET: usize = 1024;
/// Tuple indexes per parse (each covers the scans in one quote state).
const MAX_TUPLE_INDEXES: usize = 8;

impl Parser<'_> {
    /// `parenthesized_is_explicit_tuple`: whether the `(` at the index starts an
    /// explicit tuple (`()` and `(1,)` do, `(1)` does not).
    ///
    /// Short scans run directly; long ones are answered by a [`TupleIndex`]
    /// shared by every nested `(` (json_repair rescans to the end from each).
    fn parenthesized_is_explicit_tuple(&mut self) -> bool {
        if let Some(r) = self.explicit_tuple_direct(Some(DIRECT_BUDGET)) {
            return r;
        }
        let q = self.index as usize;
        // `saw_top_level_content` is set by the first char that is not a space or a
        // backslash, unless it is `)` or `,`; those end the scan (with `True`).
        let mut f = q + 1;
        while f < self.s.len() && (self.s.at(f) == '\\' || self.is_space(self.s.at(f))) {
            f += 1;
        }
        if f >= self.s.len() || matches!(self.s.at(f), ')' | ',') {
            return true;
        }
        let found = self.scan.tuples.iter().position(|ix| ix.covers(q));
        let ix = match found {
            Some(k) => &self.scan.tuples[k],
            None if self.scan.tuples.len() < MAX_TUPLE_INDEXES => {
                let s = &self.s;
                match TupleIndex::build(q, s.len(), |i| s.at(i)) {
                    Some(ix) => {
                        self.scan.tuples.push(ix);
                        self.scan.tuples.last().expect("pushed")
                    }
                    None => return self.explicit_tuple_direct(None).expect("unbounded"),
                }
            }
            None => return self.explicit_tuple_direct(None).expect("unbounded"),
        };
        // With content seen, the scan returns True only when it stops at a `,`.
        ix.terminator(q).is_some_and(|t| self.s.at(t) == ',')
    }

    /// The literal scan; `None` if it runs past `budget` chars.
    fn explicit_tuple_direct(&self, budget: Option<usize>) -> Option<bool> {
        let n = self.len();
        let mut i = self.index + 1;
        let stop = budget.map_or(n, |b| n.min(i.saturating_add(b as isize)));
        let (mut nested, mut square, mut braces) = (0usize, 0usize, 0usize);
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        let mut saw_top_level_content = false;
        while i < stop {
            let Some(ch) = self.char_at(i) else { break };
            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }
            if let Some(q) = in_quote {
                if ch == q && backslashes % 2 == 0 {
                    in_quote = None;
                }
                backslashes = 0;
                i += 1;
                continue;
            }
            if is_delim(ch) && backslashes % 2 == 0 {
                in_quote = Some(ch);
                saw_top_level_content =
                    saw_top_level_content || (nested == 0 && square == 0 && braces == 0);
                backslashes = 0;
                i += 1;
                continue;
            }
            backslashes = 0;
            let top = nested == 0 && square == 0 && braces == 0;
            if !self.is_space(ch) && ch != ',' && ch != ')' && top {
                saw_top_level_content = true;
            }
            match ch {
                '(' => nested += 1,
                ')' => {
                    if top {
                        return Some(!saw_top_level_content);
                    }
                    nested = nested.saturating_sub(1);
                }
                '[' => square += 1,
                ']' if square > 0 => square -= 1,
                '{' => braces += 1,
                '}' if braces > 0 => braces -= 1,
                ',' if top => return Some(true),
                _ => {}
            }
            i += 1;
        }
        if i < n {
            return None;
        }
        Some(!saw_top_level_content)
    }

    /// `top_level_parenthesized_can_start_value`: whether a top-level `(` looks
    /// like a standalone value rather than inline prose.
    pub(crate) fn top_level_parenthesized_can_start_value(&mut self) -> bool {
        let mut i = self.index - 1;
        while i >= 0 {
            let ch = self.s.at(i as usize);
            if ch == '\n' || ch == '\r' {
                break;
            }
            if !self.is_space(ch) {
                return false;
            }
            i -= 1;
        }
        let idx = self.scroll_whitespaces(1);
        let Some(first) = self.get_char_at(idx) else {
            return false;
        };
        // json_repair lowercases the whole rest of the input and compares its
        // first 4/5 chars; only ASCII letters lower to the letters of these
        // words, so comparing the next chars case-insensitively is the same.
        let starts_with = |word: &str| {
            word.chars().enumerate().all(|(k, w)| {
                self.get_char_at(idx + k as isize)
                    .is_some_and(|c| c.to_ascii_lowercase() == w)
            })
        };
        if !matches!(first, ')' | '{' | '[' | '(')
            && !is_delim(first)
            && !self.props.is_digit(first)
            && first != '-'
            && first != '.'
            && !starts_with("true")
            && !starts_with("null")
            && !starts_with("none")
            && !starts_with("false")
        {
            return false;
        }
        let n = self.len();
        let mut i = self.index + 1;
        let (mut nested, mut square, mut braces) = (0isize, 0usize, 0usize);
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        while i < n {
            let ch = self.s.at(i as usize);
            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }
            if let Some(q) = in_quote {
                if ch == q && backslashes % 2 == 0 {
                    in_quote = None;
                }
                backslashes = 0;
                i += 1;
                continue;
            }
            if is_delim(ch) && backslashes % 2 == 0 {
                in_quote = Some(ch);
                backslashes = 0;
                i += 1;
                continue;
            }
            backslashes = 0;
            match ch {
                '(' => nested += 1,
                ')' => {
                    if nested == 0 && square == 0 && braces == 0 {
                        i += 1;
                        while i < n {
                            let trailer = self.s.at(i as usize);
                            if trailer == '\n' || trailer == '\r' {
                                return true;
                            }
                            if !self.is_space(trailer) {
                                return false;
                            }
                            i += 1;
                        }
                        return true;
                    }
                    nested -= 1;
                }
                '[' => square += 1,
                ']' if square > 0 => square -= 1,
                '{' => braces += 1,
                '}' if braces > 0 => braces -= 1,
                _ => {}
            }
            i += 1;
        }
        true
    }

    /// `parse_parenthesized()`.
    pub(crate) fn parse_parenthesized(&mut self) -> R<Value> {
        let explicit_tuple = self.parenthesized_is_explicit_tuple();
        self.index += 1;
        let mut values = self.parse_array(')')?;
        if explicit_tuple || values.len() != 1 {
            return Ok(Value::Array(values));
        }
        Ok(values.pop().expect("one value"))
    }
}
