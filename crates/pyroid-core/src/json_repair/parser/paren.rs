// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! `parser_parenthesized.py` and `JSONParser.parse_parenthesized`.

use super::super::value::Value;
use super::{Parser, R, is_delim};

impl Parser<'_> {
    /// `parenthesized_is_explicit_tuple`: whether the `(` at the index starts an
    /// explicit tuple (`()` and `(1,)` do, `(1)` does not).
    fn parenthesized_is_explicit_tuple(&self) -> bool {
        let n = self.len();
        let mut i = self.index + 1;
        let (mut nested, mut square, mut braces) = (0usize, 0usize, 0usize);
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        let mut saw_top_level_content = false;
        while i < n {
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
                        return !saw_top_level_content;
                    }
                    nested = nested.saturating_sub(1);
                }
                '[' => square += 1,
                ']' if square > 0 => square -= 1,
                '{' => braces += 1,
                '}' if braces > 0 => braces -= 1,
                ',' if top => return true,
                _ => {}
            }
            i += 1;
        }
        !saw_top_level_content
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
