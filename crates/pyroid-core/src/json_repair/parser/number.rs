// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! `parse_number.py`.

use super::super::value::{Int, Value};
use super::{Ctx, Parser, R};

/// `NUMBER_CHARS`.
#[inline]
fn is_number_char(c: char) -> bool {
    matches!(c, '0'..='9' | '-' | '.' | 'e' | 'E' | '/' | ',' | '_')
}

/// Python's `float(text)` for text over `[0-9.eE+-]` (`None`: ValueError).
pub(crate) fn py_float(text: &str) -> Option<f64> {
    let b = text.as_bytes();
    let mut i = 0;
    if matches!(b.first(), Some(b'+' | b'-')) {
        i += 1;
    }
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let int_digits = i - int_start;
    let mut frac_digits = 0;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let f = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        frac_digits = i - f;
    }
    if int_digits + frac_digits == 0 {
        return None;
    }
    if i < b.len() && matches!(b[i], b'e' | b'E') {
        i += 1;
        if i < b.len() && matches!(b[i], b'+' | b'-') {
            i += 1;
        }
        let e = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == e {
            return None;
        }
    }
    if i != b.len() {
        return None;
    }
    text.parse::<f64>().ok()
}

impl Parser<'_> {
    /// `parse_number()`.
    pub(crate) fn parse_number(&mut self) -> R<Value> {
        let number_start_index = self.index;
        let mut number_str = String::new();
        let mut ch = self.cur();
        let is_array = self.ctx_current() == Some(Ctx::Array);
        while let Some(c) = ch {
            let ok = is_number_char(c)
                || (c == '+' && (number_str.ends_with('e') || number_str.ends_with('E')));
            if !ok || (is_array && c == ',') {
                break;
            }
            if c != '_' {
                number_str.push(c);
            }
            self.index += 1;
            ch = self.cur();
        }
        if self.cur().is_some_and(|c| self.props.is_alpha(c)) {
            // This was a string instead, sorry.
            self.index = number_start_index;
            return self.parse_string();
        }
        if number_str
            .chars()
            .last()
            .is_some_and(|c| matches!(c, '+' | '-' | 'e' | 'E' | '/' | ','))
        {
            // The number ends with a non valid character for a number/currency.
            number_str.pop();
            self.index -= 1;
        }
        if number_str.contains(',') {
            return Ok(Value::Str(number_str));
        }
        if number_str.contains(['.', 'e', 'E']) {
            return Ok(match py_float(&number_str) {
                Some(f) => Value::Float(f),
                None => Value::Str(number_str),
            });
        }
        Ok(match Int::parse_py(&number_str, self.max_int_digits) {
            Ok(i) => Value::Int(i),
            Err(_) => Value::Str(number_str),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::py_float;

    #[test]
    fn float_grammar_matches_python() {
        for (s, v) in [
            ("1.", Some(1.0)),
            (".5", Some(0.5)),
            ("-.5", Some(-0.5)),
            ("1.e5", Some(1e5)),
            ("1E+2", Some(100.0)),
            ("1e400", Some(f64::INFINITY)),
            ("-0.0", Some(-0.0)),
        ] {
            assert_eq!(py_float(s), v, "{s}");
        }
        for s in [
            ".", "e5", "1e", "1e+", "-", "1.2.3", "1-2", "1/2", "--1", ".e1", "+",
        ] {
            assert_eq!(py_float(s), None, "{s}");
        }
    }
}
