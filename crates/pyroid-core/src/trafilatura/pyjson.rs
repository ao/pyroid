// Reproduces CPython 3.11-3.15 behaviour (PSF-2.0); character data from each version's unicodedata.
//! `json.loads` (CPython 3.11-3.15, the C scanner) for the values trafilatura's baseline reads
//! from JSON-LD and Discourse preloads: structure and strings, numbers only checked.
//!
//! Python strings may hold lone surrogates (`"\ud800"`), Rust strings cannot: each lone
//! surrogate U+D800+k is decoded as the private-use character U+F0000+k. The baseline only
//! passes such strings through `html.unescape` and `remove_control_characters`, which treat
//! both alike (any non-printable character, dropped by the latter).

// index names follow the C scanner (i, j, k, s)
#![allow(clippy::many_single_char_names)]

use indexmap::IndexMap;
use rustc_hash::FxBuildHasher;

type Map = IndexMap<String, Value, FxBuildHasher>;

/// A decoded JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// A number, by its truth value (`bool(x)`: not zero; NaN and infinities are true).
    Number(bool),
    Str(String),
    List(Vec<Value>),
    /// A dict: insertion order, a repeated key keeps its first position and its last value.
    Dict(Map),
}

/// `json.loads` failed (JSONDecodeError, or ValueError for an integer of over 4300 digits).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error;

/// Nesting deeper than this is reported as an error (CPython raises RecursionError somewhere
/// around here; the exact depth depends on the interpreter and the caller's stack).
const MAX_DEPTH: usize = 1000;

struct Parser<'a> {
    s: &'a [char],
    strict: bool,
}

fn ws(s: &[char], mut i: usize) -> usize {
    while i < s.len() && matches!(s[i], ' ' | '\t' | '\n' | '\r') {
        i += 1;
    }
    i
}

impl Parser<'_> {
    fn hex4(&self, at: usize) -> Result<u32, Error> {
        let mut v = 0u32;
        for k in 0..4 {
            let d = self
                .s
                .get(at + k)
                .and_then(|c| c.to_digit(16))
                .ok_or(Error)?;
            v = v * 16 + d;
        }
        Ok(v)
    }

    /// scanstring_unicode: `i` is just after the opening quote; returns the string and the
    /// index after the closing quote.
    fn string(&self, mut i: usize) -> Result<(String, usize), Error> {
        let s = self.s;
        let mut out = String::new();
        loop {
            let c = *s.get(i).ok_or(Error)?;
            if c == '"' {
                return Ok((out, i + 1));
            }
            if c != '\\' {
                if self.strict && u32::from(c) < 0x20 {
                    return Err(Error);
                }
                out.push(c);
                i += 1;
                continue;
            }
            let e = *s.get(i + 1).ok_or(Error)?;
            if e != 'u' {
                out.push(match e {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return Err(Error),
                });
                i += 2;
                continue;
            }
            // \uXXXX: `end = next + 4; if (end >= len) error` (the closing quote must follow)
            let next = i + 2;
            let mut end = next + 4;
            if end >= s.len() {
                return Err(Error);
            }
            let mut c = self.hex4(next)?;
            if (0xD800..0xDC00).contains(&c)
                && end + 6 < s.len()
                && s[end] == '\\'
                && s[end + 1] == 'u'
            {
                let c2 = self.hex4(end + 2)?;
                if (0xDC00..0xE000).contains(&c2) {
                    c = 0x10000 + ((c - 0xD800) << 10) + (c2 - 0xDC00);
                    end += 6;
                }
            }
            let ch = if (0xD800..0xE000).contains(&c) {
                char::from_u32(0xF0000 + (c - 0xD800))
            } else {
                char::from_u32(c)
            };
            out.push(ch.ok_or(Error)?);
            i = end;
        }
    }

    /// _match_number_unicode at `i`: the index after the number, and whether it is not zero
    /// (a float literal as `float()` rounds it: `1e-400` is zero).
    fn number(&self, i: usize) -> Result<(usize, bool), Error> {
        let s = self.s;
        let mut j = i;
        if s.get(j) == Some(&'-') {
            j += 1;
        }
        let int_start = j;
        match s.get(j) {
            Some('0') => j += 1,
            Some('1'..='9') => {
                while s.get(j).is_some_and(char::is_ascii_digit) {
                    j += 1;
                }
            }
            _ => return Err(Error),
        }
        let int_digits = j - int_start;
        let mut is_float = false;
        if s.get(j) == Some(&'.') && s.get(j + 1).is_some_and(char::is_ascii_digit) {
            is_float = true;
            j += 2;
            while s.get(j).is_some_and(char::is_ascii_digit) {
                j += 1;
            }
        }
        if matches!(s.get(j), Some('e' | 'E')) {
            let mut k = j + 1;
            if matches!(s.get(k), Some('+' | '-')) {
                k += 1;
            }
            if s.get(k).is_some_and(char::is_ascii_digit) {
                while s.get(k).is_some_and(char::is_ascii_digit) {
                    k += 1;
                }
                is_float = true;
                j = k;
            }
        }
        // int() of more than 4300 digits raises ValueError
        if !is_float && int_digits > 4300 {
            return Err(Error);
        }
        let nonzero = if is_float {
            let lit: String = s[i..j].iter().collect();
            lit.parse::<f64>().is_ok_and(|x| x != 0.0)
        } else {
            s[int_start..int_start + int_digits]
                .iter()
                .any(|&c| c != '0')
        };
        Ok((j, nonzero))
    }

    fn lit(&self, i: usize, w: &str) -> bool {
        w.chars()
            .enumerate()
            .all(|(k, c)| self.s.get(i + k) == Some(&c))
    }

    /// scan_once at `i` (iteratively: containers are kept on an explicit stack).
    fn value(&self, i: usize) -> Result<(Value, usize), Error> {
        enum Frame {
            List(Vec<Value>),
            Dict(Map, String),
        }
        let s = self.s;
        let mut stack: Vec<Frame> = Vec::new();
        let mut i = i;
        loop {
            // parse one value at i (or open a container)
            let (mut v, mut j) = match s.get(i) {
                Some('"') => {
                    let (st, j) = self.string(i + 1)?;
                    (Value::Str(st), j)
                }
                Some('{') => {
                    let k = ws(s, i + 1);
                    if s.get(k) == Some(&'}') {
                        (Value::Dict(Map::with_hasher(FxBuildHasher)), k + 1)
                    } else {
                        if s.get(k) != Some(&'"') {
                            return Err(Error);
                        }
                        let (key, k) = self.string(k + 1)?;
                        let k = ws(s, k);
                        if s.get(k) != Some(&':') {
                            return Err(Error);
                        }
                        if stack.len() >= MAX_DEPTH {
                            return Err(Error);
                        }
                        stack.push(Frame::Dict(Map::with_hasher(FxBuildHasher), key));
                        i = ws(s, k + 1);
                        continue;
                    }
                }
                Some('[') => {
                    let k = ws(s, i + 1);
                    if s.get(k) == Some(&']') {
                        (Value::List(Vec::new()), k + 1)
                    } else {
                        if stack.len() >= MAX_DEPTH {
                            return Err(Error);
                        }
                        stack.push(Frame::List(Vec::new()));
                        i = k;
                        continue;
                    }
                }
                Some('n') if self.lit(i, "null") => (Value::Null, i + 4),
                Some('t') if self.lit(i, "true") => (Value::Bool(true), i + 4),
                Some('f') if self.lit(i, "false") => (Value::Bool(false), i + 5),
                Some('N') if self.lit(i, "NaN") => (Value::Number(true), i + 3),
                Some('I') if self.lit(i, "Infinity") => (Value::Number(true), i + 8),
                Some('-') if self.lit(i, "-Infinity") => (Value::Number(true), i + 9),
                Some('-' | '0'..='9') => {
                    let (j, nonzero) = self.number(i)?;
                    (Value::Number(nonzero), j)
                }
                _ => return Err(Error),
            };
            // hand the value to the enclosing containers, closing those that end here
            loop {
                match stack.last_mut() {
                    None => return Ok((v, j)),
                    Some(Frame::List(items)) => {
                        items.push(v);
                        let k = ws(s, j);
                        match s.get(k) {
                            Some(',') => {
                                i = ws(s, k + 1);
                                break;
                            }
                            Some(']') => {
                                let Some(Frame::List(items)) = stack.pop() else {
                                    return Err(Error);
                                };
                                v = Value::List(items);
                                j = k + 1;
                            }
                            _ => return Err(Error),
                        }
                    }
                    Some(Frame::Dict(map, key)) => {
                        map.insert(std::mem::take(key), v);
                        let k = ws(s, j);
                        match s.get(k) {
                            Some(',') => {
                                let k = ws(s, k + 1);
                                if s.get(k) != Some(&'"') {
                                    return Err(Error);
                                }
                                let (nk, k) = self.string(k + 1)?;
                                let k = ws(s, k);
                                if s.get(k) != Some(&':') {
                                    return Err(Error);
                                }
                                *key = nk;
                                i = ws(s, k + 1);
                                break;
                            }
                            Some('}') => {
                                let Some(Frame::Dict(map, _)) = stack.pop() else {
                                    return Err(Error);
                                };
                                v = Value::Dict(map);
                                j = k + 1;
                            }
                            _ => return Err(Error),
                        }
                    }
                }
            }
        }
    }
}

/// Whether `s` holds a lone surrogate (one of the private-use characters standing for them).
pub fn has_surrogate(s: &str) -> bool {
    s.chars().any(|c| ('\u{F0000}'..='\u{F07FF}').contains(&c))
}

/// `s.encode("utf-8", "surrogatepass")` for the Python string `s` stands for.
pub fn to_wtf8(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        let u = u32::from(c);
        if (0xF0000..=0xF07FF).contains(&u) {
            let surrogate = 0xD800 + (u - 0xF0000);
            // three bytes, as UTF-8 would write a code point in this range
            out.extend_from_slice(&[
                0xE0 | u8::try_from(surrogate >> 12).unwrap_or(0),
                0x80 | u8::try_from((surrogate >> 6) & 0x3F).unwrap_or(0),
                0x80 | u8::try_from(surrogate & 0x3F).unwrap_or(0),
            ]);
        } else {
            let mut b = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut b).as_bytes());
        }
    }
    out
}

/// `json.loads(s, strict=strict)`.
pub fn loads(s: &str, strict: bool) -> Result<Value, Error> {
    if s.starts_with('\u{feff}') {
        return Err(Error);
    }
    let cs: Vec<char> = s.chars().collect();
    let p = Parser { s: &cs, strict };
    let i = ws(&cs, 0);
    let (v, j) = p.value(i)?;
    if ws(&cs, j) != cs.len() {
        return Err(Error);
    }
    Ok(v)
}

impl Value {
    /// `dict.get(key)` (None for other values or a missing key).
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Dict(m) => m.get(key),
            _ => None,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Whether `needle in str(self)`: for a str its text; for lists and dicts their repr, which
    /// holds an ASCII-letter needle only within one of the strings (keys included); for other
    /// values never (for the needles used here).
    pub fn str_contains(&self, needle: &str) -> bool {
        let mut stack = vec![self];
        while let Some(v) = stack.pop() {
            match v {
                Value::Str(s) => {
                    if s.contains(needle) {
                        return true;
                    }
                }
                Value::List(items) => stack.extend(items.iter()),
                Value::Dict(m) => {
                    if m.keys().any(|k| k.contains(needle)) {
                        return true;
                    }
                    stack.extend(m.values());
                }
                _ => {}
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        let v = loads(
            r#" {"a": [1, 2.5e3, -0, true, null, "xé😀"], "a": "last", "b": {}} "#,
            false,
        )
        .unwrap();
        let Value::Dict(m) = &v else { panic!() };
        assert_eq!(m.keys().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(v.get("a"), Some(&Value::Str("last".into())));
        let v = loads(r#"["xé😀", "\ud800", NaN, -Infinity]"#, false).unwrap();
        let Value::List(l) = v else { panic!() };
        assert_eq!(l[0], Value::Str("xé😀".into()));
        assert_eq!(l[1], Value::Str("\u{F0000}".into()));
        assert!(loads("[1,]", false).is_err());
        assert!(loads("{\"a\":1,}", false).is_err());
        assert!(loads("\"a\u{1}\"", true).is_err());
        assert!(loads("\"a\u{1}\"", false).is_ok());
        assert!(loads("\u{feff}1", false).is_err());
        assert!(loads("01", false).is_err());
        assert!(loads("1 2", false).is_err());
        assert!(loads(&"1".repeat(4301), false).is_err());
        assert!(loads(&format!("{}.0", "1".repeat(4301)), false).is_ok());
        assert!(loads(&"[".repeat(2000), false).is_err());
        assert!(loads(r#""\x""#, false).is_err());
        assert!(loads(r#""\u12""#, false).is_err());
        let v = loads("[0, -0, 0.0, 1e-400, 1, 0.5, 1e400, -0e5, NaN]", true).unwrap();
        let Value::List(l) = v else { panic!() };
        let truth: Vec<bool> = l.iter().map(|x| matches!(x, Value::Number(true))).collect();
        assert_eq!(
            truth,
            [false, false, false, false, true, true, true, false, true]
        );
        assert!(has_surrogate("a\u{F0000}"));
        assert_eq!(
            to_wtf8("a\u{F0000}\u{F07FF}é"),
            b"a\xed\xa0\x80\xed\xbf\xbf\xc3\xa9"
        );
    }

    #[test]
    fn contains() {
        let v = loads(r#"{"@type": ["Thing", {"x": "a Product b"}]}"#, false).unwrap();
        assert!(v.get("@type").is_some_and(|t| t.str_contains("Product")));
        assert!(
            !v.get("@type")
                .is_some_and(|t| t.str_contains("VideoObject"))
        );
    }
}
