// Derived from CPython's json module (PSF-2.0) behaviour; written from scratch.
//! Python's `json.loads` / `json.JSONDecoder().raw_decode` with default
//! arguments (the C scanner, `strict=True`), producing [`Value`]s.
//!
//! One scanner serves both entry points through the small [`Source`] trait:
//! [`loads`] scans the UTF-8 bytes of a `&str` (memchr over string bodies; a
//! string without escapes is a single copy), [`raw_decode`] scans a `&[char]`.
//! Indices are byte offsets in the former and char indices in the latter.
//!
//! The scanner keeps open containers on an explicit stack, so nesting depth is
//! bounded by memory only and the time is linear. CPython itself raises
//! `RecursionError` on deep input (measured: 3.11 fails at depth 1000, 3.12 and
//! 3.13 by 20 000, 3.14 and 3.15 by 100 000); that limit is not modelled here.
//!
//! Failures are reported in CPython's scan order: the first problem met while
//! scanning left to right decides between [`DecodeError::Json`] and
//! [`DecodeError::Int`].

use memchr::memchr2;

use super::chars::{SURROGATE_BASE, surrogate_char};
use super::value::{Int, IntError, Object, Value, drop_deep, new_object};

/// Why decoding failed.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodeError {
    /// `json.JSONDecodeError` (a `ValueError`).
    Json,
    /// A plain `ValueError` from `int()`: too many digits.
    Int(IntError),
    /// A `\uXXXX\uXXXX` surrogate-pair escape decodes to a char in
    /// U+10F800..U+10FFFF, which the kernel reserves for remapped lone
    /// surrogates (see [`super::chars`]); the caller delegates the input.
    ReservedCodePoint,
}

/// `json.loads(s)`.
///
/// `s` may hold remapped lone surrogates (they pass through string bodies
/// unchanged); a `str` straight from Python must not hold U+10F800..U+10FFFF
/// (check with [`super::chars::contains_reserved`]).
pub fn loads(s: &str, max_int_digits: usize) -> Result<Value, DecodeError> {
    if s.starts_with('\u{feff}') {
        // "Unexpected UTF-8 BOM (decode using utf-8-sig)".
        return Err(DecodeError::Json);
    }
    let src = Utf8 { s };
    let (value, end) = scan(&src, skip_ws(&src, 0), max_int_digits)?;
    if skip_ws(&src, end) != s.len() {
        // "Extra data".
        drop_deep(value);
        return Err(DecodeError::Json);
    }
    Ok(value)
}

/// `json.JSONDecoder().raw_decode(text[start:])`, returning the value and the
/// absolute char index just past it.
pub fn raw_decode(
    text: &[char],
    start: usize,
    max_int_digits: usize,
) -> Result<(Value, usize), DecodeError> {
    scan(&Chars { t: text }, start, max_int_digits)
}

/// [`Source::at`] past the end (never a UTF-8 byte). Matches no grammar byte.
const END: u8 = 0xFF;
/// [`Chars::at`] for a non-ASCII char. Matches no grammar byte.
const NON_ASCII: u8 = 0x80;

/// The text being scanned.
trait Source {
    /// The unit at `i` if it is ASCII, else a byte >= 0x80 (the grammar is all
    /// ASCII, so which one does not matter); [`END`] past the end.
    fn at(&self, i: usize) -> u8;
    /// Decode the string body starting at `i` (just past the opening quote):
    /// the text and the index just past the closing quote.
    fn string(&self, i: usize) -> Result<(String, usize), DecodeError>;
    /// The (ASCII) number text in `start..end`, in `buf` if need be.
    fn number<'a>(&'a self, start: usize, end: usize, buf: &'a mut String) -> &'a str;
}

/// A `&str`, scanned as bytes.
struct Utf8<'a> {
    s: &'a str,
}

impl Source for Utf8<'_> {
    #[inline]
    fn at(&self, i: usize) -> u8 {
        self.s.as_bytes().get(i).copied().unwrap_or(END)
    }

    fn string(&self, start: usize) -> Result<(String, usize), DecodeError> {
        let b = self.s.as_bytes();
        let mut i = start;
        let mut out: Option<String> = None;
        loop {
            // Every index here sits just past an ASCII byte, so on a char boundary.
            let Some(off) = memchr2(b'"', b'\\', &b[i..]) else {
                return Err(DecodeError::Json); // unterminated
            };
            let j = i + off;
            if has_control(&b[i..j]) {
                return Err(DecodeError::Json); // "Invalid control character"
            }
            let chunk = &self.s[i..j];
            if b[j] == b'"' {
                let text = match out {
                    None => chunk.to_owned(),
                    Some(mut text) => {
                        text.push_str(chunk);
                        text
                    }
                };
                return Ok((text, j + 1));
            }
            let text = out.get_or_insert_with(|| String::with_capacity(chunk.len() + 16));
            text.push_str(chunk);
            i = escape(self, j + 1, text)?;
        }
    }

    fn number<'a>(&'a self, start: usize, end: usize, _buf: &'a mut String) -> &'a str {
        &self.s[start..end]
    }
}

/// Whether `b` holds a byte below 0x20 (branch-free, so it vectorizes).
#[inline]
fn has_control(b: &[u8]) -> bool {
    b.iter().fold(0xFF, |m: u8, &x| m.min(x)) < 0x20
}

/// A `&[char]`.
struct Chars<'a> {
    t: &'a [char],
}

impl Source for Chars<'_> {
    #[inline]
    fn at(&self, i: usize) -> u8 {
        match self.t.get(i) {
            None => END,
            Some(&c) => match u8::try_from(c) {
                Ok(b) if b.is_ascii() => b,
                _ => NON_ASCII,
            },
        }
    }

    fn string(&self, start: usize) -> Result<(String, usize), DecodeError> {
        let t = self.t;
        let mut out = String::new();
        let mut i = start;
        loop {
            let run = i;
            while let Some(&c) = t.get(i) {
                if c == '"' || c == '\\' || c < ' ' {
                    break;
                }
                i += 1;
            }
            out.reserve(i - run);
            out.extend(&t[run..i]);
            match t.get(i) {
                Some('"') => return Ok((out, i + 1)),
                Some('\\') => i = escape(self, i + 1, &mut out)?,
                // A control character, or unterminated.
                _ => return Err(DecodeError::Json),
            }
        }
    }

    fn number<'a>(&'a self, start: usize, end: usize, buf: &'a mut String) -> &'a str {
        buf.clear();
        buf.extend(&self.t[start..end]);
        buf
    }
}

/// `[ \t\n\r]*` from `i`.
#[inline]
fn skip_ws<S: Source>(src: &S, mut i: usize) -> usize {
    while matches!(src.at(i), b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    i
}

/// Whether `word` is at `i`.
#[inline]
fn word_at<S: Source>(src: &S, i: usize, word: &[u8]) -> bool {
    word.iter().zip(i..).all(|(&w, k)| src.at(k) == w)
}

/// Decode the escape whose backslash is just before `i`; returns the index past it.
fn escape<S: Source>(src: &S, i: usize, out: &mut String) -> Result<usize, DecodeError> {
    let c = match src.at(i) {
        b'"' => '"',
        b'\\' => '\\',
        b'/' => '/',
        b'b' => '\u{8}',
        b'f' => '\u{c}',
        b'n' => '\n',
        b'r' => '\r',
        b't' => '\t',
        b'u' => return unicode_escape(src, i + 1, out),
        _ => return Err(DecodeError::Json), // "Invalid \\escape"
    };
    out.push(c);
    Ok(i + 1)
}

/// Four hex digits (either case) at `i`.
#[inline]
fn hex4<S: Source>(src: &S, i: usize) -> Option<u32> {
    (i..i + 4).try_fold(0, |v, k| {
        Some((v << 4) | char::from(src.at(k)).to_digit(16)?)
    })
}

/// The `\uXXXX` escape whose hex digits start at `i`. A high surrogate directly
/// followed by a `\u` low surrogate is one char; any other surrogate is lone.
fn unicode_escape<S: Source>(src: &S, i: usize, out: &mut String) -> Result<usize, DecodeError> {
    let hi = hex4(src, i).ok_or(DecodeError::Json)?;
    let mut end = i + 4;
    let c = match hi {
        0xD800..=0xDBFF if src.at(end) == b'\\' && src.at(end + 1) == b'u' => {
            // CPython reads these four digits too (and fails on bad hex) before
            // deciding whether they make a pair.
            let lo = hex4(src, end + 2).ok_or(DecodeError::Json)?;
            if (0xDC00..=0xDFFF).contains(&lo) {
                end += 6;
                let cp = 0x1_0000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                if cp >= SURROGATE_BASE {
                    return Err(DecodeError::ReservedCodePoint);
                }
                char::from_u32(cp).ok_or(DecodeError::Json)?
            } else {
                surrogate_char(hi)
            }
        }
        0xD800..=0xDFFF => surrogate_char(hi),
        _ => char::from_u32(hi).ok_or(DecodeError::Json)?,
    };
    out.push(c);
    Ok(end)
}

/// CPython's `_match_number_unicode` at `start` (no value matched there yet).
fn number<S: Source>(
    src: &S,
    start: usize,
    max_digits: usize,
    buf: &mut String,
) -> Result<(Value, usize), DecodeError> {
    let mut i = start;
    let neg = src.at(i) == b'-';
    if neg {
        i += 1;
    }
    let first = i;
    match src.at(i) {
        b'0' => i += 1,
        b'1'..=b'9' => {
            i += 1;
            while src.at(i).is_ascii_digit() {
                i += 1;
            }
        }
        _ => return Err(DecodeError::Json), // "Expecting value"
    }
    let int_digits = i - first;
    let mut is_float = false;
    if src.at(i) == b'.' && src.at(i + 1).is_ascii_digit() {
        is_float = true;
        i += 2;
        while src.at(i).is_ascii_digit() {
            i += 1;
        }
    }
    if matches!(src.at(i), b'e' | b'E') {
        // Only `[eE][+-]?[0-9]+` is an exponent; otherwise the number ends before the `e`.
        let mut k = i + 1;
        if matches!(src.at(k), b'+' | b'-') {
            k += 1;
        }
        let digits = k;
        while src.at(k).is_ascii_digit() {
            k += 1;
        }
        if k > digits {
            is_float = true;
            i = k;
        }
    }
    if !is_float && int_digits <= 18 && (max_digits == 0 || int_digits <= max_digits) {
        let v = (first..i).fold(0i64, |v, k| v * 10 + i64::from(src.at(k) - b'0'));
        return Ok((Value::Int(Int::Small(if neg { -v } else { v })), i));
    }
    let text = src.number(start, i, buf);
    let value = if is_float {
        // Correctly rounded, like CPython's dtoa; overflow gives +-inf.
        Value::Float(text.parse().map_err(|_| DecodeError::Json)?)
    } else {
        Value::Int(Int::parse_py(text, max_digits).map_err(DecodeError::Int)?)
    };
    Ok((value, i))
}

/// An open container.
enum Frame {
    Array(Vec<Value>),
    /// The object so far and the key whose value is being scanned.
    Object(Object, String),
}

impl Frame {
    fn into_value(self) -> Value {
        match self {
            Self::Array(a) => Value::Array(a),
            Self::Object(o, _) => Value::Object(o),
        }
    }
}

/// The open containers, innermost last. Dropped iteratively (a completed
/// member may be arbitrarily deep).
struct Stack(Vec<Frame>);

impl Drop for Stack {
    fn drop(&mut self) {
        for frame in self.0.drain(..) {
            drop_deep(frame.into_value());
        }
    }
}

/// `"key" ws : ws` at `i`; returns the key and the index of the value.
fn key<S: Source>(src: &S, i: usize) -> Result<(String, usize), DecodeError> {
    if src.at(i) != b'"' {
        return Err(DecodeError::Json); // "Expecting property name enclosed in double quotes"
    }
    let (key, j) = src.string(i + 1)?;
    let j = skip_ws(src, j);
    if src.at(j) != b':' {
        return Err(DecodeError::Json); // "Expecting ':' delimiter"
    }
    Ok((key, skip_ws(src, j + 1)))
}

/// CPython's `scan_once` at `start`: one value and the index just past it.
fn scan<S: Source>(
    src: &S,
    start: usize,
    max_digits: usize,
) -> Result<(Value, usize), DecodeError> {
    let mut stack = Stack(Vec::new());
    let mut buf = String::new();
    let mut i = start;
    'value: loop {
        // Scan the value at `i`, or open a container and continue with its first member.
        let (mut value, mut j) = match src.at(i) {
            b'"' => {
                let (s, j) = src.string(i + 1)?;
                (Value::Str(s), j)
            }
            b'[' => {
                let k = skip_ws(src, i + 1);
                if src.at(k) == b']' {
                    (Value::Array(Vec::new()), k + 1)
                } else {
                    stack.0.push(Frame::Array(Vec::new()));
                    i = k;
                    continue 'value;
                }
            }
            b'{' => {
                let k = skip_ws(src, i + 1);
                if src.at(k) == b'}' {
                    (Value::Object(new_object()), k + 1)
                } else {
                    let (name, k) = key(src, k)?;
                    stack.0.push(Frame::Object(new_object(), name));
                    i = k;
                    continue 'value;
                }
            }
            b'n' if word_at(src, i + 1, b"ull") => (Value::Null, i + 4),
            b't' if word_at(src, i + 1, b"rue") => (Value::Bool(true), i + 4),
            b'f' if word_at(src, i + 1, &b"false"[1..]) => (Value::Bool(false), i + 5),
            b'N' if word_at(src, i + 1, b"aN") => (Value::Float(f64::NAN), i + 3),
            b'I' if word_at(src, i + 1, b"nfinity") => (Value::Float(f64::INFINITY), i + 8),
            b'-' if word_at(src, i + 1, b"Infinity") => (Value::Float(f64::NEG_INFINITY), i + 9),
            _ => number(src, i, max_digits, &mut buf)?,
        };
        // Hand the value to the innermost open container, closing containers
        // until one wants another member.
        loop {
            let k = skip_ws(src, j);
            match stack.0.last_mut() {
                None => return Ok((value, j)),
                Some(Frame::Array(items)) => {
                    items.push(value);
                    match src.at(k) {
                        b',' => {
                            i = skip_ws(src, k + 1);
                            continue 'value;
                        }
                        b']' => {}
                        _ => return Err(DecodeError::Json), // "Expecting ',' delimiter"
                    }
                }
                Some(Frame::Object(obj, name)) => {
                    if let Some(old) = obj.insert(std::mem::take(name), value) {
                        drop_deep(old);
                    }
                    match src.at(k) {
                        b',' => {
                            let (next, k) = key(src, skip_ws(src, k + 1))?;
                            *name = next;
                            i = k;
                            continue 'value;
                        }
                        b'}' => {}
                        _ => return Err(DecodeError::Json), // "Expecting ',' delimiter"
                    }
                }
            }
            value = stack.0.pop().expect("an open container").into_value();
            j = k + 1;
        }
    }
}

/// Helpers shared by the oracle tests of this module and `dumps`.
#[cfg(test)]
pub(super) mod testutil {
    use std::fmt::Write as _;

    use super::super::chars::{chars_from_wtf8, to_wtf8};
    use super::super::value::{Int, Value, new_object};

    pub(crate) fn hex(b: &[u8]) -> String {
        let mut s = String::with_capacity(2 * b.len());
        for x in b {
            let _ = write!(s, "{x:02x}");
        }
        s
    }

    pub(crate) fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|k| u8::from_str_radix(&s[k..k + 2], 16).expect("hex"))
            .collect()
    }

    /// A Python `str.encode("utf-8", "surrogatepass").hex()` as a kernel string.
    pub(crate) fn py_str(h: &str) -> String {
        chars_from_wtf8(&unhex(h))
            .expect("wtf-8")
            .expect("no reserved code points")
            .into_iter()
            .collect()
    }

    /// A value as a flat token list (as the fixture generators write them):
    /// `n t f i<int> d<bits> s<wtf8 hex> [ ] { k<wtf8 hex> }`. Iterative.
    pub(crate) fn tokens(v: &Value) -> Vec<String> {
        enum Item<'a> {
            Value(&'a Value),
            Key(&'a str),
            Close(&'static str),
        }
        let mut out = Vec::new();
        let mut work = vec![Item::Value(v)];
        while let Some(item) = work.pop() {
            match item {
                Item::Close(t) => out.push(t.to_owned()),
                Item::Key(k) => out.push(format!("k{}", hex(&to_wtf8(k)))),
                Item::Value(v) => match v {
                    Value::Null => out.push("n".into()),
                    Value::Bool(b) => out.push(if *b { "t" } else { "f" }.into()),
                    Value::Int(i) => {
                        let mut s = String::from("i");
                        i.write_repr(&mut s);
                        out.push(s);
                    }
                    Value::Float(f) => out.push(format!("d{:016x}", f.to_bits())),
                    Value::Str(s) => out.push(format!("s{}", hex(&to_wtf8(s)))),
                    Value::Array(a) => {
                        out.push("[".into());
                        work.push(Item::Close("]"));
                        work.extend(a.iter().rev().map(Item::Value));
                    }
                    Value::Object(o) => {
                        out.push("{".into());
                        work.push(Item::Close("}"));
                        for (k, v) in o.iter().rev() {
                            work.push(Item::Value(v));
                            work.push(Item::Key(k));
                        }
                    }
                },
            }
        }
        out
    }

    /// The inverse of [`tokens`]. Iterative.
    pub(crate) fn from_tokens(toks: &[String]) -> Value {
        enum Open {
            Array(Vec<Value>),
            Object(super::Object, Option<String>),
        }
        let mut stack: Vec<Open> = Vec::new();
        let mut toks = toks.iter();
        loop {
            let t = toks.next().expect("token");
            let (tag, rest) = t.split_at(1);
            let value = match tag {
                "[" => {
                    stack.push(Open::Array(Vec::new()));
                    continue;
                }
                "{" => {
                    stack.push(Open::Object(new_object(), None));
                    continue;
                }
                "k" => {
                    let Some(Open::Object(_, key)) = stack.last_mut() else {
                        panic!("key outside object")
                    };
                    *key = Some(py_str(rest));
                    continue;
                }
                "]" | "}" => match stack.pop().expect("open") {
                    Open::Array(a) => Value::Array(a),
                    Open::Object(o, _) => Value::Object(o),
                },
                "n" => Value::Null,
                "t" => Value::Bool(true),
                "f" => Value::Bool(false),
                "i" => Value::Int(Int::parse_py(rest, 0).expect("int")),
                "d" => Value::Float(f64::from_bits(u64::from_str_radix(rest, 16).expect("bits"))),
                "s" => Value::Str(py_str(rest)),
                _ => panic!("bad token {t}"),
            };
            match stack.last_mut() {
                None => return value,
                Some(Open::Array(a)) => a.push(value),
                Some(Open::Object(o, key)) => {
                    o.insert(key.take().expect("key"), value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{hex, py_str, tokens, unhex};
    use super::*;
    use crate::json_repair::chars::{chars_from_wtf8, surrogate_char};
    use crate::json_repair::value::is_same_object;

    fn int(v: i64) -> Value {
        Value::Int(Int::Small(v))
    }

    fn s(v: &str) -> Value {
        Value::Str(v.into())
    }

    /// `loads` and `raw_decode` at 0 must agree whenever there is no surrounding
    /// whitespace. `text` is a kernel string (it may hold remapped surrogates).
    fn both(text: &str) -> Result<Value, DecodeError> {
        let a = loads(text, 4300);
        let chars: Vec<char> = text.chars().collect();
        let b = raw_decode(&chars, 0, 4300).and_then(|(v, end)| {
            if end == chars.len() {
                Ok(v)
            } else {
                Err(DecodeError::Json)
            }
        });
        match (&a, &b) {
            (Ok(x), Ok(y)) => assert_eq!(tokens(x), tokens(y), "{text:?}"),
            _ => assert_eq!(a, b, "{text:?}"),
        }
        a
    }

    #[test]
    fn scalars() {
        assert_eq!(both("null"), Ok(Value::Null));
        assert_eq!(both("true"), Ok(Value::Bool(true)));
        assert_eq!(both("false"), Ok(Value::Bool(false)));
        assert_eq!(both("0"), Ok(int(0)));
        assert_eq!(both("-0"), Ok(int(0)));
        assert_eq!(both("-12"), Ok(int(-12)));
        assert_eq!(both("1.5"), Ok(Value::Float(1.5)));
        assert_eq!(both("1E2"), Ok(Value::Float(100.0)));
        assert_eq!(both("1e-2"), Ok(Value::Float(0.01)));
        assert_eq!(both("2e+3"), Ok(Value::Float(2000.0)));
        assert_eq!(both("1e400"), Ok(Value::Float(f64::INFINITY)));
        assert_eq!(both("-1e400"), Ok(Value::Float(f64::NEG_INFINITY)));
        assert!(matches!(both("-0.0"), Ok(Value::Float(f)) if f == 0.0 && f.is_sign_negative()));
        assert!(matches!(both("NaN"), Ok(Value::Float(f)) if f.is_nan()));
        assert_eq!(both("Infinity"), Ok(Value::Float(f64::INFINITY)));
        assert_eq!(both("-Infinity"), Ok(Value::Float(f64::NEG_INFINITY)));
        assert_eq!(both("\"a\\u00e9\\n\""), Ok(s("aé\n")));
        assert_eq!(
            both("9223372036854775808"),
            Ok(Value::Int(Int::Big("9223372036854775808".into())))
        );
        assert_eq!(both("-9223372036854775808"), Ok(int(i64::MIN)));
        assert_eq!(both("999999999999999999"), Ok(int(999_999_999_999_999_999)));
    }

    #[test]
    fn number_edges() {
        // The number ends where CPython's matcher stops; the rest is extra data.
        for bad in [
            "01", "1.", "1.e5", "1e", "1e+", "1E-", ".5", "+1", "-", "--1", "1.0e", "0x1", "1_0",
        ] {
            assert_eq!(both(bad), Err(DecodeError::Json), "{bad}");
        }
        let chars: Vec<char> = "1ex".chars().collect();
        assert_eq!(raw_decode(&chars, 0, 4300), Ok((int(1), 1)));
        let chars: Vec<char> = "1.5e+x".chars().collect();
        assert_eq!(raw_decode(&chars, 0, 4300), Ok((Value::Float(1.5), 3)));
        let chars: Vec<char> = "-0.25e-1]".chars().collect();
        assert_eq!(raw_decode(&chars, 0, 4300), Ok((Value::Float(-0.025), 8)));
        let chars: Vec<char> = "١".chars().collect(); // ARABIC-INDIC DIGIT ONE: not a JSON digit
        assert_eq!(raw_decode(&chars, 0, 4300), Err(DecodeError::Json));
        assert_eq!(both("12٣"), Err(DecodeError::Json));
    }

    #[test]
    fn int_digit_limit() {
        let long = "1".repeat(4301);
        let err = DecodeError::Int(IntError::TooManyDigits {
            limit: 4300,
            digits: 4301,
        });
        assert_eq!(loads(&long, 4300), Err(err.clone()));
        assert_eq!(loads(&format!("-{long}"), 4300), Err(err.clone()));
        assert!(loads(&long, 0).is_ok());
        assert!(loads(&"1".repeat(4300), 4300).is_ok());
        // Floats have no limit.
        assert!(loads(&format!("{long}.5"), 4300).is_ok());
        // Scan order decides: an int error before a syntax error wins, and vice versa.
        assert_eq!(loads(&format!("[{long}, }}"), 4300), Err(err.clone()));
        assert_eq!(loads(&format!("[{long}] x"), 4300), Err(err));
        assert_eq!(
            loads(&format!("[\"\x01\", {long}]"), 4300),
            Err(DecodeError::Json)
        );
    }

    #[test]
    fn strings() {
        assert_eq!(
            both(r#""\"\\\/\b\f\n\r\t""#),
            Ok(s("\"\\/\u{8}\u{c}\n\r\t"))
        );
        assert_eq!(both(r#""\uD83D\uDE00""#), Ok(s("😀")));
        assert_eq!(both(r#""\ud83d\ude00x""#), Ok(s("😀x")));
        let lone = |cp| surrogate_char(cp).to_string();
        assert_eq!(both(r#""\ud800""#), Ok(Value::Str(lone(0xD800))));
        assert_eq!(
            both(r#""\udc00\ud800""#),
            Ok(Value::Str(lone(0xDC00) + &lone(0xD800)))
        );
        assert_eq!(
            both(r#""\ud800\u0041""#),
            Ok(Value::Str(lone(0xD800) + "A"))
        );
        assert_eq!(
            both(r#""\ud800\ud800""#),
            Ok(Value::Str(lone(0xD800) + &lone(0xD800)))
        );
        assert_eq!(
            both(r#""\ud800\\u0041""#),
            Ok(Value::Str(lone(0xD800) + "\\u0041"))
        );
        assert_eq!(
            both("\"\\ud800\u{10f800}\""),
            Ok(Value::Str(lone(0xD800) + &lone(0xD800)))
        );
        assert_eq!(
            both(r#""\udbfe\udc00""#),
            Err(DecodeError::ReservedCodePoint)
        );
        assert_eq!(
            both(r#""\udbff\udfff""#),
            Err(DecodeError::ReservedCodePoint)
        );
        assert_eq!(both(r#""\udbfd\udfff""#), Ok(s("\u{10f7ff}")));
        // A raw remapped surrogate in the input passes through.
        assert_eq!(loads("\"\u{10f800}\"", 0), Ok(Value::Str(lone(0xD800))));
        assert_eq!(
            both("\"\u{7f}\u{2028}\u{feff}\""),
            Ok(s("\u{7f}\u{2028}\u{feff}"))
        );
        for bad in [
            "\"abc",
            "\"\\",
            "\"\\\"",
            "\"\x1f\"",
            "\"\n\"",
            "\"\\x41\"",
            "\"\\'\"",
            "\"\\U0041\"",
            "\"\\u004\"",
            "\"\\u00g1\"",
            "\"\\u٠٠٤١\"",
            "\"\\ud800\\u12\"",
            "\"\\ud800\\uzzzz\"",
            "\"\\ud800\\udc00",
            "'a'",
        ] {
            assert_eq!(both(bad), Err(DecodeError::Json), "{bad:?}");
        }
    }

    #[test]
    fn containers() {
        let v = both(" [1, {\"a\": [], \"b\": {}}, \"x\"] ".trim()).unwrap();
        let Value::Array(items) = &v else { panic!() };
        assert_eq!(items.len(), 3);
        assert_eq!(
            tokens(&v),
            [
                "[", "i1", "{", "k61", "[", "]", "k62", "{", "}", "}", "s78", "]"
            ]
        );
        // Duplicate keys: the last value, at the first position.
        let v = both(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        let Value::Object(o) = &v else { panic!() };
        assert_eq!(o.keys().collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(o["a"], int(3));
        assert_eq!(
            loads(" \t\n\r[ \t\n\r1 \t\n\r] \t\n\r", 0),
            Ok(Value::Array(vec![int(1)]))
        );
        for bad in [
            "[1,]",
            "[,1]",
            "[1 2]",
            "[",
            "[1",
            "{",
            "{\"a\"",
            "{\"a\":",
            "{\"a\":1",
            "{\"a\":1,}",
            "{a:1}",
            "{'a':1}",
            "{\"a\" 1}",
            "{1:2}",
            "[1]]",
            "{}}",
            "[\x0b1]",
            "[\u{a0}1]",
            "",
            " ",
            "nul",
            // A literal with one char too many / too few (spelled out for typos).
            concat!("null", "l"),
            "True",
            "nan",
            "-NaN",
            "+Infinity",
            "-Inf",
            concat!("Infini", "t"),
        ] {
            assert_eq!(both(bad), Err(DecodeError::Json), "{bad:?}");
        }
    }

    #[test]
    fn whitespace_and_bom() {
        assert_eq!(loads("\u{feff}1", 0), Err(DecodeError::Json));
        assert_eq!(loads(" \u{feff}1", 0), Err(DecodeError::Json));
        assert_eq!(loads("\"\u{feff}\"", 0), Ok(s("\u{feff}")));
        let chars: Vec<char> = "\u{feff}1".chars().collect();
        assert_eq!(raw_decode(&chars, 0, 0), Err(DecodeError::Json));
        assert_eq!(raw_decode(&chars, 1, 0), Ok((int(1), 2)));
        // raw_decode does not skip leading whitespace.
        let chars: Vec<char> = "x [1] 2".chars().collect();
        assert_eq!(raw_decode(&chars, 1, 0), Err(DecodeError::Json));
        assert_eq!(
            raw_decode(&chars, 2, 0),
            Ok((Value::Array(vec![int(1)]), 5))
        );
        assert_eq!(raw_decode(&chars, 6, 0), Ok((int(2), 7)));
        assert_eq!(raw_decode(&chars, 7, 0), Err(DecodeError::Json));
        assert_eq!(raw_decode(&chars, 99, 0), Err(DecodeError::Json));
        assert_eq!(loads("1 \x0c", 0), Err(DecodeError::Json));
    }

    #[test]
    fn deep_nesting_is_iterative() {
        let depth = 200_000;
        let text = "[".repeat(depth) + &"]".repeat(depth);
        let v = loads(&text, 0).unwrap();
        let mut probe = Value::Array(Vec::new());
        for _ in 1..depth {
            probe = Value::Array(vec![probe]);
        }
        assert!(is_same_object(&v, &probe));
        drop_deep(v);
        drop_deep(probe);
        let text = "{\"a\":".repeat(depth) + "1" + &"}".repeat(depth);
        let chars: Vec<char> = text.chars().collect();
        let (v, end) = raw_decode(&chars, 0, 0).unwrap();
        assert_eq!(end, chars.len());
        drop_deep(v);
        // Errors deep inside, and duplicate keys replacing deep values, drop iteratively too.
        let deep = "[".repeat(depth) + &"]".repeat(depth);
        assert_eq!(loads(&format!("[{deep}, x]"), 0), Err(DecodeError::Json));
        assert_eq!(
            loads(&format!("{{\"a\": {deep}, \"a\": 1}}"), 0).map(|v| tokens(&v).len()),
            Ok(4)
        );
        assert_eq!(
            loads(&("[".repeat(depth) + "1,"), 0),
            Err(DecodeError::Json)
        );
    }

    #[test]
    fn testutil_round_trip() {
        let v = loads(
            r#"{"a": [1, 2.5, "\ud800x", null, true, false, {}], "é": -7}"#,
            0,
        )
        .unwrap();
        let t = tokens(&v);
        assert_eq!(tokens(&super::testutil::from_tokens(&t)), t);
        assert_eq!(unhex(&hex(b"\x00\xffab")), b"\x00\xffab");
        assert_eq!(py_str("eda080"), surrogate_char(0xD800).to_string());
    }

    /// Differential test against a fixture written by CPython (see the report
    /// of the `jsonrepair` session for the generator): one JSON object per line.
    #[test]
    #[allow(clippy::print_stderr)]
    fn cpython_oracle() {
        let Ok(path) = std::env::var("PYROID_JSONREPAIR_PYJSON_ORACLE") else {
            return;
        };
        let text = std::fs::read_to_string(&path).expect("read fixture");
        let mut lines = text.lines();
        let header: serde_json::Value =
            serde_json::from_str(lines.next().expect("header")).unwrap();
        let (mut cases, mut reserved, mut recursion, mut failures) = (0, 0, 0, Vec::new());
        for line in lines {
            let case: serde_json::Value = serde_json::from_str(line).unwrap();
            let input = case["in"].as_str().unwrap();
            let max = usize::try_from(case["max"].as_u64().unwrap()).unwrap();
            let chars = chars_from_wtf8(&unhex(input))
                .expect("wtf-8")
                .expect("not reserved");
            let raw = case["m"] == "raw";
            let got = if raw {
                let start = usize::try_from(case["start"].as_u64().unwrap()).unwrap();
                raw_decode(&chars, start, max).map(|(v, end)| (v, Some(end)))
            } else {
                let s: String = chars.iter().collect();
                loads(&s, max).map(|v| (v, None))
            };
            cases += 1;
            let want = case["r"].as_str().unwrap();
            let ok = match (&got, want) {
                // CPython's RecursionError depends on the interpreter's stack;
                // the kernel is iterative and does not model it.
                (_, "rec") => {
                    recursion += 1;
                    true
                }
                (Err(DecodeError::ReservedCodePoint), _) => {
                    reserved += 1;
                    case["resv"] == true
                }
                (Err(DecodeError::Json), "json") => true,
                (Err(DecodeError::Int(e)), "int") => e.to_string() == case["msg"].as_str().unwrap(),
                (Ok((v, end)), "ok") => {
                    let want_toks: Vec<String> = case["v"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|t| t.as_str().unwrap().to_owned())
                        .collect();
                    let same_end =
                        !raw || end.and_then(|e| u64::try_from(e).ok()) == case["end"].as_u64();
                    same_end && tokens(v) == want_toks
                }
                _ => false,
            };
            if !ok && failures.len() < 20 {
                let shown: String = format!("{:?}", py_str(input)).chars().take(400).collect();
                let got_text: String = format!("{got:?}").chars().take(400).collect();
                let line: String = line.chars().take(400).collect();
                failures.push(format!("{line}\n  input {shown}\n  got {got_text}"));
            }
            if let Ok((v, _)) = got {
                drop_deep(v);
            }
        }
        eprintln!(
            "pyjson oracle {}: {cases} cases ({reserved} reserved-code-point delegations, \
             {recursion} CPython RecursionErrors skipped), {} failures",
            header["python"],
            failures.len()
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        assert!(cases > 0);
    }
}
