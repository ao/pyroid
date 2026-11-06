// Derived from CPython's json module (PSF-2.0) behaviour; written from scratch.
//! `json.dumps(value, **kwargs)` for [`Value`] trees, byte for byte as CPython
//! writes it (the C encoder and the pure-Python `_make_iterencode` agree on
//! everything but one error message, see [`DumpOptions::python_3_11`]).
//!
//! Iterative (an explicit stack of open containers) and linear; the output is
//! one `String`.

use std::fmt::{self, Write as _};

use super::chars::{code_point, contains_reserved};
use super::value::{Object, Value};

/// `json.dumps` keyword arguments that change the output.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)] // mirrors the keyword arguments
pub struct DumpOptions {
    pub ensure_ascii: bool,
    pub allow_nan: bool,
    pub sort_keys: bool,
    /// Already resolved: `None`, or the indent string (`" " * n` for an int).
    pub indent: Option<String>,
    pub item_separator: String,
    pub key_separator: String,
    /// Running on CPython 3.11, whose C encoder (used when `indent` is None)
    /// words the `allow_nan=False` error without the offending value:
    /// `"Out of range float values are not JSON compliant"` instead of
    /// `"...: nan"`. 3.12 and later always append `": " + repr(value)`.
    pub python_3_11: bool,
}

impl Default for DumpOptions {
    /// `json.dumps` defaults.
    fn default() -> Self {
        Self {
            ensure_ascii: true,
            allow_nan: true,
            sort_keys: false,
            indent: None,
            item_separator: ", ".into(),
            key_separator: ": ".into(),
            python_3_11: false,
        }
    }
}

/// `json.dumps` failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DumpError {
    /// `allow_nan=False` and a NaN/inf: `ValueError` with this message.
    OutOfRangeFloat(String),
}

/// `json.dumps(value, **opts)`. Remapped surrogates come out as `\udXXX` with
/// `ensure_ascii`, else as remapped chars (see `chars::to_wtf8`).
pub fn dumps(value: &Value, opts: &DumpOptions) -> Result<String, DumpError> {
    let mut enc = Encoder {
        opts,
        out: String::with_capacity(128),
        level: 0,
    };
    let mut stack: Vec<Frame<'_>> = Vec::new();
    enc.value(value, &mut stack)?;
    while let Some(top) = stack.last_mut() {
        let next = match top {
            Frame::Array { items, first } => items.next().inspect(|_| enc.separator(first)),
            Frame::Object { entries, first } => entries.next().map(|(k, v)| {
                enc.separator(first);
                write_str(&mut enc.out, k, opts.ensure_ascii);
                enc.out.push_str(&opts.key_separator);
                v
            }),
        };
        if let Some(v) = next {
            enc.value(v, &mut stack)?;
        } else {
            let close = match stack.pop() {
                Some(Frame::Object { .. }) => '}',
                _ => ']',
            };
            enc.level -= 1;
            enc.newline_indent();
            enc.out.push(close);
        }
    }
    Ok(enc.out)
}

/// An open container and what is left of it.
enum Frame<'a> {
    Array {
        items: std::slice::Iter<'a, Value>,
        first: bool,
    },
    Object {
        entries: Entries<'a>,
        first: bool,
    },
}

/// An object's items, in insertion order or sorted (`sort_keys`).
enum Entries<'a> {
    Ordered(indexmap::map::Iter<'a, String, Value>),
    Sorted(std::vec::IntoIter<(&'a String, &'a Value)>),
}

impl<'a> Iterator for Entries<'a> {
    type Item = (&'a String, &'a Value);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Ordered(it) => it.next(),
            Self::Sorted(it) => it.next(),
        }
    }
}

/// `sorted(dct.items())`: keys by code point, remapped surrogates as U+D800..U+DFFF.
fn sorted_entries(o: &Object) -> Entries<'_> {
    let mut items: Vec<(&String, &Value)> = o.iter().collect();
    if o.keys().any(|k| contains_reserved(k)) {
        items.sort_unstable_by(|a, b| a.0.chars().map(code_point).cmp(b.0.chars().map(code_point)));
    } else {
        // UTF-8 byte order is code point order.
        items.sort_unstable_by(|a, b| a.0.cmp(b.0));
    }
    Entries::Sorted(items.into_iter())
}

struct Encoder<'o> {
    opts: &'o DumpOptions,
    out: String,
    /// Open containers (`_current_indent_level`).
    level: usize,
}

impl Encoder<'_> {
    /// `"\n" + indent * level`, with an indent.
    #[inline]
    fn newline_indent(&mut self) {
        if let Some(indent) = &self.opts.indent {
            self.out.push('\n');
            for _ in 0..self.level {
                self.out.push_str(indent);
            }
        }
    }

    /// What goes before a container member.
    #[inline]
    fn separator(&mut self, first: &mut bool) {
        if *first {
            *first = false;
        } else {
            self.out.push_str(&self.opts.item_separator);
        }
        self.newline_indent();
    }

    /// Write a scalar, or open a container (pushing it on `stack`).
    fn value<'a>(&mut self, v: &'a Value, stack: &mut Vec<Frame<'a>>) -> Result<(), DumpError> {
        match v {
            Value::Null => self.out.push_str("null"),
            Value::Bool(true) => self.out.push_str("true"),
            Value::Bool(false) => self.out.push_str("false"),
            Value::Int(i) => i.write_repr(&mut self.out),
            Value::Float(f) => self.float(*f)?,
            Value::Str(s) => write_str(&mut self.out, s, self.opts.ensure_ascii),
            Value::Array(a) if a.is_empty() => self.out.push_str("[]"),
            Value::Object(o) if o.is_empty() => self.out.push_str("{}"),
            Value::Array(a) => {
                self.out.push('[');
                self.level += 1;
                stack.push(Frame::Array {
                    items: a.iter(),
                    first: true,
                });
            }
            Value::Object(o) => {
                self.out.push('{');
                self.level += 1;
                let entries = if self.opts.sort_keys {
                    sorted_entries(o)
                } else {
                    Entries::Ordered(o.iter())
                };
                stack.push(Frame::Object {
                    entries,
                    first: true,
                });
            }
        }
        Ok(())
    }

    fn float(&mut self, f: f64) -> Result<(), DumpError> {
        if f.is_finite() {
            float_repr(f, &mut self.out);
            return Ok(());
        }
        if !self.opts.allow_nan {
            let mut msg = String::from("Out of range float values are not JSON compliant");
            if !(self.opts.python_3_11 && self.opts.indent.is_none()) {
                msg.push_str(": ");
                float_repr(f, &mut msg);
            }
            return Err(DumpError::OutOfRangeFloat(msg));
        }
        self.out.push_str(if f.is_nan() {
            "NaN"
        } else if f > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        });
        Ok(())
    }
}

/// How `write_str` treats each byte with `ensure_ascii`: 0 = copy, 1 = ASCII
/// escape (C0 controls, `"` = 0x22, `\` = 0x5C, DEL), 2 = start of a non-ASCII char.
static CLASS_ASCII: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut b = 0;
    while b < 256 {
        t[b] = if b >= 0x80 {
            2
        } else if b < 0x20 || b == 0x22 || b == 0x5C || b == 0x7F {
            1
        } else {
            0
        };
        b += 1;
    }
    t
};

/// `ensure_ascii=False`: only C0 controls, `"` and `\` are escaped.
static CLASS_RAW: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut b = 0;
    while b < 0x20 {
        t[b] = 1;
        b += 1;
    }
    t[0x22] = 1;
    t[0x5C] = 1;
    t
};

/// `\uXXXX` (lowercase hex).
fn push_u(out: &mut String, cp: u32) {
    out.push_str("\\u");
    for shift in [12, 8, 4, 0] {
        out.push(char::from_digit((cp >> shift) & 0xF, 16).expect("a nibble"));
    }
}

/// `encode_basestring_ascii` / `encode_basestring`, with the quotes.
fn write_str(out: &mut String, text: &str, ensure_ascii: bool) {
    let class = if ensure_ascii {
        &CLASS_ASCII
    } else {
        &CLASS_RAW
    };
    let bytes = text.as_bytes();
    out.reserve(bytes.len() + 2);
    out.push('"');
    let mut run = 0;
    let mut pos = 0;
    while pos < bytes.len() {
        let byte = bytes[pos];
        match class[usize::from(byte)] {
            0 => {
                pos += 1;
                continue;
            }
            1 => {
                out.push_str(&text[run..pos]);
                match byte {
                    b'"' => out.push_str("\\\""),
                    b'\\' => out.push_str("\\\\"),
                    b'\n' => out.push_str("\\n"),
                    b'\r' => out.push_str("\\r"),
                    b'\t' => out.push_str("\\t"),
                    0x08 => out.push_str("\\b"),
                    0x0C => out.push_str("\\f"),
                    _ => push_u(out, u32::from(byte)),
                }
                pos += 1;
            }
            _ => {
                // A non-ASCII char under ensure_ascii: `pos` is on a char boundary.
                out.push_str(&text[run..pos]);
                let ch = text[pos..].chars().next().expect("a char");
                let cp = code_point(ch);
                if cp >= 0x1_0000 {
                    let offset = cp - 0x1_0000;
                    push_u(out, 0xD800 | (offset >> 10));
                    push_u(out, 0xDC00 | (offset & 0x3FF));
                } else {
                    push_u(out, cp);
                }
                pos += ch.len_utf8();
            }
        }
        run = pos;
    }
    out.push_str(&text[run..]);
    out.push('"');
}

/// The significant digits and decimal exponent of a `{:e}` / `{:.N$e}`
/// rendering of a finite, non-negative `f64` (up to 24 digits).
struct Sci {
    digits: [u8; 24],
    len: usize,
    exp: i32,
}

impl Sci {
    /// `{:e}` (`precision` None: the shortest round-trip digits) or
    /// `{:.precision$e}` (that many digits after the first, correctly rounded).
    fn new(abs: f64, precision: Option<usize>) -> Self {
        /// A small stack buffer for the formatter.
        struct Buf {
            b: [u8; 40],
            n: usize,
        }
        impl fmt::Write for Buf {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                let end = self.n + s.len();
                self.b
                    .get_mut(self.n..end)
                    .ok_or(fmt::Error)?
                    .copy_from_slice(s.as_bytes());
                self.n = end;
                Ok(())
            }
        }
        let mut buf = Buf { b: [0; 40], n: 0 };
        match precision {
            None => write!(buf, "{abs:e}"),
            Some(p) => write!(buf, "{abs:.p$e}"),
        }
        .expect("fits");
        let text = &buf.b[..buf.n];
        let epos = text.iter().position(|&ch| ch == b'e').expect("exponent");
        let exp = std::str::from_utf8(&text[epos + 1..])
            .ok()
            .and_then(|exp| exp.parse().ok())
            .expect("exp");
        let mut sci = Self {
            digits: [0; 24],
            len: 0,
            exp,
        };
        for &digit in text[..epos].iter().filter(|&&ch| ch != b'.') {
            sci.digits[sci.len] = digit;
            sci.len += 1;
        }
        sci
    }

    fn digits(&self) -> &[u8] {
        &self.digits[..self.len]
    }
}

/// Whether `abs` lies exactly halfway between its shortest round-trip digits
/// `sci` and the next lower candidate (last digit minus one), and that lower
/// candidate round-trips too. Rust breaks that tie upwards, CPython's dtoa
/// (`repr`) to an even last digit.
fn is_lower_tie(abs: f64, sci: &Sci) -> bool {
    let digits = sci.digits();
    let len = digits.len();
    if len == 1 && digits[0] <= b'1' {
        return false;
    }
    // The tie point: the lower candidate followed by a 5, exactly.
    let mut tie = [0u8; 25];
    tie[..len].copy_from_slice(digits);
    tie[len - 1] -= 1;
    tie[len] = b'5';
    let tie = &tie[..=len];
    // Cheap test first: the correctly rounded len + 1 digits.
    let near = Sci::new(abs, Some(len));
    if near.exp != sci.exp || near.digits() != tie {
        return false;
    }
    // Exact test: every double has at most 767 significant digits.
    let full = format!("{abs:.800e}");
    let mantissa = full
        .split('e')
        .next()
        .expect("mantissa")
        .bytes()
        .filter(|&ch| ch != b'.');
    let mut exact = mantissa
        .enumerate()
        .all(|(k, ch)| tie.get(k).map_or(ch == b'0', |&t| t == ch));
    if exact {
        let mut lower = String::with_capacity(len + 8);
        lower.push(char::from(tie[0]));
        lower.push('.');
        tie[1..len].iter().for_each(|&d| lower.push(char::from(d)));
        let _ = write!(lower, "e{}", sci.exp);
        exact = lower.parse::<f64>() == Ok(abs);
    }
    exact
}

/// `float.__repr__`: the shortest digits that round-trip (ties to even); `1e+16`
/// style when the decimal exponent is below -4 or at least 16, else fixed with
/// at least one fractional digit.
pub fn float_repr(f: f64, out: &mut String) {
    if f.is_nan() {
        out.push_str("nan");
        return;
    }
    if f.is_sign_negative() {
        out.push('-');
    }
    if f.is_infinite() {
        out.push_str("inf");
        return;
    }
    // Rust's `{:e}` is the shortest round-trip form too, but for exact ties.
    let abs = f.abs();
    let mut sci = Sci::new(abs, None);
    // ASCII digits have the parity of their value.
    if sci.digits[sci.len - 1] & 1 == 1 && is_lower_tie(abs, &sci) {
        sci.digits[sci.len - 1] -= 1;
        while sci.len > 1 && sci.digits[sci.len - 1] == b'0' {
            sci.len -= 1;
        }
    }
    let exp = sci.exp;
    let len = sci.len;
    let digits = sci.digits();
    let push = |out: &mut String, ds: &[u8]| ds.iter().for_each(|&d| out.push(char::from(d)));
    if (-4..16).contains(&exp) {
        if let Ok(point) = usize::try_from(exp).map(|e| e + 1) {
            // `point` digits before the decimal point.
            if len <= point {
                push(out, digits);
                (len..point).for_each(|_| out.push('0'));
                out.push_str(".0");
            } else {
                push(out, &digits[..point]);
                out.push('.');
                push(out, &digits[point..]);
            }
        } else {
            out.push_str("0.");
            (0..-exp - 1).for_each(|_| out.push('0'));
            push(out, digits);
        }
    } else {
        push(out, &digits[..1]);
        if len > 1 {
            out.push('.');
            push(out, &digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.unsigned_abs()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::super::chars::{surrogate_char, to_wtf8};
    use super::super::pyjson::testutil::{from_tokens, hex, py_str, unhex};
    use super::super::value::{Int, drop_deep, new_object};
    use super::*;

    fn repr(f: f64) -> String {
        let mut s = String::new();
        float_repr(f, &mut s);
        s
    }

    fn obj(items: Vec<(&str, Value)>) -> Value {
        let mut o = new_object();
        for (k, v) in items {
            o.insert(k.into(), v);
        }
        Value::Object(o)
    }

    fn int(v: i64) -> Value {
        Value::Int(Int::Small(v))
    }

    fn d(v: &Value) -> String {
        dumps(v, &DumpOptions::default()).unwrap()
    }

    #[test]
    fn float_reprs() {
        for (f, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (123_456_789_012_345.0, "123456789012345.0"),
            (1e15, "1000000000000000.0"),
            (9_999_999_999_999_998.0, "9999999999999998.0"),
            (1e16, "1e+16"),
            (1.2345e17, "1.2345e+17"),
            (1e300, "1e+300"),
            (5e-324, "5e-324"),
            (1.797_693_134_862_315_7e308, "1.7976931348623157e+308"),
            (2.225_073_858_507_201_4e-308, "2.2250738585072014e-308"),
            (-1.5, "-1.5"),
            (1e22, "1e+22"),
            (1.0 / 3.0, "0.3333333333333333"),
            // Exact ties between two shortest candidates go to the even digit (written
            // as exact sums: clippy judges literals by Rust's round-half-up digits).
            (-916_539_213_519_405.0 - 0.25, "-916539213519405.2"),
            (2_153_723_027_069_006.0 + 0.25, "2153723027069006.2"),
            (85_836_451_078_291.0 + 0.125, "85836451078291.12"),
            (161_964_452_458_649.0 + 0.625, "161964452458649.62"),
            (-219_863_665_604_542.0 - 0.625, "-219863665604542.62"),
            (0.3, "0.3"),
            (f64::NAN, "nan"),
            (-f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(repr(f), want, "{f:e}");
        }
    }

    #[test]
    fn scalars_and_strings() {
        assert_eq!(d(&Value::Null), "null");
        assert_eq!(d(&Value::Bool(true)), "true");
        assert_eq!(d(&int(-3)), "-3");
        assert_eq!(
            d(&Value::Int(Int::Big(
                "123456789012345678901234567890".into()
            ))),
            "123456789012345678901234567890"
        );
        assert_eq!(d(&Value::Float(f64::NAN)), "NaN");
        assert_eq!(d(&Value::Float(f64::NEG_INFINITY)), "-Infinity");
        let text = "a\"\\/\n\r\t\u{8}\u{c}\u{1}\u{1f}\u{7f}é\u{2028}😀";
        assert_eq!(
            d(&Value::Str(text.into())),
            r#""a\"\\/\n\r\t\b\f\u0001\u001f\u007f\u00e9\u2028\ud83d\ude00""#
        );
        let raw = DumpOptions {
            ensure_ascii: false,
            ..DumpOptions::default()
        };
        assert_eq!(
            dumps(&Value::Str(text.into()), &raw).unwrap(),
            "\"a\\\"\\\\/\\n\\r\\t\\b\\f\\u0001\\u001f\u{7f}é\u{2028}😀\""
        );
        let lone = format!("{}x{}", surrogate_char(0xD83D), surrogate_char(0xDFFF));
        assert_eq!(d(&Value::Str(lone.clone())), r#""\ud83dx\udfff""#);
        assert_eq!(
            dumps(&Value::Str(lone.clone()), &raw).unwrap(),
            format!("\"{lone}\"")
        );
    }

    #[test]
    fn containers_and_indent() {
        let v = Value::Array(vec![
            int(1),
            obj(vec![("b", Value::Array(vec![])), ("a", obj(vec![]))]),
            int(2),
        ]);
        assert_eq!(d(&v), r#"[1, {"b": [], "a": {}}, 2]"#);
        let indented = DumpOptions {
            indent: Some("  ".into()),
            item_separator: ",".into(),
            ..DumpOptions::default()
        };
        assert_eq!(
            dumps(&v, &indented).unwrap(),
            "[\n  1,\n  {\n    \"b\": [],\n    \"a\": {}\n  },\n  2\n]"
        );
        let zero = DumpOptions {
            indent: Some(String::new()),
            item_separator: ",".into(),
            sort_keys: true,
            ..DumpOptions::default()
        };
        assert_eq!(
            dumps(&v, &zero).unwrap(),
            "[\n1,\n{\n\"a\": {},\n\"b\": []\n},\n2\n]"
        );
        let compact = DumpOptions {
            item_separator: ",".into(),
            key_separator: ":".into(),
            ..DumpOptions::default()
        };
        assert_eq!(dumps(&v, &compact).unwrap(), r#"[1,{"b":[],"a":{}},2]"#);
    }

    #[test]
    fn sort_keys_by_code_point() {
        // U+E000 sorts after a surrogate, U+FFFF before an astral char; a remapped
        // surrogate is U+D800.., not U+10F800...
        let sur = surrogate_char(0xD800).to_string();
        let v = obj(vec![
            ("😀", int(1)),
            ("\u{e000}", int(2)),
            (&sur, int(3)),
            ("\u{d7ff}", int(4)),
            ("b", int(5)),
            ("a", int(6)),
        ]);
        let opts = DumpOptions {
            sort_keys: true,
            ..DumpOptions::default()
        };
        assert_eq!(
            dumps(&v, &opts).unwrap(),
            r#"{"a": 6, "b": 5, "\ud7ff": 4, "\ud800": 3, "\ue000": 2, "\ud83d\ude00": 1}"#
        );
    }

    #[test]
    fn nan_errors() {
        let v = Value::Array(vec![
            int(1),
            Value::Float(f64::INFINITY),
            Value::Float(f64::NAN),
        ]);
        let mut opts = DumpOptions {
            allow_nan: false,
            ..DumpOptions::default()
        };
        let msg = |m: &str| Err(DumpError::OutOfRangeFloat(m.into()));
        assert_eq!(
            dumps(&v, &opts),
            msg("Out of range float values are not JSON compliant: inf")
        );
        assert_eq!(
            dumps(&Value::Float(-f64::NAN), &opts),
            msg("Out of range float values are not JSON compliant: nan")
        );
        opts.python_3_11 = true;
        assert_eq!(
            dumps(&v, &opts),
            msg("Out of range float values are not JSON compliant")
        );
        opts.indent = Some(" ".into());
        assert_eq!(
            dumps(&v, &opts),
            msg("Out of range float values are not JSON compliant: inf")
        );
        assert_eq!(dumps(&Value::Float(1.0), &opts), Ok("1.0".into()));
    }

    #[test]
    fn deep_values_are_iterative() {
        let depth = 200_000;
        let mut v = Value::Array(vec![]);
        for k in 0..depth {
            v = if k % 2 == 0 {
                Value::Array(vec![v])
            } else {
                obj(vec![("k", v)])
            };
        }
        let out = dumps(&v, &DumpOptions::default()).unwrap();
        assert_eq!(out.len(), 2 + depth / 2 * 2 + depth / 2 * 7);
        drop_deep(v);
    }

    /// Differential test against a fixture written by CPython: one JSON object per line.
    #[test]
    #[allow(clippy::print_stderr)]
    fn cpython_oracle() {
        let Ok(path) = std::env::var("PYROID_JSONREPAIR_DUMPS_ORACLE") else {
            return;
        };
        let text = std::fs::read_to_string(&path).expect("read fixture");
        let mut lines = text.lines();
        let header: serde_json::Value =
            serde_json::from_str(lines.next().expect("header")).unwrap();
        let python_3_11 = header["python"].as_str().unwrap().starts_with("3.11.");
        let (mut dump_cases, mut repr_cases, mut failures) = (0, 0, Vec::new());
        for line in lines {
            let case: serde_json::Value = serde_json::from_str(line).unwrap();
            if case["m"] == "repr" {
                repr_cases += 1;
                let bits = u64::from_str_radix(case["b"].as_str().unwrap(), 16).unwrap();
                let got = repr(f64::from_bits(bits));
                if got != case["out"].as_str().unwrap() && failures.len() < 20 {
                    failures.push(format!("{line}\n  got {got}"));
                }
                continue;
            }
            dump_cases += 1;
            let toks: Vec<String> = case["v"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_str().unwrap().to_owned())
                .collect();
            let value = from_tokens(&toks);
            let o = &case["o"];
            let opts = DumpOptions {
                ensure_ascii: o["ensure_ascii"].as_bool().unwrap(),
                allow_nan: o["allow_nan"].as_bool().unwrap(),
                sort_keys: o["sort_keys"].as_bool().unwrap(),
                indent: o["indent"].as_str().map(py_str),
                item_separator: py_str(o["item_sep"].as_str().unwrap()),
                key_separator: py_str(o["key_sep"].as_str().unwrap()),
                python_3_11,
            };
            let got = dumps(&value, &opts);
            let ok = match (&got, case["r"].as_str().unwrap()) {
                (Ok(s), "ok") => hex(&to_wtf8(s)) == case["out"].as_str().unwrap(),
                (Err(DumpError::OutOfRangeFloat(m)), "err") => m == case["msg"].as_str().unwrap(),
                _ => false,
            };
            if !ok && failures.len() < 20 {
                let want = case["out"]
                    .as_str()
                    .map(|h| String::from_utf8_lossy(&unhex(h)).into_owned());
                failures.push(format!("{line}\n  got  {got:?}\n  want {want:?}"));
            }
            drop_deep(value);
        }
        eprintln!(
            "dumps oracle {}: {dump_cases} dumps cases, {repr_cases} float reprs, {} failures",
            header["python"],
            failures.len()
        );
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        assert!(dump_cases > 0);
    }
}
