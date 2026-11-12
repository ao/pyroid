// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! The value tree a repair produces: Python's JSON types, with Python's
//! semantics where json_repair relies on them.
//!
//! - [`Object`] is an insertion-ordered map with `dict` semantics: assigning an
//!   existing key keeps its position and replaces the value.
//! - [`Int`] is a Python `int`: `i64` when it fits, else canonical decimal digits.
//! - Strings are Rust `String`s in which lone surrogates (which Python strings
//!   can hold and `char` cannot) are stored remapped, see [`super::chars`].

use std::fmt;

use indexmap::IndexMap;
use rustc_hash::FxBuildHasher;

/// A JSON object: a Python `dict[str, Any]` (insertion order, last write wins).
pub type Object = IndexMap<String, Value, FxBuildHasher>;

/// The by-value iterator of an [`Object`] (for consumers outside this crate).
pub type ObjectIntoIter = indexmap::map::IntoIter<String, Value>;

/// A new, empty [`Object`].
pub fn new_object() -> Object {
    IndexMap::with_hasher(FxBuildHasher)
}

/// A Python `int`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Int {
    /// Fits in an `i64`.
    Small(i64),
    /// Does not fit in an `i64`: canonical decimal (`-` sign, no leading zeros).
    Big(Box<str>),
}

/// `int(text)` failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntError {
    /// Not `[+-]?[0-9]+` (Python raises `ValueError: invalid literal for int()`).
    Invalid,
    /// More digits than `sys.get_int_max_str_digits()` allows.
    TooManyDigits {
        /// The limit in force.
        limit: usize,
        /// The number of digits in the text (leading zeros count, as in CPython).
        digits: usize,
    },
}

impl fmt::Display for IntError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid => f.write_str("invalid literal for int() with base 10"),
            Self::TooManyDigits { limit, digits } => write!(
                f,
                "Exceeds the limit ({limit} digits) for integer string conversion: value has \
                 {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
            ),
        }
    }
}

impl std::error::Error for IntError {}

impl Int {
    /// Python's `int(text)` for ASCII `[+-]?[0-9]+` text (the only shapes
    /// json_repair passes it). `max_digits` is `sys.get_int_max_str_digits()`
    /// (0 = unlimited).
    pub fn parse_py(text: &str, max_digits: usize) -> Result<Self, IntError> {
        let (neg, digits) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            _ => (false, text),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(IntError::Invalid);
        }
        if max_digits > 0 && digits.len() > max_digits {
            return Err(IntError::TooManyDigits {
                limit: max_digits,
                digits: digits.len(),
            });
        }
        let trimmed = digits.trim_start_matches('0');
        if trimmed.is_empty() {
            return Ok(Self::Small(0));
        }
        if trimmed.len() <= 18 {
            let v: i64 = trimmed.parse().map_err(|_| IntError::Invalid)?;
            return Ok(Self::Small(if neg { -v } else { v }));
        }
        let mut s = String::with_capacity(trimmed.len() + 1);
        if neg {
            s.push('-');
        }
        s.push_str(trimmed);
        if let Ok(v) = s.parse::<i64>() {
            return Ok(Self::Small(v));
        }
        Ok(Self::Big(s.into_boxed_str()))
    }

    /// `int.__repr__`.
    pub fn write_repr(&self, out: &mut String) {
        match self {
            Self::Small(v) => {
                use std::fmt::Write;
                let _ = write!(out, "{v}");
            }
            Self::Big(s) => out.push_str(s),
        }
    }

    /// Python truthiness (`0` is falsy).
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Small(0))
    }
}

/// A Python JSON value (`dict | list | str | int | float | bool | None`).
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// `None`.
    Null,
    /// `bool`.
    Bool(bool),
    /// `int`.
    Int(Int),
    /// `float` (any `f64`, including NaN and the infinities).
    Float(f64),
    /// `str` (lone surrogates remapped, see [`super::chars`]).
    Str(String),
    /// `list`.
    Array(Vec<Value>),
    /// `dict`.
    Object(Object),
}

/// Python's `type(value)`, for json_repair's `type(a) is type(b)` checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyType {
    NoneType,
    Bool,
    Int,
    Float,
    Str,
    List,
    Dict,
}

impl Value {
    /// An empty `str`.
    pub fn empty_str() -> Self {
        Self::Str(String::new())
    }

    /// Python's `bool(value)`.
    pub fn truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Bool(b) => *b,
            Self::Int(i) => !i.is_zero(),
            Self::Float(f) => *f != 0.0,
            Self::Str(s) => !s.is_empty(),
            Self::Array(a) => !a.is_empty(),
            Self::Object(o) => !o.is_empty(),
        }
    }

    /// `type(value)`.
    pub fn py_type(&self) -> PyType {
        match self {
            Self::Null => PyType::NoneType,
            Self::Bool(_) => PyType::Bool,
            Self::Int(_) => PyType::Int,
            Self::Float(_) => PyType::Float,
            Self::Str(_) => PyType::Str,
            Self::Array(_) => PyType::List,
            Self::Object(_) => PyType::Dict,
        }
    }

    /// `value == s` for a Python `str` `s` (false for every non-`str`).
    pub fn is_str(&self, s: &str) -> bool {
        matches!(self, Self::Str(v) if v == s)
    }

    /// json_repair's `ObjectComparer.is_strictly_empty`: an empty `str`, `list` or `dict`.
    pub fn is_strictly_empty(&self) -> bool {
        match self {
            Self::Str(s) => s.is_empty(),
            Self::Array(a) => a.is_empty(),
            Self::Object(o) => o.is_empty(),
            _ => false,
        }
    }

    /// `isinstance(value, (dict, list))`.
    pub fn is_container(&self) -> bool {
        matches!(self, Self::Array(_) | Self::Object(_))
    }
}

/// json_repair's `ObjectComparer.is_same_object`: same types and the same
/// structure (dict keys and list lengths), recursively; leaf values are not
/// compared. Iterative, so deep values cannot overflow the stack.
pub fn is_same_object(a: &Value, b: &Value) -> bool {
    let mut stack: Vec<(&Value, &Value)> = vec![(a, b)];
    while let Some((x, y)) = stack.pop() {
        if x.py_type() != y.py_type() {
            return false;
        }
        match (x, y) {
            (Value::Object(ox), Value::Object(oy)) => {
                if ox.len() != oy.len() {
                    return false;
                }
                // Python iterates `obj1` in order and returns at the first
                // mismatch; the answer (a bool) does not depend on the order.
                for (k, vx) in ox {
                    match oy.get(k) {
                        None => return false,
                        Some(vy) => stack.push((vx, vy)),
                    }
                }
            }
            (Value::Array(ax), Value::Array(ay)) => {
                if ax.len() != ay.len() {
                    return false;
                }
                stack.extend(ax.iter().zip(ay));
            }
            _ => {}
        }
    }
    true
}

/// Drop a tree iteratively. The default (recursive) drop could overflow the
/// stack on deeply nested values, so callers drop whole results with this.
pub fn drop_deep(value: Value) {
    let mut stack: Vec<Value> = vec![value];
    while let Some(v) = stack.pop() {
        match v {
            Value::Array(mut a) if a.iter().any(Value::is_container) => stack.append(&mut a),
            Value::Object(o) if o.values().any(Value::is_container) => {
                stack.extend(o.into_iter().map(|(_, v)| v));
            }
            _ => {}
        }
    }
}

/// Values held while the parser can still fail, dropped with [`drop_deep`] on every path
/// (including `?` returns): they may come from `raw_decode`, which has no depth limit.
#[derive(Debug, Default)]
pub(crate) struct DeepVec(pub(crate) Vec<Value>);

impl Drop for DeepVec {
    fn drop(&mut self) {
        drop_deep(Value::Array(std::mem::take(&mut self.0)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_parse_matches_python() {
        assert_eq!(Int::parse_py("007", 4300), Ok(Int::Small(7)));
        assert_eq!(Int::parse_py("-0", 4300), Ok(Int::Small(0)));
        assert_eq!(Int::parse_py("+12", 4300), Ok(Int::Small(12)));
        assert_eq!(
            Int::parse_py("-9223372036854775808", 4300),
            Ok(Int::Small(i64::MIN))
        );
        assert_eq!(
            Int::parse_py("9223372036854775808", 4300),
            Ok(Int::Big("9223372036854775808".into()))
        );
        assert_eq!(
            Int::parse_py("-00012345678901234567890", 4300),
            Ok(Int::Big("-12345678901234567890".into()))
        );
        assert_eq!(Int::parse_py("-", 4300), Err(IntError::Invalid));
        assert_eq!(Int::parse_py("1-2", 4300), Err(IntError::Invalid));
        assert_eq!(Int::parse_py("", 4300), Err(IntError::Invalid));
        let long = "0".repeat(4300) + "1";
        assert_eq!(
            Int::parse_py(&long, 4300),
            Err(IntError::TooManyDigits {
                limit: 4300,
                digits: 4301
            })
        );
        assert!(Int::parse_py(&long, 0).is_ok());
        assert!(Int::parse_py(&("-".to_owned() + &"1".repeat(4300)), 4300).is_ok());
    }

    #[test]
    fn dict_semantics() {
        let mut o = new_object();
        o.insert("a".into(), Value::Int(Int::Small(1)));
        o.insert("b".into(), Value::Int(Int::Small(2)));
        o.insert("a".into(), Value::Int(Int::Small(3)));
        let keys: Vec<&str> = o.keys().map(String::as_str).collect();
        assert_eq!(keys, ["a", "b"]);
        assert_eq!(o["a"], Value::Int(Int::Small(3)));
    }

    #[test]
    fn same_object_and_truthiness() {
        let a = Value::Array(vec![Value::Int(Int::Small(1))]);
        let b = Value::Array(vec![Value::Int(Int::Small(2))]);
        let c = Value::Array(vec![Value::Bool(true)]);
        assert!(is_same_object(&a, &b));
        assert!(!is_same_object(&a, &c));
        assert!(!Value::Float(-0.0).truthy());
        assert!(Value::Float(f64::NAN).truthy());
        assert!(!Value::empty_str().truthy());
    }

    #[test]
    fn deep_drop_does_not_overflow() {
        let mut v = Value::Null;
        for _ in 0..200_000 {
            v = Value::Array(vec![v]);
        }
        drop_deep(v);
    }
}
