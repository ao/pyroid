//! Reading sequences for `SequenceMatcher`: which element types run
//! natively, and their keys.
//!
//! An element runs natively if its equality and hash are known without
//! calling Python: the characters of an exact `str`, items of an exact
//! `list`/`tuple` that are exact `str`, `int`, `bool` or `bytes`, and the
//! values of `bytes`/`bytearray`. Equal elements get equal keys, and keys of
//! different kinds never compare equal (as `'a' != 97 != b'a'` in Python).
//! Anything else is `None` here and runs CPython's own code.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyInt, PyList, PyString, PyTuple};

/// The code points of `s`, lone surrogates included.
pub fn code_points(s: &Bound<'_, PyString>) -> PyResult<Vec<u32>> {
    if let Ok(text) = s.to_str() {
        return Ok(text.chars().map(u32::from).collect());
    }
    let encoded = s.call_method1(
        pyo3::intern!(s.py(), "encode"),
        ("utf-32-le", "surrogatepass"),
    )?;
    let bytes = encoded.cast::<PyBytes>()?.as_bytes();
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Append code point `c` in generalized UTF-8 (surrogates as 3 bytes, as
/// `str.encode('utf-8', 'surrogatepass')` writes them).
pub fn push_wtf8(c: u32, out: &mut Vec<u8>) {
    if let Some(ch) = char::from_u32(c) {
        let mut buf = [0; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
    } else {
        #[allow(clippy::cast_possible_truncation)] // c is a surrogate, below 0x10000
        out.extend_from_slice(&[
            0xE0 | (c >> 12) as u8,
            0x80 | ((c >> 6) & 0x3F) as u8,
            0x80 | (c & 0x3F) as u8,
        ]);
    }
}

/// The bytes of `s` as a key: its UTF-8, or the `surrogatepass` encoding.
pub fn text_key(s: &Bound<'_, PyString>) -> PyResult<Box<[u8]>> {
    if let Ok(text) = s.to_str() {
        return Ok(text.as_bytes().into());
    }
    let encoded = s.call_method1(pyo3::intern!(s.py(), "encode"), ("utf-8", "surrogatepass"))?;
    Ok(encoded.cast::<PyBytes>()?.as_bytes().into())
}

/// An element's key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    /// A `str` (its generalized UTF-8).
    Text(Box<[u8]>),
    /// An `int` or `bool`.
    Int(i64),
    /// A `bytes`.
    Bytes(Box<[u8]>),
}

/// The key of a list or tuple item, or `None` if its type does not run natively.
pub fn item_key(item: &Bound<'_, PyAny>) -> PyResult<Option<Key>> {
    if let Ok(s) = item.cast_exact::<PyString>() {
        return Ok(Some(Key::Text(text_key(s)?)));
    }
    if item.is_exact_instance_of::<PyInt>() {
        return Ok(item.extract::<i64>().ok().map(Key::Int));
    }
    if let Ok(b) = item.cast_exact::<PyBool>() {
        return Ok(Some(Key::Int(i64::from(b.is_true()))));
    }
    if let Ok(b) = item.cast_exact::<PyBytes>() {
        return Ok(Some(Key::Bytes(b.as_bytes().into())));
    }
    Ok(None)
}

/// A sequence as `SequenceMatcher` reads it.
pub enum Seq<'py> {
    /// The characters of a `str`.
    Chars(Vec<u32>),
    /// The values of `bytes` or `bytearray`.
    Values(Vec<u8>),
    /// The items of a `list` (a snapshot) or `tuple`.
    Items(Bound<'py, PyTuple>),
}

impl<'py> Seq<'py> {
    /// `obj` if it is an exact `str`, `bytes`, `bytearray`, `list` or `tuple`.
    pub fn read(obj: &Bound<'py, PyAny>) -> PyResult<Option<Self>> {
        if let Ok(s) = obj.cast_exact::<PyString>() {
            return Ok(Some(Self::Chars(code_points(s)?)));
        }
        if let Ok(b) = obj.cast_exact::<PyBytes>() {
            return Ok(Some(Self::Values(b.as_bytes().to_vec())));
        }
        if let Ok(b) = obj.cast_exact::<PyByteArray>() {
            return Ok(Some(Self::Values(b.to_vec())));
        }
        if let Ok(list) = obj.cast_exact::<PyList>() {
            // One step (PyList_AsTuple): another thread may change the list.
            return Ok(Some(Self::Items(list.to_tuple())));
        }
        if let Ok(tuple) = obj.cast_exact::<PyTuple>() {
            return Ok(Some(Self::Items(tuple.clone())));
        }
        Ok(None)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            Self::Chars(c) => c.len(),
            Self::Values(v) => v.len(),
            Self::Items(t) => t.len(),
        }
    }
}

/// Which kinds of keys a sequence holds (`str` and `bytes` elements must
/// not meet: comparing them warns under `python -b`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Kinds {
    pub text: bool,
    pub bytes: bool,
}

impl Kinds {
    pub fn of(key: &Key) -> Self {
        match key {
            Key::Text(_) => Self {
                text: true,
                bytes: false,
            },
            Key::Bytes(_) => Self {
                text: false,
                bytes: true,
            },
            Key::Int(_) => Self::default(),
        }
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            text: self.text || other.text,
            bytes: self.bytes || other.bytes,
        }
    }

    /// True if `str` and `bytes` elements would be compared.
    pub fn mixed(self) -> bool {
        self.text && self.bytes
    }
}
