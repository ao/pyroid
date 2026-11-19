//! Reading sequences for `SequenceMatcher`: which element types run
//! natively, and their keys.
//!
//! An element runs natively if its equality and hash are known without
//! calling Python: the characters of a `str`, items of a `list`/`tuple` that
//! are `str`, `int` (any size), `bool` or `bytes`, and the values of
//! `bytes`/`bytearray`. Subclasses count as their base if they change none of
//! [`OVERRIDES`]. Equal elements get equal keys, and keys of different kinds
//! never compare equal (as `'a' != 97 != b'a'` in Python). Anything else is
//! `None` here and runs CPython's own code.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyByteArray, PyBytes, PyDict, PyInt, PyList, PyString, PyTuple, PyType};

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
    /// An `int` or `bool` in the signed 64-bit range.
    Int(i64),
    /// A larger `int`: its shortest two's complement bytes, little-endian.
    BigInt(Box<[u8]>),
    /// A `bytes`.
    Bytes(Box<[u8]>),
}

/// The built-in type a native object is, or behaves as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    Str,
    Int,
    Bytes,
    ByteArray,
    List,
    Tuple,
}

/// The methods a subclass must not define (anywhere before the built-in base
/// in its MRO) to run natively: difflib reaches elements through equality and
/// hashing (`b2j`, `==`), sequences through `len`, indexing and iteration, and
/// `get_close_matches` sorts its candidates.
const OVERRIDES: [&str; 10] = [
    "__eq__",
    "__ne__",
    "__hash__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__len__",
    "__getitem__",
    "__iter__",
];

/// What a subclass's type is: its base, or the first method in
/// [`OVERRIDES`] it changes.
#[derive(Clone, Debug)]
pub enum Verdict {
    Plain(Base),
    Overrides(&'static str),
    Other,
}

/// The [`Verdict`]s of the types seen while reading one sequence.
#[derive(Default)]
pub struct Types {
    seen: Vec<(usize, Verdict)>,
}

impl Types {
    /// The verdict for `obj`'s type (not one of the exact built-ins).
    pub fn verdict(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<Verdict> {
        let ty = obj.get_type();
        // The objects being read keep their types alive: the address is the type.
        let id = ty.as_ptr() as usize;
        if let Some((_, v)) = self.seen.iter().find(|(t, _)| *t == id) {
            return Ok(v.clone());
        }
        let v = verdict(&ty)?;
        if self.seen.len() < 8 {
            self.seen.push((id, v.clone()));
        }
        Ok(v)
    }
}

fn verdict(ty: &Bound<'_, PyType>) -> PyResult<Verdict> {
    let py = ty.py();
    let base = if ty.is_subclass_of::<PyString>()? {
        (Base::Str, py.get_type::<PyString>())
    } else if ty.is_subclass_of::<PyInt>()? {
        (Base::Int, py.get_type::<PyInt>())
    } else if ty.is_subclass_of::<PyBytes>()? {
        (Base::Bytes, py.get_type::<PyBytes>())
    } else if ty.is_subclass_of::<PyByteArray>()? {
        (Base::ByteArray, py.get_type::<PyByteArray>())
    } else if ty.is_subclass_of::<PyList>()? {
        (Base::List, py.get_type::<PyList>())
    } else if ty.is_subclass_of::<PyTuple>()? {
        (Base::Tuple, py.get_type::<PyTuple>())
    } else {
        return Ok(Verdict::Other);
    };
    for cls in ty
        .getattr(pyo3::intern!(py, "__mro__"))?
        .cast::<PyTuple>()?
        .iter()
    {
        if cls.is(&base.1) {
            break;
        }
        let d = cls.getattr(pyo3::intern!(py, "__dict__"))?;
        for name in OVERRIDES {
            if d.contains(name)? {
                return Ok(Verdict::Overrides(name));
            }
        }
    }
    Ok(Verdict::Plain(base.0))
}

/// The key of an `int` (`bool` included), any size.
fn int_key(item: &Bound<'_, PyAny>) -> PyResult<Key> {
    if let Ok(v) = item.extract::<i64>() {
        return Ok(Key::Int(v));
    }
    // int's own methods: a subclass may change them, not the value.
    let py = item.py();
    let int = py.get_type::<PyInt>();
    let bits: usize = int
        .call_method1(pyo3::intern!(py, "bit_length"), (item,))?
        .extract()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(pyo3::intern!(py, "signed"), true)?;
    let bytes = int.getattr(pyo3::intern!(py, "to_bytes"))?.call(
        (item, bits / 8 + 1, pyo3::intern!(py, "little")),
        Some(&kwargs),
    )?;
    Ok(Key::BigInt(bytes.cast::<PyBytes>()?.as_bytes().into()))
}

/// The key of a list or tuple item, or `None` if its type does not run natively.
pub fn item_key(item: &Bound<'_, PyAny>, types: &mut Types) -> PyResult<Option<Key>> {
    if let Ok(s) = item.cast_exact::<PyString>() {
        return Ok(Some(Key::Text(text_key(s)?)));
    }
    if item.is_exact_instance_of::<PyInt>() {
        return int_key(item).map(Some);
    }
    if let Ok(b) = item.cast_exact::<PyBool>() {
        return Ok(Some(Key::Int(i64::from(b.is_true()))));
    }
    if let Ok(b) = item.cast_exact::<PyBytes>() {
        return Ok(Some(Key::Bytes(b.as_bytes().into())));
    }
    Ok(match types.verdict(item)? {
        Verdict::Plain(Base::Str) => Some(Key::Text(text_key(item.cast::<PyString>()?)?)),
        Verdict::Plain(Base::Int) => Some(int_key(item)?),
        Verdict::Plain(Base::Bytes) => Some(Key::Bytes(item.cast::<PyBytes>()?.as_bytes().into())),
        _ => None,
    })
}

/// True under `python -b`: comparing a `str` with a `bytes` warns then, so
/// sequences that would compare them run CPython's code, which does.
pub fn bytes_warnings(py: Python<'_>) -> PyResult<bool> {
    static FLAG: PyOnceLock<bool> = PyOnceLock::new();
    FLAG.get_or_try_init(py, || -> PyResult<bool> {
        let flag: i64 = py
            .import("sys")?
            .getattr("flags")?
            .getattr("bytes_warning")?
            .extract()?;
        Ok(flag != 0)
    })
    .copied()
}

fn qualname(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.get_type()
        .getattr(pyo3::intern!(obj.py(), "__qualname__"))?
        .extract()
}

fn type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(format!("'{}'", qualname(obj)?))
}

/// Why `seq` (a matcher's `a` or `b`) does not run natively, for the
/// `FallbackWarning`; `None` if it does.
pub fn why_not(seq: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let mut types = Types::default();
    let Some(read) = Seq::read(seq, &mut types)? else {
        let t = type_name(seq)?;
        return Ok(Some(match types.verdict(seq)? {
            Verdict::Overrides(name) => format!("a sequence of type {t} (it overrides {name})"),
            _ => format!("a sequence of type {t}"),
        }));
    };
    if read.len() >= u32::MAX as usize {
        return Ok(Some("a sequence of 2**32 - 1 elements or more".into()));
    }
    if let Seq::Items(items) = read {
        for item in items.iter() {
            if item_key(&item, &mut types)?.is_none() {
                let (s, t) = (qualname(seq)?, type_name(&item)?);
                return Ok(Some(match types.verdict(&item)? {
                    Verdict::Overrides(name) => {
                        format!("a {s} with an element of type {t} (it overrides {name})")
                    }
                    _ => format!("a {s} with an element of type {t}"),
                }));
            }
        }
    }
    Ok(None)
}

/// The [`Kinds`] of `seq`'s elements (empty if it does not run natively).
pub fn kinds_of(seq: &Bound<'_, PyAny>) -> PyResult<Kinds> {
    let mut types = Types::default();
    Ok(match Seq::read(seq, &mut types)? {
        Some(Seq::Chars(_)) => Kinds {
            text: true,
            bytes: false,
        },
        Some(Seq::Items(items)) => {
            let mut kinds = Kinds::default();
            for item in items.iter() {
                if let Some(key) = item_key(&item, &mut types)? {
                    kinds = kinds.union(Kinds::of(&key));
                }
            }
            kinds
        }
        _ => Kinds::default(),
    })
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
    /// `obj` if it is a `str`, `bytes`, `bytearray`, `list` or `tuple`, or a
    /// subclass of one that keeps its indexing and comparisons.
    pub fn read(obj: &Bound<'py, PyAny>, types: &mut Types) -> PyResult<Option<Self>> {
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
        Ok(match types.verdict(obj)? {
            Verdict::Plain(Base::Str) => Some(Self::Chars(code_points(obj.cast::<PyString>()?)?)),
            Verdict::Plain(Base::Bytes) => {
                Some(Self::Values(obj.cast::<PyBytes>()?.as_bytes().to_vec()))
            }
            Verdict::Plain(Base::ByteArray) => {
                Some(Self::Values(obj.cast::<PyByteArray>()?.to_vec()))
            }
            Verdict::Plain(Base::List) => Some(Self::Items(obj.cast::<PyList>()?.to_tuple())),
            Verdict::Plain(Base::Tuple) => Some(Self::Items(obj.cast::<PyTuple>()?.clone())),
            _ => None,
        })
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
            Key::Int(_) | Key::BigInt(_) => Self::default(),
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
