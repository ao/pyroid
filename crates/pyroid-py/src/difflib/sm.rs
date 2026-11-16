//! The native state behind `pyroid.difflib.SequenceMatcher`.
//!
//! The Python class keeps CPython's instance attributes (`a`, `b`, `isjunk`,
//! `autojunk`, `opcodes`, ...) and stores three frozen objects in the
//! instance dict: `_pyroid_b` ([`BSide`], what `set_seq2` built),
//! `_pyroid_a` ([`ASide`], `a` encoded against it) and `_pyroid_mb`
//! ([`Blocks`], the matching blocks). Each records the objects it was made
//! from, and is only used while `self.a`/`self.b` are still those objects.
//! The functions here return `None` when the native path does not apply;
//! the Python side then runs CPython's method.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyInt, PyList, PySet, PyString, PyTuple};
use pyroid_core::difflib::intern::{CharInterner, Interner};
use pyroid_core::difflib::matcher::{
    BIndex, Block, Kind, NONE, Tag, calculate_ratio, matches, opcodes,
};

use super::seq::{Key, Kinds, Seq, item_key, push_wtf8};

/// How `b`'s elements were numbered.
#[derive(Debug)]
enum Keys {
    /// `b` is a `str`: by code point.
    Chars(Box<CharInterner>),
    /// By [`Key`]; `reps[id]` is the first element with that key.
    Items {
        keys: Interner<Key>,
        reps: Vec<Py<PyAny>>,
        kinds: Kinds,
    },
    /// `b` is `bytes`/`bytearray`: by value; `values[id]` is the value.
    Values {
        keys: Interner<Key>,
        values: Vec<u8>,
    },
}

/// `set_seq2(b)`'s result.
#[pyclass(frozen, module = "pyroid.difflib", name = "_BSide")]
#[derive(Debug)]
pub struct BSide {
    /// The `b` it was built from.
    #[pyo3(get)]
    b: Py<PyAny>,
    keys: Keys,
    index: BIndex,
}

/// `a`, encoded against a [`BSide`].
#[pyclass(frozen, module = "pyroid.difflib", name = "_ASide")]
#[derive(Debug)]
pub struct ASide {
    a: Py<PyAny>,
    side: Py<BSide>,
    ids: Vec<u32>,
}

/// The matching blocks of `(a, b)`.
#[pyclass(frozen, module = "pyroid.difflib", name = "_Blocks")]
#[derive(Debug)]
pub struct Blocks {
    a: Py<PyAny>,
    b: Py<PyAny>,
    found: Vec<Block>,
}

static MATCH: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static TUPLE_NEW: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `difflib.Match(a, b, size)`, made as `tuple.__new__(Match, (a, b, size))`
/// (what the namedtuple's `__new__` does).
fn new_match(py: Python<'_>, a: usize, b: usize, size: usize) -> PyResult<Bound<'_, PyAny>> {
    let cls = MATCH.get_or_try_init(py, || -> PyResult<_> {
        Ok(py.import("difflib")?.getattr("Match")?.unbind())
    })?;
    let new = TUPLE_NEW.get_or_try_init(py, || -> PyResult<_> {
        Ok(py.get_type::<PyTuple>().getattr("__new__")?.unbind())
    })?;
    new.bind(py).call1((cls.bind(py), (a, b, size)))
}

impl BSide {
    /// The element object for id `id` (CPython's `b2j` key).
    fn rep<'py>(&self, py: Python<'py>, id: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match &self.keys {
            Keys::Chars(chars) => code_point_str(py, chars.chars()[id])?.into_any(),
            Keys::Items { reps, .. } => reps[id].bind(py).clone(),
            Keys::Values { values, .. } => PyInt::new(py, values[id]).into_any(),
        })
    }

    fn ids_where(&self, keep: impl Fn(Kind) -> bool) -> Vec<usize> {
        #[allow(clippy::cast_possible_truncation)] // ids are u32
        (0..self.index.keys())
            .filter(|&id| keep(self.index.kind(id as u32)))
            .collect()
    }
}

/// The one-character `str` for code point `c` (lone surrogates included).
fn code_point_str(py: Python<'_>, c: u32) -> PyResult<Bound<'_, PyString>> {
    if let Some(ch) = char::from_u32(c) {
        let mut buf = [0; 4];
        return Ok(PyString::new(py, ch.encode_utf8(&mut buf)));
    }
    let chr = py.import("builtins")?.getattr("chr")?;
    Ok(chr.call1((c,))?.cast_into::<PyString>()?)
}

#[pymethods]
impl BSide {
    /// `b2j`: element -> positions, junk and popular elements left out.
    fn b2j<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for id in self.ids_where(|k| k == Kind::Normal) {
            #[allow(clippy::cast_possible_truncation)]
            let pos = self.index.positions(id as u32);
            d.set_item(self.rep(py, id)?, PyList::new(py, pos)?)?;
        }
        Ok(d)
    }

    /// `bjunk`.
    fn bjunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let s = PySet::empty(py)?;
        for id in self.ids_where(|k| k == Kind::Junk) {
            s.add(self.rep(py, id)?)?;
        }
        Ok(s)
    }

    /// `bpopular`.
    fn bpopular<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let s = PySet::empty(py)?;
        for id in self.ids_where(|k| k == Kind::Popular) {
            s.add(self.rep(py, id)?)?;
        }
        Ok(s)
    }

    /// `fullbcount`: element -> count in `b`.
    fn fullbcount<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for id in 0..self.index.keys() {
            #[allow(clippy::cast_possible_truncation)]
            d.set_item(self.rep(py, id)?, self.index.count(id as u32))?;
        }
        Ok(d)
    }
}

/// `b` numbered by element, before `__chain_b`'s junk and popular passes:
/// the [`BSide`] (with an empty index) and `b` as ids. `None` if `b`'s
/// elements do not run natively.
fn number_b(b: &Bound<'_, PyAny>) -> PyResult<Option<(BSide, Vec<u32>)>> {
    let Some(seq) = Seq::read(b)? else {
        return Ok(None);
    };
    if seq.len() >= u32::MAX as usize {
        return Ok(None);
    }
    let (keys, ids) = match seq {
        Seq::Chars(cps) => {
            let (chars, ids) = CharInterner::build(cps);
            (Keys::Chars(Box::new(chars)), ids)
        }
        Seq::Values(values) => {
            let mut keys = Interner::default();
            let mut distinct = Vec::new();
            let ids = values
                .iter()
                .map(|&v| {
                    let (id, new) = keys.intern(Key::Int(i64::from(v)));
                    if new {
                        distinct.push(v);
                    }
                    id
                })
                .collect();
            (
                Keys::Values {
                    keys,
                    values: distinct,
                },
                ids,
            )
        }
        Seq::Items(items) => {
            let mut keys = Interner::default();
            let mut reps = Vec::new();
            let mut kinds = Kinds::default();
            let mut ids = Vec::with_capacity(items.len());
            for item in items.iter() {
                let Some(key) = item_key(&item)? else {
                    return Ok(None);
                };
                kinds = kinds.union(Kinds::of(&key));
                let (id, new) = keys.intern(key);
                if new {
                    reps.push(item.unbind());
                }
                ids.push(id);
            }
            if kinds.mixed() {
                return Ok(None);
            }
            (Keys::Items { keys, reps, kinds }, ids)
        }
    };
    let side = BSide {
        b: b.clone().unbind(),
        keys,
        index: BIndex::new(Vec::new(), 0, &[], false),
    };
    Ok(Some((side, ids)))
}

impl BSide {
    fn distinct(&self) -> usize {
        match &self.keys {
            Keys::Chars(c) => c.chars().len(),
            Keys::Items { reps, .. } => reps.len(),
            Keys::Values { values, .. } => values.len(),
        }
    }
}

/// `__chain_b`'s isjunk pass: `isjunk` (if truthy) on each distinct element,
/// in `b2j` order. On an error, the verdicts made so far come with it.
fn junk_pass(py: Python<'_>, side: &BSide, isjunk: &Bound<'_, PyAny>) -> Result<Vec<bool>, (PyErr, Vec<bool>)> {
    let mut junk = Vec::new();
    match isjunk.is_truthy() {
        Ok(false) => return Ok(junk),
        Ok(true) => {}
        Err(e) => return Err((e, junk)),
    }
    let n = side.distinct();
    junk.reserve(n);
    for id in 0..n {
        let verdict = side
            .rep(py, id)
            .and_then(|rep| isjunk.call1((rep,)))
            .and_then(|r| r.is_truthy());
        match verdict {
            Ok(v) => junk.push(v),
            Err(e) => return Err((e, junk)),
        }
    }
    Ok(junk)
}

/// `a` as ids of `side`'s elements; `None` if it does not run natively.
fn encode_a(side: &BSide, a: &Bound<'_, PyAny>) -> PyResult<Option<Vec<u32>>> {
    let Some(seq) = Seq::read(a)? else {
        return Ok(None);
    };
    let mut buf = Vec::new();
    Ok(Some(match (&side.keys, seq) {
        (Keys::Chars(chars), Seq::Chars(cps)) => chars.encode(cps),
        (Keys::Chars(_), Seq::Values(v)) => vec![NONE; v.len()],
        (Keys::Chars(chars), Seq::Items(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            for item in items.iter() {
                match item_key(&item)? {
                    None | Some(Key::Bytes(_)) => return Ok(None),
                    Some(Key::Text(t)) => {
                        let s = std::str::from_utf8(&t).ok();
                        let mut it = s.into_iter().flat_map(str::chars);
                        ids.push(match (it.next(), it.next()) {
                            (Some(c), None) => chars.get(u32::from(c)),
                            _ => single_surrogate(&t).map_or(NONE, |c| chars.get(c)),
                        });
                    }
                    Some(Key::Int(_)) => ids.push(NONE),
                }
            }
            ids
        }
        (Keys::Items { keys, kinds, .. }, Seq::Chars(cps)) => {
            if kinds.bytes {
                return Ok(None);
            }
            cps.iter()
                .map(|&c| {
                    buf.clear();
                    push_wtf8(c, &mut buf);
                    keys.get(&Key::Text(buf.as_slice().into()))
                })
                .collect()
        }
        (Keys::Items { keys, .. } | Keys::Values { keys, .. }, Seq::Values(v)) => v
            .iter()
            .map(|&x| keys.get(&Key::Int(i64::from(x))))
            .collect(),
        (Keys::Items { keys, kinds, .. }, Seq::Items(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            let mut mine = Kinds::default();
            for item in items.iter() {
                let Some(key) = item_key(&item)? else {
                    return Ok(None);
                };
                mine = mine.union(Kinds::of(&key));
                ids.push(keys.get(&key));
            }
            if mine.union(*kinds).mixed() {
                return Ok(None);
            }
            ids
        }
        (Keys::Values { .. }, Seq::Chars(cps)) => vec![NONE; cps.len()],
        (Keys::Values { keys, .. }, Seq::Items(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            for item in items.iter() {
                match item_key(&item)? {
                    None => return Ok(None),
                    Some(key @ Key::Int(_)) => ids.push(keys.get(&key)),
                    Some(_) => ids.push(NONE),
                }
            }
            ids
        }
    }))
}

/// The code point of a one-surrogate generalized-UTF-8 key.
fn single_surrogate(t: &[u8]) -> Option<u32> {
    match *t {
        [0xED, b1 @ 0xA0..=0xBF, b2 @ 0x80..=0xBF] => {
            Some(0xD000 | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F))
        }
        _ => None,
    }
}

fn instance_dict<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(obj
        .getattr(pyo3::intern!(obj.py(), "__dict__"))?
        .cast_into::<PyDict>()?)
}

/// The instance's [`BSide`] and [`ASide`] for its current `a` and `b`
/// (encoding `a` if needed), or `None`.
fn state<'py>(
    obj: &Bound<'py, PyAny>,
    d: &Bound<'py, PyDict>,
) -> PyResult<Option<(Bound<'py, BSide>, Bound<'py, ASide>)>> {
    let py = obj.py();
    let Some(side) = d.get_item(pyo3::intern!(py, "_pyroid_b"))? else {
        return Ok(None);
    };
    let Ok(side) = side.cast_into::<BSide>() else {
        return Ok(None);
    };
    let b = obj.getattr(pyo3::intern!(py, "b"))?;
    if !side.get().b.bind(py).is(&b) {
        return Ok(None);
    }
    let a = obj.getattr(pyo3::intern!(py, "a"))?;
    if let Some(cached) = d.get_item(pyo3::intern!(py, "_pyroid_a"))? {
        if let Ok(cached) = cached.cast_into::<ASide>() {
            let c = cached.get();
            if c.a.bind(py).is(&a) && c.side.bind(py).is(&side) {
                return Ok(Some((side, cached)));
            }
        }
    }
    let Some(ids) = encode_a(side.get(), &a)? else {
        return Ok(None);
    };
    let aside = Bound::new(
        py,
        ASide {
            a: a.unbind(),
            side: side.clone().unbind(),
            ids,
        },
    )?;
    d.set_item(pyo3::intern!(py, "_pyroid_a"), &aside)?;
    Ok(Some((side, aside)))
}

/// The cached blocks, if they are for the current `a` and `b`.
fn valid_blocks<'py>(
    obj: &Bound<'py, PyAny>,
    d: &Bound<'py, PyDict>,
) -> PyResult<Option<Bound<'py, Blocks>>> {
    let py = obj.py();
    let Some(cached) = d.get_item(pyo3::intern!(py, "_pyroid_mb"))? else {
        return Ok(None);
    };
    let Ok(cached) = cached.cast_into::<Blocks>() else {
        return Ok(None);
    };
    let c = cached.get();
    let current = c.a.bind(py).is(obj.getattr(pyo3::intern!(py, "a"))?)
        && c.b.bind(py).is(obj.getattr(pyo3::intern!(py, "b"))?);
    Ok(current.then_some(cached))
}

/// The matching blocks for the current `a` and `b`: cached, or computed and
/// cached; `None` without a native state.
fn blocks<'py>(
    obj: &Bound<'py, PyAny>,
    d: &Bound<'py, PyDict>,
) -> PyResult<Option<Bound<'py, Blocks>>> {
    let py = obj.py();
    if let Some(cached) = valid_blocks(obj, d)? {
        return Ok(Some(cached));
    }
    let Some((side, aside)) = state(obj, d)? else {
        return Ok(None);
    };
    let (s, a) = (side.get(), aside.get());
    let found = if a.ids.len() + s.index.b().len() > 4096 {
        py.detach(|| s.index.matching_blocks(&a.ids))
    } else {
        s.index.matching_blocks(&a.ids)
    };
    let out = Bound::new(
        py,
        Blocks {
            a: a.a.clone_ref(py),
            b: s.b.clone_ref(py),
            found,
        },
    )?;
    d.set_item(pyo3::intern!(py, "_pyroid_mb"), &out)?;
    Ok(Some(out))
}

/// True if the instance dict holds a computed `matching_blocks` (CPython's
/// cache, which then wins: the caller may have changed that list).
fn has_list(d: &Bound<'_, PyDict>, name: &Bound<'_, PyString>) -> PyResult<bool> {
    Ok(d.get_item(name)?.is_some_and(|v| !v.is_none()))
}

fn matches_list<'py>(py: Python<'py>, blocks: &[Block]) -> PyResult<Bound<'py, PyList>> {
    let items = blocks
        .iter()
        .map(|m| new_match(py, m.a, m.b, m.size))
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

fn opcodes_list<'py>(py: Python<'py>, blocks: &[Block]) -> PyResult<Bound<'py, PyList>> {
    let tags = [
        pyo3::intern!(py, "replace"),
        pyo3::intern!(py, "delete"),
        pyo3::intern!(py, "insert"),
        pyo3::intern!(py, "equal"),
    ];
    let ops = opcodes(blocks);
    let items = ops
        .iter()
        .map(|o| {
            let tag = match o.tag {
                Tag::Replace => tags[0],
                Tag::Delete => tags[1],
                Tag::Insert => tags[2],
                Tag::Equal => tags[3],
            };
            (tag, o.i1, o.i2, o.j1, o.j2).into_pyobject(py)
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, items)
}

/// `set_seq2(b)` after the identity check: reset the caches, build the
/// index. False if `b` does not run natively (the caller chains it as
/// CPython does). If `isjunk` (or `autojunk`'s truth) raises, the instance
/// is left as CPython's `__chain_b` leaves it, without a native index.
#[pyfunction]
pub fn _sm_set_seq2(obj: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = obj.py();
    obj.setattr(pyo3::intern!(py, "b"), b)?;
    obj.setattr(pyo3::intern!(py, "opcodes"), py.None())?;
    let d = instance_dict(obj)?;
    let del = |names: &[&Bound<'_, PyString>]| -> PyResult<()> {
        for name in names {
            if d.contains(*name)? {
                d.del_item(*name)?;
            }
        }
        Ok(())
    };
    // The previous b's lazy attributes: CPython keeps bjunk and bpopular
    // until __chain_b replaces them, so a chain that fails half-way shows them.
    let old = d
        .get_item(pyo3::intern!(py, "_pyroid_b"))?
        .and_then(|o| o.cast_into::<BSide>().ok());
    let keep_old = |names: &[&str]| -> PyResult<()> {
        if let Some(old) = &old {
            for &name in names {
                if !d.contains(name)? {
                    let value = match name {
                        "bjunk" => old.get().bjunk(py)?,
                        _ => old.get().bpopular(py)?,
                    };
                    d.set_item(name, value)?;
                }
            }
        }
        Ok(())
    };
    let Some((mut side, ids)) = number_b(b)? else {
        keep_old(&["bjunk", "bpopular"])?;
        del(&[
            pyo3::intern!(py, "_pyroid_b"),
            pyo3::intern!(py, "_pyroid_a"),
            pyo3::intern!(py, "_pyroid_mb"),
            pyo3::intern!(py, "_pyroid_qr"),
        ])?;
        return Ok(false);
    };
    del(&[
        pyo3::intern!(py, "_pyroid_a"),
        pyo3::intern!(py, "_pyroid_mb"),
        pyo3::intern!(py, "_pyroid_qr"),
    ])?;
    del(&[
        pyo3::intern!(py, "matching_blocks"),
        pyo3::intern!(py, "fullbcount"),
    ])?;
    let n = side.distinct();
    let fail = |side: &mut BSide, ids: &[u32], junk: &[bool], popular: bool| -> PyResult<()> {
        // CPython's attributes at the point __chain_b raised.
        d.set_item(pyo3::intern!(py, "matching_blocks"), py.None())?;
        d.set_item(pyo3::intern!(py, "fullbcount"), py.None())?;
        let deleted = if popular { junk } else { &[] };
        side.index = BIndex::new(ids.to_vec(), n, deleted, false);
        d.set_item(pyo3::intern!(py, "b2j"), side.b2j(py)?)?;
        let found = PySet::empty(py)?;
        for (id, _) in junk.iter().enumerate().filter(|(_, j)| **j) {
            found.add(side.rep(py, id)?)?;
        }
        d.set_item(pyo3::intern!(py, "bjunk"), found)?;
        if popular {
            d.set_item(pyo3::intern!(py, "bpopular"), PySet::empty(py)?)?;
        }
        Ok(())
    };
    let junk = match obj
        .getattr(pyo3::intern!(py, "isjunk"))
        .map_err(|e| (e, Vec::new()))
        .and_then(|isjunk| junk_pass(py, &side, &isjunk))
    {
        Ok(junk) => junk,
        Err((e, junk)) => {
            keep_old(&["bpopular"])?;
            del(&[pyo3::intern!(py, "_pyroid_b")])?;
            fail(&mut side, &ids, &junk, false)?;
            return Err(e);
        }
    };
    let autojunk = match obj
        .getattr(pyo3::intern!(py, "autojunk"))
        .and_then(|v| v.is_truthy())
    {
        Ok(v) => v,
        Err(e) => {
            del(&[pyo3::intern!(py, "_pyroid_b")])?;
            fail(&mut side, &ids, &junk, true)?;
            return Err(e);
        }
    };
    side.index = BIndex::new(ids, n, &junk, autojunk);
    del(&[
        pyo3::intern!(py, "_pyroid_b"),
        pyo3::intern!(py, "b2j"),
        pyo3::intern!(py, "bjunk"),
        pyo3::intern!(py, "bpopular"),
    ])?;
    d.set_item(pyo3::intern!(py, "_pyroid_b"), Bound::new(py, side)?)?;
    Ok(true)
}

/// `ratio()`, or `None`.
#[pyfunction]
pub fn _sm_ratio(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    if has_list(&d, pyo3::intern!(py, "matching_blocks"))? {
        return Ok(None);
    }
    let Some(found) = blocks(obj, &d)? else {
        return Ok(None);
    };
    let length =
        obj.getattr(pyo3::intern!(py, "a"))?.len()? + obj.getattr(pyo3::intern!(py, "b"))?.len()?;
    Ok(Some(calculate_ratio(matches(&found.get().found), length)))
}

/// `get_matching_blocks()` when `matching_blocks` is not set: the list of
/// `Match`es (also stored as `matching_blocks`), or `None`.
#[pyfunction]
pub fn _sm_matching_blocks<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyList>>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    let Some(found) = blocks(obj, &d)? else {
        return Ok(None);
    };
    let list = matches_list(py, &found.get().found)?;
    d.set_item(pyo3::intern!(py, "matching_blocks"), &list)?;
    Ok(Some(list))
}

/// The `matching_blocks` attribute when the instance dict has none: the
/// cached blocks as a list (stored in the dict), or `None`.
#[pyfunction]
pub fn _sm_cached_blocks<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyList>>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    let Some(cached) = valid_blocks(obj, &d)? else {
        return Ok(None);
    };
    let list = matches_list(py, &cached.get().found)?;
    d.set_item(pyo3::intern!(py, "matching_blocks"), &list)?;
    Ok(Some(list))
}

/// `get_opcodes()` when `opcodes` is not set: the list (also stored as
/// `opcodes`), or `None`.
#[pyfunction]
pub fn _sm_opcodes<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyList>>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    if has_list(&d, pyo3::intern!(py, "matching_blocks"))? {
        return Ok(None);
    }
    let Some(found) = blocks(obj, &d)? else {
        return Ok(None);
    };
    let list = opcodes_list(py, &found.get().found)?;
    obj.setattr(pyo3::intern!(py, "opcodes"), &list)?;
    Ok(Some(list))
}

/// `quick_ratio()`, or `None`.
#[pyfunction]
pub fn _sm_quick_ratio(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    let Some((side, aside)) = state(obj, &d)? else {
        return Ok(None);
    };
    let (s, a) = (side.get(), aside.get());
    // CPython builds fullbcount on the first call: the attribute shows it from now on.
    d.set_item(pyo3::intern!(py, "_pyroid_qr"), true)?;
    let length = a.ids.len() + s.index.b().len();
    Ok(Some(calculate_ratio(s.index.quick_matches(&a.ids), length)))
}

/// `find_longest_match(alo, ahi, blo, bhi)`, or `None` (bounds that are not
/// plain ints within the sequences run CPython's code).
#[pyfunction]
#[allow(clippy::many_single_char_names)] // CPython's names
pub fn _sm_find_longest_match<'py>(
    obj: &Bound<'py, PyAny>,
    alo: &Bound<'py, PyAny>,
    ahi: &Bound<'py, PyAny>,
    blo: &Bound<'py, PyAny>,
    bhi: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    let Some((side, aside)) = state(obj, &d)? else {
        return Ok(None);
    };
    let (s, a) = (side.get(), aside.get());
    let (la, lb) = (a.ids.len(), s.index.b().len());
    let bound = |x: &Bound<'py, PyAny>, default: Option<usize>, len: usize| -> Option<usize> {
        if x.is_none() {
            return default;
        }
        if !x.is_exact_instance_of::<PyInt>() {
            return None;
        }
        x.extract::<usize>().ok().filter(|&v| v <= len)
    };
    let (Some(alo), Some(ahi), Some(blo), Some(bhi)) = (
        bound(alo, None, la),
        bound(ahi, Some(la), la),
        bound(blo, None, lb),
        bound(bhi, Some(lb), lb),
    ) else {
        return Ok(None);
    };
    let mut scratch = s.index.scratch();
    let (i, j, k) = s
        .index
        .find_longest_match(&a.ids, alo, ahi, blo, bhi, &mut scratch);
    Ok(Some(new_match(py, i, j, k)?))
}

/// The lazy attributes (`b2j`, `bjunk`, `bpopular`, `fullbcount`) from the
/// instance's index (stored in the instance dict), or `None` if it has none.
#[pyfunction]
pub fn _sm_chain_attr<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    let Some(side) = d.get_item(pyo3::intern!(py, "_pyroid_b"))? else {
        return Ok(None);
    };
    let Ok(side) = side.cast_into::<BSide>() else {
        return Ok(None);
    };
    let s = side.get();
    let value = match name {
        "b2j" => s.b2j(py)?.into_any(),
        "bjunk" => s.bjunk(py)?.into_any(),
        "bpopular" => s.bpopular(py)?.into_any(),
        "fullbcount" => {
            if !d.contains(pyo3::intern!(py, "_pyroid_qr"))? {
                return Ok(None);
            }
            s.fullbcount(py)?.into_any()
        }
        _ => return Ok(None),
    };
    d.set_item(name, &value)?;
    Ok(Some(value))
}
