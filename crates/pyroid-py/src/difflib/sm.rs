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
use pyroid_core::difflib::with_work;

use super::interrupt;
use super::seq::{Key, Kinds, Seq, Types, bytes_warnings, item_key, kinds_of, push_wtf8, why_not};
use super::settle::{self, Gate};

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
    /// The patch generation it was made in, for a matcher of difflib's class (else 0).
    generation: u64,
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
    /// The `b` it was built from.
    pub(super) fn b(&self) -> &Py<PyAny> {
        &self.b
    }

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
    pub(super) fn b2j<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for id in self.ids_where(|k| k == Kind::Normal) {
            #[allow(clippy::cast_possible_truncation)]
            let pos = self.index.positions(id as u32);
            d.set_item(self.rep(py, id)?, PyList::new(py, pos)?)?;
        }
        Ok(d)
    }

    /// `bjunk`.
    pub(super) fn bjunk<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let s = PySet::empty(py)?;
        for id in self.ids_where(|k| k == Kind::Junk) {
            s.add(self.rep(py, id)?)?;
        }
        Ok(s)
    }

    /// `bpopular`.
    pub(super) fn bpopular<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        let s = PySet::empty(py)?;
        for id in self.ids_where(|k| k == Kind::Popular) {
            s.add(self.rep(py, id)?)?;
        }
        Ok(s)
    }

    /// `fullbcount`: element -> count in `b`.
    pub(super) fn fullbcount<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
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
    let mut types = Types::default();
    let Some(seq) = Seq::read(b, &mut types)? else {
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
                let Some(key) = item_key(&item, &mut types)? else {
                    return Ok(None);
                };
                kinds = kinds.union(Kinds::of(&key));
                let (id, new) = keys.intern(key);
                if new {
                    reps.push(item.unbind());
                }
                ids.push(id);
            }
            if kinds.mixed() && bytes_warnings(b.py())? {
                return Ok(None);
            }
            (Keys::Items { keys, reps, kinds }, ids)
        }
    };
    let side = BSide {
        b: b.clone().unbind(),
        keys,
        index: BIndex::new(Vec::new(), 0, &[], false),
        generation: 0,
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
fn junk_pass(
    py: Python<'_>,
    side: &BSide,
    isjunk: &Bound<'_, PyAny>,
) -> Result<Vec<bool>, (PyErr, Vec<bool>)> {
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
    let mut types = Types::default();
    let Some(seq) = Seq::read(a, &mut types)? else {
        return Ok(None);
    };
    // Under `python -b`, str and bytes elements must not meet (CPython compares them and warns).
    let no_mix = || bytes_warnings(a.py());
    let mut buf = Vec::new();
    Ok(Some(match (&side.keys, seq) {
        (Keys::Chars(chars), Seq::Chars(cps)) => chars.encode(cps),
        (Keys::Chars(_), Seq::Values(v)) => vec![NONE; v.len()],
        (Keys::Chars(chars), Seq::Items(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            for item in items.iter() {
                match item_key(&item, &mut types)? {
                    None => return Ok(None),
                    Some(Key::Bytes(_)) if no_mix()? => return Ok(None),
                    Some(Key::Text(t)) => {
                        let s = std::str::from_utf8(&t).ok();
                        let mut it = s.into_iter().flat_map(str::chars);
                        ids.push(match (it.next(), it.next()) {
                            (Some(c), None) => chars.get(u32::from(c)),
                            _ => single_surrogate(&t).map_or(NONE, |c| chars.get(c)),
                        });
                    }
                    Some(Key::Int(_) | Key::BigInt(_) | Key::Bytes(_)) => ids.push(NONE),
                }
            }
            ids
        }
        (Keys::Items { keys, kinds, .. }, Seq::Chars(cps)) => {
            if kinds.bytes && no_mix()? {
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
                let Some(key) = item_key(&item, &mut types)? else {
                    return Ok(None);
                };
                mine = mine.union(Kinds::of(&key));
                ids.push(keys.get(&key));
            }
            if mine.union(*kinds).mixed() && no_mix()? {
                return Ok(None);
            }
            ids
        }
        (Keys::Values { .. }, Seq::Chars(cps)) => vec![NONE; cps.len()],
        (Keys::Values { keys, .. }, Seq::Items(items)) => {
            let mut ids = Vec::with_capacity(items.len());
            for item in items.iter() {
                match item_key(&item, &mut types)? {
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

/// `d.pop(name, None)`: another thread may have removed it already (racing
/// `set_seq*` calls on one matcher), which is not an error here.
pub(super) fn discard(d: &Bound<'_, PyDict>, name: &Bound<'_, PyString>) -> PyResult<()> {
    if !d.contains(name)? {
        return Ok(()); // the common case, without making an exception
    }
    match d.del_item(name) {
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyKeyError>(d.py()) => Ok(()),
        r => r,
    }
}

pub(super) fn instance_dict<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    Ok(obj
        .getattr(pyo3::intern!(obj.py(), "__dict__"))?
        .cast_into::<PyDict>()?)
}

/// `a`'s ids: kept on the instance (long `a`), or made for this call.
enum AIds<'py> {
    Cached(Bound<'py, ASide>),
    Fresh(Vec<u32>),
}

impl AIds<'_> {
    fn ids(&self) -> &[u32] {
        match self {
            Self::Cached(c) => &c.get().ids,
            Self::Fresh(ids) => ids,
        }
    }
}

/// Encodings of an `a` longer than this are kept on the instance.
const KEEP_A: usize = 256;

/// The instance's [`BSide`], its current `a`, and `a`'s ids.
type State<'py> = (Bound<'py, BSide>, Bound<'py, PyAny>, AIds<'py>);

/// The instance's [`State`], or `None`.
fn state<'py>(obj: &Bound<'py, PyAny>, d: &Bound<'py, PyDict>) -> PyResult<Option<State<'py>>> {
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
        if let Ok(c) = cached.cast::<ASide>() {
            let cc = c.get();
            if cc.a.bind(py).is(&a) && cc.side.bind(py).is(&side) {
                return Ok(Some((side, a, AIds::Cached(c.clone()))));
            }
        } else if let Ok(t) = cached.cast::<PyTuple>() {
            // `(a, side)`: this `a` does not run natively against this `b`.
            if t.len() == 2 && t.get_item(0)?.is(&a) && t.get_item(1)?.is(&side) {
                return Ok(None);
            }
        }
    }
    let Some(ids) = encode_a(side.get(), &a)? else {
        // Remembered: CPython's get_matching_blocks asks once per find_longest_match.
        d.set_item(pyo3::intern!(py, "_pyroid_a"), (&a, &side))?;
        return Ok(None);
    };
    if ids.len() <= KEEP_A {
        return Ok(Some((side, a, AIds::Fresh(ids))));
    }
    let aside = Bound::new(
        py,
        ASide {
            a: a.clone().unbind(),
            side: side.clone().unbind(),
            ids,
        },
    )?;
    d.set_item(pyo3::intern!(py, "_pyroid_a"), &aside)?;
    Ok(Some((side, a, AIds::Cached(aside))))
}

/// What `_pyroid_mb` says about the current `a` and `b`.
enum Cached<'py> {
    /// The blocks.
    Blocks(Bound<'py, Blocks>),
    /// `ratio()` ran and gave this (CPython has `matching_blocks` now); the
    /// blocks were not kept.
    Counted(f64),
    Nothing,
}

/// `_pyroid_mb`: a [`Blocks`], or an `(a, b, ratio)` tuple after `ratio()`; valid
/// while `self.a` and `self.b` are those objects.
fn cached_blocks<'py>(obj: &Bound<'py, PyAny>, d: &Bound<'py, PyDict>) -> PyResult<Cached<'py>> {
    let py = obj.py();
    let Some(cached) = d.get_item(pyo3::intern!(py, "_pyroid_mb"))? else {
        return Ok(Cached::Nothing);
    };
    let a = obj.getattr(pyo3::intern!(py, "a"))?;
    let b = obj.getattr(pyo3::intern!(py, "b"))?;
    if let Ok(blocks) = cached.cast::<Blocks>() {
        let c = blocks.get();
        if c.a.bind(py).is(&a) && c.b.bind(py).is(&b) {
            return Ok(Cached::Blocks(blocks.clone()));
        }
    } else if let Ok(t) = cached.cast::<PyTuple>() {
        if t.len() == 3 && t.get_item(0)?.is(&a) && t.get_item(1)?.is(&b) {
            return Ok(Cached::Counted(t.get_item(2)?.extract()?));
        }
    }
    Ok(Cached::Nothing)
}

/// The matching blocks for the current `a` and `b`: cached, or computed and
/// cached; `None` without a native state.
fn blocks<'py>(
    obj: &Bound<'py, PyAny>,
    d: &Bound<'py, PyDict>,
) -> PyResult<Option<Bound<'py, Blocks>>> {
    let py = obj.py();
    if let Cached::Blocks(blocks) = cached_blocks(obj, d)? {
        return Ok(Some(blocks));
    }
    let Some((side, a, aside)) = state(obj, d)? else {
        return Ok(None);
    };
    let (s, ids) = (side.get(), aside.ids());
    let work = || with_work(|w| s.index.matching_blocks_with(ids, w));
    let found = interrupt::run(py, ids.len() + s.index.b().len() > 4096, work)?;
    let out = Bound::new(
        py,
        Blocks {
            a: a.unbind(),
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

/// `set_seq1(a)`. `None` when the patch is off and `obj` is of difflib's
/// class: the caller runs CPython's `set_seq1`.
#[pyfunction]
pub fn _sm_set_seq1(obj: &Bound<'_, PyAny>, a: &Bound<'_, PyAny>) -> PyResult<Option<()>> {
    let py = obj.py();
    let Some(_gate) = settle::enter(obj)? else {
        return Ok(None);
    };
    if a.is(&obj.getattr(pyo3::intern!(py, "a"))?) {
        return Ok(Some(()));
    }
    obj.setattr(pyo3::intern!(py, "a"), a)?;
    obj.setattr(pyo3::intern!(py, "opcodes"), py.None())?;
    let d = instance_dict(obj)?;
    discard(&d, pyo3::intern!(py, "matching_blocks"))?;
    discard(&d, pyo3::intern!(py, "_pyroid_mb"))?;
    Ok(Some(()))
}

/// `set_seq2(b)` after the identity check: reset the caches, build the
/// index. False if `b` does not run natively (the caller chains it as
/// CPython does); `None` when the patch is off and `obj` is of difflib's
/// class (the caller runs CPython's `set_seq2`). If `isjunk` (or
/// `autojunk`'s truth) raises, the instance is left as CPython's
/// `__chain_b` leaves it, without a native index.
#[pyfunction]
pub fn _sm_set_seq2(obj: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Option<bool>> {
    let Some(mut gate) = settle::enter(obj)? else {
        return Ok(None);
    };
    set_seq2(obj, b, &mut gate).map(Some)
}

fn set_seq2(obj: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, gate: &mut Gate<'_>) -> PyResult<bool> {
    let py = obj.py();
    obj.setattr(pyo3::intern!(py, "b"), b)?;
    obj.setattr(pyo3::intern!(py, "opcodes"), py.None())?;
    chain(obj, &instance_dict(obj)?, b, false, gate)
}

/// The attributes CPython's `__chain_b` leaves when it raises: in the junk
/// pass (`popular` false: `b2j` complete, `bjunk` so far, `bpopular` the old
/// one) or at `autojunk` (`b2j` without junk, `bpopular` empty).
fn failed_chain(
    d: &Bound<'_, PyDict>,
    side: &mut BSide,
    ids: &[u32],
    n: usize,
    junk: &[bool],
    popular: bool,
) -> PyResult<()> {
    let py = d.py();
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
}

/// `__chain_b` for `b` (already `self.b`). `fresh`: the instance dict holds
/// nothing of an earlier `b` (nothing to remove first).
fn chain(
    obj: &Bound<'_, PyAny>,
    d: &Bound<'_, PyDict>,
    b: &Bound<'_, PyAny>,
    fresh: bool,
    gate: &mut Gate<'_>,
) -> PyResult<bool> {
    let py = obj.py();
    let del = |names: &[&Bound<'_, PyString>]| -> PyResult<()> {
        if fresh {
            return Ok(());
        }
        for name in names {
            discard(d, name)?;
        }
        Ok(())
    };
    // The previous b's lazy attributes: CPython keeps bjunk and bpopular
    // until __chain_b replaces them, so a chain that fails half-way shows them.
    let old = if fresh {
        None
    } else {
        d.get_item(pyo3::intern!(py, "_pyroid_b"))?
            .and_then(|o| o.cast_into::<BSide>().ok())
    };
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
    let fail = |side: &mut BSide, ids: &[u32], junk: &[bool], popular: bool| {
        failed_chain(d, side, ids, n, junk, popular)
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
    side.generation = gate.generation();
    // Recorded once per patch generation (see settle.rs).
    gate.made(
        old.as_ref()
            .is_some_and(|o| o.get().generation != 0 && o.get().generation == side.generation),
    );
    // The new index replaces the old one in one step, then the old b's attributes go: a thread
    // reading `b2j` meanwhile (racing this one) sees the old or the new, never neither.
    d.set_item(pyo3::intern!(py, "_pyroid_b"), Bound::new(py, side)?)?;
    del(&[
        pyo3::intern!(py, "b2j"),
        pyo3::intern!(py, "bjunk"),
        pyo3::intern!(py, "bpopular"),
    ])?;
    Ok(true)
}

/// `__init__(isjunk, a, b, autojunk)` when `set_seqs`, `set_seq1` and
/// `set_seq2` are not overridden. False if `b` does not run natively (the
/// caller chains it); `None` when the patch is off and `obj` is of
/// difflib's class (the caller runs CPython's `__init__`). `plain`: the
/// instance is of `SequenceMatcher` itself (no `__setattr__` or descriptors
/// of a subclass to honour).
#[pyfunction]
pub fn _sm_init(
    obj: &Bound<'_, PyAny>,
    isjunk: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    autojunk: &Bound<'_, PyAny>,
    plain: bool,
) -> PyResult<Option<bool>> {
    let Some(mut gate) = settle::enter(obj)? else {
        return Ok(None);
    };
    init(obj, isjunk, a, b, autojunk, plain, &mut gate).map(Some)
}

fn init(
    obj: &Bound<'_, PyAny>,
    isjunk: &Bound<'_, PyAny>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
    autojunk: &Bound<'_, PyAny>,
    plain: bool,
    gate: &mut Gate<'_>,
) -> PyResult<bool> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    if plain && d.is_empty() {
        // A new instance: CPython's attributes, set in its order.
        d.set_item(pyo3::intern!(py, "isjunk"), isjunk)?;
        d.set_item(pyo3::intern!(py, "a"), py.None())?;
        d.set_item(pyo3::intern!(py, "b"), py.None())?;
        d.set_item(pyo3::intern!(py, "autojunk"), autojunk)?;
        if !a.is_none() {
            d.set_item(pyo3::intern!(py, "a"), a)?;
            d.set_item(pyo3::intern!(py, "opcodes"), py.None())?;
        }
        if b.is_none() {
            return Ok(true);
        }
        d.set_item(pyo3::intern!(py, "b"), b)?;
        d.set_item(pyo3::intern!(py, "opcodes"), py.None())?;
        return chain(obj, &d, b, true, gate);
    }
    obj.setattr(pyo3::intern!(py, "isjunk"), isjunk)?;
    obj.setattr(pyo3::intern!(py, "a"), py.None())?;
    obj.setattr(pyo3::intern!(py, "b"), py.None())?;
    obj.setattr(pyo3::intern!(py, "autojunk"), autojunk)?;
    // set_seq1(a): nothing for None (it is already self.a).
    if !a.is_none() {
        obj.setattr(pyo3::intern!(py, "a"), a)?;
        obj.setattr(pyo3::intern!(py, "opcodes"), py.None())?;
        for name in [
            pyo3::intern!(py, "matching_blocks"),
            pyo3::intern!(py, "_pyroid_mb"),
        ] {
            discard(&d, name)?;
        }
    }
    if b.is_none() {
        return Ok(true);
    }
    set_seq2(obj, b, gate)
}

/// `ratio()`, or `None`.
#[pyfunction]
pub fn _sm_ratio(obj: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let py = obj.py();
    let d = instance_dict(obj)?;
    if has_list(&d, pyo3::intern!(py, "matching_blocks"))? {
        return Ok(None);
    }
    match cached_blocks(obj, &d)? {
        Cached::Blocks(blocks) => {
            let c = blocks.get();
            let last = c.found.last().map_or(0, |m| m.a + m.b); // the (la, lb, 0) sentinel
            return Ok(Some(calculate_ratio(matches(&c.found), last)));
        }
        Cached::Counted(ratio) => return Ok(Some(ratio)),
        Cached::Nothing => {}
    }
    let Some((side, a, aside)) = state(obj, &d)? else {
        return Ok(None);
    };
    let (s, ids) = (side.get(), aside.ids());
    let work = || with_work(|w| s.index.matched(ids, w));
    let matched = interrupt::run(py, ids.len() + s.index.b().len() > 4096, work)?;
    // The lengths read when a and b were encoded (the same objects).
    let ratio = calculate_ratio(matched, ids.len() + s.index.b().len());
    let marker = (a, s.b.bind(py), ratio).into_pyobject(py)?;
    d.set_item(pyo3::intern!(py, "_pyroid_mb"), marker)?;
    Ok(Some(ratio))
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
/// blocks as a list (stored in the dict) if CPython would have them, or `None`.
#[pyfunction]
pub fn _sm_cached_blocks<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyList>>> {
    let d = instance_dict(obj)?;
    let Some(list) = cached_list(obj, &d)? else {
        return Ok(None);
    };
    d.set_item(pyo3::intern!(obj.py(), "matching_blocks"), &list)?;
    Ok(Some(list))
}

/// The blocks as a list if CPython would have them as `matching_blocks`
/// (after `get_matching_blocks()`, `get_opcodes()` or `ratio()`), or `None`.
pub(super) fn cached_list<'py>(
    obj: &Bound<'py, PyAny>,
    d: &Bound<'py, PyDict>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let found = match cached_blocks(obj, d)? {
        Cached::Nothing => return Ok(None),
        Cached::Blocks(blocks) => blocks,
        Cached::Counted(_) => match blocks(obj, d)? {
            Some(blocks) => blocks,
            None => return Ok(None),
        },
    };
    matches_list(obj.py(), &found.get().found).map(Some)
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
    let Some((side, _, aside)) = state(obj, &d)? else {
        return Ok(None);
    };
    let (s, ids) = (side.get(), aside.ids());
    // CPython builds fullbcount on the first call: the attribute shows it from now on.
    d.set_item(pyo3::intern!(py, "_pyroid_qr"), true)?;
    let length = ids.len() + s.index.b().len();
    Ok(Some(calculate_ratio(
        with_work(|w| s.index.quick_matches_with(ids, w)),
        length,
    )))
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
    let Some((side, _, aside)) = state(obj, &d)? else {
        return Ok(None);
    };
    let (s, ids) = (side.get(), aside.ids());
    let (la, lb) = (ids.len(), s.index.b().len());
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
    let work = || with_work(|w| s.index.find_longest_match_with(ids, alo, ahi, blo, bhi, w));
    let (i, j, k) = interrupt::run(py, la + lb > 4096, work)?;
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

/// Why the matcher's sequences run CPython's code, for the `FallbackWarning`.
#[pyfunction]
pub fn _sm_why(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = obj.py();
    let a = obj.getattr(pyo3::intern!(py, "a"))?;
    let b = obj.getattr(pyo3::intern!(py, "b"))?;
    for seq in [&b, &a] {
        if let Some(why) = why_not(seq)? {
            return Ok(why);
        }
    }
    if bytes_warnings(py)? && kinds_of(&a)?.union(kinds_of(&b)?).mixed() {
        return Ok("a mix of str and bytes elements under python -b".into());
    }
    Ok("these sequences".into())
}
