// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! Batch helpers: `parse_batch`, `sort_versions`, `contains_batch`, `filter_batch`.
//!
//! Strings are parsed on the shared pool (detached), a chunk at a time, with a signal check
//! between chunks (Ctrl-C stops a long batch); everything that needs Python (other objects,
//! digit runs that `int()` may reject, the result objects) happens on the calling thread.

use pyo3::exceptions::PyValueError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyList, PyString, PyTuple};
use pyroid_core::packaging::specifiers::{self as spec_core, Op};
use pyroid_core::packaging::version::{self as vcore, Version};
use pyroid_core::par;

use super::config;
use super::specifiers::{PySpecifier, PySpecifierSet, coerce};
use super::version::{PyVersion, parse_str};
use crate::arrow::{Chunk, Column, ColumnType};
use crate::errors::with_note;
use crate::input::{TextInput, Threads, ThreadsArg, text_arg, threads};

/// Items parsed per detach: between chunks the batch checks for Ctrl-C.
const CHUNK: usize = 1 << 16;

/// One input item, as the batch sees it.
enum Item<'py> {
    /// A `str` (`None`: it has a lone surrogate, so it is no version).
    Str(Option<PyBackedStr>, Bound<'py, PyAny>),
    /// Any other object (a Version, or something `Version()` decides about).
    Obj(Bound<'py, PyAny>),
    /// An Arrow null.
    Null,
}

/// The input as items: a list or tuple (read through one snapshot), or an Arrow string array.
fn items<'py>(py: Python<'py>, xs: &Bound<'py, PyAny>, t: Option<Threads>) -> PyResult<(Vec<Item<'py>>, Kind)> {
    let (seq, kind) = if let Ok(l) = xs.cast::<PyList>() {
        (l.to_tuple(), Kind::List)
    } else if let Ok(tu) = xs.cast::<PyTuple>() {
        (tu.clone(), Kind::Tuple)
    } else if xs.cast::<PyString>().is_ok() {
        return Err(crate::errors::input_error(py, "expected a list, tuple or Arrow array of versions, got str".to_owned()));
    } else {
        let input = text_arg(xs, t, "versions")?;
        let TextInput::Arrow(_) = &input else {
            return Err(crate::errors::input_error(py, "expected a list, tuple or Arrow array of versions".to_owned()));
        };
        let rows = crate::batch::rows(&input).map_err(|e| e.into_pyerr(py))?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(match r {
                None => Item::Null,
                Some(s) => {
                    let obj = PyString::new(py, s);
                    Item::Str(PyBackedStr::try_from(obj.clone()).ok(), obj.into_any())
                }
            });
        }
        return Ok((out, Kind::Arrow));
    };
    let mut out = Vec::with_capacity(seq.len());
    for x in seq.iter() {
        out.push(match x.cast::<PyString>() {
            Ok(s) => Item::Str(PyBackedStr::try_from(s.clone()).ok(), x.clone()),
            Err(_) => Item::Obj(x),
        });
    }
    Ok((out, kind))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    List,
    Tuple,
    Arrow,
}

/// What parsing one item gave: a version, "not a version", or a digit run Python must check.
enum Parsed {
    Version(Version),
    Invalid,
    LongRuns,
    Null,
}

/// Parse every `str` item on the pool, a chunk at a time; other items are coerced here.
fn parse_all(py: Python<'_>, items: &[Item<'_>], t: Option<Threads>) -> PyResult<Vec<Parsed>> {
    let threads = threads(t);
    let strs: Vec<Option<&str>> = items
        .iter()
        .map(|i| match i {
            Item::Str(Some(s), _) => Some(&**s),
            _ => None,
        })
        .collect();
    let mut out: Vec<Parsed> = Vec::with_capacity(items.len());
    let mut start = 0;
    while start < items.len() {
        let end = (start + CHUNK).min(items.len());
        let slice = &strs[start..end];
        let chunk = py
            .detach(|| {
                par::map(slice.len(), threads, |k| {
                    slice[k].map(|s| match vcore::parse(s) {
                        Some(p) if p.long_runs.is_empty() => Parsed::Version(p.version),
                        Some(_) => Parsed::LongRuns,
                        None => Parsed::Invalid,
                    })
                })
            })
            .map_err(|e| crate::batch::BatchError::from(e).into_pyerr(py))?;
        for (k, p) in chunk.into_iter().enumerate() {
            out.push(match (p, &items[start + k]) {
                (Some(p), _) => p,
                (None, Item::Null) => Parsed::Null,
                (None, Item::Str(None, _)) => Parsed::Invalid,
                (None, Item::Obj(o)) => match coerce(py, o)? {
                    Some(v) => Parsed::Version(v),
                    None => Parsed::Invalid,
                },
                (None, Item::Str(Some(_), _)) => Parsed::Invalid,
            });
        }
        py.check_signals()?;
        start = end;
    }
    Ok(out)
}

/// A long digit run: parse it as `Version()` would (CPython's `int()` may raise).
fn resolve_long(py: Python<'_>, item: &Item<'_>) -> PyResult<Version> {
    match item {
        Item::Str(_, s) => parse_str(py, s.cast::<PyString>()?),
        _ => Err(PyValueError::new_err("unexpected item")),
    }
}

fn index_note(py: Python<'_>, err: PyErr, i: usize) -> PyErr {
    with_note(py, err, format!("pyroid: the version at index {i}"))
}

fn invalid(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let msg = format!("Invalid version: {}", item.repr()?);
    Ok(PyErr::from_value(config(py)?.invalid_version.bind(py).call1((msg,))?))
}

fn item_obj<'py>(py: Python<'py>, item: &Item<'py>) -> Bound<'py, PyAny> {
    match item {
        Item::Str(_, o) | Item::Obj(o) => o.clone(),
        Item::Null => py.None().into_bound(py),
    }
}

/// Parse many versions at once.
///
/// Args:
///     versions: A list or tuple of version strings (or Version objects), or an Arrow string
///         array.
///     errors: ``"raise"`` (the first invalid version raises ``InvalidVersion``, with a note
///         naming its index) or ``"none"`` (``None`` in its place).
///     threads: At most this many tasks at once on the shared pool (``None``: the whole pool).
///
/// Returns:
///     A list of ``Version`` objects (``None`` for Arrow nulls).
#[pyfunction]
#[pyo3(signature = (versions, /, *, errors = "raise", threads = None))]
fn parse_batch<'py>(py: Python<'py>, versions: &Bound<'py, PyAny>, errors: &str, threads: Option<ThreadsArg>) -> PyResult<Bound<'py, PyList>> {
    let raise = match errors {
        "raise" => true,
        "none" => false,
        other => return Err(PyValueError::new_err(format!("errors must be 'raise' or 'none', not {other:?}"))),
    };
    let t = ThreadsArg::resolve(threads)?;
    let (items, _) = items(py, versions, t)?;
    let parsed = parse_all(py, &items, t)?;
    let out = PyList::empty(py);
    for (i, (p, item)) in parsed.into_iter().zip(&items).enumerate() {
        let v = match p {
            Parsed::Version(v) => Some(v),
            Parsed::Null => None,
            Parsed::LongRuns => Some(resolve_long(py, item).map_err(|e| index_note(py, e, i))?),
            Parsed::Invalid => {
                if raise {
                    return Err(index_note(py, invalid(py, &item_obj(py, item))?, i));
                }
                None
            }
        };
        match v {
            Some(v) => out.append(Bound::new(py, PyVersion::from_core(v))?)?,
            None => out.append(py.None())?,
        }
    }
    Ok(out)
}

/// Sort versions by PEP 440 order (``sorted(versions, key=Version)``, stable).
///
/// Args:
///     versions: A list or tuple of version strings and/or Version objects, or an Arrow string
///         array.
///     reverse: Newest first.
///     threads: At most this many tasks at once on the shared pool.
///
/// Returns:
///     A list of the given objects (``str`` for Arrow input), sorted.
///
/// Raises:
///     InvalidVersion: An item is not a valid version (the first one in input order).
#[pyfunction]
#[pyo3(signature = (versions, /, *, reverse = false, threads = None))]
fn sort_versions<'py>(py: Python<'py>, versions: &Bound<'py, PyAny>, reverse: bool, threads: Option<ThreadsArg>) -> PyResult<Bound<'py, PyList>> {
    let t = ThreadsArg::resolve(threads)?;
    let (items, _) = items(py, versions, t)?;
    let parsed = parse_all(py, &items, t)?;
    let mut keys: Vec<Version> = Vec::with_capacity(items.len());
    for (p, item) in parsed.into_iter().zip(&items) {
        match p {
            Parsed::Version(v) => keys.push(v),
            Parsed::LongRuns => keys.push(resolve_long(py, item)?),
            Parsed::Invalid | Parsed::Null => return Err(invalid(py, &item_obj(py, item))?),
        }
    }
    let threads = crate::input::threads(t);
    let order = py
        .detach(|| pyroid_core::packaging::sort_indices(&keys, reverse, threads))
        .map_err(|e| crate::batch::BatchError::from(e).into_pyerr(py))?;
    py.check_signals()?;
    PyList::new(py, order.iter().map(|&i| item_obj(py, &items[i])))
}

/// `[specifiers.contains(v, prereleases, installed) for v in versions]`.
///
/// Args:
///     specifiers: A ``SpecifierSet``, ``Specifier`` or specifier string.
///     versions: A list or tuple of version strings and/or Version objects, or an Arrow string
///         array.
///     prereleases: As for ``contains``.
///     installed: As for ``SpecifierSet.contains``.
///     threads: At most this many tasks at once on the shared pool.
///
/// Returns:
///     A list (tuple for a tuple) of bools, or a ``pyroid.Column`` of bools for Arrow input
///     (nulls stay null).
#[pyfunction]
#[pyo3(signature = (specifiers, versions, /, *, prereleases = None, installed = None, threads = None))]
fn contains_batch<'py>(
    py: Python<'py>,
    specifiers: &Bound<'py, PyAny>,
    versions: &Bound<'py, PyAny>,
    prereleases: Option<&Bound<'py, PyAny>>,
    installed: Option<&Bound<'py, PyAny>>,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyAny>> {
    let t = ThreadsArg::resolve(threads)?;
    let spec = as_spec(py, specifiers)?;
    let (items, kind) = items(py, versions, t)?;
    let parsed = parse_all(py, &items, t)?;
    let mut mask: Vec<Option<bool>> = Vec::with_capacity(items.len());
    let pre = prereleases.filter(|p| !p.is_none());
    for (i, (p, item)) in parsed.into_iter().zip(&items).enumerate() {
        let r = match p {
            Parsed::Null => None,
            Parsed::Version(v) if spec.native() => Some(spec.contains_parsed(py, &item_obj(py, item), &v, pre, installed)?),
            Parsed::Invalid if spec.native() => Some(spec.contains_invalid(py, pre)?),
            _ => {
                let kw = pyo3::types::PyDict::new(py);
                if let Some(p) = pre {
                    kw.set_item("prereleases", p)?;
                }
                if let (Some(inst), true) = (installed, spec.is_set()) {
                    kw.set_item("installed", inst)?;
                }
                let r = spec.obj.call_method(intern!(py, "contains"), (item_obj(py, item),), Some(&kw)).map_err(|e| index_note(py, e, i))?;
                Some(r.is_truthy()?)
            }
        };
        mask.push(r);
    }
    Ok(match kind {
        Kind::List => PyList::new(py, mask)?.into_any(),
        Kind::Tuple => PyTuple::new(py, mask)?.into_any(),
        Kind::Arrow => {
            let chunk = pyroid_core::columnar::BoolBuffers::build(&mask);
            Bound::new(py, Column::new(ColumnType::Bool, "", vec![Chunk::Bool(chunk)]))?.into_any()
        }
    })
}

/// `list(specifiers.filter(versions, prereleases))`.
///
/// Args:
///     specifiers: A ``SpecifierSet``, ``Specifier`` or specifier string.
///     versions: A list or tuple of version strings and/or Version objects, or an Arrow string
///         array.
///     prereleases: As for ``filter``.
///     threads: At most this many tasks at once on the shared pool.
///
/// Returns:
///     A list of the matching objects (``str`` for Arrow input), in input order.
#[pyfunction]
#[pyo3(signature = (specifiers, versions, /, *, prereleases = None, threads = None))]
fn filter_batch<'py>(
    py: Python<'py>,
    specifiers: &Bound<'py, PyAny>,
    versions: &Bound<'py, PyAny>,
    prereleases: Option<&Bound<'py, PyAny>>,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyList>> {
    let t = ThreadsArg::resolve(threads)?;
    let spec = as_spec(py, specifiers)?;
    let (items, _) = items(py, versions, t)?;
    let none = py.None().into_bound(py);
    let pre = prereleases.unwrap_or(&none);
    if !spec.ranged() {
        // `===`, an empty set or foreign items: the objects' own lazy filter.
        let it = PyList::new(py, items.iter().map(|i| item_obj(py, i)))?;
        let f = spec.obj.call_method1(intern!(py, "filter"), (it, pre))?;
        return PyList::new(py, f.try_iter()?.collect::<PyResult<Vec<_>>>()?);
    }
    let parsed = parse_all(py, &items, t)?;
    let (ranges, policy) = spec.filter_plan(py, pre)?;
    let out = PyList::empty(py);
    let mut held = Vec::new();
    let mut found_final = false;
    for (p, item) in parsed.into_iter().zip(&items) {
        let v = match p {
            Parsed::Version(v) => v,
            Parsed::LongRuns => resolve_long(py, item)?,
            Parsed::Invalid | Parsed::Null => continue,
        };
        match policy {
            Policy::Pep440 => {
                if !spec_core::in_ranges(&ranges, &v) {
                    continue;
                }
                if !v.is_prerelease() {
                    found_final = true;
                    out.append(item_obj(py, item))?;
                } else if !found_final {
                    held.push(item_obj(py, item));
                }
            }
            Policy::Exclude | Policy::Keep => {
                if matches!(policy, Policy::Exclude) && v.is_prerelease() {
                    continue;
                }
                if spec_core::in_ranges(&ranges, &v) {
                    out.append(item_obj(py, item))?;
                }
            }
        }
    }
    if !found_final {
        for h in held {
            out.append(h)?;
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
pub enum Policy {
    Pep440,
    Exclude,
    Keep,
}

/// The specifier argument of a batch call.
struct Spec<'py> {
    obj: Bound<'py, PyAny>,
}

impl<'py> Spec<'py> {
    fn set(&self) -> Option<&Bound<'py, PySpecifierSet>> {
        self.obj.cast::<PySpecifierSet>().ok()
    }

    fn single(&self) -> Option<&Bound<'py, PySpecifier>> {
        self.obj.cast::<PySpecifier>().ok()
    }

    fn is_set(&self) -> bool {
        self.set().is_some()
    }

    /// Can `contains` run natively for parsed versions?
    fn native(&self) -> bool {
        if let Some(s) = self.set() {
            return s.get().is_native();
        }
        self.single().is_some_and(|s| s.get().op().is_ok_and(|op| op != Op::Arbitrary))
    }

    /// Can `filter` run natively (ranges, no `===`, not empty)?
    fn ranged(&self) -> bool {
        if let Some(s) = self.set() {
            return s.get().is_native() && !s.get().is_empty(self.obj.py());
        }
        self.single().is_some_and(|s| s.get().op().is_ok_and(|op| op != Op::Arbitrary))
    }

    fn contains_parsed(&self, py: Python<'py>, item: &Bound<'py, PyAny>, v: &Version, pre: Option<&Bound<'py, PyAny>>, installed: Option<&Bound<'py, PyAny>>) -> PyResult<bool> {
        if let Some(s) = self.set() {
            return s.get().contains_known(py, item, v, pre, installed);
        }
        let s = self.single().ok_or_else(|| PyValueError::new_err("not a specifier"))?;
        s.get().contains_known(py, v, pre)
    }

    /// `contains(x)` for an `x` that does not parse: `False`, except for an empty set (its
    /// filter yields unparsable items).
    fn contains_invalid(&self, py: Python<'py>, pre: Option<&Bound<'py, PyAny>>) -> PyResult<bool> {
        if let Some(s) = self.set() {
            return s.get().contains_unparsable(py, pre);
        }
        Ok(false)
    }

    fn filter_plan(&self, py: Python<'py>, pre: &Bound<'py, PyAny>) -> PyResult<(Vec<spec_core::Interval>, Policy)> {
        if let Some(s) = self.set() {
            return s.get().filter_plan(py, pre);
        }
        let s = self.single().ok_or_else(|| PyValueError::new_err("not a specifier"))?;
        s.get().filter_plan(py, pre)
    }
}

fn as_spec<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Spec<'py>> {
    if x.cast::<PyString>().is_ok() {
        let set = Bound::new(py, PySpecifierSet::from_text(py, x)?)?;
        return Ok(Spec { obj: set.into_any() });
    }
    Ok(Spec { obj: x.clone() })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_batch, m)?)?;
    m.add_function(wrap_pyfunction!(sort_versions, m)?)?;
    m.add_function(wrap_pyfunction!(contains_batch, m)?)?;
    m.add_function(wrap_pyfunction!(filter_batch, m)?)?;
    Ok(())
}
