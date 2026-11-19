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

use super::cell::Passed;
use super::config;
use super::specifiers::{PySpecifier, PySpecifierSet, coerce};
use super::version::{PyVersion, parse_str, plain_str};
use crate::arrow::{Chunk, Column, ColumnType};
use crate::errors::with_note;
use crate::input::{TextInput, Threads, ThreadsArg, text_arg};

/// Items parsed per detach: between chunks the batch checks for Ctrl-C.
const CHUNK: usize = 1 << 16;

/// Below this many items a chunk is parsed on the calling thread: waking the pool and freeing
/// on this thread what its workers allocated cost more than they save (2,000 versions parsed
/// in 0.53 ms on the pool and 0.40 ms here, on a 64-core machine).
const INLINE: usize = 4096;

/// The pool for `n` items: the calling thread for a small batch.
fn pool_for(n: usize, threads: par::Threads) -> par::Threads {
    if n < INLINE {
        par::Threads::ONE
    } else {
        threads
    }
}

/// The input versions: a list (one snapshot) or tuple, or an Arrow string array's rows
/// (borrowed from its buffers). Read a chunk at a time, so a batch holds at most one chunk of
/// parsed versions.
enum Rows<'a, 'py> {
    Seq(&'a Bound<'py, PyTuple>),
    Arrow(&'a [Option<&'a str>]),
}

impl<'py> Rows<'_, 'py> {
    fn len(&self) -> usize {
        match self {
            Rows::Seq(t) => t.len(),
            Rows::Arrow(r) => r.len(),
        }
    }

    /// Item `i` as a Python object (`None`: an Arrow null).
    fn obj(&self, py: Python<'py>, i: usize) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(match self {
            Rows::Seq(t) => Some(t.get_item(i)?),
            Rows::Arrow(r) => r[i].map(|s| PyString::new(py, s).into_any()),
        })
    }

    /// `f(start, texts)` for each chunk: the text of each `str` item (`None` for any other
    /// object, a `str` with a lone surrogate, a subclass that `plain_str` rejects, or an Arrow
    /// null), with a Ctrl-C check after each.
    fn for_each_chunk(
        &self,
        py: Python<'py>,
        mut f: impl FnMut(usize, &[Option<&str>]) -> PyResult<()>,
    ) -> PyResult<()> {
        let n = self.len();
        let mut start = 0;
        while start < n {
            let end = (start + CHUNK).min(n);
            match self {
                Rows::Seq(t) => {
                    let backed: Vec<Option<PyBackedStr>> = (start..end)
                        .map(|i| {
                            t.get_item(i).ok().and_then(|x| {
                                x.cast_into::<PyString>()
                                    .ok()
                                    // Subclasses with their own __iter__ or split: resolve().
                                    .filter(|s| plain_str(py, s).unwrap_or(false))
                                    .and_then(|s| PyBackedStr::try_from(s).ok())
                            })
                        })
                        .collect();
                    let texts: Vec<Option<&str>> = backed.iter().map(|b| b.as_deref()).collect();
                    f(start, &texts)?;
                }
                Rows::Arrow(r) => f(start, &r[start..end])?,
            }
            py.check_signals()?;
            start = end;
        }
        Ok(())
    }
}

/// The input: a list or tuple (its items, through one snapshot of a list), or an Arrow array.
enum Input<'py> {
    Seq(Bound<'py, PyTuple>, Kind),
    Arrow(TextInput),
}

fn input<'py>(py: Python<'py>, xs: &Bound<'py, PyAny>, t: Option<Threads>) -> PyResult<Input<'py>> {
    if let Ok(l) = xs.cast::<PyList>() {
        return Ok(Input::Seq(l.to_tuple(), Kind::List));
    }
    if let Ok(tu) = xs.cast::<PyTuple>() {
        return Ok(Input::Seq(tu.clone(), Kind::Tuple));
    }
    if xs.cast::<PyString>().is_ok() {
        return Err(crate::errors::input_error(
            py,
            "expected a list, tuple or Arrow array of versions, got str".to_owned(),
        ));
    }
    let input = text_arg(xs, t, "versions")?;
    let TextInput::Arrow(_) = &input else {
        return Err(crate::errors::input_error(
            py,
            "expected a list, tuple or Arrow array of versions".to_owned(),
        ));
    };
    Ok(Input::Arrow(input))
}

/// `f(rows, kind)` over the input's rows.
fn with_rows<'py, R>(
    py: Python<'py>,
    input: &Input<'py>,
    f: impl FnOnce(&Rows<'_, 'py>, Kind) -> PyResult<R>,
) -> PyResult<R> {
    match input {
        Input::Seq(t, kind) => f(&Rows::Seq(t), *kind),
        Input::Arrow(a) => {
            let rows = crate::batch::rows(a).map_err(|e| e.into_pyerr(py))?;
            f(&Rows::Arrow(&rows), Kind::Arrow)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    List,
    Tuple,
    Arrow,
}

/// What parsing one `str` item gave: a version, "not a version", or a digit run Python must
/// check.
enum Parsed {
    Version(Version),
    Invalid,
    LongRuns,
}

fn parse_text(s: &str) -> Parsed {
    match vcore::parse(s) {
        Some(p) if p.long_runs.is_empty() => Parsed::Version(p.version),
        Some(_) => Parsed::LongRuns,
        None => Parsed::Invalid,
    }
}

/// The version of item `i` whose text did not parse on the pool (`texts[k]` is `None` or a long
/// digit run): `Ok(None)` for "not a version" and for an Arrow null (`null` is set).
fn resolve<'py>(
    py: Python<'py>,
    rows: &Rows<'_, 'py>,
    i: usize,
    long_runs: bool,
    null: &mut bool,
) -> PyResult<Option<Version>> {
    *null = false;
    let Some(obj) = rows.obj(py, i)? else {
        *null = true;
        return Ok(None);
    };
    if long_runs {
        // A long digit run: parse it as `Version()` would (CPython's `int()` may raise).
        return parse_str(py, obj.cast::<PyString>()?).map(Some);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        if plain_str(py, s)? {
            // A str with a lone surrogate: never a version.
            return Ok(None);
        }
    }
    coerce(py, &obj)
}

fn index_note(py: Python<'_>, err: PyErr, i: usize) -> PyErr {
    with_note(py, err, format!("pyroid: the version at index {i}"))
}

fn invalid(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let msg = format!("Invalid version: {}", item.repr()?);
    Ok(PyErr::from_value(
        config(py)?.invalid_version.bind(py).call1((msg,))?,
    ))
}

fn invalid_at(py: Python<'_>, rows: &Rows<'_, '_>, i: usize) -> PyResult<PyErr> {
    let item = rows.obj(py, i)?.unwrap_or_else(|| py.None().into_bound(py));
    invalid(py, &item)
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
#[pyo3(
    signature = (versions, /, *, errors = Passed(None), threads = None),
    text_signature = "(versions, /, *, errors='raise', threads=None)"
)]
fn parse_batch<'py>(
    py: Python<'py>,
    versions: &Bound<'py, PyAny>,
    errors: Passed,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyList>> {
    let raise = match errors.0 {
        Some(e) => errors_arg(py, e.bind(py))?,
        None => true,
    };
    let t = ThreadsArg::resolve(threads)?;
    let pool = crate::input::threads(t);
    let input = input(py, versions, t)?;
    let out = PyList::empty(py);
    with_rows(py, &input, |rows, _| {
        rows.for_each_chunk(py, |start, texts| {
            let parsed = parse_chunk(py, texts, pool)?;
            for (k, p) in parsed.into_iter().enumerate() {
                let i = start + k;
                let mut null = false;
                let v = match p {
                    Some(Parsed::Version(v)) => Some(v),
                    Some(Parsed::Invalid) => None,
                    Some(Parsed::LongRuns) => {
                        resolve(py, rows, i, true, &mut null).map_err(|e| index_note(py, e, i))?
                    }
                    None => resolve(py, rows, i, false, &mut null)?,
                };
                match v {
                    Some(v) => out.append(Bound::new(py, PyVersion::from_core(v))?)?,
                    None if raise && !null => {
                        return Err(index_note(py, invalid_at(py, rows, i)?, i));
                    }
                    None => out.append(py.None())?,
                }
            }
            Ok(())
        })
    })?;
    Ok(out)
}

/// `errors=`: `"raise"` (true) or `"none"`; another value is a `ValueError`, another type an
/// `InputError`, as for the other arguments of pyroid's batch functions.
fn errors_arg(py: Python<'_>, errors: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(s) = errors.cast::<PyString>() else {
        return Err(crate::errors::input_error(
            py,
            format!(
                "errors must be 'raise' or 'none', not {}",
                errors.get_type().name()?
            ),
        ));
    };
    match &*s.to_cow()? {
        "raise" => Ok(true),
        "none" => Ok(false),
        other => Err(PyValueError::new_err(format!(
            "errors must be 'raise' or 'none', not {other:?}"
        ))),
    }
}

/// Parse one chunk's texts on the pool (`None` where there is no text).
fn parse_chunk(
    py: Python<'_>,
    texts: &[Option<&str>],
    threads: par::Threads,
) -> PyResult<Vec<Option<Parsed>>> {
    let threads = pool_for(texts.len(), threads);
    py.detach(|| par::map(texts.len(), threads, |k| texts[k].map(parse_text)))
        .map_err(|e| crate::batch::BatchError::from(e).into_pyerr(py))
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
fn sort_versions<'py>(
    py: Python<'py>,
    versions: &Bound<'py, PyAny>,
    reverse: bool,
    threads: Option<ThreadsArg>,
) -> PyResult<Bound<'py, PyList>> {
    let t = ThreadsArg::resolve(threads)?;
    let pool = crate::input::threads(t);
    let input = input(py, versions, t)?;
    with_rows(py, &input, |rows, _| {
        let mut keys: Vec<Version> = Vec::with_capacity(rows.len());
        rows.for_each_chunk(py, |start, texts| {
            for (k, p) in parse_chunk(py, texts, pool)?.into_iter().enumerate() {
                let i = start + k;
                let mut null = false;
                let v = match p {
                    Some(Parsed::Version(v)) => Some(v),
                    Some(Parsed::Invalid) => None,
                    Some(Parsed::LongRuns) => resolve(py, rows, i, true, &mut null)?,
                    None => resolve(py, rows, i, false, &mut null)?,
                };
                match v {
                    Some(v) => keys.push(v),
                    None => return Err(invalid_at(py, rows, i)?),
                }
            }
            Ok(())
        })?;
        let order = sorted_order(py, &keys, reverse, pool)?;
        let items = order
            .iter()
            .map(|&i| -> PyResult<Bound<'py, PyAny>> {
                Ok(rows.obj(py, i)?.unwrap_or_else(|| py.None().into_bound(py)))
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    })
}

/// The stable PEP 440 order of `keys`: runs of [`CHUNK`] sorted on the pool, then merged pairwise,
/// with a Ctrl-C check between the steps (each a few tens of milliseconds).
fn sorted_order(
    py: Python<'_>,
    keys: &[Version],
    reverse: bool,
    threads: par::Threads,
) -> PyResult<Vec<usize>> {
    let pool_err = |e: par::PoolError| crate::batch::BatchError::from(e).into_pyerr(py);
    let at_once = py
        .detach(|| par::max_tasks(threads))
        .map_err(pool_err)?
        .max(1);
    let n = keys.len();
    let bounds: Vec<(usize, usize)> = (0..n)
        .step_by(CHUNK)
        .map(|a| (a, (a + CHUNK).min(n)))
        .collect();
    let mut runs: Vec<Vec<usize>> = Vec::with_capacity(bounds.len());
    for group in bounds.chunks(at_once) {
        let sorted = py
            .detach(|| {
                par::map_tasks(group, threads, |&r| {
                    pyroid_core::packaging::sort_run(keys, r, reverse)
                })
            })
            .map_err(pool_err)?;
        runs.extend(sorted);
        py.check_signals()?;
    }
    while runs.len() > 1 {
        let mut next: Vec<Vec<usize>> = Vec::with_capacity(runs.len().div_ceil(2));
        let pairs = runs.chunks(2).collect::<Vec<_>>();
        for group in pairs.chunks(at_once) {
            let merged = py
                .detach(|| {
                    par::map_tasks(group, threads, |pair| match pair {
                        [l, r] => pyroid_core::packaging::merge_runs(keys, l, r, reverse),
                        [l] => l.clone(),
                        _ => Vec::new(),
                    })
                })
                .map_err(pool_err)?;
            next.extend(merged);
            py.check_signals()?;
        }
        runs = next;
    }
    Ok(runs.pop().unwrap_or_default())
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
    let pool = crate::input::threads(t);
    let spec = as_spec(py, specifiers)?;
    let input = input(py, versions, t)?;
    let pre = prereleases.filter(|p| !p.is_none());
    let plan = if let Some(set) = spec.set() {
        set.get().contains_plan(py, pre, installed)?
    } else if let Some(single) = spec.single() {
        single.get().contains_plan(py, pre)?
    } else {
        None
    };
    // Natively, on the pool, for every str item; the rest (and everything without a plan)
    // through the object's own contains().
    let (mask, kind) = with_rows(py, &input, |rows, kind| {
        let mut mask: Vec<Option<bool>> = Vec::with_capacity(rows.len());
        rows.for_each_chunk(py, |start, texts| {
            let pool = pool_for(texts.len(), pool);
            let native: Vec<Option<bool>> = match &plan {
                Some(plan) => py
                    .detach(|| {
                        par::map(texts.len(), pool, |k| {
                            texts[k].and_then(|s| match vcore::parse(s) {
                                Some(p) if p.long_runs.is_empty() => {
                                    Some(plan.contains(Some(&p.version)))
                                }
                                Some(_) => None,
                                None => Some(plan.contains(None)),
                            })
                        })
                    })
                    .map_err(|e| crate::batch::BatchError::from(e).into_pyerr(py))?,
                None => vec![None; texts.len()],
            };
            for (k, r) in native.into_iter().enumerate() {
                if r.is_some() {
                    mask.push(r);
                    continue;
                }
                let i = start + k;
                let Some(item) = rows.obj(py, i)? else {
                    mask.push(None); // an Arrow null stays null
                    continue;
                };
                let kw = pyo3::types::PyDict::new(py);
                if let Some(p) = pre {
                    kw.set_item("prereleases", p)?;
                }
                if let (Some(inst), true) = (installed, spec.is_set()) {
                    kw.set_item("installed", inst)?;
                }
                let r = spec
                    .obj
                    .call_method(intern!(py, "contains"), (item,), Some(&kw))
                    .map_err(|e| index_note(py, e, i))?;
                mask.push(Some(r.is_truthy()?));
            }
            Ok(())
        })?;
        Ok((mask, kind))
    })?;
    Ok(match kind {
        Kind::List => PyList::new(py, mask)?.into_any(),
        Kind::Tuple => PyTuple::new(py, mask)?.into_any(),
        Kind::Arrow => {
            let chunk = pyroid_core::columnar::BoolBuffers::build(&mask);
            Bound::new(
                py,
                Column::new(ColumnType::Bool, "", vec![Chunk::Bool(chunk)]),
            )?
            .into_any()
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
    let pool = crate::input::threads(t);
    let spec = as_spec(py, specifiers)?;
    let input = input(py, versions, t)?;
    let none = py.None().into_bound(py);
    let pre = prereleases.unwrap_or(&none);
    if !spec.ranged() {
        // `===`, an empty set or foreign items: the objects' own lazy filter.
        let it: Bound<'py, PyAny> = match &input {
            Input::Seq(t, _) => t.clone().into_any(),
            Input::Arrow(_) => with_rows(py, &input, |rows, _| {
                let objs = (0..rows.len())
                    .map(|i| -> PyResult<Bound<'py, PyAny>> {
                        Ok(rows.obj(py, i)?.unwrap_or_else(|| py.None().into_bound(py)))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(PyList::new(py, objs)?.into_any())
            })?,
        };
        let f = spec.obj.call_method1(intern!(py, "filter"), (it, pre))?;
        return PyList::new(py, f.try_iter()?.collect::<PyResult<Vec<_>>>()?);
    }
    let (ranges, policy) = spec.filter_plan(py, pre)?;
    // Per str item, on the pool: (in range, pre-release), or `None` when Python must decide.
    let verdict = |v: &Version| (spec_core::in_ranges(&ranges, v), v.is_prerelease());
    let out = PyList::empty(py);
    let mut held = Vec::new();
    let mut found_final = false;
    with_rows(py, &input, |rows, _| {
        rows.for_each_chunk(py, |start, texts| {
            let pool = pool_for(texts.len(), pool);
            let verdicts: Vec<Option<Option<(bool, bool)>>> = py
                .detach(|| {
                    par::map(texts.len(), pool, |k| {
                        texts[k].and_then(|s| match parse_text(s) {
                            Parsed::Version(v) => Some(Some(verdict(&v))),
                            Parsed::Invalid => Some(None),
                            Parsed::LongRuns => None,
                        })
                    })
                })
                .map_err(|e| crate::batch::BatchError::from(e).into_pyerr(py))?;
            for (k, v) in verdicts.into_iter().enumerate() {
                let i = start + k;
                let found = if let Some(found) = v {
                    found
                } else {
                    let long = texts[k].is_some();
                    let mut null = false;
                    resolve(py, rows, i, long, &mut null)?.map(|v| verdict(&v))
                };
                let Some((in_range, is_pre)) = found else {
                    continue;
                };
                let keep = match policy {
                    Policy::Pep440 => {
                        if !in_range {
                            continue;
                        }
                        if is_pre {
                            if !found_final {
                                held.push(rows.obj(py, i)?);
                            }
                            false
                        } else {
                            found_final = true;
                            held.clear();
                            true
                        }
                    }
                    Policy::Exclude => in_range && !is_pre,
                    Policy::Keep => in_range,
                };
                if keep {
                    if let Some(obj) = rows.obj(py, i)? {
                        out.append(obj)?;
                    }
                }
            }
            Ok(())
        })
    })?;
    if !found_final {
        for h in held.into_iter().flatten() {
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

    /// Can `filter` run natively (ranges, no `===`, not empty)?
    fn ranged(&self) -> bool {
        if let Some(s) = self.set() {
            return s.get().is_native(self.obj.py()) && !s.get().is_empty(self.obj.py());
        }
        self.single()
            .is_some_and(|s| s.get().op().is_ok_and(|op| op != Op::Arbitrary))
    }

    fn filter_plan(
        &self,
        py: Python<'py>,
        pre: &Bound<'py, PyAny>,
    ) -> PyResult<(Vec<spec_core::Interval>, Policy)> {
        if let Some(s) = self.set() {
            return s.get().filter_plan(py, pre);
        }
        let s = self
            .single()
            .ok_or_else(|| PyValueError::new_err("not a specifier"))?;
        s.get().filter_plan(py, pre)
    }
}

fn as_spec<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Spec<'py>> {
    if x.cast::<PyString>().is_ok() {
        let set = Bound::new(py, PySpecifierSet::from_text(py, x)?)?;
        return Ok(Spec {
            obj: set.into_any(),
        });
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
