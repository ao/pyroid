// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! `pyroid.packaging.specifiers.Specifier` and `SpecifierSet`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAssertionError, PyAttributeError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyIterator, PyList, PyString, PyTuple, PyType};
use pyroid_core::packaging::specifiers::{self as core, Interval, Op};
use pyroid_core::packaging::version::Version;

use super::cell::{Prerel, PrerelKind, StateCell};
use super::version::{PyVersion, parse_any};
use super::{claims, config};

// ---------------------------------------------------------------------------
// Coercing items to versions
// ---------------------------------------------------------------------------

/// `coerce_version(item)`: the item's version, `None` when it does not parse (`InvalidVersion`);
/// other exceptions propagate.
pub fn coerce(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Option<Version>> {
    if let Ok(v) = item.cast::<PyVersion>() {
        return v.get().core().map(Some);
    }
    let c = config(py)?;
    if let Some(cls) = &c.packaging_version {
        if item.is_instance(cls.bind(py))? {
            return super::version::from_packaging(py, item);
        }
    }
    if let Ok(s) = item.cast::<PyString>() {
        // The str path without building an InvalidVersion for each unparsable string.
        let Ok(text) = s.to_cow() else {
            return Ok(None);
        };
        let Some(parsed) = pyroid_core::packaging::version::parse(&text) else {
            return Ok(None);
        };
        if !parsed.long_runs.is_empty() {
            return super::version::parse_str(py, s).map(Some);
        }
        return Ok(Some(parsed.version));
    }
    match parse_any(py, item) {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.is_instance(py, c.invalid_version.bind(py)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Is `item` a Version object (`isinstance(item, Version)` in packaging)?
fn is_version_object(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
    if item.cast::<PyVersion>().is_ok() {
        return Ok(true);
    }
    match &config(py)?.packaging_version {
        Some(cls) => item.is_instance(cls.bind(py)),
        None => Ok(false),
    }
}

fn bool_obj(py: Python<'_>, b: bool) -> Bound<'_, PyAny> {
    PyBool::new(py, b).to_owned().into_any()
}

fn assertion() -> PyErr {
    PyAssertionError::new_err("")
}

// ---------------------------------------------------------------------------
// Specifier
// ---------------------------------------------------------------------------

/// A parsed specifier: `_spec` and what is derived from it.
#[derive(Debug)]
struct SpecData {
    op: Op,
    op_str: Py<PyString>,
    /// `_spec[1]`, as Python holds it.
    version: Py<PyString>,
    /// The same as Rust text (`None` if it has a lone surrogate, which only `===` allows).
    vtext: Option<Box<str>>,
    /// `_get_spec_version` of the version (of its base for a wildcard).
    parsed: Option<Version>,
    /// Digit runs over 19 characters in the version: packaging's lazy parse converts them with
    /// `int()`, which may raise `ValueError` (`sys.get_int_max_str_digits()`), on each use.
    long_runs: Vec<(usize, usize)>,
    ranges: OnceLock<Option<Vec<Interval>>>,
    hash: OnceLock<isize>,
    /// Whether packaging would have filled its `_spec_version` cache by now (it is lazy, and
    /// its tests look at the cache).
    used: std::sync::atomic::AtomicBool,
    spec_version: OnceLock<Py<PyTuple>>,
    /// `_ranges` / `_to_ranges()` as packaging's own objects, once asked for.
    py_ranges: OnceLock<Py<PyAny>>,
}

impl SpecData {
    fn new(py: Python<'_>, op: Op, version: Bound<'_, PyString>) -> Self {
        let vtext: Option<Box<str>> = version.to_cow().ok().map(|c| c.into());
        let found = vtext.as_deref().and_then(|t| {
            let base = if op == Op::Arbitrary { t } else { t.strip_suffix(".*").unwrap_or(t) };
            pyroid_core::packaging::version::parse(base)
        });
        let (parsed, long_runs) = match found {
            Some(p) => (Some(p.version), p.long_runs.iter().map(|r| (r.start, r.end)).collect()),
            None => (None, Vec::new()),
        };
        Self {
            op,
            op_str: PyString::intern(py, op.as_str()).unbind(),
            version: version.unbind(),
            vtext,
            parsed,
            long_runs,
            ranges: OnceLock::new(),
            hash: OnceLock::new(),
            used: std::sync::atomic::AtomicBool::new(false),
            spec_version: OnceLock::new(),
            py_ranges: OnceLock::new(),
        }
    }

    fn mark_used(&self) {
        if self.parsed.is_some() {
            self.used.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    fn is_wildcard(&self) -> bool {
        self.vtext.as_deref().is_some_and(|t| t.ends_with(".*"))
    }

    fn text(&self) -> &str {
        self.vtext.as_deref().unwrap_or("")
    }

    /// packaging's `int()` of each long digit run (each time it parses the version).
    fn check(&self, py: Python<'_>) -> PyResult<()> {
        if !self.long_runs.is_empty() {
            let int = py.get_type::<pyo3::types::PyInt>();
            for &(a, b) in &self.long_runs {
                int.call1((&self.text()[a..b],))?;
            }
        }
        Ok(())
    }

    /// `_require_spec_version`: packaging asserts the version parses (not for `===`).
    fn require(&self, py: Python<'_>) -> PyResult<&Version> {
        self.check(py)?;
        self.mark_used();
        self.parsed.as_ref().ok_or_else(assertion)
    }

    /// `_to_ranges()`.
    fn ranges(&self, py: Python<'_>) -> PyResult<&[Interval]> {
        if self.ranges.get().is_none() && self.op != Op::Arbitrary {
            self.check(py)?;
        }
        let r = self.ranges.get_or_init(|| {
            if self.op == Op::Arbitrary {
                return core::bounds_for_spec(Op::Arbitrary, "", &Version::release_only(vec![]));
            }
            let v = self.parsed.as_ref()?;
            self.mark_used();
            core::bounds_for_spec(self.op, self.text(), v)
        });
        r.as_deref().ok_or_else(assertion)
    }

    /// The `prereleases` property without the explicit setting.
    fn implied_prereleases(&self, py: Python<'_>) -> PyResult<Option<bool>> {
        if self.op == Op::NotEqual {
            return Ok(Some(false));
        }
        if self.op == Op::Equal && self.is_wildcard() {
            return Ok(Some(false));
        }
        self.check(py)?;
        self.mark_used();
        Ok(self.parsed.as_ref().map(Version::is_prerelease))
    }

    /// `_canonical_spec[1]`.
    fn canonical_version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        if self.op == Op::Arbitrary || self.is_wildcard() {
            return Ok(self.version.bind(py).clone());
        }
        let v = self.require(py)?;
        Ok(PyString::new(py, &core::canonical_version(v, self.op != Op::Compatible)))
    }
}

/// A version specifier, as `packaging.specifiers.Specifier` 26.3.
#[pyclass(frozen, subclass, module = "pyroid.packaging.specifiers", name = "Specifier")]
#[derive(Debug)]
pub struct PySpecifier {
    data: StateCell<SpecData>,
    prereleases: Prerel,
}

fn invalid_specifier(py: Python<'_>, spec: &Bound<'_, PyAny>) -> PyErr {
    let r = (|| -> PyResult<PyErr> {
        let msg = format!("Invalid specifier: {}", spec.repr()?);
        Ok(PyErr::from_value(config(py)?.invalid_specifier.bind(py).call1((msg,))?))
    })();
    r.unwrap_or_else(|e| e)
}

/// Parse a specifier string (`Specifier.__init__`'s first half).
fn parse_spec(py: Python<'_>, spec: &Bound<'_, PyAny>) -> PyResult<SpecData> {
    let Ok(s) = spec.cast::<PyString>() else {
        // re's fullmatch raises TypeError for a non-str (pyroid's copy of the pattern).
        config(py)?.spec_regex.bind(py).call_method1(intern!(py, "fullmatch"), (spec,))?;
        return Err(invalid_specifier(py, spec));
    };
    let valid = match s.to_cow() {
        Ok(text) => {
            if core::is_valid(&text) {
                let (op, version) = core::split(&text);
                return Ok(SpecData::new(py, op, PyString::new(py, version)));
            }
            false
        }
        Err(_) => {
            // A lone surrogate: only `===` takes one; ask the pattern itself.
            let m = config(py)?.spec_regex.bind(py).call_method1(intern!(py, "fullmatch"), (s,))?;
            !m.is_none()
        }
    };
    if !valid {
        return Err(invalid_specifier(py, spec));
    }
    let stripped = s.call_method0(intern!(py, "strip"))?.cast_into::<PyString>()?;
    let version = stripped
        .get_item(pyo3::types::PySlice::new(py, 3, isize::MAX, 1))?
        .call_method0(intern!(py, "strip"))?
        .cast_into::<PyString>()?;
    Ok(SpecData::new(py, Op::Arbitrary, version))
}

impl PySpecifier {
    fn data(&self) -> PyResult<super::cell::StateRef<'_, SpecData>> {
        self.data
            .get()
            .ok_or_else(|| PyAttributeError::new_err("'Specifier' object has no attribute '_spec'"))
    }

    fn new_parsed(py: Python<'_>, spec: &Bound<'_, PyAny>, prereleases: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let data = parse_spec(py, spec)?;
        Ok(Self { data: StateCell::with(data), prereleases: Prerel::new(prereleases)? })
    }

    /// The `prereleases` property.
    fn prereleases_value<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.prereleases.kind() != PrerelKind::None {
            return Ok(self.prereleases.get(py));
        }
        Ok(match self.data()?.implied_prereleases(py)? {
            Some(b) => bool_obj(py, b),
            None => py.None().into_bound(py),
        })
    }

    /// `resolve_prereleases(self._prereleases, self.prereleases)`.
    fn resolved<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.prereleases.kind() != PrerelKind::None {
            return Ok(self.prereleases.get(py));
        }
        Ok(if self.data()?.implied_prereleases(py)? == Some(true) {
            bool_obj(py, true)
        } else {
            py.None().into_bound(py)
        })
    }

    fn canonical<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let d = self.data()?;
        PyTuple::new(py, [d.op_str.bind(py).clone().into_any(), d.canonical_version(py)?.into_any()])
    }

    fn hash_value(&self, py: Python<'_>) -> PyResult<isize> {
        let d = self.data()?;
        if let Some(h) = d.hash.get() {
            return Ok(*h);
        }
        let h = self.canonical(py)?.hash()?;
        let _ = d.hash.set(h);
        Ok(h)
    }

    /// `contains(item, prereleases)` for a parsed item (not `===`).
    fn contains_parsed(&self, py: Python<'_>, d: &SpecData, parsed: &Version, prereleases: PrerelKind) -> PyResult<bool> {
        if prereleases == PrerelKind::False && parsed.is_prerelease() {
            return Ok(false);
        }
        if !d.is_wildcard() && parsed.local.is_none() {
            // _fast_match reads the specifier's version.
            d.require(py)?;
        }
        if let Some(m) = core::fast_match(d.op, d.text(), d.parsed.as_ref(), parsed) {
            return Ok(m);
        }
        Ok(core::matches_bounds_only(d.ranges(py)?, parsed))
    }

    fn contains_impl(&self, py: Python<'_>, item: &Bound<'_, PyAny>, prereleases: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        let d = self.data()?;
        if d.op == Op::Arbitrary {
            let it = PyList::new(py, [item])?;
            let pre = match prereleases {
                Some(p) => p.clone(),
                None => py.None().into_bound(py),
            };
            let mut f = self.filter_iter(py, it.into_any().try_iter()?, &pre, None)?;
            return Ok(f.next_item(py)?.is_some());
        }
        let Some(parsed) = coerce(py, item)? else {
            return Ok(false);
        };
        let kind = match prereleases {
            Some(p) if !p.is_none() => kind_of(p)?,
            _ => kind_of(&self.resolved(py)?)?,
        };
        self.contains_parsed(py, &d, &parsed, kind)
    }

    fn filter_iter(
        &self,
        py: Python<'_>,
        iterable: Bound<'_, PyIterator>,
        prereleases: &Bound<'_, PyAny>,
        key: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<FilterIter> {
        let pre = if prereleases.is_none() { self.resolved(py)? } else { prereleases.clone() };
        let d = self.data()?;
        let mode = if d.op == Op::Arbitrary {
            let lower = d.version.bind(py).call_method0(intern!(py, "lower"))?.unbind();
            Mode::Matches(Match::Arbitrary(lower), prerelease_policy(&pre)?)
        } else {
            Mode::Ranges(Arc::new(d.ranges(py)?.to_vec()), ranges_policy(&pre)?)
        };
        Ok(FilterIter::new(iterable, key, mode))
    }
}

/// The kind of a `prereleases` argument.
fn kind_of(p: &Bound<'_, PyAny>) -> PyResult<PrerelKind> {
    let py = p.py();
    Ok(if p.is_none() {
        PrerelKind::None
    } else if p.is(PyBool::new(py, false)) {
        PrerelKind::False
    } else if p.is(PyBool::new(py, true)) {
        PrerelKind::True
    } else {
        PrerelKind::Other(p.is_truthy()?)
    })
}

#[pymethods]
impl PySpecifier {
    #[new]
    #[classmethod]
    #[pyo3(signature = (spec = None, prereleases = None, *_args, **_kwargs))]
    fn py_new(
        cls: &Bound<'_, PyType>,
        spec: Option<&Bound<'_, PyAny>>,
        prereleases: Option<&Bound<'_, PyAny>>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = cls.py();
        // Like packaging's object.__new__: a subclass (or `Specifier.__new__(Specifier)`, for
        // unpickling) starts empty and __init__ fills it in.
        if let Some(s) = spec {
            if cls.is(py.get_type::<PySpecifier>()) {
                return Self::new_parsed(py, s, prereleases);
            }
        }
        Ok(Self { data: StateCell::new(), prereleases: Prerel::new(None)? })
    }

    #[pyo3(name = "__init__", signature = (spec = None, prereleases = None))]
    fn py_init(&self, py: Python<'_>, spec: Option<&Bound<'_, PyAny>>, prereleases: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        if self.data.is_set() {
            return Ok(());
        }
        let empty = PyString::new(py, "");
        let spec = spec.unwrap_or(empty.as_any());
        self.data.set(parse_spec(py, spec)?);
        if let Some(p) = prereleases {
            self.prereleases.set(p)?;
        }
        Ok(())
    }

    #[getter]
    fn prereleases<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.prereleases_value(py)
    }

    #[setter]
    fn set_prereleases(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.prereleases.set(value)
    }

    #[getter]
    fn _prereleases<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.prereleases.get(py)
    }

    #[setter(_prereleases)]
    fn set_private_prereleases(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.prereleases.set(value)
    }

    #[getter]
    fn operator<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        Ok(self.data()?.op_str.bind(py).clone())
    }

    #[getter]
    fn version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        Ok(self.data()?.version.bind(py).clone())
    }

    #[getter]
    fn _spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let d = self.data()?;
        PyTuple::new(py, [d.op_str.bind(py).clone(), d.version.bind(py).clone()])
    }

    #[getter]
    fn _spec_version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = self.data()?;
        let Some(v) = &d.parsed else {
            return Ok(py.None().into_bound(py));
        };
        if !d.used.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(py.None().into_bound(py));
        }
        if let Some(t) = d.spec_version.get() {
            return Ok(t.bind(py).clone().into_any());
        }
        let base = d.vtext.as_deref().unwrap_or("");
        let base = if d.op == Op::Arbitrary { base } else { base.strip_suffix(".*").unwrap_or(base) };
        let t = PyTuple::new(py, [PyString::new(py, base).into_any(), Bound::new(py, PyVersion::from_core(v.clone()))?.into_any()])?;
        Ok(d.spec_version.get_or_init(|| t.unbind()).bind(py).clone().into_any())
    }

    #[getter]
    fn _ranges<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let d = slf.get().data()?;
        if d.ranges.get().is_none() {
            return Ok(py.None().into_bound(py));
        }
        Self::_to_ranges(slf)
    }

    fn _get_spec_version(&self, py: Python<'_>, version: &Bound<'_, PyAny>) -> PyResult<Option<PyVersion>> {
        Ok(coerce(py, version)?.map(PyVersion::from_core))
    }

    fn _require_spec_version(&self, py: Python<'_>, version: &Bound<'_, PyAny>) -> PyResult<PyVersion> {
        coerce(py, version)?.map(PyVersion::from_core).ok_or_else(assertion)
    }

    #[getter]
    fn _canonical_spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.canonical(py)
    }

    /// packaging's own range objects for this specifier (for packaging's `ranges` module).
    fn _to_ranges<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let d = slf.get().data()?;
        if let Some(r) = d.py_ranges.get() {
            return Ok(r.bind(py).clone());
        }
        d.ranges(py)?;
        let r = config(py)?.delegate.bind(py).call1(("_to_ranges", slf))?;
        Ok(d.py_ranges.get_or_init(|| r.unbind()).bind(py).clone())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let me = slf.get();
        let pre = if me.prereleases.kind() == PrerelKind::None {
            String::new()
        } else {
            format!(", prereleases={}", me.prereleases_value(py)?.repr()?)
        };
        let name = slf.get_type().name()?;
        Ok(format!("<{name}({}{pre})>", slf.str()?.repr()?))
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let d = self.data()?;
        Ok(format!("{}{}", d.op.as_str(), d.version.bind(py).to_cow()?))
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.hash_value(py)
    }

    fn __richcmp__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>, op: CompareOp) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented().into_bound(py));
        }
        let theirs = if let Ok(s) = other.cast::<PyString>() {
            match PySpecifier::new_parsed(py, &s.str()?.into_any(), None) {
                Ok(spec) => spec.canonical(py)?.into_any(),
                Err(e) if e.is_instance(py, config(py)?.invalid_specifier.bind(py)) => {
                    return Ok(py.NotImplemented().into_bound(py));
                }
                Err(e) => return Err(e),
            }
        } else if let Ok(o) = other.cast::<PySpecifier>() {
            o.get().canonical(py)?.into_any()
        } else if is_packaging_specifier(py, other)? {
            other.getattr(intern!(py, "_canonical_spec"))?
        } else {
            return Ok(py.NotImplemented().into_bound(py));
        };
        let eq = slf.get().canonical(py)?.eq(theirs)?;
        Ok(bool_obj(py, if matches!(op, CompareOp::Eq) { eq } else { !eq }))
    }

    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains_impl(py, item, None)
    }

    /// Return whether or not the item is contained in this specifier.
    #[pyo3(signature = (item, prereleases = None))]
    fn contains(&self, py: Python<'_>, item: &Bound<'_, PyAny>, prereleases: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        self.contains_impl(py, item, prereleases)
    }

    /// Filter items in the given iterable, that match the specifier.
    #[pyo3(signature = (iterable, prereleases = None, key = None))]
    fn filter(&self, py: Python<'_>, iterable: &Bound<'_, PyAny>, prereleases: Option<&Bound<'_, PyAny>>, key: Option<&Bound<'_, PyAny>>) -> PyResult<FilterIter> {
        let none = py.None().into_bound(py);
        let key = key.filter(|k| !k.is_none());
        self.filter_iter(py, iterable.try_iter()?, prereleases.unwrap_or(&none), key)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [self._spec(py)?.into_any(), self.prereleases.get(py)])
    }

    fn __setstate__(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let fail = || -> PyResult<PyErr> { Ok(PyTypeError::new_err(format!("Cannot restore Specifier from {}", state.repr()?))) };
        let valid_pre = |p: &Bound<'_, PyAny>| p.is_none() || p.cast::<PyBool>().is_ok();
        let mut found: Option<(Bound<'_, PyAny>, Bound<'_, PyAny>)> = None;
        if let Ok(t) = state.cast::<PyTuple>() {
            if t.len() == 2 {
                let (a, b) = (t.get_item(0)?, t.get_item(1)?);
                if valid_spec(&a)?.is_some() && valid_pre(&b) {
                    found = Some((a, b));
                } else if let Ok(d) = b.cast::<PyDict>() {
                    let spec = d.get_item("_spec")?;
                    let pre = d.get_item("_prereleases")?;
                    if let (Some(s), Some(p)) = (spec, pre) {
                        if valid_spec(&s)?.is_some() && valid_pre(&p) {
                            found = Some((s, p));
                        }
                    }
                }
            }
        } else if let Ok(d) = state.cast::<PyDict>() {
            if let (Some(s), Some(p)) = (d.get_item("_spec")?, d.get_item("_prereleases")?) {
                if valid_spec(&s)?.is_some() && valid_pre(&p) {
                    found = Some((s, p));
                }
            }
        }
        let Some((spec, pre)) = found else { return Err(fail()?) };
        let Some((op, version)) = valid_spec(&spec)? else { return Err(fail()?) };
        let op = Op::from_str(&op).ok_or_else(|| PyValueError::new_err(format!("unknown operator {op:?}")))?;
        self.data.set(SpecData::new(py, op, version));
        self.prereleases.set(&pre)
    }

    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, _protocol: i64) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let state = slf.get().__getstate__(py)?;
        PyTuple::new(py, [config(py)?.newobj.bind(py).clone(), PyTuple::new(py, [slf.get_type()])?.into_any(), state.into_any()])
    }

    #[getter]
    fn __class__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if claims() && slf.get_type().is(py.get_type::<PySpecifier>()) {
            if let Some(cls) = &config(py)?.packaging_specifier {
                return Ok(cls.clone_ref(py).into_any());
            }
        }
        Ok(slf.get_type().into_any().unbind())
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, ["_str"])?.unbind())
    }

    #[getter]
    fn _str<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyString>> {
        slf.str()
    }
}

/// A pickled `_spec`: `(operator, version)`, both `str`.
fn valid_spec<'py>(s: &Bound<'py, PyAny>) -> PyResult<Option<(String, Bound<'py, PyString>)>> {
    let Ok(t) = s.cast::<PyTuple>() else { return Ok(None) };
    if t.len() != 2 {
        return Ok(None);
    }
    let (Ok(op), Ok(v)) = (t.get_item(0)?.cast_into::<PyString>(), t.get_item(1)?.cast_into::<PyString>()) else {
        return Ok(None);
    };
    Ok(Some((op.to_cow()?.into_owned(), v)))
}

fn is_packaging_specifier(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<bool> {
    match &config(py)?.packaging_specifier {
        Some(cls) => x.is_instance(cls.bind(py)),
        None => Ok(false),
    }
}

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

/// What `filter_by_ranges` does with pre-releases.
#[derive(Clone, Copy, Debug)]
enum RangesPolicy {
    /// `prereleases=None`: the PEP 440 buffer.
    Pep440,
    /// `prereleases is False`: drop them.
    Exclude,
    /// Anything else: keep them.
    Keep,
}

fn ranges_policy(p: &Bound<'_, PyAny>) -> PyResult<RangesPolicy> {
    Ok(match kind_of(p)? {
        PrerelKind::None => RangesPolicy::Pep440,
        PrerelKind::False => RangesPolicy::Exclude,
        _ => RangesPolicy::Keep,
    })
}

/// What `_apply_prereleases_filter` does: `None` (the PEP 440 filter), truthy (all), falsy.
#[derive(Clone, Copy, Debug)]
enum PrereleasePolicy {
    Pep440,
    All,
    NoPre,
}

fn prerelease_policy(p: &Bound<'_, PyAny>) -> PyResult<PrereleasePolicy> {
    Ok(if p.is_none() {
        PrereleasePolicy::Pep440
    } else if p.is_truthy()? {
        PrereleasePolicy::All
    } else {
        PrereleasePolicy::NoPre
    })
}

/// The first-stage match of `_apply_prereleases_filter`'s input.
#[derive(Debug)]
enum Match {
    /// `str(x).lower() == spec_lower` (a `===` specifier).
    Arbitrary(Py<PyAny>),
    /// Every specifier's `contains(x, prereleases=True)` (a set with `===`).
    All(Vec<Py<PyAny>>),
    /// Everything (an empty set).
    Everything,
}

#[derive(Debug)]
enum Mode {
    Ranges(Arc<Vec<Interval>>, RangesPolicy),
    Matches(Match, PrereleasePolicy),
}

/// The iterator `filter()` returns: it reads the input lazily, one item at a time, as
/// packaging's generators do.
#[pyclass(module = "pyroid.packaging.specifiers", name = "_FilterIterator")]
#[derive(Debug)]
pub struct FilterIter {
    source: Option<Py<PyIterator>>,
    key: Option<Py<PyAny>>,
    mode: Mode,
    /// Items waiting to be yielded (flushed buffers).
    pending: VecDeque<Py<PyAny>>,
    /// Pre-releases held until the input ends (PEP 440: only if no final release matched).
    held: Vec<Py<PyAny>>,
    /// `_pep440_filter_prereleases`'s unparsable strings, flushed at the first final.
    arbitrary: Vec<Py<PyAny>>,
    found_final: bool,
}

impl FilterIter {
    fn new(source: Bound<'_, PyIterator>, key: Option<&Bound<'_, PyAny>>, mode: Mode) -> Self {
        Self {
            source: Some(source.unbind()),
            key: key.map(|k| k.clone().unbind()),
            mode,
            pending: VecDeque::new(),
            held: Vec::new(),
            arbitrary: Vec::new(),
            found_final: false,
        }
    }

    fn keyed<'py>(&self, py: Python<'py>, item: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        match &self.key {
            Some(k) => k.bind(py).call1((item,)),
            None => Ok(item.clone()),
        }
    }

    fn first_stage(&self, py: Python<'_>, item: &Bound<'_, PyAny>, m: &Match) -> PyResult<bool> {
        match m {
            Match::Everything => Ok(true),
            Match::Arbitrary(lower) => {
                let x = self.keyed(py, item)?;
                let s = x.str()?.call_method0(intern!(py, "lower"))?;
                s.eq(lower.bind(py))
            }
            Match::All(specs) => {
                let t = bool_obj(py, true);
                for s in specs {
                    let x = self.keyed(py, item)?;
                    let s = s.bind(py);
                    let ok = if let Ok(spec) = s.cast::<PySpecifier>() {
                        spec.get().contains_impl(py, &x, Some(&t))?
                    } else {
                        let kw = PyDict::new(py);
                        kw.set_item("prereleases", true)?;
                        s.call_method(intern!(py, "contains"), (x,), Some(&kw))?.is_truthy()?
                    };
                    if !ok {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    /// The next item to yield, `None` when done.
    fn next_item(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        loop {
            if let Some(x) = self.pending.pop_front() {
                return Ok(Some(x));
            }
            let Some(src) = &self.source else {
                return Ok(None);
            };
            let mut src = src.bind(py).clone();
            let Some(item) = src.next() else {
                self.source = None;
                if !self.found_final {
                    let held = std::mem::take(&mut self.held);
                    self.pending.extend(held);
                }
                continue;
            };
            let item = item?;
            if let Some(out) = self.step(py, &item)? {
                return Ok(Some(out));
            }
        }
    }

    /// Process one input item: `Some(item)` to yield it now.
    fn step(&mut self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Option<Py<PyAny>>> {
        match &self.mode {
            Mode::Ranges(ranges, policy) => {
                let ranges = Arc::clone(ranges);
                let policy = *policy;
                let x = self.keyed(py, item)?;
                let Some(parsed) = coerce(py, &x)? else {
                    return Ok(None);
                };
                match policy {
                    RangesPolicy::Pep440 => {
                        if !core::in_ranges(&ranges, &parsed) {
                            return Ok(None);
                        }
                        if !parsed.is_prerelease() {
                            self.found_final = true;
                            return Ok(Some(item.clone().unbind()));
                        }
                        if !self.found_final {
                            self.held.push(item.clone().unbind());
                        }
                        Ok(None)
                    }
                    RangesPolicy::Exclude | RangesPolicy::Keep => {
                        if matches!(policy, RangesPolicy::Exclude) && parsed.is_prerelease() {
                            return Ok(None);
                        }
                        Ok(core::in_ranges(&ranges, &parsed).then(|| item.clone().unbind()))
                    }
                }
            }
            Mode::Matches(m, policy) => {
                let policy = *policy;
                // Borrow juggling: the match data is not mutated by first_stage.
                let matched = {
                    let m: &Match = m;
                    self.first_stage(py, item, m)?
                };
                if !matched {
                    return Ok(None);
                }
                match policy {
                    PrereleasePolicy::All => Ok(Some(item.clone().unbind())),
                    PrereleasePolicy::NoPre => {
                        let x = self.keyed(py, item)?;
                        let keep = coerce(py, &x)?.is_none_or(|v| !v.is_prerelease());
                        Ok(keep.then(|| item.clone().unbind()))
                    }
                    PrereleasePolicy::Pep440 => {
                        let x = self.keyed(py, item)?;
                        match coerce(py, &x)? {
                            None => {
                                if self.found_final {
                                    return Ok(Some(item.clone().unbind()));
                                }
                                self.arbitrary.push(item.clone().unbind());
                                self.held.push(item.clone().unbind());
                                Ok(None)
                            }
                            Some(v) if !v.is_prerelease() => {
                                if !self.found_final {
                                    let flushed = std::mem::take(&mut self.arbitrary);
                                    self.pending.extend(flushed);
                                    self.found_final = true;
                                }
                                self.pending.push_back(item.clone().unbind());
                                Ok(None)
                            }
                            Some(_) => {
                                if !self.found_final {
                                    self.held.push(item.clone().unbind());
                                }
                                Ok(None)
                            }
                        }
                    }
                }
            }
        }
    }
}

#[pymethods]
impl FilterIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        slf.next_item(py)
    }
}

// ---------------------------------------------------------------------------
// SpecifierSet
// ---------------------------------------------------------------------------

/// A specifier set's specifiers and what is derived from them.
#[derive(Debug)]
struct SetData {
    /// `_specs`, as the caller gave them (or as canonicalisation left them).
    specs: Py<PyTuple>,
    canonicalized: bool,
    has_arbitrary: bool,
    /// The specifiers as pyroid's, when every item is one.
    native: Option<Vec<Py<PySpecifier>>>,
    /// `_get_ranges()` (`None`: some specifier has no ranges, packaging asserts).
    ranges: OnceLock<Option<Vec<Interval>>>,
    /// The same as packaging's own objects, once asked for.
    py_ranges: OnceLock<Py<PyAny>>,
}

impl SetData {
    fn new(py: Python<'_>, specs: Bound<'_, PyTuple>, canonicalized: bool, has_arbitrary: bool) -> Self {
        let mut native = Some(Vec::with_capacity(specs.len()));
        for s in specs.iter() {
            match s.cast_into::<PySpecifier>() {
                Ok(s) => {
                    if let Some(v) = native.as_mut() {
                        v.push(s.unbind());
                    }
                }
                Err(_) => native = None,
            }
        }
        let _ = py;
        Self { specs: specs.unbind(), canonicalized, has_arbitrary, native, ranges: OnceLock::new(), py_ranges: OnceLock::new() }
    }

    /// The same set with `specs` canonicalised: packaging keeps the range caches.
    fn canonicalised(&self, py: Python<'_>, specs: Bound<'_, PyTuple>) -> Self {
        let out = Self::new(py, specs, true, self.has_arbitrary);
        if let Some(r) = self.ranges.get() {
            let _ = out.ranges.set(r.clone());
        }
        if let Some(r) = self.py_ranges.get() {
            let _ = out.py_ranges.set(r.clone_ref(py));
        }
        out
    }

    fn ranges(&self, py: Python<'_>) -> PyResult<&[Interval]> {
        if let Some(r) = self.ranges.get() {
            return r.as_deref().ok_or_else(assertion);
        }
        let native = self.native.as_ref().ok_or_else(assertion)?;
        let mut per = Vec::with_capacity(native.len());
        for s in native {
            let d = s.get().data()?;
            per.push(d.ranges(py)?.to_vec());
        }
        let r = self.ranges.get_or_init(|| Some(core::intersect_all(per.iter().map(Vec::as_slice))));
        r.as_deref().ok_or_else(assertion)
    }
}

/// A set of version specifiers, as `packaging.specifiers.SpecifierSet` 26.3.
#[pyclass(frozen, subclass, module = "pyroid.packaging.specifiers", name = "SpecifierSet")]
#[derive(Debug)]
pub struct PySpecifierSet {
    state: Mutex<Option<Arc<SetData>>>,
    prereleases: Prerel,
    /// `_is_unsatisfiable` (reset by the `prereleases` setter).
    unsat: Mutex<Option<bool>>,
}

fn has_arbitrary_items(specs: &Bound<'_, PyTuple>) -> PyResult<bool> {
    for s in specs.iter() {
        if s.str()?.to_cow().is_ok_and(|t| t.contains("===")) {
            return Ok(true);
        }
    }
    Ok(false)
}

impl PySpecifierSet {
    fn state(&self) -> PyResult<Arc<SetData>> {
        let g = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        g.as_ref()
            .map(Arc::clone)
            .ok_or_else(|| PyAttributeError::new_err("'SpecifierSet' object has no attribute '_specs'"))
    }

    fn set_state(&self, s: SetData) {
        let mut g = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *g = Some(Arc::new(s));
    }

    fn build(py: Python<'_>, specifiers: Option<&Bound<'_, PyAny>>) -> PyResult<SetData> {
        let Some(specifiers) = specifiers else {
            return Ok(SetData::new(py, PyTuple::empty(py), true, false));
        };
        if let Ok(s) = specifiers.cast::<PyString>() {
            let text = s.to_cow();
            let mut items = Vec::new();
            match &text {
                Ok(t) => {
                    for part in t.split(',') {
                        let part = core_strip(part);
                        if !part.is_empty() {
                            items.push(Bound::new(py, PySpecifier::new_parsed(py, PyString::new(py, part).as_any(), None)?)?.into_any());
                        }
                    }
                }
                Err(_) => {
                    // Lone surrogates: split and strip in Python.
                    for part in s.call_method1(intern!(py, "split"), (",",))?.try_iter()? {
                        let part = part?.call_method0(intern!(py, "strip"))?;
                        if part.is_truthy()? {
                            items.push(Bound::new(py, PySpecifier::new_parsed(py, &part, None)?)?.into_any());
                        }
                    }
                }
            }
            let has_arbitrary = match &text {
                Ok(t) => t.contains("==="),
                Err(_) => s.contains("===")?,
            };
            let n = items.len();
            return Ok(SetData::new(py, PyTuple::new(py, items)?, n <= 1, has_arbitrary));
        }
        let specs = PyTuple::new(py, specifiers.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let has_arbitrary = has_arbitrary_items(&specs)?;
        let n = specs.len();
        Ok(SetData::new(py, specs, n <= 1, has_arbitrary))
    }

    /// `_canonical_specs()`: sorted by `str`, deduplicated, stored back (packaging does).
    fn canonical_specs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let s = self.state()?;
        if s.canonicalized {
            return Ok(s.specs.bind(py).clone());
        }
        let c = config(py)?;
        let builtins = py.import("builtins")?;
        let kw = PyDict::new(py);
        kw.set_item("key", builtins.getattr("str")?)?;
        let sorted = builtins.getattr("sorted")?.call((s.specs.bind(py),), Some(&kw))?;
        let deduped = c.dict_fromkeys.bind(py).call1((sorted,))?;
        let t = PyTuple::new(py, deduped.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let mut guard = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        // Another thread may have canonicalised meanwhile; either result is the same tuple.
        let current = guard.as_ref().map(Arc::clone);
        if let Some(cur) = current {
            if cur.canonicalized {
                return Ok(cur.specs.bind(py).clone());
            }
            *guard = Some(Arc::new(cur.canonicalised(py, t.clone())));
        }
        Ok(t)
    }

    fn set_prereleases_value<'py>(&self, py: Python<'py>, s: &SetData) -> PyResult<Bound<'py, PyAny>> {
        if self.prereleases.kind() != PrerelKind::None {
            return Ok(self.prereleases.get(py));
        }
        let specs = s.specs.bind(py);
        if specs.is_empty() {
            return Ok(py.None().into_bound(py));
        }
        if let Some(native) = &s.native {
            for sp in native {
                if sp.get().prereleases_value(py)?.is_truthy()? {
                    return Ok(bool_obj(py, true));
                }
            }
            return Ok(py.None().into_bound(py));
        }
        for sp in specs.iter() {
            if sp.getattr(intern!(py, "prereleases"))?.is_truthy()? {
                return Ok(bool_obj(py, true));
            }
        }
        Ok(py.None().into_bound(py))
    }

    fn filter_impl(&self, py: Python<'_>, iterable: Bound<'_, PyIterator>, prereleases: &Bound<'_, PyAny>, key: Option<&Bound<'_, PyAny>>) -> PyResult<FilterIter> {
        let s = self.state()?;
        let mut pre = prereleases.clone();
        if pre.is_none() {
            let mine = self.set_prereleases_value(py, &s)?;
            if !mine.is_none() {
                pre = mine;
            }
        }
        let specs = s.specs.bind(py);
        let mode = if specs.is_empty() {
            Mode::Matches(Match::Everything, prerelease_policy(&pre)?)
        } else if s.has_arbitrary {
            Mode::Matches(Match::All(specs.iter().map(Bound::unbind).collect()), prerelease_policy(&pre)?)
        } else if s.native.is_some() {
            Mode::Ranges(Arc::new(s.ranges(py)?.to_vec()), ranges_policy(&pre)?)
        } else {
            return Err(foreign_items(py, specs));
        };
        Ok(FilterIter::new(iterable, key, mode))
    }

    fn contains_impl(&self, py: Python<'_>, item: &Bound<'_, PyAny>, prereleases: Option<&Bound<'_, PyAny>>, installed: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        let s = self.state()?;
        let version = coerce(py, item)?;
        let none = py.None().into_bound(py);
        let mut pre = prereleases.cloned().unwrap_or_else(|| none.clone());
        if let Some(v) = &version {
            if let Some(i) = installed {
                if i.is_truthy()? && v.is_prerelease() {
                    pre = bool_obj(py, true);
                }
            }
        }
        let specs = s.specs.bind(py);
        if let Some(v) = &version {
            if !s.has_arbitrary && v.local.is_none() && !specs.is_empty() {
                let Some(native) = &s.native else {
                    return Err(foreign_items(py, specs));
                };
                if v.is_prerelease() {
                    let k = kind_of(&pre)?;
                    if k == PrerelKind::False || (k == PrerelKind::None && self.prereleases.kind() == PrerelKind::False) {
                        return Ok(false);
                    }
                }
                if s.ranges.get().is_none() {
                    let mut needs_ranges = false;
                    for sp in native {
                        let d = sp.get().data()?;
                        if !d.is_wildcard() {
                            d.require(py)?;
                        }
                        match core::fast_match(d.op, d.text(), d.parsed.as_ref(), v) {
                            None => {
                                if d.op != Op::Arbitrary && d.parsed.is_none() && !d.is_wildcard() {
                                    return Err(assertion());
                                }
                                needs_ranges = true;
                                break;
                            }
                            Some(false) => return Ok(false),
                            Some(true) => {}
                        }
                    }
                    if !needs_ranges {
                        return Ok(true);
                    }
                }
                return Ok(core::matches_bounds_only(s.ranges(py)?, v));
            }
        }
        // packaging checks the parsed version (the caller's object when it is one), or the item
        // itself when it does not parse or a `===` must see the string as written.
        let check_item = match &version {
            Some(_) if is_version_object(py, item)? => item.clone(),
            Some(v) if !s.has_arbitrary => Bound::new(py, PyVersion::from_core(v.clone()))?.into_any(),
            _ => item.clone(),
        };
        let it = PyList::new(py, [check_item])?;
        let mut f = self.filter_impl(py, it.into_any().try_iter()?, &pre, None)?;
        Ok(f.next_item(py)?.is_some())
    }
}

fn core_strip(s: &str) -> &str {
    s.trim_matches(pyroid_core::text::is_py_whitespace)
}

/// packaging fails on items that are not specifiers (setuptools passes strings) with an
/// AttributeError from the first attribute it reads.
fn foreign_items(py: Python<'_>, specs: &Bound<'_, PyTuple>) -> PyErr {
    for s in specs.iter() {
        if s.cast::<PySpecifier>().is_err() {
            let name = s.get_type().name().map(|n| n.to_string()).unwrap_or_default();
            return PyAttributeError::new_err(format!("'{name}' object has no attribute '_spec'"));
        }
    }
    let _ = py;
    PyAttributeError::new_err("'SpecifierSet' items are not specifiers")
}

#[pymethods]
impl PySpecifierSet {
    #[new]
    #[classmethod]
    #[pyo3(signature = (specifiers = None, prereleases = None, *_args, **_kwargs))]
    fn py_new(
        cls: &Bound<'_, PyType>,
        specifiers: Option<&Bound<'_, PyAny>>,
        prereleases: Option<&Bound<'_, PyAny>>,
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = cls.py();
        let me = Self { state: Mutex::new(None), prereleases: Prerel::new(None)?, unsat: Mutex::new(None) };
        if cls.is(py.get_type::<PySpecifierSet>()) {
            me.set_state(Self::build(py, specifiers)?);
            if let Some(p) = prereleases {
                me.prereleases.set(p)?;
            }
        }
        Ok(me)
    }

    #[pyo3(name = "__init__", signature = (specifiers = None, prereleases = None))]
    fn py_init(&self, py: Python<'_>, specifiers: Option<&Bound<'_, PyAny>>, prereleases: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let empty = self.state.lock().unwrap_or_else(PoisonError::into_inner).is_none();
        if empty {
            self.set_state(Self::build(py, specifiers)?);
            if let Some(p) = prereleases {
                self.prereleases.set(p)?;
            }
        }
        Ok(())
    }

    #[getter]
    fn prereleases<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = self.state()?;
        self.set_prereleases_value(py, &s)
    }

    #[setter]
    fn set_prereleases(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.prereleases.set(value)?;
        *self.unsat.lock().unwrap_or_else(PoisonError::into_inner) = None;
        Ok(())
    }

    #[getter]
    fn _prereleases<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.prereleases.get(py)
    }

    #[setter(_prereleases)]
    fn set_private_prereleases(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.prereleases.set(value)
    }

    #[getter]
    fn _specs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(self.state()?.specs.bind(py).clone())
    }

    #[setter(_specs)]
    fn set_specs(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let cur = self.state().ok();
        let t = PyTuple::new(py, value.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
        let (canon, arb) = cur.map_or((t.len() <= 1, false), |c| (c.canonicalized, c.has_arbitrary));
        self.set_state(SetData::new(py, t, canon, arb));
        Ok(())
    }

    #[getter]
    fn _has_arbitrary(&self) -> PyResult<bool> {
        Ok(self.state()?.has_arbitrary)
    }

    #[setter(_has_arbitrary)]
    fn set_has_arbitrary(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        let cur = self.state()?;
        self.set_state(SetData::new(py, cur.specs.bind(py).clone(), cur.canonicalized, value));
        Ok(())
    }

    #[getter]
    fn _canonicalized(&self) -> PyResult<bool> {
        Ok(self.state()?.canonicalized)
    }

    #[setter(_canonicalized)]
    fn set_canonicalized(&self, py: Python<'_>, value: bool) -> PyResult<()> {
        let cur = self.state()?;
        self.set_state(SetData::new(py, cur.specs.bind(py).clone(), value, cur.has_arbitrary));
        Ok(())
    }

    fn _canonical_specs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.canonical_specs(py)
    }

    /// packaging's own range objects for this set (for packaging's `ranges` module).
    fn _get_ranges<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let s = slf.get().state()?;
        if let Some(r) = s.py_ranges.get() {
            return Ok(r.bind(py).clone());
        }
        s.ranges(py)?;
        let r = config(py)?.delegate.bind(py).call1(("_get_ranges", slf))?;
        Ok(s.py_ranges.get_or_init(|| r.unbind()).bind(py).clone())
    }

    #[getter]
    fn _ranges<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if slf.get().state()?.ranges.get().is_none() {
            return Ok(py.None().into_bound(py));
        }
        Self::_get_ranges(slf)
    }

    /// Replace the cached ranges (packaging's own interval objects, or `None` to clear).
    #[setter(_ranges)]
    fn set_ranges(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let me = slf.get();
        let cur = me.state()?;
        let next = SetData::new(py, cur.specs.bind(py).clone(), cur.canonicalized, cur.has_arbitrary);
        if !value.is_none() {
            let mut intervals = Vec::new();
            for iv in value.try_iter()? {
                intervals.push(interval_from_py(py, &iv?)?);
            }
            let _ = next.ranges.set(Some(intervals));
            let _ = next.py_ranges.set(value.clone().unbind());
        }
        me.set_state(next);
        Ok(())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let me = slf.get();
        let pre = if me.prereleases.kind() == PrerelKind::None {
            String::new()
        } else {
            let s = me.state()?;
            format!(", prereleases={}", me.set_prereleases_value(py, &s)?.repr()?)
        };
        let name = slf.get_type().name()?;
        Ok(format!("<{name}({}{pre})>", slf.str()?.repr()?))
    }

    fn __str__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        let specs = self.canonical_specs(py)?;
        let mut parts = Vec::with_capacity(specs.len());
        for s in specs.iter() {
            parts.push(s.str()?);
        }
        PyString::new(py, ",").call_method1(intern!(py, "join"), (parts,))?.cast_into::<PyString>().map_err(Into::into)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.canonical_specs(py)?.hash()
    }

    fn __and__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let other_set: Bound<'py, PySpecifierSet> = if let Ok(s) = other.cast::<PyString>() {
            Bound::new(py, PySpecifierSet { state: Mutex::new(Some(Arc::new(Self::build(py, Some(s.as_any()))?))), prereleases: Prerel::new(None)?, unsat: Mutex::new(None) })?
        } else if let Ok(o) = other.cast::<PySpecifierSet>() {
            o.clone()
        } else {
            return Ok(py.NotImplemented().into_bound(py));
        };
        let me = slf.get();
        let a = me.state()?;
        let b = other_set.get().state()?;
        let mut items: Vec<Bound<'_, PyAny>> = a.specs.bind(py).iter().collect();
        items.extend(b.specs.bind(py).iter());
        let n = items.len();
        let data = SetData::new(py, PyTuple::new(py, items)?, n <= 1, a.has_arbitrary || b.has_arbitrary);
        let (pa, pb) = (me.prereleases.get(py), other_set.get().prereleases.get(py));
        let combined = if pa.is_none() || pa.eq(&pb)? {
            pb
        } else if pb.is_none() {
            pa
        } else {
            return Err(PyValueError::new_err("Cannot combine SpecifierSets with True and False prerelease overrides."));
        };
        let out = PySpecifierSet { state: Mutex::new(Some(Arc::new(data))), prereleases: Prerel::new(Some(&combined))?, unsat: Mutex::new(None) };
        Ok(Bound::new(py, out)?.into_any())
    }

    fn __richcmp__<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>, op: CompareOp) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented().into_bound(py));
        }
        let theirs = if other.cast::<PyString>().is_ok() || other.cast::<PySpecifier>().is_ok() || is_packaging_specifier(py, other)? {
            let s = other.str()?;
            let set = PySpecifierSet { state: Mutex::new(Some(Arc::new(Self::build(py, Some(s.as_any()))?))), prereleases: Prerel::new(None)?, unsat: Mutex::new(None) };
            set.canonical_specs(py)?.into_any()
        } else if let Ok(o) = other.cast::<PySpecifierSet>() {
            o.get().canonical_specs(py)?.into_any()
        } else if is_packaging_specifier_set(py, other)? {
            other.call_method0(intern!(py, "_canonical_specs"))?
        } else {
            return Ok(py.NotImplemented().into_bound(py));
        };
        let eq = slf.get().canonical_specs(py)?.eq(theirs)?;
        Ok(bool_obj(py, if matches!(op, CompareOp::Eq) { eq } else { !eq }))
    }

    fn __len__(slf: &Bound<'_, Self>) -> PyResult<usize> {
        Ok(slf.get().state()?.specs.bind(slf.py()).len())
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.state()?.specs.bind(py).as_any().try_iter()
    }

    /// Check whether this specifier set can never be satisfied.
    fn is_unsatisfiable(&self, py: Python<'_>) -> PyResult<bool> {
        if let Some(c) = *self.unsat.lock().unwrap_or_else(PoisonError::into_inner) {
            return Ok(c);
        }
        let s = self.state()?;
        let specs = s.specs.bind(py);
        let result = if specs.is_empty() {
            false
        } else {
            let Some(native) = &s.native else {
                return Err(foreign_items(py, specs));
            };
            let ranges = s.ranges(py)?;
            let mut r = ranges.is_empty();
            if !r {
                r = check_arbitrary_unsatisfiable(py, self, &s, native)?;
            }
            if !r && self.set_prereleases_value(py, &s)?.is(PyBool::new(py, false)) {
                r = core::ranges_are_prerelease_only(ranges);
            }
            r
        };
        *self.unsat.lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
        Ok(result)
    }

    #[getter]
    fn _is_unsatisfiable(&self) -> Option<bool> {
        *self.unsat.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Return the `packaging.ranges.VersionRange` this set accepts (packaging's own code).
    fn to_range<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        config(py)?.delegate.bind(py).call1(("to_range", slf))
    }

    /// Return whether every version matching this set also matches other.
    fn is_subset<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        config(py)?.delegate.bind(py).call1(("is_subset", slf, other))
    }

    /// Return whether every version matching other also matches this set.
    fn is_superset<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        config(py)?.delegate.bind(py).call1(("is_superset", slf, other))
    }

    /// Return whether this set and other share no matching versions.
    fn is_disjoint<'py>(slf: &Bound<'py, Self>, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        config(py)?.delegate.bind(py).call1(("is_disjoint", slf, other))
    }

    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.contains_impl(py, item, None, None)
    }

    /// Return whether or not the item is contained in this SpecifierSet.
    #[pyo3(signature = (item, prereleases = None, installed = None))]
    fn contains(&self, py: Python<'_>, item: &Bound<'_, PyAny>, prereleases: Option<&Bound<'_, PyAny>>, installed: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        self.contains_impl(py, item, prereleases, installed)
    }

    /// Filter items in the given iterable, that match the specifiers in this set.
    #[pyo3(signature = (iterable, prereleases = None, key = None))]
    fn filter(&self, py: Python<'_>, iterable: &Bound<'_, PyAny>, prereleases: Option<&Bound<'_, PyAny>>, key: Option<&Bound<'_, PyAny>>) -> PyResult<FilterIter> {
        let none = py.None().into_bound(py);
        let key = key.filter(|k| !k.is_none());
        self.filter_impl(py, iterable.try_iter()?, prereleases.unwrap_or(&none), key)
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [self.state()?.specs.bind(py).clone().into_any(), self.prereleases.get(py)])
    }

    fn __setstate__(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let fail = || -> PyResult<PyErr> { Ok(PyTypeError::new_err(format!("Cannot restore SpecifierSet from {}", state.repr()?))) };
        let is_spec = |s: &Bound<'_, PyAny>| -> PyResult<bool> { Ok(s.cast::<PySpecifier>().is_ok() || is_packaging_specifier(py, s)?) };
        let valid_pre = |p: &Bound<'_, PyAny>| p.is_none() || p.cast::<PyBool>().is_ok();
        let accept = |specs: Bound<'_, PyAny>, pre: Bound<'_, PyAny>| -> PyResult<bool> {
            let specs = if specs.cast::<pyo3::types::PyFrozenSet>().is_ok() {
                let builtins = py.import("builtins")?;
                let kw = PyDict::new(py);
                kw.set_item("key", builtins.getattr("str")?)?;
                builtins.getattr("tuple")?.call1((builtins.getattr("sorted")?.call((specs,), Some(&kw))?,))?
            } else {
                specs
            };
            let Ok(t) = specs.cast_into::<PyTuple>() else { return Ok(false) };
            for s in t.iter() {
                if !is_spec(&s)? {
                    return Ok(false);
                }
            }
            if !valid_pre(&pre) {
                return Ok(false);
            }
            let n = t.len();
            let arb = has_arbitrary_items(&t)?;
            self.set_state(SetData::new(py, t, n <= 1, arb));
            self.prereleases.set(&pre)?;
            *self.unsat.lock().unwrap_or_else(PoisonError::into_inner) = None;
            Ok(true)
        };
        if let Ok(t) = state.cast::<PyTuple>() {
            if t.len() == 2 {
                let (a, b) = (t.get_item(0)?, t.get_item(1)?);
                if a.cast::<PyTuple>().is_ok() && accept(a.clone(), b.clone())? {
                    return Ok(());
                }
                if let Ok(d) = b.cast::<PyDict>() {
                    let specs = d.get_item("_specs")?.unwrap_or_else(|| PyTuple::empty(py).into_any());
                    let pre = d.get_item("_prereleases")?.unwrap_or_else(|| py.None().into_bound(py));
                    if accept(specs, pre)? {
                        return Ok(());
                    }
                }
            }
        } else if let Ok(d) = state.cast::<PyDict>() {
            let specs = d.get_item("_specs")?.unwrap_or_else(|| PyTuple::empty(py).into_any());
            let pre = d.get_item("_prereleases")?.unwrap_or_else(|| py.None().into_bound(py));
            if accept(specs, pre)? {
                return Ok(());
            }
        }
        Err(fail()?)
    }

    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, _protocol: i64) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let state = slf.get().__getstate__(py)?;
        PyTuple::new(py, [config(py)?.newobj.bind(py).clone(), PyTuple::new(py, [slf.get_type()])?.into_any(), state.into_any()])
    }

    #[getter]
    fn __class__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if claims() && slf.get_type().is(py.get_type::<PySpecifierSet>()) {
            if let Some(cls) = &config(py)?.packaging_specifier_set {
                return Ok(cls.clone_ref(py).into_any());
            }
        }
        Ok(slf.get_type().into_any().unbind())
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, ["_str"])?.unbind())
    }

    #[getter]
    fn _str<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyString>> {
        slf.str()
    }
}

/// One of packaging's `(LowerBound, UpperBound)` pairs as a native interval.
fn interval_from_py(py: Python<'_>, iv: &Bound<'_, PyAny>) -> PyResult<Interval> {
    let point = |b: &Bound<'_, PyAny>| -> PyResult<Option<core::Point>> {
        let v = b.getattr(intern!(py, "version"))?;
        if v.is_none() {
            return Ok(None);
        }
        if let Ok(kind) = v.getattr(intern!(py, "kind")) {
            let inner = v.getattr(intern!(py, "version"))?;
            let version = coerce(py, &inner)?.ok_or_else(|| PyTypeError::new_err("not a version"))?;
            let k = if kind.getattr(intern!(py, "name"))?.extract::<String>()? == "AFTER_POSTS" {
                core::Kind::AfterPosts
            } else {
                core::Kind::AfterLocals
            };
            return Ok(Some(core::Point::Boundary(version, k)));
        }
        let version = coerce(py, &v)?.ok_or_else(|| PyTypeError::new_err("not a version"))?;
        Ok(Some(core::Point::Version(version)))
    };
    let (lo, up) = (iv.get_item(0)?, iv.get_item(1)?);
    Ok((
        core::Lower { point: point(&lo)?, inclusive: lo.getattr(intern!(py, "inclusive"))?.is_truthy()? },
        core::Upper { point: point(&up)?, inclusive: up.getattr(intern!(py, "inclusive"))?.is_truthy()? },
    ))
}

fn is_packaging_specifier_set(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<bool> {
    match &config(py)?.packaging_specifier_set {
        Some(cls) => x.is_instance(cls.bind(py)),
        None => Ok(false),
    }
}

/// `SpecifierSet._check_arbitrary_unsatisfiable`.
fn check_arbitrary_unsatisfiable(py: Python<'_>, set: &PySpecifierSet, s: &SetData, native: &[Py<PySpecifier>]) -> PyResult<bool> {
    let arbitrary: Vec<_> = native.iter().filter(|sp| sp.get().data().is_ok_and(|d| d.op == Op::Arbitrary)).collect();
    let Some(first) = arbitrary.first() else {
        return Ok(false);
    };
    let lower = |sp: &Py<PySpecifier>| -> PyResult<Bound<'_, PyAny>> {
        sp.get().data()?.version.bind(py).call_method0(intern!(py, "lower"))
    };
    let first_lower = lower(first)?;
    for sp in &arbitrary[1..] {
        if !lower(sp)?.eq(&first_lower)? {
            return Ok(true);
        }
    }
    let version_obj = first.get().data()?.version.bind(py).clone();
    let candidate = coerce(py, version_obj.as_any())?;
    if set.set_prereleases_value(py, s)?.is(PyBool::new(py, false)) && candidate.as_ref().is_some_and(Version::is_prerelease) {
        return Ok(true);
    }
    let standard: Vec<_> = native.iter().filter(|sp| sp.get().data().is_ok_and(|d| d.op != Op::Arbitrary)).collect();
    if standard.is_empty() {
        return Ok(false);
    }
    let Some(candidate) = candidate else {
        return Ok(true);
    };
    let cand = Bound::new(py, PyVersion::from_core(candidate))?;
    for sp in standard {
        if !sp.get().contains_impl(py, cand.as_any(), None)? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpecifier>()?;
    m.add_class::<PySpecifierSet>()?;
    m.add_class::<FilterIter>()?;
    Ok(())
}
