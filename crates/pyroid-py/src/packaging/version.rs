// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! `pyroid.packaging.version.Version`.

use std::cmp::Ordering;
use std::sync::OnceLock;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyDeprecationWarning, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyString, PyTuple, PyType};
use pyroid_core::packaging::num::Num;
use pyroid_core::packaging::version::{self as core, Local, Pre, Version};

use super::cell::{HashCache, StateCell, StateRef};
use super::{claims, config};

/// A parsed version and the Python objects made from it on demand.
#[derive(Debug)]
struct Inner {
    v: Version,
    hash: HashCache,
    text: OnceLock<Py<PyString>>,
    release: OnceLock<Py<PyTuple>>,
}

impl Inner {
    fn new(v: Version) -> Self {
        Self { v, hash: HashCache::new(), text: OnceLock::new(), release: OnceLock::new() }
    }
}

/// A PEP 440 version, as `packaging.version.Version` 26.3.
///
/// Immutable in practice, but packaging's objects can be changed in place (`v._version = ...`,
/// which hatchling does, `__setstate__`, `__init__` again): see [`StateCell`].
#[pyclass(frozen, subclass, module = "pyroid.packaging.version", name = "Version")]
#[derive(Debug)]
pub struct PyVersion {
    inner: StateCell<Inner>,
}

pub fn num_to_py<'py>(py: Python<'py>, n: &Num) -> PyResult<Bound<'py, PyAny>> {
    match n {
        Num::Small(v) => Ok(v.into_pyobject(py)?.into_any()),
        Num::Big(d) => py.get_type::<PyInt>().call1((&**d,)),
    }
}

/// `int` (a non-negative Python int, bools included) to a [`Num`]; `None` if it is not one.
fn py_to_num(x: &Bound<'_, PyAny>) -> PyResult<Option<Num>> {
    if !x.is_instance_of::<PyInt>() {
        return Ok(None);
    }
    if let Ok(v) = x.extract::<u64>() {
        return Ok(Some(Num::Small(v)));
    }
    let s = x.str()?.to_cow()?.into_owned();
    if s.starts_with('-') {
        return Ok(None);
    }
    Ok(Some(Num::from_digits(&s)))
}

fn local_to_py<'py>(py: Python<'py>, local: &[Local]) -> PyResult<Bound<'py, PyTuple>> {
    let mut items = Vec::with_capacity(local.len());
    for seg in local {
        items.push(match seg {
            Local::Num(n) => num_to_py(py, n)?,
            Local::Str(s) => PyString::new(py, s).into_any(),
        });
    }
    PyTuple::new(py, items)
}

fn tuple_of_nums<'py>(py: Python<'py>, nums: &[Num]) -> PyResult<Bound<'py, PyTuple>> {
    let mut items = Vec::with_capacity(nums.len());
    for n in nums {
        items.push(num_to_py(py, n)?);
    }
    PyTuple::new(py, items)
}

fn invalid_version(py: Python<'_>, msg: String) -> PyErr {
    match config(py) {
        Ok(c) => match c.invalid_version.bind(py).call1((msg,)) {
            Ok(e) => PyErr::from_value(e),
            Err(e) => e,
        },
        Err(e) => e,
    }
}

fn invalid_repr(py: Python<'_>, version: &Bound<'_, PyAny>) -> PyErr {
    match version.repr() {
        Ok(r) => invalid_version(py, format!("Invalid version: {r}")),
        Err(e) => e,
    }
}

/// `Version(text)` for a `str`.
pub fn parse_str(py: Python<'_>, s: &Bound<'_, PyString>) -> PyResult<Version> {
    let Ok(text) = s.to_cow() else {
        // A lone surrogate: never part of a valid version.
        return Err(invalid_repr(py, s.as_any()));
    };
    let Some(parsed) = core::parse(&text) else {
        return Err(invalid_repr(py, s.as_any()));
    };
    // packaging converts each digit run with int(), which raises ValueError past
    // sys.get_int_max_str_digits(); only runs over 19 digits can.
    if !parsed.long_runs.is_empty() {
        let int = py.get_type::<PyInt>();
        for r in &parsed.long_runs {
            int.call1((&text[r.start..r.end],))?;
        }
    }
    Ok(parsed.version)
}

/// The state of one of packaging's own `Version` objects (its slots).
pub fn from_packaging(py: Python<'_>, v: &Bound<'_, PyAny>) -> PyResult<Option<Version>> {
    let get = |name: &str| v.getattr(name);
    let slots = (|| -> PyResult<Option<Version>> {
        from_slots(py, &get("_epoch")?, &get("_release")?, &get("_pre")?, &get("_post")?, &get("_dev")?, &get("_local")?)
    })();
    match slots {
        Ok(Some(found)) => Ok(Some(found)),
        // Something that is not packaging 26's layout: its string form.
        _ => parse_any(py, &v.str()?.into_any()).map(Some),
    }
}

/// `Version(x)` for any `x`, as packaging 26.3's `__init__` treats it.
pub fn parse_any(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Version> {
    if let Ok(s) = x.cast::<PyString>() {
        return parse_str(py, s);
    }
    // packaging's own steps (a set test, then the regex) for a non-str; they raise
    // InvalidVersion, or (for objects that pretend to be strings) return the parts.
    let parts = config(py)?.nonstr_version.bind(py).call1((x,))?;
    state_from_parts(py, &parts)?.ok_or_else(|| invalid_repr(py, x))
}

/// (epoch, release, pre, post, dev, local) as packaging stores them, to a [`Version`].
fn state_from_parts(py: Python<'_>, t: &Bound<'_, PyAny>) -> PyResult<Option<Version>> {
    let Ok(t) = t.cast::<PyTuple>() else {
        return Ok(None);
    };
    if t.len() != 6 {
        return Ok(None);
    }
    from_slots(
        py,
        &t.get_item(0)?,
        &t.get_item(1)?,
        &t.get_item(2)?,
        &t.get_item(3)?,
        &t.get_item(4)?,
        &t.get_item(5)?,
    )
}

/// packaging's slot values (`_epoch`, `_release`, `_pre`, `_post`, `_dev`, `_local`).
fn from_slots(
    _py: Python<'_>,
    epoch: &Bound<'_, PyAny>,
    release: &Bound<'_, PyAny>,
    pre: &Bound<'_, PyAny>,
    post: &Bound<'_, PyAny>,
    dev: &Bound<'_, PyAny>,
    local: &Bound<'_, PyAny>,
) -> PyResult<Option<Version>> {
    let Some(epoch) = py_to_num(epoch)? else {
        return Ok(None);
    };
    let Ok(rel) = release.cast::<PyTuple>() else {
        return Ok(None);
    };
    let mut release = Vec::with_capacity(rel.len());
    for x in rel.iter() {
        let Some(n) = py_to_num(&x)? else {
            return Ok(None);
        };
        release.push(n);
    }
    if release.is_empty() {
        return Ok(None);
    }
    let letter_pair = |x: &Bound<'_, PyAny>| -> PyResult<Option<Option<(String, Num)>>> {
        if x.is_none() {
            return Ok(Some(None));
        }
        let Ok(t) = x.cast::<PyTuple>() else {
            return Ok(None);
        };
        if t.len() != 2 {
            return Ok(None);
        }
        let Ok(l) = t.get_item(0)?.extract::<String>() else {
            return Ok(None);
        };
        let Some(n) = py_to_num(&t.get_item(1)?)? else {
            return Ok(None);
        };
        Ok(Some(Some((l, n))))
    };
    let Some(pre) = letter_pair(pre)? else {
        return Ok(None);
    };
    let pre = match pre {
        None => None,
        Some((l, n)) => match l.as_str() {
            "a" => Some((Pre::A, n)),
            "b" => Some((Pre::B, n)),
            "rc" => Some((Pre::Rc, n)),
            _ => return Ok(None),
        },
    };
    let Some(post) = letter_pair(post)? else {
        return Ok(None);
    };
    let post = match post {
        None => None,
        Some((l, n)) if l == "post" => Some(n),
        Some(_) => return Ok(None),
    };
    let Some(dev) = letter_pair(dev)? else {
        return Ok(None);
    };
    let dev = match dev {
        None => None,
        Some((l, n)) if l == "dev" => Some(n),
        Some(_) => return Ok(None),
    };
    let local = if local.is_none() {
        None
    } else {
        let Ok(t) = local.cast::<PyTuple>() else {
            return Ok(None);
        };
        let mut segs = Vec::with_capacity(t.len());
        for x in t.iter() {
            if let Ok(s) = x.cast::<PyString>() {
                segs.push(Local::Str(s.to_cow()?.into()));
            } else if let Some(n) = py_to_num(&x)? {
                segs.push(Local::Num(n));
            } else {
                return Ok(None);
            }
        }
        Some(segs)
    };
    Ok(Some(Version { epoch, release, pre, post, dev, local }))
}

impl PyVersion {
    /// A new object holding `v`.
    pub fn from_core(v: Version) -> Self {
        Self { inner: StateCell::with(Inner::new(v)) }
    }

    fn empty() -> Self {
        Self { inner: StateCell::new() }
    }

    fn inner(&self) -> PyResult<StateRef<'_, Inner>> {
        self.inner
            .get()
            .ok_or_else(|| PyAttributeError::new_err("'Version' object has no attribute '_release'"))
    }

    /// The parsed version (a copy: the object can be changed in place).
    pub fn core(&self) -> PyResult<Version> {
        Ok(self.inner()?.v.clone())
    }

    /// Run `f` on the parsed version.
    pub fn with_core<R>(&self, f: impl FnOnce(&Version) -> R) -> PyResult<R> {
        Ok(f(&self.inner()?.v))
    }

    fn is_set(&self) -> bool {
        self.inner.is_set()
    }

    /// Set the state (the first time), or replace it, as packaging's slot assignments do.
    fn set(&self, v: Version) {
        self.inner.set(Inner::new(v));
    }

    fn release_tuple<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner()?;
        if let Some(t) = inner.release.get() {
            return Ok(t.bind(py).clone());
        }
        let t = tuple_of_nums(py, &inner.v.release)?;
        let _ = inner.release.set(t.clone().unbind());
        Ok(t)
    }

    /// `_key` (`_cmpkey(...)`), built on each call (only packaging's own code reads it).
    fn key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let inner = self.inner()?;
        let v = &inner.v;
        let s = v.suffix();
        let suffix = PyTuple::new(
            py,
            [
                s.pre_rank.into_pyobject(py)?.into_any(),
                num_to_py(py, &s.pre_n)?,
                s.post_rank.into_pyobject(py)?.into_any(),
                num_to_py(py, &s.post_n)?,
                s.dev_rank.into_pyobject(py)?.into_any(),
                num_to_py(py, &s.dev_n)?,
            ],
        )?;
        let epoch = num_to_py(py, &v.epoch)?;
        let trimmed = tuple_of_nums(py, core::trim_zeros(&v.release, false))?;
        let key = match &v.local {
            None => PyTuple::new(py, [epoch, trimmed.into_any(), suffix.into_any()])?,
            Some(local) => {
                let mut segs = Vec::with_capacity(local.len());
                for seg in local {
                    segs.push(match seg {
                        Local::Num(n) => PyTuple::new(
                            py,
                            [num_to_py(py, n)?, PyString::new(py, "").into_any()],
                        )?,
                        Local::Str(st) => PyTuple::new(
                            py,
                            [(-1i64).into_pyobject(py)?.into_any(), PyString::new(py, st).into_any()],
                        )?,
                    });
                }
                PyTuple::new(
                    py,
                    [epoch, trimmed.into_any(), suffix.into_any(), PyTuple::new(py, segs)?.into_any()],
                )?
            }
        };
        Ok(key)
    }

    fn hash_value(&self, py: Python<'_>) -> PyResult<isize> {
        let inner = self.inner()?;
        if let Some(h) = inner.hash.get() {
            return Ok(h);
        }
        let h = match inner.v.py_hash() {
            #[allow(clippy::cast_possible_truncation)]
            Some(h) => h as isize,
            None => self.key(py)?.hash()?,
        };
        inner.hash.set(h);
        Ok(h)
    }

    fn text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        let inner = self.inner()?;
        if let Some(t) = inner.text.get() {
            return Ok(t.bind(py).clone());
        }
        let t = PyString::new(py, &inner.v.to_string_py());
        let _ = inner.text.set(t.clone().unbind());
        Ok(t)
    }

    /// A version of `cls` (pyroid's class or a subclass) holding `v`.
    pub fn new_of<'py>(cls: &Bound<'py, PyType>, v: Version) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        if cls.is(py.get_type::<PyVersion>()) {
            return Ok(Bound::new(py, PyVersion::from_core(v))?.into_any());
        }
        let obj = cls.call_method1(intern!(py, "__new__"), (cls,))?;
        obj.cast::<PyVersion>()?.get().set(v);
        Ok(obj)
    }
}

fn is_exact(slf: &Bound<'_, PyVersion>) -> bool {
    slf.get_type().is(slf.py().get_type::<PyVersion>())
}

fn letter_tuple<'py>(py: Python<'py>, letter: &str, n: &Num) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, [PyString::new(py, letter).into_any(), num_to_py(py, n)?])
}

fn bool_obj(py: Python<'_>, b: bool) -> Py<PyAny> {
    PyBool::new(py, b).to_owned().into_any().unbind()
}

fn compare(o: Ordering, op: CompareOp) -> bool {
    match op {
        CompareOp::Lt => o == Ordering::Less,
        CompareOp::Le => o != Ordering::Greater,
        CompareOp::Eq => o == Ordering::Equal,
        CompareOp::Ne => o != Ordering::Equal,
        CompareOp::Gt => o == Ordering::Greater,
        CompareOp::Ge => o != Ordering::Less,
    }
}

/// `str(int)` of a validation message argument, as an f-string prints it.
fn fmt_value(x: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(x.str()?.to_cow()?.into_owned())
}

fn validate_epoch(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Num> {
    // `value or 0`
    let epoch = if value.is_truthy()? { value.clone() } else { 0i64.into_pyobject(py)?.into_any() };
    if let Some(n) = py_to_num(&epoch)? {
        return Ok(n);
    }
    Err(invalid_version(py, format!("epoch must be non-negative integer, got {}", fmt_value(&epoch)?)))
}

fn validate_release(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Vec<Num>> {
    let release = if value.is_none() { PyTuple::new(py, [0i64])?.into_any() } else { value.clone() };
    if let Ok(t) = release.cast::<PyTuple>() {
        if !t.is_empty() {
            let mut out = Vec::with_capacity(t.len());
            for x in t.iter() {
                match py_to_num(&x)? {
                    Some(n) => out.push(n),
                    None => {
                        out.clear();
                        break;
                    }
                }
            }
            if out.len() == t.len() {
                return Ok(out);
            }
        }
    }
    Err(invalid_version(
        py,
        format!(
            "release must be a non-empty tuple of non-negative integers, got {}",
            fmt_value(&release)?
        ),
    ))
}

fn validate_pre(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<(Pre, Num)>> {
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(t) = value.cast::<PyTuple>() {
        if t.len() == 2 {
            let letter = t.get_item(0)?;
            let number = t.get_item(1)?;
            if let Ok(l) = letter.cast::<PyString>() {
                // normalize_pre: str.lower(), then the spellings; only ASCII can give a/b/rc.
                if let Some(k) = l.to_cow().ok().and_then(|s| Pre::from_letter(&s)) {
                    if let Some(n) = py_to_num(&number)? {
                        return Ok(Some((k, n)));
                    }
                }
            }
        }
    }
    Err(invalid_version(
        py,
        format!("pre must be a tuple of ('a'|'b'|'rc', non-negative int), got {}", fmt_value(value)?),
    ))
}

fn validate_number(py: Python<'_>, what: &str, value: &Bound<'_, PyAny>) -> PyResult<Option<Num>> {
    if value.is_none() {
        return Ok(None);
    }
    if let Some(n) = py_to_num(value)? {
        return Ok(Some(n));
    }
    Err(invalid_version(py, format!("{what} must be non-negative integer, got {}", fmt_value(value)?)))
}

fn validate_local(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<Vec<Local>>> {
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(s) = value.cast::<PyString>() {
        if let Some(l) = s.to_cow().ok().and_then(|t| core::parse_local(&t)) {
            return Ok(Some(l));
        }
    }
    Err(invalid_version(py, format!("local must be a valid version string, got {}", value.repr()?)))
}

#[pymethods]
impl PyVersion {
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = cls.py();
        // packaging's Version.__new__ is object.__new__: a subclass instance is filled in by
        // __init__; `Version.__new__(Version)` (unpickling) gives an empty object. The exact
        // class is parsed here, and its __init__ (which checks the arguments) does nothing more.
        if cls.is(py.get_type::<PyVersion>()) {
            let version = match (args.len(), kwargs) {
                (1, None) => Some(args.get_item(0)?),
                (0, Some(kw)) if kw.len() == 1 => kw.get_item("version")?,
                _ => None,
            };
            if let Some(v) = version {
                return Ok(PyVersion::from_core(parse_any(py, &v)?));
            }
        }
        Ok(PyVersion::empty())
    }

    #[pyo3(name = "__init__", signature = (version))]
    fn py_init(&self, py: Python<'_>, version: &Bound<'_, PyAny>) -> PyResult<()> {
        // The exact class was parsed by __new__ (the common case, nothing to do).
        if !self.is_set() {
            self.set(parse_any(py, version)?);
        }
        Ok(())
    }

    /// Return a new version composed of the various parts.
    #[classmethod]
    #[pyo3(signature = (*, epoch = None, release, pre = None, post = None, dev = None, local = None))]
    fn from_parts<'py>(
        cls: &Bound<'py, PyType>,
        epoch: Option<&Bound<'py, PyAny>>,
        release: &Bound<'py, PyAny>,
        pre: Option<&Bound<'py, PyAny>>,
        post: Option<&Bound<'py, PyAny>>,
        dev: Option<&Bound<'py, PyAny>>,
        local: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        let epoch = match epoch {
            Some(e) => validate_epoch(py, e)?,
            None => Num::ZERO,
        };
        let release = validate_release(py, release)?;
        let pre = match pre {
            Some(p) => validate_pre(py, p)?,
            None => None,
        };
        let post = match post {
            Some(p) => validate_number(py, "post", p)?,
            None => None,
        };
        let dev = match dev {
            Some(p) => validate_number(py, "dev", p)?,
            None => None,
        };
        let local = match local {
            Some(p) => validate_local(py, p)?,
            None => None,
        };
        PyVersion::new_of(cls, Version { epoch, release, pre, post, dev, local })
    }

    /// Return a new version with parts replaced.
    #[pyo3(signature = (**kwargs))]
    fn __replace__<'py>(
        slf: &Bound<'py, Self>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let old = slf.get().core()?;
        let mut v = old.clone();
        if let Some(kw) = kwargs {
            if let Some(x) = kw.get_item("epoch")? {
                v.epoch = validate_epoch(py, &x)?;
            }
            if let Some(x) = kw.get_item("release")? {
                v.release = validate_release(py, &x)?;
            }
            if let Some(x) = kw.get_item("pre")? {
                v.pre = validate_pre(py, &x)?;
            }
            if let Some(x) = kw.get_item("post")? {
                v.post = validate_number(py, "post", &x)?;
            }
            if let Some(x) = kw.get_item("dev")? {
                v.dev = validate_number(py, "dev", &x)?;
            }
            if let Some(x) = kw.get_item("local")? {
                v.local = validate_local(py, &x)?;
            }
        }
        if v == old {
            return Ok(slf.clone().into_any());
        }
        PyVersion::new_of(&slf.get_type(), v)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().name()?;
        let text = if is_exact(slf) { slf.get().text(slf.py())?.into_any() } else { slf.str()?.into_any() };
        Ok(format!("<{name}({})>", text.repr()?))
    }

    fn __str__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyString>> {
        let py = slf.py();
        if is_exact(slf) {
            return slf.get().text(py);
        }
        // packaging's __str__ reads the properties, which a subclass may override.
        let mut out = String::new();
        let release = slf.getattr(intern!(py, "release"))?;
        let parts: Vec<String> =
            release.try_iter()?.map(|x| x.and_then(|x| fmt_value(&x))).collect::<PyResult<_>>()?;
        out.push_str(&parts.join("."));
        let epoch = slf.getattr(intern!(py, "epoch"))?;
        if epoch.is_truthy()? {
            out = format!("{}!{out}", fmt_value(&epoch)?);
        }
        let pre = slf.getattr(intern!(py, "pre"))?;
        if !pre.is_none() {
            for x in pre.try_iter()? {
                out.push_str(&fmt_value(&x?)?);
            }
        }
        let post = slf.getattr(intern!(py, "post"))?;
        if !post.is_none() {
            out.push_str(&format!(".post{}", fmt_value(&post)?));
        }
        let dev = slf.getattr(intern!(py, "dev"))?;
        if !dev.is_none() {
            out.push_str(&format!(".dev{}", fmt_value(&dev)?));
        }
        let local = slf.getattr(intern!(py, "local"))?;
        if !local.is_none() {
            out.push_str(&format!("+{}", fmt_value(&local)?));
        }
        Ok(PyString::new(py, &out))
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.hash_value(py)
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        if let Ok(o) = other.cast::<PyVersion>() {
            let a = self.inner()?;
            let b = o.get().inner()?;
            return Ok(bool_obj(py, compare(a.v.cmp_key(&b.v), op)));
        }
        // packaging's own objects (any _BaseVersion): compare the keys, as packaging does.
        let c = config(py)?;
        if let Some(base) = &c.packaging_base_version {
            if other.is_instance(base.bind(py))? {
                let theirs = other.getattr(intern!(py, "_key"))?;
                return Ok(self.key(py)?.rich_compare(theirs, op)?.unbind());
            }
        }
        Ok(py.NotImplemented())
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self._epoch(py)?,
                self.release_tuple(py)?.into_any(),
                self._pre(py)?,
                self._post(py)?,
                self._dev(py)?,
                self._local(py)?,
            ],
        )
    }

    fn __setstate__(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let restored = (|| -> PyResult<Option<Version>> {
            if let Ok(t) = state.cast::<PyTuple>() {
                if t.len() == 6 {
                    return state_from_parts(py, state);
                }
                if t.len() == 2 {
                    if let Ok(d) = t.get_item(1)?.cast_into::<PyDict>() {
                        let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
                            Ok(d.get_item(k)?.unwrap_or_else(|| py.None().into_bound(py)))
                        };
                        let (Some(e), Some(r)) = (d.get_item("_epoch")?, d.get_item("_release")?)
                        else {
                            return Ok(None);
                        };
                        return from_slots(py, &e, &r, &get("_pre")?, &get("_post")?, &get("_dev")?, &get("_local")?);
                    }
                }
            }
            if let Ok(d) = state.cast::<PyDict>() {
                if let Some(nt) = d.get_item("_version")? {
                    if !nt.is_none() {
                        return from_slots(
                            py,
                            &nt.getattr("epoch")?,
                            &nt.getattr("release")?,
                            &nt.getattr("pre")?,
                            &nt.getattr("post")?,
                            &nt.getattr("dev")?,
                            &nt.getattr("local")?,
                        );
                    }
                }
            }
            Ok(None)
        })()?;
        match restored {
            Some(v) => {
                self.set(v);
                Ok(())
            }
            None => Err(PyTypeError::new_err(format!("Cannot restore Version from {}", state.repr()?))),
        }
    }

    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, _protocol: i64) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        // `copyreg.__newobj__` would make pickle compare `cls` with `__class__` (the claim).
        let state = slf.get().__getstate__(py)?;
        PyTuple::new(
            py,
            [config(py)?.restore.bind(py).clone(), PyTuple::new(py, [slf.get_type().into_any(), state.into_any()])?.into_any()],
        )
    }

    /// What `isinstance` sees: packaging's `Version` while packaging is imported and not
    /// patched, so packaging's own code accepts this object.
    #[getter]
    fn __class__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if claims() && is_exact(slf) {
            if let Some(cls) = &config(py)?.packaging_version {
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

    #[getter]
    fn epoch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        num_to_py(py, &self.inner()?.v.epoch)
    }

    #[getter]
    fn release<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.release_tuple(py)
    }

    #[getter]
    fn pre<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self._pre(py)
    }

    #[getter]
    fn post<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner()?.v.post {
            Some(n) => num_to_py(py, n),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    fn dev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner()?.v.dev {
            Some(n) => num_to_py(py, n),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    fn local(&self) -> PyResult<Option<String>> {
        self.with_core(Version::local_str)
    }

    #[getter]
    fn public(slf: &Bound<'_, Self>) -> PyResult<String> {
        if is_exact(slf) {
            return slf.get().with_core(Version::public);
        }
        let s = slf.str()?.to_cow()?.into_owned();
        Ok(s.split('+').next().unwrap_or("").to_owned())
    }

    #[getter]
    fn base_version(slf: &Bound<'_, Self>) -> PyResult<String> {
        if is_exact(slf) {
            return slf.get().with_core(Version::base_version);
        }
        let py = slf.py();
        let release = slf.getattr(intern!(py, "release"))?;
        let parts: Vec<String> =
            release.try_iter()?.map(|x| x.and_then(|x| fmt_value(&x))).collect::<PyResult<_>>()?;
        let seg = parts.join(".");
        let epoch = slf.getattr(intern!(py, "epoch"))?;
        Ok(if epoch.is_truthy()? { format!("{}!{seg}", fmt_value(&epoch)?) } else { seg })
    }

    #[getter]
    fn is_prerelease(slf: &Bound<'_, Self>) -> PyResult<bool> {
        if is_exact(slf) {
            return slf.get().with_core(Version::is_prerelease);
        }
        let py = slf.py();
        Ok(!slf.getattr(intern!(py, "dev"))?.is_none() || !slf.getattr(intern!(py, "pre"))?.is_none())
    }

    #[getter]
    fn is_postrelease(slf: &Bound<'_, Self>) -> PyResult<bool> {
        if is_exact(slf) {
            return slf.get().with_core(|v| v.post.is_some());
        }
        Ok(!slf.getattr(intern!(slf.py(), "post"))?.is_none())
    }

    #[getter]
    fn is_devrelease(slf: &Bound<'_, Self>) -> PyResult<bool> {
        if is_exact(slf) {
            return slf.get().with_core(|v| v.dev.is_some());
        }
        Ok(!slf.getattr(intern!(slf.py(), "dev"))?.is_none())
    }

    #[getter]
    fn major<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        release_item(slf, 0)
    }

    #[getter]
    fn minor<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        release_item(slf, 1)
    }

    #[getter]
    fn micro<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        release_item(slf, 2)
    }

    // packaging's private state, read by packaging's own code (ranges, specifiers, utils).

    #[getter]
    fn _epoch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        num_to_py(py, &self.inner()?.v.epoch)
    }

    #[getter]
    fn _release<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.release_tuple(py)
    }

    #[getter]
    fn _pre<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner()?.v.pre {
            Some((k, n)) => Ok(letter_tuple(py, k.letter(), n)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    fn _post<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner()?.v.post {
            Some(n) => Ok(letter_tuple(py, "post", n)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    fn _dev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner()?.v.dev {
            Some(n) => Ok(letter_tuple(py, "dev", n)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    fn _local<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner()?.v.local {
            Some(l) => Ok(local_to_py(py, l)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    #[getter]
    fn _key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.key(py)
    }

    #[getter]
    fn _key_cache<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.key(py)
    }

    #[getter]
    fn _hash_cache(&self, py: Python<'_>) -> PyResult<isize> {
        self.hash_value(py)
    }

    #[getter]
    fn _version<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyErr::warn(
            py,
            &py.get_type::<PyDeprecationWarning>(),
            c"Version._version is private and will be removed soon",
            1,
        )?;
        let nt = config(py)?.version_tuple.bind(py);
        nt.call1((
            self._epoch(py)?,
            self.release_tuple(py)?,
            self._dev(py)?,
            self._pre(py)?,
            self._post(py)?,
            self._local(py)?,
        ))
    }

    #[setter(_version)]
    fn set_version_tuple(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        PyErr::warn(
            py,
            &py.get_type::<PyDeprecationWarning>(),
            c"Version._version is private and will be removed soon",
            1,
        )?;
        let v = from_slots(
            py,
            &value.getattr("epoch")?,
            &value.getattr("release")?,
            &value.getattr("pre")?,
            &value.getattr("post")?,
            &value.getattr("dev")?,
            &value.getattr("local")?,
        )?;
        match v {
            Some(v) => {
                self.set(v);
                Ok(())
            }
            None => Err(PyTypeError::new_err(format!(
                "pyroid.packaging cannot store {} in a Version",
                value.repr()?
            ))),
        }
    }
}

fn release_item<'py>(slf: &Bound<'py, PyVersion>, i: usize) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    let release = if is_exact(slf) {
        slf.get().release_tuple(py)?.into_any()
    } else {
        slf.getattr(intern!(py, "release"))?
    };
    if release.len()? > i { release.get_item(i) } else { Ok(0i64.into_pyobject(py)?.into_any()) }
}

/// Parse the given version string (`packaging.version.parse`).
#[pyfunction]
fn parse(py: Python<'_>, version: &Bound<'_, PyAny>) -> PyResult<PyVersion> {
    Ok(PyVersion::from_core(parse_any(py, version)?))
}

/// Normalize the pre-release segment of a version string.
#[pyfunction]
#[pyo3(signature = (letter, /))]
fn normalize_pre(letter: &Bound<'_, PyString>) -> PyResult<String> {
    // str.lower() (Unicode, with the final-sigma rule), then the alternate spellings.
    let low = letter.call_method0(intern!(letter.py(), "lower"))?;
    Ok(core::normalize_letter(&low.cast::<PyString>()?.to_cow()?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVersion>()?;
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_pre, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_version, m)?)?;
    Ok(())
}


/// `canonicalize_version(version, strip_trailing_zero=...)`.
#[pyfunction]
#[pyo3(signature = (version, strip_trailing_zero = true))]
fn canonicalize_version<'py>(py: Python<'py>, version: &Bound<'py, PyAny>, strip_trailing_zero: bool) -> PyResult<Bound<'py, PyAny>> {
    let parsed = if version.is_instance_of::<PyString>() {
        match parse_any(py, version) {
            Ok(v) => v,
            Err(e) if e.is_instance(py, config(py)?.invalid_version.bind(py)) => return Ok(version.str()?.into_any()),
            Err(e) => return Err(e),
        }
    } else if let Ok(v) = version.cast::<PyVersion>() {
        v.get().core()?
    } else if !strip_trailing_zero {
        return Ok(version.str()?.into_any());
    } else if is_packaging_version(py, version)? {
        from_packaging(py, version)?.ok_or_else(|| invalid_repr(py, version))?
    } else {
        // _TrimmedRelease(x) for anything else runs Version.__init__(x).
        parse_any(py, version)?
    };
    let text = if strip_trailing_zero { parsed.to_string_trimmed() } else { parsed.to_string_py() };
    Ok(PyString::new(py, &text).into_any())
}

fn is_packaging_version(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<bool> {
    match &config(py)?.packaging_version_any {
        Some(cls) => x.is_instance(cls.bind(py)),
        None => Ok(false),
    }
}
