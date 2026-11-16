// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! `pyroid.packaging.markers.Marker` and the requirement parser behind
//! `pyroid.packaging.requirements.Requirement`.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyKeyError, PyRecursionError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFrozenSet, PyList, PySet, PyString, PyTuple, PyType};
use pyroid_core::packaging::markers::{self as core, Elem, List, MarkerVar, SyntaxError};
use pyroid_core::packaging::specifiers::{self as spec_core, Op};

use super::cell::{HashCache, StateCell, StateRef};
use super::specifiers::coerce;
use super::{claims, config};

// ---------------------------------------------------------------------------
// Source text
// ---------------------------------------------------------------------------

/// A Python `str` as code points (lone surrogates included).
fn code_points(s: &Bound<'_, PyString>) -> PyResult<Vec<u32>> {
    if let Ok(t) = s.to_cow() {
        return Ok(t.chars().map(u32::from).collect());
    }
    let b = s.call_method1(intern!(s.py(), "encode"), ("utf-32-le", "surrogatepass"))?;
    let bytes = b.cast::<PyBytes>()?.as_bytes();
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// A Python `str` from code points.
fn py_str<'py>(py: Python<'py>, cps: &[u32]) -> PyResult<Bound<'py, PyString>> {
    let text: Option<String> = cps.iter().map(|&c| char::from_u32(c)).collect();
    if let Some(t) = text {
        return Ok(PyString::new(py, &t));
    }
    let bytes: Vec<u8> = cps.iter().flat_map(|c| c.to_le_bytes()).collect();
    PyBytes::new(py, &bytes)
        .call_method1(intern!(py, "decode"), ("utf-32-le", "surrogatepass"))?
        .cast_into::<PyString>()
        .map_err(Into::into)
}

/// `ast.literal_eval(token)` for a quoted string: `Ok(None)` when packaging reports
/// "Invalid quoted string" (`SyntaxError` or `ValueError`).
fn quoted(py: Python<'_>, token: &[u32]) -> PyResult<Option<String>> {
    if let Some(v) = core::simple_quoted(token) {
        return Ok(Some(v));
    }
    let c = config(py)?;
    let text = py_str(py, token)?;
    match c.literal_eval.bind(py).call1((text,)) {
        Ok(v) => {
            let s = v.str()?;
            match s.to_cow() {
                Ok(t) => Ok(Some(t.into_owned())),
                // A value with a lone surrogate (from an escape): pyroid keeps values as Rust
                // text, so such a marker is rejected as packaging rejects a raw one.
                Err(_) => Ok(None),
            }
        }
        Err(e)
            if e.is_instance_of::<pyo3::exceptions::PySyntaxError>(py)
                || e.is_instance_of::<PyValueError>(py) =>
        {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// packaging's `ParserSyntaxError` for `err` in `source`, wrapped in `wrapper`
/// (`InvalidMarker`/`InvalidRequirement`) as packaging raises it (`from e`).
fn syntax_error(
    py: Python<'_>,
    err: &SyntaxError,
    source: &Bound<'_, PyString>,
    wrapper: &Bound<'_, PyAny>,
) -> PyErr {
    let r = (|| -> PyResult<PyErr> {
        let c = config(py)?;
        let kw = PyDict::new(py);
        kw.set_item("source", source)?;
        kw.set_item("span", (err.span.0, err.span.1))?;
        let inner = c
            .parser_syntax_error
            .bind(py)
            .call((err.message.as_str(),), Some(&kw))?;
        let outer = wrapper.call1((inner.str()?,))?;
        outer.setattr(intern!(py, "__cause__"), &inner)?;
        Ok(PyErr::from_value(outer))
    })();
    r.unwrap_or_else(|e| e)
}

fn unicode_version(py: Python<'_>) -> PyResult<pyroid_core::ftfy::UnicodeVersion> {
    Ok(config(py)?.unicode)
}

/// `canonicalize_name(value)` (Python's `str.lower` for non-ASCII text).
pub fn canonicalize(py: Python<'_>, value: &str) -> PyResult<String> {
    if value.is_ascii() {
        return Ok(core::canonicalize_ascii(value));
    }
    let low = PyString::new(py, value).call_method0(intern!(py, "lower"))?;
    Ok(core::canonicalize_lowered(
        &low.cast::<PyString>()?.to_cow()?,
    ))
}

/// What the core parser asks of Python: quoted strings and `canonicalize_name`.
struct PyHost<'py>(Python<'py>);

impl core::Host for PyHost<'_> {
    type Error = PyErr;

    fn quoted(&mut self, token: &[u32]) -> PyResult<Option<String>> {
        quoted(self.0, token)
    }

    fn canonicalize(&mut self, value: &str) -> PyResult<String> {
        canonicalize(self.0, value)
    }
}

/// `_parse_marker` + `_normalize_extra_values` of a `str` (syntax errors as `SyntaxError`).
fn parse_marker_str(
    py: Python<'_>,
    source: &Bound<'_, PyString>,
) -> PyResult<Result<List, SyntaxError>> {
    let cps = code_points(source)?;
    match core::parse_marker(&cps, unicode_version(py)?, &mut PyHost(py)) {
        Ok(l) => Ok(Ok(l)),
        Err(core::Error::Syntax(e)) => Ok(Err(e)),
        Err(core::Error::Host(e)) => Err(e),
    }
}

/// `parse_marker` + `_normalize_extra_values`, raising `InvalidMarker`.
fn parse_marker(py: Python<'_>, marker: &Bound<'_, PyAny>) -> PyResult<List> {
    let c = config(py)?;
    let Ok(source) = marker.cast::<PyString>() else {
        // packaging's tokenizer runs a regex on it first: TypeError for a non-str.
        c.ws_regex
            .bind(py)
            .call_method1(intern!(py, "match"), (marker, 0))?;
        return Err(PyTypeError::new_err("expected string or bytes-like object"));
    };
    match parse_marker_str(py, source)? {
        Ok(l) => Ok(l),
        Err(e) => Err(syntax_error(py, &e, source, c.invalid_marker.bind(py))),
    }
}

// ---------------------------------------------------------------------------
// Marker
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct MarkerData {
    list: List,
    text: OnceLock<Result<String, core::QuoteError>>,
    hash: HashCache,
}

impl MarkerData {
    fn new(list: List) -> Self {
        Self {
            list,
            text: OnceLock::new(),
            hash: HashCache::new(),
        }
    }

    fn text(&self) -> PyResult<&str> {
        match self
            .text
            .get_or_init(|| core::format_marker(&self.list, true))
        {
            Ok(s) => Ok(s),
            Err(core::QuoteError) => Err(PyValueError::new_err(
                "Cannot serialize marker value containing both quote characters",
            )),
        }
    }
}

/// An environment marker, as `packaging.markers.Marker` 26.3.
#[pyclass(frozen, subclass, module = "pyroid.packaging.markers", name = "Marker")]
#[derive(Debug)]
pub struct PyMarker {
    data: StateCell<MarkerData>,
}

impl PyMarker {
    fn data(&self) -> PyResult<StateRef<'_, MarkerData>> {
        // packaging sets `_markers = []` before parsing, so a failed construction still has it.
        self.data
            .get()
            .ok_or_else(|| PyAttributeError::new_err("'Marker' object has no attribute '_markers'"))
    }

    pub fn from_list(list: List) -> Self {
        Self {
            data: StateCell::with(MarkerData::new(list)),
        }
    }

    fn new_of<'py>(cls: &Bound<'py, PyType>, list: List) -> PyResult<Bound<'py, PyAny>> {
        let py = cls.py();
        if cls.is(py.get_type::<PyMarker>()) {
            return Ok(Bound::new(py, PyMarker::from_list(list))?.into_any());
        }
        let obj = cls.call_method1(intern!(py, "__new__"), (cls,))?;
        obj.cast::<PyMarker>()?
            .get()
            .data
            .set(MarkerData::new(list));
        Ok(obj)
    }

    /// The marker's tree (shared: O(1)).
    fn list(&self) -> PyResult<List> {
        Ok(self.data()?.list.clone())
    }
}

/// packaging's `_markers` objects (its `Variable`/`Value`/`Op` nodes) for `list`.
///
/// Built with a work list, whatever the depth; a group the tree shares (`m & m`) becomes one
/// Python list held in each place, as packaging's `__and__` leaves it.
fn to_py_list<'py>(py: Python<'py>, list: &List) -> PyResult<Bound<'py, PyList>> {
    let cfg = config(py)?;
    let node = |v: &MarkerVar| -> PyResult<Bound<'py, PyAny>> {
        match v {
            MarkerVar::Variable(x) => cfg.node_variable.bind(py).call1((x.as_str(),)),
            MarkerVar::Value(x) => cfg.node_value.bind(py).call1((x.as_str(),)),
        }
    };
    let mut done: HashMap<usize, Bound<'py, PyList>> = HashMap::new();
    // (the group, the next element, the Python list so far)
    let mut work: Vec<(&List, usize, Bound<'py, PyList>)> = vec![(list, 0, PyList::empty(py))];
    loop {
        let Some((src, next, out)) = work.last_mut() else {
            unreachable!("the loop returns when the outermost group is done")
        };
        let src: &List = src;
        let Some(elem) = src.get(*next) else {
            let Some((src, _, out)) = work.pop() else {
                unreachable!()
            };
            match work.last() {
                None => return Ok(out),
                Some((_, _, parent)) => parent.append(&out)?,
            }
            done.insert(src.id(), out);
            continue;
        };
        *next += 1;
        match elem {
            Elem::List(sub) => {
                if let Some(shared) = done.get(&sub.id()) {
                    out.append(shared)?;
                } else {
                    work.push((sub, 0, PyList::empty(py)));
                }
            }
            Elem::And => out.append("and")?,
            Elem::Or => out.append("or")?,
            Elem::Item(lhs, op, rhs) => {
                let item = PyTuple::new(
                    py,
                    [
                        node(lhs)?,
                        cfg.node_op.bind(py).call1((op.as_str(),))?,
                        node(rhs)?,
                    ],
                )?;
                out.append(item)?;
            }
        }
    }
}

/// One element of packaging's `_markers` that is not a group.
fn elem_from_py(py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<Elem> {
    if let Ok(s) = item.cast::<PyString>() {
        return match &*s.to_cow()? {
            "and" => Ok(Elem::And),
            "or" => Ok(Elem::Or),
            other => Err(PyTypeError::new_err(format!("Unexpected marker {other:?}"))),
        };
    }
    let Ok(t) = item.cast::<PyTuple>() else {
        return Err(PyTypeError::new_err(format!(
            "Unexpected marker {}",
            item.repr()?
        )));
    };
    if t.len() != 3 {
        return Err(PyTypeError::new_err(format!(
            "Unexpected marker {}",
            item.repr()?
        )));
    }
    let var = |n: &Bound<'_, PyAny>| -> PyResult<MarkerVar> {
        let value: String = n.getattr(intern!(py, "value"))?.extract()?;
        let kind = n.get_type().name()?.to_string();
        match kind.as_str() {
            "Variable" => Ok(MarkerVar::Variable(value)),
            "Value" => Ok(MarkerVar::Value(value)),
            _ => Err(PyTypeError::new_err(format!(
                "Unexpected marker node {}",
                n.repr()?
            ))),
        }
    };
    let op: String = t.get_item(1)?.getattr(intern!(py, "value"))?.extract()?;
    Ok(Elem::Item(var(&t.get_item(0)?)?, op, var(&t.get_item(2)?)?))
}

/// The native form of packaging's `_markers` objects.
///
/// Read with a work list, whatever the depth. A Python list held in several places becomes
/// one shared group (so packaging's `m & m` trees stay small); a list that holds itself raises
/// `RecursionError`, as packaging's own walks of it do.
fn from_py_list(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<List> {
    fn snapshot<'py>(o: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        match o.cast::<PyList>() {
            // One atomic copy (another thread may change the list meanwhile).
            Ok(l) => Ok(l.to_tuple().iter().collect()),
            Err(_) => o.try_iter()?.collect(),
        }
    }
    let mut done: HashMap<usize, List> = HashMap::new();
    let mut open: HashSet<usize> = HashSet::new();
    let root = obj.as_ptr().addr();
    open.insert(root);
    // (the Python list's id, its items, the next item, the elements so far)
    let mut work = vec![(root, snapshot(obj)?, 0usize, Vec::new())];
    loop {
        let Some((_, items, i, out)) = work.last_mut() else {
            unreachable!("the loop returns when the outermost list is done")
        };
        let Some(item) = items.get(*i).cloned() else {
            let Some((id, _, _, out)) = work.pop() else {
                unreachable!()
            };
            open.remove(&id);
            let list = List::new(out);
            match work.last_mut() {
                None => return Ok(list),
                Some((_, _, _, parent)) => parent.push(Elem::List(list.clone())),
            }
            done.insert(id, list);
            continue;
        };
        *i += 1;
        if item.cast::<PyList>().is_err() {
            out.push(elem_from_py(py, &item)?);
            continue;
        }
        let id = item.as_ptr().addr();
        if let Some(l) = done.get(&id) {
            out.push(Elem::List(l.clone()));
        } else if open.contains(&id) {
            return Err(PyRecursionError::new_err(
                "maximum recursion depth exceeded (a marker list holds itself)",
            ));
        } else {
            open.insert(id);
            let sub = snapshot(&item)?;
            work.push((id, sub, 0, Vec::new()));
        }
    }
}

fn is_marker(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<bool> {
    if x.cast::<PyMarker>().is_ok() {
        return Ok(true);
    }
    match &config(py)?.packaging_marker_any {
        Some(cls) => x.is_instance(cls.bind(py)),
        None => Ok(false),
    }
}

/// The marker list of a pyroid or packaging `Marker`.
fn marker_list(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<List> {
    if let Ok(m) = x.cast::<PyMarker>() {
        return m.get().list();
    }
    from_py_list(py, &x.getattr(intern!(py, "_markers"))?)
}

#[pymethods]
impl PyMarker {
    #[new]
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn py_new(
        cls: &Bound<'_, PyType>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let py = cls.py();
        if cls.is(py.get_type::<PyMarker>()) {
            let marker = match (args.len(), kwargs) {
                (1, None) => Some(args.get_item(0)?),
                (0, Some(kw)) if kw.len() == 1 => kw.get_item("marker")?,
                _ => None,
            };
            if let Some(m) = marker {
                return Ok(PyMarker::from_list(parse_marker(py, &m)?));
            }
        }
        Ok(PyMarker {
            data: StateCell::new(),
        })
    }

    #[pyo3(name = "__init__", signature = (marker))]
    fn py_init(&self, py: Python<'_>, marker: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.data.is_set() {
            return Ok(());
        }
        // packaging sets `_markers = []` first, so a failed __init__ leaves an empty marker.
        match parse_marker(py, marker) {
            Ok(list) => self.data.set(MarkerData::new(list)),
            Err(e) => {
                self.data.set(MarkerData::new(List::default()));
                return Err(e);
            }
        }
        Ok(())
    }

    /// Create a Marker instance from a pre-parsed marker tree.
    #[classmethod]
    fn _from_markers<'py>(
        cls: &Bound<'py, PyType>,
        markers: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        PyMarker::new_of(cls, from_py_list(cls.py(), markers)?)
    }

    #[getter]
    fn _markers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_py_list(py, &self.data()?.list)
    }

    #[setter(_markers)]
    fn set_markers(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.data.set(MarkerData::new(from_py_list(py, value)?));
        Ok(())
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(self.data()?.text()?.to_owned())
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().name()?;
        Ok(format!("<{name}({})>", slf.str()?.repr()?))
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        let d = self.data()?;
        if let Some(h) = d.hash.get() {
            return Ok(h);
        }
        let text = PyString::new(py, d.text()?);
        let h = if config(py)?.marker_hash_named {
            // As packaging 24.2 and 25.0 hash their markers.
            PyTuple::new(py, [PyString::new(py, "Marker"), text])?.hash()?
        } else {
            text.hash()?
        };
        d.hash.set(h);
        Ok(h)
    }

    fn __richcmp__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
        op: CompareOp,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) || !is_marker(py, other)? {
            return Ok(py.NotImplemented().into_bound(py));
        }
        let eq = slf.str()?.as_any().eq(other.str()?)?;
        Ok(
            PyBool::new(py, if matches!(op, CompareOp::Eq) { eq } else { !eq })
                .to_owned()
                .into_any(),
        )
    }

    fn __getstate__(&self) -> PyResult<String> {
        self.__str__()
    }

    fn __setstate__(&self, py: Python<'_>, state: &Bound<'_, PyAny>) -> PyResult<()> {
        let fail = || -> PyResult<PyErr> {
            Ok(PyTypeError::new_err(format!(
                "Cannot restore Marker from {}",
                state.repr()?
            )))
        };
        if let Ok(s) = state.cast::<PyString>() {
            let Ok(list) = parse_marker_str(py, s)? else {
                return Err(fail()?);
            };
            self.data.set(MarkerData::new(list));
            return Ok(());
        }
        let from_dict = |d: &Bound<'_, PyDict>| -> PyResult<Option<List>> {
            match d.get_item("_markers")? {
                Some(m) if m.cast::<PyList>().is_ok() => Ok(Some(from_py_list(py, &m)?)),
                _ => Ok(None),
            }
        };
        let found = if let Ok(d) = state.cast::<PyDict>() {
            from_dict(d)?
        } else if let Ok(t) = state.cast::<PyTuple>() {
            if t.len() == 2 {
                match t.get_item(1)?.cast_into::<PyDict>() {
                    Ok(d) => from_dict(&d)?,
                    Err(_) => None,
                }
            } else {
                None
            }
        } else {
            None
        };
        match found {
            Some(list) => {
                self.data.set(MarkerData::new(list));
                Ok(())
            }
            None => Err(fail()?),
        }
    }

    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, _protocol: i64) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let state = slf.get().__getstate__()?;
        PyTuple::new(
            py,
            [
                config(py)?.restore.bind(py).clone(),
                PyTuple::new(
                    py,
                    [
                        slf.get_type().into_any(),
                        PyString::new(py, &state).into_any(),
                    ],
                )?
                .into_any(),
            ],
        )
    }

    fn __and__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        combine(slf, other, Elem::And)
    }

    fn __or__<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        combine(slf, other, Elem::Or)
    }

    #[getter]
    fn __class__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if claims() {
            if let Some(cls) = &config(py)?.packaging_marker {
                return Ok(cls.clone_ref(py).into_any());
            }
        }
        Ok(slf.get_type().into_any().unbind())
    }

    /// Evaluate a marker.
    #[pyo3(signature = (environment = None, context = ContextArg(None)))]
    fn evaluate(
        &self,
        py: Python<'_>,
        environment: Option<&Bound<'_, PyAny>>,
        context: ContextArg,
    ) -> PyResult<bool> {
        let d = self.data()?;
        let ctx = context.0.as_ref().map(|c| c.bind(py).clone());
        let env = Env::new(py, environment.filter(|e| !e.is_none()), ctx.as_ref())?;
        evaluate(py, &d.list, &env)
    }
}

/// `evaluate`'s `context`: `ContextArg(None)` when not passed (packaging's default,
/// `"metadata"`), so that an explicit `None` is "another context", as in packaging.
#[derive(Debug)]
pub struct ContextArg(Option<Py<PyAny>>);

impl<'a, 'py> FromPyObject<'a, 'py> for ContextArg {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(ContextArg(Some(obj.to_owned().unbind())))
    }
}

fn combine<'py>(
    slf: &Bound<'py, PyMarker>,
    other: &Bound<'py, PyAny>,
    op: Elem,
) -> PyResult<Bound<'py, PyAny>> {
    let py = slf.py();
    if !is_marker(py, other)? {
        return Ok(py.NotImplemented().into_bound(py));
    }
    // packaging's `[self._markers, op, other._markers]`: both groups shared, not copied.
    let list = List::new(vec![
        Elem::List(slf.get().list()?),
        op,
        Elem::List(marker_list(py, other)?),
    ]);
    PyMarker::new_of(&slf.get_type(), list)
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// The marker environment: the merged dict packaging builds, looked up without building it when
/// the caller's environment is a plain dict.
struct Env<'py> {
    py: Python<'py>,
    /// `_cached_default_environment()` (shared; never changed).
    default: Bound<'py, PyAny>,
    /// The caller's dict (overrides), or the merged dict for another mapping.
    given: Option<Bound<'py, PyDict>>,
    /// `given` is the whole merged environment (built in Python for a non-dict mapping).
    merged: bool,
    context: Context,
    /// `extra` as packaging leaves it before evaluating (canonicalised when an environment is
    /// given); `None`: not in the environment.
    extra: Option<Bound<'py, PyAny>>,
    /// `python_full_version` after `_repair_python_full_version`.
    python_full_version: Bound<'py, PyAny>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Context {
    Metadata,
    LockFile,
    Other,
}

impl<'py> Env<'py> {
    fn new(
        py: Python<'py>,
        environment: Option<&Bound<'py, PyAny>>,
        context: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Self> {
        let c = config(py)?;
        let default = c.cached_default_environment.bind(py).call0()?;
        let context = match context {
            None => Context::Metadata,
            Some(ctx) => {
                if ctx.eq("lock_file")? {
                    Context::LockFile
                } else if ctx.eq("metadata")? {
                    Context::Metadata
                } else {
                    Context::Other
                }
            }
        };
        let (given, merged) = match environment {
            None => (None, false),
            Some(e) if e.is_exact_instance_of::<PyDict>() => {
                (Some(e.cast::<PyDict>()?.clone()), false)
            }
            Some(e) => {
                // packaging: current = dict(default) (+ context keys); current |= environment.
                let merged = default.cast::<PyDict>()?.copy()?;
                add_context(py, &merged, context)?;
                merged.call_method1(intern!(py, "__ior__"), (e,))?;
                (Some(merged), true)
            }
        };
        let mut env = Self {
            py,
            default,
            given,
            merged,
            context,
            extra: None,
            python_full_version: py.None().into_bound(py),
        };
        // packaging prepares these two before it evaluates anything, so their errors come
        // first, whatever the marker reads. With an environment given, a truthy `extra` is
        // canonicalised and a falsy one (None) becomes "".
        env.extra = match env.raw("extra")? {
            Some(v) if environment.is_some() => Some(if !v.is_truthy()? {
                PyString::new(py, "").into_any()
            } else if let Ok(s) = v.cast::<PyString>() {
                PyString::new(py, &canonicalize(py, &s.to_cow()?)?).into_any()
            } else {
                c.canonicalize_name.bind(py).call1((v,))?
            }),
            other => other,
        };
        // _repair_python_full_version: env["python_full_version"], with "local" after a "+".
        let Some(v) = env.raw("python_full_version")? else {
            return Err(PyKeyError::new_err("python_full_version"));
        };
        env.python_full_version = if let Ok(s) = v.cast::<PyString>() {
            let t = s.to_cow()?;
            if t.ends_with('+') {
                PyString::new(py, &format!("{t}local")).into_any()
            } else {
                v
            }
        } else if v
            .call_method1(intern!(py, "endswith"), ("+",))?
            .is_truthy()?
        {
            v.add("local")?
        } else {
            v
        };
        Ok(env)
    }

    /// `environment[key]` of the merged environment (`None`: missing).
    fn get(&self, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        match key {
            "extra" => Ok(self.extra.clone()),
            "python_full_version" => Ok(Some(self.python_full_version.clone())),
            _ => self.raw(key),
        }
    }

    fn raw(&self, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(g) = &self.given {
            if let Some(v) = g.get_item(key)? {
                return Ok(Some(v));
            }
            if self.merged {
                return Ok(None);
            }
        }
        match (key, self.context) {
            ("extra", Context::Metadata) => return Ok(Some(PyString::new(self.py, "").into_any())),
            ("extras" | "dependency_groups", Context::LockFile) => {
                return Ok(Some(PyFrozenSet::empty(self.py)?.into_any()));
            }
            _ => {}
        }
        if let Ok(d) = self.default.cast::<PyDict>() {
            return d.get_item(key);
        }
        self.default.get_item(key).map(Some).or_else(|e| {
            if e.is_instance_of::<pyo3::exceptions::PyKeyError>(self.py) {
                Ok(None)
            } else {
                Err(e)
            }
        })
    }
}

fn add_context(py: Python<'_>, env: &Bound<'_, PyDict>, context: Context) -> PyResult<()> {
    match context {
        Context::LockFile => {
            env.set_item("extras", PyFrozenSet::empty(py)?)?;
            env.set_item("dependency_groups", PyFrozenSet::empty(py)?)?;
        }
        Context::Metadata => env.set_item("extra", "")?,
        Context::Other => {}
    }
    Ok(())
}

const REQUIRING_VERSION: &[&str] = &[
    "implementation_version",
    "platform_release",
    "python_full_version",
    "python_version",
];
const ALLOWING_SET: &[&str] = &["extras", "dependency_groups"];

/// `_evaluate_markers(markers, environment)`.
///
/// packaging recurses into each group and evaluates every item (no short circuit), so errors
/// surface in the same order; this does the same with a work list.
fn evaluate(py: Python<'_>, list: &List, env: &Env<'_>) -> PyResult<bool> {
    /// A group being evaluated: `any(all(g) for g in groups)` so far, as two flags.
    struct Group<'a> {
        items: &'a [Elem],
        next: usize,
        /// Some finished `or` operand was all true.
        any: bool,
        /// Every item of the current `or` operand so far is true.
        all: bool,
    }
    fn group(items: &[Elem]) -> Group<'_> {
        Group {
            items,
            next: 0,
            any: false,
            all: true,
        }
    }
    let mut work = vec![group(list)];
    loop {
        let Some(top) = work.last_mut() else {
            unreachable!("the loop returns when the outermost group is done")
        };
        let items: &[Elem] = top.items;
        let Some(e) = items.get(top.next) else {
            let result = top.any || top.all;
            work.pop();
            match work.last_mut() {
                None => return Ok(result),
                Some(parent) => parent.all &= result,
            }
            continue;
        };
        top.next += 1;
        match e {
            Elem::List(inner) => work.push(group(inner)),
            Elem::Item(lhs, op, rhs) => {
                let v = evaluate_item(py, lhs, op, rhs, env)?;
                top.all &= v;
            }
            Elem::Or => {
                top.any |= top.all;
                top.all = true;
            }
            Elem::And => {}
        }
    }
}

fn evaluate_item(
    py: Python<'_>,
    lhs: &MarkerVar,
    op: &str,
    rhs: &MarkerVar,
    env: &Env<'_>,
) -> PyResult<bool> {
    let (key, lhs_value, rhs_value) = match lhs {
        MarkerVar::Variable(k) => (
            k.as_str(),
            lookup(py, env, k)?,
            PyString::new(py, rhs.value()).into_any(),
        ),
        MarkerVar::Value(v) => {
            let k = rhs.value();
            (k, PyString::new(py, v).into_any(), lookup(py, env, k)?)
        }
    };
    if lhs_value.cast::<PyString>().is_err() {
        let c = config(py)?;
        let key_repr = PyString::new(py, key).repr()?;
        let op_repr = PyString::new(py, op).repr()?;
        let msg = format!(
            "Set-valued marker {key_repr} can only be used with the membership form (e.g. \"<name>\" in {key}); it cannot appear on the left-hand side of {op_repr}."
        );
        return Err(PyErr::from_value(
            c.undefined_comparison.bind(py).call1((msg,))?,
        ));
    }
    let (l, r) = normalize(py, &lhs_value, &rhs_value, key)?;
    eval_op(py, &l, op, &r, key)
}

fn lookup<'py>(py: Python<'py>, env: &Env<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    match env.get(key)? {
        Some(v) => Ok(v),
        None => Err(PyErr::from_value(
            config(py)?
                .undefined_environment_name
                .bind(py)
                .call1((key,))?,
        )),
    }
}

fn normalize<'py>(
    py: Python<'py>,
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if key == "extra" || !ALLOWING_SET.contains(&key) {
        return Ok((lhs.clone(), rhs.clone()));
    }
    let canon = |x: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        match x.cast::<PyString>() {
            Ok(s) => Ok(PyString::new(py, &canonicalize(py, &s.to_cow()?)?).into_any()),
            Err(_) => config(py)?.canonicalize_name.bind(py).call1((x,)),
        }
    };
    let l = canon(lhs)?;
    if rhs.cast::<PyString>().is_ok() {
        return Ok((l, canon(rhs)?));
    }
    let set = PySet::empty(py)?;
    for v in rhs.try_iter()? {
        set.add(canon(&v?)?)?;
    }
    Ok((l, set.into_any()))
}

fn eval_op(
    py: Python<'_>,
    lhs: &Bound<'_, PyAny>,
    op: &str,
    rhs: &Bound<'_, PyAny>,
    key: &str,
) -> PyResult<bool> {
    if REQUIRING_VERSION.contains(&key) {
        // Specifier(f"{op}{rhs}") if it is one, then contains(lhs, prereleases=True).
        let text = format!("{op}{}", rhs.str()?.to_cow()?);
        if spec_core::is_valid(&text) {
            let (o, version) = spec_core::split(&text);
            if o == Op::Arbitrary {
                let spec = Bound::new(py, super::specifiers::PySpecifier::from_text(py, &text)?)?;
                let t = PyBool::new(py, true).to_owned().into_any();
                return spec
                    .call_method1(intern!(py, "contains"), (lhs, t))?
                    .is_truthy();
            }
            let Some(parsed) = coerce(py, lhs)? else {
                return Ok(false);
            };
            let spec = super::specifiers::PySpecifier::from_text(py, &text)?;
            let _ = version;
            return spec.contains_version(py, &parsed);
        }
    }
    match op {
        "in" => rhs.contains(lhs),
        "not in" => Ok(!rhs.contains(lhs)?),
        "<" | ">" => Ok(false),
        "<=" | "==" | ">=" => lhs.eq(rhs),
        "!=" => lhs.ne(rhs),
        _ => {
            let c = config(py)?;
            let op_repr = c.node_op.bind(py).call1((op,))?.repr()?;
            let msg = format!(
                "Undefined {op_repr} on {} and {}.",
                lhs.repr()?,
                rhs.repr()?
            );
            Err(PyErr::from_value(
                c.undefined_comparison.bind(py).call1((msg,))?,
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// Requirements
// ---------------------------------------------------------------------------

/// `parse_requirement(s)` with packaging's errors: `(name, url, extras, specifier, marker)`.
#[pyfunction]
fn parse_requirement<'py>(
    py: Python<'py>,
    requirement: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let c = config(py)?;
    let Ok(source) = requirement.cast::<PyString>() else {
        c.ws_regex
            .bind(py)
            .call_method1(intern!(py, "match"), (requirement, 0))?;
        return Err(PyTypeError::new_err("expected string or bytes-like object"));
    };
    let cps = code_points(source)?;
    let parsed = match core::parse_requirement(&cps, unicode_version(py)?, &mut PyHost(py)) {
        Ok(p) => p,
        Err(core::Error::Syntax(e)) => {
            return Err(syntax_error(py, &e, source, c.invalid_requirement.bind(py)));
        }
        Err(core::Error::Host(e)) => return Err(e),
    };
    let name = py_str(py, &cps[parsed.name.0..parsed.name.1])?;
    let url = match parsed.url {
        Some((a, b)) => py_str(py, &cps[a..b])?.into_any(),
        None => PyString::new(py, "").into_any(),
    };
    let extras = PyList::empty(py);
    for (a, b) in &parsed.extras {
        extras.append(py_str(py, &cps[*a..*b])?)?;
    }
    let mut spec: Vec<u32> = Vec::new();
    for (a, b) in &parsed.specifier {
        spec.extend_from_slice(&cps[*a..*b]);
    }
    let marker = match parsed.marker {
        Some(list) => Bound::new(py, PyMarker::from_list(list))?.into_any(),
        None => py.None().into_bound(py),
    };
    PyTuple::new(
        py,
        [
            name.into_any(),
            url,
            extras.into_any(),
            py_str(py, &spec)?.into_any(),
            marker,
        ],
    )
}

/// `canonicalize_name(name)` for a `str`.
#[pyfunction]
#[pyo3(signature = (name, /))]
fn canonicalize_str<'py>(
    py: Python<'py>,
    name: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(t) = name.to_cow() {
        return Ok(PyString::new(py, &canonicalize(py, &t)?).into_any());
    }
    // Lone surrogates: the same steps in Python.
    let low = name.call_method0(intern!(py, "lower"))?;
    let mut s = low
        .call_method1(intern!(py, "replace"), ("_", "-"))?
        .call_method1(intern!(py, "replace"), (".", "-"))?;
    while s.contains("--")? {
        s = s.call_method1(intern!(py, "replace"), ("--", "-"))?;
    }
    Ok(s)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMarker>()?;
    m.add_function(wrap_pyfunction!(parse_requirement, m)?)?;
    m.add_function(wrap_pyfunction!(canonicalize_str, m)?)?;
    Ok(())
}
