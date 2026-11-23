// Derived from python-dateutil 4.0.0 (BSD-3-Clause), © Gustavo Niemeyer, Paul Ganssle and
// dateutil contributors; modified.
//! `pyroid._pyroid.dateutil`: native entry points behind `pyroid.dateutil.parser`.
//!
//! The Python module (`python/pyroid/dateutil/parser.py`) owns the classes; this module runs
//! the tokenizer and the parser's resolution logic in Rust ([`pyroid_core::dateutil`]) and
//! builds the result on the calling thread, mirroring dateutil's `parser.parse`:
//!
//! - the common results (a `datetime` from a plain `datetime` default or today, naive, UTC or
//!   a `tzoffset`) are built here;
//! - everything else (`tzinfos`, a time zone name in `time.tzname`, a `default` of another type,
//!   values `datetime` rejects) runs the rest of dateutil's `parse` in Python
//!   (`parser._finish`), so the objects and messages are dateutil's and the interpreter's own;
//! - a parser or parserinfo subclass that overrides a method, and the inputs the kernel cannot
//!   reproduce, run dateutil's own algorithm (`parser._reference_parse`).

use std::cell::{Cell, OnceCell, RefCell};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, PoisonError};

use pyo3::exceptions::{PyTypeError, PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::{MutexExt, PyOnceLock};
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString, PyTuple, PyType};
use pyroid_core::dateutil::info::{Table, Tables};
use pyroid_core::dateutil::isoparser::{self as iso, Iso, IsoTz, Mode};
use pyroid_core::dateutil::{
    self as kernel, Int, ParseOptions, Parsed, ParserInfo, Props, Res, Settings,
};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::json_repair::chars;
use pyroid_core::par;
use rustc_hash::FxHashMap;

use crate::input::{Threads, threads as pool_threads};

// ---------------------------------------------------------------------------
// Set-up
// ---------------------------------------------------------------------------

/// The interpreter's Unicode tables (set once by `parser.py` at import).
static PROPS: OnceLock<Props> = OnceLock::new();

fn props() -> Props {
    *PROPS.get_or_init(|| Props::new(UnicodeVersion::LATEST))
}

/// The Unicode versions the kernel has tables for.
#[pyfunction]
fn supported_unicode_versions() -> Vec<&'static str> {
    [
        UnicodeVersion::V14_0,
        UnicodeVersion::V15_0,
        UnicodeVersion::V15_1,
        UnicodeVersion::V16_0,
        UnicodeVersion::V17_0,
    ]
    .iter()
    .map(|v| v.as_str())
    .collect()
}

/// Select the Unicode tables (`unicodedata.unidata_version`); once per process.
#[pyfunction]
fn _set_unicode_version(version: &str) -> PyResult<()> {
    let v = UnicodeVersion::from_unidata_version(version)
        .ok_or_else(|| PyValueError::new_err(format!("no tables for Unicode {version}")))?;
    let _ = PROPS.set(Props::new(v));
    Ok(())
}

/// The Python objects this module calls, from `parser.py`'s `_install`.
/// (name, pyroid's parserinfo attribute, dateutil's) for one parserinfo hook.
type InfoMethod = (Py<PyString>, Py<PyAny>, Option<Py<PyAny>>);

struct Hooks {
    module: Py<PyDict>,
    parser: Py<PyType>,
    parserinfo: Py<PyType>,
    /// pyroid's change-reporting table dict (`_reference._Table`).
    table_type: Py<PyType>,
    /// pyroid's parserinfo `__setattr__` and `__delattr__` (they report changes).
    tracked_setattr: (Py<PyAny>, Py<PyAny>),
    dateutil_parserinfo: Option<Py<PyType>>,
    isoparser: Py<PyType>,
    default_isoparser: Py<PyAny>,
    iso_module: Py<PyDict>,
    parser_error: Py<PyAny>,
    unknown_tz: Py<PyAny>,
    utc: Py<PyAny>,
    tzoffset: Py<PyAny>,
    tzlocal: Py<PyAny>,
    finish: Py<PyAny>,
    make_result: Py<PyAny>,
    reference_parse: Py<PyAny>,
    reference_private_parse: Py<PyAny>,
    reference_isoparse: Py<PyAny>,
    read_stream: Py<PyAny>,
    replay_read: Py<PyAny>,
    class_dicts: Py<PyAny>,
    datetime_module: Py<PyAny>,
    datetime_type: Py<PyAny>,
    time_module: Py<PyAny>,
    getcontext: Py<PyAny>,
    /// (name, pyroid's parser attribute) for each hook.
    parser_methods: Vec<(Py<PyString>, Py<PyAny>)>,
    /// The hooks `_parse` itself calls.
    private_methods: Vec<(Py<PyString>, Py<PyAny>)>,
    /// (name, pyroid's parserinfo attribute, dateutil's) for each hook.
    info_methods: Vec<InfoMethod>,
    iso_methods: Vec<(Py<PyString>, Py<PyAny>)>,
}

static HOOKS: PyOnceLock<Hooks> = PyOnceLock::new();

fn hooks(py: Python<'_>) -> PyResult<&Hooks> {
    HOOKS.get(py).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("pyroid.dateutil.parser is not initialised")
    })
}

fn item<'py>(d: &Bound<'py, PyDict>, k: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(k)?
        .ok_or_else(|| PyValueError::new_err(format!("_install: missing {k}")))
}

fn attrs_of(
    cls: &Bound<'_, PyAny>,
    names: &Bound<'_, PyAny>,
) -> PyResult<Vec<(Py<PyString>, Py<PyAny>)>> {
    let mut out = Vec::new();
    for n in names.try_iter()? {
        let n = n?.cast_into::<PyString>()?;
        let v = cls.getattr(&n)?;
        out.push((n.unbind(), v.unbind()));
    }
    Ok(out)
}

/// Called once by `parser.py` with the objects in [`Hooks`].
#[pyfunction]
fn _install(d: &Bound<'_, PyDict>) -> PyResult<()> {
    let py = d.py();
    let parser = item(d, "parser")?;
    let parserinfo = item(d, "parserinfo")?;
    let isoparser = item(d, "isoparser")?;
    let du_info = item(d, "dateutil_parserinfo")?;
    let tz = item(d, "tz")?;
    let datetime_module = item(d, "datetime_module")?;
    let parser_hooks = attrs_of(&parser, &item(d, "parser_hooks")?)?;
    let private_hooks = attrs_of(&parser, &item(d, "private_hooks")?)?;
    let iso_hooks = attrs_of(&isoparser, &item(d, "iso_hooks")?)?;
    let mut info_hooks = Vec::new();
    for n in item(d, "info_hooks")?.try_iter()? {
        let n = n?.cast_into::<PyString>()?;
        let ours = parserinfo.getattr(&n)?.unbind();
        let theirs = if du_info.is_none() {
            None
        } else {
            du_info.getattr(&n).ok().map(Bound::unbind)
        };
        info_hooks.push((n.unbind(), ours, theirs));
    }
    let module = item(d, "module")?.cast_into::<PyDict>()?;
    let iso_module = py.import("pyroid.dateutil._isoreference")?.dict().unbind();
    let h = Hooks {
        default_isoparser: item(&module, "DEFAULT_ISOPARSER")?.unbind(),
        module: module.unbind(),
        parser: parser.cast_into::<PyType>()?.unbind(),
        table_type: py
            .import("pyroid.dateutil._reference")?
            .getattr("_Table")?
            .cast_into::<PyType>()?
            .unbind(),
        tracked_setattr: (
            parserinfo.getattr("__setattr__")?.unbind(),
            parserinfo.getattr("__delattr__")?.unbind(),
        ),
        parserinfo: parserinfo.cast_into::<PyType>()?.unbind(),
        dateutil_parserinfo: if du_info.is_none() {
            None
        } else {
            Some(du_info.cast_into::<PyType>()?.unbind())
        },
        isoparser: isoparser.cast_into::<PyType>()?.unbind(),
        iso_module,
        parser_error: item(d, "ParserError")?.unbind(),
        unknown_tz: item(d, "UnknownTimezoneWarning")?.unbind(),
        utc: tz.getattr("UTC")?.unbind(),
        tzoffset: tz.getattr("tzoffset")?.unbind(),
        tzlocal: tz.getattr("tzlocal")?.unbind(),
        finish: item(d, "finish")?.unbind(),
        make_result: item(d, "make_result")?.unbind(),
        reference_parse: item(d, "reference_parse")?.unbind(),
        reference_private_parse: item(d, "reference_private_parse")?.unbind(),
        reference_isoparse: item(d, "reference_isoparse")?.unbind(),
        read_stream: item(d, "read_stream")?.unbind(),
        replay_read: item(d, "replay_read")?.unbind(),
        class_dicts: item(d, "class_dicts")?.unbind(),
        datetime_type: datetime_module.getattr("datetime")?.unbind(),
        datetime_module: datetime_module.unbind(),
        time_module: py.import("time")?.into_any().unbind(),
        getcontext: py.import("decimal")?.getattr("getcontext")?.unbind(),
        parser_methods: parser_hooks,
        private_methods: private_hooks,
        info_methods: info_hooks,
        iso_methods: iso_hooks,
    };
    let _ = HOOKS.set(py, h);
    Ok(())
}

// ---------------------------------------------------------------------------
// Subclass checks and the compiled parserinfo
// ---------------------------------------------------------------------------

/// The attribute dicts a [`TypeEntry`] watches, with their sizes when it was made.
type ClassDicts = Box<[(Py<PyDict>, usize)]>;

/// What a per-type cache knows about one type.
struct TypeEntry<V> {
    /// Held, so its id stays its own.
    _ty: Py<PyType>,
    /// The attribute dicts of the type and its bases that are not pyroid's or dateutil's own
    /// classes (`parser._class_dicts`), and their sizes when `verdict` was found.
    dicts: ClassDicts,
    verdict: V,
}

impl<V: Clone> TypeEntry<V> {
    /// The verdict, unless one of the dicts changed size since: an attribute was added to or
    /// deleted from the class or a base (`PI.convertyear = f` after a first parse), so the
    /// caller checks the class again.
    fn verdict(&self, py: Python<'_>) -> Option<V> {
        self.dicts
            .iter()
            .all(|(d, n)| d.bind(py).len() == *n)
            .then(|| self.verdict.clone())
    }
}

/// The dicts a [`TypeEntry`] for `ty` watches, with their sizes now.
fn class_dicts(h: &Hooks, ty: &Bound<'_, PyType>) -> PyResult<ClassDicts> {
    let py = ty.py();
    let dicts = h.class_dicts.bind(py).call1((ty,))?;
    let mut out = Vec::new();
    for d in dicts.try_iter()? {
        let d = d?.cast_into::<PyDict>()?;
        let n = d.len();
        out.push((d.unbind(), n));
    }
    Ok(out.into_boxed_slice())
}

/// Per type: the hook it overrides (`None`: none). Per thread (no lock on the hot path).
type TypeCache = RefCell<FxHashMap<usize, TypeEntry<Option<String>>>>;

thread_local! {
    static PARSER_TYPES: TypeCache = RefCell::default();
    static PRIVATE_TYPES: TypeCache = RefCell::default();
    static INFO_TYPES: TypeCache = RefCell::default();
    static ISO_TYPES: TypeCache = RefCell::default();
}

/// Whether `obj`'s type is `ty`. Compares the type pointer: a new reference to a type that every
/// thread shares (`get_type`) is an atomic write to it on free-threaded builds, which threads
/// parsing at once queue on.
fn is_type(obj: &Bound<'_, PyAny>, ty: &Py<PyType>) -> bool {
    obj.get_type_ptr().cast::<pyo3::ffi::PyObject>() == ty.as_ptr()
}

fn cached_override(
    h: &Hooks,
    cache: &'static std::thread::LocalKey<TypeCache>,
    obj: &Bound<'_, PyAny>,
    check: impl FnOnce(&Bound<'_, PyType>) -> PyResult<Option<String>>,
) -> PyResult<Option<String>> {
    let py = obj.py();
    let key = obj.get_type_ptr() as usize;
    if let Some(r) = cache.with(|c| c.borrow().get(&key).and_then(|e| e.verdict(py))) {
        return Ok(r);
    }
    let ty = obj.get_type();
    // The sizes before the check: a change made while it runs gets the class checked again.
    let dicts = class_dicts(h, &ty)?;
    let r = check(&ty)?;
    let entry = TypeEntry {
        _ty: ty.clone().unbind(),
        dicts,
        verdict: r.clone(),
    };
    let old = cache.with(|c| cache_insert(&mut c.borrow_mut(), 256, key, entry));
    drop(old);
    Ok(r)
}

/// Inserts into a per-thread cache, emptying it first when it holds more than `cap` entries, and
/// returns what it removed. The caller drops that after the `RefCell` borrow ends: dropping a
/// Python object can run any Python code (`__del__`, a type's dealloc), which may parse again and
/// borrow the same cache.
#[must_use]
fn cache_insert<K: std::hash::Hash + Eq, V>(
    c: &mut FxHashMap<K, V>,
    cap: usize,
    key: K,
    value: V,
) -> (FxHashMap<K, V>, Option<V>) {
    let old = if c.len() > cap {
        std::mem::take(c)
    } else {
        FxHashMap::default()
    };
    let replaced = c.insert(key, value);
    (old, replaced)
}

/// The first hook `ty` defines differently from `base` (`None`: none).
fn first_override(
    ty: &Bound<'_, PyType>,
    hooks: &[(Py<PyString>, Py<PyAny>)],
    what: &str,
) -> PyResult<Option<String>> {
    let py = ty.py();
    for (name, base) in hooks {
        let ours = ty.getattr(name.bind(py))?;
        if !ours.is(base.bind(py)) {
            return Ok(Some(format!(
                "a {what} subclass overriding {}",
                name.bind(py)
            )));
        }
    }
    Ok(None)
}

fn parser_override(py: Python<'_>, h: &Hooks, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if is_type(obj, &h.parser) || UPSTREAM.get(py).is_some_and(|u| is_type(obj, &u.parser)) {
        return Ok(None);
    }
    cached_override(h, &PARSER_TYPES, obj, |ty| {
        first_override(ty, &h.parser_methods, "parser")
    })
}

fn private_override(h: &Hooks, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if is_type(obj, &h.parser) {
        return Ok(None);
    }
    cached_override(h, &PRIVATE_TYPES, obj, |ty| {
        first_override(ty, &h.private_methods, "parser")
    })
}

/// Whether `obj` is exactly dateutil's or pyroid's parserinfo.
fn exact_info(h: &Hooks, obj: &Bound<'_, PyAny>) -> bool {
    is_type(obj, &h.parserinfo)
        || h.dateutil_parserinfo
            .as_ref()
            .is_some_and(|d| is_type(obj, d))
}

fn info_override(py: Python<'_>, h: &Hooks, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if exact_info(h, obj) {
        return Ok(None);
    }
    cached_override(h, &INFO_TYPES, obj, |ty| {
        for (name, ours, theirs) in &h.info_methods {
            let Ok(v) = ty.getattr(name.bind(py)) else {
                return Ok(Some(format!("a parserinfo without {}", name.bind(py))));
            };
            if !(v.is(ours.bind(py)) || theirs.as_ref().is_some_and(|t| v.is(t.bind(py)))) {
                return Ok(Some(format!(
                    "a parserinfo subclass overriding {}",
                    name.bind(py)
                )));
            }
        }
        Ok(None)
    })
}

fn iso_override(h: &Hooks, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if is_type(obj, &h.isoparser) {
        return Ok(None);
    }
    cached_override(h, &ISO_TYPES, obj, |ty| {
        first_override(ty, &h.iso_methods, "isoparser")
    })
}

/// A compiled parserinfo, or why the kernel cannot use it.
enum Compiled {
    Ok(Box<ParserInfo>),
    Unsupported(String),
}

/// The parserinfo attributes the tables come from, in order.
const TABLE_ATTRS: [&str; 7] = [
    "_jump",
    "_weekdays",
    "_months",
    "_hms",
    "_ampm",
    "_utczone",
    "_pertain",
];

/// A compiled parserinfo and the objects it was compiled from.
struct InfoEntry {
    info: usize,
    /// The 7 table dicts, `TZOFFSET` and `UTCZONE` (held, so their ids stay theirs).
    objs: [Py<PyAny>; 9],
    lens: [usize; 9],
    compiled: Arc<Compiled>,
}

thread_local! {
    /// The parserinfos this thread used last (no lock on the hot path).
    static INFOS: RefCell<Vec<InfoEntry>> = const { RefCell::new(Vec::new()) };
    /// The last pyroid parserinfo whose tables all report changes ([`INFO_GEN`]).
    static TRACKED: RefCell<Option<TrackedEntry>> = const { RefCell::new(None) };
    /// Per type: whether its instances report changes (pyroid's `__setattr__`).
    static TRACKED_TYPES: RefCell<FxHashMap<usize, TypeEntry<bool>>> = RefCell::default();
}

/// Bumped by pyroid's parserinfo on any change to one (an attribute set, a table changed).
static INFO_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Called by pyroid's parserinfo and its tables when they change.
#[pyfunction]
fn _info_changed() {
    INFO_GEN.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

/// A parserinfo checked at generation `generation`: unchanged while the generation is, apart from its
/// class-level `TZOFFSET` and `UTCZONE` (plain containers, checked each call).
struct TrackedEntry {
    info: Py<PyAny>,
    generation: u64,
    class_objs: [Py<PyAny>; 2],
    class_lens: [usize; 2],
    compiled: Arc<Compiled>,
    settings: Option<Settings>,
}

/// Whether instances of `info`'s type report their changes.
fn tracked_type(h: &Hooks, info: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = info.py();
    let key = info.get_type_ptr() as usize;
    if let Some(t) = TRACKED_TYPES.with(|c| c.borrow().get(&key).and_then(|e| e.verdict(py))) {
        return Ok(t);
    }
    let ty = info.get_type();
    let dicts = class_dicts(h, &ty)?;
    let t = ty.is_subclass(h.parserinfo.bind(py))?
        && ty.getattr("__setattr__")?.is(h.tracked_setattr.0.bind(py))
        && ty.getattr("__delattr__")?.is(h.tracked_setattr.1.bind(py));
    let entry = TypeEntry {
        _ty: ty.clone().unbind(),
        dicts,
        verdict: t,
    };
    let old = TRACKED_TYPES.with(|c| cache_insert(&mut c.borrow_mut(), 256, key, entry));
    drop(old);
    Ok(t)
}

fn container_len(o: &Bound<'_, PyAny>) -> usize {
    o.len().unwrap_or(usize::MAX)
}

/// `d` as a dict whose lookups are `dict`'s own: a `dict`, or pyroid's change-reporting table
/// (`table_type`). dateutil looks words up with `in`, `[]` and `.get`, which a dict subclass can
/// change (`__missing__`, a `get` of its own), so another one runs dateutil's code.
fn plain_dict<'py>(
    d: &Bound<'py, PyAny>,
    table_type: &Bound<'py, PyType>,
    what: &str,
) -> Result<Bound<'py, PyDict>, String> {
    let dict = d
        .cast::<PyDict>()
        .map_err(|_| format!("a parserinfo whose {what} is not a dict"))?;
    if !(d.is_exact_instance_of::<PyDict>()
        || d.get_type_ptr().cast::<pyo3::ffi::PyObject>() == table_type.as_ptr())
    {
        return Err(format!("a parserinfo whose {what} is a dict subclass"));
    }
    Ok(dict.clone())
}

fn table_of(
    d: &Bound<'_, PyAny>,
    table_type: &Bound<'_, PyType>,
    valued: bool,
    what: &str,
) -> Result<Table, String> {
    // A copy (atomic, also on free-threaded builds): another thread may change the dict while
    // this one reads it, which a dict iterator does not allow.
    let d = plain_dict(d, table_type, what)?
        .copy()
        .map_err(|e| e.to_string())?;
    let mut t = Table::default();
    for (k, v) in d.iter() {
        let k = k
            .cast::<PyString>()
            .map_err(|_| format!("a parserinfo {what} key that is not a str"))?;
        let k = k
            .to_str()
            .map_err(|_| format!("a parserinfo {what} key with lone surrogates"))?;
        let v = if valued {
            if v.is_instance_of::<pyo3::types::PyBool>() {
                return Err(format!("a parserinfo {what} value that is a bool"));
            }
            v.extract::<i64>()
                .ok()
                .filter(|v| v.abs() <= pyroid_core::dateutil::info::MAX_TABLE_VALUE)
                .ok_or_else(|| format!("a parserinfo {what} value that is not a small int"))?
        } else {
            0
        };
        t.insert(k.into(), v);
    }
    Ok(t)
}

fn compile(objs: &[Bound<'_, PyAny>], table_type: &Bound<'_, PyType>) -> Compiled {
    let build = || -> Result<ParserInfo, String> {
        let valued = [false, true, true, true, true, false, false];
        let mut tables = Vec::with_capacity(7);
        for (i, o) in objs[..7].iter().enumerate() {
            tables.push(table_of(o, table_type, valued[i], TABLE_ATTRS[i])?);
        }
        let tzd = plain_dict(&objs[7], table_type, "TZOFFSET")?
            .copy()
            .map_err(|e| e.to_string())?;
        let mut tzoffset = FxHashMap::default();
        for (k, v) in tzd.iter() {
            let Ok(k) = k.cast::<PyString>() else {
                continue; // a non-str key never matches a token
            };
            let k = k
                .to_str()
                .map_err(|_| "a TZOFFSET key with lone surrogates".to_owned())?;
            if v.is_none() {
                continue;
            }
            if v.is_instance_of::<pyo3::types::PyBool>() {
                return Err("a TZOFFSET value that is a bool".to_owned());
            }
            let v: Int = v
                .extract::<i64>()
                .map_err(|_| "a TZOFFSET value that is not an int".to_owned())?;
            tzoffset.insert(k.into(), v);
        }
        // A snapshot of the list (atomic), as for the dicts. dateutil tests `in UTCZONE`, which
        // a list subclass can change.
        let ut = if let Ok(l) = objs[8].cast_exact::<PyList>() {
            l.to_tuple().into_any()
        } else if objs[8].is_exact_instance_of::<PyTuple>() {
            objs[8].clone()
        } else if objs[8].is_instance_of::<PyList>() || objs[8].is_instance_of::<PyTuple>() {
            return Err("a parserinfo whose UTCZONE is a list subclass".to_owned());
        } else {
            return Err("a parserinfo whose UTCZONE is not a list".to_owned());
        };
        let mut utczone_list = Vec::new();
        for z in ut.try_iter().map_err(|e| e.to_string())? {
            let z = z.map_err(|e| e.to_string())?;
            if let Ok(s) = z.cast::<PyString>() {
                let s = s
                    .to_str()
                    .map_err(|_| "a UTCZONE entry with lone surrogates".to_owned())?;
                utczone_list.push(s.into());
            } else {
                return Err("a UTCZONE entry that is not a str".to_owned());
            }
        }
        let mut it = tables.into_iter();
        let mut next = || it.next().expect("seven");
        let t = Tables {
            jump: next(),
            weekdays: next(),
            months: next(),
            hms: next(),
            ampm: next(),
            utczone: next(),
            pertain: next(),
        };
        Ok(ParserInfo::new(t, tzoffset, utczone_list))
    };
    match build() {
        Ok(i) => Compiled::Ok(Box::new(i)),
        Err(e) => Compiled::Unsupported(e),
    }
}

/// `info.<name>`: from the instance dict for dateutil's and pyroid's own parserinfo (no
/// descriptors there), else `getattr`.
fn info_attr<'py>(
    info: &Bound<'py, PyAny>,
    dict: Option<&Bound<'py, PyDict>>,
    name: &Bound<'py, PyString>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(d) = dict {
        if let Some(v) = d.get_item(name)? {
            return Ok(v);
        }
    }
    info.getattr(name)
}

/// The compiled tables and settings of parserinfo `info`.
fn info_of(h: &Hooks, info: &Bound<'_, PyAny>) -> PyResult<(Arc<Compiled>, Option<Settings>)> {
    use pyo3::intern;
    let py = info.py();
    let tracked = tracked_type(h, info)?;
    let generation = INFO_GEN.load(std::sync::atomic::Ordering::Acquire);
    if tracked {
        let tz = info.getattr(intern!(py, "TZOFFSET"))?;
        let utc = info.getattr(intern!(py, "UTCZONE"))?;
        let lens = [container_len(&tz), container_len(&utc)];
        let hit = TRACKED.with(|c| {
            c.borrow().as_ref().and_then(|e| {
                (e.generation == generation
                    && e.info.bind(py).is(info)
                    && e.class_objs[0].bind(py).is(&tz)
                    && e.class_objs[1].bind(py).is(&utc)
                    && e.class_lens == lens)
                    .then(|| (e.compiled.clone(), e.settings))
            })
        });
        if let Some(hit) = hit {
            return Ok(hit);
        }
    }
    let (compiled, settings, all_tables_tracked) = info_full(h, info)?;
    if tracked && all_tables_tracked {
        let tz = info.getattr(intern!(py, "TZOFFSET"))?;
        let utc = info.getattr(intern!(py, "UTCZONE"))?;
        let class_lens = [container_len(&tz), container_len(&utc)];
        // The entry it replaces is dropped after the borrow ends (see `cache_insert`).
        let old = TRACKED.with(|c| {
            c.replace(Some(TrackedEntry {
                info: info.clone().unbind(),
                generation,
                class_objs: [tz.unbind(), utc.unbind()],
                class_lens,
                compiled: compiled.clone(),
                settings,
            }))
        });
        drop(old);
    }
    Ok((compiled, settings))
}

/// [`info_of`] reading every attribute; also whether the seven tables are all pyroid's
/// change-reporting dicts.
fn info_full(
    h: &Hooks,
    info: &Bound<'_, PyAny>,
) -> PyResult<(Arc<Compiled>, Option<Settings>, bool)> {
    use pyo3::intern;
    let py = info.py();
    let exact = exact_info(h, info);

    let dict = if exact {
        info.getattr(intern!(py, "__dict__"))?
            .cast_into::<PyDict>()
            .ok()
    } else {
        None
    };
    let d = dict.as_ref();
    let objs = [
        info_attr(info, d, intern!(py, "_jump"))?,
        info_attr(info, d, intern!(py, "_weekdays"))?,
        info_attr(info, d, intern!(py, "_months"))?,
        info_attr(info, d, intern!(py, "_hms"))?,
        info_attr(info, d, intern!(py, "_ampm"))?,
        info_attr(info, d, intern!(py, "_utczone"))?,
        info_attr(info, d, intern!(py, "_pertain"))?,
        info_attr(info, d, intern!(py, "TZOFFSET"))?,
        info_attr(info, d, intern!(py, "UTCZONE"))?,
    ];
    let lens = objs.each_ref().map(container_len);
    let settings = {
        let dayfirst = info_attr(info, d, intern!(py, "dayfirst"))?.is_truthy()?;
        let yearfirst = info_attr(info, d, intern!(py, "yearfirst"))?.is_truthy()?;
        let year = info_attr(info, d, intern!(py, "_year"))?;
        let century = info_attr(info, d, intern!(py, "_century"))?;
        let int = |o: &Bound<'_, PyAny>| -> Option<Int> {
            if o.is_instance_of::<pyo3::types::PyBool>() {
                return None;
            }
            o.extract::<i64>().ok()
        };
        int(&year)
            .zip(int(&century))
            .map(|(year, century)| Settings {
                dayfirst,
                yearfirst,
                year,
                century,
            })
    };
    let tables_tracked = {
        let table_type = h.table_type.bind(py);
        objs[..7].iter().all(|o| o.get_type().is(table_type))
    };
    let key = info.as_ptr() as usize;
    let hit = INFOS.with(|c| {
        c.borrow()
            .iter()
            .find(|e| {
                e.info == key && e.lens == lens && e.objs.iter().zip(&objs).all(|(a, b)| a.is(b))
            })
            .map(|e| e.compiled.clone())
    });
    if let Some(c) = hit {
        return Ok((c, settings, tables_tracked));
    }
    let compiled = Arc::new(compile(&objs, h.table_type.bind(py)));
    // The entries it removes are dropped after the borrow ends (see `cache_insert`).
    let evicted = INFOS.with(|c| {
        let mut c = c.borrow_mut();
        let mut evicted = Vec::new();
        let mut i = 0;
        while i < c.len() {
            if c[i].info == key {
                evicted.push(c.remove(i));
            } else {
                i += 1;
            }
        }
        if c.len() >= 8 {
            evicted.push(c.remove(0));
        }
        c.push(InfoEntry {
            info: key,
            objs: objs.map(Bound::unbind),
            lens,
            compiled: compiled.clone(),
        });
        evicted
    });
    drop(evicted);
    Ok((compiled, settings, tables_tracked))
}

// ---------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------

/// The kernel text for a Python str: its UTF-8, or (with lone surrogates) the remapped chars.
enum Text<'a> {
    Utf8(&'a str),
    Remapped(String),
    /// Lone surrogates and a char of the range they are remapped to: the reference runs.
    Unsupported,
}

impl Text<'_> {
    fn as_str(&self) -> &str {
        match self {
            Self::Utf8(s) => s,
            Self::Remapped(s) => s,
            Self::Unsupported => "",
        }
    }
}

fn text_of<'a>(text: &'a Bound<'_, PyString>) -> PyResult<Text<'a>> {
    let py = text.py();
    match text.to_str() {
        Ok(s) => Ok(Text::Utf8(s)),
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            let bytes = py
                .get_type::<PyString>()
                .call_method1("encode", (text, "utf-8", "surrogatepass"))?
                .cast_into::<PyBytes>()?;
            match chars::chars_from_wtf8(bytes.as_bytes()) {
                Some(Ok(c)) => Ok(Text::Remapped(c.into_iter().collect())),
                Some(Err(chars::InputError::ReservedCodePoint)) => Ok(Text::Unsupported),
                None => Err(PyValueError::new_err("invalid surrogatepass UTF-8")),
            }
        }
        Err(e) => Err(e),
    }
}

/// A Python str for kernel text (`remapped`: the input's lone surrogates were remapped, so
/// remapped chars become real surrogates again).
fn py_text<'py>(py: Python<'py>, s: &str, remapped: bool) -> PyResult<Bound<'py, PyString>> {
    if !remapped || !s.chars().any(chars::is_surrogate) {
        return Ok(PyString::new(py, s));
    }
    let bytes = chars::to_wtf8(s);
    PyString::from_encoded_object(
        PyBytes::new(py, &bytes).as_any(),
        Some(c"utf-8"),
        Some(c"surrogatepass"),
    )
}

/// What `_timelex` reads from `timestr`: a str to tokenize, the text of a stream, or a replay of
/// an odd stream's reads (for the reference).
enum Source<'py> {
    Str(Bound<'py, PyString>),
    Stream(Bound<'py, PyString>),
    Replay(Bound<'py, PyAny>),
}

/// What dateutil's own code reads when it runs after the kernel: the caller's `timestr`, or for
/// a stream (read to its end by [`source`]) a replay of the same reads.
fn again<'py>(
    h: &Hooks,
    timestr: &Bound<'py, PyAny>,
    streamed: Option<&Bound<'py, PyString>>,
) -> PyResult<Bound<'py, PyAny>> {
    match streamed {
        Some(text) => h.replay_read.bind(timestr.py()).call1((text, timestr)),
        None => Ok(timestr.clone()),
    }
}

fn source<'py>(h: &Hooks, timestr: &Bound<'py, PyAny>) -> PyResult<Source<'py>> {
    let py = timestr.py();
    if let Ok(s) = timestr.cast::<PyString>() {
        return Ok(Source::Str(s.clone()));
    }
    if timestr.is_instance_of::<PyBytes>() || timestr.is_instance_of::<PyByteArray>() {
        let s = timestr.call_method0(pyo3::intern!(py, "decode"))?;
        return Ok(Source::Str(s.cast_into::<PyString>()?));
    }
    if timestr
        .getattr_opt(pyo3::intern!(py, "read"))?
        .is_none_or(|r| r.is_none())
    {
        let name = timestr.get_type().name()?;
        return Err(PyTypeError::new_err(format!(
            "Parser must be a string or character stream, not {name}"
        )));
    }
    let r = h.read_stream.bind(py).call1((timestr,))?;
    match r.cast_into::<PyString>() {
        Ok(s) => Ok(Source::Stream(s)),
        Err(e) => Ok(Source::Replay(e.into_inner())),
    }
}

// ---------------------------------------------------------------------------
// parse
// ---------------------------------------------------------------------------

/// `_parse`'s keyword arguments, if `kwargs` has only those.
fn private_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Option<ParseOptions>> {
    let mut o = ParseOptions::default();
    let Some(kw) = kwargs else {
        return Ok(Some(o));
    };
    for (k, v) in kw.iter() {
        let Ok(k) = k.cast::<PyString>() else {
            return Ok(None);
        };
        let tri = |v: &Bound<'_, PyAny>| -> PyResult<Option<bool>> {
            if v.is_none() {
                Ok(None)
            } else {
                v.is_truthy().map(Some)
            }
        };
        match k.to_str()? {
            "dayfirst" => o.dayfirst = tri(&v)?,
            "yearfirst" => o.yearfirst = tri(&v)?,
            "fuzzy" => o.fuzzy = v.is_truthy()?,
            "fuzzy_with_tokens" => o.fuzzy_with_tokens = v.is_truthy()?,
            _ => return Ok(None),
        }
    }
    Ok(Some(o))
}

/// `sys.get_int_max_str_digits()`, read only when a token could reach it.
fn max_str_digits(py: Python<'_>, len: usize) -> PyResult<usize> {
    if len < 640 {
        return Ok(0);
    }
    py.import("sys")?
        .call_method0("get_int_max_str_digits")?
        .extract()
}

/// The kernel's verdict for one string.
enum Run {
    /// The kernel's result, and whether the text was remapped (lone surrogates).
    Parsed(Box<Parsed>, bool),
    /// dateutil's `(None, None)`.
    Unknown,
    /// Run dateutil's algorithm, saying why.
    Reference(String),
}

fn run_kernel(text: &Text<'_>, tables: &ParserInfo, settings: Settings, opts: ParseOptions) -> Run {
    if matches!(text, Text::Unsupported) {
        return Run::Reference(
            "a string with both lone surrogates and characters in U+10F800..U+10FFFF".to_owned(),
        );
    }
    match kernel::parse(text.as_str(), tables, settings, props(), opts) {
        Ok(Some(p)) => Run::Parsed(Box::new(p), matches!(text, Text::Remapped(_))),
        Ok(None) => Run::Unknown,
        Err(r) => Run::Reference(r.describe().to_owned()),
    }
}

/// The verdict, after the checks that need the interpreter (the `decimal` context).
///
/// dateutil computes `d % 1` and `60 * (d % 1)` in the thread's context (for every `HH:MM`).
/// They are exact and signal nothing when the precision holds their digits and the exponent
/// limits hold their exponents: from `-decimal_scale` to 0, and 1 for `60 * (d % 1)`. Then the
/// kernel's integer arithmetic gives dateutil's values. Otherwise dateutil's code runs: it
/// rounds, sets the context's flags and raises as its traps say (`decimal.Overflow` for
/// `10.5h` with `Emax = 0`, say).
///
/// For an integer (the minutes of a plain `HH:MM`) only the precision is read: reading `clamp`
/// as well added 2 to 4% to such a parse in an A/B. With `clamp = 1` and `Emax` below the
/// precision minus one, dateutil's `int % 1` sets the context's `Clamped` flag, which pyroid then
/// does not (a documented difference).
fn checked(py: Python<'_>, h: &Hooks, run: Run) -> PyResult<Run> {
    use pyo3::intern;
    let Run::Parsed(p, _) = &run else {
        return Ok(run);
    };
    if p.decimal_precision == 0 {
        return Ok(run);
    }
    let cx = h.getcontext.bind(py).call0()?;
    let int = |name: &Bound<'_, PyString>| -> PyResult<i64> { cx.getattr(name)?.extract() };
    let prec = int(intern!(py, "prec"))?;
    if prec < i64::from(p.decimal_precision) {
        return Ok(Run::Reference(format!(
            "a decimal context precision below {}",
            p.decimal_precision
        )));
    }
    if p.decimal_scale == 0 && !p.decimal_fraction {
        return Ok(run);
    }
    // Emax >= 1 for 60 * (d % 1); with clamp, Emax - prec + 1 >= 0 for exponents up to 0;
    // Emin <= -decimal_scale for d % 1.
    let emax = int(intern!(py, "Emax"))?;
    let scale = i64::try_from(p.decimal_scale).unwrap_or(i64::MAX);
    if (p.decimal_fraction && emax < 1)
        || int(intern!(py, "Emin"))? > -scale
        || (int(intern!(py, "clamp"))? != 0 && emax < prec - 1)
    {
        return Ok(Run::Reference(
            "a decimal context whose Emax, Emin or clamp this arithmetic reaches".to_owned(),
        ));
    }
    Ok(run)
}

fn parser_error<'py>(py: Python<'py>, h: &Hooks, fmt: &str, timestr: &Bound<'py, PyAny>) -> PyErr {
    match h.parser_error.bind(py).call1((fmt, timestr)) {
        Ok(e) => PyErr::from_value(e),
        Err(e) => e,
    }
}

fn opt_int(py: Python<'_>, v: Option<Int>) -> PyResult<Bound<'_, PyAny>> {
    match v {
        Some(i) => Ok(i.into_pyobject(py)?.into_any()),
        None => Ok(py.None().into_bound(py)),
    }
}

/// `parser._result` holding `res` (for dateutil's own steps).
fn result_object<'py>(
    py: Python<'py>,
    h: &Hooks,
    res: &Res,
    remapped: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let tzname = match &res.tzname {
        Some(n) => py_text(py, n, remapped)?.into_any(),
        None => py.None().into_bound(py),
    };
    let fields = PyTuple::new(
        py,
        [
            opt_int(py, res.year)?,
            opt_int(py, res.month)?,
            opt_int(py, res.day)?,
            opt_int(py, res.weekday)?,
            opt_int(py, res.hour)?,
            opt_int(py, res.minute)?,
            opt_int(py, res.second)?,
            opt_int(py, res.microsecond)?,
            tzname,
            opt_int(py, res.tzoffset)?,
            opt_int(py, res.ampm)?,
        ],
    )?;
    h.make_result
        .bind(py)
        .call1((fields, res.century_specified))
}

fn skipped_tuple<'py>(py: Python<'py>, p: &Parsed, remapped: bool) -> PyResult<Bound<'py, PyAny>> {
    match &p.skipped {
        Some(v) => {
            let items: PyResult<Vec<_>> = v.iter().map(|s| py_text(py, s, remapped)).collect();
            Ok(PyTuple::new(py, items?)?.into_any())
        }
        None => Ok(py.None().into_bound(py)),
    }
}

/// `datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)`.
fn today<'py>(py: Python<'py>, h: &Hooks) -> PyResult<Bound<'py, PyAny>> {
    let now = h
        .datetime_module
        .bind(py)
        .getattr(pyo3::intern!(py, "datetime"))?
        .call_method0(pyo3::intern!(py, "now"))?;
    let kw = PyDict::new(py);
    for k in ["hour", "minute", "second", "microsecond"] {
        kw.set_item(k, 0)?;
    }
    now.call_method("replace", (), Some(&kw))
}

/// The fields `_build_naive` reads from `default`.
struct DefaultFields<'py> {
    ymd: [Int; 3],
    hms: [Int; 4],
    tzinfo: Bound<'py, PyAny>,
    fold: bool,
}

fn default_fields<'py>(d: &Bound<'py, PyAny>) -> PyResult<DefaultFields<'py>> {
    let py = d.py();
    let g = |n: &Bound<'py, PyString>| -> PyResult<Int> { d.getattr(n)?.extract() };
    Ok(DefaultFields {
        ymd: [
            g(pyo3::intern!(py, "year"))?,
            g(pyo3::intern!(py, "month"))?,
            g(pyo3::intern!(py, "day"))?,
        ],
        hms: [
            g(pyo3::intern!(py, "hour"))?,
            g(pyo3::intern!(py, "minute"))?,
            g(pyo3::intern!(py, "second"))?,
            g(pyo3::intern!(py, "microsecond"))?,
        ],
        tzinfo: d.getattr(pyo3::intern!(py, "tzinfo"))?,
        fold: d.getattr(pyo3::intern!(py, "fold"))?.is_truthy()?,
    })
}

/// The naive fields of `_build_naive(res, default)` when `datetime` accepts them and no
/// Python-level error can occur (`None`: run dateutil's steps instead), and whether the
/// weekday step (`+ relativedelta(weekday=...)`) ran.
#[allow(clippy::many_single_char_names)]
fn naive_fields(res: &Res, ymd: [Int; 3], d: &DefaultFields<'_>) -> Option<([i64; 7], bool)> {
    let year = res.year.unwrap_or(ymd[0]);
    let month = res.month.unwrap_or(ymd[1]);
    let mut day = res.day.unwrap_or(ymd[2]);
    if res.day.is_none() {
        // If the default day exceeds the last day of the month, fall back to the end of it.
        let md = kernel::cal::month_days(year, month)?;
        if day > md {
            day = md;
        }
    }
    let f = [
        year,
        month,
        day,
        res.hour.unwrap_or(d.hms[0]),
        res.minute.unwrap_or(d.hms[1]),
        res.second.unwrap_or(d.hms[2]),
        res.microsecond.unwrap_or(d.hms[3]),
    ];
    let mut out = [0i64; 7];
    for (o, v) in out.iter_mut().zip(f) {
        *o = v;
    }
    let ord = kernel::cal::ordinal(out[0], out[1], out[2])?;
    if !(0..24).contains(&out[3])
        || !(0..60).contains(&out[4])
        || !(0..60).contains(&out[5])
        || !(0..1_000_000).contains(&out[6])
    {
        return None;
    }
    let mut jumped = false;
    if let Some(w) = res.weekday {
        if res.day.is_none_or(|d| d == 0) {
            jumped = true;
            // + relativedelta(weekday=w): dateutil indexes (MO, ..., SU) with w.
            let w = Some(w).filter(|w| (-7..7).contains(w))?;
            let w = w.rem_euclid(7);
            let jump = (7 - kernel::cal::weekday(ord) + w).rem_euclid(7);
            let [y, m, dd] = kernel::cal::from_ordinal(ord + jump)?;
            out[0] = y;
            out[1] = m;
            out[2] = dd;
        }
    }
    Some((out, jumped))
}

/// The rest of `parser.parse` after `_parse` (`_build_naive`, `_build_tzaware`), for the
/// arguments of one call or of one batch (what it reads from them is read once).
struct Finisher<'py, 'a> {
    py: Python<'py>,
    h: &'a Hooks,
    self_: &'a Bound<'py, PyAny>,
    default: &'a Bound<'py, PyAny>,
    ignoretz: bool,
    tzinfos: &'a Bound<'py, PyAny>,
    kwargs: Option<&'a Bound<'py, PyDict>>,
    /// `default`'s fields (today's time, tz and fold for `default=None`); `None`: the fast
    /// path cannot be used (another `default` type, or a patched `datetime.datetime`).
    base: OnceCell<Option<DefaultFields<'py>>>,
    /// Today's date, for `default=None`.
    today: OnceCell<[Int; 3]>,
    /// `time.tzname`.
    local_names: OnceCell<Bound<'py, PyAny>>,
    /// The local zone, when it has no DST (`None`: dateutil's steps run for it).
    local: OnceCell<Option<LocalZone<'py>>>,
    /// The parser's `info.UTCZONE`.
    utczone: &'a [Box<str>],
    offsets: TzOffsets<'py>,
}

/// `tz.tzlocal()` for a local zone without DST, where its `tzname()` is `time.tzname[0]` and
/// its offset `-time.timezone` at every date.
struct LocalZone<'py> {
    name: String,
    std_offset: i64,
    tz: Bound<'py, PyAny>,
}

/// The local-zone fields a `tzlocal()` was made under.
type LocalKey = (i64, i64, bool, String, String);

thread_local! {
    /// The `tzlocal()` handed out while the local zone stays the same (dateutil makes a new,
    /// equal one per call).
    static TZLOCAL: RefCell<Option<(LocalKey, Py<PyAny>)>> = const { RefCell::new(None) };
}

impl<'py, 'a> Finisher<'py, 'a> {
    fn new(
        h: &'a Hooks,
        self_: &'a Bound<'py, PyAny>,
        default: &'a Bound<'py, PyAny>,
        ignoretz: bool,
        tzinfos: &'a Bound<'py, PyAny>,
        kwargs: Option<&'a Bound<'py, PyDict>>,
        utczone: &'a [Box<str>],
    ) -> Self {
        Self {
            py: self_.py(),
            h,
            self_,
            default,
            ignoretz,
            tzinfos,
            kwargs,
            base: OnceCell::new(),
            today: OnceCell::new(),
            local_names: OnceCell::new(),
            local: OnceCell::new(),
            utczone,
            offsets: TzOffsets::default(),
        }
    }

    fn local_zone(&self) -> PyResult<Option<&LocalZone<'py>>> {
        if let Some(z) = self.local.get() {
            return Ok(z.as_ref());
        }
        let py = self.py;
        let t = self.h.time_module.bind(py);
        let timezone: i64 = t.getattr(pyo3::intern!(py, "timezone"))?.extract()?;
        let altzone: i64 = t.getattr(pyo3::intern!(py, "altzone"))?.extract()?;
        let daylight = t.getattr(pyo3::intern!(py, "daylight"))?.is_truthy()?;
        let names: Option<(String, String)> =
            t.getattr(pyo3::intern!(py, "tzname"))?.extract().ok();
        let z = match names {
            Some((n0, n1)) if !daylight || altzone == timezone => {
                let key: LocalKey = (timezone, altzone, daylight, n0.clone(), n1);
                let cached = TZLOCAL.with(|c| {
                    c.borrow()
                        .as_ref()
                        .filter(|(k, _)| *k == key)
                        .map(|(_, v)| v.clone_ref(py))
                });
                let tz = if let Some(v) = cached {
                    v.into_bound(py)
                } else {
                    let v = self.h.tzlocal.bind(py).call0()?;
                    let old = TZLOCAL.with(|c| c.replace(Some((key, v.clone().unbind()))));
                    drop(old);
                    v
                };
                Some(LocalZone {
                    name: n0,
                    std_offset: -timezone,
                    tz,
                })
            }
            _ => None,
        };
        Ok(self.local.get_or_init(|| z).as_ref())
    }

    fn base(&self) -> PyResult<Option<&DefaultFields<'py>>> {
        if let Some(b) = self.base.get() {
            return Ok(b.as_ref());
        }
        let py = self.py;
        let dt_type = self.h.datetime_type.bind(py);
        let b = if !(self.ignoretz || self.tzinfos.is_none()) {
            None // tzinfos: dateutil's _build_tzaware
        } else if self.default.is_none() {
            let current = self
                .h
                .datetime_module
                .bind(py)
                .getattr(pyo3::intern!(py, "datetime"))?;
            // A patched datetime.datetime (freezegun) makes dateutil's result its type.
            current.is(dt_type).then(|| DefaultFields {
                ymd: [0, 1, 1],
                hms: [0; 4],
                tzinfo: py.None().into_bound(py),
                fold: false,
            })
        } else if self.default.get_type_ptr().cast::<pyo3::ffi::PyObject>() == dt_type.as_ptr() {
            Some(default_fields(self.default)?)
        } else {
            None
        };
        Ok(self.base.get_or_init(|| b).as_ref())
    }

    fn today(&self) -> PyResult<[Int; 3]> {
        if let Some(t) = self.today.get() {
            return Ok(*t);
        }
        let py = self.py;
        let t = self
            .h
            .datetime_type
            .bind(py)
            .call_method0(pyo3::intern!(py, "now"))?;
        let ymd = [
            t.getattr(pyo3::intern!(py, "year"))?.extract()?,
            t.getattr(pyo3::intern!(py, "month"))?.extract()?,
            t.getattr(pyo3::intern!(py, "day"))?.extract()?,
        ];
        Ok(*self.today.get_or_init(|| ymd))
    }

    fn is_local_name(&self, name: &str, remapped: bool) -> PyResult<bool> {
        let py = self.py;
        let names = if let Some(n) = self.local_names.get() {
            n
        } else {
            let n = self
                .h
                .time_module
                .bind(py)
                .getattr(pyo3::intern!(py, "tzname"))?;
            self.local_names.get_or_init(|| n)
        };
        names.contains(py_text(py, name, remapped)?)
    }

    /// `parse`'s result for kernel result `p` of `timestr`.
    /// The fast path's result for kernel result `p` (`None`: dateutil's steps must run), after
    /// the checks `parse` makes first.
    fn plan(
        &self,
        p: &Parsed,
        remapped: bool,
        timestr: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Fast<'_, 'py>>> {
        let py = self.py;
        let res = &p.res;
        if res.is_empty() {
            return Err(parser_error(
                py,
                self.h,
                "String does not contain a date: %s",
                timestr,
            ));
        }
        let Some(b) = self.base()? else {
            return Ok(None);
        };
        let ymd = if !self.default.is_none() {
            b.ymd
        } else if res.year.is_none() || res.month.is_none() || res.day.is_none() {
            self.today()?
        } else {
            b.ymd
        };
        let Some((f, jumped)) = naive_fields(res, ymd, b) else {
            return Ok(None);
        };
        // `+ relativedelta(weekday=...)` adds a timedelta, which gives fold 0.
        let fold = b.fold && !jumped;
        // _build_tzaware's branches that need no Python logic.
        let tz = if self.ignoretz {
            TzPlan::Default
        } else {
            let local = res
                .tzname
                .as_deref()
                .map(|n| self.is_local_name(n, remapped))
                .transpose()?;
            if local == Some(true) {
                // tz.tzlocal(), then _assign_tzname, then UTC for a UTCZONE name it does not
                // give (GMT parsed in the UK in winter). With DST, dateutil's steps run.
                let Some(z) = self.local_zone()? else {
                    return Ok(None);
                };
                let name = res.tzname.as_deref().expect("a local name");
                if z.name != name && self.utczone.iter().any(|u| &**u == name) {
                    TzPlan::Utc
                } else {
                    TzPlan::Local
                }
            } else if res.tzoffset == Some(0) {
                TzPlan::Utc
            } else if let Some(off) = res.tzoffset {
                TzPlan::Offset(off)
            } else {
                if let Some(n) = &res.tzname {
                    warn_unknown_tz(py, self.h, n)?;
                }
                TzPlan::Default
            }
        };
        Ok(Some(Fast {
            f,
            fold,
            tz,
            base: b,
        }))
    }

    /// dateutil's own `_build_naive` / `_build_tzaware` (`parser._finish`).
    fn slow(
        &self,
        p: &Parsed,
        remapped: bool,
        timestr: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let h = self.h;
        let res_obj = result_object(py, h, &p.res, remapped)?;
        let default = if self.default.is_none() {
            today(py, h)?
        } else {
            self.default.clone()
        };
        h.finish.bind(py).call1((
            self.self_,
            res_obj,
            skipped_tuple(py, p, remapped)?,
            timestr,
            default,
            self.ignoretz,
            self.tzinfos,
            self.kwargs.map_or_else(|| PyDict::new(py), Bound::clone),
        ))
    }

    /// The fast path's `datetime` (with dateutil's tz objects).
    fn object(
        &self,
        fast: &Fast<'_, 'py>,
        p: &Parsed,
        remapped: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let h = self.h;
        let tzinfo = match fast.tz {
            TzPlan::Default => fast.base.tzinfo.clone(),
            TzPlan::Utc => h.utc.bind(py).clone(),
            TzPlan::Local => self
                .local
                .get()
                .and_then(Option::as_ref)
                .expect("planned with a local zone")
                .tz
                .clone(),
            TzPlan::Offset(off) => {
                self.offsets
                    .get(py, h, p.res.tzname.as_deref(), remapped, off)?
            }
        };
        let f = fast.f;
        let args = (f[0], f[1], f[2], f[3], f[4], f[5], f[6], tzinfo);
        let dt_type = h.datetime_type.bind(py);
        if fast.fold {
            let kw = PyDict::new(py);
            kw.set_item("fold", 1)?;
            dt_type.call(args, Some(&kw))
        } else {
            dt_type.call1(args)
        }
    }

    /// `parse`'s result for kernel result `p` of `timestr`.
    fn finish(
        &self,
        p: &Parsed,
        remapped: bool,
        timestr: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let ret = match self.plan(p, remapped, timestr)? {
            Some(fast) => self.object(&fast, p, remapped)?,
            None => return self.slow(p, remapped, timestr),
        };
        if p.skipped.is_some() {
            return Ok(PyTuple::new(py, [ret, skipped_tuple(py, p, remapped)?])?.into_any());
        }
        Ok(ret)
    }

    /// The result as an Arrow timestamp: (microseconds, aware).
    fn instant(
        &self,
        p: &Parsed,
        remapped: bool,
        timestr: &Bound<'py, PyAny>,
    ) -> PyResult<Instant> {
        if let Some(fast) = self.plan(p, remapped, timestr)? {
            let Some(naive) = micros(&fast.f) else {
                return py_instant(&self.object(&fast, p, remapped)?);
            };
            match fast.tz {
                TzPlan::Utc => return Ok(Some((naive, true))),
                TzPlan::Local => {
                    let z = self
                        .local
                        .get()
                        .and_then(Option::as_ref)
                        .expect("planned with a local zone");
                    return Ok(Some((naive - z.std_offset * 1_000_000, true)));
                }
                TzPlan::Offset(off) if off.abs() < 86_400 => {
                    return Ok(Some((naive - off * 1_000_000, true)));
                }
                TzPlan::Default if fast.base.tzinfo.is_none() => return Ok(Some((naive, false))),
                _ => {
                    let dt = self.object(&fast, p, remapped)?;
                    return py_instant(&dt);
                }
            }
        }
        py_instant(&self.slow(p, remapped, timestr)?)
    }
}

/// What a `tz.tzoffset(name, seconds)` is kept under: the factory (`Hooks::tzoffset`, by
/// address), the tzname (with `remapped` as in [`py_text`]) and the seconds.
type TzKey = (usize, Option<Box<str>>, bool, Int);

/// How many `tzoffset`s a thread keeps.
const TZOFFSETS_KEPT: usize = 64;

/// How many `tzoffset`s [`SHARED_TZOFFSETS`] keeps.
const TZOFFSETS_SHARED: usize = 1024;

thread_local! {
    /// This thread's copies of [`SHARED_TZOFFSETS`] entries. Looking one up takes no lock and
    /// writes nothing another thread reads; going to dateutil's factory on every call (a lock
    /// and two caches) made threads queue on it.
    static TZOFFSETS: RefCell<Vec<(TzKey, Py<PyAny>)>> = const { RefCell::new(Vec::new()) };
    /// Whether this thread is making a `tzoffset` for [`SHARED_TZOFFSETS`] (and the factory
    /// call ran code that parses again).
    static MAKING_TZOFFSET: Cell<bool> = const { Cell::new(false) };
}

/// The `tz.tzoffset(name, seconds)` objects pyroid hands out, one per key for every thread, and
/// each the one dateutil's factory keeps (see [`make_tzoffset`]). dateutil's factory returns the
/// same instance for a key while one is alive, and these are kept alive, so it goes on returning
/// them. Held only for lookups and inserts: no Python code runs under it, nothing is dropped
/// under it.
static SHARED_TZOFFSETS: LazyLock<Mutex<FxHashMap<TzKey, Py<PyAny>>>> =
    LazyLock::new(Mutex::default);

/// Held while a `tzoffset` for [`SHARED_TZOFFSETS`] is made, so that two threads never make one
/// at the same moment (which dateutil's factory resolves by giving one of them an instance it
/// then drops for the other's).
static MAKE_TZOFFSET: Mutex<()> = Mutex::new(());

fn shared_tzoffset(py: Python<'_>, key: &TzKey) -> Option<Py<PyAny>> {
    let shared = SHARED_TZOFFSETS
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    shared.get(key).map(|t| t.clone_ref(py))
}

/// `tz.tzoffset(name, seconds)`, asked until it returns the same object twice. When another
/// thread makes the same key at the same moment, the factory can return an instance its
/// `WeakValueDictionary.setdefault` then replaces with the other thread's; the next call returns
/// the one it keeps.
fn make_tzoffset<'py>(
    py: Python<'py>,
    h: &Hooks,
    name: Option<&str>,
    remapped: bool,
    off: Int,
) -> PyResult<Bound<'py, PyAny>> {
    let py_name = match name {
        Some(n) => py_text(py, n, remapped)?.into_any(),
        None => py.None().into_bound(py),
    };
    let factory = h.tzoffset.bind(py);
    let mut t = factory.call1((&py_name, off))?;
    for _ in 0..8 {
        let again = factory.call1((&py_name, off))?;
        if again.is(&t) {
            break;
        }
        t = again;
    }
    Ok(t)
}

/// `tz.tzoffset(name, seconds)` for one call or one batch (see [`SHARED_TZOFFSETS`]).
#[derive(Default)]
struct TzOffsets<'py>(std::marker::PhantomData<Bound<'py, PyAny>>);

impl<'py> TzOffsets<'py> {
    #[allow(clippy::unused_self)]
    fn get(
        &self,
        py: Python<'py>,
        h: &Hooks,
        name: Option<&str>,
        remapped: bool,
        off: Int,
    ) -> PyResult<Bound<'py, PyAny>> {
        let factory = h.tzoffset.as_ptr() as usize;
        let hit = TZOFFSETS.with(|c| {
            c.borrow()
                .iter()
                .find(|((f, n, r, o), _)| {
                    *o == off && *f == factory && *r == remapped && n.as_deref() == name
                })
                .map(|(_, t)| t.clone_ref(py))
        });
        if let Some(t) = hit {
            return Ok(t.into_bound(py));
        }
        let key: TzKey = (factory, name.map(Box::from), remapped, off);
        let t = if let Some(t) = shared_tzoffset(py, &key) {
            t
        } else if MAKING_TZOFFSET.get() {
            // Parsing from inside the factory call below (a `__del__` it ran): the lock is this
            // thread's, so ask the factory directly.
            return make_tzoffset(py, h, name, remapped, off);
        } else {
            // Detached while it waits: the holder runs Python code.
            let making = MAKE_TZOFFSET
                .lock_py_attached(py)
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(t) = shared_tzoffset(py, &key) {
                t
            } else {
                MAKING_TZOFFSET.set(true);
                let made = make_tzoffset(py, h, name, remapped, off);
                MAKING_TZOFFSET.set(false);
                let t = made?.unbind();
                let old = {
                    let mut shared = SHARED_TZOFFSETS
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner);
                    let old = if shared.len() >= TZOFFSETS_SHARED {
                        std::mem::take(&mut *shared)
                    } else {
                        FxHashMap::default()
                    };
                    let replaced = shared.insert(key.clone(), t.clone_ref(py));
                    (old, replaced)
                };
                drop(making);
                drop(old);
                t
            }
        };
        // The entries it drops are dropped after the borrow ends (see `cache_insert`).
        let old = TZOFFSETS.with(|c| {
            let mut c = c.borrow_mut();
            let old = if c.len() >= TZOFFSETS_KEPT {
                std::mem::take(&mut *c)
            } else {
                Vec::new()
            };
            c.push((key, t.clone_ref(py)));
            old
        });
        drop(old);
        Ok(t.into_bound(py))
    }
}

/// How `_build_tzaware` ends on the fast path.
#[derive(Clone, Copy)]
enum TzPlan {
    /// No time zone from the string: `default`'s tzinfo (or none).
    Default,
    Utc,
    /// `tz.tzlocal()` (a zone without DST).
    Local,
    /// `tz.tzoffset(tzname, seconds)`.
    Offset(Int),
}

struct Fast<'b, 'py> {
    f: [i64; 7],
    fold: bool,
    tz: TzPlan,
    base: &'b DefaultFields<'py>,
}

/// An Arrow timestamp: microseconds since the epoch, and whether it is an instant (aware).
type Instant = Option<(i64, bool)>;

/// Microseconds since 1970-01-01T00:00 of naive fields, if they are a valid date and time.
fn micros(f: &[i64; 7]) -> Option<i64> {
    let days = kernel::cal::ordinal(f[0], f[1], f[2])? - 719_163;
    if !(0..24).contains(&f[3])
        || !(0..60).contains(&f[4])
        || !(0..60).contains(&f[5])
        || !(0..1_000_000).contains(&f[6])
    {
        return None;
    }
    Some(days * 86_400_000_000 + f[3] * 3_600_000_000 + f[4] * 60_000_000 + f[5] * 1_000_000 + f[6])
}

/// [`Instant`] of a Python `datetime`.
fn py_instant(dt: &Bound<'_, PyAny>) -> PyResult<Instant> {
    let py = dt.py();
    let g = |n: &str| -> PyResult<i64> { dt.getattr(n)?.extract() };
    let f = [
        g("year")?,
        g("month")?,
        g("day")?,
        g("hour")?,
        g("minute")?,
        g("second")?,
        g("microsecond")?,
    ];
    // A datetime subclass can report any fields and offset: they are checked, not trusted.
    let Some(naive) = micros(&f) else {
        return Err(PyValueError::new_err(format!(
            "{} is not a valid date and time for an Arrow timestamp",
            dt.repr()?
        )));
    };
    let off = dt.call_method0(pyo3::intern!(py, "utcoffset"))?;
    if off.is_none() {
        return Ok(Some((naive, false)));
    }
    let part = |n: &str, unit: i64| -> PyResult<Option<i64>> {
        Ok(off.getattr(n)?.extract::<i64>()?.checked_mul(unit))
    };
    let days = part("days", 86_400_000_000)?;
    let secs = part("seconds", 1_000_000)?;
    let us = part("microseconds", 1)?;
    let instant = days
        .zip(secs)
        .zip(us)
        .and_then(|((days, secs), us)| days.checked_add(secs)?.checked_add(us))
        .and_then(|off_us| naive.checked_sub(off_us));
    match instant {
        Some(t) => Ok(Some((t, true))),
        None => Err(PyValueError::new_err(format!(
            "the UTC offset of {} is out of range for an Arrow timestamp",
            dt.repr()?
        ))),
    }
}

fn warn_unknown_tz(py: Python<'_>, h: &Hooks, name: &str) -> PyResult<()> {
    let msg = format!(
        "tzname {name} identified but not understood.  Pass `tzinfos` argument in order to \
         correctly return a timezone-aware datetime.  In a future version, this will raise an \
         exception."
    );
    let msg = std::ffi::CString::new(msg)
        .unwrap_or_else(|_| c"tzname identified but not understood.".to_owned());
    PyErr::warn(py, h.unknown_tz.bind(py), &msg, 1)
}

/// `parser.parse(self, timestr, default, ignoretz, tzinfos, **kwargs)`.
#[pyfunction]
#[pyo3(signature = (self_, timestr, default, ignoretz, tzinfos, kwargs))]
fn parser_parse<'py>(
    self_: &Bound<'py, PyAny>,
    timestr: &Bound<'py, PyAny>,
    default: &Bound<'py, PyAny>,
    ignoretz: &Bound<'py, PyAny>,
    tzinfos: &Bound<'py, PyAny>,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    parse_impl(self_, timestr, default, ignoretz, tzinfos, Some(kwargs))
}

fn parse_impl<'py>(
    self_: &Bound<'py, PyAny>,
    timestr: &Bound<'py, PyAny>,
    default: &Bound<'py, PyAny>,
    ignoretz: &Bound<'py, PyAny>,
    tzinfos: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = self_.py();
    let h = hooks(py)?;
    let reference = |reason: Option<String>, timestr: &Bound<'py, PyAny>| {
        let kw = kwargs.map_or_else(|| PyDict::new(py), Bound::clone);
        h.reference_parse
            .bind(py)
            .call1((self_, reason, timestr, default, ignoretz, tzinfos, kw))
    };
    if let Some(r) = parser_override(py, h, self_)? {
        return reference(Some(r), timestr);
    }
    let Some(mut opts) = private_kwargs(kwargs)? else {
        // An unexpected keyword: dateutil's _parse signature raises the TypeError.
        return reference(None, timestr);
    };
    let info = self_.getattr(pyo3::intern!(py, "info"))?;
    if let Some(r) = info_override(py, h, &info)? {
        return reference(Some(r), timestr);
    }
    let (compiled, settings) = info_of(h, &info)?;
    let (tables, settings) = match (&*compiled, settings) {
        (Compiled::Ok(t), Some(s)) => (t, s),
        (Compiled::Unsupported(r), _) => return reference(Some(r.clone()), timestr),
        (_, None) => {
            return reference(
                Some("a parserinfo whose _year or _century is not an int".to_owned()),
                timestr,
            );
        }
    };
    let (text, streamed) = match source(h, timestr)? {
        Source::Str(s) => (s, false),
        Source::Stream(s) => (s, true),
        Source::Replay(r) => return reference(Some("this stream".to_owned()), &r),
    };
    let t = text_of(&text)?;
    opts.max_str_digits = max_str_digits(py, t.as_str().len())?;
    let run = checked(py, h, run_kernel(&t, tables, settings, opts))?;
    match run {
        Run::Unknown => Err(parser_error(py, h, "Unknown string format: %s", timestr)),
        Run::Reference(r) => reference(Some(r), &again(h, timestr, streamed.then_some(&text))?),
        Run::Parsed(p, remapped) => Finisher::new(
            h,
            self_,
            default,
            ignoretz.is_truthy()?,
            tzinfos,
            kwargs,
            &tables.utczone_list,
        )
        .finish(&p, remapped, timestr),
    }
}

/// `parser._parse(self, timestr, dayfirst, yearfirst, fuzzy, fuzzy_with_tokens)`.
#[pyfunction]
#[pyo3(signature = (self_, timestr, dayfirst, yearfirst, fuzzy, fuzzy_with_tokens))]
fn parser_private_parse<'py>(
    self_: &Bound<'py, PyAny>,
    timestr: &Bound<'py, PyAny>,
    dayfirst: &Bound<'py, PyAny>,
    yearfirst: &Bound<'py, PyAny>,
    fuzzy: &Bound<'py, PyAny>,
    fuzzy_with_tokens: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = self_.py();
    let h = hooks(py)?;
    let reference = |reason: String, timestr: &Bound<'py, PyAny>| {
        h.reference_private_parse.bind(py).call1((
            self_,
            reason,
            timestr,
            dayfirst,
            yearfirst,
            fuzzy,
            fuzzy_with_tokens,
        ))
    };
    if let Some(r) = private_override(h, self_)? {
        return reference(r, timestr);
    }
    let info = self_.getattr(pyo3::intern!(py, "info"))?;
    if let Some(r) = info_override(py, h, &info)? {
        return reference(r, timestr);
    }
    let tri = |v: &Bound<'_, PyAny>| -> PyResult<Option<bool>> {
        if v.is_none() {
            Ok(None)
        } else {
            v.is_truthy().map(Some)
        }
    };
    let mut opts = ParseOptions {
        dayfirst: tri(dayfirst)?,
        yearfirst: tri(yearfirst)?,
        fuzzy: fuzzy.is_truthy()?,
        fuzzy_with_tokens: fuzzy_with_tokens.is_truthy()?,
        max_str_digits: 0,
    };
    let (compiled, settings) = info_of(h, &info)?;
    let (tables, settings) = match (&*compiled, settings) {
        (Compiled::Ok(t), Some(s)) => (t, s),
        (Compiled::Unsupported(r), _) => return reference(r.clone(), timestr),
        (_, None) => {
            return reference(
                "a parserinfo whose _year or _century is not an int".to_owned(),
                timestr,
            );
        }
    };
    let (text, streamed) = match source(h, timestr)? {
        Source::Str(s) => (s, false),
        Source::Stream(s) => (s, true),
        Source::Replay(r) => return reference("this stream".to_owned(), &r),
    };
    let t = text_of(&text)?;
    opts.max_str_digits = max_str_digits(py, t.as_str().len())?;
    match checked(py, h, run_kernel(&t, tables, settings, opts))? {
        Run::Unknown => Ok(PyTuple::new(py, [py.None(), py.None()])?.into_any()),
        Run::Reference(r) => reference(r, &again(h, timestr, streamed.then_some(&text))?),
        Run::Parsed(p, remapped) => {
            let res = result_object(py, h, &p.res, remapped)?;
            Ok(PyTuple::new(py, [res, skipped_tuple(py, &p, remapped)?])?.into_any())
        }
    }
}

/// The keyword arguments of the module-level `parse(timestr, parserinfo=None, **kwargs)`.
///
/// `parse` takes dateutil's keywords as named parameters rather than in a `**kwargs` dict: the
/// dict PyO3 would build per call takes a reference to each value, and a `default` that every
/// thread passes is then an atomic write per call on free-threaded builds, which threads parsing
/// at once queue on. A keyword passed as `None` is the same as one not passed, as in dateutil's
/// `parse` and `_parse`.
struct ParseKw<'a, 'py> {
    default: Option<&'a Bound<'py, PyAny>>,
    ignoretz: Option<&'a Bound<'py, PyAny>>,
    tzinfos: Option<&'a Bound<'py, PyAny>>,
    dayfirst: Option<&'a Bound<'py, PyAny>>,
    yearfirst: Option<&'a Bound<'py, PyAny>>,
    fuzzy: Option<&'a Bound<'py, PyAny>>,
    fuzzy_with_tokens: Option<&'a Bound<'py, PyAny>>,
    /// Any other keyword argument (dateutil's `_parse` raises its `TypeError`).
    rest: Option<&'a Bound<'py, PyDict>>,
}

impl<'py> ParseKw<'_, 'py> {
    /// `_parse`'s keywords that were given, as a dict (`None`: none).
    fn private(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let given = [
            ("dayfirst", self.dayfirst),
            ("yearfirst", self.yearfirst),
            ("fuzzy", self.fuzzy),
            ("fuzzy_with_tokens", self.fuzzy_with_tokens),
        ];
        if given.iter().all(|(_, v)| v.is_none()) {
            return Ok(None);
        }
        let d = PyDict::new(py);
        for (k, v) in given {
            if let Some(v) = v {
                d.set_item(k, v)?;
            }
        }
        Ok(Some(d))
    }

    /// Every keyword that was given, for another object's `parse` method.
    fn all(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in [
            ("default", self.default),
            ("ignoretz", self.ignoretz),
            ("tzinfos", self.tzinfos),
        ] {
            if let Some(v) = v {
                d.set_item(k, v)?;
            }
        }
        if let Some(p) = self.private(py)? {
            d.update(p.as_mapping())?;
        }
        if let Some(r) = self.rest {
            d.update(r.as_mapping())?;
        }
        Ok(d)
    }
}

/// dateutil's module-level `parse(timestr, parserinfo=None, **kwargs)`.
#[pyfunction]
#[pyo3(
    signature = (
        timestr, parserinfo = None, *, default = None, ignoretz = None, tzinfos = None,
        dayfirst = None, yearfirst = None, fuzzy = None, fuzzy_with_tokens = None, **kwargs
    ),
    text_signature = "(timestr, parserinfo=None, **kwargs)"
)]
#[allow(clippy::too_many_arguments)]
fn parse<'py>(
    timestr: &Bound<'py, PyAny>,
    parserinfo: Option<&Bound<'py, PyAny>>,
    default: Option<&Bound<'py, PyAny>>,
    ignoretz: Option<&Bound<'py, PyAny>>,
    tzinfos: Option<&Bound<'py, PyAny>>,
    dayfirst: Option<&Bound<'py, PyAny>>,
    yearfirst: Option<&Bound<'py, PyAny>>,
    fuzzy: Option<&Bound<'py, PyAny>>,
    fuzzy_with_tokens: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = timestr.py();
    let h = hooks(py)?;
    let kw = ParseKw {
        default,
        ignoretz,
        tzinfos,
        dayfirst,
        yearfirst,
        fuzzy,
        fuzzy_with_tokens,
        rest: kwargs,
    };
    module_parse(
        h.module.bind(py),
        h.parser.bind(py),
        timestr,
        parserinfo,
        &kw,
    )
}

/// `parse` of the module whose globals are `module` (pyroid's, or dateutil's own when
/// patched), whose `parser` class is `parser_type`: dateutil's
/// `parser(parserinfo).parse(...)` or `DEFAULTPARSER.parse(...)`.
fn module_parse<'py>(
    module: &Bound<'py, PyDict>,
    parser_type: &Bound<'py, PyType>,
    timestr: &Bound<'py, PyAny>,
    parserinfo: Option<&Bound<'py, PyAny>>,
    kw: &ParseKw<'_, 'py>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = timestr.py();
    let info = match parserinfo {
        Some(i) if i.is_truthy()? => Some(i),
        _ => None,
    };
    let dp = if let Some(info) = info {
        // `parser(parserinfo)`: the module's current `parser`.
        let cls = module
            .get_item(pyo3::intern!(py, "parser"))?
            .ok_or_else(|| PyValueError::new_err("the parser module has no parser"))?;
        if !cls.is(parser_type) {
            return cls.call1((info,))?.call_method(
                pyo3::intern!(py, "parse"),
                (timestr,),
                Some(&kw.all(py)?),
            );
        }
        cls.call1((info,))?
    } else {
        module
            .get_item(pyo3::intern!(py, "DEFAULTPARSER"))?
            .ok_or_else(|| PyValueError::new_err("the parser module has no DEFAULTPARSER"))?
    };
    if dp.get_type_ptr().cast::<pyo3::ffi::PyObject>() != parser_type.as_ptr()
        || kw.rest.is_some_and(|r| !r.is_empty())
    {
        // Another parser object's own `parse`, or a keyword dateutil's `_parse` rejects.
        return dp.call_method(pyo3::intern!(py, "parse"), (timestr,), Some(&kw.all(py)?));
    }
    let none = py.None().into_bound(py);
    let no = pyo3::types::PyBool::new(py, false).to_owned().into_any();
    let private = kw.private(py)?;
    parse_impl(
        &dp,
        timestr,
        kw.default.unwrap_or(&none),
        kw.ignoretz.unwrap_or(&no),
        kw.tzinfos.unwrap_or(&none),
        private.as_ref(),
    )
}

/// What `pyroid.patch("dateutil")` sets on python-dateutil's own modules.
struct Upstream {
    /// `dateutil.parser._parser.__dict__`.
    module: Py<PyDict>,
    parser: Py<PyType>,
    /// `dateutil.parser.isoparser.__dict__` (its `DEFAULT_ISOPARSER`, its `datetime`).
    iso_module: Py<PyDict>,
    isoparser: Py<PyType>,
    /// The installed release's isoparse.
    iso_mode: Mode,
}

static UPSTREAM: PyOnceLock<Upstream> = PyOnceLock::new();

/// Called by `parser._patched_functions` with dateutil's modules and isoparse mode.
#[pyfunction]
fn _install_upstream(d: &Bound<'_, PyDict>) -> PyResult<()> {
    let py = d.py();
    let u = Upstream {
        module: item(d, "module")?.cast_into::<PyDict>()?.unbind(),
        parser: item(d, "parser")?.cast_into::<PyType>()?.unbind(),
        iso_module: item(d, "iso_module")?.cast_into::<PyDict>()?.unbind(),
        isoparser: item(d, "isoparser")?.cast_into::<PyType>()?.unbind(),
        iso_mode: if item(d, "lenient")?.is_truthy()? {
            Mode::Lenient
        } else {
            Mode::Strict
        },
    };
    let _ = UPSTREAM.set(py, u);
    Ok(())
}

fn upstream(py: Python<'_>) -> PyResult<&Upstream> {
    UPSTREAM.get(py).ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err("pyroid.patch('dateutil') has not run")
    })
}

/// `dateutil.parser.parse` under `pyroid.patch("dateutil")`: dateutil's own `parse`, with
/// its `DEFAULTPARSER` and classes, parsed by pyroid.
#[pyfunction]
#[pyo3(
    name = "parse",
    signature = (
        timestr, parserinfo = None, *, default = None, ignoretz = None, tzinfos = None,
        dayfirst = None, yearfirst = None, fuzzy = None, fuzzy_with_tokens = None, **kwargs
    ),
    text_signature = "(timestr, parserinfo=None, **kwargs)"
)]
#[allow(clippy::too_many_arguments)]
fn upstream_parse<'py>(
    timestr: &Bound<'py, PyAny>,
    parserinfo: Option<&Bound<'py, PyAny>>,
    default: Option<&Bound<'py, PyAny>>,
    ignoretz: Option<&Bound<'py, PyAny>>,
    tzinfos: Option<&Bound<'py, PyAny>>,
    dayfirst: Option<&Bound<'py, PyAny>>,
    yearfirst: Option<&Bound<'py, PyAny>>,
    fuzzy: Option<&Bound<'py, PyAny>>,
    fuzzy_with_tokens: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = timestr.py();
    let u = upstream(py)?;
    let kw = ParseKw {
        default,
        ignoretz,
        tzinfos,
        dayfirst,
        yearfirst,
        fuzzy,
        fuzzy_with_tokens,
        rest: kwargs,
    };
    module_parse(
        u.module.bind(py),
        u.parser.bind(py),
        timestr,
        parserinfo,
        &kw,
    )
}

/// `dateutil.parser.isoparse` under `pyroid.patch("dateutil")`: dateutil's
/// `DEFAULT_ISOPARSER.isoparse`, parsed by pyroid when it can be.
#[pyfunction]
#[pyo3(name = "isoparse", signature = (*args, **kwargs), text_signature = "(dt_str)")]
fn upstream_isoparse<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = args.py();
    let h = hooks(py)?;
    let u = upstream(py)?;
    let globals = u.iso_module.bind(py);
    let p = globals
        .get_item(pyo3::intern!(py, "DEFAULT_ISOPARSER"))?
        .ok_or_else(|| {
            PyValueError::new_err("dateutil.parser.isoparser has no DEFAULT_ISOPARSER")
        })?;
    let Some(dt_str) = one_arg(args, kwargs)? else {
        return p.call_method(pyo3::intern!(py, "isoparse"), args, kwargs);
    };
    let dt_str = &dt_str;
    let native = is_type(&p, &u.isoparser)
        && globals
            .get_item(pyo3::intern!(py, "datetime"))?
            .is_some_and(|d| d.is(h.datetime_type.bind(py)));
    // A str subclass goes to dateutil's method: its `_takes_ascii` calls the object's own `read`
    // and `encode`, if it has them.
    if native {
        if let Ok(s) = dt_str.cast_exact::<PyString>() {
            if let Ok(text) = s.to_str() {
                let sep = p.getattr(pyo3::intern!(py, "_sep"))?;
                let sep = if sep.is_none() {
                    Some(None)
                } else {
                    sep.cast::<PyBytes>()
                        .ok()
                        .filter(|b| b.as_bytes().len() == 1)
                        .map(|b| Some(b.as_bytes()[0]))
                };
                if let (true, Some(sep)) = (text.is_ascii(), sep) {
                    if let Ok(iso) = iso::isoparse(text.as_bytes(), sep, u.iso_mode) {
                        if let Some(dt) = iso_build(py, h, &iso, &TzOffsets::default())? {
                            return Ok(dt);
                        }
                    }
                }
            }
        }
    }
    // dateutil's own method: its result or its exception.
    p.call_method1(pyo3::intern!(py, "isoparse"), (dt_str,))
}

// ---------------------------------------------------------------------------
// isoparse
// ---------------------------------------------------------------------------

#[allow(clippy::many_single_char_names)]
fn iso_build<'py>(
    py: Python<'py>,
    h: &Hooks,
    iso: &Iso,
    offsets: &TzOffsets<'py>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let f = iso.fields;
    let Some(mut ord) = kernel::cal::ordinal(f[0], f[1], f[2]) else {
        return Ok(None);
    };
    if !(0..24).contains(&f[3])
        || !(0..60).contains(&f[4])
        || !(0..60).contains(&f[5])
        || !(0..1_000_000).contains(&f[6])
    {
        return Ok(None);
    }
    if iso.plus_day {
        ord += 1;
    }
    let Some([y, m, d]) = kernel::cal::from_ordinal(ord) else {
        return Ok(None);
    };
    let tz = match iso.tz {
        None => py.None().into_bound(py),
        Some(IsoTz::Utc) => h.utc.bind(py).clone(),
        Some(IsoTz::Offset(s)) => offsets.get(py, h, None, false, s)?,
    };
    let dt = h
        .datetime_type
        .bind(py)
        .call1((y, m, d, f[3], f[4], f[5], f[6], tz))?;
    Ok(Some(dt))
}

/// `isoparser.isoparse` (after `_takes_ascii`) on ASCII bytes `b` for isoparser `p`.
fn iso_bytes<'py>(
    py: Python<'py>,
    h: &Hooks,
    p: &Bound<'py, PyAny>,
    b: &[u8],
    as_bytes: &dyn Fn() -> Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let reference =
        |reason: Option<String>| h.reference_isoparse.bind(py).call1((p, reason, as_bytes()));
    if let Some(r) = iso_override(h, p)? {
        return reference(Some(r));
    }
    let real = h
        .iso_module
        .bind(py)
        .get_item("datetime")?
        .is_some_and(|d| d.is(h.datetime_type.bind(py)));
    if !real {
        return reference(None);
    }
    let mode = if p.getattr(pyo3::intern!(py, "_lenient"))?.is_truthy()? {
        Mode::Lenient
    } else {
        Mode::Strict
    };
    let sep = p.getattr(pyo3::intern!(py, "_sep"))?;
    let sep = if sep.is_none() {
        None
    } else {
        match sep.cast::<PyBytes>() {
            Ok(s) if s.as_bytes().len() == 1 => Some(s.as_bytes()[0]),
            _ => return reference(None),
        }
    };
    match iso::isoparse(b, sep, mode) {
        Ok(iso) => match iso_build(py, h, &iso, &TzOffsets::default())? {
            Some(dt) => Ok(dt),
            None => reference(None),
        },
        // dateutil raises: its own code gives the exception.
        Err(_) => reference(None),
    }
}

/// The one argument of a call taken as `(*args, **kwargs)`, if it was called with exactly one
/// positional argument. Any other call goes to dateutil's method, so a wrong call raises its
/// `TypeError` (`isoparser.isoparse() missing 1 required positional argument: 'str_in'`).
fn one_arg<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if args.len() == 1 && kwargs.is_none_or(PyDictMethods::is_empty) {
        Ok(Some(args.get_item(0)?))
    } else {
        Ok(None)
    }
}

/// The module-level `isoparse(dt_str)` (`DEFAULT_ISOPARSER.isoparse`).
#[pyfunction]
#[pyo3(signature = (*args, **kwargs), text_signature = "(dt_str)")]
fn isoparse<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = args.py();
    let h = hooks(py)?;
    let p = h.default_isoparser.bind(py);
    let Some(dt_str) = one_arg(args, kwargs)? else {
        return p.call_method(pyo3::intern!(py, "isoparse"), args, kwargs);
    };
    let dt_str = &dt_str;
    // Exact str only: `_takes_ascii` calls a str subclass's own `read` and `encode`.
    if let Ok(s) = dt_str.cast_exact::<PyString>() {
        if let Ok(u) = s.to_str() {
            if u.is_ascii() {
                let mk = || PyBytes::new(py, u.as_bytes()).into_any();
                return iso_bytes(py, h, p, u.as_bytes(), &mk);
            }
        }
    }
    // Streams, bytes, str subclasses and non-ASCII strings: the method (dateutil's
    // `_takes_ascii`).
    p.call_method1(pyo3::intern!(py, "isoparse"), (dt_str,))
}

/// `isoparser.isoparse(self, dt_str)` after `_takes_ascii`.
#[pyfunction]
#[pyo3(signature = (self_, dt_str))]
fn isoparser_isoparse<'py>(
    self_: &Bound<'py, PyAny>,
    dt_str: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = self_.py();
    let h = hooks(py)?;
    match dt_str.cast::<PyBytes>() {
        Ok(b) => {
            let mk = || b.clone().into_any();
            iso_bytes(py, h, self_, b.as_bytes(), &mk)
        }
        Err(_) => h
            .reference_isoparse
            .bind(py)
            .call1((self_, py.None(), dt_str)),
    }
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

/// Rows the attached part of a batch builds between two signal checks.
const ROWS_PER_SIGNAL_CHECK: usize = 1024;

/// `par::map(n, threads, f)` detached for a batch: Ctrl-C stops it within
/// `par::POLL_INTERVAL` with the signal handler's exception (`KeyboardInterrupt`), not when
/// the batch ends.
fn map_checked<R, F>(py: Python<'_>, n: usize, threads: par::Threads, f: F) -> PyResult<Vec<R>>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    crate::interrupt::detach_batch(py, || par::map(n, threads, f))
        .map_err(|e| crate::interrupt::pool_error(&e))
}

/// `errors=`: raise the first row's exception, or give `None` for the rows that fail.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Errors {
    Raise,
    Coerce,
}

fn errors_arg(errors: &str) -> PyResult<Errors> {
    match errors {
        "raise" => Ok(Errors::Raise),
        "coerce" => Ok(Errors::Coerce),
        other => Err(PyValueError::new_err(format!(
            "errors must be 'raise' or 'coerce', not {other:?}"
        ))),
    }
}

/// One row's outcome: its value, or (coerce) `None` for a `ValueError`/`OverflowError`.
fn row_result<'py>(
    py: Python<'py>,
    errors: Errors,
    i: usize,
    r: PyResult<Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e)
            if errors == Errors::Coerce
                && (e.is_instance_of::<PyValueError>(py)
                    || e.is_instance_of::<pyo3::exceptions::PyOverflowError>(py)) =>
        {
            Ok(None)
        }
        Err(e) => Err(crate::errors::with_note(
            py,
            e,
            format!("while parsing item {i}"),
        )),
    }
}

/// A batch holding a str with lone surrogates, which pyroid's batch input refuses (`err`, its
/// `UnicodeEncodeError`): `call` on each str, as the module's single-string function takes it,
/// results in the input's kind. Only a `str`, `list` or `tuple` can hold one.
fn each_str<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
    err: PyErr,
    call: &dyn Fn(usize, Bound<'py, PyString>) -> PyResult<Option<Bound<'py, PyAny>>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = obj.py();
    let none = || py.None().into_bound(py);
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(call(0, s.clone())?.unwrap_or_else(none));
    }
    // A snapshot (atomic), so another thread changing the list cannot change the batch.
    let (items, tuple) = if let Ok(l) = obj.cast::<PyList>() {
        (l.to_tuple(), false)
    } else if let Ok(t) = obj.cast::<PyTuple>() {
        (t.clone(), true)
    } else {
        return Err(err);
    };
    let mut strs = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let Ok(s) = item.cast::<PyString>() else {
            let msg = format!(
                "expected str at index {i}, got {}",
                crate::errors::type_name(&item)
            );
            return Err(crate::input::arg_note(
                py,
                crate::errors::input_error(py, msg),
                name,
            ));
        };
        strs.push(s.clone());
    }
    let mut out = Vec::with_capacity(strs.len());
    for (i, s) in strs.into_iter().enumerate() {
        out.push(call(i, s)?.unwrap_or_else(none));
    }
    if tuple {
        Ok(PyTuple::new(py, out)?.into_any())
    } else {
        Ok(PyList::new(py, out)?.into_any())
    }
}

/// The batch's results in the caller's container kind (Arrow: a list for now).
fn output<'py>(
    py: Python<'py>,
    input: &crate::input::TextInput,
    out: Vec<Option<Bound<'py, PyAny>>>,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::input::TextInput;
    let items = out
        .into_iter()
        .map(|v| v.unwrap_or_else(|| py.None().into_bound(py)));
    match input {
        TextInput::Str(_) => Ok(items
            .into_iter()
            .next()
            .unwrap_or_else(|| py.None().into_bound(py))),
        TextInput::Tuple(_) => Ok(PyTuple::new(py, items)?.into_any()),
        TextInput::List(_) | TextInput::Arrow(_) => Ok(PyList::new(py, items)?.into_any()),
    }
}

/// A `timestamp[us]` column of `values` (`None`: null): naive when every value is, UTC
/// instants when every value is aware; `ValueError` when they are mixed.
fn timestamp_column<'py>(
    py: Python<'py>,
    input: &crate::input::TextInput,
    values: &[Instant],
    func: &str,
    hint: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use crate::arrow::{Chunk, Column, ColumnType};
    use pyroid_core::columnar::I64Buffers;
    let crate::input::TextInput::Arrow(strings) = input else {
        unreachable!("Arrow input")
    };
    let aware = values.iter().flatten().any(|&(_, a)| a);
    if aware && values.iter().flatten().any(|&(_, a)| !a) {
        return Err(PyValueError::new_err(format!(
            "pyroid.dateutil.parser.{func}: the results mix naive and aware datetimes, which one \
             Arrow timestamp column cannot hold; {hint}"
        )));
    }
    let b = I64Buffers::build(
        &values
            .iter()
            .map(|v| v.map(|(us, _)| us))
            .collect::<Vec<_>>(),
    );
    let (ty, chunk) = if aware {
        (ColumnType::TimestampUsUtc, Chunk::TimestampUsUtc(b))
    } else {
        (ColumnType::TimestampUs, Chunk::TimestampUs(b))
    };
    Ok(Bound::new(py, Column::new(ty, &strings.field.name, vec![chunk]))?.into_any())
}

/// One row's value in the shape the output needs: a Python object, or an Arrow timestamp.
fn row_value(
    arrow: bool,
    v: Option<Bound<'_, PyAny>>,
) -> PyResult<(Option<Bound<'_, PyAny>>, Instant)> {
    match v {
        None => Ok((None, None)),
        Some(v) if arrow => Ok((None, py_instant(&v)?)),
        Some(v) => Ok((Some(v), None)),
    }
}

/// `parse_batch(timestrs, parserinfo, threads, errors, kwargs)`.
#[pyfunction]
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
#[pyo3(signature = (timestrs, parserinfo, threads, errors, kwargs))]
fn parse_batch<'py>(
    timestrs: &Bound<'py, PyAny>,
    parserinfo: &Bound<'py, PyAny>,
    threads: Option<Threads>,
    errors: &str,
    kwargs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = timestrs.py();
    let h = hooks(py)?;
    let errors = errors_arg(errors)?;
    let module = h.module.bind(py);
    let p = if parserinfo.is_truthy()? {
        item(module, "parser")?.call1((parserinfo,))?
    } else {
        item(module, "DEFAULTPARSER")?
    };
    let mut default = py.None().into_bound(py);
    let mut ignoretz = false;
    let mut tzinfos = py.None().into_bound(py);
    let rest = PyDict::new(py);
    for (k, v) in kwargs.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "default" => default = v,
            "ignoretz" => ignoretz = v.is_truthy()?,
            "tzinfos" => tzinfos = v,
            "dayfirst" | "yearfirst" | "fuzzy" => rest.set_item(k, v)?,
            "fuzzy_with_tokens" => {
                return Err(PyTypeError::new_err(
                    "parse_batch() does not take fuzzy_with_tokens (use parse())",
                ));
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "parse_batch() got an unexpected keyword argument '{key}'"
                )));
            }
        }
    }
    if default.is_none() {
        // dateutil reads today's date per call; a batch reads it once.
        default = today(py, h)?;
    }
    // The rows the kernel cannot take, and anything else odd, go through `p.parse` per row.
    let call = |i: usize, s: Bound<'py, PyString>| -> PyResult<Option<Bound<'py, PyAny>>> {
        let r = p.call_method(
            pyo3::intern!(py, "parse"),
            (s,),
            Some(&{
                let kw = rest.copy()?;
                kw.set_item("default", &default)?;
                kw.set_item("ignoretz", ignoretz)?;
                kw.set_item("tzinfos", &tzinfos)?;
                kw
            }),
        );
        row_result(py, errors, i, r)
    };
    let one = |i: usize, s: &str| call(i, PyString::new(py, s));
    let input = match crate::input::text_arg(timestrs, threads, "timestrs") {
        Ok(input) => input,
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            return each_str(timestrs, "timestrs", e, &call);
        }
        Err(e) => return Err(e),
    };
    let rows = crate::batch::rows(&input).map_err(|e| e.into_pyerr(py))?;
    let opts = private_kwargs(Some(&rest))?.expect("only _parse's keywords");
    let info = p.getattr(pyo3::intern!(py, "info"))?;
    let slow_reason = match parser_override(py, h, &p)? {
        Some(r) => Some(r),
        None => info_override(py, h, &info)?,
    };
    let compiled = if slow_reason.is_none() {
        let (c, s) = info_of(h, &info)?;
        match (&*c, s) {
            (Compiled::Ok(_), Some(s)) => Some((c.clone(), s)),
            _ => None,
        }
    } else {
        None
    };
    let arrow = matches!(input, crate::input::TextInput::Arrow(_));
    let mut out = Vec::with_capacity(if arrow { 0 } else { rows.len() });
    let mut stamps: Vec<Instant> = Vec::with_capacity(if arrow { rows.len() } else { 0 });
    let Some((compiled, settings)) = compiled else {
        for (i, r) in rows.iter().enumerate() {
            let v = match r {
                None => None,
                Some(s) => one(i, s)?,
            };
            let (o, t) = row_value(arrow, v)?;
            out.push(o);
            stamps.push(t);
        }
        return if arrow {
            timestamp_column(
                py,
                &input,
                &stamps,
                "parse_batch",
                "pass ignoretz=True, or a list",
            )
        } else {
            output(py, &input, out)
        };
    };
    let Compiled::Ok(tables) = &*compiled else {
        unreachable!("checked above")
    };
    let mut opts = opts;
    let longest = rows.iter().flatten().map(|s| s.len()).max().unwrap_or(0);
    opts.max_str_digits = max_str_digits(py, longest)?;
    let pool = pool_threads(threads);
    let runs = map_checked(py, rows.len(), pool, |i| {
        rows[i].map(|s| run_kernel(&Text::Utf8(s), tables, settings, opts))
    })?;
    let fin = Finisher::new(
        h,
        &p,
        &default,
        ignoretz,
        &tzinfos,
        Some(&rest),
        &tables.utczone_list,
    );
    let coerce_none =
        |i: usize, e: PyErr| -> PyResult<()> { row_result(py, errors, i, Err(e)).map(|_| ()) };
    for (i, run) in runs.into_iter().enumerate() {
        if i % ROWS_PER_SIGNAL_CHECK == ROWS_PER_SIGNAL_CHECK - 1 {
            py.check_signals()?;
        }
        let Some(run) = run else {
            out.push(None);
            stamps.push(None);
            continue;
        };
        let s = rows[i].expect("a run per row");
        match checked(py, h, run)? {
            Run::Parsed(parsed, remapped) => {
                let timestr = PyString::new(py, s).into_any();
                if arrow {
                    match fin.instant(&parsed, remapped, &timestr) {
                        Ok(t) => stamps.push(t),
                        Err(e) => {
                            coerce_none(i, e)?;
                            stamps.push(None);
                        }
                    }
                } else {
                    out.push(row_result(
                        py,
                        errors,
                        i,
                        fin.finish(&parsed, remapped, &timestr),
                    )?);
                }
            }
            Run::Unknown => {
                let timestr = PyString::new(py, s).into_any();
                coerce_none(
                    i,
                    parser_error(py, h, "Unknown string format: %s", &timestr),
                )?;
                out.push(None);
                stamps.push(None);
            }
            Run::Reference(_) => {
                let (o, t) = row_value(arrow, one(i, s)?)?;
                out.push(o);
                stamps.push(t);
            }
        }
    }
    if arrow {
        timestamp_column(
            py,
            &input,
            &stamps,
            "parse_batch",
            "pass ignoretz=True, or a list",
        )
    } else {
        output(py, &input, out)
    }
}

/// `isoparse_batch(dt_strs, isoparser, threads, errors)`.
#[pyfunction]
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
#[pyo3(signature = (dt_strs, isoparser, threads, errors))]
fn isoparse_batch<'py>(
    dt_strs: &Bound<'py, PyAny>,
    isoparser: &Bound<'py, PyAny>,
    threads: Option<Threads>,
    errors: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let py = dt_strs.py();
    let h = hooks(py)?;
    let errors = errors_arg(errors)?;
    let call = |i: usize, s: Bound<'py, PyString>| -> PyResult<Option<Bound<'py, PyAny>>> {
        let r = isoparser.call_method1(pyo3::intern!(py, "isoparse"), (s,));
        row_result(py, errors, i, r)
    };
    let one = |i: usize, s: &str| call(i, PyString::new(py, s));
    let input = match crate::input::text_arg(dt_strs, threads, "dt_strs") {
        Ok(input) => input,
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            return each_str(dt_strs, "dt_strs", e, &call);
        }
        Err(e) => return Err(e),
    };
    let rows = crate::batch::rows(&input).map_err(|e| e.into_pyerr(py))?;
    let real = h
        .iso_module
        .bind(py)
        .get_item("datetime")?
        .is_some_and(|d| d.is(h.datetime_type.bind(py)));
    let sep = isoparser.getattr(pyo3::intern!(py, "_sep"))?;
    let sep_byte = if sep.is_none() {
        Some(None)
    } else {
        sep.cast::<PyBytes>()
            .ok()
            .filter(|b| b.as_bytes().len() == 1)
            .map(|b| Some(b.as_bytes()[0]))
    };
    let mode = if isoparser
        .getattr(pyo3::intern!(py, "_lenient"))?
        .is_truthy()?
    {
        Mode::Lenient
    } else {
        Mode::Strict
    };
    let native = real && iso_override(h, isoparser)?.is_none() && sep_byte.is_some();
    let arrow = matches!(input, crate::input::TextInput::Arrow(_));
    let mut out = Vec::with_capacity(if arrow { 0 } else { rows.len() });
    let mut stamps: Vec<Instant> = Vec::with_capacity(if arrow { rows.len() } else { 0 });
    if !native {
        for (i, r) in rows.iter().enumerate() {
            let v = match r {
                None => None,
                Some(s) => one(i, s)?,
            };
            let (o, t) = row_value(arrow, v)?;
            out.push(o);
            stamps.push(t);
        }
        return if arrow {
            timestamp_column(py, &input, &stamps, "isoparse_batch", "pass a list")
        } else {
            output(py, &input, out)
        };
    }
    let sep = sep_byte.expect("checked");
    let pool = pool_threads(threads);
    let runs = map_checked(py, rows.len(), pool, |i| {
        rows[i].map(|s| {
            if s.is_ascii() {
                iso::isoparse(s.as_bytes(), sep, mode).ok()
            } else {
                None
            }
        })
    })?;
    let offsets = TzOffsets::default();
    for (i, run) in runs.into_iter().enumerate() {
        if i % ROWS_PER_SIGNAL_CHECK == ROWS_PER_SIGNAL_CHECK - 1 {
            py.check_signals()?;
        }
        let Some(s) = rows[i] else {
            out.push(None);
            stamps.push(None);
            continue;
        };
        let iso = run.flatten();
        if arrow {
            if let Some(t) = iso.as_ref().and_then(iso_instant) {
                stamps.push(Some(t));
                continue;
            }
        }
        let built = match iso {
            Some(iso) => iso_build(py, h, &iso, &offsets)?,
            None => None,
        };
        let v = match built {
            Some(dt) => Some(dt),
            None => one(i, s)?,
        };
        let (o, t) = row_value(arrow, v)?;
        out.push(o);
        stamps.push(t);
    }
    if arrow {
        timestamp_column(py, &input, &stamps, "isoparse_batch", "pass a list")
    } else {
        output(py, &input, out)
    }
}

/// An isoparse result as an Arrow timestamp, when `datetime` accepts it.
#[allow(clippy::many_single_char_names)]
fn iso_instant(iso: &Iso) -> Option<(i64, bool)> {
    let f = iso.fields;
    let mut ord = kernel::cal::ordinal(f[0], f[1], f[2])?;
    if !(0..24).contains(&f[3])
        || !(0..60).contains(&f[4])
        || !(0..60).contains(&f[5])
        || !(0..1_000_000).contains(&f[6])
    {
        return None;
    }
    if iso.plus_day {
        ord += 1;
    }
    let [y, m, d] = kernel::cal::from_ordinal(ord)?;
    let naive = micros(&[y, m, d, f[3], f[4], f[5], f[6]])?;
    Some(match iso.tz {
        None => (naive, false),
        Some(IsoTz::Utc) => (naive, true),
        Some(IsoTz::Offset(s)) => (naive - s * 1_000_000, true),
    })
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.dateutil")?;
    let up = PyModule::new(parent.py(), "pyroid.dateutil.upstream")?;
    up.gil_used(false)?;
    up.add_function(wrap_pyfunction!(upstream_parse, &up)?)?;
    up.add_function(wrap_pyfunction!(upstream_isoparse, &up)?)?;
    m.add_submodule(&up)?;
    m.add_function(wrap_pyfunction!(_install_upstream, &m)?)?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(supported_unicode_versions, &m)?)?;
    m.add_function(wrap_pyfunction!(_set_unicode_version, &m)?)?;
    m.add_function(wrap_pyfunction!(_install, &m)?)?;
    m.add_function(wrap_pyfunction!(_info_changed, &m)?)?;
    m.add_function(wrap_pyfunction!(parse, &m)?)?;
    m.add_function(wrap_pyfunction!(parser_parse, &m)?)?;
    m.add_function(wrap_pyfunction!(parser_private_parse, &m)?)?;
    m.add_function(wrap_pyfunction!(isoparse, &m)?)?;
    m.add_function(wrap_pyfunction!(isoparser_isoparse, &m)?)?;
    m.add_function(wrap_pyfunction!(parse_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(isoparse_batch, &m)?)?;
    parent.add_submodule(&m)
}
