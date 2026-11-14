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

use std::cell::OnceCell;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use pyo3::exceptions::{PyTypeError, PyUnicodeEncodeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString, PyTuple, PyType};
use pyroid_core::dateutil::info::Table;
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
    dateutil_parserinfo: Option<Py<PyType>>,
    isoparser: Py<PyType>,
    default_isoparser: Py<PyAny>,
    iso_module: Py<PyDict>,
    parser_error: Py<PyAny>,
    unknown_tz: Py<PyAny>,
    utc: Py<PyAny>,
    tzoffset: Py<PyAny>,
    finish: Py<PyAny>,
    make_result: Py<PyAny>,
    reference_parse: Py<PyAny>,
    reference_private_parse: Py<PyAny>,
    reference_isoparse: Py<PyAny>,
    read_stream: Py<PyAny>,
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
        finish: item(d, "finish")?.unbind(),
        make_result: item(d, "make_result")?.unbind(),
        reference_parse: item(d, "reference_parse")?.unbind(),
        reference_private_parse: item(d, "reference_private_parse")?.unbind(),
        reference_isoparse: item(d, "reference_isoparse")?.unbind(),
        read_stream: item(d, "read_stream")?.unbind(),
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

/// Per type: the hook it overrides (`None`: none). Holds the types so ids stay unique.
type TypeCache = Mutex<FxHashMap<usize, (Py<PyType>, Option<String>)>>;
static PARSER_TYPES: OnceLock<TypeCache> = OnceLock::new();
static PRIVATE_TYPES: OnceLock<TypeCache> = OnceLock::new();
static INFO_TYPES: OnceLock<TypeCache> = OnceLock::new();
static ISO_TYPES: OnceLock<TypeCache> = OnceLock::new();

fn cached_override(
    cache: &'static OnceLock<TypeCache>,
    ty: &Bound<'_, PyType>,
    check: impl FnOnce() -> PyResult<Option<String>>,
) -> PyResult<Option<String>> {
    let cache = cache.get_or_init(TypeCache::default);
    let key = ty.as_ptr() as usize;
    if let Some((_, r)) = cache
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&key)
    {
        return Ok(r.clone());
    }
    let r = check()?;
    let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if c.len() > 256 {
        c.clear();
    }
    c.insert(key, (ty.clone().unbind(), r.clone()));
    Ok(r)
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
    let ty = obj.get_type();
    if ty.is(h.parser.bind(py)) {
        return Ok(None);
    }
    cached_override(&PARSER_TYPES, &ty, || {
        first_override(&ty, &h.parser_methods, "parser")
    })
}

fn private_override(py: Python<'_>, h: &Hooks, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let ty = obj.get_type();
    if ty.is(h.parser.bind(py)) {
        return Ok(None);
    }
    cached_override(&PRIVATE_TYPES, &ty, || {
        first_override(&ty, &h.private_methods, "parser")
    })
}

fn info_override(py: Python<'_>, h: &Hooks, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let ty = obj.get_type();
    if ty.is(h.parserinfo.bind(py))
        || h.dateutil_parserinfo
            .as_ref()
            .is_some_and(|d| ty.is(d.bind(py)))
    {
        return Ok(None);
    }
    cached_override(&INFO_TYPES, &ty, || {
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

fn iso_override(py: Python<'_>, h: &Hooks, obj: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let ty = obj.get_type();
    if ty.is(h.isoparser.bind(py)) {
        return Ok(None);
    }
    cached_override(&ISO_TYPES, &ty, || {
        first_override(&ty, &h.iso_methods, "isoparser")
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

struct InfoEntry {
    /// The 7 table dicts, `TZOFFSET` and `UTCZONE` (held, so their ids stay theirs).
    objs: Vec<Py<PyAny>>,
    lens: Vec<usize>,
    compiled: Arc<Compiled>,
}

static INFO_CACHE: OnceLock<Mutex<FxHashMap<usize, InfoEntry>>> = OnceLock::new();

fn container_len(o: &Bound<'_, PyAny>) -> usize {
    o.len().unwrap_or(usize::MAX)
}

fn table_of(d: &Bound<'_, PyAny>, valued: bool, what: &str) -> Result<Table, String> {
    let d = d
        .cast::<PyDict>()
        .map_err(|_| format!("a parserinfo whose {what} is not a dict"))?;
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
                .map_err(|_| format!("a parserinfo {what} value that is not an int"))?
        } else {
            0
        };
        t.insert(k.into(), v);
    }
    Ok(t)
}

fn compile(objs: &[Bound<'_, PyAny>]) -> Compiled {
    let build = || -> Result<ParserInfo, String> {
        let valued = [false, true, true, true, true, false, false];
        let mut tables = Vec::with_capacity(7);
        for (i, o) in objs[..7].iter().enumerate() {
            tables.push(table_of(o, valued[i], TABLE_ATTRS[i])?);
        }
        let tzd = objs[7]
            .cast::<PyDict>()
            .map_err(|_| "a parserinfo whose TZOFFSET is not a dict".to_owned())?;
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
                .map_err(|_| "a TZOFFSET value that is not an int".to_owned())?
                .into();
            tzoffset.insert(k.into(), v);
        }
        let ut = &objs[8];
        if !(ut.is_instance_of::<PyList>() || ut.is_instance_of::<PyTuple>()) {
            return Err("a parserinfo whose UTCZONE is not a list".to_owned());
        }
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
        Ok(ParserInfo {
            jump: next(),
            weekdays: next(),
            months: next(),
            hms: next(),
            ampm: next(),
            utczone: next(),
            pertain: next(),
            tzoffset,
            utczone_list,
        })
    };
    match build() {
        Ok(i) => Compiled::Ok(Box::new(i)),
        Err(e) => Compiled::Unsupported(e),
    }
}

/// The compiled tables and settings of parserinfo `info`.
fn info_of(info: &Bound<'_, PyAny>) -> PyResult<(Arc<Compiled>, Option<Settings>)> {
    let py = info.py();
    let mut objs = Vec::with_capacity(9);
    for a in TABLE_ATTRS {
        objs.push(info.getattr(a)?);
    }
    objs.push(info.getattr(pyo3::intern!(py, "TZOFFSET"))?);
    objs.push(info.getattr(pyo3::intern!(py, "UTCZONE"))?);
    let lens: Vec<usize> = objs.iter().map(container_len).collect();
    let settings = (|| -> PyResult<Option<Settings>> {
        let dayfirst = info.getattr(pyo3::intern!(py, "dayfirst"))?.is_truthy()?;
        let yearfirst = info.getattr(pyo3::intern!(py, "yearfirst"))?.is_truthy()?;
        let year = info.getattr(pyo3::intern!(py, "_year"))?;
        let century = info.getattr(pyo3::intern!(py, "_century"))?;
        let int = |o: &Bound<'_, PyAny>| -> Option<Int> {
            if o.is_instance_of::<pyo3::types::PyBool>() {
                return None;
            }
            o.extract::<i64>().ok().map(Int::from)
        };
        Ok(int(&year)
            .zip(int(&century))
            .map(|(year, century)| Settings {
                dayfirst,
                yearfirst,
                year,
                century,
            }))
    })()?;
    let key = info.as_ptr() as usize;
    let cache = INFO_CACHE.get_or_init(Mutex::default);
    {
        let c = cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(e) = c.get(&key) {
            if e.lens == lens && e.objs.iter().zip(&objs).all(|(a, b)| a.is(b)) {
                return Ok((e.compiled.clone(), settings));
            }
        }
    }
    let compiled = Arc::new(compile(&objs));
    let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if c.len() > 64 {
        c.clear();
    }
    c.insert(
        key,
        InfoEntry {
            objs: objs.into_iter().map(Bound::unbind).collect(),
            lens,
            compiled: compiled.clone(),
        },
    );
    Ok((compiled, settings))
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

/// What `_timelex` reads from `timestr`: a str to tokenize, or a replay of an odd stream's
/// reads (for the reference).
enum Source<'py> {
    Str(Bound<'py, PyString>),
    Replay(Bound<'py, PyAny>),
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
        Ok(s) => Ok(Source::Str(s)),
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
fn checked(py: Python<'_>, h: &Hooks, run: Run) -> PyResult<Run> {
    if let Run::Parsed(p, _) = &run {
        if p.decimal_precision > 0 {
            let prec: u32 = h
                .getcontext
                .bind(py)
                .call0()?
                .getattr(pyo3::intern!(py, "prec"))?
                .extract()?;
            if prec < p.decimal_precision {
                return Ok(Run::Reference(format!(
                    "a decimal context precision below {}",
                    p.decimal_precision
                )));
            }
        }
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
        *o = i64::try_from(v).ok()?;
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
            let w = i64::try_from(w).ok().filter(|w| (-7..7).contains(w))?;
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
    kwargs: &'a Bound<'py, PyDict>,
    /// `default`'s fields (today's time, tz and fold for `default=None`); `None`: the fast
    /// path cannot be used (another `default` type, or a patched `datetime.datetime`).
    base: OnceCell<Option<DefaultFields<'py>>>,
    /// Today's date, for `default=None`.
    today: OnceCell<[Int; 3]>,
    /// `time.tzname`.
    local_names: OnceCell<Bound<'py, PyAny>>,
}

impl<'py, 'a> Finisher<'py, 'a> {
    fn new(
        h: &'a Hooks,
        self_: &'a Bound<'py, PyAny>,
        default: &'a Bound<'py, PyAny>,
        ignoretz: bool,
        tzinfos: &'a Bound<'py, PyAny>,
        kwargs: &'a Bound<'py, PyDict>,
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
        }
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
        } else if self.default.get_type().is(dt_type) {
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
                return Ok(None); // tzlocal() and the fold logic
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
            self.kwargs,
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
            TzPlan::Offset(off) => {
                let name = match &p.res.tzname {
                    Some(n) => py_text(py, n, remapped)?.into_any(),
                    None => py.None().into_bound(py),
                };
                h.tzoffset.bind(py).call1((name, off))?
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
            let naive = micros(&fast.f);
            match fast.tz {
                TzPlan::Utc => return Ok(Some((naive, true))),
                TzPlan::Offset(off) if off.abs() < 86_400 => {
                    let off = i64::try_from(off).expect("small");
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

/// How `_build_tzaware` ends on the fast path.
#[derive(Clone, Copy)]
enum TzPlan {
    /// No time zone from the string: `default`'s tzinfo (or none).
    Default,
    Utc,
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

/// Microseconds since 1970-01-01T00:00 of valid naive fields.
fn micros(f: &[i64; 7]) -> i64 {
    let days = kernel::cal::ordinal(f[0], f[1], f[2]).expect("valid") - 719_163;
    days * 86_400_000_000 + f[3] * 3_600_000_000 + f[4] * 60_000_000 + f[5] * 1_000_000 + f[6]
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
    let naive = micros(&f);
    let off = dt.call_method0(pyo3::intern!(py, "utcoffset"))?;
    if off.is_none() {
        return Ok(Some((naive, false)));
    }
    let off_us = off.getattr("days")?.extract::<i64>()? * 86_400_000_000
        + off.getattr("seconds")?.extract::<i64>()? * 1_000_000
        + off.getattr("microseconds")?.extract::<i64>()?;
    Ok(Some((naive - off_us, true)))
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
    let py = self_.py();
    let h = hooks(py)?;
    let reference = |reason: Option<String>, timestr: &Bound<'py, PyAny>| {
        h.reference_parse
            .bind(py)
            .call1((self_, reason, timestr, default, ignoretz, tzinfos, kwargs))
    };
    if let Some(r) = parser_override(py, h, self_)? {
        return reference(Some(r), timestr);
    }
    let Some(mut opts) = private_kwargs(Some(kwargs))? else {
        // An unexpected keyword: dateutil's _parse signature raises the TypeError.
        return reference(None, timestr);
    };
    let info = self_.getattr(pyo3::intern!(py, "info"))?;
    if let Some(r) = info_override(py, h, &info)? {
        return reference(Some(r), timestr);
    }
    let (compiled, settings) = info_of(&info)?;
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
    let text = match source(h, timestr)? {
        Source::Str(s) => s,
        Source::Replay(r) => return reference(Some("this stream".to_owned()), &r),
    };
    let t = text_of(&text)?;
    opts.max_str_digits = max_str_digits(py, t.as_str().len())?;
    let run = checked(py, h, run_kernel(&t, tables, settings, opts))?;
    match run {
        Run::Unknown => Err(parser_error(py, h, "Unknown string format: %s", timestr)),
        Run::Reference(r) => reference(Some(r), timestr),
        Run::Parsed(p, remapped) => {
            Finisher::new(h, self_, default, ignoretz.is_truthy()?, tzinfos, kwargs)
                .finish(&p, remapped, timestr)
        }
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
    if let Some(r) = private_override(py, h, self_)? {
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
    let (compiled, settings) = info_of(&info)?;
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
    let text = match source(h, timestr)? {
        Source::Str(s) => s,
        Source::Replay(r) => return reference("this stream".to_owned(), &r),
    };
    let t = text_of(&text)?;
    opts.max_str_digits = max_str_digits(py, t.as_str().len())?;
    match checked(py, h, run_kernel(&t, tables, settings, opts))? {
        Run::Unknown => Ok(PyTuple::new(py, [py.None(), py.None()])?.into_any()),
        Run::Reference(r) => reference(r, timestr),
        Run::Parsed(p, remapped) => {
            let res = result_object(py, h, &p.res, remapped)?;
            Ok(PyTuple::new(py, [res, skipped_tuple(py, &p, remapped)?])?.into_any())
        }
    }
}

/// dateutil's module-level `parse(timestr, parserinfo=None, **kwargs)`.
#[pyfunction]
#[pyo3(signature = (timestr, parserinfo = None, **kwargs))]
fn parse<'py>(
    timestr: &Bound<'py, PyAny>,
    parserinfo: Option<&Bound<'py, PyAny>>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = timestr.py();
    let h = hooks(py)?;
    let module = h.module.bind(py);
    if let Some(info) = parserinfo {
        if info.is_truthy()? {
            let cls = item(module, "parser")?;
            return cls
                .call1((info,))?
                .call_method(pyo3::intern!(py, "parse"), (timestr,), kwargs);
        }
    }
    let dp = item(module, "DEFAULTPARSER")?;
    if !dp.get_type().is(h.parser.bind(py)) {
        return dp.call_method(pyo3::intern!(py, "parse"), (timestr,), kwargs);
    }
    let none = py.None().into_bound(py);
    let mut default = none.clone();
    let mut ignoretz = pyo3::types::PyBool::new(py, false).to_owned().into_any();
    let mut tzinfos = none;
    let rest = PyDict::new(py);
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            match k.cast::<PyString>().ok().map(|k| k.to_str()).transpose()? {
                Some("default") => default = v,
                Some("ignoretz") => ignoretz = v,
                Some("tzinfos") => tzinfos = v,
                Some("dayfirst" | "yearfirst" | "fuzzy" | "fuzzy_with_tokens") => {
                    rest.set_item(k, v)?;
                }
                _ => return dp.call_method(pyo3::intern!(py, "parse"), (timestr,), kwargs),
            }
        }
    }
    parser_parse(&dp, timestr, &default, &ignoretz, &tzinfos, &rest)
}

// ---------------------------------------------------------------------------
// isoparse
// ---------------------------------------------------------------------------

#[allow(clippy::many_single_char_names)]
fn iso_build<'py>(py: Python<'py>, h: &Hooks, iso: &Iso) -> PyResult<Option<Bound<'py, PyAny>>> {
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
        Some(IsoTz::Offset(s)) => h.tzoffset.bind(py).call1((py.None(), s))?,
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
    if let Some(r) = iso_override(py, h, p)? {
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
        Ok(iso) => match iso_build(py, h, &iso)? {
            Some(dt) => Ok(dt),
            None => reference(None),
        },
        // dateutil raises: its own code gives the exception.
        Err(_) => reference(None),
    }
}

/// The module-level `isoparse(dt_str)` (`DEFAULT_ISOPARSER.isoparse`).
#[pyfunction]
fn isoparse<'py>(dt_str: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = dt_str.py();
    let h = hooks(py)?;
    let p = h.default_isoparser.bind(py);
    if let Ok(s) = dt_str.cast::<PyString>() {
        if let Ok(u) = s.to_str() {
            if u.is_ascii() {
                let mk = || PyBytes::new(py, u.as_bytes()).into_any();
                return iso_bytes(py, h, p, u.as_bytes(), &mk);
            }
        }
    }
    // Streams, bytes and non-ASCII strings: the method (dateutil's `_takes_ascii`).
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
    let input = crate::input::text_arg(timestrs, threads, "timestrs")?;
    let rows = crate::batch::rows(&input).map_err(|e| e.into_pyerr(py))?;
    let opts = private_kwargs(Some(&rest))?.expect("only _parse's keywords");
    let info = p.getattr(pyo3::intern!(py, "info"))?;
    let slow_reason = match parser_override(py, h, &p)? {
        Some(r) => Some(r),
        None => info_override(py, h, &info)?,
    };
    let compiled = if slow_reason.is_none() {
        let (c, s) = info_of(&info)?;
        match (&*c, s) {
            (Compiled::Ok(_), Some(s)) => Some((c.clone(), s)),
            _ => None,
        }
    } else {
        None
    };
    // The rows the kernel cannot take, and anything else odd, go through `p.parse` per row.
    let one = |i: usize, s: &str| -> PyResult<Option<Bound<'py, PyAny>>> {
        let r = p.call_method(
            pyo3::intern!(py, "parse"),
            (PyString::new(py, s),),
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
    let runs = py
        .detach(|| {
            par::map(rows.len(), pool, |i| {
                rows[i].map(|s| run_kernel(&Text::Utf8(s), tables, settings, opts))
            })
        })
        .map_err(|e| crate::batch::BatchError::from(e).into_pyerr(py))?;
    let fin = Finisher::new(h, &p, &default, ignoretz, &tzinfos, &rest);
    let coerce_none =
        |i: usize, e: PyErr| -> PyResult<()> { row_result(py, errors, i, Err(e)).map(|_| ()) };
    for (i, run) in runs.into_iter().enumerate() {
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
#[allow(clippy::many_single_char_names)]
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
    let input = crate::input::text_arg(dt_strs, threads, "dt_strs")?;
    let rows = crate::batch::rows(&input).map_err(|e| e.into_pyerr(py))?;
    let one = |i: usize, s: &str| -> PyResult<Option<Bound<'py, PyAny>>> {
        let r = isoparser.call_method1(pyo3::intern!(py, "isoparse"), (PyString::new(py, s),));
        row_result(py, errors, i, r)
    };
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
    let native = real && iso_override(py, h, isoparser)?.is_none() && sep_byte.is_some();
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
    let runs = py
        .detach(|| {
            par::map(rows.len(), pool, |i| {
                rows[i].map(|s| {
                    if s.is_ascii() {
                        iso::isoparse(s.as_bytes(), sep, mode).ok()
                    } else {
                        None
                    }
                })
            })
        })
        .map_err(|e| crate::batch::BatchError::from(e).into_pyerr(py))?;
    for (i, run) in runs.into_iter().enumerate() {
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
            Some(iso) => iso_build(py, h, &iso)?,
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
    let naive = micros(&[y, m, d, f[3], f[4], f[5], f[6]]);
    Some(match iso.tz {
        None => (naive, false),
        Some(IsoTz::Utc) => (naive, true),
        Some(IsoTz::Offset(s)) => (naive - s * 1_000_000, true),
    })
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.dateutil")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(supported_unicode_versions, &m)?)?;
    m.add_function(wrap_pyfunction!(_set_unicode_version, &m)?)?;
    m.add_function(wrap_pyfunction!(_install, &m)?)?;
    m.add_function(wrap_pyfunction!(parse, &m)?)?;
    m.add_function(wrap_pyfunction!(parser_parse, &m)?)?;
    m.add_function(wrap_pyfunction!(parser_private_parse, &m)?)?;
    m.add_function(wrap_pyfunction!(isoparse, &m)?)?;
    m.add_function(wrap_pyfunction!(isoparser_isoparse, &m)?)?;
    m.add_function(wrap_pyfunction!(parse_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(isoparse_batch, &m)?)?;
    parent.add_submodule(&m)
}
