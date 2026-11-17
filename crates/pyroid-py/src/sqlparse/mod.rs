//! `pyroid._pyroid.sqlparse`: the native side of `pyroid.sqlparse` (sqlparse 0.6.1).
//!
//! The Python package hands over sqlparse's own token classes and token types once
//! (`_configure`); `tokenize`, `split`, `format` and `parse` then run the Rust port and build
//! real objects of those classes (`cls.__new__(cls)` and the slots set directly). The lexer's
//! table is read from the `Lexer` instance the caller passes, compiled once and cached until
//! its lists change. Every entry point returns `NotImplemented` when the Python package must
//! run sqlparse's own code instead (a lexer it cannot compile, a step where sqlparse's code
//! raises, a text holding both lone surrogates and U+10F800..).

#![allow(clippy::many_single_char_names)]

mod group;

use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyTypeError, PyUnicodeEncodeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple, PyType};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::json_repair::chars as wtf8;
use pyroid_core::sqlparse::engine::{self, Error as EngineError, FormatOptions, Stream};
use pyroid_core::sqlparse::filters::{Case, Output, TokenFilters};
use pyroid_core::sqlparse::grouping::{Grouper, Limits};
use pyroid_core::sqlparse::lexer::{self as core_lexer, Action, Config, Entry, External, Matcher};
use pyroid_core::sqlparse::regex::Regex;
use pyroid_core::sqlparse::reindent::ReindentOptions;
use pyroid_core::sqlparse::tree::{Class, Fail, Id, NO_LIST, Str, Tree};
use pyroid_core::sqlparse::ttype::{self, BUILTIN_PATHS, TT};
use rustc_hash::FxHashMap;

// ---------------------------------------------------------------------------
// Set-up: the classes and token types of the `sql` and `tokens` modules in use
// ---------------------------------------------------------------------------

struct Setup {
    classes: Vec<Py<PyType>>,
    new: Py<PyAny>,
    tokentype: Py<PyType>,
    process_as_keyword: Py<PyAny>,
    error: Py<PyType>,
    pattern: Py<PyType>,
    pause: Py<PyAny>,
    version: UnicodeVersion,
}

static SETUP: OnceLock<Setup> = OnceLock::new();

/// The token-type objects by id, and ids by object address (holding a reference, so
/// addresses stay valid).
#[derive(Default)]
struct TypeTable {
    by_id: Arc<Vec<Option<Py<PyAny>>>>,
    by_ptr: FxHashMap<usize, TT>,
}

static TYPES: Mutex<Option<TypeTable>> = Mutex::new(None);

fn setup() -> PyResult<&'static Setup> {
    SETUP
        .get()
        .ok_or_else(|| PyTypeError::new_err("pyroid.sqlparse is not configured"))
}

/// The token-type id of `obj`, registering it (`None`: not a `_TokenType` this port can
/// mirror).
fn tt_of(obj: &Bound<'_, PyAny>) -> PyResult<Option<TT>> {
    let py = obj.py();
    let s = setup()?;
    let ptr = obj.as_ptr() as usize;
    {
        let guard = TYPES
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(t) = guard.as_ref().and_then(|t| t.by_ptr.get(&ptr).copied()) {
            return Ok(Some(t));
        }
    }
    if !obj.is_instance(s.tokentype.bind(py))? {
        return Ok(None);
    }
    let tuple = obj.cast::<PyTuple>()?;
    let mut path: Vec<String> = Vec::with_capacity(tuple.len());
    for item in tuple.iter() {
        let Ok(st) = item.cast::<PyString>() else {
            return Ok(None);
        };
        path.push(st.to_str()?.to_owned());
    }
    let refs: Vec<&str> = path.iter().map(String::as_str).collect();
    let Ok(id) = ttype::intern(&refs) else {
        return Ok(None);
    };
    let mut guard = TYPES
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    let table = guard.get_or_insert_with(TypeTable::default);
    let idx = usize::from(id);
    if let Some(Some(existing)) = table.by_id.get(idx) {
        if existing.as_ptr() as usize != ptr {
            // another object with the same path: identity checks would differ
            return Ok(None);
        }
        return Ok(Some(id));
    }
    let mut v: Vec<Option<Py<PyAny>>> = table
        .by_id
        .iter()
        .map(|o| o.as_ref().map(|x| x.clone_ref(py)))
        .collect();
    if v.len() <= idx {
        v.resize_with(idx + 1, || None);
    }
    v[idx] = Some(obj.clone().unbind());
    table.by_id = Arc::new(v);
    table.by_ptr.insert(ptr, id);
    Ok(Some(id))
}

fn type_objects(py: Python<'_>) -> Arc<Vec<Option<Py<PyAny>>>> {
    let guard = TYPES
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    guard
        .as_ref()
        .map_or_else(|| Arc::new(Vec::new()), |t| Arc::clone(&t.by_id))
}

/// `_configure(classes, token_root, tokentype, process_as_keyword, error, unidata_version)`.
#[pyfunction]
#[pyo3(signature = (classes, root, tokentype, process_as_keyword, error, unidata_version, pause))]
#[allow(clippy::too_many_arguments)]
fn _configure(
    py: Python<'_>,
    classes: &Bound<'_, PyTuple>,
    root: &Bound<'_, PyAny>,
    tokentype: &Bound<'_, PyType>,
    process_as_keyword: &Bound<'_, PyAny>,
    error: &Bound<'_, PyType>,
    unidata_version: &str,
    pause: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if SETUP.get().is_some() {
        return Ok(());
    }
    let version =
        UnicodeVersion::from_unidata_version(unidata_version).unwrap_or(UnicodeVersion::LATEST);
    let mut cls = Vec::new();
    for c in classes.iter() {
        cls.push(c.cast_into::<PyType>()?.unbind());
    }
    if cls.len() != Class::ALL.len() {
        return Err(PyTypeError::new_err(
            "expected one class per sqlparse.sql class",
        ));
    }
    let new = py
        .get_type::<pyo3::types::PyAny>()
        .getattr("__new__")?
        .unbind();
    let pattern = py
        .import("re")?
        .getattr("Pattern")?
        .cast_into::<PyType>()?
        .unbind();
    let _ = SETUP.set(Setup {
        classes: cls,
        new,
        tokentype: tokentype.clone().unbind(),
        process_as_keyword: process_as_keyword.clone().unbind(),
        error: error.clone().unbind(),
        pattern,
        pause: pause.clone().unbind(),
        version,
    });
    for path in BUILTIN_PATHS {
        let mut obj = root.clone();
        for name in *path {
            obj = obj.getattr(*name)?;
        }
        if tt_of(&obj)?.is_none() {
            return Err(PyTypeError::new_err(format!(
                "token type {path:?} cannot be mirrored"
            )));
        }
    }
    Ok(())
}

fn sql_error(py: Python<'_>, msg: String) -> PyResult<PyErr> {
    Ok(PyErr::from_type(setup()?.error.bind(py).clone(), msg))
}

// ---------------------------------------------------------------------------
// The lexer snapshot
// ---------------------------------------------------------------------------

struct Snap {
    lexer: Py<PyAny>,
    regex: Py<PyAny>,
    regex_items: Vec<Py<PyAny>>,
    keywords: Py<PyAny>,
    dicts: Vec<(Py<PyAny>, usize)>,
    config: Arc<Config>,
    externals: Vec<Py<PyAny>>,
}

static SNAP: Mutex<Option<Arc<Snap>>> = Mutex::new(None);

fn snap_valid(
    lex: &Bound<'_, PyAny>,
    regex: &Bound<'_, PyAny>,
    kw: &Bound<'_, PyAny>,
    s: &Snap,
) -> PyResult<bool> {
    if !(s.lexer.is(lex) && s.regex.is(regex) && s.keywords.is(kw)) {
        return Ok(false);
    }
    let (Ok(rl), Ok(kl)) = (regex.cast::<PyList>(), kw.cast::<PyList>()) else {
        return Ok(false);
    };
    let rt = rl.to_tuple();
    if rt.len() != s.regex_items.len() || rt.iter().zip(&s.regex_items).any(|(a, b)| !b.is(&a)) {
        return Ok(false);
    }
    let kt = kl.to_tuple();
    if kt.len() != s.dicts.len() {
        return Ok(false);
    }
    for (d, (sd, n)) in kt.iter().zip(&s.dicts) {
        if !sd.is(&d) || d.len()? != *n {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The snapshot of `lex`'s table, built if needed (`None`: it needs Python's lexer).
fn snapshot(lex: &Bound<'_, PyAny>) -> PyResult<Option<Arc<Snap>>> {
    let py = lex.py();
    let regex = lex.getattr(intern!(py, "_SQL_REGEX"))?;
    let kw = lex.getattr(intern!(py, "_keywords"))?;
    {
        let guard = SNAP
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(s) = guard.as_ref() {
            if snap_valid(lex, &regex, &kw, s)? {
                return Ok(Some(Arc::clone(s)));
            }
        }
    }
    let Some(snap) = build_snapshot(lex, &regex, &kw)? else {
        return Ok(None);
    };
    let snap = Arc::new(snap);
    let mut guard = SNAP
        .lock_py_attached(py)
        .unwrap_or_else(PoisonError::into_inner);
    *guard = Some(Arc::clone(&snap));
    Ok(Some(snap))
}

fn build_snapshot(
    lex: &Bound<'_, PyAny>,
    regex: &Bound<'_, PyAny>,
    kw: &Bound<'_, PyAny>,
) -> PyResult<Option<Snap>> {
    let py = lex.py();
    let s = setup()?;
    let (Ok(rl), Ok(kl)) = (regex.cast::<PyList>(), kw.cast::<PyList>()) else {
        return Ok(None);
    };
    let rt = rl.to_tuple();
    let mut entries = Vec::with_capacity(rt.len());
    let mut externals: Vec<Py<PyAny>> = Vec::new();
    let mut items = Vec::with_capacity(rt.len());
    for item in rt.iter() {
        items.push(item.clone().unbind());
        let Ok(pair) = item.cast::<PyTuple>() else {
            return Ok(None);
        };
        if pair.len() != 2 {
            return Ok(None);
        }
        let (rexmatch, action) = (pair.get_item(0)?, pair.get_item(1)?);
        // a bound `re.Pattern.match`
        let Ok(pat) = rexmatch.getattr(intern!(py, "__self__")) else {
            return Ok(None);
        };
        if !pat.get_type().is(s.pattern.bind(py))
            || !rexmatch.eq(pat.getattr(intern!(py, "match"))?)?
        {
            return Ok(None);
        }
        let action = if action.is_instance(s.tokentype.bind(py))? {
            match tt_of(&action)? {
                Some(t) => Action::Type(t),
                None => return Ok(None),
            }
        } else if action.is(s.process_as_keyword.bind(py)) {
            Action::Keyword
        } else {
            Action::Drop
        };
        let text = pat.getattr(intern!(py, "pattern"))?;
        let flags: u32 = pat.getattr(intern!(py, "flags"))?.extract()?;
        let compiled = match text.cast::<PyString>() {
            Ok(t) => t
                .to_str()
                .ok()
                .and_then(|t| Regex::new(t, flags, s.version).ok()),
            Err(_) => None,
        };
        let matcher = if let Some(r) = compiled {
            Matcher::Native(Box::new(r))
        } else {
            externals.push(rexmatch.clone().unbind());
            Matcher::External(externals.len() - 1)
        };
        entries.push(Entry { matcher, action });
    }
    if entries.len() > usize::from(u16::MAX) {
        return Ok(None);
    }
    let kt = kl.to_tuple();
    let mut dicts_out = Vec::with_capacity(kt.len());
    let mut dicts = Vec::with_capacity(kt.len());
    for d in kt.iter() {
        let Ok(dict) = d.cast::<PyDict>() else {
            return Ok(None);
        };
        let copy = dict.copy()?;
        let mut words = Vec::with_capacity(copy.len());
        for (k, v) in copy.iter() {
            let Ok(k) = k.cast::<PyString>() else {
                continue;
            };
            if !k.get_type().is(py.get_type::<PyString>()) {
                return Ok(None);
            }
            let Ok(k) = k.to_str() else { continue };
            match tt_of(&v)? {
                Some(t) => words.push((k.to_owned(), t)),
                None => return Ok(None),
            }
        }
        dicts_out.push(words);
        dicts.push((d.clone().unbind(), copy.len()));
    }
    let config = Config::new(entries, &dicts_out, s.version);
    Ok(Some(Snap {
        lexer: lex.clone().unbind(),
        regex: regex.clone().unbind(),
        regex_items: items,
        keywords: kw.clone().unbind(),
        dicts,
        config: Arc::new(config),
        externals,
    }))
}

// ---------------------------------------------------------------------------
// Text in and out (lone surrogates)
// ---------------------------------------------------------------------------

/// The text a call works on: the str's UTF-8, or (with lone surrogates) the text with each
/// surrogate as a char in U+10F800.. (`remapped`).
struct Text {
    utf8: String,
    remapped: bool,
}

/// `None`: the str holds both lone surrogates and U+10F800.. (Python runs it).
fn text_of(s: &Bound<'_, PyString>) -> PyResult<Option<Text>> {
    let py = s.py();
    match s.to_str() {
        Ok(t) => Ok(Some(Text {
            utf8: t.to_owned(),
            remapped: false,
        })),
        Err(e) if e.is_instance_of::<PyUnicodeEncodeError>(py) => {
            let bytes = py
                .get_type::<PyString>()
                .call_method1("encode", (s, "utf-8", "surrogatepass"))?
                .cast_into::<PyBytes>()?;
            match wtf8::chars_from_wtf8(bytes.as_bytes()) {
                Some(Ok(chars)) => Ok(Some(Text {
                    utf8: chars.into_iter().collect(),
                    remapped: true,
                })),
                _ => Ok(None),
            }
        }
        Err(e) => Err(e),
    }
}

fn py_str<'py>(py: Python<'py>, s: &str, remapped: bool) -> PyResult<Bound<'py, PyString>> {
    if !remapped || !wtf8::contains_reserved(s) {
        return Ok(PyString::new(py, s));
    }
    let bytes = wtf8::to_wtf8(s);
    PyString::from_encoded_object(
        PyBytes::new(py, &bytes).as_any(),
        Some(c"utf-8"),
        Some(c"surrogatepass"),
    )
}

// ---------------------------------------------------------------------------
// Hooks: Python regexes, Ctrl-C and GIL hand-over
// ---------------------------------------------------------------------------

struct PyHooks<'a, 'py> {
    py: Python<'py>,
    text: &'a Bound<'py, PyString>,
    externals: &'a [Py<PyAny>],
    byte: usize,
    chr: usize,
    last_pause: Instant,
    pause: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> PyHooks<'a, 'py> {
    fn new(
        text: &'a Bound<'py, PyString>,
        externals: &'a [Py<PyAny>],
        pause: &'a Bound<'py, PyAny>,
    ) -> Self {
        PyHooks {
            py: text.py(),
            text,
            externals,
            byte: 0,
            chr: 0,
            last_pause: Instant::now(),
            pause,
        }
    }
}

impl External for PyHooks<'_, '_> {
    type Error = PyErr;

    fn match_at(&mut self, index: usize, text: &str, pos: usize) -> PyResult<Option<usize>> {
        if pos < self.byte {
            self.byte = 0;
            self.chr = 0;
        }
        self.chr += text[self.byte..pos].chars().count();
        self.byte = pos;
        let m = self.externals[index]
            .bind(self.py)
            .call1((self.text, self.chr))?;
        if !m.is_truthy()? {
            return Ok(None);
        }
        let end: usize = m.call_method0(intern!(self.py, "end"))?.extract()?;
        if end <= self.chr {
            return Ok(Some(pos));
        }
        let mut b = pos;
        let mut it = text[pos..].chars();
        for _ in 0..end - self.chr {
            match it.next() {
                Some(c) => b += c.len_utf8(),
                None => break,
            }
        }
        Ok(Some(b))
    }

    fn tick(&mut self) -> PyResult<()> {
        self.py.check_signals()?;
        // Long calls hold the interpreter: every 5 ms run a Python no-op, whose eval breaker
        // hands the GIL to waiting threads.
        if self.last_pause.elapsed() > Duration::from_millis(5) {
            self.pause.call0()?;
            self.last_pause = Instant::now();
        }
        Ok(())
    }
}

/// Raise `itertools.islice`'s own `ValueError` (sqlparse's `consume(iterable, -1)`).
fn empty_match_error(py: Python<'_>) -> PyErr {
    match py
        .import("itertools")
        .and_then(|m| m.getattr("islice"))
        .and_then(|f| f.call1((PyTuple::empty(py), -1)))
    {
        Err(e) => e,
        Ok(_) => PyIndexError::new_err("consume(iterable, -1)"),
    }
}

/// A DoS limit: an int, or `None` for no limit.
enum Lim {
    Is(Option<i64>),
    /// Anything else (a float, a bool, a huge int): Python decides.
    Other,
}

/// The DoS limit `v`.
fn limit(v: Option<&Bound<'_, PyAny>>) -> Lim {
    match v {
        None => Lim::Is(None),
        Some(x) if x.is_none() => Lim::Is(None),
        Some(x) if x.get_type().is(x.py().get_type::<pyo3::types::PyInt>()) => {
            x.extract::<i64>().map_or(Lim::Other, |n| Lim::Is(Some(n)))
        }
        Some(_) => Lim::Other,
    }
}

// ---------------------------------------------------------------------------
// Building the Python objects
// ---------------------------------------------------------------------------

/// Python objects for `stmt` and everything it holds, mirroring the arena.
fn build<'py>(
    py: Python<'py>,
    t: &Tree<'_>,
    stmt: Id,
    remapped: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let s = setup()?;
    let types = type_objects(py);
    let new = s.new.bind(py);
    // post-order over the lists reachable from stmt
    let mut order: Vec<Id> = Vec::new();
    let mut seen_list = vec![false; t.lists.len()];
    let mut stack: Vec<(Id, bool)> = vec![(stmt, false)];
    let mut seen_node = vec![false; t.nodes.len()];
    while let Some((id, expanded)) = stack.pop() {
        if expanded {
            order.push(id);
            continue;
        }
        if seen_node[id as usize] {
            continue;
        }
        seen_node[id as usize] = true;
        stack.push((id, true));
        let l = t.n(id).list;
        if l != NO_LIST && !seen_list[l as usize] {
            seen_list[l as usize] = true;
            for &k in t.lists[l as usize].iter().rev() {
                stack.push((k, false));
            }
        }
    }
    let mut objs: Vec<Option<Bound<'py, PyAny>>> = vec![None; t.nodes.len()];
    let mut lists: Vec<Option<Bound<'py, PyList>>> = vec![None; t.lists.len()];
    let true_ = pyo3::types::PyBool::new(py, true).to_owned().into_any();
    let false_ = pyo3::types::PyBool::new(py, false).to_owned().into_any();
    let b = |x: bool| if x { true_.clone() } else { false_.clone() };
    for &id in &order {
        let n = t.n(id);
        let cls = s.classes[n.class.index()].bind(py);
        let obj = new.call1((cls,))?;
        let value = py_str(py, t.val(id), remapped)?;
        let same = match (&n.value, &n.normalized) {
            (Str::Span(a, b), Str::Span(c, d)) => a == c && b == d,
            (Str::Owned(a), Str::Owned(b2)) => a == b2,
            _ => t.val(id) == t.norm(id),
        };
        let normalized = if same {
            value.clone()
        } else {
            py_str(py, t.norm(id), remapped)?
        };
        obj.setattr(intern!(py, "value"), &value)?;
        obj.setattr(intern!(py, "normalized"), &normalized)?;
        let ttype = match n.ttype {
            None => py.None().into_bound(py),
            Some(tt) => match types.get(usize::from(tt)).and_then(Option::as_ref) {
                Some(o) => o.bind(py).clone(),
                None => return Err(PyTypeError::new_err("unregistered token type")),
            },
        };
        obj.setattr(intern!(py, "ttype"), ttype)?;
        obj.setattr(intern!(py, "is_keyword"), b(n.is_keyword))?;
        obj.setattr(intern!(py, "is_group"), b(n.is_group))?;
        obj.setattr(intern!(py, "is_whitespace"), b(n.is_whitespace))?;
        obj.setattr(intern!(py, "is_newline"), b(n.is_newline))?;
        if n.list != NO_LIST {
            let l = n.list as usize;
            let list = if let Some(x) = &lists[l] {
                x.clone()
            } else {
                let items: Vec<Bound<'py, PyAny>> = t.lists[l]
                    .iter()
                    .map(|&k| {
                        objs[k as usize]
                            .clone()
                            .unwrap_or_else(|| py.None().into_bound(py))
                    })
                    .collect();
                let x = PyList::new(py, items)?;
                lists[l] = Some(x.clone());
                x
            };
            obj.setattr(intern!(py, "tokens"), list)?;
        }
        objs[id as usize] = Some(obj);
    }
    for &id in &order {
        let Some(obj) = &objs[id as usize] else {
            continue;
        };
        let parent = match t.n(id).parent {
            Some(p) => objs[p as usize]
                .clone()
                .unwrap_or_else(|| py.None().into_bound(py)),
            None => py.None().into_bound(py),
        };
        obj.setattr(intern!(py, "parent"), parent)?;
    }
    objs[stmt as usize]
        .clone()
        .ok_or_else(|| PyTypeError::new_err("no statement"))
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

fn not_implemented(py: Python<'_>) -> Bound<'_, PyAny> {
    py.NotImplemented().into_bound(py)
}

/// `Lexer.get_tokens(text)` for `lexer`: `(pairs, error)`, or `NotImplemented`.
#[pyfunction]
fn _tokenize<'py>(
    text: &Bound<'py, PyString>,
    lexer: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = text.py();
    let Some(snap) = snapshot(lexer)? else {
        return Ok(not_implemented(py));
    };
    let Some(src) = text_of(text)? else {
        return Ok(not_implemented(py));
    };
    let s = setup()?;
    let pause = s.pause.bind(py);
    let mut hooks = PyHooks::new(text, &snap.externals, pause);
    let (raw, err) = core_lexer::lex(&src.utf8, &snap.config, &mut hooks);
    let types = type_objects(py);
    let mut out = Vec::with_capacity(raw.len());
    for t in &raw {
        let tt = types
            .get(usize::from(t.tt))
            .and_then(Option::as_ref)
            .ok_or_else(|| PyTypeError::new_err("unregistered token type"))?;
        let v = py_str(
            py,
            &src.utf8[t.start as usize..t.end as usize],
            src.remapped,
        )?;
        out.push(PyTuple::new(py, [tt.bind(py).clone(), v.into_any()])?);
    }
    let err: Bound<'py, PyAny> = match err {
        None => py.None().into_bound(py),
        Some(core_lexer::LexError::EmptyMatch) => empty_match_error(py)
            .into_value(py)
            .into_bound(py)
            .into_any(),
        Some(core_lexer::LexError::External(e)) => e.into_value(py).into_bound(py).into_any(),
    };
    Ok(PyTuple::new(py, [PyList::new(py, out)?.into_any(), err])?.into_any())
}

fn engine_err(py: Python<'_>, e: EngineError<PyErr>) -> PyResult<Option<PyErr>> {
    Ok(Some(match e {
        EngineError::SqlParse(m) => sql_error(py, m)?,
        EngineError::Python => return Ok(None),
        EngineError::NotImplemented => PyNotImplementedError::new_err(()),
        EngineError::EmptyMatch => empty_match_error(py),
        EngineError::External(x) => x,
    }))
}

/// `split(text, strip_semicolon=...)` with `lexer`: a list of str, or `NotImplemented`.
#[pyfunction]
fn _split<'py>(
    text: &Bound<'py, PyString>,
    lexer: &Bound<'py, PyAny>,
    strip_semicolon: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let py = text.py();
    let Some(snap) = snapshot(lexer)? else {
        return Ok(not_implemented(py));
    };
    let Some(src) = text_of(text)? else {
        return Ok(not_implemented(py));
    };
    let s = setup()?;
    let pause = s.pause.bind(py);
    let mut hooks = PyHooks::new(text, &snap.externals, pause);
    match engine::split(&src.utf8, &snap.config, strip_semicolon, &mut hooks) {
        Ok(parts) => {
            let mut out = Vec::with_capacity(parts.len());
            for p in parts {
                out.push(py_str(py, &p, src.remapped)?);
            }
            Ok(PyList::new(py, out)?.into_any())
        }
        Err(e) => match engine_err(py, e)? {
            Some(err) => Err(err),
            None => Ok(not_implemented(py)),
        },
    }
}

fn case_of(s: Option<&str>) -> PyResult<Option<Case>> {
    Ok(match s {
        None => None,
        Some("upper") => Some(Case::Upper),
        Some("lower") => Some(Case::Lower),
        Some("capitalize") => Some(Case::Capitalize),
        Some(x) => return Err(PyTypeError::new_err(format!("bad case {x:?}"))),
    })
}

fn one_char(s: &str) -> Option<char> {
    let mut it = s.chars();
    let c = it.next()?;
    it.next().is_none().then_some(c)
}

/// `format(text, **options)` (options already validated by `formatter.validate_options`):
/// a str, or `NotImplemented`.
#[pyfunction]
#[pyo3(signature = (text, lexer, max_depth, max_tokens, keyword_case, identifier_case, truncate, spaces_around_operators, strip_comments, strip_whitespace, reindent, aligned, right_margin, output))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn _format<'py>(
    text: &Bound<'py, PyString>,
    lexer: &Bound<'py, PyAny>,
    max_depth: Option<&Bound<'py, PyAny>>,
    max_tokens: Option<&Bound<'py, PyAny>>,
    keyword_case: Option<&str>,
    identifier_case: Option<&str>,
    truncate: Option<(usize, Bound<'py, PyAny>)>,
    spaces_around_operators: bool,
    strip_comments: bool,
    strip_whitespace: bool,
    reindent: Option<(i64, String, i64, bool, bool, bool, bool)>,
    aligned: Option<String>,
    right_margin: bool,
    output: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = text.py();
    let (Lim::Is(depth), Lim::Is(tokens)) = (limit(max_depth), limit(max_tokens)) else {
        return Ok(not_implemented(py));
    };
    let truncate = match truncate {
        None => None,
        Some((w, ch)) => match ch.cast::<PyString>() {
            Ok(c) if c.get_type().is(py.get_type::<PyString>()) => {
                Some((w, c.to_str()?.to_owned()))
            }
            _ => return Ok(not_implemented(py)),
        },
    };
    let reindent = match reindent {
        None => None,
        Some((width, ch, wrap_after, comma_first, indent_after_first, indent_columns, compact)) => {
            let Some(ch) = one_char(&ch) else {
                return Ok(not_implemented(py));
            };
            Some(ReindentOptions {
                width,
                ch,
                wrap_after,
                comma_first,
                indent_after_first,
                indent_columns,
                compact,
            })
        }
    };
    let aligned = match aligned {
        None => None,
        Some(a) => match one_char(&a) {
            Some(c) => Some(c),
            None => return Ok(not_implemented(py)),
        },
    };
    let output = match output {
        Some("python") => Some(Output::Python),
        Some("php") => Some(Output::Php),
        _ => None,
    };
    let opts = FormatOptions {
        tokens: TokenFilters {
            keyword_case: case_of(keyword_case)?,
            identifier_case: case_of(identifier_case)?,
            truncate,
        },
        spaces_around_operators,
        strip_comments,
        strip_whitespace,
        reindent,
        aligned,
        right_margin,
        output,
    };
    let Some(snap) = snapshot(lexer)? else {
        return Ok(not_implemented(py));
    };
    let Some(src) = text_of(text)? else {
        return Ok(not_implemented(py));
    };
    let s = setup()?;
    let pause = s.pause.bind(py);
    let mut hooks = PyHooks::new(text, &snap.externals, pause);
    match engine::format(
        &src.utf8,
        &snap.config,
        Limits { depth, tokens },
        &opts,
        &mut hooks,
    ) {
        Ok(out) => Ok(py_str(py, &out, src.remapped)?.into_any()),
        Err(e) => match engine_err(py, e)? {
            Some(err) => Err(err),
            None => Ok(not_implemented(py)),
        },
    }
}

/// The statements of one `parsestream` call, grouped one per `__next__`.
#[pyclass(module = "pyroid.sqlparse")]
struct Statements {
    inner: Mutex<Option<Inner>>,
}

struct Inner {
    text: Py<PyString>,
    stream: Stream<String, PyErr>,
    remapped: bool,
    limits: Limits,
}

#[pymethods]
impl Statements {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next statement; `("python", flat_statement)` when sqlparse's own grouping must
    /// run on it (it raises there).
    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let mut guard = self
            .inner
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner);
        let Some(inner) = guard.as_mut() else {
            return Ok(None);
        };
        py.check_signals()?;
        let range = match inner.stream.next_statement() {
            Ok(Some(r)) => r,
            Ok(None) => {
                *guard = None;
                return Ok(None);
            }
            Err(EngineError::Python) => {
                *guard = None;
                return Err(PyIndexError::new_err("list index out of range"));
            }
            Err(e) => {
                *guard = None;
                return Err(engine_err(py, e)?
                    .unwrap_or_else(|| PyIndexError::new_err("list index out of range")));
            }
        };
        let remapped = inner.remapped;
        let limits = inner.limits;
        let (mut tree, stmt) = inner.stream.statement(range.clone());
        match Grouper::new(&mut tree, limits).group(stmt) {
            Ok(()) => Ok(Some(build(py, &tree, stmt, remapped)?)),
            Err(Fail::SqlParse(m)) => Err(sql_error(py, m)?),
            Err(Fail::Python) => {
                let (flat, fstmt) = inner.stream.statement(range);
                let obj = build(py, &flat, fstmt, remapped)?;
                let _ = &inner.text;
                Ok(Some(
                    PyTuple::new(py, [PyString::new(py, "python").into_any(), obj])?.into_any(),
                ))
            }
        }
    }
}

/// `parsestream(text)` with `lexer` and the limits: a `Statements` iterator, or
/// `NotImplemented`.
#[pyfunction]
fn _parse<'py>(
    text: &Bound<'py, PyString>,
    lexer: &Bound<'py, PyAny>,
    max_depth: Option<&Bound<'py, PyAny>>,
    max_tokens: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = text.py();
    let (Lim::Is(depth), Lim::Is(tokens)) = (limit(max_depth), limit(max_tokens)) else {
        return Ok(not_implemented(py));
    };
    let Some(snap) = snapshot(lexer)? else {
        return Ok(not_implemented(py));
    };
    let Some(src) = text_of(text)? else {
        return Ok(not_implemented(py));
    };
    let s = setup()?;
    let pause = s.pause.bind(py);
    let mut hooks = PyHooks::new(text, &snap.externals, pause);
    let remapped = src.remapped;
    let stream = match engine::stream(src.utf8, &snap.config, &TokenFilters::default(), &mut hooks)
    {
        Ok(st) => st,
        Err(e) => {
            return match engine_err(py, e)? {
                Some(err) => Err(err),
                None => Ok(not_implemented(py)),
            };
        }
    };
    let it = Statements {
        inner: Mutex::new(Some(Inner {
            text: text.clone().unbind(),
            stream,
            remapped,
            limits: Limits { depth, tokens },
        })),
    };
    Ok(Bound::new(py, it)?.into_any())
}

/// Testing hook: `re.compile(pattern, flags).match(text, pos)` by pyroid's engine: the end
/// (a code-point offset), `None`, or `NotImplemented` when the engine does not take the
/// pattern. `text` must not hold lone surrogates.
#[pyfunction]
fn _regex_match<'py>(
    py: Python<'py>,
    pattern: &str,
    flags: u32,
    text: &str,
    pos: usize,
    unidata_version: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let v = UnicodeVersion::from_unidata_version(unidata_version).unwrap_or(UnicodeVersion::LATEST);
    let Ok(re) = Regex::new(pattern, flags, v) else {
        return Ok(not_implemented(py));
    };
    let byte = text.char_indices().nth(pos).map_or(text.len(), |(b, _)| b);
    Ok(match re.match_at(text, byte) {
        None => py.None().into_bound(py),
        Some(end) => {
            let chars = pos + text[byte..end].chars().count();
            chars.into_pyobject(py)?.into_any()
        }
    })
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.sqlparse")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(_configure, &m)?)?;
    m.add_function(wrap_pyfunction!(_tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(_split, &m)?)?;
    m.add_function(wrap_pyfunction!(_format, &m)?)?;
    m.add_function(wrap_pyfunction!(_parse, &m)?)?;
    m.add_function(wrap_pyfunction!(_regex_match, &m)?)?;
    m.add_function(wrap_pyfunction!(group::_group, &m)?)?;
    m.add_class::<Statements>()?;
    parent.add_submodule(&m)
}
