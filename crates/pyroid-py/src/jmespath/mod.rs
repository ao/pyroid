// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! `pyroid._pyroid.jmespath`: the native side of `pyroid.jmespath`.
//!
//! Expressions are parsed by the core's port of jmespath's lexer and parser (with
//! `int()` and `json.loads` from [`host`]), turned into an [`expr::Expr`] tree holding the
//! Python values, cached as jmespath 1.1.0 caches them (the 512 most recent, oldest out
//! first), and evaluated on the caller's objects by [`interp`]. Errors are raised as the
//! exception classes `pyroid.jmespath` registers with `_configure` (jmespath's own when it is
//! installed).

mod expr;
mod functions;
mod host;
mod interp;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use pyo3::exceptions::{
    PyKeyError, PyRecursionError, PyRuntimeError, PyStopIteration, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString, PyTuple};
use pyroid_core::jmespath::{Budget, LexError, ParseError, ParseMessage, Token, TokenValue};

use expr::Expr;
use functions::Builtin;
use host::PyHost;
use interp::{Ctx, Funcs, Opts};

/// A stock `Functions` class (pyroid's, and jmespath's when it is installed): calls whose
/// `FUNCTION_TABLE` entry is one of its own run natively.
#[derive(Debug)]
struct Stock {
    cls: Py<PyAny>,
    /// The dispatch and validation methods a subclass must not override.
    methods: Vec<(Py<PyString>, Py<PyAny>)>,
    /// `(function, signature)` per [`Builtin`], from `FUNCTION_TABLE`.
    funcs: Vec<Option<(Py<PyAny>, Py<PyAny>)>>,
    /// The globals its type checks read `TYPES_MAP` and `REVERSE_TYPES_MAP` from.
    globals: Py<PyDict>,
}

impl Stock {
    /// Is `spec` (a `FUNCTION_TABLE` entry) this class's own entry for `b`?
    fn is_stock(&self, py: Python<'_>, b: Builtin, spec: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(Some((f, s))) = self.funcs.get(b as usize) else {
            return Ok(false);
        };
        Ok(spec.get_item("function")?.is(f) && spec.get_item("signature")?.is(s.bind(py)))
    }
}

/// What `_configure` registered.
#[derive(Debug)]
struct Config {
    parse_error: Py<PyAny>,
    incomplete_error: Py<PyAny>,
    lexer_error: Py<PyAny>,
    arity_error: Py<PyAny>,
    variadic_arity_error: Py<PyAny>,
    type_error: Py<PyAny>,
    empty_error: Py<PyAny>,
    unknown_function_error: Py<PyAny>,
    loads: Py<PyAny>,
    dumps: Py<PyAny>,
    dumps_kwargs: Py<PyDict>,
    warn_literal: Py<PyAny>,
    number: Py<PyAny>,
    ceil: Py<PyAny>,
    floor: Py<PyAny>,
    sum: Py<PyAny>,
    max: Py<PyAny>,
    min: Py<PyAny>,
    sorted: Py<PyAny>,
    reversed: Py<PyAny>,
    reverse_slice: Py<PyAny>,
    zero_one: Py<PyAny>,
    tree_interpreter: Py<PyAny>,
    expression_cls: Py<PyAny>,
    graphviz: Py<PyAny>,
    fallback: Py<PyAny>,
    getframe: Py<PyAny>,
    getrecursionlimit: Py<PyAny>,
    /// `str.__hash__` and `str.__eq__`: a `str` subclass with other ones is not keyed by its
    /// text in the cache.
    str_hash: Py<PyAny>,
    str_eq: Py<PyAny>,
    /// pyroid's `Functions`, then jmespath's when it is installed (the one `search` follows
    /// without custom functions).
    stocks: Vec<Stock>,
}

impl Config {
    /// The globals of the `Functions` class whose type checks the native ones stand in for.
    fn type_maps<'py>(&self, py: Python<'py>, funcs: &Funcs) -> Option<&Bound<'py, PyDict>> {
        let stock = match funcs {
            Funcs::Custom { stock: Some(i), .. } => self.stocks.get(*i),
            _ => self.stocks.last(),
        };
        stock.map(|s| s.globals.bind(py))
    }
}

static CONFIG: PyOnceLock<Config> = PyOnceLock::new();

fn config(py: Python<'_>) -> PyResult<&Config> {
    CONFIG.get(py).ok_or_else(|| {
        PyValueError::new_err("pyroid.jmespath is not configured (import pyroid.jmespath)")
    })
}

const STOCK_METHODS: &[&str] = &[
    "call_function",
    "_validate_arguments",
    "_type_check",
    "_type_check_single",
    "_get_allowed_pytypes",
    "_subtype_check",
    "_convert_to_jmespath_type",
    "_create_key_func",
];

/// Register the classes and helpers (`pyroid.jmespath` does it once at import).
#[pyfunction]
fn _configure(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<()> {
    let get = |k: &str| -> PyResult<Py<PyAny>> {
        config
            .get_item(k)?
            .map(Bound::unbind)
            .ok_or_else(|| PyValueError::new_err(format!("missing {k}")))
    };
    let mut stocks = Vec::new();
    for cls in get("stocks")?.bind(py).try_iter()? {
        let cls = cls?;
        let mut methods = Vec::new();
        for name in STOCK_METHODS {
            methods.push((
                PyString::intern(py, name).unbind(),
                cls.getattr(*name)?.unbind(),
            ));
        }
        let table = cls.getattr("FUNCTION_TABLE")?;
        let mut funcs = Vec::new();
        for b in Builtin::ALL {
            funcs.push(match table.get_item(b.name()) {
                Ok(spec) => Some((
                    spec.get_item("function")?.unbind(),
                    spec.get_item("signature")?.unbind(),
                )),
                Err(_) => None,
            });
        }
        let globals = cls
            .getattr("_type_check_single")?
            .getattr("__globals__")?
            .cast_into::<PyDict>()?
            .unbind();
        stocks.push(Stock {
            cls: cls.unbind(),
            methods,
            funcs,
            globals,
        });
    }
    let builtins = py.import("builtins")?;
    let dumps_kwargs = PyDict::new(py);
    dumps_kwargs.set_item("separators", (",", ":"))?;
    dumps_kwargs.set_item("default", builtins.getattr("str")?)?;
    let none = py.None();
    let reverse_slice = builtins
        .getattr("slice")?
        .call1((none.clone_ref(py), none, -1))?;
    let sys = py.import("sys")?;
    let c = Config {
        parse_error: get("ParseError")?,
        incomplete_error: get("IncompleteExpressionError")?,
        lexer_error: get("LexerError")?,
        arity_error: get("ArityError")?,
        variadic_arity_error: get("VariadictArityError")?,
        type_error: get("JMESPathTypeError")?,
        empty_error: get("EmptyExpressionError")?,
        unknown_function_error: get("UnknownFunctionError")?,
        loads: get("loads")?,
        dumps: get("dumps")?,
        dumps_kwargs: dumps_kwargs.unbind(),
        warn_literal: get("warn_literal")?,
        number: get("Number")?,
        ceil: get("ceil")?,
        floor: get("floor")?,
        sum: builtins.getattr("sum")?.unbind(),
        max: builtins.getattr("max")?.unbind(),
        min: builtins.getattr("min")?.unbind(),
        sorted: builtins.getattr("sorted")?.unbind(),
        reversed: builtins.getattr("reversed")?.unbind(),
        reverse_slice: reverse_slice.unbind(),
        zero_one: PyTuple::new(py, [0, 1])?.into_any().unbind(),
        tree_interpreter: get("TreeInterpreter")?,
        expression_cls: get("Expression")?,
        graphviz: get("GraphvizVisitor")?,
        fallback: get("fallback")?,
        getframe: sys.getattr("_getframe")?.unbind(),
        getrecursionlimit: sys.getattr("getrecursionlimit")?.unbind(),
        str_hash: builtins.getattr("str")?.getattr("__hash__")?.unbind(),
        str_eq: builtins.getattr("str")?.getattr("__eq__")?.unbind(),
        stocks,
    };
    let _ = CONFIG.set(py, c);
    Ok(())
}

// ---------------------------------------------------------------------------
// Recursion: where jmespath's recursive parser and interpreter would hit the limit
// ---------------------------------------------------------------------------

/// Trees and expressions this shallow are not checked (no frame walk on the fast path).
const SHALLOW: usize = 120;

/// The number of Python frames on the stack (the caller's frame included).
fn stack_depth(py: Python<'_>) -> PyResult<usize> {
    let c = config(py)?;
    let Ok(mut f) = c.getframe.bind(py).call1((0,)) else {
        return Ok(0);
    };
    let mut n = 0;
    while !f.is_none() {
        n += 1;
        f = f.getattr(pyo3::intern!(py, "f_back"))?;
    }
    Ok(n)
}

fn recursion_limit(py: Python<'_>) -> PyResult<usize> {
    config(py)?.getrecursionlimit.bind(py).call0()?.extract()
}

/// Native stack for a deep parse or evaluation. The main thread gets 8 MiB on Linux and macOS
/// (python.exe on Windows: 2 MB); other threads may be as small as the program chose, which
/// can't be read without changing it (`threading.stack_size()` resets it), so they get
/// [`pyroid_core::jmespath::STACK_BUDGET`].
fn stack_budget(py: Python<'_>) -> PyResult<usize> {
    const MAIN_THREAD: usize = if cfg!(windows) { 1 << 20 } else { 6 << 20 };
    let threading = py.import("threading")?;
    let main = threading.call_method0("main_thread")?;
    let current = threading.call_method0("current_thread")?;
    Ok(if main.is(&current) {
        MAIN_THREAD
    } else {
        pyroid_core::jmespath::STACK_BUDGET
    })
}

fn recursion_error() -> PyErr {
    PyRecursionError::new_err("maximum recursion depth exceeded")
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// A parsed tree, shared by a `ParsedResult` and the searches running on it.
struct Tree {
    root: Expr,
    /// [`expr::frames`] of the root.
    frames: usize,
    /// jmespath's dict AST (built on first access, or the dict it was made from).
    parsed: PyOnceLock<Py<PyAny>>,
}

impl Tree {
    fn new(root: Expr) -> Self {
        let frames = expr::frames(&root);
        Self::with_frames(root, frames)
    }

    fn with_frames(root: Expr, frames: usize) -> Self {
        Self {
            root,
            frames,
            parsed: PyOnceLock::new(),
        }
    }

    fn parsed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.parsed
            .get_or_try_init(py, || {
                expr::to_dict(py, &self.root).map(|d| d.into_any().unbind())
            })
            .map(|d| d.bind(py).clone())
    }
}

impl Drop for Tree {
    fn drop(&mut self) {
        if self.frames > expr::DROP_FRAMES {
            expr::free_deep(&mut self.root);
        }
    }
}

fn token_value<'py>(py: Python<'py>, t: &Token<host::Value<'py>>) -> Bound<'py, PyAny> {
    match &t.value {
        TokenValue::Text(s) => PyString::new(py, s).into_any(),
        TokenValue::Value(v) => v.clone().unwrap_or_else(|| py.None().into_bound(py)),
    }
}

/// `fmt % value`, as jmespath formats messages.
fn percent<'py>(
    py: Python<'py>,
    fmt: &str,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    PyString::new(py, fmt).call_method1("__mod__", (value,))
}

/// The exception jmespath's `_do_parse` would raise (with `expression` filled in), or
/// `None` for tokenizing (`Lexer.tokenize` leaves it unset).
fn parse_error<'py>(
    py: Python<'py>,
    err: ParseError<host::Value<'py>, PyErr>,
    expression: Option<&Bound<'py, PyAny>>,
) -> PyErr {
    let r = (|| -> PyResult<PyErr> {
        let c = config(py)?;
        Ok(match err {
            ParseError::Lex(LexError::Empty) => PyErr::from_value(c.empty_error.bind(py).call0()?),
            ParseError::Lex(LexError::Host(e)) => e,
            ParseError::Lex(LexError::Lexer {
                position,
                value,
                message,
            }) => {
                let e = c.lexer_error.bind(py).call1((
                    position,
                    host::py_str(py, &value)?,
                    host::py_str(py, &message)?,
                ))?;
                if let Some(x) = expression {
                    e.setattr("expression", x)?;
                }
                PyErr::from_value(e)
            }
            ParseError::Incomplete { token } => {
                let e = c.incomplete_error.bind(py).call1((
                    token.start,
                    token_value(py, &token),
                    token.kind.name(),
                ))?;
                if let Some(x) = expression {
                    e.call_method1("set_expression", (x,))?;
                }
                PyErr::from_value(e)
            }
            ParseError::Parse {
                lex_position,
                token,
                message,
            } => {
                let value = token_value(py, &token);
                let kind = token.kind.name();
                let msg = match message {
                    ParseMessage::InvalidToken => PyString::new(py, "invalid token").into_any(),
                    ParseMessage::SyntaxError => PyString::new(py, "syntax error").into_any(),
                    ParseMessage::UnexpectedToken => percent(py, "Unexpected token: %s", &value)?,
                    ParseMessage::Expecting(expected) => {
                        PyString::new(py, &format!("Expecting: {}, got: {kind}", expected.name()))
                            .into_any()
                    }
                    ParseMessage::ExpectingIdentifier => PyString::new(
                        py,
                        &format!(
                            "Expecting: {}, got: {kind}",
                            pyroid_core::jmespath::parser::IDENTIFIER_TYPES
                        ),
                    )
                    .into_any(),
                    ParseMessage::ExpectingDotRhs => PyString::new(
                        py,
                        &format!(
                            "Expecting: {}, got: {kind}",
                            pyroid_core::jmespath::parser::DOT_RHS_ALLOWED
                        ),
                    )
                    .into_any(),
                    ParseMessage::InvalidFunctionName => {
                        percent(py, "Invalid function name '%s'", &value)?
                    }
                    ParseMessage::QuotedFunctionName => {
                        PyString::new(py, "Quoted identifier not allowed for function names.")
                            .into_any()
                    }
                };
                let e = c
                    .parse_error
                    .bind(py)
                    .call1((lex_position, value, kind, msg))?;
                if let Some(x) = expression {
                    e.setattr("expression", x)?;
                }
                PyErr::from_value(e)
            }
            ParseError::Recursion => recursion_error(),
        })
    })();
    r.unwrap_or_else(|e| e)
}

/// Parse `text` (an expression's code points). `frames_above`: jmespath's frames between
/// the caller and `Parser._parse` (`search`/`compile`: 4, `Parser().parse`: 2 more than
/// pyroid's Python wrapper).
fn parse_text<'py>(
    py: Python<'py>,
    text: &[u32],
    expression: &Bound<'py, PyAny>,
    frames_above: usize,
) -> PyResult<Tree> {
    let c = config(py)?;
    // Each code point opens at most three frames.
    let budget = if 3 * text.len() + 30 < SHALLOW {
        Budget::UNLIMITED
    } else {
        let used = stack_depth(py)? + frames_above;
        let limit = recursion_limit(py)?;
        if used + 6 > limit {
            return Err(recursion_error());
        }
        Budget {
            frames: limit - used,
            stack: stack_budget(py)?,
        }
    };
    let mut host = PyHost { py, cfg: c };
    let mut node = pyroid_core::jmespath::parse(text, &mut host, budget)
        .map_err(|e| parse_error(py, e, Some(expression)))?;
    let (root, frames) = expr::from_node(py, &mut node)?;
    drop(node);
    Ok(Tree::with_frames(root, frames))
}

// ---------------------------------------------------------------------------
// The cache: jmespath 1.1.0's Parser._CACHE (512 entries, oldest evicted first)
// ---------------------------------------------------------------------------

const MAX_SIZE: usize = 512;

/// One cached expression, as a `_CACHE` entry.
struct Slot {
    /// The key's text, for a `str` key and a subclass that keeps `str`'s `__eq__` and
    /// `__hash__` (equal text is then equal keys); `None` for other keys.
    text: Option<Box<[u8]>>,
    /// The key object (dicts keep the first one stored).
    key: Py<PyAny>,
    /// `hash(key)` when it was stored (dicts keep it too). `None` for a key stored by its text
    /// while no odd key was cached: `str.__hash__(key)`, worked out when first needed.
    hash: Option<isize>,
    result: Py<ParsedResult>,
}

/// jmespath's `_CACHE` dict. Keys with a text are found by it in `map`. Keys of `str`
/// subclasses with their own `__eq__` or `__hash__` (`odd` of them) are found as a dict finds
/// them: by `hash()`, then identity or `==` against the keys with that hash, oldest first;
/// while any is cached, every lookup goes that way.
#[derive(Default)]
struct Cache {
    map: HashMap<Box<[u8]>, Py<ParsedResult>>,
    /// Oldest first.
    order: VecDeque<Slot>,
    odd: usize,
}

impl Cache {
    fn evict_oldest(&mut self) -> Option<(Slot, Option<Py<ParsedResult>>)> {
        let old = self.order.pop_front()?;
        let removed = if let Some(t) = &old.text {
            self.map.remove(t)
        } else {
            self.odd -= 1;
            None
        };
        Some((old, removed))
    }
}

static CACHE: LazyLock<Mutex<Cache>> = LazyLock::new(Mutex::default);

fn lock_cache() -> std::sync::MutexGuard<'static, Cache> {
    CACHE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The cache key of a `str` (UTF-8, or with surrogates passed through).
fn cache_key(s: &Bound<'_, PyString>) -> PyResult<Box<[u8]>> {
    if let Ok(t) = s.to_str() {
        return Ok(t.as_bytes().into());
    }
    let b = s.call_method1("encode", ("utf-8", "surrogatepass"))?;
    Ok(b.cast::<PyBytes>()?.as_bytes().into())
}

/// Does `s`'s class define its own `__eq__` or `__hash__` (so its text is not its dict key)?
/// Only for a subclass: a `str` never is.
#[cold]
#[inline(never)]
fn odd_key(py: Python<'_>, s: &Bound<'_, PyString>) -> PyResult<bool> {
    let c = config(py)?;
    let ty = s.get_type();
    Ok(!ty.getattr(pyo3::intern!(py, "__hash__"))?.is(&c.str_hash)
        || !ty.getattr(pyo3::intern!(py, "__eq__"))?.is(&c.str_eq))
}

/// What a dict lookup of `key` (hash `hash`) finds among the cached keys: identity or `==`
/// against each key with that hash, oldest first. Comparisons run with the cache unlocked
/// (they may run Python code, which may search).
#[cold]
#[inline(never)]
fn find_slow<'py>(
    py: Python<'py>,
    key: &Bound<'py, PyAny>,
    hash: isize,
) -> PyResult<Option<Bound<'py, ParsedResult>>> {
    let candidates = {
        let mut cache = lock_cache();
        let mut found: Vec<(Py<PyAny>, Py<ParsedResult>)> = Vec::new();
        for s in &mut cache.order {
            let h = if let Some(h) = s.hash {
                h
            } else {
                let h = text_hash(py, s.key.bind(py))?;
                s.hash = Some(h);
                h
            };
            if h == hash {
                found.push((s.key.clone_ref(py), s.result.clone_ref(py)));
            }
        }
        found
    };
    for (k, r) in candidates {
        let k = k.bind(py);
        if k.is(key) || k.eq(key)? {
            return Ok(Some(r.into_bound(py)));
        }
    }
    Ok(None)
}

/// `str.__hash__(key)`: the hash a key stored by its text had (no Python code runs).
fn text_hash(py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<isize> {
    if key.is_exact_instance_of::<PyString>() {
        return key.hash();
    }
    config(py)?.str_hash.bind(py).call1((key,))?.extract()
}

/// `hash(key)` as a dict operation reports a failure: 3.14 rewords a `TypeError` ("cannot use
/// 'X' as a dict key (...)"), so a failing key is looked up in an empty dict for its error.
fn dict_hash(key: &Bound<'_, PyAny>) -> PyResult<isize> {
    key.hash().or_else(|_| {
        let py = key.py();
        PyDict::new(py)
            .as_any()
            .get_item(key)
            .and(Err(PyKeyError::new_err(key.clone().unbind())))
    })
}

/// `_CACHE[expression]` while odd keys are about: `None` for a miss (jmespath's `except
/// KeyError: pass` takes a `KeyError` from `__hash__` or `__eq__` as one too).
#[cold]
#[inline(never)]
fn lookup_slow<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, ParsedResult>>> {
    let found = dict_hash(expression).and_then(|h| find_slow(py, expression, h));
    match found {
        Err(e) if e.is_instance_of::<PyKeyError>(py) => Ok(None),
        other => other,
    }
}

/// jmespath's insertion after a parse when odd keys are about: drop the oldest entry when
/// full (a failed `del` leaves the result uncached), then `_CACHE[expression] = result`.
#[cold]
#[inline(never)]
fn insert_slow(
    py: Python<'_>,
    expression: &Bound<'_, PyAny>,
    text: Option<Box<[u8]>>,
    result: &Bound<'_, ParsedResult>,
) -> PyResult<()> {
    let oldest = {
        let cache = lock_cache();
        if cache.order.len() >= MAX_SIZE {
            cache
                .order
                .front()
                .map(|s| (s.key.clone_ref(py), s.hash.unwrap_or(0), s.text.is_none()))
        } else {
            None
        }
    };
    if let Some((key, hash, odd)) = oldest {
        // `del _CACHE[next(iter(_CACHE))]` hashes the key again; jmespath returns the result
        // uncached on a KeyError, StopIteration or RuntimeError.
        if odd {
            match dict_hash(key.bind(py)) {
                Ok(h) if h == hash => {}
                Ok(_) => return Ok(()), // not found under its new hash: KeyError
                Err(e)
                    if e.is_instance_of::<PyKeyError>(py)
                        || e.is_instance_of::<PyStopIteration>(py)
                        || e.is_instance_of::<PyRuntimeError>(py) =>
                {
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
        let evicted = {
            let mut cache = lock_cache();
            if cache.order.front().is_some_and(|s| s.key.is(&key)) {
                cache.evict_oldest()
            } else {
                None
            }
        };
        drop(evicted);
    }
    let hash = dict_hash(expression)?;
    if let Some(same) = find_slow(py, expression, hash)? {
        // An equal key appeared (another thread, or an `__eq__` that changed its mind): the
        // dict keeps that key and takes the new value.
        let replaced = {
            let mut cache = lock_cache();
            let mut old = (None, None);
            if let Some(slot) = cache.order.iter_mut().find(|s| s.result.is(&same)) {
                let t = slot.text.clone();
                old.0 = Some(std::mem::replace(&mut slot.result, result.clone().unbind()));
                if let Some(t) = t {
                    old.1 = cache.map.insert(t, result.clone().unbind());
                }
            }
            old
        };
        // Dropped after unlocking, as below.
        drop(replaced);
        return Ok(());
    }
    let previous = {
        let mut cache = lock_cache();
        let previous = match &text {
            Some(t) => cache.map.insert(t.clone(), result.clone().unbind()),
            None => None,
        };
        let mut replaced = None;
        if previous.is_some() {
            // Another thread stored the same text meanwhile: one entry, the new value.
            if let Some(slot) = cache.order.iter_mut().find(|s| s.text == text) {
                replaced = Some(std::mem::replace(&mut slot.result, result.clone().unbind()));
            }
        } else {
            if text.is_none() {
                cache.odd += 1;
            }
            cache.order.push_back(Slot {
                text,
                key: expression.clone().unbind(),
                hash: Some(hash),
                result: result.clone().unbind(),
            });
        }
        (previous, replaced)
    };
    // Dropped after unlocking: freeing a result may run arbitrary Python code.
    drop(previous);
    Ok(())
}

/// What a non-`str` expression gets.
enum Parsed<'py> {
    Native(Bound<'py, ParsedResult>),
    /// jmespath's own result for an expression pyroid does not lex (see `fallback`).
    Foreign(Bound<'py, PyAny>),
}

/// `Parser().parse(expression)`, through the cache.
fn get_or_parse<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
    kind: &str,
    frames_above: usize,
) -> PyResult<Parsed<'py>> {
    let Ok(s) = expression.cast::<PyString>() else {
        return other_expression(py, expression, kind).map(Parsed::Foreign);
    };
    let odd = !s.is_exact_instance_of::<PyString>() && odd_key(py, s)?;
    let key = cache_key(s)?;
    let mut slow = odd;
    if !odd {
        let cache = lock_cache();
        if cache.odd == 0 {
            if let Some(hit) = cache.map.get(&key) {
                return Ok(Parsed::Native(hit.bind(py).clone()));
            }
        } else {
            slow = true;
        }
    }
    if slow {
        if let Some(hit) = lookup_slow(py, expression)? {
            return Ok(Parsed::Native(hit));
        }
    }
    let text = host::code_points(s)?;
    let tree = parse_text(py, &text, expression, frames_above)?;
    let result = Bound::new(
        py,
        ParsedResult {
            expression: Mutex::new(expression.clone().unbind()),
            tree: Mutex::new(Arc::new(tree)),
        },
    )?;
    if slow {
        insert_slow(py, expression, (!odd).then_some(key), &result)?;
        return Ok(Parsed::Native(result));
    }
    let evicted = {
        let mut cache = lock_cache();
        let mut evicted = None;
        let mut replaced = None;
        let previous = cache.map.insert(key.clone(), result.clone().unbind());
        if previous.is_none() {
            if cache.order.len() >= MAX_SIZE {
                evicted = cache.evict_oldest();
            }
            cache.order.push_back(Slot {
                text: Some(key),
                key: expression.clone().unbind(),
                hash: None,
                result: result.clone().unbind(),
            });
        } else if let Some(slot) = cache
            .order
            .iter_mut()
            .find(|s| s.text.as_deref() == Some(&*key))
        {
            // Another thread parsed it too: keep the first key, take the new value.
            replaced = Some(std::mem::replace(&mut slot.result, result.clone().unbind()));
        }
        (evicted, previous, replaced)
    };
    // Dropped after unlocking: freeing a result may run arbitrary Python code.
    drop(evicted);
    Ok(Parsed::Native(result))
}

/// A non-`str` expression: what jmespath raises for empty, unhashable, `bytes` and
/// `bytearray` ones; anything else (jmespath lexes any iterable of characters) goes to
/// jmespath. For `tokenize` (`Lexer().tokenize`), an iterable of jmespath's tokens.
fn other_expression<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
    kind: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let c = config(py)?;
    if kind != "tokenize" {
        // `Parser.parse` looks it up in its cache (a dict) first: an unhashable expression
        // raises the dict's own TypeError (whose wording changed in 3.14).
        PyDict::new(py).call_method1(pyo3::intern!(py, "get"), (expression,))?;
    }
    if !expression.is_truthy()? {
        return Err(PyErr::from_value(c.empty_error.bind(py).call0()?));
    }
    if expression.is_exact_instance_of::<PyBytes>()
        || expression.is_exact_instance_of::<PyByteArray>()
    {
        // list(b) gives ints, which match no token: "Unknown token <first byte>".
        let first = expression.get_item(0)?;
        let e =
            c.lexer_error
                .bind(py)
                .call1((0, &first, percent(py, "Unknown token %s", &first)?))?;
        if kind != "tokenize" {
            e.setattr("expression", expression)?;
        }
        return Err(PyErr::from_value(e));
    }
    c.fallback.bind(py).call1((kind, expression))
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

static DEFAULT_OPTS: LazyLock<Arc<Opts>> = LazyLock::new(|| {
    Arc::new(Opts {
        dict_cls: None,
        funcs: Funcs::Native,
        options: None,
    })
});

/// Is `obj` an instance of a stock `Functions` class that dispatches as the stock one?
fn stock_index(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    let c = config(py)?;
    let ty = obj.get_type();
    for (i, st) in c.stocks.iter().enumerate() {
        if !ty.is_subclass(st.cls.bind(py))? {
            continue;
        }
        for (name, m) in &st.methods {
            if !ty.getattr(name.bind(py))?.is(m) {
                return Ok(None);
            }
        }
        if let Some(d) = obj.getattr_opt(pyo3::intern!(py, "__dict__"))? {
            if let Ok(d) = d.cast::<PyDict>() {
                for (name, _) in &st.methods {
                    if d.contains(name.bind(py))? {
                        return Ok(None);
                    }
                }
            }
        }
        return Ok(Some(i));
    }
    Ok(None)
}

/// `TreeInterpreter(options)`: the dict class and functions to use.
fn opts_of(py: Python<'_>, options: Option<&Bound<'_, PyAny>>) -> PyResult<Arc<Opts>> {
    let Some(o) = options.filter(|o| !o.is_none()) else {
        return Ok(Arc::clone(&DEFAULT_OPTS));
    };
    let dict_cls = o.getattr(pyo3::intern!(py, "dict_cls"))?;
    let custom = o.getattr(pyo3::intern!(py, "custom_functions"))?;
    let dict_cls = if dict_cls.is_none() || dict_cls.is(py.get_type::<PyDict>()) {
        None
    } else {
        Some(dict_cls.unbind())
    };
    let funcs = if custom.is_none() {
        Funcs::Native
    } else {
        Funcs::Custom {
            stock: stock_index(py, &custom)?,
            obj: custom.unbind(),
        }
    };
    if dict_cls.is_none() && matches!(funcs, Funcs::Native) {
        return Ok(Arc::clone(&DEFAULT_OPTS));
    }
    Ok(Arc::new(Opts {
        dict_cls,
        funcs,
        options: Some(o.clone().unbind()),
    }))
}

/// Evaluate `root`. `frames_above`: jmespath's frames between the caller and the root's
/// `visit` (`search`: 2, `ParsedResult.search`: 1).
fn run<'py>(
    py: Python<'py>,
    root: &Expr,
    frames: usize,
    value: &Bound<'py, PyAny>,
    opts: &Arc<Opts>,
    frames_above: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let (limit, depth, stack) = if frames + frames_above < SHALLOW {
        (usize::MAX, 0, pyroid_core::jmespath::STACK_BUDGET)
    } else {
        (
            recursion_limit(py)?,
            stack_depth(py)? + frames_above,
            stack_budget(py)?,
        )
    };
    let ctx = Ctx::new(py, opts, limit, stack);
    interp::visit(&ctx, root, value, depth)
}

// ---------------------------------------------------------------------------
// Python classes
// ---------------------------------------------------------------------------

/// A compiled expression: ``jmespath.compile(expression)``.
#[pyclass(module = "pyroid.jmespath.parser", name = "ParsedResult", frozen, dict)]
pub(crate) struct ParsedResult {
    expression: Mutex<Py<PyAny>>,
    tree: Mutex<Arc<Tree>>,
}

impl ParsedResult {
    fn tree(&self) -> Arc<Tree> {
        Arc::clone(&self.tree.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

#[pymethods]
impl ParsedResult {
    /// A result from an expression and a dict AST, as `jmespath.parser.ParsedResult`.
    #[new]
    fn new(expression: Py<PyAny>, parsed: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            expression: Mutex::new(expression),
            tree: Mutex::new(Arc::new(tree_from_dict(parsed)?)),
        })
    }

    #[getter]
    fn get_expression(&self, py: Python<'_>) -> Py<PyAny> {
        self.expression
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone_ref(py)
    }

    #[setter]
    fn set_expression(&self, value: Py<PyAny>) {
        *self
            .expression
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = value;
    }

    /// jmespath's AST: nested ``{'type': ..., 'children': [...], 'value': ...}`` dicts.
    #[getter]
    fn get_parsed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tree().parsed(py)
    }

    #[setter]
    fn set_parsed(&self, parsed: &Bound<'_, PyAny>) -> PyResult<()> {
        let tree = Arc::new(tree_from_dict(parsed)?);
        let old = std::mem::replace(
            &mut *self.tree.lock().unwrap_or_else(PoisonError::into_inner),
            tree,
        );
        drop(old);
        Ok(())
    }

    /// Evaluate the expression on ``value``.
    #[pyo3(signature = (value, options=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        value: &Bound<'py, PyAny>,
        options: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tree = self.tree();
        let opts = opts_of(py, options)?;
        run(py, &tree.root, tree.frames, value, &opts, 1)
    }

    fn _render_dot_file<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let parsed = self.tree().parsed(py)?;
        config(py)?
            .graphviz
            .bind(py)
            .call0()?
            .call_method1("visit", (parsed,))
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        self.tree().parsed(py)?.repr()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let me = slf.get();
        let args = (me.get_expression(py), me.tree().parsed(py)?);
        PyTuple::new(
            py,
            [
                slf.get_type().into_any(),
                args.into_pyobject(py)?.into_any(),
            ],
        )
    }
}

fn tree_from_dict(parsed: &Bound<'_, PyAny>) -> PyResult<Tree> {
    let stack = stack_budget(parsed.py())?;
    let tree = Tree::new(expr::from_dict(parsed, stack)?);
    let _ = tree.parsed.set(parsed.py(), parsed.clone().unbind());
    Ok(tree)
}

/// The interpreter behind the `_Expression` objects pyroid makes: `visit(node, value)`
/// evaluates natively (the expref's own node without reading it back).
#[pyclass(
    module = "pyroid.jmespath.visitor",
    name = "_NativeInterpreter",
    frozen
)]
pub(crate) struct Interpreter {
    node: Arc<expr::ExprefNode>,
    opts: Arc<Opts>,
}

impl Interpreter {
    fn new(node: Arc<expr::ExprefNode>, opts: Arc<Opts>) -> Self {
        Self { node, opts }
    }
}

#[pymethods]
impl Interpreter {
    /// Evaluate ``node`` (a dict AST node) on ``value``.
    #[pyo3(signature = (node, *args, **kwargs))]
    fn visit<'py>(
        &self,
        py: Python<'py>,
        node: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let value = match (args.len(), kwargs.map(PyDictMethods::len)) {
            (1, None | Some(0)) => args.get_item(0)?,
            (0, Some(1)) => kwargs
                .and_then(|k| k.get_item("value").ok().flatten())
                .ok_or_else(|| PyTypeError::new_err("visit() takes a node and a value"))?,
            _ => return Err(PyTypeError::new_err("visit() takes a node and a value")),
        };
        // _Expression.visit -> TreeInterpreter.visit
        let own = self.node.dict.get(py).is_some_and(|d| node.is(d));
        if own {
            let e = &self.node.expr;
            return run(py, e, self.node.frames, &value, &self.opts, 1);
        }
        let e = expr::from_dict(node, stack_budget(py)?)?;
        run(py, &e, expr::frames(&e), &value, &self.opts, 1)
    }

    /// A pure-Python ``TreeInterpreter`` with the same options, for other attributes.
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let options = self
            .opts
            .options
            .as_ref()
            .map_or_else(|| py.None(), |o| o.clone_ref(py));
        config(py)?
            .tree_interpreter
            .bind(py)
            .call1((options,))?
            .getattr(name)
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

/// Evaluate ``expression`` on ``data`` (``jmespath.search``).
#[pyfunction]
#[pyo3(signature = (expression, data, options=None))]
fn search<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    options: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match get_or_parse(py, expression, "search", 4)? {
        Parsed::Native(p) => {
            let tree = p.get().tree();
            drop(p);
            let opts = opts_of(py, options)?;
            run(py, &tree.root, tree.frames, data, &opts, 2)
        }
        Parsed::Foreign(p) => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("options", options)?;
            p.call_method("search", (data,), Some(&kwargs))
        }
    }
}

/// Compile ``expression`` (``jmespath.compile``).
#[pyfunction]
fn compile<'py>(py: Python<'py>, expression: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match get_or_parse(py, expression, "compile", 4)? {
        Parsed::Native(p) => p.into_any(),
        Parsed::Foreign(p) => p,
    })
}

/// ``Parser().parse(expression)`` (called from pyroid's Python ``Parser.parse``).
#[pyfunction]
fn parse<'py>(py: Python<'py>, expression: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(match get_or_parse(py, expression, "parse", 2)? {
        Parsed::Native(p) => p.into_any(),
        Parsed::Foreign(p) => p,
    })
}

/// ``(tokens, error)``: jmespath's token dicts (a list, or jmespath's own generator for an
/// expression it lexes), and the exception its generator would raise after them (or ``None``).
#[pyfunction]
fn tokenize<'py>(
    py: Python<'py>,
    expression: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let Ok(s) = expression.cast::<PyString>() else {
        return match other_expression(py, expression, "tokenize") {
            Ok(r) => Ok((r, py.None().into_bound(py))),
            Err(e) => Ok((
                PyList::empty(py).into_any(),
                e.into_value(py).into_bound(py).into_any(),
            )),
        };
    };
    let text = host::code_points(s)?;
    let mut host = PyHost {
        py,
        cfg: config(py)?,
    };
    let (tokens, error) = match pyroid_core::jmespath::tokenize(&text, &mut host) {
        Ok(t) => (t, None),
        Err((t, e)) => (t, Some(e)),
    };
    let list = PyList::empty(py);
    for t in &tokens {
        let d = PyDict::new(py);
        d.set_item("type", t.kind.name())?;
        d.set_item("value", token_value(py, t))?;
        d.set_item("start", t.start)?;
        d.set_item("end", t.end)?;
        list.append(d)?;
    }
    let error = match error {
        None => py.None().into_bound(py),
        Some(LexError::Host(e)) => return Err(e),
        Some(e) => parse_error(py, ParseError::Lex(e), None)
            .into_value(py)
            .into_bound(py)
            .into_any(),
    };
    Ok((list.into_any(), error))
}

/// Clear the expression cache (``Parser.purge()``).
#[pyfunction]
fn purge() {
    let old = std::mem::take(&mut *lock_cache());
    drop(old);
}

/// How many expressions are cached.
#[pyfunction]
fn _cache_len() -> usize {
    lock_cache().order.len()
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "pyroid.jmespath")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(_configure, &m)?)?;
    m.add_function(wrap_pyfunction!(search, &m)?)?;
    m.add_function(wrap_pyfunction!(compile, &m)?)?;
    m.add_function(wrap_pyfunction!(parse, &m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, &m)?)?;
    m.add_function(wrap_pyfunction!(purge, &m)?)?;
    m.add_function(wrap_pyfunction!(_cache_len, &m)?)?;
    m.add_class::<ParsedResult>()?;
    m.add_class::<Interpreter>()?;
    parent.add_submodule(&m)
}
