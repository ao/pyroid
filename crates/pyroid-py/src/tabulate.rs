// Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
//! `pyroid._pyroid.tabulate`: the native half of `pyroid.tabulate.tabulate`.
//!
//! The Python side (`python/pyroid/tabulate/__init__.py`) normalises the input, wraps text
//! for `maxcolwidths`, and passes the rows, the headers and a call object. [`render`] then
//! runs tabulate's steps in tabulate's order: rst escaping, the ANSI and newline search, the
//! column types, cell formatting, alignment and the table text. It asks the call object for
//! everything tabulate computes from the options (per-column formats, alignments, row
//! alignments), so their errors and warnings stay tabulate's.
//!
//! Native cells: exact `str`, `int` (64-bit), `float`, `bool`, `None`, and the numpy scalar
//! types the call object lists. Any other cell goes through tabulate's own `_type` and
//! `_format`, as do strings the native tests cannot decide (see `pyroid_core::tabulate`).
//! Cells or results the Rust side cannot hold (lone surrogates) make `render` return
//! `(None, reason)`: the caller hands the call to tabulate.

use std::borrow::Cow;

use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyList, PyString, PyTuple};
use pyroid_core::tabulate::number::{self, ColType};
use pyroid_core::tabulate::pyfmt::{self, Spec};
use pyroid_core::tabulate::pystr;
use pyroid_core::tabulate::table::{
    self, Align, AlignList, Cells, Engine, Host, LineFmt, Piece, RowAlign, RowFmt, RowRef,
    Settings, TableFormat,
};
use rustc_hash::FxHashMap;

/// Why a call cannot run natively (the caller delegates), or a Python error.
enum Fail {
    Py(PyErr),
    Unsupported(&'static str),
}

impl From<PyErr> for Fail {
    fn from(e: PyErr) -> Self {
        Self::Py(e)
    }
}

type FResult<T> = Result<T, Fail>;

type Objs<'py> = Vec<Bound<'py, PyAny>>;

/// A cell as the native code sees it.
enum Val {
    None,
    Bool(bool),
    Int(i64),
    /// A numpy integer scalar that tabulate types as `int` (value from `__index__`).
    NpInt(i64),
    Float(f64),
    /// A numpy scalar that tabulate types as `float` (value from `float()`).
    NpFloat(f64),
    Str(PyBackedStr),
    /// Anything else: tabulate's `_type` and `_format` in Python.
    Obj,
}

struct Cell<'py> {
    obj: Bound<'py, PyAny>,
    val: Val,
}

/// `str(x)` of a Python str that may hold lone surrogates.
fn backed(s: &Bound<'_, PyString>) -> FResult<PyBackedStr> {
    PyBackedStr::try_from(s.clone()).map_err(|_| Fail::Unsupported("text with lone surrogates"))
}

fn to_backed(obj: &Bound<'_, PyAny>) -> FResult<PyBackedStr> {
    let s = obj
        .cast::<PyString>()
        .map_err(|_| Fail::Unsupported("a result that is not a str"))?;
    backed(s)
}

/// `Py_TYPE(obj)->tp_name`, as `"\n".join` reports it.
fn tp_name(obj: &Bound<'_, PyAny>) -> String {
    let ty = obj.get_type();
    let name = ty
        .name()
        .map_or_else(|_| "object".to_owned(), |n| n.to_string());
    let module = ty
        .module()
        .map_or_else(|_| String::new(), |m| m.to_string());
    let heap = ty
        .getattr("__flags__")
        .and_then(|f| f.extract::<u64>())
        .map_or(true, |f| f & (1 << 9) != 0);
    if heap || module == "builtins" || module.is_empty() {
        name
    } else {
        format!("{module}.{name}")
    }
}

/// `type(None)`, `bool`, `int`, `float`, `bytes`, `str` and `list`, looked up once.
static TYPES: PyOnceLock<[Py<PyAny>; 7]> = PyOnceLock::new();

fn builtin_types(py: Python<'_>) -> PyResult<&[Py<PyAny>; 7]> {
    TYPES.get_or_try_init(py, || {
        let b = py.import("builtins")?;
        Ok::<_, PyErr>([
            py.None().into_bound(py).get_type().into_any().unbind(),
            b.getattr("bool")?.unbind(),
            b.getattr("int")?.unbind(),
            b.getattr("float")?.unbind(),
            b.getattr("bytes")?.unbind(),
            b.getattr("str")?.unbind(),
            b.getattr("list")?.unbind(),
        ])
    })
}

/// The built-in formats, built once.
static BUILTIN_FORMATS: std::sync::LazyLock<FxHashMap<&'static str, TableFormat>> =
    std::sync::LazyLock::new(|| {
        table::FORMAT_NAMES
            .iter()
            .map(|&n| (n, table::builtin(n).expect("every name is built in")))
            .collect()
    });

/// What the Python call object provides.
struct Call<'py> {
    obj: Bound<'py, PyAny>,
    type_fn: Bound<'py, PyAny>,
    format_fn: Bound<'py, PyAny>,
    strip_ansi_fn: Bound<'py, PyAny>,
    afterpoint_fn: Bound<'py, PyAny>,
    to_str_fn: Bound<'py, PyAny>,
    escape_empty_fn: Bound<'py, PyAny>,
    wcswidth: Option<Bound<'py, PyAny>>,
    np_types: Vec<(usize, bool)>,
    types: [Bound<'py, PyAny>; 6],
}

impl<'py> Call<'py> {
    fn new(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let wcswidth = obj.getattr("wcswidth")?;
        let np: Vec<(Bound<'py, PyAny>, bool)> = obj.getattr("numpy_types")?.extract()?;
        let t = builtin_types(py)?;
        Ok(Self {
            type_fn: obj.getattr("type_fn")?,
            format_fn: obj.getattr("format_fn")?,
            strip_ansi_fn: obj.getattr("strip_ansi_fn")?,
            afterpoint_fn: obj.getattr("afterpoint_fn")?,
            to_str_fn: obj.getattr("to_str_fn")?,
            escape_empty_fn: obj.getattr("escape_empty_fn")?,
            wcswidth: if wcswidth.is_none() {
                None
            } else {
                Some(wcswidth)
            },
            np_types: np
                .iter()
                .map(|(t, is_int)| (t.as_ptr() as usize, *is_int))
                .collect(),
            types: [
                t[0].bind(py).clone(),
                t[1].bind(py).clone(),
                t[2].bind(py).clone(),
                t[3].bind(py).clone(),
                t[4].bind(py).clone(),
                t[5].bind(py).clone(),
            ],
            obj: obj.clone(),
        })
    }

    fn flag(&self, name: &str) -> PyResult<bool> {
        self.obj.getattr(name)?.is_truthy()
    }

    fn col_type_obj(&self, t: ColType) -> &Bound<'py, PyAny> {
        &self.types[t as usize]
    }

    /// `_more_generic`'s rank of a type `_type` returned (unknown types count as `str`).
    fn rank(&self, t: &Bound<'py, PyAny>) -> ColType {
        self.types
            .iter()
            .position(|x| x.is(t))
            .map_or(ColType::Str, |i| {
                ColType::from_rank(u8::try_from(i).unwrap_or(5))
            })
    }
}

fn classify<'py>(call: &Call<'py>, obj: Bound<'py, PyAny>) -> FResult<Cell<'py>> {
    let val = if let Ok(s) = obj.cast_exact::<PyString>() {
        Val::Str(backed(s)?)
    } else if obj.is_exact_instance_of::<PyFloat>() {
        Val::Float(obj.extract::<f64>()?)
    } else if obj.is_none() {
        Val::None
    } else if obj.is_exact_instance_of::<PyBool>() {
        Val::Bool(obj.is_truthy()?)
    } else if obj.is_exact_instance_of::<PyInt>() {
        obj.extract::<i64>().map_or(Val::Obj, Val::Int)
    } else {
        let ty = obj.get_type().as_ptr() as usize;
        match call.np_types.iter().find(|(t, _)| *t == ty) {
            Some((_, true)) => obj.extract::<i64>().map_or(Val::Obj, Val::NpInt),
            Some((_, false)) => obj.extract::<f64>().map_or(Val::Obj, Val::NpFloat),
            None => Val::Obj,
        }
    };
    Ok(Cell { obj, val })
}

/// The table's [`Host`]: widths through `wcwidth.wcswidth`, and the Python fallbacks.
struct PyHost<'a, 'py> {
    call: &'a Call<'py>,
    widths: FxHashMap<String, isize>,
    /// Custom line/row callables, by id.
    callables: Vec<Bound<'py, PyAny>>,
    /// The alignment lists the callables receive (columns, headers).
    aligns: [Option<Bound<'py, PyAny>>; 2],
}

impl PyHost<'_, '_> {
    fn py_str<'p>(&self, s: &str) -> Bound<'p, PyString>
    where
        Self: 'p,
    {
        PyString::new(self.call.obj.py(), s)
    }
}

impl Host for PyHost<'_, '_> {
    type Error = Fail;

    fn wcswidth(&mut self, s: &str) -> FResult<isize> {
        if let Some(&w) = self.widths.get(s) {
            return Ok(w);
        }
        let f = self
            .call
            .wcswidth
            .as_ref()
            .ok_or(Fail::Unsupported("wcswidth"))?;
        let w: isize = f.call1((PyString::new(f.py(), s),))?.extract()?;
        self.widths.insert(s.to_owned(), w);
        Ok(w)
    }

    fn strip_ansi(&mut self, s: &str) -> FResult<String> {
        let out = self.call.strip_ansi_fn.call1((self.py_str(s),))?;
        Ok(to_backed(&out)?.to_string())
    }

    fn afterpoint(&mut self, s: &str) -> FResult<isize> {
        Ok(self
            .call
            .afterpoint_fn
            .call1((self.py_str(s),))?
            .extract()?)
    }

    fn line(&mut self, id: usize, colwidths: &[isize], aligns: AlignList) -> FResult<Piece> {
        let py = self.call.obj.py();
        let f = &self.callables[id];
        let a = self.aligns[aligns as usize]
            .clone()
            .unwrap_or_else(|| py.None().into_bound(py));
        let out = f.call1((PyList::new(py, colwidths)?, a))?;
        piece(&out)
    }

    fn row(
        &mut self,
        id: usize,
        cells: Cells<'_>,
        colwidths: &[isize],
        aligns: AlignList,
    ) -> FResult<Piece> {
        let py = self.call.obj.py();
        let f = &self.callables[id];
        let a = self.aligns[aligns as usize]
            .clone()
            .unwrap_or_else(|| py.None().into_bound(py));
        let cells: Bound<'_, PyAny> = match cells {
            Cells::List(l) => PyList::new(py, l)?.into_any(),
            Cells::Separator => PyString::new(py, "\u{1}").into_any(),
        };
        let out = f.call1((cells, PyList::new(py, colwidths)?, a))?;
        piece(&out)
    }
}

fn piece(out: &Bound<'_, PyAny>) -> FResult<Piece> {
    if out.is_instance_of::<PyString>() {
        Ok(Piece::Str(to_backed(out)?.to_string()))
    } else {
        Ok(Piece::Other(tp_name(out)))
    }
}

/// A `TableFormat` description from Python: `("builtin", name)` or six field descriptions
/// plus padding and the four `with_header_hide` flags.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
fn table_format<'py>(
    desc: &Bound<'py, PyAny>,
    callables: &mut Vec<Bound<'py, PyAny>>,
) -> FResult<Cow<'static, TableFormat>> {
    let t = desc.cast::<PyTuple>().map_err(PyErr::from)?;
    let kind: PyBackedStr = t.get_item(0)?.extract()?;
    if &*kind == "builtin" {
        let name: PyBackedStr = t.get_item(1)?.extract()?;
        return BUILTIN_FORMATS
            .get(&*name)
            .map(Cow::Borrowed)
            .ok_or(Fail::Unsupported("an unknown built-in format"));
    }
    let mut field = |i: usize, line: bool| -> FResult<(Option<LineFmt>, Option<RowFmt>)> {
        let d = t.get_item(i)?;
        if d.is_none() {
            return Ok((Some(LineFmt::Off), Some(RowFmt::Off)));
        }
        let d = d.cast_into::<PyTuple>().map_err(PyErr::from)?;
        let tag: PyBackedStr = d.get_item(0)?.extract()?;
        let s = |j: usize| -> FResult<String> {
            Ok(d.get_item(j)?.extract::<PyBackedStr>()?.to_string())
        };
        Ok(match &*tag {
            "plain" if line => (
                Some(LineFmt::Plain {
                    begin: s(1)?,
                    fill: s(2)?,
                    sep: s(3)?,
                    end: s(4)?,
                }),
                None,
            ),
            "plain" => (
                None,
                Some(RowFmt::Plain {
                    begin: s(1)?,
                    sep: s(2)?,
                    end: s(3)?,
                }),
            ),
            "call" => {
                callables.push(d.get_item(1)?);
                let id = callables.len() - 1;
                (Some(LineFmt::Custom(id)), Some(RowFmt::Custom(id)))
            }
            "native" => {
                let code = s(1)?;
                let l = match code.as_str() {
                    "pipe" => Some(LineFmt::PipeColons),
                    "grid_colons" => Some(LineFmt::GridColons),
                    "html_begin" => Some(LineFmt::HtmlBegin),
                    "latex_begin" => Some(LineFmt::LatexBegin {
                        booktabs: false,
                        longtable: false,
                    }),
                    "latex_booktabs" => Some(LineFmt::LatexBegin {
                        booktabs: true,
                        longtable: false,
                    }),
                    "latex_longtable" => Some(LineFmt::LatexBegin {
                        booktabs: false,
                        longtable: true,
                    }),
                    "asciidoc_line" => Some(LineFmt::Asciidoc),
                    _ => None,
                };
                let r = match code.as_str() {
                    "mediawiki_header" => Some(RowFmt::Mediawiki('!')),
                    "mediawiki" => Some(RowFmt::Mediawiki('|')),
                    "moin_header" => Some(RowFmt::Moin { header: true }),
                    "moin" => Some(RowFmt::Moin { header: false }),
                    "html_th" => Some(RowFmt::Html {
                        th: true,
                        unsafe_: false,
                    }),
                    "html_td" => Some(RowFmt::Html {
                        th: false,
                        unsafe_: false,
                    }),
                    "unsafehtml_th" => Some(RowFmt::Html {
                        th: true,
                        unsafe_: true,
                    }),
                    "unsafehtml_td" => Some(RowFmt::Html {
                        th: false,
                        unsafe_: true,
                    }),
                    "latex_row" => Some(RowFmt::Latex { raw: false }),
                    "latex_raw_row" => Some(RowFmt::Latex { raw: true }),
                    "textile" => Some(RowFmt::Textile),
                    "asciidoc_header" => Some(RowFmt::Asciidoc { header: true }),
                    "asciidoc_row" => Some(RowFmt::Asciidoc { header: false }),
                    _ => None,
                };
                (l, r)
            }
            _ => (None, None),
        })
    };
    let mut lines = Vec::with_capacity(4);
    for i in 1..=4 {
        lines.push(
            field(i, true)?
                .0
                .ok_or(Fail::Unsupported("a line format"))?,
        );
    }
    let headerrow = field(5, false)?
        .1
        .ok_or(Fail::Unsupported("a row format"))?;
    let datarow = field(6, false)?
        .1
        .ok_or(Fail::Unsupported("a row format"))?;
    let padding: isize = t.get_item(7)?.extract()?;
    let hide: [bool; 4] = t.get_item(8)?.extract()?;
    let mut it = lines.into_iter();
    Ok(Cow::Owned(TableFormat {
        lineabove: it.next().expect("4"),
        linebelowheader: it.next().expect("4"),
        linebetweenrows: it.next().expect("4"),
        linebelow: it.next().expect("4"),
        headerrow,
        datarow,
        padding,
        hide,
    }))
}

/// A simple alignment value (`str`, `None` or `bool`; the Python side delegates others).
fn align_of(obj: &Bound<'_, PyAny>) -> FResult<Align> {
    if let Ok(s) = obj.cast_exact::<PyString>() {
        let s = backed(s)?;
        return Ok(match &*s {
            "right" => Align::Right,
            "center" => Align::Center,
            "decimal" => Align::Decimal,
            "left" => Align::Left,
            "" => Align::Falsy,
            _ => Align::Other,
        });
    }
    if obj.is_none() {
        return Ok(Align::Falsy);
    }
    if obj.is_exact_instance_of::<PyBool>() {
        return Ok(if obj.is_truthy()? {
            Align::Other
        } else {
            Align::Falsy
        });
    }
    Err(Fail::Unsupported("an alignment that is not a str"))
}

fn aligns_of(list: &Bound<'_, PyAny>) -> FResult<Vec<Align>> {
    let mut out = Vec::new();
    for a in list.try_iter()? {
        out.push(align_of(&a?)?);
    }
    Ok(out)
}

fn row_align_of(obj: &Bound<'_, PyAny>) -> FResult<RowAlign> {
    if let Ok(s) = obj.cast_exact::<PyString>() {
        let s = backed(s)?;
        return Ok(match &*s {
            "bottom" => RowAlign::Bottom,
            "center" => RowAlign::Center,
            _ => RowAlign::Top,
        });
    }
    if obj.is_none() || obj.is_exact_instance_of::<PyBool>() {
        return Ok(RowAlign::Top);
    }
    // `row_alignment == "bottom"` on another object.
    if obj.eq("bottom")? {
        Ok(RowAlign::Bottom)
    } else if obj.eq("center")? {
        Ok(RowAlign::Center)
    } else {
        Ok(RowAlign::Top)
    }
}

/// tabulate's `_is_separating_line` for a list, from its first two items: one is a `str` that
/// strips to `SEPARATING_LINE`.
fn is_separating_line(first: Option<&Bound<'_, PyAny>>, second: Option<&Bound<'_, PyAny>>) -> bool {
    let value = |v: Option<&Bound<'_, PyAny>>| {
        v.and_then(|v| v.cast_exact::<PyString>().ok())
            .and_then(|s| s.to_str().ok().map(|t| pystr::strip(t) == "\u{1}"))
            .unwrap_or(false)
    };
    value(first) || value(second)
}

/// `_strip_ansi(s)` for typing (`_type` always strips).
fn strip_for_type<'s>(host: &mut PyHost<'_, '_>, s: &'s str) -> FResult<Cow<'s, str>> {
    match pyroid_core::tabulate::ansi::strip(s) {
        Ok(None) => Ok(Cow::Borrowed(s)),
        Ok(Some(t)) => Ok(Cow::Owned(t)),
        Err(()) => Ok(Cow::Owned(host.strip_ansi(s)?)),
    }
}

/// `float(s)` for a str.
enum PyFloat64 {
    Value(f64),
    /// CPython raises `ValueError`.
    Invalid,
    /// Only the interpreter knows (Unicode digits or spaces).
    Ask,
}

fn py_float_of(s: &str, unicode_ok: bool) -> PyFloat64 {
    if s.is_ascii() {
        number::float_ascii(s.as_bytes()).map_or(PyFloat64::Invalid, PyFloat64::Value)
    } else if number::may_be_number(s, unicode_ok) {
        PyFloat64::Ask
    } else {
        PyFloat64::Invalid
    }
}

struct Column {
    float_spec: Option<Spec>,
    int_spec: Option<Spec>,
    int_fmt_empty: bool,
}

/// A formatted cell: borrowed from the cell, or a range of the column's text buffer.
#[derive(Clone, Copy)]
enum Fmt<'c> {
    Borrowed(&'c str),
    Range(usize, usize),
}

/// Append `text` to `arena` and return its range.
fn put<'c>(arena: &mut String, text: &str) -> Fmt<'c> {
    let start = arena.len();
    arena.push_str(text);
    Fmt::Range(start, arena.len())
}

/// Format one native cell (new text goes to `arena`); `None` means "use tabulate's `_format`".
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn format_native<'c>(
    host: &mut PyHost<'_, '_>,
    cell: &'c Cell<'_>,
    ct: ColType,
    col: &Column,
    has_invisible: bool,
    unicode_ok: bool,
    arena: &mut String,
) -> FResult<Option<Fmt<'c>>> {
    let start = arena.len();
    let range = |arena: &String| Some(Fmt::Range(start, arena.len()));
    Ok(match (&cell.val, ct) {
        (Val::Str(s), _) if s.is_empty() => Some(Fmt::Borrowed("")),
        // Other objects, bytes columns and numpy scalars in text columns: tabulate's _format.
        (Val::Obj, _) | (_, ColType::Bytes) | (Val::NpInt(_) | Val::NpFloat(_), ColType::Str) => {
            None
        }
        (Val::Str(s), ColType::Str | ColType::None | ColType::Bool) => Some(Fmt::Borrowed(s)),
        (Val::Bool(b), ColType::Str | ColType::None | ColType::Bool) => {
            Some(Fmt::Borrowed(if *b { "True" } else { "False" }))
        }
        (Val::Int(i), ColType::Str) => {
            EMPTY_INT.format_int_into(i128::from(*i), arena);
            range(arena)
        }
        (Val::Float(f), ColType::Str) => Some(put(arena, &pyfmt::float_repr(*f))),
        // An int column: str cells keep their text unless tabulate's colored-number regex
        // could match (it needs a backslash in the unicode_escape form).
        (Val::Str(s), ColType::Int) => {
            if pystr::is_printable_ascii(s) && !s.contains('\\') {
                Some(Fmt::Borrowed(s))
            } else {
                None
            }
        }
        (Val::Int(i), ColType::Int) => match &col.int_spec {
            Some(spec) => {
                spec.format_int_into(i128::from(*i), arena);
                range(arena)
            }
            None => None,
        },
        (Val::NpInt(i), ColType::Int) => {
            if col.int_fmt_empty {
                EMPTY_INT.format_int_into(i128::from(*i), arena);
                range(arena)
            } else {
                None
            }
        }
        (Val::Bool(b), ColType::Int) => {
            col.int_fmt_empty
                .then_some(Fmt::Borrowed(if *b { "True" } else { "False" }))
        }
        (Val::Str(s), ColType::Float) => {
            let Some(spec) = &col.float_spec else {
                return Ok(None);
            };
            if has_invisible {
                let raw = strip_for_type(host, s)?;
                match py_float_of(&raw, unicode_ok) {
                    PyFloat64::Ask => None,
                    PyFloat64::Invalid => Some(Fmt::Borrowed(s)),
                    PyFloat64::Value(f) => {
                        let formatted = spec.format_float(f);
                        Some(put(arena, &s.replace(&*raw, &formatted)))
                    }
                }
            } else {
                let v: Cow<'_, str> = if s.contains(',') {
                    Cow::Owned(s.replace(',', ""))
                } else {
                    Cow::Borrowed(&**s)
                };
                match py_float_of(&v, unicode_ok) {
                    PyFloat64::Ask => None,
                    PyFloat64::Invalid => Some(match v {
                        Cow::Borrowed(b) => Fmt::Borrowed(b),
                        Cow::Owned(o) => put(arena, &o),
                    }),
                    PyFloat64::Value(f) => {
                        spec.format_float_into(f, arena);
                        range(arena)
                    }
                }
            }
        }
        (
            Val::Int(_) | Val::NpInt(_) | Val::Float(_) | Val::NpFloat(_) | Val::Bool(_),
            ColType::Float,
        ) => {
            let Some(spec) = &col.float_spec else {
                return Ok(None);
            };
            #[allow(clippy::cast_precision_loss)]
            let f = match cell.val {
                Val::Int(i) | Val::NpInt(i) => i as f64,
                Val::Float(f) | Val::NpFloat(f) => f,
                Val::Bool(b) => f64::from(u8::from(b)),
                _ => unreachable!(),
            };
            spec.format_float_into(f, arena);
            range(arena)
        }
        (Val::None, _) => unreachable!("handled by the caller"),
        _ => None,
    })
}

/// `format(n, "")`.
static EMPTY_INT: std::sync::LazyLock<Spec> =
    std::sync::LazyLock::new(|| Spec::parse("", true).expect("the empty spec parses"));

/// `render(rows, headers, separating_lines, call)`: `(text, multiline, empty)` (`empty`: no
/// headers and no rows, so the text is `""` whatever the format), or `(None, reason, None)`
/// when the call has to go to tabulate.
#[pyfunction]
fn render<'py>(
    py: Python<'py>,
    rows: Bound<'py, PyList>,
    headers: Bound<'py, PyList>,
    separating_lines: Bound<'py, PyAny>,
    call: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let call = Call::new(&call)?;
    match render_inner(py, &rows, &headers, &separating_lines, &call) {
        Ok((text, multiline, empty)) => Ok(PyTuple::new(
            py,
            [
                PyString::new(py, &text).into_any(),
                PyBool::new(py, multiline).to_owned().into_any(),
                PyBool::new(py, empty).to_owned().into_any(),
            ],
        )?
        .into_any()),
        Err(Fail::Py(e)) => Err(e),
        Err(Fail::Unsupported(reason)) => Ok(PyTuple::new(
            py,
            [
                py.None().into_bound(py),
                PyString::new(py, reason).into_any(),
                py.None().into_bound(py),
            ],
        )?
        .into_any()),
    }
}

fn table_error(
    py: Python<'_>,
    e: table::Error<Fail>,
    aligns: &[Option<Bound<'_, PyAny>>; 2],
) -> Fail {
    let err = match e {
        table::Error::Host(f) => return f,
        table::Error::NegativeWidth => {
            PyValueError::new_err("Sign not allowed in string format specifier")
        }
        table::Error::AsciidocKey { list, index } => {
            let key = aligns[list as usize]
                .as_ref()
                .and_then(|l| l.get_item(index).ok())
                .unwrap_or_else(|| py.None().into_bound(py));
            PyKeyError::new_err((key.unbind(),))
        }
        table::Error::TextileSeparator => {
            PyTypeError::new_err("'str' object does not support item assignment")
        }
        table::Error::TextileEmpty | table::Error::RowAlignIndex => {
            PyIndexError::new_err("list index out of range")
        }
        table::Error::EmptyMax => {
            // max() of nothing: let the interpreter word it.
            match py
                .import("builtins")
                .and_then(|b| b.getattr("max"))
                .and_then(|m| m.call1((PyTuple::empty(py),)))
            {
                Err(e) => e,
                Ok(_) => PyValueError::new_err("max() arg is an empty sequence"),
            }
        }
        table::Error::Join { index, type_name } => PyTypeError::new_err(format!(
            "sequence item {index}: expected str instance, {type_name} found"
        )),
    };
    Fail::Py(err)
}

#[allow(clippy::too_many_lines, clippy::similar_names)]
fn render_inner<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyList>,
    headers_in: &Bound<'py, PyList>,
    separating_lines: &Bound<'py, PyAny>,
    call: &Call<'py>,
) -> FResult<(String, bool, bool)> {
    let unicode_ok = call.flag("unicode_ok")?;
    let max_digits: usize = call.obj.getattr("max_digits")?.extract()?;
    let is_rst = call.flag("is_rst")?;
    let mut callables = Vec::new();
    let fmt_plain = table_format(&call.obj.getattr("fmt_plain")?, &mut callables)?;
    let fmt_multi = {
        let d = call.obj.getattr("fmt_multi")?;
        if d.is_none() {
            None
        } else {
            Some(table_format(&d, &mut callables)?)
        }
    };
    let mut host = PyHost {
        call,
        widths: FxHashMap::default(),
        callables,
        aligns: [None, None],
    };

    // Headers (str after normalisation) and cells, as tabulate's columns (zip_longest).
    let mut headers: Vec<PyBackedStr> = Vec::with_capacity(headers_in.len());
    for h in headers_in.iter() {
        headers.push(to_backed(&h)?);
    }
    // Raw rows (a plain list given to tabulate): tabulate's `list(r)` for each row, and
    // `_remove_separating_lines` on the result, as its normalisation does.
    let raw = call.flag("raw_rows")?;
    let list_type = builtin_types(py)?[6].bind(py);
    let index_obj = call.obj.getattr("index")?;
    let index: Option<Bound<'py, PyList>> = if raw && !index_obj.is_none() {
        Some(index_obj.cast_into::<PyList>().map_err(PyErr::from)?)
    } else {
        None
    };
    let mut next_index = 0;
    let mut found_seps: Vec<usize> = Vec::new();
    let mut table_rows: Vec<Vec<Cell<'py>>> = Vec::with_capacity(rows.len());
    let mut row_lens: Vec<usize> = Vec::with_capacity(rows.len());
    let mut ncols = 0;
    for (position, row) in rows.iter().enumerate() {
        let row = match row.cast_exact::<PyList>() {
            Ok(l) => l.clone(),
            Err(_) if raw => list_type
                .call1((row,))?
                .cast_into::<PyList>()
                .map_err(PyErr::from)?,
            Err(_) => return Err(Fail::Unsupported("a row that is not a list")),
        };
        if raw && is_separating_line(row.get_item(0).ok().as_ref(), row.get_item(1).ok().as_ref()) {
            found_seps.push(position);
            continue;
        }
        // _prepend_row_index: the index goes to the rows that are not separators, and the
        // result can be one.
        let index_value = match &index {
            Some(ix) => {
                let v = ix.get_item(next_index)?;
                next_index += 1;
                if is_separating_line(Some(&v), row.get_item(0).ok().as_ref()) {
                    found_seps.push(position);
                    continue;
                }
                Some(v)
            }
            None => None,
        };
        let mut cells = Vec::with_capacity(row.len() + 1);
        for (c, item) in index_value.into_iter().chain(row.iter()).enumerate() {
            let item = if is_rst && c == 0 {
                escape_empty(call, item)?
            } else {
                item
            };
            cells.push(classify(call, item)?);
        }
        row_lens.push(cells.len());
        ncols = ncols.max(cells.len());
        table_rows.push(cells);
    }
    let separating_lines: Bound<'py, PyAny> = if raw {
        PyList::new(py, &found_seps)?.into_any()
    } else {
        separating_lines.clone()
    };
    if is_rst && !headers.is_empty() && pystr::strip(&headers[0]).is_empty() {
        headers[0] = to_backed(PyString::new(py, "..").as_any())?;
    }
    let nrows = table_rows.len();
    let mut cols: Vec<Vec<Cell<'py>>> = (0..ncols).map(|_| Vec::with_capacity(nrows)).collect();
    for row in table_rows {
        let n = row.len();
        for (j, cell) in row.into_iter().enumerate() {
            cols[j].push(cell);
        }
        for col in cols.iter_mut().skip(n) {
            col.push(Cell {
                obj: py.None().into_bound(py),
                val: Val::None,
            });
        }
    }

    // plain_text: the ANSI search and the multiline test over the tab-joined cells.
    let multiline_ok = call.flag("multiline_candidate")?;
    let mut any_esc = false;
    let mut any_newline = false;
    for h in &headers {
        let (e, n) = pystr::esc_and_newline(h);
        any_esc |= e;
        any_newline |= n;
    }
    let mut obj_texts: FxHashMap<(usize, usize), String> = FxHashMap::default();
    for (j, col) in cols.iter().enumerate() {
        for (i, cell) in col.iter().enumerate() {
            match &cell.val {
                Val::Str(s) => {
                    let (e, n) = pystr::esc_and_newline(s);
                    any_esc |= e;
                    any_newline |= n;
                }
                Val::Obj => {
                    let t = call.to_str_fn.call1((&cell.obj,))?;
                    let t = to_backed(&t)?;
                    any_esc |= t.contains('\u{1b}');
                    any_newline |= pystr::has_cr_lf(&t);
                    obj_texts.insert((i, j), t.to_string());
                }
                _ => {}
            }
        }
    }
    let has_invisible = if any_esc {
        // Rebuild "\t".join(...) in tabulate's order: headers, then the rows as normalised
        // (ragged rows are not padded here).
        let mut joined = String::new();
        let mut first = true;
        let mut push = |s: &str| {
            if !first {
                joined.push('\t');
            }
            first = false;
            joined.push_str(s);
        };
        for h in &headers {
            push(h);
        }
        for i in 0..nrows {
            for (j, col) in cols.iter().enumerate().take(row_lens[i]) {
                let cell = &col[i];
                let text: Cow<'_, str> = match &cell.val {
                    Val::Str(s) => Cow::Borrowed(&**s),
                    Val::Obj => Cow::Borrowed(obj_texts.get(&(i, j)).map_or("", String::as_str)),
                    Val::None => Cow::Borrowed("None"),
                    Val::Bool(b) => Cow::Borrowed(if *b { "True" } else { "False" }),
                    Val::Int(v) => Cow::Owned(v.to_string()),
                    Val::Float(f) => Cow::Owned(pyfmt::float_repr(*f)),
                    Val::NpInt(_) | Val::NpFloat(_) => {
                        Cow::Owned(to_backed(&cell.obj.str()?.into_any())?.to_string())
                    }
                };
                push(&text);
            }
        }
        match pyroid_core::tabulate::ansi::contains(&joined) {
            number::Tri::Yes => true,
            number::Tri::No => false,
            number::Tri::Ask => call
                .obj
                .getattr("ansi_search_fn")?
                .call1((PyString::new(py, &joined),))?
                .is_truthy()?,
        }
    } else {
        false
    };
    let is_multiline = multiline_ok && any_newline;
    let wide = call.flag("wide")?;
    let settings = Settings {
        has_invisible,
        wide,
        is_multiline,
        preserve_whitespace: call.flag("preserve_whitespace")?,
        max_digits,
        unicode_ok,
    };

    // Plain option values (no lists, nothing that can warn or raise): the option lists are
    // computed here, as the call object's methods would.
    let simple = call.flag("simple_options")?;
    let opt = |name: &str| call.obj.getattr(name);

    // Column types (`_column_type(col, numparse=np)`: `_type` strips ANSI codes always).
    let numparses: Vec<Bound<'py, PyAny>> = if simple {
        let np = PyBool::new(py, !opt("disable_numparse")?.is_truthy()?)
            .to_owned()
            .into_any();
        vec![np; ncols]
    } else {
        call.obj.call_method1("numparses", (ncols,))?.extract()?
    };
    let mut coltypes = Vec::with_capacity(ncols);
    for (j, col) in cols.iter().enumerate() {
        let np_obj = numparses
            .get(j)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
        let numparse = np_obj.is_truthy()?;
        let mut t = ColType::Bool;
        for cell in col {
            let ct = match &cell.val {
                Val::None => ColType::None,
                Val::Bool(_) => ColType::Bool,
                // `_type` checks numparse before int/float, so unparsed numbers are text.
                Val::Int(_) | Val::NpInt(_) if numparse => ColType::Int,
                Val::Float(_) | Val::NpFloat(_) if numparse => ColType::Float,
                Val::Int(_) | Val::NpInt(_) | Val::Float(_) | Val::NpFloat(_) => ColType::Str,
                Val::Str(s) => {
                    let stripped = strip_for_type(&mut host, s)?;
                    match number::str_type(&stripped, numparse, max_digits, unicode_ok) {
                        Some(ct) => ct,
                        None => call.rank(&call.type_fn.call1((&cell.obj, true, &np_obj))?),
                    }
                }
                Val::Obj => call.rank(&call.type_fn.call1((&cell.obj, true, &np_obj))?),
            };
            t = t.max(ct);
        }
        coltypes.push(t);
    }

    // Per-column formats, then the cells as text.
    let (float_formats, int_formats, missing_vals): (Objs<'py>, Objs<'py>, Objs<'py>) = if simple {
        (
            vec![opt("floatfmt")?; ncols],
            vec![opt("intfmt")?; ncols],
            vec![opt("missingval")?; ncols],
        )
    } else {
        call.obj.call_method1("formats", (ncols,))?.extract()?
    };
    let mut arenas: Vec<String> = Vec::with_capacity(ncols);
    let mut fmts: Vec<Vec<Fmt<'_>>> = Vec::with_capacity(ncols);
    for (j, col) in cols.iter().enumerate() {
        let (Some(ff), Some(fi), Some(mv)) = (
            float_formats.get(j),
            int_formats.get(j),
            missing_vals.get(j),
        ) else {
            break;
        };
        let ct = coltypes[j];
        let spec_of = |o: &Bound<'py, PyAny>, for_int: bool| -> Option<Spec> {
            o.cast_exact::<PyString>()
                .ok()
                .and_then(|s| s.to_str().ok().and_then(|s| Spec::parse(s, for_int)))
        };
        let column = Column {
            float_spec: spec_of(ff, false),
            int_spec: spec_of(fi, true),
            int_fmt_empty: fi
                .cast_exact::<PyString>()
                .is_ok_and(|s| s.to_str().is_ok_and(str::is_empty)),
        };
        let missing: Option<PyBackedStr> = match mv.cast_exact::<PyString>() {
            Ok(s) => Some(backed(s)?),
            Err(_) => None,
        };
        let ct_obj = call.col_type_obj(ct);
        let mut out = Vec::with_capacity(col.len());
        let mut arena = String::with_capacity(col.len() * 8);
        for cell in col {
            if matches!(cell.val, Val::None) {
                let m = missing
                    .as_ref()
                    .ok_or(Fail::Unsupported("a missingval that is not a str"))?;
                out.push(put(&mut arena, m));
                continue;
            }
            let f = if let Some(f) = format_native(
                &mut host,
                cell,
                ct,
                &column,
                has_invisible,
                unicode_ok,
                &mut arena,
            )? {
                f
            } else {
                let r = call
                    .format_fn
                    .call1((&cell.obj, ct_obj, ff, fi, mv, has_invisible))?;
                put(&mut arena, &to_backed(&r)?)
            };
            out.push(f);
        }
        arenas.push(arena);
        fmts.push(out);
    }
    let text_cols: Vec<Vec<Cow<'_, str>>> = fmts
        .into_iter()
        .zip(&arenas)
        .map(|(col, arena)| {
            col.into_iter()
                .map(|f| match f {
                    Fmt::Borrowed(s) => Cow::Borrowed(s),
                    Fmt::Range(a, b) => Cow::Borrowed(&arena[a..b]),
                })
                .collect()
        })
        .collect();

    // Alignments (the call object applies colglobalalign, colalign and warns like tabulate).
    let codes: Vec<u8> = coltypes.iter().map(|&t| t as u8).collect();
    let aligns_obj = if simple {
        let global = opt("colglobalalign")?;
        let (num, text) = (opt("numalign")?, opt("stralign")?);
        let items: Vec<Bound<'py, PyAny>> = coltypes
            .iter()
            .map(|t| {
                if !global.is_none() {
                    global.clone()
                } else if t.is_numeric() {
                    num.clone()
                } else {
                    text.clone()
                }
            })
            .collect();
        PyList::new(py, items)?.into_any()
    } else {
        call.obj.call_method1("aligns", (codes,))?
    };
    let aligns = aligns_of(&aligns_obj)?;
    let colon = if is_multiline {
        call.flag("colon_multi")?
    } else {
        call.flag("colon_plain")?
    };
    let min_padding: isize = call.obj.getattr("min_padding")?.extract()?;

    let mut engine = Engine {
        host: &mut host,
        s: settings,
    };
    let mut minwidths = Vec::new();
    if headers.is_empty() {
        minwidths.resize(text_cols.len(), 0);
    } else {
        for h in &headers {
            minwidths.push(
                engine
                    .width(h)
                    .map_err(|e| table_error(py, e, &[None, None]))?
                    + min_padding,
            );
        }
    }
    let n_aligned = text_cols.len().min(aligns.len()).min(minwidths.len());
    let mut aligned_cols: Vec<Vec<table::Aligned<'_>>> = Vec::with_capacity(n_aligned);
    for (j, strings) in text_cols.into_iter().enumerate().take(n_aligned) {
        let a = if colon { Align::Left } else { aligns[j] };
        aligned_cols.push(
            engine
                .align_column(strings, a, minwidths[j])
                .map_err(|e| table_error(py, e, &[None, None]))?,
        );
    }

    let mut header_texts: Vec<String> = Vec::new();
    let mut final_aligns_obj = aligns_obj.clone();
    let mut header_aligns: Vec<Align> = Vec::new();
    let mut header_aligns_obj: Option<Bound<'py, PyAny>> = None;
    let colwidths: Vec<isize> = if headers.is_empty() {
        let mut w = Vec::with_capacity(aligned_cols.len());
        for c in &aligned_cols {
            let mut m = isize::MIN;
            for cl in c {
                m = m.max(
                    engine
                        .aligned_width(cl)
                        .map_err(|e| table_error(py, e, &[None, None]))?,
                );
            }
            w.push(m);
        }
        w
    } else {
        let n_t = if aligned_cols.is_empty() {
            headers.len()
        } else {
            aligned_cols.len()
        };
        let (a2, ah): (Bound<'py, PyAny>, Bound<'py, PyAny>) = if simple {
            let global = opt("headersglobalalign")?;
            let ah = if !global.is_none() {
                PyList::new(py, vec![global; n_t])?.into_any()
            } else if aligns.is_empty() {
                PyList::new(py, vec![opt("stralign")?; headers.len()])?.into_any()
            } else {
                aligns_obj.clone()
            };
            (aligns_obj.clone(), ah)
        } else {
            call.obj
                .call_method1("header_aligns", (&aligns_obj, n_t))?
                .extract()?
        };
        final_aligns_obj = a2;
        header_aligns = aligns_of(&ah)?;
        header_aligns_obj = Some(ah);
        let mut w = Vec::with_capacity(n_t);
        for (j, &minw) in minwidths.iter().enumerate().take(n_t) {
            let m = if aligned_cols.is_empty() {
                engine
                    .width("")
                    .map_err(|e| table_error(py, e, &[None, None]))?
            } else {
                let mut m = isize::MIN;
                for cl in &aligned_cols[j] {
                    m = m.max(
                        engine
                            .aligned_width(cl)
                            .map_err(|e| table_error(py, e, &[None, None]))?,
                    );
                }
                m
            };
            w.push(minw.max(m));
        }
        let n_h = headers.len().min(header_aligns.len()).min(w.len());
        for j in 0..n_h {
            header_texts.push(
                engine
                    .align_header(&headers[j], header_aligns[j], w[j])
                    .map_err(|e| table_error(py, e, &[None, None]))?,
            );
        }
        w
    };
    let col_aligns = aligns_of(&final_aligns_obj)?;

    // rows = list(zip(*cols)), the row alignments, then the separators back in place.
    let nr = aligned_cols.iter().map(Vec::len).min().unwrap_or(0);
    let mut order: Vec<RowRef> = (0..nr).map(RowRef::Data).collect();
    // Only multiline rows compare the row alignments; the count matters either way.
    let rowaligns_obj = if simple {
        let r = opt("rowalign")?; // None or a str: _expand_iterable gives [rowalign] * nr
        PyList::new(py, vec![r; nr])?.into_any()
    } else {
        call.obj.call_method1("rowaligns", (nr,))?
    };
    let mut rowaligns = Vec::new();
    for a in rowaligns_obj.try_iter()? {
        let a = a?;
        rowaligns.push(if is_multiline {
            row_align_of(&a)?
        } else {
            RowAlign::Top
        });
    }
    if !separating_lines.is_none() {
        for idx in separating_lines.try_iter()? {
            let idx: usize = idx?.extract()?;
            let at = idx.min(order.len());
            order.insert(at, RowRef::Separator);
        }
    }

    let fmt: &TableFormat = if is_multiline {
        fmt_multi.as_ref().unwrap_or(&fmt_plain)
    } else {
        &fmt_plain
    };
    engine.host.aligns = [Some(final_aligns_obj.clone()), header_aligns_obj.clone()];
    let align_objs = [Some(final_aligns_obj), header_aligns_obj];
    let text = engine
        .format_table(
            fmt,
            &header_texts,
            &header_aligns,
            &aligned_cols,
            &order,
            &colwidths,
            &col_aligns,
            &rowaligns,
        )
        .map_err(|e| table_error(py, e, &align_objs))?;
    Ok((
        text,
        is_multiline,
        header_texts.is_empty() && order.is_empty(),
    ))
}

/// The rst first-column escape for one cell (`escape_empty`).
fn escape_empty<'py>(call: &Call<'py>, item: Bound<'py, PyAny>) -> FResult<Bound<'py, PyAny>> {
    if let Ok(s) = item.cast_exact::<PyString>() {
        let b = backed(s)?;
        if pystr::strip(&b).is_empty() {
            return Ok(PyString::new(item.py(), "..").into_any());
        }
        return Ok(item);
    }
    if item.is_instance_of::<PyString>() || item.is_instance_of::<PyBytes>() {
        return Ok(call.escape_empty_fn.call1((item,))?);
    }
    Ok(item)
}

/// `_align_column(strings, alignment, minwidth, has_invisible, enable_widechars,
/// is_multiline, preserve_whitespace)`, for tabulate's own tests of that helper; `call`
/// provides `wcswidth`, `strip_ansi_fn` and `afterpoint_fn`.
#[pyfunction]
#[pyo3(signature = (strings, alignment, minwidth, has_invisible, enable_widechars, is_multiline, preserve_whitespace, call))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn _align_column(
    py: Python<'_>,
    strings: Vec<String>,
    alignment: Bound<'_, PyAny>,
    minwidth: isize,
    has_invisible: bool,
    enable_widechars: bool,
    is_multiline: bool,
    preserve_whitespace: bool,
    call: Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let call = Call::new(&call)?;
    let mut host = PyHost {
        call: &call,
        widths: FxHashMap::default(),
        callables: Vec::new(),
        aligns: [None, None],
    };
    let settings = Settings {
        has_invisible,
        wide: enable_widechars,
        is_multiline,
        preserve_whitespace,
        max_digits: call.obj.getattr("max_digits")?.extract()?,
        unicode_ok: call.flag("unicode_ok")?,
    };
    let unsupported = |e: Fail| match e {
        Fail::Py(e) => e,
        Fail::Unsupported(r) => PyValueError::new_err(r),
    };
    let a = align_of(&alignment).map_err(unsupported)?;
    let mut engine = Engine {
        host: &mut host,
        s: settings,
    };
    let strings: Vec<Cow<'_, str>> = strings.iter().map(|s| Cow::Borrowed(s.as_str())).collect();
    let out = engine
        .align_column(strings, a, minwidth)
        .map_err(|e| unsupported(table_error(py, e, &[None, None])))?;
    Ok(out.iter().map(|c| c.padded().into_owned()).collect())
}

/// `_align_cell_veritically(text_lines, num_lines, column_width, row_alignment)`.
#[pyfunction]
fn _align_cell_vertically(
    lines: Vec<String>,
    num_lines: usize,
    width: isize,
    row_alignment: Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let a = row_align_of(&row_alignment).map_err(|e| match e {
        Fail::Py(e) => e,
        Fail::Unsupported(r) => PyValueError::new_err(r),
    })?;
    let refs: Vec<&str> = lines.iter().map(String::as_str).collect();
    Ok(table::align_cell_vertically(refs, num_lines, width, a)
        .into_iter()
        .map(Cow::into_owned)
        .collect())
}

/// The Unicode version of Rust's `char` tables (`(major, minor, update)`).
const UNICODE_VERSION: (u8, u8, u8) = char::UNICODE_VERSION;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.tabulate")?;
    m.gil_used(false)?;
    m.add_function(wrap_pyfunction!(render, &m)?)?;
    m.add_function(wrap_pyfunction!(_align_column, &m)?)?;
    m.add_function(wrap_pyfunction!(_align_cell_vertically, &m)?)?;
    m.add("UNICODE_VERSION", UNICODE_VERSION)?;
    m.add("FORMAT_NAMES", table::FORMAT_NAMES.to_vec())?;
    parent.add_submodule(&m)
}
