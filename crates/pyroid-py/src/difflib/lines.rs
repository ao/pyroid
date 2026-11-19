//! The line-based functions: `Differ.compare`/`ndiff`, `_mdiff`,
//! `unified_diff`, `context_diff`, and `get_close_matches`.
//!
//! Each takes exact `list`/`tuple` inputs of exact `str` (read in one step,
//! as another thread may change a list), or an exact `str` (one line per
//! character, as difflib reads it), and returns `None` for anything else;
//! the Python side then runs CPython's code.

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyList, PyString, PyTuple};
use pyroid_core::difflib::differ::{self, CharJunk, DiffOp, DifferVersion, Side};
use pyroid_core::difflib::format::{self, Out, Prefix};
use pyroid_core::difflib::html;
use pyroid_core::difflib::intern::Interner;
use pyroid_core::difflib::matcher::{BIndex, grouped_opcodes, opcodes};
use pyroid_core::difflib::mdiff::{self, Cell, Text};
use pyroid_core::difflib::{StrMatcher, nlargest, with_work};
use pyroid_core::par;
use pyroid_core::text::is_py_whitespace;

use super::interrupt;
use super::seq::{Base, Types, Verdict, code_points, push_wtf8};
use crate::batch::BatchError;
use crate::input::{self, Threads};

/// A line's text: its UTF-8, or (lone surrogates) the `surrogatepass` bytes.
enum LineText {
    Utf8(PyBackedStr),
    Wtf8(Box<[u8]>, Vec<u32>),
}

impl LineText {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Utf8(s) => s.as_bytes(),
            Self::Wtf8(b, _) => b,
        }
    }

    fn push_code_points(&self, out: &mut Vec<u32>) {
        match self {
            Self::Utf8(s) => out.extend(s.chars().map(u32::from)),
            Self::Wtf8(_, cps) => out.extend_from_slice(cps),
        }
    }
}

/// An exact `list`/`tuple` of exact `str`, or the characters of an exact `str`.
struct Lines<'py> {
    objs: Bound<'py, PyTuple>,
    text: Vec<LineText>,
}

impl<'py> Lines<'py> {
    fn read(obj: &Bound<'py, PyAny>) -> PyResult<Option<Self>> {
        let objs = if let Ok(list) = obj.cast_exact::<PyList>() {
            list.to_tuple()
        } else if let Ok(tuple) = obj.cast_exact::<PyTuple>() {
            tuple.clone()
        } else if let Ok(s) = obj.cast_exact::<PyString>() {
            // `ndiff("kitten", "sitting")`: difflib indexes and iterates a str, so each character
            // is a line. `tuple(s)` gives those one-character strings.
            obj.py()
                .get_type::<PyTuple>()
                .call1((s,))?
                .cast_into::<PyTuple>()?
        } else {
            return Ok(None);
        };
        let mut text = Vec::with_capacity(objs.len());
        for item in objs.iter() {
            let Ok(s) = item.cast_exact::<PyString>() else {
                return Ok(None);
            };
            text.push(line_text(s)?);
        }
        Ok(Some(Self { objs, text }))
    }

    fn len(&self) -> usize {
        self.text.len()
    }

    fn get(&self, i: usize) -> Bound<'py, PyAny> {
        self.objs.get_item(i).expect("an index within the snapshot")
    }

    /// Every line's code points, as slices of one buffer.
    fn code_points(&self) -> (Vec<u32>, Vec<usize>) {
        let mut buf = Vec::new();
        let mut ends = Vec::with_capacity(self.len());
        for t in &self.text {
            t.push_code_points(&mut buf);
            ends.push(buf.len());
        }
        (buf, ends)
    }
}

fn line_text(s: &Bound<'_, PyString>) -> PyResult<LineText> {
    if let Ok(backed) = PyBackedStr::try_from(s.clone()) {
        return Ok(LineText::Utf8(backed));
    }
    let bytes = s
        .call_method1(pyo3::intern!(s.py(), "encode"), ("utf-8", "surrogatepass"))?
        .cast_into::<PyBytes>()?
        .as_bytes()
        .into();
    Ok(LineText::Wtf8(bytes, code_points(s)?))
}

fn slices<'a>(buf: &'a [u32], ends: &[usize]) -> Vec<&'a [u32]> {
    let mut start = 0;
    ends.iter()
        .map(|&end| {
            let s = &buf[start..end];
            start = end;
            s
        })
        .collect()
}

/// Line ids: `b`'s numbered by first occurrence, `a`'s looked up.
fn line_ids(a: &Lines<'_>, b: &Lines<'_>) -> (Vec<u32>, Vec<u32>, Vec<usize>) {
    let mut keys: Interner<&[u8]> = Interner::default();
    let mut first = Vec::new();
    let bi = b
        .text
        .iter()
        .enumerate()
        .map(|(j, t)| {
            let (id, new) = keys.intern(t.bytes());
            if new {
                first.push(j);
            }
            id
        })
        .collect();
    let ai = a.text.iter().map(|t| keys.get(t.bytes())).collect();
    (ai, bi, first)
}

/// `linejunk`'s verdicts for `b`'s distinct lines (`first[id]` is the first
/// line with that id), called as CPython's `__chain_b` calls `isjunk`.
fn line_junk(
    b: &Lines<'_>,
    first: &[usize],
    kind: u8,
    linejunk: &Bound<'_, PyAny>,
) -> PyResult<Vec<bool>> {
    match kind {
        1 => {
            let mut cps = Vec::new();
            Ok(first
                .iter()
                .map(|&j| {
                    cps.clear();
                    b.text[j].push_code_points(&mut cps);
                    is_line_junk(&cps)
                })
                .collect())
        }
        2 if linejunk.is_truthy()? => first
            .iter()
            .map(|&j| linejunk.call1((b.get(j),))?.is_truthy())
            .collect(),
        _ => Ok(Vec::new()),
    }
}

/// `Differ(linejunk, charjunk).compare(a, b)`'s lines, made as they are read
/// (an ndiff of large files keeps one list of small ops, not every line), or
/// `None`. `linejunk_kind`: 0 `None`, 1 `IS_LINE_JUNK`, 2 another callable;
/// `charjunk_space_tab`: `IS_CHARACTER_JUNK` (else `None`).
#[pyfunction]
pub fn _differ_compare<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    linejunk_kind: u8,
    linejunk: &Bound<'py, PyAny>,
    charjunk_space_tab: bool,
    windowed: bool,
) -> PyResult<Option<DiffLines>> {
    let (Some(al), Some(bl)) = (Lines::read(a)?, Lines::read(b)?) else {
        return Ok(None);
    };
    let ops = compare(
        py,
        &al,
        &bl,
        linejunk_kind,
        linejunk,
        charjunk_space_tab,
        windowed,
    )?;
    Ok(Some(DiffLines {
        ops,
        a: al.objs.unbind(),
        b: bl.objs.unbind(),
        next: 0,
    }))
}

/// The iterator [`_differ_compare`] returns.
#[pyclass(module = "pyroid.difflib", name = "_DiffLines")]
#[derive(Debug)]
pub struct DiffLines {
    ops: Vec<DiffOp>,
    a: Py<PyTuple>,
    b: Py<PyTuple>,
    next: usize,
}

#[pymethods]
impl DiffLines {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        let py = slf.py();
        let this = &mut *slf;
        let Some(op) = this.ops.get(this.next) else {
            return Ok(None);
        };
        let line = match op {
            DiffOp::Line(tag, side, i) => {
                let prefix = match tag {
                    b'-' => pyo3::intern!(py, "- "),
                    b'+' => pyo3::intern!(py, "+ "),
                    _ => pyo3::intern!(py, "  "),
                };
                let lines = match side {
                    Side::A => this.a.bind(py),
                    Side::B => this.b.bind(py),
                };
                prefix.add(lines.get_item(*i)?)?
            }
            DiffOp::Tags(t) => PyString::new(py, &format!("? {t}\n")).into_any(),
        };
        this.next += 1;
        Ok(Some(line))
    }
}

struct Compared {
    ops: Vec<DiffOp>,
    a_buf: Vec<u32>,
    a_ends: Vec<usize>,
    b_buf: Vec<u32>,
    b_ends: Vec<usize>,
}

fn compare_full(
    py: Python<'_>,
    a: &Lines<'_>,
    b: &Lines<'_>,
    linejunk_kind: u8,
    linejunk: &Bound<'_, PyAny>,
    charjunk_space_tab: bool,
    windowed: bool,
) -> PyResult<Compared> {
    let (ai, bi, first) = line_ids(a, b);
    let junk = line_junk(b, &first, linejunk_kind, linejunk)?;
    let (a_buf, a_ends) = a.code_points();
    let (b_buf, b_ends) = b.code_points();
    let ops = {
        let (ac, bc) = (slices(&a_buf, &a_ends), slices(&b_buf, &b_ends));
        let lines = differ::Lines {
            a: &ai,
            b: &bi,
            b_keys: first.len(),
            b_junk: &junk,
            a_chars: &ac,
            b_chars: &bc,
        };
        let cj = if charjunk_space_tab {
            CharJunk::SpaceTab
        } else {
            CharJunk::Nothing
        };
        let version = if windowed {
            DifferVersion::Windowed
        } else {
            DifferVersion::Classic
        };
        let work = a_buf.len() + b_buf.len();
        interrupt::run(py, work > 1 << 14, || differ::compare(&lines, cj, version))?
    };
    Ok(Compared {
        ops,
        a_buf,
        a_ends,
        b_buf,
        b_ends,
    })
}

fn compare(
    py: Python<'_>,
    a: &Lines<'_>,
    b: &Lines<'_>,
    linejunk_kind: u8,
    linejunk: &Bound<'_, PyAny>,
    charjunk_space_tab: bool,
    windowed: bool,
) -> PyResult<Vec<DiffOp>> {
    Ok(compare_full(
        py,
        a,
        b,
        linejunk_kind,
        linejunk,
        charjunk_space_tab,
        windowed,
    )?
    .ops)
}

/// A `str` from code points (lone surrogates included).
fn str_from_code_points<'py>(py: Python<'py>, cps: &[u32]) -> PyResult<Bound<'py, PyAny>> {
    let text: Option<String> = cps.iter().map(|&c| char::from_u32(c)).collect();
    if let Some(text) = text {
        return Ok(PyString::new(py, &text).into_any());
    }
    let bytes: Vec<u8> = cps.iter().flat_map(|c| c.to_le_bytes()).collect();
    PyBytes::new(py, &bytes)
        .call_method1(pyo3::intern!(py, "decode"), ("utf-32-le", "surrogatepass"))
}

/// `list(_mdiff(fromlines, tolines, context, linejunk, charjunk))`, or `None`.
/// `context` is `None` or an int `>= 0`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _mdiff<'py>(
    py: Python<'py>,
    fromlines: &Bound<'py, PyAny>,
    tolines: &Bound<'py, PyAny>,
    context: Option<usize>,
    linejunk_kind: u8,
    linejunk: &Bound<'py, PyAny>,
    charjunk_space_tab: bool,
    windowed: bool,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let (Some(al), Some(bl)) = (Lines::read(fromlines)?, Lines::read(tolines)?) else {
        return Ok(None);
    };
    let c = compare_full(
        py,
        &al,
        &bl,
        linejunk_kind,
        linejunk,
        charjunk_space_tab,
        windowed,
    )?;
    let rows = {
        let (ac, bc) = (slices(&c.a_buf, &c.a_ends), slices(&c.b_buf, &c.b_ends));
        mdiff::mdiff(&c.ops, &ac, &bc, context)
    };
    let blank = ("", "\n").into_pyobject(py)?;
    let cell = |cell: &Cell| -> PyResult<Bound<'py, PyAny>> {
        Ok(match cell {
            Cell::Nothing => py.None().into_bound(py),
            Cell::Blank => blank.clone().into_any(),
            Cell::Line(num, text) => {
                let text = match text {
                    Text::Line(Side::A, i) => al.get(*i),
                    Text::Line(Side::B, j) => bl.get(*j),
                    Text::Marked(cps) => str_from_code_points(py, cps)?,
                };
                (*num, text).into_pyobject(py)?.into_any()
            }
        })
    };
    let out = PyList::empty(py);
    for row in &rows {
        let flag = match row.flag {
            None => py.None().into_bound(py),
            Some(f) => PyBool::new(py, f).to_owned().into_any(),
        };
        out.append((cell(&row.from)?, cell(&row.to)?, flag))?;
    }
    Ok(Some(out))
}

/// An int argument that runs natively: an exact `int` within +-2**40.
fn small_int(obj: &Bound<'_, PyAny>) -> Option<i64> {
    if !obj.is_exact_instance_of::<PyInt>() && !obj.is_exact_instance_of::<PyBool>() {
        return None;
    }
    obj.extract::<i64>()
        .ok()
        .filter(|n| n.unsigned_abs() < 1 << 40)
}

fn exact_str<'py>(obj: &Bound<'py, PyAny>) -> Option<Bound<'py, PyString>> {
    obj.cast_exact::<PyString>().ok().cloned()
}

/// The lines of `unified_diff(...)` (`context=False`) or `context_diff(...)`
/// (`context=True`) after `_check_types`, made as they are read, or `None`. `theme` is 3.15's
/// `(header, hunk, context, removed, added, reset)` strings for `color=True`.
#[pyfunction]
#[pyo3(signature = (a, b, fromfile, tofile, fromfiledate, tofiledate, n, lineterm, context, theme=None))]
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn _line_diff<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    fromfile: &Bound<'py, PyAny>,
    tofile: &Bound<'py, PyAny>,
    fromfiledate: &Bound<'py, PyAny>,
    tofiledate: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
    lineterm: &Bound<'py, PyAny>,
    context: bool,
    theme: Option<[String; 6]>,
) -> PyResult<Option<FormatLines>> {
    let (Some(fromfile), Some(tofile), Some(fromdate), Some(todate), Some(lineterm)) = (
        exact_str(fromfile),
        exact_str(tofile),
        exact_str(fromfiledate),
        exact_str(tofiledate),
        exact_str(lineterm),
    ) else {
        return Ok(None);
    };
    let Some(n) = small_int(n) else {
        return Ok(None);
    };
    let (Some(al), Some(bl)) = (Lines::read(a)?, Lines::read(b)?) else {
        return Ok(None);
    };
    let (ai, bi, first) = line_ids(&al, &bl);
    let (la, lb) = (al.len(), bl.len());
    let work = || {
        let index = BIndex::new(bi, first.len(), &[], true);
        let blocks = with_work(|w| index.matching_blocks_with(&ai, w));
        let codes = opcodes(&blocks);
        let groups = grouped_opcodes(&codes, n);
        if context {
            format::context(&groups, la, lb)
        } else {
            format::unified(&groups, la, lb)
        }
    };
    let plan = interrupt::run(py, la + lb > 1024, work)?;
    let theme = theme.unwrap_or_default();
    let [t_header, t_hunk, t_context, t_removed, t_added, t_reset] = theme;
    let reset = (!t_reset.is_empty()).then(|| PyString::new(py, &t_reset));
    let header = |mark: &str, name: &Bound<'py, PyString>, date: &Bound<'py, PyString>| {
        let mut s = PyString::new(py, &format!("{t_header}{mark} ")).add(name)?;
        if date.len()? > 0 {
            s = s.add("\t")?.add(date)?;
        }
        s = s.add(&lineterm)?;
        match &reset {
            Some(r) => s.add(r),
            None => Ok(s),
        }
    };
    let (m1, m2) = if context {
        ("***", "---")
    } else {
        ("---", "+++")
    };
    let headers = [
        header(m1, &fromfile, &fromdate)?.unbind(),
        header(m2, &tofile, &todate)?.unbind(),
    ];
    let prefixes = if context {
        ["  ", "- ", "+ ", "! "].map(|p| PyString::new(py, p).unbind())
    } else {
        [
            format!("{t_context} "),
            format!("{t_removed}-"),
            format!("{t_added}+"),
            String::new(),
        ]
        .map(|p| PyString::new(py, &p).unbind())
    };
    Ok(Some(FormatLines {
        plan,
        next: 0,
        second_header: false,
        headers,
        prefixes,
        hunk: t_hunk,
        context,
        lineterm: lineterm.unbind(),
        reset: reset.map(Bound::unbind),
        a: al.objs.unbind(),
        b: bl.objs.unbind(),
    }))
}

/// The iterator [`_line_diff`] returns: the diff's lines, made as they are read.
#[pyclass(module = "pyroid.difflib", name = "_FormatLines")]
#[derive(Debug)]
pub struct FormatLines {
    plan: Vec<Out>,
    next: usize,
    /// The first file header is out; the second is next.
    second_header: bool,
    headers: [Py<PyAny>; 2],
    /// Context, removed, added and changed lines' prefixes.
    prefixes: [Py<PyString>; 4],
    /// 3.15's hunk colour (or nothing).
    hunk: String,
    context: bool,
    lineterm: Py<PyString>,
    /// 3.15's colour reset after each line, if colours are on.
    reset: Option<Py<PyString>>,
    a: Py<PyTuple>,
    b: Py<PyTuple>,
}

#[pymethods]
impl FormatLines {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        let py = slf.py();
        let this = &mut *slf;
        let Some(item) = this.plan.get(this.next) else {
            return Ok(None);
        };
        let line = match item {
            Out::FileHeaders => {
                let h = this.headers[usize::from(this.second_header)]
                    .bind(py)
                    .clone();
                if this.second_header {
                    this.next += 1;
                }
                this.second_header = !this.second_header;
                return Ok(Some(h));
            }
            Out::Control(text) => {
                let s = if this.context {
                    PyString::new(py, text)
                } else {
                    PyString::new(py, &format!("{}{text}", this.hunk))
                };
                s.add(this.lineterm.bind(py))?
            }
            Out::Line(prefix, side, i) => {
                let p = this.prefixes[match prefix {
                    Prefix::Context => 0,
                    Prefix::Removed => 1,
                    Prefix::Added => 2,
                    Prefix::Changed => 3,
                }]
                .bind(py);
                let lines = match side {
                    Side::A => this.a.bind(py),
                    Side::B => this.b.bind(py),
                };
                p.add(lines.get_item(*i)?)?
            }
        };
        let line = match &this.reset {
            Some(r) => line.add(r.bind(py))?,
            None => line,
        };
        this.next += 1;
        Ok(Some(line))
    }
}

/// `obj` if it is a `str`, or a subclass that keeps str's indexing and comparisons
/// (`get_close_matches` reads its candidates and sorts them).
fn str_like<'py>(
    obj: &Bound<'py, PyAny>,
    types: &mut Types,
) -> PyResult<Option<Bound<'py, PyString>>> {
    if let Ok(s) = obj.cast_exact::<PyString>() {
        return Ok(Some(s.clone()));
    }
    Ok(match types.verdict(obj)? {
        Verdict::Plain(Base::Str) => Some(obj.cast::<PyString>()?.clone()),
        _ => None,
    })
}

/// One `get_close_matches` candidate's text.
enum Cand {
    Utf8(PyBackedStr),
    Cps(Vec<u32>),
}

impl Cand {
    fn code_points(&self) -> Vec<u32> {
        match self {
            Self::Utf8(s) => s.chars().map(u32::from).collect(),
            Self::Cps(c) => c.clone(),
        }
    }

    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            // UTF-8 byte order is code point order.
            (Self::Utf8(x), Self::Utf8(y)) => x.as_bytes().cmp(y.as_bytes()),
            _ => self.code_points().cmp(&other.code_points()),
        }
    }
}

/// Candidates in the first part of a long list (Ctrl-C is checked between parts):
/// enough for every thread of a large pool.
const FIRST_PART: usize = 1 << 16;

/// `get_close_matches(word, possibilities, n, cutoff)` after its argument
/// checks, or `None`. `possibilities` is a list or tuple.
#[pyfunction]
#[pyo3(signature = (word, possibilities, n, cutoff, threads=None))]
pub fn _get_close_matches<'py>(
    py: Python<'py>,
    word: &Bound<'py, PyAny>,
    possibilities: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
    cutoff: &Bound<'py, PyAny>,
    threads: Option<Threads>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let mut types = Types::default();
    let Some(word) = str_like(word, &mut types)? else {
        return Ok(None);
    };
    if !(n.is_exact_instance_of::<PyInt>() || n.is_exact_instance_of::<PyBool>()) {
        return Ok(None);
    }
    let n: usize = n.extract().unwrap_or(usize::MAX);
    let cutoff: f64 = if cutoff.is_exact_instance_of::<PyFloat>()
        || cutoff.is_exact_instance_of::<PyInt>()
        || cutoff.is_exact_instance_of::<PyBool>()
    {
        cutoff.extract()?
    } else {
        return Ok(None);
    };
    let objs = if let Ok(list) = possibilities.cast_exact::<PyList>() {
        list.to_tuple()
    } else if let Ok(tuple) = possibilities.cast_exact::<PyTuple>() {
        tuple.clone()
    } else {
        return Ok(None);
    };
    let mut cands = Vec::with_capacity(objs.len());
    for item in objs.iter() {
        // No extra reference per candidate: with abi3 each is a call into the interpreter.
        if !item.is_exact_instance_of::<PyString>()
            && !matches!(types.verdict(&item)?, Verdict::Plain(Base::Str))
        {
            return Ok(None);
        }
        let s = item.cast_into::<PyString>()?;
        let cand = match PyBackedStr::try_from(s) {
            Ok(b) => Cand::Utf8(b),
            // Lone surrogates: not UTF-8.
            Err(_) => Cand::Cps(code_points(
                objs.get_item(cands.len())?.cast::<PyString>()?,
            )?),
        };
        cands.push(cand);
    }
    let matcher = StrMatcher::new(&code_points(&word)?, |_| false, true);
    let t = input::threads(threads);
    let found = interrupt::in_parts(
        py,
        cands.len(),
        FIRST_PART,
        2 * par::MIN_DOCS_PER_TASK,
        |range| {
            let part = &cands[range];
            interrupt::run(py, true, || {
                par::map(part.len(), t, |i| {
                    with_work(|w| match &part[i] {
                        Cand::Utf8(s) => {
                            let la = if s.is_ascii() {
                                s.len()
                            } else {
                                s.chars().count()
                            };
                            matcher.close_match_with(la, s.chars().map(u32::from), cutoff, w)
                        }
                        Cand::Cps(cps) => {
                            matcher.close_match_with(cps.len(), cps.iter().copied(), cutoff, w)
                        }
                    })
                })
            })?
            .map_err(|e| BatchError::from(e).into_pyerr(py))
        },
    )?;
    let scored: Vec<(f64, usize)> = found
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| r.map(|r| (r, i)))
        .collect();
    let picked = nlargest(&scored, n, |x, y| cands[x].cmp(&cands[y]));
    let out = PyList::empty(py);
    for i in picked {
        out.append(objs.get_item(i)?)?;
    }
    Ok(Some(out))
}

/// Code point lines (tab-expanded `HtmlDiff` input): `compare`'s inputs.
fn compare_code_points(
    py: Python<'_>,
    a: &[Vec<u32>],
    b: &[Vec<u32>],
    linejunk_kind: u8,
    linejunk: &Bound<'_, PyAny>,
    charjunk_space_tab: bool,
    windowed: bool,
) -> PyResult<Vec<DiffOp>> {
    let key = |cps: &[u32]| {
        let mut k = Vec::with_capacity(cps.len());
        for &c in cps {
            push_wtf8(c, &mut k);
        }
        k
    };
    let mut keys: Interner<Vec<u8>> = Interner::default();
    let mut first = Vec::new();
    let bi: Vec<u32> = b
        .iter()
        .enumerate()
        .map(|(j, line)| {
            let (id, new) = keys.intern(key(line));
            if new {
                first.push(j);
            }
            id
        })
        .collect();
    let ai: Vec<u32> = a.iter().map(|line| keys.get(&key(line))).collect();
    let junk: Vec<bool> = match linejunk_kind {
        1 => first.iter().map(|&j| is_line_junk(&b[j])).collect(),
        2 if linejunk.is_truthy()? => first
            .iter()
            .map(|&j| {
                linejunk
                    .call1((str_from_code_points(py, &b[j])?,))?
                    .is_truthy()
            })
            .collect::<PyResult<_>>()?,
        _ => Vec::new(),
    };
    let ac: Vec<&[u32]> = a.iter().map(Vec::as_slice).collect();
    let bc: Vec<&[u32]> = b.iter().map(Vec::as_slice).collect();
    let lines = differ::Lines {
        a: &ai,
        b: &bi,
        b_keys: first.len(),
        b_junk: &junk,
        a_chars: &ac,
        b_chars: &bc,
    };
    let cj = if charjunk_space_tab {
        CharJunk::SpaceTab
    } else {
        CharJunk::Nothing
    };
    let version = if windowed {
        DifferVersion::Windowed
    } else {
        DifferVersion::Classic
    };
    interrupt::run(py, true, || differ::compare(&lines, cj, version))
}

/// `IS_LINE_JUNK(line)`: `line.strip() in '#'`.
fn is_line_junk(cps: &[u32]) -> bool {
    let ws = |c: u32| char::from_u32(c).is_some_and(is_py_whitespace);
    let start = cps.iter().take_while(|&&c| ws(c)).count();
    let end = cps.len() - cps[start..].iter().rev().take_while(|&&c| ws(c)).count();
    let inner = &cps[start..end];
    inner.is_empty() || inner == [u32::from(b'#')]
}

/// `HtmlDiff.make_table`'s rows before the prefixes are chosen.
#[pyclass(frozen, module = "pyroid.difflib", name = "_HtmlRows")]
#[derive(Debug)]
pub struct HtmlRows {
    rows: Vec<html::HRow>,
}

#[pymethods]
impl HtmlRows {
    /// `data_rows` for these prefixes (`self._prefix`), with `make_table`'s
    /// final replacements made.
    fn render<'py>(
        &self,
        py: Python<'py>,
        from_prefix: &str,
        to_prefix: &str,
        context: bool,
        numlines: usize,
        changed_class: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prefixes = [from_prefix, to_prefix];
        if html::has_surrogates(&self.rows) {
            let mut out: Vec<u32> = Vec::new();
            html::render(
                &mut out,
                &self.rows,
                prefixes,
                context,
                numlines,
                changed_class,
            );
            return str_from_code_points(py, &out);
        }
        let mut out = String::new();
        py.detach(|| {
            html::render(
                &mut out,
                &self.rows,
                prefixes,
                context,
                numlines,
                changed_class,
            );
        });
        Ok(PyString::new(py, &out).into_any())
    }
}

/// `make_table`'s work up to the prefixes: tab expansion, `_mdiff` (with
/// `context_lines`, `None` for full), and wrapping at `wrapcolumn` (0: none).
/// `None` for input that does not run natively.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn _html_prepare(
    py: Python<'_>,
    fromlines: &Bound<'_, PyAny>,
    tolines: &Bound<'_, PyAny>,
    tabsize: i64,
    wrapcolumn: usize,
    context_lines: Option<usize>,
    linejunk_kind: u8,
    linejunk: &Bound<'_, PyAny>,
    charjunk_space_tab: bool,
    windowed: bool,
) -> PyResult<Option<HtmlRows>> {
    let (Some(al), Some(bl)) = (Lines::read(fromlines)?, Lines::read(tolines)?) else {
        return Ok(None);
    };
    let expand = |l: &Lines<'_>| -> Vec<Vec<u32>> {
        l.text
            .iter()
            .map(|t| {
                let mut cps = Vec::new();
                t.push_code_points(&mut cps);
                html::expand_tabs(&cps, tabsize)
            })
            .collect()
    };
    let (a, b) = (expand(&al), expand(&bl));
    let ops = compare_code_points(
        py,
        &a,
        &b,
        linejunk_kind,
        linejunk,
        charjunk_space_tab,
        windowed,
    )?;
    let rows = py.detach(|| {
        let ac: Vec<&[u32]> = a.iter().map(Vec::as_slice).collect();
        let bc: Vec<&[u32]> = b.iter().map(Vec::as_slice).collect();
        let rows = mdiff::mdiff(&ops, &ac, &bc, context_lines);
        let text = |side: Side, i: usize| match side {
            Side::A => a[i].clone(),
            Side::B => b[i].clone(),
        };
        html::rows(&rows, &text, wrapcolumn)
    });
    Ok(Some(HtmlRows { rows }))
}
