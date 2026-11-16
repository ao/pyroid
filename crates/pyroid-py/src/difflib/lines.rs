//! The line-based functions: `Differ.compare`/`ndiff`, `_mdiff`,
//! `unified_diff`, `context_diff`, and `get_close_matches`.
//!
//! Each takes exact `list`/`tuple` inputs of exact `str` (read in one step,
//! as another thread may change a list) and returns `None` for anything
//! else; the Python side then runs CPython's code.

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyBool, PyBytes, PyFloat, PyInt, PyList, PyString, PyTuple};
use pyroid_core::difflib::differ::{self, CharJunk, DiffOp, DifferVersion, Side};
use pyroid_core::difflib::format::{self, Out, Prefix};
use pyroid_core::difflib::intern::Interner;
use pyroid_core::difflib::matcher::{BIndex, grouped_opcodes, opcodes};
use pyroid_core::difflib::mdiff::{self, Cell, Text};
use pyroid_core::difflib::{StrMatcher, nlargest};
use pyroid_core::par;
use pyroid_core::text::is_py_whitespace;

use super::seq::code_points;
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

/// An exact `list`/`tuple` of exact `str`.
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
            // IS_LINE_JUNK: `line.strip() in '#'`.
            let mut cps = Vec::new();
            Ok(first
                .iter()
                .map(|&j| {
                    cps.clear();
                    b.text[j].push_code_points(&mut cps);
                    let ws = |c: u32| char::from_u32(c).is_some_and(is_py_whitespace);
                    let start = cps.iter().take_while(|&&c| ws(c)).count();
                    let end = cps.len() - cps[start..].iter().rev().take_while(|&&c| ws(c)).count();
                    let inner = &cps[start..end];
                    inner.is_empty() || inner == [u32::from(b'#')]
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

/// `Differ(linejunk, charjunk).compare(a, b)` as a list of lines, or `None`.
/// `linejunk_kind`: 0 `None`, 1 `IS_LINE_JUNK`, 2 another callable;
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
) -> PyResult<Option<Bound<'py, PyList>>> {
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
    let prefixes = [
        PyString::new(py, "- "),
        PyString::new(py, "+ "),
        PyString::new(py, "  "),
    ];
    let out = PyList::empty(py);
    for op in &ops {
        match op {
            DiffOp::Line(tag, side, i) => {
                let p = &prefixes[match tag {
                    b'-' => 0,
                    b'+' => 1,
                    _ => 2,
                }];
                let line = match side {
                    Side::A => al.get(*i),
                    Side::B => bl.get(*i),
                };
                out.append(p.add(line)?)?;
            }
            DiffOp::Tags(t) => out.append(format!("? {t}\n"))?,
        }
    }
    Ok(Some(out))
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
        if work > 1 << 14 {
            py.detach(|| differ::compare(&lines, cj, version))
        } else {
            differ::compare(&lines, cj, version)
        }
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

/// `list(unified_diff(...))` (`context=False`) or `list(context_diff(...))`
/// (`context=True`) after `_check_types`, or `None`. `theme` is 3.15's
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
) -> PyResult<Option<Bound<'py, PyList>>> {
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
        let codes = opcodes(&index.matching_blocks(&ai));
        let groups = grouped_opcodes(&codes, n);
        if context {
            format::context(&groups, la, lb)
        } else {
            format::unified(&groups, la, lb)
        }
    };
    let plan = if la + lb > 1024 {
        py.detach(work)
    } else {
        work()
    };
    let theme = theme.unwrap_or_default();
    let [t_header, t_hunk, t_context, t_removed, t_added, t_reset] = &theme;
    let reset = (!t_reset.is_empty()).then(|| PyString::new(py, t_reset));
    let finish = |s: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        match &reset {
            Some(r) => s.add(r),
            None => Ok(s),
        }
    };
    let header = |mark: &str, name: &Bound<'py, PyString>, date: &Bound<'py, PyString>| {
        let mut s = PyString::new(py, &format!("{t_header}{mark} ")).add(name)?;
        if date.len()? > 0 {
            s = s.add("\t")?.add(date)?;
        }
        finish(s.add(&lineterm)?)
    };
    let prefixes = if context {
        ["  ", "- ", "+ ", "! "].map(|p| PyString::new(py, p))
    } else {
        [
            format!("{t_context} "),
            format!("{t_removed}-"),
            format!("{t_added}+"),
            String::new(),
        ]
        .map(|p| PyString::new(py, &p))
    };
    let out = PyList::empty(py);
    for line in plan {
        match line {
            Out::FileHeaders => {
                let (m1, m2) = if context {
                    ("***", "---")
                } else {
                    ("---", "+++")
                };
                out.append(header(m1, &fromfile, &fromdate)?)?;
                out.append(header(m2, &tofile, &todate)?)?;
            }
            Out::Control(text) => {
                let s = if context {
                    PyString::new(py, &text)
                } else {
                    PyString::new(py, &format!("{t_hunk}{text}"))
                };
                out.append(finish(s.add(&lineterm)?)?)?;
            }
            Out::Line(prefix, side, i) => {
                let p = &prefixes[match prefix {
                    Prefix::Context => 0,
                    Prefix::Removed => 1,
                    Prefix::Added => 2,
                    Prefix::Changed => 3,
                }];
                let line = match side {
                    Side::A => al.get(i),
                    Side::B => bl.get(i),
                };
                out.append(finish(p.add(line)?)?)?;
            }
        }
    }
    Ok(Some(out))
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

/// Candidates per chunk between checks for signals (Ctrl-C).
const CHUNK: usize = 1 << 16;

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
    let Ok(word) = word.cast_exact::<PyString>() else {
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
        let Ok(s) = item.cast_exact::<PyString>() else {
            return Ok(None);
        };
        cands.push(match PyBackedStr::try_from(s.clone()) {
            Ok(b) => Cand::Utf8(b),
            Err(_) => Cand::Cps(code_points(s)?),
        });
    }
    let matcher = StrMatcher::new(&code_points(word)?, |_| false, true);
    let t = input::threads(threads);
    let mut scored: Vec<(f64, usize)> = Vec::new();
    for (c, chunk) in cands.chunks(CHUNK).enumerate() {
        let found = py
            .detach(|| {
                par::map(chunk.len(), t, |i| {
                    let cps = match &chunk[i] {
                        Cand::Utf8(s) => s.chars().map(u32::from).collect::<Vec<_>>(),
                        Cand::Cps(cps) => cps.clone(),
                    };
                    matcher.close_match(&cps, cutoff)
                })
            })
            .map_err(|e| BatchError::from(e).into_pyerr(py))?;
        scored.extend(
            found
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| r.map(|r| (r, c * CHUNK + i))),
        );
        py.check_signals()?;
    }
    let picked = nlargest(&scored, n, |x, y| cands[x].cmp(&cands[y]));
    let out = PyList::empty(py);
    for i in picked {
        out.append(objs.get_item(i)?)?;
    }
    Ok(Some(out))
}
