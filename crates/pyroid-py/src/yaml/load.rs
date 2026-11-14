// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! Loading: `SafeConstructor` over the core's node graphs.
//!
//! PyYAML constructs collections with generators: `construct_object` returns an empty list,
//! dict or set at once and queues the generator that fills it, and `construct_document`
//! runs the queue in order until it is empty (so containers fill breadth first, and that
//! order decides which error a broken document raises). The queue below does the same.
//! Merge keys (`flatten_mapping`) mutate the graph, as PyYAML mutates its nodes.
//!
//! Scalars take a fast path for the common forms (decimal ints, plain floats, booleans,
//! timestamps); anything else goes to the Python helpers that `pyroid.yaml` registers, which
//! are PyYAML's own constructor code, so unusual explicitly tagged values (`!!int " 0x1F "`,
//! `!!float ١٫٥`, `!!bool maybe`) give PyYAML's results and exceptions.

use std::collections::VecDeque;
use std::sync::Mutex;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PySet, PyString, PyTuple};
use pyroid_core::yaml::composer::{Composer, ComposerOptions};
use pyroid_core::yaml::error::{ErrorKind, Msg, YamlError};
use pyroid_core::yaml::nodes::{Graph, NodeId, NodeKind};
use pyroid_core::yaml::reader::{self, find_non_printable, has_reserved, printable_error};
use pyroid_core::yaml::resolver::{self as res, Tz};
use pyroid_core::yaml::scanner::ScannerOptions;

use super::{Classes, Source, classes, reader_error, restore_surrogates, to_pyerr};
use crate::batch::SINGLE_DOC_DETACH_BYTES;

/// A construction failure: a core error (needs marks) or a Python exception.
enum CErr {
    Yaml(YamlError),
    Py(PyErr),
    /// A `!!python/...` tag under a full loader: pyroid.yaml hands the call to PyYAML.
    Unsupported,
}

impl From<PyErr> for CErr {
    fn from(e: PyErr) -> Self {
        Self::Py(e)
    }
}

impl From<YamlError> for CErr {
    fn from(e: YamlError) -> Self {
        Self::Yaml(e)
    }
}

type R<T> = Result<T, CErr>;

fn ctor_error(
    context: Option<&'static str>,
    context_mark: Option<pyroid_core::yaml::error::Mark>,
    problem: Msg,
    mark: pyroid_core::yaml::error::Mark,
) -> CErr {
    CErr::Yaml(YamlError::marked(
        ErrorKind::Constructor,
        context.map(Msg::text),
        context_mark,
        problem,
        Some(mark),
    ))
}

/// Python's recursion limit stands in for PyYAML's recursive `construct_scalar` and
/// `flatten_mapping` (only reachable through long `=`/merge chains).
const MAX_CHAIN: usize = 900;

enum Fill<'py> {
    Seq(NodeId, Bound<'py, PyList>),
    Map(NodeId, Bound<'py, PyDict>),
    Set(NodeId, Bound<'py, PySet>),
    /// `!!omap` (true) or `!!pairs` (false).
    Pairs(NodeId, Bound<'py, PyList>, bool),
}

struct Ctor<'g, 'py> {
    py: Python<'py>,
    c: &'py Classes,
    g: &'g mut Graph,
    objs: Vec<Option<Bound<'py, PyAny>>>,
    queue: VecDeque<Fill<'py>>,
    /// The scanner decoded `\uD800`-style escapes (scalar strings need mapping back).
    surrogates: bool,
    /// A `FullLoader`/`Loader`/`UnsafeLoader` call: `!!python/...` tags are PyYAML's.
    full: bool,
}

impl<'g, 'py> Ctor<'g, 'py> {
    fn new(
        py: Python<'py>,
        c: &'py Classes,
        g: &'g mut Graph,
        surrogates: bool,
        full: bool,
    ) -> Self {
        let n = g.nodes.len();
        Self {
            py,
            c,
            g,
            objs: vec![None; n],
            queue: VecDeque::new(),
            surrogates,
            full,
        }
    }

    /// `construct_document(node)`.
    fn document(mut self) -> R<Bound<'py, PyAny>> {
        let root = self.g.root;
        let data = self.object(root)?;
        while let Some(fill) = self.queue.pop_front() {
            match fill {
                Fill::Seq(id, list) => self.fill_seq(id, &list)?,
                Fill::Map(id, dict) => self.fill_map(id, &dict)?,
                Fill::Set(id, set) => self.fill_set(id, &set)?,
                Fill::Pairs(id, list, omap) => self.fill_pairs(id, &list, omap)?,
            }
        }
        Ok(data)
    }

    fn string(&self, s: &str) -> PyResult<Bound<'py, PyAny>> {
        if self.surrogates {
            return restore_surrogates(self.py, s);
        }
        Ok(PyString::new(self.py, s).into_any())
    }

    /// `construct_object(node)`.
    fn object(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        if let Some(o) = &self.objs[id] {
            return Ok(o.clone());
        }
        let py = self.py;
        let tag: &str = &self.g.nodes[id].tag;
        let obj = match tag {
            res::STR => {
                let s = self.scalar(id)?;
                self.string(self.value(s))?
            }
            res::MAP => {
                let d = PyDict::new(py);
                self.queue.push_back(Fill::Map(id, d.clone()));
                d.into_any()
            }
            res::SEQ => {
                let l = PyList::empty(py);
                self.queue.push_back(Fill::Seq(id, l.clone()));
                l.into_any()
            }
            res::INT => self.int(id)?,
            res::NULL => {
                self.scalar(id)?;
                py.None().into_bound(py)
            }
            res::BOOL => self.bool(id)?,
            res::FLOAT => self.float(id)?,
            res::TIMESTAMP => self.timestamp(id)?,
            res::SET => {
                let s = PySet::empty(py)?;
                self.queue.push_back(Fill::Set(id, s.clone()));
                s.into_any()
            }
            res::OMAP | res::PAIRS => {
                let l = PyList::empty(py);
                self.queue
                    .push_back(Fill::Pairs(id, l.clone(), tag == res::OMAP));
                l.into_any()
            }
            res::BINARY => self.binary(id)?,
            _ if self.full && tag.starts_with("tag:yaml.org,2002:python/") => {
                return Err(CErr::Unsupported);
            }
            _ => {
                let mark = self.g.nodes[id].start_mark;
                return Err(ctor_error(
                    None,
                    None,
                    Msg::str("could not determine a constructor for the tag %r", tag),
                    mark,
                ));
            }
        };
        self.objs[id] = Some(obj.clone());
        Ok(obj)
    }

    fn value(&self, scalar: NodeId) -> &str {
        match &self.g.nodes[scalar].kind {
            NodeKind::Scalar { value, .. } => value,
            _ => "",
        }
    }

    /// `SafeConstructor.construct_scalar(node)`: the scalar node whose value it returns
    /// (a mapping gives its `=` key's value).
    fn scalar(&self, mut id: NodeId) -> R<NodeId> {
        for _ in 0..MAX_CHAIN {
            let node = &self.g.nodes[id];
            match &node.kind {
                NodeKind::Scalar { .. } => return Ok(id),
                NodeKind::Mapping { pairs, .. } => {
                    match pairs
                        .iter()
                        .find(|(k, _)| self.g.nodes[*k].tag == res::VALUE)
                    {
                        Some(&(_, v)) => id = v,
                        None => {
                            return Err(ctor_error(
                                None,
                                None,
                                Msg::str("expected a scalar node, but found %s", "mapping"),
                                node.start_mark,
                            ));
                        }
                    }
                }
                NodeKind::Sequence { .. } => {
                    return Err(ctor_error(
                        None,
                        None,
                        Msg::str("expected a scalar node, but found %s", "sequence"),
                        node.start_mark,
                    ));
                }
            }
        }
        Err(CErr::Yaml(YamlError::Recursion))
    }

    /// Call a registered Python helper (PyYAML's own code) on the scalar value.
    fn slow(&self, helper: &str, value: &str) -> R<Bound<'py, PyAny>> {
        let f = self.c.helpers.bind(self.py).get_item(helper)?;
        Ok(f.call1((self.string(value)?,))?)
    }

    fn int(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        let value = self.value(s);
        if let Some(n) = fast_int(value) {
            return Ok(pyo3::types::PyInt::new(self.py, n).into_any());
        }
        self.slow("int", value)
    }

    fn float(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        let value = self.value(s);
        match fast_float(value) {
            Some(FastFloat::Value(f)) => Ok(PyFloat::new(self.py, f).into_any()),
            Some(FastFloat::Nan) => Ok(self.c.nan.bind(self.py).clone()),
            None => self.slow("float", value),
        }
    }

    fn bool(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        let value = self.value(s);
        if value.is_ascii() {
            let b = match value.to_ascii_lowercase().as_str() {
                "yes" | "true" | "on" => Some(true),
                "no" | "false" | "off" => Some(false),
                _ => None,
            };
            if let Some(b) = b {
                return Ok(PyBool::new(self.py, b).to_owned().into_any());
            }
        }
        self.slow("bool", value)
    }

    fn binary(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        let value = self.value(s);
        let out = self.slow("binary", value)?;
        // The helper returns the bytes, or the message of PyYAML's ConstructorError.
        if let Ok(msg) = out.cast::<PyString>() {
            let mark = self.g.nodes[id].start_mark;
            return Err(ctor_error(None, None, Msg::str("%s", msg.to_str()?), mark));
        }
        Ok(out)
    }

    fn timestamp(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        if matches!(self.g.nodes[id].kind, NodeKind::Mapping { .. }) {
            // `self.timestamp_regexp.match(node.value)` with the mapping's list of pairs.
            return Err(CErr::Py(PyTypeError::new_err(
                "expected string or bytes-like object, got 'list'",
            )));
        }
        let value = self.value(s);
        let py = self.py;
        let Some(t) = res::parse_timestamp(value) else {
            return self.slow("timestamp", value);
        };
        let num = |s: &str| -> i64 { s.bytes().fold(0i64, |a, d| a * 10 + i64::from(d - b'0')) };
        let (year, month, day) = (num(t.year), num(t.month), num(t.day));
        let Some((h, mi, se, fraction)) = t.time else {
            return Ok(self.c.date.bind(py).call1((year, month, day))?);
        };
        let mut micro = 0i64;
        if let Some(f) = fraction.filter(|f| !f.is_empty()) {
            let mut digits: String = f.chars().take(6).collect();
            while digits.len() < 6 {
                digits.push('0');
            }
            micro = num(&digits);
        }
        let tzinfo = match t.tz {
            None => py.None().into_bound(py),
            Some(Tz::Utc) => self.c.utc.bind(py).clone(),
            Some(Tz::Offset {
                minus,
                hour,
                minute,
            }) => {
                let secs = num(hour) * 3600 + minute.map_or(0, num) * 60;
                let delta = self
                    .c
                    .timedelta
                    .bind(py)
                    .call1((0, if minus { -secs } else { secs }))?;
                self.c.timezone.bind(py).call1((delta,))?
            }
        };
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("tzinfo", tzinfo)?;
        Ok(self.c.datetime.bind(py).call(
            (year, month, day, num(h), num(mi), num(se), micro),
            Some(&kwargs),
        )?)
    }

    fn is_unhashable(obj: &Bound<'py, PyAny>) -> bool {
        obj.is_instance_of::<PyList>()
            || obj.is_instance_of::<PyDict>()
            || obj.is_instance_of::<PySet>()
    }

    /// `construct_sequence` into the shell list.
    fn fill_seq(&mut self, id: NodeId, list: &Bound<'py, PyList>) -> R<()> {
        let node = &self.g.nodes[id];
        let NodeKind::Sequence { items, .. } = &node.kind else {
            return Err(ctor_error(
                None,
                None,
                Msg::str("expected a sequence node, but found %s", node.kind.id()),
                node.start_mark,
            ));
        };
        let items = items.clone();
        for child in items {
            let o = self.object(child)?;
            list.append(o)?;
        }
        Ok(())
    }

    /// `SafeConstructor.construct_mapping` up to the key check; calls `f(key, value)`.
    fn mapping(
        &mut self,
        id: NodeId,
        mut f: impl FnMut(&Bound<'py, PyAny>, Bound<'py, PyAny>) -> PyResult<()>,
    ) -> R<()> {
        if matches!(self.g.nodes[id].kind, NodeKind::Mapping { .. }) {
            self.flatten(id, 0)?;
        }
        let node = &self.g.nodes[id];
        let NodeKind::Mapping { pairs, .. } = &node.kind else {
            return Err(ctor_error(
                None,
                None,
                Msg::str("expected a mapping node, but found %s", node.kind.id()),
                node.start_mark,
            ));
        };
        let start = node.start_mark;
        let pairs = pairs.clone();
        for (k, v) in pairs {
            let key = self.object(k)?;
            if Self::is_unhashable(&key) {
                return Err(ctor_error(
                    Some("while constructing a mapping"),
                    Some(start),
                    Msg::text("found unhashable key"),
                    self.g.nodes[k].start_mark,
                ));
            }
            let value = self.object(v)?;
            f(&key, value)?;
        }
        Ok(())
    }

    fn fill_map(&mut self, id: NodeId, dict: &Bound<'py, PyDict>) -> R<()> {
        self.mapping(id, |k, v| dict.set_item(k, v))
    }

    fn fill_set(&mut self, id: NodeId, set: &Bound<'py, PySet>) -> R<()> {
        // `data.update(construct_mapping(node))`: the keys, after the values are built.
        let dict = PyDict::new(self.py);
        self.mapping(id, |k, v| dict.set_item(k, v))?;
        for k in dict.keys() {
            set.add(k)?;
        }
        Ok(())
    }

    fn fill_pairs(&mut self, id: NodeId, list: &Bound<'py, PyList>, omap: bool) -> R<()> {
        let context = if omap {
            "while constructing an ordered map"
        } else {
            "while constructing pairs"
        };
        let node = &self.g.nodes[id];
        let start = node.start_mark;
        let NodeKind::Sequence { items, .. } = &node.kind else {
            return Err(ctor_error(
                Some(context),
                Some(start),
                Msg::str("expected a sequence, but found %s", node.kind.id()),
                start,
            ));
        };
        let items = items.clone();
        for sub in items {
            let sn = &self.g.nodes[sub];
            let NodeKind::Mapping { pairs, .. } = &sn.kind else {
                return Err(ctor_error(
                    Some(context),
                    Some(start),
                    Msg::str("expected a mapping of length 1, but found %s", sn.kind.id()),
                    sn.start_mark,
                ));
            };
            if pairs.len() != 1 {
                return Err(ctor_error(
                    Some(context),
                    Some(start),
                    Msg::int(
                        "expected a single mapping item, but found %d items",
                        i64::try_from(pairs.len()).unwrap_or(i64::MAX),
                    ),
                    sn.start_mark,
                ));
            }
            let (k, v) = pairs[0];
            let key = self.object(k)?;
            let value = self.object(v)?;
            list.append(PyTuple::new(self.py, [key, value])?)?;
        }
        Ok(())
    }

    /// `flatten_mapping(node)`.
    fn flatten(&mut self, id: NodeId, depth: usize) -> R<()> {
        if depth > MAX_CHAIN {
            return Err(CErr::Yaml(YamlError::Recursion));
        }
        let mut merge: Vec<(NodeId, NodeId)> = Vec::new();
        let mut index = 0;
        loop {
            let NodeKind::Mapping { pairs, .. } = &self.g.nodes[id].kind else {
                return Ok(());
            };
            if index >= pairs.len() {
                break;
            }
            let (k, v) = pairs[index];
            let ktag: &str = &self.g.nodes[k].tag;
            if ktag == res::MERGE {
                if let NodeKind::Mapping { pairs, .. } = &mut self.g.nodes[id].kind {
                    pairs.remove(index);
                }
                let start = self.g.nodes[id].start_mark;
                match &self.g.nodes[v].kind {
                    NodeKind::Mapping { .. } => {
                        self.flatten(v, depth + 1)?;
                        if let NodeKind::Mapping { pairs, .. } = &self.g.nodes[v].kind {
                            merge.extend_from_slice(pairs);
                        }
                    }
                    NodeKind::Sequence { items, .. } => {
                        let items = items.clone();
                        let mut submerge = Vec::new();
                        for sub in items {
                            if !matches!(self.g.nodes[sub].kind, NodeKind::Mapping { .. }) {
                                let sn = &self.g.nodes[sub];
                                return Err(ctor_error(
                                    Some("while constructing a mapping"),
                                    Some(start),
                                    Msg::str(
                                        "expected a mapping for merging, but found %s",
                                        sn.kind.id(),
                                    ),
                                    sn.start_mark,
                                ));
                            }
                            self.flatten(sub, depth + 1)?;
                            if let NodeKind::Mapping { pairs, .. } = &self.g.nodes[sub].kind {
                                submerge.push(pairs.clone());
                            }
                        }
                        for value in submerge.into_iter().rev() {
                            merge.extend(value);
                        }
                    }
                    NodeKind::Scalar { .. } => {
                        let vn = &self.g.nodes[v];
                        return Err(ctor_error(
                            Some("while constructing a mapping"),
                            Some(start),
                            Msg::str(
                                "expected a mapping or list of mappings for merging, but found %s",
                                vn.kind.id(),
                            ),
                            vn.start_mark,
                        ));
                    }
                }
            } else if ktag == res::VALUE {
                self.g.nodes[k].tag = std::borrow::Cow::Borrowed(res::STR);
                index += 1;
            } else {
                index += 1;
            }
        }
        if !merge.is_empty() {
            if let NodeKind::Mapping { pairs, .. } = &mut self.g.nodes[id].kind {
                merge.append(pairs);
                *pairs = merge;
            }
        }
        Ok(())
    }
}

/// `construct_yaml_int` for the common forms; `None`: let PyYAML's code decide.
fn fast_int(value: &str) -> Option<i64> {
    let b = value.as_bytes();
    if b.is_empty() || !b.is_ascii() {
        return None;
    }
    let mut digits: Vec<u8> = b.iter().copied().filter(|&c| c != b'_').collect();
    let negative = digits.first() == Some(&b'-');
    if matches!(digits.first(), Some(b'+' | b'-')) {
        digits.remove(0);
    }
    if digits.is_empty() {
        return None;
    }
    let parse = |d: &[u8], radix: u32| -> Option<i64> {
        if d.is_empty() || d.len() > 15 {
            return None;
        }
        let s = std::str::from_utf8(d).ok()?;
        if !s.bytes().all(|c| (c as char).is_digit(radix)) {
            return None;
        }
        i64::from_str_radix(s, radix).ok()
    };
    let n = if digits == b"0" {
        0
    } else if let Some(rest) = digits.strip_prefix(b"0b") {
        parse(rest, 2)?
    } else if let Some(rest) = digits.strip_prefix(b"0x") {
        parse(rest, 16)?
    } else if digits[0] == b'0' {
        parse(&digits, 8)?
    } else if digits.contains(&b':') {
        let mut total: i64 = 0;
        let mut base: i64 = 1;
        for part in digits.rsplit(|&c| c == b':') {
            let d = parse(part, 10)?;
            total = total.checked_add(d.checked_mul(base)?)?;
            base = base.checked_mul(60)?;
        }
        total
    } else {
        parse(&digits, 10)?
    };
    Some(if negative { -n } else { n })
}

enum FastFloat {
    Value(f64),
    Nan,
}

/// `construct_yaml_float` for the common forms.
fn fast_float(value: &str) -> Option<FastFloat> {
    if value.is_empty() || !value.is_ascii() {
        return None;
    }
    let mut v: String = value
        .chars()
        .filter(|&c| c != '_')
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let negative = v.starts_with('-');
    if v.starts_with(['+', '-']) {
        v.remove(0);
    }
    let sign = if negative { -1.0 } else { 1.0 };
    if v == ".inf" {
        return Some(FastFloat::Value(sign * f64::INFINITY));
    }
    if v == ".nan" {
        return Some(FastFloat::Nan);
    }
    if v.contains(':') {
        return None;
    }
    // float() and Rust agree on [digits][.digits][e[sign]digits]; anything else goes to Python.
    let ok_start = v
        .as_bytes()
        .first()
        .is_some_and(|c| c.is_ascii_digit() || *c == b'.');
    if !ok_start
        || !v
            .bytes()
            .all(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'+' | b'-'))
    {
        return None;
    }
    let f: f64 = v.parse().ok()?;
    Some(FastFloat::Value(sign * f))
}

/// The text of a `str`/`bytes` stream, checked as PyYAML's reader checks it. `Err`: the
/// `ReaderError` to raise.
fn checked_text(
    py: Python<'_>,
    stream: &Bound<'_, PyAny>,
    name: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let text = if let Ok(s) = stream.cast::<PyString>() {
        // A str with lone surrogates raises UnicodeEncodeError here; pyroid.yaml catches it
        // and raises the ReaderError itself.
        s.to_str()?.to_owned()
    } else if let Ok(b) = stream.cast::<pyo3::types::PyBytes>() {
        let bytes = b.as_bytes();
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_owned(),
            Err(e) => return Err(reader_error(py, name, &reader::utf8_error(bytes, e))),
        }
    } else {
        return Err(PyTypeError::new_err("pyroid.yaml: expected str or bytes"));
    };
    if let Some((pos, ch)) = find_non_printable(&text) {
        return Err(reader_error(py, name, &printable_error(pos, u32::from(ch))));
    }
    Ok(text)
}

/// The options PyYAML's environment implies.
#[derive(Clone, Copy)]
struct Opts {
    scanner: ScannerOptions,
    composer: ComposerOptions,
}

fn opts(max_depth: usize, int_digits: usize) -> Opts {
    Opts {
        scanner: ScannerOptions {
            int_max_str_digits: int_digits,
        },
        composer: ComposerOptions { max_depth },
    }
}

fn construct<'py>(
    py: Python<'py>,
    graph: &mut Graph,
    surrogates: bool,
    full: bool,
) -> R<Bound<'py, PyAny>> {
    let c = classes(py)?;
    Ctor::new(py, c, graph, surrogates, full).document()
}

fn finish<'py>(
    py: Python<'py>,
    r: R<Bound<'py, PyAny>>,
    name: &Bound<'py, PyAny>,
    buffer: Option<&str>,
    pending: Option<&Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match r {
        Ok(v) => Ok(v),
        Err(CErr::Py(e)) => Err(e),
        Err(CErr::Yaml(e)) => Err(to_pyerr(py, e, &Source::new(name, buffer), pending)),
        Err(CErr::Unsupported) => Ok(py.NotImplemented().into_bound(py)),
    }
}

/// `yaml.load(stream, SafeLoader)` for a `str` or UTF-8 `bytes` stream (or a file object's
/// text with `limit`/`pending`, see `reader`). Returns `NotImplemented` for the inputs
/// pyroid.yaml hands to PyYAML.
#[pyfunction]
#[pyo3(signature = (stream, name, marks, limit=None, pending=None, max_depth=usize::MAX, int_digits=4300, full=false))]
#[allow(clippy::too_many_arguments)]
fn load<'py>(
    py: Python<'py>,
    stream: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    marks: bool,
    limit: Option<usize>,
    pending: Option<Py<PyAny>>,
    max_depth: usize,
    int_digits: usize,
    full: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let text = checked_text(py, stream, name)?;
    if has_reserved(&text) {
        return Ok(py.NotImplemented().into_bound(py));
    }
    let o = opts(max_depth, int_digits);
    let big = text.len() >= SINGLE_DOC_DETACH_BYTES;
    let mut composer = pyroid_core::yaml::loader(text, limit, o.scanner, o.composer);
    let node = if big {
        py.detach(|| composer.get_single_node())
    } else {
        composer.get_single_node()
    };
    let surrogates = composer.parser.scanner.surrogates;
    let buffer = marks.then(|| composer.parser.scanner.reader.text());
    let r = match node {
        Err(e) => Err(CErr::Yaml(e)),
        Ok(None) => Ok(py.None().into_bound(py)),
        Ok(Some(mut graph)) => construct(py, &mut graph, surrogates, full),
    };
    finish(py, r, name, buffer, pending.as_ref())
}

/// `yaml.load_all(stream, SafeLoader)`'s loader: `check_data()` and `get_data()`.
#[pyclass(frozen, module = "pyroid.yaml", name = "_LoadAll")]
struct LoadAll {
    state: Mutex<Composer>,
    name: Py<PyAny>,
    marks: bool,
    pending: Option<Py<PyAny>>,
}

#[pymethods]
impl LoadAll {
    /// `NotImplemented` (from `start`) for inputs pyroid.yaml hands to PyYAML.
    #[staticmethod]
    #[pyo3(signature = (stream, name, marks, limit=None, pending=None, max_depth=usize::MAX, int_digits=4300))]
    #[allow(clippy::too_many_arguments)]
    fn start<'py>(
        py: Python<'py>,
        stream: &Bound<'py, PyAny>,
        name: &Bound<'py, PyAny>,
        marks: bool,
        limit: Option<usize>,
        pending: Option<Py<PyAny>>,
        max_depth: usize,
        int_digits: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let text = checked_text(py, stream, name)?;
        if has_reserved(&text) {
            return Ok(py.NotImplemented().into_bound(py));
        }
        let o = opts(max_depth, int_digits);
        let composer = pyroid_core::yaml::loader(text, limit, o.scanner, o.composer);
        let me = Self {
            state: Mutex::new(composer),
            name: name.clone().unbind(),
            marks,
            pending,
        };
        Ok(Bound::new(py, me)?.into_any())
    }

    /// `check_data()`: is there another document.
    fn check_data(&self, py: Python<'_>) -> PyResult<bool> {
        let mut c = self.lock(py);
        let r = c.check_node();
        match r {
            Ok(b) => Ok(b),
            Err(e) => {
                let buffer = self.marks.then(|| c.parser.scanner.reader.text());
                Err(to_pyerr(
                    py,
                    e,
                    &Source::new(self.name.bind(py), buffer),
                    self.pending.as_ref(),
                ))
            }
        }
    }

    /// `get_data()`: construct the next document.
    fn get_data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut c = self.lock(py);
        let node = c
            .check_node()
            .and_then(|more| if more { c.get_node() } else { Ok(None) });
        let surrogates = c.parser.scanner.surrogates;
        let buffer = self.marks.then(|| c.parser.scanner.reader.text());
        let r = match node {
            Err(e) => Err(CErr::Yaml(e)),
            Ok(None) => Ok(py.None().into_bound(py)),
            Ok(Some(mut graph)) => construct(py, &mut graph, surrogates, false),
        };
        finish(py, r, self.name.bind(py), buffer, self.pending.as_ref())
    }
}

impl LoadAll {
    fn lock(&self, py: Python<'_>) -> std::sync::MutexGuard<'_, Composer> {
        use pyo3::sync::MutexExt;
        self.state
            .lock_py_attached(py)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(_scan, m)?)?;
    m.add_function(wrap_pyfunction!(_compose, m)?)?;
    m.add_class::<LoadAll>()?;
    Ok(())
}

/// The number of tokens PyYAML's scanner makes from `text` (a benchmark hook: the scanner
/// alone, as `len(list(yaml.scan(text)))`).
#[pyfunction]
fn _scan(py: Python<'_>, text: &str) -> usize {
    let text = text.to_owned();
    py.detach(|| {
        let mut s = pyroid_core::yaml::scanner::Scanner::new(
            pyroid_core::yaml::reader::Reader::new(text, None),
            ScannerOptions::default(),
        );
        let mut n = 0;
        while let Ok(Some(_)) = s.get_token() {
            n += 1;
        }
        n
    })
}

/// The number of nodes PyYAML's composer makes from `text`, all documents (a benchmark hook:
/// everything but constructing Python objects).
#[pyfunction]
fn _compose(py: Python<'_>, text: &str) -> usize {
    let text = text.to_owned();
    py.detach(|| {
        pyroid_core::yaml::compose_all(&text)
            .map_or(0, |docs| docs.iter().map(|g| g.nodes.len()).sum())
    })
}
