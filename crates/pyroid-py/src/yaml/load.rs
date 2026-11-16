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
//!
//! File objects are streams: `pyroid.yaml._Stream` reads and decodes them as PyYAML's reader
//! does, and the core's scanner asks it for more ([`PyStream`]) whenever it looks past the text
//! read so far, so documents come out of `load_all` as the stream arrives.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyList, PySet, PyString, PyTuple};
use pyroid_core::yaml::composer::{Composer, ComposerOptions};
use pyroid_core::yaml::error::{ErrorKind, Msg, YamlError};
use pyroid_core::yaml::nodes::{Graph, NodeId, NodeKind};
use pyroid_core::yaml::parser::Parser;
use pyroid_core::yaml::reader::{self, Reader, find_non_printable, is_printable, printable_error};
use pyroid_core::yaml::resolver::{self as res, Tz};
use pyroid_core::yaml::scanner::{Escapes, Scanner, ScannerOptions};

use super::{Classes, Source, classes, reader_error, to_pyerr};
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

/// How long a chain of merged mappings (`flatten_mapping`) or of `=` keys (`construct_scalar`)
/// may be before `RecursionError`. Both recurse in PyYAML, one frame a level, and run out of
/// Python frames near where its composer (two frames a level) would at twice the nesting.
fn chain_limit(max_depth: usize) -> usize {
    max_depth.saturating_mul(2).saturating_add(6).min(1 << 22)
}

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
    /// The lone surrogates of the stream's double-quoted scalars.
    escapes: &'g Escapes,
    /// A `FullLoader`/`Loader`/`UnsafeLoader` call: `!!python/...` tags are PyYAML's.
    full: bool,
    /// [`chain_limit`].
    chain: usize,
}

impl<'g, 'py> Ctor<'g, 'py> {
    fn new(py: Python<'py>, c: &'py Classes, g: &'g mut Graph, env: &'g Env<'g>) -> Self {
        let n = g.nodes.len();
        Self {
            py,
            c,
            g,
            objs: vec![None; n],
            queue: VecDeque::new(),
            escapes: env.escapes,
            full: env.full,
            chain: env.chain,
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

    /// The `str` of scalar node `id`'s value, lone surrogates from escapes included.
    fn string(&self, id: NodeId) -> PyResult<Bound<'py, PyAny>> {
        let node = &self.g.nodes[id];
        let value = self.value(id);
        match self.escapes.of(node.end_mark.index) {
            None => Ok(PyString::new(self.py, value).into_any()),
            Some(offsets) => with_surrogates(self.py, value, offsets),
        }
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
                self.string(s)?
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
        for _ in 0..self.chain {
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
    fn slow(&self, helper: &str, scalar: NodeId) -> R<Bound<'py, PyAny>> {
        let f = self.c.helpers.bind(self.py).get_item(helper)?;
        Ok(f.call1((self.string(scalar)?,))?)
    }

    fn int(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        let value = self.value(s);
        if let Some(n) = fast_int(value) {
            return Ok(pyo3::types::PyInt::new(self.py, n).into_any());
        }
        self.slow("int", s)
    }

    fn float(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        let value = self.value(s);
        match fast_float(value) {
            Some(FastFloat::Value(f)) => Ok(PyFloat::new(self.py, f).into_any()),
            Some(FastFloat::Nan) => Ok(self.c.nan.bind(self.py).clone()),
            None => self.slow("float", s),
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
        self.slow("bool", s)
    }

    fn binary(&mut self, id: NodeId) -> R<Bound<'py, PyAny>> {
        let s = self.scalar(id)?;
        let out = self.slow("binary", s)?;
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
            return self.slow("timestamp", s);
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
            self.flatten(id)?;
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
        // `data.update(construct_mapping(node))`: the keys, after the values are built. The
        // update itself, not one add per key: it sizes the table first, and the table's size
        // decides the iteration order.
        let dict = PyDict::new(self.py);
        self.mapping(id, |k, v| dict.set_item(k, v))?;
        set.call_method1(pyo3::intern!(self.py, "update"), (dict,))?;
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

    /// `flatten_mapping(node)`. PyYAML recurses into the mappings it merges; this keeps them
    /// in a stack (depth costs heap, not native stack) and works in the same order, raising
    /// `RecursionError` more than [`chain_limit`] levels down.
    #[allow(clippy::too_many_lines)] // flatten_mapping, with its recursion unrolled
    fn flatten(&mut self, root: NodeId) -> R<()> {
        let mut stack = vec![Flatten::new(root)];
        // A merged mapping was just flattened: its parent (the top) resumes.
        let mut returned = false;
        loop {
            let top = stack.last_mut().expect("a mapping is being flattened");
            let id = top.id;
            if std::mem::take(&mut returned) {
                if let Some(v) = top.map.take() {
                    // flatten_mapping(value_node); merge.extend(value_node.value)
                    if let NodeKind::Mapping { pairs, .. } = &self.g.nodes[v].kind {
                        top.merge.extend_from_slice(pairs);
                    }
                } else if let Some(seq) = &mut top.seq {
                    // flatten_mapping(subnode); submerge.append(subnode.value)
                    let sub = seq.items[seq.next];
                    if let NodeKind::Mapping { pairs, .. } = &self.g.nodes[sub].kind {
                        seq.done.push(pairs.clone());
                    }
                    seq.next += 1;
                }
            }
            // `<<: [a, b, ...]`: the next mapping of the sequence, or merge them all.
            if let Some(seq) = &mut top.seq {
                if let Some(&sub) = seq.items.get(seq.next) {
                    if !matches!(self.g.nodes[sub].kind, NodeKind::Mapping { .. }) {
                        let sn = &self.g.nodes[sub];
                        return Err(ctor_error(
                            Some("while constructing a mapping"),
                            Some(self.g.nodes[id].start_mark),
                            Msg::str("expected a mapping for merging, but found %s", sn.kind.id()),
                            sn.start_mark,
                        ));
                    }
                    if stack.len() > self.chain {
                        return Err(CErr::Yaml(YamlError::Recursion));
                    }
                    stack.push(Flatten::new(sub));
                    continue;
                }
                let seq = top.seq.take().expect("checked above");
                for value in seq.done.into_iter().rev() {
                    top.merge.extend(value);
                }
            }
            // while index < len(node.value): ...
            let mut descend = None;
            while let NodeKind::Mapping { pairs, .. } = &self.g.nodes[id].kind {
                let Some(&(k, v)) = pairs.get(top.index) else {
                    break;
                };
                let ktag: &str = &self.g.nodes[k].tag;
                if ktag == res::MERGE {
                    if let NodeKind::Mapping { pairs, .. } = &mut self.g.nodes[id].kind {
                        pairs.remove(top.index);
                    }
                    match &self.g.nodes[v].kind {
                        NodeKind::Mapping { .. } => {
                            top.map = Some(v);
                            descend = Some(v);
                            break;
                        }
                        NodeKind::Sequence { items, .. } => {
                            top.seq = Some(SeqMerge {
                                items: items.clone(),
                                next: 0,
                                done: Vec::new(),
                            });
                            break;
                        }
                        NodeKind::Scalar { .. } => {
                            let vn = &self.g.nodes[v];
                            return Err(ctor_error(
                                Some("while constructing a mapping"),
                                Some(self.g.nodes[id].start_mark),
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
                }
                top.index += 1;
            }
            if let Some(v) = descend {
                if stack.len() > self.chain {
                    return Err(CErr::Yaml(YamlError::Recursion));
                }
                stack.push(Flatten::new(v));
                continue;
            }
            if top.seq.is_some() {
                continue;
            }
            // if merge: node.value = merge + node.value
            let done = stack.pop().expect("a mapping is being flattened");
            if !done.merge.is_empty() {
                let mut merge = done.merge;
                if let NodeKind::Mapping { pairs, .. } = &mut self.g.nodes[done.id].kind {
                    merge.extend_from_slice(pairs);
                    *pairs = merge;
                }
            }
            if stack.is_empty() {
                return Ok(());
            }
            returned = true;
        }
    }
}

/// One `flatten_mapping(node)` call in [`Ctor::flatten`]'s stack.
struct Flatten {
    id: NodeId,
    merge: Vec<(NodeId, NodeId)>,
    index: usize,
    /// A `<<: mapping` being flattened.
    map: Option<NodeId>,
    /// A `<<: [mappings]` being flattened.
    seq: Option<SeqMerge>,
}

impl Flatten {
    fn new(id: NodeId) -> Self {
        Self {
            id,
            merge: Vec::new(),
            index: 0,
            map: None,
            seq: None,
        }
    }
}

/// `submerge`: the mappings of a `<<: [...]` value, the next one to flatten, and the pairs of
/// those flattened so far.
struct SeqMerge {
    items: Vec<NodeId>,
    next: usize,
    done: Vec<Vec<(NodeId, NodeId)>>,
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
    } else if let Ok(b) = stream.cast::<PyBytes>() {
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

/// The first character of `s` that PyYAML's `check_printable` rejects, for a `str` with
/// lone surrogates: `(index, code point)`.
fn non_printable_in(s: &Bound<'_, PyString>) -> PyResult<Option<(usize, u32)>> {
    let units = s.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
    let units = units.cast::<PyBytes>().map_err(PyErr::from)?.as_bytes();
    Ok(units.chunks_exact(4).enumerate().find_map(|(i, u)| {
        let cp = u32::from_le_bytes([u[0], u[1], u[2], u[3]]);
        char::from_u32(cp)
            .is_none_or(|ch| !is_printable(ch))
            .then_some((i, cp))
    }))
}

/// A `str` with lone surrogates: `value` with the stand-ins at `offsets` (see [`Escapes`])
/// turned back into them.
fn with_surrogates<'py>(
    py: Python<'py>,
    value: &str,
    offsets: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let mut units = Vec::with_capacity(value.len() * 4);
    let mut next = offsets.iter().copied().peekable();
    for (at, ch) in value.char_indices() {
        while next.next_if(|&o| o < at).is_some() {}
        let mut cp = u32::from(ch);
        if next.next_if_eq(&at).is_some() && cp >= reader::SURROGATE_BASE {
            cp = Escapes::surrogate(ch);
        }
        units.extend_from_slice(&cp.to_le_bytes());
    }
    PyBytes::new(py, &units).call_method1("decode", ("utf-32-le", "surrogatepass"))
}

/// What a read of the stream raised, kept until the scanner looks that far.
type Pending = Arc<Mutex<Option<Py<PyAny>>>>;

/// A Python file object as a [`reader::Source`]: `stream` is a `pyroid.yaml._Stream`, whose
/// `pull()` makes one of PyYAML's reads and decodes it; the printable check is done here.
#[derive(Debug)]
struct PyStream {
    stream: Py<PyAny>,
    name: Py<PyAny>,
    /// Code points decoded so far (positions of `check_printable` errors).
    chars: usize,
    pending: Pending,
}

impl PyStream {
    fn pull_attached(&mut self, py: Python<'_>) -> PyResult<(String, bool)> {
        let (data, end): (Bound<'_, PyAny>, bool) = self
            .stream
            .bind(py)
            .call_method0(pyo3::intern!(py, "pull"))?
            .extract()?;
        let s = data.cast::<PyString>().map_err(PyErr::from)?;
        let name = self.name.bind(py);
        let Ok(text) = s.to_str() else {
            let (pos, ch) = non_printable_in(s)?.unwrap_or((0, 0xD800));
            return Err(reader_error(
                py,
                name,
                &printable_error(self.chars + pos, ch),
            ));
        };
        if let Some((pos, ch)) = find_non_printable(text) {
            let e = printable_error(self.chars + pos, u32::from(ch));
            return Err(reader_error(py, name, &e));
        }
        self.chars += s.len()?;
        Ok((text.to_owned(), end))
    }
}

impl reader::Source for PyStream {
    // The scanner asks for more text on the thread that runs the load: inside `load`'s
    // `py.detach` (which this re-attaches), or attached already (`_LoadAll`). Never on a
    // worker thread.
    #[allow(clippy::disallowed_methods)]
    fn pull(&mut self) -> Result<(String, bool), ()> {
        Python::attach(|py| {
            self.pull_attached(py).map_err(|e| {
                let mut slot = self
                    .pending
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                *slot = Some(e.into_value(py).into_any());
            })
        })
    }
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

/// A composer over a `str` or `bytes` stream, or over a `pyroid.yaml._Stream` (a file
/// object), with the slot for the stream's read error.
fn open(
    py: Python<'_>,
    stream: &Bound<'_, PyAny>,
    name: &Bound<'_, PyAny>,
    o: Opts,
) -> PyResult<(Composer, Option<Pending>)> {
    if stream.is_instance_of::<PyString>() || stream.is_instance_of::<PyBytes>() {
        let text = checked_text(py, stream, name)?;
        let composer = pyroid_core::yaml::loader(text, None, o.scanner, o.composer);
        return Ok((composer, None));
    }
    let pending = Pending::default();
    let source = PyStream {
        stream: stream.clone().unbind(),
        name: name.clone().unbind(),
        chars: 0,
        pending: Arc::clone(&pending),
    };
    let reader = Reader::stream(Box::new(source));
    let parser = Parser::new(Scanner::new(reader, o.scanner));
    Ok((Composer::new(parser, o.composer), Some(pending)))
}

/// What construction needs from the loader.
struct Env<'a> {
    escapes: &'a Escapes,
    full: bool,
    chain: usize,
}

fn construct<'py>(py: Python<'py>, graph: &mut Graph, env: &Env<'_>) -> R<Bound<'py, PyAny>> {
    let c = classes(py)?;
    Ctor::new(py, c, graph, env).document()
}

/// The error a stream's read raised, if any (kept: it is raised again if asked again).
fn pending_error(py: Python<'_>, pending: Option<&Pending>) -> Option<Py<PyAny>> {
    let slot = pending?
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    slot.as_ref().map(|e| e.clone_ref(py))
}

fn finish<'py>(
    py: Python<'py>,
    r: R<Bound<'py, PyAny>>,
    name: &Bound<'py, PyAny>,
    buffer: Option<&str>,
    pending: Option<&Pending>,
) -> PyResult<Bound<'py, PyAny>> {
    match r {
        Ok(v) => Ok(v),
        Err(CErr::Py(e)) => Err(e),
        Err(CErr::Yaml(e)) => {
            let pending = pending_error(py, pending);
            Err(to_pyerr(
                py,
                e,
                &Source::new(name, buffer),
                pending.as_ref(),
            ))
        }
        Err(CErr::Unsupported) => Ok(py.NotImplemented().into_bound(py)),
    }
}

/// `yaml.load(stream, SafeLoader)` for a `str` or UTF-8 `bytes` stream, or a file object
/// wrapped in `pyroid.yaml._Stream`. Returns `NotImplemented` when a full loader meets a
/// `!!python/` tag (pyroid.yaml hands the call to PyYAML).
#[pyfunction]
#[pyo3(signature = (stream, name, marks, max_depth=usize::MAX, int_digits=4300, full=false))]
fn load<'py>(
    py: Python<'py>,
    stream: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    marks: bool,
    max_depth: usize,
    int_digits: usize,
    full: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let o = opts(max_depth, int_digits);
    let (mut composer, pending) = open(py, stream, name, o)?;
    // A stream is read while it is parsed (PyStream re-attaches for each read).
    let big =
        pending.is_some() || composer.parser.scanner.reader.text().len() >= SINGLE_DOC_DETACH_BYTES;
    let node = if big {
        py.detach(|| composer.get_single_node())
    } else {
        composer.get_single_node()
    };
    let scanner = &composer.parser.scanner;
    let buffer = marks.then(|| scanner.reader.text());
    let r = match node {
        Err(e) => Err(CErr::Yaml(e)),
        Ok(None) => Ok(py.None().into_bound(py)),
        Ok(Some(mut graph)) => {
            let env = Env {
                escapes: &scanner.escapes,
                full,
                chain: chain_limit(max_depth),
            };
            construct(py, &mut graph, &env)
        }
    };
    finish(py, r, name, buffer, pending.as_ref())
}

/// `yaml.load_all(stream, SafeLoader)`'s loader: `check_data()` and `get_data()`.
#[pyclass(frozen, module = "pyroid.yaml", name = "_LoadAll")]
struct LoadAll {
    state: Mutex<Composer>,
    name: Py<PyAny>,
    marks: bool,
    pending: Option<Pending>,
    chain: usize,
}

#[pymethods]
impl LoadAll {
    /// A loader over a `str` or `bytes` stream, or a `pyroid.yaml._Stream`.
    #[staticmethod]
    #[pyo3(signature = (stream, name, marks, max_depth=usize::MAX, int_digits=4300))]
    fn start<'py>(
        py: Python<'py>,
        stream: &Bound<'py, PyAny>,
        name: &Bound<'py, PyAny>,
        marks: bool,
        max_depth: usize,
        int_digits: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (composer, pending) = open(py, stream, name, opts(max_depth, int_digits))?;
        let me = Self {
            state: Mutex::new(composer),
            name: name.clone().unbind(),
            marks,
            pending,
            chain: chain_limit(max_depth),
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
                let pending = pending_error(py, self.pending.as_ref());
                Err(to_pyerr(
                    py,
                    e,
                    &Source::new(self.name.bind(py), buffer),
                    pending.as_ref(),
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
        let scanner = &c.parser.scanner;
        let buffer = self.marks.then(|| scanner.reader.text());
        let mut end = None;
        let r = match node {
            Err(e) => Err(CErr::Yaml(e)),
            Ok(None) => Ok(py.None().into_bound(py)),
            Ok(Some(mut graph)) => {
                end = Some(graph.nodes[graph.root].end_mark.index);
                let env = Env {
                    escapes: &scanner.escapes,
                    full: false,
                    chain: self.chain,
                };
                construct(py, &mut graph, &env)
            }
        };
        let out = finish(py, r, self.name.bind(py), buffer, self.pending.as_ref());
        if let Some(end) = end {
            // A long stream keeps no record of the documents it has given.
            c.parser.scanner.escapes.forget_until(end);
        }
        out
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
