// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! Dumping: `SafeRepresenter` (and `Representer` for plain data) over Python objects, then
//! the core's serializer and emitter.
//!
//! Objects other than `None`, `str`, `bytes`, `bool`, `int`, `float`, `list`, `tuple`,
//! `dict`, `set`, `datetime.date` and `datetime.datetime` (exact types) raise PyYAML's
//! `RepresenterError` under `SafeDumper`; under `Dumper` they make the call go to PyYAML.

use std::borrow::Cow;

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple};
use pyroid_core::yaml::emitter::{Emitter, EmitterOptions};
use pyroid_core::yaml::error::{Mark, YamlError};
use pyroid_core::yaml::nodes::{Graph, Node, NodeId, NodeKind};
use pyroid_core::yaml::represent::float_text;
use pyroid_core::yaml::resolver as res;
use pyroid_core::yaml::serializer::{self, SerializerOptions};
use rustc_hash::FxHashMap;

use super::{Classes, classes, format_msg};

enum DErr {
    Py(PyErr),
    /// A `RepresenterError("cannot represent an object", data)`.
    Undefined(Py<PyAny>),
    /// Not handled natively (the reason, for the FallbackWarning).
    Unsupported(&'static str),
    Yaml(YamlError),
}

impl From<PyErr> for DErr {
    fn from(e: PyErr) -> Self {
        Self::Py(e)
    }
}

impl From<YamlError> for DErr {
    fn from(e: YamlError) -> Self {
        Self::Yaml(e)
    }
}

type R<T> = Result<T, DErr>;

#[derive(Clone, Copy)]
struct ReprOptions {
    default_style: Option<char>,
    default_flow_style: Option<bool>,
    sort_keys: bool,
    /// `Dumper` (`Representer`): tuples are `!!python/tuple`; other types go to PyYAML.
    full: bool,
    max_depth: usize,
}

struct Repr<'py> {
    py: Python<'py>,
    c: &'py Classes,
    graph: Graph,
    represented: FxHashMap<usize, NodeId>,
    keeper: Vec<Bound<'py, PyAny>>,
    o: ReprOptions,
}

impl<'py> Repr<'py> {
    fn push(&mut self, alias_key: Option<usize>, tag: &'static str, kind: NodeKind) -> NodeId {
        let id = self.graph.push(Node {
            tag: Cow::Borrowed(tag),
            kind,
            start_mark: Mark::default(),
            end_mark: Mark::default(),
        });
        if let Some(k) = alias_key {
            self.represented.insert(k, id);
        }
        id
    }

    /// `represent_scalar(tag, value, style=None)`.
    fn scalar(
        &mut self,
        key: Option<usize>,
        tag: &'static str,
        value: String,
        style: Option<char>,
    ) -> NodeId {
        let style = style.or(self.o.default_style);
        self.push(key, tag, NodeKind::Scalar { value, style })
    }

    /// `ignore_aliases(data)`.
    fn ignore_aliases(data: &Bound<'py, PyAny>) -> bool {
        data.is_none()
            || data.cast::<PyTuple>().is_ok_and(PyTupleMethods::is_empty)
            || data.is_instance_of::<PyString>()
            || data.is_instance_of::<PyBytes>()
            || data.is_instance_of::<PyInt>()
            || data.is_instance_of::<PyFloat>()
    }

    /// `represent_data(data)`.
    fn data(&mut self, data: &Bound<'py, PyAny>, depth: usize) -> R<NodeId> {
        if depth > self.o.max_depth {
            return Err(DErr::Yaml(YamlError::Recursion));
        }
        let key = if Self::ignore_aliases(data) {
            None
        } else {
            Some(data.as_ptr().addr())
        };
        if let Some(k) = key {
            if let Some(&n) = self.represented.get(&k) {
                return Ok(n);
            }
            self.keeper.push(data.clone());
        }
        let py = self.py;
        let ty = data.get_type();
        if data.is_none() {
            return Ok(self.scalar(key, res::NULL, "null".to_owned(), None));
        }
        if ty.is(py.get_type::<PyString>()) {
            let s = data.cast::<PyString>().map_err(PyErr::from)?;
            let Ok(text) = s.to_str() else {
                return Err(DErr::Unsupported("a str with lone surrogates"));
            };
            return Ok(self.scalar(key, res::STR, text.to_owned(), None));
        }
        if ty.is(py.get_type::<PyBool>()) {
            let v = if data.is_truthy()? { "true" } else { "false" };
            return Ok(self.scalar(key, res::BOOL, v.to_owned(), None));
        }
        if ty.is(py.get_type::<PyInt>()) {
            let text = match data.extract::<i64>() {
                Ok(n) => n.to_string(),
                Err(_) => data.str()?.to_str()?.to_owned(),
            };
            return Ok(self.scalar(key, res::INT, text, None));
        }
        if ty.is(py.get_type::<PyFloat>()) {
            let x = data.cast::<PyFloat>().map_err(PyErr::from)?.value();
            return Ok(self.scalar(key, res::FLOAT, float_text(x), None));
        }
        if ty.is(py.get_type::<PyDict>()) {
            let d = data.cast::<PyDict>().map_err(PyErr::from)?;
            let items = PyList::new(py, d.items())?;
            return self.mapping(key, res::MAP, &items, depth);
        }
        if ty.is(py.get_type::<PyList>()) {
            let l = data.cast::<PyList>().map_err(PyErr::from)?.clone();
            return self.sequence(key, res::SEQ, l.iter(), depth);
        }
        if ty.is(py.get_type::<PyTuple>()) {
            let t = data.cast::<PyTuple>().map_err(PyErr::from)?.clone();
            let tag = if self.o.full {
                "tag:yaml.org,2002:python/tuple"
            } else {
                res::SEQ
            };
            return self.sequence(key, tag, t.iter(), depth);
        }
        if ty.is(py.get_type::<PyBytes>()) {
            let text: String = self
                .c
                .encodebytes
                .bind(py)
                .call1((data,))?
                .call_method1("decode", ("ascii",))?
                .extract()?;
            return Ok(self.scalar(key, res::BINARY, text, Some('|')));
        }
        if ty.is(py.get_type::<PySet>()) {
            // value = {key: None for key in data}
            let d = PyDict::new(py);
            for k in data.try_iter()? {
                d.set_item(k?, py.None())?;
            }
            let items = PyList::new(py, d.items())?;
            return self.mapping(key, res::SET, &items, depth);
        }
        if ty.is(self.c.datetime.bind(py)) {
            let text: String = data.call_method1("isoformat", (" ",))?.extract()?;
            return Ok(self.scalar(key, res::TIMESTAMP, text, None));
        }
        if ty.is(self.c.date.bind(py)) {
            let text: String = data.call_method0("isoformat")?.extract()?;
            return Ok(self.scalar(key, res::TIMESTAMP, text, None));
        }
        if self.o.full {
            return Err(DErr::Unsupported("objects other than plain data"));
        }
        Err(DErr::Undefined(data.clone().unbind()))
    }

    fn is_plain_scalar(&self, id: NodeId) -> bool {
        matches!(
            &self.graph.nodes[id].kind,
            NodeKind::Scalar { style: None, .. }
        )
    }

    /// `represent_sequence(tag, sequence)`.
    fn sequence(
        &mut self,
        key: Option<usize>,
        tag: &'static str,
        items: impl Iterator<Item = Bound<'py, PyAny>>,
        depth: usize,
    ) -> R<NodeId> {
        let id = self.push(
            key,
            tag,
            NodeKind::Sequence {
                items: Vec::new(),
                flow_style: None,
            },
        );
        let mut best_style = true;
        let mut out = Vec::new();
        for item in items {
            let n = self.data(&item, depth + 1)?;
            if !self.is_plain_scalar(n) {
                best_style = false;
            }
            out.push(n);
        }
        let flow = self.o.default_flow_style.unwrap_or(best_style);
        self.graph.nodes[id].kind = NodeKind::Sequence {
            items: out,
            flow_style: Some(flow),
        };
        Ok(id)
    }

    /// `represent_mapping(tag, mapping)` over `list(mapping.items())`.
    fn mapping(
        &mut self,
        key: Option<usize>,
        tag: &'static str,
        items: &Bound<'py, PyList>,
        depth: usize,
    ) -> R<NodeId> {
        let id = self.push(
            key,
            tag,
            NodeKind::Mapping {
                pairs: Vec::new(),
                flow_style: None,
            },
        );
        let mut items = items.clone();
        if self.o.sort_keys {
            let sorted = PyList::new(self.py, items.iter())?;
            match sorted.sort() {
                Ok(()) => items = sorted,
                Err(e) if e.is_instance_of::<PyTypeError>(self.py) => {}
                Err(e) => return Err(e.into()),
            }
        }
        let mut best_style = true;
        let mut pairs = Vec::with_capacity(items.len());
        for item in items.iter() {
            let pair = item.cast::<PyTuple>().map_err(PyErr::from)?;
            let k = self.data(&pair.get_item(0)?, depth + 1)?;
            let v = self.data(&pair.get_item(1)?, depth + 1)?;
            if !self.is_plain_scalar(k) || !self.is_plain_scalar(v) {
                best_style = false;
            }
            pairs.push((k, v));
        }
        let flow = self.o.default_flow_style.unwrap_or(best_style);
        self.graph.nodes[id].kind = NodeKind::Mapping {
            pairs,
            flow_style: Some(flow),
        };
        Ok(id)
    }
}

fn style_arg(style: Option<&str>) -> PyResult<Option<char>> {
    Ok(match style {
        None | Some("") => None,
        Some("\"") => Some('"'),
        Some("'") => Some('\''),
        Some("|") => Some('|'),
        Some(">") => Some('>'),
        Some(other) => {
            return Err(PyTypeError::new_err(format!(
                "pyroid.yaml: unsupported style {other:?}"
            )));
        }
    })
}

fn line_break_arg(lb: &str) -> &'static str {
    match lb {
        "\r" => "\r",
        "\r\n" => "\r\n",
        _ => "\n",
    }
}

/// `yaml.dump_all` up to the stream: returns `(text, writes, error, consumed)`.
///
/// - `text`: everything PyYAML would have written (also when it stopped with an error);
/// - `writes`: the end offsets (code points) of PyYAML's `stream.write` calls, if asked;
/// - `error`: `None`, the exception to raise after writing `text`, or `NotImplemented` (go
///   to PyYAML: `consumed` holds the documents already taken from the iterator).
#[pyfunction]
#[pyo3(signature = (documents, *, full, default_style, default_flow_style, canonical, best_indent, best_width, allow_unicode, line_break, encoding, explicit_start, explicit_end, version, tags, sort_keys, record_writes, max_depth))]
#[allow(
    clippy::too_many_arguments,
    clippy::fn_params_excessive_bools,
    clippy::too_many_lines
)]
fn dump<'py>(
    py: Python<'py>,
    documents: &Bound<'py, PyAny>,
    full: bool,
    default_style: Option<&str>,
    default_flow_style: Option<bool>,
    canonical: bool,
    best_indent: usize,
    best_width: u64,
    allow_unicode: bool,
    line_break: &str,
    encoding: Option<String>,
    explicit_start: bool,
    explicit_end: bool,
    version: Option<(u64, u64)>,
    tags: Option<Vec<(String, String)>>,
    sort_keys: bool,
    record_writes: bool,
    max_depth: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let c = classes(py)?;
    let o = ReprOptions {
        default_style: style_arg(default_style)?,
        default_flow_style,
        sort_keys,
        full,
        max_depth,
    };
    let mut emitter = Emitter::new(EmitterOptions {
        canonical,
        allow_unicode,
        best_indent,
        best_width,
        line_break: line_break_arg(line_break),
        record_writes,
    });
    let sopts = SerializerOptions {
        explicit_start,
        explicit_end,
        version,
        tags,
    };
    let consumed = PyList::empty(py);
    let result = (|| -> R<()> {
        serializer::open(&mut emitter, encoding)?;
        for data in documents.try_iter()? {
            let data = data?;
            consumed.append(&data)?;
            let mut r = Repr {
                py,
                c,
                graph: Graph::default(),
                represented: FxHashMap::default(),
                keeper: Vec::new(),
                o,
            };
            r.graph.root = r.data(&data, 0)?;
            serializer::serialize(&mut emitter, &r.graph, &sopts)?;
        }
        serializer::close(&mut emitter)?;
        Ok(())
    })();
    let error: Bound<'py, PyAny> = match result {
        Ok(()) => py.None().into_bound(py),
        Err(DErr::Unsupported(reason)) => {
            return PyTuple::new(
                py,
                [
                    py.None().into_bound(py),
                    py.None().into_bound(py),
                    py.NotImplemented().into_bound(py),
                    consumed.into_any(),
                    PyString::new(py, reason).into_any(),
                ],
            );
        }
        Err(DErr::Py(e)) => e.into_value(py).into_bound(py).into_any(),
        Err(DErr::Undefined(obj)) => c
            .representer_error
            .bind(py)
            .call1(("cannot represent an object", obj))?,
        Err(DErr::Yaml(YamlError::Emitter(m))) => {
            c.emitter_error.bind(py).call1((format_msg(py, &m)?,))?
        }
        Err(DErr::Yaml(YamlError::Recursion)) => {
            pyo3::exceptions::PyRecursionError::new_err("maximum recursion depth exceeded")
                .into_value(py)
                .into_bound(py)
                .into_any()
        }
        Err(DErr::Yaml(YamlError::Type(m))) => PyTypeError::new_err(m)
            .into_value(py)
            .into_bound(py)
            .into_any(),
        Err(DErr::Yaml(e)) => pyo3::exceptions::PyValueError::new_err(e.to_string())
            .into_value(py)
            .into_bound(py)
            .into_any(),
    };
    let writes = if record_writes {
        // Byte offsets into code-point offsets (Python slices the str).
        let text = emitter.output();
        let mut out = Vec::with_capacity(emitter.writes().len());
        let mut chars = 0usize;
        let mut last = 0usize;
        for &end in emitter.writes() {
            chars += text[last..end].chars().count();
            last = end;
            out.push(chars);
        }
        PyList::new(py, out)?.into_any()
    } else {
        py.None().into_bound(py)
    };
    let text = PyString::new(py, emitter.output()).into_any();
    PyTuple::new(
        py,
        [
            text,
            writes,
            error,
            consumed.into_any(),
            py.None().into_bound(py),
        ],
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dump, m)?)?;
    Ok(())
}
