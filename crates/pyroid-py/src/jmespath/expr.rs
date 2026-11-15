// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! The evaluable tree: jmespath's AST with Python values, built from the core parser's
//! nodes or from a dict AST (`ParsedResult(expression, parsed)`, `_Expression.visit(node)`),
//! and turned back into jmespath's dicts for `.parsed`.

use std::sync::Arc;

use pyo3::exceptions::PyRecursionError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyInt, PyList, PyString};
use pyroid_core::jmespath::{Comparator, Node};

use super::functions::Builtin;

/// An expref's expression, with its dict AST (shared by `.parsed` and the `_Expression`
/// objects made for it, as jmespath shares the node).
pub(crate) struct ExprefNode {
    pub expr: Expr,
    pub dict: PyOnceLock<Py<PyAny>>,
}

impl std::fmt::Debug for ExprefNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExprefNode")
            .field("expr", &self.expr)
            .finish_non_exhaustive()
    }
}

impl ExprefNode {
    pub(crate) fn new(expr: Expr) -> Arc<Self> {
        Arc::new(Self {
            expr,
            dict: PyOnceLock::new(),
        })
    }

    pub(crate) fn dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.dict
            .get_or_try_init(py, || {
                to_dict(py, &self.expr).map(|d| d.into_any().unbind())
            })
            .map(|d| d.bind(py).clone())
    }
}

/// One node. Values are jmespath's: the same objects `.parsed` holds.
#[derive(Debug)]
pub(crate) enum Expr {
    /// `None` for a dict AST whose comparator name is not one of jmespath's (a `KeyError`
    /// when visited, as in jmespath); the name object is kept for `.parsed`.
    Comparator(Option<Comparator>, Py<PyAny>, Box<Expr>, Box<Expr>),
    Current,
    Expref(Arc<ExprefNode>),
    Function {
        name: Py<PyAny>,
        builtin: Option<Builtin>,
        args: Vec<Expr>,
    },
    Field(Py<PyAny>),
    FilterProjection(Box<Expr>, Box<Expr>, Box<Expr>),
    Flatten(Box<Expr>),
    Identity,
    /// The index object, and its value when it is an `int` that fits.
    Index(Py<PyAny>, Option<isize>),
    IndexExpression(Vec<Expr>),
    KeyValPair(Py<PyAny>, Box<Expr>),
    Literal(Py<PyAny>),
    /// Children with their `child['value']` keys.
    MultiSelectDict(Vec<(Py<PyAny>, Expr)>),
    MultiSelectList(Vec<Expr>),
    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    Pipe(Vec<Expr>),
    Projection(Box<Expr>, Box<Expr>),
    Subexpression(Vec<Expr>),
    /// `children` as written and `slice(*children)`.
    Slice(Vec<Py<PyAny>>, Py<PyAny>),
    ValueProjection(Box<Expr>, Box<Expr>),
    /// A dict AST node type jmespath has no `visit_` method for.
    Unknown(Py<PyAny>),
}

/// Frames a node needs below its `visit` (an upper bound; decides when depth is checked).
pub(crate) fn frames(e: &Expr) -> usize {
    let max = |v: &[Expr]| v.iter().map(frames).max().unwrap_or(0);
    2 + match e {
        Expr::Comparator(_, _, l, r)
        | Expr::Or(l, r)
        | Expr::And(l, r)
        | Expr::Projection(l, r)
        | Expr::ValueProjection(l, r) => frames(l).max(frames(r)) + 1,
        Expr::FilterProjection(a, b, c) => frames(a).max(frames(b)).max(frames(c)) + 1,
        Expr::Expref(x) => frames(&x.expr) + 6,
        Expr::Function { args, .. } => max(args).max(6) + 2,
        Expr::Flatten(x) | Expr::Not(x) | Expr::KeyValPair(_, x) => frames(x) + 1,
        Expr::IndexExpression(v)
        | Expr::MultiSelectList(v)
        | Expr::Pipe(v)
        | Expr::Subexpression(v) => max(v) + 1,
        Expr::MultiSelectDict(v) => v.iter().map(|(_, c)| frames(c)).max().unwrap_or(0) + 1,
        _ => 1,
    }
}

/// From the core parser's nodes.
#[allow(clippy::many_single_char_names)]
pub(crate) fn from_node(py: Python<'_>, node: Node<Bound<'_, PyAny>>) -> PyResult<Expr> {
    let b = |n: Box<Node<Bound<'_, PyAny>>>| from_node(py, *n).map(Box::new);
    let v = |n: Vec<Node<Bound<'_, PyAny>>>| {
        n.into_iter()
            .map(|c| from_node(py, c))
            .collect::<PyResult<Vec<_>>>()
    };
    Ok(match node {
        Node::Comparator(c, l, r) => Expr::Comparator(Some(c), intern(py, c.name()), b(l)?, b(r)?),
        Node::Current => Expr::Current,
        Node::Expref(x) => Expr::Expref(ExprefNode::new(from_node(py, *x)?)),
        Node::FunctionExpression(name, args) => Expr::Function {
            builtin: builtin_of(&name),
            name: name.unbind(),
            args: v(args)?,
        },
        Node::Field(name) => Expr::Field(name.unbind()),
        Node::FilterProjection(l, r, c) => Expr::FilterProjection(b(l)?, b(r)?, b(c)?),
        Node::Flatten(x) => Expr::Flatten(b(x)?),
        Node::Identity => Expr::Identity,
        Node::Index(i) => index(i),
        Node::IndexExpression(c) => Expr::IndexExpression(v(c)?),
        Node::KeyValPair(k, x) => Expr::KeyValPair(k.unbind(), b(x)?),
        Node::Literal(x) => Expr::Literal(x.unbind()),
        Node::MultiSelectDict(c) => Expr::MultiSelectDict(
            c.into_iter()
                .map(|child| {
                    let key = match &child {
                        Node::KeyValPair(k, _) => k.clone().unbind(),
                        _ => py.None(),
                    };
                    Ok((key, from_node(py, child)?))
                })
                .collect::<PyResult<Vec<_>>>()?,
        ),
        Node::MultiSelectList(c) => Expr::MultiSelectList(v(c)?),
        Node::OrExpression(l, r) => Expr::Or(b(l)?, b(r)?),
        Node::AndExpression(l, r) => Expr::And(b(l)?, b(r)?),
        Node::NotExpression(x) => Expr::Not(b(x)?),
        Node::Pipe(l, r) => Expr::Pipe(vec![from_node(py, *l)?, from_node(py, *r)?]),
        Node::Projection(l, r) => Expr::Projection(b(l)?, b(r)?),
        Node::Subexpression(c) => Expr::Subexpression(v(c)?),
        Node::Slice(first, end, step) => {
            let parts: Vec<Py<PyAny>> = [first, end, step]
                .into_iter()
                .map(|p| p.map_or_else(|| py.None(), Bound::unbind))
                .collect();
            slice(py, parts)?
        }
        Node::ValueProjection(l, r) => Expr::ValueProjection(b(l)?, b(r)?),
    })
}

fn intern(py: Python<'_>, s: &str) -> Py<PyAny> {
    PyString::intern(py, s).into_any().unbind()
}

fn builtin_of(name: &Bound<'_, PyAny>) -> Option<Builtin> {
    let s = name.cast_exact::<PyString>().ok()?;
    Builtin::from_name(s.to_str().ok()?)
}

fn index(i: Bound<'_, PyAny>) -> Expr {
    let fast = if i.is_exact_instance_of::<PyInt>() {
        i.extract::<isize>().ok()
    } else {
        None
    };
    Expr::Index(i.unbind(), fast)
}

fn slice(py: Python<'_>, parts: Vec<Py<PyAny>>) -> PyResult<Expr> {
    let builtins = py.import("builtins")?;
    let args = pyo3::types::PyTuple::new(py, parts.iter().map(|p| p.bind(py)))?;
    let s = builtins.getattr("slice")?.call1(args)?;
    Ok(Expr::Slice(parts, s.unbind()))
}

/// The deepest dict AST accepted (the parser's own cap, in nodes).
const MAX_DICT_DEPTH: usize = pyroid_core::jmespath::parser::MAX_FRAMES / 2;

/// From a dict AST (`{'type': ..., 'children': [...], 'value': ...}`), as jmespath's
/// `TreeInterpreter` reads it. A malformed node raises here (jmespath raises when it visits).
pub(crate) fn from_dict(node: &Bound<'_, PyAny>, depth: usize) -> PyResult<Expr> {
    if depth > MAX_DICT_DEPTH {
        return Err(PyRecursionError::new_err(
            "maximum recursion depth exceeded while reading a jmespath AST",
        ));
    }
    let py = node.py();
    let kind = node.get_item("type")?;
    let children = || node.get_item("children");
    let child = |i: usize| -> PyResult<Expr> { from_dict(&children()?.get_item(i)?, depth + 1) };
    let child_box = |i: usize| child(i).map(Box::new);
    let all = || -> PyResult<Vec<Expr>> {
        children()?
            .try_iter()?
            .map(|c| from_dict(&c?, depth + 1))
            .collect()
    };
    let value = || node.get_item("value").map(Bound::unbind);
    let Ok(name) = kind
        .cast::<PyString>()
        .map(|s| s.to_cow().map(std::borrow::Cow::into_owned))
    else {
        return Ok(Expr::Unknown(kind.unbind()));
    };
    Ok(match name?.as_str() {
        "comparator" => {
            let v = node.get_item("value")?;
            let op = v
                .cast::<PyString>()
                .ok()
                .and_then(|s| s.to_str().ok().and_then(Comparator::from_name));
            Expr::Comparator(op, v.unbind(), child_box(0)?, child_box(1)?)
        }
        "current" => Expr::Current,
        "expref" => Expr::Expref(ExprefNode::new(child(0)?)),
        "function_expression" => {
            let name = node.get_item("value")?;
            Expr::Function {
                builtin: builtin_of(&name),
                name: name.unbind(),
                args: all()?,
            }
        }
        "field" => Expr::Field(value()?),
        "filter_projection" => Expr::FilterProjection(child_box(0)?, child_box(1)?, child_box(2)?),
        "flatten" => Expr::Flatten(child_box(0)?),
        "identity" => Expr::Identity,
        "index" => index(node.get_item("value")?),
        "index_expression" => Expr::IndexExpression(all()?),
        "key_val_pair" => Expr::KeyValPair(value()?, child_box(0)?),
        "literal" => Expr::Literal(value()?),
        "multi_select_dict" => {
            let mut out = Vec::new();
            for c in children()?.try_iter()? {
                let c = c?;
                let key = c.get_item("value")?.unbind();
                out.push((key, from_dict(&c, depth + 1)?));
            }
            Expr::MultiSelectDict(out)
        }
        "multi_select_list" => Expr::MultiSelectList(all()?),
        "or_expression" => Expr::Or(child_box(0)?, child_box(1)?),
        "and_expression" => Expr::And(child_box(0)?, child_box(1)?),
        "not_expression" => Expr::Not(child_box(0)?),
        "pipe" => Expr::Pipe(all()?),
        "projection" => Expr::Projection(child_box(0)?, child_box(1)?),
        "subexpression" => Expr::Subexpression(all()?),
        "slice" => {
            let parts = children()?
                .try_iter()?
                .map(|p| p.map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?;
            slice(py, parts)?
        }
        "value_projection" => Expr::ValueProjection(child_box(0)?, child_box(1)?),
        _ => Expr::Unknown(kind.unbind()),
    })
}

/// jmespath's dict for `e` (`ast.py`'s key order): one arm per node kind.
#[allow(clippy::too_many_lines, clippy::many_single_char_names)]
pub(crate) fn to_dict<'py>(py: Python<'py>, e: &Expr) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let list = |v: &[Expr]| -> PyResult<Bound<'py, PyList>> {
        PyList::new(
            py,
            v.iter()
                .map(|c| to_dict(py, c))
                .collect::<PyResult<Vec<_>>>()?,
        )
    };
    let pair = |a: &Expr, b: &Expr| -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, [to_dict(py, a)?, to_dict(py, b)?])
    };
    let one = |a: &Expr| -> PyResult<Bound<'py, PyList>> { PyList::new(py, [to_dict(py, a)?]) };
    let empty = || PyList::empty(py);
    match e {
        Expr::Comparator(_, name, l, r) => {
            d.set_item("type", "comparator")?;
            d.set_item("children", pair(l, r)?)?;
            d.set_item("value", name.bind(py))?;
        }
        Expr::Current => {
            d.set_item("type", "current")?;
            d.set_item("children", empty())?;
        }
        Expr::Expref(x) => {
            d.set_item("type", "expref")?;
            d.set_item("children", PyList::new(py, [x.dict(py)?])?)?;
        }
        Expr::Function { name, args, .. } => {
            d.set_item("type", "function_expression")?;
            d.set_item("children", list(args)?)?;
            d.set_item("value", name.bind(py))?;
        }
        Expr::Field(name) => {
            d.set_item("type", "field")?;
            d.set_item("children", empty())?;
            d.set_item("value", name.bind(py))?;
        }
        Expr::FilterProjection(l, r, c) => {
            d.set_item("type", "filter_projection")?;
            d.set_item(
                "children",
                PyList::new(py, [to_dict(py, l)?, to_dict(py, r)?, to_dict(py, c)?])?,
            )?;
        }
        Expr::Flatten(x) => {
            d.set_item("type", "flatten")?;
            d.set_item("children", one(x)?)?;
        }
        Expr::Identity => {
            d.set_item("type", "identity")?;
            d.set_item("children", empty())?;
        }
        Expr::Index(i, _) => {
            d.set_item("type", "index")?;
            d.set_item("value", i.bind(py))?;
            d.set_item("children", empty())?;
        }
        Expr::IndexExpression(c) => {
            d.set_item("type", "index_expression")?;
            d.set_item("children", list(c)?)?;
        }
        Expr::KeyValPair(k, x) => {
            d.set_item("type", "key_val_pair")?;
            d.set_item("children", one(x)?)?;
            d.set_item("value", k.bind(py))?;
        }
        Expr::Literal(v) => {
            d.set_item("type", "literal")?;
            d.set_item("value", v.bind(py))?;
            d.set_item("children", empty())?;
        }
        Expr::MultiSelectDict(c) => {
            d.set_item("type", "multi_select_dict")?;
            d.set_item(
                "children",
                PyList::new(
                    py,
                    c.iter()
                        .map(|(_, x)| to_dict(py, x))
                        .collect::<PyResult<Vec<_>>>()?,
                )?,
            )?;
        }
        Expr::MultiSelectList(c) => {
            d.set_item("type", "multi_select_list")?;
            d.set_item("children", list(c)?)?;
        }
        Expr::Or(l, r) => {
            d.set_item("type", "or_expression")?;
            d.set_item("children", pair(l, r)?)?;
        }
        Expr::And(l, r) => {
            d.set_item("type", "and_expression")?;
            d.set_item("children", pair(l, r)?)?;
        }
        Expr::Not(x) => {
            d.set_item("type", "not_expression")?;
            d.set_item("children", one(x)?)?;
        }
        Expr::Pipe(c) => {
            d.set_item("type", "pipe")?;
            d.set_item("children", list(c)?)?;
        }
        Expr::Projection(l, r) => {
            d.set_item("type", "projection")?;
            d.set_item("children", pair(l, r)?)?;
        }
        Expr::Subexpression(c) => {
            d.set_item("type", "subexpression")?;
            d.set_item("children", list(c)?)?;
        }
        Expr::Slice(parts, _) => {
            d.set_item("type", "slice")?;
            d.set_item(
                "children",
                PyList::new(py, parts.iter().map(|p| p.bind(py)))?,
            )?;
        }
        Expr::ValueProjection(l, r) => {
            d.set_item("type", "value_projection")?;
            d.set_item("children", pair(l, r)?)?;
        }
        Expr::Unknown(t) => {
            d.set_item("type", t.bind(py))?;
            d.set_item("children", empty())?;
        }
    }
    Ok(d)
}
