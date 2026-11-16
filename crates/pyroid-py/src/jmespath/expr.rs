// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! The evaluable tree: jmespath's AST with Python values, built from the core parser's
//! nodes or from a dict AST (`ParsedResult(expression, parsed)`, `_Expression.visit(node)`),
//! and turned back into jmespath's dicts for `.parsed`.
//!
//! Left-associative chains (`a | b | c`, `a || b`, `a == b`, `a[][]`) are parsed in a loop, so a
//! tree can be as deep as its expression is long (jmespath compiles them, and only its
//! interpreter runs out of frames). Everything here that walks a whole tree (conversion,
//! [`frames`], [`to_dict`], dropping) is a loop with its own stack, never recursion.

use std::sync::Arc;

use pyo3::exceptions::PyRecursionError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyInt, PyList, PyString};
use pyroid_core::jmespath::{Comparator, Node, stack_position};

use super::functions::Builtin;
use super::host::Value;

/// An expref's expression, with its dict AST (shared by `.parsed` and the `_Expression`
/// objects made for it, as jmespath shares the node).
pub(crate) struct ExprefNode {
    pub expr: Expr,
    /// [`frames`] of `expr`.
    pub frames: usize,
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
        let frames = frames(&expr);
        Self::with_frames(expr, frames)
    }

    fn with_frames(expr: Expr, frames: usize) -> Arc<Self> {
        Arc::new(Self {
            expr,
            frames,
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

impl Expr {
    /// No child expressions (an expref's expression is reached through its [`ExprefNode`]).
    fn is_leaf(&self) -> bool {
        match self {
            Expr::Current
            | Expr::Expref(_)
            | Expr::Field(_)
            | Expr::Identity
            | Expr::Index(..)
            | Expr::Literal(_)
            | Expr::Slice(..)
            | Expr::Unknown(_) => true,
            Expr::Function { args: v, .. }
            | Expr::IndexExpression(v)
            | Expr::MultiSelectList(v)
            | Expr::Pipe(v)
            | Expr::Subexpression(v) => v.is_empty(),
            Expr::MultiSelectDict(v) => v.is_empty(),
            _ => false,
        }
    }

    /// Call `f` on each child expression, in jmespath's order (not into exprefs).
    fn for_each_child<'a>(&'a self, mut f: impl FnMut(&'a Expr)) {
        match self {
            Expr::Comparator(_, _, l, r)
            | Expr::Or(l, r)
            | Expr::And(l, r)
            | Expr::Projection(l, r)
            | Expr::ValueProjection(l, r) => {
                f(l);
                f(r);
            }
            Expr::FilterProjection(a, b, c) => {
                f(a);
                f(b);
                f(c);
            }
            Expr::Flatten(x) | Expr::Not(x) | Expr::KeyValPair(_, x) => f(x),
            Expr::Function { args: v, .. }
            | Expr::IndexExpression(v)
            | Expr::MultiSelectList(v)
            | Expr::Pipe(v)
            | Expr::Subexpression(v) => v.iter().for_each(f),
            Expr::MultiSelectDict(v) => v.iter().for_each(|(_, c)| f(c)),
            Expr::Current
            | Expr::Expref(_)
            | Expr::Field(_)
            | Expr::Identity
            | Expr::Index(..)
            | Expr::Literal(_)
            | Expr::Slice(..)
            | Expr::Unknown(_) => {}
        }
    }

    /// Call `f` on each child expression to change it, in jmespath's order (not into exprefs).
    fn for_each_child_mut(&mut self, mut f: impl FnMut(&mut Expr)) {
        match self {
            Expr::Comparator(_, _, l, r)
            | Expr::Or(l, r)
            | Expr::And(l, r)
            | Expr::Projection(l, r)
            | Expr::ValueProjection(l, r) => {
                f(l);
                f(r);
            }
            Expr::FilterProjection(a, b, c) => {
                f(a);
                f(b);
                f(c);
            }
            Expr::Flatten(x) | Expr::Not(x) | Expr::KeyValPair(_, x) => f(x),
            Expr::Function { args: v, .. }
            | Expr::IndexExpression(v)
            | Expr::MultiSelectList(v)
            | Expr::Pipe(v)
            | Expr::Subexpression(v) => v.iter_mut().for_each(f),
            Expr::MultiSelectDict(v) => v.iter_mut().for_each(|(_, c)| f(c)),
            Expr::Current
            | Expr::Expref(_)
            | Expr::Field(_)
            | Expr::Identity
            | Expr::Index(..)
            | Expr::Literal(_)
            | Expr::Slice(..)
            | Expr::Unknown(_) => {}
        }
    }

    /// Move the children that own further nodes into `out` (an identity is left in their
    /// place), and the expression of an expref only this node holds.
    fn take_inner_children(&mut self, out: &mut Vec<Expr>) {
        let mut take = |e: &mut Expr| {
            if !e.is_leaf() || matches!(e, Expr::Expref(_)) {
                out.push(std::mem::replace(e, Expr::Identity));
            }
        };
        if let Expr::Expref(x) = self {
            if let Some(node) = Arc::get_mut(x) {
                take(&mut node.expr);
            }
            return;
        }
        self.for_each_child_mut(take);
    }
}

/// Trees that need more [`frames`] than this (at least three per level) are freed by
/// [`free_deep`]; shallower ones by the usual recursive drop.
pub(crate) const DROP_FRAMES: usize = 600;

/// Free the nodes under `e` with a loop, leaving an identity. Left-associative chains
/// (`a | b | c`, `a || b`, `a == b`, `a[][]`) are parsed in a loop, so a tree can be as deep as
/// its expression is long.
pub(crate) fn free_deep(e: &mut Expr) {
    let mut pending = vec![std::mem::replace(e, Expr::Identity)];
    while let Some(mut e) = pending.pop() {
        e.take_inner_children(&mut pending);
    }
}

impl Drop for ExprefNode {
    fn drop(&mut self) {
        if self.frames > DROP_FRAMES {
            free_deep(&mut self.expr);
        }
    }
}

/// Frames a node needs below its `visit` (an upper bound; decides when depth is checked).
pub(crate) fn frames(root: &Expr) -> usize {
    // (node, its children are done), and the children's results.
    let mut todo = Vec::with_capacity(32);
    todo.push((root, false));
    let mut done: Vec<usize> = Vec::with_capacity(16);
    while let Some((e, ready)) = todo.pop() {
        if !ready && !e.is_leaf() {
            todo.push((e, true));
            e.for_each_child(|c| todo.push((c, false)));
            continue;
        }
        let mut n = 0;
        if ready {
            e.for_each_child(|_| n += 1);
        }
        let deepest = done.drain(done.len() - n..).max().unwrap_or(0);
        done.push(frames_with(e, deepest));
    }
    done.pop().unwrap_or(0)
}

type PyNode<'py> = Node<Value<'py>>;

/// Frames of a node whose deepest child needs `deepest` (see [`frames`]).
fn frames_with(e: &Expr, deepest: usize) -> usize {
    2 + match e {
        Expr::Expref(x) => x.frames + 6,
        Expr::Function { .. } => deepest.max(6) + 2,
        _ => deepest + 1,
    }
}

/// Trees at most this deep are converted by recursion (no work stacks to allocate; at most a
/// few dozen KiB of native stack).
const RECURSION_DEPTH: usize = 48;

/// Is the tree under `n` at most `limit` nodes deep?
fn depth_at_most(n: &PyNode<'_>, limit: usize) -> bool {
    n.is_leaf() || (limit > 1 && n.children().all(|c| depth_at_most(c, limit - 1)))
}

/// From the core parser's nodes: the tree and its [`frames`]. The node's values are moved
/// out (it is left empty).
pub(crate) fn from_node<'py>(py: Python<'py>, root: &mut PyNode<'py>) -> PyResult<(Expr, usize)> {
    enum Step<'a, 'py> {
        Enter(&'a mut PyNode<'py>),
        /// A node without its children, and how many it has.
        Exit(Expr, usize),
    }
    if depth_at_most(root, RECURSION_DEPTH) {
        return from_node_shallow(py, root);
    }
    // Children are converted last to first, so the first child's result is on top of `done`.
    let mut todo = vec![Step::Enter(root)];
    let mut done: Vec<(Expr, usize)> = Vec::new();
    while let Some(step) = todo.pop() {
        match step {
            Step::Enter(n) => {
                let e = match shell(py, n) {
                    Ok(e) => e,
                    Err(err) => {
                        for (mut e, _) in done {
                            free_deep(&mut e);
                        }
                        return Err(err);
                    }
                };
                let count = n.children().count();
                todo.push(Step::Exit(e, count));
                todo.extend(PyNode::children_mut(n).map(Step::Enter));
            }
            Step::Exit(mut e, count) => {
                let children = (0..count).map(|_| done.pop().expect("a converted child"));
                let deepest = fill(&mut e, children);
                let f = frames_with(&e, deepest);
                done.push((e, f));
            }
        }
    }
    Ok(done.pop().expect("a root expression"))
}

/// [`from_node`] by recursion, for a tree at most [`RECURSION_DEPTH`] deep.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)]
fn from_node_shallow<'py>(py: Python<'py>, n: &mut PyNode<'py>) -> PyResult<(Expr, usize)> {
    let mut deepest = 0;
    let mut sub = |c: &mut PyNode<'py>| -> PyResult<(Expr, usize)> {
        let r = from_node_shallow(py, c)?;
        deepest = deepest.max(r.1);
        Ok(r)
    };
    let e = match n {
        Node::Comparator(c, l, r) => {
            let l = Box::new(sub(l)?.0);
            let r = Box::new(sub(r)?.0);
            Expr::Comparator(Some(*c), intern(py, c.name()), l, r)
        }
        Node::Current => Expr::Current,
        Node::Expref(x) => {
            let (e, f) = sub(x)?;
            Expr::Expref(ExprefNode::with_frames(e, f))
        }
        Node::FunctionExpression(name, args) => {
            let name = take(name);
            let mut converted = Vec::with_capacity(args.len());
            for a in args.iter_mut() {
                converted.push(sub(a)?.0);
            }
            Expr::Function {
                builtin: builtin_of(&name),
                name: name.unbind(),
                args: converted,
            }
        }
        Node::Field(name) => Expr::Field(take(name).unbind()),
        Node::FilterProjection(l, r, c) => {
            let l = Box::new(sub(l)?.0);
            let r = Box::new(sub(r)?.0);
            Expr::FilterProjection(l, r, Box::new(sub(c)?.0))
        }
        Node::Flatten(x) => Expr::Flatten(Box::new(sub(x)?.0)),
        Node::Identity => Expr::Identity,
        Node::Index(i) => index(take(i)),
        Node::IndexExpression(c) => Expr::IndexExpression(
            c.iter_mut()
                .map(|c| sub(c).map(|r| r.0))
                .collect::<PyResult<_>>()?,
        ),
        Node::KeyValPair(k, x) => {
            let k = take(k).unbind();
            Expr::KeyValPair(k, Box::new(sub(x)?.0))
        }
        Node::Literal(x) => Expr::Literal(take(x).unbind()),
        Node::MultiSelectDict(c) => {
            let mut pairs = Vec::with_capacity(c.len());
            for child in c.iter_mut() {
                let key = match child {
                    Node::KeyValPair(Some(k), _) => k.clone().unbind(),
                    _ => py.None(),
                };
                pairs.push((key, sub(child)?.0));
            }
            Expr::MultiSelectDict(pairs)
        }
        Node::MultiSelectList(c) => Expr::MultiSelectList(
            c.iter_mut()
                .map(|c| sub(c).map(|r| r.0))
                .collect::<PyResult<_>>()?,
        ),
        Node::OrExpression(l, r) => {
            let l = Box::new(sub(l)?.0);
            Expr::Or(l, Box::new(sub(r)?.0))
        }
        Node::AndExpression(l, r) => {
            let l = Box::new(sub(l)?.0);
            Expr::And(l, Box::new(sub(r)?.0))
        }
        Node::NotExpression(x) => Expr::Not(Box::new(sub(x)?.0)),
        Node::Pipe(l, r) => {
            let l = sub(l)?.0;
            Expr::Pipe(vec![l, sub(r)?.0])
        }
        Node::Projection(l, r) => {
            let l = Box::new(sub(l)?.0);
            Expr::Projection(l, Box::new(sub(r)?.0))
        }
        Node::Subexpression(c) => Expr::Subexpression(
            c.iter_mut()
                .map(|c| sub(c).map(|r| r.0))
                .collect::<PyResult<_>>()?,
        ),
        Node::Slice(parts) => slice_of(py, parts)?,
        Node::ValueProjection(l, r) => {
            let l = Box::new(sub(l)?.0);
            Expr::ValueProjection(l, Box::new(sub(r)?.0))
        }
    };
    let f = frames_with(&e, deepest);
    Ok((e, f))
}

/// A value of the node being converted (each is taken once).
fn take<'py>(v: &mut Value<'py>) -> Bound<'py, PyAny> {
    v.take().expect("a node value, taken once")
}

/// A slice node's `Expr`, its parts moved out.
fn slice_of<'py>(py: Python<'py>, parts: &mut [Option<Value<'py>>; 3]) -> PyResult<Expr> {
    let parts: Vec<Py<PyAny>> = parts
        .iter_mut()
        .map(|p| p.take().flatten().map_or_else(|| py.None(), Bound::unbind))
        .collect();
    slice(py, parts)
}

/// `n` as an `Expr` with identities for children (see [`fill`]), its values moved out.
fn shell<'py>(py: Python<'py>, n: &mut PyNode<'py>) -> PyResult<Expr> {
    let id = || Box::new(Expr::Identity);
    let ids = |k: usize| (0..k).map(|_| Expr::Identity).collect::<Vec<_>>();
    Ok(match n {
        Node::Comparator(c, ..) => Expr::Comparator(Some(*c), intern(py, c.name()), id(), id()),
        Node::Current => Expr::Current,
        Node::Expref(_) => Expr::Expref(ExprefNode::with_frames(Expr::Identity, 0)),
        Node::FunctionExpression(name, args) => {
            let name = take(name);
            Expr::Function {
                builtin: builtin_of(&name),
                name: name.unbind(),
                args: ids(args.len()),
            }
        }
        Node::Field(name) => Expr::Field(take(name).unbind()),
        Node::FilterProjection(..) => Expr::FilterProjection(id(), id(), id()),
        Node::Flatten(_) => Expr::Flatten(id()),
        Node::Identity => Expr::Identity,
        Node::Index(i) => index(take(i)),
        Node::IndexExpression(c) => Expr::IndexExpression(ids(c.len())),
        Node::KeyValPair(k, _) => Expr::KeyValPair(take(k).unbind(), id()),
        Node::Literal(x) => Expr::Literal(take(x).unbind()),
        // The keys are the pairs' own values: read here, before the pairs are converted.
        Node::MultiSelectDict(c) => Expr::MultiSelectDict(
            c.iter()
                .map(|child| {
                    let key = match child {
                        Node::KeyValPair(Some(k), _) => k.clone().unbind(),
                        _ => py.None(),
                    };
                    (key, Expr::Identity)
                })
                .collect(),
        ),
        Node::MultiSelectList(c) => Expr::MultiSelectList(ids(c.len())),
        Node::OrExpression(..) => Expr::Or(id(), id()),
        Node::AndExpression(..) => Expr::And(id(), id()),
        Node::NotExpression(_) => Expr::Not(id()),
        Node::Pipe(..) => Expr::Pipe(ids(2)),
        Node::Projection(..) => Expr::Projection(id(), id()),
        Node::Subexpression(c) => Expr::Subexpression(ids(c.len())),
        Node::Slice(parts) => slice_of(py, parts)?,
        Node::ValueProjection(..) => Expr::ValueProjection(id(), id()),
    })
}

/// Put `children` (converted, in order) into `e`'s places for them; the most frames any
/// child needs.
fn fill(e: &mut Expr, children: impl IntoIterator<Item = (Expr, usize)>) -> usize {
    let mut children = children.into_iter();
    if let Expr::Expref(x) = e {
        let (Some(node), Some((child, f))) = (Arc::get_mut(x), children.next()) else {
            return 0;
        };
        node.expr = child;
        node.frames = f;
        return f;
    }
    let mut deepest = 0;
    e.for_each_child_mut(|slot| {
        if let Some((child, f)) = children.next() {
            deepest = deepest.max(f);
            *slot = child;
        }
    });
    deepest
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
/// `stack`: the native stack it may use (it recurses once per level).
pub(crate) fn from_dict(node: &Bound<'_, PyAny>, stack: usize) -> PyResult<Expr> {
    dict_node(node, 0, stack_position(), stack)
}

fn dict_node(node: &Bound<'_, PyAny>, depth: usize, base: usize, stack: usize) -> PyResult<Expr> {
    if depth > MAX_DICT_DEPTH || stack_position().abs_diff(base) > stack {
        return Err(PyRecursionError::new_err(
            "maximum recursion depth exceeded while reading a jmespath AST",
        ));
    }
    let py = node.py();
    let kind = node.get_item("type")?;
    let children = || node.get_item("children");
    let child = |i: usize| -> PyResult<Expr> {
        dict_node(&children()?.get_item(i)?, depth + 1, base, stack)
    };
    let child_box = |i: usize| child(i).map(Box::new);
    let all = || -> PyResult<Vec<Expr>> {
        children()?
            .try_iter()?
            .map(|c| dict_node(&c?, depth + 1, base, stack))
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
                out.push((key, dict_node(&c, depth + 1, base, stack)?));
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

/// jmespath's dict for `root` (`ast.py`'s key order).
pub(crate) fn to_dict<'py>(py: Python<'py>, root: &Expr) -> PyResult<Bound<'py, PyDict>> {
    enum Step<'a> {
        /// A node, and whether its children's dicts are done.
        Node(&'a Expr, bool),
        /// Keep the dict on top as this expref's own.
        Expref(&'a ExprefNode),
    }
    // Children are done last to first, so the first child's dict is on top of `done`.
    let mut todo = vec![Step::Node(root, false)];
    let mut done: Vec<Bound<'py, PyDict>> = Vec::new();
    while let Some(step) = todo.pop() {
        match step {
            Step::Node(e @ Expr::Expref(x), false) if x.dict.get(py).is_none() => {
                todo.push(Step::Node(e, true));
                todo.push(Step::Expref(x));
                todo.push(Step::Node(&x.expr, false));
            }
            Step::Node(e, false) if !e.is_leaf() => {
                todo.push(Step::Node(e, true));
                e.for_each_child(|c| todo.push(Step::Node(c, false)));
            }
            Step::Node(e, _) => {
                let d = node_dict(py, e, &mut done)?;
                done.push(d);
            }
            Step::Expref(x) => {
                let d = done.pop().expect("the expref's dict");
                // If another thread set one meanwhile, that one is kept.
                let _ = x.dict.set(py, d.into_any().unbind());
            }
        }
    }
    Ok(done.pop().expect("the root's dict"))
}

/// The dict of one node, its children's dicts (in order) taken from the top of `done`: one
/// arm per node kind.
#[allow(clippy::too_many_lines)]
fn node_dict<'py>(
    py: Python<'py>,
    e: &Expr,
    done: &mut Vec<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let mut kids = |n: usize| PyList::new(py, (0..n).map(|_| done.pop().expect("a child's dict")));
    let empty = || PyList::empty(py);
    match e {
        Expr::Comparator(_, name, ..) => {
            d.set_item("type", "comparator")?;
            d.set_item("children", kids(2)?)?;
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
            d.set_item("children", kids(args.len())?)?;
            d.set_item("value", name.bind(py))?;
        }
        Expr::Field(name) => {
            d.set_item("type", "field")?;
            d.set_item("children", empty())?;
            d.set_item("value", name.bind(py))?;
        }
        Expr::FilterProjection(..) => {
            d.set_item("type", "filter_projection")?;
            d.set_item("children", kids(3)?)?;
        }
        Expr::Flatten(_) => {
            d.set_item("type", "flatten")?;
            d.set_item("children", kids(1)?)?;
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
            d.set_item("children", kids(c.len())?)?;
        }
        Expr::KeyValPair(k, _) => {
            d.set_item("type", "key_val_pair")?;
            d.set_item("children", kids(1)?)?;
            d.set_item("value", k.bind(py))?;
        }
        Expr::Literal(v) => {
            d.set_item("type", "literal")?;
            d.set_item("value", v.bind(py))?;
            d.set_item("children", empty())?;
        }
        Expr::MultiSelectDict(c) => {
            d.set_item("type", "multi_select_dict")?;
            d.set_item("children", kids(c.len())?)?;
        }
        Expr::MultiSelectList(c) => {
            d.set_item("type", "multi_select_list")?;
            d.set_item("children", kids(c.len())?)?;
        }
        Expr::Or(..) => {
            d.set_item("type", "or_expression")?;
            d.set_item("children", kids(2)?)?;
        }
        Expr::And(..) => {
            d.set_item("type", "and_expression")?;
            d.set_item("children", kids(2)?)?;
        }
        Expr::Not(_) => {
            d.set_item("type", "not_expression")?;
            d.set_item("children", kids(1)?)?;
        }
        Expr::Pipe(c) => {
            d.set_item("type", "pipe")?;
            d.set_item("children", kids(c.len())?)?;
        }
        Expr::Projection(..) => {
            d.set_item("type", "projection")?;
            d.set_item("children", kids(2)?)?;
        }
        Expr::Subexpression(c) => {
            d.set_item("type", "subexpression")?;
            d.set_item("children", kids(c.len())?)?;
        }
        Expr::Slice(parts, _) => {
            d.set_item("type", "slice")?;
            d.set_item(
                "children",
                PyList::new(py, parts.iter().map(|p| p.bind(py)))?,
            )?;
        }
        Expr::ValueProjection(..) => {
            d.set_item("type", "value_projection")?;
            d.set_item("children", kids(2)?)?;
        }
        Expr::Unknown(t) => {
            d.set_item("type", t.bind(py))?;
            d.set_item("children", empty())?;
        }
    }
    Ok(d)
}
