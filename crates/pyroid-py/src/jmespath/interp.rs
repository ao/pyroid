// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! jmespath's `TreeInterpreter` (`jmespath/visitor.py`), on Python objects.
//!
//! Each `visit_*` method is ported with its exact rules: `value.get(key)` for fields (any
//! object with a `get`), `isinstance(x, list)` for projections, falsiness by `==` against
//! `''`, `[]` and `{}`, jmespath's 0/1 versus `True`/`False` rule for `==`, ordering only for
//! numbers (`numbers.Number`, not `bool`) and strings. Exact builtin types take native paths;
//! anything else goes through the same Python operations jmespath performs.

use std::cell::Cell;
use std::sync::Arc;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyAttributeError, PyIndexError, PyKeyError, PyNotImplementedError, PyRecursionError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use pyroid_core::jmespath::Comparator;

use super::expr::{Expr, ExprefNode};
use super::functions::{self, unknown_function};
use super::{Interpreter, config};

/// How functions are called: natively, or through a custom `Functions` object.
#[derive(Debug)]
pub(crate) enum Funcs {
    Native,
    /// `options.custom_functions`; `stock` indexes the stock `Functions` class it derives
    /// from without changing how calls are dispatched (then stock entries run natively).
    Custom {
        obj: Py<PyAny>,
        stock: Option<usize>,
    },
}

/// What `TreeInterpreter.__init__` takes from the options.
#[derive(Debug)]
pub(crate) struct Opts {
    /// `None`: plain `dict`.
    pub dict_cls: Option<Py<PyAny>>,
    pub funcs: Funcs,
    /// The options object itself (for `_Expression.interpreter`).
    pub options: Option<Py<PyAny>>,
}

/// One evaluation.
pub(crate) struct Ctx<'py, 'a> {
    pub py: Python<'py>,
    pub opts: &'a Arc<Opts>,
    /// The deepest Python frame jmespath could open (`usize::MAX`: not checked).
    pub limit: usize,
    /// Where the evaluation's stack starts (see [`pyroid_core::jmespath::stack_position`]),
    /// and how much it may use.
    stack_base: usize,
    stack: usize,
    ticks: Cell<u32>,
}

impl<'py, 'a> Ctx<'py, 'a> {
    pub(crate) fn new(py: Python<'py>, opts: &'a Arc<Opts>, limit: usize, stack: usize) -> Self {
        Self {
            py,
            opts,
            limit,
            stack_base: pyroid_core::jmespath::stack_position(),
            stack,
            ticks: Cell::new(0),
        }
    }

    /// Fail as Python would if a frame at depth `frame` were opened, or before the native
    /// stack runs short.
    pub(crate) fn check(&self, frame: usize) -> PyResult<()> {
        let used = pyroid_core::jmespath::stack_position().abs_diff(self.stack_base);
        if frame > self.limit || used > self.stack {
            return Err(PyRecursionError::new_err(
                "maximum recursion depth exceeded",
            ));
        }
        Ok(())
    }

    /// Let Ctrl-C through on long loops.
    fn tick(&self) -> PyResult<()> {
        let t = self.ticks.get().wrapping_add(1);
        self.ticks.set(t);
        if t % 65_536 == 0 {
            self.py.check_signals()?;
        }
        Ok(())
    }
}

/// A function argument: a value, or an expref not yet made into an `_Expression`.
pub(crate) enum Val<'py, 'a> {
    Py(Bound<'py, PyAny>),
    Expref(&'a Arc<ExprefNode>),
}

impl<'py> Val<'py, '_> {
    pub(crate) fn to_py(&self, ctx: &Ctx<'py, '_>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Val::Py(o) => Ok(o.clone()),
            Val::Expref(e) => expref(ctx, e),
        }
    }
}

/// `_Expression(node, interpreter)`: jmespath's class (pyroid's copy without jmespath), with
/// an interpreter that evaluates natively.
fn expref<'py>(ctx: &Ctx<'py, '_>, e: &Arc<ExprefNode>) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let interpreter = Bound::new(py, Interpreter::new(Arc::clone(e), Arc::clone(ctx.opts)))?;
    config(py)?
        .expression_cls
        .bind(py)
        .call1((e.dict(py)?, interpreter))
}

/// `isinstance(x, list)`.
pub(crate) fn is_list(x: &Bound<'_, PyAny>) -> bool {
    x.is_instance_of::<PyList>()
}

/// `for element in x`, stopping when `f` returns `false`.
pub(crate) fn for_each<'py>(
    ctx: &Ctx<'py, '_>,
    x: &Bound<'py, PyAny>,
    mut f: impl FnMut(Bound<'py, PyAny>) -> PyResult<bool>,
) -> PyResult<()> {
    if let Ok(l) = x.cast_exact::<PyList>() {
        // One read of the list: another thread (or a custom function) may change it meanwhile.
        for el in crate::input::list_items(l).iter() {
            ctx.tick()?;
            if !f(el)? {
                break;
            }
        }
        return Ok(());
    }
    for el in x.try_iter()? {
        ctx.tick()?;
        if !f(el?)? {
            break;
        }
    }
    Ok(())
}

fn none(py: Python<'_>) -> Bound<'_, PyAny> {
    py.None().into_bound(py)
}

fn pybool(py: Python<'_>, b: bool) -> Bound<'_, PyAny> {
    PyBool::new(py, b).to_owned().into_any()
}

/// A type without a `get` or `values` attribute (no lookup needed).
fn is_plain_scalar_or_list(x: &Bound<'_, PyAny>) -> bool {
    x.is_none()
        || x.is_exact_instance_of::<PyList>()
        || x.is_exact_instance_of::<PyString>()
        || x.is_exact_instance_of::<PyInt>()
        || x.is_exact_instance_of::<PyFloat>()
        || x.is_instance_of::<PyBool>()
        || x.is_exact_instance_of::<PyTuple>()
}

/// `_is_actual_number`: not a bool, and an instance of `numbers.Number`.
fn is_actual_number(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    if x.is_exact_instance_of::<PyInt>() || x.is_exact_instance_of::<PyFloat>() {
        return Ok(true);
    }
    if x.is_instance_of::<PyBool>()
        || x.is_none()
        || x.is_exact_instance_of::<PyString>()
        || x.is_exact_instance_of::<PyList>()
        || x.is_exact_instance_of::<PyDict>()
    {
        return Ok(false);
    }
    let py = x.py();
    x.is_instance(config(py)?.number.bind(py))
}

/// `x in (0, 1)`.
// Exact comparisons, as Python's `==` does them.
#[allow(clippy::float_cmp)]
fn is_zero_or_one(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    if x.is_exact_instance_of::<PyInt>() {
        return Ok(matches!(x.extract::<i64>(), Ok(0 | 1)));
    }
    if let Ok(f) = x.cast_exact::<PyFloat>() {
        let v = f.value();
        return Ok(v == 0.0 || v == 1.0);
    }
    let py = x.py();
    config(py)?.zero_one.bind(py).contains(x)
}

/// An exact `int` (that fits) or `float`: compared natively, as Python compares them.
#[derive(Clone, Copy)]
enum Num {
    I(i64),
    F(f64),
}

fn num(x: &Bound<'_, PyAny>) -> Option<Num> {
    if x.is_exact_instance_of::<PyInt>() {
        return x.extract::<i64>().ok().map(Num::I);
    }
    x.cast_exact::<PyFloat>().ok().map(|f| Num::F(f.value()))
}

/// `a` against `b` as Python orders them (`None` when either is NaN).
fn num_cmp(a: Num, b: Num) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Num::I(a), Num::I(b)) => Some(a.cmp(&b)),
        (Num::F(a), Num::F(b)) => a.partial_cmp(&b),
        (Num::I(a), Num::F(b)) => (!b.is_nan()).then(|| functions::cmp_int_float(a, b)),
        (Num::F(a), Num::I(b)) => (!a.is_nan()).then(|| functions::cmp_int_float(b, a).reverse()),
    }
}

/// `x <op> y` for two exact numbers or two exact (valid UTF-8) strings, natively.
fn native_compare(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>, op: CompareOp) -> Option<bool> {
    use std::cmp::Ordering::{Equal, Greater, Less};
    let ord = if let (Some(a), Some(b)) = (num(x), num(y)) {
        num_cmp(a, b)
    } else if let (Ok(a), Ok(b)) = (x.cast_exact::<PyString>(), y.cast_exact::<PyString>()) {
        Some(a.to_str().ok()?.cmp(b.to_str().ok()?))
    } else {
        return None;
    };
    Some(match (op, ord) {
        (CompareOp::Eq, o) => o == Some(Equal),
        (CompareOp::Ne, o) => o != Some(Equal),
        (_, None) => false,
        (CompareOp::Lt, Some(o)) => o == Less,
        (CompareOp::Le, Some(o)) => o != Greater,
        (CompareOp::Gt, Some(o)) => o == Greater,
        (CompareOp::Ge, Some(o)) => o != Less,
    })
}

/// `_equals(x, y)`: `x == y`, except that 0 and 1 never equal `False` and `True`.
fn equals<'py>(x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // Two exact numbers or strings: no bool involved, so the special case can't apply.
    if let Some(b) = native_compare(x, y, CompareOp::Eq) {
        return Ok(pybool(x.py(), b));
    }
    let special = if is_actual_number(x)? && is_zero_or_one(x)? {
        y.is_instance_of::<PyBool>()
    } else if is_actual_number(y)? && is_zero_or_one(y)? {
        x.is_instance_of::<PyBool>()
    } else {
        false
    };
    if special {
        return Ok(pybool(x.py(), false));
    }
    x.rich_compare(y, CompareOp::Eq)
}

/// `_is_comparable`.
fn is_comparable(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(is_actual_number(x)? || x.is_instance_of::<PyString>())
}

/// `TreeInterpreter._is_false`.
pub(crate) fn is_false(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(s) = x.cast_exact::<PyString>() {
        return Ok(s.len()? == 0);
    }
    if let Ok(l) = x.cast_exact::<PyList>() {
        return Ok(l.is_empty());
    }
    if let Ok(d) = x.cast_exact::<PyDict>() {
        return Ok(d.is_empty());
    }
    if x.is_none() {
        return Ok(true);
    }
    if let Ok(b) = x.cast::<PyBool>() {
        return Ok(!b.is_true());
    }
    if x.is_exact_instance_of::<PyInt>() || x.is_exact_instance_of::<PyFloat>() {
        return Ok(false);
    }
    let py = x.py();
    if x.rich_compare(PyString::new(py, ""), CompareOp::Eq)?
        .is_truthy()?
        || x.rich_compare(PyList::empty(py), CompareOp::Eq)?
            .is_truthy()?
        || x.rich_compare(PyDict::new(py), CompareOp::Eq)?
            .is_truthy()?
    {
        return Ok(true);
    }
    Ok(x.is_none() || x.is(PyBool::new(py, false)))
}

/// `visit_field`: `value.get(key)`, `None` on `AttributeError`.
fn field<'py>(value: &Bound<'py, PyAny>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    if let Ok(d) = value.cast_exact::<PyDict>() {
        return Ok(d.get_item(key)?.unwrap_or_else(|| none(py)));
    }
    if is_plain_scalar_or_list(value) {
        return Ok(none(py));
    }
    let attribute = |e: &PyErr| e.is_instance_of::<PyAttributeError>(py);
    match value.getattr(pyo3::intern!(py, "get")) {
        Ok(get) => match get.call1((key,)) {
            Ok(v) => Ok(v),
            Err(e) if attribute(&e) => Ok(none(py)),
            Err(e) => Err(e),
        },
        Err(e) if attribute(&e) => Ok(none(py)),
        Err(e) => Err(e),
    }
}

/// `base.values()`, or `None` on `AttributeError`.
fn values_of<'py>(base: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = base.py();
    if let Ok(d) = base.cast_exact::<PyDict>() {
        return Ok(Some(d.values().into_any()));
    }
    if is_plain_scalar_or_list(base) {
        return Ok(None);
    }
    let attribute = |e: &PyErr| e.is_instance_of::<PyAttributeError>(py);
    match base.getattr(pyo3::intern!(py, "values")) {
        Ok(values) => match values.call0() {
            Ok(v) => Ok(Some(v)),
            Err(e) if attribute(&e) => Ok(None),
            Err(e) => Err(e),
        },
        Err(e) if attribute(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Visit `node` with `value`. `depth`: the frames open before this node's `visit`.
///
/// Each `visit_*` method is its own function (never inlined), so a level of nesting costs
/// only the frame of the method that recurses.
pub(crate) fn visit<'py>(
    ctx: &Ctx<'py, '_>,
    node: &Expr,
    value: &Bound<'py, PyAny>,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    // visit -> visit_<type> -> a helper.
    ctx.check(depth + 3)?;
    let inner = depth + 2;
    match node {
        Expr::Subexpression(children) | Expr::IndexExpression(children) | Expr::Pipe(children) => {
            visit_chain(ctx, children, value, inner)
        }
        Expr::Field(key) => field(value, key.bind(py)),
        Expr::Comparator(op, name, l, r) => visit_comparator(ctx, *op, name, l, r, value, inner),
        Expr::Current | Expr::Identity => Ok(value.clone()),
        Expr::Expref(e) => expref(ctx, e),
        Expr::Function {
            name,
            builtin,
            args,
        } => call_function(ctx, name.bind(py), *builtin, args, value, depth),
        Expr::FilterProjection(left, right, cond) => {
            visit_filter_projection(ctx, left, right, cond, value, inner)
        }
        Expr::Flatten(x) => visit_flatten(ctx, x, value, inner),
        Expr::Index(index, fast) => visit_index(value, index.bind(py), *fast),
        Expr::KeyValPair(_, x) => visit(ctx, x, value, inner),
        Expr::Literal(v) => Ok(v.bind(py).clone()),
        Expr::MultiSelectDict(children) => visit_multi_select_dict(ctx, children, value, inner),
        Expr::MultiSelectList(children) => visit_multi_select_list(ctx, children, value, inner),
        Expr::Or(l, r) => visit_or(ctx, l, r, value, inner),
        Expr::And(l, r) => visit_and(ctx, l, r, value, inner),
        Expr::Not(x) => visit_not(ctx, x, value, inner),
        Expr::Projection(l, r) => visit_projection(ctx, l, r, value, inner),
        Expr::Slice(_, s) => {
            if !is_list(value) {
                return Ok(none(py));
            }
            value.get_item(s.bind(py))
        }
        Expr::ValueProjection(l, r) => visit_value_projection(ctx, l, r, value, inner),
        Expr::Unknown(t) => Err(PyNotImplementedError::new_err(t.clone_ref(py))),
    }
}

/// `visit_subexpression`, `visit_index_expression`, `visit_pipe`.
#[inline(never)]
fn visit_chain<'py>(
    ctx: &Ctx<'py, '_>,
    children: &[Expr],
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let step = |c: &Expr, v: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        if let Expr::Field(key) = c {
            // A field opens no further frames: only the frame limit can stop it.
            if inner + 3 > ctx.limit {
                ctx.check(inner + 3)?;
            }
            field(v, key.bind(ctx.py))
        } else {
            visit(ctx, c, v, inner)
        }
    };
    let Some((first, rest)) = children.split_first() else {
        return Ok(value.clone());
    };
    let mut result = step(first, value)?;
    for c in rest {
        result = step(c, &result)?;
    }
    Ok(result)
}

#[inline(never)]
fn visit_comparator<'py>(
    ctx: &Ctx<'py, '_>,
    op: Option<Comparator>,
    name: &Py<PyAny>,
    l: &Expr,
    r: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let Some(op) = op else {
        return Err(PyKeyError::new_err(name.clone_ref(py)));
    };
    let left = visit(ctx, l, value, inner)?;
    let right = visit(ctx, r, value, inner)?;
    match op {
        Comparator::Eq => equals(&left, &right),
        Comparator::Ne => Ok(pybool(py, !equals(&left, &right)?.is_truthy()?)),
        _ => {
            // Ordering is only for numbers and strings; anything else gives None.
            if !(is_comparable(&left)? && is_comparable(&right)?) {
                return Ok(none(py));
            }
            let op = match op {
                Comparator::Lt => CompareOp::Lt,
                Comparator::Gt => CompareOp::Gt,
                Comparator::Lte => CompareOp::Le,
                _ => CompareOp::Ge,
            };
            if let Some(b) = native_compare(&left, &right, op) {
                return Ok(pybool(py, b));
            }
            left.rich_compare(&right, op)
        }
    }
}

#[inline(never)]
fn visit_filter_projection<'py>(
    ctx: &Ctx<'py, '_>,
    left: &Expr,
    right: &Expr,
    cond: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let base = visit(ctx, left, value, inner)?;
    if !is_list(&base) {
        return Ok(none(py));
    }
    let mut collected = Vec::new();
    for_each(ctx, &base, |el| {
        if !is_false(&visit(ctx, cond, &el, inner)?)? {
            let current = visit(ctx, right, &el, inner)?;
            if !current.is_none() {
                collected.push(current);
            }
        }
        Ok(true)
    })?;
    Ok(PyList::new(py, collected)?.into_any())
}

#[inline(never)]
fn visit_flatten<'py>(
    ctx: &Ctx<'py, '_>,
    x: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let base = visit(ctx, x, value, inner)?;
    if !is_list(&base) {
        // Can't flatten the object if it's not a list.
        return Ok(none(py));
    }
    let mut merged = Vec::new();
    for_each(ctx, &base, |el| {
        if let Ok(l) = el.cast_exact::<PyList>() {
            merged.extend(crate::input::list_items(l).iter());
        } else if is_list(&el) {
            // list.extend iterates a list subclass with its own __iter__.
            for item in el.try_iter()? {
                merged.push(item?);
            }
        } else {
            merged.push(el);
        }
        Ok(true)
    })?;
    Ok(PyList::new(py, merged)?.into_any())
}

fn visit_index<'py>(
    value: &Bound<'py, PyAny>,
    index: &Bound<'py, PyAny>,
    fast: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    if !is_list(value) {
        return Ok(none(py));
    }
    if let (Ok(l), Some(i)) = (value.cast_exact::<PyList>(), fast) {
        // One read of one item (the list can change between a len() and a get_item()); a
        // negative index goes through list.__getitem__, which reads the length with the item.
        if let Ok(i) = usize::try_from(i) {
            return Ok(l.get_item(i).unwrap_or_else(|_| none(py)));
        }
    }
    match value.get_item(index) {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyIndexError>(py) => Ok(none(py)),
        Err(e) => Err(e),
    }
}

#[inline(never)]
fn visit_multi_select_dict<'py>(
    ctx: &Ctx<'py, '_>,
    children: &[(Py<PyAny>, Expr)],
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    if value.is_none() {
        return Ok(none(py));
    }
    let d = match &ctx.opts.dict_cls {
        None => PyDict::new(py).into_any(),
        Some(cls) => cls.bind(py).call0()?,
    };
    let exact = d.cast_exact::<PyDict>().ok();
    for (key, child) in children {
        let v = visit(ctx, child, value, inner)?;
        match exact {
            Some(e) => e.set_item(key.bind(py), v)?,
            None => d.set_item(key.bind(py), v)?,
        }
    }
    Ok(d)
}

#[inline(never)]
fn visit_multi_select_list<'py>(
    ctx: &Ctx<'py, '_>,
    children: &[Expr],
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    if value.is_none() {
        return Ok(none(py));
    }
    let mut collected = Vec::with_capacity(children.len());
    for c in children {
        collected.push(visit(ctx, c, value, inner)?);
    }
    Ok(PyList::new(py, collected)?.into_any())
}

#[inline(never)]
fn visit_or<'py>(
    ctx: &Ctx<'py, '_>,
    l: &Expr,
    r: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let matched = visit(ctx, l, value, inner)?;
    if is_false(&matched)? {
        return visit(ctx, r, value, inner);
    }
    Ok(matched)
}

#[inline(never)]
fn visit_and<'py>(
    ctx: &Ctx<'py, '_>,
    l: &Expr,
    r: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let matched = visit(ctx, l, value, inner)?;
    if is_false(&matched)? {
        return Ok(matched);
    }
    visit(ctx, r, value, inner)
}

#[inline(never)]
fn visit_not<'py>(
    ctx: &Ctx<'py, '_>,
    x: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let original = visit(ctx, x, value, inner)?;
    // !0 is false: 0 is not special-cased in jmespath.
    if is_actual_number(&original)? {
        let zero = if original.is_exact_instance_of::<PyInt>() {
            matches!(original.extract::<i64>(), Ok(0))
        } else if let Ok(f) = original.cast_exact::<PyFloat>() {
            f.value() == 0.0
        } else {
            original.rich_compare(0, CompareOp::Eq)?.is_truthy()?
        };
        if zero {
            return Ok(pybool(py, false));
        }
    }
    Ok(pybool(py, !original.is_truthy()?))
}

#[inline(never)]
fn visit_projection<'py>(
    ctx: &Ctx<'py, '_>,
    l: &Expr,
    r: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let base = visit(ctx, l, value, inner)?;
    if !is_list(&base) {
        return Ok(none(py));
    }
    // A projection usually keeps most elements: one allocation for all of them.
    let mut collected =
        Vec::with_capacity(base.cast_exact::<PyList>().map_or(0, PyListMethods::len));
    if matches!(r, Expr::Identity) {
        for_each(ctx, &base, |el| {
            if !el.is_none() {
                collected.push(el);
            }
            Ok(true)
        })?;
    } else {
        for_each(ctx, &base, |el| {
            let current = visit(ctx, r, &el, inner)?;
            if !current.is_none() {
                collected.push(current);
            }
            Ok(true)
        })?;
    }
    Ok(PyList::new(py, collected)?.into_any())
}

#[inline(never)]
fn visit_value_projection<'py>(
    ctx: &Ctx<'py, '_>,
    l: &Expr,
    r: &Expr,
    value: &Bound<'py, PyAny>,
    inner: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let base = visit(ctx, l, value, inner)?;
    let Some(values) = values_of(&base)? else {
        return Ok(none(py));
    };
    let mut collected = Vec::new();
    for_each(ctx, &values, |el| {
        let current = visit(ctx, r, &el, inner)?;
        if !current.is_none() {
            collected.push(current);
        }
        Ok(true)
    })?;
    Ok(PyList::new(py, collected)?.into_any())
}

/// `visit_function_expression`: the arguments, then `call_function(name, args)`.
#[inline(never)]
fn call_function<'py, 'a>(
    ctx: &Ctx<'py, '_>,
    name: &Bound<'py, PyAny>,
    builtin: Option<functions::Builtin>,
    args: &'a [Expr],
    value: &Bound<'py, PyAny>,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let inner = depth + 2;
    let mut resolved: Vec<Val<'py, 'a>> = Vec::with_capacity(args.len());
    for a in args {
        resolved.push(match a {
            Expr::Expref(e) => {
                ctx.check(inner + 3)?;
                Val::Expref(e)
            }
            _ => Val::Py(visit(ctx, a, value, inner)?),
        });
    }
    match &ctx.opts.funcs {
        Funcs::Native => match builtin {
            Some(b) => functions::call(ctx, b, name, &resolved, inner),
            None => table_function(ctx, name, &resolved),
        },
        Funcs::Custom { obj, stock } => {
            let obj = obj.bind(py);
            if let Some(i) = stock {
                let table = obj.getattr(pyo3::intern!(py, "FUNCTION_TABLE"))?;
                let spec = match table.get_item(name) {
                    Ok(s) => s,
                    Err(e) if e.is_instance_of::<PyKeyError>(py) => {
                        return Err(unknown_function(py, name));
                    }
                    Err(e) => return Err(e),
                };
                if let Some(b) = builtin {
                    if config(py)?.stocks[*i].is_stock(py, b, &spec)? {
                        return functions::call(ctx, b, name, &resolved, inner);
                    }
                }
            }
            obj.call_method1(
                pyo3::intern!(py, "call_function"),
                (name, args_list(ctx, &resolved)?),
            )
        }
    }
}

/// The resolved arguments as the list `call_function` takes.
fn args_list<'py>(ctx: &Ctx<'py, '_>, resolved: &[Val<'py, '_>]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(
        ctx.py,
        resolved
            .iter()
            .map(|v| v.to_py(ctx))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

/// A function that is not a built-in, without custom functions: one added to a stock
/// class's `FUNCTION_TABLE` (jmespath's first, then pyroid's) runs as jmespath's
/// `Functions().call_function(name, args)` would run it; any other name raises jmespath's
/// `UnknownFunctionError`. Only names that are not built-ins come here, so the tables are
/// read on this path only.
fn table_function<'py>(
    ctx: &Ctx<'py, '_>,
    name: &Bound<'py, PyAny>,
    resolved: &[Val<'py, '_>],
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    for stock in config(py)?.stocks.iter().rev() {
        let cls = stock.cls.bind(py);
        let table = cls.getattr(pyo3::intern!(py, "FUNCTION_TABLE"))?;
        match table.get_item(name) {
            Ok(_) => {
                return cls.call0()?.call_method1(
                    pyo3::intern!(py, "call_function"),
                    (name, args_list(ctx, resolved)?),
                );
            }
            Err(e) if e.is_instance_of::<PyKeyError>(py) => {}
            Err(e) => return Err(e),
        }
    }
    Err(unknown_function(py, name))
}
