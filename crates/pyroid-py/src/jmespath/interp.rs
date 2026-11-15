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

use super::expr::Expr;
use super::functions::{self, unknown_function};
use super::{ExprRef, config};

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
    ticks: Cell<u32>,
}

impl<'py, 'a> Ctx<'py, 'a> {
    pub(crate) fn new(py: Python<'py>, opts: &'a Arc<Opts>, limit: usize) -> Self {
        Self {
            py,
            opts,
            limit,
            ticks: Cell::new(0),
        }
    }

    /// Fail as Python would if a frame at depth `frame` were opened.
    pub(crate) fn check(&self, frame: usize) -> PyResult<()> {
        if frame > self.limit {
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
    Expref(&'a Arc<Expr>),
}

impl<'py> Val<'py, '_> {
    pub(crate) fn to_py(&self, ctx: &Ctx<'py, '_>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Val::Py(o) => Ok(o.clone()),
            Val::Expref(e) => expref(ctx, e),
        }
    }
}

fn expref<'py>(ctx: &Ctx<'py, '_>, e: &Arc<Expr>) -> PyResult<Bound<'py, PyAny>> {
    Ok(Bound::new(ctx.py, ExprRef::new(Arc::clone(e), Arc::clone(ctx.opts)))?.into_any())
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
        // As list iteration: the length is read again at every step.
        let mut i = 0;
        while i < l.len() {
            ctx.tick()?;
            if !f(l.get_item(i)?)? {
                break;
            }
            i += 1;
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
        || x.is_exact_instance_of::<ExprRef>()
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

/// `_equals(x, y)`: `x == y`, except that 0 and 1 never equal `False` and `True`.
fn equals<'py>(x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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
    if x.is_exact_instance_of::<PyInt>()
        || x.is_exact_instance_of::<PyFloat>()
        || x.is_exact_instance_of::<ExprRef>()
    {
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
/// One arm per `visit_*` method.
#[allow(clippy::too_many_lines)]
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
            let mut result = value.clone();
            for c in children {
                result = visit(ctx, c, &result, inner)?;
            }
            Ok(result)
        }
        Expr::Field(key) => field(value, key.bind(py)),
        Expr::Comparator(op, name, l, r) => {
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
                    left.rich_compare(&right, op)
                }
            }
        }
        Expr::Current | Expr::Identity => Ok(value.clone()),
        Expr::Expref(e) => expref(ctx, e),
        Expr::Function {
            name,
            builtin,
            args,
        } => call_function(ctx, name.bind(py), *builtin, args, value, depth),
        Expr::FilterProjection(left, right, cond) => {
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
        Expr::Flatten(x) => {
            let base = visit(ctx, x, value, inner)?;
            if !is_list(&base) {
                // Can't flatten the object if it's not a list.
                return Ok(none(py));
            }
            let mut merged = Vec::new();
            for_each(ctx, &base, |el| {
                if let Ok(l) = el.cast_exact::<PyList>() {
                    merged.extend(l.iter());
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
        Expr::Index(index, fast) => {
            if !is_list(value) {
                return Ok(none(py));
            }
            if let (Ok(l), Some(i)) = (value.cast_exact::<PyList>(), fast) {
                let n = isize::try_from(l.len())?;
                let j = if *i < 0 { i + n } else { *i };
                return match usize::try_from(j) {
                    Ok(j) if j < l.len() => l.get_item(j),
                    _ => Ok(none(py)),
                };
            }
            match value.get_item(index.bind(py)) {
                Ok(v) => Ok(v),
                Err(e) if e.is_instance_of::<PyIndexError>(py) => Ok(none(py)),
                Err(e) => Err(e),
            }
        }
        Expr::KeyValPair(_, x) => visit(ctx, x, value, inner),
        Expr::Literal(v) => Ok(v.bind(py).clone()),
        Expr::MultiSelectDict(children) => {
            if value.is_none() {
                return Ok(none(py));
            }
            match &ctx.opts.dict_cls {
                None => {
                    let d = PyDict::new(py);
                    for (key, child) in children {
                        let v = visit(ctx, child, value, inner)?;
                        d.set_item(key.bind(py), v)?;
                    }
                    Ok(d.into_any())
                }
                Some(cls) => {
                    let d = cls.bind(py).call0()?;
                    for (key, child) in children {
                        let v = visit(ctx, child, value, inner)?;
                        d.set_item(key.bind(py), v)?;
                    }
                    Ok(d)
                }
            }
        }
        Expr::MultiSelectList(children) => {
            if value.is_none() {
                return Ok(none(py));
            }
            let mut collected = Vec::with_capacity(children.len());
            for c in children {
                collected.push(visit(ctx, c, value, inner)?);
            }
            Ok(PyList::new(py, collected)?.into_any())
        }
        Expr::Or(l, r) => {
            let matched = visit(ctx, l, value, inner)?;
            if is_false(&matched)? {
                return visit(ctx, r, value, inner);
            }
            Ok(matched)
        }
        Expr::And(l, r) => {
            let matched = visit(ctx, l, value, inner)?;
            if is_false(&matched)? {
                return Ok(matched);
            }
            visit(ctx, r, value, inner)
        }
        Expr::Not(x) => {
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
        Expr::Projection(l, r) => {
            let base = visit(ctx, l, value, inner)?;
            if !is_list(&base) {
                return Ok(none(py));
            }
            let mut collected = Vec::new();
            if matches!(**r, Expr::Identity) {
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
        Expr::Slice(_, s) => {
            if !is_list(value) {
                return Ok(none(py));
            }
            value.get_item(s.bind(py))
        }
        Expr::ValueProjection(l, r) => {
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
        Expr::Unknown(t) => Err(PyNotImplementedError::new_err(t.clone_ref(py))),
    }
}

/// `visit_function_expression`: the arguments, then `call_function(name, args)`.
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
            None => Err(unknown_function(py, name)),
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
            let list = PyList::new(
                py,
                resolved
                    .iter()
                    .map(|v| v.to_py(ctx))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
            obj.call_method1(pyo3::intern!(py, "call_function"), (name, list))
        }
    }
}
