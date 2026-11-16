// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! jmespath's built-in functions (`jmespath/functions.py`), with `Functions`'s argument
//! checks: arity, then types by `type(x).__name__` (so subclasses of `list` or `dict` are
//! rejected, `OrderedDict` is an object), as jmespath does them.
//!
//! Numbers go through CPython's own `sum`, `abs`, `math.ceil`, `math.floor`, `max`, `min`
//! and `/`, so results (int or float, compensated float sums on 3.12+) are the interpreter's.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::sync::Arc;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

use super::config;
use super::interp::{self, Ctx, Val, is_list};

macro_rules! builtins {
    ($($variant:ident = $name:literal: [$($arg:expr),*] $(variadic $v:literal)?;)*) => {
        /// A built-in function.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub(crate) enum Builtin { $($variant),* }

        impl Builtin {
            pub(crate) fn from_name(name: &str) -> Option<Self> {
                match name { $($name => Some(Self::$variant),)* _ => None }
            }

            pub(crate) fn name(self) -> &'static str {
                match self { $(Self::$variant => $name),* }
            }

            /// Each argument's `types` list, and whether the last is variadic.
            fn signature(self) -> (&'static [&'static [&'static str]], bool) {
                match self {
                    $(Self::$variant => (&[$($arg),*], false $(|| $v)?)),*
                }
            }

            pub(crate) const ALL: &'static [Builtin] = &[$(Self::$variant),*];
        }
    };
}

const ANY: &[&str] = &[];

builtins! {
    Abs = "abs": [&["number"]];
    Avg = "avg": [&["array-number"]];
    NotNull = "not_null": [ANY] variadic true;
    ToArray = "to_array": [ANY];
    ToString = "to_string": [ANY];
    ToNumber = "to_number": [ANY];
    Contains = "contains": [&["array", "string"], ANY];
    Length = "length": [&["string", "array", "object"]];
    EndsWith = "ends_with": [&["string"], &["string"]];
    StartsWith = "starts_with": [&["string"], &["string"]];
    Reverse = "reverse": [&["array", "string"]];
    Ceil = "ceil": [&["number"]];
    Floor = "floor": [&["number"]];
    Join = "join": [&["string"], &["array-string"]];
    Map = "map": [&["expref"], &["array"]];
    Max = "max": [&["array-number", "array-string"]];
    Merge = "merge": [&["object"]] variadic true;
    Min = "min": [&["array-number", "array-string"]];
    Sort = "sort": [&["array-string", "array-number"]];
    Sum = "sum": [&["array-number"]];
    Keys = "keys": [&["object"]];
    Values = "values": [&["object"]];
    Type = "type": [ANY];
    SortBy = "sort_by": [&["array"], &["expref"]];
    MinBy = "min_by": [&["array"], &["expref"]];
    MaxBy = "max_by": [&["array"], &["expref"]];
}

/// `REVERSE_TYPES_MAP[t]`.
fn reverse_type(t: &str) -> &'static [&'static str] {
    match t {
        "boolean" => &["bool"],
        "array" => &["list", "_Projection"],
        "object" => &["dict", "OrderedDict"],
        "null" => &["NoneType"],
        "string" => &["unicode", "str"],
        "number" => &["float", "int", "long"],
        "expref" => &["_Expression"],
        _ => &[],
    }
}

/// `TYPES_MAP.get(name, 'unknown')`.
pub(crate) fn jmespath_type(name: &str) -> &'static str {
    match name {
        "bool" => "boolean",
        "list" | "_Projection" => "array",
        "dict" | "OrderedDict" => "object",
        "NoneType" => "null",
        "unicode" | "str" => "string",
        "float" | "int" | "long" => "number",
        "_Expression" => "expref",
        _ => "unknown",
    }
}

/// `type(x).__name__`, without a lookup for the builtin types.
pub(crate) fn type_name(v: &Val<'_, '_>) -> PyResult<Cow<'static, str>> {
    match v {
        Val::Expref(_) => Ok(Cow::Borrowed("_Expression")),
        Val::Py(o) => py_type_name(o),
    }
}

/// [`type_name`] of a Python value.
fn py_type_name(o: &Bound<'_, PyAny>) -> PyResult<Cow<'static, str>> {
    Ok(Cow::Borrowed(if o.is_none() {
        "NoneType"
    } else if o.is_instance_of::<PyBool>() {
        "bool"
    } else if o.is_exact_instance_of::<PyString>() {
        "str"
    } else if o.is_exact_instance_of::<PyDict>() {
        "dict"
    } else if o.is_exact_instance_of::<PyList>() {
        "list"
    } else if o.is_exact_instance_of::<PyInt>() {
        "int"
    } else if o.is_exact_instance_of::<PyFloat>() {
        "float"
    } else {
        return Ok(Cow::Owned(o.get_type().name()?.to_cow()?.into_owned()));
    }))
}

fn py_list<'py>(py: Python<'py>, items: &[&str]) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, items)
}

/// `JMESPathTypeError(function_name, current_value, actual_type, expected_types)`.
fn type_error<'py>(
    ctx: &Ctx<'py, '_>,
    name: &Bound<'py, PyAny>,
    current: &Bound<'py, PyAny>,
    actual: &str,
    expected: Bound<'py, PyAny>,
) -> PyErr {
    let py = ctx.py;
    match config(py).and_then(|c| {
        c.type_error
            .bind(py)
            .call1((name, current, actual, expected))
    }) {
        Ok(e) => PyErr::from_value(e),
        Err(e) => e,
    }
}

/// `Functions._validate_arguments`.
fn validate<'py>(
    ctx: &Ctx<'py, '_>,
    f: Builtin,
    name: &Bound<'py, PyAny>,
    args: &[Val<'py, '_>],
    depth: usize,
) -> PyResult<()> {
    let py = ctx.py;
    let c = config(py)?;
    let (sig, variadic) = f.signature();
    if variadic {
        if args.len() < sig.len() {
            let e = c
                .variadic_arity_error
                .bind(py)
                .call1((sig.len(), args.len(), name))?;
            return Err(PyErr::from_value(e));
        }
    } else if args.len() != sig.len() {
        let e = c
            .arity_error
            .bind(py)
            .call1((sig.len(), args.len(), name))?;
        return Err(PyErr::from_value(e));
    }
    // _type_check -> _type_check_single -> _subtype_check
    ctx.check(depth + 3)?;
    for (types, arg) in sig.iter().zip(args) {
        if !types.is_empty() {
            type_check_single(ctx, arg, types, name)?;
        }
    }
    Ok(())
}

fn type_check_single<'py>(
    ctx: &Ctx<'py, '_>,
    current: &Val<'py, '_>,
    types: &[&str],
    name: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let py = ctx.py;
    // _get_allowed_pytypes, without building the lists: `array-number` allows the array
    // types, with `number` as a subtype.
    let base = |t: &&str| reverse_type(t.split_once('-').map_or(*t, |(b, _)| b));
    let subtypes = || {
        types
            .iter()
            .filter_map(|t| t.split_once('-').map(|(_, sub)| reverse_type(sub)))
    };
    let actual = type_name(current)?;
    if !types.iter().any(|t| base(t).contains(&actual.as_ref())) {
        let cur = current.to_py(ctx)?;
        return Err(type_error(
            ctx,
            name,
            &cur,
            jmespath_type(&actual),
            py_list(py, types)?.into_any(),
        ));
    }
    let n_subtypes = subtypes().count();
    if n_subtypes == 0 {
        return Ok(());
    }
    // Arrays are the only types with subtypes.
    let Val::Py(current) = current else {
        return Ok(());
    };
    let raw_error = |el: &Bound<'py, PyAny>, actual: &str| -> PyResult<PyErr> {
        Ok(type_error(
            ctx,
            name,
            el,
            actual,
            py_list(py, types)?.into_any(),
        ))
    };
    if n_subtypes == 1 {
        let allowed = subtypes().next().unwrap_or_default();
        let mut err = None;
        interp::for_each(ctx, current, |el| {
            let actual = py_type_name(&el)?;
            if allowed.contains(&actual.as_ref()) {
                Ok(true)
            } else {
                err = Some(raw_error(&el, &actual)?);
                Ok(false)
            }
        })?;
        return err.map_or(Ok(()), Err);
    }
    if n_subtypes > 1 && current.is_truthy()? {
        // The first element's type picks the subtype the others must have.
        let first_el = current.get_item(0)?;
        let first = type_name(&Val::Py(first_el.clone()))?;
        let Some(allowed) = subtypes().find(|s| s.contains(&first.as_ref())) else {
            return Err(raw_error(&first_el, &first)?);
        };
        let mut err = None;
        interp::for_each(ctx, current, |el| {
            let actual = py_type_name(&el)?;
            if allowed.contains(&actual.as_ref()) {
                Ok(true)
            } else {
                err = Some(raw_error(&el, &actual)?);
                Ok(false)
            }
        })?;
        return err.map_or(Ok(()), Err);
    }
    Ok(())
}

/// `UnknownFunctionError("Unknown function: %s()" % name)`, raised while handling the
/// `KeyError` (as jmespath's `call_function` does).
pub(crate) fn unknown_function<'py>(py: Python<'py>, name: &Bound<'py, PyAny>) -> PyErr {
    let r = (|| -> PyResult<PyErr> {
        let msg = PyString::new(py, "Unknown function: %s()").call_method1("__mod__", (name,))?;
        let e = config(py)?.unknown_function_error.bind(py).call1((msg,))?;
        let key_error = PyKeyError::new_err(name.clone().unbind()).into_value(py);
        e.setattr("__context__", key_error)?;
        Ok(PyErr::from_value(e))
    })();
    r.unwrap_or_else(|e| e)
}

/// Validate and run a built-in (`call_function` with a stock `FUNCTION_TABLE` entry): one
/// arm per `_func_*` method.
/// `depth` is the frame of the function node's `visit_function_expression`.
#[allow(clippy::too_many_lines)]
pub(crate) fn call<'py>(
    ctx: &Ctx<'py, '_>,
    f: Builtin,
    name: &Bound<'py, PyAny>,
    args: &[Val<'py, '_>],
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    // call_function -> _validate_arguments
    ctx.check(depth + 2)?;
    validate(ctx, f, name, args, depth + 2)?;
    let py = ctx.py;
    let c = config(py)?;
    let arg = |i: usize| args[i].to_py(ctx);
    let none = || Ok(py.None().into_bound(py));
    match f {
        Builtin::Abs => arg(0)?.abs(),
        Builtin::Avg => {
            let a = arg(0)?;
            if a.is_truthy()? {
                let total = c.sum.bind(py).call1((&a,))?;
                total.div(a.len()?)
            } else {
                none()
            }
        }
        Builtin::NotNull => {
            for a in args {
                let v = a.to_py(ctx)?;
                if !v.is_none() {
                    return Ok(v);
                }
            }
            none()
        }
        Builtin::ToArray => {
            let a = arg(0)?;
            if is_list(&a) {
                Ok(a)
            } else {
                Ok(PyList::new(py, [a])?.into_any())
            }
        }
        Builtin::ToString => {
            let a = arg(0)?;
            if a.is_instance_of::<PyString>() {
                Ok(a)
            } else {
                c.dumps.bind(py).call((a,), Some(c.dumps_kwargs.bind(py)))
            }
        }
        Builtin::ToNumber => to_number(ctx, arg(0)?),
        Builtin::Contains => {
            let subject = arg(0)?;
            let search = arg(1)?;
            Ok(PyBool::new(py, subject.contains(search)?)
                .to_owned()
                .into_any())
        }
        Builtin::Length => Ok(arg(0)?.len()?.into_pyobject(py)?.into_any()),
        Builtin::EndsWith => arg(0)?.call_method1(pyo3::intern!(py, "endswith"), (arg(1)?,)),
        Builtin::StartsWith => arg(0)?.call_method1(pyo3::intern!(py, "startswith"), (arg(1)?,)),
        Builtin::Reverse => {
            let a = arg(0)?;
            if a.is_instance_of::<PyString>() {
                a.get_item(c.reverse_slice.bind(py))
            } else if let Ok(l) = a.cast_exact::<PyList>() {
                let mut items: Vec<Bound<'py, PyAny>> = l.iter().collect();
                items.reverse();
                Ok(PyList::new(py, items)?.into_any())
            } else {
                let r = c.reversed.bind(py).call1((&a,))?;
                Ok(py.get_type::<PyList>().call1((r,))?)
            }
        }
        Builtin::Ceil => c.ceil.bind(py).call1((arg(0)?,)),
        Builtin::Floor => c.floor.bind(py).call1((arg(0)?,)),
        Builtin::Join => arg(0)?.call_method1(pyo3::intern!(py, "join"), (arg(1)?,)),
        Builtin::Map => {
            // _func_map -> expref.visit -> visit
            let a = arg(1)?;
            let mut out = Vec::new();
            match &args[0] {
                Val::Expref(e) => {
                    let e = Arc::clone(e);
                    interp::for_each(ctx, &a, |el| {
                        out.push(interp::visit(ctx, &e.expr, &el, depth + 3)?);
                        Ok(true)
                    })?;
                }
                Val::Py(x) => {
                    let expression = x.getattr("expression")?;
                    interp::for_each(ctx, &a, |el| {
                        out.push(x.call_method1("visit", (&expression, el))?);
                        Ok(true)
                    })?;
                }
            }
            Ok(PyList::new(py, out)?.into_any())
        }
        Builtin::Max | Builtin::Min => {
            let a = arg(0)?;
            if a.is_truthy()? {
                let f = if f == Builtin::Max { &c.max } else { &c.min };
                f.bind(py).call1((a,))
            } else {
                none()
            }
        }
        Builtin::Merge => {
            let merged = PyDict::new(py);
            for a in args {
                let a = a.to_py(ctx)?;
                if let Ok(d) = a.cast_exact::<PyDict>() {
                    merged.update(d.as_mapping())?;
                } else {
                    merged.call_method1(pyo3::intern!(py, "update"), (a,))?;
                }
            }
            Ok(merged.into_any())
        }
        Builtin::Sort => sort(ctx, &arg(0)?),
        Builtin::Sum => c.sum.bind(py).call1((arg(0)?,)),
        Builtin::Keys => {
            let a = arg(0)?;
            if let Ok(d) = a.cast_exact::<PyDict>() {
                Ok(d.keys().into_any())
            } else {
                let k = a.call_method0(pyo3::intern!(py, "keys"))?;
                Ok(py.get_type::<PyList>().call1((k,))?)
            }
        }
        Builtin::Values => {
            let a = arg(0)?;
            if let Ok(d) = a.cast_exact::<PyDict>() {
                Ok(d.values().into_any())
            } else {
                let v = a.call_method0(pyo3::intern!(py, "values"))?;
                Ok(py.get_type::<PyList>().call1((v,))?)
            }
        }
        Builtin::Type => {
            let a = arg(0)?;
            let t = if a.is_instance_of::<PyString>() {
                "string"
            } else if a.is_instance_of::<PyBool>() {
                "boolean"
            } else if a.is_instance_of::<PyList>() {
                "array"
            } else if a.is_instance_of::<PyDict>() {
                "object"
            } else if a.is_instance_of::<PyFloat>() || a.is_instance_of::<PyInt>() {
                "number"
            } else if a.is_none() {
                "null"
            } else {
                return none();
            };
            Ok(PyString::new(py, t).into_any())
        }
        Builtin::SortBy => sort_by(ctx, &args[0], &args[1], depth),
        Builtin::MinBy | Builtin::MaxBy => min_max_by(ctx, f, &args[0], &args[1], depth),
    }
}

fn to_number<'py>(ctx: &Ctx<'py, '_>, a: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    if a.is_instance_of::<PyList>() || a.is_instance_of::<PyDict>() || a.is_instance_of::<PyBool>()
    {
        return Ok(py.None().into_bound(py));
    }
    if a.is_none() {
        return Ok(a);
    }
    if a.is_instance_of::<PyInt>() || a.is_instance_of::<PyFloat>() {
        return Ok(a);
    }
    match py.get_type::<PyInt>().call1((&a,)) {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
            match py.get_type::<PyFloat>().call1((&a,)) {
                Ok(v) => Ok(v),
                Err(e) if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {
                    Ok(py.None().into_bound(py))
                }
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(e),
    }
}

/// A sort key the native sort can compare: Python's order on exact `int`, `float` and `str`.
enum Key<'py> {
    Int(i64),
    Float(f64),
    Str(&'py str),
}

impl Key<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Key::Int(a), Key::Int(b)) => a.cmp(b),
            (Key::Float(a), Key::Float(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Key::Int(a), Key::Float(b)) => cmp_int_float(*a, *b),
            (Key::Float(a), Key::Int(b)) => cmp_int_float(*b, *a).reverse(),
            (Key::Str(a), Key::Str(b)) => a.cmp(b),
            // Never mixed: `native_keys` gives all strings or all numbers.
            (Key::Str(_), _) => Ordering::Greater,
            (_, Key::Str(_)) => Ordering::Less,
        }
    }
}

/// Python's exact comparison of an `int` with a (non-NaN) `float`.
pub(crate) fn cmp_int_float(i: i64, f: f64) -> Ordering {
    const TWO_63: f64 = 9_223_372_036_854_775_808.0;
    if f >= TWO_63 {
        return Ordering::Less;
    }
    if f < -TWO_63 {
        return Ordering::Greater;
    }
    let t = f.trunc();
    // |t| < 2^63, so the conversion is exact.
    #[allow(clippy::cast_possible_truncation)]
    let ti = t as i64;
    match i.cmp(&ti) {
        Ordering::Equal => {
            let frac = f - t;
            if frac > 0.0 {
                Ordering::Less
            } else if frac < 0.0 {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
        o => o,
    }
}

/// Keys the native sort handles exactly (a total order, as `sorted` sees it), or `None`.
fn native_keys<'a>(keys: &'a [Bound<'_, PyAny>]) -> Option<Vec<Key<'a>>> {
    let mut out = Vec::with_capacity(keys.len());
    let mut strings = 0usize;
    for k in keys {
        if let Ok(s) = k.cast_exact::<PyString>() {
            out.push(Key::Str(s.to_str().ok()?));
            strings += 1;
        } else if k.is_exact_instance_of::<PyInt>() {
            out.push(Key::Int(k.extract::<i64>().ok()?));
        } else if let Ok(f) = k.cast_exact::<PyFloat>() {
            let v = f.value();
            if v.is_nan() {
                return None;
            }
            out.push(Key::Float(v));
        } else {
            return None;
        }
    }
    if strings != 0 && strings != keys.len() {
        return None;
    }
    Some(out)
}

/// The elements of `items` in the order of `sorted(range(n), key=keys.__getitem__)`.
fn sorted_by_keys<'py>(
    ctx: &Ctx<'py, '_>,
    items: Vec<Bound<'py, PyAny>>,
    keys: &[Bound<'py, PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    // A stable sort on a total order gives the same order as CPython's Timsort.
    let order: Vec<usize> = if let Some(native) = native_keys(keys) {
        if native.iter().all(|k| matches!(k, Key::Int(_))) {
            let mut pairs: Vec<(i64, usize)> = native
                .iter()
                .enumerate()
                .map(|(i, k)| (if let Key::Int(v) = k { *v } else { 0 }, i))
                .collect();
            pairs.sort_by_key(|p| p.0);
            pairs.into_iter().map(|p| p.1).collect()
        } else if native.iter().all(|k| matches!(k, Key::Str(_))) {
            let mut pairs: Vec<(&str, usize)> = native
                .iter()
                .enumerate()
                .map(|(i, k)| (if let Key::Str(v) = k { *v } else { "" }, i))
                .collect();
            pairs.sort_by(|a, b| a.0.cmp(b.0));
            pairs.into_iter().map(|p| p.1).collect()
        } else {
            let mut idx: Vec<usize> = (0..items.len()).collect();
            idx.sort_by(|&a, &b| native[a].cmp(&native[b]));
            idx
        }
    } else {
        let c = config(py)?;
        let key_list = PyList::new(py, keys)?;
        let getitem = key_list.getattr(pyo3::intern!(py, "__getitem__"))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("key", getitem)?;
        let r = c.sorted.bind(py).call(
            (pyo3::types::PyRange::new(py, 0, items.len().try_into()?)?,),
            Some(&kwargs),
        )?;
        r.try_iter()?
            .map(|i| i?.extract::<usize>())
            .collect::<PyResult<Vec<_>>>()?
    };
    let mut slots: Vec<Option<Bound<'py, PyAny>>> = items.into_iter().map(Some).collect();
    let out: Vec<Bound<'py, PyAny>> = order.into_iter().filter_map(|i| slots[i].take()).collect();
    Ok(PyList::new(py, out)?.into_any())
}

fn sort<'py>(ctx: &Ctx<'py, '_>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let mut items = Vec::new();
    if let Ok(l) = a.cast_exact::<PyList>() {
        items.extend(l.iter());
    } else {
        // `list(sorted(arg))` on something that is not a list: Python's own sort.
        let r = config(py)?.sorted.bind(py).call1((a,))?;
        return py.get_type::<PyList>().call1((r,));
    }
    let keys = items.clone();
    sorted_by_keys(ctx, items, &keys)
}

/// The expref's value for `x` (`expref.visit(expref.expression, x)`).
fn apply<'py>(
    ctx: &Ctx<'py, '_>,
    expref: &Val<'py, '_>,
    x: &Bound<'py, PyAny>,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    match expref {
        Val::Expref(e) => interp::visit(ctx, &e.expr, x, depth),
        Val::Py(o) => {
            let expression = o.getattr("expression")?;
            o.call_method1("visit", (expression, x))
        }
    }
}

fn items_of<'py>(ctx: &Ctx<'py, '_>, a: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut items = Vec::new();
    interp::for_each(ctx, a, |el| {
        items.push(el);
        Ok(true)
    })?;
    Ok(items)
}

fn sort_by<'py>(
    ctx: &Ctx<'py, '_>,
    array: &Val<'py, '_>,
    expref: &Val<'py, '_>,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let array = array.to_py(ctx)?;
    if !array.is_truthy()? {
        return Ok(array);
    }
    // _func_sort_by -> expref.visit -> visit: the first element's type decides.
    let first = array.get_item(0)?;
    let required = jmespath_type(&type_name(&Val::Py(apply(
        ctx,
        expref,
        &first,
        depth + 3,
    )?))?);
    if required != "number" && required != "string" {
        return Err(type_error(
            ctx,
            &PyString::new(py, "sort_by").into_any(),
            &first,
            required,
            py_list(py, &["string", "number"])?.into_any(),
        ));
    }
    // sorted() lists the array, then computes every key (keyfunc -> expref.visit -> visit).
    let items = if let Ok(l) = array.cast_exact::<PyList>() {
        l.iter().collect()
    } else {
        items_of(ctx, &array)?
    };
    let mut keys = Vec::with_capacity(items.len());
    for (n, x) in items.iter().enumerate() {
        if n % 4096 == 4095 {
            py.check_signals()?;
        }
        let k = apply(ctx, expref, x, depth + 4)?;
        let t = jmespath_type(&type_name(&Val::Py(k.clone()))?);
        if t != required {
            return Err(type_error(
                ctx,
                &PyString::new(py, "sort_by").into_any(),
                &k,
                t,
                py_list(py, &[required])?.into_any(),
            ));
        }
        keys.push(k);
    }
    sorted_by_keys(ctx, items, &keys)
}

/// `a < b` (or `a > b`) as `min`/`max` compare keys.
fn key_less<'py>(a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>, max: bool) -> PyResult<bool> {
    if let (Some(x), Some(y)) = (native_key(a), native_key(b)) {
        // A str against a number goes to Python, which raises the TypeError.
        if matches!(x, Key::Str(_)) == matches!(y, Key::Str(_)) {
            let o = x.cmp(&y);
            return Ok(if max {
                o == Ordering::Greater
            } else {
                o == Ordering::Less
            });
        }
    }
    let op = if max {
        pyo3::basic::CompareOp::Gt
    } else {
        pyo3::basic::CompareOp::Lt
    };
    a.rich_compare(b, op)?.is_truthy()
}

fn native_key<'a>(k: &'a Bound<'_, PyAny>) -> Option<Key<'a>> {
    if let Ok(s) = k.cast_exact::<PyString>() {
        return s.to_str().ok().map(Key::Str);
    }
    if k.is_exact_instance_of::<PyInt>() {
        return k.extract::<i64>().ok().map(Key::Int);
    }
    if let Ok(f) = k.cast_exact::<PyFloat>() {
        let v = f.value();
        return (!v.is_nan()).then_some(Key::Float(v));
    }
    None
}

fn min_max_by<'py>(
    ctx: &Ctx<'py, '_>,
    f: Builtin,
    array: &Val<'py, '_>,
    expref: &Val<'py, '_>,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let py = ctx.py;
    let fname = PyString::new(py, f.name()).into_any();
    let array = array.to_py(ctx)?;
    if !array.is_truthy()? {
        return Ok(py.None().into_bound(py));
    }
    let max = f == Builtin::MaxBy;
    let mut best: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = None;
    let mut n = 0usize;
    interp::for_each(ctx, &array, |x| {
        n += 1;
        if n % 4096 == 0 {
            py.check_signals()?;
        }
        // _func_min_by -> min -> keyfunc -> expref.visit -> visit
        let k = apply(ctx, expref, &x, depth + 4)?;
        let t = jmespath_type(&type_name(&Val::Py(k.clone()))?);
        if t != "number" && t != "string" {
            return Err(type_error(
                ctx,
                &fname,
                &k,
                t,
                py_list(py, &["number", "string"])?.into_any(),
            ));
        }
        match &best {
            None => best = Some((x, k)),
            Some((_, bk)) => {
                if key_less(&k, bk, max)? {
                    best = Some((x, k));
                }
            }
        }
        Ok(true)
    })?;
    Ok(best.map_or_else(|| py.None().into_bound(py), |(x, _)| x))
}
