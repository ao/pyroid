"""Helpers for the pyroid.jmespath tests: Hypothesis strategies and outcome encodings."""

from __future__ import annotations

import decimal
import re
from collections import OrderedDict
from typing import Any

from hypothesis import strategies as st

KEYS = ["a", "b", "c", "foo", "bar", "ü", "a b", "", "0", "true"]
FUNCTIONS = [
    "abs", "avg", "ceil", "contains", "ends_with", "floor", "join", "keys", "length", "map",
    "max", "max_by", "merge", "min", "min_by", "not_null", "reverse", "sort", "sort_by",
    "starts_with", "sum", "to_array", "to_string", "to_number", "type", "values", "nope",
]  # fmt: skip
# Calls with jmespath's arity, so they get past the arity check and reach the type checks.
CALLS = {
    "abs": "abs({0})",
    "avg": "avg({0})",
    "ceil": "ceil({0})",
    "floor": "floor({0})",
    "contains": "contains({0}, {1})",
    "ends_with": "ends_with({0}, {1})",
    "starts_with": "starts_with({0}, {1})",
    "join": "join({0}, {1})",
    "keys": "keys({0})",
    "values": "values({0})",
    "length": "length({0})",
    "map": "map(&{0}, {1})",
    "max": "max({0})",
    "min": "min({0})",
    "sum": "sum({0})",
    "sort": "sort({0})",
    "reverse": "reverse({0})",
    "merge": "merge({0}, {1})",
    "not_null": "not_null({0}, {1})",
    "to_array": "to_array({0})",
    "to_string": "to_string({0})",
    "to_number": "to_number({0})",
    "type": "type({0})",
    "sort_by": "sort_by({0}, &{1})",
    "max_by": "max_by({0}, &{1})",
    "min_by": "min_by({0}, &{1})",
}

identifiers = st.sampled_from(
    ["a", "b", "c", "foo", "bar", '"ü"', '"a b"', '""', '"0"', '"\\u00fc"', '"true"', '"\\ud800"']
)
literals = st.sampled_from(
    [
        "`1`", "`0`", "`-1`", "`1.5`", "`-0.0`", '`"x"`', '`""`', "`[1, 2]`", "`[]`", "`{}`",
        '`{"a": 1}`', "`null`", "`true`", "`false`", "'s'", "''", "'a'", "`foo`", "`1e400`",
        "`NaN`", "`[0, true]`", '`"2016"`', "`12345678901234567890123`", "`[3, 1, 2]`",
        '`["b", "a"]`', "`[1.5, 2]`", "' x'", "`\\u00e9`", '`"\\ud800"`', "'\\''",
    ]
)  # fmt: skip
numbers = st.sampled_from(["0", "1", "-1", "2", "-2", "3", "10", "99999999999999999999"])
COMPARATORS = ["==", "!=", "<", "<=", ">", ">="]


def _extend(children: Any) -> Any:
    return st.one_of(
        st.tuples(children, identifiers).map(lambda t: f"{t[0]}.{t[1]}"),
        st.tuples(children, numbers).map(lambda t: f"{t[0]}[{t[1]}]"),
        st.tuples(children, st.lists(st.one_of(st.none(), numbers), min_size=1, max_size=3)).map(
            lambda t: f"{t[0]}[{':'.join('' if p is None else p for p in t[1])}]"
        ),
        children.map(lambda c: f"{c}[*]"),
        children.map(lambda c: f"{c}[]"),
        children.map(lambda c: f"{c}.*"),
        st.tuples(children, children).map(lambda t: f"{t[0]}[?{t[1]}]"),
        st.tuples(children, children).map(lambda t: f"{t[0]} | {t[1]}"),
        st.tuples(children, children).map(lambda t: f"{t[0]} || {t[1]}"),
        st.tuples(children, children).map(lambda t: f"{t[0]} && {t[1]}"),
        children.map(lambda c: f"!{c}"),
        children.map(lambda c: f"({c})"),
        st.lists(children, min_size=1, max_size=3).map(lambda v: "[" + ", ".join(v) + "]"),
        st.lists(st.tuples(identifiers, children), min_size=1, max_size=3).map(
            lambda v: "{" + ", ".join(f"{k}: {x}" for k, x in v) + "}"
        ),
        st.tuples(
            st.sampled_from(FUNCTIONS),
            st.lists(st.one_of(children, children.map(lambda c: "&" + c)), max_size=3),
        ).map(lambda t: f"{t[0]}({', '.join(t[1])})"),
        st.tuples(st.sampled_from(sorted(CALLS)), children, children).map(
            lambda t: CALLS[t[0]].format(t[1], t[2])
        ),
        st.tuples(children, st.sampled_from(COMPARATORS), children).map(
            lambda t: f"{t[0]} {t[1]} {t[2]}"
        ),
        st.tuples(children, children).map(lambda t: f"{t[0]}.{t[1]}"),
    )


valid_expressions = st.recursive(
    st.one_of(identifiers, literals, st.just("@"), st.just("*"), st.just("[]")),
    _extend,
    max_leaves=12,
)


def _mutate(s: str) -> Any:
    return st.tuples(
        st.integers(0, len(s)),
        st.sampled_from(list(".[]{}()|&!<>=@*?:,'`\" -1a\\")),
        st.booleans(),
    ).map(lambda t: s[: t[0]] + t[1] + s[t[0] :] if t[2] else s[: t[0]] + s[t[0] + 1 :])


#: Valid expressions, mutated ones, and short random text in jmespath's alphabet.
expressions = st.one_of(
    valid_expressions,
    valid_expressions,
    valid_expressions.flatmap(_mutate),
    st.text(alphabet=".[]{}()|&!<>=@*?:,'`\" -01abü\\", min_size=1, max_size=12),
)


class ListSub(list):  # type: ignore[type-arg]
    """A list subclass (projections accept it; type checks by name reject it)."""


class DictSub(dict):  # type: ignore[type-arg]
    """A dict subclass (fields use its get; type checks by name reject it)."""


class Getter:
    """An object with a ``get`` method: jmespath's field lookup duck-types it."""

    def get(self, key: Any) -> Any:
        return ("got", key)


scalars = st.one_of(
    st.integers(-3, 3),
    st.integers(),
    st.sampled_from([2**64, -(2**70), 2**53 + 1]),
    st.floats(allow_nan=True, allow_infinity=True),
    st.sampled_from([0.0, -0.0, 1.0, float("nan"), 0.5]),
    st.booleans(),
    st.none(),
    st.text(max_size=4),
    st.text(st.characters(codec="utf-8") | st.characters(categories=["Cs"]), max_size=3),
    st.sampled_from(KEYS),
    st.sampled_from([decimal.Decimal("1"), decimal.Decimal("2.5")]),
)


def _containers(children: Any) -> Any:
    return st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.sampled_from(KEYS), children, max_size=4),
        st.fixed_dictionaries({k: children for k in ("a", "b", "foo")}),
        st.lists(st.fixed_dictionaries({"a": children, "b": children}), max_size=4),
        st.lists(children, max_size=3).map(tuple),
        st.dictionaries(st.sampled_from(KEYS), children, max_size=3).map(OrderedDict),
        st.lists(children, max_size=3).map(ListSub),
        st.dictionaries(st.sampled_from(KEYS), children, max_size=3).map(DictSub),
        st.just(Getter()),
    )


#: JSON-like data with the keys the expressions use, plus Python-only types.
data = st.recursive(scalars, _containers, max_leaves=24)

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def normalize(text: str) -> str:
    """Drop object addresses and the module of ``_Expression`` from a message."""
    return _ADDRESS.sub("0x?", text).replace(
        "pyroid.jmespath.visitor._Expression", "jmespath.visitor._Expression"
    )


def index(value: Any, ids: dict[int, Any] | None = None, path: tuple[Any, ...] = ()) -> Any:
    """``{id(obj): path}`` for the containers and objects inside ``value``."""
    ids = {} if ids is None else ids
    ids.setdefault(id(value), path)
    if isinstance(value, dict):
        for k, v in value.items():
            index(v, ids, (*path, k))
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            index(v, ids, (*path, i))
    return ids


def encode(x: Any, ids: dict[int, Any]) -> Any:
    """A comparable rendering: types, key order, float bits, and which input objects come back."""
    if id(x) in ids and not isinstance(x, int | float | str | bool | type(None)):
        return ("input", ids[id(x)])
    t = type(x).__name__
    if isinstance(x, float):
        return (t, "nan" if x != x else repr(x))
    if isinstance(x, dict):
        return (t, [(encode(k, ids), encode(v, ids)) for k, v in x.items()])
    if isinstance(x, list | tuple):
        return (t, [encode(v, ids) for v in x])
    return (t, normalize(repr(x)))


def error(e: BaseException, ids: dict[int, Any]) -> Any:
    """An exception's class, ``str()`` and attributes."""
    attrs = {k: encode(v, ids) for k, v in sorted(vars(e).items())} if hasattr(e, "__dict__") else {}
    return ("err", type(e).__name__, normalize(str(e)), encode(e.args, ids), attrs)


def outcome(search: Any, expression: Any, value: Any, options: Any, ids: dict[int, Any]) -> Any:
    """``search(expression, value, options)``'s result or exception, encoded."""
    try:
        return ("ok", encode(search(expression, value, options), ids))
    except RecursionError:
        return ("recursion",)
    except Exception as e:  # noqa: BLE001 - every exception is compared
        return error(e, ids)
