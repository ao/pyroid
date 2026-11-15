"""pyroid.jmespath against jmespath 1.1.0.

Every comparison covers the result (types, key order, float bits, and which of the input's own
objects come back) or the exception (class, ``str()``, ``args`` and attributes), and the
compiled AST (``.parsed``) when the expression parses. Inputs: the JMESPath compliance suite and
jmespath.py's legacy cases, jmespath.py's own unit tests run against pyroid, and Hypothesis
expressions from the grammar (valid, mutated and random) over data with Python-only types.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import sys
import threading
import types
import unittest
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _helpers() -> Any:
    """python/tests/_jmespath_util.py (CI imports tests with --import-mode=importlib)."""
    name = "_jmespath_util"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("_jmespath_util.py")
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_u = _helpers()

jmespath: Any = pytest.importorskip("jmespath")
if importlib.metadata.version("jmespath") != "1.1.0":
    pytest.skip("needs jmespath 1.1.0", allow_module_level=True)
importlib.import_module("jmespath.functions")

import pyroid.jmespath as pj  # noqa: E402
import pyroid.jmespath.functions as pjf  # noqa: E402

# Both emit it for the deprecated `foo` literal form; test_jmespath.py checks the warning.
pytestmark = pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")

DATA = Path(__file__).resolve().parents[2] / "parity" / "data" / "jmespath-1.1.0"
CASE_FILES = sorted(DATA.glob("compliance/*.json")) + sorted(DATA.glob("legacy/*.json"))


def compliance_cases() -> list[Any]:
    cases = []
    for path in CASE_FILES:
        suites = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=OrderedDict)
        for n, suite in enumerate(suites):
            for m, case in enumerate(suite["cases"]):
                cases.append(
                    pytest.param(suite["given"], case, id=f"{path.parent.name}/{path.stem}-{n}-{m}")
                )
    return cases


CASES = compliance_cases()


def compare(expression: Any, value: Any, jopts: Any = None, popts: Any = None) -> None:
    ids = _u.index(value)
    expected = _u.outcome(jmespath.search, expression, value, jopts, ids)
    actual = _u.outcome(pj.search, expression, value, popts, ids)
    assert actual == expected, expression
    try:
        parsed = jmespath.compile(expression).parsed
    except Exception:
        return
    assert repr(pj.compile(expression).parsed) == repr(parsed)
    assert pj.compile(expression).parsed == parsed


@pytest.mark.skipif(not CASES, reason="needs parity/data/jmespath-1.1.0")
@pytest.mark.parametrize(("given", "case"), CASES)
def test_compliance_suite(given: Any, case: dict[str, Any]) -> None:
    expression = case["expression"]
    compare(expression, given)
    compare(
        expression,
        given,
        jmespath.Options(dict_cls=OrderedDict),
        pj.Options(dict_cls=OrderedDict),
    )
    if "result" in case:
        # As jmespath.py's own runner checks it.
        result = pj.compile(expression).search(given, options=pj.Options(dict_cls=OrderedDict))
        assert result == case["result"]
    elif "error" in case:
        with pytest.raises(ValueError):  # noqa: PT011 - the runner's rule: any ValueError
            pj.compile(expression).search(given)


def test_compliance_suite_is_complete() -> None:
    if not CASES:
        pytest.skip("needs parity/data/jmespath-1.1.0")
    # 16 compliance files and the legacy literals, as jmespath.py 1.1.0 ships them.
    assert len(CASE_FILES) == 17
    assert len(CASES) == 917


SCALARS = [
    0, 1, -1, 2**53, 2**53 + 1, 2**64, -(2**63), 1.0, 0.0, -0.0, 1.5, 2.0**53, float("nan"),
    float("inf"), -float("inf"), True, False, None, "", "a", "b", "é", "\ud800", "\U0001f600",
]  # fmt: skip


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_comparisons(op: str) -> None:
    for a in SCALARS:
        for b in SCALARS:
            compare(f"a {op} b", {"a": a, "b": b})
            compare(f"[?a {op} b] | length(@)", [{"a": a, "b": b}])
            compare(f"!(a {op} b)", {"a": a, "b": b})


# ---------------------------------------------------------------------------
# Real queries: botocore's paginator and waiter expressions and AWS CLI examples
# ---------------------------------------------------------------------------


def _decode(x: Any) -> Any:
    """Undo bench/jmespath/real_queries.py's JSON encoding of datetimes and bytes."""
    import datetime

    if isinstance(x, dict):
        if set(x) == {"$datetime"}:
            return datetime.datetime.fromisoformat(x["$datetime"])
        if set(x) == {"$bytes"}:
            return x["$bytes"].encode()
        return {k: _decode(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v) for v in x]
    return x


REAL = DATA / "real-queries.json"
REAL_CASES = json.loads(REAL.read_text(encoding="utf-8")) if REAL.is_file() else []


@pytest.mark.skipif(not REAL_CASES, reason="needs parity/data/jmespath-1.1.0/real-queries.json")
def test_real_queries() -> None:
    # A sample of bench/jmespath/real_queries.py's corpus (the whole corpus is in the report).
    assert len(REAL_CASES) > 300
    for case in REAL_CASES:
        compare(case["expression"], _decode(case["data"]))


# ---------------------------------------------------------------------------
# jmespath.py's own unit tests, against pyroid.jmespath
# ---------------------------------------------------------------------------

UPSTREAM = ["test_lexer", "test_parser", "test_functions", "test_search", "test_custom_functions"]


def _load_upstream(name: str) -> types.ModuleType:
    """Import one of jmespath.py's test modules with ``jmespath`` meaning ``pyroid.jmespath``."""
    aliases = {
        "jmespath": pj,
        "jmespath.ast": pj.ast,
        "jmespath.exceptions": pj.exceptions,
        "jmespath.functions": pj.functions,
        "jmespath.lexer": pj.lexer,
        "jmespath.parser": pj.parser,
        "jmespath.visitor": pj.visitor,
    }
    saved = {k: sys.modules.get(k) for k in [*aliases, "tests"]}
    try:
        sys.modules.update(aliases)
        package = types.ModuleType("tests")
        package.__path__ = [str(DATA / "tests")]
        exec(
            compile((DATA / "tests" / "__init__.py").read_text(), "tests/__init__.py", "exec"),
            package.__dict__,
        )
        sys.modules["tests"] = package
        module = types.ModuleType(f"tests.{name}")
        path = DATA / "tests" / f"{name}.py"
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
        return module
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.mark.thread_unsafe(reason="swaps sys.modules while importing")
@pytest.mark.skipif(not (DATA / "tests").is_dir(), reason="needs parity/data/jmespath-1.1.0")
@pytest.mark.parametrize("name", UPSTREAM)
def test_upstream_unit_tests(name: str) -> None:
    module = _load_upstream(name)
    suite = unittest.TestLoader().loadTestsFromModule(module)
    result = unittest.TestResult()
    suite.run(result)
    problems = [f"{t}: {tb}" for t, tb in result.errors + result.failures]
    assert not problems, "\n".join(problems)
    assert result.testsRun > 0


# ---------------------------------------------------------------------------
# Hypothesis
# ---------------------------------------------------------------------------


@settings(suppress_health_check=list(HealthCheck), deadline=None)
@given(_u.expressions, _u.data, st.booleans())
def test_fuzz_against_jmespath(expression: str, value: Any, ordered: bool) -> None:
    if ordered:
        compare(
            expression,
            value,
            jmespath.Options(dict_cls=OrderedDict),
            pj.Options(dict_cls=OrderedDict),
        )
    else:
        compare(expression, value)


Decorator = Callable[[Callable[..., Any]], Callable[..., Any]]


def _custom(base: Any, signature: Callable[..., Decorator]) -> tuple[Any, Any]:
    class Plus(base):  # type: ignore[misc]
        @signature({"types": ["number"]}, {"types": ["number"]})
        def _func_add(self, x: Any, y: Any) -> Any:
            return x + y

        @signature({"types": ["array"]}, {"types": ["expref"]})
        def _func_my_map(self, array: Any, expref: Any) -> Any:
            return [expref.visit(expref.expression, x) for x in array]

        @signature({"types": ["string", "array", "object", "null"]})
        def _func_length(self, s: Any) -> Any:  # replaces a built-in
            return -1 if s is None else len(s)

    class Dispatch(Plus):
        def call_function(self, function_name: Any, resolved_args: Any) -> Any:
            if function_name == "abs":
                return "abs!"
            return super().call_function(function_name, resolved_args)

    return Plus, Dispatch


J_PLUS, J_DISPATCH = _custom(jmespath.functions.Functions, jmespath.functions.signature)
P_PLUS, P_DISPATCH = _custom(pjf.Functions, pjf.signature)
CUSTOM_CALLS = st.sampled_from(["add({0}, {1})", "my_map({0}, &{1})", "length({0})", "abs({0})"])
custom_expressions = st.one_of(
    _u.expressions,
    st.tuples(CUSTOM_CALLS, _u.valid_expressions, _u.valid_expressions).map(
        lambda t: t[0].format(t[1], t[2])
    ),
)


@settings(suppress_health_check=list(HealthCheck), deadline=None)
@given(
    custom_expressions,
    _u.data,
    st.sampled_from(["plus", "dispatch", "plain"]),
    st.booleans(),
)
def test_fuzz_custom_functions(expression: str, value: Any, kind: str, jmespath_base: bool) -> None:
    jcls = {"plus": J_PLUS, "dispatch": J_DISPATCH, "plain": jmespath.functions.Functions}[kind]
    pool = (
        {"plus": J_PLUS, "dispatch": J_DISPATCH, "plain": jmespath.functions.Functions}
        if jmespath_base
        else {"plus": P_PLUS, "dispatch": P_DISPATCH, "plain": pjf.Functions}
    )
    ids = _u.index(value)
    expected = _u.outcome(
        jmespath.search, expression, value, jmespath.Options(custom_functions=jcls()), ids
    )
    actual = _u.outcome(
        pj.search, expression, value, pj.Options(custom_functions=pool[kind]()), ids
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# Nesting: RecursionError where jmespath's recursive parser and interpreter raise it
# ---------------------------------------------------------------------------

NESTING = {
    "parens": lambda n: "(" * n + "a" + ")" * n,
    "not": lambda n: "!" * n + "a",
    "list": lambda n: "[" * n + "a" + "]" * n,
    "hash": lambda n: "{a:" * n + "a" + "}" * n,
    "filter": lambda n: "[?" * n + "a" + "]" * n,
    "function": lambda n: "not_null(" * n + "a" + ")" * n,
    "expref": lambda n: "map(&" * n + "a" + ", @)" * n,
    "or": lambda n: "(a || " * n + "a" + ")" * n,
    "projection": lambda n: "a" + "[*].a" * n,
    "comparator": lambda n: "(a == " * n + "a" + ")" * n,
}


def _deepest(search: Any, build: Any) -> int:
    lo, hi = 0, 1200
    while lo < hi:
        mid = (lo + hi + 1) // 2
        pj.parser.Parser.purge()
        try:
            search(build(mid), {"a": {"a": [1]}})
        except RecursionError:
            hi = mid - 1
        except Exception:
            lo = mid
        else:
            lo = mid
    return lo


@pytest.mark.thread_unsafe(reason="depends on the stack depth and recursion limit")
@pytest.mark.skipif(sys.version_info < (3, 12), reason="3.11 counts C calls in the limit")
@pytest.mark.parametrize("kind", sorted(NESTING))
def test_recursion_depth_matches(kind: str) -> None:
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("other threads get a smaller native stack budget")
    build = NESTING[kind]
    assert _deepest(pj.search, build) == _deepest(jmespath.search, build)
