"""pyroid.jmespath: API, identity, the parse cache, options, custom functions, errors, limits."""

from __future__ import annotations

import copy
import importlib.util
import math
import pickle
import subprocess
import sys
import sysconfig
import textwrap
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest

import pyroid
import pyroid.jmespath as jp
from pyroid._pyroid import jmespath as native
from pyroid.jmespath import ast, exceptions, functions, lexer, parser, visitor

HAVE_JMESPATH = importlib.util.find_spec("jmespath") is not None


def oracle() -> Any:
    return pytest.importorskip("jmespath")


def test_search_and_compile() -> None:
    data = {"foo": {"bar": [{"baz": 1}, {"baz": 2}, {"qux": 3}]}}
    assert jp.search("foo.bar[*].baz", data) == [1, 2]
    compiled = jp.compile("foo.bar[?baz > `1`].baz | [0]")
    assert compiled.search(data) == 2
    assert compiled.expression == "foo.bar[?baz > `1`].baz | [0]"
    assert jp.search("length(foo.bar)", data) == 3
    assert jp.__version__ == jp.JMESPATH_VERSION == "1.1.0"


def test_results_are_the_inputs_own_objects() -> None:
    inner = {"x": [1, 2]}
    items = [inner, {"x": None}, {"x": [3]}]
    data = {"a": inner, "items": items}
    assert jp.search("a", data) is inner
    assert jp.search("@", data) is data
    assert jp.search("items[0]", data) is inner
    projected = jp.search("items[*].x", data)
    assert projected == [[1, 2], [3]]
    assert projected[0] is inner["x"]
    assert jp.search("items[*]", data) is not items  # a projection builds a new list
    # Literals are the compiled AST's objects, as in jmespath.
    literal = jp.compile("`[1, 2]`")
    assert literal.search({}) is literal.search({}) is literal.parsed["value"]


@pytest.mark.thread_unsafe(reason="the parse cache is process-wide")
def test_parse_cache() -> None:
    parser.Parser.purge()
    first = jp.compile("cache.test")
    assert jp.compile("cache.test") is first
    assert parser.Parser().parse("cache.test") is first
    assert native._cache_len() == 1
    # 512 entries; the oldest goes first (jmespath 1.1.0).
    for i in range(parser.Parser._MAX_SIZE):
        jp.compile(f"cache{i}")
    assert native._cache_len() == 512
    assert jp.compile("cache.test") is not first
    assert jp.compile("cache1") is jp.compile("cache1")
    parser.Parser.purge()
    assert native._cache_len() == 0


class _AnyKey(str):
    """Every instance hashes alike and equals everything (P67's jm_strsub_eq)."""

    __slots__ = ()

    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return True


class _CaseFolded(str):
    """A case-insensitive key."""

    __slots__ = ()

    def __hash__(self) -> int:
        return hash(self.lower())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, str) and self.lower() == other.lower()


class _Unhashable(str):
    __slots__ = ()
    __hash__ = None  # type: ignore[assignment]


def _purge_both() -> Any:
    parser.Parser.purge()
    if HAVE_JMESPATH:
        real = oracle()
        real.parser.Parser.purge()
        return real
    return None


@pytest.mark.thread_unsafe(reason="the parse cache is process-wide")
def test_str_subclass_keys_with_their_own_eq_or_hash() -> None:
    # jmespath's cache is a dict keyed by the expression object: a str subclass with its own
    # __eq__/__hash__ finds what a dict finds, not what its text would.
    data = {"a": 1, "b": 2, "A": 3}
    try:
        real = _purge_both()
        got = [jp.search(_AnyKey("a"), data), jp.search(_AnyKey("b"), data)]
        assert got == [1, 1]
        if real is not None:
            assert got == [real.search(_AnyKey("a"), data), real.search(_AnyKey("b"), data)]
        real = _purge_both()
        first = jp.compile(_CaseFolded("A"))
        assert jp.compile(_CaseFolded("a")) is first
        # A plain str with the same hash: the dict compares it through _CaseFolded.__eq__.
        assert jp.compile("a") is first
        assert jp.search("a", data) == 3
        assert type(first.expression) is _CaseFolded
        assert native._cache_len() == 1
        if real is not None:
            real.compile(_CaseFolded("A"))
            assert real.search("a", data) == 3
        # Unhashable: the dict's own TypeError (3.14 words it differently from hash()).
        lookup: dict[Any, Any] = {}
        with pytest.raises(TypeError) as caught:
            lookup.get(_Unhashable("a"))
        with pytest.raises(TypeError) as raised:
            jp.search(_Unhashable("a"), data)
        assert str(raised.value) == str(caught.value)
        # Eviction: 512 entries, the oldest out first, odd keys included.
        _purge_both()
        jp.compile(_AnyKey("z"))
        for i in range(parser.Parser._MAX_SIZE - 1):
            jp.compile(f"odd{i}")
        assert native._cache_len() == 512
        assert jp.search(_AnyKey("b"), data) is None  # _AnyKey("z")'s entry
        jp.compile("odd_last")
        assert native._cache_len() == 512
        assert jp.search(_AnyKey("b"), data) == 2
    finally:
        _purge_both()
    # With no odd keys cached, lookups go by text again.
    assert jp.compile("a") is jp.compile("a")
    assert jp.compile(_TextOnly("a")) is jp.compile("a")


class _TextOnly(str):
    """A str subclass that keeps str's __eq__ and __hash__: keyed by its text."""

    __slots__ = ()


def test_parsed_ast() -> None:
    assert jp.compile("foo.bar").parsed == ast.subexpression([ast.field("foo"), ast.field("bar")])
    parsed = jp.compile("a[0][1:2]").parsed
    assert list(parsed) == ["type", "children"]
    assert parsed["children"][1] == {"type": "index", "value": 0, "children": []}
    assert parsed["children"][2] == {"type": "slice", "children": [1, 2, None]}
    compiled = jp.compile("{a: `1`}")
    assert compiled.parsed is compiled.parsed
    assert repr(compiled) == repr(compiled.parsed)
    assert jp.compile("foo")._render_dot_file() == ('digraph AST {\nfield1 [label="field(foo)"]\n}')


def test_parsed_result_from_a_dict_ast() -> None:
    tree = ast.projection(ast.field("items"), ast.field("x"))
    result = parser.ParsedResult("items[*].x", tree)
    assert result.parsed is tree
    assert result.search({"items": [{"x": 1}, {"y": 2}, {"x": 3}]}) == [1, 3]
    result.parsed = ast.field("y")
    assert result.search({"y": 5}) == 5
    result.expression = "y"
    assert result.expression == "y"
    clone = pickle.loads(pickle.dumps(jp.compile("a.b[?c == `1`]")))
    assert clone.search({"a": {"b": [{"c": 1}, {"c": 2}]}}) == [{"c": 1}]
    assert copy.copy(jp.compile("a")).search({"a": 1}) == 1
    with pytest.raises(NotImplementedError, match="nope"):
        parser.ParsedResult("?", {"type": "nope", "children": []}).search({})


def test_options() -> None:
    data = {"c": 3, "b": 2, "a": 1}
    result = jp.search("{a: a, b: b}", data, jp.Options(dict_cls=OrderedDict))
    assert type(result) is OrderedDict
    assert list(result) == ["a", "b"]
    assert type(jp.search("{a: a}", data, jp.Options(dict_cls=dict))) is dict

    class Duck:
        dict_cls = OrderedDict
        custom_functions = None

    assert type(jp.search("{a: a}", data, Duck())) is OrderedDict
    with pytest.raises(AttributeError):
        jp.search("a", data, object())
    if HAVE_JMESPATH:
        real = oracle()
        assert type(jp.search("{a: a}", data, real.Options(dict_cls=OrderedDict))) is OrderedDict


def test_errors() -> None:
    with pytest.raises(exceptions.ParseError) as parse_error:
        jp.compile("foo]baz")
    e = parse_error.value
    assert (e.lex_position, e.token_value, e.token_type, e.expression) == (
        3,
        "]",
        "RBRACKET",
        "foo]baz",
    )
    assert str(e) == (
        'Unexpected token: ]: Parse error at column 3, token "]" (RBRACKET), for expression:\n'
        '"foo]baz"\n    ^'
    )
    with pytest.raises(exceptions.IncompleteExpressionError) as incomplete:
        jp.compile("length(@,")
    assert incomplete.value.lex_position == 9
    assert incomplete.value.token_type is None
    with pytest.raises(exceptions.LexerError) as lexer_error:
        jp.compile('foo."bar')
    assert (
        str(lexer_error.value) == 'Bad jmespath expression: Unclosed " delimiter:\nfoo."bar\n    ^'
    )
    with pytest.raises(exceptions.EmptyExpressionError):
        jp.search("", {})
    with pytest.raises(exceptions.ArityError, match=r"Expected 1 argument for function length"):
        jp.search("length(@, @)", [])
    with pytest.raises(exceptions.VariadictArityError):
        jp.search("not_null()", [])
    with pytest.raises(exceptions.JMESPathTypeError) as type_error:
        jp.search("length(@)", 2)
    assert type_error.value.current_value == 2
    assert type_error.value.actual_type == "number"
    with pytest.raises(exceptions.UnknownFunctionError) as unknown:
        jp.search("nope(@)", 1)
    assert isinstance(unknown.value.__context__, KeyError)
    assert issubclass(exceptions.JMESPathError, ValueError)
    if HAVE_JMESPATH:
        real = oracle()
        assert exceptions.ParseError is real.exceptions.ParseError
        with pytest.raises(real.exceptions.JMESPathError):
            jp.search("a[", {})


def test_python_comparisons_raise_as_in_jmespath() -> None:
    with pytest.raises(TypeError, match="'<' not supported"):
        jp.search("a < b", {"a": 1, "b": "x"})
    with pytest.raises(TypeError, match="requires string as left operand"):
        jp.search("contains('abc', `1`)", {})
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        jp.search("[::0]", [1])
    assert jp.search("[?a >= `1`].a", [{"a": Decimal("3")}]) == [Decimal("3")]
    assert jp.search("a < b", {"a": "2016", "b": "2017"}) is True


def test_numbers() -> None:
    assert jp.search("sum(@)", [1, 2, 3]) == 6
    assert type(jp.search("sum(@)", [1, 2, 3])) is int
    assert jp.search("sum(@)", [0.1] * 10) == sum([0.1] * 10)
    assert jp.search("avg(@)", [1, 2]) == 1.5
    assert jp.search("avg(@)", [2**60 + 1, 1]) == (2**60 + 2) / 2
    assert jp.search("ceil(@)", 1.5) == 2
    assert type(jp.search("floor(@)", 1.5)) is int
    assert jp.search("abs(@)", -(2**70)) == 2**70
    assert [jp.search("to_number(@)", s) for s in ["1_000", " 12 ", "1e3", "x", "1.5"]] == [
        1000,
        12,
        1000.0,
        None,
        1.5,
    ]
    assert math.isnan(jp.search("to_number(@)", "nan"))
    assert jp.search("to_string(@)", {"é": [1, None, True]}) == '{"\\u00e9":[1,null,true]}'
    assert jp.search("max(@)", [1, 2.5, 2]) == 2.5
    assert jp.search("sort(@)", [3, 1.5, 2]) == [1.5, 2, 3]


def test_sort_by_is_stable() -> None:
    rows = [{"k": k, "i": i} for i, k in enumerate([2, 1, 2, 1, 1.0, 0])]
    assert [r["i"] for r in jp.search("sort_by(@, &k)", rows)] == [5, 1, 3, 4, 0, 2]
    words = [{"k": k, "i": i} for i, k in enumerate(["b", "a", "b", "ä", "a"])]
    assert [r["i"] for r in jp.search("sort_by(@, &k)", words)] == [1, 4, 0, 2, 3]
    assert jp.search("max_by(@, &k).i", rows) == 0
    assert jp.search("min_by(@, &k).i", rows) == 5
    if HAVE_JMESPATH:
        real = oracle()
        nan = float("nan")
        values = [{"k": v} for v in [3.0, nan, 1.0, 2.0, nan, 0.5]]
        expected = real.search("sort_by(@, &k)", values)
        assert [id(x) for x in jp.search("sort_by(@, &k)", values)] == [id(x) for x in expected]


_SORT_MEMORY = r"""
import random, resource, sys
import pyroid.jmespath as jp

def peak():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024

n = 1_000_000
data = list(range(1000, 1000 + n))
random.Random(0).shuffle(data)
compiled = jp.compile(EXPRESSION)
compiled.search(data[:100])
before = peak()
result = compiled.search(data)
assert result == sorted(data)
print((peak() - before) / n)
"""


@pytest.mark.thread_unsafe(reason="a subprocess")
@pytest.mark.skipif(sys.platform == "win32", reason="no resource module")
@pytest.mark.parametrize("expression", ["sort_by(@, &@)", "sort(@)"])
def test_sort_peak_memory(run_python: Any, expression: str) -> None:
    # J8: sorting 1M records took +61 MB at peak (jmespath: +24 MB), about 64 bytes per element.
    r = run_python(f"EXPRESSION = {expression!r}\n" + _SORT_MEMORY, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    # Now about 36 bytes for int keys (items, keys, (key, index) pairs, the order) on Linux. Peak
    # RSS follows the platform's allocator: macOS's malloc measured 51 to 64 bytes for the same
    # sort, so the tight bound applies where it was measured and a loose one elsewhere.
    bound = 50 if sys.platform.startswith("linux") else 80
    assert float(r.stdout) < bound, r.stdout


class _Plus(functions.Functions):
    @functions.signature({"types": ["number"]}, {"types": ["number"]})
    def _func_add(self, x: Any, y: Any) -> Any:
        return x + y

    @functions.signature({"types": ["array"]}, {"types": ["expref"]})
    def _func_twice(self, array: Any, expref: Any) -> Any:
        assert type(expref).__name__ == "_Expression"
        assert expref.expression["type"] == "field"
        via_interpreter = [expref.interpreter.visit(expref.expression, x) for x in array]
        assert via_interpreter == [expref.visit(expref.expression, x) for x in array]
        return [expref.visit(expref.expression, x) * 2 for x in array]


def test_custom_functions() -> None:
    options = jp.Options(custom_functions=_Plus())
    assert jp.search("add(`1`, `2`)", {}, options) == 3
    assert jp.search("twice(@, &a)", [{"a": 1}, {"a": 2}], options) == [2, 4]
    assert jp.search("length(@)", [1, 2], options) == 2
    with pytest.raises(exceptions.UnknownFunctionError):
        jp.search("add(`1`, `2`)", {})

    class Dispatch(_Plus):
        def call_function(self, function_name: Any, resolved_args: Any) -> Any:
            if function_name == "length":
                return -1
            return super().call_function(function_name, resolved_args)

    dispatch = jp.Options(custom_functions=Dispatch())
    assert jp.search("length(@)", [1, 2], dispatch) == -1
    assert jp.search("sort_by(@, &a)[*].a", [{"a": 2}, {"a": 1}], dispatch) == [1, 2]


def test_custom_functions_from_jmespath() -> None:
    real = oracle()

    base: Any = real.functions.Functions
    signature: Any = real.functions.signature

    class Custom(base):  # type: ignore[misc]
        @signature({"types": ["string", "array", "object", "null"]})  # type: ignore[untyped-decorator]
        def _func_length0(self, s: Any) -> Any:
            return 0 if s is None else len(s)

    options = real.Options(custom_functions=Custom())
    data = {"a": {"b": [1, 2, 3]}}
    assert jp.search("length0(a.b)", data, options) == 3
    assert jp.search("length0(a.c)", data, options) == 0
    assert jp.search("sort(a.b)", data, options) == [1, 2, 3]


def _shout(self: Any, s: Any) -> Any:
    return s.upper()


@pytest.mark.thread_unsafe(reason="edits the stock FUNCTION_TABLE")
def test_functions_added_to_the_stock_function_table() -> None:
    # P67's jm_function_table_add: jmespath's default Functions() looks every call up in its
    # class's FUNCTION_TABLE, so a function added there is found. pyroid reads the tables for
    # names that are not built-ins (jmespath's class first, then pyroid's).
    classes: list[Any] = [functions.Functions]
    if HAVE_JMESPATH:
        classes.append(oracle().functions.Functions)
    for cls in classes:
        table = cls.FUNCTION_TABLE
        table["shout"] = {"function": _shout, "signature": [{"types": ["string"]}]}
        try:
            assert jp.search("shout('a')", {}) == "A"
            assert jp.compile("[*].shout(@)").search(["a", "b"]) == ["A", "B"]
            with pytest.raises(exceptions.JMESPathTypeError):
                jp.search("shout(`1`)", {})
            with pytest.raises(exceptions.ArityError):
                jp.search("shout('a', 'b')", {})
            if cls is not functions.Functions:
                real = oracle()
                assert real.search("[*].shout(@)", ["a", "b"]) == ["A", "B"]
        finally:
            del table["shout"]
        with pytest.raises(exceptions.UnknownFunctionError, match=r"shout\(\)"):
            jp.search("shout('a')", {})


@pytest.mark.thread_unsafe(reason="edits the stock FUNCTION_TABLE")
def test_replaced_builtin_entries_are_not_read() -> None:
    # Documented: the built-ins run natively without reading FUNCTION_TABLE (reading it would
    # cost every search that calls one).
    table = functions.Functions.FUNCTION_TABLE
    saved = table["abs"]
    table["abs"] = {"function": lambda self, x: "ABS", "signature": saved["signature"]}
    try:
        assert jp.search("abs(`-1`)", {}) == 1
    finally:
        table["abs"] = saved


def test_expref_visit_with_other_arguments() -> None:
    # P67's jm_visit: anything but (node, value) raises what jmespath's visit_<type>(node,
    # value) raises.
    expref: Any = jp.search("&a", {})
    node = expref.expression
    assert expref.visit(node, {"a": 5}) == 5
    assert expref.visit(node, value={"a": 6}) == 6
    calls: list[Callable[[Any], Any]] = [
        lambda e: e.visit(e.expression),
        lambda e: e.visit(e.expression, {}, {}),
        lambda e: e.visit(e.expression, nope=1),
        lambda e: e.visit(e.expression, {}, value={}),
        lambda e: e.visit(e.expression, value={}, nope=1),
        lambda e: e.visit({"type": "nope", "children": []}),
    ]
    got = [_raised(c, expref) for c in calls]
    assert got[0] == (
        "TypeError",
        "TreeInterpreter.visit_field() missing 1 required positional argument: 'value'",
    )
    assert got[-1] == ("NotImplementedError", "nope")
    if HAVE_JMESPATH:
        real_expref = oracle().search("&a", {})
        assert got == [_raised(c, real_expref) for c in calls]


def _raised(f: Callable[[Any], Any], arg: Any) -> tuple[str, str]:
    try:
        f(arg)
    except Exception as e:
        return type(e).__name__, str(e)
    return "no error", ""


class _TaggedStr(str):
    __slots__ = ()


class _LazyList(list[Any]):
    __slots__ = ()


class _LazyDict(dict[Any, Any]):
    __slots__ = ()


_ANSIBLE_CASES = [
    "length(@)",
    "length(servers)",
    "servers[?starts_with(name, 'server')].name",
    "servers[?contains(name, '1')].name",
    "keys(@)",
    "values(servers[0])",
    "merge(servers[0], servers[1])",
    "servers[].name | join(', ', @)",
    "map(&name, servers)",
    "reverse(servers[].name)",
    "max(servers[].name)",
    "sort(servers[].name)",
    "sort_by(servers, &name)[].name",
    "max_by(servers, &name).name",
    "servers[].name | [?ends_with(@, 'db')]",
]


def _ansible_data() -> Any:
    servers = _LazyList(
        [_LazyDict(name=_TaggedStr("server11"), n=1), _LazyDict(name=_TaggedStr("db"), n=2)]
    )
    return _LazyDict(servers=servers)


def _outcome(search: Any, expression: str, data: Any) -> Any:
    try:
        return search(expression, data)
    except Exception as e:
        return (type(e).__name__, str(e))


@pytest.mark.thread_unsafe(reason="changes the module-wide type maps")
def test_type_maps_take_additions_as_ansible_makes_them() -> None:
    # J2: Ansible's json_query adds its tagged str/list/dict classes to REVERSE_TYPES_MAP.
    reverse = functions.REVERSE_TYPES_MAP
    saved = dict(reverse)
    data = _ansible_data()
    before = {e: _outcome(jp.search, e, data) for e in _ANSIBLE_CASES}
    assert before["length(@)"][0] == "JMESPathTypeError"
    try:
        reverse["string"] = reverse["string"] + ("_TaggedStr",)
        reverse["array"] = reverse["array"] + ("_LazyList",)
        reverse["object"] = reverse["object"] + ("_LazyDict",)
        assert jp.search("length(@)", _TaggedStr("web-prod")) == 8
        after = {e: _outcome(jp.search, e, data) for e in _ANSIBLE_CASES}
        assert after["length(servers)"] == 2
        assert after["servers[].name | join(', ', @)"] == "server11, db"
        assert after["max(servers[].name)"] == "server11"
        # sort_by's keys go by TYPES_MAP, which Ansible leaves alone: jmespath's error.
        assert after["sort_by(servers, &name)[].name"][0] == "JMESPathTypeError"
        if HAVE_JMESPATH:
            real = oracle()
            assert functions.REVERSE_TYPES_MAP is real.functions.REVERSE_TYPES_MAP
            assert functions.TYPES_MAP is real.functions.TYPES_MAP
            assert after == {e: _outcome(real.search, e, data) for e in _ANSIBLE_CASES}

        # A custom Functions subclass reads the same maps.
        class Custom(functions.Functions):
            pass

        options = jp.Options(custom_functions=Custom())
        assert jp.search("length(servers)", data, options) == 2
    finally:
        reverse.clear()
        reverse.update(saved)
    assert _outcome(jp.search, "length(servers)", data) == before["length(servers)"]


@pytest.mark.thread_unsafe(reason="changes the module-wide type maps")
def test_types_map_takes_additions() -> None:
    saved = dict(functions.TYPES_MAP)
    data = {
        "servers": [{"name": _TaggedStr("server11"), "n": 1}, {"name": _TaggedStr("db"), "n": 2}]
    }
    try:
        functions.TYPES_MAP["_TaggedStr"] = "string"
        assert jp.search("sort_by(servers, &name)[].name", data) == ["db", "server11"]
        assert jp.search("max_by(servers, &name).n", data) == 1
        with pytest.raises(exceptions.JMESPathTypeError, match='received: "string"'):
            jp.search("abs(@)", _TaggedStr("x"))
    finally:
        functions.TYPES_MAP.clear()
        functions.TYPES_MAP.update(saved)


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_tokenize_is_lazy() -> None:
    tokens = lexer.Lexer().tokenize("a.b ^")
    assert next(tokens) == {"type": "unquoted_identifier", "value": "a", "start": 0, "end": 1}
    assert next(tokens)["type"] == "dot"
    assert next(tokens)["value"] == "b"
    with pytest.raises(exceptions.LexerError):
        next(tokens)
    with pytest.raises(exceptions.EmptyExpressionError):
        list(lexer.Lexer().tokenize(""))
    with pytest.warns(PendingDeprecationWarning, match="deprecated string literal syntax"):
        assert next(iter(lexer.Lexer().tokenize("`foo`")))["value"] == "foo"


@pytest.mark.thread_unsafe(reason="warnings")
def test_other_expression_types() -> None:
    search: Any = jp.search  # these expressions are not str on purpose
    for falsy in (None, b"", 0):
        with pytest.raises(exceptions.EmptyExpressionError):
            search(falsy, {})
    with pytest.raises(TypeError, match="unhashable"):
        search(["a"], {})
    with pytest.raises(exceptions.LexerError, match="Unknown token 97"):
        search(b"a.b", {})
    if HAVE_JMESPATH:
        getattr(jp, "_WARNED").discard("search")  # noqa: B009 - once per process
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert search(("a", ".", "b"), {"a": {"b": 1}}) == 1
        assert any(w.category is pyroid.FallbackWarning for w in caught)
    else:
        with pytest.raises(pyroid.PyroidError):
            search(("a",), {})


def test_unhashable_expressions_raise_the_cache_lookups_error() -> None:
    # J5: jmespath looks the expression up in its cache dict; 3.14 words that error differently
    # from hash().
    search: Any = jp.search
    compile_: Any = jp.compile
    cache: dict[Any, Any] = {}
    for expression in (["a"], bytearray(b"a"), {"a": 1}):
        with pytest.raises(TypeError) as caught:
            cache.get(expression)
        with pytest.raises(TypeError) as raised:
            search(expression, {})
        assert str(raised.value) == str(caught.value)
        with pytest.raises(TypeError) as raised:
            compile_(expression)
        assert str(raised.value) == str(caught.value)


@pytest.mark.thread_unsafe(reason="warnings")
def test_tokenize_takes_what_jmespath_lexes() -> None:
    # J4: Lexer().tokenize() does not hash its expression (jmespath lexes any iterable).
    lexer_: Any = lexer.Lexer()
    with pytest.raises(exceptions.LexerError, match="Unknown token 110") as e:
        list(lexer_.tokenize(bytearray(b"n")))
    assert (e.value.lex_position, e.value.token_value, e.value.expression) == (0, 110, None)
    with pytest.raises(exceptions.EmptyExpressionError):
        list(lexer_.tokenize([]))
    if HAVE_JMESPATH:
        real = oracle()
        getattr(jp, "_WARNED").discard("tokenize")  # noqa: B009 - once per process
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert list(lexer_.tokenize(["n"])) == list(real.lexer.Lexer().tokenize(["n"]))
            # Tokens up to a bad character come out first, as from jmespath's generator.
            tokens = lexer_.tokenize(["a", ".", "~"])
            assert next(tokens)["value"] == "a"
            assert next(tokens)["type"] == "dot"
            with pytest.raises(exceptions.LexerError):
                next(tokens)
        assert any(w.category is pyroid.FallbackWarning for w in caught)
    else:
        with pytest.raises(pyroid.PyroidError):
            list(lexer_.tokenize(["n"]))


@pytest.mark.thread_unsafe(reason="warnings and the parse cache")
def test_deprecated_literal_warning_points_at_the_caller() -> None:
    # J7: the warning pointed at pyroid/jmespath/__init__.py.
    calls: list[Callable[[str], Any]] = [
        lambda e: jp.search(e, {}),
        lambda e: jp.compile(e),
        lambda e: parser.Parser().parse(e),
        lambda e: list(lexer.Lexer().tokenize(e)),
    ]
    for i, call in enumerate(calls):
        parser.Parser.purge()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call(f"`literal_{i}`")
        [w] = [w for w in caught if w.category is PendingDeprecationWarning]
        assert str(w.message) == "deprecated string literal syntax"
        assert w.filename == __file__, (i, w.filename)


def test_lone_surrogates() -> None:
    data = {"\ud800": 1, "x\udfff": 2}
    assert jp.search('"\\ud800"', data) == 1
    assert jp.search('"x\udfff"', data) == 2
    assert jp.search("'\ud800'", {}) == "\ud800"
    with pytest.raises(exceptions.LexerError) as e:
        jp.compile("a.\ud800")
    assert e.value.lex_position == 2


@pytest.mark.thread_unsafe(reason="a subprocess")
def test_deep_nesting_on_a_small_stack_raises() -> None:
    code = textwrap.dedent(
        """
        import sys, threading
        import pyroid.jmespath as jp
        sys.setrecursionlimit(1_000_000)
        out = []
        def work():
            for e in ["(" * 3000 + "a" + ")" * 3000, "[" * 3000 + "a" + "]" * 3000,
                      "{a:" * 3000 + "a" + "}" * 3000, "!" * 3000 + "a"]:
                try:
                    jp.search(e, {"a": 1})
                except RecursionError:
                    out.append("RecursionError")
        threading.stack_size(256 * 1024)
        t = threading.Thread(target=work)
        t.start()
        t.join()
        print(out)
        """
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == str(["RecursionError"] * 4)


_LONG_CHAINS = r"""
import sys, threading
import pyroid.jmespath as jp
from pyroid.jmespath import exceptions

CHAINS = {
    "pipe": lambda n: " | ".join(["a"] * n),
    "or_expression": lambda n: " || ".join(["a"] * n),
    "and_expression": lambda n: " && ".join(["a"] * n),
    "comparator": lambda n: " == ".join(["a"] * n),
    "projection": lambda n: "a" + "[]" * n,
}

def run(n):
    out = []
    for kind, build in CHAINS.items():
        expression = build(n)
        # jmespath parses these chains in a loop and compiles them; only its interpreter runs
        # out of frames.
        compiled = jp.compile(expression)
        assert compiled.parsed["type"] == kind, compiled.parsed["type"]
        try:
            jp.search(expression, {"a": [1]})
            out.append("ok")
        except RecursionError:
            out.append("RecursionError")
        # A syntax error after a long chain.
        try:
            jp.compile(expression + " ]")
        except exceptions.ParseError:
            out.append("ParseError")
    return out

print(run(20_000))
sys.setrecursionlimit(100_000)
print(run(30_000))
if STACK:
    threading.stack_size(STACK)
    results = []
    t = threading.Thread(target=lambda: results.append(run(30_000)))
    t.start()
    t.join()
    print(results[0])
"""


#: CPython 3.14+ on musl (Alpine) cannot itself build and drop such deep data (the `.parsed`
#: dicts) in a thread with a 256 KiB stack (see test_yaml.py); that case says nothing about pyroid.
_MUSL_SMALL_STACK = pytest.mark.skipif(
    sys.version_info >= (3, 14) and "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or ""),
    reason="CPython 3.14+ on musl overflows its own C stack at this depth in a 256 KiB thread",
)


@pytest.mark.thread_unsafe(reason="a subprocess")
@pytest.mark.parametrize("stack", [0, pytest.param(256 << 10, marks=_MUSL_SMALL_STACK)])
def test_long_chains_compile_and_raise_recursion_error(run_python: Any, stack: int) -> None:
    # J1: `a || a || ...`, `a[][]...` and the like (20,000 terms, 40 KB) crashed the interpreter.
    r = run_python(f"STACK = {stack}\n" + _LONG_CHAINS, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    runs = [eval(line) for line in r.stdout.splitlines()]
    assert len(runs) == (3 if stack else 2)
    # At the default limit, as in jmespath; with a raised one the native stack decides.
    assert runs[0] == ["RecursionError", "ParseError"] * 5
    for run in runs[1:]:
        assert run[1::2] == ["ParseError"] * 5
        assert set(run[::2]) <= {"ok", "RecursionError"}


@pytest.mark.thread_unsafe(reason="starts its own threads")
def test_threads_share_the_cache() -> None:
    expressions = [f"k{i}.v" for i in range(2000)]
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for i, e in enumerate(expressions):
                assert jp.search(e, {f"k{i}": {"v": i}}) == i
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert native._cache_len() <= 512


def test_free_threading_keeps_the_gil_off() -> None:
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled is None or sysconfig_gil():
        pytest.skip("not a free-threaded build")
    assert not is_gil_enabled()


def sysconfig_gil() -> bool:
    import sysconfig

    return not sysconfig.get_config_var("Py_GIL_DISABLED")


def test_modules() -> None:
    assert visitor.Options is jp.Options
    assert hasattr(visitor, "TreeInterpreter")
    interpreter = visitor.TreeInterpreter()
    assert interpreter.visit(jp.compile("a.b").parsed, {"a": {"b": 3}}) == 3
    assert parser.Parser.BINDING_POWER["lparen"] == 60
    assert functions.TYPES_MAP["OrderedDict"] == "object"
