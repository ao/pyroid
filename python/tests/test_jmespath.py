"""pyroid.jmespath: API, identity, the parse cache, options, custom functions, errors, limits."""

from __future__ import annotations

import copy
import importlib.util
import math
import pickle
import subprocess
import sys
import textwrap
import threading
import warnings
from collections import OrderedDict
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
