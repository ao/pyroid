"""pyroid.sqlparse: behaviour, interop with sqlparse's objects, limits, safety and batches.

The differential tests against sqlparse 0.6.1 are in ``test_sqlparse_parity.py``, the patch
in ``test_sqlparse_patch.py``; sqlparse's own suite runs from ``parity/data/sqlparse-0.6.1``.
"""

from __future__ import annotations

import io
import os
import pickle
import platform
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from typing import Any

import pytest

import pyroid
import pyroid.sqlparse as P  # noqa: N812
from pyroid.sqlparse import engine, lexer
from pyroid.sqlparse import tokens as T  # noqa: N812

S: Any = pytest.importorskip("sqlparse")
SAME = getattr(S, "__version__", None) == "0.6.1"
needs_061 = pytest.mark.skipif(not SAME, reason="sqlparse 0.6.1 is the reference")

QUERIES = [
    "select a, b from t where x = 1 and y = 2",
    "SELECT a.id, COUNT(b.x) AS cnt, SUM(CASE WHEN b.y > 10 THEN 1 ELSE 0 END) AS big\n"
    "FROM accounts a LEFT JOIN orders b ON a.id = b.account_id\n"
    "WHERE a.created_at >= '2024-01-01' AND a.status IN ('open', 'pending')\n"
    "GROUP BY a.id HAVING COUNT(b.x) > 2 ORDER BY cnt DESC LIMIT 50;",
    "insert into t (a, b) values (1, 2), (3, 4); update t set a = 1 where b = 2; delete from t",
    "create table foo (id int primary key, name varchar(10) not null) /* c */ -- x\n",
    "CREATE FUNCTION f() RETURNS int AS $$ BEGIN RETURN 1; END; $$ LANGUAGE plpgsql;",
    "with x as (select 1) select * from x",
    "select 1.5a, 123abc, 0x1F, -2.5E-3, 'it''s', \"q\", `b`, [c], :p, %(n)s, ?, $1",
    "",
]


def tree(t: Any) -> tuple[Any, ...]:
    """A token object's state, recursively (parents by class name)."""
    return (
        type(t).__name__,
        t.ttype,
        t.value,
        t.normalized,
        t.is_keyword,
        t.is_group,
        t.is_whitespace,
        t.is_newline,
        type(t.parent).__name__ if t.parent is not None else None,
        tuple(tree(c) for c in t.tokens) if t.is_group else None,
    )


def run_py(code: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


# ---------------------------------------------------------------------------
# Objects: sqlparse's own classes and token types
# ---------------------------------------------------------------------------


def test_shares_sqlparse_objects() -> None:
    assert P.sql.Identifier is S.sql.Identifier
    assert P.sql.Statement is S.sql.Statement
    assert P.tokens.Keyword is S.tokens.Keyword
    assert P.tokens.Keyword.DML is S.tokens.Keyword.DML
    assert P.exceptions.SQLParseError is S.exceptions.SQLParseError
    (stmt,) = P.parse("select a from b")
    assert isinstance(stmt, S.sql.Statement)
    assert stmt.tokens[0].ttype is S.tokens.Keyword.DML
    assert any(isinstance(t, S.sql.Identifier) for t in stmt.tokens)


@needs_061
def test_shares_the_lexer_with_sqlparse_061() -> None:
    assert P.lexer.Lexer is S.lexer.Lexer
    assert P.keywords.KEYWORDS is S.keywords.KEYWORDS


@pytest.mark.parametrize("q", QUERIES)
def test_trees_round_trip(q: str) -> None:
    stmts = P.parse(q)
    assert "".join(str(s) for s in stmts) == q or q.strip() == ""
    for s in stmts:
        for tok in s.flatten():
            node = tok
            while node.parent is not None:
                assert any(x is node for x in node.parent.tokens)
                node = node.parent
            assert node is s


@needs_061
@pytest.mark.parametrize("q", QUERIES)
def test_same_as_sqlparse(q: str) -> None:
    assert [tree(s) for s in P.parse(q)] == [tree(s) for s in S.parse(q)]
    assert list(P.lexer.tokenize(q)) == list(S.lexer.tokenize(q))
    assert P.split(q) == S.split(q)
    assert P.split(q, strip_semicolon=True) == S.split(q, strip_semicolon=True)
    for opts in ({}, {"reindent": True}, {"keyword_case": "upper", "reindent_aligned": True}):
        assert P.format(q, **opts) == S.format(q, **opts)


def test_trees_mix_with_sqlparse_code() -> None:
    (stmt,) = P.parse("select a, b from t where x = 1")
    # sqlparse's own filters on pyroid's tree
    S.filters.StripWhitespaceFilter().process(stmt)
    S.filters.SpacesAroundOperatorsFilter().process(stmt)
    assert str(stmt) == "select a, b from t where x = 1"
    # sqlparse's tree through pyroid's filters
    (theirs,) = S.parse("select  a  from t")
    P.filters.StripWhitespaceFilter().process(theirs)
    assert str(theirs) == "select a from t"


def test_pickle_and_copy_trees() -> None:
    import copy

    (stmt,) = P.parse(QUERIES[1])
    # round trip of objects this test just made (trusted data)
    for clone in (pickle.loads(pickle.dumps(stmt)), copy.deepcopy(stmt)):
        assert tree(clone) == tree(stmt)


def test_own_classes_without_sqlparse() -> None:
    r = run_py(
        """
        import pickle, sys
        sys.modules["sqlparse"] = None
        import pyroid.sqlparse as P  # noqa: N812
        assert P._backend.REAL is None
        (s,) = P.parse("select a, b from t where x = 1")
        assert type(s).__module__ == "pyroid.sqlparse.sql", type(s).__module__
        assert s.tokens[0].ttype is P.tokens.Keyword.DML
        print(P.format("select a, b from t where x = 1", reindent=True))
        assert pickle.loads(pickle.dumps(s)).value == s.value
        assert P.split("select 1; select 2") == ["select 1;", "select 2"]
        """
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout == "select a,\n       b\nfrom t\nwhere x = 1\n"


# ---------------------------------------------------------------------------
# Input handling and laziness
# ---------------------------------------------------------------------------


def test_parsestream_is_lazy() -> None:
    gen = P.parsestream(123)  # type: ignore[arg-type]
    assert type(gen).__name__ == "generator"
    with pytest.raises(TypeError, match=r"Expected text or file-like object, got <class 'int'>"):
        next(gen)
    stmts = list(P.parsestream(io.StringIO("select 1; select 2")))
    assert [str(s) for s in stmts] == ["select 1; ", "select 2"]


@pytest.mark.thread_unsafe
@pytest.mark.skipif(sys.platform == "win32", reason="the resource module")
@pytest.mark.parametrize(("call", "most"), [("split", 3.0), ("format", 3.5), ("parsestream", 3.0)])
def test_big_text_memory_stays_near_the_text_size(call: str, most: float) -> None:
    # The text is lexed as the splitter asks and each statement's tokens are dropped after
    # it, so split, format and parsestream hold about one statement's tokens, not the text's
    # (they took about 9 times the text's size). format returns a str of the text's size,
    # split a list of strs holding all of it. A fresh process: ru_maxrss is the peak so far.
    r = run_py(
        """
        import gc, resource, sys
        import pyroid.sqlparse as P
        call, n = sys.argv[1], int(sys.argv[2])
        P.format("select 1"), P.split("select 1"), list(P.parsestream("select 1"))
        text = "insert into t values (1, 'abc', 2.5, null), (2, 'de', -3, 'x');\\n" * n
        gc.collect()
        scale = 1 if sys.platform == "darwin" else 1024
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        if call == "split":
            P.split(text)
        elif call == "format":
            P.format(text)
        else:
            # Each statement is a tree with parent links, freed by the cycle collector; how much of
            # that garbage waits for a collection depends on the build (free-threaded CPython on
            # macOS kept 5.6 times the text, and 3.2 times with a collection every 1,000
            # statements). Collecting every 100 measures what pyroid itself holds.
            for i, _ in enumerate(P.parsestream(text)):
                if i % 100 == 99:
                    gc.collect()
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
        print(f"{(after - before) / len(text):.2f}")
        """,
        call,
        # parsestream makes Python objects for every statement: a smaller text
        str((2 if call == "parsestream" else 8) * 2**20 // 64),
        timeout=600,
    )
    assert r.returncode == 0, r.stderr
    assert float(r.stdout) < most, f"{call}: {r.stdout.strip()} times the text's size"


@pytest.mark.thread_unsafe
def test_parsestream_reads_the_limits_for_each_statement() -> None:
    deep = "select " + "(" * 150 + "1" + ")" * 150
    grouping = engine.grouping
    saved = grouping.MAX_GROUPING_DEPTH
    try:
        stmts = P.parsestream(f"select 1; {deep}; {deep}")
        assert str(next(stmts)) == "select 1; "
        grouping.MAX_GROUPING_DEPTH = None  # type: ignore[assignment]  # read for the next one
        assert str(next(stmts)) == f"{deep}; "
        grouping.MAX_GROUPING_DEPTH = 100
        with pytest.raises(P.exceptions.SQLParseError, match="depth exceeded"):
            next(stmts)
    finally:
        grouping.MAX_GROUPING_DEPTH = saved


def test_the_namespace_is_sqlparses() -> None:
    public = {n for n in dir(P) if not n.startswith("_")}
    assert not public & {"Any", "Generator", "annotations"}
    assert set(P.__all__) <= public


def test_bytes_and_encodings() -> None:
    assert P.format(b"select '\xc3\xa9'") == "select 'é'"
    assert P.format(b"select '\xe9'") == S.format(b"select '\xe9'")  # not utf-8: unicode-escape
    assert P.format("select 'é'".encode("cp1252"), encoding="cp1252") == "select 'é'"
    with pytest.raises(LookupError):
        P.format(b"select 1", encoding="no-such-codec")


@needs_061
def test_lone_surrogates() -> None:
    for q in ["select '\ud800' as x", "select \udfff from t", "select 1 -- \ud83d\n"]:
        assert P.format(q, reindent=True) == S.format(q, reindent=True)
        assert [tree(s) for s in P.parse(q)] == [tree(s) for s in S.parse(q)]
        assert P.split(q) == S.split(q)
    # a surrogate and a character from U+10F800 on: sqlparse's own code runs
    q = "select '\ud800', '\U0010f800'"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert P.format(q) == S.format(q)
        assert [tree(s) for s in P.parse(q)] == [tree(s) for s in S.parse(q)]


# ---------------------------------------------------------------------------
# DoS limits and deep input
# ---------------------------------------------------------------------------


def test_depth_limit_message() -> None:
    q = "select " + "(" * 150 + "1" + ")" * 150
    with pytest.raises(
        P.exceptions.SQLParseError, match=r"^Maximum grouping depth exceeded \(100\)\.$"
    ):
        P.parse(q)
    with pytest.raises(P.exceptions.SQLParseError, match="depth"):
        P.format(q, reindent=True)
    assert P.split(q) == [q]  # split does not group


def test_token_limit_message() -> None:
    q = "select " + ", ".join(["1"] * 6000)
    with pytest.raises(
        P.exceptions.SQLParseError, match=r"^Maximum number of tokens exceeded \(10000\)\.$"
    ):
        P.parse(q)


@pytest.mark.thread_unsafe
def test_limits_follow_the_module_globals() -> None:
    g: Any = engine.grouping
    saved = g.MAX_GROUPING_DEPTH, g.MAX_GROUPING_TOKENS
    try:
        g.MAX_GROUPING_DEPTH = 5
        with pytest.raises(P.exceptions.SQLParseError, match=r"\(5\)"):
            P.parse("select ((((((((1))))))))")
        g.MAX_GROUPING_DEPTH = None
        g.MAX_GROUPING_TOKENS = None
        (s,) = P.parse("select " + "(" * 400 + "1" + ")" * 400)
        assert str(s).count("(") == 400
        # far past CPython's recursion limit: sqlparse's own error
        with pytest.raises(P.exceptions.SQLParseError, match=r"^Maximum recursion depth exceeded$"):
            P.parse("select " + "(" * 5000 + "1" + ")" * 5000)
    finally:
        g.MAX_GROUPING_DEPTH, g.MAX_GROUPING_TOKENS = saved


@needs_061
@pytest.mark.thread_unsafe
def test_limits_disabled_like_sqlparse() -> None:
    mods: list[Any] = [engine.grouping, S.engine.grouping]
    saved = [(m, m.MAX_GROUPING_DEPTH, m.MAX_GROUPING_TOKENS) for m in mods]
    try:
        for m, _, _ in saved:
            m.MAX_GROUPING_DEPTH = m.MAX_GROUPING_TOKENS = None
        q = "select " + "(" * 300 + "a + 1" + ")" * 300
        assert [tree(s) for s in P.parse(q)] == [tree(s) for s in S.parse(q)]
        assert P.format(q, reindent=True) == S.format(q, reindent=True)
    finally:
        for m, d, t in saved:
            m.MAX_GROUPING_DEPTH, m.MAX_GROUPING_TOKENS = d, t


@pytest.mark.thread_unsafe
@pytest.mark.skipif(
    platform.libc_ver()[0] != "glibc" and sys.platform == "linux",
    reason="small thread stacks crash CPython itself on musl",
)
def test_deep_input_in_a_small_thread() -> None:
    g: Any = engine.grouping
    saved = g.MAX_GROUPING_DEPTH, g.MAX_GROUPING_TOKENS
    out: list[Any] = []

    def work() -> None:
        try:
            out.append(P.format("select " + "(" * 99 + "1" + ")" * 99, reindent=True))
            g.MAX_GROUPING_DEPTH = None
            g.MAX_GROUPING_TOKENS = None
            try:
                P.format("select " + "(" * 50_000 + "1" + ")" * 50_000, reindent=True)
            except P.exceptions.SQLParseError as e:
                out.append(str(e))
        finally:
            g.MAX_GROUPING_DEPTH, g.MAX_GROUPING_TOKENS = saved

    old = threading.stack_size(512 * 1024)
    try:
        t = threading.Thread(target=work)
        t.start()
        t.join()
    finally:
        threading.stack_size(old)
    assert out[0].startswith("select (")
    assert out[1:] in ([], ["Maximum recursion depth exceeded"])


# ---------------------------------------------------------------------------
# Options and errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("opts", "message"),
    [
        ({"keyword_case": "foo"}, "Invalid value for keyword_case: 'foo'"),
        ({"output_format": "java"}, "Unknown output format: 'java'"),
        ({"indent_width": 0}, "indent_width requires a positive integer"),
        ({"truncate_strings": 1}, "Invalid value for truncate_strings: 1"),
        ({"right_margin": 5}, "right_margin requires an integer > 10"),
    ],
)
def test_invalid_options(opts: dict[str, Any], message: str) -> None:
    with pytest.raises(P.exceptions.SQLParseError) as e:
        P.format("select 1", **opts)
    assert str(e.value) == message


def test_right_margin_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        P.format("select 1", right_margin=20)
    assert P.format("", right_margin=20) == ""


# ---------------------------------------------------------------------------
# The lexer: customisation, overrides
# ---------------------------------------------------------------------------


@pytest.mark.thread_unsafe
def test_lexer_customisation_is_followed() -> None:
    lex = lexer.Lexer.get_default_instance()
    try:
        lex.clear()
        lex.set_SQL_REGEX([(r"ZORDER\s+BY\b", T.Keyword), *P.keywords.SQL_REGEX])
        for d in (P.keywords.KEYWORDS_COMMON, P.keywords.KEYWORDS):
            lex.add_keywords(d)
        lex.add_keywords({"BAR": T.Keyword})
        toks: list[Any] = [
            (t.ttype, t.value) for t in P.parse("select * from foo zorder by bar")[0].flatten()
        ]
        assert (T.Keyword, "zorder by") in toks
        assert (T.Keyword, "bar") in toks
    finally:
        lex.default_initialization()
    values = [t.value for t in P.parse("select * from foo zorder by bar")[0].flatten()]
    assert "zorder by" not in values


@pytest.mark.thread_unsafe
def test_lexer_patterns_pyroid_does_not_compile_run_in_re() -> None:
    lex = lexer.Lexer.get_default_instance()
    try:
        lex.set_SQL_REGEX([(r"(?x) ZORDER \s+ BY \b", T.Keyword), *P.keywords.SQL_REGEX])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyroid.FallbackWarning)
            toks = [t.value for t in P.parse("select 1 zorder  by x")[0].flatten()]
        assert "zorder  by" in toks
    finally:
        lex.default_initialization()


@pytest.mark.thread_unsafe
def test_an_overridden_lexer_runs_python() -> None:
    class Loud(lexer.Lexer):
        def get_tokens(self, text: Any, encoding: Any = None) -> Any:
            for ttype, value in super().get_tokens(text, encoding):
                yield ttype, value.upper() if ttype in T.Keyword else value

    saved = lexer.Lexer._default_instance
    try:
        loud = Loud()
        loud.default_initialization()
        lexer.Lexer._default_instance = loud
        from pyroid.sqlparse import _core

        _core._WARNED.clear()  # once per process: --iterations reruns this test
        with pytest.warns(pyroid.FallbackWarning):
            out = P.format("select a from b")
        assert out == "SELECT a FROM b"
    finally:
        lexer.Lexer._default_instance = saved


LEXED = "select foo, zz from t"


def _lexed() -> tuple[Any, ...]:
    """What each entry point makes of LEXED with keyword_case="upper" (FallbackWarnings off)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        return (
            P.format(LEXED, keyword_case="upper"),
            [(t, v) for t, v in lexer.tokenize(LEXED) if v.strip()],
            [(t.ttype, t.value) for t in P.parse(LEXED)[0].flatten() if t.value.strip()],
            P.format_batch([LEXED], keyword_case="upper"),
        )


def _zz_is_a_keyword(got: tuple[Any, ...]) -> None:
    assert got[0] == "SELECT foo, ZZ FROM t"
    assert (T.Keyword, "zz") in got[1]
    assert (T.Keyword, "zz") in got[2]
    assert got[3] == ["SELECT foo, ZZ FROM t"]


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("method", ["is_keyword", "get_tokens"])
def test_a_method_replaced_on_the_lexer_class_runs_python(method: str) -> None:
    # Set on the class itself (not a subclass or the instance): lexing goes to that code.
    cls = lexer.Lexer
    orig = cls.__dict__[method]

    def is_keyword(self: Any, value: str) -> Any:
        return (T.Keyword, value) if value.upper() == "ZZ" else orig(self, value)

    def get_tokens(self: Any, text: Any, encoding: Any = None) -> Any:
        for ttype, value in orig(self, text, encoding):
            yield (T.Keyword, value) if value.upper() == "ZZ" else (ttype, value)

    setattr(cls, method, is_keyword if method == "is_keyword" else get_tokens)
    try:
        from pyroid.sqlparse import _core

        _core._WARNED.clear()  # once per process: --iterations reruns this test
        with pytest.warns(pyroid.FallbackWarning, match="overridden methods"):
            P.split(LEXED)
        _zz_is_a_keyword(_lexed())
        if SAME and lexer.Lexer is S.lexer.Lexer:  # one class for both: sqlparse agrees
            assert S.format(LEXED, keyword_case="upper") == "SELECT foo, ZZ FROM t"
    finally:
        setattr(cls, method, orig)
    assert _lexed()[0] == "SELECT foo, zz FROM t"


@pytest.mark.thread_unsafe
def test_a_keyword_dict_subclass_is_read_through_its_methods() -> None:
    class AlwaysZZ(dict):  # type: ignore[type-arg]
        def __contains__(self, key: object) -> bool:
            return isinstance(key, str) and key.startswith("ZZ")

        def __getitem__(self, key: object) -> Any:
            return T.Keyword

    class Plain(dict):  # type: ignore[type-arg]
        pass

    lex = lexer.Lexer.get_default_instance()
    try:
        lex.add_keywords(AlwaysZZ())
        _zz_is_a_keyword(_lexed())
        lex.default_initialization()
        lex.add_keywords(Plain({"ZZ": T.Keyword}))  # dict's own lookups: read natively
        with warnings.catch_warnings():
            warnings.simplefilter("error", pyroid.FallbackWarning)
            _zz_is_a_keyword(_lexed())
    finally:
        lex.default_initialization()


@pytest.mark.thread_unsafe
def test_keyword_dict_edits_that_keep_its_size_are_seen() -> None:
    lex = lexer.Lexer.get_default_instance()
    up = {"keyword_case": "upper"}
    d: dict[str, Any] = {"FOO": T.Keyword}
    big = {f"K{i}": T.Keyword for i in range(300)}
    try:
        lex.add_keywords(d)
        lex.add_keywords(big)
        assert P.format(LEXED, **up) == "SELECT FOO, zz FROM t"
        del d["FOO"]
        d["ZZ"] = T.Keyword
        assert P.format(LEXED, **up) == "SELECT foo, ZZ FROM t"
        d.popitem()
        d["FOO"] = T.Keyword
        assert P.format(LEXED, **up) == "SELECT FOO, zz FROM t"
        # many deletes and as many adds, so the dict compacts its entries
        for i in range(200):
            del big[f"K{i}"]
        big.update({f"J{i}": T.Keyword for i in range(199)})
        big["ZZ"] = T.Keyword
        assert P.format(LEXED, **up) == "SELECT FOO, ZZ FROM t"
        # a value replaced in place (not the last key) is seen after the next add_keywords
        # (documented)
        d["FOO"] = T.Name
        lex.add_keywords({})
        assert P.format(LEXED, **up) == "SELECT foo, ZZ FROM t"
    finally:
        lex.default_initialization()


# ---------------------------------------------------------------------------
# group() on statements of Python objects
# ---------------------------------------------------------------------------


@needs_061
def test_filterstack_with_python_filters_groups_natively() -> None:
    class Bold:
        def process(self, stream: Any) -> Any:
            for ttype, value in stream:
                yield ttype, f"<b>{value}</b>" if ttype in T.Keyword else value

    def run(mod: Any) -> list[str]:
        stack = mod.engine.FilterStack()
        stack.enable_grouping()
        stack.preprocess.append(Bold())
        stack.stmtprocess.append(mod.filters.StripWhitespaceFilter())
        stack.postprocess.append(mod.filters.SerializerUnicode())
        return list(stack.run(QUERIES[1]))

    assert run(P) == run(S)


@needs_061
def test_group_keeps_object_identity() -> None:
    stmt = next(S.engine.StatementSplitter().process(S.lexer.tokenize("select a.b from t")))
    leaves, lst = list(stmt.tokens), stmt.tokens
    assert P.engine.grouping.group(stmt) is stmt
    assert stmt.tokens is lst
    assert {id(x) for x in stmt.flatten()} == {id(x) for x in leaves}
    ref = next(S.engine.StatementSplitter().process(S.lexer.tokenize("select a.b from t")))
    S.engine.grouping.group(ref)
    assert tree(stmt) == tree(ref)


# ---------------------------------------------------------------------------
# Batches
# ---------------------------------------------------------------------------


def test_batches_equal_the_loop() -> None:
    docs = QUERIES * 50
    for opts in ({}, {"reindent": True}, {"keyword_case": "upper", "strip_comments": True}):
        want = [P.format(d, **opts) for d in docs]
        for threads in (None, 1, 2):
            assert P.format_batch(docs, threads=threads, **opts) == want
            assert P.format_batch(tuple(docs), threads=threads, **opts) == tuple(want)
    assert P.split_batch(docs) == [P.split(d) for d in docs]
    assert P.split_batch(docs, strip_semicolon=True) == [
        P.split(d, strip_semicolon=True) for d in docs
    ]


def test_batches_on_arrow_without_pyarrow() -> None:
    from pyroid._pyroid import _testing

    col = _testing.column([QUERIES[0], None, QUERIES[2]], chunks=2)
    out = P.format_batch(col, reindent=True)
    assert out.to_pylist() == [
        P.format(QUERIES[0], reindent=True),
        None,
        P.format(QUERIES[2], reindent=True),
    ]
    assert P.split_batch(col) == [P.split(QUERIES[0]), None, P.split(QUERIES[2])]


def test_batch_raises_what_the_loop_raises() -> None:
    deep = "select " + "(" * 150 + "1" + ")" * 150
    with pytest.raises(P.exceptions.SQLParseError, match="depth"):
        P.format_batch(["select 1", deep, "select 2"], reindent=True)
    # the loop's own TypeError (not pyroid.InputError), for lists and tuples
    for bad in (None, 2, 1.5):
        with pytest.raises(TypeError) as one:
            P.format(bad)  # type: ignore[arg-type]
        assert str(one.value) == f"Expected text or file-like object, got {type(bad)!r}"
        batches: list[Any] = [P.format_batch, P.split_batch]
        for batch in batches:
            for kind in (list, tuple):
                with pytest.raises(TypeError) as many:
                    batch(kind(["select 1", bad]))
                assert type(many.value) is TypeError
                assert str(many.value) == str(one.value)


def test_batches_take_what_format_and_split_take() -> None:
    def docs() -> list[Any]:  # streams are read once: a fresh list per call
        return ["select a", b"select 'b'", b"select '\xe9'", io.StringIO("select c; select d")]

    for opts in ({}, {"keyword_case": "upper"}, {"reindent": True}):
        want = [P.format(d, **opts) for d in docs()]
        assert P.format_batch(docs(), **opts) == want
        assert P.format_batch(tuple(docs()), **opts) == tuple(want)
    assert P.split_batch(docs()) == [P.split(d) for d in docs()]
    assert P.split_batch(tuple(docs()), strip_semicolon=True) == tuple(
        P.split(d, strip_semicolon=True) for d in docs()
    )


@pytest.mark.timeout(120)
def test_big_batch() -> None:
    docs = [f"select a{i}, b from t where c = {i}" for i in range(20_000)]
    out = P.format_batch(docs, reindent=True)
    assert out[123] == P.format(docs[123], reindent=True)


# ---------------------------------------------------------------------------
# Threads and Ctrl-C
# ---------------------------------------------------------------------------


@pytest.mark.thread_unsafe
def test_threads_while_the_lexer_changes() -> None:
    # While another thread re-initialises the lexer, a call can see its table half built
    # (sqlparse has the same race); nothing may crash, and the results are right afterwards.
    want = P.format(QUERIES[1], reindent=True)
    stop = time.monotonic() + 2
    bad: list[Any] = []

    def user() -> None:
        while time.monotonic() < stop:
            try:
                P.format(QUERIES[1], reindent=True)
                P.parse(QUERIES[1])
            except Exception as e:
                bad.append(e)
                return

    def changer() -> None:
        lex = lexer.Lexer.get_default_instance()
        while time.monotonic() < stop:
            lex.add_keywords({"ZZZ": T.Keyword})
            lex.default_initialization()

    threads = [threading.Thread(target=user) for _ in range(6)] + [threading.Thread(target=changer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lexer.Lexer.get_default_instance().default_initialization()
    assert bad == []
    assert P.format(QUERIES[1], reindent=True) == want


@pytest.mark.thread_unsafe
@pytest.mark.parametrize(
    "send",
    [
        # Trips CPython's SIGINT flag as the signal handler does, on every platform.
        "_thread.interrupt_main()",
        # A real signal. On Windows os.kill(pid, SIGINT) is TerminateProcess(pid, 2).
        pytest.param(
            "os.kill(os.getpid(), signal.SIGINT)",
            marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals"),
        ),
    ],
    ids=["interrupt_main", "kill"],
)
def test_ctrl_c_stops_a_long_format(send: str) -> None:
    code = f"""
        import _thread, os, signal, threading, time
        import pyroid.sqlparse as P  # noqa: N812
        q = "select a, b, c from t where x = 1 and y in (1, 2, 3);\\n" * 200_000
        threading.Timer(0.5, lambda: {send}).start()
        t = time.monotonic()
        try:
            P.format(q, reindent=True)
            print("finished", time.monotonic() - t)
        except KeyboardInterrupt:
            print("interrupted", time.monotonic() - t)
        """
    r = run_py(code, timeout=300)
    assert r.returncode == 0, r.stderr
    what, secs = r.stdout.split()
    assert what == "interrupted" or float(secs) < 1.0, r.stdout
    assert float(secs) < 3.0, r.stdout


def _dist_info(root: Any, version: str) -> None:
    """Metadata that makes pyroid._versions see sqlparse `version` installed under `root`."""
    info = root / f"sqlparse-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: sqlparse\nVersion: {version}\n")


def test_version_warning_for_another_release(tmp_path: Any) -> None:
    src = os.path.dirname(S.__file__)
    import shutil

    dst = tmp_path / "sqlparse"
    shutil.copytree(src, dst)
    init = dst / "__init__.py"
    init.write_text(
        init.read_text().replace(f'__version__ = "{S.__version__}"', '__version__ = "0.5.3"')
    )
    _dist_info(tmp_path, "0.5.3")
    r = run_py(
        """
        import sys, warnings
        sys.path.insert(0, sys.argv[1])
        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as w:
            import pyroid.sqlparse as P  # noqa: N812
        import sqlparse
        assert P.sql.Identifier is sqlparse.sql.Identifier
        print("|".join(f"{x.category.__name__}: {x.message}" for x in w))
        """,
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    # One warning, pyroid's VersionMismatchWarning: the classes are still sqlparse's own.
    assert r.stdout.strip() == (
        "VersionMismatchWarning: pyroid.sqlparse reproduces sqlparse 0.6.1, but sqlparse 0.5.3 "
        "is installed: results can differ from it. Install sqlparse==0.6.1 to compare like for "
        'like (or silence this with warnings.simplefilter("ignore", '
        "pyroid.VersionMismatchWarning))."
    )


def test_classes_warning_for_a_release_older_than_0_5(tmp_path: Any) -> None:
    # A stand-in sqlparse 0.4.4: its sql module lacks the classes pyroid builds trees from.
    pkg = tmp_path / "sqlparse"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "0.4.4"\n')
    for sub in ("sql", "tokens", "exceptions", "keywords", "lexer"):
        (pkg / f"{sub}.py").write_text("")
    _dist_info(tmp_path, "0.4.4")
    r = run_py(
        """
        import sys, warnings
        sys.path.insert(0, sys.argv[1])
        warnings.simplefilter("always")
        with warnings.catch_warnings(record=True) as w:
            import pyroid.sqlparse as P  # noqa: N812
        assert P.sql.Identifier.__module__ == "pyroid.sqlparse.sql"
        print("|".join(f"{x.category.__name__}: {x.message}" for x in w))
        """,
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    got = r.stdout.strip().split("|")
    assert len(got) == 2, got
    # The classes are chosen (and the RuntimeWarning given) before the version check runs.
    assert got[0] == (
        "RuntimeWarning: sqlparse 0.4.4 is older than 0.5.0, so pyroid.sqlparse builds its trees "
        "from its own copies of sqlparse 0.6.1's token classes: isinstance checks against "
        "sqlparse.sql fail on them, and pyroid.patch('sqlparse') refuses to patch it."
    )
    assert got[1].startswith(
        "VersionMismatchWarning: pyroid.sqlparse reproduces sqlparse 0.6.1, but sqlparse 0.4.4 "
        "is installed: results can differ from it."
    )
