"""Differential tests: pyroid.sqlparse must give sqlparse 0.6.1's output exactly.

Every input goes through both libraries and the results must match: ``lexer.tokenize`` (token
type identity and value), ``parse`` trees field by field (class, ``ttype`` identity, ``value``,
``normalized``, the ``is_*`` flags, parent links by position, the children) and ``str`` of each
statement, ``split`` with and without ``strip_semicolon``, ``format`` under random option sets
(valid and invalid values; exceptions by class and message), the DoS limits
(``engine.grouping.MAX_GROUPING_DEPTH`` and ``MAX_GROUPING_TOKENS`` set on both libraries'
modules), ``parsestream`` on text streams and bytes, a customised lexer table, and pyroid's regex
engine (``_regex_match``) against ``re`` on every pattern of ``keywords.SQL_REGEX`` and on random
patterns.

Inputs: SQL-ish text from a grammar (keywords in any case, including spellings with a dotless
i, a long s or the Kelvin sign that IGNORECASE folds; plain, unicode and quoted identifiers;
strings, numbers, placeholders, comments and dollar quotes, closed and unclosed; operators;
nested parentheses, brackets, CASE, IF, BEGIN and loops; statement separators; odd whitespace;
lone surrogates and characters from U+10F800 up), random mutations of real statements, and
arbitrary text.

``PYROID_SQLPARSE_FUZZ_EXAMPLES=n`` sets the Hypothesis examples of every test (long runs).
``PYROID_SQLPARSE_OWN=1`` hides sqlparse while pyroid.sqlparse is imported, so pyroid builds its
trees from its own copies of sqlparse's classes and token types (then compared by name and
path); run this file alone in that mode.
"""

from __future__ import annotations

import contextlib
import functools
import io
import os
import re
import sys
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import pytest

OWN = os.environ.get("PYROID_SQLPARSE_OWN") == "1"
if OWN and ("pyroid.sqlparse" in sys.modules or "sqlparse" in sys.modules):
    pytest.skip("PYROID_SQLPARSE_OWN=1 needs a fresh process", allow_module_level=True)
try:
    if OWN:
        sys.modules["sqlparse"] = None  # type: ignore[assignment]
    import pyroid.sqlparse as ps
    from pyroid._pyroid import sqlparse as native
except ImportError as e:  # not built yet
    pytest.skip(f"pyroid.sqlparse is not importable: {e}", allow_module_level=True)
finally:
    if OWN:
        del sys.modules["sqlparse"]

sqlparse: Any = pytest.importorskip("sqlparse")
if getattr(sqlparse, "__version__", None) != "0.6.1":
    pytest.skip("needs sqlparse 0.6.1", allow_module_level=True)

import sqlparse.engine.grouping  # type: ignore[no-redef]  # noqa: E402
import sqlparse.keywords  # type: ignore[no-redef]  # noqa: E402
import sqlparse.lexer  # type: ignore[no-redef]  # noqa: E402
from hypothesis import HealthCheck, assume, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

#: pyroid builds trees from sqlparse's own classes and token types.
SHARED = ps.tokens.Token is sqlparse.tokens.Token
if OWN and SHARED:
    pytest.skip(
        "PYROID_SQLPARSE_OWN=1 but pyroid shares sqlparse's objects", allow_module_level=True
    )
if not OWN and not SHARED:
    pytest.skip("pyroid.sqlparse does not share sqlparse 0.6.1's objects", allow_module_level=True)

pytestmark = [
    # Which path ran (MODULES.md), not what the result is.
    pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning"),
]

T = sqlparse.tokens
SK = sqlparse.keywords
UNIDATA = unicodedata.unidata_version

_EXAMPLES = os.environ.get("PYROID_SQLPARSE_FUZZ_EXAMPLES")
_HEALTH = [
    HealthCheck.too_slow,
    HealthCheck.data_too_large,
    HealthCheck.filter_too_much,
    HealthCheck.large_base_example,
]


def fuzz(n: int) -> settings:
    """``@settings`` with `n` examples, or ``PYROID_SQLPARSE_FUZZ_EXAMPLES``."""
    return settings(
        max_examples=int(_EXAMPLES) if _EXAMPLES else n,
        deadline=None,
        suppress_health_check=_HEALTH,
    )


@pytest.fixture(autouse=True)
def _oracle_is_sqlparse() -> None:
    """The oracle is sqlparse itself, not pyroid.patch("sqlparse")'s functions."""
    patched = [
        f"{m.__name__}.{n}"
        for m, n in (
            (sqlparse, "parse"),
            (sqlparse, "format"),
            (sqlparse, "split"),
            (sqlparse.lexer, "tokenize"),
            (sqlparse.engine.grouping, "group"),
        )
        if getattr(m, n).__module__ != m.__name__
    ]
    if patched:
        pytest.skip(f"sqlparse is patched ({', '.join(patched)})")


# ---- outcomes -----------------------------------------------------------------------------------


def tt_key(tt: Any) -> Any:
    """A token type: its path, and its identity when the libraries share token types."""
    if tt is None:
        return None
    if not isinstance(tt, tuple):
        return ("not a token type", type(tt).__name__, repr(tt))
    return (repr(tt), id(tt)) if SHARED else repr(tt)


def cls_key(cls: type) -> Any:
    if SHARED or cls.__module__ == "builtins":
        return (cls.__module__, cls.__qualname__, id(cls))
    return cls.__qualname__


def exc_key(e: BaseException) -> tuple[Any, ...]:
    cause = e.__cause__
    return (
        "raise",
        cls_key(type(e)),
        str(e),
        None if cause is None else (type(cause).__qualname__, str(cause)),
    )


class Node(NamedTuple):
    """One token of a tree: its fields, its parent by position."""

    path: tuple[int, ...]
    cls: Any
    ttype: Any
    value: Any
    normalized: Any
    is_keyword: Any
    is_group: Any
    is_whitespace: Any
    is_newline: Any
    parent: Any
    children: Any
    extra: Any


_MISSING = "<missing>"


def tree(stmt: Any) -> list[Node]:
    """`stmt` and every token under it, depth first (iterative: trees can be deep)."""
    order: list[tuple[tuple[int, ...], Any]] = []
    where: dict[int, tuple[int, ...]] = {}
    stack: list[tuple[tuple[int, ...], Any]] = [((), stmt)]
    shared: set[tuple[int, ...]] = set()  # an object, or a tokens list, met twice
    lists: set[int] = set()
    while stack:
        path, tok = stack.pop()
        if id(tok) in where:
            shared.add(path)
        where[id(tok)] = path
        order.append((path, tok))
        kids = getattr(tok, "tokens", None)
        if isinstance(kids, list):
            if id(kids) in lists:
                shared.add(path)
            lists.add(id(kids))
            stack.extend(((*path, i), kids[i]) for i in reversed(range(len(kids))))
    out = []
    for path, tok in order:
        parent = getattr(tok, "parent", _MISSING)
        if parent is None or parent is _MISSING:
            ppath: Any = parent
        elif id(parent) in where:
            ppath = where[id(parent)]
        else:
            ppath = ("outside the tree", type(parent).__name__, str(parent))
        kids = getattr(tok, "tokens", _MISSING)
        value = getattr(tok, "value", _MISSING)
        out.append(
            Node(
                path,
                cls_key(type(tok)),
                tt_key(getattr(tok, "ttype", _MISSING)),
                (type(value).__name__, value),
                getattr(tok, "normalized", _MISSING),
                getattr(tok, "is_keyword", _MISSING),
                getattr(tok, "is_group", _MISSING),
                getattr(tok, "is_whitespace", _MISSING),
                getattr(tok, "is_newline", _MISSING),
                ppath,
                len(kids) if isinstance(kids, list) else (type(kids).__name__, kids),
                (sorted(vars(tok)) if hasattr(tok, "__dict__") else None, path in shared),
            )
        )
    return out


def stmt_key(stmt: Any) -> tuple[Any, ...]:
    return (str(stmt), tree(stmt))


def drain(make: Callable[[], Any], key: Callable[[Any], Any]) -> tuple[Any, ...]:
    """The items `make()` yields (by `key`) and the exception that ends it, if any."""
    items: list[Any] = []
    try:
        it = make()
        kind = type(it).__name__
        items.extend(key(x) for x in it)  # keeps the items before an exception
    except Exception as e:  # compared, not handled
        return ("items", items, exc_key(e))
    return ("items", kind, items)


def call(fn: Callable[..., Any], *args: Any, **kw: Any) -> tuple[Any, ...]:
    try:
        r = fn(*args, **kw)
    except Exception as e:  # compared, not handled
        return exc_key(e)
    return ("ok", type(r).__name__, r)


def pair_key(p: Any) -> Any:
    if type(p) is not tuple or len(p) != 2:
        return ("not a pair", type(p).__name__, repr(p))
    return (tt_key(p[0]), type(p[1]).__name__, p[1])


def tokenize_out(lib: Any, text: Any, encoding: str | None = None) -> Any:
    return drain(lambda: lib.lexer.tokenize(text, encoding), pair_key)


def parse_out(lib: Any, text: Any, encoding: str | None = None) -> Any:
    try:
        stmts = lib.parse(text, encoding)
    except Exception as e:  # compared, not handled
        return exc_key(e)
    return ("ok", type(stmts).__name__, [stmt_key(s) for s in stmts])


def parsestream_out(lib: Any, stream: Any, encoding: str | None = None) -> Any:
    return drain(lambda: lib.parsestream(stream, encoding), stmt_key)


def first_diff(a: Any, b: Any, path: str = "") -> str:
    """Where `a` (pyroid) and `b` (sqlparse) first differ."""
    if isinstance(a, tuple | list) and isinstance(b, tuple | list) and type(a) is type(b):
        names = getattr(a, "_fields", None)
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                return first_diff(x, y, f"{path}.{names[i]}" if names else f"{path}[{i}]")
        if len(a) != len(b):
            extra = a[len(b)] if len(a) > len(b) else b[len(a)]
            return f"{path}: length {len(a)} (pyroid) != {len(b)} (sqlparse); next {extra!r:.300}"
    return f"{path or '<top>'}:\n  pyroid:   {a!r:.600}\n  sqlparse: {b!r:.600}"


def same(got: Any, want: Any, what: str, data: Any) -> None:
    """Fail with the first difference unless pyroid's outcome `got` equals sqlparse's `want`."""
    if got != want:
        pytest.fail(f"{what} differs for {data!r:.2000}\n{first_diff(got, want)}", pytrace=False)


# ---- generators: SQL-ish text -------------------------------------------------------------------

_KW_DICT_NAMES = (
    "KEYWORDS_COMMON",
    "KEYWORDS_ORACLE",
    "KEYWORDS_MYSQL",
    "KEYWORDS_PLPGSQL",
    "KEYWORDS_HQL",
    "KEYWORDS_MSACCESS",
    "KEYWORDS_SNOWFLAKE",
    "KEYWORDS_BIGQUERY",
    "KEYWORDS",
)
KEYWORD_WORDS = sorted({k for n in _KW_DICT_NAMES for k in getattr(SK, n)})
# The words and phrases SQL_REGEX matches itself, and the splitter's block words.
PHRASES = [
    "CASE", "IN", "VALUES", "USING", "FROM", "AS", "JOIN", "LEFT JOIN", "LEFT OUTER JOIN",
    "RIGHT\tJOIN", "FULL  OUTER JOIN", "INNER JOIN", "CROSS JOIN", "NATURAL JOIN",
    "STRAIGHT JOIN", "STRAIGHT_JOIN", "LEFT\nOUTER\nJOIN", "END", "END IF", "END LOOP",
    "END WHILE", "END FOR", "END CASE", "END  REPEAT", "IF EXISTS", "IF NOT EXISTS",
    "IF\nNOT\nEXISTS", "NOT NULL", "ASC", "DESC", "ASC NULLS FIRST", "DESC NULLS LAST",
    "NULLS FIRST", "NULLS LAST", "UNION ALL", "UNION", "CREATE", "CREATE OR REPLACE",
    "DOUBLE PRECISION", "GROUP BY", "ORDER BY", "PRIMARY KEY", "HANDLER FOR", "GO", "GO 2",
    "GO 10", "GO\t3", "LATERAL VIEW EXPLODE", "LATERAL VIEW INLINE", "LATERAL VIEW POSEXPLODE",
    "LATERAL VIEW STACK", "LATERAL VIEW PARSE_URL_TUPLE", "AT TIME ZONE 'UTC'",
    "WITH' TIME ZONE 'x'", "LIKE", "NOT LIKE", "ILIKE", "NOT ILIKE", "RLIKE", "REGEXP",
    "NOT REGEXP", "REGEXP BINARY", "NOT REGEXP BINARY", "BEGIN TRANSACTION", "BEGIN WORK",
    "BEGIN TRAN", "BEGIN DEFERRED", "BEGIN IMMEDIATE", "BEGIN EXCLUSIVE",
    "BEGIN DISTRIBUTED TRANSACTION", "COMMIT", "ROLLBACK", "DECLARE", "LOOP", "WHILE", "FOR",
    "DO", "THEN", "ELSE", "ELSIF", "WHEN", "OVER", "WHERE", "HAVING", "LIMIT", "RETURNING",
    "WINDOW", "QUALIFY", "TIMESTAMP", "DATE", "INTERVAL", "TIME", "BETWEEN", "AND", "OR", "NOT",
    "IS", "NULL", "TRUE", "FALSE", "DISTINCT", "SELECT", "INSERT", "UPDATE", "DELETE", "SET",
]  # fmt: skip
# Spellings that IGNORECASE or str.upper() fold to ASCII letters, or nearly: dotless i, dotted
# capital I, long s, Kelvin sign, sharp s and its capital, the fi/fl/ff/st ligatures, Angstrom
# sign, e acute, capital theta.
FOLDS = [
    ("I", "\u0131"), ("I", "\u0130"), ("S", "\u017f"), ("K", "\u212a"), ("SS", "\xdf"),
    ("SS", "\u1e9e"), ("FI", "\ufb01"), ("FL", "\ufb02"), ("FF", "\ufb00"), ("ST", "\ufb06"),
    ("A", "\u212b"), ("E", "\xe9"), ("O", "\u0398"),
]  # fmt: skip


@st.composite
def keyword_s(draw: Any) -> str:
    w = draw(st.sampled_from(KEYWORD_WORDS + PHRASES))
    mode = draw(st.integers(0, 4))
    if mode == 1:
        w = w.lower()
    elif mode == 2:
        mask = draw(st.integers(0, (1 << len(w)) - 1))
        w = "".join(c.lower() if mask >> i & 1 else c for i, c in enumerate(w))
    elif mode == 3:
        src, dst = draw(st.sampled_from(FOLDS))
        if src in w:
            w = w.replace(src, dst, draw(st.integers(1, 3)))
        if draw(st.booleans()):
            w = w.lower()
    return str(w)


ID_ALPHABET = (
    "abcxyz_ABCXYZ0189$#\xe9\xf1\xf8\xc6\xdf\u0131\u017f\u0130\u212a\u212b\xc0\xdc\xd7\xf7"
    "\xe0\xfc\xff\u0178\xb5\u0438\u0416\u03a9\u03bc\u03c2\u03a3\u540d\u524d\u0663\u096a\u07c0"
    "\u01c5\u0301\u0308\U0001d518\U0001d7d8\uff21"
)
names = st.one_of(
    st.sampled_from(
        [
            "a", "t", "t1", "users", "x1", "_x", "caf\xe9", "\u0438\u043c\u044f",
            "\u540d\u524d", "\u03a9mega", "\U0001d518ser", "a$b", "a#b", "@var", "@@rowcount",
            "##temp", "#temp", "@\xe9", "#\xc01", "@1", "\u017felect", "\u0131d", "\u212aey",
            "\u212bngstrom", "\xd7", "\xf7", "\xc0\xdc", "a\u0663", "\u0663a", "x\u0301",
            "public.t", "s.t.c", "t .c", "t. c", 't."c"', "t.*", "f(x)", "count(*)",
        ]
    ),
    st.text(alphabet=ID_ALPHABET, min_size=1, max_size=8),
)  # fmt: skip
quoted = st.one_of(
    names.map(lambda n: f'"{n}"'),
    names.map(lambda n: f"`{n}`"),
    names.map(lambda n: f"\xb4{n}\xb4"),
    names.map(lambda n: f"[{n}]"),
    st.sampled_from(
        [
            '""', '"a""b"', '"a\\"b"', '"a\\\\"', '"abc', '"a\nb"', "``", "`a``b`", "`abc",
            "\xb4\xb4", "\xb4a\xb4\xb4b\xb4", "\xb4abc", "[a b]", "[]", "[a[b]", "[abc", "a[1]",
            "a [1]", "x][1]", ")[1]", "[[x]]", '"\\"', '"x"y"',
        ]
    ),
)  # fmt: skip
strings = st.one_of(
    st.sampled_from(
        [
            "''", "'a'", "'it''s'", "'a\\'b'", "'a\\\\'", "'unclosed", "'multi\nline'", "N'x'",
            "E'\\n'", "B'0101'", "X'FF'", "''''", "'\\'", "'%s'", "'$$'", "'/*'", "'--'",
            "'\ud800'", "'\U0010f800'", "'a''", "'''", "'\\''", "'\xe9'", "''x''",
            "'abcdefghijkl'", "''abcdefghijklm''",
        ]
    ),
    st.text(alphabet="ab '\\\n%$-*/", max_size=12).map(lambda s: "'" + s + "'"),
)  # fmt: skip
numbers = st.one_of(
    st.sampled_from(
        [
            "1", "0", "-1", "1.5", "1.5a", "123abc", ".5", "-.5", "5.", "-5.", "-0x1F", "0x1f",
            "0xg", "0x", "1E-3", "1e10", "1.5E3", "1.5E+3", "-2.5e-7", "1_000",
            "\u0661\u0662\u0663", "\u0663.\u0665", "1.2.3", "00", "9" * 25, "1a", "a1", "1\xe9",
            "1\xc0", "1\xdc", "1\xfc", "1\xd7", "1.e5", "1e", "-", "--1", "1-1", "1.5.", "x.5",
            "t.5", "BETWEEN .5", "1.5_", "_1", "1\u212a", "1\u0131",
        ]
    ),
    st.integers(-(10**6), 10**6).map(str),
)  # fmt: skip
placeholders = st.sampled_from(
    [
        "?", "%s", "%(name)s", "%(n_1)s", "%(\xe9)s", "%()s", ":x", ":1", ":name", "$1", "$x",
        "?1", "?name", "%d", "%", ":", "$", "@p", "x:y", "x$1", "x?y", ":\xe9", "$\xe9", "%(a)",
        "\\x", "\\dt", "\\", "\\\\",
    ]
)  # fmt: skip
comments = st.sampled_from(
    [
        "-- c\n", "--\n", "-- c", "--+ hint\n", "--+hint", "# c\n", "#c", "# c\r\n", "# c\r",
        "#+ h\n", "/* c */", "/**/", "/*+ hint */", "/* unclosed", "*/", "/* a /* b */ c */",
        "/*\n*/", "--c\r", "-- \U0001d518\n", "/*/", "/*+", "/*/*/", "--\r\n", "-- a -- b\n",
        "/* $$ */", "-- $$\n", "/* ' */", "-- '\n", "#", "# ", "#+", "##x", "-- c\x85",
        "-- c\u2028x",
    ]
)  # fmt: skip
dollars = st.sampled_from(
    [
        "$$", "$$ x $$", "$tag$ x $tag$", "$a$ $b$ y $b$ $a$", "$$$$", "$tag$ unclosed",
        "x$$y$$", '"$$"', "$1$", "$_$x$_$", "$\xc0$x$\xc0$", "$\xe9$ $\xe9$", "$$\n$$",
        "$TAG$ $tag$", "$ab$c$ab$", "$$$", "$$ $$ $$", "a$$", "$x$$x$", "$\xdc1$ $\xdc1$",
        "$\u0131$ $\u0131$", "$\u212a$ $k$", "$$x$$y", "$1", "$$'$$'", "$$--$$\n", "$$/*$$*/",
        "$\xd7$ $\xd7$", "$\u212b$ $\u212b$",
    ]
)  # fmt: skip
operators = st.sampled_from(
    [
        "->", "->>", "#>", "#>>", "@>", "<@", "?|", "?&", "#-", "||", "::", ":=", "=", "==",
        "<>", "!=", ">=", "<=", "~", "!~*", "~~", "+", "-", "*", "/", "%", "^", "&", "|",
        "<=>", "=>", "!", "@", "#", "-->", "|/", ">>", "<<", "*=", "+=", "-=", "::text",
    ]
)  # fmt: skip
punct = st.sampled_from(["(", ")", "[", "]", ",", ";", ".", ":", "{", "}", "'", '"', "`", "\xb4"])
WHITESPACE = [
    " ", "  ", "\n", "\r\n", "\r", "\t", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
    "\xa0", "\u1680", "\u2000", "\u2028", "\u2029", "\u202f", "\u3000", "\u200b", "\ufeff",
    " \n ", "\n\n", "\r\r\n",
]  # fmt: skip
whitespace = st.sampled_from(WHITESPACE)
WEIRD = [
    "\ud800", "\udc00", "\udfff", "\ud83d", "\U0010f800", "\U0010f801", "\U0010ffff",
    "\U0010fffd", "\U0010f8ff", "\x00", "\x7f", "\u0301", "\u0130", "\ufffd", "\U0001d7d8",
    "\xad", "\u2066",
]  # fmt: skip
weird = st.sampled_from(WEIRD)

leaf = st.one_of(
    keyword_s(),
    names,
    quoted,
    strings,
    numbers,
    placeholders,
    comments,
    dollars,
    operators,
    punct,
    whitespace,
    weird,
)
sep = st.sampled_from([" ", " ", " ", "", "\n", "\t", "\r\n", "  "])

TEMPLATES = [
    "({})", "[{}]", "f({})", "COUNT({})", "{}({})", "CASE {} WHEN {} THEN {} ELSE {} END",
    "CASE WHEN {} THEN {} END", "IF {} THEN {} ELSE {} END IF", "IF ({}) {}",
    "BEGIN {}; {} END", "BEGIN {} END;", "FOR {} IN {} LOOP {} END LOOP",
    "WHILE {} LOOP {} END LOOP", "LOOP {} END LOOP", "WHILE {} DO {} END WHILE",
    "REPEAT {} UNTIL {} END REPEAT", "DECLARE {}; BEGIN {} END",
    "CREATE FUNCTION f() RETURNS int AS $$ {} $$ LANGUAGE sql",
    "CREATE OR REPLACE PROCEDURE p() BEGIN {} END", "SELECT {} FROM {} WHERE {}",
    "SELECT {} FROM {} GROUP BY {} HAVING {} ORDER BY {}", "INSERT INTO t ({}) VALUES ({}), ({})",
    "UPDATE {} SET {} = {} WHERE {}", "WITH x AS ({}) SELECT {}",
    "{} OVER (PARTITION BY {} ORDER BY {})", "{} OVER w", "CAST({} AS {})", "{}::{}",
    "{} AS {}", "{} {}", "{} BETWEEN {} AND {}", "{} IN ({})", "{} IS NOT NULL", "{}.{}",
    "{} = {}", "{} := {}", "{} < {}", "/* {} */", "-- {}\n", "'{}'", "$$ {} $$",
    "{}, {}, {}", "{};{}", "{}\nGO\n{}", "{} AND {} OR {}", "NOT {}", "{} JOIN {} ON {}",
    "{} LEFT JOIN {} USING ({})", "SELECT {} FROM {} WHERE {} IN (SELECT {} FROM {})",
    "{} WHERE {}", "VALUES ({}, {})", "{} ASC, {} DESC", "TIMESTAMP '{}'", "{} AT TIME ZONE 'UTC'",
    "{} -> {}", "{} || {}", "{}[{}]", "ARRAY[{}]", "(SELECT {}) AS {}", "EXISTS ({})",
    "{} UNION ALL {}", "{} LIMIT {}", "SELECT {} FROM {} LATERAL VIEW EXPLODE({}) t AS {}",
    "BEGIN TRANSACTION; {}; COMMIT", "BEGIN; {}; END;", "DECLARE x int; {}",
    "CREATE TABLE {} ({} int, {} varchar(10))", "{} = ANY({})", "INTERVAL '{}' DAY",
]  # fmt: skip


def _templated(children: Any) -> Any:
    @st.composite
    def fill(draw: Any) -> str:
        tpl = draw(st.sampled_from(TEMPLATES))
        parts = tpl.split("{}")
        out = [parts[0]]
        for p in parts[1:]:
            out.append(draw(children))
            out.append(p)
        if draw(st.integers(0, 9)) == 0:  # an unbalanced construct now and then
            cut = draw(st.integers(0, sum(map(len, out))))
            return "".join(out)[:cut]
        return "".join(out)

    @st.composite
    def seq(draw: Any) -> str:
        items = draw(st.lists(children, min_size=1, max_size=5))
        return "".join(x + draw(sep) for x in items)

    return st.one_of(fill(), seq())


structured = st.recursive(leaf, _templated, max_leaves=25)


@st.composite
def statements(draw: Any) -> str:
    parts = draw(st.lists(structured, min_size=1, max_size=4))
    seps = [";", ";\n", "\nGO\n", "\nGO 2\n", " ", "\n", ";;", "\r\n", "; ", ";\n\n", "\ngo\n"]
    return "".join(p + draw(st.sampled_from(seps)) for p in parts)


@st.composite
def soup(draw: Any) -> str:
    items = draw(st.lists(st.one_of(leaf, sep), max_size=40))
    return "".join(items)


REAL_SQL = [
    "SELECT a, b AS bee, c.d FROM t1 AS x LEFT OUTER JOIN t2 y ON x.id = y.id WHERE a > 1 AND b IN "
    "(1, 2, 3) GROUP BY a ORDER BY b DESC NULLS LAST LIMIT 10 OFFSET 5;",
    "select count(*), sum(x) over (partition by y order by z rows between unbounded preceding and "
    "current row) from t window w as (partition by q)",
    "INSERT INTO users (id, name, email) VALUES (1, 'Ann', 'a@x.org'), (2, 'Bob''s', NULL) ON "
    "CONFLICT (id) DO UPDATE SET name = EXCLUDED.name RETURNING id;",
    "UPDATE accounts SET balance = balance - 100.50, updated_at = NOW() WHERE id = :id AND status "
    "<> 'closed'",
    "DELETE FROM logs WHERE created_at < CURRENT_DATE - INTERVAL '30 days'",
    "CREATE TABLE IF NOT EXISTS public.orders (id bigserial PRIMARY KEY, customer_id int NOT NULL "
    "REFERENCES customers(id) ON DELETE CASCADE, total numeric(10,2) DEFAULT 0.00, note text, "
    "created timestamp with time zone DEFAULT now());",
    "CREATE OR REPLACE FUNCTION add(a integer, b integer) RETURNS integer AS $$ BEGIN RETURN a + b;"
    " END; $$ LANGUAGE plpgsql IMMUTABLE;",
    "CREATE FUNCTION f() RETURNS trigger AS $body$\nDECLARE\n  r record;\nBEGIN\n  FOR r IN "
    "SELECT * FROM t LOOP\n    IF r.x > 0 THEN\n      INSERT INTO u VALUES (r.x);\n"
    "    ELSIF r.x < 0 THEN\n      CONTINUE;\n    END IF;\n  END LOOP;\n  RETURN NEW;\nEND;\n"
    "$body$ LANGUAGE plpgsql;",
    "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 100) SELECT sum(n) "
    "FROM t;",
    "SELECT CASE WHEN a = 1 THEN 'one' WHEN a = 2 THEN 'two' ELSE 'many' END AS label, CAST(b AS "
    "varchar(10)), c::int, d::timestamp AT TIME ZONE 'UTC' FROM t",
    "SELECT data->>'name', data->'tags'->0, data #> '{a,b}', data #>> '{a}', data @> '{\"x\":1}', "
    "tags <@ ARRAY['a'], data ? 'k', data ?| array['a','b'], data ?& array['c'], data #- '{a}' "
    "FROM docs",
    "select * from t where name like 'a%' and descr not ilike '%b' and code regexp '^[0-9]+$' and "
    "x not regexp binary 'y' and z rlike 'w'",
    "SELECT TOP 10 [Order ID], [Customer Name] FROM [dbo].[Orders] WHERE [Total] > 100\nGO\n"
    "SELECT 1\nGO 2",
    "BEGIN TRANSACTION; UPDATE t SET a = 1; COMMIT; BEGIN; DELETE FROM t; ROLLBACK; BEGIN WORK; "
    "SELECT 1; END;",
    "DECLARE @x int = 5; SET @x = @x + 1; SELECT @x, @@ROWCOUNT, #temp.col FROM ##global_temp;",
    "SELECT `user`.`id`, `order`.`total` FROM `user` JOIN `order` USING (`id`) WHERE "
    "`user`.`name` = 'x' -- trailing\n",
    "select a from b where c = ? and d = %s and e = %(name)s and f = :param and g = $1 and h = ?1",
    "/* header */ SELECT /*+ INDEX(t idx) */ a FROM t -- end\n# mysql comment\nSELECT 2 #  c\n",
    "SELECT e.name, d.name FROM emp e NATURAL JOIN dept d CROSS JOIN loc l STRAIGHT_JOIN x FULL "
    "JOIN y ON 1=1 RIGHT JOIN z ON true",
    "CREATE PROCEDURE p() BEGIN DECLARE i INT DEFAULT 0; WHILE i < 10 DO SET i = i + 1; END WHILE; "
    "REPEAT SET i = i - 1; UNTIL i = 0 END REPEAT; END",
    "SELECT explode(arr) FROM t LATERAL VIEW EXPLODE(arr) a AS x LATERAL VIEW posexplode(m) b AS p,"
    " v",
    "SELECT 1.5, .5, 5., -0x1F, 0xABCDEF, 1E-3, 1.5e+10, 123abc, 1.5a, 1_000, 'str', N'nat', "
    "E'\\n', B'0101', X'FF'",
    "SELECT a FROM t WHERE x BETWEEN 1 AND 10 AND y IS NOT NULL AND z IN (SELECT id FROM u WHERE "
    "u.k = t.k) AND EXISTS (SELECT 1) HAVING count(*) > 1",
    "ALTER TABLE t ADD COLUMN c double precision, DROP CONSTRAINT k, ALTER COLUMN d SET DATA TYPE "
    "varchar(20) USING d::varchar",
    "CREATE INDEX CONCURRENTLY idx ON t USING btree (lower(name) text_pattern_ops) WHERE "
    "deleted_at IS NULL;",
    "SELECT x FROM t ORDER BY a ASC, b DESC, c NULLS FIRST, d ASC NULLS LAST",
    "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET v = s.v WHEN NOT MATCHED "
    "THEN INSERT (id, v) VALUES (s.id, s.v);",
    "SELECT array[1,2,3][1], arr[2:3], (a).b, f(x)[1], t.col[0] FROM t",
    "COPY t (a, b) FROM STDIN WITH (FORMAT csv);\n\\copy t to 'f.csv'\n\\dt",
    "SELECT name, salary, RANK() OVER w, LAG(salary, 1) OVER (ORDER BY salary) FROM emp WINDOW w "
    "AS (PARTITION BY dept ORDER BY salary DESC)",
    "IF EXISTS (SELECT 1 FROM t) BEGIN UPDATE t SET a = 1 END ELSE BEGIN INSERT INTO t VALUES (1) "
    "END",
    "select date '2020-01-01', timestamp '2020-01-01 00:00', interval '1 day', time '12:00' from "
    "dual",
    "SELECT a := 1, b = 2, c == 3, d != 4, e <> 5, f >= 6, g <= 7, h <=> 8, i || 'x', j ~ 'y', "
    "k !~* 'z' FROM t",
    "SELECT * FROM t1 UNION ALL SELECT * FROM t2 UNION SELECT * FROM t3 INTERSECT SELECT * FROM t4 "
    "EXCEPT SELECT * FROM t5",
    "CREATE VIEW v AS SELECT a, b FROM t; DROP VIEW IF EXISTS v CASCADE; TRUNCATE TABLE t RESTART "
    "IDENTITY;",
    'select "Weird ""Name""", \'it\'\'s\', \'a\\\'b\' from "Schema"."Table"',
    "SELECT caf\xe9, \u0438\u043c\u044f, \u540d\u524d, \u03a9mega FROM donn\xe9es WHERE \u0131d = 1"
    " AND \u017ftatus = 'ok' AND \u212aey = 2",
    "select\r\n  a,\r\n  b\r\nfrom\tt\x0bwhere\x0cx\xa0=\u30001",
    "CREATE TRIGGER trg AFTER INSERT ON t FOR EACH ROW EXECUTE PROCEDURE f();",
    "SELECT GROUP_CONCAT(DISTINCT a ORDER BY a SEPARATOR ','), IF(x > 0, 'p', 'n'), IFNULL(y, 0) "
    "FROM t",
    "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO app_user; REVOKE ALL ON t FROM PUBLIC;",
    "SELECT a FROM t FOR UPDATE SKIP LOCKED; LOCK TABLE t IN ACCESS EXCLUSIVE MODE;",
    "CREATE TABLE t (a INT AUTO_INCREMENT, b ENUM('x','y'), PRIMARY KEY (a)) ENGINE=InnoDB DEFAULT"
    " CHARSET=utf8mb4;",
    "SELECT * FROM TABLE(FLATTEN(input => parse_json(col))) f, LATERAL (SELECT 1) l QUALIFY "
    "ROW_NUMBER() OVER (ORDER BY 1) = 1",
    "DO $$ BEGIN IF NOT EXISTS (SELECT 1) THEN RAISE NOTICE 'x'; END IF; END $$;",
    "SELECT 1; ; ;SELECT 2;\n\n-- only a comment\n;",
    "CASE x WHEN 1 THEN BEGIN SELECT 1; END; END CASE;",
    "SELECT\n    a,\n    b\nFROM\n    t\nWHERE\n    c = 1\n    AND d = 2\n",
    "select a, b, c, d, e, f, g, h, i, j, k, l, m, n from aaaaaaaaaa, bbbbbbbbbb, cccccccccc where "
    "aaaaaaaaaa.x = bbbbbbbbbb.y and cccccccccc.z in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)",
    "insert into t values (1, 'a'), (2, 'b'), (3, 'c'); insert into t (a) select a from u",
    'SELECT a AS "x", b "y", c z, d AS e.f, (SELECT 1) AS g, h::text AS i FROM t AS u',
    "SELECT * FROM t WHERE a = 1 -- c1\n  AND b = 2 /* c2 */ OR c = 3\n-- c3\nORDER BY 1",
    "SELECT x FROM t WHERE y = 'O''Reilly' AND z = 'line1\nline2' AND w = 'abcdefghijklmnopqr'",
    "SELECT sum(a) FILTER (WHERE b > 0) OVER (PARTITION BY c), percentile_cont(0.5) WITHIN GROUP "
    "(ORDER BY d) FROM t GROUP BY ROLLUP (a, b), CUBE (c)",
    "UPDATE t SET (a, b) = (SELECT x, y FROM u WHERE u.id = t.id) WHERE EXISTS (SELECT 1 FROM u)",
    "CREATE TABLE t (id INT, CONSTRAINT pk PRIMARY KEY (id), CHECK (id > 0)) PARTITION BY RANGE "
    "(id)",
]


def _upstream_files() -> list[str]:
    d = Path(__file__).resolve().parents[2] / "parity" / "data" / "sqlparse-0.6.1" / "tests"
    out = []
    for p in sorted((d / "files").glob("*.sql")) if d.is_dir() else []:
        try:
            out.append(p.read_bytes().decode("utf-8"))
        except UnicodeDecodeError:
            continue
    return out


UPSTREAM_FILES = _upstream_files()
SEEDS = REAL_SQL + [s for s in UPSTREAM_FILES if len(s) <= 4000]


@st.composite
def mutated(draw: Any) -> str:
    """A real statement with random spans deleted, duplicated, recased or inserted."""
    s = draw(st.sampled_from(SEEDS))
    for _ in range(draw(st.integers(0, 5))):
        op = draw(st.integers(0, 5))
        i = draw(st.integers(0, len(s)))
        j = draw(st.integers(i, min(len(s), i + 25)))
        if op == 0:
            s = s[:i] + s[j:]
        elif op == 1:
            s = s[:i] + s[i:j] * draw(st.integers(2, 4)) + s[j:]
        elif op == 2:
            s = s[:i] + draw(leaf) + s[i:]
        elif op == 3:
            s = s[:i] + s[i:j].swapcase() + s[j:]
        elif op == 4:
            s = s[:i] + draw(whitespace) + s[j:]
        else:
            s = s[:i] + draw(structured) + s[j:]
    return str(s)


raw_text = st.lists(
    st.one_of(st.characters(), st.sampled_from(WEIRD + WHITESPACE)), max_size=40
).map("".join)

sql_text = st.one_of(statements(), soup(), mutated(), raw_text)

EXAMPLES = [
    "",
    "$$$$",
    "\u0131f exists",
    "\u0130F NOT EXISTS t",
    "select \u017fum(x) from \u212aelvin",
    "SELECT 'a\\'b', 'it''s' FROM t",
    "/*+ hint */ select 1 /* unclosed",
    "x\x1cy\x1d z\x85",
    "select '\ud800' as a, \udfff\U0010f800 from t",
    "$tag$ a $tag$ $tag$",
    "SELECT 1.5a, 123abc, .5, -0x1F, 1E-3",
    "begin transaction; select 1; commit;",
    "BEGIN IF x THEN SELECT 1; END IF; END; SELECT 2",
    "select a from t\nGO\nselect b\nGO 2\n",
]


# ---- tokenize, parse, split ---------------------------------------------------------------------


@fuzz(150)
@given(text=sql_text)
@example(text="")
@example(text="%\x00")  # sqlparse: Operator '%', Error '\x00'
@example(text="LO\x00N x")
def test_tokenize(text: str) -> None:
    same(tokenize_out(ps, text), tokenize_out(sqlparse, text), "lexer.tokenize", text)


_SEED_TEXTS = EXAMPLES + REAL_SQL + UPSTREAM_FILES


@pytest.mark.parametrize("text", _SEED_TEXTS, ids=[f"seed{i}" for i in range(len(_SEED_TEXTS))])
def test_seeds(text: str) -> None:
    same(tokenize_out(ps, text), tokenize_out(sqlparse, text), "lexer.tokenize", text)
    same(parse_out(ps, text), parse_out(sqlparse, text), "parse", text)
    for semi in (False, True):
        got = call(ps.split, text, strip_semicolon=semi)
        same(got, call(sqlparse.split, text, strip_semicolon=semi), f"split({semi})", text)
    for opts in FORMAT_SETS:
        got = call(ps.format, text, **opts)
        same(got, call(sqlparse.format, text, **opts), f"format(**{opts!r})", text)


@fuzz(120)
@given(text=sql_text)
def test_parse(text: str) -> None:
    same(parse_out(ps, text), parse_out(sqlparse, text), "parse", text)


@fuzz(150)
@given(text=sql_text)
def test_split(text: str) -> None:
    for semi in (False, True):
        got = call(ps.split, text, strip_semicolon=semi)
        same(got, call(sqlparse.split, text, strip_semicolon=semi), f"split({semi})", text)


# ---- format -------------------------------------------------------------------------------------

_BOOL_OPTS = (
    "strip_comments",
    "use_space_around_operators",
    "strip_whitespace",
    "indent_columns",
    "reindent",
    "reindent_aligned",
    "indent_after_first",
    "indent_tabs",
    "comma_first",
    "compact",
)
_CASES = [None, "upper", "lower", "capitalize"]
VALID_OPTIONS = {
    **{k: st.booleans() for k in _BOOL_OPTS},
    "keyword_case": st.sampled_from(_CASES),
    "identifier_case": st.sampled_from(_CASES),
    "output_format": st.sampled_from([None, "sql", "python", "php"]),
    "truncate_strings": st.sampled_from([None, 2, 3, 5, 10, 40]),
    "truncate_char": st.sampled_from(["[...]", "", "~", "\u2026", "\U0010f800", "''"]),
    "indent_width": st.integers(1, 8),
    "wrap_after": st.sampled_from([0, 1, 5, 10, 20, 80]),
}
_ANY_BOOL: list[Any] = [True, False, 0, 1, 1.0, None, "yes", 2, "", [], "True"]
ANY_OPTIONS = {
    **{k: st.sampled_from(_ANY_BOOL) for k in _BOOL_OPTS},
    "keyword_case": st.sampled_from([*_CASES, "UPPER", "title", "", 0, 1, "swapcase"]),
    "identifier_case": st.sampled_from([*_CASES, "Lower", "casefold", 2]),
    "output_format": st.sampled_from([None, "sql", "python", "php", "PHP", "json", "", 1]),
    "truncate_strings": st.sampled_from(
        [None, 2, 3, 10, "4", 1, 0, -3, 2.7, "x", True, False, [1], "1e3", " 5 ", 10**30]
    ),
    "truncate_char": st.sampled_from(["[...]", "", "\u2026", 5, None, b"x", "\ud800"]),
    "indent_width": st.sampled_from([1, 2, 4, "3", 0, -1, 2.5, "a", None, True, " 2 ", 10**6]),
    "wrap_after": st.sampled_from([0, 1, 10, "5", -1, "b", None, 3.9, True]),
    "right_margin": st.sampled_from([None, 10, 20, 79, "30", 9, 0, "x", 10.5, False]),
    "unknown_option": st.sampled_from([1, None]),
}
FORMAT_SETS: list[dict[str, Any]] = [
    {},
    {"reindent": True},
    {"reindent_aligned": True},
    {"keyword_case": "upper", "identifier_case": "lower"},
    {"strip_comments": True},
    {"use_space_around_operators": True},
    {"reindent": True, "comma_first": True, "wrap_after": 20},
    {"reindent": True, "indent_tabs": True, "indent_after_first": True, "indent_columns": True},
    {"reindent": True, "compact": True, "indent_width": 4},
    {"output_format": "python", "reindent": True},
    {"output_format": "php"},
    {"truncate_strings": 5},
    {"strip_whitespace": True, "keyword_case": "capitalize"},
]


@fuzz(100)
@given(text=sql_text, opts=st.fixed_dictionaries({}, optional=VALID_OPTIONS))
@example(text="'abc' \ud800", opts={"truncate_strings": 2, "truncate_char": "\U0010f800"})
def test_format(text: str, opts: dict[str, Any]) -> None:
    got = call(ps.format, text, **opts)
    same(got, call(sqlparse.format, text, **opts), f"format(**{opts!r})", text)


@fuzz(100)
@given(text=st.one_of(mutated(), soup()), opts=st.fixed_dictionaries({}, optional=ANY_OPTIONS))
@example(text="select 'abc', a", opts={"truncate_strings": 2**64})
@example(text="select 'abc', a", opts={"reindent": True, "indent_width": 2**63})
@example(text="select 'abc', a", opts={"reindent": True, "wrap_after": 2**63})
def test_format_any_options(text: str, opts: dict[str, Any]) -> None:
    got = call(ps.format, text, **opts)
    same(got, call(sqlparse.format, text, **opts), f"format(**{opts!r})", text)


@fuzz(80)
@given(text=mutated(), i=st.integers(0, len(FORMAT_SETS) - 1))
def test_format_sets(text: str, i: int) -> None:
    opts = FORMAT_SETS[i]
    got = call(ps.format, text, **opts)
    same(got, call(sqlparse.format, text, **opts), f"format(**{opts!r})", text)


# ---- streams, bytes and argument types ----------------------------------------------------------

_FEEDS = ("str", "stringio", "utf8", "utf8-enc", "utf16", "latin1-raw", "latin1-enc", "ascii-enc")


def _feed(text: str, how: str) -> tuple[Any, str | None]:
    if how == "str":
        return text, None
    if how == "stringio":
        return io.StringIO(text), "utf-8"
    if how == "utf8":
        return text.encode("utf-8", "surrogatepass"), None
    if how == "utf8-enc":
        return text.encode("utf-8", "surrogatepass"), "utf-8"
    if how == "utf16":
        return text.encode("utf-16", "surrogatepass"), "utf-16"
    if how == "latin1-raw":  # not UTF-8 in general: sqlparse decodes it as unicode-escape
        return text.encode("latin-1", "replace"), None
    if how == "latin1-enc":
        return text.encode("latin-1", "replace"), "latin-1"
    return text.encode("utf-8", "surrogatepass"), "ascii"


API = {
    "tokenize": lambda lib, x, enc: tokenize_out(lib, x, enc),
    "parse": lambda lib, x, enc: parse_out(lib, x, enc),
    "parsestream": lambda lib, x, enc: parsestream_out(lib, x, enc),
    "split": lambda lib, x, enc: call(lib.split, x, enc),
    "format": lambda lib, x, enc: call(lib.format, x, enc, reindent=True),
}


@fuzz(100)
@given(text=sql_text, how=st.sampled_from(_FEEDS), api=st.sampled_from(sorted(API)))
@example(text="select '\\x' from t \xe9", how="latin1-raw", api="parse")
@example(text="select '\\N{x}' from t \xe9", how="latin1-raw", api="split")
def test_streams_and_bytes(text: str, how: str, api: str) -> None:
    fn = API[api]
    got = fn(ps, *_feed(text, how))
    same(got, fn(sqlparse, *_feed(text, how)), f"{api} on {how}", text)


BAD_INPUTS = [
    None,
    1,
    1.5,
    bytearray(b"select 1"),
    memoryview(b"select 1"),
    ["select 1"],
    ("select", 1),
    object,
]


@pytest.mark.parametrize("api", sorted(API))
@pytest.mark.parametrize(
    "make",
    [*BAD_INPUTS, "bytesio", "nope-encoding", "str-subclass"],
    ids=[*(type(x).__name__ for x in BAD_INPUTS), "bytesio", "nope-encoding", "str-subclass"],
)
def test_argument_types(api: str, make: Any) -> None:
    fn = API[api]

    def arg() -> tuple[Any, str | None]:
        if make == "bytesio":
            return io.BytesIO(b"select 1"), None
        if make == "nope-encoding":
            return b"select 1", "nope"
        if make == "str-subclass":
            return type("S", (str,), {})("select a from b"), None
        return make, None

    same(fn(ps, *arg()), fn(sqlparse, *arg()), f"{api}({make!r})", make)


# ---- the DoS limits -----------------------------------------------------------------------------

_GROUPING_MODULES = [sqlparse.engine.grouping, ps.engine.grouping]


@contextlib.contextmanager
def limits(depth: Any, tokens: Any) -> Iterator[None]:
    saved = [(m, m.MAX_GROUPING_DEPTH, m.MAX_GROUPING_TOKENS) for m in _GROUPING_MODULES]
    try:
        for m in _GROUPING_MODULES:
            m.MAX_GROUPING_DEPTH = depth
            m.MAX_GROUPING_TOKENS = tokens
        yield
    finally:
        for m, d, t in saved:
            m.MAX_GROUPING_DEPTH = d
            m.MAX_GROUPING_TOKENS = t


NESTS = {
    "paren": ("(", ")"),
    "bracket": ("[", "]"),
    "func": ("f(", ")"),
    "case": ("CASE WHEN 1 THEN ", " END"),
    "begin": ("BEGIN ", " END"),
    "if": ("IF x THEN ", " END IF"),
    "for": ("FOR x IN y LOOP ", " END LOOP"),
    "sub": ("(SELECT a FROM ", " AS t)"),
}


def nested(kinds: list[str], core: str = "x") -> str:
    head = "".join(NESTS[k][0] for k in kinds)
    tail = "".join(NESTS[k][1] for k in reversed(kinds))
    return f"SELECT {head}{core}{tail} FROM t"


LIMIT_CHECKS = {
    "parse": lambda lib, x: parse_out(lib, x),
    "parsestream": lambda lib, x: parsestream_out(lib, io.StringIO(x)),
    "split": lambda lib, x: call(lib.split, x),
    "format-reindent": lambda lib, x: call(lib.format, x, reindent=True),
    "format-aligned": lambda lib, x: call(lib.format, x, reindent_aligned=True),
    "format-strip_comments": lambda lib, x: call(lib.format, x, strip_comments=True),
    "format-keyword_case": lambda lib, x: call(lib.format, x, keyword_case="upper"),
}


def _check_limits(text: str, checks: Any = LIMIT_CHECKS) -> None:
    for name, fn in checks.items():
        same(fn(ps, text), fn(sqlparse, text), name, text)


LIMIT_VALUES = [None, 0, 1, 2, 3, 5, 8, -1, True, False, 2.5, "3", 10**20]


@pytest.mark.thread_unsafe(reason="sets the grouping limits on both libraries' modules")
@fuzz(80)
@given(
    depth=st.sampled_from(LIMIT_VALUES),
    tokens=st.sampled_from([*LIMIT_VALUES, 10, 20, 50]),
    text=st.one_of(
        st.lists(st.sampled_from(sorted(NESTS)), max_size=10).map(nested),
        statements(),
        mutated(),
    ),
)
def test_limits(depth: Any, tokens: Any, text: str) -> None:
    with limits(depth, tokens):
        _check_limits(text)


@pytest.mark.thread_unsafe(reason="deep inputs (the stack of a thread is small)")
@pytest.mark.parametrize("kind", [*sorted(NESTS), "mixed"])
@pytest.mark.parametrize("n", [99, 100, 101, 102])
def test_depth_limit_boundary(kind: str, n: int) -> None:
    order = sorted(NESTS)
    kinds = [order[i % len(order)] for i in range(n)] if kind == "mixed" else [kind] * n
    _check_limits(nested(kinds))


def _flat(n: int) -> str:
    """A statement of exactly `n` tokens."""
    return ("x " * (n // 2) + "x" * (n % 2)).rstrip(" ") + (" " if n % 2 == 0 else "")


@pytest.mark.thread_unsafe(reason="10,000-token inputs")
@pytest.mark.parametrize("n", [9999, 10000, 10001])
def test_token_limit_boundary(n: int) -> None:
    text = _flat(n)
    assert len(list(sqlparse.lexer.tokenize(text))) == n
    _check_limits(text, {k: LIMIT_CHECKS[k] for k in ("parse", "format-reindent", "split")})


@pytest.mark.thread_unsafe(reason="sets the grouping limits on both libraries' modules")
@pytest.mark.parametrize(
    ("depth", "tokens", "text"),
    [
        (None, None, nested(["paren"] * 150)),
        (None, None, nested(["paren"] * 2000)),  # sqlparse: RecursionError -> SQLParseError
        (None, None, nested(["case", "func"] * 1000)),
        (None, 10000, nested(["paren"] * 300)),
        (200, None, nested(["paren"] * 250)),
        (None, None, _flat(12001)),
        (None, None, "SELECT " + ", ".join(["a"] * 1500) + " FROM t"),
        (5, 3, "SELECT a FROM t; SELECT (b) FROM u; SELECT ((c)) FROM v"),
    ],
    ids=["150", "2000", "case-func-2000", "300", "250-of-200", "flat-12001", "list-1500", "small"],
)
def test_limits_off(depth: Any, tokens: Any, text: str) -> None:
    with limits(depth, tokens):
        _check_limits(text, {k: LIMIT_CHECKS[k] for k in ("parse", "parsestream", "split")})
        _check_limits(text, {"format-reindent": LIMIT_CHECKS["format-reindent"]})


# ---- a customised lexer -------------------------------------------------------------------------

CUSTOM_PATTERNS = [
    r"\w+", r"x*", r"(?=a)", r"[0-9]+", r"FOO\b", r"[^\s]+", r"\s+", r".", r"'[^']*'",
    r"--[^\n]*", r"(?>ab|a)c", r"a++", r"(?i:select)", r"(?x) a  b", r"\N{LATIN SMALL LETTER A}",
    r"(?<=a)b", r"(a)(?(1)b|c)", r"$", r"\Z", "\u0131+", "[\u212a]", "[A-Z\xc0-\xdc]+",
    r"(\w)\1", r"\$\w*\$", r"/\*.*?\*/", r"(?s)/\*.*?\*/", r"(?m)^\w+", r"[^\W\d_]+",
    r"\d+(?:\.\d+)?", r"[\u0131]+",
]  # fmt: skip
CUSTOM_TTYPES = [
    T.Keyword, T.Keyword.DML, T.Keyword.DDL, T.Name, T.Name.Builtin, T.Name.Placeholder,
    T.Literal.String.Single, T.Literal.Number.Integer, T.Comment.Single, T.Comment.Multiline,
    T.Punctuation, T.Whitespace, T.Newline, T.Error, T.Operator, T.Operator.Comparison, T.Text,
    T.Keyword.Order, T.Keyword.CTE, T.Wildcard, T.Assignment,
]  # fmt: skip
CUSTOM_ACTIONS = [*CUSTOM_TTYPES, SK.PROCESS_AS_KEYWORD, "not an action"]


def _own_tt(tt: Any) -> Any:
    """pyroid's token type or keyword marker for one of sqlparse's."""
    if tt is SK.PROCESS_AS_KEYWORD:
        return ps.keywords.PROCESS_AS_KEYWORD
    if isinstance(tt, sqlparse.tokens._TokenType):
        return functools.reduce(getattr, tt, ps.tokens.Token)
    return tt


def _own_dict(d: dict[str, Any]) -> dict[str, Any]:
    for n in _KW_DICT_NAMES:
        if d is getattr(SK, n):
            return getattr(ps.keywords, n)  # type: ignore[no-any-return]
    return {k: _own_tt(v) for k, v in d.items()}


@contextlib.contextmanager
def custom_lexer(table: list[tuple[str, Any]], kwdicts: list[dict[str, Any]]) -> Iterator[None]:
    lexers = [(sqlparse.lexer.Lexer.get_default_instance(), lambda x: x, lambda d: d)]
    if not SHARED:
        lexers.append((ps.lexer.Lexer.get_default_instance(), _own_tt, _own_dict))
    try:
        for lex, conv, conv_dict in lexers:
            lex.clear()
            lex.set_SQL_REGEX([(rx, conv(tt)) for rx, tt in table])
            for d in kwdicts:
                lex.add_keywords(conv_dict(d))
        yield
    finally:
        for lex, _, _ in lexers:
            lex.default_initialization()


def _describe_table(table: list[tuple[str, Any]]) -> Any:
    """`table`, with sqlparse's own entries as their index in SQL_REGEX."""
    return [
        f"SQL_REGEX[{SK.SQL_REGEX.index(e)}]" if e in SK.SQL_REGEX else (e[0], repr(e[1]))
        for e in table
    ]


def _describe_dict(d: dict[str, Any]) -> Any:
    for n in _KW_DICT_NAMES:
        if d is getattr(SK, n):
            return n
    return {k: repr(v) for k, v in d.items()}


@st.composite
def lexer_tables(draw: Any) -> list[tuple[str, Any]]:
    custom = st.tuples(st.sampled_from(CUSTOM_PATTERNS), st.sampled_from(CUSTOM_ACTIONS))
    if draw(st.booleans()):  # sqlparse's table with a few entries added or dropped
        table = list(SK.SQL_REGEX)
        for _ in range(draw(st.integers(0, 3))):
            i = draw(st.integers(0, len(table)))
            if draw(st.booleans()) and i < len(table):
                del table[i]
            else:
                table.insert(i, draw(custom))
        return table
    table_: list[tuple[str, Any]] = draw(
        st.lists(st.one_of(st.sampled_from(SK.SQL_REGEX), custom), max_size=12)
    )
    return table_


_KW_CUSTOM_WORDS = [
    "FOO", "SELECT", "FROM", "X", "ID", "\u0130D", "\u017fELECT", "SUM", "1", "BEGIN", "END",
]  # fmt: skip
kw_dicts = st.lists(
    st.one_of(
        st.sampled_from([getattr(SK, n) for n in _KW_DICT_NAMES]),
        st.dictionaries(st.sampled_from(_KW_CUSTOM_WORDS), st.sampled_from(CUSTOM_TTYPES)),
    ),
    max_size=4,
)


@pytest.mark.thread_unsafe(reason="changes the Lexer singleton")
@fuzz(80)
@given(table=lexer_tables(), kwdicts=kw_dicts, text=st.one_of(mutated(), soup()))
@example(table=[(r"x*", T.Name)], kwdicts=[], text="a")  # sqlparse yields (Name, '') first
@example(table=[(r"x*", T.Keyword)], kwdicts=[], text="a")  # IndexError in the splitter
@example(
    table=[(r"(?m)\Ax", T.Keyword), (r"[^x]", T.Name), ("x", T.Error)], kwdicts=[], text="y\nx"
)
def test_custom_lexer(
    table: list[tuple[str, Any]], kwdicts: list[dict[str, Any]], text: str
) -> None:
    data = (_describe_table(table), [_describe_dict(d) for d in kwdicts], text)
    with custom_lexer(table, kwdicts):
        same(tokenize_out(ps, text), tokenize_out(sqlparse, text), "lexer.tokenize", data)
        same(parse_out(ps, text), parse_out(sqlparse, text), "parse", data)
        same(call(ps.split, text), call(sqlparse.split, text), "split", data)
        opts = {"reindent": True, "keyword_case": "upper"}
        same(call(ps.format, text, **opts), call(sqlparse.format, text, **opts), "format", data)


# ---- the regex engine ---------------------------------------------------------------------------

SQL_PATTERNS = [rx for rx, _ in SK.SQL_REGEX]
OWN_PATTERNS = [
    SK._DOLLAR_QUOTE_DELIM.pattern, SK._DOLLAR_QUOTE_OPENER_OK.pattern, r"/\*\+", r"/\*(?!\+)",
    r"\*/", r"(?<=\.)\w+", "\\$(?:[_A-Z\xc0-\xdc]\\w*)?\\$", r"[^\W\d]+", r"\bAND\b",
    r"(a|ab)(c|bcd)(d*)", r".*?$", "[\u0131-\u017f]+", r"[\u0131-\u017f]+", r"[k]", r"[K]",
    "[\u212a]", "[\u212b]", r"[^a-z]+", r"x{2,3}?", r"\Bx", r"^a", r"a$", r"\Z", r"\A\w",
    r"(?!a)\w", r"[\s\S]*?\*/", r"(?:x|y)+?z", r"(\w)\1", r"[-a]", r"[a\-z]+", r"[\]]", r"[\d.]+",
    "[\xc0-\xdc\xe0-\xfc]+", "[\xb5]", "[\u03a3\u03c3\u03c2]", "\u01c5", "\ufb00", "\u0130",
    "\xdf", "\u017f", "[^\xc0-\xdc]", r"\W+", r"\D+", r"\S+", r"[\w$#]+", r"(?<![\w\])])\[",
    r"[^']*'", r"\s*?x", r"(?:ab)*?c", r"(a|b|)+", r"(?:a?)*b", r"[\x00-\x1f]+",
    "[\u0300-\u036f]", r"\d+E-?\d+", r"[\xc0-\xdc]", r"\u212a", r"\U0001d518",
]  # fmt: skip
TRICKY = (
    "\u0131\u017f\u0130iIsSkK\u212a\u212b\xc5\xe5\xdf\u1e9e\xb5\u03bc\u039c\u03a3\u03c3\u03c2"
    "\xd7\xf7\xc0\xdc\xe0\xfc\xff\u0178\u01c5\u01c4\u01c6\ufb00\ufb01\u0301\u0308\u0438\u0416"
    "\u540d\u0663\u096a\u07c0\U0001d7d8\U0001d518\x1c\x1d\x1e\x1f\x85\xa0\u2028\u3000\t\n\r"
    "\x0b\x0c \x00\x01\x7f\u03b2\u03d0\u0412\u1c80\ua64b\u1e61\u1e9b\ufb05\ufb06\u1fbe\u0345"
)
RX_CHARS = TRICKY + "abcdefxyz0189_-.()'\"`\xb4$#@:?%\\*/+<>=!~|^&[],;"
rx_text = st.one_of(
    st.text(alphabet=RX_CHARS, max_size=16),
    st.lists(st.one_of(leaf, st.sampled_from(list(TRICKY))), max_size=8).map("".join),
    mutated(),
).filter(lambda s: not any("\ud800" <= c <= "\udfff" for c in s))
FLAGS = re.IGNORECASE | re.UNICODE


def _regex_same(pattern: str, flags: int, text: str, pos: int) -> bool:
    """Whether pyroid's engine took `pattern` (it then must agree with ``re``)."""
    got = native._regex_match(pattern, flags, text, pos, UNIDATA)
    if got is NotImplemented:
        return False
    m = re.compile(pattern, flags).match(text, pos)
    want = None if m is None else m.end()
    if got != want:
        pytest.fail(
            f"re.compile({pattern!r}, {flags!r}).match({text!r}, {pos}): "
            f"pyroid {got!r}, re {want!r}",
            pytrace=False,
        )
    return True


@fuzz(400)
@given(pattern=st.sampled_from(SQL_PATTERNS + OWN_PATTERNS), text=rx_text, data=st.data())
def test_regex_table_patterns(pattern: str, text: str, data: Any) -> None:
    pos = data.draw(st.integers(0, len(text)))
    taken = _regex_same(pattern, FLAGS, text, pos)
    assert taken or pattern not in SQL_PATTERNS, f"pyroid's engine refused {pattern!r}"


@pytest.mark.parametrize("pattern", SQL_PATTERNS)
def test_regex_table_patterns_everywhere(pattern: str) -> None:
    for text in [*SEEDS[:8], RX_CHARS, TRICKY * 2, " ".join(PHRASES)]:
        clean = "".join(c for c in text if not "\ud800" <= c <= "\udfff")
        for pos in range(len(clean) + 1):
            assert _regex_same(pattern, FLAGS, clean, pos)


REGEX_CASES = [
    ("s", FLAGS, "\x00", 0),
    ("\u03c3", FLAGS, "\x00", 0),
    ("\xb5", FLAGS, "\x00", 0),
    ("[s]", FLAGS, "\x00", 0),
    (r"%(\(\w+\))?s", FLAGS, "%\x00", 0),
    (r"(CASE|IN|VALUES|USING|FROM|AS)\b", FLAGS, "\x00N", 0),
    (r"\A", re.MULTILINE | re.UNICODE, "a\nb", 2),
    (r"^", re.MULTILINE | re.UNICODE, "a\nb", 2),
    (r"\Z", re.MULTILINE | re.UNICODE, "a\nb", 1),
    (r"$", re.MULTILINE | re.UNICODE, "a\nb", 1),
]


@pytest.mark.parametrize(
    ("pattern", "flags", "text", "pos"), REGEX_CASES, ids=[str(i) for i in range(len(REGEX_CASES))]
)
def test_regex_cases(pattern: str, flags: int, text: str, pos: int) -> None:
    _regex_same(pattern, flags, text, pos)


RX_LITERALS = [*TRICKY, *"aAbB_1 .$'()-*+?[]{}|^\\"]
RX_CLASS_ITEMS = [
    "a-z", "A-Z", "\xc0-\xdc", "\xe0-\xfc", "0-9", r"\w", r"\W", r"\d", r"\D", r"\s", r"\S",
    "\u0131-\u017f", "\u0100-\u017f", "\u03a3-\u03c3", "\u2120-\u2130", r"\x00-\x1f", r"\-",
    r"\]", "_", "$", "#", "'", "k", "s", "i", r"\u212a", r"\xb5",
]  # fmt: skip


@st.composite
def rx_class(draw: Any) -> str:
    items = draw(
        st.lists(
            st.one_of(st.sampled_from(RX_CLASS_ITEMS), st.sampled_from(TRICKY).map(re.escape)),
            min_size=1,
            max_size=4,
        )
    )
    return "[" + ("^" if draw(st.booleans()) else "") + "".join(items) + "]"


rx_atom = st.one_of(
    st.sampled_from(RX_LITERALS).map(re.escape),
    rx_class(),
    st.sampled_from(
        [".", r"\w", r"\W", r"\d", r"\D", r"\s", r"\S", r"\b", r"\B", "^", "$", r"\A", r"\Z"]
    ),
)
RX_QUANT = ["*", "+", "?", "*?", "+?", "??", "{2}", "{1,3}", "{0,2}?", "{2,}", "{,2}"]


def _rx_extend(inner: Any) -> Any:
    return st.one_of(
        st.lists(inner, min_size=2, max_size=3).map("".join),
        st.tuples(inner, inner).map("|".join),
        inner.map(lambda x: f"({x})"),
        inner.map(lambda x: f"(?:{x})"),
        inner.map(lambda x: f"(?={x})"),
        inner.map(lambda x: f"(?!{x})"),
        st.tuples(inner, st.sampled_from(RX_QUANT)).map(lambda t: f"(?:{t[0]}){t[1]}"),
        st.tuples(rx_atom, st.sampled_from(RX_QUANT)).map("".join),
        st.tuples(st.sampled_from(["(?<=", "(?<!"]), st.one_of(rx_class(), rx_atom), inner).map(
            lambda t: f"{t[0]}{t[1]}){t[2]}"
        ),
        st.tuples(inner, inner).map(lambda t: f"({t[0]}){t[1]}\\1"),
    )


rx_pattern = st.recursive(rx_atom, _rx_extend, max_leaves=8)


@fuzz(400)
@given(
    pattern=rx_pattern,
    flags=st.sampled_from([0, re.I, re.M, re.S, re.I | re.M | re.S, re.I | re.S]),
    text=st.text(alphabet=RX_CHARS, max_size=12),
    data=st.data(),
)
def test_regex_random_patterns(pattern: str, flags: int, text: str, data: Any) -> None:
    try:
        re.compile(pattern, flags)
    except re.error:
        assume(False)
    pos = data.draw(st.integers(0, len(text)))
    _regex_same(pattern, flags | re.UNICODE, text, pos)


# ---- mixed: the real API on mutated real SQL -----------------------------------------------------


@fuzz(60)
@given(text=mutated())
def test_mutated_real_sql(text: str) -> None:
    same(parse_out(ps, text), parse_out(sqlparse, text), "parse", text)
    same(call(ps.split, text), call(sqlparse.split, text), "split", text)
    for opts in ({"reindent": True}, {"reindent_aligned": True, "keyword_case": "lower"}):
        same(call(ps.format, text, **opts), call(sqlparse.format, text, **opts), "format", text)


def test_module_surface() -> None:
    """The names sqlparse's modules define exist on pyroid's."""
    for name in ("parse", "parsestream", "format", "split"):
        assert callable(getattr(ps, name))
    for mod in ("sql", "tokens", "keywords", "lexer", "engine", "filters", "formatter", "utils"):
        real = sys.modules[f"sqlparse.{mod}"]
        mine = getattr(ps, mod)
        own = [
            n
            for n in dir(real)
            if not n.startswith("_")
            and (n.isupper() or getattr(getattr(real, n), "__module__", None) == real.__name__)
        ]
        missing = [n for n in own if not hasattr(mine, n)]
        assert not missing, f"pyroid.sqlparse.{mod} lacks {missing}"
