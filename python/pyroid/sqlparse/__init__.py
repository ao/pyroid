"""sqlparse 0.6.1 with a Rust core: ``parse``, ``parsestream``, ``format`` and ``split``.

``pyroid.sqlparse`` gives the output of sqlparse 0.6.1 (the final sqlparse release): the same
tokens, the same token trees, the same formatted text and the same exceptions, including the
DoS limits ``engine.grouping.MAX_GROUPING_DEPTH`` and ``MAX_GROUPING_TOKENS``.

The trees are objects of sqlparse's own classes when sqlparse 0.5.0 or newer is installed
(``isinstance(t, sqlparse.sql.Identifier)``, ``t.ttype is sqlparse.tokens.Keyword`` and
``except sqlparse.exceptions.SQLParseError`` work on them), else of pyroid's copies of
sqlparse 0.6.1's classes. The submodules (``sql``, ``tokens``, ``keywords``, ``lexer``,
``engine``, ``filters``, ``formatter``, ``utils``, ``exceptions``, ``cli``) are sqlparse
0.6.1's, with the lexer, the grouping and the formatting filters running in Rust.

The modules ``engine``, ``filters``, ``formatter``, ``utils``, ``cli`` and the ``_own_*``
copies are vendored from sqlparse 0.6.1 (BSD-3-Clause, see ``LICENSES/``), with their imports
pointed at ``pyroid.sqlparse``, ``engine.grouping.group`` and ``lexer.tokenize`` running the
Rust port, and ``FilterStack`` taking a grouping function (``_group``).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

from pyroid.sqlparse import _backend, exceptions, keywords, sql, tokens
from pyroid.sqlparse import lexer
from pyroid.sqlparse import engine, filters, formatter, utils
from pyroid.sqlparse import cli
from pyroid.sqlparse._core import native as _native
from pyroid.sqlparse._core import warn_fallback as _warn_fallback

__version__ = "0.6.1"
#: The sqlparse release pyroid.sqlparse reproduces.
SQLPARSE_VERSION = _backend.SQLPARSE_VERSION

__all__ = [
    "SQLPARSE_VERSION",
    "cli",
    "engine",
    "exceptions",
    "filters",
    "format",
    "formatter",
    "keywords",
    "lexer",
    "parse",
    "parsestream",
    "split",
    "sql",
    "tokens",
    "utils",
]


# ---------------------------------------------------------------------------
# The pipelines, for a given grouping module (it holds the DoS limits)
# ---------------------------------------------------------------------------


def _python_stream(stream: Any, encoding: str | None, grouping: Any) -> Generator[Any, None, None]:
    stack = engine.FilterStack()
    stack.enable_grouping()
    stack._group = grouping._py_group
    yield from stack.run(stream, encoding)


def _group_python(stmt: Any, grouping: Any) -> Any:
    try:
        return grouping._py_group(stmt)
    except RecursionError as err:
        raise exceptions.SQLParseError("Maximum recursion depth exceeded") from err


def _parsestream(stream: Any, encoding: str | None, grouping: Any) -> Generator[Any, None, None]:
    lex = lexer.Lexer.get_default_instance()
    if not lexer.plain(lex):
        _warn_fallback("parsestream", "the default Lexer has overridden methods", 4)
        yield from _python_stream(stream, encoding, grouping)
        return
    text = lexer.read(stream, encoding)
    it = _native._parse(text, lex, grouping.MAX_GROUPING_DEPTH, grouping.MAX_GROUPING_TOKENS)
    if it is NotImplemented:
        yield from _python_stream(text, None, grouping)
        return
    for stmt in it:
        if type(stmt) is tuple:  # sqlparse's own grouping raises on this one
            yield _group_python(stmt[1], grouping)
        else:
            yield stmt


def _format(sql: Any, encoding: str | None, options: dict[str, Any], grouping: Any) -> str:
    lex = lexer.Lexer.get_default_instance()
    if lexer.plain(lex):
        text = lexer.read(sql, encoding)
        get = options.get
        truncate = (get("truncate_strings"), get("truncate_char")) if get("truncate_strings") else None
        reindent = None
        if get("reindent"):
            reindent = (
                options["indent_width"],
                options["indent_char"],
                options["wrap_after"],
                bool(options["comma_first"]),
                bool(options["indent_after_first"]),
                bool(options["indent_columns"]),
                bool(options["compact"]),
            )
        output = get("output_format")
        r = _native._format(
            text,
            lex,
            grouping.MAX_GROUPING_DEPTH,
            grouping.MAX_GROUPING_TOKENS,
            get("keyword_case") or None,
            get("identifier_case") or None,
            truncate,
            bool(get("use_space_around_operators", False)),
            bool(get("strip_comments")),
            bool(get("strip_whitespace") or get("reindent")),
            reindent,
            options["indent_char"] if get("reindent_aligned", False) else None,
            bool(get("right_margin")),
            output.lower() if output else None,
        )
        if r is not NotImplemented:
            return r  # type: ignore[no-any-return]
        sql, encoding = text, None
    else:
        _warn_fallback("format", "the default Lexer has overridden methods", 4)
    stack = engine.FilterStack()
    stack._group = grouping._py_group
    stack = formatter.build_filter_stack(stack, options)
    stack.postprocess.append(filters.SerializerUnicode())
    return "".join(stack.run(sql, encoding))


def _split(sql: Any, encoding: str | None, strip_semicolon: bool) -> list[str]:
    lex = lexer.Lexer.get_default_instance()
    if lexer.plain(lex):
        text = lexer.read(sql, encoding)
        r = _native._split(text, lex, bool(strip_semicolon))
        if r is not NotImplemented:
            return r  # type: ignore[no-any-return]
        sql, encoding = text, None
    else:
        _warn_fallback("split", "the default Lexer has overridden methods", 4)
    stack = engine.FilterStack(strip_semicolon=strip_semicolon)
    return [str(stmt).strip() for stmt in stack.run(sql, encoding)]


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


def parse(sql: Any, encoding: str | None = None) -> tuple[Any, ...]:
    """Parse sql and return a list of statements.

    Args:
        sql: A string containing one or more SQL statements.
        encoding: The encoding of the statement (optional).

    Returns:
        A tuple of ``sql.Statement`` instances.
    """
    return tuple(_parsestream(sql, encoding, engine.grouping))


def parsestream(stream: Any, encoding: str | None = None) -> Generator[Any, None, None]:
    """Parse sql statements from file-like object.

    Args:
        stream: A file-like object.
        encoding: The encoding of the stream contents (optional).

    Returns:
        A generator of ``sql.Statement`` instances.
    """
    return _parsestream(stream, encoding, engine.grouping)


def format(sql: Any, encoding: str | None = None, **options: Any) -> str:  # noqa: A001
    """Format *sql* according to *options*.

    Args:
        sql: The SQL text.
        encoding: The encoding of the statement (optional).
        **options: sqlparse's formatting options.

    Returns:
        The formatted SQL statement as string.
    """
    options = formatter.validate_options(options)
    return _format(sql, encoding, options, engine.grouping)


def split(sql: Any, encoding: str | None = None, strip_semicolon: bool = False) -> list[str]:
    """Split *sql* into single statements.

    Args:
        sql: A string containing one or more SQL statements.
        encoding: The encoding of the statement (optional).
        strip_semicolon: If True, remove trailing semicolons (default: False).

    Returns:
        A list of strings.
    """
    return _split(sql, encoding, strip_semicolon)
