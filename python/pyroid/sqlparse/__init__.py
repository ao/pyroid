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

from pyroid import _versions
from pyroid.sqlparse import (
    _backend,
    cli,
    engine,
    exceptions,
    filters,
    formatter,
    keywords,
    lexer,
    sql,
    tokens,
    utils,
)
from pyroid.sqlparse._core import native as _native
from pyroid.sqlparse._core import warn_fallback as _warn_fallback

__version__ = "0.6.1"
#: The sqlparse release pyroid.sqlparse reproduces.
SQLPARSE_VERSION = _backend.SQLPARSE_VERSION
_versions.check(__name__, "sqlparse", f"sqlparse=={SQLPARSE_VERSION}")

__all__ = [
    "SQLPARSE_VERSION",
    "cli",
    "engine",
    "exceptions",
    "filters",
    "format",
    "format_batch",
    "formatter",
    "keywords",
    "lexer",
    "parse",
    "parsestream",
    "split",
    "split_batch",
    "sql",
    "tokens",
    "utils",
]


# ---------------------------------------------------------------------------
# The pipelines. `limits` is the module whose MAX_GROUPING_DEPTH / MAX_GROUPING_TOKENS apply
# (None: pyroid.sqlparse.engine.grouping; an installed sqlparse's under pyroid.patch).
# ---------------------------------------------------------------------------


def _limit_values(limits: Any) -> tuple[Any, Any]:
    src = engine.grouping if limits is None else limits
    return getattr(src, "MAX_GROUPING_DEPTH", 100), getattr(src, "MAX_GROUPING_TOKENS", 10000)


def _python_group(limits: Any) -> Any:
    def group(stmt: Any) -> Any:
        return engine.grouping._py_group(stmt, limits)

    return group


def _python_stream(stream: Any, encoding: str | None, limits: Any) -> Generator[Any, None, None]:
    stack = engine.FilterStack()
    stack.enable_grouping()
    stack._group = _python_group(limits)
    yield from stack.run(stream, encoding)


def _group_python(stmt: Any, limits: Any) -> Any:
    try:
        return engine.grouping._py_group(stmt, limits)
    except RecursionError as err:
        raise exceptions.SQLParseError("Maximum recursion depth exceeded") from err


def _parsestream(stream: Any, encoding: str | None, limits: Any) -> Generator[Any, None, None]:
    lex = lexer.Lexer.get_default_instance()
    if not lexer.plain(lex):
        _warn_fallback("parsestream", "the default Lexer has overridden methods", 4)
        yield from _python_stream(stream, encoding, limits)
        return
    text = lexer.read(stream, encoding)
    # the limits are read for each statement, as sqlparse's group() reads them
    it = _native._parse(text, lex, lambda: _limit_values(limits))
    if it is NotImplemented:
        yield from _python_stream(text, None, limits)
        return
    for stmt in it:
        if type(stmt) is tuple:  # sqlparse's own grouping: it raises, or reads the limits
            yield _group_python(stmt[1], limits)
        else:
            yield stmt


# Option values past this go to sqlparse's own code (Python ints have no bound).
_BIG = 2**62


def _format_args(options: dict[str, Any]) -> tuple[Any, ...] | None:
    """The validated options as `_native._format` takes them (`build_filter_stack`'s reading).

    None for integer options too large for the Rust port.
    """
    get = options.get
    for key in ("truncate_strings", "indent_width", "wrap_after"):
        value = get(key)
        if isinstance(value, int) and not -_BIG < value < _BIG:
            return None
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
    return (
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


def _format(sql: Any, encoding: str | None, options: dict[str, Any], limits: Any) -> str:
    lex = lexer.Lexer.get_default_instance()
    if lexer.plain(lex):
        text = lexer.read(sql, encoding)
        args = _format_args(options)
        r = (
            NotImplemented
            if args is None
            else _native._format(text, lex, *_limit_values(limits), args)
        )
        if r is not NotImplemented:
            return r  # type: ignore[no-any-return]
        sql, encoding = text, None
    else:
        _warn_fallback("format", "the default Lexer has overridden methods", 4)
    stack = engine.FilterStack()
    stack._group = _python_group(limits)
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


def _batch_items(sqls: Any) -> list[Any] | None:
    """One copy of a list or tuple batch's items (the caller's list may change meanwhile)."""
    return list(sqls) if isinstance(sqls, (list, tuple)) else None


def _batch_texts(items: list[Any] | None) -> list[Any] | None:
    """The items as `format` and `split` read them, or None when they are all str already.

    Bytes are decoded and text streams read; the first item they reject raises their
    TypeError.
    """
    if items is None or all(type(x) is str for x in items):
        return None
    return [lexer.read(x, None) for x in items]


def _batch_loop(sqls: Any, items: list[Any] | None, one: Any) -> Any:
    """``[one(sql) for sql in sqls]`` in the batch's container kind (Arrow nulls stay None)."""
    if isinstance(sqls, str):
        return one(sqls)
    if items is not None:
        out = [one(x) for x in items]
        return tuple(out) if isinstance(sqls, tuple) else out
    rows = sqls.to_pylist() if hasattr(sqls, "to_pylist") else list(sqls)  # pyarrow, Column
    return [None if x is None else one(x) for x in rows]


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


def parse(sql: Any, encoding: str | None = None) -> tuple[Any, ...]:
    """Parse `sql` and return its statements.

    Args:
        sql: A string (or bytes, or a text stream) holding one or more SQL statements.
        encoding: The encoding of bytes input (default: UTF-8, else ``unicode-escape``).

    Returns:
        A tuple of ``sql.Statement`` objects.

    Raises:
        sqlparse.exceptions.SQLParseError: A statement nests deeper than
            ``engine.grouping.MAX_GROUPING_DEPTH`` or a token list holds more than
            ``MAX_GROUPING_TOKENS`` tokens.
    """
    return tuple(_parsestream(sql, encoding, None))


def parsestream(stream: Any, encoding: str | None = None) -> Generator[Any, None, None]:
    """Parse the statements of a file-like object, one at a time.

    Nothing is read until the first ``next()``; each statement is grouped when it is reached.

    Args:
        stream: A text stream (or a str or bytes).
        encoding: The encoding of bytes input.

    Returns:
        A generator of ``sql.Statement`` objects.
    """
    return _parsestream(stream, encoding, None)


def format(sql: Any, encoding: str | None = None, **options: Any) -> str:
    """Format `sql` according to `options`.

    Args:
        sql: The SQL text.
        encoding: The encoding of bytes input.
        **options: sqlparse's options: ``keyword_case``, ``identifier_case``
            (``"upper"``, ``"lower"``, ``"capitalize"``), ``strip_comments``,
            ``strip_whitespace``, ``reindent``, ``reindent_aligned``,
            ``use_space_around_operators``, ``indent_tabs``, ``indent_width``,
            ``indent_after_first``, ``indent_columns``, ``wrap_after``, ``comma_first``,
            ``compact``, ``output_format`` (``"sql"``, ``"python"``, ``"php"``),
            ``truncate_strings``, ``truncate_char``. ``right_margin`` raises
            ``NotImplementedError``, as in sqlparse 0.6.1.

    Returns:
        The formatted SQL.

    Raises:
        sqlparse.exceptions.SQLParseError: An option value is invalid, or a statement passes
            a DoS limit.
    """
    options = formatter.validate_options(options)
    return _format(sql, encoding, options, None)


def split(sql: Any, encoding: str | None = None, strip_semicolon: bool = False) -> list[str]:
    """Split `sql` into its statements.

    Args:
        sql: A string holding one or more SQL statements.
        encoding: The encoding of bytes input.
        strip_semicolon: Remove trailing semicolons.

    Returns:
        The statements, each stripped of surrounding whitespace.
    """
    return _split(sql, encoding, strip_semicolon)


def format_batch(sqls: Any, *, threads: int | None = None, **options: Any) -> Any:
    """Format every SQL text of `sqls` with `options`, in parallel.

    The result equals ``[format(sql, **options) for sql in sqls]``, in the caller's kind: a
    list for a list, a tuple for a tuple, a ``pyroid.Column`` of strings for an Arrow array or
    column (nulls stay null), a str for a str (one text). A list or tuple holds what ``format``
    takes (str, bytes, text streams), and an item it rejects raises its ``TypeError``, as the
    loop does; other errors are the loop's too.

    Args:
        sqls: The SQL texts: a list or tuple, or an Arrow string array or column.
        threads: At most this many tasks at once on the shared pool (``None``: all of it).
        **options: ``format``'s options.

    Returns:
        The formatted texts.
    """
    options = formatter.validate_options(options)
    lex = lexer.Lexer.get_default_instance()
    args = _format_args(options)
    native = lexer.plain(lex) and args is not None
    if native:
        r = _native._format_batch(sqls, lex, *_limit_values(None), args, threads)
        if r is not NotImplemented:
            return r
    items = _batch_items(sqls)
    texts = _batch_texts(items)
    if native and texts is not None:  # bytes or streams, read as format reads them
        again = tuple(texts) if isinstance(sqls, tuple) else texts
        r = _native._format_batch(again, lex, *_limit_values(None), args, threads)
        if r is not NotImplemented:
            return r

    def one(sql: Any) -> Any:
        return _format(sql, None, dict(options), None)

    return _batch_loop(sqls, items if texts is None else texts, one)


def split_batch(sqls: Any, *, threads: int | None = None, strip_semicolon: bool = False) -> Any:
    """Split every SQL text of `sqls` into statements, in parallel.

    The result equals ``[split(sql, strip_semicolon=strip_semicolon) for sql in sqls]``: a
    list of lists (a tuple of lists for a tuple; ``None`` for an Arrow null). A list or tuple
    holds what ``split`` takes (str, bytes, text streams), and an item it rejects raises its
    ``TypeError``, as the loop does.

    Args:
        sqls: The SQL texts: a list or tuple, or an Arrow string array or column.
        threads: At most this many tasks at once on the shared pool (``None``: all of it).
        strip_semicolon: Remove trailing semicolons, as ``split`` does.

    Returns:
        The statements of each text.
    """
    lex = lexer.Lexer.get_default_instance()
    native = lexer.plain(lex)
    if native:
        r = _native._split_batch(sqls, lex, bool(strip_semicolon), threads)
        if r is not NotImplemented:
            return r
    items = _batch_items(sqls)
    texts = _batch_texts(items)
    if native and texts is not None:  # bytes or streams, read as split reads them
        again = tuple(texts) if isinstance(sqls, tuple) else texts
        r = _native._split_batch(again, lex, bool(strip_semicolon), threads)
        if r is not NotImplemented:
            return r

    def one(sql: Any) -> Any:
        return _split(sql, None, strip_semicolon)

    return _batch_loop(sqls, items if texts is None else texts, one)


def __dir__() -> list[str]:
    """``dir(pyroid.sqlparse)``: sqlparse's names, not this module's typing imports."""
    return sorted(set(globals()) - {"annotations", "Any", "Generator"})
