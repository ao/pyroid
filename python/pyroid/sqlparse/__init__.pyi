# Type stubs for pyroid.sqlparse: the signatures of sqlparse 0.6.1. The module docstring is kept
# here on purpose: the API docs are built from this stub.
"""sqlparse 0.6.1 with a Rust core: ``parse``, ``parsestream``, ``format`` and ``split``.

``pyroid.sqlparse`` gives sqlparse 0.6.1's output (0.6.1 is the final sqlparse release): the
same tokens, token trees, formatted text and exceptions, including the DoS limits
``engine.grouping.MAX_GROUPING_DEPTH`` (100) and ``MAX_GROUPING_TOKENS`` (10,000).

The trees are objects of sqlparse's own classes when sqlparse 0.5.0 or newer is installed, so
``isinstance(t, sqlparse.sql.Identifier)``, ``t.ttype is sqlparse.tokens.Keyword`` and
``except sqlparse.exceptions.SQLParseError`` hold, and sqlparse's own filters run on them.
Without sqlparse they are pyroid's copies of sqlparse 0.6.1's classes. The submodules
(``sql``, ``tokens``, ``keywords``, ``lexer``, ``engine``, ``filters``, ``formatter``,
``utils``, ``exceptions``, ``cli``) are sqlparse 0.6.1's, with the lexer, the grouping
(``engine.grouping.group``) and the formatting filters running in Rust.

Calls that only sqlparse's own Python code handles run it (pyroid ships it; sqlparse need not
be installed), with one ``pyroid.FallbackWarning`` per reason: a ``Lexer`` with overridden
methods, lexer patterns pyroid's matcher does not reproduce (those patterns only), and a text
holding both lone surrogates and characters from U+10F800 on.
"""

from collections.abc import Generator
from typing import IO, Any, Final, TypeAlias, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings
from pyroid.sqlparse import (
    cli as cli,
)
from pyroid.sqlparse import (
    engine as engine,
)
from pyroid.sqlparse import (
    exceptions as exceptions,
)
from pyroid.sqlparse import (
    filters as filters,
)
from pyroid.sqlparse import (
    formatter as formatter,
)
from pyroid.sqlparse import (
    keywords as keywords,
)
from pyroid.sqlparse import (
    lexer as lexer,
)
from pyroid.sqlparse import (
    sql as sql,
)
from pyroid.sqlparse import (
    tokens as tokens,
)
from pyroid.sqlparse import (
    utils as utils,
)

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

__version__: Final[str]
#: The sqlparse release pyroid.sqlparse reproduces: ``"0.6.1"``.
SQLPARSE_VERSION: Final[str]

def parse(sql: str | bytes | IO[str], encoding: str | None = None) -> tuple[sql.Statement, ...]:
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

def parsestream(
    stream: str | bytes | IO[str], encoding: str | None = None
) -> Generator[sql.Statement]:
    """Parse the statements of a file-like object, one at a time.

    Nothing is read until the first ``next()``; each statement is grouped when it is reached.

    Args:
        stream: A text stream (or a str or bytes).
        encoding: The encoding of bytes input.

    Returns:
        A generator of ``sql.Statement`` objects.
    """

def format(sql: str | bytes | IO[str], encoding: str | None = None, **options: Any) -> str:
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

def split(
    sql: str | bytes | IO[str], encoding: str | None = None, strip_semicolon: bool = False
) -> list[str]:
    """Split `sql` into its statements.

    Args:
        sql: A string holding one or more SQL statements.
        encoding: The encoding of bytes input.
        strip_semicolon: Remove trailing semicolons.

    Returns:
        The statements, each stripped of surrounding whitespace.
    """

# What format and split take, one text.
_Text: TypeAlias = str | bytes | IO[str]

@overload
def format_batch(sqls: str, *, threads: int | None = None, **options: Any) -> str:
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

@overload
def format_batch(
    sqls: list[str] | list[bytes] | list[_Text], *, threads: int | None = None, **options: Any
) -> list[str]: ...
@overload
def format_batch(
    sqls: tuple[_Text, ...], *, threads: int | None = None, **options: Any
) -> tuple[str, ...]: ...
@overload
def format_batch(
    sqls: ArrowStrings, *, threads: int | None = None, **options: Any
) -> Column[str]: ...
@overload
def split_batch(
    sqls: str, *, threads: int | None = None, strip_semicolon: bool = False
) -> list[str]:
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

@overload
def split_batch(
    sqls: tuple[_Text, ...], *, threads: int | None = None, strip_semicolon: bool = False
) -> tuple[list[str], ...]: ...
@overload
def split_batch(
    sqls: list[str] | list[bytes] | list[_Text] | ArrowStrings,
    *,
    threads: int | None = None,
    strip_semicolon: bool = False,
) -> list[list[str] | None]: ...
