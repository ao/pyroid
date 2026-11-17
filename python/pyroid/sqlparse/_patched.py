"""The functions ``pyroid.patch("sqlparse")`` sets on an installed sqlparse.

They are pyroid.sqlparse's, with the DoS limits read from ``sqlparse.engine.grouping`` (where
code using sqlparse sets them). Module-level functions, so they pickle by reference.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from typing import Any

import pyroid.sqlparse as _p

__all__ = ["format", "parse", "parsestream", "split", "tokenize"]


def _limits() -> Any:
    return sys.modules.get("sqlparse.engine.grouping")


def parse(sql: Any, encoding: str | None = None) -> tuple[Any, ...]:
    """``sqlparse.parse``, run by pyroid."""
    return tuple(_p._parsestream(sql, encoding, _limits()))


def parsestream(stream: Any, encoding: str | None = None) -> Generator[Any, None, None]:
    """``sqlparse.parsestream``, run by pyroid."""
    return _p._parsestream(stream, encoding, _limits())


def format(sql: Any, encoding: str | None = None, **options: Any) -> str:  # noqa: A001
    """``sqlparse.format``, run by pyroid."""
    options = _p.formatter.validate_options(options)
    return _p._format(sql, encoding, options, _limits())


def split(sql: Any, encoding: str | None = None, strip_semicolon: bool = False) -> list[str]:
    """``sqlparse.split``, run by pyroid."""
    return _p._split(sql, encoding, strip_semicolon)


tokenize = _p.lexer.tokenize
