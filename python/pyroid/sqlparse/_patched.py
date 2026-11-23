"""The functions ``pyroid.patch("sqlparse")`` sets on an installed sqlparse.

They are pyroid.sqlparse's, with the DoS limits read from ``sqlparse.engine.grouping`` (where
code using sqlparse sets them). Module-level functions, so they pickle by reference.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

# the implementation module (its private helpers are not in the stub)
_p: Any = importlib.import_module("pyroid.sqlparse")

__all__ = ["format", "group", "parse", "parsestream", "split", "tokenize"]


def _limits() -> Any:
    return sys.modules.get("sqlparse.engine.grouping")


def parse(sql: Any, encoding: str | None = None) -> Any:
    """``sqlparse.parse``, run by pyroid."""
    return tuple(_p._parsestream(sql, encoding, _limits()))


def parsestream(stream: Any, encoding: str | None = None) -> Any:
    """``sqlparse.parsestream``, run by pyroid."""
    return _p._parsestream(stream, encoding, _limits())


def format(sql: Any, encoding: str | None = None, **options: Any) -> Any:
    """``sqlparse.format``, run by pyroid."""
    options = _p.formatter.validate_options(options)
    return _p._format(sql, encoding, options, _limits())


def split(sql: Any, encoding: str | None = None, strip_semicolon: bool = False) -> Any:
    """``sqlparse.split``, run by pyroid."""
    return _p._split(sql, encoding, strip_semicolon)


def group(stmt: Any) -> Any:
    """``sqlparse.engine.grouping.group``, run by pyroid."""
    limits = _limits()
    depth, tokens = _p._limit_values(limits)
    r = _p._native._group(stmt, depth, tokens)
    if r is NotImplemented:
        return _p.engine.grouping._py_group(stmt, limits)
    return r


tokenize = _p.lexer.tokenize
