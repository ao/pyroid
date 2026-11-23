"""The native side of pyroid.sqlparse, configured with the token classes and types in use."""

from __future__ import annotations

import unicodedata
import warnings
from typing import Any

from pyroid._pyroid import FallbackWarning
from pyroid._pyroid import sqlparse as native
from pyroid.sqlparse import _backend, exceptions, keywords, sql, tokens

__all__ = ["native", "warn_fallback"]


def _pause() -> None:
    """Called by long native runs now and then: the call lets other threads take the GIL."""


native._configure(
    tuple(getattr(sql, name) for name in _backend.CLASSES),
    tokens.Token,
    tokens._TokenType,
    keywords.PROCESS_AS_KEYWORD,
    exceptions.SQLParseError,
    unicodedata.unidata_version,
    _pause,
    _backend.NEWLINE_SLOT,
)

_WARNED: set[str] = set()


def warn_fallback(function: str, reason: str, stacklevel: int = 3) -> None:
    """Warn once per process per function and reason that Python's code runs (slow path)."""
    key = f"{function}:{reason}"
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"pyroid.sqlparse.{function}: {reason}, so this call runs sqlparse's Python code "
        "(slow path)",
        FallbackWarning,
        stacklevel=stacklevel,
    )


def any_type(x: Any) -> str:
    """`type(x)!r` as sqlparse's TypeError spells it."""
    return repr(type(x))
