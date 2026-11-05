"""Whitespace normalisation and length filtering (``pyroid.text``)."""

from typing import overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings

__all__ = ["min_chars", "normalize_whitespace"]

@overload
def normalize_whitespace(x: str, /, *, threads: int | None = None) -> str:
    """Collapse every run of whitespace to one space and strip both ends.

    Same result as ``" ".join(s.split())`` for every document (whitespace is
    exactly what ``str.isspace()`` accepts). Unchanged documents are returned
    as the original objects; Arrow nulls stay null.

    Args:
        x: A str, a list or tuple of str, or an Arrow string array/column
            (``string``, ``large_string`` or ``string_view``).
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.

    Returns:
        The same kind as ``x``: str, list, tuple, or a :class:`pyroid.Column`
        of strings (``large_string`` for ``large_string`` input, else
        ``string``).

    Raises:
        pyroid.InputError: ``x`` is not one of the accepted kinds, or holds a
            non-str item or a non-string Arrow type.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def normalize_whitespace(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def normalize_whitespace(
    x: tuple[str, ...], /, *, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def normalize_whitespace(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def min_chars(x: str, /, n: int, *, threads: int | None = None) -> bool:
    """True for each document with at least ``n`` characters (``len(s) >= n``).

    A predicate: it never builds new strings, it returns a mask.

    Args:
        x: A str, a list or tuple of str, or an Arrow string array/column.
        n: Minimum number of characters (Unicode code points), ``>= 0``.
        threads: As for :func:`normalize_whitespace`.

    Returns:
        bool, ``list[bool]``, ``tuple[bool, ...]``, or a :class:`pyroid.Column`
        of bools (nulls stay null), matching ``x``.

    Raises:
        pyroid.InputError: ``x`` or ``n`` has the wrong type.
        ValueError: ``n`` is negative.
    """

@overload
def min_chars(x: list[str], /, n: int, *, threads: int | None = None) -> list[bool]: ...
@overload
def min_chars(x: tuple[str, ...], /, n: int, *, threads: int | None = None) -> tuple[bool, ...]: ...
@overload
def min_chars(x: ArrowStrings, /, n: int, *, threads: int | None = None) -> Column[bool]: ...
