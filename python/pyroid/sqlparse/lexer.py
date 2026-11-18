"""SQL lexer (``sqlparse.lexer``): ``tokenize`` runs pyroid's port of sqlparse 0.6.1's lexer.

``Lexer`` is sqlparse's own class when sqlparse 0.6.1 is installed (one singleton for both
libraries), else pyroid's copy. Its table (``set_SQL_REGEX``, ``add_keywords``, ``clear``,
``default_initialization``) is read on each call and compiled once per change.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import TextIOBase
from typing import Any

from pyroid.sqlparse import _backend

if _backend.SHARE_LEXER and _backend.REAL is not None:
    Lexer: Any = _backend.REAL.lexer.Lexer
else:
    from pyroid.sqlparse._own_lexer import Lexer

    Lexer.__module__ = __name__

from pyroid.sqlparse import _core

__all__ = ["Lexer", "tokenize"]


def _original(name: str) -> Any:
    """``Lexer.<name>`` if it is sqlparse 0.6.1's own (or pyroid's copy), else None."""
    f = getattr(Lexer, name)
    own = getattr(f, "__qualname__", None) == f"Lexer.{name}" and getattr(
        f, "__module__", None
    ) in ("sqlparse.lexer", "pyroid.sqlparse._own_lexer")
    return f if own else None


# The methods the lexing runs: replaced on the class, the Python code runs instead.
_GET_TOKENS = _original("get_tokens")
_IS_KEYWORD = _original("is_keyword")


def plain(lex: Any) -> bool:
    """Whether `lex` is a `Lexer` whose behaviour is only its table.

    Nothing is overridden: not on the instance, in a subclass, or on the class itself.
    """
    return (
        type(lex) is Lexer
        and len(vars(lex)) == 2
        and Lexer.get_tokens is _GET_TOKENS
        and Lexer.is_keyword is _IS_KEYWORD
    )


def read(text: Any, encoding: str | None) -> Any:
    """`Lexer.get_tokens`'s input handling: the text of a str, bytes or text stream."""
    if isinstance(text, TextIOBase):
        text = text.read()
    if isinstance(text, str):
        pass
    elif isinstance(text, bytes):
        if encoding:
            text = text.decode(encoding)
        else:
            try:
                text = text.decode("utf-8")
            except UnicodeDecodeError:
                text = text.decode("unicode-escape")
    else:
        raise TypeError(f"Expected text or file-like object, got {type(text)!r}")
    return text


def _tokens(lex: Any, sql: Any, encoding: str | None) -> Iterator[tuple[Any, str]]:
    text = read(sql, encoding)
    r = _core.native._tokenize(text, lex)
    if r is NotImplemented or r is None:
        why = (
            "the text holds both lone surrogates and characters from U+10F800 on"
            if r is None
            else "the lexer's table holds entries pyroid cannot read"
        )
        _core.warn_fallback("lexer.tokenize", why, 4)
        yield from Lexer.get_tokens(lex, text)
        return
    first, rest = r
    yield from first
    if rest is not None:
        for pairs in rest:  # lexed as they are asked for, a list at a time
            yield from pairs


def tokenize(sql: Any, encoding: str | None = None) -> Iterator[tuple[Any, str]]:
    """Tokenize `sql` with the default `Lexer`: ``(token type, value)`` pairs."""
    lex = Lexer.get_default_instance()
    if not plain(lex):
        return lex.get_tokens(sql, encoding)  # type: ignore[no-any-return]
    return _tokens(lex, sql, encoding)
