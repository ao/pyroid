# Type stubs for pyroid.sqlparse.lexer (sqlparse 0.6.1).
"""The lexer (``sqlparse.lexer``): ``tokenize`` runs pyroid's port of sqlparse 0.6.1's lexer.

``Lexer`` is sqlparse's own class when sqlparse 0.6.1 is installed (one singleton for both
libraries), else pyroid's copy. Its table (``set_SQL_REGEX``, ``add_keywords``, ``clear``,
``default_initialization``) is read on each call and compiled once per change; patterns
pyroid's matcher does not reproduce run in Python's ``re`` (one ``pyroid.FallbackWarning``).
"""

from collections.abc import Iterator
from typing import IO, Any, ClassVar

from pyroid.sqlparse.tokens import _TokenType

__all__ = ["Lexer", "tokenize"]

class Lexer:
    """The lexer singleton's class; configure it with ``Lexer.get_default_instance()``."""

    _default_instance: ClassVar[Lexer | None]
    @classmethod
    def get_default_instance(cls) -> Lexer: ...
    def default_initialization(self) -> None: ...
    def clear(self) -> None: ...
    def set_SQL_REGEX(self, SQL_REGEX: Any) -> None: ...  # noqa: N802, N803
    def add_keywords(self, keywords: dict[str, _TokenType]) -> None: ...
    def is_keyword(self, value: str) -> tuple[_TokenType, str]: ...
    def get_tokens(
        self, text: str | bytes | IO[str], encoding: str | None = None
    ) -> Iterator[tuple[_TokenType, str]]: ...

def plain(lex: Any) -> bool:
    """Whether `lex` is a `Lexer` whose behaviour is only its table (nothing overridden)."""

def read(text: Any, encoding: str | None) -> Any:
    """`Lexer.get_tokens`'s input handling: the text of a str, bytes or text stream."""

def tokenize(
    sql: str | bytes | IO[str], encoding: str | None = None
) -> Iterator[tuple[_TokenType, str]]:
    """Tokenize `sql` with the default `Lexer`: ``(token type, value)`` pairs."""
