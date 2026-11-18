# Type stubs for pyroid.sqlparse.sql (sqlparse 0.6.1's token classes).
"""Token classes: ``sqlparse.sql``'s own classes when sqlparse 0.5.0 or newer is installed.

``parse`` builds trees of these classes, so ``isinstance(t, sqlparse.sql.Identifier)`` and
sqlparse's own code work on them; their methods are the installed release's (pyroid's copies
of sqlparse 0.6.1's without sqlparse).
"""

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, TextIO

from pyroid.sqlparse.tokens import _TokenType

class NameAliasMixin:
    def get_real_name(self) -> str | None: ...
    def get_alias(self) -> str | None: ...

class Token:
    value: str
    ttype: _TokenType | None
    parent: TokenList | None
    normalized: str
    is_keyword: bool
    is_group: bool
    is_whitespace: bool
    is_newline: bool
    def __init__(self, ttype: _TokenType | None, value: object) -> None: ...
    def flatten(self) -> Iterator[Token]: ...
    def match(self, ttype: _TokenType | None, values: Any, regex: bool = False) -> bool: ...
    def within(self, group_cls: type[Any] | tuple[type[Any], ...]) -> bool: ...
    def is_child_of(self, other: Token) -> bool: ...
    def has_ancestor(self, other: Token) -> bool: ...

class TokenList(Token):
    tokens: list[Token]
    def __init__(self, tokens: list[Token] | None = None) -> None: ...
    def __iter__(self) -> Iterator[Token]: ...
    def __getitem__(self, item: int) -> Token: ...
    def _pprint_tree(
        self, max_depth: int | None = None, depth: int = 0, f: TextIO | None = None, _pre: str = ""
    ) -> None: ...
    def get_token_at_offset(self, offset: int) -> Token | None: ...
    def get_sublists(self) -> Iterator[TokenList]: ...
    def token_first(self, skip_ws: bool = True, skip_cm: bool = False) -> Token | None: ...
    def token_next_by(
        self, i: Any = None, m: Any = None, t: Any = None, idx: int = -1, end: int | None = None
    ) -> tuple[int | None, Token | None]: ...
    def token_not_matching(
        self, funcs: Any, idx: int | None
    ) -> tuple[int | None, Token | None]: ...
    def token_matching(self, funcs: Any, idx: int | None) -> Token | None: ...
    def token_prev(
        self, idx: int | None, skip_ws: bool = True, skip_cm: bool = False
    ) -> tuple[int | None, Token | None]: ...
    def token_next(
        self, idx: int | None, skip_ws: bool = True, skip_cm: bool = False, _reverse: bool = False
    ) -> tuple[int | None, Token | None]: ...
    def token_index(self, token: Token, start: int | Token = 0) -> int: ...
    def group_tokens(
        self,
        grp_cls: type[TokenList],
        start: int,
        end: int,
        include_end: bool = True,
        extend: bool = False,
    ) -> TokenList: ...
    def insert_before(self, where: int | Token, token: Token) -> None: ...
    def insert_after(self, where: int | Token, token: Token, skip_ws: bool = True) -> None: ...
    def has_alias(self) -> bool: ...
    def get_alias(self) -> str | None: ...
    def get_name(self) -> str | None: ...
    def get_real_name(self) -> str | None: ...
    def get_parent_name(self) -> str | None: ...
    def _token_matching(
        self,
        funcs: Callable[[Token], bool]
        | list[Callable[[Token], bool]]
        | tuple[Callable[[Token], bool], ...],
        start: int | None = 0,
        end: int | None = None,
        reverse: bool = False,
    ) -> Any: ...
    def _get_first_name(
        self,
        idx: int | None = None,
        reverse: bool = False,
        keywords: bool = False,
        real_name: bool = False,
    ) -> str | None: ...

class Statement(TokenList):
    def get_type(self) -> str: ...

class Identifier(NameAliasMixin, TokenList):
    def is_wildcard(self) -> bool: ...
    def get_typecast(self) -> str | None: ...
    def get_ordering(self) -> str | None: ...
    def get_array_indices(self) -> Iterator[list[Token]]: ...

class IdentifierList(TokenList):
    def get_identifiers(self) -> Iterator[Token]: ...

class TypedLiteral(TokenList):
    M_OPEN: ClassVar[list[tuple[_TokenType, str | None]]]
    M_CLOSE: ClassVar[tuple[_TokenType, None]]
    M_EXTEND: ClassVar[tuple[_TokenType, tuple[str, ...]]]

class Parenthesis(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]
    M_CLOSE: ClassVar[tuple[_TokenType, str]]

class SquareBrackets(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]
    M_CLOSE: ClassVar[tuple[_TokenType, str]]

class Assignment(TokenList): ...

class If(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]
    M_CLOSE: ClassVar[tuple[_TokenType, str]]

class For(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, tuple[str, ...]]]
    M_CLOSE: ClassVar[tuple[_TokenType, str]]

class Comparison(TokenList):
    @property
    def left(self) -> Token: ...
    @property
    def right(self) -> Token: ...

class Comment(TokenList):
    def is_multiline(self) -> bool | list[Token]: ...

class Where(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]
    M_CLOSE: ClassVar[tuple[_TokenType, tuple[str, ...]]]

class Over(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]

class Having(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]
    M_CLOSE: ClassVar[tuple[_TokenType, tuple[str, ...]]]

class Case(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]
    M_CLOSE: ClassVar[tuple[_TokenType, str]]
    def get_cases(self, skip_ws: bool = False) -> list[tuple[list[Token] | None, list[Token]]]: ...

class Function(NameAliasMixin, TokenList):
    def get_parameters(self) -> list[Token] | Iterator[Token]: ...
    def get_window(self) -> Token | None: ...

class Begin(TokenList):
    M_OPEN: ClassVar[tuple[_TokenType, str]]
    M_CLOSE: ClassVar[tuple[_TokenType, str]]

class Operation(TokenList): ...
class Values(TokenList): ...
class Command(TokenList): ...
