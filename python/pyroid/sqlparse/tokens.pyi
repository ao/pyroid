# Type stubs for pyroid.sqlparse.tokens (sqlparse 0.6.1's token types).
"""Token types: ``sqlparse.tokens``'s own objects when sqlparse is installed.

A token type is a ``_TokenType`` (a tuple of names); ``ttype in Keyword`` tests the prefix,
``ttype is Keyword.DML`` the identity. Sub-types are made on first attribute access.
"""

from typing import Any

class _TokenType(tuple[Any, ...]):
    parent: _TokenType | None
    def __contains__(self, item: object) -> bool: ...
    def __getattr__(self, name: str) -> _TokenType: ...

Token: _TokenType
Text: _TokenType
Whitespace: _TokenType
Newline: _TokenType
Error: _TokenType
Other: _TokenType
Keyword: _TokenType
Name: _TokenType
Literal: _TokenType
String: _TokenType
Number: _TokenType
Punctuation: _TokenType
Operator: _TokenType
Comparison: _TokenType
Wildcard: _TokenType
Comment: _TokenType
Assignment: _TokenType
Generic: _TokenType
Command: _TokenType
DML: _TokenType
DDL: _TokenType
CTE: _TokenType
