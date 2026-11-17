# Type stubs for pyroid.sqlparse.keywords (sqlparse 0.6.1).
"""The lexer's regex table and keyword dictionaries (``sqlparse.keywords``).

sqlparse's own objects when sqlparse 0.6.1 is installed (changing them changes both
libraries), else pyroid's copies of sqlparse 0.6.1's.
"""

from typing import Any, Final

from pyroid.sqlparse.tokens import _TokenType

PROCESS_AS_KEYWORD: Final[object]
SQL_REGEX: list[tuple[str, Any]]
KEYWORDS: dict[str, _TokenType]
KEYWORDS_COMMON: dict[str, _TokenType]
KEYWORDS_ORACLE: dict[str, _TokenType]
KEYWORDS_MYSQL: dict[str, _TokenType]
KEYWORDS_PLPGSQL: dict[str, _TokenType]
KEYWORDS_HQL: dict[str, _TokenType]
KEYWORDS_MSACCESS: dict[str, _TokenType]
KEYWORDS_SNOWFLAKE: dict[str, _TokenType]
KEYWORDS_BIGQUERY: dict[str, _TokenType]

class DelimitedSpans:
    openers: dict[int, tuple[int, str, _TokenType]]
    def __init__(self, openers: dict[int, tuple[int, str, _TokenType]], closers: Any) -> None: ...
    def resolve(self, pos: int) -> tuple[int, _TokenType] | None: ...

def find_delimited_spans(text: str) -> DelimitedSpans: ...
