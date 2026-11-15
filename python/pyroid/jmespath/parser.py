# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's parser (``jmespath.parser``), parsing natively.

``Parser().parse(expression)`` returns a :class:`ParsedResult` from the cache that
:func:`pyroid.jmespath.compile` and :func:`pyroid.jmespath.search` share: jmespath 1.1.0's 512
most recent expressions, oldest out first.
"""

from __future__ import annotations

from typing import Any

from pyroid._pyroid import jmespath as _native

__all__ = ["ParsedResult", "Parser"]

ParsedResult = _native.ParsedResult


class Parser:
    """jmespath's ``Parser``: ``parse()`` compiles an expression (cached)."""

    BINDING_POWER = {  # noqa: RUF012 - jmespath's class attribute
        "eof": 0,
        "unquoted_identifier": 0,
        "quoted_identifier": 0,
        "literal": 0,
        "rbracket": 0,
        "rparen": 0,
        "comma": 0,
        "rbrace": 0,
        "number": 0,
        "current": 0,
        "expref": 0,
        "colon": 0,
        "pipe": 1,
        "or": 2,
        "and": 3,
        "eq": 5,
        "gt": 5,
        "lt": 5,
        "gte": 5,
        "lte": 5,
        "ne": 5,
        "flatten": 9,
        # Everything above stops a projection.
        "star": 20,
        "filter": 21,
        "dot": 40,
        "not": 45,
        "lbrace": 50,
        "lbracket": 55,
        "lparen": 60,
    }
    # The maximum binding power for a token that can stop
    # a projection.
    _PROJECTION_STOP = 10
    # The _MAX_SIZE most recent expressions are cached.
    _MAX_SIZE = 512

    def __init__(self, lookahead: int = 2) -> None:
        self.tokenizer: Any = None
        self._tokens: list[Any] = [None] * lookahead
        self._buffer_size = lookahead
        self._index = 0

    def parse(self, expression: Any) -> Any:
        """Compile ``expression`` (or return the cached result)."""
        return _native.parse(expression)

    @classmethod
    def purge(cls) -> None:
        """Clear the expression compilation cache."""
        _native.purge()
