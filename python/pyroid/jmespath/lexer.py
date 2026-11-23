# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's lexer (``jmespath.lexer``), tokenizing natively."""

from __future__ import annotations

import string
from collections.abc import Iterator
from typing import Any

from pyroid._pyroid import jmespath as _native

__all__ = ["Lexer"]


class Lexer:
    """Split an expression into jmespath's tokens."""

    START_IDENTIFIER = set(string.ascii_letters + "_")  # noqa: RUF012 - jmespath's class attribute
    VALID_IDENTIFIER = set(string.ascii_letters + string.digits + "_")  # noqa: RUF012 - jmespath's class attribute
    VALID_NUMBER = set(string.digits)  # noqa: RUF012 - jmespath's class attribute
    WHITESPACE = set(" \t\n\r")  # noqa: RUF012 - jmespath's class attribute
    SIMPLE_TOKENS = {  # noqa: RUF012 - jmespath's class attribute
        ".": "dot",
        "*": "star",
        "]": "rbracket",
        ",": "comma",
        ":": "colon",
        "@": "current",
        "(": "lparen",
        ")": "rparen",
        "{": "lbrace",
        "}": "rbrace",
    }

    def tokenize(self, expression: Any) -> Iterator[dict[str, Any]]:
        """Yield ``{'type', 'value', 'start', 'end'}`` dicts, ending with ``eof``.

        Raises ``LexerError`` (or ``EmptyExpressionError``) when the iteration reaches the bad
        character, as jmespath's generator does.
        """
        tokens, error = _native.tokenize(expression)
        yield from tokens
        if error is not None:
            raise error
