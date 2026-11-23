# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's exception classes (`jmespath.exceptions`).

With jmespath installed these are jmespath's own classes, so `except
jmespath.exceptions.ParseError` catches errors from pyroid too; without it, copies with the same
attributes and messages.
"""

from typing import Any

__all__ = [
    "ArityError",
    "EmptyExpressionError",
    "IncompleteExpressionError",
    "JMESPathError",
    "JMESPathTypeError",
    "LexerError",
    "ParseError",
    "UnknownFunctionError",
    "VariadictArityError",
]

class JMESPathError(ValueError):
    """The base class of jmespath's errors."""

class ParseError(JMESPathError):
    """An expression that does not parse."""

    lex_position: Any
    token_value: Any
    token_type: Any
    msg: Any
    expression: Any
    def __init__(
        self, lex_position: Any, token_value: Any, token_type: Any, msg: Any = ...
    ) -> None: ...

class IncompleteExpressionError(ParseError):
    """An expression that ends too early."""

    def set_expression(self, expression: Any) -> None:
        """Record the expression (and move the position to its end)."""

class LexerError(ParseError):
    """An expression with a character or literal the lexer rejects."""

    lexer_position: Any
    lexer_value: Any
    message: Any
    def __init__(
        self, lexer_position: Any, lexer_value: Any, message: Any, expression: Any = None
    ) -> None: ...

class ArityError(ParseError):
    """A function called with the wrong number of arguments."""

    expected_arity: Any
    actual_arity: Any
    function_name: Any
    def __init__(self, expected: Any, actual: Any, name: Any) -> None: ...

class VariadictArityError(ArityError):
    """A variadic function called with too few arguments."""

class JMESPathTypeError(JMESPathError):
    """A function argument of the wrong type."""

    function_name: Any
    current_value: Any
    actual_type: Any
    expected_types: Any
    def __init__(
        self, function_name: Any, current_value: Any, actual_type: Any, expected_types: Any
    ) -> None: ...

class EmptyExpressionError(JMESPathError):
    """An empty expression."""

    def __init__(self) -> None: ...

class UnknownFunctionError(JMESPathError):
    """A call to a function that does not exist."""
