# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""Copies of jmespath's exception classes, for when jmespath is not installed."""

from __future__ import annotations

from typing import Any


class JMESPathError(ValueError):
    """The base class of jmespath's errors."""


class ParseError(JMESPathError):
    """An expression that does not parse."""

    _ERROR_MESSAGE = "Invalid jmespath expression"

    def __init__(
        self, lex_position: Any, token_value: Any, token_type: Any, msg: Any = _ERROR_MESSAGE
    ) -> None:
        super().__init__(lex_position, token_value, token_type)
        self.lex_position = lex_position
        self.token_value = token_value
        self.token_type = token_type.upper()
        self.msg = msg
        # Whatever catches the ParseError can fill in the full expression
        self.expression: Any = None

    def __str__(self) -> str:
        # self.lex_position +1 to account for the starting double quote char.
        underline = " " * (self.lex_position + 1) + "^"
        return (
            '%s: Parse error at column %s, token "%s" (%s), for expression:\n"%s"\n%s'  # noqa: UP031
            % (
                self.msg,
                self.lex_position,
                self.token_value,
                self.token_type,
                self.expression,
                underline,
            )
        )


class IncompleteExpressionError(ParseError):
    """An expression that ends too early."""

    def set_expression(self, expression: Any) -> None:
        """Record the expression (and move the position to its end)."""
        self.expression = expression
        self.lex_position = len(expression)
        self.token_type = None
        self.token_value = None

    def __str__(self) -> str:
        # self.lex_position +1 to account for the starting double quote char.
        underline = " " * (self.lex_position + 1) + "^"
        return 'Invalid jmespath expression: Incomplete expression:\n"%s"\n%s' % (  # noqa: UP031
            self.expression,
            underline,
        )


class LexerError(ParseError):
    """An expression with a character or literal the lexer rejects."""

    def __init__(
        self, lexer_position: Any, lexer_value: Any, message: Any, expression: Any = None
    ) -> None:
        self.lexer_position = lexer_position
        self.lexer_value = lexer_value
        self.message = message
        super().__init__(lexer_position, lexer_value, message)
        # Whatever catches LexerError can set this.
        self.expression = expression

    def __str__(self) -> str:
        underline = " " * self.lexer_position + "^"
        return "Bad jmespath expression: %s:\n%s\n%s" % (  # noqa: UP031
            self.message,
            self.expression,
            underline,
        )


class ArityError(ParseError):
    """A function called with the wrong number of arguments."""

    def __init__(self, expected: Any, actual: Any, name: Any) -> None:
        self.expected_arity = expected
        self.actual_arity = actual
        self.function_name = name
        self.expression = None

    def __str__(self) -> str:
        return "Expected %s %s for function %s(), received %s" % (  # noqa: UP031
            self.expected_arity,
            self._pluralize("argument", self.expected_arity),
            self.function_name,
            self.actual_arity,
        )

    def _pluralize(self, word: str, count: Any) -> str:
        if count == 1:
            return word
        return word + "s"


class VariadictArityError(ArityError):
    """A variadic function called with too few arguments."""

    def __str__(self) -> str:
        return "Expected at least %s %s for function %s(), received %s" % (  # noqa: UP031
            self.expected_arity,
            self._pluralize("argument", self.expected_arity),
            self.function_name,
            self.actual_arity,
        )


class JMESPathTypeError(JMESPathError):
    """A function argument of the wrong type."""

    def __init__(
        self, function_name: Any, current_value: Any, actual_type: Any, expected_types: Any
    ) -> None:
        self.function_name = function_name
        self.current_value = current_value
        self.actual_type = actual_type
        self.expected_types = expected_types

    def __str__(self) -> str:
        return (
            'In function %s(), invalid type for value: %s, expected one of: %s, received: "%s"'  # noqa: UP031
            % (self.function_name, self.current_value, self.expected_types, self.actual_type)
        )


class EmptyExpressionError(JMESPathError):
    """An empty expression."""

    def __init__(self) -> None:
        super().__init__("Invalid JMESPath expression: cannot be empty.")


class UnknownFunctionError(JMESPathError):
    """A call to a function that does not exist."""
