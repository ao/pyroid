# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's exception classes.

With jmespath installed these are jmespath's own classes (``jmespath.exceptions``), so
``except jmespath.exceptions.ParseError`` catches errors from pyroid too. Without it they are
copies with the same attributes and messages.
"""

from __future__ import annotations

from pyroid.jmespath import _find

if _find.jmespath() is not None:
    from jmespath.exceptions import (
        ArityError,
        EmptyExpressionError,
        IncompleteExpressionError,
        JMESPathError,
        JMESPathTypeError,
        LexerError,
        ParseError,
        UnknownFunctionError,
        VariadictArityError,
    )
else:
    from pyroid.jmespath._exceptions import (
        ArityError,
        EmptyExpressionError,
        IncompleteExpressionError,
        JMESPathError,
        JMESPathTypeError,
        LexerError,
        ParseError,
        UnknownFunctionError,
        VariadictArityError,
    )

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
