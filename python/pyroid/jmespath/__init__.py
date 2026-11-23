# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""A drop-in, Rust-backed replacement for jmespath 1.1.0.

``import pyroid.jmespath as jmespath`` and use ``jmespath.search(expression, data)`` or
``jmespath.compile(expression).search(data)``. Results, compiled ASTs and errors are
jmespath's.
"""

from __future__ import annotations

import json
import math
import numbers
import sys
import warnings
from typing import Any

from pyroid import _versions
from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import jmespath as _native
from pyroid.jmespath import _find, ast, exceptions, functions, lexer, parser, visitor
from pyroid.jmespath.visitor import Options

#: The jmespath release whose behaviour this module reproduces.
JMESPATH_VERSION = "1.1.0"
__version__ = JMESPATH_VERSION
_versions.check(__name__, "jmespath", f"jmespath=={JMESPATH_VERSION}")

__all__ = [
    "JMESPATH_VERSION",
    "Options",
    "ast",
    "compile",
    "exceptions",
    "functions",
    "lexer",
    "parser",
    "search",
    "visitor",
]


def _warn_literal() -> None:
    warnings.warn(
        "deprecated string literal syntax",
        PendingDeprecationWarning,
        stacklevel=_caller_stacklevel(),
    )


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points ``warnings.warn`` at the first frame outside this package."""
    frame = sys._getframe(1)  # the function that calls warnings.warn: stacklevel 1
    level = 1
    while frame.f_back is not None:
        name = str(frame.f_globals.get("__name__", ""))
        if name != __name__ and not name.startswith(__name__ + "."):
            break
        frame = frame.f_back
        level += 1
    return level


_WARNED: set[str] = set()


def _fallback(kind: str, expression: Any) -> Any:
    """Jmespath's own lexer for a non-``str`` expression (it lexes any iterable)."""
    real = _find.jmespath()
    if real is None:
        raise PyroidError(
            f"pyroid.jmespath only lexes str expressions (got {type(expression).__name__}); "
            f"jmespath lexes other iterables: pip install jmespath=={JMESPATH_VERSION}"
        )
    if kind not in _WARNED:
        _WARNED.add(kind)
        warnings.warn(
            f"pyroid.jmespath: a {type(expression).__name__} expression is handed to jmespath "
            "(slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=3,
        )
    import jmespath.lexer
    import jmespath.parser

    if kind == "tokenize":
        return jmespath.lexer.Lexer().tokenize(expression)
    return jmespath.parser.Parser().parse(expression)


def _stocks() -> list[Any]:
    found = [functions.Functions]
    if _find.jmespath() is not None:
        import jmespath.functions

        found.append(jmespath.functions.Functions)
    return found


_native._configure(
    {
        "ParseError": exceptions.ParseError,
        "IncompleteExpressionError": exceptions.IncompleteExpressionError,
        "LexerError": exceptions.LexerError,
        "ArityError": exceptions.ArityError,
        "VariadictArityError": exceptions.VariadictArityError,
        "JMESPathTypeError": exceptions.JMESPathTypeError,
        "EmptyExpressionError": exceptions.EmptyExpressionError,
        "UnknownFunctionError": exceptions.UnknownFunctionError,
        "loads": json.loads,
        "dumps": json.dumps,
        "warn_literal": _warn_literal,
        "Number": numbers.Number,
        "ceil": math.ceil,
        "floor": math.floor,
        "TreeInterpreter": visitor.TreeInterpreter,
        "Expression": visitor._Expression,
        "GraphvizVisitor": visitor.GraphvizVisitor,
        "fallback": _fallback,
        "stocks": _stocks(),
    }
)

compile = _native.compile
search = _native.search
