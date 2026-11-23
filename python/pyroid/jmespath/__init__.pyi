# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""A drop-in for jmespath 1.1.0, in Rust.

```python
import pyroid.jmespath as jmespath          # was: import jmespath

jmespath.search("Reservations[].Instances[].InstanceId", response)
expression = jmespath.compile("items[?price > `10`].name | sort(@)")
expression.search(data)
```

Results, compiled ASTs and errors are jmespath 1.1.0's: the same objects (the input's own
sub-objects wherever jmespath returns them, `int` or `float` as jmespath computes them, key
order, `sort_by` order), the same `.parsed` dicts, and the same exception classes, messages
and attributes. With jmespath installed the exception classes *are* jmespath's, so
`except jmespath.exceptions.JMESPathError` catches errors from both.

Everything runs natively: the lexer, the parser, the interpreter (on your Python objects,
without converting them) and the 26 built-in functions. Python is called where jmespath's
semantics are Python's: `__eq__`, `__lt__`, `get` and `values` of objects other than the
builtin types, `json.loads` for literals (once per expression), `json.dumps` in `to_string`,
and custom functions.

`Options(custom_functions=...)` takes a subclass of `pyroid.jmespath.functions.Functions` or of
jmespath's own `Functions`: the built-ins it inherits run natively, its own functions run in
Python with jmespath's arguments (expression references are jmespath's `_Expression` objects).

Compiled expressions are cached as jmespath 1.1.0 caches them: the 512 most recent, oldest out
first, shared by `search`, `compile` and `parser.Parser().parse`.
"""

from typing import Any, Final

from pyroid.jmespath import ast as ast
from pyroid.jmespath import exceptions as exceptions
from pyroid.jmespath import functions as functions
from pyroid.jmespath import lexer as lexer
from pyroid.jmespath import parser as parser
from pyroid.jmespath import visitor as visitor
from pyroid.jmespath.parser import ParsedResult
from pyroid.jmespath.visitor import Options as Options

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

#: The jmespath release whose behaviour this module reproduces.
JMESPATH_VERSION: Final[str]
__version__: Final[str]

def compile(expression: str) -> ParsedResult:
    """Compile `expression` (or return its cached compilation).

    Args:
        expression: A JMESPath expression.

    Returns:
        A `ParsedResult`: `.search(data, options=None)` evaluates it, `.parsed` is jmespath's
        AST and `.expression` the text. The same object comes back for the same expression
        while it is in the cache.

    Raises:
        jmespath.exceptions.ParseError: The expression does not parse (`LexerError` and
            `IncompleteExpressionError` are subclasses).
        jmespath.exceptions.EmptyExpressionError: `expression` is empty.
    """

def search(expression: str, data: Any, options: Any = None) -> Any:
    """Evaluate `expression` on `data`.

    Args:
        expression: A JMESPath expression (compiled once and cached).
        data: Any Python object: dicts and lists are walked as jmespath walks them, other
            objects are used through the same operations jmespath applies to them.
        options: `Options(dict_cls=..., custom_functions=...)`, jmespath's or pyroid's (any
            object with those two attributes).

    Returns:
        The result, as jmespath returns it.

    Raises:
        jmespath.exceptions.JMESPathError: A parse error, an unknown function, a wrong number
            of arguments or a wrong argument type, as in jmespath.
    """
