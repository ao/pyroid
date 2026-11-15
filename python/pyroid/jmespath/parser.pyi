# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's parser (`jmespath.parser`), parsing natively."""

from typing import Any, ClassVar, Final

__all__ = ["ParsedResult", "Parser"]

class ParsedResult:
    """A compiled expression."""

    expression: Any
    """The expression text."""

    parsed: Any
    """jmespath's AST: nested `{'type': ..., 'children': [...], 'value': ...}` dicts.

    Built on first access, in jmespath's key order. Assigning a new AST changes what
    `search` evaluates; editing the dicts in place does not.
    """

    def __init__(self, expression: Any, parsed: Any) -> None:
        """Make a result from an expression and a dict AST (as jmespath's `ParsedResult`)."""

    def search(self, value: Any, options: Any = None) -> Any:
        """Evaluate the expression on `value` (see `pyroid.jmespath.search`)."""

    def _render_dot_file(self) -> str: ...
    def __reduce__(self) -> tuple[Any, ...]: ...

class Parser:
    """jmespath's `Parser`: `parse()` compiles an expression (cached)."""

    BINDING_POWER: ClassVar[dict[str, int]]
    _PROJECTION_STOP: ClassVar[int]
    _MAX_SIZE: Final = 512
    tokenizer: Any
    def __init__(self, lookahead: int = 2) -> None: ...
    def parse(self, expression: Any) -> ParsedResult:
        """Compile `expression`, or return the cached `ParsedResult`."""

    @classmethod
    def purge(cls) -> None:
        """Clear the expression compilation cache."""
