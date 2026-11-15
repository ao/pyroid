# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's AST node constructors (``jmespath.ast``).

Nodes are dicts: ``{"type": <node type>, "children": [...], "value": ...}``, the shape of
``compile(expression).parsed``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "and_expression",
    "comparator",
    "current_node",
    "expref",
    "field",
    "filter_projection",
    "flatten",
    "function_expression",
    "identity",
    "index",
    "index_expression",
    "key_val_pair",
    "literal",
    "multi_select_dict",
    "multi_select_list",
    "not_expression",
    "or_expression",
    "pipe",
    "projection",
    "slice",
    "subexpression",
    "value_projection",
]

Node = dict[str, Any]


def comparator(name: Any, first: Node, second: Node) -> Node:
    """A comparison (``name`` is ``'eq'``, ``'ne'``, ``'lt'``, ``'gt'``, ``'lte'``, ``'gte'``)."""
    return {"type": "comparator", "children": [first, second], "value": name}


def current_node() -> Node:
    """``@``."""
    return {"type": "current", "children": []}


def expref(expression: Node) -> Node:
    """``&expression``."""
    return {"type": "expref", "children": [expression]}


def function_expression(name: Any, args: list[Node]) -> Node:
    """A function call."""
    return {"type": "function_expression", "children": args, "value": name}


def field(name: Any) -> Node:
    """A field lookup."""
    return {"type": "field", "children": [], "value": name}


def filter_projection(left: Node, right: Node, comparator: Node) -> Node:
    """``left[?comparator].right``."""
    return {"type": "filter_projection", "children": [left, right, comparator]}


def flatten(node: Node) -> Node:
    """``node[]``."""
    return {"type": "flatten", "children": [node]}


def identity() -> Node:
    """The value itself."""
    return {"type": "identity", "children": []}


def index(index: Any) -> Node:
    """``[index]``."""
    return {"type": "index", "value": index, "children": []}


def index_expression(children: list[Node]) -> Node:
    """Index steps applied in turn."""
    return {"type": "index_expression", "children": children}


def key_val_pair(key_name: Any, node: Node) -> Node:
    """One ``key: node`` of a multi-select hash."""
    return {"type": "key_val_pair", "children": [node], "value": key_name}


def literal(literal_value: Any) -> Node:
    """A literal value."""
    return {"type": "literal", "value": literal_value, "children": []}


def multi_select_dict(nodes: list[Node]) -> Node:
    """``{key: expression, ...}``."""
    return {"type": "multi_select_dict", "children": nodes}


def multi_select_list(nodes: list[Node]) -> Node:
    """``[expression, ...]``."""
    return {"type": "multi_select_list", "children": nodes}


def or_expression(left: Node, right: Node) -> Node:
    """``left || right``."""
    return {"type": "or_expression", "children": [left, right]}


def and_expression(left: Node, right: Node) -> Node:
    """``left && right``."""
    return {"type": "and_expression", "children": [left, right]}


def not_expression(expr: Node) -> Node:
    """``!expr``."""
    return {"type": "not_expression", "children": [expr]}


def pipe(left: Node, right: Node) -> Node:
    """``left | right``."""
    return {"type": "pipe", "children": [left, right]}


def projection(left: Node, right: Node) -> Node:
    """``left[*].right``."""
    return {"type": "projection", "children": [left, right]}


def subexpression(children: list[Node]) -> Node:
    """``a.b.c``."""
    return {"type": "subexpression", "children": children}


def slice(start: Any, end: Any, step: Any) -> Node:  # noqa: A001 - jmespath's name
    """``[start:end:step]``."""
    return {"type": "slice", "children": [start, end, step]}


def value_projection(left: Node, right: Node) -> Node:
    """``left.*.right``."""
    return {"type": "value_projection", "children": [left, right]}
