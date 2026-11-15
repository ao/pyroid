# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's ``Options`` and its tree interpreter (``jmespath.visitor``).

:func:`pyroid.jmespath.search` and ``ParsedResult.search`` evaluate natively. The
``TreeInterpreter`` here is jmespath's Python code, for programs that drive the interpreter
themselves (``TreeInterpreter(options).visit(parsed, data)``) or subclass it.
"""

from __future__ import annotations

import importlib
import operator
from numbers import Number
from typing import Any

from pyroid.jmespath import _find, functions

__all__ = ["GraphvizVisitor", "Options", "TreeInterpreter", "Visitor"]


def _equals(x: Any, y: Any) -> Any:
    if _is_special_number_case(x, y):
        return False
    return x == y


def _is_special_number_case(x: Any, y: Any) -> Any:
    # We need to special case comparing 0 or 1 to
    # True/False.  While normally comparing any
    # integer other than 0/1 to True/False will always
    # return False.  However 0/1 have this:
    # >>> 0 == True
    # False
    # >>> 0 == False
    # True
    # >>> 1 == True
    # True
    # >>> 1 == False
    # False
    #
    # Also need to consider that:
    # >>> 0 in [True, False]
    # True
    if _is_actual_number(x) and x in (0, 1):
        return isinstance(y, bool)
    if _is_actual_number(y) and y in (0, 1):
        return isinstance(x, bool)
    return None


def _is_comparable(x: Any) -> bool:
    # The spec doesn't officially support string types yet,
    # but enough people are relying on this behavior that
    # it's been added back.  This should eventually become
    # part of the official spec.
    return _is_actual_number(x) or isinstance(x, str)


def _is_actual_number(x: Any) -> bool:
    # We need to handle python's quirkiness with booleans,
    # specifically:
    #
    # >>> isinstance(False, int)
    # True
    # >>> isinstance(True, int)
    # True
    if isinstance(x, bool):
        return False
    return isinstance(x, Number)


class Options:
    """Options for evaluating an expression.

    Args:
        dict_cls: The class of the dicts a multi-select hash (``{a: a}``) builds, for example
            ``collections.OrderedDict``. ``None`` means ``dict``.
        custom_functions: An instance of a ``pyroid.jmespath.functions.Functions`` subclass
            (or of jmespath's own ``Functions``) with extra ``_func_*`` functions.
    """

    def __init__(self, dict_cls: Any = None, custom_functions: Any = None) -> None:
        #: The class to use when creating a dict.  The interpreter
        #  may create dictionaries during the evaluation of a JMESPath
        #  expression.  For example, a multi-select hash will
        #  create a dictionary.  By default we use a dict() type.
        #  You can set this value to change what dict type is used.
        #  The most common reason you would change this is if you
        #  want to set a collections.OrderedDict so that you can
        #  have predictable key ordering.
        self.dict_cls = dict_cls
        self.custom_functions = custom_functions


class _Expression:
    """An expression reference (``&expr``) as functions receive it."""

    def __init__(self, expression: Any, interpreter: Any) -> None:
        self.expression = expression
        self.interpreter = interpreter

    def visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate ``node`` (normally ``self.expression``) on a value."""
        return self.interpreter.visit(node, *args, **kwargs)


if _find.jmespath() is not None:
    # jmespath's own class, so isinstance checks and reprs in custom functions see it.
    _Expression = importlib.import_module("jmespath.visitor")._Expression


class Visitor:
    """Dispatch ``visit(node)`` to ``visit_<node type>``."""

    def __init__(self) -> None:
        self._method_cache: dict[Any, Any] = {}

    def visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Visit ``node``."""
        node_type = node["type"]
        method = self._method_cache.get(node_type)
        if method is None:
            method = getattr(self, "visit_%s" % node["type"], self.default_visit)  # noqa: UP031
            self._method_cache[node_type] = method
        return method(node, *args, **kwargs)

    def default_visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Called for a node type without a ``visit_<type>`` method."""
        raise NotImplementedError("default_visit")


class TreeInterpreter(Visitor):
    """jmespath's interpreter, in Python."""

    COMPARATOR_FUNC: dict[str, Any] = {  # noqa: RUF012 - jmespath's class attribute
        "eq": _equals,
        "ne": lambda x, y: not _equals(x, y),
        "lt": operator.lt,
        "gt": operator.gt,
        "lte": operator.le,
        "gte": operator.ge,
    }
    _EQUALITY_OPS = ("eq", "ne")
    MAP_TYPE = dict

    def __init__(self, options: Any = None) -> None:
        super().__init__()
        self._dict_cls: Any = self.MAP_TYPE
        if options is None:
            options = Options()
        self._options = options
        if options.dict_cls is not None:
            self._dict_cls = self._options.dict_cls
        if options.custom_functions is not None:
            self._functions = self._options.custom_functions
        else:
            self._functions = functions.Functions()

    def default_visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Called for a node type without a ``visit_<type>`` method."""
        raise NotImplementedError(node["type"])

    def visit_subexpression(self, node: Any, value: Any) -> Any:  # noqa: D102
        result = value
        for child in node["children"]:
            result = self.visit(child, result)
        return result

    def visit_field(self, node: Any, value: Any) -> Any:  # noqa: D102
        try:
            return value.get(node["value"])
        except AttributeError:
            return None

    def visit_comparator(self, node: Any, value: Any) -> Any:  # noqa: D102
        # Common case: comparator is == or !=
        comparator_func = self.COMPARATOR_FUNC[node["value"]]
        if node["value"] in self._EQUALITY_OPS:
            return comparator_func(
                self.visit(node["children"][0], value), self.visit(node["children"][1], value)
            )
        # Ordering operators are only valid for numbers.
        # Evaluating any other type with a comparison operator
        # will yield a None value.
        left = self.visit(node["children"][0], value)
        right = self.visit(node["children"][1], value)
        if not (_is_comparable(left) and _is_comparable(right)):
            return None
        return comparator_func(left, right)

    def visit_current(self, node: Any, value: Any) -> Any:  # noqa: D102
        return value

    def visit_expref(self, node: Any, value: Any) -> Any:  # noqa: D102
        return _Expression(node["children"][0], self)

    def visit_function_expression(self, node: Any, value: Any) -> Any:  # noqa: D102
        resolved_args = []
        for child in node["children"]:
            current = self.visit(child, value)
            resolved_args.append(current)
        return self._functions.call_function(node["value"], resolved_args)

    def visit_filter_projection(self, node: Any, value: Any) -> Any:  # noqa: D102
        base = self.visit(node["children"][0], value)
        if not isinstance(base, list):
            return None
        comparator_node = node["children"][2]
        collected = []
        for element in base:
            if self._is_true(self.visit(comparator_node, element)):
                current = self.visit(node["children"][1], element)
                if current is not None:
                    collected.append(current)
        return collected

    def visit_flatten(self, node: Any, value: Any) -> Any:  # noqa: D102
        base = self.visit(node["children"][0], value)
        if not isinstance(base, list):
            # Can't flatten the object if it's not a list.
            return None
        merged_list: list[Any] = []
        for element in base:
            if isinstance(element, list):
                merged_list.extend(element)
            else:
                merged_list.append(element)
        return merged_list

    def visit_identity(self, node: Any, value: Any) -> Any:  # noqa: D102
        return value

    def visit_index(self, node: Any, value: Any) -> Any:  # noqa: D102
        # Even though we can index strings, we don't
        # want to support that.
        if not isinstance(value, list):
            return None
        try:
            return value[node["value"]]
        except IndexError:
            return None

    def visit_index_expression(self, node: Any, value: Any) -> Any:  # noqa: D102
        result = value
        for child in node["children"]:
            result = self.visit(child, result)
        return result

    def visit_slice(self, node: Any, value: Any) -> Any:  # noqa: D102
        if not isinstance(value, list):
            return None
        s = slice(*node["children"])
        return value[s]

    def visit_key_val_pair(self, node: Any, value: Any) -> Any:  # noqa: D102
        return self.visit(node["children"][0], value)

    def visit_literal(self, node: Any, value: Any) -> Any:  # noqa: D102
        return node["value"]

    def visit_multi_select_dict(self, node: Any, value: Any) -> Any:  # noqa: D102
        if value is None:
            return None
        collected = self._dict_cls()
        for child in node["children"]:
            collected[child["value"]] = self.visit(child, value)
        return collected

    def visit_multi_select_list(self, node: Any, value: Any) -> Any:  # noqa: D102
        if value is None:
            return None
        collected = []
        for child in node["children"]:
            collected.append(self.visit(child, value))  # noqa: PERF401
        return collected

    def visit_or_expression(self, node: Any, value: Any) -> Any:  # noqa: D102
        matched = self.visit(node["children"][0], value)
        if self._is_false(matched):
            matched = self.visit(node["children"][1], value)
        return matched

    def visit_and_expression(self, node: Any, value: Any) -> Any:  # noqa: D102
        matched = self.visit(node["children"][0], value)
        if self._is_false(matched):
            return matched
        return self.visit(node["children"][1], value)

    def visit_not_expression(self, node: Any, value: Any) -> Any:  # noqa: D102
        original_result = self.visit(node["children"][0], value)
        if _is_actual_number(original_result) and original_result == 0:
            # Special case for 0, !0 should be false, not true.
            # 0 is not a special cased integer in jmespath.
            return False
        return not original_result

    def visit_pipe(self, node: Any, value: Any) -> Any:  # noqa: D102
        result = value
        for child in node["children"]:
            result = self.visit(child, result)
        return result

    def visit_projection(self, node: Any, value: Any) -> Any:  # noqa: D102
        base = self.visit(node["children"][0], value)
        if not isinstance(base, list):
            return None
        collected = []
        for element in base:
            current = self.visit(node["children"][1], element)
            if current is not None:
                collected.append(current)
        return collected

    def visit_value_projection(self, node: Any, value: Any) -> Any:  # noqa: D102
        base = self.visit(node["children"][0], value)
        try:
            base = base.values()
        except AttributeError:
            return None
        collected = []
        for element in base:
            current = self.visit(node["children"][1], element)
            if current is not None:
                collected.append(current)
        return collected

    def _is_false(self, value: Any) -> bool:
        # This looks weird, but we're explicitly using equality checks
        # because the truth/false values are different between
        # python and jmespath.
        return value == "" or value == [] or value == {} or value is None or value is False

    def _is_true(self, value: Any) -> bool:
        return not self._is_false(value)


class GraphvizVisitor(Visitor):
    """Render an AST as a Graphviz dot file."""

    def __init__(self) -> None:
        super().__init__()
        self._lines: list[str] = []
        self._count = 1

    def visit(self, node: Any, *args: Any, **kwargs: Any) -> Any:
        """Render ``node``."""
        self._lines.append("digraph AST {")
        current = "%s%s" % (node["type"], self._count)  # noqa: UP031
        self._count += 1
        self._visit(node, current)
        self._lines.append("}")
        return "\n".join(self._lines)

    def _visit(self, node: Any, current: Any) -> None:
        self._lines.append(
            '%s [label="%s(%s)"]' % (current, node["type"], node.get("value", ""))  # noqa: UP031
        )
        for child in node.get("children", []):
            child_name = "%s%s" % (child["type"], self._count)  # noqa: UP031
            self._count += 1
            self._lines.append("  %s -> %s" % (current, child_name))  # noqa: UP031
            self._visit(child, child_name)
