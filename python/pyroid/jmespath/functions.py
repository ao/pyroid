# Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
"""jmespath's ``Functions`` class and ``signature`` decorator, for custom functions.

Subclass :class:`Functions`, add ``_func_<name>`` methods decorated with :func:`signature`,
and pass an instance as ``Options(custom_functions=...)``. Subclasses of jmespath's own
``jmespath.functions.Functions`` work the same way. The built-in functions run natively in
pyroid; the methods here are jmespath's Python code, used when a subclass calls them itself.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from pyroid.jmespath import _find, exceptions

__all__ = ["REVERSE_TYPES_MAP", "TYPES_MAP", "FunctionRegistry", "Functions", "signature"]

# python types -> jmespath types
TYPES_MAP = {
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
    "unicode": "string",
    "str": "string",
    "float": "number",
    "int": "number",
    "long": "number",
    "OrderedDict": "object",
    "_Projection": "array",
    "_Expression": "expref",
}


# jmespath types -> python types
REVERSE_TYPES_MAP = {
    "boolean": ("bool",),
    "array": ("list", "_Projection"),
    "object": (
        "dict",
        "OrderedDict",
    ),
    "null": ("NoneType",),
    "string": ("unicode", "str"),
    "number": ("float", "int", "long"),
    "expref": ("_Expression",),
}

if _find.jmespath() is not None:
    # The same dicts as jmespath's: code that adds its own types to one (Ansible's json_query
    # adds its tagged str, list and dict classes) changes both, and the native type checks
    # follow it.
    _real = importlib.import_module("jmespath.functions")
    TYPES_MAP = _real.TYPES_MAP
    REVERSE_TYPES_MAP = _real.REVERSE_TYPES_MAP

_F = TypeVar("_F", bound=Callable[..., Any])


def signature(*arguments: dict[str, Any]) -> Callable[[_F], _F]:
    """Declare a custom function's argument types: ``{'types': [...]}`` per argument."""

    def _record_signature(func: _F) -> _F:
        func.signature = arguments  # type: ignore[attr-defined]
        return func

    return _record_signature


def _get_methods(cls: type) -> Iterator[tuple[str, Any]]:
    yield from inspect.getmembers(cls, predicate=inspect.isfunction)


class FunctionRegistry(type):
    """Builds ``FUNCTION_TABLE`` from the ``_func_*`` methods that have a signature."""

    def __init__(cls, name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> None:
        cls._populate_function_table()
        super().__init__(name, bases, attrs)

    def _populate_function_table(cls) -> None:
        function_table = {}
        # Any method with a @signature decorator that also
        # starts with "_func_" is registered as a function.
        # _func_max_by -> max_by function.
        for name, method in _get_methods(cls):
            if not name.startswith("_func_"):
                continue
            sig = getattr(method, "signature", None)
            if sig is not None:
                function_table[name[6:]] = {
                    "function": method,
                    "signature": sig,
                }
        cls.FUNCTION_TABLE = function_table


class Functions(metaclass=FunctionRegistry):
    """jmespath's built-in functions; subclass it to add your own."""

    FUNCTION_TABLE: dict[str, dict[str, Any]] = {}  # noqa: RUF012 - set per class by the metaclass

    def call_function(self, function_name: Any, resolved_args: Any) -> Any:
        """Check ``resolved_args`` against the function's signature and call it."""
        try:
            spec = self.FUNCTION_TABLE[function_name]
        except KeyError:
            raise exceptions.UnknownFunctionError(  # noqa: B904 - jmespath chains the KeyError
                "Unknown function: %s()" % function_name  # noqa: UP031
            )
        function = spec["function"]
        sig = spec["signature"]
        self._validate_arguments(resolved_args, sig, function_name)
        return function(self, *resolved_args)

    def _validate_arguments(self, args: Any, sig: Any, function_name: Any) -> None:
        if sig and sig[-1].get("variadic"):
            if len(args) < len(sig):
                raise exceptions.VariadictArityError(len(sig), len(args), function_name)
        elif len(args) != len(sig):
            raise exceptions.ArityError(len(sig), len(args), function_name)
        return self._type_check(args, sig, function_name)

    def _type_check(self, actual: Any, sig: Any, function_name: Any) -> None:
        for i in range(len(sig)):
            allowed_types = sig[i]["types"]
            if allowed_types:
                self._type_check_single(actual[i], allowed_types, function_name)

    def _type_check_single(self, current: Any, types: Any, function_name: Any) -> None:
        # Type checking involves checking the top level type,
        # and in the case of arrays, potentially checking the types
        # of each element.
        allowed_types, allowed_subtypes = self._get_allowed_pytypes(types)
        # We're not using isinstance() on purpose.
        # The type model for jmespath does not map
        # 1-1 with python types (booleans are considered
        # integers in python for example).
        actual_typename = type(current).__name__
        if actual_typename not in allowed_types:
            raise exceptions.JMESPathTypeError(
                function_name, current, self._convert_to_jmespath_type(actual_typename), types
            )
        # If we're dealing with a list type, we can have
        # additional restrictions on the type of the list
        # elements (for example a function can require a
        # list of numbers or a list of strings).
        # Arrays are the only types that can have subtypes.
        if allowed_subtypes:
            self._subtype_check(current, allowed_subtypes, types, function_name)

    def _get_allowed_pytypes(self, types: Any) -> tuple[list[Any], list[Any]]:
        allowed_types: list[Any] = []
        allowed_subtypes: list[Any] = []
        for t in types:
            type_ = t.split("-", 1)
            if len(type_) == 2:
                type_, subtype = type_
                allowed_subtypes.append(REVERSE_TYPES_MAP[subtype])
            else:
                type_ = type_[0]
            allowed_types.extend(REVERSE_TYPES_MAP[type_])
        return allowed_types, allowed_subtypes

    def _subtype_check(
        self, current: Any, allowed_subtypes: Any, types: Any, function_name: Any
    ) -> None:
        if len(allowed_subtypes) == 1:
            # The easy case, we know up front what type
            # we need to validate.
            allowed_subtypes = allowed_subtypes[0]
            for element in current:
                actual_typename = type(element).__name__
                if actual_typename not in allowed_subtypes:
                    raise exceptions.JMESPathTypeError(
                        function_name, element, actual_typename, types
                    )
        elif len(allowed_subtypes) > 1 and current:
            # Dynamic type validation.  Based on the first
            # type we see, we validate that the remaining types
            # match.
            first = type(current[0]).__name__
            for subtypes in allowed_subtypes:
                if first in subtypes:
                    allowed = subtypes
                    break
            else:
                raise exceptions.JMESPathTypeError(function_name, current[0], first, types)
            for element in current:
                actual_typename = type(element).__name__
                if actual_typename not in allowed:
                    raise exceptions.JMESPathTypeError(
                        function_name, element, actual_typename, types
                    )

    @signature({"types": ["number"]})
    def _func_abs(self, arg: Any) -> Any:
        return abs(arg)

    @signature({"types": ["array-number"]})
    def _func_avg(self, arg: Any) -> Any:
        if arg:
            return sum(arg) / len(arg)
        return None

    @signature({"types": [], "variadic": True})
    def _func_not_null(self, *arguments: Any) -> Any:
        for argument in arguments:
            if argument is not None:
                return argument
        return None

    @signature({"types": []})
    def _func_to_array(self, arg: Any) -> Any:
        if isinstance(arg, list):
            return arg
        return [arg]

    @signature({"types": []})
    def _func_to_string(self, arg: Any) -> Any:
        if isinstance(arg, str):
            return arg
        return json.dumps(arg, separators=(",", ":"), default=str)

    @signature({"types": []})
    def _func_to_number(self, arg: Any) -> Any:
        if isinstance(arg, list | dict | bool):
            return None
        if arg is None:
            return None
        if isinstance(arg, int | float):
            return arg
        try:
            return int(arg)
        except ValueError:
            try:
                return float(arg)
            except ValueError:
                return None

    @signature({"types": ["array", "string"]}, {"types": []})
    def _func_contains(self, subject: Any, search: Any) -> Any:
        return search in subject

    @signature({"types": ["string", "array", "object"]})
    def _func_length(self, arg: Any) -> Any:
        return len(arg)

    @signature({"types": ["string"]}, {"types": ["string"]})
    def _func_ends_with(self, search: Any, suffix: Any) -> Any:
        return search.endswith(suffix)

    @signature({"types": ["string"]}, {"types": ["string"]})
    def _func_starts_with(self, search: Any, suffix: Any) -> Any:
        return search.startswith(suffix)

    @signature({"types": ["array", "string"]})
    def _func_reverse(self, arg: Any) -> Any:
        if isinstance(arg, str):
            return arg[::-1]
        return list(reversed(arg))

    @signature({"types": ["number"]})
    def _func_ceil(self, arg: Any) -> Any:
        return math.ceil(arg)

    @signature({"types": ["number"]})
    def _func_floor(self, arg: Any) -> Any:
        return math.floor(arg)

    @signature({"types": ["string"]}, {"types": ["array-string"]})
    def _func_join(self, separator: Any, array: Any) -> Any:
        return separator.join(array)

    @signature({"types": ["expref"]}, {"types": ["array"]})
    def _func_map(self, expref: Any, arg: Any) -> Any:
        result = []
        for element in arg:
            result.append(expref.visit(expref.expression, element))  # noqa: PERF401
        return result

    @signature({"types": ["array-number", "array-string"]})
    def _func_max(self, arg: Any) -> Any:
        if arg:
            return max(arg)
        return None

    @signature({"types": ["object"], "variadic": True})
    def _func_merge(self, *arguments: Any) -> Any:
        merged: dict[Any, Any] = {}
        for arg in arguments:
            merged.update(arg)
        return merged

    @signature({"types": ["array-number", "array-string"]})
    def _func_min(self, arg: Any) -> Any:
        if arg:
            return min(arg)
        return None

    @signature({"types": ["array-string", "array-number"]})
    def _func_sort(self, arg: Any) -> Any:
        return sorted(arg)

    @signature({"types": ["array-number"]})
    def _func_sum(self, arg: Any) -> Any:
        return sum(arg)

    @signature({"types": ["object"]})
    def _func_keys(self, arg: Any) -> Any:
        # To be consistent with .values()
        # should we also return the indices of a list?
        return list(arg.keys())

    @signature({"types": ["object"]})
    def _func_values(self, arg: Any) -> Any:
        return list(arg.values())

    @signature({"types": []})
    def _func_type(self, arg: Any) -> Any:
        if isinstance(arg, str):
            return "string"
        if isinstance(arg, bool):
            return "boolean"
        if isinstance(arg, list):
            return "array"
        if isinstance(arg, dict):
            return "object"
        if isinstance(arg, float | int):
            return "number"
        if arg is None:
            return "null"
        return None

    @signature({"types": ["array"]}, {"types": ["expref"]})
    def _func_sort_by(self, array: Any, expref: Any) -> Any:
        if not array:
            return array
        # sort_by allows for the expref to be either a number of
        # a string, so we have some special logic to handle this.
        # We evaluate the first array element and verify that it's
        # either a string of a number.  We then create a key function
        # that validates that type, which requires that remaining array
        # elements resolve to the same type as the first element.
        required_type = self._convert_to_jmespath_type(
            type(expref.visit(expref.expression, array[0])).__name__
        )
        if required_type not in ["number", "string"]:
            raise exceptions.JMESPathTypeError(
                "sort_by", array[0], required_type, ["string", "number"]
            )
        keyfunc = self._create_key_func(expref, [required_type], "sort_by")
        return sorted(array, key=keyfunc)

    @signature({"types": ["array"]}, {"types": ["expref"]})
    def _func_min_by(self, array: Any, expref: Any) -> Any:
        keyfunc = self._create_key_func(expref, ["number", "string"], "min_by")
        if array:
            return min(array, key=keyfunc)
        return None

    @signature({"types": ["array"]}, {"types": ["expref"]})
    def _func_max_by(self, array: Any, expref: Any) -> Any:
        keyfunc = self._create_key_func(expref, ["number", "string"], "max_by")
        if array:
            return max(array, key=keyfunc)
        return None

    def _create_key_func(
        self, expref: Any, allowed_types: Any, function_name: Any
    ) -> Callable[[Any], Any]:
        def keyfunc(x: Any) -> Any:
            result = expref.visit(expref.expression, x)
            actual_typename = type(result).__name__
            jmespath_type = self._convert_to_jmespath_type(actual_typename)
            # allowed_types is in term of jmespath types, not python types.
            if jmespath_type not in allowed_types:
                raise exceptions.JMESPathTypeError(
                    function_name, result, jmespath_type, allowed_types
                )
            return result

        return keyfunc

    def _convert_to_jmespath_type(self, pyobject: Any) -> Any:
        return TYPES_MAP.get(pyobject, "unknown")
