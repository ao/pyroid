# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""``packaging.specifiers``: version specifiers (:class:`Specifier`, :class:`SpecifierSet`)."""

from __future__ import annotations

import abc
from typing import Any

from pyroid._pyroid import packaging as _native
from pyroid.packaging import _compat
from pyroid.packaging._compat import InvalidSpecifier

__all__ = ["BaseSpecifier", "InvalidSpecifier", "Specifier", "SpecifierSet"]

Specifier = _native.Specifier
SpecifierSet = _native.SpecifierSet

if _compat._ps is not None:
    BaseSpecifier: Any = _compat._ps.BaseSpecifier
else:

    class BaseSpecifier(metaclass=abc.ABCMeta):  # type: ignore[no-redef]  # noqa: B024
        """Abstract base class for :class:`Specifier` and :class:`SpecifierSet`."""

        __slots__ = ()


# isinstance(x, BaseSpecifier) holds for pyroid's specifiers (also under the patch, where the
# __class__ claim is off).
BaseSpecifier.register(Specifier)
BaseSpecifier.register(SpecifierSet)

Specifier.__slots__ = ("_prereleases", "_ranges", "_spec", "_spec_version")
SpecifierSet.__slots__ = (
    "_canonicalized",
    "_has_arbitrary",
    "_is_unsatisfiable",
    "_prereleases",
    "_ranges",
    "_specs",
)
Specifier._regex = _compat.SPECIFIER_REGEX
Specifier._specifier_regex_str = _compat.SPECIFIER_REGEX_STR
# packaging keeps this unused table for backward compatibility; pip-requirements-parser (and so
# pip-audit) reads it at import.
Specifier._operators = {
    "~=": "compatible",
    "==": "equal",
    "!=": "not_equal",
    "<=": "less_than_equal",
    ">=": "greater_than_equal",
    "<": "less_than",
    ">": "greater_than",
    "===": "arbitrary",
}


def __dir__() -> list[str]:
    return __all__
