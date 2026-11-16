# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""packaging's own classes when it is installed, else pyroid's copies; native configuration.

With packaging installed, pyroid raises packaging's exception classes, so ``except
packaging.version.InvalidVersion`` catches pyroid's errors too.
"""

from __future__ import annotations

import importlib
import re
import sys
import warnings
from typing import Any, NamedTuple

from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import packaging as _native

def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


_pv = _import("packaging.version")
#: Whether packaging itself is importable (its classes are used where pyroid's would be copies).
HAVE_PACKAGING = _pv is not None and not getattr(_pv, "__pyroid__", False)
_ps = _import("packaging.specifiers") if HAVE_PACKAGING else None
#: The installed packaging release ("" without it).
INSTALLED = getattr(_import("packaging"), "__version__", "") if HAVE_PACKAGING else ""
# pyroid objects claim packaging's classes (`__class__`) only for the releases whose private
# attributes they reproduce.
_CLAIM = HAVE_PACKAGING and INSTALLED.startswith("26.")


class _InvalidVersion(ValueError):
    """Raised when a version string is not a valid version."""


class _InvalidSpecifier(ValueError):
    """Raised when attempting to create a :class:`Specifier` with an invalid specifier string."""


class _VersionTuple(NamedTuple):
    epoch: int
    release: tuple[int, ...]
    dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    post: tuple[str, int] | None
    local: tuple[int | str, ...] | None


for _cls, _mod, _name in (
    (_InvalidVersion, "pyroid.packaging.version", "InvalidVersion"),
    (_VersionTuple, "pyroid.packaging.version", "_Version"),
    (_InvalidSpecifier, "pyroid.packaging.specifiers", "InvalidSpecifier"),
):
    _cls.__module__ = _mod
    _cls.__name__ = _cls.__qualname__ = _name

InvalidVersion: type[ValueError] = _pv.InvalidVersion if HAVE_PACKAGING else _InvalidVersion
_Version: Any = getattr(_pv, "_Version", _VersionTuple) if HAVE_PACKAGING else _VersionTuple
InvalidSpecifier: type[ValueError] = (
    _ps.InvalidSpecifier if _ps is not None else _InvalidSpecifier
)

# packaging.version.VERSION_PATTERN, verbatim.
_VERSION_PATTERN = r"""
    v?+                                                   # optional leading v
    (?a:
        (?:(?P<epoch>[0-9]+)!)?+                          # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*+)                 # release segment
        (?P<pre>                                          # pre-release
            [._-]?+
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [._-]?+
            (?P<pre_n>[0-9]+)?
        )?+
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [._-]?
                (?P<post_l>post|rev|r)
                [._-]?
                (?P<post_n2>[0-9]+)?
            )
        )?+
        (?P<dev>                                          # dev release
            [._-]?+
            (?P<dev_l>dev)
            [._-]?+
            (?P<dev_n>[0-9]+)?
        )?+
    )
    (?a:\+
        (?P<local>                                        # local version
            [a-z0-9]+
            (?:[._-][a-z0-9]+)*+
        )
    )?+
"""

_VERSION_PATTERN_OLD = _VERSION_PATTERN.replace("*+", "*").replace("?+", "?")

VERSION_PATTERN = (
    _VERSION_PATTERN_OLD
    if (sys.implementation.name == "cpython" and sys.version_info < (3, 11, 5))
    or (sys.implementation.name == "pypy" and sys.version_info < (3, 11, 13))
    or sys.version_info < (3, 11)
    else _VERSION_PATTERN
)

_REGEX = re.compile(r"\s*" + VERSION_PATTERN + r"\s*", re.VERBOSE | re.IGNORECASE)
_SIMPLE_VERSION_INDICATORS = frozenset(".0123456789")
_LETTERS = {"alpha": "a", "beta": "b", "c": "rc", "pre": "rc", "preview": "rc", "rev": "post", "r": "post"}
_SEPARATORS = re.compile(r"[\._-]")


def _letter(letter: str | None, number: Any) -> tuple[str, int] | None:
    if letter:
        letter = letter.lower()
        letter = _LETTERS.get(letter, letter)
        return letter, int(number or 0)
    if number:
        return "post", int(number)
    return None


def _nonstr_version(version: Any) -> tuple[Any, ...]:
    """packaging 26.3's ``Version.__init__`` for a non-``str`` argument: raise as it does."""
    try:
        is_simple = _SIMPLE_VERSION_INDICATORS.issuperset(version)
    except TypeError:
        raise InvalidVersion(f"Invalid version: {version!r}") from None
    if is_simple:
        try:
            release = tuple(map(int, version.split(".")))
        except AttributeError:
            raise InvalidVersion(f"Invalid version: {version!r}") from None
        except ValueError:
            if "" in version.split("."):
                raise InvalidVersion(f"Invalid version: {version!r}") from None
            raise
        return (0, release, None, None, None, None)
    try:
        match = _REGEX.fullmatch(version)
    except TypeError:
        raise InvalidVersion(f"Invalid version: {version!r}") from None
    if not match:
        raise InvalidVersion(f"Invalid version: {version!r}")
    local = match.group("local")
    return (
        int(match.group("epoch")) if match.group("epoch") else 0,
        tuple(map(int, match.group("release").split("."))),
        _letter(match.group("pre_l"), match.group("pre_n")),
        _letter(match.group("post_l"), match.group("post_n1") or match.group("post_n2")),
        _letter(match.group("dev_l"), match.group("dev_n")),
        None
        if local is None
        else tuple(p.lower() if not p.isdigit() else int(p) for p in _SEPARATORS.split(local)),
    )


# Specifier._specifier_regex_str, verbatim.
SPECIFIER_REGEX_STR = r"""
        (?:
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                ===  # Only match for the identity operator
                \s*
                [^\s;)]*  # The arbitrary version can be just about anything,
                          # we match everything except for whitespace, a
                          # semi-colon for marker support, and a closing paren
                          # since versions can be enclosed in them.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?:==|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release

                # You cannot use a wild card and a pre-release, post-release, a dev or
                # local version together so group them with a | and make them optional.
                (?:
                    \.\*  # Wild card syntax of .*
                    |
                    (?a:                                  # pre release
                        [-_\.]?
                        (alpha|beta|preview|pre|a|b|c|rc)
                        [-_\.]?
                        [0-9]*
                    )?
                    (?a:                                  # post release
                        (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                    )?
                    (?a:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?a:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?:~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?:<=|>=|<|>)

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?a:                   # pre release
                    [-_\.]?
                    (alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?a:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(post|rev|r)[-_\.]?[0-9]*)
                )?
                (?a:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """

SPECIFIER_REGEX = re.compile(r"\s*" + SPECIFIER_REGEX_STR + r"\s*", re.VERBOSE | re.IGNORECASE)

_WARNED: set[str] = set()


def _fallback_warning(what: str) -> None:
    if what not in _WARNED:
        _WARNED.add(what)
        warnings.warn(
            f"pyroid.packaging: {what} runs packaging's own code (slow path). This warning is "
            "shown once.",
            FallbackWarning,
            stacklevel=4,
        )


def _ranges_module() -> Any:
    ranges = _import("packaging._ranges") if HAVE_PACKAGING else None
    if ranges is None:
        raise PyroidError(
            "SpecifierSet.to_range(), is_subset(), is_superset() and is_disjoint() use "
            "packaging's version-range code: pip install packaging==26.3"
        )
    return ranges


def _delegate(name: str, obj: Any, *args: Any) -> Any:
    """Calls pyroid hands to packaging's own code (its version-range algebra)."""
    ranges = _ranges_module()
    if name == "_to_ranges":
        op, version = obj.operator, obj.version
        if op == "===":
            return ranges.FULL_RANGE
        # The class packaging's ranges module uses (pyroid's under the patch).
        version_cls = sys.modules["packaging.version"].Version
        return ranges.bounds_for_spec(op, version, version_cls(version.removesuffix(".*")))
    if name == "_get_ranges":
        return ranges.intersect_specifier_bounds(s._to_ranges() for s in obj._specs)
    from packaging.ranges import VersionRange

    if name == "to_range":
        _fallback_warning("SpecifierSet.to_range()")
        return VersionRange._from_specifier_set(obj)
    # is_subset / is_superset / is_disjoint: SpecifierSet._check_relation_operand, then ranges.
    (other,) = args
    from pyroid.packaging.specifiers import SpecifierSet

    packaging_set = getattr(_ps, "SpecifierSet", SpecifierSet)
    if not isinstance(other, (SpecifierSet, packaging_set)):
        raise TypeError("expected a SpecifierSet")
    if obj._has_arbitrary or other._has_arbitrary:
        raise ValueError("set relations do not support === specifiers")
    _fallback_warning(f"SpecifierSet.{name}()")
    return getattr(obj.to_range(), name)(other.to_range())


_native._configure(
    {
        "InvalidVersion": InvalidVersion,
        "nonstr_version": _nonstr_version,
        "_Version": _Version,
        "packaging_Version": _pv.Version if _CLAIM else None,
        "packaging_BaseVersion": getattr(_pv, "_BaseVersion", None) if HAVE_PACKAGING else None,
        "InvalidSpecifier": InvalidSpecifier,
        "spec_regex": SPECIFIER_REGEX,
        "delegate": _delegate,
        "packaging_Specifier": _ps.Specifier if _CLAIM else None,
        "packaging_SpecifierSet": _ps.SpecifierSet if _CLAIM else None,
    }
)
