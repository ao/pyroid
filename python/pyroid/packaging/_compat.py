# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""packaging's own classes when it is installed, else pyroid's copies; native configuration.

With packaging installed, pyroid raises packaging's exception classes, so ``except
packaging.version.InvalidVersion`` catches pyroid's errors too.

pyroid's versions compare and hash with packaging's through ``Version._key``, which pyroid
builds in the installed release's format: 26.1 and later use tuples of ints, 26.0 and earlier a
tuple holding packaging's ``Infinity`` objects. At import, pyroid checks its keys against the
installed packaging's on a set of versions that covers every branch of the key; if neither
format matches, pyroid's objects and packaging's are kept apart (they do not compare, and
pyroid's do not claim packaging's classes), with a ``RuntimeWarning``.
"""

from __future__ import annotations

import functools
import importlib
import os
import platform
import re
import sys
import warnings
from typing import Any, NamedTuple, cast

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
_pm = _import("packaging.markers") if HAVE_PACKAGING else None
_pr = _import("packaging.requirements") if HAVE_PACKAGING else None
_pt = _import("packaging._tokenizer") if HAVE_PACKAGING else None
_pp = _import("packaging._parser") if HAVE_PACKAGING else None
#: The installed packaging release ("" without it).
INSTALLED = getattr(_import("packaging"), "__version__", "") if HAVE_PACKAGING else ""
# pyroid objects claim packaging's classes (`__class__`) only for the releases whose private
# attributes they reproduce (and whose comparison key they build: see _choose_key_format).
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


class _InvalidMarker(ValueError):
    """Raised when attempting to create a :class:`Marker` from invalid input."""


class _UndefinedComparison(ValueError):
    """Raised when evaluating an unsupported marker comparison."""


class _UndefinedEnvironmentName(KeyError):
    """Raised when evaluating a marker that references a missing environment key."""


class _InvalidRequirement(ValueError):
    """An invalid requirement was found, users should refer to PEP 508."""


class _ParserSyntaxError(Exception):
    """The provided source text could not be parsed correctly."""

    def __init__(self, message: str, *, source: str, span: tuple[int, int]) -> None:
        self.span = span
        self.message = message
        self.source = source
        super().__init__()

    def __str__(self) -> str:
        marker = " " * self.span[0] + "~" * (self.span[1] - self.span[0]) + "^"
        return f"{self.message}\n    {self.source}\n    {marker}"


class _Node:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.value!r})>"

    def serialize(self) -> str:
        raise NotImplementedError


class _Variable(_Node):
    __slots__ = ()

    def serialize(self) -> str:
        return str(self)


class _Value(_Node):
    __slots__ = ()

    def serialize(self) -> str:
        value = str(self)
        if '"' not in value:
            return f'"{value}"'
        if "'" not in value:
            return f"'{value}'"
        raise ValueError("Cannot serialize marker value containing both quote characters")


class _Op(_Node):
    __slots__ = ()

    def serialize(self) -> str:
        return str(self)


for _cls, _mod, _name in (
    (_InvalidVersion, "pyroid.packaging.version", "InvalidVersion"),
    (_VersionTuple, "pyroid.packaging.version", "_Version"),
    (_InvalidSpecifier, "pyroid.packaging.specifiers", "InvalidSpecifier"),
    (_InvalidMarker, "pyroid.packaging.markers", "InvalidMarker"),
    (_UndefinedComparison, "pyroid.packaging.markers", "UndefinedComparison"),
    (_UndefinedEnvironmentName, "pyroid.packaging.markers", "UndefinedEnvironmentName"),
    (_InvalidRequirement, "pyroid.packaging.requirements", "InvalidRequirement"),
    (_ParserSyntaxError, "pyroid.packaging._compat", "ParserSyntaxError"),
    (_Node, "pyroid.packaging._compat", "Node"),
    (_Variable, "pyroid.packaging._compat", "Variable"),
    (_Value, "pyroid.packaging._compat", "Value"),
    (_Op, "pyroid.packaging._compat", "Op"),
):
    _cls.__module__ = _mod
    _cls.__name__ = _cls.__qualname__ = _name
del _cls, _mod, _name

InvalidVersion: type[ValueError] = _pv.InvalidVersion if HAVE_PACKAGING else _InvalidVersion
_Version: Any = getattr(_pv, "_Version", _VersionTuple) if HAVE_PACKAGING else _VersionTuple
InvalidSpecifier: type[ValueError] = _ps.InvalidSpecifier if _ps is not None else _InvalidSpecifier
InvalidMarker: type[ValueError] = getattr(_pm, "InvalidMarker", _InvalidMarker)
UndefinedComparison: type[ValueError] = getattr(_pm, "UndefinedComparison", _UndefinedComparison)
# packaging 26.3 raises its UndefinedEnvironmentName, a KeyError, for a missing environment key.
# Earlier releases raise a plain KeyError (their UndefinedEnvironmentName, a ValueError, is never
# raised): with one of those, pyroid raises a subclass of both, so `except KeyError` and `except
# packaging.markers.UndefinedEnvironmentName` both catch it.
UndefinedEnvironmentName: type[Exception] = getattr(
    _pm, "UndefinedEnvironmentName", _UndefinedEnvironmentName
)
if not issubclass(UndefinedEnvironmentName, KeyError):
    UndefinedEnvironmentName = type(
        "UndefinedEnvironmentName",
        (UndefinedEnvironmentName, KeyError),
        {
            "__module__": "pyroid.packaging.markers",
            "__doc__": _UndefinedEnvironmentName.__doc__,
        },
    )
InvalidRequirement: type[ValueError] = getattr(_pr, "InvalidRequirement", _InvalidRequirement)
ParserSyntaxError: Any = getattr(_pt, "ParserSyntaxError", _ParserSyntaxError)
Variable: Any = getattr(_pp, "Variable", _Variable)
Value: Any = getattr(_pp, "Value", _Value)
Op: Any = getattr(_pp, "Op", _Op)

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
_LETTERS = {
    "alpha": "a",
    "beta": "b",
    "c": "rc",
    "pre": "rc",
    "preview": "rc",
    "rev": "post",
    "r": "post",
}
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
    """Packaging 26.3's ``Version.__init__`` for a non-``str`` argument: raise as it does."""
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


def _restore(cls: Any, state: Any) -> Any:
    """Unpickle a pyroid packaging object: ``cls.__new__(cls)``, then ``__setstate__(state)``.

    packaging's own objects pickle through ``copyreg.__newobj__``; pyroid's cannot (pickle
    checks ``args[0] is obj.__class__``, and ``__class__`` is packaging's class while
    unpatched).
    """
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj


_restore.__module__ = "pyroid.packaging"


def canonicalize_name(name: str, *, validate: bool = False) -> str:
    """Packaging's ``canonicalize_name`` (fast for ``str``; other types fail as there)."""
    if type(name) is str and not validate:
        return cast("str", _native.canonicalize_str(name))
    from pyroid.packaging.utils import canonicalize_name as full

    return full(name, validate=validate)


def _format_full_version(info: Any) -> str:
    version = f"{info.major}.{info.minor}.{info.micro}"
    kind = info.releaselevel
    if kind != "final":
        version += kind[0] + str(info.serial)
    return version


@functools.cache
def _cached_default_environment() -> dict[str, str]:
    """Packaging 26.3's default marker environment, built once (cleared by ``cache_clear``)."""
    iver = _format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
    return {
        "implementation_name": implementation_name,
        "implementation_version": iver,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


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
_PYROID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _caller_stacklevel() -> int:
    """The ``stacklevel`` of the first frame outside pyroid and the import system."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and (
        frame.f_code.co_filename.startswith(_PYROID_DIR)
        or frame.f_code.co_filename.startswith("<frozen importlib")
    ):
        frame = frame.f_back
        level += 1
    return level


def _fallback_warning(what: str) -> None:
    if what not in _WARNED:
        _WARNED.add(what)
        warnings.warn(
            f"pyroid.packaging: {what} runs packaging's own code (slow path). This warning is "
            "shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
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
    version_range = importlib.import_module("packaging.ranges").VersionRange
    if name == "to_range":
        _fallback_warning("SpecifierSet.to_range()")
        return version_range._from_specifier_set(obj)
    # is_subset / is_superset / is_disjoint: SpecifierSet._check_relation_operand, then ranges.
    (other,) = args
    from pyroid.packaging.specifiers import SpecifierSet

    packaging_set = getattr(_ps, "SpecifierSet", SpecifierSet)
    if not isinstance(other, (SpecifierSet, packaging_set)):
        raise TypeError("expected a SpecifierSet")
    if obj._has_arbitrary or cast(Any, other)._has_arbitrary:
        raise ValueError("set relations do not support === specifiers")
    _fallback_warning(f"SpecifierSet.{name}()")
    # The ranges directly: the caller asked for the relation, not for to_range().
    theirs = (
        version_range._from_specifier_set(other)
        if isinstance(other, SpecifierSet)
        else other.to_range()
    )
    return getattr(version_range._from_specifier_set(obj), name)(theirs)


_native._configure(
    {
        "InvalidVersion": InvalidVersion,
        "restore": _restore,
        "nonstr_version": _nonstr_version,
        "_Version": _Version,
        "packaging_Version": _pv.Version if _CLAIM else None,
        "packaging_Version_any": _pv.Version if HAVE_PACKAGING else None,
        "packaging_BaseVersion": getattr(_pv, "_BaseVersion", None) if HAVE_PACKAGING else None,
        "key_cache_slot": HAVE_PACKAGING
        and "_key_cache" in getattr(_pv.Version, "__slots__", ()),
        "InvalidSpecifier": InvalidSpecifier,
        "spec_regex": SPECIFIER_REGEX,
        "delegate": _delegate,
        "packaging_Specifier": getattr(_ps, "Specifier", None) if _CLAIM else None,
        "packaging_SpecifierSet": getattr(_ps, "SpecifierSet", None) if _CLAIM else None,
        "packaging_Specifier_any": getattr(_ps, "Specifier", None),
        "packaging_SpecifierSet_any": getattr(_ps, "SpecifierSet", None),
        "InvalidMarker": InvalidMarker,
        "UndefinedComparison": UndefinedComparison,
        "UndefinedEnvironmentName": UndefinedEnvironmentName,
        "InvalidRequirement": InvalidRequirement,
        "ParserSyntaxError": ParserSyntaxError,
        "cached_default_environment": _cached_default_environment,
        "canonicalize_name": canonicalize_name,
        "Variable": Variable,
        "Value": Value,
        "Op": Op,
        "packaging_Marker": getattr(_pm, "Marker", None) if _CLAIM else None,
        "packaging_Marker_any": getattr(_pm, "Marker", None),
    }
)


# Versions whose keys take every branch of both key formats: pre, post and dev alone and
# together, dev-only, local segments of both kinds, an epoch, trailing zeros, a number past
# 2**64.
_KEY_PROBES = (
    "1",
    "1.0.0",
    "0",
    "2!1.0",
    "1.0a1",
    "1.0b2",
    "1.0rc3",
    "1.0.post4",
    "1.0.dev5",
    "1.0a1.dev2",
    "1.0.post1.dev2",
    "1.0rc1.post2.dev3",
    "1.0+abc.5",
    "1.0+5.abc",
    "1.0.dev1+local",
    "1.2.3." + "9" * 25,
)


def _keys_match() -> bool:
    """Do pyroid's ``_key`` and hash equal the installed packaging's on every probe?"""
    for text in _KEY_PROBES:
        theirs, mine = _pv.Version(text), _native.Version(text)
        if mine._key != theirs._key or hash(mine) != hash(theirs):
            return False
    return True


def _choose_key_format() -> str:
    """Pick the installed packaging's key format; ``"apart"`` (and a warning) for neither."""
    if not HAVE_PACKAGING:
        return "current"
    try:
        if _keys_match():
            return "current"
        structures = importlib.import_module("packaging._structures")
        _native._set_key_format("legacy", structures.Infinity, structures.NegativeInfinity)
        if _keys_match():
            return "legacy"
    except Exception:  # noqa: BLE001  # a packaging whose versions pyroid cannot read
        pass
    _native._set_key_format("apart")
    warnings.warn(
        f"pyroid.packaging: packaging {INSTALLED or '(unknown release)'} orders versions with "
        "a comparison key pyroid does not reproduce, so pyroid's objects and packaging's do not "
        "compare with each other (pyroid's results are packaging 26.3's)",
        RuntimeWarning,
        stacklevel=_caller_stacklevel(),
    )
    return "apart"


#: How pyroid's ``Version._key`` is built for the installed packaging: ``"current"`` (26.1 and
#: later, or no packaging), ``"legacy"`` (26.0 and earlier), or ``"apart"`` (a release whose key
#: pyroid does not reproduce: no comparisons across the two libraries).
KEY_FORMAT = _choose_key_format()
