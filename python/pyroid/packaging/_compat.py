# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""packaging's own classes when it is installed, else pyroid's copies; native configuration.

With packaging installed, pyroid raises packaging's exception classes, so ``except
packaging.version.InvalidVersion`` catches pyroid's errors too.
"""

from __future__ import annotations

import re
import sys
from typing import Any, NamedTuple

from pyroid._pyroid import packaging as _native

try:
    import packaging.version as _pv
except ImportError:  # pragma: no cover - exercised by the no-packaging CI job
    _pv = None  # type: ignore[assignment]

#: Whether packaging itself is importable (its classes are used where pyroid's would be copies).
HAVE_PACKAGING = _pv is not None and not getattr(_pv, "__pyroid__", False)


class _InvalidVersion(ValueError):
    """Raised when a version string is not a valid version.

    >>> Version("invalid")
    Traceback (most recent call last):
        ...
    packaging.version.InvalidVersion: Invalid version: 'invalid'
    """


_InvalidVersion.__module__ = "pyroid.packaging.version"
_InvalidVersion.__name__ = _InvalidVersion.__qualname__ = "InvalidVersion"


class _VersionTuple(NamedTuple):
    epoch: int
    release: tuple[int, ...]
    dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    post: tuple[str, int] | None
    local: tuple[int | str, ...] | None


_VersionTuple.__module__ = "pyroid.packaging.version"
_VersionTuple.__name__ = _VersionTuple.__qualname__ = "_Version"

if HAVE_PACKAGING:
    InvalidVersion: type[ValueError] = _pv.InvalidVersion
    _Version: Any = getattr(_pv, "_Version", _VersionTuple)
    PACKAGING_VERSION_CLASS: type | None = _pv.Version
    PACKAGING_BASE_VERSION: type | None = getattr(_pv, "_BaseVersion", None)
else:
    InvalidVersion = _InvalidVersion
    _Version = _VersionTuple
    PACKAGING_VERSION_CLASS = None
    PACKAGING_BASE_VERSION = None

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


_native._configure(
    {
        "InvalidVersion": InvalidVersion,
        "nonstr_version": _nonstr_version,
        "_Version": _Version,
        "packaging_Version": PACKAGING_VERSION_CLASS,
        "packaging_BaseVersion": PACKAGING_BASE_VERSION,
    }
)
