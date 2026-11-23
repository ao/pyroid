# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""A drop-in for packaging 26.3's versions, specifiers, markers and requirements, in Rust.

``from pyroid.packaging.version import Version`` (and ``specifiers``, ``markers``,
``requirements``, ``utils``) gives classes that behave as packaging's: the same parsing,
ordering, hashing, string forms and exceptions. With packaging installed, the exceptions are
packaging's own classes and ``isinstance(v, packaging.version.Version)`` is true for pyroid's
versions.
"""

from __future__ import annotations

from pyroid import _versions
from pyroid._pyroid import packaging as _native
from pyroid.packaging import _compat, markers, requirements, specifiers, utils, version

# Pickles of pyroid's objects refer to `pyroid.packaging._restore`.
_restore = _compat._restore

#: The packaging release whose behaviour this module reproduces.
PACKAGING_VERSION = "26.3"
__version__ = PACKAGING_VERSION
_versions.check(__name__, "packaging", f"packaging=={PACKAGING_VERSION}")

#: The packaging submodules pyroid ports (``pyroid.patch("packaging")`` without packaging installed
#: provides shims for these).
PORTED = ("version", "specifiers", "markers", "requirements", "utils")

contains_batch = _native.contains_batch
filter_batch = _native.filter_batch
parse_batch = _native.parse_batch
sort_versions = _native.sort_versions

__all__ = [
    "PACKAGING_VERSION",
    "PORTED",
    "contains_batch",
    "filter_batch",
    "markers",
    "parse_batch",
    "requirements",
    "sort_versions",
    "specifiers",
    "utils",
    "version",
]

del _compat


def _patch_targets() -> list[tuple[str, str, object]]:
    """``(module, name, pyroid's object)`` for each packaging object ``patch()`` replaces.

    ``pyroid.patch("packaging")`` replaces the original object everywhere a packaging module holds
    it (``packaging.specifiers.Version`` too), so packaging's own modules use pyroid's classes.
    """
    return [
        ("packaging.version", "Version", version.Version),
        ("packaging.version", "parse", version.parse),
        ("packaging.specifiers", "Specifier", specifiers.Specifier),
        ("packaging.specifiers", "SpecifierSet", specifiers.SpecifierSet),
        ("packaging.markers", "Marker", markers.Marker),
        ("packaging.markers", "default_environment", markers.default_environment),
        ("packaging.markers", "_cached_default_environment", markers._cached_default_environment),
        ("packaging.requirements", "Requirement", requirements.Requirement),
        ("packaging.utils", "canonicalize_name", utils.canonicalize_name),
        ("packaging.utils", "canonicalize_version", utils.canonicalize_version),
        ("packaging.utils", "is_normalized_name", utils.is_normalized_name),
        ("packaging.utils", "parse_wheel_filename", utils.parse_wheel_filename),
        ("packaging.utils", "parse_sdist_filename", utils.parse_sdist_filename),
    ]
