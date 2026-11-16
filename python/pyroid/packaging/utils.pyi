# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""`packaging.utils`: name, version and filename helpers."""

from typing import Any, NewType, TypeAlias

from pyroid.packaging.version import Version

__all__ = [
    "BuildTag",
    "InvalidName",
    "InvalidSdistFilename",
    "InvalidWheelFilename",
    "NormalizedName",
    "canonicalize_name",
    "canonicalize_version",
    "is_normalized_name",
    "parse_sdist_filename",
    "parse_wheel_filename",
]

BuildTag: TypeAlias = tuple[()] | tuple[int, str]
NormalizedName = NewType("NormalizedName", str)

class InvalidName(ValueError):
    """An invalid distribution name (packaging's own class when installed)."""

class InvalidWheelFilename(ValueError):
    """An invalid wheel filename (packaging's own class when installed)."""

class InvalidSdistFilename(ValueError):
    """An invalid sdist filename (packaging's own class when installed)."""

def canonicalize_name(name: str, *, validate: bool = False) -> NormalizedName:
    """Normalise a project or extra name (PEP 503).

    Raises:
        InvalidName: `validate` is true and `name` is not a valid distribution name.
    """

def is_normalized_name(name: str) -> bool:
    """Whether `name` is a normalised project name."""

def canonicalize_version(version: Version | str, *, strip_trailing_zero: bool = True) -> str:
    """The canonical form of a version (an invalid version string comes back unchanged)."""

def parse_wheel_filename(
    filename: str, *, validate_order: bool = False
) -> tuple[NormalizedName, Version, BuildTag, frozenset[Any]]:
    """`(name, version, build tag, tags)` of a wheel filename (packaging's `Tag` objects).

    Raises:
        InvalidWheelFilename: The filename does not follow the wheel specification.
    """

def parse_sdist_filename(filename: str) -> tuple[NormalizedName, Version]:
    """`(name, version)` of an sdist filename.

    Raises:
        InvalidSdistFilename: The filename is not a valid sdist filename.
    """
