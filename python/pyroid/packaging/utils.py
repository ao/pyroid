# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""``packaging.utils``: name, version and filename helpers."""

from __future__ import annotations

import re
from typing import Any, NewType, cast

from pyroid._pyroid import packaging as _native
from pyroid.packaging import _compat
from pyroid.packaging.version import InvalidVersion, Version

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

_pu = _compat._import("packaging.utils") if _compat.HAVE_PACKAGING else None
_ptags = _compat._import("packaging.tags") if _compat.HAVE_PACKAGING else None
if _ptags is None or not hasattr(_ptags, "InvalidTag"):
    from pyroid.packaging import _tags as _ptags

Tag = _ptags.Tag
parse_tag = _ptags.parse_tag
InvalidTag: Any = _ptags.InvalidTag
UnsortedTagsError: Any = getattr(_ptags, "UnsortedTagsError", ValueError)

BuildTag = tuple[()] | tuple[int, str]
NormalizedName = NewType("NormalizedName", str)


class _InvalidName(ValueError):
    """An invalid distribution name; users should refer to the packaging user guide."""


class _InvalidWheelFilename(ValueError):
    """An invalid wheel filename was found, users should refer to PEP 427."""


class _InvalidSdistFilename(ValueError):
    """An invalid sdist filename was found, users should refer to the packaging user guide."""


for _cls, _name in (
    (_InvalidName, "InvalidName"),
    (_InvalidWheelFilename, "InvalidWheelFilename"),
    (_InvalidSdistFilename, "InvalidSdistFilename"),
):
    _cls.__module__ = __name__
    _cls.__name__ = _cls.__qualname__ = _name

InvalidName: type[ValueError] = getattr(_pu, "InvalidName", _InvalidName)
InvalidWheelFilename: type[ValueError] = getattr(_pu, "InvalidWheelFilename", _InvalidWheelFilename)
InvalidSdistFilename: type[ValueError] = getattr(_pu, "InvalidSdistFilename", _InvalidSdistFilename)

_validate_regex = re.compile(r"[a-z0-9]|[a-z0-9][a-z0-9._-]*[a-z0-9]", re.IGNORECASE | re.ASCII)
_normalized_regex = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.ASCII)
_build_tag_regex = re.compile(r"(\d+)(.*)", re.ASCII)
_wheel_name_regex = re.compile(r"^[\w._]+\Z", re.UNICODE)
_canonicalize_str = _native.canonicalize_str


def canonicalize_name(name: str, *, validate: bool = False) -> NormalizedName:
    """Normalise a project or extra name (PEP 503).

    Raises:
        InvalidName: `validate` is true and `name` is not a valid distribution name.
    """
    if validate and not _validate_regex.fullmatch(name):
        raise InvalidName(f"name is invalid: {name!r}")
    if type(name) is str:
        return cast("NormalizedName", _canonicalize_str(name))
    value = name.lower().replace("_", "-").replace(".", "-")
    while "--" in value:
        value = value.replace("--", "-")
    return cast("NormalizedName", value)


def is_normalized_name(name: str) -> bool:
    """Whether `name` is a normalised project name."""
    return _normalized_regex.fullmatch(name) is not None


def canonicalize_version(version: Version | str, *, strip_trailing_zero: bool = True) -> str:
    """The canonical form of a version (an invalid version string comes back unchanged)."""
    return _native.canonicalize_version(version, strip_trailing_zero)


def parse_wheel_filename(
    filename: str,
    *,
    validate_order: bool = False,
) -> tuple[NormalizedName, Version, BuildTag, frozenset[Any]]:
    """`(name, version, build tag, tags)` of a wheel filename (packaging's `Tag` objects).

    Raises:
        InvalidWheelFilename: The filename does not follow the wheel specification.
    """
    if not filename.endswith(".whl"):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (extension must be '.whl'): {filename!r}"
        )

    filename = filename[:-4]
    dashes = filename.count("-")
    if dashes not in (4, 5):
        raise InvalidWheelFilename(f"Invalid wheel filename (wrong number of parts): {filename!r}")

    parts = filename.split("-", dashes - 2)
    name_part = parts[0]
    if "__" in name_part or _wheel_name_regex.match(name_part) is None:
        raise InvalidWheelFilename(f"Invalid project name: {filename!r}")
    name = canonicalize_name(name_part)

    try:
        version = Version(parts[1])
    except InvalidVersion as e:
        raise InvalidWheelFilename(f"Invalid wheel filename (invalid version): {filename!r}") from e

    if dashes == 5:
        build_part = parts[2]
        build_match = _build_tag_regex.match(build_part)
        if build_match is None:
            raise InvalidWheelFilename(f"Invalid build number: {build_part} in {filename!r}")
        build = cast("BuildTag", (int(build_match.group(1)), build_match.group(2)))
    else:
        build = ()
    tag_str = parts[-1]
    try:
        tags = parse_tag(tag_str, validate_order=validate_order)
    except UnsortedTagsError:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (compressed tag set components must be in "
            f"sorted order per PEP 425): {filename!r}"
        ) from None
    except InvalidTag:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (invalid tag component): {filename!r}"
        ) from None
    return (name, version, build, tags)


def parse_sdist_filename(filename: str) -> tuple[NormalizedName, Version]:
    """`(name, version)` of an sdist filename.

    Raises:
        InvalidSdistFilename: The filename is not a valid sdist filename.
    """
    if filename.endswith(".tar.gz"):
        file_stem = filename[: -len(".tar.gz")]
    elif filename.endswith(".zip"):
        file_stem = filename[: -len(".zip")]
    else:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (extension must be '.tar.gz' or '.zip'): {filename!r}"
        )

    name_part, sep, version_part = file_stem.rpartition("-")
    if not sep:
        raise InvalidSdistFilename(f"Invalid sdist filename: {filename!r}")
    if not name_part:
        raise InvalidSdistFilename(f"Invalid sdist filename (empty project name): {filename!r}")

    name = canonicalize_name(name_part)

    try:
        version = Version(version_part)
    except InvalidVersion as e:
        raise InvalidSdistFilename(f"Invalid sdist filename (invalid version): {filename!r}") from e

    return (name, version)


def __dir__() -> list[str]:
    return __all__
