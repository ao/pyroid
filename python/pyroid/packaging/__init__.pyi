# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""A drop-in for packaging 26.3's versions, specifiers, markers and requirements, in Rust.

```python
from pyroid.packaging.version import Version            # was: from packaging.version import Version
from pyroid.packaging.specifiers import SpecifierSet

Version("1.0rc1") < Version("1.0")                    # True
SpecifierSet(">=1.0,!=1.5.*").contains("1.6.2")       # True
```

`pyroid.packaging.version`, `specifiers`, `markers`, `requirements` and `utils` give packaging
26.3's results: the same parsing, ordering, hashes, strings, pre-release rules, marker values and
exceptions. With packaging installed, the exception classes are packaging's own, pyroid's
objects compare and hash equal to packaging's, and `isinstance(v, packaging.version.Version)`
is true for pyroid's versions. `pyroid.patch("packaging")` makes code that imports packaging use
these classes.

The batch functions below parse on pyroid's thread pool.
"""

from collections.abc import Sequence
from typing import Any, Final, Literal, TypeAlias, TypeVar, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings as ArrowArray
from pyroid.packaging import markers as markers
from pyroid.packaging import requirements as requirements
from pyroid.packaging import specifiers as specifiers
from pyroid.packaging import utils as utils
from pyroid.packaging import version as version
from pyroid.packaging.specifiers import Specifier, SpecifierSet
from pyroid.packaging.version import Version

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

#: The packaging release whose behaviour this module reproduces.
PACKAGING_VERSION: Final[str]
__version__: Final[str]
#: The packaging submodules pyroid ports.
PORTED: Final[tuple[str, ...]]

_T = TypeVar("_T")
_Specs: TypeAlias = SpecifierSet | Specifier | str

@overload
def parse_batch(
    versions: Sequence[str | Version] | ArrowArray,
    /,
    *,
    errors: Literal["raise"] = "raise",
    threads: int | None = None,
) -> list[Version]: ...
@overload
def parse_batch(
    versions: Sequence[str | Version] | ArrowArray,
    /,
    *,
    errors: Literal["none"],
    threads: int | None = None,
) -> list[Version | None]:
    """Parse many versions at once.

    Args:
        versions: Version strings (or `Version` objects) in a list or tuple, or an Arrow string
            array.
        errors: `"raise"`: the first invalid version raises `InvalidVersion`, with a note naming
            its index. `"none"`: `None` in its place.
        threads: At most this many tasks at once on pyroid's shared pool (`None`: all of it).

    Returns:
        One `Version` per item (`None` for an Arrow null, or an invalid one with
        `errors="none"`).
    """

def sort_versions(
    versions: Sequence[_T] | ArrowArray, /, *, reverse: bool = False, threads: int | None = None
) -> list[_T]:
    """Sort versions in PEP 440 order, as `sorted(versions, key=Version)` does (stable).

    Args:
        versions: Version strings and/or `Version` objects (pyroid's or packaging's) in a list
            or tuple, or an Arrow string array.
        reverse: Newest first (equal versions keep their order, as with `sorted`).
        threads: At most this many tasks at once on pyroid's shared pool.

    Returns:
        The given objects, sorted (new `str` objects for Arrow input).

    Raises:
        InvalidVersion: An item is not a version (the first one in input order).
    """

@overload
def contains_batch(
    specifiers: _Specs,
    versions: list[Any],
    /,
    *,
    prereleases: bool | None = None,
    installed: bool | None = None,
    threads: int | None = None,
) -> list[bool]: ...
@overload
def contains_batch(
    specifiers: _Specs,
    versions: tuple[Any, ...],
    /,
    *,
    prereleases: bool | None = None,
    installed: bool | None = None,
    threads: int | None = None,
) -> tuple[bool, ...]: ...
@overload
def contains_batch(
    specifiers: _Specs,
    versions: ArrowArray,
    /,
    *,
    prereleases: bool | None = None,
    installed: bool | None = None,
    threads: int | None = None,
) -> Column[bool]:
    """`[specifiers.contains(v, prereleases, installed) for v in versions]`, natively.

    Args:
        specifiers: A `SpecifierSet`, a `Specifier` or a specifier string.
        versions: Version strings and/or `Version` objects in a list or tuple, or an Arrow
            string array.
        prereleases: As for `contains`.
        installed: As for `SpecifierSet.contains`.
        threads: At most this many tasks at once on pyroid's shared pool.

    Returns:
        A list (a tuple for a tuple) of bools, or a `pyroid.Column` of bools for Arrow input,
        where nulls stay null.
    """

def filter_batch(
    specifiers: _Specs,
    versions: Sequence[_T] | ArrowArray,
    /,
    *,
    prereleases: bool | None = None,
    threads: int | None = None,
) -> list[_T]:
    """`list(specifiers.filter(versions, prereleases))`, natively.

    Args:
        specifiers: A `SpecifierSet`, a `Specifier` or a specifier string.
        versions: Version strings and/or `Version` objects in a list or tuple, or an Arrow
            string array.
        prereleases: As for `filter`: `None` keeps pre-releases only when nothing else matches.
        threads: At most this many tasks at once on pyroid's shared pool.

    Returns:
        The matching objects, in input order (new `str` objects for Arrow input).
    """
