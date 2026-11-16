# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""`packaging.specifiers`: version specifiers."""

import abc
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Final, TypeVar, overload

from pyroid.packaging.version import Version

__all__ = ["BaseSpecifier", "InvalidSpecifier", "Specifier", "SpecifierSet"]

_T = TypeVar("_T")
_V = TypeVar("_V", str, Version)

class InvalidSpecifier(ValueError):
    """Raised for an invalid specifier string (packaging's own class when installed)."""

class BaseSpecifier(metaclass=abc.ABCMeta):
    """packaging's abstract base class (pyroid's `Specifier` and `SpecifierSet` are registered)."""

class Specifier:
    """One version specifier (`>=1.0`), as `packaging.specifiers.Specifier` 26.3."""

    __match_args__: Final[tuple[str, ...]]
    def __init__(self, spec: str = "", prereleases: bool | None = None) -> None:
        """Parse `spec`.

        Raises:
            InvalidSpecifier: `spec` is not a valid specifier.
        """
    @property
    def prereleases(self) -> bool | None:
        """Whether pre-releases match (`None`: PEP 440's default)."""
    @prereleases.setter
    def prereleases(self, value: bool | None) -> None: ...
    @property
    def operator(self) -> str:
        """The operator (`"=="`, `"~="`, ...)."""
    @property
    def version(self) -> str:
        """The version string, as written."""
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __contains__(self, item: str | Version) -> bool: ...
    def contains(self, item: str | Version, prereleases: bool | None = None) -> bool:
        """Whether `item` matches (an unparsable item does not)."""
    @overload
    def filter(self, iterable: Iterable[_V], prereleases: bool | None = None, key: None = None) -> Iterator[_V]: ...
    @overload
    def filter(
        self, iterable: Iterable[_T], prereleases: bool | None = None, key: Callable[[_T], str | Version] = ...
    ) -> Iterator[_T]: ...
    def __getstate__(self) -> tuple[Any, ...]: ...
    def __setstate__(self, state: object) -> None: ...

class SpecifierSet:
    """A set of specifiers (`>=1.0,!=1.5.*`), as `packaging.specifiers.SpecifierSet` 26.3."""

    __match_args__: Final[tuple[str, ...]]
    def __init__(self, specifiers: str | Iterable[Specifier] = "", prereleases: bool | None = None) -> None:
        """Parse a comma-separated specifier string, or take the given specifiers as they are.

        Raises:
            InvalidSpecifier: One of the specifiers is invalid.
        """
    @property
    def prereleases(self) -> bool | None:
        """Whether pre-releases match (`None`: PEP 440's default)."""
    @prereleases.setter
    def prereleases(self, value: bool | None) -> None: ...
    def __hash__(self) -> int: ...
    def __and__(self, other: SpecifierSet | str) -> SpecifierSet: ...
    def __eq__(self, other: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Specifier]: ...
    def __contains__(self, item: str | Version) -> bool: ...
    def contains(
        self, item: str | Version, prereleases: bool | None = None, installed: bool | None = None
    ) -> bool:
        """Whether `item` matches every specifier."""
    @overload
    def filter(self, iterable: Iterable[_V], prereleases: bool | None = None, key: None = None) -> Iterator[_V]: ...
    @overload
    def filter(
        self, iterable: Iterable[_T], prereleases: bool | None = None, key: Callable[[_T], str | Version] = ...
    ) -> Iterator[_T]: ...
    def is_unsatisfiable(self) -> bool:
        """Whether no version can match."""
    def to_range(self) -> Any:
        """The `packaging.ranges.VersionRange` this set accepts (packaging's own code; needs packaging)."""
    def is_subset(self, other: SpecifierSet) -> bool:
        """Whether every version matching this set matches `other` (packaging's own code)."""
    def is_superset(self, other: SpecifierSet) -> bool:
        """Whether every version matching `other` matches this set (packaging's own code)."""
    def is_disjoint(self, other: SpecifierSet) -> bool:
        """Whether no version matches both sets (packaging's own code)."""
    def __getstate__(self) -> tuple[Any, ...]: ...
    def __setstate__(self, state: object) -> None: ...
