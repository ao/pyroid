# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""``packaging.requirements``: PEP 508 dependency specifiers (:class:`Requirement`)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pyroid._pyroid import packaging as _native
from pyroid.packaging import _compat
from pyroid.packaging._compat import InvalidRequirement, InvalidSpecifier
from pyroid.packaging.markers import Marker
from pyroid.packaging.specifiers import SpecifierSet

__all__ = ["InvalidRequirement", "Requirement"]

_parse = _native.parse_requirement
_canonicalize = _native.canonicalize_str
_PACKAGING_REQUIREMENT = _compat._pr.Requirement if _compat._CLAIM else None


class Requirement:
    """Parse a requirement.

    Parse a given requirement string into its parts, such as name, specifier, URL, and extras.
    Raises InvalidRequirement on a badly-formed requirement string.
    """

    __slots__ = ("extras", "marker", "name", "specifier", "url")

    def __init__(self, requirement_string: str) -> None:
        name, url, extras, specifier, marker = _parse(requirement_string)
        self.name: str = name
        self.url: str | None = url or None
        self.extras: set[str] = set(extras)
        try:
            self.specifier: SpecifierSet = SpecifierSet(specifier)
        except InvalidSpecifier as e:
            raise InvalidRequirement(str(e)) from e
        self.marker: Marker | None = marker

    def _iter_parts(self, name: str) -> Iterator[str]:
        yield name

        if self.extras:
            formatted_extras = ",".join(sorted(self.extras))
            yield f"[{formatted_extras}]"

        if self.specifier:
            yield str(self.specifier)

        if self.url:
            yield f" @ {self.url}"
            if self.marker:
                yield " "

        if self.marker:
            yield f"; {self.marker}"

    def __getstate__(self) -> tuple[str, bool | None]:
        return (str(self), self.specifier._prereleases)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, str):
            requirement_string: str = state
            prereleases: bool | None = None
        elif (
            isinstance(state, tuple)
            and len(state) == 2
            and isinstance(state[0], str)
            and (state[1] is None or isinstance(state[1], bool))
        ):
            requirement_string, prereleases = state
        elif isinstance(state, dict) and state.keys() >= set(self.__slots__):
            for key in self.__slots__:
                setattr(self, key, state[key])
            return
        else:
            raise TypeError(f"Cannot restore Requirement from {state!r}")

        try:
            tmp = Requirement(requirement_string)
        except InvalidRequirement as exc:
            raise TypeError(f"Cannot restore Requirement from {state!r}") from exc
        self.name = tmp.name
        self.url = tmp.url
        self.extras = tmp.extras
        self.specifier = tmp.specifier
        self.specifier._prereleases = prereleases
        self.marker = tmp.marker

    def __reduce_ex__(self, protocol: Any) -> tuple[Any, ...]:
        # Not copyreg.__newobj__: pickle compares its class argument with __class__ (the claim).
        return (_compat._restore, (type(self), self.__getstate__()))

    def __str__(self) -> str:
        return "".join(self._iter_parts(self.name))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({str(self)!r})>"

    def __hash__(self) -> int:
        return hash(
            (
                _canonical(self.name),
                frozenset(_canonical(e) for e in self.extras),
                self.specifier,
                self.url,
                self.marker,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _REQUIREMENT_TYPES):
            return NotImplemented

        self_extras = frozenset(_canonical(e) for e in self.extras)
        other_extras = frozenset(_canonical(e) for e in other.extras)  # type: ignore[attr-defined]
        return (
            _canonical(self.name) == _canonical(other.name)  # type: ignore[attr-defined]
            and self_extras == other_extras
            and self.specifier == other.specifier  # type: ignore[attr-defined]
            and self.url == other.url  # type: ignore[attr-defined]
            and self.marker == other.marker  # type: ignore[attr-defined]
        )

    @property  # type: ignore[misc]
    def __class__(self) -> Any:
        # isinstance(r, packaging.requirements.Requirement) for pyroid's (unpatched; not for
        # subclasses).
        cls = type(self)
        if cls is Requirement and _PACKAGING_REQUIREMENT is not None and _native._claims():
            return _PACKAGING_REQUIREMENT
        return cls


def _canonical(name: Any) -> str:
    return _canonicalize(name) if type(name) is str else _compat.canonicalize_name(name)


_REQUIREMENT_TYPES: tuple[type, ...] = (Requirement,) + (
    (_compat._pr.Requirement,) if _compat._pr is not None else ()
)


def __dir__() -> list[str]:
    return __all__
