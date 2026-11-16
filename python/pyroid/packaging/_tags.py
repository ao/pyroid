# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""packaging 26.3's ``Tag`` and ``parse_tag``, for ``parse_wheel_filename`` without packaging.

With packaging installed, :mod:`pyroid.packaging.utils` uses ``packaging.tags`` itself.
"""

from __future__ import annotations


class UnsortedTagsError(ValueError):
    """Raised when a tag component is not in sorted order per PEP 425."""


class InvalidTag(ValueError):
    """Raised when an interpreter component is not an identifier, a tag component is empty, or
    a tag does not have exactly three components."""


class TooManyTagsError(ValueError):
    """Raised when a compressed tag set exceeds the configured limit."""


class Tag:
    """A representation of the tag triple for a wheel."""

    __slots__ = ["_abi", "_hash", "_interpreter", "_platform"]

    def __init__(self, interpreter: str, abi: str, platform: str) -> None:
        self._interpreter = interpreter.lower()
        self._abi = abi.lower()
        self._platform = platform.lower()
        self._hash = hash((self._interpreter, self._abi, self._platform))

    @property
    def interpreter(self) -> str:
        """The interpreter name, e.g. ``"py"``."""
        return self._interpreter

    @property
    def abi(self) -> str:
        """The supported ABI."""
        return self._abi

    @property
    def platform(self) -> str:
        """The OS/platform."""
        return self._platform

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented
        return (
            (self._hash == other._hash)
            and (self._platform == other._platform)
            and (self._abi == other._abi)
            and (self._interpreter == other._interpreter)
        )

    def __hash__(self) -> int:
        return self._hash

    def __str__(self) -> str:
        return f"{self._interpreter}-{self._abi}-{self._platform}"

    def __repr__(self) -> str:
        return f"<{self} @ {id(self)}>"

    def __getstate__(self) -> tuple[str, str, str]:
        return (self._interpreter, self._abi, self._platform)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, tuple) and len(state) == 3 and all(isinstance(s, str) for s in state):
            self._interpreter, self._abi, self._platform = state
            self._hash = hash((self._interpreter, self._abi, self._platform))
            return
        raise TypeError(f"Cannot restore Tag from {state!r}")


def parse_tag(tag: str, *, validate_order: bool = False, limit: int | None = None) -> frozenset[Tag]:
    """Parse a tag (e.g. ``py3-none-any``) into a frozenset of :class:`Tag` instances."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    component_parts = [component.split(".") for component in tag.split("-")]
    for parts in component_parts:
        if "" in parts:
            component = ".".join(parts)
            raise InvalidTag(f"Tag {tag!r} has an empty component: {component!r}")
        if validate_order and parts != sorted(parts):
            component = ".".join(parts)
            raise UnsortedTagsError(f"Tag component {component!r} is not in sorted order per PEP 425")
    tag_count = 1
    for parts in component_parts:
        tag_count *= len(parts)
    if limit is not None and tag_count > limit:
        raise TooManyTagsError(
            f"Compressed tag set would generate {tag_count} tags, exceeding limit {limit}"
        )
    try:
        interpreters, abis, platforms = component_parts
    except ValueError as exc:
        raise InvalidTag(f"Tag {tag!r} must have exactly three components") from exc
    for interpreter in interpreters:
        if not interpreter.isidentifier():
            raise InvalidTag(f"Tag {tag!r} has an invalid interpreter: {interpreter!r}")
    return frozenset(
        Tag(interpreter, abi, platform_)
        for interpreter in interpreters
        for abi in abis
        for platform_ in platforms
    )
