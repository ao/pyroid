# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""`packaging.version`: PEP 440 versions."""

from typing import Any, Final, Literal

from typing_extensions import Self

__all__ = ["VERSION_PATTERN", "InvalidVersion", "Version", "normalize_pre", "parse"]

#: The regular expression packaging matches versions with (`re.VERBOSE | re.IGNORECASE`).
VERSION_PATTERN: Final[str]

class InvalidVersion(ValueError):
    """Raised when a version string is not a valid version (packaging's own class when installed)."""

class Version:
    """A PEP 440 version, as `packaging.version.Version` 26.3.

    Comparable, hashable and immutable (`__replace__` makes changed copies). It compares and
    hashes equal to packaging's `Version` for the same version, and with packaging installed
    `isinstance(v, packaging.version.Version)` is true.
    """

    __match_args__: Final[tuple[str, ...]]
    def __init__(self, version: str) -> None:
        """Parse and normalise `version`.

        Raises:
            InvalidVersion: `version` is not a valid PEP 440 version.
        """
    @classmethod
    def from_parts(
        cls,
        *,
        epoch: int = 0,
        release: tuple[int, ...],
        pre: tuple[str, int] | None = None,
        post: int | None = None,
        dev: int | None = None,
        local: str | None = None,
    ) -> Self:
        """Return a new version composed of the various parts (the pre-release letter is normalised)."""
    def __replace__(self, **kwargs: Any) -> Self:
        """Return a new version with parts replaced (`None` clears one); `self` when nothing changes."""
    def __hash__(self) -> int: ...
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __getstate__(self) -> tuple[Any, ...]: ...
    def __setstate__(self, state: object) -> None: ...
    @property
    def epoch(self) -> int:
        """The epoch of the version."""
    @property
    def release(self) -> tuple[int, ...]:
        """The components of the release segment (trailing zeros kept)."""
    @property
    def pre(self) -> tuple[Literal["a", "b", "rc"], int] | None:
        """The pre-release segment."""
    @property
    def post(self) -> int | None:
        """The post-release number."""
    @property
    def dev(self) -> int | None:
        """The development release number."""
    @property
    def local(self) -> str | None:
        """The local version label."""
    @property
    def public(self) -> str:
        """The version without its local label."""
    @property
    def base_version(self) -> str:
        """The epoch and release segment."""
    @property
    def is_prerelease(self) -> bool:
        """Whether this is a pre-release (pre or dev)."""
    @property
    def is_postrelease(self) -> bool:
        """Whether this is a post-release."""
    @property
    def is_devrelease(self) -> bool:
        """Whether this is a development release."""
    @property
    def major(self) -> int:
        """The first item of `release`, or 0."""
    @property
    def minor(self) -> int:
        """The second item of `release`, or 0."""
    @property
    def micro(self) -> int:
        """The third item of `release`, or 0."""

def parse(version: str) -> Version:
    """Parse the given version string (the same as `Version(version)`).

    Raises:
        InvalidVersion: `version` is not a valid PEP 440 version.
    """

def normalize_pre(letter: str, /) -> str:
    """Normalise a pre-release letter (`"alpha"` to `"a"`, `"c"` to `"rc"`; other text lowercased)."""
