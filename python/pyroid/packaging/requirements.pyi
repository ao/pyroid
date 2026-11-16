# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""`packaging.requirements`: PEP 508 dependency specifiers."""

from pyroid.packaging.markers import Marker
from pyroid.packaging.specifiers import SpecifierSet

__all__ = ["InvalidRequirement", "Requirement"]

class InvalidRequirement(ValueError):
    """Raised for an invalid requirement string (packaging's own class when installed)."""

class Requirement:
    """A parsed requirement (`name[extras] specifier ; marker`), as packaging 26.3's.

    The attributes are writable, as packaging's are.
    """

    name: str
    url: str | None
    extras: set[str]
    specifier: SpecifierSet
    marker: Marker | None
    def __init__(self, requirement_string: str) -> None:
        """Parse `requirement_string`.

        Raises:
            InvalidRequirement: The string is not a valid PEP 508 requirement.
        """
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __getstate__(self) -> tuple[str, bool | None]: ...
    def __setstate__(self, state: object) -> None: ...
