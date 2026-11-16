# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""`packaging.markers`: environment markers."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from typing import Any, Literal, TypeAlias, TypedDict

__all__ = [
    "Environment",
    "EvaluateContext",
    "InvalidMarker",
    "Marker",
    "UndefinedComparison",
    "UndefinedEnvironmentName",
    "default_environment",
]

EvaluateContext: TypeAlias = Literal["metadata", "lock_file", "requirement"]

class Environment(TypedDict):
    """A marker environment, as `default_environment()` returns it."""

    implementation_name: str
    implementation_version: str
    os_name: str
    platform_machine: str
    platform_release: str
    platform_system: str
    platform_version: str
    python_full_version: str
    platform_python_implementation: str
    python_version: str
    sys_platform: str

class InvalidMarker(ValueError):
    """Raised for an invalid marker string (packaging's own class when installed)."""

class UndefinedComparison(ValueError):
    """Raised for a comparison markers do not define (packaging's own class when installed)."""

class UndefinedEnvironmentName(KeyError):
    """Raised when a marker names a variable the environment lacks (packaging's own class)."""

class Marker:
    """A parsed environment marker, as `packaging.markers.Marker` 26.3."""

    def __init__(self, marker: str) -> None:
        """Parse `marker`.

        Raises:
            InvalidMarker: `marker` is not a valid marker expression.
        """
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __and__(self, other: Marker) -> Marker: ...
    def __or__(self, other: Marker) -> Marker: ...
    def evaluate(
        self,
        environment: Mapping[str, str | AbstractSet[str]] | None = None,
        context: EvaluateContext = "metadata",
    ) -> bool:
        """Evaluate the marker in the current environment, updated with `environment`.

        Raises:
            UndefinedComparison: A comparison markers do not define.
            UndefinedEnvironmentName: The marker names a variable the environment lacks.
        """
    def __getstate__(self) -> str: ...
    def __setstate__(self, state: object) -> None: ...

def default_environment() -> Environment:
    """A copy of the current process's marker environment (computed once per process)."""

def _cached_default_environment() -> Any: ...
