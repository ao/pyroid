# Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
# contributors; modified.
"""``packaging.markers``: environment markers (:class:`Marker`, :func:`default_environment`)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict, cast

from pyroid._pyroid import packaging as _native
from pyroid.packaging import _compat
from pyroid.packaging._compat import (
    InvalidMarker,
    UndefinedComparison,
    UndefinedEnvironmentName,
    _cached_default_environment,
)

__all__ = [
    "Environment",
    "EvaluateContext",
    "InvalidMarker",
    "Marker",
    "UndefinedComparison",
    "UndefinedEnvironmentName",
    "default_environment",
]

Marker = _native.Marker

if _compat._pm is not None and hasattr(_compat._pm, "Environment"):
    Environment: Any = _compat._pm.Environment
    EvaluateContext: Any = getattr(
        _compat._pm, "EvaluateContext", Literal["metadata", "lock_file", "requirement"]
    )
else:

    class Environment(TypedDict):  # type: ignore[no-redef]
        """A marker environment, as :func:`default_environment` returns it."""

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

    EvaluateContext = Literal["metadata", "lock_file", "requirement"]


def default_environment() -> Environment:
    """Return the default marker environment for the current Python process.

    A fresh copy of the environment computed once per process (as packaging 26.3 caches it).
    """
    return cast("Environment", dict(_cached_default_environment()))


def __dir__() -> list[str]:
    return __all__
