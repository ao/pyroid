"""Find the installed jmespath (not pyroid's shim of it)."""

from __future__ import annotations

from typing import Any

_UNSET: Any = object()
_found: Any = _UNSET


def jmespath() -> Any:
    """The real ``jmespath`` module, or ``None`` when it is not installed."""
    global _found
    if _found is _UNSET:
        try:
            import jmespath as real
        except ImportError:
            real = None
        # Under pyroid.patch("jmespath") without jmespath, `jmespath` is pyroid's shim.
        if real is not None and getattr(real, "__pyroid__", False):
            real = getattr(real, "__wrapped__", None)
        _found = real
    return _found
