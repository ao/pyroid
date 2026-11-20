# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Error messages keep datasketch's %-formatting, so they stay byte-identical and the code
# stays easy to diff against upstream.
# ruff: noqa: UP031
"""Helpers shared by the pyroid.datasketch modules (not public API)."""

from __future__ import annotations

import importlib
import sys
import warnings
from types import ModuleType
from typing import Any

from pyroid import _versions
from pyroid._pyroid import FallbackWarning, PyroidError


# Every pyroid.datasketch module imports this one first (through the package), so a missing numpy
# fails here with an install hint, as pyroid.fasttext does.
def _require_numpy() -> None:
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyroid.datasketch needs numpy, as datasketch does: "
            "pip install 'pyroid[datasketch]' (or numpy)",
            name="numpy",
        ) from exc


_require_numpy()

__all__ = [
    "DATASKETCH_VERSION",
    "_SCHEME_AFFINE32",
    "_SCHEME_AFFINE64",
    "_SCHEME_LEGACY",
    "_VALID_SCHEMES",
    "_check_scheme_consistency",
    "fallback",
    "warn_once",
]

#: The datasketch release pyroid.datasketch reproduces.
DATASKETCH_VERSION = "2.0.0"

# Every pyroid.datasketch module imports this one first, through the package.
_versions.check("pyroid.datasketch", "datasketch", f"datasketch=={DATASKETCH_VERSION}")

_SCHEME_LEGACY = "legacy"
_SCHEME_AFFINE32 = "affine32"
_SCHEME_AFFINE64 = "affine64"
_VALID_SCHEMES = (_SCHEME_AFFINE32, _SCHEME_AFFINE64, _SCHEME_LEGACY)


def _check_scheme_consistency(known: str | None, minhash: Any) -> str | None:
    """Check the scheme of a MinHash given to an index against the one it has seen.

    As datasketch's ``datasketch.minhash._check_scheme_consistency``: sketch
    types without a ``scheme`` attribute (WeightedMinHash) are exempt.
    """
    scheme = getattr(minhash, "scheme", None)
    if scheme is None:
        return known
    if known is None:
        return scheme  # type: ignore[no-any-return]
    if scheme != known:
        raise ValueError(
            "MinHash scheme %r does not match scheme %r of the MinHash previously given to "
            "this index" % (scheme, known)
        )
    return known


# Keys already warned about. Two threads racing on a key can both warn; that is harmless.
_WARNED: set[str] = set()


def _caller_stacklevel() -> int:
    """The stacklevel of the first frame outside pyroid.datasketch."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith("pyroid.datasketch"):
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    return level


def warn_once(key: str, message: str) -> None:
    """Emit ``message`` as a :class:`pyroid.FallbackWarning`, once per process per ``key``."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, FallbackWarning, stacklevel=_caller_stacklevel())


def fallback(name: str, reason: str) -> ModuleType:
    """Import datasketch to delegate ``name`` to it, warning once; PyroidError if missing."""
    try:
        real: ModuleType | None = importlib.import_module("datasketch")
    except ImportError:
        real = None
    # Under a pyroid patch shim, use the real module it wraps, never the shim.
    if real is not None and getattr(real, "__pyroid__", False):
        real = real.__wrapped__
    if real is None:
        raise PyroidError(
            f"pyroid.datasketch.{name}: {reason} is not supported by pyroid's Rust port, and "
            f"datasketch is not installed to fall back to "
            f"(pip install datasketch=={DATASKETCH_VERSION})"
        )
    warn_once(
        f"{name}:{reason}",
        f"pyroid.datasketch.{name}: {reason} is not supported by the Rust port; "
        "delegating to datasketch (slow path). This warning is shown once.",
    )
    return real
