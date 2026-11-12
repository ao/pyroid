"""Calls that ``pyroid.nltk`` hands to the real NLTK 3.10.4, with a one-time FallbackWarning."""

from __future__ import annotations

import importlib
import sys
import threading
import warnings
from typing import Any

from pyroid._pyroid import FallbackWarning, PyroidError

_LOCK = threading.Lock()
_WARNED: set[str] = set()
#: NLTK's own objects that pyroid.patch("nltk") replaced, by "<module>:<name>": what the
#: fallbacks call while the patch is active.
ORIGINALS: dict[str, Any] = {}


def stacklevel() -> int:
    """``stacklevel`` for warnings.warn that points at the first frame outside pyroid."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and str(frame.f_globals.get("__name__", "")).startswith(
        "pyroid."
    ):
        frame = frame.f_back
        level += 1
    return level


def module(name: str, what: str, *, warn: bool = True) -> Any:
    """``importlib.import_module(name)`` (an NLTK module), after a one-time warning for ``what``.

    Raises:
        PyroidError: NLTK is not installed.
    """
    real = sys.modules.get(name)
    if real is None or getattr(real, "__pyroid__", False):
        try:
            real = importlib.import_module(name)
        except ImportError as e:
            raise PyroidError(
                f"pyroid.nltk does not port {what}; it delegates that to NLTK, which is not "
                "installed (pip install nltk==3.10.4)"
            ) from e
    real = getattr(real, "__wrapped__", real)
    if real is None:  # pyroid.patch("nltk") shim without NLTK behind it
        raise PyroidError(
            f"pyroid.nltk does not port {what}; it delegates that to NLTK, which is not "
            "installed (pip install nltk==3.10.4)"
        )
    if not warn:
        return real
    with _LOCK:
        first = what not in _WARNED
        _WARNED.add(what)
    if first:
        version = getattr(sys.modules.get("nltk"), "__version__", "?")
        warnings.warn(
            f"pyroid.nltk: {what} is not ported to Rust; calling NLTK {version} (slow path). "
            "This warning is shown once.",
            FallbackWarning,
            stacklevel=stacklevel(),
        )
    return real


def attr(path: str, what: str, *, warn: bool = True) -> Any:
    """``nltk.<module>.<name>`` for ``path`` = ``"<module>:<name>"``, via :func:`module`.

    ``warn=False`` for a lookup that is part of another fallback (which warns once itself).
    """
    mod, _, name = path.partition(":")
    real = module(mod, what, warn=warn)
    original = ORIGINALS.get(path)
    return original if original is not None else getattr(real, name)
