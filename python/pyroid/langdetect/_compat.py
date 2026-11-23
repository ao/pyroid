# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""The classes pyroid shares with an installed langdetect, and the installed-version check.

With langdetect installed, :class:`Language`, :class:`LangDetectException` and
:data:`ErrorCode` are langdetect's own classes, so ``except
langdetect.LangDetectException`` and ``isinstance(x, langdetect.language.Language)`` hold
for what pyroid raises and returns. Without it they are copies of langdetect 1.0.9's
(:mod:`pyroid.langdetect._classes`).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Any

from pyroid.langdetect import _classes

#: The langdetect release whose behaviour pyroid.langdetect reproduces.
LANGDETECT_VERSION = "1.0.9"
#: Other langdetect releases that detect exactly as 1.0.9 does (the same detection code and
#: profiles): no version warning for them.
SAME_RELEASES = ("1.0.7", "1.0.8")


def _installed_package_dir() -> str | None:
    """The installed langdetect's package directory (nothing is imported)."""
    current = sys.modules.get("langdetect")
    if current is not None and getattr(current, "__pyroid__", False):
        real = getattr(current, "__wrapped__", None)
        return os.path.dirname(real.__file__) if real is not None and real.__file__ else None
    try:
        spec = importlib.util.find_spec("langdetect")
    except (ImportError, ValueError):
        return None
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    return next(iter(locations), None) if locations else None


def _import_real() -> Any:
    """The installed langdetect's classes (None: not installed, or it does not import)."""
    if _installed_package_dir() is None:
        return None
    current = sys.modules.get("langdetect")
    if current is not None and getattr(current, "__pyroid__", False):
        return None  # pyroid's shims (patch() without langdetect): use pyroid's copies
    try:
        # By name: langdetect ships no types, and pyroid's stubs never depend on it.
        lang_detect_exception = importlib.import_module("langdetect.lang_detect_exception")
        language = importlib.import_module("langdetect.language")
    except Exception:  # a broken install: pyroid works without it
        return None
    return lang_detect_exception.ErrorCode, lang_detect_exception.LangDetectException, language


_real = _import_real()
ErrorCode: Any = _real[0] if _real is not None else _classes.ErrorCode
LangDetectException: Any = _real[1] if _real is not None else _classes.LangDetectException
Language: Any = _real[2].Language if _real is not None else _classes.Language


def _profiles_dir() -> str | None:
    """The profile directory of the installed langdetect, if any."""
    pkg = _installed_package_dir()
    return os.path.join(pkg, "profiles") if pkg is not None else None
