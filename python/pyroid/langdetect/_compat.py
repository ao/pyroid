# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""The classes pyroid shares with an installed langdetect, and the installed-version check.

With langdetect installed, :class:`Language`, :class:`LangDetectException` and
:data:`ErrorCode` are langdetect's own classes, so ``except
langdetect.LangDetectException`` and ``isinstance(x, langdetect.language.Language)`` hold
for what pyroid raises and returns. Without it they are copies of langdetect 1.0.9's.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from typing import Any

#: The langdetect release whose behaviour pyroid.langdetect reproduces.
LANGDETECT_VERSION = "1.0.9"


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


# Copies of langdetect 1.0.9's classes, used when langdetect is not installed.
_ErrorCode: Any = type(
    "ErrorCode",
    (),
    {
        "NoTextError": 0,
        "FormatError": 1,
        "FileLoadError": 2,
        "DuplicateLangError": 3,
        "NeedLoadProfileError": 4,
        "CantDetectError": 5,
        "CantOpenTrainData": 6,
        "TrainDataFormatError": 7,
        "InitParamError": 8,
        "__module__": "pyroid.langdetect.lang_detect_exception",
    },
)


class _LangDetectException(Exception):  # noqa: N818 (langdetect's name)
    """langdetect's exception; ``code`` is one of :data:`ErrorCode`'s values."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code

    def get_code(self) -> int:
        """The error code."""
        return self.code


class _Language:
    """A detected language: ``lang`` and ``prob``, ordered by ``prob``."""

    def __init__(self, lang: Any, prob: Any) -> None:
        self.lang = lang
        self.prob = prob

    def __repr__(self) -> str:
        if self.lang is None:
            return ""
        return "%s:%s" % (self.lang, self.prob)  # noqa: UP031 (langdetect's formatting)

    def __lt__(self, other: Any) -> Any:
        return self.prob < other.prob


for _cls, _name, _module in (
    (_LangDetectException, "LangDetectException", "pyroid.langdetect.lang_detect_exception"),
    (_Language, "Language", "pyroid.langdetect.language"),
):
    _cls.__name__ = _cls.__qualname__ = _name
    _cls.__module__ = _module
del _cls, _name, _module

_real = _import_real()
ErrorCode: Any = _real[0] if _real is not None else _ErrorCode
LangDetectException: Any = _real[1] if _real is not None else _LangDetectException
Language: Any = _real[2].Language if _real is not None else _Language


def _profiles_dir() -> str | None:
    """The profile directory of the installed langdetect, if any."""
    pkg = _installed_package_dir()
    return os.path.join(pkg, "profiles") if pkg is not None else None


_WARNED = False


def check_version() -> None:
    """Warn once per process if the installed langdetect is not 1.0.9.

    The shape follows json_repair's version warning until ``pyroid._versions.check`` lands.
    """
    global _WARNED
    if _WARNED or _real is None:
        return
    _WARNED = True
    pkg = _installed_package_dir()
    if pkg is None:
        return
    parent = os.path.dirname(pkg)
    if os.path.isdir(os.path.join(parent, f"langdetect-{LANGDETECT_VERSION}.dist-info")):
        return
    from importlib import metadata

    try:
        version = metadata.version("langdetect")
    except metadata.PackageNotFoundError:
        return
    if version == LANGDETECT_VERSION:
        return
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and str(frame.f_globals.get("__name__", "")).startswith(
        ("pyroid", "importlib")
    ):
        frame = frame.f_back
        level += 1
    warnings.warn(
        f"pyroid.langdetect reproduces langdetect {LANGDETECT_VERSION}, but langdetect {version} "
        f"is installed: detections follow {LANGDETECT_VERSION}, and Language, "
        f"LangDetectException and ErrorCode are the installed release's classes.",
        RuntimeWarning,
        stacklevel=level,
    )
