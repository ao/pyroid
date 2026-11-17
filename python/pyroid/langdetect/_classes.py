# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""Copies of langdetect 1.0.9's ``Language``, ``LangDetectException`` and ``ErrorCode``.

pyroid uses them when langdetect is not installed (see :mod:`pyroid.langdetect._compat`). They
carry langdetect's names, so messages read ``Language.__init__() missing ...`` as langdetect's
do, and pyroid's module names, so they pickle as ``pyroid.langdetect.language.Language`` and
``pyroid.langdetect.lang_detect_exception.LangDetectException``.
"""

from __future__ import annotations

from typing import Any

ErrorCode: Any = type(
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


class LangDetectException(Exception):  # noqa: N818 (langdetect's name)
    """langdetect's exception; ``code`` is one of :data:`ErrorCode`'s values."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code

    def get_code(self) -> int:
        """The error code."""
        return self.code


class Language:
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


LangDetectException.__module__ = "pyroid.langdetect.lang_detect_exception"
Language.__module__ = "pyroid.langdetect.language"
