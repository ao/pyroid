"""Keyword extraction (TF-IDF and TextRank), delegated to ``jieba.analyse`` (jieba 0.42.1).

pyroid does not port ``jieba.analyse``. Every call here goes to jieba, with a
one-time :class:`pyroid.FallbackWarning` (:class:`pyroid.PyroidError` if jieba
is not installed), after the words added to ``pyroid.jieba.dt`` are replayed
onto ``jieba.dt``.
"""

from __future__ import annotations

from typing import Any

from pyroid.jieba import _delegate

__all__ = ["extract_tags", "set_idf_path", "set_stop_words", "textrank"]


def _analyse() -> Any:
    return _delegate.submodule("analyse")


def extract_tags(*args: Any, **kwargs: Any) -> Any:
    """``jieba.analyse.extract_tags`` (TF-IDF)."""
    return _analyse().extract_tags(*args, **kwargs)


def textrank(*args: Any, **kwargs: Any) -> Any:
    """``jieba.analyse.textrank``."""
    return _analyse().textrank(*args, **kwargs)


def set_idf_path(idf_path: str) -> None:
    """``jieba.analyse.set_idf_path``."""
    _analyse().set_idf_path(idf_path)


def set_stop_words(stop_words_path: str) -> None:
    """``jieba.analyse.set_stop_words``."""
    _analyse().set_stop_words(stop_words_path)


def __getattr__(name: str) -> Any:
    # Private and dunder lookups (hasattr, help(), star imports, stubtest) must not import jieba.
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_analyse(), name)
