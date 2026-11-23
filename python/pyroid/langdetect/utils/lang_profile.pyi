# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``LangProfile``: one language's n-gram counts."""

from collections import defaultdict
from re import Pattern
from typing import Any, ClassVar

__all__ = ["LangProfile"]

class LangProfile:
    """One language's n-gram frequencies (``name``, ``freq``, ``n_words``)."""

    MINIMUM_FREQ: ClassVar[int]
    LESS_FREQ_RATIO: ClassVar[int]
    ROMAN_CHAR_RE: ClassVar[Pattern[str]]
    ROMAN_SUBSTR_RE: ClassVar[Pattern[str]]
    name: Any
    freq: defaultdict[Any, int]
    n_words: Any
    def __init__(self, name: Any = None, freq: Any = None, n_words: Any = None) -> None: ...
    def add(self, gram: Any) -> None:
        """Count one n-gram."""
    def omit_less_freq(self) -> None:
        """Drop rare n-grams, and Latin ones from a profile with few Latin letters."""
    def update(self, text: Any) -> None:
        """Count the n-grams of `text`."""
