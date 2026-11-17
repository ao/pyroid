# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``NGram``: the n-gram buffer and character normalisation."""

from re import Pattern
from typing import Any, ClassVar

__all__ = ["NGram"]

class NGram:
    """The last three normalised characters of a text, as langdetect's ``NGram``."""

    LATIN1_EXCLUDED: ClassVar[str]
    N_GRAM: ClassVar[int]
    TO_NORMALIZE_VI_CHARS: ClassVar[str]
    DMARK_CLASS: ClassVar[str]
    ALPHABET_WITH_DMARK: ClassVar[Pattern[str]]
    grams: str
    capitalword: bool
    def __init__(self) -> None: ...
    def add_char(self, ch: str) -> None:
        """Append one character."""
    def get(self, n: int) -> str | None:
        """The last `n` characters (`n` = 1, 2 or 3), or ``None``."""
    @classmethod
    def normalize(cls, ch: Any) -> str:
        """langdetect's normalisation of one character."""
    @classmethod
    def normalize_vi(cls, text: Any) -> str:
        """Fold a Vietnamese letter and a following combining mark into one character."""
