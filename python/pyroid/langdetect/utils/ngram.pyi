# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``NGram``: the n-gram buffer and character normalisation."""

from re import Pattern
from typing import Any, ClassVar

__all__ = ["NGram"]

class NGram:
    """The last three normalised characters of a text, as langdetect's ``NGram``.

    The class attributes are langdetect's tables. Changing one (``N_GRAM``,
    ``LATIN1_EXCLUDED``, an entry of ``CJK_MAP``, ...) changes the detections as in
    langdetect, which then run langdetect's Python algorithm (langdetect's speed).
    """

    LATIN1_EXCLUDED: ClassVar[str]
    N_GRAM: ClassVar[int]
    NORMALIZED_VI_CHARS: ClassVar[list[str]]
    TO_NORMALIZE_VI_CHARS: ClassVar[str]
    DMARK_CLASS: ClassVar[str]
    ALPHABET_WITH_DMARK: ClassVar[Pattern[str]]
    CJK_CLASS: ClassVar[list[str]]
    #: CJK ideograph -> the representative of its class (6,667 entries).
    CJK_MAP: ClassVar[dict[str, str]]
    grams: str
    capitalword: bool
    def __init__(self) -> None: ...
    def add_char(self, ch: str) -> None:
        """Append one character."""
    def get(self, n: int) -> str | None:
        """The last `n` characters (`n` = 1, 2 or 3), or ``None``."""
    @classmethod
    def normalize(cls, ch: Any) -> str:
        """One character as langdetect normalises it."""
    @classmethod
    def normalize_vi(cls, text: Any) -> str:
        """Fold a Vietnamese letter and a following combining mark into one character."""
