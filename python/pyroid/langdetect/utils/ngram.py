# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``NGram``: the n-gram buffer and character normalisation (in Rust)."""

from __future__ import annotations

import re
from typing import Any

from pyroid._pyroid import langdetect as _native

__all__ = ["NGram"]


class NGram:
    """The last three normalised characters of a text, as langdetect's ``NGram``."""

    LATIN1_EXCLUDED = "\u00a0\u00ab\u00b0\u00bb"
    N_GRAM = 3

    TO_NORMALIZE_VI_CHARS = (
        "AEIOUYaeiouy\u00c2\u00ca\u00d4\u00e2\u00ea\u00f4\u0102\u0103\u01a0\u01a1\u01af\u01b0"
    )
    DMARK_CLASS = "\u0300\u0301\u0303\u0309\u0323"
    ALPHABET_WITH_DMARK = re.compile(
        "([" + TO_NORMALIZE_VI_CHARS + "])([" + DMARK_CLASS + "])", re.UNICODE
    )

    def __init__(self) -> None:
        self.grams = " "
        self.capitalword = False

    def add_char(self, ch: str) -> None:
        """Append a character into the n-gram buffer."""
        ch = self.normalize(ch)
        last_char = self.grams[-1]
        if last_char == " ":
            self.grams = " "
            self.capitalword = False
            if ch == " ":
                return
        elif len(self.grams) >= self.N_GRAM:
            self.grams = self.grams[1:]
        self.grams += ch

        if ch.isupper():
            if last_char.isupper():
                self.capitalword = True
        else:
            self.capitalword = False

    def get(self, n: int) -> str | None:
        """The last `n` characters (`n` = 1, 2 or 3), or ``None``."""
        if self.capitalword:
            return None
        if n < 1 or n > self.N_GRAM or len(self.grams) < n:
            return None
        if n == 1:
            ch = self.grams[-1]
            if ch == " ":
                return None
            return ch
        return self.grams[-n:]

    @classmethod
    def normalize(cls, ch: Any) -> str:
        """langdetect's normalisation of one character."""
        if not isinstance(ch, str):
            ord(ch)  # raises what langdetect raises
        return _native.normalize(ch)  # type: ignore[no-any-return]

    @classmethod
    def normalize_vi(cls, text: Any) -> str:
        """Fold a Vietnamese letter followed by a combining mark into one character."""
        if not isinstance(text, str):
            return cls.ALPHABET_WITH_DMARK.sub("", text)  # raises what langdetect raises
        return _native.normalize_vi(text)  # type: ignore[no-any-return]
