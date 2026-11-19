# Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
"""langdetect's ``LangProfile``: one language's n-gram counts."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from pyroid.langdetect.utils.ngram import NGram

__all__ = ["LangProfile"]


class LangProfile:
    """One language's n-gram frequencies (``name``, ``freq``, ``n_words``)."""

    MINIMUM_FREQ = 2
    LESS_FREQ_RATIO = 100000

    ROMAN_CHAR_RE = re.compile(r"^[A-Za-z]$")
    ROMAN_SUBSTR_RE = re.compile(r".*[A-Za-z].*")

    def __init__(self, name: Any = None, freq: Any = None, n_words: Any = None) -> None:
        self.freq: defaultdict[Any, int] = defaultdict(int)
        if freq is not None:
            self.freq.update(freq)

        if n_words is None:
            n_words = [0] * NGram.N_GRAM

        self.name = name
        self.n_words = n_words

    def add(self, gram: Any) -> None:
        """Add one n-gram."""
        if self.name is None or gram is None:  # Illegal
            return
        length = len(gram)
        if length < 1 or length > NGram.N_GRAM:  # Illegal
            return
        self.n_words[length - 1] += 1
        self.freq[gram] += 1

    def omit_less_freq(self) -> None:
        """Remove rare n-grams, and Latin ones from a profile with few Latin letters."""
        if self.name is None:  # Illegal
            return
        threshold = max(self.n_words[0] // self.LESS_FREQ_RATIO, self.MINIMUM_FREQ)

        roman = 0
        for key, count in list(self.freq.items()):
            if count <= threshold:
                self.n_words[len(key) - 1] -= count
                del self.freq[key]
            elif self.ROMAN_CHAR_RE.match(key):
                roman += count

        # roman check
        if roman < self.n_words[0] // 3:
            for key, count in list(self.freq.items()):
                if self.ROMAN_SUBSTR_RE.match(key):
                    self.n_words[len(key) - 1] -= count
                    del self.freq[key]

    def update(self, text: Any) -> None:
        """Count the n-grams of `text`."""
        if text is None:
            return
        text = NGram.normalize_vi(text)
        gram = NGram()
        for ch in text:
            gram.add_char(ch)
            for n in range(1, NGram.N_GRAM + 1):
                self.add(gram.get(n))
