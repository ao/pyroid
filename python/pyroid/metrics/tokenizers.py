# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
"""rouge-score's tokenizer classes (``rouge_score.tokenizers``), Rust-backed.

The RougeScorer class can be instantiated with the tokenizers defined here. New
tokenizers can be defined by creating a subclass of the Tokenizer abstract class
and overriding the tokenize() method.
"""

from __future__ import annotations

import abc
from typing import Any

from pyroid import _versions
from pyroid.metrics.tokenize import _NLTK, _ROUGE_SCORE, _tokens

__all__ = ["DefaultTokenizer", "Tokenizer"]

_versions.check(__name__, "rouge_score", _ROUGE_SCORE)
_versions.check(__name__, "nltk", _NLTK)  # DefaultTokenizer(use_stemmer=True)


class Tokenizer(abc.ABC):
    """Abstract base class for a tokenizer.

    Subclasses of Tokenizer must implement the tokenize() method.
    """

    @abc.abstractmethod
    def tokenize(self, text: Any) -> list[str]:
        """Split ``text`` into tokens."""
        raise NotImplementedError("Tokenizer must override tokenize() method")


class DefaultTokenizer(Tokenizer):
    """Default tokenizer which tokenizes on whitespace."""

    def __init__(self, use_stemmer: bool = False) -> None:
        """Constructor for DefaultTokenizer.

        Args:
          use_stemmer: boolean, indicating whether Porter stemmer should be used to
          strip word suffixes to improve matching.
        """
        # rouge-score keeps nltk's PorterStemmer() here, or None; pyroid stems in Rust.
        self._use_stemmer = bool(use_stemmer)

    def tokenize(self, text: Any) -> list[str]:
        """rouge-score's tokens of ``text`` (stemmed if ``use_stemmer``)."""
        return _tokens(text, self._use_stemmer)
