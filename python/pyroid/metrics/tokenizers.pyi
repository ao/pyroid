# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
# Type stubs for pyroid.metrics.tokenizers. The module docstring is kept here on
# purpose: the API docs are built from this stub.
"""rouge-score's tokenizer classes (``rouge_score.tokenizers``), Rust-backed.

``from pyroid.metrics import tokenizers`` replaces ``from rouge_score import
tokenizers``. :class:`DefaultTokenizer` tokenizes exactly as rouge-score 0.1.2's:
lowercase, keep runs of ``a-z0-9``, and with ``use_stemmer=True`` stem tokens
longer than 3 characters with nltk 3.10.4's Porter stemmer (ported to Rust;
nltk is not needed).

Pass a :class:`Tokenizer` subclass (or any object with a ``tokenize`` method)
as ``RougeScorer(tokenizer=...)`` to tokenize differently; it is called from
Python where rouge-score calls it.

Differences from rouge-score
----------------------------
- ``DefaultTokenizer`` has no ``_stemmer`` attribute (rouge-score keeps an
  nltk ``PorterStemmer`` there); stemming runs in Rust.
- A ``str`` with lone surrogates (U+D800..U+DFFF) raises ``UnicodeEncodeError``
  (rouge-score drops them like any other non-alphanumeric character).
"""

import abc
from typing import Any

__all__ = ["DefaultTokenizer", "Tokenizer"]

class Tokenizer(abc.ABC):
    """Abstract base class for a tokenizer.

    Subclasses of Tokenizer must implement the tokenize() method.
    """

    @abc.abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Split ``text`` into tokens."""

class DefaultTokenizer(Tokenizer):
    """Default tokenizer which tokenizes on whitespace (rouge-score's).

    Example:
        ```python
        from pyroid.metrics import tokenizers

        tokenizers.DefaultTokenizer(use_stemmer=True).tokenize("the friends had a meeting")
        # ['the', 'friend', 'had', 'a', 'meet']
        ```
    """

    def __init__(self, use_stemmer: bool = False) -> None:
        """Constructor for DefaultTokenizer.

        Args:
            use_stemmer: boolean, indicating whether Porter stemmer should be used to
                strip word suffixes to improve matching.
        """

    def tokenize(self, text: str) -> list[str]:
        """rouge-score's tokens of ``text``.

        ``text.lower()``, then every run of characters other than ``a-z0-9``
        separates tokens (Unicode lowercasing included: ``"İ"`` gives ``"i"``,
        the Kelvin sign gives ``"k"``). With ``use_stemmer``, tokens longer
        than 3 are Porter-stemmed. ``bytes`` and other objects with a
        ``lower()`` method are handled as rouge-score handles them.

        Args:
            text: The text to tokenize.

        Returns:
            The tokens, in order.

        Raises:
            UnicodeEncodeError: ``text`` holds a lone surrogate.
        """

    _use_stemmer: Any
