# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
"""rouge-score's tokenizing function (``rouge_score.tokenize``), Rust-backed.

``tokenize(text, stemmer)`` lowercases, replaces every run of characters other
than ``a-z0-9`` with a space and splits, exactly as rouge-score does. The
tokenizing runs in Rust; a ``stemmer`` given here is called from Python (use
``tokenizers.DefaultTokenizer(use_stemmer=True)`` for the Rust port of nltk's
Porter stemmer).
"""

from __future__ import annotations

import re
from typing import Any

from pyroid import _versions
from pyroid._pyroid import metrics as _native

__all__ = [
    "NON_ALPHANUM_PATTERN",
    "NON_ALPHANUM_RE",
    "SPACES_PATTERN",
    "SPACES_RE",
    "VALID_TOKEN_PATTERN",
    "VALID_TOKEN_RE",
    "tokenize",
]

# What the ROUGE modules reproduce: rouge-score, and the NLTK whose Porter stemmer it uses.
_ROUGE_SCORE = "rouge-score==0.1.2"
_NLTK = "nltk==3.10.4"
_versions.check(__name__, "rouge_score", _ROUGE_SCORE)

# Pre-compile regexes that are use often
NON_ALPHANUM_PATTERN = r"[^a-z0-9]+"
NON_ALPHANUM_RE = re.compile(NON_ALPHANUM_PATTERN)
SPACES_PATTERN = r"\s+"
SPACES_RE = re.compile(SPACES_PATTERN)
VALID_TOKEN_PATTERN = r"^[a-z0-9]+$"
VALID_TOKEN_RE = re.compile(VALID_TOKEN_PATTERN)


def _ensure_str(s: Any) -> Any:
    """``six.ensure_str(s)`` (Python 3): str as is, bytes decoded as UTF-8."""
    if type(s) is str:
        return s
    if isinstance(s, bytes):
        return s.decode("utf-8", "strict")
    if not isinstance(s, str):
        raise TypeError(f"not expecting type '{type(s)}'")
    return s


def _tokens(text: Any, use_stemmer: bool) -> list[str]:
    """rouge-score's tokenizer with nltk's Porter stemmer or none, for any input.

    A str is lowercased in Rust. Anything else goes through its own ``lower()``
    first, as in rouge-score (``bytes.lower()`` is ASCII-only), then
    ``six.ensure_str``.
    """
    if type(text) is str:
        return _native.rouge_tokenize(text, use_stemmer)  # type: ignore[no-any-return]
    lowered = _ensure_str(text.lower())
    return _native.rouge_tokenize(lowered, use_stemmer, True)  # type: ignore[no-any-return]


def tokenize(text: Any, stemmer: Any) -> list[str]:
    """Tokenize input text into a list of tokens.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
      text: A text blob to tokenize.
      stemmer: An optional stemmer.

    Returns:
      A list of string tokens extracted from input text.
    """
    tokens = _tokens(text, False)
    if stemmer:
        # Only stem words more than 3 characters long.
        stemmed = [_ensure_str(stemmer.stem(x)) if len(x) > 3 else x for x in tokens]
        # One final check to drop any empty or invalid tokens.
        tokens = [x for x in stemmed if VALID_TOKEN_RE.match(x)]
    return tokens
