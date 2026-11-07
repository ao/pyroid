# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
# Type stubs for pyroid.metrics.tokenize. The module docstring is kept here on
# purpose: the API docs are built from this stub.
"""rouge-score's tokenizing function (``rouge_score.tokenize``), Rust-backed.

``from pyroid.metrics import tokenize`` replaces ``from rouge_score import
tokenize``: :func:`tokenize` and the module's regex constants.

Differences from rouge-score
----------------------------
- A ``str`` with lone surrogates (U+D800..U+DFFF) raises ``UnicodeEncodeError``.
- A ``stemmer`` passed here is called from Python for every token longer than
  3, as in rouge-score. :class:`pyroid.metrics.tokenizers.DefaultTokenizer`
  with ``use_stemmer=True`` stems in Rust instead.
"""

import re
from typing import Any, Final, Protocol, type_check_only

__all__ = [
    "NON_ALPHANUM_PATTERN",
    "NON_ALPHANUM_RE",
    "SPACES_PATTERN",
    "SPACES_RE",
    "VALID_TOKEN_PATTERN",
    "VALID_TOKEN_RE",
    "tokenize",
]

NON_ALPHANUM_PATTERN: Final[str]
NON_ALPHANUM_RE: Final[re.Pattern[str]]
SPACES_PATTERN: Final[str]
SPACES_RE: Final[re.Pattern[str]]
VALID_TOKEN_PATTERN: Final[str]
VALID_TOKEN_RE: Final[re.Pattern[str]]

@type_check_only
class _Stemmer(Protocol):
    def stem(self, word: str, /) -> str: ...

def tokenize(text: str, stemmer: _Stemmer | None) -> list[str]:
    """Tokenize input text into a list of tokens, as rouge-score does.

    This approach aims to replicate the approach taken by Chin-Yew Lin in
    the original ROUGE implementation.

    Args:
        text: A text blob to tokenize.
        stemmer: An optional stemmer (e.g. ``nltk.stem.porter.PorterStemmer()``),
            applied to tokens longer than 3 characters.

    Returns:
        A list of string tokens extracted from input text.
    """

def _ensure_str(s: Any) -> Any: ...
def _tokens(text: Any, use_stemmer: bool) -> list[str]: ...
