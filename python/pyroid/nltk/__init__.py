# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
"""NLTK 3.10.4's sentence and word tokenizers, in Rust (``import pyroid.nltk as nltk``).

:func:`sent_tokenize` and :class:`PunktSentenceTokenizer` (Punkt, with the installed
``punkt_tab`` models), :func:`word_tokenize`, :class:`TreebankWordTokenizer`,
:class:`NLTKWordTokenizer` and :class:`TreebankWordDetokenizer` give the same results as
NLTK 3.10.4. NLTK itself is optional; the ``punkt_tab`` data is not bundled
(``nltk.download('punkt_tab')``). So do :func:`wordpunct_tokenize`, :class:`RegexpTokenizer`
(for common patterns), :class:`WhitespaceTokenizer` and :class:`BlanklineTokenizer`. Training
(``PunktTrainer``), other patterns and anything else not ported is handed to NLTK with a
one-time :class:`pyroid.FallbackWarning`.

Beyond NLTK, the tokenizers also take a list or tuple of str, or an Arrow string column,
and return one list per document, computed in parallel on pyroid's shared thread pool
(``threads=``).
"""

from __future__ import annotations

import functools
from typing import Any

from pyroid._pyroid import nltk as _native
from pyroid.nltk._punkt import (
    DEBUG_DECISION_FMT,
    REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC,
    REASON_ABBR_WITH_SENTENCE_STARTER,
    REASON_DEFAULT_DECISION,
    REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC,
    REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC,
    REASON_KNOWN_COLLOCATION,
    REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC,
    UNICODE_VERSION,
    PunktLanguageVars,
    PunktParameters,
    PunktSentenceTokenizer,
    PunktTokenizer,
    PunktTrainer,
    format_debug_decision,
    load_punkt_params,
)
from pyroid.nltk._regexp import (
    BlanklineTokenizer,
    RegexpTokenizer,
    WhitespaceTokenizer,
    WordPunctTokenizer,
    blankline_tokenize,
    regexp_tokenize,
    wordpunct_tokenize,
)
from pyroid.nltk._treebank import (
    MacIntyreContractions,
    NLTKWordTokenizer,
    TreebankWordDetokenizer,
    TreebankWordTokenizer,
    align_tokens,
)

__all__ = [
    "NLTK_VERSION",
    "BlanklineTokenizer",
    "MacIntyreContractions",
    "NLTKWordTokenizer",
    "PunktLanguageVars",
    "PunktParameters",
    "PunktSentenceTokenizer",
    "PunktTokenizer",
    "PunktTrainer",
    "RegexpTokenizer",
    "TreebankWordDetokenizer",
    "TreebankWordTokenizer",
    "WhitespaceTokenizer",
    "WordPunctTokenizer",
    "align_tokens",
    "blankline_tokenize",
    "format_debug_decision",
    "load_punkt_params",
    "punkt_model_available",
    "regexp_tokenize",
    "sent_tokenize",
    "word_tokenize",
    "wordpunct_tokenize",
]

#: The NLTK release whose tokenizers this module reproduces.
NLTK_VERSION = "3.10.4"
#: The ``regex`` release whose Unicode tables NLTK 3.10.4's patterns are reproduced with.
REGEX_VERSION = "2026.9.29"

_ = (
    DEBUG_DECISION_FMT,
    REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC,
    REASON_ABBR_WITH_SENTENCE_STARTER,
    REASON_DEFAULT_DECISION,
    REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC,
    REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC,
    REASON_KNOWN_COLLOCATION,
    REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC,
)


@functools.lru_cache
def _get_punkt_tokenizer(language: str = "english") -> PunktTokenizer:
    """The :class:`PunktTokenizer` for `language`, loaded once (NLTK's lru-cached constructor)."""
    return PunktTokenizer(language)


def punkt_model_available(language: str = "english") -> bool:
    """Whether the Punkt model :func:`sent_tokenize` uses for `language` is installed.

    The model is loaded as :func:`sent_tokenize` loads it, and kept for it.

    Raises:
        TypeError: `language` is not a str.
        ValueError: `language` cannot name one model.
    """
    try:
        _get_punkt_tokenizer(language)
    except LookupError:
        return False
    return True


def sent_tokenize(text: Any, language: str = "english", *, threads: int | None = None) -> Any:
    """Split `text` into sentences with the Punkt model for `language`.

    Args:
        text: A str, or a list or tuple of str, or an Arrow string array or column.
        language: The ``punkt_tab`` model name.
        threads: For a batch: ``None`` uses the whole shared pool, ``n`` at most ``n`` tasks,
            ``1`` the calling thread.

    Returns:
        The sentences (a list of str); for a batch, one list per document (``None`` for an
        Arrow null).

    Raises:
        LookupError: The model is not installed (``nltk.download('punkt_tab')``).
        pyroid.InputError: `text` is not text.
    """
    return _get_punkt_tokenizer(language).tokenize(text, threads=threads)


_treebank_word_tokenizer = NLTKWordTokenizer()


def word_tokenize(
    text: Any,
    language: str = "english",
    preserve_line: bool = False,
    *,
    threads: int | None = None,
) -> Any:
    """Split `text` into words: Punkt sentences, then :class:`NLTKWordTokenizer` on each.

    Args:
        text: A str, or a batch as for :func:`sent_tokenize`.
        language: The ``punkt_tab`` model name.
        preserve_line: Do not split sentences first (no model is needed then).
        threads: For a batch, as for :func:`sent_tokenize`.

    Returns:
        The tokens (a list of str); for a batch, one list per document.

    Raises:
        LookupError: The model is not installed (``nltk.download('punkt_tab')``).
        pyroid.InputError: `text` is not text.
    """
    if preserve_line:
        if _treebank_word_tokenizer._twin() is not None:
            return [*_treebank_word_tokenizer.tokenize(text)]
        return _native.treebank(text, True, False, UNICODE_VERSION, threads=threads)
    punkt = _get_punkt_tokenizer(language)
    native = punkt._impl()
    if native is None or _treebank_word_tokenizer._twin() is not None:
        sentences = sent_tokenize(text, language)
        return [tok for sent in sentences for tok in _treebank_word_tokenizer.tokenize(sent)]
    return native.word_tokenize(text, threads=threads)
