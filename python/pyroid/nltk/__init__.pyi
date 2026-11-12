# Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
r"""NLTK 3.10.4's sentence and word tokenizers, in Rust.

```python
import pyroid.nltk as nltk

nltk.sent_tokenize("Mr. Smith went to Washington. He liked it.")
# ['Mr. Smith went to Washington.', 'He liked it.']
nltk.word_tokenize("They'll pay $3.88 (roughly).")
# ['They', "'ll", 'pay', '$', '3.88', '(', 'roughly', ')', '.']
nltk.sent_tokenize(docs, language="german", threads=8)  # one list per document
nltk.TreebankWordDetokenizer().detokenize(["I", "ca", "n't", "go", "."])  # "I can't go."
```

:func:`sent_tokenize`, :func:`word_tokenize`, :class:`PunktSentenceTokenizer`,
:class:`PunktTokenizer`, :class:`TreebankWordTokenizer`, :class:`NLTKWordTokenizer`,
:class:`TreebankWordDetokenizer`, :func:`align_tokens`, :func:`wordpunct_tokenize`,
:class:`RegexpTokenizer` (for common patterns), :class:`WhitespaceTokenizer` and
:class:`BlanklineTokenizer` give NLTK 3.10.4's results, str for str and offset for offset.
NLTK runs its patterns with the third-party ``regex`` module (``nltk.redos``) and, around them,
Python's ``str`` methods; both are reproduced: ``regex`` 2026.9.29's character classes, and
``str.lower()`` / ``isupper()`` / ``islower()`` of the running CPython (3.11 to 3.15).

The Punkt models are NLTK's ``punkt_tab`` data, which pyroid does not bundle (it has no
licence): install it with ``nltk.download('punkt_tab')`` or
``python -m nltk.downloader punkt_tab``. With NLTK installed, the model is found and read
by NLTK's own ``nltk.data.find`` and ``load_punkt_params`` (so ``nltk.data.path`` and its
checks apply); without NLTK, pyroid looks in NLTK's default locations (``NLTK_DATA``,
``~/nltk_data``, ``sys.prefix``'s and the system's ``nltk_data`` directories) for
``tokenizers/punkt_tab/<language>/`` or ``tokenizers/punkt_tab.zip``. NLTK itself is not
needed for anything else.

Handed to NLTK, with one :class:`pyroid.FallbackWarning` per feature
(:class:`pyroid.PyroidError` if NLTK is not installed): training (:func:`PunktTrainer`,
``PunktSentenceTokenizer(train_text)``), a custom ``token_cls``, a
:class:`PunktLanguageVars` subclass that changes more than ``sent_end_chars``, parameters
that are not sets of str, :meth:`PunktSentenceTokenizer.dump`, saving parameters, a
Treebank tokenizer whose pattern lists were changed or overridden, and a
:class:`RegexpTokenizer` pattern that is not one of the native ones.

pyroid extensions: the tokenizers also take a list or tuple of str, or an Arrow string
column, and return one list per document (``None`` for an Arrow null), computed in parallel
on pyroid's shared pool (``threads=``: ``None`` for the whole pool, ``n`` for at most ``n``
tasks, ``1`` for the calling thread).

Differences from NLTK:

- Lone surrogates (``"\ud800"``) raise ``UnicodeEncodeError``, as everywhere in pyroid.
- No pattern runs under NLTK's 5-second ``regex`` timeout, so nothing raises
  ``TimeoutError`` on very long inputs.
- Results follow ``regex`` 2026.9.29's Unicode tables; with another ``regex`` release
  installed, NLTK can differ on characters whose properties changed between the two.
- Looking up an unseen word in ``PunktParameters.ortho_context`` does not add a 0 entry to
  it (NLTK's ``defaultdict`` grows with every new word it tokenizes).
- An NLTK ``PunktParameters`` given to a tokenizer is copied (into a pyroid
  :class:`PunktParameters`, which ``tok._params`` returns); edit it through ``tok._params``,
  since later edits of the NLTK object are not seen.
- NLTK compiles a ``PunktLanguageVars`` object's patterns at first use, so it keeps using the
  old ``sent_end_chars`` in part of the algorithm after they are changed; pyroid uses the new
  ones throughout.
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Final, TypeAlias, overload

from pyroid._typing import ArrowStrings

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

NLTK_VERSION: Final[str]
"""The NLTK release whose tokenizers this module reproduces (``"3.10.4"``)."""
REGEX_VERSION: Final[str]
"""The ``regex`` release whose Unicode tables the patterns follow (``"2026.9.29"``)."""
UNICODE_VERSION: Final[str]
"""This CPython's ``unicodedata.unidata_version``, which ``str`` case tests follow."""

_Batch: TypeAlias = list[str] | tuple[str, ...]

DEBUG_DECISION_FMT: Final[str]
REASON_DEFAULT_DECISION: Final[str]
REASON_KNOWN_COLLOCATION: Final[str]
REASON_ABBR_WITH_ORTHOGRAPHIC_HEURISTIC: Final[str]
REASON_ABBR_WITH_SENTENCE_STARTER: Final[str]
REASON_INITIAL_WITH_ORTHOGRAPHIC_HEURISTIC: Final[str]
REASON_NUMBER_WITH_ORTHOGRAPHIC_HEURISTIC: Final[str]
REASON_INITIAL_WITH_SPECIAL_ORTHOGRAPHIC_HEURISTIC: Final[str]

@overload
def sent_tokenize(text: str, language: str = "english", *, threads: int | None = None) -> list[str]:
    """Split `text` into sentences with the Punkt model for `language` (NLTK's ``sent_tokenize``).

    The tokenizer for each language is loaded once and kept, as in NLTK.

    Args:
        text: A str, or a list or tuple of str, or an Arrow string array or column.
        language: The ``punkt_tab`` model name (``"english"``, ``"german"``, ...).
        threads: For a batch: ``None`` uses the whole shared pool, ``n`` at most ``n`` tasks,
            ``1`` the calling thread.

    Returns:
        The sentences; for a batch, one list per document (``None`` for an Arrow null).

    Raises:
        LookupError: The model is not installed (``nltk.download('punkt_tab')``).
        TypeError: `language` is not a str.
        ValueError: `language` cannot name one model.
        pyroid.InputError: `text` is not text.
    """

@overload
def sent_tokenize(
    text: _Batch, language: str = "english", *, threads: int | None = None
) -> list[list[str]]: ...
@overload
def sent_tokenize(
    text: ArrowStrings, language: str = "english", *, threads: int | None = None
) -> list[list[str] | None]: ...
@overload
def word_tokenize(
    text: str, language: str = "english", preserve_line: bool = False, *, threads: int | None = None
) -> list[str]:
    """Split `text` into words (NLTK's ``word_tokenize``).

    The sentences of :func:`sent_tokenize`, each split by :class:`NLTKWordTokenizer`.

    Args:
        text: A str, or a batch as for :func:`sent_tokenize`.
        language: The ``punkt_tab`` model name.
        preserve_line: Do not split sentences first (then no model is needed).
        threads: For a batch, as for :func:`sent_tokenize`.

    Returns:
        The tokens; for a batch, one list per document.

    Raises:
        LookupError: The model is not installed (``nltk.download('punkt_tab')``).
        pyroid.InputError: `text` is not text.
    """

@overload
def word_tokenize(
    text: _Batch,
    language: str = "english",
    preserve_line: bool = False,
    *,
    threads: int | None = None,
) -> list[list[str]]: ...
@overload
def word_tokenize(
    text: ArrowStrings,
    language: str = "english",
    preserve_line: bool = False,
    *,
    threads: int | None = None,
) -> list[list[str] | None]: ...
def punkt_model_available(language: str = "english") -> bool:
    """Whether the Punkt model :func:`sent_tokenize` uses for `language` is installed.

    The model is loaded as :func:`sent_tokenize` loads it, and kept for it.

    Raises:
        TypeError: `language` is not a str.
        ValueError: `language` cannot name one model.
    """

class PunktParameters:
    """Stores data used to perform sentence boundary detection with Punkt.

    The attributes are NLTK's. Change them in place (``params.abbrev_types.add("dr")``) or
    assign new ones; tokenizers using these parameters see the change at their next call.
    """

    abbrev_types: set[str]
    """Lowercase abbreviations, without their final period (``"dr"``)."""
    collocations: set[tuple[str, str]]
    """Word pairs whose first word ends in a period that is not a sentence break."""
    sent_starters: set[str]
    """Words that often start a sentence."""
    ortho_context: defaultdict[str, int]
    """Orthographic contexts (bit flags) each word type was seen in."""

    def __init__(self) -> None: ...
    def clear_abbrevs(self) -> None:
        """Replace ``abbrev_types`` with an empty set."""

    def clear_collocations(self) -> None:
        """Replace ``collocations`` with an empty set."""

    def clear_sent_starters(self) -> None:
        """Replace ``sent_starters`` with an empty set."""

    def clear_ortho_context(self) -> None:
        """Replace ``ortho_context`` with an empty ``defaultdict(int)``."""

    def add_ortho_context(self, typ: str, flag: int) -> None:
        """Add `flag` to the orthographic contexts of `typ`."""

class PunktLanguageVars:
    """NLTK's language-dependent variables of the Punkt algorithm.

    Subclass it and change ``sent_end_chars`` (a tuple of one-character strings) to break
    sentences at other characters; that still runs in Rust. Overriding anything else (the
    patterns, :meth:`word_tokenize`) hands the tokenizer to NLTK, with a
    :class:`pyroid.FallbackWarning`.
    """

    sent_end_chars: tuple[str, ...]
    """Characters which are candidates for sentence boundaries (``(".", "?", "!")``)."""
    internal_punctuation: str
    """Sentence-internal punctuation (``",:;"``; used in training)."""

    def word_tokenize(self, s: str) -> list[str]:
        """Punkt's word tokenization of `s` (runs in NLTK)."""

    def period_context_re(self) -> Any:
        """NLTK's compiled period-context pattern (from NLTK)."""

class PunktSentenceTokenizer:
    """NLTK's ``PunktSentenceTokenizer``, running in Rust.

    Args:
        train_text: Training text (trained with NLTK's ``PunktTrainer``: a
            :class:`pyroid.FallbackWarning`), or a :class:`PunktParameters` (pyroid's or
            NLTK's) to use. ``None`` or empty: empty parameters.
        verbose: Passed to the trainer.
        lang_vars: A :class:`PunktLanguageVars` (pyroid's or NLTK's).
        token_cls: NLTK's ``PunktToken`` or a subclass; anything but the default runs in
            NLTK.
    """

    PUNCTUATION: tuple[str, ...]
    _params: Any
    _lang_vars: Any

    def __init__(
        self,
        train_text: Any = None,
        verbose: bool = False,
        lang_vars: Any = None,
        token_cls: Any = ...,
    ) -> None: ...
    def train(self, train_text: Any, verbose: bool = False) -> Any:
        """Return parameters trained on `train_text` (with NLTK), or `train_text` if not a str."""

    @overload
    def tokenize(
        self, text: str, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[str]:
        """Split `text` into sentences.

        Args:
            text: A str, or a batch as for :func:`sent_tokenize`.
            realign_boundaries: Move closing quotes and brackets that follow a sentence break
                into the sentence they close.
            threads: For a batch, as for :func:`sent_tokenize`.
        """

    @overload
    def tokenize(
        self, text: _Batch, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[list[str]]: ...
    @overload
    def tokenize(
        self, text: ArrowStrings, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[list[str] | None]: ...
    @overload
    def sentences_from_text(
        self, text: str, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[str]:
        """The same as :meth:`tokenize`."""

    @overload
    def sentences_from_text(
        self, text: _Batch, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[list[str]]: ...
    @overload
    def sentences_from_text(
        self, text: ArrowStrings, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[list[str] | None]: ...
    @overload
    def span_tokenize(
        self, text: str, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> Iterator[tuple[int, int]]:
        """The ``(start, end)`` offsets of the sentences in `text`.

        A generator for a str, as NLTK's; for a batch, a list with one list of tuples per
        document. With ``realign_boundaries=False`` a span can be empty, as in NLTK.
        """

    @overload
    def span_tokenize(
        self, text: _Batch, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[list[tuple[int, int]]]: ...
    @overload
    def span_tokenize(
        self, text: ArrowStrings, realign_boundaries: bool = True, *, threads: int | None = None
    ) -> list[list[tuple[int, int]] | None]: ...
    def tokenize_sents(
        self, strings: Sequence[str], *, threads: int | None = None
    ) -> list[list[str]]:
        """:meth:`tokenize` of each string (in parallel)."""

    def span_tokenize_sents(
        self, strings: Sequence[str], *, threads: int | None = None
    ) -> Iterator[list[tuple[int, int]]]:
        """:meth:`span_tokenize` of each string, as lists."""

    def text_contains_sentbreak(self, text: str) -> bool:
        """Whether `text` contains a sentence break that is not at its last token."""

    def sentences_from_text_legacy(self, text: str) -> Iterator[str]:
        """The sentences of `text`, found by annotating every token (NLTK's older method)."""

    def sentences_from_tokens(self, tokens: Iterable[str]) -> Iterator[list[str]]:
        """Group word tokens into sentences (lists of the same tokens)."""

    def debug_decisions(self, text: str) -> Iterator[dict[str, Any]]:
        """One dict per candidate sentence break, with the evidence for the decision.

        The keys and values are NLTK's (``period_index`` is a code point offset); format one
        with :func:`format_debug_decision`.
        """

    def dump(self, tokens: Any) -> str:
        """NLTK's debugging dump of annotated tokens (runs in NLTK)."""

class PunktTokenizer(PunktSentenceTokenizer):
    """A :class:`PunktSentenceTokenizer` with the parameters of an installed ``punkt_tab`` model.

    Args:
        lang: The model name (``"english"``, ``"german"``, ...).

    Raises:
        TypeError: `lang` is not a str.
        ValueError: `lang` cannot name one model.
        LookupError: The model is not installed (``nltk.download('punkt_tab')``).
    """

    def __init__(self, lang: str = "english") -> None: ...
    def load_lang(self, lang: str = "english") -> None:
        """Load the ``punkt_tab`` model for `lang` into this tokenizer."""

    @property
    def save_dir(self) -> str:
        """NLTK's private directory for saved parameters (created by NLTK)."""

    def save_params(self) -> str:
        """Write the parameters as tab files under :attr:`save_dir` (with NLTK); return it."""

def PunktTrainer(*args: Any, **kwargs: Any) -> Any:  # noqa: N802
    """NLTK's ``PunktTrainer`` (training is not ported: a :class:`pyroid.FallbackWarning`).

    Its ``get_params()`` result can be given to :class:`PunktSentenceTokenizer`.
    """

def load_punkt_params(lang_dir: Any) -> PunktParameters:
    """Read the parameters of a ``punkt_tab`` model directory.

    A path is read by pyroid; an NLTK path pointer (from ``nltk.data.find``) by NLTK, with a
    :class:`pyroid.FallbackWarning`.
    """

def format_debug_decision(d: dict[str, Any]) -> str:
    """Format one :meth:`PunktSentenceTokenizer.debug_decisions` dict as NLTK does."""

class MacIntyreContractions:
    """List of contractions adapted from Robert MacIntyre's tokenizer (NLTK's)."""

    CONTRACTIONS2: list[str]
    CONTRACTIONS3: list[str]
    CONTRACTIONS4: list[str]

class _WordTokenizer:
    STARTING_QUOTES: Any
    PUNCTUATION: Any
    PARENS_BRACKETS: Any
    CONVERT_PARENTHESES: Any
    DOUBLE_DASHES: Any
    ENDING_QUOTES: Any
    CONTRACTIONS2: Any
    CONTRACTIONS3: Any

    @overload
    def tokenize(
        self,
        text: str,
        convert_parentheses: bool = False,
        return_str: bool = False,
        *,
        threads: int | None = None,
    ) -> list[str]:
        """Split `text` (one sentence, or several) into tokens.

        Args:
            text: A str, or a batch as for :func:`sent_tokenize`.
            convert_parentheses: Replace brackets with their PTB symbols (``(`` with
                ``-LRB-``, ...).
            return_str: Deprecated, as in NLTK (warns; the result is still a list).
            threads: For a batch, as for :func:`sent_tokenize`.
        """

    @overload
    def tokenize(
        self,
        text: _Batch,
        convert_parentheses: bool = False,
        return_str: bool = False,
        *,
        threads: int | None = None,
    ) -> list[list[str]]: ...
    @overload
    def tokenize(
        self,
        text: ArrowStrings,
        convert_parentheses: bool = False,
        return_str: bool = False,
        *,
        threads: int | None = None,
    ) -> list[list[str] | None]: ...
    def span_tokenize(self, text: str) -> Iterator[tuple[int, int]]:
        """The ``(start, end)`` offsets of the tokens in `text` (NLTK's ``align_tokens``).

        Raises:
            ValueError: A token cannot be found in `text`, as in NLTK.
        """

    def tokenize_sents(
        self, strings: Sequence[str], *, threads: int | None = None
    ) -> list[list[str]]:
        """:meth:`tokenize` of each string (in parallel)."""

    def span_tokenize_sents(self, strings: Sequence[str]) -> Iterator[list[tuple[int, int]]]:
        """:meth:`span_tokenize` of each string, as lists."""

class TreebankWordTokenizer(_WordTokenizer):
    """NLTK's ``TreebankWordTokenizer``: the Penn Treebank tokenizer's rules, in Rust.

    Splits contractions (``don't`` -> ``do n't``), most punctuation, and a period at the end
    of the text. Changing its pattern lists (on the class, a subclass or an instance) hands
    the call to NLTK with those patterns.
    """

class NLTKWordTokenizer(_WordTokenizer):
    """NLTK's ``NLTKWordTokenizer``, the improved Treebank rules :func:`word_tokenize` uses.

    It also splits Unicode quotes and guillemets, dashes, ellipses and some clitics.
    """

class TreebankWordDetokenizer:
    """NLTK's ``TreebankWordDetokenizer``: undo the Treebank tokenizer's padding, in Rust."""

    CONTRACTIONS2: Any
    CONTRACTIONS3: Any
    ENDING_QUOTES: Any
    DOUBLE_DASHES: Any
    CONVERT_PARENTHESES: Any
    PARENS_BRACKETS: Any
    PUNCTUATION: Any
    STARTING_QUOTES: Any

    def tokenize(self, tokens: Sequence[str], convert_parentheses: bool = False) -> str:
        """Join `tokens` into text, undoing the tokenizer's padding.

        Args:
            tokens: The tokens of one text.
            convert_parentheses: Replace PTB symbols (``-LRB-``, ...) with brackets.
        """

    def detokenize(self, tokens: Sequence[str], convert_parentheses: bool = False) -> str:
        """The same as :meth:`tokenize`."""

def align_tokens(tokens: Sequence[str], sentence: str) -> list[tuple[int, int]]:
    """The ``(start, end)`` offsets of `tokens`, found one after another in `sentence`.

    Raises:
        ValueError: A token is not in `sentence` after the previous one.
    """

class RegexpTokenizer:
    r"""NLTK's ``RegexpTokenizer``: the matches of a pattern, or the text between them.

    These patterns run in Rust when spelled exactly so and compiled with no flags beyond the
    default ``re.UNICODE | re.MULTILINE | re.DOTALL``: ``\w+``, ``\w+|[^\w\s]+``,
    ``\w+|\$[\d\.]+|\S+``, ``[\w']+``, ``\S+``, ``\s+``, ``[a-zA-Z]+`` and
    :class:`BlanklineTokenizer`'s. Any other pattern runs in NLTK, with one
    :class:`pyroid.FallbackWarning`.

    Args:
        pattern: The pattern (a str, or a compiled pattern whose ``.pattern`` is used).
        gaps: The pattern matches the separators, not the tokens.
        discard_empty: With `gaps`, drop empty tokens.
        flags: The flags the pattern is compiled with.
    """

    def __init__(
        self, pattern: Any, gaps: bool = False, discard_empty: bool = True, flags: int = ...
    ) -> None: ...
    @overload
    def tokenize(self, text: str, *, threads: int | None = None) -> list[str]:
        """The tokens of `text`; for a batch (as for :func:`sent_tokenize`), one list per doc."""

    @overload
    def tokenize(self, text: _Batch, *, threads: int | None = None) -> list[list[str]]: ...
    @overload
    def tokenize(
        self, text: ArrowStrings, *, threads: int | None = None
    ) -> list[list[str] | None]: ...
    def span_tokenize(self, text: str) -> Iterator[tuple[int, int]]:
        """The ``(start, end)`` offsets of the tokens in `text` (a generator, as NLTK's)."""

    def tokenize_sents(
        self, strings: Sequence[str], *, threads: int | None = None
    ) -> list[list[str]]:
        """:meth:`tokenize` of each string (in parallel)."""

    def span_tokenize_sents(self, strings: Sequence[str]) -> Iterator[list[tuple[int, int]]]:
        """:meth:`span_tokenize` of each string, as lists."""

class WhitespaceTokenizer(RegexpTokenizer):
    r"""Split `text` at runs of whitespace (``\s+`` with ``gaps=True``, ``regex``'s ``\s``)."""

    def __init__(self) -> None: ...

class BlanklineTokenizer(RegexpTokenizer):
    """Split `text` at blank lines (lines of nothing but whitespace)."""

    def __init__(self) -> None: ...

class WordPunctTokenizer(RegexpTokenizer):
    r"""Split `text` into runs of word characters and of other non-space characters.

    The pattern is ``\w+|[^\w\s]+``.
    """

    def __init__(self) -> None: ...

@overload
def wordpunct_tokenize(text: str, *, threads: int | None = None) -> list[str]:
    """:meth:`WordPunctTokenizer.tokenize`."""

@overload
def wordpunct_tokenize(text: _Batch, *, threads: int | None = None) -> list[list[str]]: ...
@overload
def wordpunct_tokenize(
    text: ArrowStrings, *, threads: int | None = None
) -> list[list[str] | None]: ...
@overload
def blankline_tokenize(text: str, *, threads: int | None = None) -> list[str]:
    """:meth:`BlanklineTokenizer.tokenize`."""

@overload
def blankline_tokenize(text: _Batch, *, threads: int | None = None) -> list[list[str]]: ...
@overload
def blankline_tokenize(
    text: ArrowStrings, *, threads: int | None = None
) -> list[list[str] | None]: ...
def regexp_tokenize(
    text: Any, pattern: Any, gaps: bool = False, discard_empty: bool = True, flags: int = ...
) -> Any:
    """:meth:`RegexpTokenizer.tokenize` with a tokenizer made from the arguments."""
