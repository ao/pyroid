# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
# Type stubs for pyroid.filters. The module docstring is kept here on purpose:
# the API docs are built from this stub.
"""datatrove 0.10.1's quality filters in Rust, with the same decisions (``pyroid.filters``).

``GopherQualityFilter``, ``GopherRepetitionFilter``, ``C4QualityFilter`` and
``FineWebQualityFilter`` keep and drop the same documents as datatrove, give
the same ``filter_reason`` strings, and leave the same text after C4's line
removal. Parameters have datatrove's names and defaults; pass them as keyword
arguments, as a config object (:class:`GopherQuality` etc.), or pass a
datatrove filter instance, whose settings are read.

- Predicates (``gopher_quality`` ...) return a keep mask in the caller's kind.
- ``*_reasons`` return datatrove's reason per document, ``None`` when kept
  (null in an Arrow column, as are null inputs).
- :func:`c4_lines` is C4's line removal as a transform: each document's
  ``doc.text`` after the filter (unchanged and dropped documents come back as
  the original objects).
- :func:`apply` chains filters as datatrove's executor does (the first drop
  wins and C4's rewritten text is what later filters see), in one pass with
  one shared tokenization.

Words and languages
-------------------
datatrove splits text with the word tokenizer of the filter's ``language``
(``load_word_tokenizer(language)``; the default ``"eng"`` is spaCy 3.8.18's
``spacy.blank("en")`` plus the ``sentencizer``). ``language`` takes any of
datatrove's 4,343 codes (``"deu_Latn"``, ``"deu"``, ``"de"``, ...;
:func:`languages` lists them), as datatrove does:

- **Native** (3,999 codes), at Rust speed on every thread:

  - the 3,742 codes datatrove tokenizes with a rule-based
    ``spacy.blank(code)``: the 57 spaCy languages ``af`` ... ``yo``,
    including ``xx`` (spaCy's multi-language tokenizer, which datatrove
    assigns to 1,995 codes such as ``"uzn_Latn"``). pyroid ports spaCy's
    prefix, suffix, infix and URL rules, each language's tokenizer
    exceptions, the special-case matcher and the sentencizer;
  - the 129 codes of ``IndicNLPTokenizer`` (``"hin_Deva"``, ``"ben_Beng"``,
    ``"urd_Arab"``, ...: indic-nlp-library's ``trivial_tokenize`` and
    ``sentence_split``);
  - the 128 codes of ``WhitespaceTokenizer`` (``"kat_Geor"``, ...: spaCy
    ``xx`` words and datatrove's regex sentence splitter).

  ``split_into_words`` and ``split_into_sentences`` match datatrove's.
- **Opt-in Python tokenizer** (the other 344 codes): datatrove uses a
  segmenter pyroid does not port: ``StanzaTokenizer`` (255 codes),
  ``SpaCyTokenizer("zh")`` (jieba), ``SpaCyTokenizer("ja")`` (SudachiPy),
  ``SpaCyTokenizer("vi")`` (pyvi), ``ThaiTokenizer``, ``BurmeseTokenizer``,
  ``TibetanTokenizer``, ``LaoTokenizer``, ``KhmerTokenizer`` and
  ``KiwiTokenizer`` (see :func:`languages`). Calling a filter with such a code raises
  ``NotImplementedError`` unless you pass ``python_tokenizer=True``. Then each
  document is tokenized by **datatrove's own tokenizer, in Python, on the
  calling thread** (it needs datatrove 0.10.1 and the segmenter installed;
  ``pyroid.PyroidError`` says what is missing), and the filter math runs in
  Rust on those words, so the decisions are datatrove's. That path runs at the
  reference tokenizer's speed (tokenization dominates the filters' cost), only
  tokenizes the documents that reach a word-based check, and emits one
  :class:`pyroid.FallbackWarning` per function and language (C4 steps, which
  run as datatrove's own filter, get their own).
  ``python_tokenizer=True`` also works for native codes (it then uses
  datatrove's tokenizer, slower and no more exact); it is only needed for
  the others.
- ``language`` may also be a word tokenizer object, as datatrove allows: a
  datatrove ``WordTokenizer`` instance, or any object with
  ``word_tokenize(text)`` (and ``span_tokenize(text)`` for C4). It is called
  in Python as above, and needs no ``python_tokenizer=True``.
- A code datatrove does not know raises datatrove's ``ValueError``, when the
  call starts.

On the Python-tokenizer path, a :class:`C4Quality` step (which splits
sentences with the tokenizer) runs as datatrove's own ``C4QualityFilter``; in
:func:`apply`, later steps then see (and tokenize) its rewritten text, as in a
datatrove pipeline.

Python's ``str`` semantics (``isalpha``, ``isspace``, ``splitlines``,
``lower``, the ``re`` digit and word classes) follow the running interpreter's
Unicode version (14.0 to 17.0, CPython 3.11 to 3.15), as datatrove's do.

Differences from datatrove
--------------------------
- A ``top_n_grams`` / ``dup_n_grams`` size below 1 raises ``ValueError``
  (datatrove loops forever on ``dup_n_grams`` sizes below 1).
- Values are checked when the call starts: a non-number threshold raises
  ``TypeError``, and an unknown ``language`` ``ValueError``, before any
  document is filtered (datatrove raises the latter at the first document
  that needs words); a failing document raises for the whole batch (the
  exception names the document).
- ``exclusion_writer`` is not supported (use the ``*_reasons`` functions).
- A document with a token that is exactly ``IS_ALPHA`` (e.g. in
  ``"IS_ALPHA."``, not in ``"xIS_ALPHA"``) is tokenized like any other text,
  as datatrove does the first time in a process. After that, spaCy fails on
  such documents (its string store maps ``IS_ALPHA`` to id 1, which is its
  hash map's deleted-key marker), and datatrove re-tokenizes
  ``text.replace("IS_ALPHA", "")`` instead, or raises if that still has the
  token. So datatrove's words for these documents depend on what it saw
  before; pyroid's do not.
"""

from collections.abc import Iterable, Sequence
from typing import (
    Any,
    Final,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypedDict,
    Unpack,
    overload,
    type_check_only,
)

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings

__all__ = [
    "DATATROVE_VERSION",
    "STOP_WORDS",
    "C4Quality",
    "FilterResult",
    "FineWebQuality",
    "GopherQuality",
    "GopherRepetition",
    "Language",
    "apply",
    "c4_lines",
    "c4_quality",
    "c4_quality_reasons",
    "english_languages",
    "fineweb_quality",
    "fineweb_quality_reasons",
    "gopher_quality",
    "gopher_quality_reasons",
    "gopher_repetition",
    "gopher_repetition_reasons",
    "languages",
]

DATATROVE_VERSION: Final[str]
STOP_WORDS: Final[tuple[str, ...]]

@type_check_only
class _WordTokenizer(Protocol):
    """A word tokenizer object: a datatrove ``WordTokenizer``, or anything with this method."""

    def word_tokenize(self, text: str, /) -> Sequence[str]: ...

@type_check_only
class _SentenceTokenizer(_WordTokenizer, Protocol):
    """A word tokenizer that also splits sentences (what C4 needs)."""

    def span_tokenize(self, text: str, /) -> Sequence[tuple[int, int]]: ...

class GopherQuality(NamedTuple):
    """``GopherQualityFilter``'s parameters and defaults.

    A falsy threshold (``None``, ``0``) disables its check, as in datatrove.
    ``stop_words=None`` means :data:`STOP_WORDS`.
    """

    min_doc_words: float | None = 50
    max_doc_words: float | None = 100000
    min_avg_word_length: float | None = 3
    max_avg_word_length: float | None = 10
    max_symbol_word_ratio: float | None = 0.1
    max_bullet_lines_ratio: float | None = 0.9
    max_ellipsis_lines_ratio: float | None = 0.3
    max_non_alpha_words_ratio: float | None = 0.8
    min_stop_words: float | None = 2
    stop_words: Iterable[str] | None = None
    language: str | _WordTokenizer = "eng"

class GopherRepetition(NamedTuple):
    """``GopherRepetitionFilter``'s parameters and defaults.

    A falsy fraction disables its check. ``top_n_grams`` and ``dup_n_grams``
    are ``(n, max_fraction)`` pairs, ``n >= 1``.
    """

    dup_line_frac: float | None = 0.3
    dup_para_frac: float | None = 0.3
    dup_line_char_frac: float | None = 0.2
    dup_para_char_frac: float | None = 0.2
    top_n_grams: Sequence[tuple[int, float]] = ...
    dup_n_grams: Sequence[tuple[int, float]] = ...
    language: str | _WordTokenizer = "eng"

class C4Quality(NamedTuple):
    """``C4QualityFilter``'s parameters and defaults.

    ``-1`` disables ``min_num_sentences``, ``min_words_per_line`` or
    ``max_word_length``. ``split_paragraph=False`` filters sentences instead
    of lines (and joins the kept ones with spaces).
    """

    split_paragraph: bool = True
    remove_citations: bool = True
    filter_no_terminal_punct: bool = True
    min_num_sentences: int = 5
    min_words_per_line: int = 3
    max_word_length: int = 1000
    filter_lorem_ipsum: bool = True
    filter_javascript: bool = True
    filter_curly_bracket: bool = True
    filter_policy: bool = True
    language: str | _SentenceTokenizer = "eng"

class FineWebQuality(NamedTuple):
    """``FineWebQualityFilter``'s parameters and defaults.

    ``stop_chars=None`` means datatrove's ``TERMINAL_PUNCTUATION``.
    """

    line_punct_thr: float = 0.12
    line_punct_exclude_zero: bool = False
    stop_chars: str | Iterable[str] | None = None
    short_line_thr: float = 0.67
    short_line_length: int = 30
    char_duplicates_ratio: float = 0.01
    new_line_ratio: float = 0.3
    language: str | _WordTokenizer = "eng"

class FilterResult(NamedTuple):
    """What :func:`apply` returns, each in the caller's container kind."""

    keep: Any
    """Keep mask (as the predicates return)."""
    reason: Any
    """Drop reason per document, ``None`` when kept (as ``*_reasons``)."""
    text: Any
    """Each document's text after the chain (as :func:`c4_lines`)."""

@type_check_only
class _GopherQualityKwargs(TypedDict, total=False):
    min_doc_words: float | None
    max_doc_words: float | None
    min_avg_word_length: float | None
    max_avg_word_length: float | None
    max_symbol_word_ratio: float | None
    max_bullet_lines_ratio: float | None
    max_ellipsis_lines_ratio: float | None
    max_non_alpha_words_ratio: float | None
    min_stop_words: float | None
    stop_words: Iterable[str] | None
    language: str | _WordTokenizer

@type_check_only
class _GopherRepetitionKwargs(TypedDict, total=False):
    dup_line_frac: float | None
    dup_para_frac: float | None
    dup_line_char_frac: float | None
    dup_para_char_frac: float | None
    top_n_grams: Sequence[tuple[int, float]]
    dup_n_grams: Sequence[tuple[int, float]]
    language: str | _WordTokenizer

@type_check_only
class _C4QualityKwargs(TypedDict, total=False):
    split_paragraph: bool
    remove_citations: bool
    filter_no_terminal_punct: bool
    min_num_sentences: int
    min_words_per_line: int
    max_word_length: int
    filter_lorem_ipsum: bool
    filter_javascript: bool
    filter_curly_bracket: bool
    filter_policy: bool
    language: str | _SentenceTokenizer

@type_check_only
class _FineWebQualityKwargs(TypedDict, total=False):
    line_punct_thr: float
    line_punct_exclude_zero: bool
    stop_chars: str | Iterable[str] | None
    short_line_thr: float
    short_line_length: int
    char_duplicates_ratio: float
    new_line_ratio: float
    language: str | _WordTokenizer

# ------------------------------------------------------------ Gopher quality

@overload
def gopher_quality(
    texts: str,
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> bool:
    """``GopherQualityFilter``: True for each document datatrove keeps.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column
            (``string``, ``large_string`` or ``string_view``).
        config: A :class:`GopherQuality` (or a datatrove
            ``GopherQualityFilter``); keyword arguments override its fields.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread. On the
            Python-tokenizer path it applies to the Rust part only.
        python_tokenizer: Tokenize with datatrove's own tokenizer for
            ``language``, in Python on the calling thread, and run the filter
            math in Rust on its words (see "Words and languages" above). Needed
            for the languages pyroid does not tokenize natively; emits
            :class:`pyroid.FallbackWarning` once per function and language.
        **kwargs: ``GopherQualityFilter``'s parameters (``language``: a
            datatrove language code or a word tokenizer object).

    Returns:
        bool, ``list[bool]``, ``tuple[bool, ...]``, or a :class:`pyroid.Column`
        of bools (nulls stay null), matching ``texts``.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: A parameter has the wrong type.
        ValueError: datatrove has no tokenizer for ``language`` (datatrove's
            message), or a document of 2**30 characters or more (spaCy's
            limit).
        NotImplementedError: pyroid does not tokenize ``language`` natively
            and ``python_tokenizer`` is not set (the message names
            datatrove's tokenizer and what to install).
        pyroid.PyroidError: The Python-tokenizer path needs datatrove or the
            tokenizer's package, and it is not installed.
        ZeroDivisionError: As datatrove, e.g. ``min_doc_words=None`` and a
            document without words.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def gopher_quality(
    texts: list[str],
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> list[bool]: ...
@overload
def gopher_quality(
    texts: tuple[str, ...],
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> tuple[bool, ...]: ...
@overload
def gopher_quality(
    texts: ArrowStrings,
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> Column[bool]: ...
@overload
def gopher_quality_reasons(
    texts: str,
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> str | None:
    """``GopherQualityFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``gopher_short_doc``, ``gopher_long_doc``,
    ``gopher_below_avg_threshold``, ``gopher_above_avg_threshold``,
    ``gopher_too_many_hashes``, ``gopher_too_many_ellipsis``,
    ``gopher_too_many_bullets``, ``gopher_too_many_end_ellipsis``,
    ``gopher_below_alpha_threshold``, ``gopher_enough_stop_words``.
    Arguments and errors as for :func:`gopher_quality`; Arrow input gives a
    ``string`` :class:`pyroid.Column` (null when kept or null).
    """

@overload
def gopher_quality_reasons(
    texts: list[str],
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> list[str | None]: ...
@overload
def gopher_quality_reasons(
    texts: tuple[str, ...],
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> tuple[str | None, ...]: ...
@overload
def gopher_quality_reasons(
    texts: ArrowStrings,
    config: GopherQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherQualityKwargs],
) -> Column[str]: ...

# --------------------------------------------------------- Gopher repetition

@overload
def gopher_repetition(
    texts: str,
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> bool:
    """``GopherRepetitionFilter``: True for each document datatrove keeps.

    Arguments, returns and errors as for :func:`gopher_quality`, with
    :class:`GopherRepetition` / ``GopherRepetitionFilter`` parameters.
    """

@overload
def gopher_repetition(
    texts: list[str],
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> list[bool]: ...
@overload
def gopher_repetition(
    texts: tuple[str, ...],
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> tuple[bool, ...]: ...
@overload
def gopher_repetition(
    texts: ArrowStrings,
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> Column[bool]: ...
@overload
def gopher_repetition_reasons(
    texts: str,
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> str | None:
    """``GopherRepetitionFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``empty``, ``dup_para_frac``, ``dup_para_char_frac``,
    ``dup_line_frac``, ``dup_line_char_frac``, ``top_{n}_gram``,
    ``duplicated_{n}_n_grams``.
    """

@overload
def gopher_repetition_reasons(
    texts: list[str],
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> list[str | None]: ...
@overload
def gopher_repetition_reasons(
    texts: tuple[str, ...],
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> tuple[str | None, ...]: ...
@overload
def gopher_repetition_reasons(
    texts: ArrowStrings,
    config: GopherRepetition | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_GopherRepetitionKwargs],
) -> Column[str]: ...

# ------------------------------------------------------------------------ C4

@overload
def c4_quality(
    texts: str,
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> bool:
    """``C4QualityFilter``'s document decision: True for each document kept.

    Arguments, returns and errors as for :func:`gopher_quality`, with
    :class:`C4Quality` / ``C4QualityFilter`` parameters. The text after line
    removal is :func:`c4_lines`.
    """

@overload
def c4_quality(
    texts: list[str],
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> list[bool]: ...
@overload
def c4_quality(
    texts: tuple[str, ...],
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> tuple[bool, ...]: ...
@overload
def c4_quality(
    texts: ArrowStrings,
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> Column[bool]: ...
@overload
def c4_quality_reasons(
    texts: str,
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> str | None:
    """``C4QualityFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``lorem_ipsum``, ``curly_bracket``, ``too_few_sentences``.
    """

@overload
def c4_quality_reasons(
    texts: list[str],
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> list[str | None]: ...
@overload
def c4_quality_reasons(
    texts: tuple[str, ...],
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> tuple[str | None, ...]: ...
@overload
def c4_quality_reasons(
    texts: ArrowStrings,
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> Column[str]: ...
@overload
def c4_lines(
    texts: str,
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> str:
    """``C4QualityFilter``'s line removal: ``doc.text`` after the filter.

    For a kept document, the lines (or sentences) C4 keeps, stripped and
    re-joined, citations removed; for a dropped one, the input (datatrove
    leaves it untouched). Unchanged documents are the original objects;
    Arrow nulls stay null.

    Returns:
        str, list, tuple, or a :class:`pyroid.Column` of strings
        (``large_string`` for ``large_string`` input), matching ``texts``.
    """

@overload
def c4_lines(
    texts: list[str],
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> list[str]: ...
@overload
def c4_lines(
    texts: tuple[str, ...],
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> tuple[str, ...]: ...
@overload
def c4_lines(
    texts: ArrowStrings,
    config: C4Quality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_C4QualityKwargs],
) -> Column[str]: ...

# ------------------------------------------------------------------- FineWeb

@overload
def fineweb_quality(
    texts: str,
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> bool:
    """``FineWebQualityFilter``: True for each document datatrove keeps.

    Arguments, returns and errors as for :func:`gopher_quality`, with
    :class:`FineWebQuality` / ``FineWebQualityFilter`` parameters.
    """

@overload
def fineweb_quality(
    texts: list[str],
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> list[bool]: ...
@overload
def fineweb_quality(
    texts: tuple[str, ...],
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> tuple[bool, ...]: ...
@overload
def fineweb_quality(
    texts: ArrowStrings,
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> Column[bool]: ...
@overload
def fineweb_quality_reasons(
    texts: str,
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> str | None:
    """``FineWebQualityFilter``: datatrove's drop reason, ``None`` when kept.

    Reasons: ``empty``, ``line_punct_ratio``, ``short_line_ratio``,
    ``char_dup_ratio``, ``list_ratio``.
    """

@overload
def fineweb_quality_reasons(
    texts: list[str],
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> list[str | None]: ...
@overload
def fineweb_quality_reasons(
    texts: tuple[str, ...],
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> tuple[str | None, ...]: ...
@overload
def fineweb_quality_reasons(
    texts: ArrowStrings,
    config: FineWebQuality | None = None,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
    **kwargs: Unpack[_FineWebQualityKwargs],
) -> Column[str]: ...

# --------------------------------------------------------------------- chain

def apply(
    texts: str | list[str] | tuple[str, ...] | ArrowStrings,
    filters: Iterable[GopherQuality | GopherRepetition | C4Quality | FineWebQuality | Any]
    | GopherQuality
    | GopherRepetition
    | C4Quality
    | FineWebQuality,
    /,
    *,
    threads: int | None = None,
    python_tokenizer: bool = False,
) -> FilterResult:
    """Run several filters in order, in one pass over each document.

    The same as running the datatrove filters one after another in a
    pipeline: a document dropped by one filter is not seen by the next, and
    a :class:`C4Quality` step's rewritten text is what later steps filter.
    Steps that see the same text tokenize it once.

    Args:
        texts: As for :func:`gopher_quality`.
        filters: Config objects, or datatrove filter instances (or one of either)
            (``GopherQualityFilter`` etc.) whose settings are read. Each has its
            own ``language``.
        threads: As for :func:`gopher_quality`.
        python_tokenizer: As for :func:`gopher_quality`, for every step.
            Steps are then run in datatrove's order: consecutive steps with the
            same tokenizer share one tokenization per document, and a C4 step
            runs as datatrove's ``C4QualityFilter``, whose rewritten text the
            later steps tokenize again.

    Returns:
        ``FilterResult(keep, reason, text)``, each in the caller's kind.

    Raises the same exceptions as :func:`gopher_quality`.
    """

class Language(NamedTuple):
    """How datatrove 0.10.1 tokenizes one language code (an entry of :func:`languages`)."""

    tokenizer: str
    """datatrove's tokenizer class: ``"SpaCyTokenizer"``, ``"StanzaTokenizer"``,
    ``"IndicNLPTokenizer"``, ``"WhitespaceTokenizer"``, ``"KiwiTokenizer"``, ..."""
    arg: str
    """The argument it is built with (a spaCy, Stanza or Indic NLP language code), or ``""``."""
    native: bool
    """pyroid tokenizes this code natively; if not, it needs ``python_tokenizer=True``."""

def languages() -> dict[str, Language]:
    """Every language code datatrove 0.10.1 knows, with its tokenizer.

    The keys are what ``language=`` accepts: datatrove's ``tokenizer_assignment.csv``
    as ``load_word_tokenizer`` reads it (``"deu_Latn"``, ``"deu"``, ``"de"``, ...).
    ``pyroid.dedup.languages`` is the same function.

    Example::

        >>> pyroid.filters.languages()["cmn_Hani"]
        Language(tokenizer='SpaCyTokenizer', arg='zh', native=False)

    Returns:
        ``{code: Language(tokenizer, arg, native)}``, sorted by code.
    """

def english_languages() -> list[str]:
    """The datatrove language codes whose tokenizer is ``SpaCyTokenizer("en")``."""

# ======================================================================================
# More datatrove 0.10.1 steps: one section each (each extends __all__).
# ======================================================================================

# ---- PIIFormatter ----
__all__ += ["PII", "PII_EMAIL_REPLACEMENT", "PII_IP_REPLACEMENT", "PIIReplacer", "pii_replace"]

PII_EMAIL_REPLACEMENT: Final[tuple[str, ...]]
"""``PIIFormatter``'s default ``email_replacement``."""
PII_IP_REPLACEMENT: Final[tuple[str, ...]]
"""``PIIFormatter``'s default ``ip_replacement`` (addresses that did not answer pings)."""

class PII(NamedTuple):
    """``PIIFormatter``'s parameters and defaults.

    Attributes:
        remove_emails: Replace email addresses.
        remove_ips: Replace IPv4 addresses.
        only_remove_public_ips: Replace only addresses ``ipaddress`` calls
            ``is_global`` (the running interpreter's tables, as datatrove).
        email_replacement: Replacements, used in turn (a str is one).
        ip_replacement: The same for IP addresses.
    """

    remove_emails: bool = True
    remove_ips: bool = True
    only_remove_public_ips: bool = True
    email_replacement: tuple[str, ...] | str = ...
    ip_replacement: tuple[str, ...] | str = ...

@type_check_only
class _PIIKwargs(TypedDict, total=False):
    remove_emails: bool
    remove_ips: bool
    only_remove_public_ips: bool
    email_replacement: tuple[str, ...] | str
    ip_replacement: tuple[str, ...] | str

class PIIReplacer:
    """datatrove's ``PIIFormatter``, keeping its state across calls.

    Each call rewrites a batch of documents exactly as ``PIIFormatter.format``
    would, one after another: every match of datatrove's email regex becomes
    the next email replacement, then, in that text, every match of its IPv4
    regex that ``ipaddress.ip_address`` accepts (and, by default, calls
    ``is_global``) becomes the next IP replacement. "Next" cycles through each
    replacement tuple and carries over from one document, and one call, to the
    next, so feeding a corpus batch by batch gives datatrove's output.

    The matches are found and replaced in Rust, in parallel; the calling
    thread numbers the replacements and asks ``ipaddress`` about each distinct
    address once.

    Attributes:
        config: The :class:`PII` settings.
        email_index: Index of the next email replacement (datatrove's
            ``emails_replacer._replace_i``); assignable.
        ip_index: The same for IP replacements (``ip_replacer._replace_i``).
    """

    config: PII
    email_index: int
    ip_index: int

    def __init__(self, config: PII | None = None, /, **kwargs: Unpack[_PIIKwargs]) -> None:
        """A formatter whose counters start at 0.

        Args:
            config: A :class:`PII`; keyword arguments override its fields.
            **kwargs: ``PIIFormatter``'s parameters.

        Raises:
            TypeError: A replacement is not a str.
        """

    @overload
    def __call__(self, texts: str, /, *, threads: int | None = None) -> str:
        """Rewrite each document; unchanged documents come back as the same objects.

        Args:
            texts: A str, a list or tuple of str, or an Arrow string array/column.
            threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
                ``n`` tasks at once; ``1`` runs on the calling thread.

        Returns:
            The texts in the caller's kind (a :class:`pyroid.Column` of strings for
            Arrow input; nulls stay null).

        Raises:
            pyroid.InputError: ``texts`` is not one of the accepted kinds.
            IndexError: A match needs a replacement and the tuple is empty (as datatrove).
            UnicodeEncodeError: A str contains a lone surrogate.
            UnicodeDecodeError: An Arrow row is not valid UTF-8.
        """

    @overload
    def __call__(self, texts: list[str], /, *, threads: int | None = None) -> list[str]: ...
    @overload
    def __call__(
        self, texts: tuple[str, ...], /, *, threads: int | None = None
    ) -> tuple[str, ...]: ...
    @overload
    def __call__(self, texts: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...

@overload
def pii_replace(
    texts: str,
    config: PII | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_PIIKwargs],
) -> str:
    """``PIIFormatter``: each document's text with emails and IP addresses replaced.

    A fresh formatter per call, so the replacement counters start at 0: one
    call over a whole corpus gives datatrove's output. To process a corpus in
    batches, use one :class:`PIIReplacer`.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`PII`; keyword arguments override its fields.
        threads: As for :meth:`PIIReplacer.__call__`.
        **kwargs: ``PIIFormatter``'s parameters.

    Returns:
        As :meth:`PIIReplacer.__call__`.
    """

@overload
def pii_replace(
    texts: list[str],
    config: PII | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_PIIKwargs],
) -> list[str]: ...
@overload
def pii_replace(
    texts: tuple[str, ...],
    config: PII | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_PIIKwargs],
) -> tuple[str, ...]: ...
@overload
def pii_replace(
    texts: ArrowStrings,
    config: PII | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_PIIKwargs],
) -> Column[str]: ...

# ---- end PIIFormatter ----

# ---- C4BadWordsFilter ----
import os  # noqa: E402
from collections.abc import Mapping  # noqa: E402

__all__ += [
    "C4_BADWORDS_LANGUAGES",
    "BadWordsFilter",
    "BadWordsResult",
    "C4BadWords",
    "c4_badwords",
    "c4_badwords_reasons",
    "load_c4_badwords",
]

C4_BADWORDS_LANGUAGES: Final[tuple[str, ...]]
"""The 28 languages datatrove 0.10.1 has a word list for (``_BADWORDS_LANGS``)."""

def load_c4_badwords(
    language: str, *, cache_dir: str | os.PathLike[str] | None = None, download: bool = True
) -> frozenset[str]:
    """Datatrove's bad-word set for ``language``.

    The lists are the LDNOOBW "List of Dirty, Naughty, Obscene, and Otherwise
    Bad Words" (https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words,
    CC BY 4.0), at the two commits datatrove pins. pyroid does not ship them:
    like datatrove, it downloads a list the first time it is needed, into
    ``cache_dir`` (default ``$PYROID_CACHE_DIR``, else ``$XDG_CACHE_HOME/pyroid``,
    else ``~/.cache/pyroid``) under ``c4_badwords/``, and checks its sha256.
    The set is built as datatrove builds it: each line ``strip()``-ed, minus
    every language's allowlist (``sm``, ``グロ``, ``女の子``, ``性``).

    Args:
        language: One of :data:`C4_BADWORDS_LANGUAGES`.
        cache_dir: Where the lists are kept.
        download: Fetch a missing list (else ``FileNotFoundError``).

    Raises:
        ValueError: No list for ``language`` (datatrove's message).
        OSError: The download failed, or the file's sha256 is not the pinned one.
    """

class C4BadWords(NamedTuple):
    """``C4BadWordsFilter``'s parameters and defaults (``exclusion_writer`` aside).

    Attributes:
        keep_fraction: The fraction of documents with bad words to keep anyway,
            decided per document by ``numpy.random.default_rng(seed).uniform()``.
        fail_on_missing_language: Raise for a language without a list (else keep
            its documents).
        seed: ``default_rng``'s seed: a non-negative int, or ``None`` for fresh OS
            entropy (as numpy).
        default_language: The language of documents given none.
    """

    keep_fraction: float = 0.0
    fail_on_missing_language: bool = True
    seed: int | None = None
    default_language: str = "en"

@type_check_only
class _C4BadWordsKwargs(TypedDict, total=False):
    keep_fraction: float
    fail_on_missing_language: bool
    seed: int | None
    default_language: str

class BadWordsResult(NamedTuple):
    """What a :class:`BadWordsFilter` call returns, each in the caller's container kind."""

    keep: Any
    """True for each kept document."""
    reason: Any
    """``"document_removed_with_badwords"`` for each dropped document, else ``None``."""

_Languages: TypeAlias = str | Sequence[str | None] | None

class BadWordsFilter:
    r"""datatrove's ``C4BadWordsFilter``, keeping its random state across calls.

    A document is dropped when ``doc.text.lower()`` contains a word of its
    language's list, flanked by non-word characters (``re``'s ``\W``) or the
    text's ends (``ja``, ``th`` and ``zh`` words match anywhere), unless
    ``keep_fraction > 0`` and the next ``uniform()`` draw of numpy's
    ``default_rng(seed)`` is below it. The draws are numpy's PCG64 stream, bit
    for bit, one per flagged document in document order, continuing across
    calls (:attr:`rng_state`). The search runs in Rust, in parallel.

    Attributes:
        config: The :class:`C4BadWords` settings.
        rng_state: ``(state, inc)`` of the PCG64 generator (as numpy's
            ``bit_generator.state["state"]``); assignable.
    """

    config: C4BadWords
    rng_state: tuple[int, int]

    def __init__(
        self,
        config: C4BadWords | None = None,
        /,
        *,
        words: Mapping[str, Iterable[str]] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        download: bool = True,
        **kwargs: Unpack[_C4BadWordsKwargs],
    ) -> None:
        """A filter whose generator is ``default_rng(seed)``.

        Args:
            config: A :class:`C4BadWords`; keyword arguments override its fields.
            words: Word lists to use instead of datatrove's, by language (any
                language code; matched literally, not lowercased). Languages
                not given are loaded with :func:`load_c4_badwords` when first needed.
            cache_dir: As for :func:`load_c4_badwords`.
            download: As for :func:`load_c4_badwords`.
            **kwargs: ``C4BadWordsFilter``'s parameters.

        Raises:
            TypeError: ``seed`` is not an int or ``None``.
            ValueError: ``seed`` is negative.
        """

    def __call__(
        self, texts: Any, /, *, languages: _Languages = None, threads: int | None = None
    ) -> BadWordsResult:
        """Filter ``texts``: keep mask and drop reasons from one pass.

        Args:
            texts: A str, a list or tuple of str, or an Arrow string array/column.
            languages: Each document's language (datatrove's ``doc.metadata["language"]``):
                one str for all, or one entry per document (``None``: the default).
                Default: ``default_language``.
            threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
                ``n`` tasks at once; ``1`` runs on the calling thread.

        Returns:
            ``BadWordsResult(keep, reason)`` in the caller's kind (Arrow nulls stay null).

        Raises:
            ValueError: A language has no list and ``fail_on_missing_language``
                (datatrove's message), or ``languages`` has the wrong length.
            OSError: A list could not be downloaded.
            pyroid.InputError: ``texts`` is not one of the accepted kinds.
            UnicodeEncodeError: A str contains a lone surrogate.
            UnicodeDecodeError: An Arrow row is not valid UTF-8.
        """

    @overload
    def keep(
        self, texts: str, /, *, languages: _Languages = None, threads: int | None = None
    ) -> bool:
        """The keep mask only (draws as :meth:`__call__`)."""

    @overload
    def keep(
        self, texts: list[str], /, *, languages: _Languages = None, threads: int | None = None
    ) -> list[bool]: ...
    @overload
    def keep(
        self,
        texts: tuple[str, ...],
        /,
        *,
        languages: _Languages = None,
        threads: int | None = None,
    ) -> tuple[bool, ...]: ...
    @overload
    def keep(
        self, texts: ArrowStrings, /, *, languages: _Languages = None, threads: int | None = None
    ) -> Column[bool]: ...
    @overload
    def reasons(
        self, texts: str, /, *, languages: _Languages = None, threads: int | None = None
    ) -> str | None:
        """The drop reasons only (draws as :meth:`__call__`)."""

    @overload
    def reasons(
        self, texts: list[str], /, *, languages: _Languages = None, threads: int | None = None
    ) -> list[str | None]: ...
    @overload
    def reasons(
        self,
        texts: tuple[str, ...],
        /,
        *,
        languages: _Languages = None,
        threads: int | None = None,
    ) -> tuple[str | None, ...]: ...
    @overload
    def reasons(
        self, texts: ArrowStrings, /, *, languages: _Languages = None, threads: int | None = None
    ) -> Column[str]: ...
    @overload
    def found(
        self, texts: str, /, *, languages: _Languages = None, threads: int | None = None
    ) -> bool:
        """Whether each document contains a bad word (no draws; missing lists give False)."""

    @overload
    def found(
        self, texts: list[str], /, *, languages: _Languages = None, threads: int | None = None
    ) -> list[bool]: ...
    @overload
    def found(
        self,
        texts: tuple[str, ...],
        /,
        *,
        languages: _Languages = None,
        threads: int | None = None,
    ) -> tuple[bool, ...]: ...
    @overload
    def found(
        self, texts: ArrowStrings, /, *, languages: _Languages = None, threads: int | None = None
    ) -> Column[bool]: ...

@overload
def c4_badwords(
    texts: str,
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> bool:
    """``C4BadWordsFilter``: True for each document datatrove keeps.

    A fresh :class:`BadWordsFilter` per call, so the draws start from
    ``default_rng(seed)``: one call over a whole corpus gives datatrove's
    decisions. For batches, use one :class:`BadWordsFilter`.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`C4BadWords`; keyword arguments override its fields.
        languages: As for :meth:`BadWordsFilter.__call__`.
        words: As for :class:`BadWordsFilter`.
        threads: As for :meth:`BadWordsFilter.__call__`.
        **kwargs: ``C4BadWordsFilter``'s parameters.

    Returns:
        A keep mask in the caller's kind.
    """

@overload
def c4_badwords(
    texts: list[str],
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> list[bool]: ...
@overload
def c4_badwords(
    texts: tuple[str, ...],
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> tuple[bool, ...]: ...
@overload
def c4_badwords(
    texts: ArrowStrings,
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> Column[bool]: ...
@overload
def c4_badwords_reasons(
    texts: str,
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> str | None:
    """``C4BadWordsFilter``: datatrove's drop reason per document, ``None`` when kept.

    As :func:`c4_badwords`, returning ``"document_removed_with_badwords"`` or ``None``.
    """

@overload
def c4_badwords_reasons(
    texts: list[str],
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> list[str | None]: ...
@overload
def c4_badwords_reasons(
    texts: tuple[str, ...],
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> tuple[str | None, ...]: ...
@overload
def c4_badwords_reasons(
    texts: ArrowStrings,
    config: C4BadWords | None = None,
    /,
    *,
    languages: _Languages = None,
    words: Mapping[str, Iterable[str]] | None = None,
    threads: int | None = None,
    **kwargs: Unpack[_C4BadWordsKwargs],
) -> Column[str]: ...

# ---- end C4BadWordsFilter ----

# ---- URLFilter ----
__all__ += [
    "TLDEXTRACT_VERSION",
    "DatatroveURLLists",
    "URLFilter",
    "URLFilterConfig",
    "datatrove_url_lists",
    "url_filter",
    "url_filter_reasons",
]

#: The tldextract release whose ``extract_str`` the URL filter reproduces.
TLDEXTRACT_VERSION: Final[str]

class URLFilterConfig(NamedTuple):
    """datatrove 0.10.1 ``URLFilter``'s parameters and defaults.

    The lists are read as datatrove's ``parse_list`` reads them: entries starting
    with ``#`` are skipped (the test is ``x[0] != "#"``, so an empty string raises
    ``IndexError``, as in datatrove); domains and URLs are ``strip()``-ped; words,
    subwords and soft words are normalized (every character but ``[a-zA-Z0-9]``
    removed, then lowercased), so ``"--"`` becomes ``""``. A falsy list adds nothing.
    """

    soft_word_threshold: Any = 2
    extra_domains: Iterable[str] | None = None
    extra_urls: Iterable[str] | None = None
    banned_words: Iterable[str] | None = None
    banned_subwords: Iterable[str] | None = None
    soft_banned_words: Iterable[str] | None = None
    use_integrated_lists: bool = True

@type_check_only
class _URLFilterKwargs(TypedDict, total=False):
    soft_word_threshold: float
    extra_domains: Iterable[str] | None
    extra_urls: Iterable[str] | None
    banned_words: Iterable[str] | None
    banned_subwords: Iterable[str] | None
    soft_banned_words: Iterable[str] | None
    use_integrated_lists: bool

class DatatroveURLLists(NamedTuple):
    """The files of datatrove's integrated URL block lists (see :func:`datatrove_url_lists`)."""

    domains: str
    urls: str
    banned_words: str
    banned_subwords: str
    soft_banned_words: str

def datatrove_url_lists() -> DatatroveURLLists:
    """Extract datatrove's integrated URL block lists as datatrove does; return their paths.

    datatrove 0.10.1 ships them in its package (``datatrove/assets``):
    ``url_filterblacklistsv0_3_0.tar.gz`` (``domains``: 4,558,940 lines; ``urls``:
    19,587), ``banned_words.txt``, ``banned_subwords.txt`` and
    ``soft_banned_words.txt``. The archive is extracted once into datatrove's own cache
    (``huggingface_hub.cached_assets_path("datatrove", "filters", "url_filter")``),
    under datatrove's lock and ``.completed`` marker, so datatrove and pyroid share
    one copy. pyroid does not ship these lists: they are not under pyroid's licence.

    Raises:
        pyroid.PyroidError: datatrove (or huggingface_hub) is not installed.
        ImportError: fasteners is not installed (datatrove's lock needs it).
    """

class URLFilter:
    """datatrove 0.10.1's ``URLFilter``, built once: the same decisions and drop reasons.

    Build it once and reuse it: with the integrated lists it holds 4.56 million
    blocked domains (about 200 MB, in one Rust string arena with a hash table),
    and building it takes a second or two. Each URL is checked as datatrove does,
    in its order, and the first hit is the reason:

    1. ``"domain"``: tldextract's registered domain
       (``top_domain_under_public_suffix``) is a blocked domain;
    2. ``"subdomain"``: its FQDN is a blocked domain;
    3. ``"url"``: the URL itself is blocked;
    4. ``"hard_blacklisted"``: a word of ``re.split(r"[^a-zA-Z0-9]+", url)`` (case
       sensitive; ``""`` when the URL starts or ends with another character) is
       a banned word;
    5. ``"soft_blacklisted"``: at least ``soft_word_threshold`` distinct words are
       soft-banned;
    6. ``"blacklisted_subword"``: a banned subword is a substring of the URL with
       every character but ``[a-zA-Z0-9]`` removed, lowercased.

    Domains come from tldextract 5.4.0's ``extract_str``, ported to Rust:
    ``lenient_netloc`` (scheme, userinfo, ``[IPv6]``, port, ``strip()``, trailing
    dots), U+3002, U+FF0E and U+FF61 as dots, the IPv6 literal test (``ipaddress.IPv6Address``), and
    the Public Suffix List walk (wildcards such as ``*.ck``, exceptions such as
    ``!www.ck``), labels lowered with ``str.lower()`` of the running interpreter's
    Unicode version and ``xn--`` labels decoded as ``idna`` does. The suffix lists
    are the ones ``tld_extractor`` uses (its cached copy of the Public Suffix List,
    or tldextract's bundled snapshot), read from it when the filter is built;
    with its ``include_psl_private_domains`` setting.

    Args:
        config: A :class:`URLFilterConfig`; keyword arguments override its fields.
        tld_extractor: A ``tldextract.TLDExtract`` (default: a new
            ``TLDExtract()``, as datatrove builds).
        **kwargs: ``URLFilter``'s parameters (see :class:`URLFilterConfig`).

    Raises:
        TypeError: ``config`` is not a :class:`URLFilterConfig`, or
            ``soft_word_threshold`` does not compare with an int (datatrove raises
            this ``TypeError`` at the first URL that reaches the soft-word check).
        IndexError: A list has an empty string (as datatrove).
        pyroid.PyroidError: tldextract is not installed, or
            ``use_integrated_lists=True`` and datatrove is not installed.

    The methods take URLs as a str, a list or tuple of str, or an Arrow string
    array/column, with ``threads=`` as everywhere in pyroid, and raise
    ``ValueError`` for an empty URL (datatrove's ``assert url``),
    ``AttributeError`` as datatrove when the banned subwords are only ``""``
    (pyahocorasick's automaton then has no words) and a URL reaches that check,
    ``pyroid.InputError`` for other input kinds, ``UnicodeEncodeError`` for a lone
    surrogate and ``UnicodeDecodeError`` for invalid UTF-8 in Arrow input.
    """

    config: URLFilterConfig | None
    def __init__(
        self,
        config: URLFilterConfig | None = None,
        /,
        *,
        tld_extractor: Any = None,
        **kwargs: Unpack[_URLFilterKwargs],
    ) -> None: ...
    @overload
    def keep(self, urls: str, /, *, threads: int | None = None) -> bool:
        """True for each URL datatrove keeps.

        Returns:
            bool, ``list[bool]``, ``tuple[bool, ...]``, or a :class:`pyroid.Column`
            of bools (nulls stay null), matching ``urls``.
        """

    @overload
    def keep(self, urls: list[str], /, *, threads: int | None = None) -> list[bool]: ...
    @overload
    def keep(self, urls: tuple[str, ...], /, *, threads: int | None = None) -> tuple[bool, ...]: ...
    @overload
    def keep(self, urls: ArrowStrings, /, *, threads: int | None = None) -> Column[bool]: ...
    @overload
    def reasons(self, urls: str, /, *, threads: int | None = None) -> str | None:
        """The drop reason (datatrove's) per URL, ``None`` when kept.

        Reasons: ``domain``, ``subdomain``, ``url``, ``hard_blacklisted``,
        ``soft_blacklisted``, ``blacklisted_subword``. Arrow input gives a
        ``string`` :class:`pyroid.Column` (null when kept or null).
        """

    @overload
    def reasons(self, urls: list[str], /, *, threads: int | None = None) -> list[str | None]: ...
    @overload
    def reasons(
        self, urls: tuple[str, ...], /, *, threads: int | None = None
    ) -> tuple[str | None, ...]: ...
    @overload
    def reasons(self, urls: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
    @overload
    def domains(self, urls: str, /, *, threads: int | None = None) -> tuple[str, str]:
        """The ``(top_domain_under_public_suffix, fqdn)`` tldextract gives each URL.

        What datatrove looks up in the blocked domains (``registered_domain`` is
        ``top_domain_under_public_suffix``). Both are ``""`` for IP addresses,
        hosts without a public suffix, and bare suffixes.

        Returns:
            A pair, each in the kind of ``urls`` (Arrow: ``large_string`` columns
            for ``large_string`` input, else ``string``; nulls stay null).
        """

    @overload
    def domains(
        self, urls: list[str], /, *, threads: int | None = None
    ) -> tuple[list[str], list[str]]: ...
    @overload
    def domains(
        self, urls: tuple[str, ...], /, *, threads: int | None = None
    ) -> tuple[tuple[str, ...], tuple[str, ...]]: ...
    @overload
    def domains(
        self, urls: ArrowStrings, /, *, threads: int | None = None
    ) -> tuple[Column[str], Column[str]]: ...
    @property
    def tld_extractor(self) -> Any:
        """The ``tldextract.TLDExtract`` whose suffix lists are used."""

    @property
    def num_domains(self) -> int:
        """Number of blocked domains."""

    @property
    def num_urls(self) -> int:
        """Number of blocked URLs."""

@overload
def url_filter(
    urls: str,
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> bool:
    """``URLFilter``: True for each URL datatrove keeps.

    ``config`` is a :class:`URLFilterConfig` (keyword arguments override its
    fields) or a built :class:`URLFilter`. The last two filters built from configs
    are cached (configs whose lists are lists, tuples, sets or strings), so repeated
    calls do not reload the block lists.

    Returns:
        As :meth:`URLFilter.keep`.
    """

@overload
def url_filter(
    urls: list[str],
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> list[bool]: ...
@overload
def url_filter(
    urls: tuple[str, ...],
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> tuple[bool, ...]: ...
@overload
def url_filter(
    urls: ArrowStrings,
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> Column[bool]: ...
@overload
def url_filter_reasons(
    urls: str,
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> str | None:
    """``URLFilter``: datatrove's drop reason per URL, ``None`` when kept.

    Arguments as for :func:`url_filter`; results as :meth:`URLFilter.reasons`.
    """

@overload
def url_filter_reasons(
    urls: list[str],
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> list[str | None]: ...
@overload
def url_filter_reasons(
    urls: tuple[str, ...],
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> tuple[str | None, ...]: ...
@overload
def url_filter_reasons(
    urls: ArrowStrings,
    config: URLFilterConfig | URLFilter | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_URLFilterKwargs],
) -> Column[str]: ...

# ---- end URLFilter ----

# ---- SymbolLinesFormatter ----
__all__ += ["SymbolLines", "symbol_lines"]

class SymbolLines(NamedTuple):
    r"""``SymbolLinesFormatter``'s parameters and defaults.

    Attributes:
        symbols_to_remove: The symbols a removed line may consist of (with
            ``" "``); ``None`` is datatrove's ``PUNCTUATION_SET``.
        replace_char: What a run of removed lines becomes (``""``: nothing;
            FineWeb-2 uses ``"\n"``).
    """

    symbols_to_remove: Iterable[str] | None = None
    replace_char: str = ""

@type_check_only
class _SymbolLinesKwargs(TypedDict, total=False):
    symbols_to_remove: Iterable[str] | None
    replace_char: str

@overload
def symbol_lines(
    texts: str,
    config: SymbolLines | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_SymbolLinesKwargs],
) -> str:
    r"""``SymbolLinesFormatter``: each document without its symbol-only lines.

    As datatrove: over ``text.splitlines()``, a non-blank line made only of
    ``symbols_to_remove`` and spaces is dropped, a run of them becomes one
    ``replace_char`` (if not empty), and the kept lines are joined with
    ``"\n"`` (so ``"\r\n"`` and a trailing newline come out as datatrove's do).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`SymbolLines`; keyword arguments override its fields.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        **kwargs: ``SymbolLinesFormatter``'s parameters.

    Returns:
        The texts in the caller's kind; a document that comes out equal comes
        back as the same object (Arrow: a :class:`pyroid.Column`, nulls stay null).

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: ``replace_char`` is not a str.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def symbol_lines(
    texts: list[str],
    config: SymbolLines | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_SymbolLinesKwargs],
) -> list[str]: ...
@overload
def symbol_lines(
    texts: tuple[str, ...],
    config: SymbolLines | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_SymbolLinesKwargs],
) -> tuple[str, ...]: ...
@overload
def symbol_lines(
    texts: ArrowStrings,
    config: SymbolLines | None = None,
    /,
    *,
    threads: int | None = None,
    **kwargs: Unpack[_SymbolLinesKwargs],
) -> Column[str]: ...

# ---- end SymbolLinesFormatter ----
