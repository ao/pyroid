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

    Raises:
        As for :func:`gopher_quality`.
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
# ---- end PIIFormatter ----

# ---- C4BadWordsFilter ----
# ---- end C4BadWordsFilter ----

# ---- URLFilter ----
# ---- end URLFilter ----

# ---- SymbolLinesFormatter ----
# ---- end SymbolLinesFormatter ----

# ---- UnigramLogProbFilter ----
# ---- end UnigramLogProbFilter ----
