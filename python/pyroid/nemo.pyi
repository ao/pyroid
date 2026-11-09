"""NeMo Curator 1.4.0's CPU text heuristics in Rust, with the same scores (``pyroid.nemo``).

Each score function returns what the named NeMo filter's ``score_document`` returns, bit for bit:
a float or int for a str, and a float64 or int64 numpy array (one score per document) for a list,
a tuple or an Arrow string array/column. Where NeMo raises for a document, the same exception
type is raised; in a batch, for the first such document, with a note naming its index. NeMo
then applies the keep rule given in each docstring to the score.

Errors shared by every function: ``pyroid.InputError`` when ``texts`` is not one of the accepted
kinds or holds a non-str item, ``UnicodeEncodeError`` for a str with a lone surrogate,
``UnicodeDecodeError`` for invalid UTF-8 in an Arrow row, and ``ValueError`` for an Arrow null
(NeMo needs a str for every document) or ``threads <= 0``.
"""

from collections.abc import Sequence
from typing import Any, Final, Literal, TypeAlias, overload

import numpy as np
import numpy.typing as npt

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings
from pyroid.fasttext import Model

__all__ = [
    "NEMO_CURATOR_VERSION",
    # filter scores and fastText helpers
    "boilerplate_string",
    "bullets",
    "common_english_words",
    "ellipsis",
    "fasttext_lang_id",
    "fasttext_quality",
    "lang_id_strings",
    "long_word",
    "mean_word_length",
    "non_alpha_numeric",
    "numbers",
    "parentheses",
    "pornographic_urls",
    "punctuation",
    "repeated_lines",
    "repeated_lines_by_char",
    "repeated_paragraphs",
    "repeated_paragraphs_by_char",
    "repeating_duplicate_ngrams",
    "repeating_top_ngrams",
    "substring",
    "symbols_to_words",
    "urls",
    "white_space",
    "word_count",
    "words_without_alphabets",
    # string modifiers
    "fasttext_label",
    "normalize_newlines",
    "remove_boilerplate_strings",
    "remove_lines",
    "remove_markdown",
    "remove_quotation",
    "remove_urls",
    "slice_text",
]

NEMO_CURATOR_VERSION: Final[str]

# --- filter scores and fastText helpers -------------------------------------------------------

_F64: TypeAlias = npt.NDArray[np.float64]
_I64: TypeAlias = npt.NDArray[np.int64]

@overload
def non_alpha_numeric(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``NonAlphaNumericFilter`` does.

    The share of characters outside ``[a-zA-Z0-9\n?!,.]``: ``(n - k) / n`` for ``n``
    characters (code points), 1.0 for an empty document. NeMo keeps a document when
    ``score <= max_non_alpha_numeric_to_text_ratio`` (0.25).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column (``string``,
            ``large_string`` or ``string_view``).
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most ``n`` tasks at
            once; ``1`` runs on the calling thread.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def non_alpha_numeric(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def non_alpha_numeric(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def non_alpha_numeric(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def symbols_to_words(texts: str, /, *, lang: str = "en", threads: int | None = None) -> float:
    """Score each document as NeMo's ``SymbolsToWordsFilter`` does.

    The share of words (of ``text.strip()``) that are an ellipsis mark (``...``, ``[...]``,
    ``…``, ``(...)``, ``[…]``, ``-»``, ``read more..``, ``read more``) or whose runs of ``#``
    number more than half their length. NeMo keeps a document when
    ``score <= max_symbol_to_word_ratio`` (0.1).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        lang: NeMo's ``lang``, compared as ``lang.lower()``: ``"zh"`` splits words with jieba
            0.42.1's default dictionary (``jieba.cut``, through :mod:`pyroid.jieba`); any other
            code but ``"ja"`` splits on whitespace (``str.split()``).
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no words, or (``lang="zh"``) a word that strips to
            ``""``, as in NeMo.
        NotImplementedError: ``lang`` is ``"ja"`` (NeMo's MeCab splitter is not ported).
        ValueError: An Arrow element is null.
    """

@overload
def symbols_to_words(
    texts: list[str], /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def symbols_to_words(
    texts: tuple[str, ...], /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def symbols_to_words(
    texts: ArrowStrings, /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def numbers(texts: str, /, *, threads: int | None = None) -> float:
    """Score each document as NeMo's ``NumbersFilter`` does.

    The share of characters that are ASCII digits (NeMo's ``[[:digit:]]``), 1.0 for an empty
    document. NeMo keeps a document when ``score <= max_number_to_text_ratio`` (0.15).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def numbers(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def numbers(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def numbers(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def urls(texts: str, /, *, threads: int | None = None) -> float:
    """Score each document as NeMo's ``UrlsFilter`` does, with its default URL pattern.

    The share of characters inside matches of NeMo's ``regex_url``
    (``https?://`` and a run of URL characters or ``%XX`` escapes), 1.0 for an empty document.
    NeMo keeps a document when ``score <= max_url_to_text_ratio`` (0.2). A custom
    ``url_regex`` is not supported.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def urls(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def urls(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def urls(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def white_space(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``WhiteSpaceFilter`` does.

    The share of characters that are ``\t``, ``\n``, ``\r``, ``\b`` or a space, 1.0 for an
    empty document. NeMo keeps a document when ``score <= max_white_space_ratio`` (0.25).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def white_space(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def white_space(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def white_space(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def parentheses(texts: str, /, *, threads: int | None = None) -> float:
    """Score each document as NeMo's ``ParenthesesFilter`` does.

    The share of characters in ``{ } ⟨ ⟩ [ ] ( )``, 1.0 for an empty document. NeMo keeps a
    document when ``score <= max_parentheses_ratio`` (0.1).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def parentheses(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def parentheses(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def parentheses(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def boilerplate_string(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``BoilerPlateStringFilter`` does.

    1.0 if a paragraph (``text.split("\n\n")``) contains "lorem ipsum" in any case; otherwise
    the share of paragraphs whose ``strip().lower()`` contains one of NeMo's policy strings
    ("terms of use", "privacy policy", "use cookies", ...). NeMo keeps a document when
    ``score <= max_boilerplate_string_ratio`` (0.4).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def boilerplate_string(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def boilerplate_string(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def boilerplate_string(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_lines(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``RepeatedLinesFilter`` does.

    Distinct lines over lines, where the lines are the non-blank pieces of
    ``text.split("\n")``, unstripped. NeMo keeps a document when
    ``score >= max_repeated_line_fraction`` (0.7).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no non-blank line.
        ValueError: An Arrow element is null.
    """

@overload
def repeated_lines(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_lines(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_lines(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_paragraphs(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``RepeatedParagraphsFilter`` does.

    Distinct paragraphs over paragraphs (``text.split("\n\n")``, empty ones included). NeMo
    keeps a document when ``score >= max_repeated_paragraphs_ratio`` (0.7).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def repeated_paragraphs(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_paragraphs(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_paragraphs(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_lines_by_char(texts: str, /, *, threads: int | None = None) -> float:
    """Score each document as NeMo's ``RepeatedLinesByCharFilter`` does.

    The characters of the distinct lines over the characters of all lines (lines as for
    :func:`repeated_lines`). NeMo keeps a document when
    ``score >= max_repeated_lines_char_ratio`` (0.8).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no non-blank line.
        ValueError: An Arrow element is null.
    """

@overload
def repeated_lines_by_char(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_lines_by_char(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_lines_by_char(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_paragraphs_by_char(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``RepeatedParagraphsByCharFilter`` does.

    The characters of the distinct paragraphs over the characters of all paragraphs. NeMo keeps
    a document when ``score >= max_repeated_paragraphs_char_ratio`` (0.8).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: Every paragraph is empty (``""``, ``"\n\n"``).
        ValueError: An Arrow element is null.
    """

@overload
def repeated_paragraphs_by_char(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def repeated_paragraphs_by_char(
    texts: tuple[str, ...], /, *, threads: int | None = None
) -> _F64: ...
@overload
def repeated_paragraphs_by_char(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def word_count(texts: str, /, *, lang: str = "en", threads: int | None = None) -> int:
    """Score each document as NeMo's ``WordCountFilter`` does: its number of words.

    NeMo keeps a document when ``min_words <= score <= max_words`` (50 and 100000).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        lang: As for :func:`symbols_to_words`.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        An int for a str; an int64 numpy array for a batch.

    Raises:
        NotImplementedError: ``lang`` is ``"ja"``.
        ValueError: An Arrow element is null.
    """

@overload
def word_count(texts: list[str], /, *, lang: str = "en", threads: int | None = None) -> _I64: ...
@overload
def word_count(
    texts: tuple[str, ...], /, *, lang: str = "en", threads: int | None = None
) -> _I64: ...
@overload
def word_count(texts: ArrowStrings, /, *, lang: str = "en", threads: int | None = None) -> _I64: ...
@overload
def punctuation(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``PunctuationFilter`` does.

    The share of lines (non-blank pieces of ``text.split("\n")``) whose stripped text does
    not end with ``.``, ``?``, ``!``, ``"`` or ``'``. NeMo keeps a document when
    ``score <= max_num_sentences_without_endmark_ratio`` (0.85).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no non-blank line.
        ValueError: An Arrow element is null.
    """

@overload
def punctuation(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def punctuation(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def punctuation(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def words_without_alphabets(
    texts: str, /, *, lang: str = "en", threads: int | None = None
) -> float:
    """Score each document as NeMo's ``WordsWithoutAlphabetsFilter`` does.

    The share of words with at least one character in ``[[:alpha:]]`` as the ``regex`` module
    defines it (its Alphabetic property, Unicode 18.0.0 in regex 2026.9.29 and 2026.11.9; not
    ``str.isalpha()``). NeMo keeps a document when ``score >= min_words_with_alphabets`` (0.8).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        lang: As for :func:`symbols_to_words`.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no words.
        NotImplementedError: ``lang`` is ``"ja"``.
        ValueError: An Arrow element is null.
    """

@overload
def words_without_alphabets(
    texts: list[str], /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def words_without_alphabets(
    texts: tuple[str, ...], /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def words_without_alphabets(
    texts: ArrowStrings, /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def common_english_words(
    texts: str,
    /,
    *,
    min_num_common_words: int = 2,
    stop_at_false: bool = True,
    threads: int | None = None,
) -> int:
    """Score each document as NeMo's ``CommonEnglishWordsFilter`` does.

    The number of words of ``text.split()`` that are exactly "the", "be", "to", "of", "and",
    "that", "have" or "with". With ``stop_at_false``, counting stops as soon as the count
    reaches ``min_num_common_words`` (checked after every word, as NeMo does). NeMo keeps a
    document when ``score >= min_num_common_words``.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        min_num_common_words: NeMo's ``min_num_common_words``.
        stop_at_false: NeMo's ``stop_at_false``.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        An int for a str; an int64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def common_english_words(
    texts: list[str],
    /,
    *,
    min_num_common_words: int = 2,
    stop_at_false: bool = True,
    threads: int | None = None,
) -> _I64: ...
@overload
def common_english_words(
    texts: tuple[str, ...],
    /,
    *,
    min_num_common_words: int = 2,
    stop_at_false: bool = True,
    threads: int | None = None,
) -> _I64: ...
@overload
def common_english_words(
    texts: ArrowStrings,
    /,
    *,
    min_num_common_words: int = 2,
    stop_at_false: bool = True,
    threads: int | None = None,
) -> _I64: ...
@overload
def mean_word_length(texts: str, /, *, lang: str = "en", threads: int | None = None) -> float:
    """Score each document as NeMo's ``MeanWordLengthFilter`` does: the mean word length.

    Lengths are in characters (code points); empty words are left out. NeMo keeps a document
    when ``min_mean_word_length <= score <= max_mean_word_length`` (3 and 10).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        lang: As for :func:`symbols_to_words`.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no words.
        NotImplementedError: ``lang`` is ``"ja"``.
        ValueError: An Arrow element is null.
    """

@overload
def mean_word_length(
    texts: list[str], /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def mean_word_length(
    texts: tuple[str, ...], /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def mean_word_length(
    texts: ArrowStrings, /, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def long_word(texts: str, /, *, lang: str = "en", threads: int | None = None) -> int:
    """Score each document as NeMo's ``LongWordFilter`` does: its longest word's length.

    NeMo keeps a document when ``score <= max_word_length`` (1000).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        lang: As for :func:`symbols_to_words`.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        An int for a str; an int64 numpy array for a batch.

    Raises:
        ValueError: The document has no words (``max()`` of nothing, as in NeMo), or an Arrow
            element is null.
        NotImplementedError: ``lang`` is ``"ja"``.
    """

@overload
def long_word(texts: list[str], /, *, lang: str = "en", threads: int | None = None) -> _I64: ...
@overload
def long_word(
    texts: tuple[str, ...], /, *, lang: str = "en", threads: int | None = None
) -> _I64: ...
@overload
def long_word(texts: ArrowStrings, /, *, lang: str = "en", threads: int | None = None) -> _I64: ...
@overload
def ellipsis(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``EllipsisFilter`` does.

    The share of lines (non-blank pieces of ``text.split("\n")``) whose ``strip().lower()``
    ends with an ellipsis mark (see :func:`symbols_to_words`). NeMo keeps a document when
    ``score <= max_num_lines_ending_with_ellipsis_ratio`` (0.3).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no non-blank line.
        ValueError: An Arrow element is null.
    """

@overload
def ellipsis(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def ellipsis(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def ellipsis(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def repeating_top_ngrams(
    texts: str, /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> float:
    """Score each document as NeMo's ``RepeatingTopNGramsFilter(n)`` does.

    The share of characters that removing the most frequent word n-gram (its words joined
    with single spaces, every non-overlapping occurrence in the text) would remove; 1.0 for a
    document with fewer than ``n`` words. NeMo keeps a document when
    ``score <= max_repeating_ngram_ratio`` (0.2).

    When several n-grams share the top count, NeMo takes the first of them in the iteration
    order of a Python ``set``, which depends on the process's string hash seed
    (``PYTHONHASHSEED``). For those documents this function builds the same set in Python, so
    the result is the one NeMo gives in the same process; with a fixed ``PYTHONHASHSEED`` it is
    the same in every run.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        n: The n-gram length. Below 1 there are no n-grams and NeMo raises ValueError.
        lang: As for :func:`symbols_to_words`.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ValueError: ``n < 1`` (``max()`` of nothing, as in NeMo), or an Arrow element is null.
        NotImplementedError: ``lang`` is ``"ja"``.
    """

@overload
def repeating_top_ngrams(
    texts: list[str], /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def repeating_top_ngrams(
    texts: tuple[str, ...], /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def repeating_top_ngrams(
    texts: ArrowStrings, /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def repeating_duplicate_ngrams(
    texts: str, /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> float:
    """Score each document as NeMo's ``RepeatingDuplicateNGramsFilter(n)`` does.

    The share of characters in word n-grams seen earlier in the document (overlapping repeats
    counted once, with the spaces between their words); 1.0 below ``n`` words. For ``n < 1``
    there are no n-grams: 0.0 (1.0 for an empty document). NeMo keeps a document when
    ``score <= max_repeating_duplicate_ngram_ratio`` (0.2).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        n: The n-gram length.
        lang: As for :func:`symbols_to_words`.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        NotImplementedError: ``lang`` is ``"ja"``.
        ValueError: An Arrow element is null.
    """

@overload
def repeating_duplicate_ngrams(
    texts: list[str], /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def repeating_duplicate_ngrams(
    texts: tuple[str, ...], /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def repeating_duplicate_ngrams(
    texts: ArrowStrings, /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> _F64: ...
@overload
def bullets(texts: str, /, *, threads: int | None = None) -> float:
    r"""Score each document as NeMo's ``BulletsFilter`` does.

    The share of lines (non-blank pieces of ``text.split("\n")``) whose stripped text starts
    with one of NeMo's twelve bullets (U+2022 ``•``, U+2023, U+2043, U+204C, U+204D,
    U+2219, U+25CB, U+25CF, U+25D8, U+25E6, U+29BE, U+29BF). NeMo keeps a document when
    ``score <= max_bullet_lines_ratio`` (0.9).

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        ZeroDivisionError: The document has no non-blank line.
        ValueError: An Arrow element is null.
    """

@overload
def bullets(texts: list[str], /, *, threads: int | None = None) -> _F64: ...
@overload
def bullets(texts: tuple[str, ...], /, *, threads: int | None = None) -> _F64: ...
@overload
def bullets(texts: ArrowStrings, /, *, threads: int | None = None) -> _F64: ...
@overload
def pornographic_urls(texts: str, /, *, threads: int | None = None) -> int:
    """Score each document as NeMo's ``PornographicUrlsFilter`` does, with its URL pattern.

    1 if a URL (as for :func:`urls`) contains "porn" (case-sensitive), else 0. NeMo keeps a
    document when ``score != 1``.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        An int for a str; an int64 numpy array for a batch.

    Raises:
        ValueError: An Arrow element is null.
    """

@overload
def pornographic_urls(texts: list[str], /, *, threads: int | None = None) -> _I64: ...
@overload
def pornographic_urls(texts: tuple[str, ...], /, *, threads: int | None = None) -> _I64: ...
@overload
def pornographic_urls(texts: ArrowStrings, /, *, threads: int | None = None) -> _I64: ...
@overload
def substring(
    texts: str,
    /,
    substring: str,
    position: Literal["prefix", "suffix", "any"],
    *,
    threads: int | None = None,
) -> int:
    """Score each document as NeMo's ``SubstringFilter(substring, position)`` does.

    1 if the document starts with (``"prefix"``), ends with (``"suffix"``) or contains
    (``"any"``) ``substring``, else 0. NeMo keeps a document when ``score == 1``.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        substring: The substring to look for.
        position: ``"prefix"``, ``"suffix"`` or ``"any"``.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        An int for a str; an int64 numpy array for a batch.

    Raises:
        ValueError: ``position`` is not one of the three (NeMo's message, before any document
            is read), or an Arrow element is null.
    """

@overload
def substring(
    texts: list[str],
    /,
    substring: str,
    position: Literal["prefix", "suffix", "any"],
    *,
    threads: int | None = None,
) -> _I64: ...
@overload
def substring(
    texts: tuple[str, ...],
    /,
    substring: str,
    position: Literal["prefix", "suffix", "any"],
    *,
    threads: int | None = None,
) -> _I64: ...
@overload
def substring(
    texts: ArrowStrings,
    /,
    substring: str,
    position: Literal["prefix", "suffix", "any"],
    *,
    threads: int | None = None,
) -> _I64: ...
@overload
def fasttext_lang_id(
    model: Model, texts: str, /, *, threads: int | None = None
) -> tuple[float, str]:
    r"""Score each document as NeMo's ``FastTextLangId`` does, as a probability and a language.

    The line ``text.strip().replace("\n", " ")`` goes to ``model``'s top-1 prediction; the
    score is its probability (fastText's float32, widened) and the language is its label
    without the ``__label__`` prefix. NeMo stores ``str([score, lang])``
    (:func:`lang_id_strings`) and keeps a document when ``score >= min_langid_score`` (0.3)
    and, with a ``lang`` filter, the language matches (case-insensitively; ``"en"`` matches
    any ``"en_..."`` script).

    Args:
        model: A :class:`pyroid.fasttext.Model` (for example ``lid.176.bin``).
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        ``(score, lang)`` for a str; for a batch, a float64 numpy array of scores and a list of
        languages.

    Raises:
        IndexError: The model predicts no label for a line (no known token), as NeMo's
            ``score[0][0]`` does.
        ValueError: The model is not supervised, or an Arrow element is null.
        RuntimeError: The model's weights give NaN.
        UnicodeDecodeError: The predicted label is not valid UTF-8.
    """

@overload
def fasttext_lang_id(
    model: Model, texts: list[str], /, *, threads: int | None = None
) -> tuple[_F64, list[str]]: ...
@overload
def fasttext_lang_id(
    model: Model, texts: tuple[str, ...], /, *, threads: int | None = None
) -> tuple[_F64, list[str]]: ...
@overload
def fasttext_lang_id(
    model: Model, texts: ArrowStrings, /, *, threads: int | None = None
) -> tuple[_F64, list[str]]: ...
@overload
def fasttext_quality(
    model: Model, texts: str, /, label: str = "__label__hq", *, threads: int | None = None
) -> float:
    r"""Score each document as NeMo's ``FastTextQualityFilter`` does.

    The line ``text.replace("\n", " ").replace("__label__", " ")`` goes to ``model``'s top-1
    prediction with probability ``p`` (fastText's float32, widened); the score is ``p`` if its
    label is ``label``, else ``1 - p``. NeMo keeps a document when
    ``np.random.pareto(alpha) > 1 - score``, a random draw (``alpha`` 3).

    Args:
        model: A :class:`pyroid.fasttext.Model`.
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        label: The label that means high quality.
        threads: As for :func:`non_alpha_numeric`.

    Returns:
        A float for a str; a float64 numpy array for a batch.

    Raises:
        IndexError: The model predicts no label for a line.
        ValueError: The model is not supervised, or an Arrow element is null.
        RuntimeError: The model's weights give NaN.
        UnicodeDecodeError: The predicted label is not valid UTF-8.
    """

@overload
def fasttext_quality(
    model: Model,
    texts: list[str],
    /,
    label: str = "__label__hq",
    *,
    threads: int | None = None,
) -> _F64: ...
@overload
def fasttext_quality(
    model: Model,
    texts: tuple[str, ...],
    /,
    label: str = "__label__hq",
    *,
    threads: int | None = None,
) -> _F64: ...
@overload
def fasttext_quality(
    model: Model,
    texts: ArrowStrings,
    /,
    label: str = "__label__hq",
    *,
    threads: int | None = None,
) -> _F64: ...
def lang_id_strings(
    scores: Sequence[float] | npt.NDArray[np.floating[Any]], langs: Sequence[str]
) -> list[str]:
    """Format :func:`fasttext_lang_id`'s results as NeMo's ``FastTextLangId`` scores.

    Each is ``str([score, lang])``, e.g. ``"[0.98, 'en']"``: Python's ``repr`` of the float and
    of the str.

    Args:
        scores: The probabilities (a float64 array, or any sequence of floats).
        langs: The languages, as many as ``scores``.

    Returns:
        One str per score.

    Raises:
        ValueError: ``scores`` and ``langs`` differ in length.
        TypeError: An item of ``langs`` is not a str.
    """

# --- string modifiers ---------------------------------------------------------------------------
# Each takes a str, a list or tuple of str, or an Arrow string array/column, and returns the same
# kind (a pyroid.Column for Arrow input); unchanged documents come back as the original objects
# and Arrow nulls stay null.

@overload
def remove_boilerplate_strings(
    x: str, /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> str:
    r"""Drop boilerplate paragraphs, as ``BoilerPlateStringModifier`` does.

    Paragraphs are ``text.split("\n\n")``. A paragraph containing "lorem ipsum" (any case)
    makes the document empty. A paragraph with one of NeMo's policy strings ("terms of use",
    "privacy policy", "uses cookies", ...) is boilerplate: with
    ``remove_if_at_top_or_bottom=False`` it makes the document empty; with ``True`` the
    boilerplate paragraphs are removed when they form one block at the top or the bottom of the
    document, and the document is left as it is otherwise (also when every paragraph is
    boilerplate).

    Args:
        x: The documents.
        remove_if_at_top_or_bottom: As for ``BoilerPlateStringModifier``.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most ``n`` tasks at once;
            ``1`` runs on the calling thread.

    Returns:
        The documents, the same kind as ``x``.

    Raises:
        pyroid.InputError: ``x`` is not one of the accepted kinds.
    """

@overload
def remove_boilerplate_strings(
    x: list[str], /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> list[str]: ...
@overload
def remove_boilerplate_strings(
    x: tuple[str, ...], /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def remove_boilerplate_strings(
    x: ArrowStrings, /, *, remove_if_at_top_or_bottom: bool = True, threads: int | None = None
) -> Column[str]: ...
@overload
def remove_markdown(x: str, /, *, threads: int | None = None) -> str:
    """Remove Markdown emphasis and links, as ``MarkdownRemover`` does.

    Line by line, ``**a**``, then ``*a*``, then ``_a_`` become ``a``, and ``[a](b)`` becomes ``b``
    (NeMo's four ``re.sub`` calls with lazy groups, in that order).

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_markdown(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def remove_markdown(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def remove_markdown(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def normalize_newlines(x: str, /, *, threads: int | None = None) -> str:
    r"""Shorten newline runs, as ``NewlineNormalizer`` does.

    Three or more ``\n`` in a row become ``\n\n``; then three or more ``\r\n`` in a row become
    ``\r\n\r\n``.

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def normalize_newlines(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def normalize_newlines(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def normalize_newlines(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def remove_quotation(x: str, /, *, threads: int | None = None) -> str:
    """Remove the quotes around a quoted document, as ``QuotationRemover`` does.

    A document that starts and ends with ``"`` and has more than 2 characters after
    ``strip()`` loses its first and last character, unless it contains a newline and its first
    line ends with ``"``.

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_quotation(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def remove_quotation(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def remove_quotation(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def remove_urls(x: str, /, *, threads: int | None = None) -> str:
    r"""Delete URLs, as ``UrlRemover`` does.

    The same as ``re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)``, including
    ``re``'s case rules (U+017F LATIN SMALL LETTER LONG S matches ``s``).

    Args:
        x: The documents.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_urls(x: list[str], /, *, threads: int | None = None) -> list[str]: ...
@overload
def remove_urls(x: tuple[str, ...], /, *, threads: int | None = None) -> tuple[str, ...]: ...
@overload
def remove_urls(x: ArrowStrings, /, *, threads: int | None = None) -> Column[str]: ...
@overload
def remove_lines(x: str, /, patterns: Sequence[str], *, threads: int | None = None) -> str:
    r"""Drop the lines equal to one of ``patterns``, as ``LineRemover(patterns)`` does.

    Lines are ``text.split("\n")``; a line is dropped when it equals a pattern exactly.

    Args:
        x: The documents.
        patterns: The lines to drop.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def remove_lines(
    x: list[str], /, patterns: Sequence[str], *, threads: int | None = None
) -> list[str]: ...
@overload
def remove_lines(
    x: tuple[str, ...], /, patterns: Sequence[str], *, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def remove_lines(
    x: ArrowStrings, /, patterns: Sequence[str], *, threads: int | None = None
) -> Column[str]: ...
@overload
def slice_text(
    x: str,
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> str:
    """Slice each document, as ``Slicer(left, right, include_left, include_right, strip)`` does.

    An int bound is a Python slice index (in characters). A str ``left`` starts the slice at its
    first occurrence (after it when ``include_left`` is false), a str ``right`` ends it at its
    last occurrence (after it when ``include_right`` is true); the result is ``""`` when either
    string is missing.
    ``None`` means the start or the end. ``strip`` strips the result.

    Args:
        x: The documents.
        left: The start bound.
        right: The end bound.
        include_left: Keep a str ``left`` in the result.
        include_right: Keep a str ``right`` in the result.
        strip: Strip whitespace from the result.
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def slice_text(
    x: list[str],
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> list[str]: ...
@overload
def slice_text(
    x: tuple[str, ...],
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> tuple[str, ...]: ...
@overload
def slice_text(
    x: ArrowStrings,
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> Column[str]: ...
@overload
def fasttext_label(x: str, /, label: object, *, threads: int | None = None) -> str:
    r"""Prefix each document with a fastText label, as ``FastTextLabelModifier(label)`` does.

    The result is ``f"{label} {text}"`` after replacing ``\n`` and ``__label__`` in the text by
    spaces. Every document changes.

    Args:
        x: The documents.
        label: The label (formatted with ``format(label, "")``, as an f-string does).
        threads: As for :func:`remove_boilerplate_strings`.

    Returns:
        The documents, the same kind as ``x``.
    """

@overload
def fasttext_label(x: list[str], /, label: object, *, threads: int | None = None) -> list[str]: ...
@overload
def fasttext_label(
    x: tuple[str, ...], /, label: object, *, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def fasttext_label(
    x: ArrowStrings, /, label: object, *, threads: int | None = None
) -> Column[str]: ...
