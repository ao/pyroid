# Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
"""NeMo Curator 1.4.0's CPU text heuristics in Rust, with the same scores (``pyroid.nemo``).

One function per heuristic ``DocumentFilter`` (its ``score_document``) and per string
``DocumentModifier`` (its ``modify_document``), plus the fastText stages' scoring. NeMo
Curator itself is not needed. Types and docs are in ``nemo.pyi``; the drop-in pipeline
stages are in :mod:`pyroid.integrations.nemo_curator`.
"""

from __future__ import annotations

import functools
import sys
from typing import Any

from pyroid._pyroid import nemo as _native

#: The NeMo Curator release whose results this module reproduces.
NEMO_CURATOR_VERSION = "1.4.0"

# --- filter scores and fastText helpers -------------------------------------------------------

non_alpha_numeric = _native.non_alpha_numeric
symbols_to_words = _native.symbols_to_words
numbers = _native.numbers
urls = _native.urls
white_space = _native.white_space
parentheses = _native.parentheses
boilerplate_string = _native.boilerplate_string
repeated_lines = _native.repeated_lines
repeated_paragraphs = _native.repeated_paragraphs
repeated_lines_by_char = _native.repeated_lines_by_char
repeated_paragraphs_by_char = _native.repeated_paragraphs_by_char
word_count = _native.word_count
punctuation = _native.punctuation
words_without_alphabets = _native.words_without_alphabets
common_english_words = _native.common_english_words
mean_word_length = _native.mean_word_length
long_word = _native.long_word
ellipsis = _native.ellipsis
repeating_duplicate_ngrams = _native.repeating_duplicate_ngrams
bullets = _native.bullets
pornographic_urls = _native.pornographic_urls
substring = _native.substring
fasttext_lang_id = _native.fasttext_lang_id
fasttext_quality = _native.fasttext_quality
lang_id_strings = _native.lang_id_strings


def _tied_score(text: str, n: int, starts: list[int] | None, words: list[str] | None) -> float:
    """NeMo's score for a document whose most frequent n-gram is tied.

    ``RepeatingTopNGramsFilter`` keeps the first of the tied n-grams in the iteration order of
    ``set(ngrams)``. That order depends on this process's string hashes, so the set is built
    here exactly as NeMo builds it: the same tuples of str, inserted in the same order. Used
    when the Rust kernel cannot resolve ties itself (see :func:`_hash_key`).
    """
    if words is None:
        words = text.strip().split()
    ngrams = list(zip(*[words[i:] for i in range(n)], strict=False))
    unique = set(ngrams)
    if starts is None:  # every n-gram is tied
        top = next(iter(unique))
    else:
        tied = {ngrams[j] for j in starts}
        top = next(filter(tied.__contains__, unique))
    most = " ".join(top)
    nchar = len(text)
    return (nchar - len(text.replace(most, ""))) / nchar


def _secret() -> bytes:
    """The siphash key of CPython's string hashes: the first 16 bytes of ``_Py_HashSecret``."""
    import ctypes

    return bytes((ctypes.c_ubyte * 24).in_dll(ctypes.pythonapi, "_Py_HashSecret"))[:16]


#: Characters of every PEP 393 width: 1 byte (ASCII, Latin-1), 2 bytes and 4 bytes.
_WIDTHS = (
    "az~ \x00\x7f",
    "\x80\xe9\xff",
    "\u0100\u4e2d\u6587\ud7ff\ue000\uffff",
    "\U0001f600\U00010000\U0010ffff",
)
#: ``hash(x) == hash(x + _MODULUS)`` for ints: equal hashes, different items.
_MODULUS = (1 << 61) - 1


def _self_check(key: bytes) -> bool:
    """Whether pyroid's str hashes and set order, with ``key``, are this interpreter's.

    Compares ``hash()`` of strings of every width with lengths 0 to 24 (siphash13's tail and
    block paths); ``list(set(ngrams))`` for 200 lists of 0 to about 2000 n-grams, with repeats,
    through the code that resolves the ties; and ``list(set(ints))`` for lists whose items
    share slots or whole hashes, and for one past the size at which CPython grows its table
    2x instead of 4x (78643 items).
    """
    import random

    rng = random.Random(0x6E656D6F)
    pools = ["".join(_WIDTHS[:widest]) for widest in range(1, len(_WIDTHS) + 1)]
    strings = [""]
    strings += ["".join(rng.choices(p, k=length)) for length in range(1, 25) for p in pools]
    if _native._str_hashes(strings, key) != [hash(s) for s in strings]:
        return False
    vocab = strings[:48]
    for _ in range(200):
        n = rng.randint(1, 4)
        words = rng.choices(vocab, k=int(10 ** rng.uniform(0, 3.3)) - 1 + n)
        ngrams = list(zip(*[words[i:] for i in range(n)], strict=False))
        # The first index of each n-gram: later (reversed) assignments win.
        first = dict(zip(reversed(ngrams), reversed(range(len(ngrams))), strict=True))
        if _native._ngram_set_order(words, n, key) != [first[g] for g in set(ngrams)]:
            return False
    mix = 0x9E3779B97F4A7C15  # odd: i * mix is a bijection modulo 2**60
    for size, extra in ((3, 1), (40, 10), (700, 175), (6000, 1500), (79_000, 500)):
        base = [i * mix & ((1 << 60) - 1) for i in range(1, size + 1)]
        some = base[:extra]
        twins = [b + _MODULUS for b in some]  # same hash as b, a different item
        xs = [*twins[::2], *base, *twins[1::2], -1, -2, *(b >> 12 << 12 for b in some), *some]
        order = _native._set_order([hash(x) for x in xs], xs)
        if [xs[i] for i in order] != list(set(xs)):
            return False
    return True


@functools.cache
def _hash_key() -> bytes | None:
    """This process's string hash key, if pyroid reproduces its set order (checked once).

    ``None`` (ties are then resolved in Python, with the same result) on interpreters other
    than 64-bit CPython with siphash13, if the key cannot be read, or if the check fails.
    """
    info = sys.hash_info
    if (
        sys.implementation.name != "cpython"
        or info.algorithm != "siphash13"
        or info.cutoff != 0
        or info.hash_bits != 64
    ):
        return None
    try:
        key = _secret()
        return key if _self_check(key) else None
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return None


def repeating_top_ngrams(
    texts: Any, /, n: int = 2, *, lang: str = "en", threads: int | None = None
) -> Any:
    """``RepeatingTopNGramsFilter(n)``'s score; see ``nemo.pyi``."""
    scores, ties = _native._repeating_top_ngrams(
        texts, n, lang=lang, threads=threads, hash_key=_hash_key()
    )
    if isinstance(texts, str):
        if ties:
            _, text, starts, words = ties[0]
            return _tied_score(text, n, starts, words)
        return scores
    for i, text, starts, words in ties:
        scores[i] = _tied_score(text, n, starts, words)
    return scores


# --- string modifiers ---------------------------------------------------------------------------

remove_boilerplate_strings = _native.remove_boilerplate_strings
remove_markdown = _native.remove_markdown
normalize_newlines = _native.normalize_newlines
remove_quotation = _native.remove_quotation
remove_urls = _native.remove_urls
remove_lines = _native.remove_lines

# Python slice indices beyond this are the same as this, for any text pyroid can hold.
_INDEX_LIMIT = 1 << 62


def _slice_bound(value: object) -> tuple[int | None, str | None]:
    # Slicer's own isinstance checks, in its order (bool is an int).
    if isinstance(value, int):
        return max(-_INDEX_LIMIT, min(_INDEX_LIMIT, value)), None
    if isinstance(value, str):
        return None, value
    return None, None


def slice_text(
    x: Any,
    /,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
    *,
    threads: int | None = None,
) -> Any:
    """Slice each document as ``Slicer(...).modify_document`` does (see ``nemo.pyi``)."""
    li, ls = _slice_bound(left)
    ri, rs = _slice_bound(right)
    return _native._slice_text(
        x, li, ls, ri, rs, bool(include_left), bool(include_right), bool(strip), threads=threads
    )


def fasttext_label(x: Any, /, label: object, *, threads: int | None = None) -> Any:
    """Prefix each document with ``label`` as ``FastTextLabelModifier`` does (see ``nemo.pyi``)."""
    # FastTextLabelModifier builds f"{label} {text}".
    return _native._fasttext_label(x, format(label, ""), threads=threads)


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
