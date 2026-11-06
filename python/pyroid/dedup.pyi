# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
# Type stubs for pyroid.dedup. The module docstring is kept here on purpose:
# the API docs are built from this stub.
"""MinHash near-duplicate and exact deduplication (``pyroid.dedup``).

The MinHash pipeline makes **the same removals as datatrove 0.10.1's**
``MinhashDedupSignature`` -> ``MinhashDedupBuckets`` -> ``MinhashDedupCluster``
-> ``MinhashDedupFilter``, in memory and multi-threaded:

1. *Signatures.* Each document is normalised with datatrove's
   ``simplify_text``, split into words by the word tokenizer of ``language``
   (datatrove's ``word_tokenize``; spaCy English by default), and every ``n_grams``-word window is
   hashed (``hash_fc``); ``num_buckets * hashes_per_bucket`` permutations
   ``(a * x + b) mod (2**61 - 1)``, with ``a`` and ``b`` drawn from numpy's
   ``RandomState(seed)`` exactly as datatrove draws them, give the signature.
   Documents with fewer than ``n_grams`` words have no signature and are
   always kept.
2. *Buckets.* Two documents are a duplicate pair when all
   ``hashes_per_bucket`` values of one bucket are equal.
3. *Clusters.* Pairs are merged with datatrove's union-find (union by size).
   **The document kept from each cluster is the union-find root**: the root
   of the larger side wins each merge, and ties go to the earlier document
   of the pair. That is usually, but not always, the cluster's first
   document; :meth:`MinHashDedup.clusters` lists the kept one first.
4. *Filter.* Every other member of a cluster is removed.

Documents are numbered in the order they are added, as datatrove numbers the
documents of one task. Configurations are datatrove's: pass a
:class:`MinhashConfig`, datatrove's own ``MinhashConfig``, a dict, or keyword
arguments (``hash_fc=`` and ``precision=`` set ``hash_config``). The default
is datatrove's default (xxhash/64); FineWeb used
``MinhashConfig(hash_config=HashConfig("sha1", 64))``.

Languages
---------
``language=`` is ``MinhashDedupSignature(language=...)``: any of datatrove's
language codes (:func:`languages`), as in ``pyroid.filters``. 3,999 of the
4,343 codes are tokenized natively, in Rust, on every thread: those datatrove
tokenizes with a rule-based ``spacy.blank(code)`` (57 spaCy languages, ``xx``
included), ``IndicNLPTokenizer`` or ``WhitespaceTokenizer``. The other 344
(Chinese with jieba, Japanese with SudachiPy, Vietnamese with pyvi, Stanza,
Thai, Kiwi, ...) raise ``NotImplementedError`` unless
``python_tokenizer=True``: then pyroid's ``simplify_text`` runs in Rust,
**datatrove's own tokenizer splits each document in Python, on the calling
thread**, and shingling, MinHash, bucketing and clustering run in Rust on
those words, with one :class:`pyroid.FallbackWarning` per function and
language. That path runs at the reference tokenizer's speed. ``language`` may
also be a word tokenizer object (a datatrove ``WordTokenizer``, or anything
with ``word_tokenize(text)``), called the same way.

Known differences from datatrove
--------------------------------
- A str with a lone surrogate raises ``UnicodeEncodeError`` (pyroid's policy);
  datatrove raises it too unless the document has fewer than ``n_grams``
  words.
- ``n_grams``, ``num_buckets`` or ``hashes_per_bucket`` below 1 raise
  ``ValueError`` (datatrove silently produces no signatures or fails later).
- Index files (``index_folder``, ``create_index_name``,
  ``only_dedup_in_index``) are not supported.
- An unknown ``language`` raises datatrove's ``ValueError`` when the call
  starts, as ``MinhashDedupSignature`` does when it is built.
"""

from dataclasses import dataclass
from typing import Any, Final, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings
from pyroid.filters import Language as Language
from pyroid.filters import _WordTokenizer
from pyroid.filters import languages as languages

__all__ = [
    "DATATROVE_VERSION",
    "ExactDedup",
    "HashConfig",
    "Language",
    "MinHashDedup",
    "MinhashConfig",
    "TextNormConfig",
    "exact_dedup",
    "languages",
    "minhash_dedup",
    "minhash_signatures",
    "simplify_text",
]

DATATROVE_VERSION: Final[str]
_UNICODE_VERSION: Final[str]

def _native_config(cfg: MinhashConfig, language: str | None = None) -> Any: ...

@dataclass(frozen=True)
class HashConfig:
    """datatrove's ``HashConfig``: the shingle hash function and its width.

    Attributes:
        precision: 64 or 32 bits.
        hash_fc: ``"xxhash"`` (xxHash64/xxHash32, seed 0) or ``"sha1"`` (the
            first 8 or 4 bytes of SHA-1, little-endian).
    """

    precision: int = 64
    hash_fc: str = "xxhash"

@dataclass(frozen=True)
class TextNormConfig:
    """datatrove's ``TextNormConfig``: what :func:`simplify_text` does, in order.

    Attributes:
        lowercase: ``str.lower()``.
        norm_numbers: Every run of digits (any script, with at most one
            ``.``/``,``-style separator inside) becomes ``0``.
        norm_weekdays: English weekday names become ``WEEKDAY``.
        norm_monthnames: English month names become ``MONTH``.
        remove_punctuation: datatrove's punctuation set becomes spaces.
        norm_whitespace: Strip, and collapse whitespace runs to one space.
        norm_unicode_diacritics: NFD, then drop combining marks (``Mn``).
    """

    lowercase: bool = True
    norm_whitespace: bool = True
    remove_punctuation: bool = True
    norm_unicode_diacritics: bool = True
    norm_numbers: bool = True
    norm_weekdays: bool = False
    norm_monthnames: bool = False

@dataclass(frozen=True)
class MinhashConfig:
    """datatrove's ``MinhashConfig``, with the same fields and defaults.

    Attributes:
        n_grams: Words per shingle.
        num_buckets: Bands of the LSH index.
        hashes_per_bucket: Hashes per band; a pair needs all of them equal.
        seed: Seed of numpy's ``RandomState`` that draws the permutations
            (0 to 2**32 - 1).
        norm_config: The :class:`TextNormConfig` applied before shingling.
        hash_config: The :class:`HashConfig` of the shingle hash.
    """

    n_grams: int = 5
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    seed: int = 1
    norm_config: TextNormConfig = ...
    hash_config: HashConfig = ...

@overload
def minhash_signatures(
    texts: str,
    config: Any = None,
    *,
    threads: int | None = None,
    language: str | _WordTokenizer = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> list[int] | None:
    """MinHash signature of each document, identical to datatrove's.

    Each signature is ``num_buckets * hashes_per_bucket`` ints, bucket by
    bucket (datatrove's ``get_signature`` flattened); ``None`` for a document
    with fewer than ``n_grams`` words, which datatrove gives no signature.
    Arrow nulls give ``None`` too.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`MinhashConfig`, datatrove's ``MinhashConfig``, a
            dict of its fields, or ``None`` for the defaults.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        language: datatrove's ``MinhashDedupSignature(language=...)``: the
            language code whose word tokenizer splits the simplified text (see
            "Languages" above), or a word tokenizer object.
        python_tokenizer: Tokenize with datatrove's own tokenizer, in Python
            on the calling thread (needed for the languages pyroid does not
            tokenize natively; emits :class:`pyroid.FallbackWarning` once per
            function and language). ``simplify_text``, shingling and MinHash
            still run in Rust.
        **kwargs: ``MinhashConfig`` fields to override, plus ``hash_fc`` and
            ``precision``.

    Returns:
        One signature (str input) or a list with one entry per document.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: An unknown or mistyped config field, or a ``language``
            that is neither a str nor a word tokenizer.
        ValueError: A config value out of range, or a ``language`` datatrove
            has no tokenizer for (datatrove's message).
        NotImplementedError: pyroid does not tokenize ``language`` natively
            and ``python_tokenizer`` is not set.
        pyroid.PyroidError: The Python-tokenizer path needs datatrove or the
            tokenizer's package, and it is not installed.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def minhash_signatures(
    texts: list[str] | tuple[str, ...] | ArrowStrings,
    config: Any = None,
    *,
    threads: int | None = None,
    language: str | _WordTokenizer = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> list[list[int] | None]: ...
@overload
def minhash_dedup(
    texts: str,
    config: Any = None,
    *,
    threads: int | None = None,
    language: str | _WordTokenizer = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> bool:
    """Keep mask of datatrove's MinHash dedup over ``texts`` (one corpus).

    ``minhash_dedup(texts, cfg)`` is ``MinHashDedup(cfg).fit(texts)``: ``False``
    exactly for the documents datatrove's four stages remove.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: As for :func:`minhash_signatures`.
        threads: As for :func:`minhash_signatures`.
        language: As for :func:`minhash_signatures`.
        python_tokenizer: As for :func:`minhash_signatures`.
        **kwargs: As for :func:`minhash_signatures`.

    Returns:
        A mask in the same kind as ``texts``: bool, list, tuple, or a
        :class:`pyroid.Column` of bools (Arrow nulls stay null).

    Raises:
        As for :func:`minhash_signatures`.
    """

@overload
def minhash_dedup(
    texts: list[str],
    config: Any = None,
    *,
    threads: int | None = None,
    language: str | _WordTokenizer = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> list[bool]: ...
@overload
def minhash_dedup(
    texts: tuple[str, ...],
    config: Any = None,
    *,
    threads: int | None = None,
    language: str | _WordTokenizer = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> tuple[bool, ...]: ...
@overload
def minhash_dedup(
    texts: ArrowStrings,
    config: Any = None,
    *,
    threads: int | None = None,
    language: str | _WordTokenizer = "eng",
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Column[bool]: ...
@overload
def simplify_text(texts: str, config: Any = None, *, threads: int | None = None) -> str:
    """The normalisation MinHash shingles see: datatrove's ``simplify_text``.

    Byte-identical to ``datatrove.utils.text.simplify_text`` on the same
    CPython version (its ``unicodedata``) with the ``regex`` module's digit
    class. Unchanged documents are returned as the original objects.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`TextNormConfig`, datatrove's ``TextNormConfig``, a
            dict of its fields, or ``None`` for the defaults.
        threads: As for :func:`minhash_signatures`.

    Returns:
        The same kind as ``texts``: str, list, tuple, or a :class:`pyroid.Column`
        of strings.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def simplify_text(
    texts: list[str], config: Any = None, *, threads: int | None = None
) -> list[str]: ...
@overload
def simplify_text(
    texts: tuple[str, ...], config: Any = None, *, threads: int | None = None
) -> tuple[str, ...]: ...
@overload
def simplify_text(
    texts: ArrowStrings, config: Any = None, *, threads: int | None = None
) -> Column[str]: ...
@overload
def exact_dedup(texts: str, *, normalize: Any = False, threads: int | None = None) -> bool:
    """Keep mask of exact duplicates: ``True`` for the first copy of each text.

    Texts are compared by a 128-bit XXH3 hash (for 10⁹ documents the chance
    of any false match is about 10⁻²¹). Arrow nulls stay null and match
    nothing.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        normalize: ``False`` compares the raw text; ``True`` compares
            :func:`simplify_text` with the default :class:`TextNormConfig`;
            a ``TextNormConfig`` (pyroid's or datatrove's) or dict uses that.
        threads: As for :func:`minhash_signatures`.

    Returns:
        A mask in the same kind as ``texts``.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def exact_dedup(
    texts: list[str], *, normalize: Any = False, threads: int | None = None
) -> list[bool]: ...
@overload
def exact_dedup(
    texts: tuple[str, ...], *, normalize: Any = False, threads: int | None = None
) -> tuple[bool, ...]: ...
@overload
def exact_dedup(
    texts: ArrowStrings, *, normalize: Any = False, threads: int | None = None
) -> Column[bool]: ...

class MinHashDedup:
    """An in-memory MinHash index that makes datatrove's removals.

    Add documents in batches with :meth:`add` (only their signatures are
    kept, ``8 * num_buckets * hashes_per_bucket`` bytes each, in Rust), then
    read the decisions for everything added so far. Decisions are global:
    a later batch can change which document of a cluster is kept, so masks
    are only final once every batch is in. :meth:`fit` does one corpus in a
    single call. Methods may be called from several threads.

    Example::

        d = MinHashDedup(hash_fc="sha1")       # FineWeb's configuration
        for batch in batches:
            d.add(batch)
        keep = d.keep_mask()                   # one bool per document added

    Args:
        config: As for :func:`minhash_signatures`.
        threads: Default ``threads=`` for :meth:`add`, :meth:`fit` and the
            clustering.
        language: As for :func:`minhash_signatures`, for every document added.
        python_tokenizer: As for :func:`minhash_signatures` (the warning is
            emitted once, when the index is made).
        **kwargs: As for :func:`minhash_signatures`.
    """

    _state: Any
    def __init__(
        self,
        config: Any = None,
        *,
        threads: int | None = None,
        language: str | _WordTokenizer = "eng",
        python_tokenizer: bool = False,
        **kwargs: Any,
    ) -> None: ...
    @property
    def config(self) -> MinhashConfig:
        """The configuration (a :class:`MinhashConfig`)."""

    @property
    def language(self) -> str | _WordTokenizer:
        """The ``language`` the documents are tokenized for (a code or a tokenizer object)."""

    def add(
        self, texts: str | list[str] | tuple[str, ...] | ArrowStrings, *, threads: int | None = None
    ) -> range:
        """Add a batch of documents.

        Returns:
            The indices given to the batch's documents (Arrow nulls get one
            too, and are always kept).
        """

    @overload
    def fit(self, texts: str, *, threads: int | None = None) -> bool:
        """Replace the index with ``texts`` and return their keep mask.

        Returns:
            A mask in the same kind as ``texts`` (see :func:`minhash_dedup`).
        """

    @overload
    def fit(self, texts: list[str], *, threads: int | None = None) -> list[bool]: ...
    @overload
    def fit(self, texts: tuple[str, ...], *, threads: int | None = None) -> tuple[bool, ...]: ...
    @overload
    def fit(self, texts: ArrowStrings, *, threads: int | None = None) -> Column[bool]: ...
    def keep_mask(self) -> list[bool]:
        """``True`` for each document to keep, over every document added."""

    def duplicates(self) -> list[int]:
        """Indices of the documents to remove, ascending (datatrove's ``.remove`` file)."""

    def clusters(self) -> list[list[int]]:
        """Every cluster of two or more near-duplicates.

        Each cluster lists the kept document first, then the removed ones in
        ascending order. Clusters are ordered by their smallest member, so
        ``clusters()[k]`` is datatrove's cluster id ``k``.
        """

    def cluster_ids(self) -> list[int]:
        """Per-document ``minhash_cluster_id``, as datatrove writes it (``-1``: in no cluster)."""

    def cluster_sizes(self) -> list[int]:
        """Per-document ``minhash_cluster_size``, as datatrove writes it (``1``: no cluster)."""

    def representatives(self) -> list[int]:
        """For each document, the index of its cluster's kept document (itself if kept)."""

    def signature(self, index: int) -> list[int] | None:
        """The stored signature of document ``index`` (``None``: no shingles).

        Raises:
            IndexError: ``index`` is not a document added so far.
        """

    def reset(self) -> None:
        """Forget every document."""

    def __len__(self) -> int: ...

class ExactDedup:
    """Streaming exact dedup: keep the first copy of each text, across batches.

    Holds one 128-bit hash per distinct text (see :func:`exact_dedup`), so
    each batch's mask is final as soon as :meth:`add` returns.

    Args:
        normalize: As for :func:`exact_dedup`.
    """

    def __init__(self, *, normalize: Any = False) -> None: ...
    @overload
    def add(self, texts: str, *, threads: int | None = None) -> bool:
        """Keep mask for one more batch: ``True`` for each first copy so far.

        Returns:
            A mask in the same kind as ``texts`` (Arrow nulls stay null).
        """

    @overload
    def add(self, texts: list[str], *, threads: int | None = None) -> list[bool]: ...
    @overload
    def add(self, texts: tuple[str, ...], *, threads: int | None = None) -> tuple[bool, ...]: ...
    @overload
    def add(self, texts: ArrowStrings, *, threads: int | None = None) -> Column[bool]: ...
    def reset(self) -> None:
        """Forget every text seen."""

    def __len__(self) -> int:
        """Number of distinct texts seen."""

# ======================================================================================
# More datatrove 0.10.1 steps: one section each (each extends __all__).
# ======================================================================================

# ---- SentenceDedup ----

from typing import Generic, NamedTuple, TypeVar  # noqa: E402

__all__ += ["SentDedupConfig", "SentenceDedup", "SentenceDedupResult", "sentence_dedup"]

_SentT = TypeVar("_SentT")
_SentM = TypeVar("_SentM")

_native: Any

def _sent_config(config: object, kwargs: dict[str, Any]) -> SentDedupConfig: ...
def _native_sent(cfg: SentDedupConfig, language: str) -> Any: ...

@dataclass(frozen=True)
class SentDedupConfig:
    """datatrove's ``SentDedupConfig``, with the same fields and defaults.

    Attributes:
        n_sentences: Sentences per window (C4's three); at least 1.
        split_sentences: ``True``: the language tokenizer's sentences (datatrove's
            ``span_tokenize``, ported); ``False``: lines (``str.splitlines()``).
        only_dedup_in_index: With index files (the datatrove drop-ins only),
            remove only the windows found in the index. No effect in memory.
        min_doc_words: A document whose text changed is dropped if it has
            fewer words (``<= 0``: no minimum).
        min_num_sentences: A document whose text changed is dropped if it has
            fewer sentences (``<= 0``: no minimum).
        min_words_to_remove_span: Restore a removed run whose first sentence
            has fewer words, as datatrove does: only that first sentence comes
            back (``<= 0``: never).
        norm_config: The :class:`TextNormConfig` applied to each sentence
            before hashing.
        hash_config: The :class:`HashConfig` of the window hash.
    """

    n_sentences: int = 3
    split_sentences: bool = True
    only_dedup_in_index: bool = True
    min_doc_words: int = 50
    min_num_sentences: int = 3
    min_words_to_remove_span: int = 0
    norm_config: TextNormConfig = ...
    hash_config: HashConfig = ...

class SentenceDedupResult(NamedTuple, Generic[_SentT, _SentM]):
    """What :func:`sentence_dedup` and the :class:`SentenceDedup` methods return.

    Both fields are in the caller's container kind (str/bool, list, tuple, or a
    :class:`pyroid.Column`; Arrow nulls stay null).
    """

    text: _SentT
    """Each document's text with its duplicate windows removed (the original
    object when nothing was removed). Dropped documents get their text too."""
    keep: _SentM
    """``True`` for each document datatrove keeps."""
    def __class_getitem__(cls, params: Any) -> Any: ...  # Generic's, as at runtime

@overload
def sentence_dedup(
    texts: str,
    config: Any = None,
    *,
    language: str = "eng",
    threads: int | None = None,
    **kwargs: Any,
) -> SentenceDedupResult[str, bool]:
    """The three sentence-dedup stages of datatrove over one corpus.

    ``sentence_dedup(texts, cfg)`` is ``SentenceDedup(cfg).fit(texts)``: the
    texts and keep decisions of datatrove's ``SentenceDedupSignature`` ->
    ``SentenceFindDedups`` -> ``SentenceDedupFilter`` run as one task (see
    :class:`SentenceDedup`). The input is read once.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: A :class:`SentDedupConfig`, datatrove's ``SentDedupConfig``, a
            dict of its fields, or ``None`` for the defaults.
        language: datatrove's language code (as its ``language=``), for the sentence and
            word tokenizer: any code ``pyroid.filters.languages()`` marks native
            (spaCy rule-based languages, the whitespace and IndicNLP tokenizers).
            Codes whose datatrove tokenizer pyroid does not port (jieba, SudachiPy,
            Stanza, ...) raise ``NotImplementedError``; use the datatrove drop-ins.
        threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at most
            ``n`` tasks at once; ``1`` runs on the calling thread.
        **kwargs: ``SentDedupConfig`` fields to override, plus ``hash_fc`` and
            ``precision``.

    Returns:
        A :class:`SentenceDedupResult` ``(text, keep)``.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: An unknown or mistyped config field.
        ValueError: A config value out of range, ``language`` is not a datatrove
            code (datatrove's message), or a text of 2**30 characters or more.
        NotImplementedError: No native word tokenizer for ``language``.
        OverflowError: A document has more than 65,536 sentence windows
            (datatrove's ``<u2`` window index).
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def sentence_dedup(
    texts: list[str],
    config: Any = None,
    *,
    language: str = "eng",
    threads: int | None = None,
    **kwargs: Any,
) -> SentenceDedupResult[list[str], list[bool]]: ...
@overload
def sentence_dedup(
    texts: tuple[str, ...],
    config: Any = None,
    *,
    language: str = "eng",
    threads: int | None = None,
    **kwargs: Any,
) -> SentenceDedupResult[tuple[str, ...], tuple[bool, ...]]: ...
@overload
def sentence_dedup(
    texts: ArrowStrings,
    config: Any = None,
    *,
    language: str = "eng",
    threads: int | None = None,
    **kwargs: Any,
) -> SentenceDedupResult[Column[str], Column[bool]]: ...

class SentenceDedup:
    """datatrove's sentence-level dedup, in memory and multi-threaded.

    C4's rule ("we discarded all but one of any three-sentence span occurring
    more than once"), with **the same removals and keep decisions as datatrove
    0.10.1's** ``SentenceDedupSignature`` -> ``SentenceFindDedups`` ->
    ``SentenceDedupFilter`` run as one task:

    1. *Signatures* (:meth:`add`). A document's sentences (datatrove's
       ``sent_tokenize``: spaCy's sentences, stripped, empty ones dropped; or,
       with ``split_sentences=False``, its lines) each go through
       :func:`simplify_text`; every window of ``n_sentences`` consecutive
       ones, joined by a space, is hashed (``hash_fc``) unless it is blank.
       A document with fewer than ``n_sentences`` sentences has none.
    2. *Duplicates* (:meth:`duplicates`). For each hash seen more than once,
       every window but the first (lowest document, then window) is removed.
    3. *Filter* (:meth:`filter`). The ``n_sentences`` sentences of each
       removed window are cut out. Sentences are now spaCy's
       ``span_tokenize`` spans, each with the whitespace before it, which
       (unlike step 1) include a whitespace-only last sentence; the result is
       left-stripped (lines are rejoined with newlines). A document whose
       text changed is dropped if it has fewer than ``min_doc_words`` words
       or ``min_num_sentences`` sentences; an empty text is always dropped,
       even with nothing removed, as in datatrove.

    datatrove's quirks are kept: ``min_words_to_remove_span`` only counts,
    and only restores, the *first* sentence of a removed run.

    Documents are numbered in the order they are added, as datatrove numbers
    the documents of one task. Add every batch first, then filter them (pass
    ``start=`` for batches after the first); :meth:`fit` does one corpus in a
    single call. Only the signatures are kept (16 bytes per window, in Rust).
    Methods may be called from several threads.

    Example::

        d = SentenceDedup(min_doc_words=50)          # datatrove's example config
        ranges = [d.add(batch) for batch in batches]
        for batch, r in zip(batches, ranges):
            texts, keep = d.filter(batch, start=r.start)

    Known differences from datatrove: a str with a lone surrogate raises
    ``UnicodeEncodeError`` (so does datatrove, except with
    ``split_sentences=False`` on a document with fewer than ``n_sentences``
    lines); ``n_sentences`` below 1 raises
    ``ValueError``; a hashed window index past 65,535 raises datatrove's
    ``OverflowError`` (numpy's ``<u2``) when its batch is added, and the
    batch is not added (datatrove fails when it saves the task's
    signatures). Index files: use the datatrove drop-ins
    (:mod:`pyroid.integrations.datatrove`).

    Args:
        config: As for :func:`sentence_dedup`.
        language: As for :func:`sentence_dedup`; any code is accepted when no
            tokenizer is needed (``split_sentences=False`` and the three
            word/sentence thresholds ``<= 0``).
        threads: Default ``threads=`` for :meth:`add`, :meth:`filter` and
            :meth:`fit`.
        **kwargs: As for :func:`sentence_dedup`.
    """

    _state: Any
    def __init__(
        self,
        config: Any = None,
        *,
        language: str = "eng",
        threads: int | None = None,
        **kwargs: Any,
    ) -> None: ...
    @property
    def config(self) -> SentDedupConfig:
        """The configuration (a :class:`SentDedupConfig`)."""

    @property
    def language(self) -> str:
        """The datatrove language code of the word tokenizer."""

    def add(
        self, texts: str | list[str] | tuple[str, ...] | ArrowStrings, *, threads: int | None = None
    ) -> range:
        """Stage 1 for a batch of documents.

        Returns:
            The indices given to the batch's documents (Arrow nulls get one,
            and no signatures).

        Raises:
            OverflowError: A window index past 65,535 (nothing is added).
        """

    def duplicates(self) -> list[tuple[int, int]]:
        """Stage 2: the ``(document, window)`` pairs to remove, sorted.

        These are the records of datatrove's ``.c4_dup`` files for the task.
        """

    @overload
    def filter(
        self, texts: str, *, start: int = 0, threads: int | None = None
    ) -> SentenceDedupResult[str, bool]:
        """Stage 3 for documents ``start``, ``start + 1``, ... (as added).

        Pass the same documents, in the same order, as to :meth:`add`.
        Decisions are global, so filter once every batch has been added.

        Returns:
            A :class:`SentenceDedupResult` ``(text, keep)`` in the caller's
            kind.
        """

    @overload
    def filter(
        self, texts: list[str], *, start: int = 0, threads: int | None = None
    ) -> SentenceDedupResult[list[str], list[bool]]: ...
    @overload
    def filter(
        self, texts: tuple[str, ...], *, start: int = 0, threads: int | None = None
    ) -> SentenceDedupResult[tuple[str, ...], tuple[bool, ...]]: ...
    @overload
    def filter(
        self, texts: ArrowStrings, *, start: int = 0, threads: int | None = None
    ) -> SentenceDedupResult[Column[str], Column[bool]]: ...
    @overload
    def fit(self, texts: str, *, threads: int | None = None) -> SentenceDedupResult[str, bool]:
        """Replace the documents with ``texts`` and run the three stages on them.

        Returns:
            A :class:`SentenceDedupResult` ``(text, keep)`` (see
            :func:`sentence_dedup`).
        """

    @overload
    def fit(
        self, texts: list[str], *, threads: int | None = None
    ) -> SentenceDedupResult[list[str], list[bool]]: ...
    @overload
    def fit(
        self, texts: tuple[str, ...], *, threads: int | None = None
    ) -> SentenceDedupResult[tuple[str, ...], tuple[bool, ...]]: ...
    @overload
    def fit(
        self, texts: ArrowStrings, *, threads: int | None = None
    ) -> SentenceDedupResult[Column[str], Column[bool]]: ...
    def reset(self) -> None:
        """Forget every document."""

    def __len__(self) -> int:
        """Number of documents added."""

# ---- end SentenceDedup ----
