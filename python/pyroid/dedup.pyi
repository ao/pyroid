# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
# Type stubs for pyroid.dedup. The module docstring is kept here on purpose:
# the API docs are built from this stub.
"""MinHash near-duplicate and exact deduplication (``pyroid.dedup``).

The MinHash pipeline makes **the same removals as datatrove 0.10.1's**
``MinhashDedupSignature`` -> ``MinhashDedupBuckets`` -> ``MinhashDedupCluster``
-> ``MinhashDedupFilter``, in memory and multi-threaded:

1. *Signatures.* Each document is normalised with datatrove's
   ``simplify_text``, split into words by pyroid's port of spaCy's English
   tokenizer (datatrove's ``word_tokenize``), and every ``n_grams``-word window is
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

Known differences from datatrove
--------------------------------
- A str with a lone surrogate raises ``UnicodeEncodeError`` (pyroid's policy);
  datatrove raises it too unless the document has fewer than ``n_grams``
  words.
- ``n_grams``, ``num_buckets`` or ``hashes_per_bucket`` below 1 raise
  ``ValueError`` (datatrove silently produces no signatures or fails later).
- Index files (``index_folder``, ``create_index_name``,
  ``only_dedup_in_index``) are not supported.
"""

from dataclasses import dataclass
from typing import Any, Final, overload

from pyroid._pyroid import Column
from pyroid._typing import ArrowStrings

__all__ = [
    "DATATROVE_VERSION",
    "ExactDedup",
    "HashConfig",
    "MinHashDedup",
    "MinhashConfig",
    "TextNormConfig",
    "exact_dedup",
    "minhash_dedup",
    "minhash_signatures",
    "simplify_text",
]

DATATROVE_VERSION: Final[str]
_UNICODE_VERSION: Final[str]

def _native_config(cfg: MinhashConfig) -> Any: ...

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
    texts: str, config: Any = None, *, threads: int | None = None, **kwargs: Any
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
        **kwargs: ``MinhashConfig`` fields to override, plus ``hash_fc`` and
            ``precision``.

    Returns:
        One signature (str input) or a list with one entry per document.

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: An unknown or mistyped config field.
        ValueError: A config value out of range.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def minhash_signatures(
    texts: list[str] | tuple[str, ...] | ArrowStrings,
    config: Any = None,
    *,
    threads: int | None = None,
    **kwargs: Any,
) -> list[list[int] | None]: ...
@overload
def minhash_dedup(
    texts: str, config: Any = None, *, threads: int | None = None, **kwargs: Any
) -> bool:
    """Keep mask of datatrove's MinHash dedup over ``texts`` (one corpus).

    ``minhash_dedup(texts, cfg)`` is ``MinHashDedup(cfg).fit(texts)``: ``False``
    exactly for the documents datatrove's four stages remove.

    Args:
        texts: A str, a list or tuple of str, or an Arrow string array/column.
        config: As for :func:`minhash_signatures`.
        threads: As for :func:`minhash_signatures`.
        **kwargs: As for :func:`minhash_signatures`.

    Returns:
        A mask in the same kind as ``texts``: bool, list, tuple, or a
        :class:`pyroid.Column` of bools (Arrow nulls stay null).

    Raises:
        pyroid.InputError: ``texts`` is not one of the accepted kinds.
        TypeError: An unknown or mistyped config field.
        ValueError: A config value out of range.
        UnicodeEncodeError: A str contains a lone surrogate.
        UnicodeDecodeError: An Arrow row is not valid UTF-8.
    """

@overload
def minhash_dedup(
    texts: list[str], config: Any = None, *, threads: int | None = None, **kwargs: Any
) -> list[bool]: ...
@overload
def minhash_dedup(
    texts: tuple[str, ...], config: Any = None, *, threads: int | None = None, **kwargs: Any
) -> tuple[bool, ...]: ...
@overload
def minhash_dedup(
    texts: ArrowStrings, config: Any = None, *, threads: int | None = None, **kwargs: Any
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
        **kwargs: As for :func:`minhash_signatures`.
    """

    _state: Any
    def __init__(
        self, config: Any = None, *, threads: int | None = None, **kwargs: Any
    ) -> None: ...
    @property
    def config(self) -> MinhashConfig:
        """The configuration (a :class:`MinhashConfig`)."""

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
