# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
"""MinHash LSH, as ``datasketch.lsh``, with the index in Rust. Types and docs are in ``lsh.pyi``.

The classes keep datasketch's attributes, checks and error messages. The
in-memory (``"dict"``) index lives in ``_LSHNative``; ``keys`` and
``hashtables`` are read-only views of it. Redis or Cassandra storage, and an
LSH ``hashfunc``, are handled by datasketch itself: ``MinHashLSH(...)`` then
returns a ``datasketch.MinHashLSH`` (with a ``pyroid.FallbackWarning``).
"""

from __future__ import annotations

import operator
import pickle
import struct
import sys
from collections.abc import Callable, Hashable, Iterable
from types import TracebackType
from typing import Any, Self

from pyroid._pyroid import datasketch as _native
from pyroid.datasketch import _common
from pyroid.datasketch._common import _check_scheme_consistency
from pyroid.datasketch.storage import (
    DictListStorageView,
    DictSetStorageView,
    OrderedStorage,
    UnorderedStorage,
    _random_name,
    unordered_storage,
)

# Pickles of the native index name it by this module.
_LSHNative = _native._LSHNative

__all__ = ["MinHashLSH", "MinHashLSHDeletionSession", "MinHashLSHInsertionSession"]


def _false_positive_probability(threshold: float, b: int, r: int) -> float:
    """The probability that a set below ``threshold`` shares a band (integrated)."""
    return _native.lsh_false_positive(threshold, b, r)  # type: ignore[no-any-return]


def _false_negative_probability(threshold: float, b: int, r: int) -> float:
    """The probability that a set above ``threshold`` shares no band (integrated)."""
    return _native.lsh_false_negative(threshold, b, r)  # type: ignore[no-any-return]


def _optimal_param(
    threshold: float, num_perm: int, false_positive_weight: float, false_negative_weight: float
) -> tuple[int, int]:
    """The ``(b, r)`` that minimizes the weighted false positive and negative probabilities."""
    # datasketch loops over range(1, num_perm + 1): the same TypeError for a
    # non-integer, and (0, 0) when the range is empty.
    if operator.index(num_perm) < 1:
        return (0, 0)
    return _native.lsh_optimal_param(  # type: ignore[no-any-return]
        threshold, num_perm, false_positive_weight, false_negative_weight
    )


def _delegation_reason(storage_config: Any, hashfunc: Any) -> str | None:
    """Why datasketch must build this index, or None. Never raises."""
    config = storage_config if storage_config else {"type": "dict"}
    try:
        tp = config["type"]
    except Exception:
        return None  # __init__ raises datasketch's error at datasketch's point
    if not isinstance(tp, str):
        return None
    if tp in ("redis", "cassandra"):
        return f"storage_config type {tp!r}"
    if tp == "dict" and hashfunc:
        return "hashfunc"
    return None


def _band_size(r: Any) -> int | None:
    """``r`` as a non-negative int for the native band keys, or None (Python slices)."""
    try:
        r = operator.index(r)
    except TypeError:
        return None
    return r if r >= 0 else None


def _ndarray(x: Any) -> bool:
    np = sys.modules.get("numpy")
    return np is not None and isinstance(x, np.ndarray)


def _rows(values: Any, h: Any) -> Any:
    """A batch of signatures (axis 0) as the C-contiguous, native-order array the kernel reads."""
    np = sys.modules["numpy"]
    if values.ndim < 2:
        raise ValueError(
            f"expected a 2-D array of signatures (one per row), got {values.ndim} dimension(s)"
        )
    if values.shape[0] and values.shape[1] != h:
        raise _length_error(h, values.shape[1])
    values = np.ascontiguousarray(values)
    # The band keys depend on the raw bytes only (byteswap()), so a non-native
    # array is read as the native dtype of the same bytes.
    if not values.dtype.isnative:
        values = values.view(values.dtype.newbyteorder("="))
    return values


def _length_error(h: Any, n: Any) -> ValueError:
    # %d as datasketch (it also formats a float num_perm as an int).
    return ValueError("Expecting minhash with length %d, got %d" % (h, n))  # noqa: UP031


def _same_length(keys: list[Any], n: int) -> None:
    if len(keys) != n:
        raise ValueError(f"keys and minhashes differ in length ({len(keys)} and {n})")


class MinHashLSH:
    """The MinHash LSH index, for queries with a Jaccard similarity threshold.

    Args:
        threshold: The Jaccard similarity threshold, in ``[0.0, 1.0]``. The
            band layout is chosen to minimize the weighted false positive and
            false negative probabilities at this threshold.
        num_perm: The number of permutations of the MinHashes to index (the
            ``sample_size`` of a weighted MinHash).
        weights: ``(false_positive_weight, false_negative_weight)``; they sum
            to 1.0.
        params: ``(b, r)``, the number of bands and rows per band. Skips the
            search; ``threshold`` and ``weights`` are then ignored.
        storage_config: ``{"type": "dict"}`` (the default) is the in-memory
            index. ``"redis"`` and ``"cassandra"`` are delegated to
            datasketch. ``"basename"`` names the storage.
        prepickle: Pickle keys to bytes before storing them. Default: true
            for Redis storage only.
        hashfunc: Hash function applied to band keys. Delegated to
            datasketch.

    Raises:
        ValueError: For the same invalid arguments as datasketch, with its
            messages (``"Unknown storage type: ..."`` included).

    Attributes:
        h: ``num_perm``.
        b: The number of bands (band tables).
        r: The number of rows per band.
        hashranges: ``[(i * r, (i + 1) * r) for i in range(b)]``.
        prepickle: Whether keys are pickled before they are stored.
        hashfunc: ``None`` (an index with a hash function is datasketch's).
        hashtables: One view per band: band key -> set of keys.
        keys: View: key -> its band keys.
    """

    def __new__(
        cls,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        params: tuple[int, int] | None = None,
        storage_config: dict[str, Any] | None = None,
        prepickle: bool | None = None,
        hashfunc: Callable[[bytes], bytes] | None = None,
    ) -> Self:
        """Create the index, or datasketch's for storage or a hashfunc pyroid does not port."""
        reason = _delegation_reason(storage_config, hashfunc)
        if reason is not None:
            real = _common.fallback("MinHashLSH", reason)
            return real.MinHashLSH(  # type: ignore[no-any-return]
                threshold=threshold,
                num_perm=num_perm,
                weights=weights,
                params=params,
                storage_config=storage_config,
                prepickle=prepickle,
                hashfunc=hashfunc,
            )
        return super().__new__(cls)

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        params: tuple[int, int] | None = None,
        storage_config: dict[str, Any] | None = None,
        prepickle: bool | None = None,
        hashfunc: Callable[[bytes], bytes] | None = None,
    ) -> None:
        storage_config = storage_config if storage_config else {"type": "dict"}
        self._buffer_size = 50000
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is not None:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError(
                    "The product of b and r in params is "
                    f"{self.b} * {self.r} = {self.b * self.r} -- "
                    f"it must be less than num_perm {num_perm}. "
                    "Did you forget to specify num_perm?"
                )
        else:
            false_positive_weight, false_negative_weight = weights
            self.b, self.r = _optimal_param(
                threshold, num_perm, false_positive_weight, false_negative_weight
            )
        if self.b < 2:
            raise ValueError("The number of bands are too small (b < 2)")

        self.prepickle = storage_config["type"] == "redis" if prepickle is None else prepickle
        self._require_bytes_keys = not (storage_config["type"] == "dict" or self.prepickle)

        self.hashfunc = hashfunc
        if hashfunc:
            self._H = self._hashed_byteswap
        else:
            self._H = self._byteswap

        # datasketch draws the name even when "basename" is given (and with
        # dict storage, which ignores it): the draw advances `random`.
        basename = storage_config.get("basename", _random_name(11))
        tp = storage_config["type"]
        for i in range(self.b):
            # The storage names are built as datasketch builds them, for its errors.
            b"".join([basename, b"_bucket_", struct.pack(">H", i)])
            if tp != "dict":
                raise ValueError(f"Unknown storage type: {tp}")
        self.hashranges = [(i * self.r, (i + 1) * self.r) for i in range(self.b)]
        self._index = _LSHNative(len(self.hashranges), _band_size(self.r))
        self.hashtables: list[UnorderedStorage] = [
            DictSetStorageView(self._index, i) for i in range(len(self.hashranges))
        ]
        self.keys: OrderedStorage = DictListStorageView(self._index)
        # The permutation scheme of the indexed MinHash, learned from the
        # first insert.
        self._minhash_scheme: str | None = None

    @property
    def buffer_size(self) -> int:
        """The storage buffer size (used by Redis and Cassandra storage)."""
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self.keys.buffer_size = value
        for t in self.hashtables:
            t.buffer_size = value
        self._buffer_size = value

    def insert(self, key: Hashable, minhash: Any, check_duplication: bool = True) -> None:
        """Insert ``key`` with the MinHash (or weighted MinHash) of its set.

        Args:
            key: The key of the set (any hashable; keys compare with ``==``).
            minhash: Its MinHash, of length ``num_perm``.
            check_duplication: Raise if ``key`` is already in the index. With
                ``False``, inserting a key again adds the new band keys to its
                entry, as datasketch does.

        Raises:
            ValueError: The MinHash has the wrong length or another
                permutation scheme than earlier ones, or ``key`` exists.
        """
        self._insert(key, minhash, check_duplication=check_duplication, buffer=False)

    def insert_batch(
        self,
        keys: Iterable[Hashable],
        minhashes: Any,
        *,
        check_duplication: bool = True,
        threads: int | None = None,
    ) -> None:
        """Insert ``keys[i]`` with ``minhashes[i]`` for every ``i`` (pyroid extension).

        The index ends up as after ``insert`` of each item in order, but
        nothing is inserted if an item is invalid: the error is the one
        ``insert`` would raise for the first invalid item, with a note naming
        it. Band tables are filled in parallel.

        Args:
            keys: The keys.
            minhashes: MinHash-like objects, or a numpy array whose rows
                (axis 0) are ``hashvalues`` arrays, e.g. ``(n, num_perm)``
                ``uint64`` or ``uint32``. Array rows have no ``scheme``, so
                they are not checked against it (as weighted MinHashes).
            check_duplication: As for :meth:`insert`; a key repeated within
                the batch is a duplicate too.
            threads: At most this many tasks at once on pyroid's thread pool
                (``None``: the whole pool, ``1``: the calling thread).

        Raises:
            ValueError: ``keys`` and ``minhashes`` differ in length, or as
                :meth:`insert`.
            TypeError: An array of another dtype than integers or floats.
        """
        keys = list(keys)
        if _ndarray(minhashes):
            values = _rows(minhashes, self.h)
            _same_length(keys, values.shape[0])
            if self._require_bytes_keys:
                for key in keys:
                    self._check_bytes_key(key)
            prepare = pickle.dumps if self.prepickle else None
            self._index.insert_array(keys, values, prepare, bool(check_duplication), threads)
            return
        minhashes = list(minhashes)
        _same_length(keys, len(minhashes))
        known = getattr(self, "_minhash_scheme", None)

        def prepare_item(key: Any, minhash: Any) -> Any:
            nonlocal known
            if len(minhash) != self.h:
                raise _length_error(self.h, len(minhash))
            known = _check_scheme_consistency(known, minhash)
            self._check_bytes_key(key)
            return pickle.dumps(key) if self.prepickle else key

        self._index.insert_many(
            keys, minhashes, prepare_item, self._band_keys, bool(check_duplication), threads
        )
        self._minhash_scheme = known

    def merge(self, other: MinHashLSH, check_overlap: bool = False) -> None:
        """Merge ``other`` into this index, making this one the union of both.

        Only ``num_perm``, ``b`` and ``r`` are compared, as in datasketch.

        Args:
            other: Another pyroid ``MinHashLSH``.
            check_overlap: Raise if a key is in both indexes (nothing is
                merged then).

        Raises:
            ValueError: The indexes differ in type, parameters or MinHash
                scheme, or ``check_overlap`` found a common key.
        """
        self._merge(other, check_overlap=check_overlap, buffer=False)

    def insertion_session(self, buffer_size: int = 50000) -> MinHashLSHInsertionSession:
        """Return a context manager whose ``insert`` inserts into this index."""
        return MinHashLSHInsertionSession(self, buffer_size=buffer_size)

    def deletion_session(self, buffer_size: int = 50000) -> MinHashLSHDeletionSession:
        """Return a context manager whose ``remove`` removes from this index."""
        return MinHashLSHDeletionSession(self, buffer_size=buffer_size)

    def _check_bytes_key(self, key: Any) -> None:
        if self._require_bytes_keys and not isinstance(key, bytes):
            raise TypeError(
                "prepickle=False requires bytes keys for non-dict storage, "
                f"got {type(key).__name__}. "
                "Either pass bytes keys or use prepickle=True for automatic serialization."
            )

    def _band_keys(self, minhash: Any, n: int) -> list[bytes]:
        # datasketch's expression, for hashvalues the kernel does not read.
        return [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges[:n]]

    def _insert(
        self,
        key: Hashable,
        minhash: Any,
        check_duplication: bool = True,
        buffer: bool = False,
    ) -> None:
        if len(minhash) != self.h:
            raise _length_error(self.h, len(minhash))
        self._minhash_scheme = _check_scheme_consistency(
            getattr(self, "_minhash_scheme", None), minhash
        )
        self._check_bytes_key(key)
        if self.prepickle:
            key = pickle.dumps(key)
        self._index.insert(key, minhash, bool(check_duplication), self._band_keys)

    def __equivalent(self, other: MinHashLSH) -> bool:
        return (
            type(self) is type(other)
            and self.h == other.h
            and self.b == other.b
            and self.r == other.r
        )

    def _merge(self, other: MinHashLSH, check_overlap: bool = False, buffer: bool = False) -> None:
        if self.__equivalent(other):
            known, other_known = (
                getattr(self, "_minhash_scheme", None),
                getattr(other, "_minhash_scheme", None),
            )
            if known is not None and other_known is not None and known != other_known:
                raise ValueError(
                    f"Cannot merge MinHashLSH indexed with MinHash scheme {other_known!r} "
                    f"into one indexed with scheme {known!r}"
                )
            self._index.merge(other._index, bool(check_overlap))
            if known is None:
                self._minhash_scheme = other_known
        else:
            if type(self) is not type(other):
                raise ValueError(f"Cannot merge type MinHashLSH and type {type(other).__name__}.")
            raise ValueError("Cannot merge MinHashLSH with different initialization parameters.")

    def query(self, minhash: Any) -> list[Hashable]:
        """Return the keys whose sets likely have a Jaccard similarity above the threshold.

        Args:
            minhash: The MinHash of the query set.

        Returns:
            The keys that share a band with ``minhash``, each once, in no
            particular order.

        Raises:
            ValueError: The MinHash has the wrong length or another scheme.
        """
        if len(minhash) != self.h:
            raise _length_error(self.h, len(minhash))
        _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        candidates = self._index.query(minhash, len(self.hashranges), self._band_keys)
        if self.prepickle:
            # prepickle=True: the stored keys are pickle.dumps() of the caller's keys,
            # made by this index (datasketch's behaviour).
            return [pickle.loads(key) for key in candidates]
        return candidates  # type: ignore[no-any-return]

    def query_batch(self, minhashes: Any, *, threads: int | None = None) -> list[list[Hashable]]:
        """Return ``[self.query(m) for m in minhashes]``, in parallel (pyroid extension).

        Args:
            minhashes: MinHash-like objects, or a numpy array of signatures
                (rows), as for :meth:`insert_batch`.
            threads: As for :meth:`insert_batch`.
        """
        if _ndarray(minhashes):
            results = self._index.query_array(_rows(minhashes, self.h), threads)
        else:
            known = getattr(self, "_minhash_scheme", None)

            def prepare_item(minhash: Any) -> None:
                if len(minhash) != self.h:
                    raise _length_error(self.h, len(minhash))
                _check_scheme_consistency(known, minhash)

            results = self._index.query_many(
                list(minhashes), prepare_item, self._band_keys, threads
            )
        if self.prepickle:
            return [[pickle.loads(key) for key in keys] for keys in results]
        return results  # type: ignore[no-any-return]

    def add_to_query_buffer(self, minhash: Any) -> None:
        """Buffer a query for :meth:`collect_query_buffer`.

        The buckets are read now (a snapshot), not when the buffer is collected.
        """
        if len(minhash) != self.h:
            raise _length_error(self.h, len(minhash))
        _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        hs = self._index.band_keys(minhash, len(self.hashranges), self._band_keys)
        for h, hashtable in zip(hs, self.hashtables, strict=False):
            hashtable.add_to_select_buffer([h])

    def collect_query_buffer(self) -> list[Hashable]:
        """Run the buffered queries; return the keys every one of them found."""
        collected_result_lists = [
            hashtable.collect_select_buffer() for hashtable in self.hashtables
        ]
        if not any(collected_result_lists):
            return []

        # Each buffered query contributes one result list per hashtable. We first
        # union candidates across bands for each query, then intersect across the
        # buffered queries to match repeated calls to `query()`.
        per_query_result_sets = [
            set().union(*query_result_lists)
            for query_result_lists in zip(*collected_result_lists, strict=False)
        ]
        if not per_query_result_sets:
            return []

        candidates = set.intersection(*per_query_result_sets)
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        return list(candidates)

    def __contains__(self, key: Hashable) -> bool:
        if self.prepickle:
            key = pickle.dumps(key)
        return self._index.contains(key)  # type: ignore[no-any-return]

    def remove(self, key: Hashable) -> None:
        """Remove ``key`` from the index.

        Raises:
            ValueError: ``key`` is not in the index.
        """
        self._remove(key, buffer=False)

    def _remove(self, key: Hashable, buffer: bool = False) -> None:
        if self.prepickle:
            key = pickle.dumps(key)
        self._index.remove(key)

    def is_empty(self) -> bool:
        """Return True if some band table is empty (datasketch's definition)."""
        return self._index.is_empty()  # type: ignore[no-any-return]

    def _byteswap(self, hs: Any) -> bytes:
        return bytes(hs.byteswap().data)

    def _hashed_byteswap(self, hs: Any) -> bytes:
        if self.hashfunc is None:
            raise RuntimeError("Hash function not configured.")
        return self.hashfunc(bytes(hs.byteswap().data))

    def _query_b(self, minhash: Any, b: int) -> set[Hashable]:
        if len(minhash) != self.h:
            raise _length_error(self.h, len(minhash))
        _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        if b > len(self.hashtables):
            raise ValueError("b must be less or equal to the number of hash tables")
        n = min(len(self.hashranges[:b]), len(self.hashtables[:b]))
        candidates = self._index.query(minhash, n, self._band_keys)
        if self.prepickle:
            return {pickle.loads(key) for key in candidates}
        return set(candidates)

    def get_counts(self) -> list[dict[Hashable, int]]:
        """Return, per band table, ``{band key: number of keys in the bucket}``."""
        return [hashtable.itemcounts() for hashtable in self.hashtables]

    def get_subset_counts(self, *keys: Hashable) -> list[dict[Hashable, int]]:
        """Return :meth:`get_counts` restricted to ``keys``."""
        key_set = [pickle.dumps(key) for key in set(keys)] if self.prepickle else list(set(keys))
        hashtables = [unordered_storage({"type": "dict"}) for _ in range(self.b)]
        hss = self.keys.getmany(*key_set)
        for key, hs in zip(key_set, hss, strict=False):
            for h, hashtable in zip(hs, hashtables, strict=False):
                hashtable.insert(h, key)
        return [hashtable.itemcounts() for hashtable in hashtables]


class MinHashLSHInsertionSession:
    """Context manager for inserting many keys into a :class:`MinHashLSH`."""

    def __init__(self, lsh: MinHashLSH, buffer_size: int) -> None:
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Flush the buffers (nothing to do for the in-memory index)."""
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def insert(self, key: Hashable, minhash: Any, check_duplication: bool = True) -> None:
        """Insert ``key`` with its MinHash, as :meth:`MinHashLSH.insert`."""
        self.lsh._insert(key, minhash, check_duplication=check_duplication, buffer=True)


class MinHashLSHDeletionSession:
    """Context manager for removing many keys from a :class:`MinHashLSH`."""

    def __init__(self, lsh: MinHashLSH, buffer_size: int) -> None:
        self.lsh = lsh
        self.lsh.buffer_size = buffer_size

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Flush the buffers (nothing to do for the in-memory index)."""
        self.lsh.keys.empty_buffer()
        for hashtable in self.lsh.hashtables:
            hashtable.empty_buffer()

    def remove(self, key: Hashable) -> None:
        """Remove ``key``, as :meth:`MinHashLSH.remove`."""
        self.lsh._remove(key, buffer=True)
