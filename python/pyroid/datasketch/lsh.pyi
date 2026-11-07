# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.lsh. The module docstring is kept here on purpose:
# the API docs are built from this stub.
"""MinHash LSH (``pyroid.datasketch.lsh``), as datasketch 2.0.0's ``datasketch.lsh``.

``MinHashLSH`` has datasketch's constructor, methods, attributes and error
messages, and returns the same query results (as sets: result lists are in
no particular order, as in datasketch). The in-memory index is in Rust:

```python
from pyroid.datasketch.lsh import MinHashLSH

lsh = MinHashLSH(threshold=0.5, num_perm=128)
for key, m in minhashes.items():
    lsh.insert(key, m)
lsh.query(minhashes["doc-1"])
lsh.insert_batch(keys, list_of_minhashes, threads=8)  # pyroid extension
lsh.query_batch(signatures_2d_array)  # pyroid extension
```

The index reads ``hashvalues``, ``len()`` and ``scheme`` of the sketches it is
given, so datasketch's ``MinHash``, ``LeanMinHash`` and ``WeightedMinHash``
work as well as pyroid's. Band keys are computed in Rust when ``hashvalues``
is a C-contiguous numpy array of an integer or float dtype in native byte
order; anything else runs datasketch's own expression
``bytes(hashvalues[start:end].byteswap().data)``.

Differences from datasketch:

- ``storage_config`` of type ``"redis"`` or ``"cassandra"``, and a
  ``hashfunc``, are handled by datasketch: ``MinHashLSH(...)`` returns a
  ``datasketch.MinHashLSH`` and emits a ``pyroid.FallbackWarning`` (once).
- ``keys`` and ``hashtables[i]`` are read-only views
  (:class:`~pyroid.datasketch.storage.DictListStorageView`,
  :class:`~pyroid.datasketch.storage.DictSetStorageView`). Their values are
  new lists and sets on every access (a snapshot), so a query buffered with
  :meth:`MinHashLSH.add_to_query_buffer` sees the index as it was then. The
  mutating storage methods raise ``NotImplementedError``. Band tables iterate
  in no particular order (``keys`` iterates in insertion order, as a dict).
- When several equal keys of different types are involved (``1``, ``1.0``,
  ``True``), the object returned may be another one of the equal keys than
  datasketch returns.

The ``(b, r)`` search for ``threshold`` and ``weights`` is a Rust port of
datasketch's (``scipy.integrate.quad``) and picks the same parameters; it
needs neither datasketch nor scipy.
"""

from collections.abc import Callable, Hashable, Iterable, Sequence
from types import TracebackType
from typing import Any, Protocol, Self

from pyroid.datasketch.storage import OrderedStorage, UnorderedStorage

__all__ = ["MinHashLSH", "MinHashLSHDeletionSession", "MinHashLSHInsertionSession"]

def _false_positive_probability(threshold: float, b: int, r: int) -> float: ...
def _false_negative_probability(threshold: float, b: int, r: int) -> float: ...
def _optimal_param(
    threshold: float, num_perm: int, false_positive_weight: float, false_negative_weight: float
) -> tuple[int, int]: ...

class _Sketch(Protocol):
    """What the index reads from a MinHash: ``hashvalues`` and ``len()``."""

    @property
    def hashvalues(self) -> Any: ...
    def __len__(self) -> int: ...

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

    h: int
    b: int
    r: int
    hashranges: list[tuple[int, int]]
    prepickle: bool
    hashfunc: Callable[[bytes], bytes] | None
    hashtables: list[UnorderedStorage]
    keys: OrderedStorage
    _buffer_size: int
    _require_bytes_keys: bool
    _minhash_scheme: str | None
    _H: Callable[[Any], bytes]
    _index: Any

    def __new__(
        cls,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        params: tuple[int, int] | None = None,
        storage_config: dict[str, Any] | None = None,
        prepickle: bool | None = None,
        hashfunc: Callable[[bytes], bytes] | None = None,
    ) -> Self: ...
    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        params: tuple[int, int] | None = None,
        storage_config: dict[str, Any] | None = None,
        prepickle: bool | None = None,
        hashfunc: Callable[[bytes], bytes] | None = None,
    ) -> None: ...
    @property
    def buffer_size(self) -> int:
        """The storage buffer size (used by Redis and Cassandra storage)."""
    @buffer_size.setter
    def buffer_size(self, value: int) -> None: ...
    def insert(self, key: Hashable, minhash: _Sketch, check_duplication: bool = True) -> None:
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
    def insert_batch(
        self,
        keys: Iterable[Hashable],
        minhashes: Sequence[_Sketch] | Any,
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
    def insertion_session(self, buffer_size: int = 50000) -> MinHashLSHInsertionSession:
        """Return a context manager whose ``insert`` inserts into this index."""
    def deletion_session(self, buffer_size: int = 50000) -> MinHashLSHDeletionSession:
        """Return a context manager whose ``remove`` removes from this index."""
    def query(self, minhash: _Sketch) -> list[Hashable]:
        """Return the keys whose sets likely have a Jaccard similarity above the threshold.

        Args:
            minhash: The MinHash of the query set.

        Returns:
            The keys that share a band with ``minhash``, each once, in no
            particular order.

        Raises:
            ValueError: The MinHash has the wrong length or another scheme.
        """
    def query_batch(
        self, minhashes: Sequence[_Sketch] | Any, *, threads: int | None = None
    ) -> list[list[Hashable]]:
        """Return ``[self.query(m) for m in minhashes]``, in parallel (pyroid extension).

        Args:
            minhashes: MinHash-like objects, or a numpy array of signatures
                (rows), as for :meth:`insert_batch`.
            threads: As for :meth:`insert_batch`.
        """
    def add_to_query_buffer(self, minhash: _Sketch) -> None:
        """Buffer a query for :meth:`collect_query_buffer`.

        The buckets are read now (a snapshot), not when the buffer is collected.
        """
    def collect_query_buffer(self) -> list[Hashable]:
        """Run the buffered queries; return the keys every one of them found."""
    def _insert(
        self,
        key: Hashable,
        minhash: _Sketch,
        check_duplication: bool = True,
        buffer: bool = False,
    ) -> None: ...
    def _merge(
        self, other: MinHashLSH, check_overlap: bool = False, buffer: bool = False
    ) -> None: ...
    def _remove(self, key: Hashable, buffer: bool = False) -> None: ...
    def _query_b(self, minhash: _Sketch, b: int) -> set[Hashable]: ...
    def _byteswap(self, hs: Any) -> bytes: ...
    def _hashed_byteswap(self, hs: Any) -> bytes: ...
    def __contains__(self, key: Hashable) -> bool: ...
    def remove(self, key: Hashable) -> None:
        """Remove ``key`` from the index.

        Raises:
            ValueError: ``key`` is not in the index.
        """
    def is_empty(self) -> bool:
        """Return True if some band table is empty (datasketch's definition)."""
    def get_counts(self) -> list[dict[Hashable, int]]:
        """Return, per band table, ``{band key: number of keys in the bucket}``."""
    def get_subset_counts(self, *keys: Hashable) -> list[dict[Hashable, int]]:
        """Return :meth:`get_counts` restricted to ``keys``."""

class MinHashLSHInsertionSession:
    """Context manager for inserting many keys into a :class:`MinHashLSH`."""

    lsh: MinHashLSH
    def __init__(self, lsh: MinHashLSH, buffer_size: int) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def close(self) -> None:
        """Flush the buffers (nothing to do for the in-memory index)."""
    def insert(self, key: Hashable, minhash: _Sketch, check_duplication: bool = True) -> None:
        """Insert ``key`` with its MinHash, as :meth:`MinHashLSH.insert`."""

class MinHashLSHDeletionSession:
    """Context manager for removing many keys from a :class:`MinHashLSH`."""

    lsh: MinHashLSH
    def __init__(self, lsh: MinHashLSH, buffer_size: int) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...
    def close(self) -> None:
        """Flush the buffers (nothing to do for the in-memory index)."""
    def remove(self, key: Hashable) -> None:
        """Remove ``key``, as :meth:`MinHashLSH.remove`."""
