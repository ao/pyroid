# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
"""MinHash LSH Forest, as ``datasketch.lshforest``, with the index in Rust.

Types and docs are in ``lshforest.pyi``. ``keys``, ``hashtables`` and
``sorted_hashtables`` are views of the native index with the API of the dict,
``defaultdict(list)`` and list datasketch keeps there.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import (
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from typing import Any, overload

from pyroid._pyroid import datasketch as _native
from pyroid.datasketch._common import _check_scheme_consistency

# Pickles of the native index name it by this module.
_ForestNative = _native._ForestNative

__all__ = ["MinHashLSHForest"]


def _band_key_list(value: Iterable[Any]) -> list[bytes]:
    out = []
    for h in value:
        if not isinstance(h, (bytes, bytearray, memoryview)):
            raise TypeError(
                "pyroid.datasketch: MinHashLSHForest.keys values must be lists of bytes, "
                f"got an item of type {type(h).__name__}"
            )
        out.append(bytes(h))
    return out


class _ForestKeys(MutableMapping[Hashable, list[bytes]]):
    """``forest.keys``: key -> its band keys, in insertion order (a dict's API)."""

    def __init__(self, index: Any) -> None:
        self._index = index

    def __getitem__(self, key: Hashable) -> list[bytes]:
        hs = self._index.keys_get(key)
        if hs is None:
            raise KeyError(key)
        return hs  # type: ignore[no-any-return]

    def __setitem__(self, key: Hashable, value: Iterable[bytes]) -> None:
        self._index.keys_set(key, _band_key_list(value))

    def __delitem__(self, key: Hashable) -> None:
        if not self._index.keys_del(key):
            raise KeyError(key)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self._index.keys_list())

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        return self._index.contains(key)  # type: ignore[no-any-return]

    def get(self, key: Hashable, default: Any = None) -> Any:
        hs = self._index.keys_get(key)
        return default if hs is None else hs

    def __repr__(self) -> str:
        return repr(dict(self.items()))


class _ForestTable(Mapping[bytes, list[Hashable]]):
    """``forest.hashtables[i]``: band key -> the keys added with it, in order."""

    def __init__(self, index: Any, table: int) -> None:
        self._index = index
        self._table = table

    def __getitem__(self, key: bytes) -> list[Hashable]:
        # A defaultdict(list): a missing bucket reads as [] (it is not created).
        if not isinstance(key, bytes):
            hash(key)
            return []
        found = self._index.table_get(self._table, key)
        return [] if found is None else found

    def get(self, key: bytes, default: Any = None) -> Any:
        if not isinstance(key, bytes):
            hash(key)
            return default
        found = self._index.table_get(self._table, key)
        return default if found is None else found

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._index.table_keys(self._table))

    def __len__(self) -> int:
        return self._index.table_len(self._table)  # type: ignore[no-any-return]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, bytes):
            hash(key)
            return False
        return self._index.table_contains(self._table, key)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return repr(dict(self.items()))


class _SortedTable(Sequence[bytes]):
    """``forest.sorted_hashtables[i]``: the band keys sorted by the last ``index()``."""

    def __init__(self, index: Any, table: int) -> None:
        self._index = index
        self._table = table

    def __len__(self) -> int:
        return self._index.sorted_len(self._table)  # type: ignore[no-any-return]

    @overload
    def __getitem__(self, i: int) -> bytes: ...
    @overload
    def __getitem__(self, i: slice) -> list[bytes]: ...
    def __getitem__(self, i: int | slice) -> bytes | list[bytes]:
        if isinstance(i, slice):
            return self._index.sorted_list(self._table)[i]  # type: ignore[no-any-return]
        i = operator.index(i)
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError("list index out of range")
        return self._index.sorted_item(self._table, i)  # type: ignore[no-any-return]

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._index.sorted_list(self._table))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, _SortedTable)):
            return list(self) == list(other)
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(list(self))


class MinHashLSHForest:
    """The LSH Forest for MinHash: top-``k`` queries by Jaccard similarity.

    Band keys are kept in sorted arrays, one per prefix tree, as in
    datasketch. It also works with weighted MinHashes.

    Args:
        num_perm: The number of permutations of the MinHashes to index (the
            ``sample_size`` of a weighted MinHash).
        l: The number of prefix trees.

    Raises:
        ValueError: ``num_perm`` or ``l`` is not positive, or ``l > num_perm``.

    Attributes:
        l: The number of prefix trees.
        k: The maximum depth of a tree, ``int(num_perm / l)``.
        hashranges: ``[(i * k, (i + 1) * k) for i in range(l)]``.
        keys: Key -> its band keys (a dict view).
        hashtables: Per tree: band key -> keys added with it, in order.
        sorted_hashtables: Per tree: its band keys, sorted at the last
            :meth:`index`.
    """

    def __init__(self, num_perm: int = 128, l: int = 8) -> None:  # noqa: E741
        if l <= 0 or num_perm <= 0:
            raise ValueError("num_perm and l must be positive")
        if l > num_perm:
            raise ValueError("l cannot be greater than num_perm")
        # Number of prefix trees
        self.l = l
        # Maximum depth of the prefix tree
        self.k = int(num_perm / l)
        tables = range(self.l)
        self._index = _ForestNative(len(tables), self.k)
        self.hashtables: list[Mapping[bytes, list[Hashable]]] = [
            _ForestTable(self._index, i) for i in tables
        ]
        self.hashranges = [(i * self.k, (i + 1) * self.k) for i in range(self.l)]
        self.keys: MutableMapping[Hashable, list[bytes]] = _ForestKeys(self._index)
        # This is the sorted array implementation for the prefix trees
        self.sorted_hashtables: list[Sequence[bytes]] = [
            _SortedTable(self._index, i) for i in range(self.l)
        ]
        # The permutation scheme of the indexed MinHash, learned on first add.
        self._minhash_scheme: str | None = None

    def add(self, key: Hashable, minhash: Any) -> None:
        """Add ``key`` with the MinHash of its set.

        The key is searchable after the next :meth:`index` (or now, through
        band keys that are already indexed).

        Raises:
            ValueError: The MinHash is shorter than ``k * l``, has another
                scheme than earlier ones, or ``key`` was already added.
        """
        if len(minhash) < self.k * self.l:
            raise ValueError("The num_perm of MinHash out of range")
        self._minhash_scheme = _check_scheme_consistency(
            getattr(self, "_minhash_scheme", None), minhash
        )
        self._index.add(key, minhash, self._band_keys)

    def _band_keys(self, minhash: Any) -> list[bytes]:
        # datasketch's expression, for hashvalues the kernel does not read.
        return [self._H(minhash.hashvalues[start:end]) for start, end in self.hashranges]

    def index(self) -> None:
        """Index all the keys added so far and make them searchable."""
        self._index.index()

    def _query(self, minhash: Any, r: int, b: int) -> Iterator[Hashable]:
        if r > self.k or r <= 0 or b > self.l or b <= 0:
            raise ValueError("parameter outside range")
        # Generate prefixes of concatenated hash values
        hps = self._index.prefixes(minhash, r) if type(r) is int else None
        if hps is None:
            hps = [self._H(minhash.hashvalues[start : start + r]) for start, _ in self.hashranges]
        # Set the prefix length for look-ups in the sorted hash values list
        prefix_size = len(hps[0])
        for i, hp in enumerate(hps[: len(self.sorted_hashtables)]):
            yield from self._index.prefix_keys(i, hp, prefix_size)

    def query(self, minhash: Any, k: int) -> list[Hashable]:
        """Return up to ``k`` keys with (approximately) the highest Jaccard similarity.

        Args:
            minhash: The MinHash of the query set.
            k: The maximum number of keys to return.

        Returns:
            The keys, in no particular order.

        Raises:
            ValueError: ``k <= 0``, or the MinHash is too short or has
                another scheme.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if len(minhash) < self.k * self.l:
            raise ValueError("The num_perm of MinHash out of range")
        _check_scheme_consistency(getattr(self, "_minhash_scheme", None), minhash)
        if type(k) is int:
            found = self._index.query(minhash, min(k, sys.maxsize))
            if found is not None:
                return found  # type: ignore[no-any-return]
        results = set()
        r = self.k
        while r > 0:
            for key in self._query(minhash, r, self.l):
                results.add(key)
                if len(results) >= k:
                    return list(results)
            r -= 1
        return list(results)

    def get_minhash_hashvalues(self, key: Hashable) -> Any:
        """Return the hash values stored for ``key`` (a numpy array).

        Raises:
            KeyError: ``key`` is not in the forest.
            ValueError: The stored band keys are not whole 4- or 8-byte values.
        """
        import numpy as np

        byteslist = self.keys.get(key, None)
        if byteslist is None:
            raise KeyError(f"The provided key does not exist in the LSHForest: {key}")
        # Each stored segment packs self.k hash values; derive the value byte
        # width from the segment size (4 for the "affine32" MinHash scheme,
        # 8 for "affine64" and "legacy"). Validate before trusting it, so
        # corrupted or truncated storage raises instead of being silently
        # reinterpreted with the wrong width.
        segment_size = len(byteslist[0])
        bytes_per_value, remainder = divmod(segment_size, self.k)
        if remainder != 0 or bytes_per_value not in (4, 8):
            raise ValueError(
                f"Stored hash segment of {segment_size:d} bytes is not {self.k:d} values of a "
                "supported width (4 or 8 bytes)"
            )
        if any(len(item) != segment_size for item in byteslist):
            raise ValueError("Stored hash segments have inconsistent sizes")
        dtype = np.uint32 if bytes_per_value == 4 else np.uint64
        values_per_segment = segment_size // np.dtype(dtype).itemsize
        hashvalues = np.empty(len(byteslist) * values_per_segment, dtype=dtype)
        for index, item in enumerate(byteslist):
            # unswap the bytes, as their representation is flipped during storage
            hv_segment = np.frombuffer(item, dtype=dtype).byteswap()
            curr_index = index * values_per_segment
            hashvalues[curr_index : curr_index + values_per_segment] = hv_segment
        return hashvalues

    def _binary_search(self, n: int, func: Callable[[int], bool]) -> int:
        """https://golang.org/src/sort/search.go?s=2247:2287#L49."""
        i, j = 0, n
        while i < j:
            h = int(i + (j - i) / 2)
            if not func(h):
                i = h + 1
            else:
                j = h
        return i

    def is_empty(self) -> bool:
        """Return True if some tree has nothing searchable (before :meth:`index`)."""
        return self._index.is_empty()  # type: ignore[no-any-return]

    def _H(self, hs: Any) -> bytes:  # noqa: N802
        return bytes(hs.byteswap().data)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.keys
