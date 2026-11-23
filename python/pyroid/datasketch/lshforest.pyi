# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.lshforest. The module docstring is kept here on purpose:
# the API docs are built from this stub.
"""MinHash LSH Forest (``pyroid.datasketch.lshforest``), as datasketch 2.0.0's.

``MinHashLSHForest`` has datasketch's methods, attributes and errors, and
``query(minhash, k)`` returns the same key set as datasketch, including where
it stops at ``k`` keys: buckets and their keys are visited in datasketch's
order. The index is in Rust:

```python
from pyroid.datasketch.lshforest import MinHashLSHForest

forest = MinHashLSHForest(num_perm=128)
for key, m in minhashes.items():
    forest.add(key, m)
forest.index()
forest.query(minhashes["doc-1"], 10)
```

Keys added after :meth:`MinHashLSHForest.index` are found when they share an
indexed band key, as in datasketch.

Differences from datasketch: ``keys``, ``hashtables[i]`` and
``sorted_hashtables[i]`` are views of the native index with the API of a
dict, a ``defaultdict(list)`` and a list. Values read from them are new
lists. ``keys`` accepts assignment and deletion of entries, whose values must
be lists of bytes; reading a missing bucket of ``hashtables[i]`` returns
``[]`` without creating it.
"""

from collections.abc import Callable, Hashable, Iterator, Mapping, MutableMapping, Sequence
from typing import Any, Protocol

__all__ = ["MinHashLSHForest"]

class _Sketch(Protocol):
    @property
    def hashvalues(self) -> Any: ...
    def __len__(self) -> int: ...

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

    l: int
    k: int
    hashranges: list[tuple[int, int]]
    keys: MutableMapping[Hashable, list[bytes]]
    hashtables: list[Mapping[bytes, list[Hashable]]]
    sorted_hashtables: list[Sequence[bytes]]
    _minhash_scheme: str | None
    _index: Any

    def __init__(self, num_perm: int = 128, l: int = 8) -> None: ...
    def add(self, key: Hashable, minhash: _Sketch) -> None:
        """Add ``key`` with the MinHash of its set.

        The key is searchable after the next :meth:`index` (or now, through
        band keys that are already indexed).

        Raises:
            ValueError: The MinHash is shorter than ``k * l``, has another
                scheme than earlier ones, or ``key`` was already added.
        """
    def index(self) -> None:
        """Index all the keys added so far and make them searchable."""
    def query(self, minhash: _Sketch, k: int) -> list[Hashable]:
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
    def get_minhash_hashvalues(self, key: Hashable) -> Any:
        """Return the hash values stored for ``key`` (a numpy array).

        Raises:
            KeyError: ``key`` is not in the forest.
            ValueError: The stored band keys are not whole 4- or 8-byte values.
        """
    def _query(self, minhash: _Sketch, r: int, b: int) -> Iterator[Hashable]: ...
    def _H(self, hs: Any) -> bytes: ...  # noqa: N802
    def _binary_search(self, n: int, func: Callable[[int], bool]) -> int: ...
    def is_empty(self) -> bool:
        """Return True if some tree has nothing searchable (before :meth:`index`)."""
    def __contains__(self, key: Hashable) -> bool: ...
