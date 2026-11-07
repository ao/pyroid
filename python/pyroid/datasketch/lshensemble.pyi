# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.lshensemble. The API docs are built from this stub.
"""LSH Ensemble, as datasketch 2.0.0's ``datasketch.lshensemble``.

:class:`MinHashLSHEnsemble` answers containment queries: which indexed sets
contain at least `threshold` of the query set. It is datasketch's class on top
of pyroid's :class:`~pyroid.datasketch.MinHashLSH`, so the partitions, the
``(b, r)`` parameters per partition and the query results are datasketch's.
The parameter search (scipy's ``quad`` in datasketch) runs in Rust, ten
searches in parallel, so constructing an ensemble takes milliseconds.

Example:
    ```python
    from pyroid.datasketch import MinHashLSHEnsemble

    ens = MinHashLSHEnsemble(threshold=0.8, num_perm=128, num_part=16)
    ens.index([(key, minhash, size) for key, minhash, size in entries])
    set(ens.query(query_minhash, query_size))
    ```

Redis and Cassandra storage go through pyroid's ``MinHashLSH``, which hands them
to datasketch (with a one-time :class:`pyroid.FallbackWarning`).
"""

import logging
from collections.abc import Generator, Hashable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pyroid.datasketch.lsh import MinHashLSH
from pyroid.datasketch.minhash import MinHash

logger: logging.Logger

def _false_positive_probability(threshold: float, b: int, r: int, xq: float) -> float: ...
def _false_negative_probability(threshold: float, b: int, r: int, xq: float) -> float: ...
def _optimal_param(
    threshold: float,
    num_perm: int,
    max_r: int,
    xq: float,
    false_positive_weight: float,
    false_negative_weight: float,
) -> tuple[int, int]: ...

class MinHashLSHEnsemble:
    """The LSH Ensemble index for containment queries.

    Args:
        threshold: The containment threshold, in [0.0, 1.0].
        num_perm: The number of permutations of the indexed MinHash.
        num_part: The number of partitions by set size.
        m: The maximum band size `r` over the inner LSH indexes, in
            ``[2, num_perm]``.
        weights: ``(false_positive_weight, false_negative_weight)``, summing to 1.0.
        storage_config: As for :class:`~pyroid.datasketch.MinHashLSH`.
        prepickle: As for :class:`~pyroid.datasketch.MinHashLSH`.

    Raises:
        ValueError: As datasketch, for parameters out of range.
    """

    threshold: float
    h: int
    m: int
    xqs: NDArray[np.float64]
    params: NDArray[Any]
    indexes: list[dict[int, MinHashLSH]]
    lowers: list[Any]
    uppers: list[Any]
    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        num_part: int = 16,
        m: int = 8,
        weights: tuple[float, float] = (0.5, 0.5),
        storage_config: dict[str, Any] | None = None,
        prepickle: bool | None = None,
    ) -> None: ...
    def index(self, entries: Iterable[tuple[Hashable, MinHash, int]]) -> None:
        """Index every ``(key, minhash, size)`` of `entries`; callable once.

        Raises:
            ValueError: The index is not empty, `entries` is empty, or a size is
                not positive.
        """

    def query(self, minhash: MinHash, size: int) -> Generator[Hashable]:
        """Yield the keys of sets likely to contain `threshold` of the query set.

        Args:
            minhash: The MinHash of the query set.
            size: The number of distinct items in the query set.
        """

    def __contains__(self, key: Hashable) -> bool: ...
    def is_empty(self) -> bool:
        """Return whether nothing has been indexed."""
