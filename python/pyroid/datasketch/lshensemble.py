# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Messages and code stay as in datasketch, so they are byte-identical and easy to diff.
# ruff: noqa: D100, D205, UP045, UP028, C402, PIE808
import logging
import struct
from collections import Counter, deque
from collections.abc import Generator, Hashable, Iterable
from typing import Optional

import numpy as np

from pyroid._pyroid import datasketch as _native
from pyroid.datasketch.lsh import MinHashLSH
from pyroid.datasketch.lshensemble_partition import optimal_partitions
from pyroid.datasketch.minhash import MinHash
from pyroid.datasketch.storage import _random_name

__all__ = ["MinHashLSHEnsemble"]

logger = logging.getLogger(__name__)


def _false_positive_probability(threshold, b, r, xq):
    """Compute the false positive probability given the containment threshold.
    xq is the ratio of x/q.

    pyroid: datasketch's ``scipy.integrate.quad`` call, ported to Rust (QUADPACK's
    ``dqagse``, as scipy 1.18.1 runs it), so the value and every chosen ``(b, r)``
    are datasketch's.
    """
    return _native.ensemble_false_positive(threshold, b, r, xq)


def _false_negative_probability(threshold, b, r, xq):
    """Compute the false negative probability given the containment threshold."""
    return _native.ensemble_false_negative(threshold, b, r, xq)


def _optimal_param(threshold, num_perm, max_r, xq, false_positive_weight, false_negative_weight):
    """Compute the optimal parameters that minimizes the weighted sum
    of probabilities of false positive and false negative.
    xq is the ratio of x/q.
    """
    return _native.ensemble_optimal_param(
        threshold, num_perm, max_r, xq, false_positive_weight, false_negative_weight
    )


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

    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        num_part: int = 16,
        m: int = 8,
        weights: tuple[float, float] = (0.5, 0.5),
        storage_config: Optional[dict] = None,
        prepickle: Optional[bool] = None,
    ) -> None:
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if num_part < 1:
            raise ValueError("num_part must be at least 1")
        if m < 2 or m > num_perm:
            raise ValueError("m must be in the range of [2, num_perm]")
        if any(w < 0.0 or w > 1.0 for w in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.threshold = threshold
        self.h = num_perm
        self.m = m
        rs = self._init_optimal_params(weights)
        # Initialize multiple LSH indexes for each partition
        storage_config = storage_config if storage_config else {"type": "dict"}
        basename = storage_config.get("basename", _random_name(11))
        self.indexes = [
            dict(
                (
                    r,
                    MinHashLSH(
                        num_perm=self.h,
                        params=(int(self.h / r), r),
                        storage_config=self._get_storage_config(
                            basename, storage_config, partition, r
                        ),
                        prepickle=prepickle,
                    ),
                )
                for r in rs
            )
            for partition in range(0, num_part)
        ]
        self.lowers = [None for _ in self.indexes]
        self.uppers = [None for _ in self.indexes]

    def _init_optimal_params(self, weights):
        false_positive_weight, false_negative_weight = weights
        self.xqs = np.exp(np.linspace(-5, 5, 10))
        # pyroid: datasketch's `_optimal_param` per xq, the ten searches in parallel.
        self.params = np.array(
            _native.ensemble_optimal_params(
                self.threshold,
                self.h,
                self.m,
                self.xqs.tolist(),
                false_positive_weight,
                false_negative_weight,
            ),
            dtype=int,
        )
        # Find all unique r
        rs = set()
        for _, r in self.params:
            rs.add(r)
        return rs

    def _get_optimal_param(self, x, q):
        i = np.searchsorted(self.xqs, float(x) / float(q), side="left")
        if i == len(self.params):
            i = i - 1
        return self.params[i]

    def _get_storage_config(self, basename, base_config, partition, r):
        config = dict(base_config)
        config["basename"] = b"-".join(
            [basename, struct.pack(">H", partition), struct.pack(">H", r)]
        )
        return config

    def index(self, entries: Iterable[tuple[Hashable, MinHash, int]]) -> None:
        """Index every ``(key, minhash, size)`` of `entries`; callable once.

        Raises:
            ValueError: The index is not empty, `entries` is empty, or a size is
                not positive.
        """
        if not self.is_empty():
            raise ValueError("Cannot call index again on a non-empty index")
        if not isinstance(entries, list):
            queue = deque()
            for key, minhash, size in entries:
                if size <= 0:
                    raise ValueError("Set size must be positive")
                queue.append((key, minhash, size))
            entries = list(queue)
        if len(entries) == 0:
            raise ValueError("entries is empty")
        # Create optimal partitions.
        sizes, counts = np.array(sorted(Counter(e[2] for e in entries).most_common())).T
        partitions = optimal_partitions(sizes, counts, len(self.indexes))
        for i, (lower, upper) in enumerate(partitions):
            self.lowers[i], self.uppers[i] = lower, upper
        # Insert into partitions.
        entries.sort(key=lambda e: e[2])
        curr_part = 0
        for key, minhash, size in entries:
            u = self.uppers[curr_part]
            if size > u:
                curr_part += 1
            for r in self.indexes[curr_part]:
                self.indexes[curr_part][r].insert(key, minhash)

    def query(self, minhash: MinHash, size: int) -> Generator[Hashable, None, None]:
        """Yield the keys of sets likely to contain `threshold` of the query set.

        Args:
            minhash: The MinHash of the query set.
            size: The number of distinct items in the query set.
        """
        for i, index in enumerate(self.indexes):
            u = self.uppers[i]
            if u is None:
                continue
            b, r = self._get_optimal_param(u, size)
            # The permutation-scheme guard lives in the inner MinHashLSH
            # (_query_b calls _check_scheme_consistency), so it is checked
            # lazily as this generator is iterated and only for partitions
            # that have had an insert. A query reaching only empty partitions
            # returns nothing regardless, so it cannot yield wrong results.
            for key in index[r]._query_b(minhash, b):
                yield key

    def __contains__(self, key: Hashable) -> bool:
        """Args:
            key (hashable): The unique identifier of a set.

        Returns:
            bool: True only if the key exists in the index.

        """
        return any(any(key in index[r] for r in index) for index in self.indexes)

    def is_empty(self) -> bool:
        """Return whether nothing has been indexed."""
        return all(all(index[r].is_empty() for r in index) for index in self.indexes)
