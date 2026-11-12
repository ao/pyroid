# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.lshensemble_partition.
"""The optimal set-size partitions of LSH Ensemble (datasketch's ``lshensemble_partition``)."""

from typing import Any

__all__ = ["optimal_partitions"]

def optimal_partitions(sizes: Any, counts: Any, num_part: int) -> list[tuple[int, int]]:
    """Return `num_part` ``(lower, upper)`` set-size intervals that minimise false positives.

    Args:
        sizes: The distinct set sizes, ascending.
        counts: The number of sets of each size.
        num_part: The number of partitions.
    """
