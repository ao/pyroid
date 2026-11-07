# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
"""Exact drop-in for datasketch 2.0.0's MinHash family, with the work in Rust.

``from pyroid.datasketch import MinHash, MinHashLSH`` replaces
``from datasketch import MinHash, MinHashLSH``: hash values, LeanMinHash
bytes, LSH parameters and query results are datasketch 2.0.0's. The
submodules mirror datasketch's (``pyroid.datasketch.minhash``,
``.lsh``, ...). ``MinHash.bulk(..., threads=)``, :func:`minhash_batch`
and the LSH batch methods are pyroid extensions that run on pyroid's
thread pool.

Classes pyroid does not port (``bBitMinHash``, ``HNSW``, ``MinHashLSHBloom``,
``AsyncMinHashLSH``, ...) are datasketch's own, imported on first use with a
one-time :class:`pyroid.FallbackWarning`; this needs datasketch installed.
"""

from __future__ import annotations

from typing import Any

from pyroid.datasketch._common import DATASKETCH_VERSION, fallback
from pyroid.datasketch.hashfunc import sha1_hash32, sha1_hash64
from pyroid.datasketch.lean_minhash import LeanMinHash
from pyroid.datasketch.lsh import MinHashLSH
from pyroid.datasketch.lshensemble import MinHashLSHEnsemble
from pyroid.datasketch.lshforest import MinHashLSHForest
from pyroid.datasketch.minhash import MinHash, minhash_batch

# Alias, as in datasketch
WeightedMinHashLSH = MinHashLSH
WeightedMinHashLSHForest = MinHashLSHForest

__all__ = [
    "DATASKETCH_VERSION",
    "LeanMinHash",
    "MinHash",
    "MinHashLSH",
    "MinHashLSHEnsemble",
    "MinHashLSHForest",
    "WeightedMinHashLSH",
    "WeightedMinHashLSHForest",
    "minhash_batch",
    "sha1_hash32",
    "sha1_hash64",
]

# datasketch's public names that pyroid hands to datasketch itself.
_DELEGATED = frozenset(
    {
        "AsyncMinHashLSH",
        "HNSW",
        "HyperLogLog",
        "HyperLogLogPlusPlus",
        "MinHashLSHBloom",
        "WeightedMinHash",
        "WeightedMinHashGenerator",
        "bBitMinHash",
    }
)


def __getattr__(name: str) -> Any:
    if name in _DELEGATED:
        return getattr(fallback(name, "this class"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
