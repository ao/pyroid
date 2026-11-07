# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch. The API docs are built from this stub.
"""An exact drop-in for datasketch 2.0.0's MinHash family, with the work in Rust.

```python
from pyroid.datasketch import MinHash, MinHashLSH  # instead of: from datasketch import ...
```

Hash values, LeanMinHash bytes, the LSH ``(b, r)`` parameters and query results
are datasketch 2.0.0's, for all three permutation schemes. The submodules mirror
datasketch's (``pyroid.datasketch.minhash``, ``.lean_minhash``, ``.lsh``, ...), so
``from datasketch.lsh import MinHashLSH`` becomes
``from pyroid.datasketch.lsh import MinHashLSH``. numpy is required, as for
datasketch; scipy is not.

pyroid extensions run on pyroid's thread pool: ``MinHash.bulk(..., threads=)``,
:func:`minhash_batch` (a signature matrix from token lists or texts) and the
``MinHashLSH`` batch methods.

datasketch classes that pyroid does not port (``bBitMinHash``, ``HNSW``,
``MinHashLSHBloom``, ``AsyncMinHashLSH``, ``WeightedMinHash``,
``WeightedMinHashGenerator``, ``HyperLogLog``, ``HyperLogLogPlusPlus``) are
datasketch's own: getting one from this module imports datasketch, with a
one-time :class:`pyroid.FallbackWarning` (:class:`pyroid.PyroidError` if
datasketch is not installed).
"""

from typing import Any, Final

from pyroid.datasketch.hashfunc import sha1_hash32 as sha1_hash32
from pyroid.datasketch.hashfunc import sha1_hash64 as sha1_hash64
from pyroid.datasketch.lean_minhash import LeanMinHash as LeanMinHash
from pyroid.datasketch.lsh import MinHashLSH as MinHashLSH
from pyroid.datasketch.lshensemble import MinHashLSHEnsemble as MinHashLSHEnsemble
from pyroid.datasketch.lshforest import MinHashLSHForest as MinHashLSHForest
from pyroid.datasketch.minhash import MinHash as MinHash
from pyroid.datasketch.minhash import minhash_batch as minhash_batch

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

WeightedMinHashLSH = MinHashLSH
WeightedMinHashLSHForest = MinHashLSHForest

#: The datasketch release whose results pyroid.datasketch reproduces.
DATASKETCH_VERSION: Final[str]

def __getattr__(name: str) -> Any: ...
