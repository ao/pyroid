# `pyroid.datasketch`: drop-in datasketch

`pyroid.datasketch` reproduces [datasketch](https://github.com/ekzhu/datasketch) 2.0.0 by Eric
Zhu (MIT): `MinHash`, `LeanMinHash`, `MinHashLSH`, `MinHashLSHForest` and `MinHashLSHEnsemble`.
The classes are datasketch's own Python code, with the same attributes, pickles and error
messages; the hashing, the permutations, the LSH index and the `(b, r)` parameter search run in
Rust. Hash values, LeanMinHash bytes and LSH query results are datasketch's for all three
permutation schemes ([parity](../compatibility.md#datasketch)).

```bash
pip install "pyroid[datasketch]"   # numpy, which pyroid.datasketch needs as datasketch does
```

## Change one import

```diff
-from datasketch import MinHash, MinHashLSH
+from pyroid.datasketch import MinHash, MinHashLSH

 m1, m2 = MinHash(num_perm=128), MinHash(num_perm=128)
 m1.update_batch([b"minhash", b"is", b"a", b"probabilistic", b"data", b"structure"])
 m2.update_batch([b"minhash", b"is", b"a", b"probability", b"data", b"structure"])
 m1.jaccard(m2)  # 0.65625, as with datasketch

 lsh = MinHashLSH(threshold=0.5, num_perm=128)
 lsh.insert("m2", m2)
 lsh.query(m1)  # ['m2']
```

The submodules have datasketch's names too: `from datasketch.lsh import MinHashLSH` becomes
`from pyroid.datasketch.lsh import MinHashLSH`. To switch code you do not own, use
[`pyroid.patch("datasketch")`](#patching-code-you-do-not-own). scipy is not needed.

## MinHash

`MinHash(num_perm=128, seed=1, scheme=...)` takes datasketch's arguments. The scheme is
`"affine32"` by default (datasketch 2.0.0's), `"affine64"`, or `"legacy"` for sketches made by
datasketch before 2.0.0. `hashvalues` is a numpy array of the scheme's dtype (`uint32` for
`"affine32"`, `uint64` otherwise), exactly as datasketch computes it.

- `update(b)` and `update_batch(values)` hash `bytes` with SHA1 in Rust, then apply the 128
  permutations there. Other bytes-like values (`bytearray`, `memoryview`, numpy scalars) run
  datasketch's numpy code, with the same result.
- `hashfunc=`: pyroid's and datasketch's `sha1_hash32` and `sha1_hash64` run natively. Any other
  function is called from Python for every value, as in datasketch, with a one-time
  `pyroid.FallbackWarning`; the permutations still run in Rust when it returns ints in range.
- The permutation parameters are drawn as datasketch draws them with numpy's legacy
  `RandomState`, in Rust, and cached per seed, so creating a `MinHash` is 65 to 240 times faster
  than with datasketch (Docker/OrbStack VM, indicative).
- `jaccard`, `count`, `merge`, `union`, `copy`, `digest`, `is_empty`, `clear` and pickling are
  datasketch's code. A `MinHash` pickled by datasketch unpickles under `pyroid.patch("datasketch")`.

`MinHash.bulk(docs, threads=None, **kwargs)` and `MinHash.generator(...)` hash documents in
parallel on pyroid's thread pool. `threads=` is pyroid's addition; the results equal datasketch's
`bulk`, and all the MinHash objects share one `permutations` array, as datasketch's do.

```python
from pyroid.datasketch import MinHash

docs = [[b"token1", b"token2", b"token3"], [b"token4", b"token5"]]
minhashes = MinHash.bulk(docs, num_perm=64, threads=8)
```

## Signatures for many documents: `minhash_batch`

`minhash_batch(docs, num_perm=128, seed=1, scheme=None, ngram=1, threads=None)` returns a
`(len(docs), num_perm)` numpy array whose rows are the `hashvalues` datasketch would compute. It
takes token lists (`bytes`, or `str` hashed as UTF-8), or texts (`str`, `list[str]` or an Arrow
string column), whose tokens are their word `ngram`-grams:

```python
text, ngram = "a quick brown fox", 2
words = text.split()
tokens = [" ".join(words[i : i + ngram]).encode("utf-8") for i in range(len(words) - ngram + 1)]
# [b'a quick', b'quick brown', b'brown fox']
```

A text with fewer than `ngram` words has no tokens, and its row stays at the scheme's maximum
value (an empty MinHash), as do Arrow nulls. For texts, the shingling runs in Rust as well, which
is where most of datasketch's time goes.

```python
import pyroid.datasketch as ds

sigs = ds.minhash_batch(texts, ngram=5, threads=8)  # uint32, shape (len(texts), 128)
lean = ds.LeanMinHash(seed=1, hashvalues=sigs[0], scheme="affine32")
```

## MinHashLSH

`MinHashLSH(threshold, num_perm, weights, params, storage_config, prepickle, hashfunc)` takes
datasketch's arguments and picks the same `(b, r)`: datasketch minimises a weighted sum of two
`scipy.integrate.quad` integrals, and pyroid ports scipy's QUADPACK code operation by operation,
so every integral is the same double. `insert`, `query`, `remove`, `merge`, `__contains__`,
`insertion_session`, `deletion_session`, `add_to_query_buffer` and the counts behave as
datasketch's in-memory index. Query results are lists in no particular order, as datasketch's
are.

The index takes any object with `hashvalues`, `len()` and `scheme`: pyroid's or datasketch's
`MinHash` and `LeanMinHash`, and datasketch's `WeightedMinHash`. Schemes cannot be mixed in one
index, as in datasketch. `WeightedMinHashLSH` and `WeightedMinHashLSHForest` are aliases of
`MinHashLSH` and `MinHashLSHForest`, as in datasketch.

`insert_batch(keys, minhashes, check_duplication=True, threads=None)` and
`query_batch(minhashes, threads=None)` are pyroid's additions. `minhashes` is a list of sketches
or a signature matrix such as `minhash_batch` returns. The results equal per-item `insert` and
`query`; `insert_batch` inserts nothing if any item is invalid.

```python
from pyroid.datasketch import MinHashLSH, minhash_batch

sigs = minhash_batch(texts, ngram=5, threads=8)
lsh = MinHashLSH(threshold=0.8, num_perm=128)
lsh.insert_batch(range(len(texts)), sigs, threads=8)
candidates = lsh.query_batch(sigs, threads=8)  # one list of keys per text
```

`storage_config` with `"type": "redis"` or `"cassandra"`, and an LSH `hashfunc=`, are
datasketch's: `MinHashLSH(...)` returns a `datasketch.MinHashLSH` (needs datasketch installed)
with a one-time `FallbackWarning`.

## LSH Forest, LSH Ensemble and LeanMinHash

- `MinHashLSHForest(num_perm, l)`: `add`, `index`, `query(minhash, k)` and
  `get_minhash_hashvalues`, with the same top-k key sets as datasketch's, including keys added
  after `index()` that share an indexed prefix. The sorted prefixes live in Rust.
- `MinHashLSHEnsemble(threshold, num_perm, num_part, m)`: datasketch's containment index on
  pyroid's `MinHashLSH`. Its ten `(b, r)` searches run in parallel in Rust, so building one takes
  milliseconds.
- `LeanMinHash`: `serialize`, `deserialize` and pickling write and read datasketch 2.0.0's byte
  format (the pre-2.0.0 format for `"legacy"`), so the bytes go both ways between pyroid and
  datasketch. `bytes` and `bytearray` buffers are handled in Rust, other buffers by datasketch's
  `struct` code.

## Patching code you do not own

```python
import pyroid

pyroid.patch("datasketch")
import datasketch  # pyroid's drop-in from here on
from datasketch.lsh import MinHashLSH  # pyroid.datasketch.lsh.MinHashLSH
```

The shim covers `datasketch` and its `minhash`, `lean_minhash`, `hashfunc`, `lsh`, `lshforest`
and `lshensemble` submodules. Everything else (`WeightedMinHash`, `HyperLogLog`, `bBitMinHash`,
`storage`, ...) is the installed datasketch's. See [`pyroid.patch()`](patch.md).

## What stays datasketch's

`WeightedMinHash`, `WeightedMinHashGenerator`, `HyperLogLog`, `HyperLogLogPlusPlus`, `bBitMinHash`,
`HNSW`, `MinHashLSHBloom` and `AsyncMinHashLSH` are not ported. Getting one from
`pyroid.datasketch` returns datasketch's class with a one-time `FallbackWarning`, or raises
`pyroid.PyroidError` if datasketch is not installed.

## Differences from datasketch

From the module's documentation (`pyroid/datasketch/*.pyi`):

- `MinHashLSH.keys` and `hashtables[i]` are read-only views with datasketch's `DictListStorage`
  and `DictSetStorage` read methods. Their values are new lists and sets on every access, so a
  query buffered with `add_to_query_buffer` sees the index as it was then. The mutating storage
  methods raise `NotImplementedError`.
- With `check_duplication=False` and keys that are equal but of different types (`1`, `1.0`,
  `True`), the object returned for that key may be another of the equal keys.
- `gpu_mode` is accepted, but pyroid computes on the CPU in every mode, with the same values.
  `gpu_mode="always"` still raises `RuntimeError` without CuPy and a CUDA device.
- `MinHash.generator` reads up to 2,048 documents ahead to hash them in parallel. The MinHash
  objects and any error come out in datasketch's order.

## Parity

There were 0 differences in 1,882,064 comparisons on 103,232 documents: the 100,000-document
FineWeb slice plus the 3,232-document adversarial set, word 5-gram shingles. The suite compares
the signatures of all three schemes, `minhash_batch`, LeanMinHash bytes, `(b, r)`, every
document's query result for four LSH configurations, `query_batch`, removals, merges, LSH Forest
top-10 and LSH Ensemble containment. datasketch's own test suite also passes against pyroid's
classes. Details: [Compatibility](../compatibility.md#datasketch).

## Speed

These timings come from a Docker/OrbStack VM on Apple Silicon (12 vCPUs, shared), indicative.
CPython 3.15t with the GIL off, the first 10,000 FineWeb dev documents, 128 permutations,
`"affine32"`; every row was first checked for identical results:

| Run | datasketch, 1 thread | pyroid, 1 thread | pyroid, 8 threads |
|---|---:|---:|---:|
| `MinHash.bulk` on word 5-gram shingles | 3.13 s | 0.39 s (8.0×) | 0.15 s (21×) |
| From raw text: shingling and signatures (`minhash_batch`) | 4.43 s | 0.35 s (12.5×) | 0.052 s (85×) |
| `MinHashLSH(threshold=0.9)`, the `(b, r)` search | 13.0 ms | 1.5 ms (8.9×) | |
| LSH insert, threshold 0.5 (`insert_batch` for 8 threads) | 525 ms | 42.9 ms (12×) | 19.2 ms (27×) |
| LSH query, threshold 0.5 (`query_batch` for 8 threads) | 235 ms | 26.7 ms (8.8×) | 12.3 ms (19×) |

datasketch split over 8 threads (the GIL is off) took 2.31 s and 2.43 s for the first two rows.
`MinHash.bulk` scales less than `minhash_batch` because copying the tokens out of Python objects
runs on one thread. More rows: [Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

API: [`pyroid.datasketch` reference](../reference/datasketch.md).
