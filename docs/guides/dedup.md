# `pyroid.dedup`: MinHash and exact dedup *(preview)*

!!! warning "Preview"
    `pyroid.dedup` is the newest module; names and details may still change before 1.0. See its
    [API reference](../reference/dedup.md).

`pyroid.dedup` makes **the same removals as datatrove 0.10.1's MinHash deduplication**, the
four-stage `MinhashDedupSignature` → `MinhashDedupBuckets` → `MinhashDedupCluster` →
`MinhashDedupFilter` pipeline that FineWeb was deduplicated with, but in memory, in one process
and multi-threaded. It also has a fast exact-duplicate filter.

## From datatrove's four stages to one call

```python
import pyroid.dedup as D

keep = D.minhash_dedup(texts, hash_fc="sha1")  # FineWeb's configuration (sha1, 64-bit)
deduped = [t for t, k in zip(texts, keep) if k]
```

`keep[i]` is `False` exactly for the documents datatrove's four stages remove. Configurations are
datatrove's: pass pyroid's `MinhashConfig`, datatrove's own `MinhashConfig`, a dict, or keyword
arguments (`hash_fc=` and `precision=` set the hash config). The default is datatrove's default
(xxhash, 64-bit, 5-word shingles, 14 buckets × 8 hashes, seed 1).

## Corpora that don't fit in one call

`MinHashDedup` is an in-memory index. Add documents in batches (only their signatures are kept:
8 bytes × 112 hashes per document with the default configuration), then read the decisions:

```python
d = D.MinHashDedup(hash_fc="sha1")
for batch in batches:  # lists, tuples or Arrow columns
    d.add(batch)  # returns the range of indices given to the batch
keep = d.keep_mask()  # one bool per document added, in add order
d.duplicates()  # indices to remove (datatrove's .remove file)
d.clusters()  # each cluster: the kept document first, then the removed ones
d.cluster_ids(), d.cluster_sizes()  # as datatrove writes minhash_cluster_id / _size
```

Decisions are global, so a later batch can change which document of a cluster is kept: masks are
final once every batch is in. The kept document of each cluster is datatrove's union-find root.
That is usually, but not always, the cluster's first document.

## Exact duplicates

```python
keep = D.exact_dedup(texts)  # True for the first copy of each text
keep = D.exact_dedup(texts, normalize=True)  # compare datatrove's simplify_text() instead

seen = D.ExactDedup()  # streaming, across batches
for batch in batches:
    keep = seen.add(batch)  # final as soon as add() returns
```

Texts are compared by a 128-bit hash. On a plain `list[str]`, Python's own `set` is faster than
`exact_dedup`; it earns its place on Arrow input, with normalization, and as compact streaming
state (16 bytes per distinct text).

## Arrow and threads

Every function takes a `str`, `list`, `tuple` or Arrow string column. The dedup functions return
a keep mask of the same kind (Arrow nulls stay null and are always kept), `simplify_text` returns
text of the same kind, and `minhash_signatures` returns a list. `threads=` works as everywhere in pyroid
([Threads](../concepts/threads.md)).

```python
import pyarrow as pa

keep = D.minhash_dedup(table["text"], hash_fc="sha1", threads=8)  # pyroid.Column of bools
table = table.filter(pa.chunked_array(keep))
```

`minhash_signatures(texts)` returns datatrove's signatures themselves (a flat list of
`num_buckets * hashes_per_bucket` ints per document, `None` for documents with fewer than
`n_grams` words, which datatrove never removes), and `simplify_text(texts)` the normalized text
the shingles are built from.

## Differences from datatrove

From the module's documentation (`pyroid/dedup.pyi`):

- A `str` with a lone surrogate raises `UnicodeEncodeError` (pyroid's policy); datatrove raises it
  too unless the document has fewer than `n_grams` words.
- `n_grams`, `num_buckets` or `hashes_per_bucket` below 1 raise `ValueError` (datatrove silently
  produces no signatures or fails later).
- Index files (`index_folder`, `create_index_name`, `only_dedup_in_index`) are not supported.

Documents are numbered in the order they are added, as datatrove numbers the documents of one
task. Multi-worker hash ranges are not needed: one process does all the work.

## Speed

Signatures from raw text measured about 64× faster than datatrove (0.094–0.099 against 6.17 ms
per document, tokenization included) and 1.7–1.8× faster than rensa, with datatrove's decisions;
the whole four-stage pipeline on 10k documents took 1.21 s on one thread against datatrove's
99.5 s. *Docker/OrbStack VM on Apple Silicon, indicative, preliminary*: see
[Benchmarks](../benchmarks.md#pyroiddedup-preview).

Signatures and removals match datatrove on the parity corpus: all 13,232 signatures and the
445-document removed set, from raw text ([Compatibility](../compatibility.md#dedup)).
