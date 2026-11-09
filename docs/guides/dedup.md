# `pyroid.dedup`: MinHash, sentence and exact dedup

`pyroid.dedup` makes the same removals as datatrove 0.10.1's dedup pipelines, but in memory, in
one process and multi-threaded:

- MinHash near-duplicate dedup: the four-stage `MinhashDedupSignature` →
  `MinhashDedupBuckets` → `MinhashDedupCluster` → `MinhashDedupFilter` pipeline that FineWeb was
  deduplicated with;
- sentence dedup, C4's rule: `SentenceDedupSignature` → `SentenceFindDedups` →
  `SentenceDedupFilter`, which cuts every repeated three-sentence span from all but its first
  document.

It also has a fast exact-duplicate filter.

Every MinHash step is datatrove's, exactly:

- `simplify_text` normalization, for the interpreter's Unicode version (CPython 3.11 to 3.15);
- words from the word tokenizer of `language`: spaCy's English by default, with the same ports
  [`pyroid.filters`](filters.md) uses, native for 4,013 of datatrove's 4,343 language codes
  (Japanese with an installed SudachiDict dictionary);
- `n_grams`-word shingles and the same hash functions;
- the permutations drawn from numpy's `RandomState(seed)`, as datatrove draws them;
- the same buckets and the same union-find.

So the same document of each cluster is kept ([parity](../compatibility.md#dedup)).

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

`MinHashDedup` is an in-memory index. Add documents in batches, then read the decisions. Only
the signatures are kept: 8 bytes × 112 hashes per document with the default configuration.

```python
d = D.MinHashDedup(hash_fc="sha1")
for batch in batches:  # lists, tuples or Arrow columns
    d.add(batch)  # returns the range of indices given to the batch
keep = d.keep_mask()  # one bool per document added, in add order
d.duplicates()  # indices to remove (datatrove's .remove file)
d.clusters()  # each cluster: the kept document first, then the removed ones
d.cluster_ids(), d.cluster_sizes()  # as datatrove writes minhash_cluster_id / _size
```

Decisions are global, so a later batch can change which document of a cluster is kept. Masks are
final once every batch is in. The kept document of each cluster is datatrove's union-find root,
which is usually, but not always, the cluster's first document.

## Languages

`language=` is datatrove's `MinhashDedupSignature(language=...)` and
`SentenceDedupSignature(language=...)`. It takes any of datatrove's language codes
(`D.languages()`, the same table as `pyroid.filters.languages()`). The native codes (4,013 with a
Japanese dictionary installed, 4,009 without) are tokenized in Rust, on every thread:

```python
import pyroid.dedup as D

text = (
    "Die Stadt liegt am Ufer des Flusses und war seit dem Mittelalter wichtig für Kaufleute "
    "aus vielen Ländern. Heute leben hier etwa achtzigtausend Menschen, und die Altstadt "
    "mit ihren engen Gassen zieht jedes Jahr viele Besucher an."
)
docs = [
    text,
    text + " Mehr dazu morgen.",
    "Dieser ganz andere Text beschreibt das Wetter in den Bergen.",
]
D.minhash_dedup(docs, hash_fc="sha1", language="deu_Latn")  # [True, False, True]
```

Japanese (`SpaCyTokenizer("ja")`, SudachiPy ported to Rust) is native when a SudachiDict
dictionary is installed: `pip install sudachidict-core`, or the `system.dic` that
`PYROID_SUDACHI_DICT` names ([Languages](languages.md#japanese)). Without one, Japanese is one of
the codes below.

For MinHash, the other 330 codes (Vietnamese with pyvi, Stanza, Thai, Kiwi, ...) need
`python_tokenizer=True`. Then datatrove's own tokenizer splits each document in Python, on
the calling thread. `simplify_text` still runs in Rust, and so do shingling, MinHash, bucketing
and clustering on its words, with one [`FallbackWarning`](../concepts/fallbacks.md) per function
and language. That path runs at the tokenizer's speed. `language` may also be a word tokenizer
object: a datatrove `WordTokenizer`, or anything with `word_tokenize(text)`.

Sentence dedup takes native codes only. Other codes raise `NotImplementedError` (there is no
Python-tokenizer path), unless no tokenizer is needed. The language table, the choices and the
per-language speed are in [Languages](languages.md).

## Sentence dedup

`sentence_dedup` runs datatrove's three sentence-dedup stages over one corpus. It returns each
document's text with its duplicate spans removed, and the keep mask:

```python
import pyroid.dedup as D

boiler = "Subscribe to our newsletter. Follow us on social media. All rights reserved."
docs = [
    "The river flooded the old town last spring. Volunteers filled sandbags for three days. "
    "The water fell within a week. " + boiler,
    "A new bakery opened on the main square. It sells bread from a wood-fired oven. "
    "The queue starts at seven in the morning. " + boiler,
    boiler,
]
result = D.sentence_dedup(docs, min_doc_words=10)
result.keep  # [True, True, False]
result.text[1]
# 'A new bakery opened on the main square. It sells bread from a wood-fired oven. The queue
# starts at seven in the morning.'
```

The three stages work as in datatrove:

1. Signatures: a document's sentences each go through `simplify_text`, and every window of
   `n_sentences` consecutive ones is hashed. The sentences come from datatrove's tokenizer for
   `language`, or are the document's lines with `split_sentences=False`.
2. Duplicates: for each hash seen more than once, every window but the first (lowest document,
   then window) is removed.
3. Filter: the sentences of each removed window are cut out. A document whose text changed is
   dropped if it has fewer than `min_doc_words` words or `min_num_sentences` sentences, and an
   empty text is always dropped.

So the first document keeps its boilerplate, the second loses it, and the third, which is only
boilerplate, is dropped. Configurations are datatrove's `SentDedupConfig`: by default 3 sentences
per window, `min_doc_words=50`, `min_num_sentences=3` and xxhash/64. Pass pyroid's
`SentDedupConfig`, datatrove's own, a dict, or keyword arguments (with `hash_fc=` and
`precision=`).

For a corpus in batches, `SentenceDedup` keeps only the signatures, in Rust (16 bytes per
window). Decisions are global, so add every batch first, then filter each one:

```python
d = D.SentenceDedup(min_doc_words=10)
first = d.add(docs[:2])  # stage 1: range(0, 2), the indices given to the batch
rest = d.add(docs[2:])  # range(2, 3)
d.duplicates()  # stage 2: [(1, 3), (2, 0)], the (document, window) pairs to remove
d.filter(docs[:2], start=first.start).keep  # stage 3: [True, True]
d.filter(docs[2:], start=rest.start).keep  # [False]
d.fit(docs).keep  # all three stages on one corpus: [True, True, False]
```

Pass `filter` the same documents, in the same order, as `add`. `duplicates()` are the records
of datatrove's `.c4_dup` files. For datatrove's files themselves (signatures, duplicates, and
the index files of `SentenceDedupBuildIndex` and `only_dedup_in_index`), use the datatrove
drop-in steps in [datatrove steps](datatrove.md), which write the same files, byte for byte.

The signature, duplicate and index files are byte-identical to datatrove's, and the same
documents come out. That holds on the 13,232-document parity corpus with 7 configurations, and on
an 8,300-document corpus with injected duplicates with 9. Hypothesis runs add 300 cases and 100
three-stage ones, and six FineWeb-2 languages have their own cases
([Compatibility](../compatibility.md#dedup)).

## Exact duplicates

```python
keep = D.exact_dedup(texts)  # True for the first copy of each text
keep = D.exact_dedup(texts, normalize=True)  # compare datatrove's simplify_text() instead

seen = D.ExactDedup()  # streaming, across batches
for batch in batches:
    keep = seen.add(batch)  # final as soon as add() returns
```

Texts are compared by a 128-bit hash. On a plain `list[str]`, Python's own `set` is faster than
`exact_dedup`. `exact_dedup` earns its place on Arrow input, with normalization, and as compact
streaming state (16 bytes per distinct text).

## Arrow and threads

Every function takes a `str`, `list`, `tuple` or Arrow string column. The dedup functions return
a keep mask of the same kind, and Arrow nulls stay null and are always kept. `sentence_dedup`
returns text and mask of the same kind, `simplify_text` returns text of the same kind, and
`minhash_signatures` returns a list. `threads=` works as everywhere in pyroid
([Threads](../concepts/threads.md)).

```python
import pyarrow as pa

keep = D.minhash_dedup(table["text"], hash_fc="sha1", threads=8)  # pyroid.Column of bools
table = table.filter(pa.chunked_array(keep))
```

`minhash_signatures(texts)` returns datatrove's signatures themselves: a flat list of
`num_buckets * hashes_per_bucket` ints per document, or `None` for documents with fewer than
`n_grams` words, which datatrove never removes. `simplify_text(texts)` returns the normalized
text the shingles are built from.

## Differences from datatrove

From the module's documentation (`pyroid/dedup.pyi`):

- A str with a lone surrogate raises `UnicodeEncodeError` (pyroid's policy). datatrove raises it
  too, unless the document has fewer than `n_grams` words.
- `n_grams`, `num_buckets` or `hashes_per_bucket` below 1 raise `ValueError` (datatrove silently
  produces no signatures or fails later).
- Index files (`index_folder`, `create_index_name`, `only_dedup_in_index`) are not supported.
- An unknown `language` raises datatrove's `ValueError` when the call starts, as
  `MinhashDedupSignature` does when it is built.

For sentence dedup, from `SentenceDedup`'s documentation:

- A str with a lone surrogate raises `UnicodeEncodeError` (so does datatrove, except with
  `split_sentences=False` on a document with fewer than `n_sentences` lines).
- `n_sentences` below 1 raises `ValueError`.
- A hashed window index past 65,535 raises datatrove's `OverflowError` (numpy's `<u2`) when its
  batch is added, and the batch is not added (datatrove fails when it saves the task's
  signatures).
- Index files: use the datatrove drop-ins (`pyroid.integrations.datatrove`).

Also for sentence dedup: codes whose datatrove tokenizer pyroid does not port (pyvi, Stanza, ...,
and Japanese without a SudachiDict dictionary) raise `NotImplementedError`, and `language` must be
a code, not a tokenizer object. datatrove's quirks are kept. `min_words_to_remove_span` only counts, and only restores, the first
sentence of a removed run. Stage 1 strips sentences while stage 3 does not, so window indices can
shift.

Also: `simplify_text`'s decimal-digit class comes from `regex` 2026.9.29's data, which another
`regex` release could change. On a Python newer than Unicode 17.0, pyroid warns once and uses its
17.0 tables. MinHash calls Python only for a Python tokenizer (`python_tokenizer=True`, or a
tokenizer object as `language`), which emits `FallbackWarning`.

Documents are numbered in the order they are added, as datatrove numbers the documents of one
task. Multi-worker hash ranges are not needed: one process does all the work.

## Speed

On an AWS c7i.4xlarge (Intel Xeon Platinum 8488C), MinHash signatures from raw text took 146 µs
per document on one thread, against 4,940 µs for datatrove: 34× faster. With 8 threads they took
28.0 µs. That is sha1/64 with tokenization included, on the first 1,000 FineWeb documents, with
pyroid on free-threaded CPython 3.15t and datatrove on CPython 3.13. rensa 0.5.0 took 144–161 µs
and datasketch 2.0.0 436 µs per document, with other hash families and without datatrove's
decisions. On a c8g.4xlarge (AWS Graviton4) pyroid was 47× faster than datatrove. See
[Benchmarks](../benchmarks.md#per-stage).

The rest of this section comes from earlier runs on a Docker/OrbStack VM on Apple Silicon,
indicative.

In the module's preliminary run, the whole four-stage pipeline on 10k documents took 1.21 s on
one thread against datatrove's 99.5 s ([details](../benchmarks.md#pyroiddedup)).

Beyond English, MinHash signatures with FineWeb-2's configuration ran 36–64× faster than
datatrove on one thread for German, French, Spanish, Russian and Arabic (Hindi: 10×), 44× for
Chinese and 6.4× for Japanese ([Languages](languages.md#speed)).

Sentence dedup ran on the 13,232-document parity corpus with datatrove's example configuration
(release wheel, CPython 3.13, load average not recorded):

- stages 1 and 3 through the datatrove drop-in steps: 5,664 documents/s on one thread (58×) and
  10,457 with 8 threads (107×), against datatrove's 98 documents/s in one process;
- stage 1 alone: 1.03 s, against 25.4 s for datatrove at 8 processes;
- stage 2: 0.008 s, against 0.30 s;
- the in-memory `fit`: 6,059 documents/s on one thread and 17,316 with 8.

See [Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

Signatures and removals match datatrove on the parity corpus: all 13,232 signatures and the
445-document removed set, from raw text ([Compatibility](../compatibility.md#dedup)).

API: [`pyroid.dedup` reference](../reference/dedup.md).
