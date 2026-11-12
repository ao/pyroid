# Large datasets

`pyroid clean` and the other commands run on datasets of thousands of Parquet or JSONL files, on
one machine or many, with memory that does not grow with the dataset. A run that stops part way
picks up where it left off.

```console
$ pip install "pyroid[cli]"
$ pyroid clean data/ out/ --recipe fineweb --workers 8 --progress
```

With a directory as OUT, every input file gets its own output file under the same relative name
(`data/2024/part-00017.parquet` becomes `out/2024/part-00017.parquet`). `--workers 8` runs eight
processes on one machine, one input file each, and splits the CPUs between them. The output is
the same as a single-process run.

## Inputs

IN can be any mix of:

- files: `part-0001.parquet`, `docs.jsonl.zst`;
- directories, read recursively; names starting with `.` or `_` are skipped (`_SUCCESS`,
  `.pyroid/`);
- globs: `data/*.parquet`, `data/**/part-*.jsonl.gz`;
- URIs: `s3://bucket/prefix/`, `gs://bucket/*.parquet`,
  `hf://datasets/HuggingFaceFW/fineweb/sample/10BT/*.parquet`, read through fsspec when it is
  installed (with `s3fs`, `gcsfs` or `huggingface_hub` for those schemes), otherwise through
  pyarrow's own S3 and GCS filesystems; any other URI fsspec can open works too;
- `--files-from list.txt`: one of the above per line.

Files are listed, then sorted by name within each IN, and each file's position in that list is its
*rank*. Files are streamed in Arrow batches of `--batch-size` rows (10,000 by default), one Parquet
row group at a time, and the next batch is read while the current one is processed: memory holds
about one row group and two batches per process.

OUT, and the folders of the [MinHash stages](#minhash-dedup-across-shards), are local paths (or a
shared filesystem such as NFS, EFS or Lustre). Outputs are not written to object stores directly.

## Output layout

| OUT | What is written |
|---|---|
| `out.parquet`, `out.jsonl.gz`, ... (a data suffix) | one file with every kept row, in input order; with `--shard`, one per shard: `out-00003-of-00016.parquet` |
| `out/` (a directory) | one file per input file, same relative name; `--format` changes the format |
| `out/ --merge` | one file per shard: `out/part-00000-of-00001.parquet` |

`--excluded` follows OUT: a file when OUT is a file, a directory (outside OUT) when OUT is one.

Every output is written under a temporary name and renamed when complete, so a crash never
leaves a half-written file under its final name. Renames and deletes happen only after every
handle on the file is closed, and are retried while Windows reports the file as in use (a virus
scanner, an indexer).

## Resuming

A directory OUT records each finished input file in `OUT/.pyroid/done/`. Run the same command
again after a crash, a kill or a lost machine, and it skips what is done:

```console
$ pyroid clean data/ out/ --recipe fineweb --workers 8
^C
$ pyroid clean data/ out/ --recipe fineweb --workers 8   # redoes only the unfinished files
```

Each record holds a fingerprint of the command, its options, the input list (names relative to
IN, and sizes) and the pyroid version, plus the files the unit wrote, as paths relative to OUT. A
file counts as done only if its outputs are still there and are the ones the command would write
now, so a deleted output, another `--format` or another `--excluded` redoes that file. If the
fingerprint changes, pyroid refuses to mix the two runs; `--overwrite` redoes everything. Since
nothing depends on absolute paths, a machine that mounts the data and OUT somewhere else resumes
the same run. A file OUT keeps the same records next to it (`.out.parquet.pyroid/`) while it runs
and removes them when it is done.

## Many machines: `--shard`

`--shard I/N` takes input files I, I+N, I+2N, ... (datatrove's split). Give each machine its own
I and the same OUT on a shared filesystem, or separate OUTs:

```console
$ pyroid langid s3://bucket/raw/ /mnt/shared/langid/ --shard 3/16 --workers 32
```

`--stats-json PATH` writes one shard's counts; each shard also leaves
`OUT/.pyroid/stats-0000I-of-0000N.json`. `pyroid stats OUT` sums every finished file of every
shard:

```console
$ pyroid stats /mnt/shared/langid/
{
 "command": "langid",
 "docs_in": ...,
 ...
}
```

`--progress` prints one status line to stderr (files done, documents, rate, time left). Without
it the commands print nothing unless something fails.

## MinHash dedup across shards

Dedup is a decision over the whole dataset, so `pyroid clean --shard` can't do it inside one
shard. Run the chain without it, then datatrove's four MinHash stages on the result:

```console
# 1. on each machine i = 0..15: the chain up to the quality filters
$ pyroid clean s3://bucket/raw/ /mnt/shared/clean/ --recipe fineweb --no-dedup --shard $i/16
# 2. on each machine: signatures of the cleaned files (FineWeb's settings: sha1, 64-bit)
$ pyroid dedup signatures /mnt/shared/clean/ /mnt/shared/sigs/ --hash-fc sha1 --shard $i/16
# 3. on up to 14 machines (one per bucket), or a multiple of 14 to split buckets by hash range
$ pyroid dedup buckets /mnt/shared/sigs/ /mnt/shared/dups/ --shard $i/14
# 4. on one machine
$ pyroid dedup cluster /mnt/shared/dups/ /mnt/shared/removed/
# 5. on each machine: copy the cleaned files without the duplicates
$ pyroid dedup filter /mnt/shared/clean/ /mnt/shared/final/ --removed /mnt/shared/removed/ --shard $i/16
```

Each stage checks that the one before it finished on every shard, and that the MinHash settings
and the input list match. `signatures`, `buckets` and `filter` resume like the per-file
commands; `cluster` runs whole again.

The folders have datatrove 0.10.1's layout and byte format (`bucket_XXX/RANK.minhash.sig`,
`BUCKET_WORKER.dups`, `RANK.remove`), so either tool can run any stage on the other's files. The
rank of a file is its position in the sorted input list, which is what datatrove does with one
task per input file. Documents are numbered in input order, so the result equals one
single-process `pyroid clean` (or `pyroid.dedup.MinHashDedup` over all files in order): the same
documents are removed, and the same one of each cluster is kept.

On one machine, `pyroid clean` and `pyroid dedup` (without `--shard`) run the same four stages
themselves, in a work directory next to OUT (`--tmp-dir` moves it).

| Stage | Memory |
|---|---|
| `signatures` | `--sort-buffer-mb` (64 MiB) per process; past it, sorted runs spill to disk (952 bytes per document with 14 × 8 hashes) |
| `buckets` | 64 MiB of read buffers per bucket, one bucket per thread (`--threads`); past the open-file limit, files are merged in groups first |
| `cluster` | one entry per document that has a near-duplicate, as in datatrove (one process) |
| `filter` | one row group plus one batch |

## Speed and memory

On 4,188,430 FineWeb documents (`sample/10BT` files `000` to `003`, 8.6 GB of Parquet with about
1,000 rows per row group), on AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, 16 vCPU, 32 GiB),
Amazon Linux 2023, CPython 3.13, pyroid at commit 8d8359b7. Peak RSS is the largest single
process (GNU `time`).

| Run | Wall | Peak RSS |
|---|---|---|
| `pyroid clean data/ kept.parquet --recipe fineweb`, one process | 454 s | 1.7 GiB |
| the same with `--workers 4`, a directory OUT | 349 s | 1.0 GiB per worker |
| `clean --no-dedup --shard I/4`, one shard (one file) | 64-67 s | 1.2 GiB |
| `dedup signatures --shard I/4`, one shard | 17 s | 0.8 GiB |
| `dedup buckets --shard I/2`, seven buckets | 0.4 s | not recorded |
| `dedup cluster` | 0.2 s | 75 MiB |
| `dedup filter --shard I/4`, one shard | 25-27 s | 0.7 GiB |
| `clean --no-dedup --workers 4` of two files read from `hf://` | 135 s | 1.0 GiB |

Every layout kept the same 3,547,373 documents with the same text, and
the `hf://` run equals the same two files read from disk. The shards above ran one after another
on one machine; on four machines the stages would add up to about 110 s (65 + 17 + 0.4 + 0.2 +
27 s, not measured as such). A run stopped with Ctrl-C after
two of four files had passed the filters took 263 s to finish, against 349 s from scratch.

On a smaller machine, the per-stage peaks on 1,000,000 short synthetic documents
(`python/tests/test_cli_memory.py`, `--sort-buffer-mb 64`, release build, Docker VM on an M3 Pro,
indicative) were 237 MiB for `signatures`, 118 MiB for `buckets`, 69 MiB for `cluster`, 198 MiB
for `filter`, and 694 MiB for `clean --recipe fineweb` with the LID model. The signatures alone of
`pyroid.dedup.MinHashDedup`'s in-memory index would take 854 MiB there.

## Proof of identity

The tests run each of these on the 13,232-document parity corpus (FineWeb dev plus the
adversarial set, 335 MinHash removals), split into five files, and compare the per-document
digest of `bench/pipeline.py` with the single-process run, whose digest is datatrove's FineWeb
chain (`c0c6b746…`):

- `pyroid clean parts/ out/ --workers 2`;
- `pyroid clean --no-dedup --shard I/3`, then `dedup signatures --shard I/3`,
  `buckets --shard I/2`, `cluster` and `filter --shard I/3`.

The 100,000-document benchmark slice, split into 16 files and run as four shards of two workers
each, gives the digest of datatrove's FineWeb chain on that slice (`da897999…`, from the
benchmark), and datatrove's own `MinhashDedupSignature` (16 tasks), `MinhashDedupBuckets` (14
tasks) and `MinhashDedupCluster` write byte-identical `.minhash.sig`, `.dups` and `.remove` files
from the same cleaned files (`python/tests/test_cli_scale_100k.py`, with `PYROID_TEST_SCALE=1`).
Smaller cases cover several ranks, both hash precisions, an empty rank and split buckets
(`python/tests/test_dedup_files.py`).

## What differs from datatrove

- Ranks are input files, never a group of files: `--shard I/N` gives a process several files, but
  each keeps its own rank. datatrove with fewer tasks than files numbers documents per task, which
  can change which copy of a duplicate is kept (not how many are removed).
- Index files (`index_folder`, `create_index_name`, `only_dedup_in_index`) are not supported.
- `dedup filter --load-cluster-ids` adds the column to every file; datatrove skips files that had
  no removals.
