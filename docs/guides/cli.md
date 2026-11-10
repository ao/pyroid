# The `pyroid` command line

The `pyroid` command cleans Parquet and JSONL files without writing any Python. It is the
quickest way to try pyroid on your own data:

```console
$ pip install "pyroid[cli]"     # pyarrow and numpy
$ pyroid clean in.parquet out.parquet --recipe fineweb
```

`pyroid clean --recipe fineweb` runs FineWeb's cleaning chain with datatrove 0.10.1's stages,
settings and order. It keeps the same documents, with the same text, as datatrove's own FineWeb
pipeline. The per-document digest of a `pyroid clean` run equals the stock pipeline's on the
10,000-document FineWeb dev slice and on the 13,232-document parity corpus
([parity](../compatibility.md#integrations)).

Files are streamed in Arrow batches, so memory holds about one row group and two batches per
process, whatever the size of the dataset; MinHash dedup sorts its signatures on disk.
`python -m pyroid` is the same program. For thousands of files and many machines, see
[Large datasets](large-datasets.md).

## Commands

```console
$ pyroid --help
usage: pyroid [-h] [--version] COMMAND ...

Clean text datasets for LLM training, in Parquet or JSON Lines files.

positional arguments:
  COMMAND
    clean     run a cleaning recipe (ftfy, LID, quality filters, MinHash; or FineWeb from WARC)
    langid    add fastText language columns; optionally keep some languages
    dedup     drop near-duplicates (datatrove's MinHash) or exact duplicates
    translit  transliterate to ASCII (anyascii data, ISC)
    stats     sum the stats of every finished file of a directory OUT (all shards)

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit

pyroid run [-p NAMES] (SCRIPT | -m MODULE) [ARGS...] runs a Python program with
pyroid.integrations.patch() applied; see 'pyroid run --help'.
```

| Command | What it does |
|---|---|
| [`pyroid clean`](#pyroid-clean-the-fineweb-recipe) | ftfy, fastText language ID, the Gopher, C4 and FineWeb quality filters, MinHash dedup |
| [`pyroid clean --recipe fineweb-warc`](#pyroid-clean-fineweb-from-warc-files) | datatrove's `examples/fineweb.py` from WARC files: the reader, URL filter, Trafilatura, language ID, the same filters, MinHash, PII |
| [`pyroid langid`](#pyroid-langid) | adds `language` / `language_score` columns; optionally keeps some languages |
| [`pyroid dedup`](#pyroid-dedup) | drops near-duplicates (datatrove's MinHash) or exact duplicates |
| [`pyroid translit`](#pyroid-translit) | transliterates a column to ASCII |
| [`pyroid stats`](large-datasets.md#many-machines-shard) | sums the counts of every shard of a directory OUT |
| [`pyroid run`](#pyroid-run) | runs a Python program with `import ftfy` / `import fasttext` (or the modules `-p` names) redirected to pyroid |

## `pyroid clean`: the FineWeb recipe

The `fineweb` recipe is the chain pyroid's benchmark checks against datatrove:

1. `ftfy.fix_text` with ftfy's default config;
2. fastText language ID with `lid.176.bin`: keep English with probability above 0.65
   (datatrove's `LanguageFilter(languages=["en"], language_threshold=0.65)`);
3. `GopherRepetitionFilter()`, `GopherQualityFilter()`,
   `C4QualityFilter(filter_no_terminal_punct=False)` and `FineWebQualityFilter()`, in that order,
   with datatrove's decisions and drop reasons;
4. MinHash dedup over the survivors, in input order, with datatrove's FineWeb settings (sha1,
   64-bit, 14 buckets of 8 hashes, word 5-grams).

On the first 200 documents of the FineWeb dev slice:

```console
$ pyroid clean sample.parquet clean.parquet --recipe fineweb --lid-model lid.176.bin --stats
pyroid clean: 200 in, 170 out, … s
  dropped gopher_quality:gopher_below_alpha_threshold         24   12.0%
  dropped fineweb_quality:line_punct_ratio                     3    1.5%
  dropped gopher_quality:gopher_too_many_end_ellipsis          3    1.5%
```

The wall time is elided here; see [Speed](#speed). The kept rows keep every input column. The
text column holds the cleaned text (after ftfy and C4's line removal). datatrove's `language` and
`language_score` columns are set, replacing columns of that name (FineWeb's own files have both).

```console
$ pyroid clean --help
usage: pyroid clean [-h] [--files-from PATH] [--text-column TEXT_COLUMN] [--threads THREADS]
                    [--batch-size BATCH_SIZE] [--shard I/N] [--workers N] [--merge] [--format FMT]
                    [--excluded PATH] [--stats] [--stats-json PATH] [--progress]
                    [--compression COMPRESSION] [--overwrite] --recipe {fineweb,fineweb-warc}
                    [--dump NAME] [--limit N] [--lid-model LID_MODEL] [--no-dedup]
                    [--tmp-dir TMP_DIR] [--sort-buffer-mb MB]
                    [IN ...] OUT

Run a cleaning recipe and write the kept rows. The fineweb recipe is datatrove's FineWeb chain on
text: ftfy, fastText LID, the Gopher, C4 and FineWeb quality filters, then MinHash dedup. The
fineweb-warc recipe is datatrove's examples/fineweb.py on WARC files: datatrove's WarcReader, URL
filter, Trafilatura, fastText LID, the same filters, MinHash dedup, then PII replacement.

positional arguments:
  IN                    input files (.parquet/.pq or .jsonl/.ndjson/.json[.gz|.bz2|.zst|.lz4];
                        fineweb-warc: .warc, .arc, .wet[.gz]), directories, globs or URIs
  OUT                   output file, or a directory for one output per input file (see --merge)

options:
  -h, --help            show this help message and exit
  --files-from PATH     read more inputs from PATH, one per line
  --text-column TEXT_COLUMN
                        the text column (default: text)
  --threads THREADS     threads per process (default: 0, the whole pool, sized by
                        $PYROID_NUM_THREADS; with --workers, an equal share of the CPUs)
  --batch-size BATCH_SIZE
                        rows per Arrow batch (default: 10000; --recipe fineweb-warc: pages per
                        batch, 256)
  --shard I/N           process input files I, I+N, I+2N, ...
  --workers N           worker processes, one input file each (default: 1)
  --merge               with a directory OUT, write one file per shard (part-I-of-N) instead of
                        one per input file
  --format FMT          with a directory OUT, the output format: parquet, jsonl, jsonl.gz,
                        jsonl.bz2, jsonl.zst, jsonl.lz4 (default: each input's; parquet for WARC
                        inputs)
  --excluded PATH       also write the dropped rows here (a file, or a directory like OUT)
  --stats               print row counts and drops by reason to stderr
  --stats-json PATH     write the stats to PATH as JSON
  --progress            show a progress line on stderr
  --compression COMPRESSION
                        Parquet compression (default: pyarrow's, snappy)
  --overwrite           redo finished work and replace existing outputs
  --recipe {fineweb,fineweb-warc}
                        fineweb (Parquet or JSONL text) or fineweb-warc (WARC, ARC or WET files)
  --dump NAME           fineweb-warc: add a dump column holding NAME (datatrove's
                        default_metadata)
  --limit N             fineweb-warc: read at most N documents per WARC file (WarcReader's limit)
  --lid-model LID_MODEL
                        fastText's lid.176.bin (default: $PYROID_LID_MODEL, else a cached
                        download)
  --no-dedup            skip the MinHash stage (needed with --shard; run the 'pyroid dedup' stages
                        after)
  --tmp-dir TMP_DIR     where rows and signatures wait for the MinHash decision (default: next to
                        OUT)
  --sort-buffer-mb MB   signatures sorted in memory per process before spilling to disk (default:
                        64)
```

`--threads` is pyroid's `threads=` for every stage: omitted or `0` uses the whole thread pool
(sized by `PYROID_NUM_THREADS`), `1` runs on one thread ([Threads](../concepts/threads.md)).
The results do not depend on `--threads`, `--batch-size`, `--workers` or `--shard`. The options
for many files and many machines are described in [Large datasets](large-datasets.md).

### Dropped rows and `filter_reason`

`--excluded PATH` writes every dropped row to a second file, with all its columns (the text as the
dropping stage saw it) plus a `filter_reason` column:

| `filter_reason` | Dropped by |
|---|---|
| `lid` | language ID: not English, or English with probability 0.65 or less |
| `gopher_repetition:<reason>` | `GopherRepetitionFilter`, e.g. `gopher_repetition:dup_line_frac` |
| `gopher_quality:<reason>` | `GopherQualityFilter`, e.g. `gopher_quality:gopher_short_doc` |
| `c4_quality:<reason>` | `C4QualityFilter`, e.g. `c4_quality:too_few_sentences` |
| `fineweb_quality:<reason>` | `FineWebQualityFilter`, e.g. `fineweb_quality:line_punct_ratio` |
| `minhash` | MinHash: a near-duplicate (each cluster keeps the document datatrove keeps) |
| `null` | a null text |

`<reason>` is datatrove's own `filter_reason` string for that filter. The excluded file has the
same format rules as the output (its own suffix decides), so you can keep the output in Parquet
and the dropped rows in JSONL:

```console
$ pyroid clean sample.parquet clean.parquet --recipe fineweb --lid-model lid.176.bin \
    --overwrite --excluded dropped.jsonl --stats-json stats.json
```

### Stats

`--stats` prints the counts to stderr: documents in and out, the wall time, and one line per drop
reason with its share of the input. `--stats-json PATH` writes the same counts as JSON, plus the
counts per stage; for the run above:

```text
{
 "command": "clean",
 "docs_in": 200,
 "docs_out": 170,
 "removed": {
  "gopher_quality:gopher_below_alpha_threshold": 24,
  "fineweb_quality:line_punct_ratio": 3,
  "gopher_quality:gopher_too_many_end_ellipsis": 3
 },
 "removed_by_stage": {
  "gopher_quality": 27,
  "fineweb_quality": 3
 },
 "wall_s": …,
 "pyroid": "1.0.0rc1",
 "files": 1
}
```

`clean`, `langid`, `dedup` (and its four stages) and `translit` take `--stats` and
`--stats-json`.

### The MinHash work directory

Which near-duplicate to drop depends on the whole corpus, so `clean` can't write a survivor until
it has seen every document. Survivors wait in Arrow IPC files, and their MinHash signatures in
datatrove's on-disk format, in a work directory next to OUT (`.OUT.pyroid/` for a file OUT,
`OUT/.pyroid/` for a directory), or under `--tmp-dir`. Then the survivors are written in input
order and the work directory is deleted. Plan disk space there for the survivors plus about 1 KB
of signatures per survivor. If the run fails or is interrupted, the work directory stays, and the
same command resumes from it. Under `--tmp-dir` the work directory is created with mode 0700, and
pyroid refuses to reuse one that is a link, belongs to another user or is writable by others.
`--no-dedup` skips MinHash, and the output is then the chain up to `FineWebQualityFilter`.

## `pyroid clean`: FineWeb from WARC files

The `fineweb-warc` recipe is datatrove's `examples/fineweb.py`, from Common Crawl WARC files to
the deduplicated, PII-replaced documents, with no datatrove executor:

1. the documents datatrove's `WarcReader` reads, as [`pyroid.warc`](warc.md) reads them: HTML
   (and WET text) records, decoded as datatrove decodes them;
2. `URLFilter()`, with datatrove's block lists;
3. `Trafilatura(favour_precision=True)`: [`pyroid.trafilatura`](trafilatura.md) as trafilatura
   1.11.0, the release `datatrove[processing]` installs, with its dedup cache emptied for each
   input file, as datatrove's per-task step empties it;
4. `LanguageFilter(languages=["en"], language_threshold=0.65)` and the four quality filters of
   the `fineweb` recipe (FineWeb has no ftfy step);
5. MinHash dedup over the survivors of every file, in input order, with FineWeb's settings;
6. `PIIFormatter()` on what is left, with a fresh replacement rotation for each file.

Each input file is one datatrove task, so the output does not depend on `--workers`. On two WARC
files of FineWeb pages, the final documents (ids and texts) are those of the benchmark's native
chain reading the files with warcio and datatrove's own `process_record`
(`bench/fineweb_warc.py native --reader warcio`).

```console
$ pip install "pyroid[cli,warc,datatrove]" tldextract fasteners
$ pyroid clean warc/ fineweb.parquet --recipe fineweb-warc --lid-model lid.176.bin \
      --dump CC-MAIN-2026-39 --workers 8 --stats
```

- IN: WARC, ARC or WET files, gzip (one member per record, as Common Crawl writes them) or
  plain. A directory or glob gives its `.warc`, `.arc` and `.wet` files, with or without `.gz`;
  a file named directly is read whatever its name. URIs (`s3://`, `hf://`, ...) are read through
  fsspec.
- OUT: one file (Parquet or JSONL), or a directory with one Parquet file per input
  (`CC-MAIN-...-00000.warc.gz` gives `CC-MAIN-...-00000.parquet`; `--format` picks another).
- Columns: `text`, `id`, `dump` (with `--dump NAME`), `url`, `date`, `file_path`, `language`
  and `language_score`, the metadata datatrove's writer keeps, in its order. One row per kept
  document.
- `--excluded` gets the dropped documents with a `filter_reason`: `url_filter:<reason>`
  (`url_filter:domain`, ...), `trafilatura` (no text extracted), `trafilatura:broken_process`
  (the extraction raised; datatrove's child process dies on such a page), `lid`,
  `<stage>:<datatrove reason>` as for the `fineweb` recipe, and `minhash`. Their `text` is the
  text the dropping stage saw: the HTML for the URL filter and Trafilatura.
- `--limit N` reads at most N documents per file (datatrove's `WarcReader(limit=N)` with one
  task per file). `--batch-size` is the pages per extraction batch (default 256).
- `--no-dedup` skips MinHash, as for `fineweb`. PII replacement still runs, on each file's
  survivors, so later `pyroid dedup` stages see the replaced text (datatrove's MinHash runs before
  PII).
- Needs datatrove (for the URL block lists), tldextract and fasteners, as datatrove's URL filter
  does, and faust-cchardet and python-magic (with the libmagic library) for the decoding, as
  datatrove's reader does. trafilatura, warcio and spaCy are not needed.

On 8 CC-MAIN-2026-39 WARC files (AWS EC2 c7i.4xlarge, `--workers 8`), the recipe took 79.6 s
(600 CPU s), where stock datatrove 0.10.1 took 848.3 s (6,593 CPU s), and kept the same documents
as datatrove with pyroid's steps ([details](warc.md#speed)).

## Inputs and outputs

These rules hold for every command:

- The inputs are files, directories (read recursively, in name order), globs or URIs, as
  [Large datasets](large-datasets.md#inputs) describes. Files without a Parquet or JSONL suffix
  are ignored, and so are names starting with `.` or `_`. Each file is read with its own schema.
- Formats follow the file name, for input and output alike. `.parquet` or `.pq` is Parquet.
  `.jsonl`, `.ndjson` or `.json` is JSON Lines, one object per line, optionally compressed as
  `.gz`, `.bz2`, `.zst` or `.lz4` (e.g. `docs.jsonl.gz`). `--compression` sets the Parquet codec
  (default: pyarrow's, snappy).
- OUT is one file when its name has one of these suffixes, holding every kept row in input order.
  Otherwise it is a directory with one output file per input file, under the same relative name
  ([layouts](large-datasets.md#output-layout)). Every output is written under a hidden temporary
  name and renamed when complete, so a failed or interrupted run never leaves a truncated file.
  An existing OUT file is not replaced unless you pass `--overwrite`, and an output path equal to
  the input path is refused.
- `--text-column` names the text column (default `text`). Every other column is passed through
  unchanged.
- `--batch-size` sets the rows per Arrow batch (default 10,000).

Errors exit with status 1 and a one-line message, for example:

```console
$ pyroid clean sample.parquet clean.parquet --recipe fineweb
pyroid: clean.parquet exists (pass --overwrite to replace it)
$ pyroid clean sample.parquet x.parquet --recipe fineweb --text-column body
pyroid: sample.parquet: no column 'body' (columns: text, id, dump, url, date, file_path, language, language_score, token_count)
```

For errors from reading or writing the files (`FileNotFoundError`, `ValueError`),
`PYROID_DEBUG=1` shows the traceback instead. Bad options exit with status 2 (argparse's usage
error).

A directory of shards in, compressed JSONL out, two threads:

```console
$ pyroid clean shards/ clean.jsonl.zst --recipe fineweb --lid-model lid.176.bin --threads 2
```

## The language ID model

`clean` (`--lid-model`) and `langid` (`--model`) look for fastText's `lid.176.bin` in this order:

1. the path given on the command line;
2. `$PYROID_LID_MODEL`;
3. `lid.176.bin` in pyroid's cache directory: `$PYROID_CACHE_DIR`, else
   `$XDG_CACHE_HOME/pyroid`, else `~/.cache/pyroid`. If it is not there, it is downloaded once
   from `dl.fbaipublicfiles.com` (125 MiB, CC BY-SA 3.0, by Facebook AI Research) and its
   sha256 is checked.

To run without network, download the model once and point `PYROID_LID_MODEL` at it.

## `pyroid langid`

```text
usage: pyroid langid [-h] [--files-from PATH] [--text-column TEXT_COLUMN] [--threads THREADS]
                     [--batch-size BATCH_SIZE] [--shard I/N] [--workers N] [--merge]
                     [--format FMT] [--excluded PATH] [--stats] [--stats-json PATH] [--progress]
                     [--compression COMPRESSION] [--overwrite] [--model MODEL] [--keep LANGS]
                     [--min-score MIN_SCORE] [--top-k TOP_K]
                     [IN ...] OUT
```

Adds fastText's top label (without `__label__`) as `language` and its probability as
`language_score`, replacing columns of those names. Each document is predicted as one line
(newlines become spaces), as datatrove's `LanguageFilter` does. The text column is not changed.

| Option | Meaning |
|---|---|
| `--model MODEL` | a fastText model file (`.bin` or `.ftz`); default: `lid.176.bin`, found as [above](#the-language-id-model) |
| `--keep LANGS` | keep only documents in these languages, comma-separated (`en,fr`); without it, every row but null texts is kept |
| `--min-score MIN_SCORE` | with `--keep`: datatrove's `language_threshold` (default 0.65); a document is kept if one of the languages has a probability above it |
| `--top-k TOP_K` | above 1: also add `top_languages` and `top_language_scores`, lists of the `TOP_K` best labels and their probabilities |

With `--keep`, dropped rows go to `--excluded` with `filter_reason` `lid` (or `null` for a null
text).

```console
$ pyroid langid sample.parquet tagged.parquet --model lid.176.bin --keep en,fr --top-k 3
```

## `pyroid dedup`

```text
usage: pyroid dedup [-h] [--files-from PATH] [--text-column TEXT_COLUMN] [--threads THREADS]
                    [--batch-size BATCH_SIZE] [--shard I/N] [--workers N] [--merge] [--format FMT]
                    [--excluded PATH] [--stats] [--stats-json PATH] [--progress]
                    [--compression COMPRESSION] [--overwrite] [--method {minhash,exact}]
                    [--normalize] [--tmp-dir TMP_DIR] [--n-grams N_GRAMS]
                    [--num-buckets NUM_BUCKETS] [--hashes-per-bucket HASHES_PER_BUCKET]
                    [--seed SEED] [--hash-fc {xxhash,sha1}] [--precision {32,64}]
                    [--sort-buffer-mb MB]
                    [IN ...] OUT
```

| Option | Meaning |
|---|---|
| `--method minhash` | the default: datatrove's MinHash dedup ([`pyroid.dedup`](dedup.md)); one document of each cluster of near-duplicates is kept, the one datatrove keeps |
| `--method exact` | drop exact repeats of an earlier document |
| `--normalize` | with `exact`: compare the texts after datatrove's `simplify_text` normalization |
| `--n-grams`, `--num-buckets`, `--hashes-per-bucket`, `--seed`, `--hash-fc`, `--precision` | datatrove `MinhashConfig` fields; each defaults to datatrove's default (xxhash, 64-bit, 5-grams, 14 × 8, seed 1) |

The `filter_reason` of a dropped row is the method's name (`minhash` or `exact`). `--method
minhash` reads the input twice (signatures first, then the rows), so it must not change during the
run; the signatures are sorted on disk, in the work directory. FineWeb's own settings are
`--hash-fc sha1`. `--method exact` runs in one process over all inputs in order and writes one
file. Across machines, run datatrove's four stages one by one: `pyroid dedup signatures`,
`buckets`, `cluster` and `filter` ([Large datasets](large-datasets.md#minhash-dedup-across-shards)).

```console
$ pyroid dedup dups.parquet dedup.jsonl --method exact --stats --excluded dups-out.jsonl
pyroid dedup: 3 in, 2 out, … s
  dropped exact          1   33.3%
$ cat dedup.jsonl dups-out.jsonl
{"id": "a", "text": "Hello, world!"}
{"id": "b", "text": "hello world"}
{"id": "c", "text": "Hello, world!", "filter_reason": "exact"}
```

With `--normalize`, `"hello world"` counts as a repeat of `"Hello, world!"` too.

## `pyroid translit`

```text
usage: pyroid translit [-h] [--files-from PATH] [--text-column TEXT_COLUMN] [--threads THREADS]
                       [--batch-size BATCH_SIZE] [--shard I/N] [--workers N] [--merge]
                       [--format FMT] [--stats] [--stats-json PATH] [--progress]
                       [--compression COMPRESSION] [--overwrite] [--mode {anyascii,unidecode}]
                       [--errors ERRORS] [--replace-str REPLACE_STR]
                       [--output-column OUTPUT_COLUMN]
                       [IN ...] OUT
```

Transliterates the text column to ASCII with [`pyroid.translit`](translit.md) (anyascii's data,
ISC). `--mode anyascii` is the default. `--mode unidecode` takes Unidecode's `--errors`
(`ignore`, `replace`, `strict` or `preserve`; default `ignore`) and `--replace-str` (default
`?`). `--output-column` writes a new column instead of replacing the text. No row is dropped, so
there is no `--excluded`.

```console
$ pyroid translit names.parquet ascii.jsonl --text-column name --output-column name_ascii
$ cat ascii.jsonl
{"name": "Björk Guðmundsdóttir", "name_ascii": "Bjork Gudmundsdottir"}
{"name": "北京", "name_ascii": "BeiJing"}
{"name": "Ελλάδα", "name_ascii": "Ellada"}
```

## `pyroid run`

`pyroid run [-p NAMES] (SCRIPT | -m MODULE) [ARGS...]` runs a Python program with
`import ftfy` / `import fasttext` giving pyroid's drop-ins, in the program and in the Python
processes it starts; `-p` picks the modules from `ftfy`, `fasttext`, `anyascii`, `datasketch`,
`trafilatura` and `nltk`. It is described with
[`pyroid.patch()`](patch.md#pyroid-run-programs-and-their-child-processes).

## From Python: `pyroid.integrations.recipes`

`pyroid clean` is a thin driver around `pyroid.integrations.recipes.FineWeb`, which you can use
directly on Arrow batches (it needs the same `pyroid[cli]` extra). `annotate` runs every stage
but the dedup on one batch and returns Arrow columns, without building a Python `str` per
document. Dedup is a property of the whole corpus. Feed each batch's survivors, in order, to the
index from `new_dedup`, and read its keep mask at the end.

```python
import pyarrow.parquet as pq

from pyroid.integrations.recipes import FineWeb

recipe = FineWeb(lid_model="lid.176.bin")
recipe.stages()
# ['lid', 'gopher_repetition', 'gopher_quality', 'c4_quality', 'fineweb_quality', 'minhash']
dedup = recipe.new_dedup()
survivors = []
for batch in pq.ParquetFile("sample.parquet").iter_batches(batch_size=100, columns=["text"]):
    ann = recipe.annotate(batch.column("text"))
    kept = ann.text.filter(ann.keep)  # the cleaned text of the survivors
    dedup.add(kept)
    survivors += kept.to_pylist()
keep = dedup.keep_mask()  # False for MinHash's near-duplicates
clean = [t for t, k in zip(survivors, keep) if k]
len(clean)  # 170
```

`annotate` returns an `Annotation` of five Arrow columns, one value per input document:

- `text`, as it leaves the chain; for a dropped document, the text the dropping stage saw;
- `keep`;
- `reason`: `None` or a [`filter_reason`](#dropped-rows-and-filter_reason) value other than
  `minhash`;
- `language` and `language_score`.

```python
ann = FineWeb(lid_model="lid.176.bin", minhash=None).annotate(["Guten Morgen!", None])
ann.keep.to_pylist()  # [False, False]
ann.reason.to_pylist()  # ['lid', 'null']
```

`FineWeb`'s keyword arguments change the chain:

- `languages` (default `("en",)`; `None` for any) and `language_threshold` (0.65);
- `ftfy_config`: a `TextFixerConfig`, `None` for ftfy's defaults, or `False` to skip ftfy;
- `filters`: `pyroid.filters` configs or datatrove filter instances, in order;
- `minhash`: a dict of `MinhashConfig` fields, a `MinhashConfig`, or `None` for no dedup;
- `lid_model`: a path, a loaded `pyroid.fasttext.Model`, or `None` for the lookup
  [above](#the-language-id-model), loaded on first use;
- `threads`.

```python
import pyroid.filters as F
from pyroid.integrations.recipes import FineWeb

recipe = FineWeb(lid_model="lid.176.bin", filters=[F.GopherQuality(min_doc_words=10)])
recipe.stages()  # ['lid', 'gopher_quality', 'minhash']
```

`lid_model_path(path=None, *, download=True)` is the model lookup on its own, and
`MINHASH_CONFIG` holds FineWeb's MinHash settings. The [Hugging Face helpers](huggingface.md) run
the same recipe over a `datasets.Dataset`.

`FineWebWarc` holds the pieces of the `fineweb-warc` recipe: `documents(path, limit=-1)` (the
reader), `new_task()` (a `TaskExtractor`, one per WARC file: datatrove's Trafilatura step with
its own dedup cache), `annotate(docs, task)` (the URL filter, Trafilatura, LID and the quality
filters on one batch of documents, as Python lists), `new_dedup()` and `new_pii()` (one per file):

```python
from pyroid.integrations.recipes import FineWebWarc

recipe = FineWebWarc(lid_model="lid.176.bin")
task = recipe.new_task()
docs = list(recipe.documents("CC-MAIN-20260920-00000.warc.gz", limit=100))
ann = recipe.annotate(docs, task)
kept = [(d["id"], t) for d, t, k in zip(docs, ann.text, ann.keep) if k]
```


## Speed

On the FineWeb dev slice (10,000 documents), `pyroid clean --recipe fineweb` with 8 threads took
8.7 s and 2.2 s in two runs. Stock datatrove 0.10.1's FineWeb pipeline with 8 processes, run in
the same minute, took 498.8 s and 213.4 s, so `pyroid clean` used 58× and 99× less wall time.
With one thread it took 11.4 s and 7.5 s (44× and 29× against the 8-process runs). These runs
were on a Docker/OrbStack VM on Apple Silicon, indicative. The VM was heavily shared (load
average 18 to 30), which inflates stock datatrove most, so each ratio comes from a same-minute
pair (stock, then pyroid). pyroid's times include interpreter start-up, IO and writing both
outputs. See [Benchmarks](../benchmarks.md#other-modules-languages-and-integrations).

The benchmark's own timings on an AWS c7i.4xlarge (Intel Xeon Platinum 8488C) are the better
guide to an unloaded machine. They time the same chain run natively in one process (the
benchmark's `chain.py`, not the command line), with identical output to datatrove. On the dev
slice it took 1.95 s with 8 threads, 18× faster than datatrove at 8 processes. On 100,000
FineWeb documents it took 10.2 s, about 27× faster. See
[Benchmarks](../benchmarks.md#end-to-end-the-fineweb-chain).

API: [`pyroid.integrations.recipes` reference](../reference/integrations.md#recipes).
