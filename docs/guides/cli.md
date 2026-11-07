# The `pyroid` command line

The `pyroid` command cleans Parquet and JSONL files without writing any Python. It is the
quickest way to try pyroid on your own data:

```console
$ pip install "pyroid[cli]"     # pyarrow and numpy
$ pyroid clean in.parquet out.parquet --recipe fineweb
```

`pyroid clean --recipe fineweb` runs FineWeb's cleaning chain with datatrove 0.10.1's stages,
settings and order, and keeps **the same documents with the same text** as datatrove's own
FineWeb pipeline: the per-document digest of a `pyroid clean` run equals the stock pipeline's on
the 10,000-document FineWeb dev slice and on the 13,232-document parity corpus
([parity](../compatibility.md#integrations)).

Files are streamed in Arrow batches: one batch of documents is in memory at a time (MinHash
dedup also keeps its index of the survivors). `python -m pyroid` is the same program.

## Commands

```console
$ pyroid --help
usage: pyroid [-h] [--version] COMMAND ...

Rust-fast text cleaning for LLM datasets: Parquet/JSONL in and out.

positional arguments:
  COMMAND
    clean     run a cleaning recipe (ftfy, LID, quality filters, MinHash)
    langid    add fastText language columns; optionally keep some languages
    dedup     drop near-duplicates (MinHash, datatrove's) or exact duplicates
    translit  transliterate to ASCII (anyascii data, ISC)

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit

Also: pyroid run [-p ftfy,fasttext] (SCRIPT | -m MODULE) [ARGS...] runs a program with
pyroid.integrations.patch() applied.
```

| Command | What it does |
|---|---|
| [`pyroid clean`](#pyroid-clean-the-fineweb-recipe) | ftfy, fastText language ID, the Gopher, C4 and FineWeb quality filters, MinHash dedup |
| [`pyroid langid`](#pyroid-langid) | adds `language` / `language_score` columns; optionally keeps some languages |
| [`pyroid dedup`](#pyroid-dedup) | drops near-duplicates (datatrove's MinHash) or exact duplicates |
| [`pyroid translit`](#pyroid-translit) | transliterates a column to ASCII |
| [`pyroid run`](#pyroid-run) | runs a Python program with `import ftfy` / `import fasttext` redirected to pyroid |

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

(The wall time is elided here; for speed, see [below](#speed).) The kept rows keep every input
column. The text column holds the cleaned text (after ftfy and C4's line removal), and
datatrove's `language` and `language_score` columns are set, replacing columns of that name
(FineWeb's own files have both).

```console
$ pyroid clean --help
usage: pyroid clean [-h] [--text-column TEXT_COLUMN] [--threads THREADS] [--batch-size BATCH_SIZE]
                    [--excluded PATH] [--stats] [--stats-json PATH] [--compression COMPRESSION]
                    [--overwrite] --recipe {fineweb} [--lid-model LID_MODEL] [--no-dedup]
                    [--tmp-dir TMP_DIR]
                    input output

positional arguments:
  input                 input file (.parquet/.pq or .jsonl/.ndjson/.json[.gz|.bz2|.zst|.lz4]), or
                        a directory of such files
  output                output file (.parquet/.pq or .jsonl/.ndjson/.json[.gz|.bz2|.zst|.lz4])

options:
  -h, --help            show this help message and exit
  --text-column TEXT_COLUMN
                        the text column (default: text)
  --threads THREADS     pyroid threads (default 0: the whole pool, sized by PYROID_NUM_THREADS)
  --batch-size BATCH_SIZE
                        rows per Arrow batch (default 10000)
  --excluded PATH       also write the dropped rows here
  --stats               print kept/dropped counts by reason
  --stats-json PATH     write the stats as JSON
  --compression COMPRESSION
                        Parquet compression (default: pyarrow's, snappy)
  --overwrite           replace an existing output file
  --recipe {fineweb}
  --lid-model LID_MODEL
                        fastText lid.176.bin (default: $PYROID_LID_MODEL or download)
  --no-dedup            skip the MinHash stage
  --tmp-dir TMP_DIR     where survivors wait for the dedup decision (default: OUT's dir)
```

`--threads` is pyroid's `threads=` for every stage: omitted or `0` uses the whole thread pool
(sized by `PYROID_NUM_THREADS`), `1` runs on one thread ([Threads](../concepts/threads.md)).
The results do not depend on `--threads` or `--batch-size`.

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
 "pyroid": "1.0.0.dev0"
}
```

Every command takes `--stats` and `--stats-json`.

### The MinHash spool

Which near-duplicate to drop depends on the whole corpus, so `clean` can't write a survivor until
it has seen every document. Survivors wait in an Arrow IPC file (`.pyroid-*.arrow`) in the output
file's directory, or in `--tmp-dir`, until the MinHash decision is made; then they are written in
input order and the spool is deleted, also when the run fails. Plan disk space for the
survivors there. `--no-dedup` skips MinHash (and the spool): the output is then the chain up to
`FineWebQualityFilter`.

## Inputs and outputs

These rules hold for every command:

- **Input** is a file or a directory. A directory is read as one dataset: every Parquet or JSON
  Lines file in it (by the suffixes below), in name order (other files are ignored; mixing
  Parquet and JSONL is an error). JSONL files after the first are parsed with the first file's
  schema.
- **Formats** follow the file name, for input and output alike: `.parquet` or `.pq` is Parquet;
  `.jsonl`, `.ndjson` or `.json` is JSON Lines, one object per line, optionally compressed as
  `.gz`, `.bz2`, `.zst` or `.lz4` (e.g. `docs.jsonl.gz`). `--compression` sets the Parquet codec
  (default: pyarrow's, snappy).
- **Output** is one file. It is written to `OUT.partial` and renamed to `OUT` when the command
  finishes, so a failed or interrupted run never leaves a truncated `OUT`; the partial file is
  removed on failure. An existing `OUT` is not replaced unless you pass `--overwrite`, and an
  output path equal to the input path is refused.
- **Columns:** `--text-column` names the text column (default `text`); every other column is
  passed through unchanged.
- **Batches:** `--batch-size` rows per Arrow batch (default 10,000), across file boundaries.

Errors exit with status 1 and a one-line message, for example:

```console
$ pyroid clean sample.parquet clean.parquet --recipe fineweb
pyroid: clean.parquet exists (pass --overwrite to replace it)
$ pyroid clean sample.parquet x.parquet --recipe fineweb --text-column body
pyroid: no column 'body' (columns: text, id, dump, url, date, file_path, language, language_score, token_count)
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
usage: pyroid langid [-h] [--text-column TEXT_COLUMN] [--threads THREADS]
                     [--batch-size BATCH_SIZE] [--excluded PATH] [--stats] [--stats-json PATH]
                     [--compression COMPRESSION] [--overwrite] [--model MODEL] [--keep LANGS]
                     [--min-score MIN_SCORE] [--top-k TOP_K]
                     input output
```

Adds fastText's top label (without `__label__`) as `language` and its probability as
`language_score`, replacing columns of those names. Each document is predicted as one line
(newlines become spaces), as datatrove's `LanguageFilter` does. The text column is not changed.

| Option | Meaning |
|---|---|
| `--model MODEL` | a fastText model file (`.bin` or `.ftz`); default: `lid.176.bin`, found as [above](#the-language-id-model) |
| `--keep LANGS` | keep only documents in these languages, comma-separated (`en,fr`); without it, every row but null texts is kept |
| `--min-score MIN_SCORE` | with `--keep`: datatrove's `language_threshold` (default 0.65); a document is kept if one of the languages has a probability above it |
| `--top-k K` | also add `top_languages` and `top_language_scores`: lists of the `K` best labels and their probabilities |

With `--keep`, dropped rows go to `--excluded` with `filter_reason` `lid` (or `null` for a null
text).

```console
$ pyroid langid sample.parquet tagged.parquet --model lid.176.bin --keep en,fr --top-k 3
```

## `pyroid dedup`

```text
usage: pyroid dedup [-h] [--text-column TEXT_COLUMN] [--threads THREADS] [--batch-size BATCH_SIZE]
                    [--excluded PATH] [--stats] [--stats-json PATH] [--compression COMPRESSION]
                    [--overwrite] [--method {minhash,exact}] [--normalize] [--n-grams N_GRAMS]
                    [--num-buckets NUM_BUCKETS] [--hashes-per-bucket HASHES_PER_BUCKET]
                    [--seed SEED] [--hash-fc {xxhash,sha1}] [--precision {32,64}]
                    input output
```

| Option | Meaning |
|---|---|
| `--method minhash` | the default: datatrove's MinHash dedup ([`pyroid.dedup`](dedup.md)); one document of each cluster of near-duplicates is kept, the one datatrove keeps |
| `--method exact` | drop exact repeats of an earlier document |
| `--normalize` | with `exact`: compare the texts after datatrove's `simplify_text` normalization |
| `--n-grams`, `--num-buckets`, `--hashes-per-bucket`, `--seed`, `--hash-fc`, `--precision` | datatrove `MinhashConfig` fields; each defaults to datatrove's default (xxhash, 64-bit, 5-grams, 14 × 8, seed 1) |

The `filter_reason` of a dropped row is the method's name (`minhash` or `exact`). `--method
minhash` reads the input twice (signatures first, then the rows), so it must not change during the
run; nothing is spooled. FineWeb's own settings are `--hash-fc sha1`.

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
usage: pyroid translit [-h] [--text-column TEXT_COLUMN] [--threads THREADS]
                       [--batch-size BATCH_SIZE] [--stats] [--stats-json PATH]
                       [--compression COMPRESSION] [--overwrite] [--mode {anyascii,unidecode}]
                       [--errors ERRORS] [--replace-str REPLACE_STR]
                       [--output-column OUTPUT_COLUMN]
                       input output
```

Transliterates the text column to ASCII with [`pyroid.translit`](translit.md) (anyascii's data,
ISC): `--mode anyascii` (the default) or `--mode unidecode` with Unidecode's `--errors`
(`ignore`, `replace`, `strict`, `preserve`; default `ignore`) and `--replace-str` (default `?`).
`--output-column` writes a new column instead of replacing the text. No row is dropped, so there
is no `--excluded`.

```console
$ pyroid translit names.parquet ascii.jsonl --text-column name --output-column name_ascii
$ cat ascii.jsonl
{"name": "Björk Guðmundsdóttir", "name_ascii": "Bjork Gudmundsdottir"}
{"name": "北京", "name_ascii": "BeiJing"}
{"name": "Ελλάδα", "name_ascii": "Ellada"}
```

## `pyroid run`

`pyroid run [-p NAMES] (SCRIPT | -m MODULE) [ARGS...]` runs a Python program with
`import ftfy` / `import fasttext` (and optionally `import anyascii`) giving pyroid's drop-ins, in
the program and in the Python processes it starts. It is described with
[`pyroid.patch()`](patch.md#pyroid-run-programs-and-their-child-processes).

## From Python: `pyroid.integrations.recipes`

`pyroid clean` is a thin driver around `pyroid.integrations.recipes.FineWeb`, which you can use
directly on Arrow batches (it needs the same `pyroid[cli]` extra). `annotate` runs every stage
but the dedup on one batch and returns Arrow columns, without building a Python `str` per
document. Dedup is a property of the whole corpus: feed each batch's survivors, in order, to the
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

`annotate` returns an `Annotation` of five Arrow columns, one value per input document: `text`
(as it leaves the chain; for a dropped document, the text the dropping stage saw), `keep`,
`reason` (`None` or a [`filter_reason`](#dropped-rows-and-filter_reason) value other than
`minhash`), `language` and `language_score`:

```python
ann = FineWeb(lid_model="lid.176.bin", minhash=None).annotate(["Guten Morgen!", None])
ann.keep.to_pylist()  # [False, False]
ann.reason.to_pylist()  # ['lid', 'null']
```

`FineWeb`'s keyword arguments change the chain: `languages` (default `("en",)`; `None` for any)
and `language_threshold` (0.65), `ftfy_config` (a `TextFixerConfig`, `None` for ftfy's defaults,
`False` to skip ftfy), `filters` (`pyroid.filters` configs or datatrove filter instances, in
order), `minhash` (a dict of `MinhashConfig` fields, a `MinhashConfig`, or `None` for no dedup),
`lid_model` (a path, a loaded `pyroid.fasttext.Model`, or `None` for the lookup
[above](#the-language-id-model), loaded on first use) and `threads`.

```python
import pyroid.filters as F
from pyroid.integrations.recipes import FineWeb

recipe = FineWeb(lid_model="lid.176.bin", filters=[F.GopherQuality(min_doc_words=10)])
recipe.stages()  # ['lid', 'gopher_quality', 'minhash']
```

`lid_model_path(path=None, *, download=True)` is the model lookup on its own, `MINHASH_CONFIG`
FineWeb's MinHash settings. The [Hugging Face helpers](huggingface.md) run the same recipe over a
`datasets.Dataset`.

## Speed

On the FineWeb dev slice (10,000 documents), `pyroid clean --recipe fineweb` with 8 threads took
8.7 s and 2.2 s in two runs, against 498.8 s and 213.4 s for stock datatrove 0.10.1's FineWeb
pipeline with 8 processes run in the same minute: **58× and 99× less wall time**. With one thread
it took 11.4 s and 7.5 s (44× and 29× against the 8-process runs). *Docker/OrbStack VM on Apple
Silicon (12 vCPU), indicative*: the VM was heavily shared (load average 18 to 30), which inflates
stock datatrove most, so each ratio comes from a same-minute pair (stock, then pyroid). pyroid's
times include interpreter start-up, IO and writing both outputs. See
[Benchmarks](../benchmarks.md#datatroves-fineweb-pipeline-with-pyroids-steps-swapped-in).

API: [`pyroid.integrations.recipes` reference](../reference/integrations.md#recipes).
