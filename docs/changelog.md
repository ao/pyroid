# Changelog

The changelog is kept in
[`CHANGELOG.md`](https://github.com/ao/pyroid/blob/main/CHANGELOG.md) in the repository, and
release notes are on the [GitHub releases](https://github.com/ao/pyroid/releases) page. From 1.0
on, both are generated from [Conventional Commits](https://www.conventionalcommits.org/) by
release-please.

## 1.0.0

pyroid 1.0 is a rewrite: Rust versions of popular Python libraries, each with the API and the
output of the release it names, and most of them many times faster. Cleaning LLM training data is
where it has been measured end to end. The 0.7.x API is removed: 0.7 added general-purpose helpers
that mostly ran in Python, while 1.0 speeds up code that uses the libraries it replaces. 0.7.x
stays installable as `pip install "pyroid<1"`, its source is on the
[`legacy`](https://github.com/ao/pyroid/tree/legacy) branch, and
[Migrating from 0.7](migrating-from-0.7.md) lists replacements.

### Modules

| Module | Replaces |
|---|---|
| [`pyroid.ftfy`](guides/ftfy.md) | ftfy 6.3.1 |
| [`pyroid.fasttext`](guides/fasttext.md) | fastText 0.9.2's inference, as fasttext-predict 0.9.2.4 runs it, on `.bin` and `.ftz` models (no training) |
| [`pyroid.filters`](guides/filters.md) | datatrove 0.10.1's `GopherRepetitionFilter`, `GopherQualityFilter`, `C4QualityFilter`, `FineWebQualityFilter`, `PIIFormatter`, `C4BadWordsFilter`, `SymbolLinesFormatter` and `URLFilter` |
| [`pyroid.dedup`](guides/dedup.md) | datatrove 0.10.1's MinHash dedup and sentence dedup, plus exact dedup |
| [`pyroid.trafilatura`](guides/trafilatura.md) | Trafilatura 2.3.1's text extraction, and 1.11.0's with `compat="1.11.0"`, including libxml2 2.14.6's HTML parser as lxml 6.1.3 drives it, and jusText 3.0.2 |
| [`pyroid.translit`](guides/translit.md) | anyascii 0.3.3; `unidecode()` has Unidecode's signature and `errors=` handling, with anyascii's output |
| [`pyroid.metrics`](guides/metrics.md) | rouge-score 0.1.2; sacrebleu 2.6.0's BLEU, chrF and chrF++ |
| [`pyroid.json_repair`](guides/json_repair.md) | json_repair 0.64.5 |
| [`pyroid.datasketch`](guides/datasketch.md) | datasketch 2.0.0's `MinHash`, `LeanMinHash`, `MinHashLSH`, `MinHashLSHForest` and `MinHashLSHEnsemble` |
| [`pyroid.jieba`](guides/jieba.md) | jieba 0.42.1's word segmentation, with its default dictionary and user dictionaries |
| [`pyroid.nltk`](guides/nltk.md) | NLTK 3.10.4's Punkt sentence tokenizer, `word_tokenize`, the Treebank tokenizers and detokenizer, and the common regexp tokenizers |
| [`pyroid.warc`](guides/warc.md) | warcio 1.8.2's WARC and ARC reader (`ArchiveIterator`), and datatrove 0.10.1's WARC reading, including CPython's `gzip.GzipFile` read path |
| [`pyroid.nemo`](guides/nemo-curator.md#pyroidnemo) | NeMo Curator 1.4.0's heuristic document filters and string modifiers, as functions that don't need NeMo |

The filters and dedup tokenize
[4,049 of datatrove's 4,343 language codes](guides/languages.md) natively, with datatrove's
tokenizer for each: spaCy's rule-based tokenizers, IndicNLP, the whitespace tokenizer, spaCy's
Chinese tokenizer over jieba, datatrove's Japanese tokenizer over SudachiPy 0.7.0 (with an
installed sudachidict-core dictionary), spaCy's Vietnamese tokenizer over pyvi 0.1.1, and
datatrove's Thai tokenizer over PyThaiNLP 5.4.0 (with an installed pythainlp). The other codes
run datatrove's own tokenizer with `python_tokenizer=True`.

### Integrations

- The [`pyroid` command line](guides/cli.md): `clean` (datatrove's FineWeb pipeline over Parquet
  or JSONL, or from WARC files with `--recipe fineweb-warc`), `langid`, `dedup`, `translit`, `stats` and `run`. Inputs can be many files,
  directories, globs, and `s3://`, `gs://`, `hf://` or other fsspec URIs; `--workers` and
  `--shard I/N` split the work ([Large datasets](guides/large-datasets.md)); runs resume after a
  stop; `pyroid dedup signatures`, `buckets`, `cluster` and `filter` run MinHash dedup across
  shards in datatrove's file format.
- [`pyroid.patch()`](guides/patch.md) and `pyroid run -p`: `import ftfy`, `fasttext`, `anyascii`,
  `datasketch`, `trafilatura`, `nltk`, `warcio` and `nemo_curator` give pyroid's functions.
- [`pyroid.integrations.datatrove`](guides/datatrove.md): 17 drop-in datatrove steps with
  datatrove's names and arguments, `Trafilatura` and `WarcReader` among them.
- [`pyroid.integrations.nemo_curator`](guides/nemo-curator.md): NeMo Curator 1.4.0's CPU text stages (`ScoreFilter`, `Score`, `Filter`, `Modify`, the 23
  heuristic filters, the two fastText filters, `UnicodeReformatter` and the string modifiers),
  subclassing NeMo's classes with the same arguments.
- [`pyroid.integrations.hf`](guides/huggingface.md) (Hugging Face `datasets`) and
  `pyroid.integrations.recipes.FineWeb`.

### Shared behaviour

- Batch functions take a `str`, a `list` or `tuple` of `str`, or an Arrow string column, and
  return the same kind. They release the GIL and run on one thread pool (`threads=`).
- An option or input a module does not port goes to the incumbent library with a
  `pyroid.FallbackWarning`, or raises.
- Wheels for CPython 3.11 to 3.15 and free-threaded 3.14t and 3.15t, on Linux (glibc and musl),
  macOS and Windows, x86-64 and ARM64. No runtime dependencies; the extras are `[fasttext]`,
  `[datasketch]`, `[cli]`, `[datatrove]`, `[hf]`, `[warc]`, `[nemo]` and `[all]`.
- Licence expression `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND PSF-2.0
  AND Unicode-3.0 AND Zlib`: pyroid's own code is MIT, and the ported parts keep their
  licences ([NOTICE](https://github.com/ao/pyroid/blob/main/NOTICE) lists them).
