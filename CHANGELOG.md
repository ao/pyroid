# Changelog

All notable changes to pyroid are documented here. From 1.0.0 on, [release-please](https://github.com/googleapis/release-please) maintains this file from [Conventional Commits](https://www.conventionalcommits.org/).

## [1.1.0](https://github.com/ao/pyroid/compare/v1.0.1...v1.1.0) (2026-11-24)


### Features

* **dateutil:** pyroid.dateutil.parser, a drop-in for python-dateutil's parse and isoparse ([#22](https://github.com/ao/pyroid/issues/22)) ([37c5ff6](https://github.com/ao/pyroid/commit/37c5ff6dd95104c20ee741f823b62302a7b0f347))
* **difflib:** pyroid.difflib, the standard library's difflib in Rust ([#23](https://github.com/ao/pyroid/issues/23)) ([b0f850f](https://github.com/ao/pyroid/commit/b0f850f3403103151c491f471c592841050aefa8))
* **jmespath:** pyroid.jmespath, a drop-in for jmespath 1.1.0 ([#20](https://github.com/ao/pyroid/issues/20)) ([e51a6dd](https://github.com/ao/pyroid/commit/e51a6dd22a9726e7a601596364353532d5d7c413))
* **langdetect:** pyroid.langdetect, a drop-in for langdetect 1.0.9 ([#29](https://github.com/ao/pyroid/issues/29)) ([445a4ec](https://github.com/ao/pyroid/commit/445a4ec28915ad29caddd56e0e137e05116adc62))
* match json-repair 0.64.9, warn when an installed incumbent is another release, watch upstream weekly ([#15](https://github.com/ao/pyroid/issues/15)) ([675b0f9](https://github.com/ao/pyroid/commit/675b0f921121c4587e47bb364ead80a2a2229b2d))
* **packaging:** pyroid.packaging, a drop-in for packaging 26.3 ([#24](https://github.com/ao/pyroid/issues/24)) ([b3b3539](https://github.com/ao/pyroid/commit/b3b3539cccf21beb4661780284f5b5100560b957))
* pyroid.tabulate, a drop-in for tabulate 0.10.0 ([#19](https://github.com/ao/pyroid/issues/19)) ([1d1e13b](https://github.com/ao/pyroid/commit/1d1e13befb319218787fdf706f77bde9a5d34372))
* **sqlparse:** pyroid.sqlparse, a drop-in for sqlparse 0.6.1 ([#27](https://github.com/ao/pyroid/issues/27)) ([1c0301f](https://github.com/ao/pyroid/commit/1c0301f9303f1fe52e27a72e915676f1c9f731e8))
* **yaml:** add pyroid.yaml, a drop-in for PyYAML's safe loader and dumper ([#17](https://github.com/ao/pyroid/issues/17)) ([ee062f7](https://github.com/ao/pyroid/commit/ee062f7422970b8bfdab7a4aa7ba9707f9d51e06))


### Bug Fixes

* containers changed by other threads during a call, Ctrl-C in minhash_dedup, pyroid run without pyarrow ([#21](https://github.com/ao/pyroid/issues/21)) ([1f9e0d9](https://github.com/ao/pyroid/commit/1f9e0d94894b6cbffb02ca672d9ddae5eb7f5a04))
* Ctrl-C and SIGTERM, sentence dedup, datasketch pickles, memory and the rest of the outside evaluation ([#18](https://github.com/ao/pyroid/issues/18)) ([3dc300b](https://github.com/ao/pyroid/commit/3dc300b6753510607e3eb3dbda2c37d2f16289d6))
* the outside evaluation's findings (json_repair Arrow output, patched anyascii, first-run friction, fastText speed) ([#16](https://github.com/ao/pyroid/issues/16)) ([1455d38](https://github.com/ao/pyroid/commit/1455d38ca795043195458a9e2f06f8b71f347817))
* Windows open errors, Ctrl-C during patch(), pyroid.Column as a sequence, CLI and json_repair fixes ([#36](https://github.com/ao/pyroid/issues/36)) ([3131d11](https://github.com/ao/pyroid/commit/3131d11cabb0598206a966299e23b26edf03d128))


### Performance Improvements

* **cli:** compress clean's spooled rows on a background thread ([#14](https://github.com/ao/pyroid/issues/14)) ([e02798d](https://github.com/ao/pyroid/commit/e02798dac80b505fb902f2acb3ff7d37c41e2776))

## [1.0.1](https://github.com/ao/pyroid/compare/v1.0.0...v1.0.1) (2026-11-14)


### Bug Fixes

* **thai:** error instead of panic on huge texts; warn on another pythainlp version ([#11](https://github.com/ao/pyroid/issues/11)) ([132c60f](https://github.com/ao/pyroid/commit/132c60f6d964e7eb84f36a5013997b74229b3f4e))
* **warc:** damaged brotli payloads (`Content-Encoding: br`) now decode exactly as brotlicffi 1.2.1.0 does; brotli-decompressor 6.3.1 fixes the 41 cases where pyroid decoded further ([#9](https://github.com/ao/pyroid/issues/9))

## [1.0.0](https://github.com/ao/pyroid/compare/v0.7.0...v1.0.0) (2026-11-13)

pyroid 1.0 is a rewrite. It has Rust versions of popular Python libraries, each with the API and
output of the release named below, and most are many times faster. The end-to-end benchmarks are
on cleaning LLM training data.

The 0.7.x API is removed. 0.7 added general-purpose helpers that mostly ran in Python; 1.0 speeds
up code that uses the libraries it replaces. 0.7.x stays installable with `pip install "pyroid<1"`,
and its source is on the [`legacy`](https://github.com/ao/pyroid/tree/legacy) branch.
[Migrating from 0.7](https://ataiva.com/open-source/pyroid/docs/migrating-from-0.7/) lists replacements.

### Modules

| Module | Replaces |
|---|---|
| `pyroid.ftfy` | ftfy 6.3.1 |
| `pyroid.fasttext` | fastText 0.9.2's inference, as fasttext-predict 0.9.2.4 runs it, on `.bin` and `.ftz` models (no training) |
| `pyroid.filters` | datatrove 0.10.1's `GopherRepetitionFilter`, `GopherQualityFilter`, `C4QualityFilter`, `FineWebQualityFilter`, `PIIFormatter`, `C4BadWordsFilter`, `SymbolLinesFormatter` and `URLFilter` |
| `pyroid.dedup` | datatrove 0.10.1's MinHash dedup and sentence dedup, plus exact dedup |
| `pyroid.trafilatura` | Trafilatura 2.3.1's text extraction, and 1.11.0's with `compat="1.11.0"`, including libxml2 2.14.6's HTML parser as lxml 6.1.3 drives it, and jusText 3.0.2 |
| `pyroid.translit` | anyascii 0.3.3; `unidecode()` has Unidecode's signature and `errors=` handling, with anyascii's output |
| `pyroid.metrics` | rouge-score 0.1.2; sacrebleu 2.6.0's BLEU, chrF and chrF++ |
| `pyroid.json_repair` | json_repair 0.64.5 |
| `pyroid.datasketch` | datasketch 2.0.0's `MinHash`, `LeanMinHash`, `MinHashLSH`, `MinHashLSHForest` and `MinHashLSHEnsemble` |
| `pyroid.jieba` | jieba 0.42.1's word segmentation, with its default dictionary and user dictionaries |
| `pyroid.nltk` | NLTK 3.10.4's Punkt sentence tokenizer, `word_tokenize`, the Treebank tokenizers and detokenizer, and the common regexp tokenizers |
| `pyroid.warc` | warcio 1.8.2's WARC and ARC reader (`ArchiveIterator`), and datatrove 0.10.1's WARC reading, including CPython's `gzip.GzipFile` read path |
| `pyroid.nemo` | NeMo Curator 1.4.0's heuristic document filters and string modifiers, as functions that don't need NeMo |

The filters and dedup tokenize 4,049 of datatrove's 4,343 language codes natively, with
datatrove's tokenizer for each: spaCy's rule-based tokenizers, IndicNLP, the whitespace tokenizer,
spaCy's Chinese tokenizer over jieba, datatrove's Japanese tokenizer over SudachiPy 0.7.0 (with
an installed sudachidict-core dictionary), spaCy's Vietnamese tokenizer over pyvi 0.1.1, and
datatrove's Thai tokenizer over PyThaiNLP 5.4.0 (with an installed pythainlp). The other codes run
datatrove's own tokenizer with `python_tokenizer=True`.

### Integrations

- The `pyroid` command line: `clean` (datatrove's FineWeb pipeline over Parquet or JSONL, or
  from WARC files with `--recipe fineweb-warc`), `langid`, `dedup`, `translit`, `stats` and
  `run`. Inputs can be many files, directories, globs, and `s3://`, `gs://`, `hf://` or other
  fsspec URIs. `--workers` and `--shard I/N` split the work, and runs resume after a stop.
  `pyroid dedup signatures`, `buckets`, `cluster` and `filter` run MinHash dedup across shards in
  datatrove's file format.
- `pyroid.patch()` and `pyroid run -p`: `import ftfy`, `fasttext`, `anyascii`, `datasketch`,
  `trafilatura`, `nltk`, `warcio` and `nemo_curator` give pyroid's functions.
- `pyroid.integrations.datatrove`: 17 drop-in datatrove steps with datatrove's names and
  arguments, `Trafilatura` and `WarcReader` among them.
- `pyroid.integrations.nemo_curator`: NeMo Curator 1.4.0's CPU text stages (`ScoreFilter`,
  `Score`, `Filter`, `Modify`, the 23 heuristic filters, the two fastText filters,
  `UnicodeReformatter` and the string modifiers), subclassing NeMo's classes with the same
  arguments.
- `pyroid.integrations.hf` (Hugging Face `datasets`) and `pyroid.integrations.recipes.FineWeb`.

### Shared behaviour

- Batch functions take a `str`, a `list` or `tuple` of `str`, or an Arrow string column, and
  return the same kind. They release the GIL and run on one thread pool (`threads=`).
- An option or input a module does not port goes to the incumbent library with a
  `pyroid.FallbackWarning`, or raises.
- Wheels for CPython 3.11 to 3.15 and free-threaded 3.14t and 3.15t, on Linux (glibc and musl),
  macOS and Windows, x86-64 and ARM64. No runtime dependencies; the extras are `[fasttext]`,
  `[datasketch]`, `[cli]`, `[datatrove]`, `[hf]`, `[warc]`, `[nemo]` and `[all]`.
- pyroid's own code is MIT, and the ported parts keep their licences (NOTICE lists them). The
  licence expression is
  `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND PSF-2.0 AND Unicode-3.0 AND Zlib`.
