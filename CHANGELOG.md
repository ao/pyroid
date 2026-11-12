# Changelog

All notable changes to pyroid are documented here. From 1.0.0 on, [release-please](https://github.com/googleapis/release-please) maintains this file from [Conventional Commits](https://www.conventionalcommits.org/).

## 1.0.0 (unreleased)

pyroid 1.0 is a rewrite: Rust versions of popular Python libraries, each with the API and the
output of the release it names, and most of them many times faster. The largest measured gains
are in cleaning LLM training data. The 0.7.x API is removed: 0.7 added general-purpose helpers
that mostly ran in Python, while 1.0 speeds up code that uses the libraries it replaces. 0.7.x
stays installable as `pip install "pyroid<1"`, its source is on the
[`legacy`](https://github.com/ao/pyroid/tree/legacy) branch, and
[Migrating from 0.7](https://ao.github.io/pyroid/migrating-from-0.7/) lists replacements.

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

The filters and dedup tokenize 4,049 of datatrove's 4,343 language codes natively, with
datatrove's tokenizer for each: spaCy's rule-based tokenizers, IndicNLP, the whitespace tokenizer,
spaCy's Chinese tokenizer over jieba, datatrove's Japanese tokenizer over SudachiPy 0.7.0 (with
an installed sudachidict-core dictionary), spaCy's Vietnamese tokenizer over pyvi 0.1.1, and
datatrove's Thai tokenizer over PyThaiNLP 5.4.0 (with an installed pythainlp). The other codes run
datatrove's own tokenizer with `python_tokenizer=True`.

### Integrations

- The `pyroid` command line: `clean` (datatrove's FineWeb pipeline over Parquet or JSONL, or
  from WARC files with `--recipe fineweb-warc`), `langid`, `dedup`, `translit`, `stats` and `run`. Inputs can be many files, directories, globs,
  and `s3://`, `gs://`, `hf://` or other fsspec URIs; `--workers` and `--shard I/N` split the
  work; runs resume after a stop; `pyroid dedup signatures`, `buckets`, `cluster` and `filter` run
  MinHash dedup across shards in datatrove's file format.
- `pyroid.patch()` and `pyroid run -p`: `import ftfy`, `fasttext`, `anyascii`, `datasketch`,
  `trafilatura`, `nltk` and `warcio` give pyroid's functions.
- `pyroid.integrations.datatrove`: 17 drop-in datatrove steps with datatrove's names and
  arguments, `Trafilatura` and `WarcReader` among them.
- `pyroid.integrations.hf` (Hugging Face `datasets`) and `pyroid.integrations.recipes.FineWeb`.

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
  licences (NOTICE lists them).
