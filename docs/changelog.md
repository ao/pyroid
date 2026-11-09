# Changelog

The changelog is kept in
[`CHANGELOG.md`](https://github.com/ao/pyroid/blob/main/CHANGELOG.md) in the repository, and
release notes are on the [GitHub releases](https://github.com/ao/pyroid/releases) page. From 1.0
on, both are generated from [Conventional Commits](https://www.conventionalcommits.org/) by
release-please.

## 1.0.0 (unreleased)

pyroid 1.0 is a rewrite: Rust ports of the Python libraries used to clean LLM training data and
score model output, each with the API and the output of the release it names. The 0.7.x API is
removed. It stays installable as `pip install "pyroid<1"`, its source is on the
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

The filters and dedup tokenize
[4,013 of datatrove's 4,343 language codes](guides/languages.md) natively, with datatrove's
tokenizer for each: spaCy's rule-based tokenizers, IndicNLP, the whitespace tokenizer, spaCy's
Chinese tokenizer over jieba, and datatrove's Japanese tokenizer over SudachiPy 0.7.0 (with an
installed sudachidict-core dictionary). The other codes run datatrove's own tokenizer with
`python_tokenizer=True`.

### Integrations

- The [`pyroid` command line](guides/cli.md): `clean` (datatrove's FineWeb pipeline over Parquet
  or JSONL), `langid`, `dedup`, `translit`, `stats` and `run`. Inputs can be many files,
  directories, globs, and `s3://`, `gs://`, `hf://` or other fsspec URIs; `--workers` and
  `--shard I/N` split the work ([Large datasets](guides/large-datasets.md)); runs resume after a
  stop; `pyroid dedup signatures`, `buckets`, `cluster` and `filter` run MinHash dedup across
  shards in datatrove's file format.
- [`pyroid.patch()`](guides/patch.md) and `pyroid run -p`: `import ftfy`, `fasttext`, `anyascii`,
  `datasketch`, `trafilatura` and `nltk` give pyroid's functions.
- [`pyroid.integrations.datatrove`](guides/datatrove.md): 16 drop-in datatrove steps with
  datatrove's names and arguments, `Trafilatura` among them.
- [`pyroid.integrations.hf`](guides/huggingface.md) (Hugging Face `datasets`) and
  `pyroid.integrations.recipes.FineWeb`.

### Shared behaviour

- Batch functions take a `str`, a `list` or `tuple` of `str`, or an Arrow string column, and
  return the same kind. They release the GIL and run on one thread pool (`threads=`).
- An option or input a module does not port goes to the incumbent library with a
  `pyroid.FallbackWarning`, or raises.
- Wheels for CPython 3.11 to 3.15 and free-threaded 3.14t and 3.15t, on Linux (glibc and musl),
  macOS and Windows, x86-64 and ARM64. No runtime dependencies; the extras are `[fasttext]`,
  `[datasketch]`, `[cli]`, `[datatrove]` and `[hf]`.
- Licence expression `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND PSF-2.0
  AND Unicode-3.0`: pyroid's own code is MIT, and the ported parts keep their licences
  ([NOTICE](https://github.com/ao/pyroid/blob/main/NOTICE) lists them).
