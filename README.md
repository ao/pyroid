# pyroid

[![PyPI](https://img.shields.io/pypi/v/pyroid)](https://pypi.org/project/pyroid/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyroid)](https://pypi.org/project/pyroid/)
[![License](https://img.shields.io/pypi/l/pyroid)](https://github.com/ao/pyroid/blob/main/LICENSE)
[![CI](https://github.com/ao/pyroid/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ao/pyroid/actions/workflows/ci.yml)

[Docs](https://ao.github.io/pyroid/) | [Benchmarks](https://ao.github.io/pyroid/benchmarks/) | [Compatibility](https://ao.github.io/pyroid/compatibility/)

pyroid is a Rust implementation of the Python libraries used to clean LLM training data and score
model output, with the same API and the same output.

From Common Crawl WARC files to deduplicated documents, datatrove's FineWeb pipeline runs 8.4×
faster with pyroid's steps: 104 s instead of 871 s on 8 workers, for 171,595 pages in 8 WARC files
of CC-MAIN-2026-39 on AWS EC2 c7i.4xlarge
([benchmarks](https://ao.github.io/pyroid/benchmarks/#from-warc-files-datatroves-fineweb-pipeline)).
It keeps the same documents, apart from the pages stock datatrove drops at its 1 s extraction
timeout.

On 100,000 FineWeb documents, with the text already extracted, pyroid's cleaning chain is 27×
faster than datatrove on 8 processes and 56× on one (also AWS EC2 c7i.4xlarge).

```console
$ pip install "pyroid[cli]"
$ pyroid clean in.parquet out.parquet --recipe fineweb   # datatrove's FineWeb pipeline
```

pyroid 1.0 is not on PyPI yet. Until it is, `pip install pyroid` installs the unrelated 0.7.x
toolkit ([see below](#the-07x-releases)).

In Python, change the import and keep the rest of your code:

```python
import pyroid.ftfy as ftfy  # was: import ftfy

ftfy.fix_text("naÃ¯ve rÃ©sumÃ©")  # 'naïve résumé', as ftfy returns it
ftfy.fix_text_batch(["naÃ¯ve rÃ©sumÃ©", "ok"])  # ['naïve résumé', 'ok'], on all cores
```

| Module | Replaces |
|---|---|
| [`pyroid.ftfy`](https://ao.github.io/pyroid/guides/ftfy/) | ftfy 6.3.1 |
| [`pyroid.fasttext`](https://ao.github.io/pyroid/guides/fasttext/) | fastText inference (fasttext, fasttext-wheel, fasttext-predict) |
| [`pyroid.filters`](https://ao.github.io/pyroid/guides/filters/) | datatrove 0.10.1's Gopher, C4, FineWeb, PII, C4 bad-words, symbol-lines and URL filters |
| [`pyroid.dedup`](https://ao.github.io/pyroid/guides/dedup/) | datatrove 0.10.1's MinHash and sentence dedup |
| [`pyroid.trafilatura`](https://ao.github.io/pyroid/guides/trafilatura/) | Trafilatura's text extraction: 2.3.1, or 1.11.0 as datatrove 0.10.1 installs it (with lxml 6.1.3's HTML parser and jusText 3.0.2) |
| [`pyroid.translit`](https://ao.github.io/pyroid/guides/translit/) | anyascii; a permissively licensed alternative to Unidecode |
| [`pyroid.metrics`](https://ao.github.io/pyroid/guides/metrics/) | rouge-score 0.1.2; sacrebleu 2.6.0's BLEU, chrF and chrF++ |
| [`pyroid.json_repair`](https://ao.github.io/pyroid/guides/json_repair/) | json_repair 0.64.5 |
| [`pyroid.datasketch`](https://ao.github.io/pyroid/guides/datasketch/) | datasketch 2.0.0's MinHash, LeanMinHash, MinHashLSH, LSH Forest and LSH Ensemble |
| [`pyroid.jieba`](https://ao.github.io/pyroid/guides/jieba/) | jieba 0.42.1 (Chinese word segmentation) |
| [`pyroid.nltk`](https://ao.github.io/pyroid/guides/nltk/) | NLTK 3.10.4's `sent_tokenize` (Punkt), `word_tokenize` and Treebank tokenizers |
| [`pyroid.warc`](https://ao.github.io/pyroid/guides/warc/) | warcio 1.8.2's WARC/ARC reader (`ArchiveIterator`), and datatrove 0.10.1's WARC reading (`WarcReader`) |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ao/pyroid/main/docs/assets/hero-dark.svg">
  <img alt="Wall time to clean the 10,000-document FineWeb dev slice with identical output on AWS EC2 c7i.4xlarge: pyroid takes 1.9 s with 8 threads, 18× faster than datatrove's 35 s with 8 processes, and 5.6 s with 1 thread, 38× faster than datatrove's 213 s with 1 process" src="https://raw.githubusercontent.com/ao/pyroid/main/docs/assets/hero-light.svg">
</picture>

<sub>The FineWeb chain without text extraction (ftfy, fastText language ID, the Gopher, C4 and
FineWeb quality filters, MinHash dedup) on the first 10,000 documents of the pinned FineWeb slice,
with 1 and 8 processes or threads pinned to 8 CPUs. datatrove 0.10.1 (ftfy 6.3.1,
fasttext-predict 0.9.2.4, spaCy 3.8.18) on CPython 3.13.18; pyroid's native chain on free-threaded
CPython 3.15.2. The output is the same document for document. On the 100k corpus the ratios are
27× and 56×.</sub>

## Same output, checked

`pyroid.ftfy` is byte-identical to ftfy 6.3.1. `pyroid.fasttext` gives the same labels and
probabilities as fastText's C++ code on the same CPU architecture. The filters and dedup make
datatrove's keep/drop decisions, with its reasons and dedup files, and `pyroid.trafilatura`
extracts the same text as Trafilatura 2.3.1 or 1.11.0, byte for byte. The metrics match
rouge-score's and sacrebleu's scores to the last digit, `pyroid.json_repair` makes json_repair's
repairs, `pyroid.datasketch` gives datasketch's hash values and LSH query results, and
`pyroid.jieba` and `pyroid.nltk` give jieba's and NLTK's words, sentences and offsets.
`pyroid.warc` reads the same records, documents and errors as warcio 1.8.2 and datatrove's
WARC reader.

Each module is tested against the library it replaces, on pinned real data and with
property-based fuzzing ([parity counts](https://ao.github.io/pyroid/compatibility/)). Every suite
runs on Linux aarch64. The ftfy, fastText, filters, dedup, metrics and json_repair suites also ran
natively on AWS EC2 x86_64 and aarch64 (c7i.4xlarge and c8g.4xlarge), and the Trafilatura pages
on x86_64. macOS and Windows are checked in CI before each release.

Where a module meets an option or input it doesn't port, it hands the call to the real library
with a `pyroid.FallbackWarning`, or raises. It never returns a different result silently.

pyroid's results don't depend on the number of threads or workers, or on how loaded the machine
is. Stock datatrove's do: its Trafilatura step gives up on a page after 1 s, so two runs over the
same WARC files can keep different documents. pyroid has no timeout.

## Differences from the originals

Batch functions take a `str`, a `list` of `str` or an Arrow column (pyarrow, polars, Hugging Face
`datasets`), and return text and masks in the same kind. They release the GIL and run on all
cores, so you use threads instead of worker processes, and one loaded fastText model serves every
thread instead of one copy per process.

Wheels are built for CPython 3.11 to 3.15 and free-threaded 3.14t and 3.15t, on Linux (glibc and
musl), macOS and Windows, for x86-64 and ARM64, and need no compiler. The core has no runtime
dependencies; `pyroid.fasttext` and `pyroid.datasketch` need numpy (`pip install "pyroid[fasttext]"`,
`"pyroid[datasketch]"`).

## Speed by stage

AWS EC2 c7i.4xlarge, the 10,000-document dev slice, pyroid on one thread against the incumbent in
one process:

| pyroid | Against | Speedup |
|---|---|---:|
| `ftfy.fix_text_batch` | ftfy 6.3.1 | 108× |
| `fasttext` `predict`, `lid.176.bin` / `lid.176.ftz` | fasttext-predict 0.9.2.4 | 1.6× / 3.6× |
| `filters`, Gopher, C4 and FineWeb quality | datatrove 0.10.1 | 35–69× |
| `dedup`, MinHash signatures | datatrove 0.10.1 | 34× |
| `translit.anyascii` | anyascii 0.3.3 | 53× |

With pyroid's stages swapped into datatrove's own executor, the chain ran 11.2× faster at 8
processes on the dev slice and 13.6× on 100k documents. From WARC files, where text extraction
takes most of the time, the whole pipeline ran 8.4× faster at 8 workers and 9.0× at one.

The other modules have only been measured in a Docker VM on a Mac so far, so these numbers are
indicative (one thread unless stated):

| pyroid | Against | Speedup |
|---|---|---:|
| `trafilatura.extract`, default options (`compat="2.3.1"` / `"1.11.0"`) | Trafilatura 2.3.1 / 1.11.0 | 7.2× / 8.6× |
| `metrics`, ROUGE | rouge-score 0.1.2 | 59–65× |
| `metrics`, corpus BLEU / chrF | sacrebleu 2.6.0 | 11× / 19× |
| `json_repair.repair_json`, broken JSON | json_repair 0.64.5 | about 30× |
| `datasketch`, MinHash signatures, 1 / 8 threads | datasketch 2.0.0 | 8× / 21× |
| `jieba.lcut`, a batch, 1 / 8 threads | jieba 0.42.1 | 24× / 58× |
| `nltk.sent_tokenize` / `word_tokenize` | NLTK 3.10.4 | 60× / 31× |
| `filters`, FineWeb-2 settings, Chinese / Japanese | datatrove 0.10.1 | 44× / 7.8× |

`pyroid.json_repair` runs in linear time where json_repair takes quadratic time, so its lead grows
with the input. The [benchmarks page](https://ao.github.io/pyroid/benchmarks/) gives each row's
corpus, interpreter and source.

## Integrations

| Entry point | What it does |
|---|---|
| [`pyroid clean`](https://ao.github.io/pyroid/guides/cli/) | datatrove's FineWeb pipeline as one command over Parquet or JSONL, or from WARC files |
| [Large datasets](https://ao.github.io/pyroid/guides/large-datasets/) | `pyroid clean` over thousands of files: worker processes, `--shard I/N` across machines, `s3://`, `gs://` and `hf://` inputs, resumable runs, and MinHash dedup across shards with datatrove's files |
| [`pyroid.patch()`](https://ao.github.io/pyroid/guides/patch/) | makes `import ftfy`, `fasttext`, `anyascii`, `datasketch`, `trafilatura`, `nltk` and `warcio` give pyroid's functions, in code you don't own |
| [datatrove steps](https://ao.github.io/pyroid/guides/datatrove/) | datatrove's pipeline steps with the same names and signatures, `Trafilatura` included |
| [Hugging Face `datasets`](https://ao.github.io/pyroid/guides/huggingface/) | batched callables for `Dataset.map` and `Dataset.filter` |

```python
import pyroid

pyroid.patch("ftfy", "fasttext")  # later imports of ftfy and fasttext get pyroid's
```

The filters and dedup cover 4,013 of datatrove's 4,343 language codes natively, Chinese included,
and Japanese with an installed SudachiDict dictionary (`pip install sudachidict-core`). The rest
go through datatrove's own tokenizers
([languages](https://ao.github.io/pyroid/guides/languages/)).

## The 0.7.x releases

pyroid 1.0 is a rewrite. The previous general-purpose toolkit (0.1 to 0.7) is not carried
forward, and the 1.0 API is incompatible with it.

- 0.7.x stays installable from PyPI: `pip install "pyroid<1"`.
- Its source is kept on the [`legacy`](https://github.com/ao/pyroid/tree/legacy) branch.
- [Migrating from 0.7](https://ao.github.io/pyroid/migrating-from-0.7/) lists replacements.

## Development

The extension is built with [maturin](https://www.maturin.rs/):

```console
$ uv venv && source .venv/bin/activate
$ uv pip install maturin --group test
$ maturin develop --uv     # builds pyroid._pyroid into the venv (editable)
$ pytest python/tests
$ cargo test               # pure-Rust core (crates/pyroid-core)
```

The code is laid out as follows:

- `crates/pyroid-core`: pure Rust, no Python;
- `crates/pyroid-py`: the PyO3 bindings, built as `pyroid._pyroid`;
- `python/pyroid`: the typed Python package;
- `bench/`: the benchmark harness;
- `parity/`: differential tests against the incumbents;
- `docs/`: the documentation.

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

pyroid's own code is MIT. The ported parts keep their licences, so the package's licence
expression is `MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND PSF-2.0 AND Unicode-3.0`:

- Apache-2.0: ported from ftfy, datatrove, rouge-score, NLTK (the Porter stemmer and the
  tokenizers), sacrebleu, Trafilatura, SudachiPy and warcio;
- MIT: ported from spaCy, indic-nlp-library, urduhack, json_repair, datasketch, jieba (with its
  dictionary) and libxml2 (the HTML parser);
- BSD-2-Clause: jusText (Trafilatura's paragraph classifier and its stoplists);
- BSD-3-Clause: numpy's random generator (for C4 bad words), scipy's QUADPACK integration (for
  MinHashLSH's parameters), tldextract and lxml;
- PSF-2.0: CPython's `ipaddress` (IPv6 parsing);
- ISC: anyascii's transliteration data;
- Unicode-3.0: data from the Unicode Character Database.

See [LICENSE](LICENSE), [LICENSES/](LICENSES) and [NOTICE](NOTICE), which lists the ported files.
