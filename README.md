# pyroid

[![PyPI](https://img.shields.io/pypi/v/pyroid)](https://pypi.org/project/pyroid/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyroid)](https://pypi.org/project/pyroid/)
[![License](https://img.shields.io/pypi/l/pyroid)](https://github.com/ao/pyroid/blob/main/LICENSE)
[![CI](https://github.com/ao/pyroid/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ao/pyroid/actions/workflows/ci.yml)

[Docs](https://ataiva.com/open-source/pyroid/docs/) | [Benchmarks](https://ataiva.com/open-source/pyroid/docs/benchmarks/) | [Compatibility](https://ataiva.com/open-source/pyroid/docs/compatibility/)

pyroid reimplements popular Python libraries in Rust, with the same API and the same output. Most
of its modules are many times faster than the library they replace.

```console
$ pip install pyroid
```

Change the import. The output is the same:

```python
import pyroid.ftfy as ftfy  # was: import ftfy
import pyroid.json_repair as json_repair  # was: import json_repair

ftfy.fix_text("naÃ¯ve rÃ©sumÃ©")  # 'naïve résumé'
ftfy.fix_text_batch(["naÃ¯ve rÃ©sumÃ©", "ok"])  # ['naïve résumé', 'ok'], on all cores
json_repair.repair_json("{'a': 1, 'b': [2, 3")  # '{"a": 1, "b": [2, 3]}'
```

| Area | Module | Replaces |
|---|---|---|
| Text and encoding | [`pyroid.ftfy`](https://ataiva.com/open-source/pyroid/docs/guides/ftfy/) | ftfy 6.3.1 |
| | [`pyroid.translit`](https://ataiva.com/open-source/pyroid/docs/guides/translit/) | anyascii 0.3.3; a permissively licensed alternative to Unidecode |
| Language ID and NLP | [`pyroid.fasttext`](https://ataiva.com/open-source/pyroid/docs/guides/fasttext/) | fastText 0.9.2 inference (fasttext, fasttext-wheel, fasttext-predict) |
| | [`pyroid.nltk`](https://ataiva.com/open-source/pyroid/docs/guides/nltk/) | NLTK 3.10.4's `sent_tokenize` (Punkt), `word_tokenize` and Treebank tokenizers |
| | [`pyroid.jieba`](https://ataiva.com/open-source/pyroid/docs/guides/jieba/) | jieba 0.42.1 (Chinese word segmentation) |
| Web and archives | [`pyroid.trafilatura`](https://ataiva.com/open-source/pyroid/docs/guides/trafilatura/) | Trafilatura's text extraction: 2.3.1, or 1.11.0 as datatrove 0.10.1 installs it (with lxml 6.1.3's HTML parser and jusText 3.0.2) |
| | [`pyroid.warc`](https://ataiva.com/open-source/pyroid/docs/guides/warc/) | warcio 1.8.2's WARC/ARC reader (`ArchiveIterator`), and datatrove 0.10.1's WARC reading (`WarcReader`) |
| Dedup and similarity | [`pyroid.datasketch`](https://ataiva.com/open-source/pyroid/docs/guides/datasketch/) | datasketch 2.0.0's MinHash, LeanMinHash, MinHashLSH, LSH Forest and LSH Ensemble |
| | [`pyroid.dedup`](https://ataiva.com/open-source/pyroid/docs/guides/dedup/) | datatrove 0.10.1's MinHash and sentence dedup |
| Quality filters | [`pyroid.filters`](https://ataiva.com/open-source/pyroid/docs/guides/filters/) | datatrove 0.10.1's Gopher, C4, FineWeb, PII, C4 bad-words, symbol-lines and URL filters |
| | [`pyroid.nemo`](https://ataiva.com/open-source/pyroid/docs/guides/nemo-curator/#pyroidnemo) | NeMo Curator 1.4.0's heuristic document filters and string modifiers |
| Evaluation | [`pyroid.metrics`](https://ataiva.com/open-source/pyroid/docs/guides/metrics/) | rouge-score 0.1.2; sacrebleu 2.6.0's BLEU, chrF and chrF++ |
| JSON and YAML | [`pyroid.json_repair`](https://ataiva.com/open-source/pyroid/docs/guides/json_repair/) | json_repair 0.64.5 |
| | [`pyroid.yaml`](https://ataiva.com/open-source/pyroid/docs/guides/yaml/) | PyYAML 6.0.3's `safe_load` and `safe_dump` (its pure-Python `SafeLoader` and `SafeDumper`) |

## Example: cleaning LLM training data

The end-to-end benchmarks run datatrove's FineWeb pipeline. Without text extraction, that is a
chain of ftfy, fastText language ID, the Gopher, C4 and FineWeb quality filters and MinHash dedup.
On 100,000 FineWeb documents on AWS EC2 c7i.4xlarge, pyroid's version of the chain is 27× faster
than datatrove on 8 processes, with the same output.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ao/pyroid/main/docs/assets/hero-dark.svg">
  <img alt="Wall time to clean the 10,000-document FineWeb dev slice with identical output on AWS EC2 c7i.4xlarge: pyroid takes 1.9 s with 8 threads, 18× faster than datatrove's 35 s with 8 processes, and 5.6 s with 1 thread, 38× faster than datatrove's 213 s with 1 process" src="https://raw.githubusercontent.com/ao/pyroid/main/docs/assets/hero-light.svg">
</picture>

<sub>The chain without text extraction, on the first 10,000 documents of the pinned FineWeb slice,
with 1 and 8 processes or threads pinned to 8 CPUs. datatrove 0.10.1 (ftfy 6.3.1,
fasttext-predict 0.9.2.4, spaCy 3.8.18) on CPython 3.13.18; pyroid's native chain on free-threaded
CPython 3.15.2. Same output, document for document. On the 100k corpus the ratios are 27× and
56×.</sub>

`pyroid clean` runs the chain over Parquet or JSONL files:

```console
$ pip install "pyroid[cli]"
$ pyroid clean in.parquet out.parquet --recipe fineweb   # datatrove's FineWeb pipeline
```

pyroid's stages also drop into datatrove's own executor. There the chain runs 11.2× faster at 8
processes on the dev slice, and 13.6× on 100k documents.

From Common Crawl WARC files, the pipeline also extracts the text with Trafilatura. On 8 WARC files
of CC-MAIN-2026-39 (171,595 pages), on a c7i.4xlarge with 8 workers, stock datatrove took 848.3 s.
With pyroid's steps in datatrove it took 89.4 s (9.5×), and `pyroid clean --recipe fineweb-warc`
took 79.6 s (10.7×). Both keep the same documents as stock datatrove, apart from the pages it
drops at its 1 s extraction timeout
([benchmarks](https://ataiva.com/open-source/pyroid/docs/benchmarks/#from-warc-files-datatroves-fineweb-pipeline)).

## Same output, checked

Each module is tested against the library it replaces, on pinned real data and with
property-based fuzzing. The comparisons are exact: `pyroid.ftfy` is byte-identical to ftfy 6.3.1,
and the metrics match rouge-score and sacrebleu to the last digit. fastText's labels and
probabilities are bit-identical to its C++ code on Linux (glibc) and Windows x64; on macOS the
probabilities can differ in the last bits. The
[compatibility page](https://ataiva.com/open-source/pyroid/docs/compatibility/) has the counts for every module,
and each guide lists its known differences.

CI runs the `parity/` suites on Linux x86_64 and aarch64, and most of them have also run natively
on AWS EC2 ([platforms](https://ataiva.com/open-source/pyroid/docs/compatibility/#platforms)). On macOS, Windows
and musl Linux, every release's wheels pass pyroid's test suite before they are published.

Where a module doesn't port an option or input, it hands the call to the real library with a
`pyroid.FallbackWarning`, or raises.

Results don't depend on the number of threads or workers, or on how loaded the machine is. Stock
datatrove's do: its Trafilatura step gives up on a page after 1 s, so two runs over the same WARC
files can keep different documents. pyroid has no timeout.

## Differences from the originals

Batch functions take a `str`, a `list` of `str` or an Arrow column (pyarrow, polars, Hugging Face
`datasets`) and return text and masks in the same kind. They release the GIL and run on all
cores, so threads replace worker processes, and every thread shares one loaded fastText model
instead of a copy per process.

Wheels cover CPython 3.11 to 3.15 and free-threaded 3.14t and 3.15t, on Linux (glibc and musl),
macOS and Windows, for x86-64 and ARM64, so no compiler is needed. The core has no runtime
dependencies; `pyroid.fasttext` and `pyroid.datasketch` need numpy (`pip install "pyroid[fasttext]"`,
`"pyroid[datasketch]"`).

## Speed by module

pyroid on one thread against the library it replaces in one process, on the 10,000-document
FineWeb dev slice, AWS EC2 c7i.4xlarge:

| pyroid | Against | Speedup |
|---|---|---:|
| `ftfy.fix_text_batch` | ftfy 6.3.1 | 108× |
| `fasttext` `predict`, `lid.176.bin` / `lid.176.ftz` | fasttext-predict 0.9.2.4 | 1.6× / 3.6× |
| `filters`, Gopher, C4 and FineWeb quality | datatrove 0.10.1 | 35–69× |
| `dedup`, MinHash signatures | datatrove 0.10.1 | 34× |
| `translit.anyascii` | anyascii 0.3.3 | 53× |

The other modules have only been measured in a Docker VM on a Mac, so treat these as indicative
(one thread unless stated):

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

`pyroid.yaml` was measured on a shared AWS EC2 c8g.16xlarge, also indicative, on 11,878 real
YAML files (343 MB: GitHub Actions workflows, Kubernetes and Helm files, OpenAPI specs, model
cards): `safe_load` runs 6.8× faster than PyYAML's `CSafeLoader` and 44× faster than its default
`SafeLoader`; `safe_dump` 6.1× faster than `CSafeDumper` and 25× faster than `SafeDumper`.

`pyroid.json_repair` stays linear on 13 input families where json_repair is quadratic, so the gap
grows with the input. The [benchmarks page](https://ataiva.com/open-source/pyroid/docs/benchmarks/) has each
row's corpus, interpreter and source.

## Integrations

| Entry point | What it does |
|---|---|
| [`pyroid clean`](https://ataiva.com/open-source/pyroid/docs/guides/cli/) | datatrove's FineWeb pipeline as one command over Parquet or JSONL, or from WARC files |
| [Large datasets](https://ataiva.com/open-source/pyroid/docs/guides/large-datasets/) | `pyroid clean` over thousands of files: worker processes, `--shard I/N` across machines, `s3://`, `gs://` and `hf://` inputs, resumable runs, and MinHash dedup across shards with datatrove's files |
| [`pyroid.patch()`](https://ataiva.com/open-source/pyroid/docs/guides/patch/) | makes `import ftfy`, `fasttext`, `anyascii`, `datasketch`, `trafilatura`, `nltk`, `warcio`, `nemo_curator` and `yaml` give pyroid's functions, in code you don't own |
| [datatrove steps](https://ataiva.com/open-source/pyroid/docs/guides/datatrove/) | datatrove's pipeline steps with the same names and signatures, `Trafilatura` included |
| [NeMo Curator stages](https://ataiva.com/open-source/pyroid/docs/guides/nemo-curator/) | NeMo Curator 1.4.0's CPU text stages (`ScoreFilter`, the heuristic filters, the modifiers) with the same classes and arguments |
| [Hugging Face `datasets`](https://ataiva.com/open-source/pyroid/docs/guides/huggingface/) | batched callables for `Dataset.map` and `Dataset.filter` |

```python
import pyroid

pyroid.patch("ftfy", "fasttext")  # later imports of ftfy and fasttext get pyroid's
```

The filters and dedup tokenize 4,049 of datatrove's 4,343 language codes natively, Chinese and
Vietnamese included. Native Japanese needs `pip install sudachidict-core`, and native Thai
`pip install pythainlp`. The rest go through datatrove's own tokenizers
([languages](https://ataiva.com/open-source/pyroid/docs/guides/languages/)).

## The 0.7.x releases

pyroid 1.0 is a rewrite, and its API is incompatible with 0.1 to 0.7. Those releases tried to make
Python faster with general-purpose helpers, most of which ran in Python. 1.0 replaces existing
libraries exactly, so it speeds up code that uses them, not Python in general.

0.7.x is still on PyPI (`pip install "pyroid<1"`), and its source is on the
[`legacy`](https://github.com/ao/pyroid/tree/legacy) branch.
[Migrating from 0.7](https://ataiva.com/open-source/pyroid/docs/migrating-from-0.7/) lists replacements.

## Development

The extension is built with [maturin](https://www.maturin.rs/):

```console
$ uv venv && source .venv/bin/activate
$ uv pip install maturin --group test
$ maturin develop --uv     # builds pyroid._pyroid into the venv (editable)
$ pytest python/tests
$ cargo test               # pure-Rust core (crates/pyroid-core)
```

Layout:

- `crates/pyroid-core`: pure Rust, no Python;
- `crates/pyroid-py`: the PyO3 bindings, built as `pyroid._pyroid`;
- `python/pyroid`: the typed Python package;
- `bench/`: the benchmark harness;
- `parity/`: differential tests against the incumbents;
- `docs/`: the documentation.

See [CONTRIBUTING.md](https://github.com/ao/pyroid/blob/main/CONTRIBUTING.md).

## License

pyroid's own code is MIT. The ported parts keep their licences, so the package's licence
expression is
`MIT AND Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause AND ISC AND PSF-2.0 AND Unicode-3.0 AND Zlib`:

- Apache-2.0: ported from ftfy, datatrove, rouge-score, NLTK (the Porter stemmer and the
  tokenizers), sacrebleu, Trafilatura, SudachiPy and warcio;
- MIT: ported from fastText, spaCy, indic-nlp-library, urduhack, json_repair, datasketch, jieba
  (with its dictionary), libxml2 (the HTML parser) and PyYAML;
- BSD-2-Clause: jusText (Trafilatura's paragraph classifier and its stoplists);
- BSD-3-Clause: numpy's random generator (for C4 bad words), scipy's QUADPACK integration (for
  MinHashLSH's parameters), tldextract and lxml;
- PSF-2.0: CPython's `ipaddress` (IPv6 parsing);
- ISC: anyascii's transliteration data;
- Unicode-3.0: data from the Unicode Character Database;
- Zlib: zlib-rs, the Rust crate behind `pyroid.warc`'s gzip decoding.

See [LICENSE](https://github.com/ao/pyroid/blob/main/LICENSE),
[LICENSES/](https://github.com/ao/pyroid/tree/main/LICENSES) and
[NOTICE](https://github.com/ao/pyroid/blob/main/NOTICE), which lists the ported files.
[LICENSES/THIRD-PARTY-RUST.txt](https://github.com/ao/pyroid/blob/main/LICENSES/THIRD-PARTY-RUST.txt)
has the licences of the Rust crates the extension links.
