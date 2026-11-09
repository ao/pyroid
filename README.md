# pyroid

[![PyPI](https://img.shields.io/pypi/v/pyroid)](https://pypi.org/project/pyroid/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyroid)](https://pypi.org/project/pyroid/)
[![License](https://img.shields.io/pypi/l/pyroid)](https://github.com/ao/pyroid/blob/main/LICENSE)
[![CI](https://github.com/ao/pyroid/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ao/pyroid/actions/workflows/ci.yml)

[Docs](https://ao.github.io/pyroid/) | [Benchmarks](https://ao.github.io/pyroid/benchmarks/) | [Compatibility](https://ao.github.io/pyroid/compatibility/)

pyroid is a Rust implementation of the Python libraries used to clean LLM training data and score
model output, with the same API and the same output.

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
| [`pyroid.translit`](https://ao.github.io/pyroid/guides/translit/) | anyascii; a permissively licensed alternative to Unidecode |
| [`pyroid.metrics`](https://ao.github.io/pyroid/guides/metrics/) | rouge-score 0.1.2; sacrebleu 2.6.0's BLEU, chrF and chrF++ |
| [`pyroid.json_repair`](https://ao.github.io/pyroid/guides/json_repair/) | json_repair 0.64.5 |
| [`pyroid.datasketch`](https://ao.github.io/pyroid/guides/datasketch/) | datasketch 2.0.0's MinHash, LeanMinHash, MinHashLSH, LSH Forest and LSH Ensemble |
| [`pyroid.jieba`](https://ao.github.io/pyroid/guides/jieba/) | jieba 0.42.1 (Chinese word segmentation) |

On 100,000 FineWeb documents, pyroid runs datatrove's FineWeb cleaning chain about 27× faster
than datatrove on 8 processes and 56× faster on one thread, with identical output (AWS EC2
c7i.4xlarge; [benchmarks](https://ao.github.io/pyroid/benchmarks/)).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ao/pyroid/main/docs/assets/hero-dark.svg">
  <img alt="Wall time to clean the 10,000-document FineWeb dev slice with identical output on AWS EC2 c7i.4xlarge: pyroid takes 1.9 s with 8 threads, 18× faster than datatrove's 35 s with 8 processes, and 5.6 s with 1 thread, 38× faster than datatrove's 213 s with 1 process" src="https://raw.githubusercontent.com/ao/pyroid/main/docs/assets/hero-light.svg">
</picture>

<sub>The FineWeb chain (ftfy, fastText language ID, the Gopher, C4 and FineWeb quality filters,
MinHash dedup) on the first 10,000 documents of the pinned FineWeb slice, with 1 and 8 processes
or threads pinned to 8 CPUs. datatrove 0.10.1 (ftfy 6.3.1, fasttext-predict 0.9.2.4, spaCy
3.8.18) on CPython 3.13.18; pyroid's native chain on free-threaded CPython 3.15.2. The output is
the same document for document. On the 100k corpus the ratios are 27× and 56×.</sub>

## Same output, checked

`pyroid.ftfy` is byte-identical to ftfy 6.3.1. `pyroid.fasttext` gives the same labels and
probabilities as fastText's C++ code on the same CPU architecture. The filters and dedup make
datatrove's keep/drop decisions, with its reasons and dedup files. The metrics match
rouge-score's and sacrebleu's scores to the last digit, `pyroid.json_repair` makes
json_repair's repairs, and `pyroid.datasketch` gives datasketch's hash values and LSH query results.

Each module is tested against the library it replaces, on pinned real data and with
property-based fuzzing ([parity counts](https://ao.github.io/pyroid/compatibility/)). The suites
run on Linux x86_64 and aarch64, including natively on AWS EC2 (c7i.4xlarge and c8g.4xlarge);
macOS and Windows are checked in CI before each release.

Where a module meets an option or input it doesn't port, it hands the call to the real library
with a `pyroid.FallbackWarning`, or raises. It never returns a different result silently.

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

With pyroid's stages swapped into datatrove's own executor, the pipeline ran 11.2× faster at 8
processes on the dev slice and 13.6× on 100k documents.

The other modules have only been measured in a Docker VM on a Mac so far, so their numbers are
indicative. On one thread, ROUGE is 59–65× faster than rouge-score, corpus BLEU 11× and chrF 19×
faster than sacrebleu, and `pyroid.json_repair` about 30× faster than json_repair on broken JSON,
where it runs in linear time and json_repair in quadratic. `pyroid.datasketch` computes MinHash
signatures 8× faster than datasketch on one thread and 21× on eight (85× from raw text), and
MinHashLSH inserts 8–27× and queries 6–19× faster (10,000 FineWeb documents, CPython 3.15t).

## Integrations

| Entry point | What it does |
|---|---|
| [`pyroid clean`](https://ao.github.io/pyroid/guides/cli/) | datatrove's FineWeb pipeline as one command over Parquet or JSONL |
| [`pyroid.patch()`](https://ao.github.io/pyroid/guides/patch/) | makes `import ftfy`, `import fasttext`, `import anyascii` and `import datasketch` give pyroid's functions, in code you don't own |
| [datatrove steps](https://ao.github.io/pyroid/guides/datatrove/) | datatrove's pipeline steps with the same names and signatures |
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
expression is `MIT AND Apache-2.0 AND BSD-3-Clause AND ISC AND PSF-2.0 AND Unicode-3.0`:

- Apache-2.0: ported from ftfy, datatrove, rouge-score, NLTK's Porter stemmer and sacrebleu;
- MIT: ported from spaCy, indic-nlp-library, urduhack and json_repair;
- BSD-3-Clause: numpy's random generator (for C4 bad words) and tldextract;
- PSF-2.0: CPython's `ipaddress` (IPv6 parsing);
- ISC: anyascii's transliteration data;
- Unicode-3.0: data from the Unicode Character Database.

See [LICENSE](LICENSE), [LICENSES/](LICENSES) and [NOTICE](NOTICE), which lists the ported files.
