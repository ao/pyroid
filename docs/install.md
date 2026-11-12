# Install

```console
$ pip install pyroid
```

To stay on the unrelated 0.7.x toolkit, pin `pyroid<1` ([Migrating from 0.7](migrating-from-0.7.md)).

pyroid has no runtime dependencies. Extras:

| Extra | Adds | Needed for |
|---|---|---|
| `pyroid[fasttext]` | `numpy>=1.26` | `pyroid.fasttext`, which returns numpy arrays as fastText does. Without numpy, `import pyroid.fasttext` raises a `ModuleNotFoundError` for numpy (an `ImportError`) that says how to fix it. |
| `pyroid[datasketch]` | `numpy>=1.26` | [`pyroid.datasketch`](guides/datasketch.md), whose sketches hold numpy arrays as datasketch's do. Without numpy, `import pyroid.datasketch` raises a `ModuleNotFoundError` for numpy that says how to fix it. |
| `pyroid[cli]` | `pyarrow>=18`, `numpy>=1.26` | the [`pyroid` command line](guides/cli.md) (Parquet and JSONL in and out, the FineWeb recipe) and `pyroid.integrations.recipes` |
| `pyroid[datatrove]` | `datatrove>=0.10.1`, `regex`, `tokenizers`, `numpy>=1.26`, `orjson>=3.2` | the [datatrove drop-in steps](guides/datatrove.md) in `pyroid.integrations.datatrove` (`regex` and `tokenizers` are imported by datatrove's filters but not declared by datatrove; `orjson` is for its JSONL reader and writer, from datatrove's `io` extra) |
| `pyroid[hf]` | `datasets>=3`, `pyarrow>=18`, `numpy>=1.26` | the [Hugging Face `datasets` helpers](guides/huggingface.md) in `pyroid.integrations.hf` |
| `pyroid[warc]` | `faust-cchardet`, `python-magic` | datatrove's page decoding in [`pyroid.warc.iter_documents`](guides/warc.md), its `WarcReader` drop-in and `pyroid clean --recipe fineweb-warc`: faust-cchardet guesses the charset of pages that are not UTF-8, and python-magic (which needs the libmagic library) types records without `WARC-Identified-Payload-Type`. Both are imported only when a record needs them, as datatrove imports them. `pyroid.warc.ArchiveIterator` needs neither. |
| `pyroid[nemo]` | `numpy>=1.26` | [`pyroid.nemo`](guides/nemo-curator.md), which returns numpy arrays of scores. NeMo Curator itself is not pulled in. |
| `pyroid[all]` | all of the above | everything; on CPython 3.15 it leaves out `datatrove`, `hf` and `warc` (see below) |

Extras combine: `pip install "pyroid[cli,fasttext]"`. `pyroid.ftfy`, `pyroid.filters`,
`pyroid.dedup`, `pyroid.translit`, `pyroid.metrics`, `pyroid.json_repair`, `pyroid.trafilatura`,
`pyroid.nltk`, `pyroid.jieba`, `pyroid.warc`'s reader and `pyroid.patch()` need none of them.

Some dependencies of the `datatrove`, `hf` and `warc` extras (orjson, tokenizers, PyYAML and
faust-cchardet) publish no wheels for CPython 3.15 and 3.15t yet, and orjson none for 3.14t. There
pip builds them from source: faust-cchardet needs a C++ compiler, and orjson and tokenizers a Rust
toolchain. `pyroid[all]` skips those three extras on 3.15; name them to build them anyway.
pyroid's own wheels cover every interpreter below.

Optional libraries, needed only by the features that use them:

- pyarrow, polars, pandas, Hugging Face `datasets` or any other Arrow library: pass their string
  columns straight in ([Inputs and outputs](concepts/inputs.md)).
- The incumbents, only for the few features pyroid hands back to them
  ([Fallbacks](concepts/fallbacks.md)):
    - ftfy 6.3.1;
    - datatrove 0.10.1 and a language's segmenter (Stanza, Kiwi, ...) for
      `python_tokenizer=True`;
    - trafilatura 2.3.1 (or 1.11.0 with `compat="1.11.0"`) for the delegated extraction options
      (other output formats, metadata, links, ...);
    - nltk 3.10.4 for Punkt training and the other `pyroid.nltk` features it hands back;
    - jieba 0.42.1 for `pyroid.jieba.posseg`, `pyroid.jieba.analyse` and paddle mode;
    - rouge-score 0.1.2 and sacrebleu 2.6.0 for the delegated metrics features;
    - json_repair 0.64.5 for `schema=`;
    - datasketch 2.0.0 for its classes pyroid does not port (`WeightedMinHash`, `HyperLogLog`,
      ...) and for Redis or Cassandra LSH storage.

    Without them, those calls raise `pyroid.PyroidError`.
- NLTK's `punkt_tab` data for `pyroid.nltk`'s Punkt and ROUGE's `split_summaries=True`
  (`python -m nltk.downloader punkt_tab`; pyroid does not bundle it, [details](guides/nltk.md)).
- numpy for `pyroid.metrics.scoring.BootstrapAggregator`, which runs rouge-score's numpy code.
- tldextract for `pyroid.filters.URLFilter`. Its suffix lists are read from a
  `tldextract.TLDExtract` when the filter is built; the matching itself runs in Rust.
- sudachidict-core (20261015 or newer) for native Japanese in `pyroid.filters` and
  `pyroid.dedup`: `pip install sudachidict-core`. pyroid does not ship the dictionary (its
  `system.dic` is 202 MB): it reads the installed one, or the file that `PYROID_SUDACHI_DICT`
  names, without importing SudachiPy. Without it the Japanese codes need `python_tokenizer=True`
  ([Languages](guides/languages.md#japanese)).
- pythainlp for native Thai in `pyroid.filters` and `pyroid.dedup`: `pip install pythainlp`.
  pyroid reads its word list and sentence model (or the corpus directory that
  `PYROID_PYTHAINLP_CORPUS` names) without importing it. With both, 4,049 of datatrove's 4,343
  language codes are native; without either, 4,025 ([Languages](guides/languages.md#thai)).

## Supported platforms

Each release has three wheels per platform, so every supported interpreter gets a prebuilt
binary and no compiler is needed:

| Wheel | Installs on |
|---|---|
| `cp311-abi3` | CPython 3.11, 3.12, 3.13, 3.14 |
| `cp314-cp314t` | free-threaded CPython 3.14t |
| `cp315-abi3.abi3t` | CPython 3.15 and free-threaded 3.15t |

| OS | Architectures | Wheel tag |
|---|---|---|
| Linux (glibc 2.17+) | x86-64, aarch64 | `manylinux2014` |
| Linux (musl 1.2+, e.g. Alpine) | x86-64, aarch64 | `musllinux_1_2` |
| macOS | x86-64, arm64 | `macosx` |
| Windows | x64, ARM64 | `win_amd64`, `win_arm64` |

That is 24 wheels plus an sdist per release. Before a release is published, the release
workflow installs each wheel in a clean virtual environment and runs pyroid's test suite with it,
on CPython 3.11, 3.14t, 3.15 and 3.15t, on every platform above. PyPy and GraalPy are not
supported.

!!! note "Where parity has been checked"
    Every parity suite has run on Linux aarch64, and CI runs the `parity/` suites on Linux x86_64
    and aarch64. The ftfy, fastText, filters, dedup, metrics and
    json_repair suites also ran natively on AWS EC2, on x86_64 (Intel Sapphire Rapids) and
    aarch64 (AWS Graviton4), with 0 differences. `pyroid.trafilatura` matched trafilatura on
    27,585 Common Crawl pages in 8 option sets, in both modes, on x86_64 (AWS EC2 c7i.4xlarge).
    fastText's and the metrics' references are pinned per platform, because the incumbents' own
    output can differ by architecture and C library. On macOS, Windows and musl, the release
    workflow runs pyroid's test suite, which compares against the installed incumbents, but not
    the `parity/` suites on pinned data ([Compatibility](compatibility.md#platforms)).

On free-threaded builds, importing pyroid does not re-enable the GIL. See
[Threads and free-threading](concepts/threads.md).

## Check the install

```console
$ python -c "import pyroid; print(pyroid.__version__)"
```

It prints the installed version as pip shows it (PEP 440, e.g. `1.0.0`).

If the native extension is missing or was built for another Python, `import pyroid` fails with an
`ImportError` that says so. pyroid has no pure-Python fallback.

## Build from source

Where no wheel fits (or with `pip install --no-binary pyroid pyroid`), pip builds the sdist. That
needs no Rust install: if `cargo` is not on `PATH`, maturin first downloads a Rust toolchain
(about 580 MB, into `~/.cache/puccinialin`), so the build needs network access and takes a few
minutes.

To build a checkout, you need a Rust toolchain and [maturin](https://www.maturin.rs/) 1.16 or
newer:

```console
$ git clone https://github.com/ao/pyroid && cd pyroid
$ uv venv && source .venv/bin/activate
$ uv pip install maturin
$ maturin develop --uv --release      # builds pyroid._pyroid into the venv
```

Without `--release` you get an unoptimized debug build, which is much slower. [`CONTRIBUTING.md`](https://github.com/ao/pyroid/blob/main/CONTRIBUTING.md)
covers the test and lint setup.
