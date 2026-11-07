# Install

!!! warning "Pre-release"
    pyroid 1.0 is not on PyPI yet. Until it is, `pip install pyroid` installs 0.7.x, an unrelated
    older toolkit ([Migrating from 0.7](migrating-from-0.7.md)). To try 1.0 now, build it from a
    checkout (below).

```console
$ pip install pyroid
```

pyroid has no runtime dependencies. Extras:

| Extra | Adds | Needed for |
|---|---|---|
| `pyroid[fasttext]` | `numpy>=1.26` | `pyroid.fasttext`, which returns numpy arrays as fastText does. Without numpy, `import pyroid.fasttext` raises a `ModuleNotFoundError` for numpy (an `ImportError`) that says how to fix it. |
| `pyroid[datasketch]` | `numpy>=1.26` | [`pyroid.datasketch`](guides/datasketch.md), whose sketches hold numpy arrays as datasketch's do. Without numpy, `import pyroid.datasketch` raises a `ModuleNotFoundError` for numpy that says how to fix it. |
| `pyroid[cli]` | `pyarrow>=18`, `numpy>=1.26` | the [`pyroid` command line](guides/cli.md) (Parquet and JSONL in and out, the FineWeb recipe) and `pyroid.integrations.recipes` |
| `pyroid[datatrove]` | `datatrove>=0.10.1`, `regex`, `tokenizers`, `numpy>=1.26` | the [datatrove drop-in steps](guides/datatrove.md) in `pyroid.integrations.datatrove` (`regex` and `tokenizers` are imported by datatrove's filters but not declared by datatrove) |
| `pyroid[hf]` | `datasets>=3`, `pyarrow>=18`, `numpy>=1.26` | the [Hugging Face `datasets` helpers](guides/huggingface.md) in `pyroid.integrations.hf` |

Extras combine: `pip install "pyroid[cli,fasttext]"`. `pyroid.ftfy`, `pyroid.filters`,
`pyroid.dedup`, `pyroid.translit`, `pyroid.metrics`, `pyroid.json_repair` and `pyroid.patch()`
need none of them.

Optional libraries, needed only by the features that use them:

- pyarrow, polars, pandas, Hugging Face `datasets` or any other Arrow library: pass their string
  columns straight in ([Inputs and outputs](concepts/inputs.md)).
- The incumbents, only for the few features pyroid hands back to them
  ([Fallbacks](concepts/fallbacks.md)):
    - ftfy 6.3.1;
    - datatrove 0.10.1 and a language's segmenter (SudachiPy, pyvi, ...) for
      `python_tokenizer=True`;
    - jieba 0.42.1 for `pyroid.jieba.posseg`, `pyroid.jieba.analyse` and paddle mode;
    - rouge-score, nltk and sacrebleu 2.6.0 for the delegated metrics features;
    - json_repair 0.64.5 for `schema=`;
    - datasketch 2.0.0 for its classes pyroid does not port (`WeightedMinHash`, `HyperLogLog`,
      ...) and for Redis or Cassandra LSH storage.

    Without them, those calls raise `pyroid.PyroidError`.
- numpy for `pyroid.metrics.scoring.BootstrapAggregator`, which runs rouge-score's numpy code.
- tldextract for `pyroid.filters.URLFilter`. Its suffix lists are read from a
  `tldextract.TLDExtract` when the filter is built; the matching itself runs in Rust.

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

That is 24 wheels plus an sdist per release, each tested in a clean virtual environment before
it is published. Before the first release, the Linux wheels have been built and tested on aarch64
and x86_64: natively on AWS EC2, and earlier in a Docker VM and under emulation. The release
workflow builds the rest. PyPy and GraalPy are not supported.

!!! note "Where parity has been checked"
    Every parity suite has run on Linux aarch64. The ftfy, fastText, filters, dedup, metrics and
    json_repair suites also ran natively on AWS EC2, on x86_64 (Intel Sapphire Rapids) and
    aarch64 (AWS Graviton4), with 0 differences. fastText's and the metrics' references are
    pinned per platform, because the incumbents' own output can differ by architecture. macOS,
    Windows and musl are unverified until the first CI run on them
    ([Compatibility](compatibility.md#platforms)).

On free-threaded builds, importing pyroid does not re-enable the GIL. See
[Threads and free-threading](concepts/threads.md).

## Check the install

```console
$ python -c "import pyroid; print(pyroid.__version__)"
```

It prints the installed version as pip shows it (PEP 440, e.g. `1.0.0.dev0`).

If the native extension is missing or was built for another Python, `import pyroid` fails with an
`ImportError` that says so. pyroid has no pure-Python fallback.

## Build from source

You need a Rust toolchain and [maturin](https://www.maturin.rs/) 1.16 or newer:

```console
$ git clone https://github.com/ao/pyroid && cd pyroid
$ uv venv && source .venv/bin/activate
$ uv pip install maturin
$ maturin develop --uv --release      # builds pyroid._pyroid into the venv
```

Without `--release` you get an unoptimized debug build, which is much slower. [`CONTRIBUTING.md`](https://github.com/ao/pyroid/blob/main/CONTRIBUTING.md)
covers the test and lint setup.
