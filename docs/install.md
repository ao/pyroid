# Install

!!! warning "Pre-release"
    pyroid 1.0 is not on PyPI yet. Until it is, `pip install pyroid` installs 0.7.x, an unrelated
    older toolkit ([Migrating from 0.7](migrating-from-0.7.md)). To try 1.0 now, build it from a
    checkout (below).

```console
$ pip install pyroid
```

pyroid has **no runtime dependencies**. Extras:

| Extra | Adds | Needed for |
|---|---|---|
| `pyroid[fasttext]` | `numpy>=1.26` | `pyroid.fasttext`, which returns numpy arrays as fastText does. Without numpy, `import pyroid.fasttext` raises an `ImportError` that says how to fix it. |

Optional libraries pyroid works with but never requires:

- **pyarrow, polars, pandas, Hugging Face `datasets`** or any other Arrow library: pass their string
  columns straight in ([Inputs and outputs](concepts/inputs.md)).
- **ftfy** 6.3.1: only for the few ftfy features `pyroid.ftfy` hands back to ftfy
  ([Fallbacks](concepts/fallbacks.md)). Without it those calls raise `pyroid.PyroidError`.

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
it is published. Before the first release, wheels have only been built and tested on Linux
aarch64; the release workflow builds the rest. PyPy and GraalPy are not supported.

!!! note "Where parity has been checked"
    The parity suites have so far run on Linux aarch64 only. fastText's floating-point rounding
    and tie order are matched per architecture, and x86-64, macOS and Windows parity is
    unverified until the first CI run on those platforms
    ([Compatibility](compatibility.md#platforms)).

On free-threaded builds, importing pyroid does not re-enable the GIL. See
[Threads and free-threading](concepts/threads.md).

## Check the install

```console
$ python -c "import pyroid; print(pyroid.__version__)"
```

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
