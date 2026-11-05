# pyroid

[![PyPI](https://img.shields.io/pypi/v/pyroid)](https://pypi.org/project/pyroid/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyroid)](https://pypi.org/project/pyroid/)
[![License](https://img.shields.io/pypi/l/pyroid)](https://github.com/ao/pyroid/blob/main/LICENSE)
[![CI](https://github.com/ao/pyroid/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ao/pyroid/actions/workflows/ci.yml)

[Docs](https://ao.github.io/pyroid/) | [Benchmarks](https://ao.github.io/pyroid/benchmarks/) | [Compatibility](https://ao.github.io/pyroid/compatibility/)

**Drop-in Rust replacements for ftfy, fastText inference and datatrove's quality filters, for
cleaning LLM datasets: the same API, the same output, a fraction of the CPU time.**

> [!NOTE]
> pyroid 1.0 is not on PyPI yet: until it is, `pip install pyroid` gives the unrelated 0.7.x
> toolkit ([see below](#the-07x-releases)). Speed numbers are preliminary, from a Docker VM.

<!-- TODO(bare-metal): hero chart from the bare-metal harness run, light and dark SVGs rendered
     by bench/chart.py and committed to docs/assets/. Caption: corpus, cores, versions.
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.svg">
  <img alt="Time to clean the FineWeb sample, pyroid vs the incumbents" src="docs/assets/hero-light.svg">
</picture>
-->

```console
$ pip install pyroid               # add [fasttext] for pyroid.fasttext (numpy)
```

```python
import pyroid.ftfy as ftfy  # was: import ftfy (byte-identical output)
import pyroid.fasttext as fasttext  # was: import fasttext (same .bin/.ftz models)
import pyroid.filters as filters  # datatrove's Gopher/C4/FineWeb filters (same decisions)

docs = ftfy.fix_text_batch(docs)  # all cores, GIL released
lid = fasttext.load_model("lid.176.bin")
labels, probs = lid.predict([d.replace("\n", " ") for d in docs], k=1)
fineweb_filters = [
    filters.GopherRepetition(),
    filters.GopherQuality(),
    filters.C4Quality(filter_no_terminal_punct=False),
    filters.FineWebQuality(),
]
result = filters.apply(docs, fineweb_filters)  # in datatrove's order, one tokenization per doc
clean = [text for text, keep in zip(result.text, result.keep) if keep]
```

Batch functions take a `str`, a `list` of `str` or an Arrow column (pyarrow, polars, Hugging Face
`datasets`); text and masks come back in the same kind.

**Replaces:** ftfy · fastText inference (fasttext, fasttext-wheel, fasttext-predict) · datatrove's
Gopher, C4 and FineWeb filters · datatrove's MinHash dedup · Unidecode (permissive,
anyascii-based)

## Why

- **Faster, measured against the incumbents** (single-threaded; Docker VM on Apple Silicon,
  indicative, preliminary: [methodology](https://ao.github.io/pyroid/benchmarks/)):
  `fix_text` 95–167× faster than ftfy on FineWeb documents; FineWeb's four quality filters about
  86× and MinHash signatures about 64× faster than datatrove; fastText language ID 1.5–5.4×
  faster; transliteration 37× faster than anyascii and 76× faster than Unidecode.
- **Same output, checked:** byte-identical to ftfy 6.3.1, bit-identical fastText labels and
  probabilities, and datatrove's keep/drop decisions and reasons, on pinned FineWeb documents
  and property-based fuzzing ([parity counts](https://ao.github.io/pyroid/compatibility/)). So
  far verified on Linux aarch64; the other platforms are checked in CI before release.
- **Installs everywhere:** wheels for CPython 3.11–3.15 and free-threaded 3.14t/3.15t, on Linux
  (glibc and musl), macOS and Windows, x86-64 and ARM64. No compiler, no runtime dependencies.
- **Threads, not processes:** batch calls release the GIL and run on all cores, and one loaded
  fastText model serves every thread (one copy in memory instead of one per worker process).
- **Never silently different:** `pyroid.ftfy` hands the few options it doesn't port to the real
  ftfy, with a `pyroid.FallbackWarning`; the other modules raise rather than approximate.

| Module | Replaces | Status |
|---|---|---|
| [`pyroid.ftfy`](https://ao.github.io/pyroid/guides/ftfy/) | ftfy 6.3.1 | ready |
| [`pyroid.fasttext`](https://ao.github.io/pyroid/guides/fasttext/) | fastText inference | ready |
| [`pyroid.filters`](https://ao.github.io/pyroid/guides/filters/) | datatrove 0.10.1 Gopher, C4, FineWeb filters | ready |
| [`pyroid.dedup`](https://ao.github.io/pyroid/guides/dedup/) | datatrove 0.10.1 MinHash dedup | ready |
| [`pyroid.translit`](https://ao.github.io/pyroid/guides/translit/) | anyascii; a permissive alternative to Unidecode | ready |

## The 0.7.x releases

pyroid 1.0 is a rewrite. The previous general-purpose toolkit (0.1–0.7) is **not** carried
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

Layout: `crates/pyroid-core` (pure Rust, no Python), `crates/pyroid-py` (PyO3 bindings, built as
`pyroid._pyroid`), `python/pyroid` (the typed Python package), `bench/` (benchmark harness),
`parity/` (differential tests against the incumbents) and `docs/` (this documentation). See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. Parts are ported from ftfy and datatrove (Apache-2.0) and from spaCy (MIT), and keep their
licences: see [LICENSE](LICENSE) and [NOTICE](NOTICE).
