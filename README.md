# pyroid

> **Status: 1.0 is a ground-up rewrite in progress. Nothing here is usable yet.**

Pyroid 1.0 aims to be a Rust-fast text-cleaning core for LLM datasets, as a
Python package with prebuilt wheels and no runtime dependencies.

## Planned scope

- **`ftfy`, drop-in:** output byte-identical to [ftfy](https://github.com/rspeer/python-ftfy)
  for the supported options.
- **fastText inference:** load `.bin`/`.ftz` models, `predict`, sentence vectors,
  with batched calls that release the GIL.
- **Quality filters:** Gopher quality/repetition, C4 and FineWeb filters, returning masks.
- **Deduplication:** MinHash-LSH and exact dedup.
- **Transliteration:** under a permissive license.

The target is CPython 3.11+, including free-threaded builds (3.14t and 3.15t).
Correctness is checked against the incumbent libraries with differential tests.
This README makes no performance claims: numbers will be published only once
they are measured reproducibly.

## The 0.7.x releases

The previous general-purpose toolkit (0.1–0.7) is **not** carried forward, and
the 1.0 API is incompatible with it.

- 0.7.x stays installable from PyPI: `pip install "pyroid<1"`.
- Its source is kept on the `legacy` branch.

## Development

The extension is built with [maturin](https://www.maturin.rs/):

```console
$ uv venv && source .venv/bin/activate
$ uv pip install maturin pytest packaging
$ maturin develop --uv     # builds pyroid._pyroid into the venv (editable)
$ pytest python/tests
$ cargo test               # pure-Rust core (crates/pyroid-core)
```

Layout: `crates/pyroid-core` (pure Rust, no Python), `crates/pyroid-py` (PyO3
bindings, built as `pyroid._pyroid`) and `python/pyroid` (the typed Python package).

## License

MIT. See [LICENSE](LICENSE).
