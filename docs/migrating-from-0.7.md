# Migrating from 0.7

pyroid 1.0 is a different library that keeps the name. The 0.x releases (0.1 to 0.7) were a
general-purpose toolkit (math, strings, collections, file and network I/O, images, basic machine
learning). **That API was removed in 1.0**, and nothing from it is carried forward. 1.0 is a
text-cleaning core for LLM datasets.

## Staying on 0.7

0.7.x stays on PyPI, and its source is kept on the
[`legacy`](https://github.com/ao/pyroid/tree/legacy) branch. Pin it:

```console
$ pip install "pyroid<1"
```

```toml
# pyproject.toml
dependencies = ["pyroid<1"]
```

0.7.x is frozen: no further releases are planned.

## Moving off 0.7

An audit before the rewrite found that 0.7's public functions ran in Python, not Rust, so the
standard library or a mainstream package is a like-for-like replacement:

| 0.7 | Use instead |
|---|---|
| `pyroid.math.Vector`, `pyroid.math.Matrix` | [numpy](https://numpy.org/) arrays |
| `pyroid.math.stats.mean`, `.median`, `.calc_std`, ... | the [`statistics`](https://docs.python.org/3/library/statistics.html) module, or numpy |
| `pyroid.text.reverse`, `.to_uppercase`, `.to_lowercase` | `s[::-1]`, `s.upper()`, `s.lower()` |
| `pyroid.text.base64_encode`, `.base64_decode` | the [`base64`](https://docs.python.org/3/library/base64.html) module |
| `pyroid.data.filter`, `.map`, `.reduce`, `.sort` | built-in `filter`, `map`, `sorted`, and `functools.reduce` |
| `pyroid.data.DataFrame`, `.apply`, `.groupby_aggregate` | [polars](https://pola.rs/) or [pandas](https://pandas.pydata.org/) |
| `pyroid.io.read_file`, `.write_file`, `.read_files` | `pathlib.Path.read_text` / `.write_text` |
| `pyroid.io.get` and the other HTTP helpers, `pyroid.AsyncClient` | [httpx](https://www.python-httpx.org/) or `urllib.request` |
| `pyroid.image.basic` | [Pillow](https://python-pillow.org/) |
| `pyroid.ml.basic` (k-means, linear regression, ...) | [scikit-learn](https://scikit-learn.org/) |

!!! note "`pyroid.text` is not the 0.7 module"
    1.0 has a `pyroid.text` module, but it is unrelated to 0.7's: it holds
    `normalize_whitespace` and `min_chars`. Code that calls `pyroid.text.reverse` fails with an
    `AttributeError` on 1.0.

## Starting with 1.0

See the [guides](guides/ftfy.md) for what 1.0 does, and [Install](install.md) for its wheels.
