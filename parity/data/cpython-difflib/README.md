# CPython's difflib tests

Copied unchanged from the CPython repository (https://github.com/python/cpython, PSF-2.0, see
`LICENSE`), one directory per minor version, each at the tag of the newest patch release when
copied:

| Directory | Tag |
|---|---|
| `3.11/` | `v3.11.14` |
| `3.12/` | `v3.12.16` |
| `3.13/` | `v3.13.18` |
| `3.14/` | `v3.14.10` |
| `3.15/` | `v3.15.2` |

- `test_difflib.py`: `Lib/test/test_difflib.py`.
- `test_difflib_expect.html`: `Lib/test/test_difflib_expect.html`, the `HtmlDiff` output
  `test_html_diff` compares with.

`python/tests/test_difflib_cpython.py` runs the running interpreter's version against
`pyroid.difflib` and against the standard library under `pyroid.patch("difflib")`, with a small
stand-in for `test.support` (python-build-standalone interpreters ship no `test` package).
`SHA256SUMS` lists the files' digests.
