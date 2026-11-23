# sqlparse 0.6.1 test suite

`tests/` is copied unchanged from the sqlparse 0.6.1 sdist on PyPI
(`sqlparse-0.6.1.tar.gz`, sha256 ca722573f848ff2d30e40ea30f53a104b7dea7635ea9285891263eb2c58fb6e7;
https://github.com/andialbrecht/sqlparse, BSD-3-Clause, see `LICENSE`).

`python -m pytest -p _sqlparse_upstream parity/data/sqlparse-0.6.1/tests` (with `python/tests`
on `sys.path`) runs it with `pyroid.patch("sqlparse")` on: with sqlparse installed its
`parse`, `parsestream`, `format`, `split` and `lexer.tokenize` are pyroid's; without it the
whole `sqlparse` package is `pyroid.sqlparse`.
