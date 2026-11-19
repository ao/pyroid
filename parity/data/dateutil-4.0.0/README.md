Vendored from python-dateutil 4.0.0 (the sdist on PyPI): `tests/test_parser.py`, `tests/test_isoparser.py`, `tests/_common.py`, `tests/conftest.py` and `tests/__init__.py` (as `upstream_tests/`, unmodified) and `LICENSE`.
License: BSD-3-Clause, and Apache-2.0 for contributions after 2017-12-01 (see `LICENSE`, sha256 `ba00f51a0d92823b5a1cde27d8b5b9d2321e67ed8da9bc163eff96d5e17e577e`).
`ruff.toml` keeps ruff from linting or reformatting the vendored tests.

`tests/test_dateutil.py` (check `upstream-4.0.0`) runs these tests twice when python-dateutil 4.0.0 is installed: as they are, and with `from dateutil.parser import` pointing at the candidate. Every test must have the same outcome (passed, failed, xfailed, skipped).
