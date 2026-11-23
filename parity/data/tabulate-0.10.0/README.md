Vendored from the tabulate 0.10.0 sdist on PyPI (`tabulate-0.10.0.tar.gz`, sha256 `e2cfde8f79420f6deeffdeda9aaec3b6bc5abce947655d17ac662b126e48a60d`): `test/` (unmodified) and `LICENSE`.
License: MIT, Copyright (c) 2011-2020 Sergey Astanin and contributors (see `LICENSE`, sha256 `cdfab50d37d8ead06468fb76a3abafcd05c16749ec8a18eebfa50fd48c7cb2d2`).

| file | what |
|---|---|
| `test/` | tabulate 0.10.0's own test suite |
| `conftest.py` | patches `tabulate` with `pyroid.tabulate` before the tests import it (`PYROID_TABULATE_IMPL=tabulate` runs them against the installed tabulate) |
| `pytest.ini` | keeps the run apart from the repository's pytest configuration |
| `cli/tabulate/__init__.py` | stands in for the `tabulate/__init__.py` script that `test/test_cli.py` runs, so the CLI tests run `python -m pyroid.tabulate`'s code |

Run from `cli/` (`python/tests/test_tabulate_upstream.py` does this):

```bash
cd parity/data/tabulate-0.10.0/cli
PYTHONSAFEPATH=1 python -m pytest -q -c ../pytest.ini --rootdir .. ../test
```

With numpy, pandas and wcwidth installed: 318 passed, 1 skipped (a test tabulate skips unconditionally), against tabulate 0.10.0 and against pyroid alike.
