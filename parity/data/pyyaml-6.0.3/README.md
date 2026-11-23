Copied from the PyYAML 6.0.3 sdist on PyPI (`pyyaml-6.0.3.tar.gz`, sha256 `d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f`): `tests/legacy_tests/data/` (as `data/`, all 574 files, unmodified) and `LICENSE`.
License: MIT, Copyright (c) 2017-2021 Ingy döt Net, Copyright (c) 2006-2016 Kirill Simonov (see `LICENSE`, sha256 `8d3928f9dc4490fd635707cb88eb26bd764102a7282954307d3e5167a577e8a4`).

The harness that reads these files (`tests/legacy_tests/test_*.py`, `canonical.py`) is not copied. Each test function there names the extensions it needs (`test_x.unittest = ['.data', '.canonical']`) and runs once per base name that has all of them. Several `.code`, `.dumper-error` and `.recursive` files are Python expressions evaluated in the namespace of `test_constructor.py`, `test_errors.py` or `test_recursive.py`.

| extension | files | read by |
|---|---|---|
| `.data` | 202 | most tests: the YAML input |
| `.canonical` | 91 | emitter, parser, composer, constructor and scanner tests (the same document in canonical form) |
| `.loader-error` | 82 | `test_errors.py`: input that must raise `YAMLError` on load, as a stream and as a string |
| `.code` | 55 | `test_constructor.py`, `test_multi_constructor.py`, `test_representer.py`: the expected Python value |
| `.tokens`, `.structure` | 28, 28 | `test_tokens.py`, `test_structure.py`: expected token and node shapes |
| `.emitter-error` | 16 | `test_errors.py`: event lists the emitter must reject |
| `.error`, `.empty`, `.skip-ext` | 13, 7, 7 | markers: spec examples that are errors, documents that are empty, cases the C extension tests skip |
| `.detect` | 9 | `test_resolver.py`: expected implicit tag |
| `.events` | 6 | `test_emitter.py`: event lists in canonical form |
| `.dumper-error` | 6 | `test_errors.py`: Python values the dumper must reject |
| `.recursive` | 5 | `test_recursive.py` |
| `.unicode`, `.stream-error`, `.single-loader-error` | 3, 3, 3 | `test_input_output.py`, `test_reader.py`, `test_errors.py` |
| others | 1 each | `.marks`, `.path`, `.multi`, `.sort`/`.sorted`, `.schema`/`.schema-skip` (`yaml11.schema`), `.subclass_blacklist`, `.former-loader-error`, `.former-dumper-error` |

Many inputs need loaders other than `SafeLoader` (`construct-python-*`, `.path`, `.multi`, `.subclass_blacklist`); a `safe_load`/`safe_dump` drop-in uses the rest. `latin.unicode` (59 KB) is the largest file; the whole directory is 147 KB.
