Vendored from https://github.com/mangiucugna/json_repair at tag `v0.64.9`: `tests/` (as `upstream_tests/`, unmodified) and `LICENSE`.
License: MIT, Copyright (c) 2023 Stefano Baccianella (see `LICENSE`, sha256 `c2b8d0a3c32136b34289c5ed31edcc1e64be7f1f00990935b9ef004308a2155f`).
`ruff.toml` keeps ruff from linting or reformatting the vendored tests.

| file | what |
|---|---|
| `upstream_tests/` | json_repair 0.64.9's own test suite (imports `src.json_repair`; run only through the recorder) |
| `upstream-calls.jsonl` | 821 unique outermost public calls those tests make (691 `repair_json`, 54 `loads`, 65 `load`, 11 `from_file`), each with json_repair's outcome on CPython 3.12; header line first |
| `corpus-outcomes.json` | sha256 of the generated corpus (20,000 cases, `pyroid_parity/json_repair_corpus.py`) and json_repair's outcome digests per option set, one per 250 cases, pinned on CPython 3.12 |

- **Recording** (`python -m pyroid_parity.json_repair_record`, see its docstring): 534 upstream tests run and pass; `test_performance.py`, `test_type_inference.py` and `test_docs_app_schema.py` are not run (pytest-benchmark, mypy, the Flask demo app). Two calls are dropped: a schema nested past the stack limit, and a schema holding a Python object. Outcomes come from a clean replay with `PYTHONHASHSEED=0`.
- **Format** (`pyroid_parity/json_repair_compare.py`): JSON with type-exact tags (`$int`, `$float`, `$str`, `$tuple`, `$dict`, `$rep`/`$cat`/`$runs` for long repetitive values, `$textio`/`$path` for stream and file arguments; `"seekable": false` marks a stream that cannot seek).
- **Tags**: `schema` (170 calls; pyroid delegates `schema=` to json_repair), `schema-pydantic` (12), `via-JSONParser` (10), `patched-in-test` (2: the test monkeypatched json_repair, so the stored outcome is the clean replay), `recursion` (3, skipped: the outcome depends on the interpreter's recursion limit and the caller's stack depth), `hash-seed` (4, skipped: json_repair lists missing required schema properties in set order).
- **Kernel golden file**: `python -m pyroid_parity.json_repair_kernel_cases` writes `crates/pyroid-core/src/json_repair/testdata/upstream_cases.json` (490 inputs from these calls, 4 option sets) for `cargo test`.
- **Regenerate** (Docker, parity env): `uv run --frozen -p 3.12 --with pydantic python -m pyroid_parity.json_repair_record [--check]`, `uv run --frozen -p 3.12 python -m pyroid_parity.json_repair_kernel_cases [--check]` and `PYTHONHASHSEED=0 uv run --frozen -p 3.12 python -m pyroid_parity.json_repair_corpus --pin`.
