# trafilatura 1.11.0 parity data

The `trafilatura` suite's 1.11.0 mode (`--trafilatura-compat 1.11.0`, pyroid's `compat="1.11.0"`): trafilatura
1.11.0 as a fresh `pip install datatrove[processing]==0.10.1` resolves it, with lxml 6.1.3, justext 3.0.2,
lxml-html-clean 0.4.6, charset-normalizer 3.5.2, courlan 1.4.0 and htmldate 1.11.0 (the `trafilatura111`
dependency group). Nothing third-party is committed here; the files live in `$PYROID_DATA`, pinned in
[`bench/manifest.json`](../../../bench/manifest.json):

| file | made by | what |
|---|---|---|
| `trafilatura/CC-MAIN-*.parquet` | `bench/fetch.py` | the same 7,192 pages as the 2.3.1 mode ([`../trafilatura-2.3.1`](../trafilatura-2.3.1/README.md)) |
| `trafilatura/upstream-tests-1.11.0.parquet` | `bench/fetch.py` | Trafilatura's `tests/` at the commit tagged v1.11.0 (`60647e5e...`), without `tests/eval/`, `evaldata.json` and `evaldata.py` |
| `parity/trafilatura-1.11.0-ref.parquet` | `python -m pyroid_parity.refs trafilatura111` | per page and option set: status, text length and outcome digest |
| `parity/trafilatura-1.11.0-upstream.parquet`, `-blobs.parquet` | `python -m pyroid_parity.refs trafilatura111_upstream` | the `extract` / `bare_extraction` calls 1.11.0's tests make, with golden outcomes |

Both `refs` steps need the 1.11.0 env: `uv run --no-group trafilatura --group trafilatura111 python -m
pyroid_parity.refs trafilatura111 trafilatura111_upstream`.

## How 1.11.0 differs from 2.3.1, as far as the suite goes

- No `fast`: the matrix uses `no_fallback=True` instead. `extract` takes unknown keywords (`**kwargs`) and ignores
  them, so `fast=True` changes nothing there (edge case `fast_ignored`), while `bare_extraction` rejects it with a
  TypeError (`bare:fast`).
- `include_formatting` defaults to False, `bare_extraction` to `as_dict=True` (it returns a dict).
- `max_tree_size` is a working option, not a ValueError; `LRUCache` is a linked list, and `deduplicate=` does not
  take one (the module-level `LRU_TEST` is always the cache).

## Recording 1.11.0's tests

`*_tests.py`, as 1.11.0's `pytest.ini` collects them, except the CLI tests and those that reach the network
unmocked (downloads, feeds, sitemaps, spider; none of them calls `extract` or `bare_extraction`). The run here has
57 tests: 54 pass and 3 fail with these pins (`realworld_tests.py::test_extract[False-True]`,
`unit_tests.py::test_external`, `unit_tests.py::test_table_processing`); the failing tests' calls are recorded like
the others, with the clean replay as their golden outcome.

`unit_tests.py` and `filters_tests.py` change trafilatura's `DEFAULT_CONFIG` in place at import (MIN_OUTPUT_SIZE
and MIN_EXTRACTED_SIZE set to 0), so every call of the run sees those thresholds, including those that pass no
`config=`. The recorder takes a snapshot of the pristine config before any test module is imported and records
each call with the config it actually saw (tag `default-config-mutated`). Each such call is recorded a second
time with trafilatura's own default (tag `pristine-config`), in a replay group of its own so the original
sequences keep their dedup state: those variants exercise the default thresholds that pyroid ports.

Result: 633 calls (625 `extract`, 8 `bare_extraction`): 318 from 29 tests as recorded, 315 pristine-config
variants; 128 pass lxml trees. No replay differs from what its test saw and none depends on the hash seed. A
second recording gives the same bytes.
