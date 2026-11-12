# trafilatura 2.3.1 parity data

Nothing third-party is committed here. The `trafilatura` suite's inputs and references live in `$PYROID_DATA`,
pinned by sha256 in [`bench/manifest.json`](../../../bench/manifest.json):

| file | made by | what |
|---|---|---|
| `trafilatura/CC-MAIN-2026-39.parquet` | `bench/fetch.py` | 4,677 pages (id, url, html): bytes [0, 196,203,906) of WARC `...-00100` of CC-MAIN-2026-39. The P10 sample, byte for byte (sha256 `a9cadd81...`); its splits are rows 0-1999 (eval), 2000-2999 (dev), 3000-4676 (hold) |
| `trafilatura/CC-MAIN-2023-50.parquet` | `bench/fetch.py` | 926 pages from three 10 MB slices (20 %, 40 %, 60 % into WARC `...-00100`) |
| `trafilatura/CC-MAIN-2019-35.parquet` | `bench/fetch.py` | 1,589 pages from three 10 MB slices of WARC `...-00100` |
| `trafilatura/upstream-tests-2.3.1.parquet` | `bench/fetch.py` | Trafilatura's `tests/` at the commit tagged v2.3.1 (`e2f85c96...`), without `tests/eval/`: path, git blob sha1, bytes |
| `parity/trafilatura-ref.parquet` | `python -m pyroid_parity.refs trafilatura` | per page and option set: status, text length and the sha256 of the outcome (`trafilatura_compare.digest`), never the text |
| `parity/trafilatura-upstream.parquet`, `-blobs.parquet` | `python -m pyroid_parity.refs trafilatura_upstream` | the `extract` / `bare_extraction` calls Trafilatura's tests make, with their golden outcomes |

The pages are what datatrove 0.10.1 hands to its Trafilatura step: each slice is an HTTP range of a Common Crawl
WARC file on gzip member boundaries, read by `WarcReader.read_file` and filtered by `URLFilter()`
(`pyroid_parity.trafilatura_cc`). They are third-party web content under the
[Common Crawl terms of use](https://commoncrawl.org/terms-of-use); `tests/cache/` in the upstream tests holds
third-party pages too. Trafilatura's tests are Apache-2.0, © Adrien Barbaresi and contributors.

## Recording Trafilatura's tests

`pyroid_parity.trafilatura_record` runs the upstream `*_tests.py` files with a plugin that records every
outermost public `extract` and `bare_extraction` call: its arguments as given (lxml trees node by node,
`ConfigParser`, `Extractor` and `LRUCache` objects with their state at call time; strings over 4 KiB in the blob
table) and the pytest node id. The golden outcome is a clean replay in a fresh process, test by test after
`trafilatura.meta.reset_caches()`, because the module-level dedup cache carries state from call to call.

Recorded on CPython 3.13 (linux-aarch64): the upstream run has 4,189 passed, 37 xfailed and 2 skipped tests;
3,938 of them make 4,553 calls (4,540 `extract`, 13 `bare_extraction`). 2,669 use only options pyroid ports
(`upstream` check), 1,884 use others (`upstream-delegated`), 101 pass lxml trees. None depend on the hash seed.
2 differ from what the test saw, because the test changed state between calls in ways a replay of public calls
does not see: `test_dedup_reset_caches` calls `reset_caches()` itself, `test_settings_element_lists` edits
`MANUALLY_CLEANED` in place. Their golden outcomes are the clean replay's.

Not run, and why:

| file | reason |
|---|---|
| `cli_tests.py` | the command-line interface: subprocesses and the CLI's worker pools, not direct calls |
| `downloads_tests.py` | HTTP downloads |
| `feeds_tests.py`, `sitemaps_tests.py`, `spider_tests.py` | no `extract` or `bare_extraction` calls |
| `eval_common_tests.py`, `eval_gate_tests.py` | the evaluation harness; needs `tests/eval/` (923 pages, 135 MB), not fetched |

`py3langid` is not in the oracle group, so the tests that need it skip, as they do upstream without it.
