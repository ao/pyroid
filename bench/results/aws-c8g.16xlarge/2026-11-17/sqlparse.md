# sqlparse 0.6.1 vs pyroid.sqlparse

EC2 c8g.16xlarge devbox (Graviton4, 64 vCPU), shared, indicative; CPython 3.12.16; pyroid 1.0.1 release build (not PGO) at feat/sqlparse e6c2dd1a; sqlparse 0.6.1. Both libraries in one process, interleaved, best of 7 rounds. Load average 3.1 before, 4.3 after. `python bench/sqlparse/sqlparse_bench.py --corpus bench-sample.jsonl --corpus-limit 20000 --out sqlparse.json`.

The corpus rows are 20,000 documents (7.2 MB) drawn at random (seed 52) from the 216,108 of the parity corpus (`bench/sqlparse/corpus.py`, all 17 sources), each library timed once over all of them.

| Input | Call | sqlparse (µs) | pyroid (µs) | Speedup |
|---|---|---:|---:|---:|
| 1-line SELECT | `format()` | 91.7 | 6.41 | 14.3x |
| 1-line SELECT | `format(reindent=True)` | 376.2 | 12.96 | 29.0x |
| 1-line SELECT | `format(reindent, keyword_case)` | 379.7 | 13.69 | 27.7x |
| 1-line SELECT | `split()` | 94.7 | 2.99 | 31.6x |
| 1-line SELECT | `parse()` | 300.6 | 20.17 | 14.9x |
| 4-line SELECT | `format()` | 721.9 | 18.95 | 38.1x |
| 4-line SELECT | `format(reindent=True)` | 2,766.3 | 65.27 | 42.4x |
| 4-line SELECT | `format(reindent, keyword_case)` | 2,809.1 | 70.79 | 39.7x |
| 4-line SELECT | `split()` | 678.1 | 11.04 | 61.4x |
| 4-line SELECT | `parse()` | 2,222.6 | 128.10 | 17.4x |
| INSERT, 200 rows | `format()` | 14,441.1 | 189.07 | 76.4x |
| INSERT, 200 rows | `format(reindent=True)` | 42,690.5 | 888.88 | 48.0x |
| INSERT, 200 rows | `format(reindent, keyword_case)` | 43,501.7 | 913.86 | 47.6x |
| INSERT, 200 rows | `split()` | 14,901.7 | 98.33 | 151.6x |
| INSERT, 200 rows | `parse()` | 38,631.9 | 2,145.38 | 18.0x |
| CREATE TABLE, 40 columns | `format()` | 2,762.9 | 68.49 | 40.3x |
| CREATE TABLE, 40 columns | `format(reindent=True)` | 10,116.1 | 381.47 | 26.5x |
| CREATE TABLE, 40 columns | `format(reindent, keyword_case)` | 10,188.5 | 395.60 | 25.8x |
| CREATE TABLE, 40 columns | `split()` | 2,651.5 | 46.74 | 56.7x |
| CREATE TABLE, 40 columns | `parse()` | 7,658.1 | 486.91 | 15.7x |
| migration, 120 statements | `format()` | 14,371.9 | 330.68 | 43.5x |
| migration, 120 statements | `format(reindent=True)` | 56,059.3 | 1,254.68 | 44.7x |
| migration, 120 statements | `format(reindent, keyword_case)` | 56,744.6 | 1,458.38 | 38.9x |
| migration, 120 statements | `split()` | 13,907.2 | 163.63 | 85.0x |
| migration, 120 statements | `parse()` | 44,903.3 | 2,884.54 | 15.6x |

| Batch | sqlparse loop | pyroid, all cores | pyroid, 1 thread | Speedup | Speedup, 1 thread |
|---|---:|---:|---:|---:|---:|
| 20,000 statements, `format(reindent=True)` | 18.85 s | 37.8 ms | 439.8 ms | 499x | 43x |
| 20,000 statements, `split()` | 4.88 s | 13.7 ms | 71.9 ms | 357x | 68x |

| Call, 20,000 corpus documents | sqlparse | pyroid | Speedup |
|---|---:|---:|---:|
| `format()` | 12.3 s | 0.37 s | 33x |
| `format(reindent=True)` | 30.4 s | 0.97 s | 31x |
| `format(reindent, keyword_case)` | 30.8 s | 1.03 s | 30x |
| `split()` | 11.5 s | 0.21 s | 54x |
| `parse()` | 27.8 s | 3.29 s | 8x |
