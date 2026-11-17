# sqlparse 0.6.1 vs pyroid.sqlparse

EC2 c8g.16xlarge devbox (Graviton4, 64 vCPU), shared, indicative; CPython 3.12.16; pyroid 1.0.1 release build (not PGO); sqlparse 0.6.1. Both libraries in one process, interleaved, best of 7 rounds. Load average 2.1 before, 2.2 after. `python bench/sqlparse/sqlparse_bench.py --out sqlparse.json`.

| Input | Call | sqlparse (µs) | pyroid (µs) | Speedup |
|---|---|---:|---:|---:|
| 1-line SELECT | `format()` | 96.1 | 6.26 | 15.3x |
| 1-line SELECT | `format(reindent=True)` | 375.2 | 12.76 | 29.4x |
| 1-line SELECT | `format(reindent, keyword_case)` | 379.2 | 13.49 | 28.1x |
| 1-line SELECT | `split()` | 93.8 | 2.98 | 31.5x |
| 1-line SELECT | `parse()` | 300.4 | 19.96 | 15.0x |
| 4-line SELECT | `format()` | 742.6 | 18.82 | 39.5x |
| 4-line SELECT | `format(reindent=True)` | 2,804.0 | 64.74 | 43.3x |
| 4-line SELECT | `format(reindent, keyword_case)` | 2,848.7 | 70.76 | 40.3x |
| 4-line SELECT | `split()` | 716.4 | 11.00 | 65.1x |
| 4-line SELECT | `parse()` | 2,270.8 | 124.16 | 18.3x |
| INSERT, 200 rows | `format()` | 15,259.2 | 185.99 | 82.0x |
| INSERT, 200 rows | `format(reindent=True)` | 42,600.9 | 884.81 | 48.1x |
| INSERT, 200 rows | `format(reindent, keyword_case)` | 43,545.0 | 905.33 | 48.1x |
| INSERT, 200 rows | `split()` | 15,345.5 | 97.33 | 157.7x |
| INSERT, 200 rows | `parse()` | 38,729.1 | 2,127.18 | 18.2x |
| CREATE TABLE, 40 columns | `format()` | 2,814.1 | 68.43 | 41.1x |
| CREATE TABLE, 40 columns | `format(reindent=True)` | 9,982.5 | 381.31 | 26.2x |
| CREATE TABLE, 40 columns | `format(reindent, keyword_case)` | 10,186.4 | 394.20 | 25.8x |
| CREATE TABLE, 40 columns | `split()` | 2,581.4 | 46.95 | 55.0x |
| CREATE TABLE, 40 columns | `parse()` | 7,634.1 | 482.71 | 15.8x |
| migration, 120 statements | `format()` | 14,474.3 | 331.36 | 43.7x |
| migration, 120 statements | `format(reindent=True)` | 56,259.6 | 1,247.77 | 45.1x |
| migration, 120 statements | `format(reindent, keyword_case)` | 56,511.7 | 1,444.62 | 39.1x |
| migration, 120 statements | `split()` | 13,974.8 | 163.41 | 85.5x |
| migration, 120 statements | `parse()` | 44,747.8 | 2,859.61 | 15.6x |

| Batch | sqlparse loop | pyroid, all cores | pyroid, 1 thread | Speedup | Speedup, 1 thread |
|---|---:|---:|---:|---:|---:|
| 20,000 statements, `format(reindent=True)` | 19.03 s | 37.9 ms | 440.0 ms | 502x | 43x |
| 20,000 statements, `split()` | 4.89 s | 13.9 ms | 71.0 ms | 352x | 69x |
