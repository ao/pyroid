# sqlparse 0.6.1 vs pyroid.sqlparse

AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, 16 vCPU: 8 cores, 2 threads each), dedicated, Amazon Linux 2023, Docker `pyroid-dev:amd64`; CPython 3.12.16; pyroid 1.0.1 release build (`maturin build --release`, not PGO) at feat/sqlparse 62fc7bb3, built on the instance; sqlparse 0.6.1. Both libraries in one process, interleaved, best of 7 rounds; one benchmark process at a time, nothing else running. Load average 0.19 before, 0.99 after. `python bench/sqlparse/sqlparse_bench.py --corpus bench-sample.jsonl --corpus-limit 20000 --out sqlparse.json`.

The corpus rows are 20,000 documents (7.6 MB) drawn at random (`random.Random(52).sample` of the lines) from the 216,175 of the real-SQL corpus, rebuilt on the instance on 2026-11-17 with `bench/sqlparse/corpus.py` (all 17 sources, downloaded that day), each library timed once over all of them. The shared-devbox run of the same day (`../../aws-c8g.16xlarge/2026-11-17/`) used a sample of the corpus as built earlier (216,108 documents, 7.2 MB).

A second run (`sqlparse-run2.json`, load 0.47 before, 1.03 after) gave timings within 10% of these (median 0.6%) and speedups within 9%.

| Input | Call | sqlparse (µs) | pyroid (µs) | Speedup |
|---|---|---:|---:|---:|
| 1-line SELECT | `format()` | 85.1 | 5.65 | 15.1x |
| 1-line SELECT | `format(reindent=True)` | 384.4 | 11.50 | 33.4x |
| 1-line SELECT | `format(reindent, keyword_case)` | 388.4 | 12.16 | 32.0x |
| 1-line SELECT | `split()` | 80.4 | 2.76 | 29.2x |
| 1-line SELECT | `parse()` | 284.2 | 18.19 | 15.6x |
| 4-line SELECT | `format()` | 630.1 | 17.53 | 35.9x |
| 4-line SELECT | `format(reindent=True)` | 2,507.1 | 62.70 | 40.0x |
| 4-line SELECT | `format(reindent, keyword_case)` | 2,530.3 | 67.69 | 37.4x |
| 4-line SELECT | `split()` | 607.8 | 10.49 | 57.9x |
| 4-line SELECT | `parse()` | 1,951.5 | 118.46 | 16.5x |
| INSERT, 200 rows | `format()` | 12,585.8 | 183.97 | 68.4x |
| INSERT, 200 rows | `format(reindent=True)` | 34,111.2 | 760.34 | 44.9x |
| INSERT, 200 rows | `format(reindent, keyword_case)` | 34,628.7 | 780.22 | 44.4x |
| INSERT, 200 rows | `split()` | 12,086.7 | 89.93 | 134.4x |
| INSERT, 200 rows | `parse()` | 30,239.4 | 2,423.07 | 12.5x |
| CREATE TABLE, 40 columns | `format()` | 2,279.0 | 65.90 | 34.6x |
| CREATE TABLE, 40 columns | `format(reindent=True)` | 8,273.2 | 332.33 | 24.9x |
| CREATE TABLE, 40 columns | `format(reindent, keyword_case)` | 8,320.5 | 349.61 | 23.8x |
| CREATE TABLE, 40 columns | `split()` | 2,221.8 | 44.02 | 50.5x |
| CREATE TABLE, 40 columns | `parse()` | 6,278.9 | 437.72 | 14.3x |
| migration, 120 statements | `format()` | 12,366.9 | 303.22 | 40.8x |
| migration, 120 statements | `format(reindent=True)` | 53,608.0 | 1,140.96 | 47.0x |
| migration, 120 statements | `format(reindent, keyword_case)` | 54,100.7 | 1,395.09 | 38.8x |
| migration, 120 statements | `split()` | 11,090.6 | 154.21 | 71.9x |
| migration, 120 statements | `parse()` | 39,516.1 | 2,707.02 | 14.6x |

| Batch | sqlparse loop | pyroid, all 16 vCPUs | pyroid, 1 thread | Speedup | Speedup, 1 thread |
|---|---:|---:|---:|---:|---:|
| 20,000 statements, `format(reindent=True)` | 18.17 s | 59.9 ms | 395.1 ms | 303x | 46x |
| 20,000 statements, `split()` | 4.17 s | 14.2 ms | 67.7 ms | 294x | 62x |

| Call, 20,000 corpus documents | sqlparse | pyroid | Speedup |
|---|---:|---:|---:|
| `format()` | 10.8 s | 0.38 s | 28x |
| `format(reindent=True)` | 27.9 s | 1.01 s | 28x |
| `format(reindent, keyword_case)` | 28.4 s | 1.10 s | 26x |
| `split()` | 10.0 s | 0.23 s | 44x |
| `parse()` | 26.1 s | 5.06 s | 5x |

## Corpus `parse()`: time in the cycle collector

A separate run: one library in a fresh process (same wheels and venv), the same 20,000 documents
(sha256 of the sample file `ff68a9d0...`), 200 warm-up calls and a `gc.collect()`, then one
`parse()` per document, timed once, with the collector's time summed from `gc.callbacks`.
"Kept" stores every tree in a list until the loop ends, as the benchmark's corpus rows do;
"dropped" lets each tree go before the next call. The c8g.16xlarge rows are the shared devbox
(load average 10 to 15, indicative), same sample, a release build of the same commit.

| Machine | Library | Trees | Wall | In the collector | Collections (gen 0 / 1 / 2) |
|---|---|---|---:|---:|---|
| c7i.4xlarge | pyroid | kept | 3.67 s | 2.35 s | 2,513 / 228 / 10 |
| c7i.4xlarge | pyroid | dropped | 1.77 s | 0.53 s | 2,361 / 214 / 19 |
| c7i.4xlarge | sqlparse | kept | 28.15 s | 3.46 s | 3,854 / 350 / 15 |
| c7i.4xlarge | sqlparse | dropped | 23.28 s | 0.84 s | 3,836 / 348 / 31 |
| c8g.16xlarge, shared | pyroid | kept | 2.53 s | 1.22 s | 2,513 / 228 / 10 |
| c8g.16xlarge, shared | pyroid | dropped | 1.68 s | 0.45 s | 2,361 / 214 / 19 |
| c8g.16xlarge, shared | sqlparse | kept | 27.46 s | 1.82 s | 3,854 / 350 / 15 |
| c8g.16xlarge, shared | sqlparse | dropped | 25.80 s | 0.68 s | 3,836 / 348 / 31 |

On the c7i, `perf stat` over the kept pyroid loop counts 47% of cycles stalled on L2 misses
(`memory_activity.stalls_l2_miss`), against 10% for the dropped one.
