# langdetect 1.0.9 vs pyroid.langdetect

AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, 16 vCPU: 8 cores, 2 threads each), dedicated,
Amazon Linux 2023, Docker `pyroid-dev:amd64`; CPython 3.12.16; pyroid 1.0.1 release build
(`maturin build --release`, not PGO) at feat/langdetect 5125fe57, built on the instance;
langdetect 1.0.9, pyarrow 26.0.1. One benchmark process at a time, nothing else running. Load
average 0.46 before, 1.90 after.

    python bench/langdetect_bench.py --fineweb /data/fineweb/fineweb-dev-10k.parquet \
        --fw2 fw2 --label "AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, 16 vCPU), dedicated" \
        --out langdetect.json

`fw2` holds the FineWeb-2 test split (`data/<subset>/test/000_00000.parquet`) of each of the 54
profile languages other than English, the subsets of `parity/langdetect_corpus.py`, downloaded on
2026-11-17 (2.8 GiB). Every text's `detect_langs` (languages and `float.hex` of each probability,
seed 0) was equal in both libraries before timing. µs per text, median of 5 rounds:

| Run | 50 characters (2,000 texts) | 1,500 characters (1,000) | 12,000 characters (250) |
|---|---:|---:|---:|
| langdetect, one thread | 1,366 | 2,687 | 15,191 |
| langdetect, 8 processes | 200 (7×) | 348 (8×) | 1,994 (8×) |
| pyroid, one call per text | 39 (35×) | 58 (46×) | 309 (49×) |
| pyroid `detect_batch`, threads=1 | 37 (37×) | 53 (51×) | 289 (52×) |
| pyroid `detect_batch`, threads=8 | 8.7 (158×) | 12 (220×) | 62 (246×) |

Patched `langdetect.detect` on the 1,500-character texts: 55.2 µs (49×). Start-up to the first
detection in a fresh interpreter: pyroid 112 ms, langdetect 237 ms.

A second run (`langdetect-run2.json`, load average 1.18 after)
gave medians within 5.3% of these (median 1.5%).

## `threads=8` and the c7i's hardware threads

The process's pool has 16 threads, one per vCPU, and vCPUs `n` and `n + 8` are the two hardware
threads of one core. `threads=8` runs 8 tasks on whichever 8 pool threads take them, so some
share a core. Time per 12,000-character text for one `detect_batch(texts, threads=8)` call
repeated 20 times, in a separate script with the same texts:

| Placement | µs per text |
|---|---:|
| default (pool of 16 on all vCPUs) | 59.1, 59.4, 61.3 |
| `taskset -c 0-7` (pool of 8, one thread per core) | 41.2, 42.0, 45.6 |
| `taskset -c 0-3,8-11` (pool of 8 on 4 cores) | 61.0 |

The instructions are the same in each case (3.42 million per text); the cycles per text are 1.12
million pinned to 8 cores, 1.45 million by default and 1.65 million on 4 cores (`perf stat`).
