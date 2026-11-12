# Consolidated benchmark (Docker/OrbStack VM on Apple Silicon, indicative)

Machine: `orbstack-aarch64-12cpu`; run folder `2026-11-05-consolidated`. Corpus: FineWeb dev slice (first 10,000 docs of the pinned 100k headline slice). Every row passed its equivalence gate before it was timed (byte-identical ftfy/anyascii output, fastText labels with |dp| <= 1e-6, datatrove filter decisions and text, datatrove MinHash signatures).

## End to end: the FineWeb chain (ftfy -> LID -> 4 filters -> MinHash dedup), 10k docs

| run | wall s, median [IQR] | CPU s | peak RSS of one process, MiB | vs datatrove, 1 proc | vs datatrove, 8 procs |
|---|---:|---:|---:|---:|---:|
| pyroid native chain, 8 threads (CPython 3.15t) | 1.61 [1.57-1.71] | 6.1 | 605 | 157.6x | 26.1x |
| pyroid stages in datatrove's executor, 8 processes (3.13) | 2.85 [2.83-2.92] | 13.0 | 385 | 88.9x | 14.7x |
| pyroid native chain, 1 thread (CPython 3.15t) | 5.16 [5.15-5.41] | 5.8 | 503 | 49.1x | 8.1x |
| pyroid stages in datatrove's executor, 1 process (3.13) | 7.48 [7.24-7.52] | 8.1 | 421 | 33.9x | 5.6x |
| datatrove 0.10.1, 8 processes (CPython 3.13) | 41.93 [41.27-42.18] | 314.5 | 367 | 6.0x | 1.0x |
| datatrove 0.10.1, 1 process (CPython 3.13) | 253.20 [252.48-253.45] | 254.3 | 438 | 1.0x | 0.2x |

Digests: datatrove at 1 and 8 workers and datatrove's executor with pyroid's stages: identical (digest-w1, digest-candidate-w1, digest-w8, digest-candidate-w8). pyroid's native chain (chain.py, 3.15t): digest `ef314b8d6ae9...`, identical to datatrove.

## Per stage: microseconds per document, median [IQR] of pyperf values

Incumbents: one process, CPython 3.13, per-doc loop over their largest measured batch (10k docs for fix_text, LID and translit; 1k for the filters and MinHash). pyroid: one call over the same docs (list[str]); `pool` = RAYON_NUM_THREADS. One-stage e2e: datatrove's executor running only that stage at 8 processes (wall time per doc, start-up and parquet IO included).

| stage | incumbent, 1 proc | incumbent, 8 procs (one-stage e2e) | pyroid 3.13, 1 thread | pyroid 3.15t, pool 1 | pyroid 3.15t, pool 8 | speedup, 1 thread vs 1 proc |
|---|---:|---:|---:|---:|---:|---:|
| fix_text, 10,000 docs | 834 [831-838] | 205 | 5.47 [5.40-5.80] | 5.66 [5.63-5.77] | 1.72 [1.68-2.11] | 147× |
| lid (lid.176.bin), 10,000 docs | 290 [284-302] | 187 | 144 [141-149] | 141 [139-148] | 20.2 [19.7-25.9] | 2.1× |
| lid (lid.176.ftz), 10,000 docs | 451 [444-457] | 187 | 105 [104-106] | 105 [105-106] | 16.0 [15.9-17.2] | 4.3× |
| gopher_repetition, 1,000 docs | 7,295 [7,267-7,383] | 1,257 | 141 [139-147] | 141 [140-147] | 29.9 [27.4-38.1] | 52× |
| gopher_quality, 1,000 docs | 5,739 [5,720-5,762] | 973 | 76.5 [75.8-82.6] | 79.8 [78.4-83.8] | 16.0 [14.4-16.7] | 72× |
| c4_quality, 1,000 docs | 5,131 [5,099-5,212] | - | 47.0 [46.7-59.0] | 48.2 [46.7-61.1] | 16.1 [14.0-22.0] | 107× |
| c4_quality@fineweb, 1,000 docs | 6,635 [6,590-6,774] | 1,162 | 84.3 [83.5-92.0] | 88.2 [84.3-108] | 20.0 [17.7-22.6] | 75× |
| fineweb_quality, 1,000 docs | 5,415 [5,383-5,441] | 943 | 65.5 [64.2-76.9] | 67.0 [64.4-70.1] | 14.2 [13.4-15.3] | 81× |
| minhash, 1,000 docs | 5,820 [5,786-5,870] | 1,046 | 91.0 [89.3-103] | 91.7 [90.4-99.9] | 22.7 [21.8-24.3] | 63× |
| translit, 10,000 docs | 128 [126-129] | - | 1.29 [1.27-1.40] | 1.62 [1.50-1.64] | 1.00 [0.98-1.12] | 79× |

MinHash estimators with other hash families (gated on near-duplicate precision/recall >= 0.90, not on identical signatures): rensa 0.5.0 RMinHash.update(112) per doc: 149 us/doc; rensa 0.5.0 RMinHash.from_token_sets(112): 159 us/doc; datasketch 2.0.0 MinHash.bulk(num_perm=112): 476 us/doc.

pyroid on the filters and MinHash with 10k docs per call (3.15t, list[str]; larger batches use the pool better): c4_quality pool 1: 79.1; c4_quality pool 8: 12.6; c4_quality@fineweb pool 1: 93.3; c4_quality@fineweb pool 8: 14.3; fineweb_quality pool 1: 72.6; fineweb_quality pool 8: 13.2; gopher_quality pool 1: 85.2; gopher_quality pool 8: 13.6; gopher_repetition pool 1: 150; gopher_repetition pool 8: 22.1; minhash pool 1: 107; minhash pool 8: 17.1 us/doc.

pyroid with Arrow input (3.15t, 10k docs): pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] [3.15t, pool 1]: 5.16 us/doc; pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] [3.15t, pool 8]: 1.08 us/doc; pyroid.translit 1.0.0.dev0 anyascii [arrow] [3.15t, pool 1]: 1.15 us/doc; pyroid.translit 1.0.0.dev0 anyascii [arrow] [3.15t, pool 8]: 0.41 us/doc.

## Charts

- `charts/consolidated-e2e-{light,dark}.svg`: the end-to-end table.
- `charts/consolidated-stages-{light,dark}.svg`: per stage at the incumbents' largest batch (10k fast stages, 1k filters/MinHash), with pyroid's 3.13 and 3.15t rows.
- `charts/hero-{light,dark}.svg`: the README hero, pyroid's native chain vs datatrove at 1 and 8 workers (copied to `docs/assets/`).

