# Consolidated benchmark (AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, 16 vCPU), Amazon Linux 2023; CPython 3.13.18 and 3.15.2t, datatrove 0.10.1)

Machine: `aws-c7i.4xlarge`; run folder `dev`. Corpus: FineWeb dev slice (first 10,000 docs of the pinned 100k headline slice). Every row passed its equivalence gate before it was timed (byte-identical ftfy/anyascii output, fastText labels with |dp| <= 1e-6, datatrove filter decisions and text, datatrove MinHash signatures).

pyroid git sha: `f35bd9e4cbc0e75cd5066fc722c826c52dfce8cb`.

## End to end: the FineWeb chain (ftfy -> LID -> 4 filters -> MinHash dedup), 10k docs

| run | wall s, median [IQR] | CPU s | peak RSS of one process, MiB | vs datatrove, 1 proc | vs datatrove, 8 procs |
|---|---:|---:|---:|---:|---:|
| pyroid native chain, 8 threads (CPython 3.15t) | 1.95 [1.90-2.01] | 6.6 | 593 | 109.4x | 18.2x |
| pyroid stages in datatrove's executor, 8 processes (3.13) | 3.15 [3.14-3.18] | 11.3 | 377 | 67.5x | 11.2x |
| pyroid native chain, 1 thread (CPython 3.15t) | 5.61 [5.59-5.62] | 6.4 | 509 | 37.9x | 6.3x |
| pyroid stages in datatrove's executor, 1 process (3.13) | 8.52 [8.48-8.54] | 9.3 | 422 | 25.0x | 4.2x |
| datatrove 0.10.1, 8 processes (CPython 3.13) | 35.37 [35.24-35.52] | 263.8 | 378 | 6.0x | 1.0x |
| datatrove 0.10.1, 1 process (CPython 3.13) | 212.76 [210.27-239.77] | 213.5 | 443 | 1.0x | 0.2x |

Digests: datatrove at 1 and 8 workers and datatrove's executor with pyroid's stages: identical (digest-w1, digest-candidate-w1, digest-w8, digest-candidate-w8). pyroid's native chain (chain.py, 3.15t): digest `ef314b8d6ae9...`, identical to datatrove.

## Per stage: microseconds per document, median [IQR] of pyperf values

Incumbents: one process, CPython 3.13, per-doc loop over their largest measured batch (10k docs for fix_text, LID and translit; 1k for the filters and MinHash). pyroid: one call over the same docs (list[str]); `pool` = RAYON_NUM_THREADS. One-stage e2e: datatrove's executor running only that stage at 8 processes (wall time per doc, start-up and parquet IO included).

| stage | incumbent, 1 proc | incumbent, 8 procs (one-stage e2e) | pyroid 3.13, 1 thread | pyroid 3.15t, pool 1 | pyroid 3.15t, pool 8 | speedup, 1 thread vs 1 proc |
|---|---:|---:|---:|---:|---:|---:|
| fix_text, 10,000 docs | 724 [722-726] | 183 | 5.32 [5.31-5.34] | 6.70 [6.68-6.73] | 2.06 [2.04-2.08] | 108× |
| lid (lid.176.bin), 10,000 docs | 185 [184-186] | 140 | 89.2 [88.9-89.7] | 114 [113-115] | 16.8 [16.4-17.4] | 1.6× |
| lid (lid.176.ftz), 10,000 docs | 373 [372-374] | 140 | 93.3 [93.2-93.4] | 103 [102-103] | 16.3 [16.2-16.4] | 3.6× |
| gopher_repetition, 1,000 docs | 6,105 [6,016-6,136] | 1,065 | 142 [141-142] | 174 [172-175] | 26.4 [25.8-28.3] | 35× |
| gopher_quality, 1,000 docs | 4,615 [4,601-4,643] | 850 | 82.9 [82.5-83.2] | 99.9 [98.8-105] | 16.3 [15.8-18.0] | 46× |
| c4_quality, 1,000 docs | 4,508 [4,478-4,542] | - | 52.8 [52.5-53.2] | 65.7 [65.0-66.4] | 14.5 [14.2-14.8] | 69× |
| c4_quality@fineweb, 1,000 docs | 5,967 [5,868-6,012] | 1,075 | 94.5 [94.3-94.7] | 114 [113-116] | 19.5 [17.4-20.7] | 52× |
| fineweb_quality, 1,000 docs | 4,287 [4,266-4,305] | 801 | 72.2 [71.8-72.4] | 90.8 [89.7-93.9] | 13.2 [12.5-15.4] | 47× |
| minhash, 1,000 docs | 4,940 [4,911-4,976] | 938 | 123 [122-124] | 146 [146-150] | 28.0 [27.8-28.2] | 34× |
| translit, 10,000 docs | 92.2 [92.0-92.5] | - | 2.18 [2.08-2.24] | 1.74 [1.71-1.77] | 1.01 [1.00-1.04] | 53× |

MinHash estimators with other hash families (gated on near-duplicate precision/recall >= 0.90, not on identical signatures): rensa 0.5.0 RMinHash.update(112) per doc: 144 us/doc; rensa 0.5.0 RMinHash.from_token_sets(112): 161 us/doc; datasketch 2.0.0 MinHash.bulk(num_perm=112): 436 us/doc.

pyroid on the filters and MinHash with 10k docs per call (3.15t, list[str]; larger batches use the pool better): c4_quality pool 1: 95.5; c4_quality pool 8: 14.4; c4_quality@fineweb pool 1: 118; c4_quality@fineweb pool 8: 16.7; fineweb_quality pool 1: 92.2; fineweb_quality pool 8: 12.7; gopher_quality pool 1: 101; gopher_quality pool 8: 14.1; gopher_repetition pool 1: 173; gopher_repetition pool 8: 23.8; minhash pool 1: 157; minhash pool 8: 23.9 us/doc.

pyroid with Arrow input (3.15t, 10k docs): pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] [3.15t, pool 1]: 6.63 us/doc; pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] [3.15t, pool 8]: 1.31 us/doc; pyroid.translit 1.0.0.dev0 anyascii [arrow] [3.15t, pool 1]: 1.79 us/doc; pyroid.translit 1.0.0.dev0 anyascii [arrow] [3.15t, pool 8]: 0.60 us/doc.

## Charts

- `charts/consolidated-e2e-{light,dark}.svg`: the end-to-end table.
- `charts/consolidated-stages-{light,dark}.svg`: per stage at the incumbents' largest batch (10k fast stages, 1k filters/MinHash), with pyroid's 3.13 and 3.15t rows.
- `charts/hero-{light,dark}.svg`: the README hero, pyroid's native chain vs datatrove at 1 and 8 workers (copied to `docs/assets/`).

