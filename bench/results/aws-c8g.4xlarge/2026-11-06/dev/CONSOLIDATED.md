# Consolidated benchmark (AWS EC2 c8g.4xlarge (AWS Graviton4, 16 vCPU), Amazon Linux 2023; CPython 3.13.18 and 3.15.2t, datatrove 0.10.1)

Machine: `aws-c8g.4xlarge`; run folder `dev`. Corpus: FineWeb dev slice (first 10,000 docs of the pinned 100k headline slice). Every row passed its equivalence gate before it was timed (byte-identical ftfy/anyascii output, fastText labels with |dp| <= 1e-6, datatrove filter decisions and text, datatrove MinHash signatures).

pyroid git sha: `f35bd9e4cbc0e75cd5066fc722c826c52dfce8cb`.

## End to end: the FineWeb chain (ftfy -> LID -> 4 filters -> MinHash dedup), 10k docs

| run | wall s, median [IQR] | CPU s | peak RSS of one process, MiB | vs datatrove, 1 proc | vs datatrove, 8 procs |
|---|---:|---:|---:|---:|---:|
| pyroid native chain, 8 threads (CPython 3.15t) | 1.64 [1.64-1.65] | 5.4 | 604 | 141.5x | 19.6x |
| pyroid stages in datatrove's executor, 8 processes (3.13) | 2.68 [2.68-2.71] | 9.6 | 351 | 86.8x | 12.0x |
| pyroid native chain, 1 thread (CPython 3.15t) | 5.15 [5.10-5.16] | 5.6 | 506 | 45.2x | 6.2x |
| pyroid stages in datatrove's executor, 1 process (3.13) | 7.52 [7.49-7.57] | 8.0 | 414 | 30.9x | 4.3x |
| datatrove 0.10.1, 8 processes (CPython 3.13) | 32.15 [32.14-32.18] | 239.8 | 366 | 7.2x | 1.0x |
| datatrove 0.10.1, 1 process (CPython 3.13) | 232.51 [231.77-232.99] | 233.0 | 446 | 1.0x | 0.1x |

Digests: datatrove at 1 and 8 workers and datatrove's executor with pyroid's stages: identical (digest-w1, digest-candidate-w1, digest-w8, digest-candidate-w8). pyroid's native chain (chain.py, 3.15t): digest `ef314b8d6ae9...`, identical to datatrove.

## Per stage: microseconds per document, median [IQR] of pyperf values

Incumbents: one process, CPython 3.13, per-doc loop over their largest measured batch (10k docs for fix_text, LID and translit; 1k for the filters and MinHash). pyroid: one call over the same docs (list[str]); `pool` = RAYON_NUM_THREADS. One-stage e2e: datatrove's executor running only that stage at 8 processes (wall time per doc, start-up and parquet IO included).

| stage | incumbent, 1 proc | incumbent, 8 procs (one-stage e2e) | pyroid 3.13, 1 thread | pyroid 3.15t, pool 1 | pyroid 3.15t, pool 8 | speedup, 1 thread vs 1 proc |
|---|---:|---:|---:|---:|---:|---:|
| fix_text, 10,000 docs | 818 [818-818] | 168 | 5.45 [5.44-5.46] | 5.54 [5.53-5.55] | 1.64 [1.64-1.64] | 148× |
| lid (lid.176.bin), 10,000 docs | 284 [282-290] | 130 | 112 [109-115] | 116 [112-120] | 14.4 [14.3-14.5] | 2.4× |
| lid (lid.176.ftz), 10,000 docs | 500 [497-504] | 130 | 88.6 [88.4-88.8] | 89.2 [89.0-89.4] | 13.1 [13.1-13.4] | 5.6× |
| gopher_repetition, 1,000 docs | 6,494 [6,476-6,525] | 961 | 154 [154-155] | 158 [156-159] | 19.2 [19.1-19.6] | 41× |
| gopher_quality, 1,000 docs | 5,125 [5,116-5,151] | 780 | 87.1 [86.3-89.0] | 92.1 [89.7-95.9] | 9.34 [9.03-9.48] | 56× |
| c4_quality, 1,000 docs | 4,618 [4,571-4,633] | - | 55.7 [55.3-55.9] | 55.8 [55.1-56.2] | 9.34 [9.17-9.47] | 83× |
| c4_quality@fineweb, 1,000 docs | 5,969 [5,932-5,990] | 901 | 101 [100-103] | 101 [101-102] | 11.0 [10.6-11.1] | 59× |
| fineweb_quality, 1,000 docs | 4,748 [4,734-4,771] | 730 | 81.6 [81.3-82.7] | 78.5 [77.4-81.5] | 7.85 [7.52-8.10] | 60× |
| minhash, 1,000 docs | 5,541 [5,501-5,557] | 862 | 116 [116-116] | 117 [116-117] | 20.7 [20.7-20.8] | 47× |
| translit, 10,000 docs | 109 [109-110] | - | 1.63 [1.57-1.68] | 1.39 [1.37-1.40] | 0.87 [0.86-0.88] | 79× |

MinHash estimators with other hash families (gated on near-duplicate precision/recall >= 0.90, not on identical signatures): rensa 0.5.0 RMinHash.update(112) per doc: 135 us/doc; rensa 0.5.0 RMinHash.from_token_sets(112): 154 us/doc; datasketch 2.0.0 MinHash.bulk(num_perm=112): 473 us/doc.

pyroid on the filters and MinHash with 10k docs per call (3.15t, list[str]; larger batches use the pool better): c4_quality pool 1: 85.6; c4_quality pool 8: 12.3; c4_quality@fineweb pool 1: 105; c4_quality@fineweb pool 8: 13.8; fineweb_quality pool 1: 85.2; fineweb_quality pool 8: 11.3; gopher_quality pool 1: 92.6; gopher_quality pool 8: 13.0; gopher_repetition pool 1: 159; gopher_repetition pool 8: 21.4; minhash pool 1: 130; minhash pool 8: 19.5 us/doc.

pyroid with Arrow input (3.15t, 10k docs): pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] [3.15t, pool 1]: 5.15 us/doc; pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] [3.15t, pool 8]: 0.80 us/doc; pyroid.translit 1.0.0.dev0 anyascii [arrow] [3.15t, pool 1]: 1.13 us/doc; pyroid.translit 1.0.0.dev0 anyascii [arrow] [3.15t, pool 8]: 0.29 us/doc.

## Charts

- `charts/consolidated-e2e-{light,dark}.svg`: the end-to-end table.
- `charts/consolidated-stages-{light,dark}.svg`: per stage at the incumbents' largest batch (10k fast stages, 1k filters/MinHash), with pyroid's 3.13 and 3.15t rows.
- `charts/hero-{light,dark}.svg`: the README hero, pyroid's native chain vs datatrove at 1 and 8 workers (copied to `docs/assets/`).

