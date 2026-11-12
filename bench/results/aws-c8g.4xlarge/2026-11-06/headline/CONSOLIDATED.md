# Consolidated benchmark (AWS EC2 c8g.4xlarge (AWS Graviton4, 16 vCPU), Amazon Linux 2023; CPython 3.13.18 and 3.15.2t, datatrove 0.10.1)

Machine: `aws-c8g.4xlarge`; run folder `headline`. Corpus: FineWeb headline slice (the pinned 100,000 docs). Every row passed the full-chain digest gate below before it was timed.

pyroid git sha: `f35bd9e4cbc0e75cd5066fc722c826c52dfce8cb`.

## End to end: the FineWeb chain (ftfy -> LID -> 4 filters -> MinHash dedup), 100k docs

| run | wall s, median [IQR] | CPU s | peak RSS of one process, MiB | vs datatrove, 1 proc | vs datatrove, 8 procs |
|---|---:|---:|---:|---:|---:|
| pyroid native chain, 8 threads (CPython 3.15t) | 10.16 [10.10-10.17] | 45.7 | 2222 | 229.8x | 29.4x |
| pyroid stages in datatrove's executor, 8 processes (3.13) | 20.26 [20.26-20.26] | 75.5 | 1511 | 115.3x | 14.7x |
| pyroid native chain, 1 thread (CPython 3.15t) | 46.83 [46.32-47.13] | 47.6 | 2086 | 49.9x | 6.4x |
| pyroid stages in datatrove's executor, 1 process (3.13) | 69.64 [69.64-69.64] | 70.3 | 1608 | 33.5x | 4.3x |
| datatrove 0.10.1, 8 processes (CPython 3.13) | 298.46 [298.46-298.46] | 2335.1 | 447 | 7.8x | 1.0x |
| datatrove 0.10.1, 1 process (CPython 3.13) | 2334.87 [2334.87-2334.87] | 2335.4 | 700 | 1.0x | 0.1x |

Digests: datatrove at 8 workers and datatrove's executor with pyroid's stages: identical (digest-w8, digest-candidate-w8). pyroid's native chain (chain.py, 3.15t): digest `da8979997cd2...`, identical to datatrove.

Per-stage rows: not measured in this run (no `stages/*.json` or no `ft315t/stages/*.json`).

## Charts

- `charts/consolidated-e2e-{light,dark}.svg`: the end-to-end table.
- `charts/hero-{light,dark}.svg`: the README hero, pyroid's native chain vs datatrove at 1 and 8 workers (copied to `docs/assets/`).

