# Consolidated benchmark (AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, 16 vCPU), Amazon Linux 2023; CPython 3.13.18 and 3.15.2t, datatrove 0.10.1)

Machine: `aws-c7i.4xlarge`; run folder `headline`. Corpus: FineWeb headline slice (the pinned 100,000 docs). Every row passed the full-chain digest gate below before it was timed.

pyroid git sha: `f35bd9e4cbc0e75cd5066fc722c826c52dfce8cb`.

## End to end: the FineWeb chain (ftfy -> LID -> 4 filters -> MinHash dedup), 100k docs

| run | wall s, median [IQR] | CPU s | peak RSS of one process, MiB | vs datatrove, 1 proc | vs datatrove, 8 procs |
|---|---:|---:|---:|---:|---:|
| pyroid native chain, 8 threads (CPython 3.15t) | 10.17 [10.15-10.23] | 45.3 | 2234 | 240.3x | 26.7x |
| pyroid stages in datatrove's executor, 8 processes (3.13) | 19.96 [19.96-19.96] | 69.7 | 1515 | 122.5x | 13.6x |
| pyroid native chain, 1 thread (CPython 3.15t) | 43.37 [43.36-43.70] | 44.6 | 2123 | 56.4x | 6.3x |
| pyroid stages in datatrove's executor, 1 process (3.13) | 64.42 [64.42-64.42] | 65.1 | 1621 | 37.9x | 4.2x |
| datatrove 0.10.1, 8 processes (CPython 3.13) | 272.09 [272.09-272.09] | 2122.2 | 449 | 9.0x | 1.0x |
| datatrove 0.10.1, 1 process (CPython 3.13) | 2444.38 [2444.38-2444.38] | 2445.1 | 725 | 1.0x | 0.1x |

Digests: datatrove at 8 workers and datatrove's executor with pyroid's stages: identical (digest-w8, digest-candidate-w8). pyroid's native chain (chain.py, 3.15t): digest `da8979997cd2...`, identical to datatrove.

Per-stage rows: not measured in this run (no `stages/*.json` or no `ft315t/stages/*.json`).

## Charts

- `charts/consolidated-e2e-{light,dark}.svg`: the end-to-end table.
- `charts/hero-{light,dark}.svg`: the README hero, pyroid's native chain vs datatrove at 1 and 8 workers (copied to `docs/assets/`).

