## Benchmark results: aws-c8g.4xlarge, 2026-11-06

*AWS EC2 c8g.4xlarge (a VM on real hardware, not bare metal).* CPU: ARM aarch64 (model name hidden by the VM), 16 logical CPUs (affinity 16); kernel 6.18.56-125.164.amzn2023.aarch64; Python 3.13.18 (GIL on); git f35bd9e4cbc0 (git); manifest sha256 2ee2667b67b4.

Load average (1/5/15 min) before: 2.25 / 2.72 / 2.74; after: 6.52 / 5.24 / 3.07. Other sessions share this VM.

Versions: astral-dev-toolchain-hyperfine 2.0.0, datasketch 2.0.0, datatrove 0.10.1, fasttext-predict 0.9.2.4, ftfy 6.3.1, numpy 2.5.4, pyarrow 26.0.1, pyperf 2.10.0, pyroid 1.0.0.dev0, regex 2026.9.29, rensa 0.5.0, spacy 3.8.18, tokenizers 0.23.3.

### Pipeline equivalence runs

| label | corpus | docs | workers | stages | impl | digest | check-refs mismatches | coverage |
|---|---|---|---|---|---|---|---|---|
| digest-w8 | headline | 100000 | 8 | all | datatrove,fasttext,ftfy | da8979997cd283aa | 0 | 5081 full / 4919 partial / 90000 no ref |
| digest-candidate-w8 | headline | 100000 | 8 | all | pyroid.dedup,pyroid.fasttext,pyroid.filters,pyroid.ftfy | da8979997cd283aa | 0 | 5081 full / 4919 partial / 90000 no ref |

Digest check: identical across digest-w8, digest-candidate-w8.

### End to end: hyperfine-e2e.json

| command | median wall (s) | IQR (s) | mean ± sd (s) | CPU-s (median) | CPU-s / wall | largest process RSS (MiB) | cgroup anon peak, sampled (MiB) | sum of worker maxrss (MiB, bound) | wall ms / doc | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| datatrove-0.10.1-w1 | 2334.87 | 2334.87 to 2334.87 | 2334.87 ± 0.00 | 2335.4 | 1.00 | 700 |  | 700 | 23.35 | 1 |
| pyroid-w1 | 69.64 | 69.64 to 69.64 | 69.64 ± 0.00 | 70.3 | 1.01 | 1608 |  | 446 | 0.70 | 1 |
| datatrove-0.10.1-w8 | 298.46 | 298.46 to 298.46 | 298.46 ± 0.00 | 2335.1 | 7.82 | 447 |  | 3486 | 2.98 | 1 |
| pyroid-w8 | 20.26 | 20.26 to 20.26 | 20.26 ± 0.00 | 75.5 | 3.73 | 1511 |  | 4637 | 0.20 | 1 |

hyperfine `memory_peak_resident` is the largest single process, not the simultaneous total; the cgroup column is the container's anonymous memory sampled every 0.1 s during the run (it can miss short spikes; the median includes the warmup run), and the last memory column sums per-worker peaks (an upper bound).
