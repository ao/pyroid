## Benchmark results: aws-c7i.4xlarge, 2026-11-06

*AWS EC2 c7i.4xlarge (a VM on real hardware, not bare metal).* CPU: Intel(R) Xeon(R) Platinum 8488C, 16 logical CPUs (affinity 16); kernel 6.18.56-125.164.amzn2023.x86_64; Python 3.13.18 (GIL on); git f35bd9e4cbc0 (git); manifest sha256 c69d5b111fdb.

Load average (1/5/15 min) before: 2.03 / 2.50 / 2.67; after: 7.00 / 5.14 / 2.95. Other sessions share this VM.

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
| datatrove-0.10.1-w1 | 2444.38 | 2444.38 to 2444.38 | 2444.38 ± 0.00 | 2445.1 | 1.00 | 725 |  | 725 | 24.44 | 1 |
| pyroid-w1 | 64.42 | 64.42 to 64.42 | 64.42 ± 0.00 | 65.1 | 1.01 | 1621 |  | 457 | 0.64 | 1 |
| datatrove-0.10.1-w8 | 272.09 | 272.09 to 272.09 | 272.09 ± 0.00 | 2122.2 | 7.80 | 449 |  | 3603 | 2.72 | 1 |
| pyroid-w8 | 19.96 | 19.96 to 19.96 | 19.96 ± 0.00 | 69.7 | 3.49 | 1515 |  | 4719 | 0.20 | 1 |

hyperfine `memory_peak_resident` is the largest single process, not the simultaneous total; the cgroup column is the container's anonymous memory sampled every 0.1 s during the run (it can miss short spikes; the median includes the warmup run), and the last memory column sums per-worker peaks (an upper bound).
