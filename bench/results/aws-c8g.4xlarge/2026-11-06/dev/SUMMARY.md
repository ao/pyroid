## Benchmark results: aws-c8g.4xlarge, 2026-11-06

*AWS EC2 c8g.4xlarge (a VM on real hardware, not bare metal).* CPU: ARM aarch64 (model name hidden by the VM), 16 logical CPUs (affinity 16); kernel 6.18.56-125.164.amzn2023.aarch64; Python 3.13.18 (GIL on); git f35bd9e4cbc0 (git); manifest sha256 2ee2667b67b4.

Load average (1/5/15 min) before: 1.07 / 2.25 / 2.03; after: 7.37 / 5.28 / 3.06. Other sessions share this VM.

Versions: astral-dev-toolchain-hyperfine 2.0.0, datasketch 2.0.0, datatrove 0.10.1, fasttext-predict 0.9.2.4, ftfy 6.3.1, numpy 2.5.4, pyarrow 26.0.1, pyperf 2.10.0, pyroid 1.0.0.dev0, regex 2026.9.29, rensa 0.5.0, spacy 3.8.18, tokenizers 0.23.3.

### Equivalence gates (parity, 13232 docs; all passed)

| gate | implementation | docs | result |
|---|---|---|---|
| ftfy | ftfy 6.3.1 | 13232 | 0 mismatches (13232 pinned, 0 live) |
| lid | fasttext-predict 0.9.2.4 lid.176.bin | 13232 | 0 mismatches (13232 pinned, 0 live) |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz | 13232 | 0 mismatches (13232 pinned, 0 live) |
| lid_filter | datatrove 0.10.1 LanguageFilter | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | datatrove gopher_repetition | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | datatrove gopher_quality | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | datatrove c4_quality | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | datatrove c4_quality@fineweb | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | datatrove fineweb_quality | 13232 | 0 mismatches (13232 pinned, 0 live) |
| minhash_signature | datatrove MinhashDedupSignature | 13232 | 0 mismatches (13232 pinned, 0 live) |
| minhash_estimator | datasketch 2.0.0 MinHash.bulk | 600 | P 0.9649, R 0.9649, mean abs err 0.0244 (600 pairs) |
| minhash_estimator | rensa 0.5.0 RMinHash.from_token_sets | 600 | P 0.9611, R 0.9544, mean abs err 0.0213 (600 pairs) |
| minhash_estimator_info | datatrove signature (own shingles, not gated) | 600 | P 0.9614, R 0.9614, mean abs err 0.0237 (600 pairs) |
| ftfy | pyroid.ftfy 1.0.0.dev0 | 13232 | 0 mismatches (13232 pinned, 0 live) |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin | 13232 | 0 mismatches (13232 pinned, 0 live) |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | pyroid.filters 1.0.0.dev0 gopher_repetition | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | pyroid.filters 1.0.0.dev0 gopher_quality | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | pyroid.filters 1.0.0.dev0 c4_quality | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | pyroid.filters 1.0.0.dev0 c4_quality@fineweb | 13232 | 0 mismatches (13232 pinned, 0 live) |
| filter | pyroid.filters 1.0.0.dev0 fineweb_quality | 13232 | 0 mismatches (13232 pinned, 0 live) |
| minhash_signature | pyroid.dedup 1.0.0.dev0 | 13232 | 0 mismatches (13232 pinned, 0 live) |
| minhash_dedup | pyroid.dedup 1.0.0.dev0 | 13232 | 0 mismatches (13232 pinned, 0 live) |

### Pipeline equivalence runs

| label | corpus | docs | workers | stages | impl | digest | check-refs mismatches | coverage |
|---|---|---|---|---|---|---|---|---|
| refs-fix_text | parity | 13232 | 8 | fix_text | ftfy | 0405fd3bdbf7b982 | 0 | 13232 full / 0 partial / 0 no ref |
| refs-lid | parity | 13232 | 8 | lid | fasttext | c3b3374bdc9e83f4 | 0 | 13232 full / 0 partial / 0 no ref |
| refs-gopher_repetition | parity | 13232 | 8 | gopher_repetition | datatrove | 82029db1fc622421 | 0 | 13232 full / 0 partial / 0 no ref |
| refs-gopher_quality | parity | 13232 | 8 | gopher_quality | datatrove | 53f6ce2b85762db0 | 0 | 13232 full / 0 partial / 0 no ref |
| refs-c4_quality | parity | 13232 | 8 | c4_quality | datatrove | ee6cbd251576f185 | 0 | 13232 full / 0 partial / 0 no ref |
| refs-fineweb_quality | parity | 13232 | 8 | fineweb_quality | datatrove | 120b754bb137cd45 | 0 | 13232 full / 0 partial / 0 no ref |
| refs-minhash | parity | 13232 | 8 | minhash | datatrove | 5a7cd0ebcb0cc385 | 0 | 13232 full / 0 partial / 0 no ref |
| digest-w1 | dev | 10000 | 1 | all | datatrove,fasttext,ftfy | ef314b8d6ae95cc6 | 0 | 5081 full / 4919 partial / 0 no ref |
| digest-candidate-w1 | dev | 10000 | 1 | all | pyroid.dedup,pyroid.fasttext,pyroid.filters,pyroid.ftfy | ef314b8d6ae95cc6 | 0 | 5081 full / 4919 partial / 0 no ref |
| digest-w8 | dev | 10000 | 8 | all | datatrove,fasttext,ftfy | ef314b8d6ae95cc6 | 0 | 5081 full / 4919 partial / 0 no ref |
| digest-candidate-w8 | dev | 10000 | 8 | all | pyroid.dedup,pyroid.fasttext,pyroid.filters,pyroid.ftfy | ef314b8d6ae95cc6 | 0 | 5081 full / 4919 partial / 0 no ref |

Digest check: identical across digest-w1, digest-candidate-w1, digest-w8, digest-candidate-w8.

### End to end: hyperfine-e2e.json

| command | median wall (s) | IQR (s) | mean ± sd (s) | CPU-s (median) | CPU-s / wall | largest process RSS (MiB) | cgroup anon peak, sampled (MiB) | sum of worker maxrss (MiB, bound) | wall ms / doc | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| datatrove-0.10.1-w1 | 232.51 | 232.14 to 232.75 | 232.42 ± 0.62 | 233.0 | 1.00 | 446 |  | 446 | 23.25 | 3 |
| pyroid-w1 | 7.52 | 7.50 to 7.55 | 7.53 ± 0.04 | 8.0 | 1.07 | 413 |  | 413 | 0.75 | 3 |
| datatrove-0.10.1-w8 | 32.15 | 32.15 to 32.16 | 32.16 ± 0.02 | 239.8 | 7.46 | 365 |  | 2986 | 3.22 | 3 |
| pyroid-w8 | 2.68 | 2.68 to 2.69 | 2.69 ± 0.02 | 9.6 | 3.57 | 349 |  | 2926 | 0.27 | 3 |

### End to end: hyperfine-stages-w8.json

| command | median wall (s) | IQR (s) | mean ± sd (s) | CPU-s (median) | CPU-s / wall | largest process RSS (MiB) | cgroup anon peak, sampled (MiB) | sum of worker maxrss (MiB, bound) | wall ms / doc | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text-w8 | 1.68 | 1.68 to 1.68 | 1.68 ± 0.00 | 9.8 | 5.84 | 190 |  | 1515 | 0.17 | 3 |
| lid-w8 | 1.30 | 1.30 to 1.31 | 1.31 ± 0.01 | 6.8 | 5.21 | 319 |  | 2581 | 0.13 | 3 |
| gopher_repetition-w8 | 9.61 | 9.61 to 9.62 | 9.61 ± 0.01 | 70.9 | 7.38 | 225 |  | 1865 | 0.96 | 3 |
| gopher_quality-w8 | 7.80 | 7.79 to 7.86 | 7.83 ± 0.07 | 57.4 | 7.36 | 222 |  | 1864 | 0.78 | 3 |
| c4_quality-w8 | 9.01 | 9.00 to 9.01 | 9.00 ± 0.02 | 66.4 | 7.37 | 221 |  | 1832 | 0.90 | 3 |
| fineweb_quality-w8 | 7.30 | 7.30 to 7.33 | 7.32 ± 0.03 | 53.5 | 7.33 | 226 |  | 1861 | 0.73 | 3 |
| minhash-w8 | 8.62 | 8.62 to 8.63 | 8.62 ± 0.01 | 62.2 | 7.21 | 265 |  | 2110 | 0.86 | 3 |

hyperfine `memory_peak_resident` is the largest single process, not the simultaneous total; the cgroup column is the container's anonymous memory sampled every 0.1 s during the run (it can miss short spikes; the median includes the warmup run), and the last memory column sums per-worker peaks (an upper bound).

### Per stage (pyperf; one value = one call over the batch)

| stage | implementation | docs / call | median / call | IQR / call | stdev | per doc (median) | per doc (IQR) | values | peak RSS MiB (track-memory) | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 1 | 738 ns | 737 ns to 739 ns | 2.38 ns | 738 ns | 737 ns to 739 ns | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 1 | 972 ns | 969 ns to 975 ns | 5.53 ns | 972 ns | 969 ns to 975 ns | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 1 | 2.44 µs | 2.42 µs to 2.45 µs | 17.2 ns | 2.44 µs | 2.42 µs to 2.45 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 1 | 96.6 µs | 96.6 µs to 96.8 µs | 418 ns | 96.6 µs | 96.6 µs to 96.8 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 100 | 355 µs | 354 µs to 356 µs | 1.32 µs | 3.55 µs | 3.54 µs to 3.56 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 100 | 357 µs | 356 µs to 358 µs | 1.6 µs | 3.57 µs | 3.56 µs to 3.58 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 100 | 360 µs | 359 µs to 361 µs | 1.02 µs | 3.6 µs | 3.59 µs to 3.61 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 100 | 64.3 ms | 64.2 ms to 64.3 ms | 82.1 µs | 643 µs | 642 µs to 643 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 1000 | 4.25 ms | 4.25 ms to 4.25 ms | 8.62 µs | 4.25 µs | 4.25 µs to 4.25 µs | 20 | 148 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 1000 | 4.35 ms | 4.33 ms to 4.38 ms | 171 µs | 4.35 µs | 4.33 µs to 4.38 µs | 20 | 141 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 1000 | 4.51 ms | 4.5 ms to 4.53 ms | 22 µs | 4.51 µs | 4.5 µs to 4.53 µs | 20 | 154 | 0 diff |
| fix_text | ftfy 6.3.1 | 1000 | 739 ms | 739 ms to 740 ms | 1.25 ms | 739 µs | 739 µs to 740 µs | 20 | 139 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 10000 | 51.1 ms | 51.1 ms to 51.2 ms | 87.3 µs | 5.11 µs | 5.11 µs to 5.12 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 10000 | 54.5 ms | 54.4 ms to 54.6 ms | 102 µs | 5.45 µs | 5.44 µs to 5.46 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 10000 | 57.7 ms | 57.4 ms to 57.7 ms | 177 µs | 5.77 µs | 5.74 µs to 5.77 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 10000 | 8.18 s | 8.18 s to 8.18 s | 9.46 ms | 818 µs | 818 µs to 818 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 1 | 9.69 µs | 9.55 µs to 9.75 µs | 171 ns | 9.69 µs | 9.55 µs to 9.75 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 1 | 9.71 µs | 9.65 µs to 9.77 µs | 114 ns | 9.71 µs | 9.65 µs to 9.77 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 1 | 9.77 µs | 9.7 µs to 9.86 µs | 77 ns | 9.77 µs | 9.7 µs to 9.86 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 1 | 9.79 µs | 9.78 µs to 9.81 µs | 47.5 ns | 9.79 µs | 9.78 µs to 9.81 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 1 | 10.6 µs | 10.6 µs to 10.7 µs | 72.6 ns | 10.6 µs | 10.6 µs to 10.7 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 1 | 11.2 µs | 11.1 µs to 11.2 µs | 100 ns | 11.2 µs | 11.1 µs to 11.2 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 1 | 11.9 µs | 11.9 µs to 12 µs | 73.9 ns | 11.9 µs | 11.9 µs to 12 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 1 | 12.9 µs | 12.7 µs to 12.9 µs | 132 ns | 12.9 µs | 12.7 µs to 12.9 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 1 | 13.3 µs | 13.1 µs to 13.3 µs | 89.1 ns | 13.3 µs | 13.1 µs to 13.3 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 1 | 13.8 µs | 13.8 µs to 13.8 µs | 12.5 ns | 13.8 µs | 13.8 µs to 13.8 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1 | 37.9 µs | 37.8 µs to 38.2 µs | 189 ns | 37.9 µs | 37.8 µs to 38.2 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1 | 41.5 µs | 41.3 µs to 41.6 µs | 191 ns | 41.5 µs | 41.3 µs to 41.6 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1 | 59.1 µs | 59 µs to 59.3 µs | 391 ns | 59.1 µs | 59 µs to 59.3 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1 | 62.3 µs | 62 µs to 62.6 µs | 500 ns | 62.3 µs | 62 µs to 62.6 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1 | 145 µs | 144 µs to 146 µs | 1.2 µs | 145 µs | 144 µs to 146 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 100 | 8.1 ms | 8.08 ms to 8.1 ms | 16.8 µs | 81 µs | 80.8 µs to 81 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 100 | 8.17 ms | 8.16 ms to 8.18 ms | 15.8 µs | 81.7 µs | 81.6 µs to 81.8 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 100 | 8.29 ms | 8.27 ms to 8.29 ms | 24.8 µs | 82.9 µs | 82.7 µs to 82.9 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 100 | 8.39 ms | 8.37 ms to 8.39 ms | 12.2 µs | 83.9 µs | 83.7 µs to 83.9 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 100 | 8.63 ms | 8.61 ms to 8.64 ms | 28.3 µs | 86.3 µs | 86.1 µs to 86.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 100 | 10 ms | 9.72 ms to 10.2 ms | 490 µs | 100 µs | 97.2 µs to 102 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 100 | 10.4 ms | 10.2 ms to 10.6 ms | 325 µs | 104 µs | 102 µs to 106 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 100 | 10.5 ms | 10.4 ms to 10.7 ms | 267 µs | 105 µs | 104 µs to 107 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 100 | 10.7 ms | 10.6 ms to 11 ms | 254 µs | 107 µs | 106 µs to 110 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 100 | 10.9 ms | 10.9 ms to 11.1 ms | 296 µs | 109 µs | 109 µs to 111 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 100 | 25.4 ms | 25 ms to 25.7 ms | 500 µs | 254 µs | 250 µs to 257 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 100 | 26.7 ms | 26.5 ms to 27.2 ms | 378 µs | 267 µs | 265 µs to 272 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 100 | 37.4 ms | 37.2 ms to 37.7 ms | 516 µs | 374 µs | 372 µs to 377 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 100 | 45.1 ms | 44.9 ms to 45.7 ms | 658 µs | 451 µs | 449 µs to 457 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 100 | 45.6 ms | 44.9 ms to 46 ms | 739 µs | 456 µs | 449 µs to 460 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 1000 | 87.1 ms | 87 ms to 87.2 ms | 197 µs | 87.1 µs | 87 µs to 87.2 µs | 20 | 143 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 1000 | 87.9 ms | 87.8 ms to 88 ms | 201 µs | 87.9 µs | 87.8 µs to 88 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 1000 | 88.8 ms | 88.6 ms to 89.1 ms | 326 µs | 88.8 µs | 88.6 µs to 89.1 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 1000 | 90.3 ms | 90.2 ms to 90.4 ms | 204 µs | 90.3 µs | 90.2 µs to 90.4 µs | 20 | 143 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 1000 | 92.8 ms | 92.5 ms to 93.2 ms | 443 µs | 92.8 µs | 92.5 µs to 93.2 µs | 20 | 137 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 1000 | 109 ms | 108 ms to 111 ms | 2.8 ms | 109 µs | 108 µs to 111 µs | 20 | 273 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 1000 | 114 ms | 111 ms to 116 ms | 3.74 ms | 114 µs | 111 µs to 116 µs | 20 | 266 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 1000 | 117 ms | 114 ms to 119 ms | 3.92 ms | 117 µs | 114 µs to 119 µs | 20 | 266 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 1000 | 117 ms | 116 ms to 119 ms | 2.25 ms | 117 µs | 116 µs to 119 µs | 20 | 267 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 1000 | 118 ms | 117 ms to 121 ms | 3.23 ms | 118 µs | 117 µs to 121 µs | 20 | 273 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1000 | 272 ms | 266 ms to 277 ms | 6.94 ms | 272 µs | 266 µs to 277 µs | 20 | 266 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1000 | 272 ms | 269 ms to 274 ms | 3.48 ms | 272 µs | 269 µs to 274 µs | 20 | 266 | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1000 | 375 ms | 365 ms to 380 ms | 8.23 ms | 375 µs | 365 µs to 380 µs | 20 | 286 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1000 | 479 ms | 475 ms to 485 ms | 6.81 ms | 479 µs | 475 µs to 485 µs | 20 | 136 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1000 | 489 ms | 484 ms to 495 ms | 8.27 ms | 489 µs | 484 µs to 495 µs | 20 | 137 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 10000 | 886 ms | 884 ms to 887 ms | 2.13 ms | 88.6 µs | 88.4 µs to 88.7 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 10000 | 895 ms | 893 ms to 896 ms | 2.3 ms | 89.5 µs | 89.3 µs to 89.6 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 10000 | 900 ms | 899 ms to 902 ms | 2.22 ms | 90 µs | 89.9 µs to 90.2 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 10000 | 921 ms | 920 ms to 923 ms | 2.18 ms | 92.1 µs | 92 µs to 92.3 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 10000 | 949 ms | 944 ms to 956 ms | 6.86 ms | 94.9 µs | 94.4 µs to 95.6 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 10000 | 1.12 s | 1.09 s to 1.15 s | 31.2 ms | 112 µs | 109 µs to 115 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 10000 | 1.14 s | 1.13 s to 1.16 s | 29.1 ms | 114 µs | 113 µs to 116 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 10000 | 1.14 s | 1.13 s to 1.16 s | 42.9 ms | 114 µs | 113 µs to 116 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 10000 | 1.16 s | 1.15 s to 1.21 s | 44.1 ms | 116 µs | 115 µs to 121 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 10000 | 1.24 s | 1.22 s to 1.26 s | 31.8 ms | 124 µs | 122 µs to 126 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 10000 | 2.84 s | 2.82 s to 2.89 s | 54.1 ms | 284 µs | 282 µs to 289 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 10000 | 2.84 s | 2.83 s to 2.88 s | 55.5 ms | 284 µs | 283 µs to 288 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 10000 | 3.84 s | 3.81 s to 3.93 s | 73.5 ms | 384 µs | 381 µs to 393 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 10000 | 5 s | 4.98 s to 5.04 s | 62.4 ms | 500 µs | 498 µs to 504 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 10000 | 5.09 s | 5.05 s to 5.12 s | 78.8 ms | 509 µs | 505 µs to 512 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 1 | 26.5 µs | 26.5 µs to 26.5 µs | 41.4 ns | 26.5 µs | 26.5 µs to 26.5 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 1 | 27.1 µs | 27 µs to 27.1 µs | 79.7 ns | 27.1 µs | 27 µs to 27.1 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 1 | 28.2 µs | 28.1 µs to 28.2 µs | 93.8 ns | 28.2 µs | 28.1 µs to 28.2 µs | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1 | 1.1 ms | 1.09 ms to 1.1 ms | 9.41 µs | 1.1 ms | 1.09 ms to 1.1 ms | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 100 | 9.97 ms | 9.96 ms to 10 ms | 35.3 µs | 99.7 µs | 99.6 µs to 100 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 100 | 9.99 ms | 9.96 ms to 10 ms | 30.6 µs | 99.9 µs | 99.6 µs to 100 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 100 | 11.4 ms | 11.3 ms to 11.4 ms | 51.3 µs | 114 µs | 113 µs to 114 µs | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 100 | 612 ms | 609 ms to 614 ms | 3.07 ms | 6.12 ms | 6.09 ms to 6.14 ms | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 1000 | 154 ms | 154 ms to 155 ms | 1.14 ms | 154 µs | 154 µs to 155 µs | 20 | 150 | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 1000 | 155 ms | 154 ms to 156 ms | 1.42 ms | 155 µs | 154 µs to 156 µs | 20 | 150 | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 1000 | 173 ms | 171 ms to 173 ms | 1.45 ms | 173 µs | 171 µs to 173 µs | 20 | 159 | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1000 | 6.49 s | 6.48 s to 6.52 s | 32.9 ms | 6.49 ms | 6.48 ms to 6.52 ms | 20 | 196 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 1 | 12.8 µs | 12.7 µs to 12.8 µs | 50.8 ns | 12.8 µs | 12.7 µs to 12.8 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 1 | 13.3 µs | 13.3 µs to 13.4 µs | 83.4 ns | 13.3 µs | 13.3 µs to 13.4 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 1 | 14.6 µs | 14.5 µs to 14.6 µs | 56.8 ns | 14.6 µs | 14.5 µs to 14.6 µs | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1 | 905 µs | 899 µs to 908 µs | 8.33 µs | 905 µs | 899 µs to 908 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 100 | 4.11 ms | 4.1 ms to 4.11 ms | 8.07 µs | 41.1 µs | 41 µs to 41.1 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 100 | 4.12 ms | 4.11 ms to 4.12 ms | 8.72 µs | 41.2 µs | 41.1 µs to 41.2 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 100 | 5.05 ms | 5.03 ms to 5.06 ms | 25.6 µs | 50.5 µs | 50.3 µs to 50.6 µs | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 100 | 486 ms | 486 ms to 489 ms | 3.57 ms | 4.86 ms | 4.86 ms to 4.89 ms | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 1000 | 87.1 ms | 86.4 ms to 88.6 ms | 1.36 ms | 87.1 µs | 86.4 µs to 88.6 µs | 20 | 148 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 1000 | 87.7 ms | 86.8 ms to 88.9 ms | 1.14 ms | 87.7 µs | 86.8 µs to 88.9 µs | 20 | 148 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 1000 | 99.6 ms | 98.8 ms to 101 ms | 1.33 ms | 99.6 µs | 98.8 µs to 101 µs | 20 | 148 | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1000 | 5.13 s | 5.12 s to 5.14 s | 26.6 ms | 5.13 ms | 5.12 ms to 5.14 ms | 20 | 196 | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1 | 9.16 µs | 9.14 µs to 9.17 µs | 33.3 ns | 9.16 µs | 9.14 µs to 9.17 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 1 | 10.3 µs | 10.3 µs to 10.3 µs | 40.8 ns | 10.3 µs | 10.3 µs to 10.3 µs | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1 | 424 µs | 422 µs to 430 µs | 5.4 µs | 424 µs | 422 µs to 430 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 100 | 3.71 ms | 3.7 ms to 3.72 ms | 11.3 µs | 37.1 µs | 37 µs to 37.2 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 100 | 3.73 ms | 3.71 ms to 3.73 ms | 11.6 µs | 37.3 µs | 37.1 µs to 37.3 µs | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 100 | 401 ms | 401 ms to 404 ms | 2.65 ms | 4.01 ms | 4.01 ms to 4.04 ms | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1000 | 55.7 ms | 55.3 ms to 55.9 ms | 448 µs | 55.7 µs | 55.3 µs to 55.9 µs | 20 | 152 | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 1000 | 56.2 ms | 56 ms to 56.4 ms | 379 µs | 56.2 µs | 56 µs to 56.4 µs | 20 | 152 | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1000 | 4.62 s | 4.57 s to 4.63 s | 36.2 ms | 4.62 ms | 4.57 ms to 4.63 ms | 20 | 184 | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1 | 12.5 µs | 12.5 µs to 12.5 µs | 43 ns | 12.5 µs | 12.5 µs to 12.5 µs | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 1 | 13.7 µs | 13.7 µs to 13.7 µs | 47.7 ns | 13.7 µs | 13.7 µs to 13.7 µs | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1 | 1 ms | 998 µs to 1.01 ms | 10.2 µs | 1 ms | 998 µs to 1.01 ms | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 100 | 5 ms | 5 ms to 5.02 ms | 16.4 µs | 50 µs | 50 µs to 50.2 µs | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 100 | 5.01 ms | 4.99 ms to 5.01 ms | 15 µs | 50.1 µs | 49.9 µs to 50.1 µs | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 100 | 578 ms | 576 ms to 581 ms | 5.08 ms | 5.78 ms | 5.76 ms to 5.81 ms | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 1000 | 99.1 ms | 97.5 ms to 100 ms | 1.67 ms | 99.1 µs | 97.5 µs to 100 µs | 20 | 148 | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1000 | 101 ms | 100 ms to 102 ms | 1.43 ms | 101 µs | 100 µs to 102 µs | 20 | 148 | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1000 | 5.97 s | 5.95 s to 5.99 s | 65.8 ms | 5.97 ms | 5.95 ms to 5.99 ms | 20 | 186 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 1 | 8.55 µs | 8.55 µs to 8.56 µs | 17.7 ns | 8.55 µs | 8.55 µs to 8.56 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 1 | 9.21 µs | 9.18 µs to 9.23 µs | 33.1 ns | 9.21 µs | 9.18 µs to 9.23 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 1 | 10.3 µs | 10.3 µs to 10.3 µs | 35.5 ns | 10.3 µs | 10.3 µs to 10.3 µs | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1 | 842 µs | 836 µs to 848 µs | 9.75 µs | 842 µs | 836 µs to 848 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 100 | 3.36 ms | 3.36 ms to 3.36 ms | 9.11 µs | 33.6 µs | 33.6 µs to 33.6 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 100 | 3.38 ms | 3.38 ms to 3.39 ms | 7.64 µs | 33.8 µs | 33.8 µs to 33.9 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 100 | 3.87 ms | 3.87 ms to 3.88 ms | 10 µs | 38.7 µs | 38.7 µs to 38.8 µs | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 100 | 446 ms | 445 ms to 448 ms | 1.47 ms | 4.46 ms | 4.45 ms to 4.48 ms | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 1000 | 81.1 ms | 80.3 ms to 81.6 ms | 1.02 ms | 81.1 µs | 80.3 µs to 81.6 µs | 20 | 148 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 1000 | 81.6 ms | 81.3 ms to 82.7 ms | 883 µs | 81.6 µs | 81.3 µs to 82.7 µs | 20 | 148 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 1000 | 88.2 ms | 87.7 ms to 89 ms | 1.29 ms | 88.2 µs | 87.7 µs to 89 µs | 20 | 147 | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1000 | 4.75 s | 4.74 s to 4.77 s | 27.9 ms | 4.75 ms | 4.74 ms to 4.77 ms | 20 | 197 | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1 | 13.5 µs | 13.5 µs to 13.5 µs | 54 ns | 13.5 µs | 13.5 µs to 13.5 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 1 | 27.3 µs | 27.2 µs to 27.4 µs | 153 ns | 27.3 µs | 27.2 µs to 27.4 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 1 | 28 µs | 27.9 µs to 28 µs | 63.3 ns | 28 µs | 27.9 µs to 28 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 1 | 28.1 µs | 28.1 µs to 28.2 µs | 69.1 ns | 28.1 µs | 28.1 µs to 28.2 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 1 | 30.9 µs | 30.8 µs to 31 µs | 138 ns | 30.9 µs | 30.8 µs to 31 µs | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1 | 35.2 µs | 34.7 µs to 36.3 µs | 1.2 µs | 35.2 µs | 34.7 µs to 36.3 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1 | 171 µs | 171 µs to 172 µs | 705 ns | 171 µs | 171 µs to 172 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1 | 866 µs | 860 µs to 869 µs | 6.68 µs | 866 µs | 860 µs to 869 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 100 | 9.59 ms | 9.58 ms to 9.6 ms | 14.1 µs | 95.9 µs | 95.8 µs to 96 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 100 | 9.78 ms | 9.77 ms to 9.78 ms | 15.5 µs | 97.8 µs | 97.7 µs to 97.8 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 100 | 10 ms | 10 ms to 10 ms | 27.5 µs | 100 µs | 100 µs to 100 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 100 | 11.3 ms | 11.3 ms to 11.3 ms | 31.2 µs | 113 µs | 113 µs to 113 µs | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 100 | 12.2 ms | 12.2 ms to 12.3 ms | 91 µs | 122 µs | 122 µs to 123 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 100 | 13.3 ms | 13.2 ms to 13.3 ms | 168 µs | 133 µs | 132 µs to 133 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 100 | 44.5 ms | 44.3 ms to 44.8 ms | 313 µs | 445 µs | 443 µs to 448 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 100 | 514 ms | 512 ms to 515 ms | 2 ms | 5.14 ms | 5.12 ms to 5.15 ms | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 1000 | 112 ms | 111 ms to 112 ms | 548 µs | 112 µs | 111 µs to 112 µs | 20 | 145 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 1000 | 116 ms | 116 ms to 116 ms | 460 µs | 116 µs | 116 µs to 116 µs | 20 | 147 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 1000 | 119 ms | 119 ms to 120 ms | 386 µs | 119 µs | 119 µs to 120 µs | 20 | 150 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 1000 | 134 ms | 134 ms to 134 ms | 717 µs | 134 µs | 134 µs to 134 µs | 20 | 147 | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1000 | 135 ms | 134 ms to 136 ms | 1.29 ms | 135 µs | 134 µs to 136 µs | 20 | 147 | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1000 | 154 ms | 153 ms to 154 ms | 893 µs | 154 µs | 153 µs to 154 µs | 20 | 161 | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1000 | 473 ms | 472 ms to 475 ms | 2.41 ms | 473 µs | 472 µs to 475 µs | 20 | 219 | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1000 | 5.54 s | 5.5 s to 5.55 s | 34.1 ms | 5.54 ms | 5.5 ms to 5.55 ms | 20 | 216 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 1 | 344 ns | 342 ns to 348 ns | 3.66 ns | 344 ns | 342 ns to 348 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 1 | 376 ns | 375 ns to 378 ns | 2.31 ns | 376 ns | 375 ns to 378 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 1 | 1.84 µs | 1.84 µs to 1.85 µs | 9.91 ns | 1.84 µs | 1.84 µs to 1.85 µs | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 1 | 20.2 µs | 20.2 µs to 20.3 µs | 71.2 ns | 20.2 µs | 20.2 µs to 20.3 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 100 | 56.4 µs | 56.3 µs to 56.5 µs | 127 ns | 564 ns | 563 ns to 565 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 100 | 68.9 µs | 68.7 µs to 69 µs | 339 ns | 689 ns | 687 ns to 690 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 100 | 76.5 µs | 76.3 µs to 76.7 µs | 336 ns | 765 ns | 763 ns to 767 ns | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 100 | 8.88 ms | 8.86 ms to 8.9 ms | 20.2 µs | 88.8 µs | 88.6 µs to 89 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 1000 | 794 µs | 793 µs to 795 µs | 3.48 µs | 794 ns | 793 ns to 795 ns | 20 | 139 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 1000 | 839 µs | 833 µs to 845 µs | 9.08 µs | 839 ns | 833 ns to 845 ns | 20 | 150 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 1000 | 1.17 ms | 1.16 ms to 1.18 ms | 15.8 µs | 1.17 µs | 1.16 µs to 1.18 µs | 20 | 142 | 0 diff |
| translit | anyascii 0.3.3 | 1000 | 105 ms | 105 ms to 105 ms | 258 µs | 105 µs | 105 µs to 105 µs | 20 | 131 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 10000 | 11.2 ms | 11.1 ms to 11.3 ms | 86.8 µs | 1.12 µs | 1.11 µs to 1.13 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 10000 | 13.5 ms | 13.4 ms to 13.5 ms | 137 µs | 1.35 µs | 1.34 µs to 1.35 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 10000 | 16.3 ms | 15.7 ms to 16.8 ms | 792 µs | 1.63 µs | 1.57 µs to 1.68 µs | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 10000 | 1.09 s | 1.09 s to 1.1 s | 5.66 ms | 109 µs | 109 µs to 110 µs | 20 |  | 0 diff |
