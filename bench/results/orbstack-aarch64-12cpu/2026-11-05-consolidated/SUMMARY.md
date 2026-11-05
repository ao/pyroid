## Benchmark results: orbstack-aarch64-12cpu, 2026-11-05

*Docker/OrbStack VM, indicative (not README numbers).* CPU: Apple aarch64 (model name hidden by the VM), 12 logical CPUs (affinity 12); kernel 7.2.9-orbstack-00391-g2b298d05fbd4; Python 3.13.18 (GIL on); git cd77e4e (BENCH_GIT_SHA); manifest sha256 28b2499b62de.

Load average (1/5/15 min) before: 1.80 / 1.93 / 1.83; after: 6.98 / 5.66 / 3.50. Other sessions share this VM.

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
| datatrove-0.10.1-w1 | 253.20 | 252.84 to 253.32 | 253.04 ± 0.50 | 254.3 | 1.00 | 426 | 413 | 432 | 25.32 | 3 |
| pyroid-w1 | 7.48 | 7.36 to 7.50 | 7.41 ± 0.15 | 8.1 | 1.09 | 412 | 384 | 412 | 0.75 | 3 |
| datatrove-0.10.1-w8 | 41.93 | 41.60 to 42.06 | 41.79 ± 0.47 | 314.5 | 7.50 | 366 | 2356 | 2998 | 4.19 | 3 |
| pyroid-w8 | 2.85 | 2.84 to 2.88 | 2.87 ± 0.05 | 13.0 | 4.58 | 373 | 2022 | 2857 | 0.28 | 3 |

### End to end: hyperfine-stages-w8.json

| command | median wall (s) | IQR (s) | mean ± sd (s) | CPU-s (median) | CPU-s / wall | largest process RSS (MiB) | cgroup anon peak, sampled (MiB) | sum of worker maxrss (MiB, bound) | wall ms / doc | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text-w8 | 2.05 | 2.01 to 2.10 | 2.06 ± 0.09 | 13.1 | 6.40 | 166 | 831 | 1409 | 0.20 | 3 |
| lid-w8 | 1.87 | 1.86 to 1.87 | 1.87 ± 0.01 | 11.4 | 6.12 | 310 | 1927 | 2528 | 0.19 | 3 |
| gopher_repetition-w8 | 12.57 | 12.35 to 12.60 | 12.44 ± 0.27 | 93.4 | 7.43 | 226 | 1235 | 1868 | 1.26 | 3 |
| gopher_quality-w8 | 9.73 | 9.70 to 9.83 | 9.78 ± 0.13 | 72.7 | 7.47 | 227 | 1230 | 1875 | 0.97 | 3 |
| c4_quality-w8 | 11.62 | 11.59 to 11.86 | 11.76 ± 0.30 | 86.9 | 7.47 | 222 | 1223 | 1848 | 1.16 | 3 |
| fineweb_quality-w8 | 9.43 | 9.42 to 9.51 | 9.47 ± 0.10 | 70.3 | 7.45 | 225 | 1234 | 1871 | 0.94 | 3 |
| minhash-w8 | 10.46 | 10.44 to 10.56 | 10.51 ± 0.14 | 76.8 | 7.34 | 270 | 1316 | 2115 | 1.05 | 3 |

hyperfine `memory_peak_resident` is the largest single process, not the simultaneous total; the cgroup column is the container's anonymous memory sampled every 0.1 s during the run (it can miss short spikes; the median includes the warmup run), and the last memory column sums per-worker peaks (an upper bound).

### Per stage (pyperf; one value = one call over the batch)

| stage | implementation | docs / call | median / call | IQR / call | stdev | per doc (median) | per doc (IQR) | values | peak RSS MiB (track-memory) | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 1 | 706 ns | 696 ns to 737 ns | 23.8 ns | 706 ns | 696 ns to 737 ns | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 1 | 961 ns | 952 ns to 1 µs | 73.7 ns | 961 ns | 952 ns to 1 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 1 | 2.13 µs | 2.09 µs to 2.15 µs | 106 ns | 2.13 µs | 2.09 µs to 2.15 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 1 | 98.8 µs | 95.8 µs to 102 µs | 3.81 µs | 98.8 µs | 95.8 µs to 102 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 100 | 265 µs | 261 µs to 268 µs | 12.2 µs | 2.65 µs | 2.61 µs to 2.68 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 100 | 272 µs | 265 µs to 275 µs | 14.5 µs | 2.72 µs | 2.65 µs to 2.75 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 100 | 279 µs | 270 µs to 286 µs | 15.5 µs | 2.79 µs | 2.7 µs to 2.86 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 100 | 64.3 ms | 63.9 ms to 66.8 ms | 3.11 ms | 643 µs | 639 µs to 668 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 1000 | 4.06 ms | 4.04 ms to 4.32 ms | 351 µs | 4.06 µs | 4.04 µs to 4.32 µs | 20 | 147 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 1000 | 4.14 ms | 4.09 ms to 4.42 ms | 279 µs | 4.14 µs | 4.09 µs to 4.42 µs | 20 | 140 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 1000 | 4.27 ms | 4.2 ms to 4.47 ms | 323 µs | 4.27 µs | 4.2 µs to 4.47 µs | 20 | 138 | 0 diff |
| fix_text | ftfy 6.3.1 | 1000 | 754 ms | 750 ms to 759 ms | 7.5 ms | 754 µs | 750 µs to 759 µs | 20 | 139 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 10000 | 51.5 ms | 51.3 ms to 52.1 ms | 1.9 ms | 5.15 µs | 5.13 µs to 5.21 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 10000 | 54.7 ms | 54.1 ms to 57.3 ms | 4.32 ms | 5.47 µs | 5.41 µs to 5.73 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 10000 | 57.8 ms | 57.2 ms to 62.6 ms | 6.89 ms | 5.78 µs | 5.72 µs to 6.26 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 10000 | 8.34 s | 8.31 s to 8.37 s | 92.1 ms | 834 µs | 831 µs to 837 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 1 | 10.3 µs | 10.3 µs to 10.4 µs | 102 ns | 10.3 µs | 10.3 µs to 10.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 1 | 10.4 µs | 10.3 µs to 10.5 µs | 224 ns | 10.4 µs | 10.3 µs to 10.5 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 1 | 11.2 µs | 11.1 µs to 11.4 µs | 175 ns | 11.2 µs | 11.1 µs to 11.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 1 | 11.4 µs | 11.3 µs to 11.5 µs | 747 ns | 11.4 µs | 11.3 µs to 11.5 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 1 | 11.6 µs | 11.5 µs to 12.1 µs | 369 ns | 11.6 µs | 11.5 µs to 12.1 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 1 | 12.2 µs | 12.1 µs to 12.4 µs | 383 ns | 12.2 µs | 12.1 µs to 12.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 1 | 12.2 µs | 12.1 µs to 12.3 µs | 414 ns | 12.2 µs | 12.1 µs to 12.3 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 1 | 13.3 µs | 13.2 µs to 13.4 µs | 402 ns | 13.3 µs | 13.2 µs to 13.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 1 | 13.4 µs | 13.3 µs to 13.6 µs | 442 ns | 13.4 µs | 13.3 µs to 13.6 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 1 | 14.5 µs | 14.4 µs to 14.8 µs | 739 ns | 14.5 µs | 14.4 µs to 14.8 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1 | 37.2 µs | 36.3 µs to 38.6 µs | 2.4 µs | 37.2 µs | 36.3 µs to 38.6 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1 | 39.4 µs | 39 µs to 40.1 µs | 2.53 µs | 39.4 µs | 39 µs to 40.1 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1 | 57.6 µs | 57.4 µs to 59 µs | 1.95 µs | 57.6 µs | 57.4 µs to 59 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1 | 60.9 µs | 60.3 µs to 62 µs | 2.12 µs | 60.9 µs | 60.3 µs to 62 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1 | 143 µs | 141 µs to 145 µs | 7.09 µs | 143 µs | 141 µs to 145 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 100 | 9.15 ms | 9.12 ms to 9.23 ms | 404 µs | 91.5 µs | 91.2 µs to 92.3 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 100 | 9.17 ms | 9.11 ms to 9.61 ms | 1.49 ms | 91.7 µs | 91.1 µs to 96.1 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 100 | 9.24 ms | 9.22 ms to 9.29 ms | 485 µs | 92.4 µs | 92.2 µs to 92.9 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 100 | 9.41 ms | 9.39 ms to 9.89 ms | 1.66 ms | 94.1 µs | 93.9 µs to 98.9 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 100 | 9.62 ms | 9.6 ms to 10.3 ms | 590 µs | 96.2 µs | 96 µs to 103 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 100 | 11.4 ms | 11.1 ms to 12.4 ms | 1.56 ms | 114 µs | 111 µs to 124 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 100 | 11.4 ms | 11.2 ms to 13.3 ms | 3.65 ms | 114 µs | 112 µs to 133 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 100 | 11.7 ms | 11.2 ms to 15.2 ms | 3.37 ms | 117 µs | 112 µs to 152 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 100 | 11.7 ms | 11.3 ms to 12.2 ms | 1.73 ms | 117 µs | 113 µs to 122 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 100 | 12.6 ms | 11.8 ms to 13.2 ms | 1.43 ms | 126 µs | 118 µs to 132 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 100 | 25.1 ms | 23.7 ms to 30.8 ms | 4.36 ms | 251 µs | 237 µs to 308 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 100 | 27.1 ms | 24.8 ms to 32.2 ms | 5.04 ms | 271 µs | 248 µs to 322 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 100 | 37.8 ms | 35.4 ms to 48.3 ms | 8.43 ms | 378 µs | 354 µs to 483 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 100 | 38.8 ms | 38.5 ms to 40.3 ms | 3.82 ms | 388 µs | 385 µs to 403 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 100 | 40.6 ms | 38.9 ms to 49 ms | 6.73 ms | 406 µs | 389 µs to 490 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 1000 | 99.8 ms | 99.4 ms to 101 ms | 6.04 ms | 99.8 µs | 99.4 µs to 101 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 1000 | 99.9 ms | 99.3 ms to 114 ms | 11.6 ms | 99.9 µs | 99.3 µs to 114 µs | 20 | 142 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 1000 | 101 ms | 101 ms to 101 ms | 14.6 ms | 101 µs | 101 µs to 101 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 1000 | 102 ms | 102 ms to 110 ms | 9.79 ms | 102 µs | 102 µs to 110 µs | 20 | 142 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 1000 | 105 ms | 104 ms to 106 ms | 10.6 ms | 105 µs | 104 µs to 106 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 1000 | 129 ms | 128 ms to 133 ms | 14.2 ms | 129 µs | 128 µs to 133 µs | 20 | 266 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 1000 | 132 ms | 132 ms to 133 ms | 2.34 ms | 132 µs | 132 µs to 133 µs | 20 | 272 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 1000 | 134 ms | 130 ms to 198 ms | 40.5 ms | 134 µs | 130 µs to 198 µs | 20 | 272 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 1000 | 138 ms | 135 ms to 141 ms | 13 ms | 138 µs | 135 µs to 141 µs | 20 | 266 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 1000 | 164 ms | 129 ms to 222 ms | 55 ms | 164 µs | 129 µs to 222 µs | 20 | 266 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1000 | 263 ms | 259 ms to 311 ms | 48.6 ms | 263 µs | 259 µs to 311 µs | 20 | 268 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1000 | 265 ms | 258 ms to 301 ms | 52.5 ms | 265 µs | 258 µs to 301 µs | 20 | 268 | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1000 | 365 ms | 361 ms to 458 ms | 57.7 ms | 365 µs | 361 µs to 458 µs | 20 | 268 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1000 | 414 ms | 412 ms to 442 ms | 39.7 ms | 414 µs | 412 µs to 442 µs | 20 | 137 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1000 | 420 ms | 409 ms to 443 ms | 52.9 ms | 420 µs | 409 µs to 443 µs | 20 | 137 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 10000 | 1.05 s | 1.04 s to 1.06 s | 156 ms | 105 µs | 104 µs to 106 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 10000 | 1.06 s | 1.05 s to 1.07 s | 8.71 ms | 106 µs | 105 µs to 107 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 10000 | 1.07 s | 1.04 s to 1.09 s | 39.4 ms | 107 µs | 104 µs to 109 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 10000 | 1.08 s | 1.08 s to 1.1 s | 21.8 ms | 108 µs | 108 µs to 110 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 10000 | 1.12 s | 1.09 s to 1.13 s | 33.3 ms | 112 µs | 109 µs to 113 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 10000 | 1.44 s | 1.41 s to 1.49 s | 54.1 ms | 144 µs | 141 µs to 149 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 10000 | 1.48 s | 1.43 s to 1.51 s | 45 ms | 148 µs | 143 µs to 151 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 10000 | 1.48 s | 1.45 s to 1.54 s | 93.5 ms | 148 µs | 145 µs to 154 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 10000 | 1.49 s | 1.45 s to 1.5 s | 40.8 ms | 149 µs | 145 µs to 150 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 10000 | 1.54 s | 1.52 s to 1.56 s | 50.1 ms | 154 µs | 152 µs to 156 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 10000 | 2.9 s | 2.84 s to 3.01 s | 98.1 ms | 290 µs | 284 µs to 301 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 10000 | 2.97 s | 2.91 s to 3 s | 72.7 ms | 297 µs | 291 µs to 300 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 10000 | 4.02 s | 3.98 s to 4.05 s | 71.8 ms | 402 µs | 398 µs to 405 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 10000 | 4.51 s | 4.45 s to 4.56 s | 228 ms | 451 µs | 445 µs to 456 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 10000 | 4.55 s | 4.5 s to 4.59 s | 71.9 ms | 455 µs | 450 µs to 459 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 1 | 24.3 µs | 24.2 µs to 25.1 µs | 868 ns | 24.3 µs | 24.2 µs to 25.1 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 1 | 24.8 µs | 24.6 µs to 25.9 µs | 1.81 µs | 24.8 µs | 24.6 µs to 25.9 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 1 | 25.5 µs | 25.4 µs to 26.6 µs | 1.94 µs | 25.5 µs | 25.4 µs to 26.6 µs | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1 | 1.17 ms | 1.15 ms to 1.27 ms | 110 µs | 1.17 ms | 1.15 ms to 1.27 ms | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 100 | 9.92 ms | 9.84 ms to 10.1 ms | 382 µs | 99.2 µs | 98.4 µs to 101 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 100 | 9.92 ms | 9.85 ms to 10.2 ms | 370 µs | 99.2 µs | 98.5 µs to 102 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 100 | 11.4 ms | 11.1 ms to 12 ms | 695 µs | 114 µs | 111 µs to 120 µs | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 100 | 688 ms | 681 ms to 698 ms | 15.7 ms | 6.88 ms | 6.81 ms to 6.98 ms | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 1000 | 141 ms | 139 ms to 147 ms | 15.3 ms | 141 µs | 139 µs to 147 µs | 20 | 148 | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 1000 | 141 ms | 138 ms to 147 ms | 13.9 ms | 141 µs | 138 µs to 147 µs | 20 | 148 | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 1000 | 155 ms | 151 ms to 163 ms | 17.3 ms | 155 µs | 151 µs to 163 µs | 20 | 148 | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1000 | 7.29 s | 7.27 s to 7.38 s | 92.5 ms | 7.29 ms | 7.27 ms to 7.38 ms | 20 | 199 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 1 | 11.8 µs | 11.8 µs to 12.1 µs | 284 ns | 11.8 µs | 11.8 µs to 12.1 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 1 | 12 µs | 11.9 µs to 12.2 µs | 435 ns | 12 µs | 11.9 µs to 12.2 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 1 | 12.9 µs | 12.8 µs to 13.1 µs | 455 ns | 12.9 µs | 12.8 µs to 13.1 µs | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1 | 982 µs | 976 µs to 1.01 ms | 71 µs | 982 µs | 976 µs to 1.01 ms | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 100 | 4.23 ms | 4.21 ms to 4.35 ms | 341 µs | 42.3 µs | 42.1 µs to 43.5 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 100 | 4.23 ms | 4.2 ms to 4.32 ms | 172 µs | 42.3 µs | 42 µs to 43.2 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 100 | 5.05 ms | 5.01 ms to 5.29 ms | 249 µs | 50.5 µs | 50.1 µs to 52.9 µs | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 100 | 538 ms | 528 ms to 550 ms | 14.9 ms | 5.38 ms | 5.28 ms to 5.5 ms | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 1000 | 76.5 ms | 75.8 ms to 80.7 ms | 9.57 ms | 76.5 µs | 75.8 µs to 80.7 µs | 20 | 146 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 1000 | 77.6 ms | 75.4 ms to 79.9 ms | 6.67 ms | 77.6 µs | 75.4 µs to 79.9 µs | 20 | 146 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 1000 | 87 ms | 85.5 ms to 92.4 ms | 20.1 ms | 87 µs | 85.5 µs to 92.4 µs | 20 | 146 | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1000 | 5.74 s | 5.72 s to 5.76 s | 46.6 ms | 5.74 ms | 5.72 ms to 5.76 ms | 20 | 196 | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1 | 8.01 µs | 7.96 µs to 8.17 µs | 257 ns | 8.01 µs | 7.96 µs to 8.17 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 1 | 8.92 µs | 8.86 µs to 9.33 µs | 416 ns | 8.92 µs | 8.86 µs to 9.33 µs | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1 | 468 µs | 460 µs to 496 µs | 36 µs | 468 µs | 460 µs to 496 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 100 | 3.72 ms | 3.69 ms to 3.88 ms | 321 µs | 37.2 µs | 36.9 µs to 38.8 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 100 | 3.74 ms | 3.7 ms to 3.83 ms | 226 µs | 37.4 µs | 37 µs to 38.3 µs | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 100 | 444 ms | 431 ms to 457 ms | 15.8 ms | 4.44 ms | 4.31 ms to 4.57 ms | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1000 | 47 ms | 46.8 ms to 52.7 ms | 9.69 ms | 47 µs | 46.8 µs to 52.7 µs | 20 | 150 | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 1000 | 47.2 ms | 46.9 ms to 49.4 ms | 14.2 ms | 47.2 µs | 46.9 µs to 49.4 µs | 20 | 150 | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1000 | 5.13 s | 5.1 s to 5.2 s | 78.5 ms | 5.13 ms | 5.1 ms to 5.2 ms | 20 | 185 | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1 | 11.4 µs | 11.1 µs to 13.1 µs | 1.22 µs | 11.4 µs | 11.1 µs to 13.1 µs | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 1 | 12.8 µs | 12 µs to 13.4 µs | 940 ns | 12.8 µs | 12 µs to 13.4 µs | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1 | 1.08 ms | 1.07 ms to 1.12 ms | 107 µs | 1.08 ms | 1.07 ms to 1.12 ms | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 100 | 5.03 ms | 4.99 ms to 5.06 ms | 154 µs | 50.3 µs | 49.9 µs to 50.6 µs | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 100 | 5.09 ms | 5.05 ms to 5.15 ms | 587 µs | 50.9 µs | 50.5 µs to 51.5 µs | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 100 | 640 ms | 614 ms to 694 ms | 42.8 ms | 6.4 ms | 6.14 ms to 6.94 ms | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1000 | 84.3 ms | 83.5 ms to 89 ms | 5.46 ms | 84.3 µs | 83.5 µs to 89 µs | 20 | 146 | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 1000 | 95.7 ms | 86.9 ms to 117 ms | 15.8 ms | 95.7 µs | 86.9 µs to 117 µs | 20 | 146 | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1000 | 6.63 s | 6.6 s to 6.76 s | 109 ms | 6.63 ms | 6.6 ms to 6.76 ms | 20 | 187 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 1 | 7.5 µs | 7.47 µs to 7.58 µs | 228 ns | 7.5 µs | 7.47 µs to 7.58 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 1 | 7.81 µs | 7.73 µs to 8 µs | 557 ns | 7.81 µs | 7.73 µs to 8 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 1 | 8.63 µs | 8.6 µs to 8.84 µs | 425 ns | 8.63 µs | 8.6 µs to 8.84 µs | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1 | 927 µs | 923 µs to 937 µs | 63.1 µs | 927 µs | 923 µs to 937 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 100 | 3.24 ms | 3.2 ms to 3.35 ms | 327 µs | 32.4 µs | 32 µs to 33.5 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 100 | 3.29 ms | 3.2 ms to 3.41 ms | 459 µs | 32.9 µs | 32 µs to 34.1 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 100 | 3.66 ms | 3.61 ms to 3.81 ms | 270 µs | 36.6 µs | 36.1 µs to 38.1 µs | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 100 | 505 ms | 490 ms to 519 ms | 15.8 ms | 5.05 ms | 4.9 ms to 5.19 ms | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 1000 | 65.5 ms | 64.3 ms to 73.8 ms | 9.01 ms | 65.5 µs | 64.3 µs to 73.8 µs | 20 | 146 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 1000 | 66.5 ms | 64.5 ms to 68.8 ms | 10.7 ms | 66.5 µs | 64.5 µs to 68.8 µs | 20 | 146 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 1000 | 71.2 ms | 69.8 ms to 75.8 ms | 9.93 ms | 71.2 µs | 69.8 µs to 75.8 µs | 20 | 146 | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1000 | 5.41 s | 5.39 s to 5.44 s | 47.5 ms | 5.41 ms | 5.39 ms to 5.44 ms | 20 | 197 | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1 | 12.7 µs | 12.6 µs to 12.9 µs | 380 ns | 12.7 µs | 12.6 µs to 12.9 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 1 | 22.1 µs | 21.9 µs to 22.6 µs | 1.24 µs | 22.1 µs | 21.9 µs to 22.6 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 1 | 23.2 µs | 22.7 µs to 24.1 µs | 759 ns | 23.2 µs | 22.7 µs to 24.1 µs | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1 | 23.3 µs | 22.8 µs to 24.6 µs | 994 ns | 23.3 µs | 22.8 µs to 24.6 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 1 | 23.6 µs | 22.9 µs to 24.8 µs | 1.19 µs | 23.6 µs | 22.9 µs to 24.8 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 1 | 25.8 µs | 25.7 µs to 26.1 µs | 990 ns | 25.8 µs | 25.7 µs to 26.1 µs | 20 |  | 0 diff |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1 | 212 µs | 208 µs to 214 µs | 4.07 µs | 212 µs | 208 µs to 214 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1 | 872 µs | 866 µs to 918 µs | 46.6 µs | 872 µs | 866 µs to 918 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 100 | 7.75 ms | 7.73 ms to 7.91 ms | 494 µs | 77.5 µs | 77.3 µs to 79.1 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 100 | 7.9 ms | 7.84 ms to 8.44 ms | 694 µs | 79 µs | 78.4 µs to 84.4 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 100 | 8.3 ms | 8.14 ms to 8.67 ms | 550 µs | 83 µs | 81.4 µs to 86.7 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 100 | 9.13 ms | 9.1 ms to 9.37 ms | 1.2 ms | 91.3 µs | 91 µs to 93.7 µs | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 100 | 13.4 ms | 13.1 ms to 13.5 ms | 364 µs | 134 µs | 131 µs to 135 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 100 | 14.1 ms | 13.6 ms to 15.6 ms | 1.92 ms | 141 µs | 136 µs to 156 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 100 | 42.8 ms | 42.2 ms to 47.3 ms | 4.14 ms | 428 µs | 422 µs to 473 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 100 | 540 ms | 527 ms to 561 ms | 19 ms | 5.4 ms | 5.27 ms to 5.61 ms | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 1000 | 88 ms | 87.8 ms to 92.7 ms | 6.89 ms | 88 µs | 87.8 µs to 92.7 µs | 20 | 143 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 1000 | 91 ms | 89.5 ms to 96.6 ms | 8.17 ms | 91 µs | 89.5 µs to 96.6 µs | 20 | 147 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 1000 | 93.3 ms | 92.5 ms to 94.9 ms | 6.82 ms | 93.3 µs | 92.5 µs to 94.9 µs | 20 | 147 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 1000 | 104 ms | 103 ms to 113 ms | 17.1 ms | 104 µs | 103 µs to 113 µs | 20 | 146 | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1000 | 149 ms | 147 ms to 153 ms | 7.63 ms | 149 µs | 147 µs to 153 µs | 20 | 136 | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1000 | 159 ms | 157 ms to 160 ms | 5.27 ms | 159 µs | 157 µs to 160 µs | 20 | 177 | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1000 | 476 ms | 456 ms to 491 ms | 18.2 ms | 476 µs | 456 µs to 491 µs | 20 | 219 | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1000 | 5.82 s | 5.79 s to 5.87 s | 100 ms | 5.82 ms | 5.79 ms to 5.87 ms | 20 | 224 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 1 | 302 ns | 295 ns to 309 ns | 17.9 ns | 302 ns | 295 ns to 309 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 1 | 356 ns | 349 ns to 362 ns | 20.3 ns | 356 ns | 349 ns to 362 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 1 | 1.48 µs | 1.47 µs to 1.5 µs | 72.8 ns | 1.48 µs | 1.47 µs to 1.5 µs | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 1 | 22.6 µs | 22.6 µs to 22.7 µs | 557 ns | 22.6 µs | 22.6 µs to 22.7 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 100 | 58.6 µs | 58.3 µs to 59 µs | 2.14 µs | 586 ns | 583 ns to 590 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 100 | 58.9 µs | 58.5 µs to 59.1 µs | 3.27 µs | 589 ns | 585 ns to 591 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 100 | 62.6 µs | 62.2 µs to 63.5 µs | 4.05 µs | 626 ns | 622 ns to 635 ns | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 100 | 10.2 ms | 10.1 ms to 10.2 ms | 324 µs | 102 µs | 101 µs to 102 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 1000 | 689 µs | 680 µs to 849 µs | 125 µs | 689 ns | 680 ns to 849 ns | 20 | 149 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 1000 | 721 µs | 702 µs to 845 µs | 153 µs | 721 ns | 702 ns to 845 ns | 20 | 138 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 1000 | 825 µs | 820 µs to 915 µs | 191 µs | 825 ns | 820 ns to 915 ns | 20 | 141 | 0 diff |
| translit | anyascii 0.3.3 | 1000 | 120 ms | 120 ms to 121 ms | 1.97 ms | 120 µs | 120 µs to 121 µs | 20 | 135 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 10000 | 10.9 ms | 10.6 ms to 11.7 ms | 994 µs | 1.09 µs | 1.06 µs to 1.17 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 10000 | 12.9 ms | 12.7 ms to 13.8 ms | 1.89 ms | 1.29 µs | 1.27 µs to 1.38 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 10000 | 13.6 ms | 12.8 ms to 14.5 ms | 1.72 ms | 1.36 µs | 1.28 µs to 1.45 µs | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 10000 | 1.28 s | 1.26 s to 1.29 s | 15.9 ms | 128 µs | 126 µs to 129 µs | 20 |  | 0 diff |
