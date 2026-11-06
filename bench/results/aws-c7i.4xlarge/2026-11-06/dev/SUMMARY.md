## Benchmark results: aws-c7i.4xlarge, 2026-11-06

*AWS EC2 c7i.4xlarge (a VM on real hardware, not bare metal).* CPU: Intel(R) Xeon(R) Platinum 8488C, 16 logical CPUs (affinity 16); kernel 6.18.56-125.164.amzn2023.x86_64; Python 3.13.18 (GIL on); git f35bd9e4cbc0 (git); manifest sha256 c69d5b111fdb.

Load average (1/5/15 min) before: 1.07 / 2.66 / 2.43; after: 7.79 / 5.57 / 3.32. Other sessions share this VM.

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
| datatrove-0.10.1-w1 | 212.76 | 211.52 to 226.26 | 220.93 ± 16.36 | 213.5 | 1.00 | 442 |  | 442 | 21.28 | 3 |
| pyroid-w1 | 8.52 | 8.50 to 8.53 | 8.51 ± 0.03 | 9.3 | 1.09 | 420 |  | 420 | 0.85 | 3 |
| datatrove-0.10.1-w8 | 35.37 | 35.31 to 35.45 | 35.38 ± 0.14 | 263.8 | 7.46 | 376 |  | 3069 | 3.54 | 3 |
| pyroid-w8 | 3.15 | 3.15 to 3.17 | 3.16 ± 0.02 | 11.3 | 3.58 | 364 |  | 2991 | 0.32 | 3 |

### End to end: hyperfine-stages-w8.json

| command | median wall (s) | IQR (s) | mean ± sd (s) | CPU-s (median) | CPU-s / wall | largest process RSS (MiB) | cgroup anon peak, sampled (MiB) | sum of worker maxrss (MiB, bound) | wall ms / doc | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text-w8 | 1.83 | 1.82 to 1.84 | 1.83 ± 0.01 | 11.0 | 6.03 | 183 |  | 1551 | 0.18 | 3 |
| lid-w8 | 1.40 | 1.39 to 1.41 | 1.40 ± 0.02 | 7.4 | 5.30 | 320 |  | 2651 | 0.14 | 3 |
| gopher_repetition-w8 | 10.65 | 10.64 to 10.66 | 10.65 ± 0.02 | 78.7 | 7.39 | 235 |  | 1935 | 1.06 | 3 |
| gopher_quality-w8 | 8.50 | 8.49 to 8.53 | 8.51 ± 0.04 | 62.5 | 7.35 | 236 |  | 1937 | 0.85 | 3 |
| c4_quality-w8 | 10.75 | 10.74 to 10.76 | 10.75 ± 0.02 | 79.5 | 7.39 | 230 |  | 1916 | 1.08 | 3 |
| fineweb_quality-w8 | 8.01 | 7.97 to 8.04 | 8.01 ± 0.08 | 59.0 | 7.37 | 235 |  | 1928 | 0.80 | 3 |
| minhash-w8 | 9.38 | 9.37 to 9.45 | 9.42 ± 0.09 | 68.4 | 7.29 | 275 |  | 2183 | 0.94 | 3 |

hyperfine `memory_peak_resident` is the largest single process, not the simultaneous total; the cgroup column is the container's anonymous memory sampled every 0.1 s during the run (it can miss short spikes; the median includes the warmup run), and the last memory column sums per-worker peaks (an upper bound).

### Per stage (pyperf; one value = one call over the batch)

| stage | implementation | docs / call | median / call | IQR / call | stdev | per doc (median) | per doc (IQR) | values | peak RSS MiB (track-memory) | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 1 | 643 ns | 642 ns to 644 ns | 1.76 ns | 643 ns | 642 ns to 644 ns | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 1 | 839 ns | 834 ns to 840 ns | 4.39 ns | 839 ns | 834 ns to 840 ns | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 1 | 1.85 µs | 1.84 µs to 1.87 µs | 22.2 ns | 1.85 µs | 1.84 µs to 1.87 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 1 | 88.5 µs | 88.2 µs to 89.1 µs | 965 ns | 88.5 µs | 88.2 µs to 89.1 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 100 | 372 µs | 371 µs to 372 µs | 833 ns | 3.72 µs | 3.71 µs to 3.72 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 100 | 541 µs | 501 µs to 558 µs | 55.9 µs | 5.41 µs | 5.01 µs to 5.58 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 100 | 556 µs | 504 µs to 568 µs | 47.1 µs | 5.56 µs | 5.04 µs to 5.68 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 100 | 57.4 ms | 57.2 ms to 57.6 ms | 303 µs | 574 µs | 572 µs to 576 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 1000 | 4.46 ms | 4.46 ms to 4.47 ms | 6.99 µs | 4.46 µs | 4.46 µs to 4.47 µs | 20 | 155 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 1000 | 4.49 ms | 4.48 ms to 4.5 ms | 15 µs | 4.49 µs | 4.48 µs to 4.5 µs | 20 | 148 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 1000 | 4.55 ms | 4.54 ms to 4.57 ms | 218 µs | 4.55 µs | 4.54 µs to 4.57 µs | 20 | 141 | 0 diff |
| fix_text | ftfy 6.3.1 | 1000 | 656 ms | 653 ms to 659 ms | 3.9 ms | 656 µs | 653 µs to 659 µs | 20 | 155 | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch | 10000 | 53.2 ms | 53.1 ms to 53.3 ms | 256 µs | 5.32 µs | 5.31 µs to 5.33 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text_batch [arrow] | 10000 | 54.2 ms | 53.8 ms to 54.4 ms | 607 µs | 5.42 µs | 5.38 µs to 5.44 µs | 20 |  | 0 diff |
| fix_text | pyroid.ftfy 1.0.0.dev0 fix_text per doc | 10000 | 56.4 ms | 56.1 ms to 56.7 ms | 376 µs | 5.64 µs | 5.61 µs to 5.67 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 10000 | 7.24 s | 7.23 s to 7.26 s | 30.8 ms | 724 µs | 723 µs to 726 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 1 | 8.49 µs | 8.47 µs to 8.51 µs | 61.8 ns | 8.49 µs | 8.47 µs to 8.51 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 1 | 8.63 µs | 8.6 µs to 8.66 µs | 59.8 ns | 8.63 µs | 8.6 µs to 8.66 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 1 | 9.34 µs | 9.32 µs to 9.37 µs | 48.6 ns | 9.34 µs | 9.32 µs to 9.37 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 1 | 9.41 µs | 9.41 µs to 9.45 µs | 29.6 ns | 9.41 µs | 9.41 µs to 9.45 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 1 | 9.52 µs | 9.5 µs to 9.54 µs | 30.9 ns | 9.52 µs | 9.5 µs to 9.54 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 1 | 10.4 µs | 10.4 µs to 10.4 µs | 28.5 ns | 10.4 µs | 10.4 µs to 10.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 1 | 10.6 µs | 10.5 µs to 10.6 µs | 39.4 ns | 10.6 µs | 10.5 µs to 10.6 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 1 | 11.5 µs | 11.5 µs to 11.5 µs | 29.5 ns | 11.5 µs | 11.5 µs to 11.5 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 1 | 12.3 µs | 12.2 µs to 12.4 µs | 228 ns | 12.3 µs | 12.2 µs to 12.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 1 | 12.8 µs | 12.8 µs to 12.8 µs | 49 ns | 12.8 µs | 12.8 µs to 12.8 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1 | 27.3 µs | 27.2 µs to 27.5 µs | 560 ns | 27.3 µs | 27.2 µs to 27.5 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1 | 30 µs | 29.9 µs to 30.1 µs | 131 ns | 30 µs | 29.9 µs to 30.1 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1 | 44.7 µs | 44.6 µs to 44.9 µs | 222 ns | 44.7 µs | 44.6 µs to 44.9 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1 | 47.4 µs | 47.1 µs to 47.5 µs | 277 ns | 47.4 µs | 47.1 µs to 47.5 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1 | 120 µs | 120 µs to 121 µs | 1.9 µs | 120 µs | 120 µs to 121 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 100 | 7.9 ms | 7.83 ms to 8 ms | 133 µs | 79 µs | 78.3 µs to 80 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 100 | 7.97 ms | 7.85 ms to 8.01 ms | 111 µs | 79.7 µs | 78.5 µs to 80.1 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 100 | 8.05 ms | 7.95 ms to 8.12 ms | 99.2 µs | 80.5 µs | 79.5 µs to 81.2 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 100 | 8.3 ms | 8.22 ms to 8.37 ms | 105 µs | 83 µs | 82.2 µs to 83.7 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 100 | 8.51 ms | 8.46 ms to 8.55 ms | 114 µs | 85.1 µs | 84.6 µs to 85.5 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 100 | 8.54 ms | 8.54 ms to 8.55 ms | 9.92 µs | 85.4 µs | 85.4 µs to 85.5 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 100 | 8.66 ms | 8.65 ms to 8.68 ms | 16.5 µs | 86.6 µs | 86.5 µs to 86.8 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 100 | 8.79 ms | 8.72 ms to 8.81 ms | 68.3 µs | 87.9 µs | 87.2 µs to 88.1 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 100 | 9.04 ms | 9.03 ms to 9.04 ms | 10.6 µs | 90.4 µs | 90.3 µs to 90.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 100 | 9.1 ms | 9.03 ms to 9.12 ms | 61.9 µs | 91 µs | 90.3 µs to 91.2 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 100 | 16.6 ms | 16.4 ms to 16.7 ms | 201 µs | 166 µs | 164 µs to 167 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 100 | 16.9 ms | 16.7 ms to 17.1 ms | 203 µs | 169 µs | 167 µs to 171 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 100 | 26 ms | 25.8 ms to 26.1 ms | 226 µs | 260 µs | 258 µs to 261 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 100 | 34 ms | 33.9 ms to 34.2 ms | 220 µs | 340 µs | 339 µs to 342 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 100 | 34.5 ms | 34.3 ms to 34.7 ms | 248 µs | 345 µs | 343 µs to 347 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 1000 | 87.2 ms | 86.3 ms to 87.9 ms | 1.09 ms | 87.2 µs | 86.3 µs to 87.9 µs | 20 | 273 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 1000 | 88.1 ms | 87.4 ms to 88.5 ms | 1.3 ms | 88.1 µs | 87.4 µs to 88.5 µs | 20 | 267 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 1000 | 90.4 ms | 89.7 ms to 92 ms | 1.55 ms | 90.4 µs | 89.7 µs to 92 µs | 20 | 267 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 1000 | 91 ms | 90.2 ms to 91.8 ms | 1.41 ms | 91 µs | 90.2 µs to 91.8 µs | 20 | 274 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 1000 | 91.4 ms | 91.3 ms to 91.6 ms | 205 µs | 91.4 µs | 91.3 µs to 91.6 µs | 20 | 143 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 1000 | 92 ms | 91.9 ms to 92 ms | 128 µs | 92 µs | 91.9 µs to 92 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 1000 | 93 ms | 92.9 ms to 93.1 ms | 135 µs | 93 µs | 92.9 µs to 93.1 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 1000 | 94.6 ms | 93 ms to 95.7 ms | 1.64 ms | 94.6 µs | 93 µs to 95.7 µs | 20 | 268 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 1000 | 94.7 ms | 94.6 ms to 94.7 ms | 131 µs | 94.7 µs | 94.6 µs to 94.7 µs | 20 | 143 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 1000 | 97.7 ms | 97 ms to 98.1 ms | 613 µs | 97.7 µs | 97 µs to 98.1 µs | 20 | 137 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1000 | 181 ms | 179 ms to 183 ms | 2.69 ms | 181 µs | 179 µs to 183 µs | 20 | 267 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1000 | 185 ms | 183 ms to 186 ms | 13.4 ms | 185 µs | 183 µs to 186 µs | 20 | 280 | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1000 | 274 ms | 272 ms to 275 ms | 3.33 ms | 274 µs | 272 µs to 275 µs | 20 | 320 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1000 | 361 ms | 360 ms to 364 ms | 2.45 ms | 361 µs | 360 µs to 364 µs | 20 | 147 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1000 | 366 ms | 365 ms to 368 ms | 1.64 ms | 366 µs | 365 µs to 368 µs | 20 | 136 | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 batch | 10000 | 892 ms | 889 ms to 897 ms | 7.45 ms | 89.2 µs | 88.9 µs to 89.7 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.bin k=1 per doc | 10000 | 922 ms | 920 ms to 926 ms | 7.46 ms | 92.2 µs | 92 µs to 92.6 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 batch | 10000 | 926 ms | 918 ms to 940 ms | 11.3 ms | 92.6 µs | 91.8 µs to 94 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 batch | 10000 | 933 ms | 932 ms to 934 ms | 1.19 ms | 93.3 µs | 93.2 µs to 93.4 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=1 | 10000 | 937 ms | 934 ms to 941 ms | 5.4 ms | 93.7 µs | 93.4 µs to 94.1 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=1 | 10000 | 948 ms | 947 ms to 948 ms | 1.48 ms | 94.8 µs | 94.7 µs to 94.8 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 predict lid.176.ftz k=1 per doc | 10000 | 959 ms | 940 ms to 960 ms | 28.4 ms | 95.9 µs | 94 µs to 96 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 batch | 10000 | 965 ms | 964 ms to 965 ms | 933 µs | 96.5 µs | 96.4 µs to 96.5 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.bin k=5 | 10000 | 988 ms | 977 ms to 995 ms | 15.9 ms | 98.8 µs | 97.7 µs to 99.5 µs | 20 |  | 0 diff |
| lid | pyroid.fasttext 1.0.0.dev0 lid.176.ftz k=5 | 10000 | 1 s | 993 ms to 1.01 s | 9.53 ms | 100 µs | 99.3 µs to 101 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 10000 | 1.85 s | 1.84 s to 1.86 s | 12.8 ms | 185 µs | 184 µs to 186 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 10000 | 1.94 s | 1.94 s to 1.95 s | 15.3 ms | 194 µs | 194 µs to 195 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 10000 | 2.82 s | 2.79 s to 2.83 s | 29.5 ms | 282 µs | 279 µs to 283 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 10000 | 3.73 s | 3.72 s to 3.73 s | 15 ms | 373 µs | 372 µs to 373 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 10000 | 3.79 s | 3.78 s to 3.79 s | 7.33 ms | 379 µs | 378 µs to 379 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 1 | 26.3 µs | 26.2 µs to 26.9 µs | 809 ns | 26.3 µs | 26.2 µs to 26.9 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 1 | 27.4 µs | 27 µs to 27.7 µs | 897 ns | 27.4 µs | 27 µs to 27.7 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 1 | 28.5 µs | 28.3 µs to 28.7 µs | 652 ns | 28.5 µs | 28.3 µs to 28.7 µs | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1 | 1.29 ms | 1.27 ms to 1.3 ms | 21.2 µs | 1.29 ms | 1.27 ms to 1.3 ms | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 100 | 11.1 ms | 11.1 ms to 11.4 ms | 178 µs | 111 µs | 111 µs to 114 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 100 | 11.3 ms | 11.2 ms to 11.3 ms | 109 µs | 113 µs | 112 µs to 113 µs | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 100 | 12.2 ms | 12.2 ms to 12.3 ms | 150 µs | 122 µs | 122 µs to 123 µs | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 100 | 635 ms | 631 ms to 641 ms | 7.01 ms | 6.35 ms | 6.31 ms to 6.41 ms | 20 |  | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 apply([GopherRepetition]) direct | 1000 | 142 ms | 141 ms to 142 ms | 686 µs | 142 µs | 141 µs to 142 µs | 20 | 149 | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition | 1000 | 142 ms | 141 ms to 143 ms | 819 µs | 142 µs | 141 µs to 143 µs | 20 | 150 | 0 diff |
| gopher_repetition | pyroid.filters 1.0.0.dev0 gopher_repetition_reasons per doc | 1000 | 155 ms | 155 ms to 156 ms | 1.04 ms | 155 µs | 155 µs to 156 µs | 20 | 154 | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1000 | 6.1 s | 6.06 s to 6.13 s | 168 ms | 6.1 ms | 6.06 ms to 6.13 ms | 20 | 198 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 1 | 11.2 µs | 11.2 µs to 11.3 µs | 126 ns | 11.2 µs | 11.2 µs to 11.3 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 1 | 11.9 µs | 11.9 µs to 12 µs | 120 ns | 11.9 µs | 11.9 µs to 12 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 1 | 12.9 µs | 12.8 µs to 13 µs | 180 ns | 12.9 µs | 12.8 µs to 13 µs | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1 | 882 µs | 878 µs to 894 µs | 10.2 µs | 882 µs | 878 µs to 894 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 100 | 4.5 ms | 4.5 ms to 4.54 ms | 36.2 µs | 45 µs | 45 µs to 45.4 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 100 | 4.5 ms | 4.48 ms to 4.55 ms | 66.2 µs | 45 µs | 44.8 µs to 45.5 µs | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 100 | 5.13 ms | 5.12 ms to 5.13 ms | 16 µs | 51.3 µs | 51.2 µs to 51.3 µs | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 100 | 438 ms | 436 ms to 439 ms | 2.93 ms | 4.38 ms | 4.36 ms to 4.39 ms | 20 |  | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality | 1000 | 82.7 ms | 82.5 ms to 83.3 ms | 492 µs | 82.7 µs | 82.5 µs to 83.3 µs | 20 | 148 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 apply([GopherQuality]) direct | 1000 | 82.9 ms | 82.5 ms to 83.2 ms | 489 µs | 82.9 µs | 82.5 µs to 83.2 µs | 20 | 148 | 0 diff |
| gopher_quality | pyroid.filters 1.0.0.dev0 gopher_quality_reasons per doc | 1000 | 91.2 ms | 90.9 ms to 91.6 ms | 479 µs | 91.2 µs | 90.9 µs to 91.6 µs | 20 | 147 | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1000 | 4.61 s | 4.6 s to 4.64 s | 25.9 ms | 4.61 ms | 4.6 ms to 4.64 ms | 20 | 197 | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1 | 8.25 µs | 8.18 µs to 8.34 µs | 142 ns | 8.25 µs | 8.18 µs to 8.34 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 1 | 9.35 µs | 9.24 µs to 9.54 µs | 157 ns | 9.35 µs | 9.24 µs to 9.54 µs | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1 | 427 µs | 416 µs to 436 µs | 10.3 µs | 427 µs | 416 µs to 436 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 100 | 4.25 ms | 4.24 ms to 4.32 ms | 82.2 µs | 42.5 µs | 42.4 µs to 43.2 µs | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 100 | 4.29 ms | 4.23 ms to 4.42 ms | 100 µs | 42.9 µs | 42.3 µs to 44.2 µs | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 100 | 393 ms | 391 ms to 394 ms | 2.9 ms | 3.93 ms | 3.91 ms to 3.94 ms | 20 |  | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1000 | 52.8 ms | 52.5 ms to 53.2 ms | 1.92 ms | 52.8 µs | 52.5 µs to 53.2 µs | 20 | 152 | 0 diff |
| c4_quality | pyroid.filters 1.0.0.dev0 c4_quality | 1000 | 52.8 ms | 52.2 ms to 53.4 ms | 720 µs | 52.8 µs | 52.2 µs to 53.4 µs | 20 | 152 | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1000 | 4.51 s | 4.48 s to 4.53 s | 55.3 ms | 4.51 ms | 4.48 ms to 4.53 ms | 20 | 185 | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1 | 11 µs | 11 µs to 11.2 µs | 276 ns | 11 µs | 11 µs to 11.2 µs | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 1 | 12.2 µs | 12.1 µs to 12.3 µs | 109 ns | 12.2 µs | 12.1 µs to 12.3 µs | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1 | 1.08 ms | 1.07 ms to 1.09 ms | 10.1 µs | 1.08 ms | 1.07 ms to 1.09 ms | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 100 | 5.46 ms | 5.45 ms to 5.46 ms | 33.3 µs | 54.6 µs | 54.5 µs to 54.6 µs | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 100 | 5.49 ms | 5.48 ms to 5.55 ms | 40.1 µs | 54.9 µs | 54.8 µs to 55.5 µs | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 100 | 580 ms | 577 ms to 589 ms | 8.55 ms | 5.8 ms | 5.77 ms to 5.89 ms | 20 |  | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 apply([C4Quality]) direct | 1000 | 94.5 ms | 94.3 ms to 94.7 ms | 375 µs | 94.5 µs | 94.3 µs to 94.7 µs | 20 | 147 | 0 diff |
| c4_quality@fineweb | pyroid.filters 1.0.0.dev0 c4_quality | 1000 | 94.8 ms | 94.4 ms to 95.3 ms | 700 µs | 94.8 µs | 94.4 µs to 95.3 µs | 20 | 147 | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1000 | 5.97 s | 5.87 s to 6.01 s | 72.2 ms | 5.97 ms | 5.87 ms to 6.01 ms | 20 | 199 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 1 | 7.57 µs | 7.51 µs to 7.65 µs | 97.2 ns | 7.57 µs | 7.51 µs to 7.65 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 1 | 8.22 µs | 8.19 µs to 8.27 µs | 90.8 ns | 8.22 µs | 8.19 µs to 8.27 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 1 | 9.16 µs | 9.08 µs to 9.3 µs | 146 ns | 9.16 µs | 9.08 µs to 9.3 µs | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1 | 832 µs | 820 µs to 834 µs | 11.6 µs | 832 µs | 820 µs to 834 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 100 | 3.65 ms | 3.64 ms to 3.68 ms | 43.7 µs | 36.5 µs | 36.4 µs to 36.8 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 100 | 3.67 ms | 3.66 ms to 3.7 ms | 34.8 µs | 36.7 µs | 36.6 µs to 37 µs | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 100 | 3.91 ms | 3.9 ms to 3.92 ms | 16.5 µs | 39.1 µs | 39 µs to 39.2 µs | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 100 | 403 ms | 399 ms to 407 ms | 5.14 ms | 4.03 ms | 3.99 ms to 4.07 ms | 20 |  | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 apply([FineWebQuality]) direct | 1000 | 72.2 ms | 71.8 ms to 72.3 ms | 392 µs | 72.2 µs | 71.8 µs to 72.3 µs | 20 | 148 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality | 1000 | 72.5 ms | 72.3 ms to 72.7 ms | 434 µs | 72.5 µs | 72.3 µs to 72.7 µs | 20 | 148 | 0 diff |
| fineweb_quality | pyroid.filters 1.0.0.dev0 fineweb_quality_reasons per doc | 1000 | 77.6 ms | 77.3 ms to 77.8 ms | 346 µs | 77.6 µs | 77.3 µs to 77.8 µs | 20 | 147 | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1000 | 4.29 s | 4.27 s to 4.3 s | 33.6 ms | 4.29 ms | 4.27 ms to 4.3 ms | 20 | 197 | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1 | 13.2 µs | 13 µs to 13.5 µs | 351 ns | 13.2 µs | 13 µs to 13.5 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 1 | 26.2 µs | 26.1 µs to 26.5 µs | 324 ns | 26.2 µs | 26.1 µs to 26.5 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 1 | 26.5 µs | 26.5 µs to 26.6 µs | 179 ns | 26.5 µs | 26.5 µs to 26.6 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 1 | 26.5 µs | 26.4 µs to 26.7 µs | 193 ns | 26.5 µs | 26.4 µs to 26.7 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 1 | 28.7 µs | 28.6 µs to 28.8 µs | 185 ns | 28.7 µs | 28.6 µs to 28.8 µs | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1 | 55.7 µs | 53.4 µs to 59.1 µs | 4.06 µs | 55.7 µs | 53.4 µs to 59.1 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1 | 158 µs | 158 µs to 159 µs | 1.24 µs | 158 µs | 158 µs to 159 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1 | 816 µs | 804 µs to 818 µs | 10.4 µs | 816 µs | 804 µs to 818 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 100 | 11.1 ms | 10.9 ms to 11.1 ms | 104 µs | 111 µs | 109 µs to 111 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 100 | 11.1 ms | 11.1 ms to 11.2 ms | 69.6 µs | 111 µs | 111 µs to 112 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 100 | 11.2 ms | 11.1 ms to 11.2 ms | 70.2 µs | 112 µs | 111 µs to 112 µs | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 100 | 12.1 ms | 12 ms to 12.1 ms | 20.6 µs | 121 µs | 120 µs to 121 µs | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 100 | 13 ms | 12.9 ms to 13.1 ms | 292 µs | 130 µs | 129 µs to 131 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 100 | 13.2 ms | 13.2 ms to 13.3 ms | 80.9 µs | 132 µs | 132 µs to 133 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 100 | 39.2 ms | 39.1 ms to 39.4 ms | 236 µs | 392 µs | 391 µs to 394 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 100 | 457 ms | 456 ms to 459 ms | 2.02 ms | 4.57 ms | 4.56 ms to 4.59 ms | 20 |  | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 MinHashDedup.fit direct | 1000 | 121 ms | 120 ms to 121 ms | 512 µs | 121 µs | 120 µs to 121 µs | 20 | 144 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures direct | 1000 | 123 ms | 122 ms to 124 ms | 890 µs | 123 µs | 122 µs to 124 µs | 20 | 146 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures | 1000 | 125 ms | 124 ms to 125 ms | 903 µs | 125 µs | 124 µs to 125 µs | 20 | 148 | 0 diff |
| minhash | pyroid.dedup 1.0.0.dev0 minhash_signatures per doc | 1000 | 137 ms | 136 ms to 138 ms | 1.06 ms | 137 µs | 136 µs to 138 µs | 20 | 147 | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1000 | 144 ms | 143 ms to 146 ms | 2.09 ms | 144 µs | 143 µs to 146 µs | 20 | 134 | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1000 | 161 ms | 159 ms to 163 ms | 2.42 ms | 161 µs | 159 µs to 163 µs | 20 | 141 | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1000 | 436 ms | 435 ms to 439 ms | 2.96 ms | 436 µs | 435 µs to 439 µs | 20 | 219 | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1000 | 4.94 s | 4.91 s to 4.97 s | 40.5 ms | 4.94 ms | 4.91 ms to 4.97 ms | 20 | 225 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 1 | 275 ns | 273 ns to 276 ns | 10.7 ns | 275 ns | 273 ns to 276 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 1 | 284 ns | 282 ns to 286 ns | 2.71 ns | 284 ns | 282 ns to 286 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 1 | 1.3 µs | 1.28 µs to 1.32 µs | 27 ns | 1.3 µs | 1.28 µs to 1.32 µs | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 1 | 15.6 µs | 15.6 µs to 15.7 µs | 111 ns | 15.6 µs | 15.6 µs to 15.7 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 100 | 55.6 µs | 55.1 µs to 55.8 µs | 598 ns | 556 ns | 551 ns to 558 ns | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 100 | 127 µs | 127 µs to 128 µs | 21.2 µs | 1.27 µs | 1.27 µs to 1.28 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 100 | 134 µs | 128 µs to 214 µs | 81.1 µs | 1.34 µs | 1.28 µs to 2.14 µs | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 100 | 7.5 ms | 7.43 ms to 7.6 ms | 127 µs | 75 µs | 74.3 µs to 76 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 1000 | 863 µs | 859 µs to 872 µs | 11.3 µs | 863 ns | 859 ns to 872 ns | 20 | 148 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 1000 | 1.14 ms | 1.12 ms to 1.7 ms | 292 µs | 1.14 µs | 1.12 µs to 1.7 µs | 20 | 142 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 1000 | 1.28 ms | 1.27 ms to 1.3 ms | 18.6 µs | 1.28 µs | 1.27 µs to 1.3 µs | 20 | 149 | 0 diff |
| translit | anyascii 0.3.3 | 1000 | 88.8 ms | 88 ms to 90.6 ms | 1.98 ms | 88.8 µs | 88 µs to 90.6 µs | 20 | 131 | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii per doc | 10000 | 14.2 ms | 14.1 ms to 14.3 ms | 249 µs | 1.42 µs | 1.41 µs to 1.43 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii [arrow] | 10000 | 16.7 ms | 16.1 ms to 17 ms | 606 µs | 1.67 µs | 1.61 µs to 1.7 µs | 20 |  | 0 diff |
| translit | pyroid.translit 1.0.0.dev0 anyascii | 10000 | 21.8 ms | 21 ms to 22.4 ms | 1.11 ms | 2.18 µs | 2.1 µs to 2.24 µs | 20 |  | 0 diff |
| translit | anyascii 0.3.3 | 10000 | 922 ms | 921 ms to 925 ms | 17 ms | 92.2 µs | 92.1 µs to 92.5 µs | 20 |  | 0 diff |
