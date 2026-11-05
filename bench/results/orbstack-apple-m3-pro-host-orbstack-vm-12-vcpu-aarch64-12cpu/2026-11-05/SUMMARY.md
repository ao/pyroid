## Benchmark results: orbstack-apple-m3-pro-host-orbstack-vm-12-vcpu-aarch64-12cpu, 2026-11-05

*Docker/OrbStack VM, indicative (not README numbers).* CPU: Apple M3 Pro (host; OrbStack VM 12 vCPU), 12 logical CPUs (affinity 12); kernel 7.2.9-orbstack-00391-g2b298d05fbd4; Python 3.13.18 (GIL on); git beaa569ad7a8 (BENCH_GIT_SHA); manifest sha256 28b2499b62de.

Load average (1/5/15 min) before: 3.32 / 1.78 / 1.20; after: 9.86 / 10.14 / 7.43. Other sessions share this VM.

Versions: astral-dev-toolchain-hyperfine 2.0.0, datasketch 2.0.0, datatrove 0.10.1, fasttext-predict 0.9.2.4, ftfy 6.3.1, numpy 2.5.4, pyarrow 26.0.1, pyperf 2.10.0, regex 2026.9.29, rensa 0.5.0, spacy 3.8.18, tokenizers 0.23.3.

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
| digest-w8 | dev | 10000 | 8 | all | datatrove,fasttext,ftfy | ef314b8d6ae95cc6 | 0 | 5081 full / 4919 partial / 0 no ref |

Digest check: identical across digest-w1, digest-w8.

### End to end: hyperfine-e2e.json

| command | median wall (s) | IQR (s) | mean ± sd (s) | CPU-s (median) | CPU-s / wall | largest process RSS (MiB) | cgroup anon peak, sampled (MiB) | sum of worker maxrss (MiB, bound) | wall ms / doc | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| datatrove-0.10.1-w1 | 399.20 | 359.08 to 449.40 | 405.92 ± 90.51 | 401.8 | 1.01 | 403 | 434 | 408 | 39.92 | 3 |
| datatrove-0.10.1-w8 | 63.17 | 60.62 to 66.13 | 63.44 ± 5.51 | 464.5 | 7.35 | 365 | 2363 | 2995 | 6.32 | 3 |

### End to end: hyperfine-stages-w8.json

| command | median wall (s) | IQR (s) | mean ± sd (s) | CPU-s (median) | CPU-s / wall | largest process RSS (MiB) | cgroup anon peak, sampled (MiB) | sum of worker maxrss (MiB, bound) | wall ms / doc | runs |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text-w8 | 2.87 | 2.65 to 3.19 | 2.94 ± 0.54 | 18.9 | 6.57 | 167 | 841 | 1415 | 0.29 | 3 |
| lid-w8 | 3.12 | 2.92 to 3.12 | 2.99 ± 0.24 | 14.7 | 4.71 | 299 | 1872 | 2460 | 0.31 | 3 |
| gopher_repetition-w8 | 19.90 | 18.42 to 19.93 | 18.93 ± 1.72 | 134.9 | 6.78 | 226 | 1246 | 1876 | 1.99 | 3 |
| gopher_quality-w8 | 14.86 | 14.50 to 15.76 | 15.22 ± 1.30 | 108.8 | 7.32 | 226 | 1237 | 1876 | 1.49 | 3 |
| c4_quality-w8 | 16.88 | 16.76 to 17.87 | 17.45 ± 1.22 | 127.5 | 7.55 | 222 | 1228 | 1847 | 1.69 | 3 |
| fineweb_quality-w8 | 12.91 | 12.66 to 13.89 | 13.40 ± 1.30 | 94.9 | 7.35 | 226 | 1236 | 1868 | 1.29 | 3 |
| minhash-w8 | 13.78 | 13.60 to 14.67 | 14.25 ± 1.15 | 103.3 | 7.49 | 271 | 1325 | 2121 | 1.38 | 3 |

hyperfine `memory_peak_resident` is the largest single process, not the simultaneous total; the cgroup column is the container's anonymous memory sampled every 0.1 s during the run (it can miss short spikes; the median includes the warmup run), and the last memory column sums per-worker peaks (an upper bound).

### Per stage (pyperf; one value = one call over the batch)

| stage | implementation | docs / call | median / call | IQR / call | stdev | per doc (median) | per doc (IQR) | values | peak RSS MiB (track-memory) | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| fix_text | ftfy 6.3.1 | 1 | 110 µs | 100 µs to 132 µs | 20.7 µs | 110 µs | 100 µs to 132 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 100 | 83.7 ms | 74.4 ms to 96.9 ms | 16.6 ms | 837 µs | 744 µs to 969 µs | 20 | 129 | 0 diff |
| fix_text | ftfy 6.3.1 | 1000 | 878 ms | 836 ms to 940 ms | 75.2 ms | 878 µs | 836 µs to 940 µs | 20 |  | 0 diff |
| fix_text | ftfy 6.3.1 | 10000 | 10 s | 9.24 s to 12.3 s | 3.25 s | 1 ms | 924 µs to 1.23 ms | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1 | 39.3 µs | 36.6 µs to 45 µs | 4.69 µs | 39.3 µs | 36.6 µs to 45 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1 | 40.5 µs | 39.4 µs to 46.6 µs | 6.46 µs | 40.5 µs | 39.4 µs to 46.6 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1 | 62.9 µs | 60.5 µs to 71.1 µs | 11 µs | 62.9 µs | 60.5 µs to 71.1 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1 | 70.3 µs | 64.7 µs to 88.4 µs | 14.9 µs | 70.3 µs | 64.7 µs to 88.4 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1 | 160 µs | 150 µs to 169 µs | 31 µs | 160 µs | 150 µs to 169 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 100 | 28 ms | 25.2 ms to 32 ms | 8.11 ms | 280 µs | 252 µs to 320 µs | 20 | 260 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 100 | 31.3 ms | 27.6 ms to 37.2 ms | 8.59 ms | 313 µs | 276 µs to 372 µs | 20 | 260 | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 100 | 34.9 ms | 34.1 ms to 37.9 ms | 7.84 ms | 349 µs | 341 µs to 379 µs | 20 | 261 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 100 | 39.6 ms | 38.9 ms to 40.3 ms | 4.07 ms | 396 µs | 389 µs to 403 µs | 20 | 130 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 100 | 42.5 ms | 40.2 ms to 46.3 ms | 13.1 ms | 425 µs | 402 µs to 463 µs | 20 | 130 | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 1000 | 281 ms | 262 ms to 296 ms | 38.9 ms | 281 µs | 262 µs to 296 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 1000 | 295 ms | 270 ms to 321 ms | 42.6 ms | 295 µs | 270 µs to 321 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 1000 | 387 ms | 356 ms to 443 ms | 53.5 ms | 387 µs | 356 µs to 443 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 1000 | 483 ms | 430 ms to 547 ms | 69.1 ms | 483 µs | 430 µs to 547 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 1000 | 490 ms | 435 ms to 531 ms | 115 ms | 490 µs | 435 µs to 531 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=5 | 10000 | 3.07 s | 3.03 s to 3.36 s | 280 ms | 307 µs | 303 µs to 336 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.bin k=1 | 10000 | 3.21 s | 3.08 s to 3.32 s | 179 ms | 321 µs | 308 µs to 332 µs | 20 |  | 0 diff |
| lid | datatrove 0.10.1 LanguageFilter(en, 0.65, ft176) | 10000 | 4.33 s | 4.24 s to 4.51 s | 228 ms | 433 µs | 424 µs to 451 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=5 | 10000 | 4.93 s | 4.82 s to 5.16 s | 550 ms | 493 µs | 482 µs to 516 µs | 20 |  | 0 diff |
| lid | fasttext-predict 0.9.2.4 lid.176.ftz k=1 | 10000 | 4.99 s | 4.85 s to 5.31 s | 337 ms | 499 µs | 485 µs to 531 µs | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1 | 1.3 ms | 1.18 ms to 1.39 ms | 361 µs | 1.3 ms | 1.18 ms to 1.39 ms | 20 |  | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 100 | 749 ms | 708 ms to 783 ms | 104 ms | 7.49 ms | 7.08 ms to 7.83 ms | 20 | 181 | 0 diff |
| gopher_repetition | datatrove 0.10.1 GopherRepetitionFilter() | 1000 | 8.65 s | 8.47 s to 9.2 s | 1.14 s | 8.65 ms | 8.47 ms to 9.2 ms | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1 | 1.39 ms | 1.04 ms to 2.11 ms | 659 µs | 1.39 ms | 1.04 ms to 2.11 ms | 20 |  | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 100 | 709 ms | 649 ms to 859 ms | 184 ms | 7.09 ms | 6.49 ms to 8.59 ms | 20 | 180 | 0 diff |
| gopher_quality | datatrove 0.10.1 GopherQualityFilter() | 1000 | 7.21 s | 6.72 s to 7.68 s | 1.84 s | 7.21 ms | 6.72 ms to 7.68 ms | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1 | 494 µs | 471 µs to 584 µs | 131 µs | 494 µs | 471 µs to 584 µs | 20 |  | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 100 | 523 ms | 446 ms to 677 ms | 171 ms | 5.23 ms | 4.46 ms to 6.77 ms | 20 | 177 | 0 diff |
| c4_quality | datatrove 0.10.1 C4QualityFilter() | 1000 | 6.18 s | 5.95 s to 7.14 s | 878 ms | 6.18 ms | 5.95 ms to 7.14 ms | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1 | 1.24 ms | 1.15 ms to 1.39 ms | 178 µs | 1.24 ms | 1.15 ms to 1.39 ms | 20 |  | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 100 | 798 ms | 730 ms to 911 ms | 156 ms | 7.98 ms | 7.3 ms to 9.11 ms | 20 | 178 | 0 diff |
| c4_quality@fineweb | datatrove 0.10.1 C4QualityFilter(filter_no_terminal_punct=False) | 1000 | 8.42 s | 7.81 s to 8.87 s | 853 ms | 8.42 ms | 7.81 ms to 8.87 ms | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1 | 1.05 ms | 936 µs to 1.3 ms | 353 µs | 1.05 ms | 936 µs to 1.3 ms | 20 |  | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 100 | 594 ms | 532 ms to 711 ms | 132 ms | 5.94 ms | 5.32 ms to 7.11 ms | 20 | 181 | 0 diff |
| fineweb_quality | datatrove 0.10.1 FineWebQualityFilter() | 1000 | 6.54 s | 6.23 s to 6.99 s | 521 ms | 6.54 ms | 6.23 ms to 6.99 ms | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1 | 13 µs | 12.7 µs to 14.5 µs | 1.95 µs | 13 µs | 12.7 µs to 14.5 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1 | 24.9 µs | 23.4 µs to 27.3 µs | 5.1 µs | 24.9 µs | 23.4 µs to 27.3 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1 | 254 µs | 221 µs to 317 µs | 50.2 µs | 254 µs | 221 µs to 317 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1 | 1.01 ms | 921 µs to 1.15 ms | 275 µs | 1.01 ms | 921 µs to 1.15 ms | 20 |  | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 100 | 14.5 ms | 13.6 ms to 15.2 ms | 1.97 ms | 145 µs | 136 µs to 152 µs | 20 | 128 | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 100 | 15.8 ms | 14.2 ms to 17.6 ms | 2.24 ms | 158 µs | 142 µs to 176 µs | 20 | 136 | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 100 | 51.6 ms | 45.7 ms to 58.9 ms | 8.74 ms | 516 µs | 457 µs to 589 µs | 20 | 171 | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 100 | 637 ms | 596 ms to 679 ms | 111 ms | 6.37 ms | 5.96 ms to 6.79 ms | 20 | 186 | 0 diff |
| minhash | rensa 0.5.0 RMinHash.update(112) per doc | 1000 | 150 ms | 147 ms to 159 ms | 20.9 ms | 150 µs | 147 µs to 159 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | rensa 0.5.0 RMinHash.from_token_sets(112) | 1000 | 159 ms | 156 ms to 169 ms | 18.3 ms | 159 µs | 156 µs to 169 µs | 20 |  | P 0.9611 R 0.9544 |
| minhash | datasketch 2.0.0 MinHash.bulk(num_perm=112) | 1000 | 498 ms | 474 ms to 521 ms | 33.6 ms | 498 µs | 474 µs to 521 µs | 20 |  | P 0.9649 R 0.9649 |
| minhash | datatrove 0.10.1 MinhashDedupSignature | 1000 | 6.88 s | 6.72 s to 7.07 s | 703 ms | 6.88 ms | 6.72 ms to 7.07 ms | 20 |  | 0 diff |
