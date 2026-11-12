# datasketch 2.0.0 vs pyroid.datasketch (Docker/OrbStack VM, indicative)

Machine: Apple M3 Pro host, OrbStack VM with 12 vCPUs (aarch64), shared with other sessions (load average 7-10 during the run).
CPython 3.15.2 free-threaded (GIL off for the whole run), numpy 2.5.4, datasketch 2.0.0, pyroid release wheel built from `feat/datasketch` @ f01a181c (the last library change before the run).
Corpus: the first 10,000 docs of the pinned FineWeb dev slice, word 5-gram shingles (5.1M shingles in total); 128 permutations, seed 1.
Every row passed the equivalence gate first: identical signatures, identical LSH query result sets. Best of 3 wall-clock runs.
The datasketch "8-thread pool" rows split the documents over a ThreadPoolExecutor (possible because the GIL is off).

Command: `python bench/datasketch_bench.py --docs 10000 --repeats 3 --threads 8 --out datasketch-315t.json`

| group | row | threads | ms | items/s | vs datasketch (1 thread) |
|---|---|---:|---:|---:|---:|
| sig/affine32 | datasketch MinHash.bulk | 1 | 3128.1 | 3197 | 1.0x |
| sig/affine32 | datasketch MinHash.bulk, 8-thread pool | 8 | 2306.8 | 4335 | 1.4x |
| sig/affine32 | pyroid MinHash.bulk(threads=1) | 1 | 391.7 | 25533 | 8.0x |
| sig/affine32 | pyroid MinHash.bulk(threads=8) | 8 | 147.9 | 67602 | 21.1x |
| text/affine32 | datasketch shingles + MinHash.bulk (from text) | 1 | 4427.9 | 2258 | 1.0x |
| text/affine32 | datasketch from text, 8-thread pool | 8 | 2432.7 | 4111 | 1.8x |
| text/affine32 | pyroid minhash_batch(texts, threads=1) | 1 | 353.4 | 28293 | 12.5x |
| text/affine32 | pyroid minhash_batch(texts, threads=8) | 8 | 52.0 | 192267 | 85.1x |
| sig/legacy | datasketch MinHash.bulk | 1 | 4514.9 | 2215 | 1.0x |
| sig/legacy | datasketch MinHash.bulk, 8-thread pool | 8 | 2217.6 | 4509 | 2.0x |
| sig/legacy | pyroid MinHash.bulk(threads=1) | 1 | 779.2 | 12834 | 5.8x |
| sig/legacy | pyroid MinHash.bulk(threads=8) | 8 | 258.1 | 38750 | 17.5x |
| text/legacy | datasketch shingles + MinHash.bulk (from text) | 1 | 5783.4 | 1729 | 1.0x |
| text/legacy | datasketch from text, 8-thread pool | 8 | 2360.9 | 4236 | 2.4x |
| text/legacy | pyroid minhash_batch(texts, threads=1) | 1 | 597.0 | 16749 | 9.7x |
| text/legacy | pyroid minhash_batch(texts, threads=8) | 8 | 91.6 | 109163 | 63.1x |
| sig/affine64 | datasketch MinHash.bulk | 1 | 3657.7 | 2734 | 1.0x |
| sig/affine64 | datasketch MinHash.bulk, 8-thread pool | 8 | 2273.1 | 4399 | 1.6x |
| sig/affine64 | pyroid MinHash.bulk(threads=1) | 1 | 683.2 | 14637 | 5.4x |
| sig/affine64 | pyroid MinHash.bulk(threads=8) | 8 | 226.3 | 44182 | 16.2x |
| text/affine64 | datasketch shingles + MinHash.bulk (from text) | 1 | 4925.5 | 2030 | 1.0x |
| text/affine64 | datasketch from text, 8-thread pool | 8 | 2472.3 | 4045 | 2.0x |
| text/affine64 | pyroid minhash_batch(texts, threads=1) | 1 | 561.9 | 17796 | 8.8x |
| text/affine64 | pyroid minhash_batch(texts, threads=8) | 8 | 102.2 | 97868 | 48.2x |
| lsh/init | datasketch MinHashLSH(threshold=0.9) | 1 | 13.0 | 77 | 1.0x |
| lsh/init | pyroid MinHashLSH(threshold=0.9) | 1 | 1.5 | 685 | 8.9x |
| lsh-insert/t0.9 | datasketch insert per doc | 1 | 119.3 | 83819 | 1.0x |
| lsh-insert/t0.9 | pyroid insert per doc | 1 | 15.5 | 644707 | 7.7x |
| lsh-insert/t0.9 | pyroid insert_batch(threads=1) | 1 | 11.3 | 885886 | 10.6x |
| lsh-insert/t0.9 | pyroid insert_batch(threads=8) | 8 | 11.2 | 894597 | 10.7x |
| lsh-query/t0.9 | datasketch query per doc | 1 | 58.5 | 170870 | 1.0x |
| lsh-query/t0.9 | pyroid query per doc | 1 | 10.5 | 953410 | 5.6x |
| lsh-query/t0.9 | pyroid query_batch(threads=1) | 1 | 4.4 | 2260290 | 13.2x |
| lsh-query/t0.9 | pyroid query_batch(threads=8) | 8 | 5.0 | 2015231 | 11.8x |
| lsh-insert/t0.5 | datasketch insert per doc | 1 | 524.7 | 19058 | 1.0x |
| lsh-insert/t0.5 | pyroid insert per doc | 1 | 42.9 | 233295 | 12.2x |
| lsh-insert/t0.5 | pyroid insert_batch(threads=1) | 1 | 40.7 | 245922 | 12.9x |
| lsh-insert/t0.5 | pyroid insert_batch(threads=8) | 8 | 19.2 | 520935 | 27.3x |
| lsh-query/t0.5 | datasketch query per doc | 1 | 235.2 | 42523 | 1.0x |
| lsh-query/t0.5 | pyroid query per doc | 1 | 26.7 | 374112 | 8.8x |
| lsh-query/t0.5 | pyroid query_batch(threads=1) | 1 | 17.9 | 558815 | 13.1x |
| lsh-query/t0.5 | pyroid query_batch(threads=8) | 8 | 12.3 | 810994 | 19.1x |
