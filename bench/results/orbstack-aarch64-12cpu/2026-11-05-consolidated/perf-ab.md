# P4 perf A/B comparisons (Docker/OrbStack VM on Apple Silicon, indicative)

Raw output of `bench/perf_matrix.py compare A B` for each idea in the P4 report: candidate rows of
`bench/stages.py` (gated before timing), dev slice unless stated, median us/doc, pool = RAYON_NUM_THREADS
(or Python threads for per-doc rows). Positive = B faster. Identical wheels varied by up to +-10% (pool 1)
and +-20% (pool 8) between runs in this shared VM.

## 1. Allocator: base vs mimalloc, 10k docs (round 1)

### b315t
| row | pool | A us/doc | B us/doc | B vs A | (A=base-b315t, B=mimalloc-b315t) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures direct | 1 | 107.56 | 107.74 | -0.2% |  |
| pyroid.dedup minhash_signatures direct | 8 | 17.40 | 22.21 | -21.7% | slower >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 1 | 145.87 | 119.77 | +21.8% | faster >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 8 | 21.73 | 17.64 | +23.2% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 1 | 108.90 | 107.14 | +1.6% |  |
| pyroid.fasttext lid.176.ftz k=1 batch | 8 | 18.34 | 17.09 | +7.3% | faster >=5% |
| pyroid.filters apply([GopherRepetition]) direct | 1 | 152.61 | 153.92 | -0.9% |  |
| pyroid.filters apply([GopherRepetition]) direct | 8 | 24.45 | 25.25 | -3.2% |  |
| pyroid.ftfy fix_text_batch | 1 | 5.62 | 6.25 | -10.1% | slower >=5% |
| pyroid.ftfy fix_text_batch | 8 | 1.56 | 1.73 | -9.7% | slower >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 1 | 5.17 | 5.68 | -9.1% | slower >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.10 | 1.16 | -5.0% |  |
| pyroid.translit anyascii | 1 | 1.47 | 1.64 | -10.5% | slower >=5% |
| pyroid.translit anyascii | 8 | 0.97 | 1.08 | -10.0% | slower >=5% |
| pyroid.translit anyascii [arrow] | 1 | 1.21 | 1.37 | -11.6% | slower >=5% |
| pyroid.translit anyascii [arrow] | 8 | 0.41 | 0.60 | -32.0% | slower >=5% |
### b312
| row | pool | A us/doc | B us/doc | B vs A | (A=base-b312, B=mimalloc-b312) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures direct | 1 | 109.19 | 104.54 | +4.5% |  |
| pyroid.dedup minhash_signatures direct | 8 | 18.09 | 18.43 | -1.9% |  |
| pyroid.fasttext lid.176.bin k=1 batch | 1 | 146.83 | 125.57 | +16.9% | faster >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 8 | 23.76 | 17.15 | +38.6% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 1 | 104.81 | 107.92 | -2.9% |  |
| pyroid.fasttext lid.176.ftz k=1 batch | 8 | 16.58 | 18.30 | -9.4% | slower >=5% |
| pyroid.filters apply([GopherRepetition]) direct | 1 | 153.25 | 148.80 | +3.0% |  |
| pyroid.filters apply([GopherRepetition]) direct | 8 | 24.91 | 22.50 | +10.7% | faster >=5% |
| pyroid.ftfy fix_text_batch | 1 | 5.77 | 5.78 | -0.3% |  |
| pyroid.ftfy fix_text_batch | 8 | 1.46 | 1.63 | -10.6% | slower >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 1 | 5.16 | 5.36 | -3.8% |  |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.01 | 1.00 | +0.8% |  |
| pyroid.translit anyascii | 1 | 1.36 | 1.41 | -3.4% |  |
| pyroid.translit anyascii | 8 | 0.81 | 0.83 | -2.4% |  |
| pyroid.translit anyascii [arrow] | 1 | 1.11 | 1.24 | -10.7% | slower >=5% |
| pyroid.translit anyascii [arrow] | 8 | 0.40 | 0.45 | -11.1% | slower >=5% |

## 1. Allocator: 100k headline docs, pool 8 (base vs mimalloc; base vs glibc tunables; dedup on 3.13)

| row | pool | A us/doc | B us/doc | B vs A | (A=base, B=mimalloc) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures direct | 8 | 20.28 | 18.48 | +9.7% | faster >=5% |
| pyroid.ftfy fix_text_batch | 8 | 1.73 | 1.71 | +1.1% |  |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 0.96 | 0.86 | +11.4% | faster >=5% |
| pyroid.translit anyascii | 8 | 1.07 | 1.07 | -0.4% |  |
| pyroid.translit anyascii [arrow] | 8 | 0.41 | 0.39 | +6.4% | faster >=5% |
| row | pool | A us/doc | B us/doc | B vs A | (A=base, B=tunables) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures direct | 8 | 20.28 | 19.59 | +3.5% |  |
| pyroid.ftfy fix_text_batch | 8 | 1.73 | 1.69 | +2.0% |  |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 0.96 | 0.92 | +4.2% |  |
| pyroid.translit anyascii | 8 | 1.07 | 0.95 | +12.1% | faster >=5% |
| pyroid.translit anyascii [arrow] | 8 | 0.41 | 0.41 | +0.2% |  |

## 1. Allocator round 2, 3.15t: base vs mimalloc; base vs GLIBC_TUNABLES=glibc.malloc.hugetlb=1

| row | pool | A us/doc | B us/doc | B vs A | (A=base, B=mimalloc) |
|---|---:|---:|---:|---:|---|
| pyroid.fasttext lid.176.bin k=1 batch | 1 | 150.53 | 122.53 | +22.9% | faster >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 8 | 23.22 | 18.36 | +26.5% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 1 | 109.77 | 106.60 | +3.0% |  |
| pyroid.fasttext lid.176.ftz k=1 batch | 8 | 17.36 | 16.82 | +3.2% |  |
| pyroid.ftfy fix_text_batch | 1 | 5.62 | 5.79 | -3.0% |  |
| pyroid.ftfy fix_text_batch | 8 | 1.68 | 1.58 | +6.2% | faster >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 1 | 5.18 | 5.22 | -0.7% |  |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.09 | 1.28 | -14.9% | slower >=5% |
| pyroid.translit anyascii | 1 | 1.45 | 1.62 | -10.4% | slower >=5% |
| pyroid.translit anyascii | 8 | 0.90 | 1.00 | -10.1% | slower >=5% |
| pyroid.translit anyascii [arrow] | 1 | 1.22 | 1.38 | -11.9% | slower >=5% |
| pyroid.translit anyascii [arrow] | 8 | 0.40 | 0.41 | -1.5% |  |
| row | pool | A us/doc | B us/doc | B vs A | (A=base, B=hugetlb) |
|---|---:|---:|---:|---:|---|
| pyroid.fasttext lid.176.bin k=1 batch | 1 | 150.53 | 132.48 | +13.6% | faster >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 8 | 23.22 | 19.16 | +21.2% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 1 | 109.77 | 108.88 | +0.8% |  |
| pyroid.fasttext lid.176.ftz k=1 batch | 8 | 17.36 | 17.80 | -2.5% |  |
| pyroid.ftfy fix_text_batch | 1 | 5.62 | 5.63 | -0.3% |  |
| pyroid.ftfy fix_text_batch | 8 | 1.68 | 1.63 | +2.8% |  |
| pyroid.ftfy fix_text_batch [arrow] | 1 | 5.18 | 5.16 | +0.3% |  |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.09 | 1.21 | -10.3% | slower >=5% |
| pyroid.translit anyascii | 1 | 1.45 | 1.54 | -5.7% | slower >=5% |
| pyroid.translit anyascii | 8 | 0.90 | 0.89 | +0.2% |  |
| pyroid.translit anyascii [arrow] | 1 | 1.22 | 1.12 | +8.9% | faster >=5% |
| pyroid.translit anyascii [arrow] | 8 | 0.40 | 0.40 | -0.2% |  |

## 2. Parallel list extraction, 3.15t: fresh strs (BENCH_FRESH_STRS=1), then warm

| row | pool | A us/doc | B us/doc | B vs A | (A=fresh-base, B=fresh-ext) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures direct | 1 | 114.66 | 117.45 | -2.4% |  |
| pyroid.dedup minhash_signatures direct | 8 | 21.36 | 21.17 | +0.9% |  |
| pyroid.filters apply([GopherRepetition]) direct | 1 | 167.04 | 163.40 | +2.2% |  |
| pyroid.filters apply([GopherRepetition]) direct | 8 | 28.24 | 25.33 | +11.5% | faster >=5% |
| pyroid.ftfy fix_text_batch | 1 | 7.41 | 7.79 | -4.8% |  |
| pyroid.ftfy fix_text_batch | 8 | 3.39 | 2.43 | +39.4% | faster >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 1 | 5.24 | 5.81 | -9.8% | slower >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.26 | 1.27 | -0.6% |  |
| pyroid.translit anyascii | 1 | 3.15 | 3.21 | -1.9% |  |
| pyroid.translit anyascii | 8 | 2.79 | 1.72 | +62.8% | faster >=5% |
| pyroid.translit anyascii [arrow] | 1 | 1.34 | 1.31 | +2.6% |  |
| pyroid.translit anyascii [arrow] | 8 | 0.54 | 0.56 | -3.8% |  |
| row | pool | A us/doc | B us/doc | B vs A | (A=warm-base, B=warm-ext) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures direct | 8 | 19.50 | 19.39 | +0.6% |  |
| pyroid.ftfy fix_text_batch | 8 | 1.71 | 1.79 | -4.3% |  |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.24 | 1.12 | +11.2% | faster >=5% |
| pyroid.translit anyascii | 8 | 1.04 | 0.99 | +5.2% | faster >=5% |
| pyroid.translit anyascii [arrow] | 8 | 0.44 | 0.46 | -4.2% |  |

## 3. Detach threshold: 4 KiB vs always detach (per-doc rows, 5k docs)

### b312
| row | pool | A us/doc | B us/doc | B vs A | (A=cur-b312, B=detach0-b312) |
|---|---:|---:|---:|---:|---|
| pyroid.fasttext predict lid.176.bin k=1 per doc | 1 | 138.33 | 148.84 | -7.1% | slower >=5% |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 8 | 80.35 | 43.95 | +82.8% | faster >=5% |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 1 | 105.74 | 105.67 | +0.1% |  |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 8 | 79.97 | 27.74 | +188.3% | faster >=5% |
| pyroid.filters gopher_repetition_reasons per doc | 1 | 179.14 | 171.47 | +4.5% |  |
| pyroid.filters gopher_repetition_reasons per doc | 8 | 99.36 | 58.83 | +68.9% | faster >=5% |
| pyroid.ftfy fix_text per doc | 1 | 5.43 | 5.52 | -1.5% |  |
| pyroid.ftfy fix_text per doc | 8 | 5.54 | 8.17 | -32.3% | slower >=5% |
### b315t
| row | pool | A us/doc | B us/doc | B vs A | (A=cur-b315t, B=detach0-b315t) |
|---|---:|---:|---:|---:|---|
| pyroid.fasttext predict lid.176.bin k=1 per doc | 1 | 153.14 | 152.09 | +0.7% |  |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 8 | 22.96 | 27.05 | -15.1% | slower >=5% |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 1 | 102.77 | 104.42 | -1.6% |  |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 8 | 15.73 | 15.31 | +2.8% |  |
| pyroid.filters gopher_repetition_reasons per doc | 1 | 165.62 | 164.50 | +0.7% |  |
| pyroid.filters gopher_repetition_reasons per doc | 8 | 25.44 | 25.58 | -0.6% |  |
| pyroid.ftfy fix_text per doc | 1 | 5.56 | 5.36 | +3.8% |  |
| pyroid.ftfy fix_text per doc | 8 | 1.19 | 1.17 | +2.1% |  |

## 3. Detach: 4 KiB vs the kept GIL-aware DocCost rule (per-doc rows, 5k docs)

### b312
| row | pool | A us/doc | B us/doc | B vs A | (A=cur-b312, B=detach-b312) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures per doc | 1 | 118.35 | 122.52 | -3.4% |  |
| pyroid.dedup minhash_signatures per doc | 8 | 70.09 | 55.47 | +26.4% | faster >=5% |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 1 | 153.56 | 143.67 | +6.9% | faster >=5% |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 8 | 70.79 | 46.32 | +52.8% | faster >=5% |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 1 | 101.18 | 102.14 | -0.9% |  |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 8 | 55.46 | 28.53 | +94.4% | faster >=5% |
| pyroid.filters gopher_repetition_reasons per doc | 1 | 169.18 | 160.50 | +5.4% | faster >=5% |
| pyroid.filters gopher_repetition_reasons per doc | 8 | 91.89 | 66.50 | +38.2% | faster >=5% |
| pyroid.ftfy fix_text per doc | 1 | 5.55 | 5.40 | +2.9% |  |
| pyroid.ftfy fix_text per doc | 8 | 5.81 | 5.21 | +11.6% | faster >=5% |
### b315t
| row | pool | A us/doc | B us/doc | B vs A | (A=cur-b315t, B=detach-b315t) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup minhash_signatures per doc | 1 | 114.62 | 122.47 | -6.4% | slower >=5% |
| pyroid.dedup minhash_signatures per doc | 8 | 21.56 | 24.78 | -13.0% | slower >=5% |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 1 | 141.60 | 152.38 | -7.1% | slower >=5% |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 8 | 20.80 | 21.94 | -5.2% | slower >=5% |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 1 | 102.03 | 106.64 | -4.3% |  |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 8 | 15.53 | 16.36 | -5.1% | slower >=5% |

## 4. PGO: same commit without vs with PGO

### b315t
| row | pool | A us/doc | B us/doc | B vs A | (A=detach-b315t, B=pgo-b315t) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup MinHashDedup.fit direct | 1 | 106.05 | 97.29 | +9.0% | faster >=5% |
| pyroid.dedup MinHashDedup.fit direct | 8 | 19.28 | 14.32 | +34.7% | faster >=5% |
| pyroid.dedup minhash_signatures direct | 1 | 108.44 | 103.33 | +4.9% |  |
| pyroid.dedup minhash_signatures direct | 8 | 17.85 | 20.39 | -12.5% | slower >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 1 | 148.25 | 131.61 | +12.6% | faster >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 8 | 22.49 | 20.57 | +9.4% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 1 | 106.31 | 95.05 | +11.9% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 8 | 15.97 | 14.35 | +11.3% | faster >=5% |
| pyroid.filters apply([GopherRepetition]) direct | 1 | 151.11 | 138.49 | +9.1% | faster >=5% |
| pyroid.filters apply([GopherRepetition]) direct | 8 | 22.00 | 20.60 | +6.8% | faster >=5% |
| pyroid.ftfy fix_text_batch | 1 | 5.71 | 6.00 | -4.8% |  |
| pyroid.ftfy fix_text_batch | 8 | 1.74 | 1.70 | +2.3% |  |
| pyroid.ftfy fix_text_batch [arrow] | 1 | 5.27 | 5.12 | +2.9% |  |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.04 | 1.07 | -2.1% |  |
| pyroid.translit anyascii | 1 | 1.53 | 1.70 | -10.4% | slower >=5% |
| pyroid.translit anyascii | 8 | 1.01 | 0.98 | +2.9% |  |
| pyroid.translit anyascii [arrow] | 1 | 1.19 | 1.10 | +8.2% | faster >=5% |
| pyroid.translit anyascii [arrow] | 8 | 0.41 | 0.38 | +7.7% | faster >=5% |
### b312
| row | pool | A us/doc | B us/doc | B vs A | (A=detach-b312, B=pgo-b312) |
|---|---:|---:|---:|---:|---|
| pyroid.dedup MinHashDedup.fit direct | 1 | 102.27 | 98.96 | +3.4% |  |
| pyroid.dedup MinHashDedup.fit direct | 8 | 15.45 | 14.90 | +3.7% |  |
| pyroid.dedup minhash_signatures direct | 1 | 108.27 | 102.39 | +5.7% | faster >=5% |
| pyroid.dedup minhash_signatures direct | 8 | 17.58 | 16.84 | +4.4% |  |
| pyroid.fasttext lid.176.bin k=1 batch | 1 | 149.61 | 130.98 | +14.2% | faster >=5% |
| pyroid.fasttext lid.176.bin k=1 batch | 8 | 21.66 | 19.66 | +10.2% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 1 | 105.36 | 93.45 | +12.7% | faster >=5% |
| pyroid.fasttext lid.176.ftz k=1 batch | 8 | 16.72 | 14.90 | +12.2% | faster >=5% |
| pyroid.filters apply([GopherRepetition]) direct | 1 | 151.63 | 138.88 | +9.2% | faster >=5% |
| pyroid.filters apply([GopherRepetition]) direct | 8 | 23.70 | 21.24 | +11.6% | faster >=5% |
| pyroid.ftfy fix_text_batch | 1 | 5.51 | 5.23 | +5.4% | faster >=5% |
| pyroid.ftfy fix_text_batch | 8 | 1.51 | 1.37 | +10.9% | faster >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 1 | 5.36 | 5.10 | +5.1% | faster >=5% |
| pyroid.ftfy fix_text_batch [arrow] | 8 | 1.15 | 1.03 | +11.0% | faster >=5% |
| pyroid.translit anyascii | 1 | 1.49 | 1.27 | +17.6% | faster >=5% |
| pyroid.translit anyascii | 8 | 0.79 | 0.77 | +2.7% |  |
| pyroid.translit anyascii [arrow] | 1 | 1.10 | 1.12 | -1.9% |  |
| pyroid.translit anyascii [arrow] | 8 | 0.38 | 0.44 | -13.8% | slower >=5% |

## 3. Detach threshold (cont.): dedup and translit per-doc rows, 4 KiB vs always detach (rerun)

### b312
| pyroid.dedup minhash_signatures per doc | 1 | 129.40 | 124.03 | +4.3% |  |
| pyroid.dedup minhash_signatures per doc | 8 | 81.22 | 56.49 | +43.8% | faster >=5% |
| pyroid.translit anyascii per doc | 1 | 1.21 | 1.24 | -2.7% |  |
| pyroid.translit anyascii per doc | 8 | 2.35 | 3.60 | -34.6% | slower >=5% |
### b315t
| pyroid.dedup minhash_signatures per doc | 1 | 120.80 | 118.77 | +1.7% |  |
| pyroid.dedup minhash_signatures per doc | 8 | 22.51 | 33.26 | -32.3% | slower >=5% |
| pyroid.translit anyascii per doc | 1 | 1.25 | 1.13 | +10.8% | faster >=5% |
| pyroid.translit anyascii per doc | 8 | 0.43 | 0.45 | -3.3% |  |

## 3. Detach, 3.15t ABA check: the same 4 KiB wheel twice, then 4 KiB vs the DocCost rule (2nd round)

### cur (1st) vs cur (2nd), identical wheels
| pyroid.dedup minhash_signatures per doc | 1 | 114.62 | 124.86 | -8.2% | slower >=5% |
| pyroid.dedup minhash_signatures per doc | 8 | 21.56 | 27.42 | -21.4% | slower >=5% |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 1 | 141.60 | 147.58 | -4.0% |  |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 8 | 20.80 | 19.19 | +8.4% | faster >=5% |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 1 | 102.03 | 106.57 | -4.3% |  |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 8 | 15.53 | 16.80 | -7.6% | slower >=5% |
### cur (2nd) vs detach (2nd)
| pyroid.dedup minhash_signatures per doc | 1 | 124.86 | 120.28 | +3.8% |  |
| pyroid.dedup minhash_signatures per doc | 8 | 27.42 | 29.16 | -6.0% | slower >=5% |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 1 | 147.58 | 148.54 | -0.6% |  |
| pyroid.fasttext predict lid.176.bin k=1 per doc | 8 | 19.19 | 24.09 | -20.3% | slower >=5% |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 1 | 106.57 | 107.24 | -0.6% |  |
| pyroid.fasttext predict lid.176.ftz k=1 per doc | 8 | 16.80 | 16.09 | +4.5% |  |

## 5a. dedup in place: MinHashDedup.fit row (3.15t) and peak RSS of fit on FineWeb 100k (pool 8)

| pyroid.dedup MinHashDedup.fit direct | 1 | 105.34 | 103.61 | +1.7% |  |
| pyroid.dedup MinHashDedup.fit direct | 8 | 15.10 | 15.12 | -0.1% |  |

Peak RSS over the loaded corpus (`ru_maxrss` delta around `MinHashDedup(**cfg).fit(docs)`, 2 runs each):
list[str] 553-555 MiB before, 485 MiB after; Arrow column 262 MiB before, 186 MiB after.
