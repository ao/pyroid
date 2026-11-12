# pyroid.metrics vs rouge-score 0.1.2 / sacrebleu 2.6.0 (Docker/OrbStack VM, indicative)

Generated from `metrics.json` (written by `bench/metrics.py`). Wall time in ms, median of the runs (1 run for
reference rows over 100k pairs). Every row's results were compared with the 1-process reference's before it
was reported (exact `repr`/`str`/signature strings): all identical.

- CPython 3.13.18, aarch64, 12 vCPUs shared with other sessions; load average (1/5/15 min) 10.4, 18.8, 18.2 before, 4.3, 8.3, 12.7 after.
- pyroid 1.0.0-dev release wheel; references rouge-score 0.1.2, sacrebleu 2.6.0, nltk 3.10.4, numpy 2.5.4, regex 2026.9.29.
- ROUGE: CNN/DM test, lead-3 vs highlights, rouge1/rouge2/rougeL/rougeLsum (100k = the 11,490 pairs repeated).
- BLEU/chrF: WMT21 newstest2021 segments in file order (distinct), first reference; default settings.
- x8: references in 8 fork processes (corpus scores via sacrebleu's `_extract_corpus_statistics` per chunk);
  pyroid at `threads=8` in one process.

| workload | pairs | reference x1 | reference x8 | pyroid per-call loop | pyroid batch t=1 | pyroid batch t=8 | x1 speedup | x8 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rouge | 1,000 | 852 | 238 | 21.2 | 22.4 | 18.3 | 38x | 13x |
| rouge+stem | 1,000 | 2,260 | 359 | 37.0 | 29.4 | 13.4 | 77x | 27x |
| rouge | 10,000 | 15,998 | 4,457 | 388 | 394 | 244 | 41x | 18x |
| rouge+stem | 10,000 | 32,216 | 7,991 | 513 | 497 | 311 | 65x | 26x |
| rouge | 100,000 | 178,492 | 34,741 | 2,661 | 3,039 | 1,179 | 59x | 29x |
| rouge+stem | 100,000 | 306,034 | 47,777 | 4,030 | 4,715 | 1,359 | 65x | 35x |
| bleu-sentence | 1,000 | 127 | 80.4 | 19.5 | 15.8 | 12.7 | 8x | 6x |
| bleu-sentence | 10,000 | 1,249 | 291 | 206 | 160 | 109 | 8x | 3x |
| bleu-sentence | 100,000 | 9,741 | 2,434 | 2,286 | 1,862 | 1,327 | 5x | 2x |
| bleu-corpus | 1,000 | 115 | 101 | n/a | 8.1 | 3.0 | 14x | 33x |
| bleu-corpus | 10,000 | 898 | 310 | n/a | 94.2 | 41.3 | 10x | 8x |
| bleu-corpus | 100,000 | 7,827 | 1,716 | n/a | 681 | 276 | 11x | 6x |
| chrf-sentence | 1,000 | 281 | 129 | 28.0 | 22.3 | 9.4 | 13x | 14x |
| chrf-sentence | 10,000 | 2,824 | 612 | 273 | 230 | 82.9 | 12x | 7x |
| chrf-sentence | 100,000 | 22,661 | 4,361 | 2,371 | 1,987 | 915 | 11x | 5x |
| chrf-corpus | 1,000 | 278 | 144 | n/a | 17.7 | 5.1 | 16x | 28x |
| chrf-corpus | 10,000 | 3,023 | 646 | n/a | 186 | 34.8 | 16x | 19x |
| chrf-corpus | 100,000 | 27,476 | 5,213 | n/a | 1,421 | 288 | 19x | 18x |

Batch = `RougeScorer.score_batch`, `sentence_bleu_batch`/`sentence_chrf_batch`, or `BLEU`/`CHRF.corpus_score(threads=)`.
Per-sentence BLEU/chrF time is dominated by building sacrebleu's Python `BLEUScore`/`CHRFScore` objects (kept in Python for exactness).
