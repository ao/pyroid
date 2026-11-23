# pyroid.difflib vs the standard library's difflib

EC2 c8g.16xlarge devbox (Graviton4, 64 vCPU), shared, indicative; CPython 3.12.16, pyroid 1.0.1 release build; load average 0.99 before and 1.35 after. bench/difflib/difflib_bench.py, best of 7 rounds (1 for 1,000,000 names and close_matches_batch, 3 for the whole standard library), both libraries interleaved in one process, results checked identical first.

| Case | Variant | Best | Median |
|---|---|---:|---:|
| SequenceMatcher(None, a, b).ratio(), names | difflib | 9.55 us | 9.61 us |
| SequenceMatcher(None, a, b).ratio(), names | pyroid.difflib | 2.02 us | 2.02 us |
| SequenceMatcher(None, a, b).ratio(), names | ratio_batch, 1 thread | 0.56 us | 0.56 us |
| SequenceMatcher(None, a, b).ratio(), names | ratio_batch | 0.11 us | 0.11 us |
| SequenceMatcher(None, a, b).ratio(), names | rapidfuzz fuzz.ratio (another score) | 0.24 us | 0.24 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | difflib | 435.86 us | 436.10 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | pyroid.difflib | 22.93 us | 22.94 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | ratio_batch, 1 thread | 19.63 us | 19.64 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | ratio_batch | 2.70 us | 2.72 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | rapidfuzz fuzz.ratio (another score) | 4.77 us | 4.79 us |
| get_close_matches, 10,000 place names | difflib | 13.52 ms | 13.54 ms |
| get_close_matches, 10,000 place names | pyroid.difflib, 1 thread | 0.90 ms | 0.91 ms |
| get_close_matches, 10,000 place names | pyroid.difflib | 0.58 ms | 0.63 ms |
| get_close_matches, 100,000 place names | difflib | 142.00 ms | 142.13 ms |
| get_close_matches, 100,000 place names | pyroid.difflib, 1 thread | 8.01 ms | 8.03 ms |
| get_close_matches, 100,000 place names | pyroid.difflib | 3.09 ms | 3.11 ms |
| get_close_matches, 1,000,000 place names | difflib | 1,756.79 ms | 1,756.79 ms |
| get_close_matches, 1,000,000 place names | pyroid.difflib, 1 thread | 100.45 ms | 100.45 ms |
| get_close_matches, 1,000,000 place names | pyroid.difflib | 32.19 ms | 32.19 ms |
| close_matches_batch, 1,000 queries x 10,000 names | difflib | 11,727.42 ms | 11,727.42 ms |
| close_matches_batch, 1,000 queries x 10,000 names | close_matches_batch, 1 thread | 482.67 ms | 482.67 ms |
| close_matches_batch, 1,000 queries x 10,000 names | close_matches_batch | 62.78 ms | 62.78 ms |
| unified_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 6.02 ms | 6.09 ms |
| unified_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 0.64 ms | 0.65 ms |
| context_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 6.23 ms | 6.25 ms |
| context_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 0.70 ms | 0.71 ms |
| ndiff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 30.41 ms | 30.47 ms |
| ndiff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 2.60 ms | 2.63 ms |
| HtmlDiff().make_file, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 52.92 ms | 53.28 ms |
| HtmlDiff().make_file, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 8.45 ms | 8.47 ms |
| unified_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 8.53 ms | 8.56 ms |
| unified_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 0.90 ms | 0.91 ms |
| context_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 8.75 ms | 8.79 ms |
| context_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 0.98 ms | 1.00 ms |
| ndiff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 107.16 ms | 107.26 ms |
| ndiff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 6.35 ms | 6.38 ms |
| HtmlDiff().make_file, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 136.93 ms | 137.48 ms |
| HtmlDiff().make_file, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 13.69 ms | 13.70 ms |
| unified_diff, the standard library 3.12 -> 3.14 (591 files, 594,900 lines) | difflib | 378.75 ms | 378.96 ms |
| unified_diff, the standard library 3.12 -> 3.14 (591 files, 594,900 lines) | pyroid.difflib | 52.39 ms | 52.42 ms |
