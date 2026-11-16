# pyroid.difflib vs the standard library's difflib

AWS EC2 c7i.4xlarge (Intel Xeon Platinum 8488C, 16 vCPU), dedicated, Amazon Linux 2023, Docker; CPython 3.12.16, pyroid 1.0.1 release wheel built on the instance; load average 0.41 before and 0.70 after. bench/difflib/difflib_bench.py, best of 7 rounds (1 for 1,000,000 names and close_matches_batch, 3 for the whole standard library), both libraries interleaved in one process, results checked identical first.

| Case | Variant | Best | Median |
|---|---|---:|---:|
| SequenceMatcher(None, a, b).ratio(), names | difflib | 9.98 us | 10.03 us |
| SequenceMatcher(None, a, b).ratio(), names | pyroid.difflib | 2.07 us | 2.08 us |
| SequenceMatcher(None, a, b).ratio(), names | ratio_batch, 1 thread | 0.64 us | 0.65 us |
| SequenceMatcher(None, a, b).ratio(), names | ratio_batch | 0.19 us | 0.21 us |
| SequenceMatcher(None, a, b).ratio(), names | rapidfuzz fuzz.ratio (another score) | 0.25 us | 0.26 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | difflib | 397.66 us | 398.46 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | pyroid.difflib | 26.70 us | 26.79 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | ratio_batch, 1 thread | 23.22 us | 23.27 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | ratio_batch | 4.97 us | 5.09 us |
| SequenceMatcher(None, a, b).ratio(), 500 characters | rapidfuzz fuzz.ratio (another score) | 4.50 us | 4.56 us |
| get_close_matches, 10,000 place names | difflib | 13.59 ms | 13.77 ms |
| get_close_matches, 10,000 place names | pyroid.difflib, 1 thread | 1.09 ms | 1.12 ms |
| get_close_matches, 10,000 place names | pyroid.difflib | 0.79 ms | 0.99 ms |
| get_close_matches, 100,000 place names | difflib | 144.40 ms | 145.23 ms |
| get_close_matches, 100,000 place names | pyroid.difflib, 1 thread | 9.93 ms | 9.99 ms |
| get_close_matches, 100,000 place names | pyroid.difflib | 4.83 ms | 4.96 ms |
| get_close_matches, 1,000,000 place names | difflib | 1,800.78 ms | 1,800.78 ms |
| get_close_matches, 1,000,000 place names | pyroid.difflib, 1 thread | 136.33 ms | 136.33 ms |
| get_close_matches, 1,000,000 place names | pyroid.difflib | 58.24 ms | 58.24 ms |
| close_matches_batch, 1,000 queries x 10,000 names | difflib | 11,895.28 ms | 11,895.28 ms |
| close_matches_batch, 1,000 queries x 10,000 names | close_matches_batch, 1 thread | 584.00 ms | 584.00 ms |
| close_matches_batch, 1,000 queries x 10,000 names | close_matches_batch | 102.68 ms | 102.68 ms |
| unified_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 5.62 ms | 5.67 ms |
| unified_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 0.78 ms | 0.79 ms |
| context_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 5.86 ms | 5.93 ms |
| context_diff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 0.86 ms | 0.88 ms |
| ndiff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 30.55 ms | 30.71 ms |
| ndiff, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 2.74 ms | 2.78 ms |
| HtmlDiff().make_file, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | difflib | 52.45 ms | 52.60 ms |
| HtmlDiff().make_file, argparse.py 3.12 -> 3.14 (2,650 / 2,789 lines) | pyroid.difflib | 8.46 ms | 8.54 ms |
| unified_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 7.89 ms | 7.96 ms |
| unified_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 1.07 ms | 1.10 ms |
| context_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 8.11 ms | 8.21 ms |
| context_diff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 1.14 ms | 1.17 ms |
| ndiff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 106.96 ms | 107.23 ms |
| ndiff, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 6.98 ms | 7.09 ms |
| HtmlDiff().make_file, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | difflib | 138.28 ms | 138.90 ms |
| HtmlDiff().make_file, typing.py 3.12 -> 3.14 (3,494 / 3,866 lines) | pyroid.difflib | 14.74 ms | 14.99 ms |
| unified_diff, the standard library 3.12 -> 3.14 (591 files, 594,900 lines) | difflib | 370.71 ms | 372.43 ms |
| unified_diff, the standard library 3.12 -> 3.14 (591 files, 594,900 lines) | pyroid.difflib | 65.84 ms | 66.03 ms |
