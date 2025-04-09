# Pyroid Benchmark Results

Generated on: 2025-04-09 15:04:09

| Operation | Pure Python | NumPy | pyroid | Speedup vs Python | Speedup vs NumPy |
|-----------|------------|-------|--------|-------------------|------------------|
| Sum 1,000 numbers | 0.00ms | 0.05ms | 0.00ms | infx | N/A |
| Mean 1,000 numbers | 0.00ms | 0.05ms | 0.00ms | infx | N/A |
| Std 1,000 numbers | 0.07ms | 0.08ms | 0.00ms | 75.50x | 83.75x |
| Matrix multiply 31x31 | 1.14ms | 0.04ms | 0.00ms | infx | N/A |
| Sum 10,000 numbers | 0.02ms | 0.21ms | 0.00ms | infx | N/A |
| Mean 10,000 numbers | 0.02ms | 0.26ms | 0.00ms | infx | N/A |
| Std 10,000 numbers | 0.72ms | 0.25ms | 0.00ms | 337.44x | 114.67x |
| Matrix multiply 100x100 | 36.52ms | 0.05ms | 0.00ms | 38296.00x | 48.25x |
| Sum 100,000 numbers | 0.19ms | 1.85ms | 0.00ms | infx | N/A |
| Mean 100,000 numbers | 0.21ms | 2.03ms | 0.00ms | 178.00x | 1707.00x |
| Std 100,000 numbers | 6.46ms | 2.07ms | 0.00ms | 3009.22x | 965.22x |
| Regex replace 13,000 chars | 0.12ms | N/A | 0.04ms | 2.89x | N/A |
| Text cleanup 1,000 texts | 0.45ms | N/A | 0.42ms | 1.05x | N/A |
| Base64 encode 61,000 chars | 0.08ms | N/A | 0.06ms | 1.33x | N/A |
| Base64 decode 84,000 chars | 0.01ms | N/A | 0.01ms | 1.43x | N/A |
| Regex replace 130,000 chars | 0.34ms | N/A | 0.31ms | 1.12x | N/A |
| Text cleanup 10,000 texts | 3.92ms | N/A | 3.85ms | 1.02x | N/A |
| Base64 encode 610,000 chars | 0.72ms | N/A | 0.65ms | 1.11x | N/A |
| Base64 decode 840,000 chars | 0.05ms | N/A | 0.01ms | 3.76x | N/A |
| Filter 1,000 items | 0.07ms | N/A | 0.04ms | 1.69x | N/A |
| Map 1,000 items | 0.04ms | N/A | 0.04ms | 1.10x | N/A |
| Reduce 1,000 items | 0.01ms | N/A | 0.03ms | 0.20x | N/A |
| Sort 1,000 items | 0.06ms | N/A | 0.04ms | 1.43x | N/A |
| Sort 1,000 items with key | 0.09ms | N/A | 0.09ms | 1.04x | N/A |
| Filter 10,000 items | 0.38ms | N/A | 0.39ms | 0.95x | N/A |
| Map 10,000 items | 0.34ms | N/A | 0.34ms | 1.00x | N/A |
| Reduce 10,000 items | 0.05ms | N/A | 0.34ms | 0.13x | N/A |
| Sort 10,000 items | 0.69ms | N/A | 0.65ms | 1.07x | N/A |
| Sort 10,000 items with key | 1.11ms | N/A | 0.97ms | 1.15x | N/A |
| Fetch single URL | 1008.87ms | N/A | 1562.67ms | 0.65x | N/A |
| Fetch 25 URLs | 2292.23ms | N/A | 2066.63ms | 1.11x | N/A |
| Async sleep | 501.22ms | N/A | 501.20ms | 1.00x | N/A |
| Async file read | 0.57ms | N/A | 0.14ms | 4.09x | N/A |
| Gather tasks | 301.47ms | N/A | 301.15ms | 1.00x | N/A |
| Zero-copy buffer | 96.83ms | N/A | 0.13ms | 734.38x | N/A |
| Parallel processing | 9735.32ms | N/A | 0.14ms | 67159.38x | N/A |
| Unified runtime | 2251.65ms | N/A | 0.25ms | 8960.23x | N/A |
| Web Scraping | 1352.23ms | N/A | 2235.80ms | 0.60x | N/A |
| Data Processing Pipeline | 20.75ms | N/A | 22.67ms | 0.92x | N/A |
| Text Processing Pipeline | 26.56ms | N/A | 27.49ms | 0.97x | N/A |
| Scientific Computing | 96.95ms | 21.51ms | 101.59ms | 0.95x | 0.21x |
| Web Scraping | 1386.86ms | N/A | 2574.20ms | 0.54x | N/A |
| High-Throughput Data Processing | 63.74ms | N/A | 1068.72ms | 0.06x | N/A |
