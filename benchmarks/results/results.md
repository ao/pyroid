# Pyroid Benchmark Results

Generated on: 2025-04-10 01:06:39

| Operation | Pure Python | NumPy | pyroid | Speedup vs Python | Speedup vs NumPy |
|-----------|------------|-------|--------|-------------------|------------------|
| Fetch single URL | 2254.59ms | N/A | 815.13ms | 2.77x | N/A |
| Fetch 25 URLs | 2988.16ms | N/A | 2608.96ms | 1.15x | N/A |
| Async sleep | 501.60ms | N/A | 501.61ms | 1.00x | N/A |
| Async file read | 0.38ms | N/A | 0.72ms | 0.53x | N/A |
| Gather tasks | 301.69ms | N/A | 301.99ms | 1.00x | N/A |
| Zero-copy buffer | 108.28ms | N/A | 78.92ms | 1.37x | N/A |
| Parallel processing | 9741.54ms | N/A | 3422.02ms | 2.85x | N/A |
| Unified runtime | 2469.07ms | N/A | 5467.34ms | 0.45x | N/A |
| Web Scraping | 1695.25ms | N/A | 1852.95ms | 0.91x | N/A |
