bench/yaml/small_docs_bench.py on AWS EC2 c7i.4xlarge (dedicated), CPython 3.12.16, PyYAML 6.0.3 with libyaml, 2026-11-14. Load average before: 1.07 (the benchmark's own process).

| Document | pyroid load | CSafeLoader | SafeLoader | pyroid dump | CSafeDumper | SafeDumper |
|---|---:|---:|---:|---:|---:|---:|
| 5 bytes | 3.4 µs | 8.0 µs | 48.7 µs | 4.6 µs | 8.4 µs | 31.9 µs |
| 300 bytes | 18.6 µs | 105.0 µs | 842.7 µs | 8.0 µs | 30.9 µs | 131.3 µs |
| 2 KB workflow | 59.7 µs | 320.7 µs | 2913.4 µs | 22.5 µs | 111.7 µs | 517.7 µs |
