# json_repair 0.64.9 vs pyroid.json_repair, 2026-11-14

EC2 c8g.16xlarge devbox (Graviton4, 64 vCPU), shared with other work, so indicative. Release
wheels of pyroid 1.0.0 with the json_repair 0.64.9 port (commit "feat(json_repair): match
json-repair 0.64.9"), `bench/json_repair_bench.py`:

| File | Run | Load average |
|---|---|---|
| `json_repair-0.64.9-3.12.json`, `.md` | `--sections docs,patho`, CPython 3.12 | 0.1 before, 4.7 after |
| `json_repair-0.64.9-3.15t-batch.json` | `--sections batch`, free-threaded CPython 3.15t | 30 to 40 |
| `json_repair-0.64.5-3.12-docs.json` | json_repair 0.64.5 alone, `--sections docs`, CPython 3.12 | 5 to 6 |

Every pyroid result was gated against json_repair 0.64.9's before it was timed. The files were
written before the script took its label from `PYROID_BENCH_LABEL`; their label was then set by
hand to this machine (it said "Docker/OrbStack VM").
