# pyroid benchmark harness

Per-stage microbenchmarks (pyperf) and end-to-end runs (hyperfine 2.0) for the FineWeb-style cleaning chain:
- `ftfy.fix_text`
- fastText LID
- the Gopher, C4 and FineWeb filters
- MinHash dedup

These rules follow R2 §5 and are enforced in code:
1. **Equivalence before timing.** Every timed row first passes a gate in `gates.py` against the pinned references in `/data/parity` (see `../parity`); a mismatch aborts the run.
   - ftfy: byte-identical.
   - LID: same top-k labels, |Δp| ≤ 1e-6.
   - Filters: same keep/drop, reason and rewritten text.
   - MinHash: datatrove signatures identical. datasketch and rensa are gated on near-duplicate precision/recall ≥ 0.90 and mean |Ĵ−J| ≤ 0.04 against exact Jaccard.
2. **The baseline is 8 processes.** The end-to-end runs datatrove's own `LocalPipelineExecutor` with tasks = workers = N, as in its `examples/fineweb.py`. The 8-process run is pinned to 8 CPUs with `taskset -c 0-7`. A 1-process run is reported alongside.
3. **CPU-seconds and peak RSS are always recorded.**
   - CPU: hyperfine `time_cpu` plus getrusage over self and children.
   - Memory: hyperfine's largest single process, the sum of per-worker maxrss (an upper bound on the simultaneous total), and the container's anon memory sampled every 0.1 s.
   - pyperf memory is a separate `--track-memory` pass, because it replaces the timing values.
4. **The GIL guard aborts** if an extension re-enables the GIL on a free-threaded interpreter (rensa 0.5.0 does this on 3.14t).
5. **Charts:**
   - Deterministic light and dark SVGs, fastest on top and never re-ordered by hand.
   - Candidates in the accent colour; overflow is labelled, never clipped.
   - Each chart's caption names the corpus, cores and versions.

## Run it (Docker only; never on the macOS host)

```bash
cd /Users/devel/src/ao/pyroid/.claude/worktrees/harness          # or any checkout of this branch
SHA=$(git rev-parse HEAD)
DX_NAME=harness /Users/devel/src/ao/pyroid/.orchestration/dx.sh "cd bench && \
  UV_PROJECT_ENVIRONMENT=/target/venv-bench uv run --frozen python fetch.py"          # pinned inputs -> /data
DX_NAME=harness /Users/devel/src/ao/pyroid/.orchestration/dx.sh "cd bench && \
  UV_PROJECT_ENVIRONMENT=/target/venv-bench BENCH_GIT_SHA=$SHA BENCH_E2E_CORPUS=dev BENCH_E2E_RUNS=3 \
  BENCH_MEMORY_SIZES=100 uv run --frozen bash run_baselines.sh"
```

- Results go to `results/<machine>/<date>/`:
  - `machine.json`, `gates.json`
  - `stages/*.json` (pyperf) and `memory/*.json`
  - `e2e/hyperfine-*.json`, `e2e/runs.jsonl`, `e2e/equivalence.json`
  - `charts/*-{light,dark}.svg`
  - `SUMMARY.md`, built by `summarize.py`
- All the knobs are env vars documented at the top of `run_baselines.sh`.
- Use `BENCH_RIGOROUS=1` on a quiet bare-metal box. The default is pyperf `--fast`, because in the shared VM the noise comes from other sessions rather than from too few samples.
- The full chain at one process costs about 40 ms/doc. That is why the default e2e corpus for VM runs is the 10k dev slice (`BENCH_E2E_CORPUS=dev`); the 100k headline slice is for bare metal.

Individual pieces:

```bash
uv run --frozen python gates.py --workers 8 --out /target/gates.json            # all gates, dev + adversarial
uv run --frozen python stages.py --list                                          # rows + where the Python/Rust boundary sits
uv run --frozen python stages.py --stage fix_text --docs 1000 --fast -o /target/fix_text.json
uv run --frozen python pipeline.py --corpus dev --docs 500 --workers 8 --check-refs --out /target/pipe
uv run --frozen python chart.py e2e results/.../e2e/hyperfine-e2e.json --out /target/e2e --title "..."
```

## Plugging in a pyroid module

Each command takes `--impl SUITE=MODULE`, resolved through `pyroid_parity.impl`, so the contract is the same as in
`../parity/README.md`. Install the built wheel into the run with `uv run --frozen --with ../dist/pyroid-*.whl ...`.

- `stages.py --impl ftfy=pyroid.ftfy` adds a gated row next to the incumbents.
- `pipeline.py --impl ...` swaps a stage, and the digest must equal the incumbent's.
- `run_baselines.sh` reads `BENCH_IMPL="ftfy=pyroid.ftfy fasttext=pyroid.fasttext"`.

Thread scaling: pass `BENCH_THREADS` / `RAYON_NUM_THREADS`; workers inherit them.

## Inputs (pinned in `manifest.json`, fetched by `fetch.py` into `/data`, ≤ 1.5 GB total; currently 324 MB)

| file | source |
|---|---|
| `fineweb/fineweb-100k.parquet` | headline slice: first 100,000 rows of `HuggingFaceFW/fineweb@9bb295ddab0e05d785b879661af7260fed5140fc` `sample/10BT/000_00000.parquet`. Content sha256 `1a933782…` reproduces R2's pin |
| `fineweb/fineweb-dev-10k.parquet` | dev slice: its first 10,000 rows |
| `fineweb2/<lang>.parquet` | first 250 rows of the FineWeb-2 test split (`af9c1333…`) for fra, deu, spa, rus, cmn, jpn, arb, hin |
| `models/lid.176.{bin,ftz}` | fastText language ID (`dl.fbaipublicfiles.com`), CC BY-SA 3.0 |
| `parity/*.parquet` | adversarial set and reference outputs from `python -m pyroid_parity.refs all --pin` |

- `fetch.py --verify` re-hashes everything.
- `content_sha256` is computed over `id\0text\0…`, so it stays the same however the parquet file is written.
- FineWeb and FineWeb-2 are released under ODC-By 1.0: © their authors, as distributed by Hugging Face.

## Caveats
- **datatrove 0.10.1 + fasttext-predict:** `LanguageFilter` doesn't run unmodified. It requires the `fasttext-numpy2-wheel` and `fasteners` distributions, downloads the model by URL, and calls `.item()` on plain floats.
  - `gates.PinnedLanguageFilter` keeps datatrove's `filter()`/`predict()` code, loads the pinned `.bin`, and returns numpy probabilities.
  - It matches fasttext-ref with 0 mismatches.
- **datatrove's English tokenizer is spaCy `blank("en")`.** Per-stage filter numbers include tokenization; in the chained pipeline datatrove's `lru_cache(5)` on `split_into_words` shares it across filters.
- **VM numbers are indicative only.** Every number here is labelled "Docker/OrbStack VM, indicative". README numbers need the bare-metal run (PLAN Phase 3).
