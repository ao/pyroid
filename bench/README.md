# pyroid benchmark harness

Per-stage microbenchmarks (pyperf) and end-to-end runs (hyperfine 2.0) for the FineWeb-style cleaning chain:
- `ftfy.fix_text`
- fastText LID
- the Gopher, C4 and FineWeb filters
- MinHash dedup

These rules are enforced in code:
1. Equivalence comes before timing. Every timed row first passes a gate in `gates.py` against the pinned references in `$PYROID_DATA/parity` (see `../parity`); a mismatch aborts the run.
   - ftfy: byte-identical.
   - LID: same top-k labels, |Δp| ≤ 1e-6.
   - Filters: same keep/drop, reason and rewritten text.
   - MinHash: datatrove signatures identical. datasketch and rensa are gated on near-duplicate precision/recall ≥ 0.90 and mean |Ĵ−J| ≤ 0.04 against exact Jaccard.
2. The baseline is 8 processes. The end-to-end run uses datatrove's own `LocalPipelineExecutor` with tasks = workers = N, as in its `examples/fineweb.py`. The 8-process run is pinned to 8 CPUs with `taskset -c 0-7`. A 1-process run is reported alongside.
3. CPU-seconds and peak RSS are always recorded.
   - CPU: hyperfine `time_cpu` plus getrusage over self and children.
   - Memory: hyperfine's largest single process, the sum of per-worker maxrss (an upper bound on the simultaneous total), and the container's anon memory sampled every 0.1 s.
   - pyperf memory is a separate `--track-memory` pass, because it replaces the timing values.
4. The GIL guard aborts if an extension re-enables the GIL on a free-threaded interpreter (rensa 0.5.0 does this on 3.14t).
5. Charts:
   - Deterministic light and dark SVGs, fastest on top and never re-ordered by hand.
   - Candidates in the accent colour; overflow is labelled, never clipped.
   - Each chart's caption names the corpus, cores and versions.

## Run it

On Linux, or in a Linux container, from a checkout. The harness pins CPUs with `taskset` and reads its inputs from
`$PYROID_DATA` (default `/data`):

```bash
cd bench
uv run --frozen python fetch.py          # pinned inputs -> $PYROID_DATA
BENCH_GIT_SHA=$(git rev-parse HEAD) BENCH_E2E_CORPUS=dev BENCH_E2E_RUNS=3 BENCH_MEMORY_SIZES=100 \
  uv run --frozen bash run_baselines.sh
```

- Results go to `results/<machine>/<date>/`:
  - `machine.json`, `gates.json`
  - `stages/*.json` (pyperf) and `memory/*.json`
  - `e2e/hyperfine-*.json`, `e2e/runs.jsonl`, `e2e/equivalence.json`
  - `charts/*-{light,dark}.svg`
  - `SUMMARY.md`, built by `summarize.py`
- All the knobs are env vars documented at the top of `run_baselines.sh`. Pipeline scratch goes to `$BENCH_WORK` (default `/target/bench-work`).
- Use `BENCH_RIGOROUS=1` on a quiet, dedicated machine. The default is pyperf `--fast`, because on a shared VM the noise comes from other workloads rather than from too few samples.
- The full chain in one datatrove process costs about 40 ms/doc in the Docker VM; over the 100k headline slice, one such run took 2,444.4 s on an AWS c7i.4xlarge. Use the 10k dev slice (`BENCH_E2E_CORPUS=dev`) for quick runs and the headline slice on a dedicated machine.
- The published numbers come from AWS EC2: `results/aws-c7i.4xlarge/2026-11-06/` and `results/aws-c8g.4xlarge/2026-11-06/`, each with a `dev/` (10k) and a `headline/` (100k) consolidated run. [`ec2/README.md`](ec2/README.md) describes the scripts that ran them.

Individual pieces:

```bash
uv run --frozen python gates.py --workers 8 --out /target/gates.json            # all gates, dev + adversarial
uv run --frozen python stages.py --list                                          # rows + where the Python/Rust boundary sits
uv run --frozen python stages.py --stage fix_text --docs 1000 --fast -o /target/fix_text.json
uv run --frozen python pipeline.py --corpus dev --docs 500 --workers 8 --check-refs --out /target/pipe
uv run --frozen python chart.py e2e results/.../e2e/hyperfine-e2e.json --out /target/e2e --title "..."
uv run --frozen python chart.py hero $R/e2e/hyperfine-e2e.json $R/ft315t/hyperfine-chain.json --run $R --out /target/hero --png  # R: a consolidated run
```

## Plugging in a pyroid module

Each command takes `--impl SUITE=MODULE`, resolved through `pyroid_parity.impl`, so the contract is the same as in
`../parity/README.md`. Install the built wheel into the run with `uv run --frozen --with ../dist/pyroid-*.whl ...`.

- `stages.py --impl ftfy=pyroid.ftfy` adds a gated row next to the incumbents.
- `pipeline.py --impl ...` swaps a stage, and the digest must equal the incumbent's.
- `run_baselines.sh` reads `BENCH_IMPL="ftfy=pyroid.ftfy fasttext=pyroid.fasttext"`.

For thread scaling, pass `BENCH_THREADS` / `RAYON_NUM_THREADS`; workers inherit them.
- CPU pinning: pyperf pins its workers to the kernel's isolated CPUs when there are any. The OrbStack VM
  isolates CPU 11, which would run every thread of a multi-thread row on one core, so `stages.py` passes
  `--affinity=$BENCH_CPUS` (default `0-7`, the e2e set) whenever a row uses more than one thread.
- Candidate rows per stage (all gated): the parity-contract row; `[arrow]` (input as a pyarrow `string`
  array, built untimed); LID `batch` (`predict(list)`); `direct` (the module's own batch API, no adapter in the
  timed region); `MinHashDedup.fit`; `per doc` (the single-str call in a Python loop over `BENCH_THREADS`
  contiguous chunks). `translit` is a stages-only stage (anyascii 0.3.3 is its incumbent and live oracle).
- `BENCH_FRESH_STRS=1` gives every timed call new str objects (built untimed). CPython caches a non-ASCII str's
  UTF-8 on first use, so by default every call after the first sees warm caches.
- `BENCH_ROW_SUFFIX=" [3.15t, pool 8]"` is appended to candidate row names, so runs can share a chart.
- `--rows a,b,!c` keeps rows containing `a` or `b` and drops rows containing `c`.

## Performance work

| tool | what |
|---|---|
| `perf_matrix.py run/table/compare` | candidate-only A/B: every module's rows at pool sizes 1 and 8 (`--per-doc`: Python threads), gated first; `compare A B` flags changes of 5% or more |
| `chain.py` | the FineWeb chain natively in pyroid, one process, Arrow in, `threads=N`; `--digest --expect D` gates it on `pipeline.py`'s per-doc digest of the datatrove chain |
| `pgo_train.py`, `pgo_build.sh` | profile-guided release wheels: training on headline[10000:20000] (never the benchmarked dev slice), then `-Cprofile-use` |
| `run_consolidated.sh`, `consolidate.py` | the consolidated benchmark: `run_baselines.sh` with all five pyroid modules on 3.13, pyroid on 3.15t at pools 1 and 8, `chain.py` under hyperfine, then `CONSOLIDATED.md`, `consolidated.json` (docs' benchmarks.json row schema) and light/dark charts, among them the README hero (`charts/hero-{light,dark}.svg`; `--png` / `BENCH_PNG=1` add PNG previews). The dev slice by default; the 100k headline slice with `BENCH_E2E_CORPUS=headline BENCH_EQUIV_DOCS=100000 BENCH_EQUIV_WORKERS=8` (knobs at the top of the script). On AWS EC2, `machine_info.py` reads the instance type from DMI (results under `aws-<instance type>/`) and the labels name the instance, CPU and OS |

## json_repair (`json_repair_bench.py`)

This script stands alone: it needs only the stdlib and `json-repair==0.64.5`, not the bench uv env.
json_repair is the oracle and the baseline; every other `--impl` (such as `pyroid.json_repair`) is
optional and skipped with a warning if it does not import.

```bash
uv venv -p 3.12 /target/jr-312 && uv pip install --python /target/jr-312 json-repair==0.64.5 dist/pyroid-*.whl
/target/jr-312/bin/python bench/json_repair_bench.py --impl json_repair --impl pyroid.json_repair --out /target/jr-bench.json
/target/jr-312/bin/python bench/json_repair_bench.py --list        # the pathological families and their cause
/target/jr-312/bin/python bench/json_repair_bench.py --impl json_repair --impl json_repair --quick   # gate smoke test
```

- It gates first. For every input, a candidate's `repair_json(s)` must equal json_repair's byte for byte, and its
  `loads(s)` must be type-exactly equal, including dict key order. A mismatch aborts the run (exit 1).
- Sections (`--sections docs,patho,batch`):
  - `docs`: seeded ~1 KB and ~1 MB valid and broken LLM/API docs, `repair_json` and `loads`, best and median per call.
  - `patho`: the super-linear families (truncated long strings, escaped code, markdown in arrays, unescaped
    quotes, `(1)` list lines, missing `}, {`, continued table rows, four adversarial ones (two of them
    unclosed nesting, where every level rescans the rest of the input), and four linear controls) at doubling sizes from 2 KiB to 1 MiB, with the log-log slope. Each point runs in a subprocess:
    a call over `--timeout` (20 s) is killed, and an impl stops doubling after a call over `--stop-after` (10 s).
  - `batch`: 10k ~1 KB broken docs, json_repair in a loop, plus an 8-thread pool on free-threaded builds,
    vs the candidate's `repair_json_batch(docs, threads=1|8)`.
- Output: a Markdown summary on stdout and every timing in `--out`, with the machine info, `sys._is_gil_enabled()`
  and the load average before and after.

## FineWeb from WARC files (`fineweb_warc.py`)

datatrove 0.10.1's `examples/fineweb.py` from Common Crawl WARC files to the deduplicated, PII-replaced output:
WarcReader, URLFilter, Trafilatura(favour_precision=True), LanguageFilter(languages=["en"]), the Gopher, C4 and
FineWeb filters with fineweb.py's exclusion writers, then the four MinHash stages (sha1/64, 14 buckets x 8, 5-grams)
and PIIFormatter. It stands alone (not the bench uv env), in three modes:

```bash
python bench/fineweb_warc.py stock        --warcs W --out O/stock  --workers 8
python bench/fineweb_warc.py pyroid-steps --warcs W --out O/steps  --workers 8 --lid-model lid.176.bin
python bench/fineweb_warc.py native       --warcs W --out O/native --workers 8 --lid-model lid.176.bin
python bench/fineweb_warc.py compare O/stock O/steps O/native     # exit 1 unless the digests are identical
```

- `stock`: datatrove's own steps on its `LocalPipelineExecutor`, in an env with `datatrove[io,processing]==0.10.1`
  as pip resolves it (trafilatura 1.11.0, fasttext-numpy2-wheel) plus spaCy.
- `pyroid-steps`: the same executor with `pyroid.integrations.datatrove`'s drop-ins where pyroid has one; the same
  env plus the pyroid wheel.
- `native`: no executor. `pyroid.trafilatura.extract_batch(compat="1.11.0")`, `pyroid.filters` and the
  `FineWeb` recipe per WARC file in a process pool, then MinHash and PII over the files in order.
- Every mode runs one task per WARC file, so the output does not depend on `--workers`. `OUT/summary.json` has wall
  seconds per phase, CPU seconds, peak RSS, seconds and documents per step, and sha256 digests of the documents
  (sorted by id) after MinhashDedupFilter and after PIIFormatter.
- datatrove's Trafilatura step gives up on a page after 1 s and restarts its child process, which empties the dedup
  cache, so `stock` can lose pages on a slow or loaded machine. It records them in `OUT/events.jsonl`;
  `native --timeouts-of O/stock` drops the same pages the same way, for an exact comparison with that run.
- fasttext-numpy2-wheel 0.9.2 has no cp313 wheels, so it needs g++. Without one, `stock --lid-model` swaps in
  fasttext-predict with `gates.py`'s shim. The EC2 run (`ec2/run_warc.sh`) uses datatrove's own LanguageFilter.

## Inputs (pinned in `manifest.json`, fetched by `fetch.py` into `$PYROID_DATA`, ≤ 1.5 GB total; currently 324 MB)

| file | source |
|---|---|
| `fineweb/fineweb-100k.parquet` | headline slice: first 100,000 rows of `HuggingFaceFW/fineweb@9bb295ddab0e05d785b879661af7260fed5140fc` `sample/10BT/000_00000.parquet`. Content sha256 `1a933782…` |
| `fineweb/fineweb-dev-10k.parquet` | dev slice: its first 10,000 rows |
| `fineweb2/<lang>.parquet` | first 250 rows of the FineWeb-2 test split (`af9c1333…`) for fra, deu, spa, rus, cmn, jpn, arb, hin |
| `models/lid.176.{bin,ftz}` | fastText language ID (`dl.fbaipublicfiles.com`), CC BY-SA 3.0 |
| `parity/*.parquet` | adversarial set and reference outputs from `python -m pyroid_parity.refs all --pin` |

- `fetch.py --verify` re-hashes the inputs; `fetch.py --verify-all` also re-hashes the generated `parity/` references.
- `content_sha256` is computed over `id\0text\0…`, so it stays the same however the parquet file is written.
- FineWeb and FineWeb-2 are released under ODC-By 1.0: © their authors, as distributed by Hugging Face.

## Caveats
- datatrove 0.10.1's `LanguageFilter` doesn't run unmodified with fasttext-predict. It requires the `fasttext-numpy2-wheel` and `fasteners` distributions, downloads the model by URL, and calls `.item()` on plain floats.
  - `gates.PinnedLanguageFilter` keeps datatrove's `filter()`/`predict()` code, loads the pinned `.bin`, and returns numpy probabilities.
  - It matches fasttext-ref with 0 mismatches.
- datatrove's English tokenizer is spaCy `blank("en")`. Per-stage filter numbers include tokenization; in the chained pipeline datatrove's `lru_cache(5)` on `split_into_words` shares it across filters.
- Every run is labelled with where it ran. Runs in the Docker/OrbStack VM on Apple Silicon are labelled "Docker/OrbStack VM, indicative". The AWS EC2 runs are VMs on real hardware, not bare metal, and are labelled with the instance type, CPU and OS.
