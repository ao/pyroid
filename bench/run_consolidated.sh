#!/usr/bin/env bash
# The consolidated benchmark: every incumbent and every pyroid module, gated, in one results folder.
# Run INSIDE ONE dx.sh call from the worktree root, with the pyroid wheel installed in all three envs:
#
#   DX_NAME=perf .orchestration/dx.sh 'BENCH_GIT_SHA=<sha> bash bench/run_consolidated.sh'
#
# 1. run_baselines.sh in the oracle env (CPython 3.13, $BENCH_PY313): gates, pyperf stages (incumbents and
#    pyroid at RAYON_NUM_THREADS=1), memory, pipeline equivalence, hyperfine e2e at 1 and 8 processes for
#    datatrove and for datatrove's executor running pyroid's stages, one-stage e2e at 8 processes.
# 2. pyroid alone on free-threaded CPython ($BENCH_PY315T): the same gated stage rows at pools of 1 and 8
#    (RAYON_NUM_THREADS), suffixed " [3.15t, pool N]" so charts and tables keep them apart.
# 3. chain.py: the whole chain natively in pyroid on 3.15t at threads=1 and 8; its digest must equal the
#    datatrove chain's from step 1, then hyperfine times it (taskset $BENCH_CPUS, like the e2e).
# 4. consolidate.py: CONSOLIDATED.md, consolidated.json and the light/dark charts, among them the README hero
#    (charts/hero-{light,dark}.svg; copy it to docs/assets/).
#
# Knobs: BENCH_OUT [results/<machine>/<date>-consolidated], BENCH_E2E_RUNS [3], BENCH_CHAIN_RUNS [5],
#   BENCH_PY313 [/target/venv-bench/bin/python], BENCH_PY315T [/target/venv-b315t/bin/python],
#   BENCH_SKIP_BASELINES=1 (reuse step 1 in BENCH_OUT), BENCH_PYPERF_FT [-p 5 -n 3 -w 1],
#   BENCH_PNG=1 (PNG previews next to the charts).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
PY313=${BENCH_PY313:-/target/venv-bench/bin/python}
PY315T=${BENCH_PY315T:-/target/venv-b315t/bin/python}
HF=${BENCH_HYPERFINE:-$(dirname "$PY313")/hyperfine}
MACHINE=${BENCH_MACHINE:-$("$PY313" machine_info.py --slug)}
OUT=${BENCH_OUT:-$HERE/results/$MACHINE/$(date -u +%Y-%m-%d)-consolidated}
CPUS=${BENCH_CPUS:-0-7}
FT_PYPERF=${BENCH_PYPERF_FT:--p 5 -n 3 -w 1}
export PYROID_MANIFEST=${PYROID_MANIFEST:-$HERE/manifest.json}
log() { echo "[run_consolidated $(date -u +%H:%M:%S)] $*"; }
IMPLS="ftfy=pyroid.ftfy fasttext=pyroid.fasttext filters=pyroid.filters dedup=pyroid.dedup translit=pyroid.translit"

if [[ "${BENCH_SKIP_BASELINES:-0}" != 1 ]]; then
    log "1/4 run_baselines.sh on $("$PY313" -V) -> $OUT"
    BENCH_OUT=$OUT BENCH_PYTHON=$PY313 BENCH_HYPERFINE=$HF BENCH_IMPL="$IMPLS" BENCH_IMPL_NAME=pyroid \
        BENCH_FAST_STAGES="fix_text lid translit" BENCH_E2E_CORPUS=dev BENCH_E2E_RUNS=${BENCH_E2E_RUNS:-3} \
        BENCH_MEMORY_SIZES=1000 RAYON_NUM_THREADS=1 bash run_baselines.sh
fi

FT=$OUT/ft315t
mkdir -p "$FT/stages"
log "2/4 pyroid stage rows on $("$PY315T" -V) at pools 1 and 8"
# Batch rows only (list and Arrow; no parity adapters, no per-doc loops). Sizes match the incumbents'
# largest batch per stage (10k fast, 1k slow) for the charts, plus 10k everywhere for the tables.
rows_for() {
    case $1 in
        fix_text) echo "fix_text_batch" ;;
        lid) echo "k=1 batch,k=5 batch" ;;
        minhash) echo "direct" ;;
        translit) echo "pyroid.translit,!per doc" ;;
        *) echo "direct" ;;
    esac
}
for st in fix_text lid translit gopher_repetition gopher_quality c4_quality c4_quality@fineweb fineweb_quality minhash; do
    case $st in fix_text|lid|translit) sizes="10000" ;; *) sizes="1000 10000" ;; esac
    for n in $sizes; do
        for pool in 1 8; do
            f=$FT/stages/$st-d$n-p$pool.json
            rm -f "$f"
            log "  $st, $n docs, pool $pool"
            RAYON_NUM_THREADS=$pool BENCH_ROW_SUFFIX=" [3.15t, pool $pool]" "$PY315T" stages.py --stage "$st" \
                --docs "$n" --rows "$(rows_for "$st")" $(for s in $IMPLS; do echo "--impl $s"; done) \
                $FT_PYPERF --quiet -o "$f"
        done
    done
done

log "3/4 chain.py on 3.15t: digest gate, then hyperfine"
EXPECT=$("$PY313" -c "import json,sys; d=json.load(open(sys.argv[1]))['digests']; print(d['digest-w1'])" "$OUT/e2e/equivalence.json")
"$PY315T" chain.py --corpus dev --threads 8 --expect "$EXPECT" --label "pyroid chain 3.15t threads=8 (digest gate)" \
    --summary-log "$FT/chain-runs.jsonl" > "$FT/chain-digest.json"
pin() { if command -v taskset >/dev/null; then taskset -c "$CPUS" "$@"; else "$@"; fi; }
pin "$HF" -N --warmup 1 --runs "${BENCH_CHAIN_RUNS:-5}" \
    --metrics time_wall_clock,time_cpu,memory_peak_resident --export-json "$FT/hyperfine-chain.json" \
    -n "pyroid chain 3.15t threads=1" "$PY315T $HERE/chain.py --corpus dev --threads 1 --summary-log $FT/chain-runs.jsonl" \
    -n "pyroid chain 3.15t threads=8" "$PY315T $HERE/chain.py --corpus dev --threads 8 --summary-log $FT/chain-runs.jsonl"

log "4/4 consolidate.py"
"$PY313" consolidate.py "$OUT" $([[ "${BENCH_PNG:-0}" == 1 ]] && echo --png)
log "done: $OUT/CONSOLIDATED.md"
