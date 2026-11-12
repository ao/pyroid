#!/usr/bin/env bash
# The consolidated benchmark: every incumbent and every pyroid module, gated, in one results folder.
# Run it on Linux (or in one Linux container) from the repository root, with the pyroid wheel installed in all three
# envs ($BENCH_PY313, $BENCH_PY315T and the bench env):
#
#   BENCH_GIT_SHA=$(git rev-parse HEAD) bash bench/run_consolidated.sh
#
# 1. run_baselines.sh in the oracle env (CPython 3.13, $BENCH_PY313): gates, pyperf stages (incumbents and
#    pyroid at RAYON_NUM_THREADS=1), memory, pipeline equivalence, hyperfine e2e at 1 and 8 processes for
#    datatrove and for datatrove's executor running pyroid's stages, one-stage e2e at 8 processes.
# 2. pyroid alone on free-threaded CPython ($BENCH_PY315T): the same gated stage rows at pools of 1 and 8
#    (RAYON_NUM_THREADS), suffixed " [3.15t, pool N]" so charts and tables keep them apart.
# 3. chain.py: the whole chain natively in pyroid on 3.15t at threads=1 and 8; its digest must equal the
#    datatrove chain's from step 1 on the same docs, then hyperfine times it (taskset $BENCH_CPUS, like the e2e).
# 4. consolidate.py: CONSOLIDATED.md, consolidated.json and the light/dark charts, among them the README hero
#    (charts/hero-{light,dark}.svg; copy it to docs/assets/).
#
# Knobs: BENCH_OUT [results/<machine>/<date>-consolidated], BENCH_E2E_RUNS [3], BENCH_CHAIN_RUNS [5],
#   BENCH_CHAIN_WARMUP [1], BENCH_PY313 [/target/venv-bench/bin/python], BENCH_PY315T [/target/venv-b315t/bin/python],
#   BENCH_SKIP_BASELINES=1 (reuse step 1 in BENCH_OUT), BENCH_PYPERF_FT [-p 5 -n 3 -w 1],
#   BENCH_PNG=1 (PNG previews next to the charts).
#   BENCH_E2E_CORPUS [dev], BENCH_E2E_DOCS [0 = all]: the e2e corpus and docs of step 1; step 3's chain.py runs
#     use the ones step 1 recorded in run-config.json (resolved.e2e_corpus / e2e_docs), the digest gate the
#     docs the equivalence digests covered (resolved.equiv_docs).
#   BENCH_MEMORY_SIZES [1000] (step 1's --track-memory batch sizes).
#   BENCH_FT_SIZES_FAST [10000] (fix_text, lid, translit), BENCH_FT_SIZES_SLOW [1000 10000] (the filters and
#     minhash): docs per call of the 3.15t stage rows; BENCH_SKIP_FT_STAGES=1 skips step 2.
#   Every other run_baselines.sh knob (BENCH_SKIP, BENCH_SIZES_FAST/SLOW, BENCH_GATE_LIMIT, BENCH_E2E_WARMUP,
#   BENCH_E2E_STAGES, BENCH_EQUIV_DOCS, BENCH_EQUIV_WORKERS, ...) passes through from the environment; the
#   3.15t rows use run_baselines.sh's stage corpus, BENCH_CORPUS [dev].
#
# The hero calls pyroid's output identical on every e2e doc, so the full-chain digests must cover them all:
# BENCH_EQUIV_DOCS (run_baselines.sh default 10000) below the e2e docs stops the run before step 1. The 100k
# headline slice, end to end only (the stage rows stay with a dev run; a 1-worker datatrove digest of 100k
# docs would take most of an hour, so the digests run at 8 workers only):
#
#   BENCH_E2E_CORPUS=headline BENCH_EQUIV_DOCS=100000 BENCH_EQUIV_WORKERS=8 BENCH_SKIP=stages,memory \
#       BENCH_SKIP_FT_STAGES=1 bash bench/run_consolidated.sh
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
E2E_CORPUS=${BENCH_E2E_CORPUS:-dev}
FT_SIZES_FAST=${BENCH_FT_SIZES_FAST:-10000}
FT_SIZES_SLOW=${BENCH_FT_SIZES_SLOW:-1000 10000}
export PYROID_MANIFEST=${PYROID_MANIFEST:-$HERE/manifest.json}
log() { echo "[run_consolidated $(date -u +%H:%M:%S)] $*"; }
IMPLS="ftfy=pyroid.ftfy fasttext=pyroid.fasttext filters=pyroid.filters dedup=pyroid.dedup translit=pyroid.translit"

if [[ "${BENCH_SKIP_BASELINES:-0}" != 1 ]]; then
    e2e_n=${BENCH_E2E_DOCS:-0}
    if [[ "$e2e_n" == 0 ]]; then
        e2e_n=$("$PY313" -c "import gates; print(gates.corpus_len('$E2E_CORPUS'))")
    fi
    if (( ${BENCH_EQUIV_DOCS:-10000} < e2e_n )); then
        echo "BENCH_EQUIV_DOCS=${BENCH_EQUIV_DOCS:-10000} is below the $e2e_n e2e docs: the hero would refuse" \
            "an output-identical claim the digests do not cover (set BENCH_EQUIV_DOCS=$e2e_n)" >&2
        exit 2
    fi
    log "1/4 run_baselines.sh on $("$PY313" -V) -> $OUT"
    BENCH_OUT=$OUT BENCH_PYTHON=$PY313 BENCH_HYPERFINE=$HF BENCH_IMPL="$IMPLS" BENCH_IMPL_NAME=pyroid \
        BENCH_FAST_STAGES="fix_text lid translit" BENCH_E2E_CORPUS=$E2E_CORPUS BENCH_E2E_RUNS=${BENCH_E2E_RUNS:-3} \
        BENCH_MEMORY_SIZES=${BENCH_MEMORY_SIZES:-1000} RAYON_NUM_THREADS=1 bash run_baselines.sh
fi

# What step 1 ran (also when it is reused): the stage corpus, the e2e corpus and docs, the docs the
# equivalence digests covered, and the datatrove chain's digest (digest-w1, else the first datatrove one).
GATE=$("$PY313" -c '
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))["resolved"]
eq = json.load(open(sys.argv[2], encoding="utf-8"))
lb = "digest-w1" if "digest-w1" in eq["digests"] else next(x for x in eq["labels"] if not x.startswith("digest-candidate"))
print(cfg["corpus"], cfg["e2e_corpus"], cfg["e2e_docs"], cfg["equiv_docs"], eq["digests"][lb])
' "$OUT/run-config.json" "$OUT/e2e/equivalence.json")
read -r STAGE_CORPUS CHAIN_CORPUS CHAIN_DOCS EQUIV_N EXPECT <<< "$GATE"

FT=$OUT/ft315t
mkdir -p "$FT"
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
if [[ "${BENCH_SKIP_FT_STAGES:-0}" == 1 ]]; then
    log "2/4 skipped (BENCH_SKIP_FT_STAGES=1)"
else
    mkdir -p "$FT/stages"
    log "2/4 pyroid stage rows on $("$PY315T" -V) at pools 1 and 8"
    for st in fix_text lid translit gopher_repetition gopher_quality c4_quality c4_quality@fineweb fineweb_quality minhash; do
        case $st in fix_text|lid|translit) sizes=$FT_SIZES_FAST ;; *) sizes=$FT_SIZES_SLOW ;; esac
        for n in $sizes; do
            for pool in 1 8; do
                f=$FT/stages/$st-d$n-p$pool.json
                rm -f "$f"
                log "  $st, $n docs, pool $pool"
                RAYON_NUM_THREADS=$pool BENCH_ROW_SUFFIX=" [3.15t, pool $pool]" "$PY315T" stages.py --stage "$st" \
                    --corpus "$STAGE_CORPUS" --docs "$n" --rows "$(rows_for "$st")" \
                    $(for s in $IMPLS; do echo "--impl $s"; done) $FT_PYPERF --quiet -o "$f"
            done
        done
    done
fi

log "3/4 chain.py on 3.15t ($CHAIN_CORPUS, $CHAIN_DOCS docs): digest gate on $EQUIV_N docs, then hyperfine"
"$PY315T" chain.py --corpus "$CHAIN_CORPUS" --docs "$EQUIV_N" --threads 8 --expect "$EXPECT" \
    --label "pyroid chain 3.15t threads=8 (digest gate)" --summary-log "$FT/chain-runs.jsonl" > "$FT/chain-digest.json"
pin() { if command -v taskset >/dev/null; then taskset -c "$CPUS" "$@"; else "$@"; fi; }
CHAIN="$PY315T $HERE/chain.py --corpus $CHAIN_CORPUS --docs $CHAIN_DOCS"
pin "$HF" -N --warmup "${BENCH_CHAIN_WARMUP:-1}" --runs "${BENCH_CHAIN_RUNS:-5}" \
    --metrics time_wall_clock,time_cpu,memory_peak_resident --export-json "$FT/hyperfine-chain.json" \
    -n "pyroid chain 3.15t threads=1" "$CHAIN --threads 1 --summary-log $FT/chain-runs.jsonl" \
    -n "pyroid chain 3.15t threads=8" "$CHAIN --threads 8 --summary-log $FT/chain-runs.jsonl"

log "4/4 consolidate.py"
"$PY313" consolidate.py "$OUT" $([[ "${BENCH_PNG:-0}" == 1 ]] && echo --png)
log "done: $OUT/CONSOLIDATED.md"
