#!/usr/bin/env bash
# Reproducible baseline runner (R2 section 5). Run it INSIDE ONE dx.sh call, from the worktree root:
#
#   DX_NAME=harness .orchestration/dx.sh 'cd bench && UV_PROJECT_ENVIRONMENT=/target/venv-bench \
#       BENCH_GIT_SHA=<sha from the host> uv run --frozen bash run_baselines.sh'
#
# Order (equivalence before timing): machine.json -> gates.py (all pinned refs) -> pyperf stages (each gated
# on its timed docs) -> pyperf --track-memory pass -> pipeline equivalence (one stage at a time vs the refs,
# then the full chain at every worker count: same digest + check-refs) -> hyperfine e2e -> hyperfine per stage
# at 8 workers -> machine.json "after" -> charts -> SUMMARY.md.
#
# Output: $BENCH_OUT (default bench/results/<machine>/<UTC date>/): run-config.json, machine.json, gates.json,
#   stages/<stage>-d<N>.json (+ .gates.json), memory/<stage>-d<N>.json, e2e/hyperfine-e2e.json,
#   e2e/hyperfine-stages-w8.json, e2e/runs.jsonl, e2e/equivalence.{jsonl,json}, charts/*-{light,dark}.svg,
#   SUMMARY.md. Pipeline scratch goes to $BENCH_WORK (default /target/bench-work), never into results.
#
# pyperf mode: --fast by default (10 processes x 2 values + 1 warmup each). In this shared VM the dominant
# noise is other sessions, which more processes do not remove, while every value of a slow datatrove stage
# costs a full batch (about 10 ms/doc); 20 values give a usable median + IQR. BENCH_RIGOROUS=1 -> --rigorous
# (40 processes) for a quiet bare-metal run.
#
# Knobs (env, defaults in brackets):
#   BENCH_OUT, BENCH_WORK, BENCH_MACHINE [machine_info --slug], BENCH_DATE [UTC today], BENCH_OVERWRITE [0]
#   BENCH_PYTHON [python on PATH], BENCH_HYPERFINE [hyperfine on PATH], BENCH_CPUS [0-7] (taskset for e2e)
#   BENCH_CORPUS [dev]           corpus for pyperf stages
#   BENCH_FAST_STAGES [fix_text lid], BENCH_SIZES_FAST [1 100 1000 10000]
#   BENCH_SLOW_STAGES [gopher_repetition gopher_quality c4_quality c4_quality@fineweb fineweb_quality minhash]
#   BENCH_SIZES_SLOW [1 100 1000]
#   BENCH_PYPERF [--fast | --rigorous if BENCH_RIGOROUS=1], BENCH_MEMORY [1], BENCH_MEMORY_SIZES [largest size]
#   BENCH_GATE_LIMIT [0 = all docs], BENCH_GATE_WORKERS [8]
#   BENCH_E2E_CORPUS [headline], BENCH_E2E_DOCS [0 = all], BENCH_E2E_WORKERS [1 8], BENCH_E2E_RUNS [5],
#   BENCH_E2E_WARMUP [1], BENCH_E2E_STAGES [all seven stages; "" skips per-stage e2e], BENCH_E2E_STAGE_WORKERS [8]
#   BENCH_EQUIV_DOCS [10000] (full-chain digest runs; the headline's first 10k = dev, so check-refs covers them)
#   BENCH_EQUIV_STAGE_CORPUS [parity] (one-stage runs vs refs: dev + adversarial, 13,232 docs)
#   BENCH_IMPL [""]  space-separated SUITE=MODULE candidates: extra pyperf rows + extra e2e commands
#                    (translit=MODULE only adds pyperf rows: gates.py and pipeline.py have no translit stage)
#   BENCH_IMPL_NAME [candidate]  hyperfine name prefix of the candidate e2e commands (charts accent "pyroid*"/"candidate*")
#   BENCH_SKIP [""]  comma list of steps: gates,stages,memory,equiv,e2e,e2e_stages,charts
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
PY=${BENCH_PYTHON:-$(command -v python)}
HF=${BENCH_HYPERFINE:-$(command -v hyperfine || true)}

err=$("$PY" -c 'import regex, datatrove.pipeline.filters, datatrove.pipeline.dedup, fasttext, ftfy, pyperf, datasketch, rensa, matplotlib' 2>&1) || {
    echo "preflight: the bench env cannot import what the harness needs: ${err##*$'\n'}" >&2
    echo "  (run inside the bench uv env: UV_PROJECT_ENVIRONMENT=/target/venv-bench uv run --frozen bash run_baselines.sh)" >&2
    exit 2
}

MACHINE=${BENCH_MACHINE:-$("$PY" machine_info.py --slug)}
DATE=${BENCH_DATE:-$(date -u +%Y-%m-%d)}
OUT=${BENCH_OUT:-$HERE/results/$MACHINE/$DATE}
WORK=${BENCH_WORK:-/target/bench-work}
CPUS=${BENCH_CPUS:-0-7}
CORPUS=${BENCH_CORPUS:-dev}
FAST_STAGES=${BENCH_FAST_STAGES-fix_text lid}
SLOW_STAGES=${BENCH_SLOW_STAGES-gopher_repetition gopher_quality c4_quality c4_quality@fineweb fineweb_quality minhash}
SIZES_FAST=${BENCH_SIZES_FAST:-1 100 1000 10000}
SIZES_SLOW=${BENCH_SIZES_SLOW:-1 100 1000}
if [[ "${BENCH_RIGOROUS:-0}" == 1 ]]; then PYPERF=${BENCH_PYPERF:---rigorous}; else PYPERF=${BENCH_PYPERF:---fast}; fi
MEMORY=${BENCH_MEMORY:-1}
GATE_LIMIT=${BENCH_GATE_LIMIT:-0}
GATE_WORKERS=${BENCH_GATE_WORKERS:-8}
E2E_CORPUS=${BENCH_E2E_CORPUS:-headline}
E2E_DOCS=${BENCH_E2E_DOCS:-0}
E2E_WORKERS=${BENCH_E2E_WORKERS:-1 8}
E2E_RUNS=${BENCH_E2E_RUNS:-5}
E2E_WARMUP=${BENCH_E2E_WARMUP:-1}
E2E_STAGES=${BENCH_E2E_STAGES-fix_text lid gopher_repetition gopher_quality c4_quality fineweb_quality minhash}
E2E_STAGE_WORKERS=${BENCH_E2E_STAGE_WORKERS:-8}
EQUIV_DOCS=${BENCH_EQUIV_DOCS:-10000}
EQUIV_STAGE_CORPUS=${BENCH_EQUIV_STAGE_CORPUS:-parity}
IMPL=${BENCH_IMPL:-}
CAND=${BENCH_IMPL_NAME:-candidate}  # e2e command name for the BENCH_IMPL candidate (e.g. pyroid-0.8.0)
SKIP=${BENCH_SKIP:-}

skip() { [[ ",$SKIP," == *",$1,"* ]]; }
log() { echo "[run_baselines $(date -u +%H:%M:%S)] $*"; }
pin() { if command -v taskset >/dev/null; then taskset -c "$CPUS" "$@"; else "$@"; fi; }
impl_args=()   # stages.py: every candidate
pipe_args=()   # gates.py / pipeline.py: only the suites they know (translit is a stages.py-only stage)
for s in $IMPL; do
    impl_args+=(--impl "$s")
    case "$s" in translit=*|*.translit) ;; *) pipe_args+=(--impl "$s") ;; esac
done

if [[ -d "$OUT" && -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
    if [[ "${BENCH_OVERWRITE:-0}" == 1 ]]; then rm -rf "$OUT"; else
        echo "$OUT exists and is not empty (set BENCH_OVERWRITE=1 or another BENCH_OUT)" >&2; exit 2; fi
fi
mkdir -p "$OUT"/{stages,memory,e2e,charts} "$WORK"
DTV=$("$PY" -c 'import importlib.metadata as m; print(m.version("datatrove"))')
if [[ "$E2E_DOCS" == 0 ]]; then E2E_N=$("$PY" -c "import gates; print(gates.corpus_len('$E2E_CORPUS'))"); else E2E_N=$E2E_DOCS; fi
EQUIV_N=$(( EQUIV_DOCS < E2E_N ? EQUIV_DOCS : E2E_N ))

"$PY" - "$OUT/run-config.json" <<EOF
import json, os, sys
cfg = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("BENCH_", "RAYON_", "PYROID_"))}
cfg["resolved"] = {"python": "$PY", "hyperfine": "$HF", "machine": "$MACHINE", "date": "$DATE", "out": "$OUT",
    "work": "$WORK", "cpus": "$CPUS", "corpus": "$CORPUS", "fast_stages": "$FAST_STAGES", "slow_stages": "$SLOW_STAGES",
    "sizes_fast": "$SIZES_FAST", "sizes_slow": "$SIZES_SLOW", "pyperf": "$PYPERF", "memory": "$MEMORY",
    "gate_limit": "$GATE_LIMIT", "e2e_corpus": "$E2E_CORPUS", "e2e_docs": $E2E_N, "e2e_workers": "$E2E_WORKERS",
    "e2e_runs": "$E2E_RUNS", "e2e_warmup": "$E2E_WARMUP", "e2e_stages": "$E2E_STAGES",
    "e2e_stage_workers": "$E2E_STAGE_WORKERS", "equiv_docs": $EQUIV_N, "impl": "$IMPL", "skip": "$SKIP"}
open(sys.argv[1], "w").write(json.dumps(cfg, indent=1) + "\n")
EOF

log "machine info -> $OUT/machine.json"
"$PY" machine_info.py --out "$OUT/machine.json"

if ! skip gates; then
    log "gates (all pinned refs, $GATE_WORKERS workers)"
    "$PY" gates.py --workers "$GATE_WORKERS" --limit "$GATE_LIMIT" "${pipe_args[@]}" --out "$OUT/gates.json"
fi

sizes_for() { if [[ " $FAST_STAGES " == *" $1 "* ]]; then echo "$SIZES_FAST"; else echo "$SIZES_SLOW"; fi; }
largest() { local m=0; for n in $1; do (( n > m )) && m=$n; done; echo "$m"; }

if ! skip stages; then
    for st in $FAST_STAGES $SLOW_STAGES; do
        for n in $(sizes_for "$st"); do
            log "pyperf $PYPERF: $st, $n docs/call"
            "$PY" stages.py --stage "$st" --docs "$n" --corpus "$CORPUS" --gate-workers "$GATE_WORKERS" "${impl_args[@]}" \
                $PYPERF -o "$OUT/stages/$st-d$n.json"
        done
    done
fi

if ! skip memory && [[ "$MEMORY" == 1 ]]; then
    for st in $FAST_STAGES $SLOW_STAGES; do
        for n in ${BENCH_MEMORY_SIZES:-$(largest "$(sizes_for "$st")")}; do
            gate="$OUT/stages/$st-d$n.gates.json"
            reuse=(); [[ -f "$gate" ]] && reuse=(--reuse-gates "$gate")
            log "pyperf --track-memory: $st, $n docs/call"
            "$PY" stages.py --stage "$st" --docs "$n" --corpus "$CORPUS" --gate-workers "$GATE_WORKERS" "${impl_args[@]}" \
                "${reuse[@]}" $PYPERF --track-memory -o "$OUT/memory/$st-d$n.json"
        done
    done
fi

EQ="$OUT/e2e/equivalence.jsonl"
run_pipe() {  # label, then pipeline args; the summary goes to $EQ
    local label=$1; shift
    pin "$PY" pipeline.py "$@" --quiet --label "$label" --summary-log "$EQ" --out "$WORK/$label" >/dev/null
}
if ! skip equiv; then
    for s in $E2E_STAGES; do
        log "equivalence: --stages $s on $EQUIV_STAGE_CORPUS at $E2E_STAGE_WORKERS workers vs pinned refs"
        run_pipe "refs-$s" --stages "$s" --corpus "$EQUIV_STAGE_CORPUS" --workers "$E2E_STAGE_WORKERS" --check-refs
    done
    labels=()
    for w in $E2E_WORKERS; do
        log "equivalence: full chain, $E2E_CORPUS[:$EQUIV_N], $w workers (digest + check-refs)"
        run_pipe "digest-w$w" --corpus "$E2E_CORPUS" --docs "$EQUIV_N" --workers "$w" --check-refs
        labels+=("digest-w$w")
        if [[ -n "$IMPL" ]]; then
            run_pipe "digest-candidate-w$w" --corpus "$E2E_CORPUS" --docs "$EQUIV_N" --workers "$w" --check-refs "${pipe_args[@]}"
            labels+=("digest-candidate-w$w")
        fi
    done
    "$PY" - "$EQ" "$OUT/e2e/equivalence.json" "${labels[@]}" <<'EOF'
import json, sys
log, out, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
runs = {r["label"]: r for r in map(json.loads, open(log, encoding="utf-8")) if r["label"] in labels}
digests = {lb: runs[lb]["digest"] for lb in labels}
ok = len(set(digests.values())) == 1
json.dump({"ok": ok, "labels": labels, "digests": digests}, open(out, "w"), indent=1)
print(("digests identical: " if ok else "DIGEST MISMATCH: ") + json.dumps(digests))
sys.exit(0 if ok else 1)
EOF
fi

hf() {  # export file, then hyperfine -n/command pairs
    local export=$1; shift
    [[ -n "$HF" ]] || { echo "hyperfine not found (set BENCH_HYPERFINE)" >&2; exit 2; }
    pin "$HF" -N --warmup "$E2E_WARMUP" --runs "$E2E_RUNS" \
        --metrics time_wall_clock,time_cpu,memory_peak_resident --export-json "$export" "$@"
}
PIPE="$PY $HERE/pipeline.py --corpus $E2E_CORPUS --docs $E2E_N --no-digest --quiet --summary-log $OUT/e2e/runs.jsonl"
if ! skip e2e; then
    cmds=()
    for w in $E2E_WORKERS; do
        cmds+=(-n "datatrove-$DTV-w$w" "$PIPE --workers $w --label datatrove-$DTV-w$w --out $WORK/e2e-w$w")
        if [[ -n "$IMPL" ]]; then
            cmds+=(-n "$CAND-w$w" "$PIPE --workers $w --label $CAND-w$w --out $WORK/e2e-cand-w$w ${pipe_args[*]}")
        fi
    done
    log "hyperfine e2e: $E2E_CORPUS, $E2E_N docs, workers {$E2E_WORKERS}, $E2E_RUNS runs (+$E2E_WARMUP warmup), taskset $CPUS"
    hf "$OUT/e2e/hyperfine-e2e.json" "${cmds[@]}"
fi

if ! skip e2e_stages && [[ -n "$E2E_STAGES" ]]; then
    cmds=()
    for s in $E2E_STAGES; do
        w=$E2E_STAGE_WORKERS
        cmds+=(-n "$s-w$w" "$PIPE --stages $s --workers $w --label $s-w$w --out $WORK/stage-$s")
    done
    log "hyperfine per stage at $E2E_STAGE_WORKERS workers: $E2E_STAGES"
    hf "$OUT/e2e/hyperfine-stages-w$E2E_STAGE_WORKERS.json" "${cmds[@]}"
fi

"$PY" machine_info.py --update "$OUT/machine.json"

if ! skip charts; then
    log "charts"
    cap="$E2E_CORPUS ($E2E_N docs), datatrove $DTV, taskset $CPUS; median of $E2E_RUNS runs, IQR whiskers"
    [[ -f "$OUT/e2e/hyperfine-e2e.json" ]] && "$PY" chart.py e2e "$OUT/e2e/hyperfine-e2e.json" --out "$OUT/charts/e2e" \
        --title "end to end: $cap"
    for f in "$OUT"/e2e/hyperfine-stages-w*.json; do
        [[ -f "$f" ]] && "$PY" chart.py e2e "$f" --out "$OUT/charts/$(basename "$f" .json)" --title "one stage per run: $cap"
    done
    shopt -s nullglob
    stage_files=("$OUT"/stages/*-d*[0-9].json)
    if (( ${#stage_files[@]} )); then
        "$PY" chart.py stages "${stage_files[@]}" --out "$OUT/charts/stages" \
            --title "per-doc time at the largest batch per stage ($CORPUS corpus, pyperf $PYPERF, 1 process)"
        for n in $(echo "$SIZES_FAST $SIZES_SLOW" | tr ' ' '\n' | sort -nu); do
            "$PY" chart.py stages "${stage_files[@]}" --docs "$n" --out "$OUT/charts/stages-d$n" \
                --title "per-doc time, batches of $n docs where measured ($CORPUS corpus, pyperf $PYPERF)"
        done
        "$PY" chart.py sweep "${stage_files[@]}" --out "$OUT/charts/sweep" --cols 3
    fi
fi

"$PY" summarize.py "$OUT" --out "$OUT/SUMMARY.md" >/dev/null
log "done: $OUT (SUMMARY.md, charts/)"
