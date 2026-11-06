#!/usr/bin/env bash
# One pyroid benchmark + parity run on an EC2 instance, natively (no Docker), as root, after bootstrap.sh.
#
#   PYROID_S3=s3://<bucket>/ec2bench bash run.sh          # normally started by `ec2.sh start` under systemd-run
#
# Steps, in order (each leaves $STATE/<step>.done; a re-run skips finished steps, so the script is resumable):
#   meta            lscpu (-e, -J), /proc/cpuinfo, DMI, IMDSv2 identity, glibc hwcaps diagnostics, THP/SMT, versions
#   code            git clone of $PYROID_BRANCH [next] from the S3 bundle; $PYROID_OVERLAY_PATHS (the bench tooling)
#                   checked out from $PYROID_OVERLAY_REF [feat/ec2bench] when the bundle carries that branch
#   wheels          maturin release wheels, as ci.yml: cp311-abi3 (built with 3.12) and cp315-abi3.abi3t (with 3.15)
#   envs            bench env (3.13, bench/uv.lock), 3.15t bench env (as P4's venv-b315t), parity envs 3.13/3.12/3.15t
#   fetch           bench/fetch.py (pinned inputs, from the upstream URLs) + --verify
#   refs            python -m pyroid_parity.refs all, json_repair_record --check, then refdiff.py check: mismatches
#                   against the pins are quantified against the old bytes in $S3/pins/ and re-pinned on the instance
#                   (manifest before/after kept), then fetch.py --verify-all
#   parity          every parity suite on 3.12 and 3.15t (the CI parity job's command), the two interpreters in parallel
#   bench_dev       bench/run_consolidated.sh on the dev slice (10k docs)
#   bench_headline  bench/run_consolidated.sh e2e on the 100k headline slice (stage sweep reused from bench_dev); skipped
#                   with a reason if the shutdown deadline leaves too little time
#   refs_hwcaps     x86_64 only, when a per-platform ref changed: regenerate it with glibc's AVX2/FMA ifuncs masked
#                   (GLIBC_TUNABLES) and check it against the old (emulated, SSE2) pin, to attribute the delta
# After every step: results packed to $S3/results/<machine>/results.tar.zst (+ status.json, run.log), so a partial run
# survives the instance's self-termination. A guard packs once more 15 min before the scheduled shutdown.
#
# Knobs (env): PYROID_S3 (s3://... or a local dir for tests), PYROID_PREFIX [/opt/pyroid], PYROID_STEPS [all],
#   PYROID_FORCE ["" | space-separated steps to redo], PYROID_MACHINE [aws-<instance type> | hostname],
#   PYROID_DATE [UTC today, fixed at the first start], PYROID_DATA_SEED (tests: symlink pinned files from this dir
#   instead of downloading), WHEEL_PROFILE [release], REFS_WHAT [all], REFS_WORKERS [nproc],
#   PARITY_SUITES [ftfy fasttext filters dedup metrics json_repair], PARITY_PYTHONS [3.12 3.15t], PARITY_LIMIT [0],
#   PARITY_HYPOTHESIS [ci], DEV_ENV / HEADLINE_ENV (extra VAR=value words for run_consolidated.sh),
#   HEADLINE_RUNS [1], HEADLINE_WARMUP [0], HEADLINE_FORCE [0] (run even if the time estimate says no),
#   PYROID_DEADLINE (epoch s; default: the scheduled shutdown in /run/systemd/shutdown/scheduled),
#   REFS_HWCAPS_ARCHES [x86_64] (tests: run refs_hwcaps on other arches too; the masking itself is x86-only).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=${PYROID_PREFIX:-/opt/pyroid}
# shellcheck source=/dev/null
[[ -f $PREFIX/env.sh ]] && source "$PREFIX/env.sh"
export HOME=${HOME:-/root}
# Every env and project below is named explicitly (a dev container exports its own defaults for these).
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-west-2}
S3=${PYROID_S3:?set PYROID_S3 to s3://<bucket>/<prefix> (or a local directory for tests)}
S3=${S3%/}
STATE=$PREFIX/state
WORK=$PREFIX/work
SRC=$PREFIX/src/pyroid
export PYROID_DATA=${PYROID_DATA:-$PREFIX/data}
mkdir -p "$STATE" "$WORK" "$PREFIX/logs" "$PYROID_DATA"

dmi() { cat "/sys/devices/virtual/dmi/id/$1" 2>/dev/null || true; }
if [[ -z ${PYROID_MACHINE:-} ]]; then
    if [[ "$(dmi sys_vendor)" == "Amazon EC2" && -n "$(dmi product_name)" ]]; then
        PYROID_MACHINE=aws-$(dmi product_name)
    else
        PYROID_MACHINE=$(hostname -s 2>/dev/null || echo unknown)
    fi
fi
MACHINE=$PYROID_MACHINE
[[ -f $STATE/date ]] || echo "${PYROID_DATE:-$(date -u +%Y-%m-%d)}" >"$STATE/date"
DATE=$(cat "$STATE/date")
RES=$PREFIX/results/$MACHINE/$DATE # becomes bench/results/<machine>/<date>/ in the repo
mkdir -p "$RES"
LOG=$PREFIX/logs/run.log
exec > >(tee -a "$LOG") 2>&1

STEPS_ALL="meta code wheels envs fetch refs parity bench_dev bench_headline refs_hwcaps"
STEPS=${PYROID_STEPS:-$STEPS_ALL}
FORCE=${PYROID_FORCE:-}
SUITES=${PARITY_SUITES:-ftfy fasttext filters dedup metrics json_repair}
PYTHONS_PARITY=${PARITY_PYTHONS:-3.12 3.15t}
NPROC=$(nproc)
ARCH=$(uname -m)
log() { echo "[run $(date -u +%H:%M:%S)] $*"; }
PY313=$WORK/venv-bench/bin/python
PY315T=$WORK/venv-b315t/bin/python
PYPAR=$WORK/venv-parity/bin/python

deadline() {  # epoch seconds of the scheduled shutdown, or empty
    if [[ -n ${PYROID_DEADLINE:-} ]]; then echo "$PYROID_DEADLINE"; return; fi
    local f=/run/systemd/shutdown/scheduled usec
    [[ -r $f ]] || return 0
    usec=$(sed -n 's/^USEC=//p' "$f")
    [[ -n $usec ]] && echo $((usec / 1000000))
    return 0
}
DEADLINE=$(deadline)

s3put() {  # local file -> $S3/<key>
    if [[ $S3 == s3://* ]]; then aws s3 cp --only-show-errors "$1" "$S3/$2"; else mkdir -p "$(dirname "$S3/$2")" && cp "$1" "$S3/$2"; fi
}
s3get() {  # $S3/<key> -> local file; with a trailing / on both: the whole prefix
    if [[ $S3 == s3://* ]]; then
        if [[ $1 == */ ]]; then aws s3 cp --only-show-errors --recursive "$S3/$1" "$2"; else aws s3 cp --only-show-errors "$S3/$1" "$2"; fi
    else
        if [[ $1 == */ ]]; then mkdir -p "$2" && { cp -r "$S3/$1." "$2" 2>/dev/null || true; }; else cp "$S3/$1" "$2"; fi
    fi
}

status() {  # state [step]
    local now; now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    printf '{"machine": "%s", "date": "%s", "state": "%s", "step": "%s", "done": "%s", "failed": "%s", "skipped": "%s", "updated": "%s", "deadline": "%s", "loadavg": "%s"}\n' \
        "$MACHINE" "$DATE" "$1" "${2:-}" "$(names "done")" \
        "$(names failed)" "$(names skipped)" \
        "$now" "${DEADLINE:+$(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ)}" "$(cut -d' ' -f1-3 /proc/loadavg)" \
        >"$STATE/status.json"
    cp "$STATE/status.json" "$RES/status.json"
    s3put "$STATE/status.json" "results/$MACHINE/status.json" || log "status upload failed"
}

package() {  # pack $RES (+ refs-new) and upload; safe to call concurrently
    (
        flock 9
        mkdir -p "$RES/logs" "$PREFIX/out"
        cp "$LOG" "$RES/logs/run.log" 2>/dev/null || true
        cp "$PREFIX"/logs/bootstrap.{log,json} "$RES/logs/" 2>/dev/null || true
        cp "$STATE/durations.tsv" "$RES/durations.tsv" 2>/dev/null || true
        tar -C "$PREFIX/results" --zstd --warning=no-file-changed -cf "$PREFIX/out/results.tar.zst" "$MACHINE/$DATE" || true
        s3put "$PREFIX/out/results.tar.zst" "results/$MACHINE/results.tar.zst" || log "results upload failed"
        if [[ -d $PREFIX/refs-new ]] && [[ -n "$(ls -A "$PREFIX/refs-new")" ]]; then
            tar -C "$PREFIX" --zstd -cf "$PREFIX/out/refs-new.tar.zst" refs-new
            s3put "$PREFIX/out/refs-new.tar.zst" "results/$MACHINE/refs-new.tar.zst" || log "refs upload failed"
        fi
        s3put "$LOG" "results/$MACHINE/run.log" || true
    ) 9>"$STATE/package.lock"
}

names() { find "$STATE" -maxdepth 1 -name "*.$1" -printf "%f\n" | sed "s/\.$1\$//" | sort | xargs; }
want() { [[ " $STEPS " == *" $1 "* ]]; }
run_step() {  # name critical(0|1)
    local name=$1 crit=$2 t0=$SECONDS rc
    if ! want "$name"; then log "skip $name (not in PYROID_STEPS)"; return 0; fi
    if [[ -f $STATE/$name.done && " $FORCE " != *" $name "* ]]; then log "skip $name (done earlier)"; return 0; fi
    rm -f "$STATE/$name.failed" "$STATE/$name.skipped" "$STATE/$name.done"
    log "=== $name"
    CURRENT_STEP=$name
    status running "$name"
    set +e
    (set -euo pipefail; "step_$name")
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' "$name" "$((SECONDS - t0))" "$rc" >>"$STATE/durations.tsv"
    if ((rc == 0)); then
        [[ -f $STATE/$name.skipped ]] || touch "$STATE/$name.done"
        log "=== $name: ok ($((SECONDS - t0)) s)"
    else
        echo "$rc" >"$STATE/$name.failed"
        log "=== $name: FAILED rc=$rc ($((SECONDS - t0)) s)"
    fi
    status running "$name"
    package
    if ((rc != 0 && crit == 1)); then log "critical step $name failed: stopping"; status failed "$name"; exit 1; fi
    return 0
}

wheel() {  # abi3 | abi3t -> path
    local pat
    case $1 in abi3) pat='-cp311-abi3-' ;; abi3t) pat='-cp315-abi3.abi3t-' ;; esac
    find "$WORK/dist" -maxdepth 1 -name "pyroid-*${pat}*.whl" | head -n 1
}

# Eight CPUs on distinct physical cores (lscpu's CORE ids): on c7i the second hyperthreads are left out.
pick_cpus() {
    lscpu -p=CPU,CORE 2>/dev/null | grep -v '^#' | sort -t, -k2,2n -k1,1n | awk -F, '!seen[$2]++ {print $1}' |
        head -n 8 | sort -n | paste -sd, -
}
export BENCH_CPUS=${BENCH_CPUS:-$(pick_cpus)}
[[ -n $BENCH_CPUS ]] || BENCH_CPUS=0-7

# ---------------------------------------------------------------------------------------------------------- steps
step_meta() {
    local d=$RES/machine
    mkdir -p "$d"
    lscpu >"$d/lscpu.txt"
    lscpu -e >"$d/lscpu-e.txt"
    lscpu -J >"$d/lscpu.json"
    awk 'NF == 0 {exit} {print}' /proc/cpuinfo >"$d/cpuinfo-cpu0.txt"
    uname -a >"$d/uname.txt"
    cp /etc/os-release "$d/os-release.txt"
    for f in sys_vendor product_name board_asset_tag bios_vendor bios_version; do echo "$f: $(dmi $f)"; done >"$d/dmi.txt"
    local tok
    tok=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300' || true)
    if [[ -n $tok ]]; then
        for k in instance-id instance-type ami-id placement/availability-zone placement/region instance-life-cycle; do
            echo "$k: $(curl -s -m 2 -H "X-aws-ec2-metadata-token: $tok" "http://169.254.169.254/latest/meta-data/$k")"
        done >"$d/imds.txt"
    fi
    local ldso
    for ldso in /lib64/ld-linux-x86-64.so.2 /lib/ld-linux-aarch64.so.1; do
        [[ -x $ldso ]] && "$ldso" --list-diagnostics >"$d/ld-diagnostics.txt" 2>&1 && break
    done
    {
        for f in /sys/kernel/mm/transparent_hugepage/{enabled,defrag} /sys/devices/system/cpu/smt/{active,control} \
            /sys/devices/system/cpu/cpu0/cpufreq/{scaling_driver,scaling_governor,scaling_max_freq} \
            /sys/devices/system/cpu/isolated /proc/sys/kernel/perf_event_paranoid; do
            [[ -r $f ]] && echo "$f: $(cat "$f")"
        done
        echo "nproc: $NPROC"
        echo "bench_cpus: $BENCH_CPUS"
        echo "loadavg: $(cat /proc/loadavg)"
        echo "deadline: ${DEADLINE:+$(date -u -d "@$DEADLINE" +%FT%TZ)}"
        command -v free >/dev/null && free -m
        df -h /
    } >"$d/system.txt"
    command -v systemctl >/dev/null && systemctl list-timers --all --no-pager >"$d/timers.txt" 2>&1 || true
    {
        rustc -vV
        cargo -V
        uv --version
        maturin --version
        gcc --version | head -n 1
        ldd --version | head -n 1
        uv python list --only-installed
    } >"$d/versions.txt" 2>&1
    cat "$d/lscpu.txt" "$d/system.txt"
}

step_code() {
    local b=$PREFIX/code/pyroid-next.bundle
    mkdir -p "$PREFIX/code"
    s3get code/pyroid-next.bundle "$b"
    git bundle list-heads "$b"
    rm -rf "$SRC"
    git clone --quiet --branch "${PYROID_BRANCH:-next}" "$b" "$SRC"
    local ov=${PYROID_OVERLAY_REF-feat/ec2bench} ov_sha=""
    local paths=${PYROID_OVERLAY_PATHS:-bench/ec2 bench/run_consolidated.sh bench/run_baselines.sh bench/consolidate.py bench/chart.py bench/machine_info.py bench/summarize.py}
    if [[ -n $ov ]] && git -C "$SRC" rev-parse -q --verify "origin/$ov" >/dev/null; then
        ov_sha=$(git -C "$SRC" rev-parse "origin/$ov")
        local p have=""
        for p in $paths; do
            if [[ -n "$(git -C "$SRC" ls-tree "origin/$ov" -- "$p")" ]]; then have+=" $p"; fi
        done
        paths=${have# }
        # shellcheck disable=SC2086
        [[ -z $paths ]] || git -C "$SRC" checkout "origin/$ov" -- $paths
        log "overlaid ${paths:-nothing} from $ov @ $ov_sha"
    fi
    {
        echo "branch: ${PYROID_BRANCH:-next}"
        echo "sha: $(git -C "$SRC" rev-parse HEAD)"
        echo "subject: $(git -C "$SRC" log -1 --format=%s)"
        echo "overlay_ref: $ov"
        echo "overlay_sha: $ov_sha"
        echo "overlay_paths: $paths"
        echo "status:"
        git -C "$SRC" status --porcelain
        echo "diffstat vs HEAD:"
        git -C "$SRC" diff --cached --stat HEAD
    } | tee "$RES/code.txt"
}

step_wheels() {
    cd "$SRC"
    rm -rf "$WORK/dist"
    mkdir -p "$WORK/dist"
    local prof=(--release)
    [[ ${WHEEL_PROFILE:-release} == release ]] || prof=(--profile "$WHEEL_PROFILE")
    local p312 p315 r1=0 r2=0
    p312=$(uv python find 3.12)
    p315=$(uv python find 3.15)
    "$p315" -c 'import sysconfig, sys; sys.exit(1 if sysconfig.get_config_var("Py_GIL_DISABLED") else 0)' ||
        { echo "uv python find 3.15 gave a free-threaded interpreter: $p315" >&2; return 1; }
    # Two runs on purpose (ci.yml): one maturin run picks one ABI family; with 3.15 in it, 3.12 would not be abi3.
    # In parallel, each with its own target dir: fat LTO with one codegen unit links on one core.
    CARGO_TARGET_DIR=$WORK/cargo-abi3 maturin build "${prof[@]}" --locked --out "$WORK/dist" -i "$p312" \
        >"$RES/wheel-abi3.log" 2>&1 &
    local j1=$!
    CARGO_TARGET_DIR=$WORK/cargo-abi3t maturin build "${prof[@]}" --locked --out "$WORK/dist" -i "$p315" \
        >"$RES/wheel-abi3t.log" 2>&1 &
    local j2=$!
    wait "$j1" || r1=$?
    wait "$j2" || r2=$?
    tail -n 3 "$RES/wheel-abi3.log" "$RES/wheel-abi3t.log"
    ((r1 == 0 && r2 == 0)) || { echo "maturin failed: abi3 rc=$r1, abi3t rc=$r2" >&2; return 1; }
    [[ -n "$(wheel abi3)" && -n "$(wheel abi3t)" ]] || { ls -l "$WORK/dist"; echo "missing a wheel flavour" >&2; return 1; }
    (cd "$WORK/dist" && sha256sum ./*.whl && ls -l ./*.whl) | tee "$RES/wheels.txt"
    rm -rf "$WORK/cargo-abi3" "$WORK/cargo-abi3t" # ~2 GB each; the wheels are all later steps need
}

step_envs() {
    (cd "$SRC/bench" && UV_PROJECT_ENVIRONMENT=$WORK/venv-bench uv sync --frozen)
    uv pip install --python "$PY313" --no-index --no-deps --reinstall "$(wheel abi3)"
    rm -rf "$WORK/venv-b315t"
    uv venv -q -p 3.15t "$WORK/venv-b315t"
    # The 3.15t bench env as P4 built it (venv-b315t); stages.py/gates.py import datatrove's pure-Python parts.
    uv pip install --python "$PY315T" pyperf==2.10.0 pyarrow==26.0.1 numpy==2.5.4 ftfy==6.3.1 datatrove==0.10.1 \
        regex==2026.9.29 psutil anyascii==0.3.3 -e "$SRC/parity"
    uv pip install --python "$PY315T" --no-index --no-deps --reinstall "$(wheel abi3t)"
    (cd "$SRC/parity" && UV_PROJECT_ENVIRONMENT=$WORK/venv-parity uv sync --frozen)
    (cd "$SRC/parity" && UV_PROJECT_ENVIRONMENT=$WORK/venv-parity-312 uv sync --frozen -p 3.12)
    (cd "$SRC/parity" && UV_PROJECT_ENVIRONMENT=$WORK/venv-parity-315t uv sync --frozen -p 3.15t --no-default-groups)
    local py
    for py in "$PY313" "$PY315T"; do
        "$py" -c 'import sys, pyroid, pyroid._pyroid as m; print(sys.version.split()[0], "GIL", getattr(sys, "_is_gil_enabled", lambda: True)(), pyroid.__version__, m.__file__)'
    done
    for py in "$PY313" "$PY315T" "$PYPAR" "$WORK/venv-parity-312/bin/python" "$WORK/venv-parity-315t/bin/python"; do
        echo "== $py"
        uv pip freeze --python "$py"
    done >"$RES/envs-freeze.txt"
}

seed_data() {  # tests only: symlink every pinned file that exists in $PYROID_DATA_SEED (refs regenerate over the links)
    local seed=$1 rel
    for rel in $("$PY313" -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["files"]))' "$SRC/bench/manifest.json"); do
        if [[ -e $seed/$rel && ! -e $PYROID_DATA/$rel ]]; then
            mkdir -p "$(dirname "$PYROID_DATA/$rel")"
            ln -s "$seed/$rel" "$PYROID_DATA/$rel"
        fi
    done
}

step_fetch() {
    cd "$SRC/bench"
    [[ -n ${PYROID_DATA_SEED:-} ]] && seed_data "$PYROID_DATA_SEED"
    local i
    for i in 1 2 3; do
        "$PY313" fetch.py && break
        ((i < 3)) || return 1
        log "fetch attempt $i failed; retrying in 60 s"
        sleep 60
    done
    "$PY313" fetch.py --verify
}

step_refs() {
    local d=$RES/refs rc=0 jr=0
    mkdir -p "$d" "$PREFIX/pins"
    s3get pins/ "$PREFIX/pins/" || log "no pins/ in $S3 (deltas will not be computed)"
    ls -l "$PREFIX/pins" || true
    cp "$SRC/bench/manifest.json" "$d/manifest.before.json"
    cd "$SRC"
    # shellcheck disable=SC2086
    "$PYPAR" -m pyroid_parity.refs ${REFS_WHAT:-all} --workers "${REFS_WORKERS:-$NPROC}" 2>&1 | tee "$d/refs.log" || rc=$?
    echo "refs exit $rc (1 = some file differs from its pin; refdiff.py check says which and by how much)" | tee -a "$d/refs.log"
    "$PYPAR" -m pyroid_parity.json_repair_record --check >"$d/json_repair_record.log" 2>&1 || jr=$?
    echo "json_repair_record --check exit $jr" | tee -a "$d/refs.log"
    "$PYPAR" "$HERE/refdiff.py" check --pins "$PREFIX/pins" --copy-new "$PREFIX/refs-new" --repin --out "$d/refs-check.json" |
        tee "$d/refs-check.txt"
    cp "$SRC/bench/manifest.json" "$d/manifest.after.json"
    diff -u "$d/manifest.before.json" "$d/manifest.after.json" >"$d/manifest.diff" || true
    "$PY313" bench/fetch.py --verify-all 2>&1 | tee "$d/verify-all.log"
    # Per-platform refs that moved were re-pinned above (expected on new hardware): later steps run against this
    # hardware's oracle outputs. The step fails only for a missing ref, a non-platform ref that moved, or json_repair.
    "$PY313" - "$d/refs-check.json" "$jr" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
bad = [f"missing {m['rel']}" for m in c["missing"]] + [f"unexpected {u}" for u in c["unexpected"]]
bad += ["json_repair_record --check failed"] if sys.argv[2] != "0" else []
moved = [m["rel"] for m in c["mismatched"] if m["platform_specific"]]
print(f"refs: {len(c['checked'])} checked, per-platform re-pinned: {moved or 'none'}; problems: {bad or 'none'}")
sys.exit(1 if bad else 0)
PY
}


parity_one() {  # python suite
    local py=$1 suite=$2 whl groups=() envdir
    if [[ $py == *t ]]; then whl=$(wheel abi3t); groups=(--no-default-groups); else whl=$(wheel abi3); fi
    envdir=$WORK/venv-parity-${py//./}
    (cd "$SRC" && UV_PROJECT_ENVIRONMENT=$envdir HYPOTHESIS_PROFILE=${PARITY_HYPOTHESIS:-ci} \
        uv run --frozen --project parity --python "$py" "${groups[@]}" --with "$whl" \
        pytest parity/tests -m "$suite" --impl "pyroid.$suite" -p no:cacheprovider -q \
        --parity-limit "${PARITY_LIMIT:-0}" --parity-report "$RES/parity/$suite-$py.json")
}

step_parity() {
    local d=$RES/parity py suite rc=0
    mkdir -p "$d"
    : >"$d/status.tsv"
    for py in $PYTHONS_PARITY; do
        (
            for suite in $SUITES; do
                local t0=$SECONDS r=0
                parity_one "$py" "$suite" >"$d/$suite-$py.log" 2>&1 || r=$?
                printf '%s\t%s\t%s\t%s\n' "$py" "$suite" "$r" "$((SECONDS - t0))" >>"$d/status.tsv"
                echo "[parity $py] $suite exit $r ($((SECONDS - t0)) s): $(tail -n 1 "$d/$suite-$py.log")"
            done
        ) &
    done
    wait
    for py in $PYTHONS_PARITY; do
        shopt -s nullglob
        local files=("$d"/*-"$py".json)
        if ((${#files[@]})); then
            "$PYPAR" -m pyroid_parity.report merge "${files[@]}" -o "$d/parity-$py.merged.json" || log "report merge failed"
            : >"$d/PARITY-$py.md"
            local f
            for f in "${files[@]}"; do  # render takes one report (merge output is {"runs": [...]})
                "$PYPAR" -m pyroid_parity.report render "$f" >>"$d/PARITY-$py.md" || log "render $f failed"
                echo >>"$d/PARITY-$py.md"
            done
        fi
    done
    cat "$d/status.tsv"
    awk -F'\t' '$3 != 0 {bad = 1} END {exit bad}' "$d/status.tsv" || rc=1
    return "$rc"
}

quiet_box() {
    if command -v systemctl >/dev/null && [[ -d /run/systemd/system ]]; then
        systemctl stop dnf-makecache.timer 2>/dev/null || true
    fi
    log "before benchmarking: load $(cut -d' ' -f1-3 /proc/loadavg), BENCH_CPUS=$BENCH_CPUS"
}

consolidated() {  # out-dir, then extra VAR=value words
    local out=$1
    shift
    cd "$SRC"
    export BENCH_OVERWRITE=1 # a step that runs again starts its folder afresh
    env BENCH_OUT="$out" BENCH_PY313="$PY313" BENCH_PY315T="$PY315T" BENCH_WORK="$WORK/bench-work" \
        BENCH_MACHINE="$MACHINE" BENCH_DATE="$DATE" BENCH_GIT_SHA="$(git -C "$SRC" rev-parse HEAD)" \
        "$@" bash bench/run_consolidated.sh
    rm -rf "$WORK/bench-work"
}

step_bench_dev() {
    quiet_box
    # shellcheck disable=SC2086
    consolidated "$RES/dev" ${DEV_ENV:-}
}

headline_fits() {  # 0 if the headline e2e fits before the deadline (estimate from the dev run's own medians)
    [[ ${HEADLINE_FORCE:-0} == 1 || -z $DEADLINE ]] && return 0
    local need
    # The dev run's medians scaled to the headline's docs; 100 min if they cannot be read.
    need=$("$PY313" - "$RES/dev" "${HEADLINE_RUNS:-1}" "${HEADLINE_WARMUP:-0}" "${BENCH_E2E_DOCS:-0}" <<'PY' || echo 6000
import json, statistics, sys
from pathlib import Path
dev = Path(sys.argv[1])
res = {}
for r in json.loads((dev / "e2e" / "hyperfine-e2e.json").read_text())["results"]:
    wall = [m["time_wall_clock"]["value"] for m in r.get("measurements", [])] or r.get("times", [])
    res[r.get("name") or r["command"]] = statistics.median(wall)
dev_docs = json.loads((dev / "run-config.json").read_text())["resolved"]["e2e_docs"]
head_docs = int(sys.argv[4]) or 100_000
scale = head_docs / dev_docs
runs = int(sys.argv[2]) + int(sys.argv[3])
e2e = scale * runs * sum(res.values())  # every e2e command, 1 and 8 workers
digest = scale * sum(v for k, v in res.items() if k.endswith("-w8"))  # the 8-worker digest runs
chain = scale * 7 * sum(v for k, v in res.items() if k.startswith("pyroid-"))  # chain.py: gate + 6 timed runs
print(int(e2e + digest + chain + 10 * 60))  # + 10 min margin
PY
)
    local left=$((DEADLINE - $(date +%s) - 20 * 60)) # keep 20 min for refs_hwcaps and the last upload
    log "headline estimate ${need}s, time left before the deadline (minus 20 min) ${left}s"
    ((need <= left))
}

step_bench_headline() {
    if ! headline_fits; then
        log "headline skipped: not enough time before the scheduled shutdown (HEADLINE_FORCE=1 overrides)"
        touch "$STATE/bench_headline.skipped"
        return 0
    fi
    quiet_box
    # Stage sweep, gates and memory come from bench_dev (same corpus slices); the digest runs at 8 workers on all docs.
    # shellcheck disable=SC2086
    consolidated "$RES/headline" BENCH_E2E_CORPUS=headline BENCH_SKIP=gates,stages,memory,e2e_stages,equiv_stages \
        BENCH_SKIP_FT_STAGES=1 BENCH_EQUIV_DOCS=100000 BENCH_EQUIV_WORKERS=8 \
        BENCH_E2E_RUNS="${HEADLINE_RUNS:-1}" BENCH_E2E_WARMUP="${HEADLINE_WARMUP:-0}" ${HEADLINE_ENV:-}
}

step_refs_hwcaps() {
    local check=$RES/refs/refs-check.json d=$RES/refs/hwcaps
    if [[ " ${REFS_HWCAPS_ARCHES:-x86_64} " != *" $ARCH "* || ! -f $check ]]; then log "refs_hwcaps: not on ${REFS_HWCAPS_ARCHES:-x86_64}, or no refs-check.json"; touch "$STATE/refs_hwcaps.skipped"; return 0; fi
    local rels
    rels=$("$PY313" -c 'import json, sys; c = json.load(open(sys.argv[1])); print(" ".join(sorted(r["rel"] for r in c.get("mismatched", []) if r.get("platform_specific"))))' "$check")
    if [[ -z $rels ]]; then log "refs_hwcaps: every per-platform ref reproduced; nothing to attribute"; touch "$STATE/refs_hwcaps.skipped"; return 0; fi
    mkdir -p "$d"
    local alt=$PREFIX/data-hwcaps rel f suites=""
    rm -rf "$alt"
    mkdir -p "$alt/parity"
    for f in "$PYROID_DATA"/*; do [[ $(basename "$f") == parity ]] || ln -s "$f" "$alt/"; done
    for f in "$PYROID_DATA"/parity/*; do ln -s "$f" "$alt/parity/"; done
    for rel in $rels; do
        rm -f "$alt/$rel"
        case $(basename "$rel") in fasttext-ref*) suites+=" fasttext" ;; metrics-ref*) suites+=" metrics" ;; esac
    done
    # glibc 2.34's x86 tunables: masking these "usable" bits sends libm's exp/log/... ifuncs to their SSE2 variants.
    local tun=glibc.cpu.hwcaps=-AVX512F,-AVX2,-FMA,-FMA4,-AVX
    GLIBC_TUNABLES=$tun /lib64/ld-linux-x86-64.so.2 --list-diagnostics >"$d/ld-diagnostics-masked.txt" 2>&1 || true
    for s in $suites; do
        log "refs_hwcaps: $s with GLIBC_TUNABLES=$tun against the pins before this run"
        local rc=0
        (cd "$SRC" && PYROID_DATA=$alt PYROID_MANIFEST=$RES/refs/manifest.before.json GLIBC_TUNABLES=$tun \
            "$PYPAR" -m pyroid_parity.refs "$s" --workers "$NPROC") >"$d/refs-$s.log" 2>&1 || rc=$?
        echo "$s: exit $rc ($( ((rc == 0)) && echo 'reproduces the old pin' || echo 'differs from the old pin'))" | tee -a "$d/summary.txt"
    done
    for rel in $rels; do
        local b; b=$(basename "$rel")
        [[ -f $PREFIX/pins/$b ]] && "$PYPAR" "$HERE/refdiff.py" diff "$PREFIX/pins/$b" "$alt/$rel" --out "$d/delta-masked-vs-old-$b.json" | tee -a "$d/summary.txt"
        "$PYPAR" "$HERE/refdiff.py" diff "$alt/$rel" "$PYROID_DATA/$rel" --out "$d/delta-masked-vs-native-$b.json" | tee -a "$d/summary.txt"
    done
    rm -rf "$alt"
}

# ------------------------------------------------------------------------------------------------------------ main
uploader() { while sleep 600; do s3put "$LOG" "results/$MACHINE/run.log" || true; done; }
guard() {  # pack once more shortly before the scheduled shutdown, whatever is running then
    [[ -n $DEADLINE ]] || return 0
    local wait=$((DEADLINE - $(date +%s) - 15 * 60))
    ((wait > 0)) && sleep "$wait"
    log "deadline guard: packing results 15 min before the scheduled shutdown"
    status running "deadline-guard"
    package
}
uploader &
UP=$!
guard &
GUARD=$!
finish() {
    local rc=$?
    kill "$UP" "$GUARD" 2>/dev/null || true
    if ((rc == 0)); then status "done" ""; else status failed "${CURRENT_STEP:-}"; fi
    package
    log "exit $rc; results: $S3/results/$MACHINE/results.tar.zst"
}
trap finish EXIT

log "machine $MACHINE ($ARCH, $NPROC CPUs), date $DATE, steps: $STEPS"
log "deadline: $([[ -n $DEADLINE ]] && date -u -d "@$DEADLINE" +%FT%TZ || echo none); results -> $S3/results/$MACHINE/"
run_step meta 0
run_step code 1
run_step wheels 1
run_step envs 1
run_step fetch 1
run_step refs 0
run_step parity 0
run_step bench_dev 0
run_step bench_headline 0
run_step refs_hwcaps 0
failed=$(names failed)
log "finished; failed steps: ${failed:-none}"
[[ -z $failed ]]
