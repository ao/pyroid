#!/usr/bin/env bash
# The end-to-end FineWeb-from-WARC benchmark (bench/fineweb_warc.py) on one EC2 instance, natively (no Docker), as
# root, on Amazon Linux 2023. Launched and shut down by the lead (the shutdown time comes from the user data).
#
#   HTMLX_S3=s3://<bucket>/<prefix> bash run_warc.sh
#
# Resumable: every step leaves $HTMLX_PREFIX/state/<step>.done and a re-run skips the steps already done. Results
# (small files only) go to $HTMLX_S3/results/ after every step, with status.json and run.log.
#
# Settings (env):
#   HTMLX_S3        s3:// prefix: code/pyroid.bundle (git bundle) is read from it, results/ written to it (required)
#   HTMLX_BRANCH    branch to check out of the bundle [feat/warc]
#   HTMLX_WARCS     WARC files of $HTMLX_CRAWL to process, from the first in its warc.paths.gz [8]
#   HTMLX_CRAWL     Common Crawl crawl [CC-MAIN-2026-39]
#   HTMLX_CPUS      CPUs the runs are pinned to (taskset) [0-7: one thread per core on a c7i.4xlarge]
#   HTMLX_PREFIX    work folder [/opt/htmlx]
#   HTMLX_STEPS     steps to run (default: all but the 1-worker runs, in the order below); HTMLX_FORCE: steps to
#                   redo even if done
#   HTMLX_ONE_WORKER  1: also run native-1, pyroid-1, stock-1 and compare (about 2.5 h more) [0]
#   HTMLX_PARITY_ARGS  extra pyroid_parity.warc_scale arguments for the parity step, e.g. "--limit 500" [none];
#                   HTMLX_PARITY_MINUTES [30]: its estimate for the deadline
#   HTMLX_HOOK      shell commands for the hook step; else the file HTMLX_HOOK_FILE; else $HTMLX_S3/code/hook.sh if
#                   it exists; else the hook step is skipped. HTMLX_HOOK_MINUTES [60]: its estimate for the deadline.
#   HTMLX_DEADLINE  epoch seconds to finish by [the scheduled shutdown in /run/systemd/shutdown/scheduled]; a run
#                   step whose estimate would end within 10 min of it is skipped (with the reason in run.log)
#   HTMLX_POWEROFF  1: power off 1 min after the last step [0]
#   HTMLX_RUN_ARGS  extra fineweb_warc.py arguments for every run, e.g. "--limit 200" (tests) [none]
#   HTMLX_STOCK_ARGS  extra arguments for stock runs only (tests on machines without g++ for fasttext-numpy2-wheel:
#                   "--lid-model PATH" swaps in fasttext-predict; never on EC2) [none]
#
# Steps, most important first (estimates for 8 WARC files on a c7i.4xlarge, from P11's run there: stock datatrove
# 871 s at 8 workers and 6,976 s at 1, the pyroid modes 104-110 s at 8 and 739-776 s at 1):
#   code 1 min      dnf git gcc gcc-c++ tar zstd util-linux file-libs; clone $HTMLX_BRANCH from the bundle; machine info
#   bootstrap 5     bench/ec2/bootstrap.sh with PYTHONS=3.13.18 (rustup, uv, CPython, maturin)
#   wheel 8         maturin build --release --locked (the cargo target is deleted afterwards)
#   envs 3          venv-stock: datatrove[io,processing]==0.10.1 as pip resolves it (trafilatura 1.11.0,
#                   fasttext-numpy2-wheel) + spaCy 3.8.18 (datatrove's English tokenizer needs it; [processing] does
#                   not install it); venv-pyroid: the same pinned set (venv-stock's freeze) + the pyroid wheel.
#                   Warms datatrove's caches (URL block lists, lid.176.bin, the Public Suffix List) once.
#   fetch 5         the WARC files from data.commoncrawl.org (sizes checked against Content-Length) and lid.176.bin
#                   (sha256 checked)
#   reader 6        fineweb_warc.py reader: warcio + process_record, pyroid.warc.iter_documents and iter_batches on
#                   each file, one process on the first CPU of $HTMLX_CPUS; the readers' digests must agree
#   stock-8 15      fineweb_warc.py stock, 8 workers, datatrove's real LanguageFilter
#   pyroid-8 2      fineweb_warc.py pyroid-steps, 8 workers
#   native-8 2      fineweb_warc.py native, 8 workers (reading with pyroid.warc)
#   cli-8 3         fineweb_warc.py cli: pyroid clean --recipe fineweb-warc --workers 8
#   compare-8 1-4   digests of the four; if stock-8 lost pages to Trafilatura's timeout, also native --timeouts-of
#                   stock-8 (native-8x) compared with it
#   parity 30       parity/pyroid_parity/warc_scale.py over the 8 files: every record and document, warcio 1.8.2
#                   and datatrove 0.10.1 vs pyroid.warc (8 workers), results in parity/warc-scale.json
#   hook ?          $HTMLX_HOOK, unpinned; it gets the variables listed at step_hook
#   native-1 13, pyroid-1 14, stock-1 117       the same at 1 worker (only with HTMLX_ONE_WORKER=1)
#   compare 1-4     every run against native-8; native-1x (native --timeouts-of stock-1) if stock-1 lost pages
set -euo pipefail

S3=${HTMLX_S3:?set HTMLX_S3 to the s3:// prefix holding code/pyroid.bundle}
S3=${S3%/}
BRANCH=${HTMLX_BRANCH:-feat/warc}
NWARC=${HTMLX_WARCS:-8}
CRAWL=${HTMLX_CRAWL:-CC-MAIN-2026-39}
CPUS=${HTMLX_CPUS:-0-7}
W=${HTMLX_PREFIX:-/opt/htmlx}
RUNS="reader stock-8 pyroid-8 native-8 cli-8 compare-8 parity hook native-1 pyroid-1 stock-1 compare"
STEPS_ALL="code bootstrap wheel envs fetch reader stock-8 pyroid-8 native-8 cli-8 compare-8 parity hook"
if [[ ${HTMLX_ONE_WORKER:-0} == 1 ]]; then STEPS_ALL="$STEPS_ALL native-1 pyroid-1 stock-1 compare"; fi
STEPS=${HTMLX_STEPS:-$STEPS_ALL}
FORCE=${HTMLX_FORCE:-}
LID_SHA256=7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e # pyroid.integrations.recipes.LID_176_SHA256
R=$W/repo
RES=$W/results
STATE=$W/state
FW=$R/bench/fineweb_warc.py
mkdir -p "$W" "$RES/runs" "$STATE"
exec > >(tee -a "$RES/run.log") 2>&1
export HOME=${HOME:-/root}
export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-$(curl -s -m 2 -H "X-aws-ec2-metadata-token: $(curl -s -m 2 -X PUT \
    http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" \
    http://169.254.169.254/latest/meta-data/placement/region || true)}
# One cache for every env: datatrove's assets (URL block lists, lid.176.bin) and tldextract's Public Suffix List,
# so all runs read the same lists.
export HF_HOME=$W/hf TLDEXTRACT_CACHE=$W/tldextract TOKENIZERS_PARALLELISM=false
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT
log() { echo "[warc $(date -u +%H:%M:%S)] $*"; }
CURRENT=""

deadline() {
    if [[ -n ${HTMLX_DEADLINE:-} ]]; then echo "$HTMLX_DEADLINE"; return; fi
    local f=/run/systemd/shutdown/scheduled usec
    [[ -r $f ]] || return 0
    usec=$(sed -n 's/^USEC=//p' "$f")
    [[ -n $usec ]] && echo $((usec / 1000000))
    return 0
}
DEADLINE=$(deadline)

names() { find "$STATE" -maxdepth 1 -name "*.$1" -printf "%f\n" | sed "s/\.$1\$//" | sort | xargs; }
up() {
    aws s3 cp --only-show-errors --recursive "$RES/" "$S3/results/" || log "results upload failed"
}
status() {  # state [step]
    printf '{"state": "%s", "step": "%s", "done": "%s", "failed": "%s", "skipped": "%s", "updated": "%s", "deadline": "%s", "loadavg": "%s"}\n' \
        "$1" "${2:-}" "$(names done)" "$(names failed)" "$(names skipped)" "$(date -u +%FT%TZ)" \
        "${DEADLINE:+$(date -u -d "@$DEADLINE" +%FT%TZ)}" "$(cut -d' ' -f1-3 /proc/loadavg)" >"$RES/status.json"
}
want() { [[ " $STEPS " == *" $1 "* ]]; }
skip_step() {  # name reason: mark the running step skipped
    echo "$2" >"$STATE/$1.skipped"
    log "skip $1: $2"
}
run_step() {  # name critical(0|1)
    local name=$1 crit=$2 t0=$SECONDS rc
    if ! want "$name"; then return 0; fi
    if [[ -f $STATE/$name.done && " $FORCE " != *" $name "* ]]; then log "skip $name (done earlier)"; return 0; fi
    rm -f "$STATE/$name".{done,failed,skipped}
    CURRENT=$name
    log "=== $name (load $(cut -d' ' -f1-3 /proc/loadavg); $(df -h --output=avail "$W" | tail -n 1 | xargs) free)"
    status running "$name"
    up
    set +e
    (set -euo pipefail; "step_${name//-/_}")
    rc=$?
    set -e
    printf '%s\t%s\t%s\n' "$name" "$((SECONDS - t0))" "$rc" >>"$RES/durations.tsv"
    if ((rc == 0)); then
        [[ -f $STATE/$name.skipped ]] || touch "$STATE/$name.done"
        log "=== $name: ok ($((SECONDS - t0)) s)"
    else
        echo "$rc" >"$STATE/$name.failed"
        log "=== $name: FAILED rc=$rc ($((SECONDS - t0)) s)"
    fi
    status running "$name"
    up
    if ((rc != 0 && crit == 1)); then log "critical step $name failed: stopping"; status failed "$name"; up; exit 1; fi
    return 0
}

# Does a step of $1 minutes fit before the deadline (with 10 min to spare)?
fits() {
    [[ -z $DEADLINE ]] && return 0
    (($(date +%s) + $1 * 60 + 600 <= DEADLINE))
}
wall_of() {  # run name -> its measured wall seconds, or empty
    local f=$W/runs/$1/summary.json
    [[ -f $f ]] && "$W/venv-stock/bin/python" -c 'import json, sys; print(int(json.load(open(sys.argv[1]))["wall_seconds"]["total"]))' "$f"
    return 0
}
estimate() {  # run name -> minutes: P10's numbers scaled to $NWARC, then 8x the 8-worker run once it is measured
    local base w
    case $1 in
        stock-8) base=15 ;; pyroid-8 | native-8 | native-8x | native-1x) base=2 ;; cli-8) base=3 ;; reader) base=6 ;;
        native-1) base=13 ;; pyroid-1) base=14 ;; stock-1) base=117 ;; *) base=10 ;;
    esac
    if [[ $1 == *-1 ]] && w=$(wall_of "${1%-1}-8") && [[ -n $w ]]; then
        echo $((w * 8 / 60 + 2))
    else
        echo $(((base * NWARC + 7) / 8))
    fi
}
PY=""
use_tools() {
    # shellcheck source=/dev/null
    source /opt/pyroid/env.sh
    PY=$(uv python find 3.13.18)
}

# ---------------------------------------------------------------------------------------------------------- steps
step_code() {
    dnf install -y -q git gcc gcc-c++ tar zstd util-linux file-libs
    aws s3 cp --only-show-errors "$S3/code/pyroid.bundle" "$W/pyroid.bundle"
    rm -rf "$R"
    git clone -q -b "$BRANCH" "$W/pyroid.bundle" "$R"
    {
        echo "branch: $BRANCH"
        echo "sha: $(git -C "$R" rev-parse HEAD)"
        echo "subject: $(git -C "$R" log -1 --format=%s)"
    } | tee "$RES/code.txt"
    mkdir -p "$RES/machine"
    lscpu >"$RES/machine/lscpu.txt"
    lscpu -e >"$RES/machine/lscpu-e.txt"
    local tok
    tok=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
    curl -s -m 2 -H "X-aws-ec2-metadata-token: $tok" http://169.254.169.254/latest/meta-data/instance-type \
        >"$RES/machine/instance-type.txt" || true
    {
        echo "nproc: $(nproc)"
        echo "cpus: $CPUS (cores: $(lscpu -p=CPU,CORE | grep -v '^#' | awk -F, -v l="$CPUS" 'BEGIN { n = split(l, r, ",");
            for (i = 1; i <= n; i++) { if (split(r[i], ab, "-") == 2) { for (c = ab[1]; c <= ab[2]; c++) s[c] = 1 } else s[r[i]] = 1 } }
            ($1 in s) { k[$2] = 1 } END { print length(k) }') distinct)"
        uname -r
        sed -n 's/^PRETTY_NAME=//p' /etc/os-release
        free -m
        df -h "$W"
    } | tee "$RES/machine/system.txt"
}

step_bootstrap() {
    PYTHONS=3.13.18 bash "$R/bench/ec2/bootstrap.sh" >"$RES/bootstrap.log" 2>&1 || { tail -n 40 "$RES/bootstrap.log"; return 1; }
    use_tools
    "$PY" -V
}

step_wheel() {
    use_tools
    rm -rf "$W/dist"
    (cd "$R" && CARGO_TARGET_DIR=$W/cargo maturin build --release --locked --out "$W/dist" -i "$PY" >"$RES/wheel.log" 2>&1) ||
        { tail -n 40 "$RES/wheel.log"; return 1; }
    rm -rf "$W/cargo"
    ls -l "$W"/dist/*.whl | tee -a "$RES/wheel.log"
}

step_envs() {
    use_tools
    local whl
    whl=$(ls "$W"/dist/pyroid-*.whl | head -n 1)
    uv venv -q --clear -p "$PY" "$W/venv-stock"
    VIRTUAL_ENV=$W/venv-stock uv pip install -q "datatrove[io,processing]==0.10.1" "spacy==3.8.18"
    VIRTUAL_ENV=$W/venv-stock uv pip freeze >"$RES/freeze-stock.txt"
    grep -q '^trafilatura==1.11.0$' "$RES/freeze-stock.txt" || { log "trafilatura is not 1.11.0"; grep -i trafilatura "$RES/freeze-stock.txt"; return 1; }
    grep -qiE '^fasttext[-_]numpy2[-_]wheel==' "$RES/freeze-stock.txt" || { log "no fasttext-numpy2-wheel"; return 1; }
    uv venv -q --clear -p "$PY" "$W/venv-pyroid"
    VIRTUAL_ENV=$W/venv-pyroid uv pip install -q -r "$RES/freeze-stock.txt" "$whl"
    VIRTUAL_ENV=$W/venv-pyroid uv pip freeze >"$RES/freeze-pyroid.txt"
    uv cache clean -q || true
    # datatrove's caches, filled once here so that no timed run pays for them: the URL block lists (extracted from
    # datatrove's package), lid.176.bin (datatrove downloads it), the Public Suffix List (tldextract fetches it).
    "$W/venv-stock/bin/python" - <<'EOF' | tee "$RES/caches.txt"
import hashlib
from datatrove.io import cached_asset_path_or_download
from datatrove.pipeline.filters import URLFilter
from datatrove.utils.lid import FT176LID
from tldextract import TLDExtract

f = URLFilter()
f.download_data()
print("url block lists:", len(f.block_listed_domains), "domains")
print("psl:", TLDExtract()("http://www.example.co.uk/"))
p = cached_asset_path_or_download(FT176LID.MODEL_URL, namespace="lid", subfolder=FT176LID.MODEL_SUBFOLDER,
                                  desc="fast-text language identifier model")
print("datatrove lid.176.bin:", hashlib.sha256(open(p, "rb").read()).hexdigest())
import spacy
spacy.blank("en")
EOF
    "$W/venv-pyroid/bin/python" - <<'EOF' | tee -a "$RES/caches.txt"
import importlib.metadata
import pyroid.trafilatura as pt
from pyroid.integrations.datatrove import Trafilatura
assert pt.extract("<html><body><p>x</p></body></html>", compat="1.11.0") == "x"
mode = Trafilatura()._mode()
print("pyroid", importlib.metadata.version("pyroid"), "Trafilatura drop-in mode:", mode)
assert mode == "1.11.0", mode
EOF
}

step_fetch() {
    mkdir -p "$W/warc"
    curl -sSf --retry 5 -o "$W/warc.paths.gz" "https://data.commoncrawl.org/crawl-data/$CRAWL/warc.paths.gz"
    gzip -dc "$W/warc.paths.gz" | sed -n "1,${NWARC}p" >"$RES/warc-paths.txt"
    # shellcheck disable=SC2016
    xargs -P 8 -I{} sh -c 'f="$1/warc/$(basename "$2")"
        n=$(curl -sSfI --retry 5 "https://data.commoncrawl.org/$2" | tr -d "\r" | sed -n "s/^[Cc]ontent-[Ll]ength: //p" | tail -n 1)
        if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$n" ]; then exit 0; fi
        curl -sSf --retry 8 --retry-all-errors -o "$f" "https://data.commoncrawl.org/$2"
        [ "$(stat -c %s "$f")" = "$n" ] || { echo "$f: $(stat -c %s "$f") bytes, expected $n"; exit 1; }' _ "$W" {} <"$RES/warc-paths.txt"
    ls -l "$W/warc" | tee "$RES/warc-files.txt"
    [[ $(find "$W/warc" -name '*.warc.gz' | wc -l) -eq $NWARC ]]
    [[ -f $W/lid.176.bin ]] || curl -sSf --retry 5 -o "$W/lid.176.bin" https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    local sha
    sha=$(sha256sum "$W/lid.176.bin" | cut -d' ' -f1)
    echo "lid.176.bin: $sha" | tee -a "$RES/warc-files.txt"
    [[ $sha == "$LID_SHA256" ]] || { log "lid.176.bin sha256 $sha, expected $LID_SHA256"; rm -f "$W/lid.176.bin"; return 1; }
}

label() {
    local it model
    it=$(cat "$RES/machine/instance-type.txt" 2>/dev/null || echo unknown)
    model=$(lscpu | sed -n 's/^Model name: *//p' | head -n 1)
    echo "AWS EC2 $it ($model, $(nproc) vCPU), $(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | tr -d '"'), CPUs $CPUS"
}

run_mode() {  # name mode workers [extra fineweb_warc.py args]
    local name=$1 mode=$2 n=$3 est
    shift 3
    est=$(estimate "$name")
    if ! fits "$est"; then skip_step "$name" "estimated $est min would end within 10 min of the deadline"; return 0; fi
    local py=$W/venv-pyroid/bin/python lid=(--lid-model "$W/lid.176.bin")
    # shellcheck disable=SC2206
    local extra=(${HTMLX_RUN_ARGS:-})
    # shellcheck disable=SC2206
    if [[ $mode == stock ]]; then py=$W/venv-stock/bin/python; lid=(${HTMLX_STOCK_ARGS:-}); fi
    log "$name: $mode, $n workers, CPUs $CPUS, estimated $est min"
    rm -rf "$W/runs/$name" "$RES/runs/$name"
    mkdir -p "$W/runs" "$RES/runs/$name"
    BENCH_LABEL=$(label) BENCH_GIT_SHA=$(git -C "$R" rev-parse HEAD) taskset -c "$CPUS" "$py" "$FW" "$mode" \
        --warcs "$W/warc" --out "$W/runs/$name" --workers "$n" --dump "$CRAWL" "${lid[@]}" "${extra[@]}" "$@" \
        >"$RES/runs/$name/run.log" 2>&1 || { tail -n 40 "$RES/runs/$name/run.log"; return 1; }
    tail -n 30 "$RES/runs/$name/run.log"
    local f
    for f in summary.json docs.tsv.gz events.jsonl timing logs cli-stats.json; do
        [[ -e $W/runs/$name/$f ]] && cp -r "$W/runs/$name/$f" "$RES/runs/$name/"
    done
    rm -rf "$RES/runs/$name"/logs/*/completions
    return 0
}
lost() {  # run name -> pages its Trafilatura step lost (timeouts, dead children), or empty
    local f=$W/runs/$1/summary.json
    [[ -f $f ]] && "$W/venv-stock/bin/python" -c 'import json, sys; t = json.load(open(sys.argv[1]))["trafilatura"]; print(t["timeout"] + t["broken_process"] + t["clean_error"])' "$f"
    return 0
}
compare() {  # out-name run...
    local out=$1 r args=()
    shift
    for r in "$@"; do [[ -f $W/runs/$r/summary.json ]] && args+=("$W/runs/$r"); done
    ((${#args[@]} >= 2)) || { log "compare $out: fewer than 2 finished runs"; return 0; }
    "$W/venv-stock/bin/python" "$FW" compare "${args[@]}" --json "$RES/$out.json" | tee "$RES/$out.txt" || true
}
emulated() {  # stock run -> run native with its lost pages dropped, if it lost any, and compare the two
    local s=$1 x=native-${1#stock-}x n
    n=$(lost "$s")
    if [[ -z $n || $n == 0 ]]; then log "$s lost no pages: no emulated run needed"; return 0; fi
    log "$s lost $n pages to Trafilatura's timeout or a dead child: $x drops the same pages"
    run_mode "$x" native 8 --timeouts-of "$W/runs/$s"
    compare "compare-$s-emulated" "$s" "$x"
}

step_reader() {
    local est first=${CPUS%%[-,]*}
    est=$(estimate reader)
    if ! fits "$est"; then skip_step reader "estimated $est min would end within 10 min of the deadline"; return 0; fi
    rm -rf "$W/runs/reader" "$RES/runs/reader"
    mkdir -p "$RES/runs/reader"
    # shellcheck disable=SC2206
    local extra=(${HTMLX_RUN_ARGS:-})
    BENCH_LABEL="$(label), CPU $first" BENCH_GIT_SHA=$(git -C "$R" rev-parse HEAD) taskset -c "$first" \
        "$W/venv-pyroid/bin/python" "$FW" reader --warcs "$W/warc" --out "$W/runs/reader" "${extra[@]}" \
        >"$RES/runs/reader/run.log" 2>&1 || { tail -n 40 "$RES/runs/reader/run.log"; return 1; }
    tail -n 30 "$RES/runs/reader/run.log"
    cp "$W/runs/reader/summary.json" "$RES/runs/reader/"
}
step_stock_8() { run_mode stock-8 stock 8; }
step_pyroid_8() { run_mode pyroid-8 pyroid-steps 8; }
step_native_8() { run_mode native-8 native 8; }
step_cli_8() { run_mode cli-8 cli 8; }
step_compare_8() {
    compare compare-8 stock-8 pyroid-8 native-8 cli-8
    emulated stock-8
}
step_parity() {
    local m=${HTMLX_PARITY_MINUTES:-30}
    if ! fits "$m"; then skip_step parity "estimated $m min would end within 10 min of the deadline"; return 0; fi
    mkdir -p "$RES/parity"
    # shellcheck disable=SC2206
    local extra=(${HTMLX_PARITY_ARGS:-})
    (cd "$R/parity" && taskset -c "$CPUS" "$W/venv-pyroid/bin/python" -m pyroid_parity.warc_scale \
        "$RES/parity/warc-scale.json" "$W"/warc/*.warc.gz --impl pyroid.warc --workers 8 "${extra[@]}") \
        >"$RES/parity/run.log" 2>&1 || { tail -n 40 "$RES/parity/run.log"; return 1; }
    tail -n 30 "$RES/parity/run.log"
}
step_hook() {
    local cmd=${HTMLX_HOOK:-} file
    if [[ -z $cmd && -n ${HTMLX_HOOK_FILE:-} ]]; then cmd=$(cat "$HTMLX_HOOK_FILE"); fi
    if [[ -z $cmd ]] && aws s3 cp --only-show-errors "$S3/code/hook.sh" "$W/hook.sh" 2>/dev/null; then cmd=$(cat "$W/hook.sh"); fi
    if [[ -z $cmd ]]; then skip_step hook "no HTMLX_HOOK, HTMLX_HOOK_FILE or $S3/code/hook.sh"; return 0; fi
    if ! fits "${HTMLX_HOOK_MINUTES:-60}"; then skip_step hook "estimated ${HTMLX_HOOK_MINUTES:-60} min would end within 10 min of the deadline"; return 0; fi
    use_tools
    mkdir -p "$RES/hook"
    file=$W/hook-run.sh
    printf '%s\n' "$cmd" >"$file"
    cp "$file" "$RES/hook/hook.sh"
    # What the hook gets: the checkout, the interpreter, both envs, the wheel, the data, a results folder (uploaded
    # with the rest: keep it small), the CPU list the runs use. uv and maturin are on PATH (/opt/pyroid/env.sh).
    (
        cd "$R"
        export HTMLX_W=$W HTMLX_REPO=$R HTMLX_PY=$PY HTMLX_VENV_STOCK=$W/venv-stock HTMLX_VENV_PYROID=$W/venv-pyroid \
            HTMLX_WHEEL=$(ls "$W"/dist/pyroid-*.whl | head -n 1) HTMLX_WARC_DIR=$W/warc HTMLX_LID_MODEL=$W/lid.176.bin \
            HTMLX_HOOK_RES=$RES/hook HTMLX_CPUS=$CPUS
        bash -euo pipefail "$file"
    ) >"$RES/hook/hook.log" 2>&1 || { tail -n 40 "$RES/hook/hook.log"; return 1; }
    tail -n 20 "$RES/hook/hook.log"
}
step_native_1() { run_mode native-1 native 1; }
step_pyroid_1() { run_mode pyroid-1 pyroid-steps 1; }
step_stock_1() { run_mode stock-1 stock 1; }
step_compare() {
    # Reference native-8: the pyroid runs must all match it; a stock run that lost pages matches its emulated run.
    compare compare native-8 pyroid-8 cli-8 native-1 pyroid-1 stock-8 stock-1
    emulated stock-1
}

# ----------------------------------------------------------------------------------------------------------- main
trap 'rc=$?; if ((rc != 0)); then status failed "$CURRENT"; up; fi' EXIT
log "start: branch $BRANCH, $NWARC WARC files of $CRAWL, CPUs $CPUS, deadline ${DEADLINE:+$(date -u -d "@$DEADLINE" +%FT%TZ)}"
run_step code 1
run_step bootstrap 1
run_step wheel 1
run_step envs 1
run_step fetch 1
for s in $RUNS; do
    run_step "$s" 0
done
du -sh "$W"/* 2>/dev/null | sort -h | tail -n 12 >"$RES/disk.txt" || true
status done
up
log "done"
if [[ ${HTMLX_POWEROFF:-0} == 1 ]]; then shutdown -h +1; fi
