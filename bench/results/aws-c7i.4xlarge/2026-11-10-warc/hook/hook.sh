# pyroid warc parity on EC2: the hook step of bench/ec2/run_warc.sh (bash -euo pipefail, as root, cwd = the
# checkout). It gets HTMLX_W, HTMLX_REPO, HTMLX_PY, HTMLX_WHEEL, HTMLX_HOOK_RES, HTMLX_CPUS (see step_hook).
#
# 1. env:    parity/ synced (uv sync --frozen, default groups) into $HTMLX_W/warc-parity/venv,
#            plus the instance's pyroid wheel; libmagic is AL2023's file-libs.
# 2. corpus: bench/fetch.py --only warc (86.9 MB of Common Crawl slices + warcio's test files, sha256-checked).
# 3. suite:  pytest -m warc --impl pyroid.warc --parity-limit 0 (every record), natively on this CPU, with
#            Hypothesis's ci profile; parity-report.json and the inputs of any diff (PYROID_PARITY_REPROS) go to the
#            results folder.
# 4. scale:  whole files the run's 8 CC-MAIN-2026-39 WARCs do not cover (two older WARC files, one without
#            WARC-Identified-Payload-Type, two WET, two WAT, the 2012 ARC; 3.8 GB), downloaded and size-checked,
#            pyroid_parity.warc_scale on them (4 record configs + the datatrove level), then deleted.
#
# Phases are timed in $RES/hook-phases.json. A phase that fails is recorded and the next one still runs; the hook
# exits 1 if any phase failed. Overrides for tests: WARC_HOOK_SYNC_ARGS (extra uv sync arguments),
# WARC_HOOK_FILES (paths under data.commoncrawl.org, one per line), WARC_HOOK_PYTEST_K (a pytest -k expression),
# WARC_HOOK_PYTEST_ARGS (more pytest arguments), WARC_HOOK_SKIP (phases to skip, e.g. "suite").
set -euo pipefail

R=${HTMLX_REPO:?}
W=${HTMLX_W:?}
RES=${HTMLX_HOOK_RES:?}
PY=${HTMLX_PY:?}
WHEEL=${HTMLX_WHEEL:?}
CPUS=${HTMLX_CPUS:-}
NPROC=$(nproc)
T=$W/warc-parity
VENV=$T/venv
mkdir -p "$T" "$RES"
export PYROID_DATA=$T/data UV_PROJECT_ENVIRONMENT=$VENV UV_PYTHON=$PY HYPOTHESIS_PROFILE=ci

CC=https://data.commoncrawl.org
FILES=${WARC_HOOK_FILES:-"crawl-data/CC-MAIN-2016-40/segments/1474738659496.36/warc/CC-MAIN-20160924173739-00001-ip-10-143-35-109.ec2.internal.warc.gz
crawl-data/CC-MAIN-2019-35/segments/1566027312025.20/warc/CC-MAIN-20190817203056-20190817225056-00001.warc.gz
crawl-data/CC-MAIN-2013-20/segments/1368696381249/wet/CC-MAIN-20130516092621-00001-ip-10-60-113-184.ec2.internal.warc.wet.gz
crawl-data/CC-MAIN-2023-50/segments/1700679099281.67/wet/CC-MAIN-20231128083443-20231128113443-00001.warc.wet.gz
crawl-data/CC-MAIN-2018-30/segments/1531676588961.14/wat/CC-MAIN-20180715183800-20180715203800-00001.warc.wat.gz
crawl-data/CC-MAIN-2026-43/segments/1791170053823.21/wat/CC-MAIN-20261005105623-20261005135623-00001.warc.wat.gz
parse-output/segment/1346823845675/1346864469604_0.arc.gz"}
SKIP=" ${WARC_HOOK_SKIP:-} "

declare -A SECONDS_OF STATUS_OF
FAILED=0
phase() {
    local name=$1
    shift
    if [[ $SKIP == *" $name "* ]]; then
        STATUS_OF[$name]=skipped
        echo "[$(date -u +%T)] $name: skipped"
        return 0
    fi
    echo "[$(date -u +%T)] $name: start"
    local t0=$SECONDS rc=0
    "$@" || rc=$?
    SECONDS_OF[$name]=$((SECONDS - t0))
    STATUS_OF[$name]=$([[ $rc == 0 ]] && echo ok || echo "exit $rc")
    [[ $rc == 0 ]] || FAILED=1
    echo "[$(date -u +%T)] $name: ${STATUS_OF[$name]} in ${SECONDS_OF[$name]} s"
    write_phases
}
write_phases() {
    {
        echo "{"
        local first=1 k
        for k in env corpus suite scale; do
            [[ -n ${STATUS_OF[$k]:-} ]] || continue
            [[ $first == 1 ]] || echo ","
            first=0
            printf '  "%s": {"status": "%s", "seconds": %s}' "$k" "${STATUS_OF[$k]}" "${SECONDS_OF[$k]:-null}"
        done
        printf '\n}\n'
    } >"$RES/hook-phases.json"
}

do_env() {
    # The default groups: the warc group, and datatrove's tokenizers etc. that pyroid.integrations.datatrove imports.
    # shellcheck disable=SC2086  # extra arguments are words
    (cd "$R/parity" && uv sync --frozen -q ${WARC_HOOK_SYNC_ARGS:-})
    uv pip install -q --python "$VENV/bin/python" --no-deps --reinstall "$WHEEL"
    "$VENV/bin/python" -c 'import magic, pyroid.warc as pw; print("libmagic", magic.version(), "pyroid.warc", pw.WARCIO_VERSION)'
    uv pip freeze --python "$VENV/bin/python" >"$RES/freeze-warc-parity.txt"
}

do_corpus() {
    (cd "$R/bench" && "$VENV/bin/python" fetch.py --only warc | tail -n 3)
}

do_suite() {
    local rc=0 select=()
    if [[ -n ${WARC_HOOK_PYTEST_K:-} ]]; then select=(-k "$WARC_HOOK_PYTEST_K"); fi
    # shellcheck disable=SC2086  # extra arguments are words
    (cd "$R/parity" && PYROID_PARITY_REPROS="$RES/repros" "$VENV/bin/python" -m pytest -m warc -q -p no:cacheprovider \
        --impl pyroid.warc --parity-limit 0 --parity-report "$RES/parity-report.json" "${select[@]}" ${WARC_HOOK_PYTEST_ARGS:-}) \
        >"$RES/pytest.log" 2>&1 || rc=$?
    grep -E "^(FAILED|ERROR)|passed|failed" "$RES/pytest.log" | tail -n 25 || true
    if [[ -f $RES/parity-report.json ]]; then
        (cd "$R/parity" && "$VENV/bin/python" -m pyroid_parity.report render "$RES/parity-report.json" >"$RES/parity-report.md") || true
    fi
    # Keep the results small: diff notes always, the inputs only up to 1 MB each and 50 MB in all.
    if [[ -d $RES/repros ]]; then
        find "$RES/repros" -type f ! -name '*.json' -size +1M -delete
        if (($(du -sm "$RES/repros" | cut -f1) > 50)); then find "$RES/repros" -type f ! -name '*.json' -delete; fi
    fi
    return "$rc"
}

do_scale() {
    local d=$T/whole rc=0 p f want got
    mkdir -p "$d"
    local paths=()
    while IFS= read -r p; do
        [[ -n $p ]] || continue
        f=$d/$(basename "$p")
        want=$(curl -sSfI --retry 8 --retry-delay 5 "$CC/$p" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
        curl -sSf --retry 8 --retry-delay 5 -o "$f" "$CC/$p"
        got=$(stat -c %s "$f")
        if [[ $got != "$want" ]]; then echo "$p: $got bytes, Content-Length $want"; return 1; fi
        echo "$(basename "$p"): $got bytes"
        paths+=("$f")
    done <<<"$FILES"
    (cd "$R/parity" && "$VENV/bin/python" -m pyroid_parity.warc_scale "$RES/warc-scale-extra.json" "${paths[@]}" \
        --impl pyroid.warc --workers "$NPROC" --configs content/file,content/datatrove,random/file,digests-log/file) \
        >"$RES/warc-scale-extra.log" 2>&1 || rc=$?
    tail -n 3 "$RES/warc-scale-extra.log"
    rm -rf "$d"
    return "$rc"
}

echo "warc parity hook: $NPROC CPUs (all of them; the run steps use ${CPUS:-all}), $PY, $(basename "$WHEEL")"
phase env do_env
if [[ ${STATUS_OF[env]} == ok || ${STATUS_OF[env]} == skipped ]]; then
    phase corpus do_corpus
    phase suite do_suite
    phase scale do_scale
fi
write_phases
cat "$RES/hook-phases.json"
exit "$FAILED"
