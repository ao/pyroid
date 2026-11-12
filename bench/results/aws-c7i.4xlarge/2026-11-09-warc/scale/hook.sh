#!/usr/bin/env bash
# Hook for bench/ec2/run_warc.sh: pyroid.trafilatura vs trafilatura on 100,000 pages from several crawls, per mode.
set -euo pipefail
cd "$HTMLX_REPO/parity"
for mode in 2.3.1 1.11.0; do
  if [[ $mode == 2.3.1 ]]; then grp=(--group trafilatura); extra=(); else grp=(--group trafilatura111); extra=(--compat 1.11.0); fi
  UV_PROJECT_ENVIRONMENT="$HTMLX_W/venv-scale-$mode" uv run --frozen --no-default-groups "${grp[@]}" --group cc \
    --with "$HTMLX_WHEEL" python -m pyroid_parity.trafilatura_scale "$HTMLX_HOOK_RES/scale-$mode" \
    --impl pyroid.trafilatura "${extra[@]}" --pages 100000 --workers "$(nproc)" --slice-mb 100 \
    >"$HTMLX_HOOK_RES/scale-$mode.log" 2>&1 || echo "scale $mode failed: $?" >>"$HTMLX_HOOK_RES/scale-failed.txt"
  tail -n 20 "$HTMLX_HOOK_RES/scale-$mode.log"
  rm -rf "$HTMLX_W/venv-scale-$mode"
done
