#!/usr/bin/env bash
# django-debug-toolbar 8.0.0: its SQL panel formatting (toolbar_paths.py) and its own test suite,
# with sqlparse and patched, compared. Needs `setup.sh toolbar` and /target/djangosql.jsonl
# (django_sql.py generate).
#   bash bench/sqlparse/consumers/run_toolbar.sh [paths|suite]   (in the dev container, from /work)
set -euo pipefail
WHAT=${1:-paths suite}
PY=/target/cons/toolbar/venv/bin/python
cd /work/bench/sqlparse/consumers
. ./_env.sh
HERE=$PWD
if [[ $WHAT == *paths* ]]; then
  export PROBE_LOG=$OUT/toolbar-paths.probe; rm -f $PROBE_LOG
  timed sqlparse plain "$PY" toolbar_paths.py /target/djangosql.jsonl $OUT/toolbar-sqlparse.json ${EVERY:-10}
  timed pyroid patched "$PY" toolbar_paths.py /target/djangosql.jsonl $OUT/toolbar-pyroid.json ${EVERY:-10}
  "$PY" compare.py probe $PROBE_LOG
  "$PY" compare.py json $OUT/toolbar-sqlparse.json $OUT/toolbar-pyroid.json || true
fi
if [[ $WHAT == *suite* ]]; then
  cd /target/src/django-debug-toolbar-8.0.0/django_debug_toolbar-8.0.0
  for mode in sqlparse pyroid; do
    export PROBE_LOG=$OUT/toolbar-suite-$mode.probe; rm -f $PROBE_LOG
    run=(plain); [ $mode = pyroid ] && run=(patched)
    PYTHONPATH=$PWD DJANGO_SETTINGS_MODULE=tests.settings timed "toolbar-suite-$mode" \
      "${run[@]}" "$PY" -b -m django test -v 2 tests > $OUT/toolbar-suite-$mode.log 2>&1
    grep -E "^Ran |^OK|^FAILED|^TIME" $OUT/toolbar-suite-$mode.log | tail -3
    "$PY" $HERE/compare.py probe $PROBE_LOG
  done
  "$PY" $HERE/compare.py unittest $OUT/toolbar-suite-sqlparse.log $OUT/toolbar-suite-pyroid.log || true
fi
