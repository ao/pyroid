#!/usr/bin/env bash
# Django's sqlparse call sites (django_paths.py) with sqlparse, then patched, compared byte for
# byte. Needs `setup.sh django`.
#   bash bench/sqlparse/consumers/run_django_paths.sh   (in the dev container, from /work)
set -euo pipefail
PY=/target/cons/django/venv/bin/python
cd /work/bench/sqlparse/consumers
. ./_env.sh
export PROBE_LOG=$OUT/django-paths.probe
rm -f $PROBE_LOG
timed sqlparse plain "$PY" django_paths.py $OUT/django-sqlparse.json > $OUT/django-paths-sqlparse.log 2>&1
timed pyroid patched "$PY" django_paths.py $OUT/django-pyroid.json > $OUT/django-paths-pyroid.log 2>&1
"$PY" compare.py probe $PROBE_LOG
tail -1 $OUT/django-paths-sqlparse.log; tail -1 $OUT/django-paths-pyroid.log
grep -c PatchWarning $OUT/django-paths-pyroid.log || true
"$PY" compare.py json $OUT/django-sqlparse.json $OUT/django-pyroid.json
