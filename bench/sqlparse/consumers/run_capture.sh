#!/usr/bin/env bash
# Capture what a consumer's test run feeds sqlparse (unpatched), then replay it through sqlparse
# and pyroid.sqlparse (replay.py). Consumers: django, toolbar, pgcli, mycli, litecli, sqllineage.
#   bash bench/sqlparse/consumers/run_capture.sh pgcli   (in the dev container, from /work)
set -euo pipefail
NAME=${1:?consumer}
PY=/target/cons/$NAME/venv/bin/python
HERE=/work/bench/sqlparse/consumers
cd $HERE
. ./_env.sh
# One process at a time where there is a choice: appends of long lines from several could interleave.
export PROBE_CAPTURE=$OUT/$NAME.capture.jsonl PROBE_LOG=/dev/null
rm -f $PROBE_CAPTURE
case $NAME in
  django)
    cd /target/src/django-6.1.3/django-6.1.3/tests
    plain "$PY" runtests.py --settings=test_sqlite --parallel=1 backends migrations test_runner \
      dbshell introspection inspectdb schema model_options admin_scripts bash_completion \
      constraints indexes > $OUT/$NAME-capture.log 2>&1 || true
    cd $HERE; plain "$PY" django_paths.py $OUT/django-capture.json >> $OUT/$NAME-capture.log 2>&1 ;;
  toolbar)
    cd /target/src/django-debug-toolbar-8.0.0/django_debug_toolbar-8.0.0
    PYTHONPATH=$PWD DJANGO_SETTINGS_MODULE=tests.settings plain "$PY" -m django test tests \
      > $OUT/$NAME-capture.log 2>&1 || true ;;
  pgcli) cd /target/src/pgcli-4.7.1/pgcli-4.7.1
    plain timeout 280 "$PY" -m pytest -p no:cacheprovider -q tests --ignore=tests/features > $OUT/$NAME-capture.log 2>&1 || true ;;
  mycli) cd /target/src/mycli-2.31.0/mycli-2.31.0
    plain timeout 280 "$PY" -m pytest -p no:cacheprovider -q mycli_test/pytests -o addopts= > $OUT/$NAME-capture.log 2>&1 || true ;;
  litecli) cd /target/src/litecli-1.17.1/litecli-1.17.1
    plain timeout 280 "$PY" -m pytest -p no:cacheprovider -q tests > $OUT/$NAME-capture.log 2>&1 || true ;;
  sqllineage) cd /target/src/sqllineage-1.5.9/sqllineage-1.5.9
    plain timeout 280 "$PY" -m pytest -p no:cacheprovider -q tests > $OUT/$NAME-capture.log 2>&1 || true ;;
esac
tail -n 2 $OUT/$NAME-capture.log
cd $HERE
"$PY" replay.py $PROBE_CAPTURE
