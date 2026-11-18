#!/usr/bin/env bash
# A consumer's own pytest suite with sqlparse and patched, compared test by test (JUnit XML).
#   bash bench/sqlparse/consumers/run_pytest.sh pgcli   (in the dev container, from /work)
# Consumers: pgcli, mycli, litecli, sqllineage (each needs `setup.sh NAME`). Extra arguments go
# to pytest (e.g. -x, -k ...).
set -euo pipefail
NAME=${1:?pgcli, mycli, litecli or sqllineage}; shift
PY=/target/cons/$NAME/venv/bin/python
HERE=/work/bench/sqlparse/consumers
cd $HERE
. ./_env.sh
case $NAME in
  # tests/features are behave scenarios against a live server; the DB tests skip without one.
  pgcli) DIR=/target/src/pgcli-4.7.1/pgcli-4.7.1; ARGS=(tests --ignore=tests/features) ;;
  # pytest-random-order is not installed: plain order, so both runs see the same sequence.
  mycli) DIR=/target/src/mycli-2.31.0/mycli-2.31.0; ARGS=(mycli_test/pytests -o addopts=) ;;
  litecli) DIR=/target/src/litecli-1.17.1/litecli-1.17.1; ARGS=(tests) ;;
  sqllineage) DIR=/target/src/sqllineage-1.5.9/sqllineage-1.5.9; ARGS=(tests) ;;
  *) echo "unknown consumer $NAME" >&2; exit 2 ;;
esac
cd $DIR
for mode in sqlparse pyroid; do
  export PROBE_LOG=$OUT/$NAME-$mode.probe; rm -f $PROBE_LOG
  run=(plain); [ $mode = pyroid ] && run=(patched)
  timed "$NAME-$mode" "${run[@]}" timeout 280 "$PY" -m pytest -p no:cacheprovider -q -rfE \
    --junitxml=$OUT/$NAME-$mode.xml "${ARGS[@]}" "$@" > $OUT/$NAME-$mode.log 2>&1
  tail -1 $OUT/$NAME-$mode.log
  "$PY" $HERE/compare.py probe $PROBE_LOG
done
"$PY" $HERE/compare.py junit $OUT/$NAME-sqlparse.xml $OUT/$NAME-pyroid.xml || true
