#!/usr/bin/env bash
# The parts of Django's own test suite that reach sqlparse, on SQLite, with sqlparse (MODE=sqlparse)
# or patched (MODE=pyroid); the -v 2 log goes to /target/cons/out/django-suite-MODE.log. Run both
# (one call each: a run takes a few minutes), then compare:
#   bash bench/sqlparse/consumers/run_django_suite.sh sqlparse   (in the dev container, from /work)
#   bash bench/sqlparse/consumers/run_django_suite.sh pyroid   (in the dev container, from /work)
#   bash bench/sqlparse/consumers/run_django_suite.sh compare   (in the dev container, from /work)
# Needs `setup.sh django`. LABELS overrides the test labels.
set -euo pipefail
MODE=${1:?sqlparse, pyroid or compare}
PY=/target/cons/django/venv/bin/python
cd /work/bench/sqlparse/consumers
. ./_env.sh
if [ "$MODE" = compare ]; then
  exec "$PY" compare.py unittest $OUT/django-suite-sqlparse.log $OUT/django-suite-pyroid.log
fi
# Every app whose tests or code paths call sqlparse (grep -rn sqlparse django/ tests/).
LABELS=${LABELS:-"backends migrations test_runner dbshell introspection inspectdb schema \
model_options admin_scripts bash_completion constraints indexes"}
cd /target/src/django-6.1.3/django-6.1.3/tests
export PROBE_LOG=$OUT/django-suite-$MODE.probe
rm -f $PROBE_LOG
run=(plain "$PY" runtests.py --settings=test_sqlite --parallel=16 -v 2 $LABELS)
[ "$MODE" = pyroid ] && run=(patched "${run[@]:1}")
timed "django-suite-$MODE" "${run[@]}" > $OUT/django-suite-$MODE.log 2>&1
grep -E "^Ran |^OK|^FAILED|^TIME" $OUT/django-suite-$MODE.log | tail -3
"$PY" /work/bench/sqlparse/consumers/compare.py probe $PROBE_LOG
