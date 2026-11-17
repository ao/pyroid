# Sourced by the run_*.sh scripts (after setting PY).
#   patched CMD...   run CMD with pyroid.patch("sqlparse") applied at start-up in it and in its
#                    child processes: the PYROID_PATCH + PYTHONPATH hook that `pyroid run` sets
#                    up (`pyroid run` itself imports pyarrow, which the consumer venvs lack)
#   plain CMD...     run CMD as is (sqlparse's own functions)
# Both put probe/ on PYTHONPATH: each process logs which sqlparse it used, and patched ones how
# many calls reached pyroid's native core, to $PROBE_LOG (summary: compare.py probe LOG).
#   timed LABEL CMD...   print the wall time of CMD
HOOK=$("$PY" -c 'import pyroid.integrations._patch as p; print(p.HOOK_DIR)')
PROBE=/work/bench/sqlparse/consumers/probe
OUT=/target/cons/out
mkdir -p $OUT
# Django builds some IN (...) lists from sets: the same order in both runs.
export PYTHONHASHSEED=0
patched() { PYROID_PATCH=sqlparse PYTHONPATH="$HOOK:$PROBE${PYTHONPATH:+:$PYTHONPATH}" "$@"; }
plain() { PYTHONPATH="$PROBE${PYTHONPATH:+:$PYTHONPATH}" "$@"; }
timed() {
  local label=$1; shift
  local s e rc=0
  s=$(date +%s%N); "$@" || rc=$?; e=$(date +%s%N)
  echo "TIME $label: $(( (e - s) / 1000000 )) ms (exit $rc)"
  return 0
}
