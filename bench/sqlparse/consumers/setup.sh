#!/usr/bin/env bash
# Build the consumer venvs on the devbox (python 3.12): /target/cons/<name>/venv, each with
# sqlparse 0.6.1, the consumer, its test dependencies and pyroid from the worktree (/work/python,
# through a .pth file: build it first with `maturin develop --uv` in /target/venv). Sources
# (the test suites) are the consumers' sdists, unpacked in /target/src by fetch.py.
#
#   bash bench/sqlparse/consumers/setup.sh django toolbar pgcli mycli litecli sqllineage   (in the dev container, from /work)
set -euo pipefail
cd /work
HERE=/work/bench/sqlparse/consumers
CONS=/target/cons
SRC=/target/src
export UV_CACHE_DIR=${UV_CACHE_DIR:-/target/uv-cache}

DJANGO=django==6.1.3
TOOLBAR=django-debug-toolbar==8.0.0
PGCLI=pgcli==4.7.1
MYCLI=mycli==2.31.0
LITECLI=litecli==1.17.1
SQLLINEAGE=sqllineage==1.5.9

ls /work/python/pyroid/_pyroid*.so > /dev/null  # pyroid must be built in the worktree

venv() {  # venv NAME -> prints the python path
  local v=$CONS/$1/venv
  [ -x "$v/bin/python" ] || uv venv -q -p 3.12 "$v" >&2
  echo /work/python > "$v/lib/python3.12/site-packages/_pyroid_worktree.pth"
  echo "$v/bin/python"
}
src() { python -I "$HERE/fetch.py" "$1" | sed 's/.* -> //'; }

for name in "$@"; do
  echo "== $name"
  case $name in
    django)
      py=$(venv django); d=$(src $DJANGO)
      # The Django test suite's requirements that the sqlparse-related apps need, and the
      # drivers django_sql.py imports to compile SQL for PostgreSQL, MySQL and Oracle.
      uv pip install -q --python "$py" -e "$d" sqlparse==0.6.1 asgiref tblib pyyaml pillow jinja2 \
        tzdata docutils argon2-cffi bcrypt numpy black 'psycopg[binary]' pymysql oracledb ;;
    toolbar)
      py=$(venv toolbar); d=$(src $TOOLBAR)
      uv pip install -q --python "$py" $DJANGO sqlparse==0.6.1 "$d" django-csp \
        django-template-partials html5lib jinja2 pygments whitenoise ;;
    pgcli)
      py=$(venv pgcli); d=$(src $PGCLI)
      uv pip install -q --python "$py" "$d" sqlparse==0.6.1 pytest mock 'psycopg[binary]' \
        keyrings.alt sshtunnel pexpect ;;
    mycli)
      py=$(venv mycli); d=$(src $MYCLI)
      uv pip install -q --python "$py" "$d" sqlparse==0.6.1 'pytest~=9.2.0' pexpect ;;
    litecli)
      py=$(venv litecli); d=$(src $LITECLI)
      uv pip install -q --python "$py" "$d" sqlparse==0.6.1 pytest pexpect mock ;;
    sqllineage)
      # The sdist builds sqllineage's JavaScript front end (npm): install the wheel, test the sdist.
      py=$(venv sqllineage); src $SQLLINEAGE > /dev/null
      uv pip install -q --python "$py" $SQLLINEAGE sqlparse==0.6.1 pytest ;;
    *) echo "unknown consumer $name" >&2; exit 2 ;;
  esac
  "$py" -c "import sqlparse, pyroid; print('sqlparse', sqlparse.__version__, 'pyroid', pyroid.__version__)"
done
