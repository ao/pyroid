"""django-debug-toolbar's SQL panel formatting, with sqlparse or with pyroid.patch("sqlparse").

The SQL panel formats every captured query with ``debug_toolbar.panels.sql.utils.reformat_sql``:
sqlparse's own ``FilterStack`` with grouping, ``AlignedIndentFilter`` and the toolbar's
``BoldKeywordFilter`` / ``EscapedStringSerializer`` (and, for the collapsed view, a
``ElideSelectListsFilter`` token-stream filter without grouping). Under the patch the stack's
``lexer.tokenize`` and ``grouping.group`` are pyroid's, the filters stay Python. This script
formats a sample of the Django SQL corpus (``django_sql.py generate``) both ways, with
``PRETTIFY_SQL`` on and off, and writes the results as JSON::

    python toolbar_paths.py /target/djangosql.jsonl OUT.json [EVERY]

``run_toolbar.sh`` runs it with and without the patch and compares.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EXTRA = [
    "SELECT 1",
    "SELECT a.b, c FROM t1 a INNER JOIN t2 ON a.id = t2.a_id WHERE c IN (1, 2) ORDER BY c DESC",
    "SELECT COUNT(*) AS `__count` FROM `auth_user`",
    "SELECT (1) AS `a` FROM `auth_user` WHERE `id` = 1 LIMIT 1",
    "select x.a from x union select y.b from y except select z.c from z",
    "SELECT '<script>alert(1)</script>' AS \"x & y\" FROM t WHERE a < 1 AND b > 2",
    "SELECT " + "(" * 150 + "1" + ")" * 150,
    "SELECT " + ", ".join(f"t.c{i}" for i in range(11000)) + " FROM t",
    "INSERT INTO t (a) VALUES ('é'), ('日本'), ('\U0001f600');",
]


def main() -> None:
    corpus, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    import django
    from django.conf import settings

    settings.configure(
        DEBUG=True,
        SECRET_KEY="x",
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django.contrib.staticfiles",
            "debug_toolbar",
        ],
        STATIC_URL="/static/",
        DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
        DEBUG_TOOLBAR_CONFIG={"PRETTIFY_SQL": True},
    )
    django.setup()
    from debug_toolbar.panels.sql import utils
    from django.test.utils import override_settings

    import sqlparse

    sqls = sorted({json.loads(line)["sql"] for line in corpus.open()})[::every] + EXTRA
    t0 = time.perf_counter()
    res: dict[str, object] = {
        "functions": {
            "group": sqlparse.engine.grouping.group.__module__,
            "tokenize": sqlparse.lexer.tokenize.__module__,
        }
    }
    res["reformat"] = [utils.reformat_sql(s, with_toggle=True) for s in sqls]
    with override_settings(DEBUG_TOOLBAR_CONFIG={"PRETTIFY_SQL": False}):
        res["no_prettify"] = [utils.reformat_sql(s) for s in sqls[::5]]
    res["calls"] = {"statements": len(sqls)}
    out_path.write_text(json.dumps(res, ensure_ascii=False, sort_keys=True))
    print(
        f"{out_path}: {len(sqls)} statements, group from {res['functions']['group']}, "
        f"{time.perf_counter() - t0:.1f} s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
