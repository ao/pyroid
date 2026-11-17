"""Django's sqlparse call sites, run once with sqlparse and once with pyroid.patch("sqlparse").

Django 6.1 calls sqlparse in ``BaseDatabaseOperations.prepare_sql_script`` (``RunSQL``
migrations, ``sqlmigrate``), ``format_debug_sql`` (the test runner's ``--debug-sql`` and
``QueryFormatter``), the SQLite introspection (constraints, index orders and collations parsed
from ``sqlite_master``, so ``inspectdb`` too) and the MySQL introspection of CHECK clauses. This
script drives each of them on a throwaway app and writes what Django returned as JSON, so two
runs compare byte for byte::

    python django_paths.py /target/cons/out/django-sqlparse.json
    python -m pyroid run -p sqlparse django_paths.py /target/cons/out/django-pyroid.json
    python compare.py json /target/cons/out/django-sqlparse.json /target/cons/out/django-pyroid.json

``run_django_paths.sh`` does all three in the ``django`` consumer venv (see ``setup.sh``),
patching through the start-up hook (``PYROID_PATCH`` and ``PYTHONPATH``, see ``_env.sh``).
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Any

APP_MODELS = """
from django.db import models
from django.db.models import Q, F
from django.db.models.functions import Lower


class Author(models.Model):
    name = models.CharField(max_length=100, db_collation="NOCASE", db_comment="who")
    email = models.EmailField(unique=True)
    age = models.PositiveIntegerField(default=0)
    rating = models.DecimalField(max_digits=5, decimal_places=2, null=True)

    class Meta:
        constraints = [
            models.CheckConstraint(condition=Q(age__lt=200), name="age_lt_200"),
            models.UniqueConstraint(Lower("name"), "email", name="lower_name_email"),
            models.UniqueConstraint(fields=["name", "age"], name="name_age",
                                    condition=Q(age__gt=10)),
        ]
        indexes = [models.Index(F("age").desc(), "name", name="age_desc_name")]


class Book(models.Model):
    title = models.CharField(max_length=200, db_index=True)
    author = models.ForeignKey(Author, models.CASCADE, related_name="books")
    coauthors = models.ManyToManyField(Author, related_name="cobooks")
    pages = models.IntegerField(db_default=1)
    published = models.DateField(null=True)
    data = models.JSONField(default=dict)
    total = models.GeneratedField(expression=F("pages") * 2, output_field=models.IntegerField(),
                                  db_persist=True)

    class Meta:
        unique_together = [("title", "author")]
        ordering = ["-published", "title"]


class Note(models.Model):  # as migration 0002 leaves it
    id = models.AutoField(primary_key=True)
    body = models.TextField(db_collation="NOCASE")
    rank = models.IntegerField(unique=True)
    flag = models.BooleanField(default=False)
"""

MIGRATION_0001 = '''
from django.db import migrations

SQL = """
-- a comment line
CREATE TABLE consapp_raw (id integer PRIMARY KEY, v text DEFAULT 'a;b' CHECK (v <> ''));
/* a block comment; with a semicolon */
INSERT INTO consapp_raw (id, v) VALUES (1, 'x -- not a comment'), (2, 'y /* nor this */');
CREATE INDEX consapp_raw_v ON consapp_raw (v DESC);
CREATE TRIGGER consapp_raw_t AFTER INSERT ON consapp_raw BEGIN
    UPDATE consapp_raw SET v = upper(v) WHERE id = new.id;
END;
"""


class Migration(migrations.Migration):
    initial = True
    operations = [
        migrations.RunSQL(SQL, reverse_sql="DROP TABLE consapp_raw; -- gone\\n"),
        migrations.RunSQL(
            ["CREATE TABLE consapp_list (a integer)", ("INSERT INTO consapp_list VALUES (%s)", [7])],
            reverse_sql=["DROP TABLE consapp_list"],
        ),
    ]
'''

MIGRATION_0002 = """
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("consapp", "0001_initial")]
    operations = [
        migrations.CreateModel(
            "Note",
            [
                ("id", models.AutoField(primary_key=True)),
                ("body", models.TextField(db_collation="RTRIM")),
                ("rank", models.IntegerField(unique=True)),
            ],
        ),
        migrations.RunSQL(
            "UPDATE consapp_note SET rank = rank + 1 WHERE body LIKE '%;%'; "
            "-- trailing\\nDELETE FROM consapp_note WHERE rank < 0;",
            migrations.RunSQL.noop,
        ),
        migrations.AddField("note", "flag", models.BooleanField(default=False)),
        migrations.AlterField("note", "body", models.TextField(db_collation="NOCASE")),
    ]
"""

SCRIPTS = [
    "",
    "   ",
    "SELECT 1",
    "SELECT 1;SELECT 2;",
    "-- only a comment",
    "/* only a block */",
    "SELECT 'a;b'; SELECT \"c;d\"; SELECT `e;f`",
    "SELECT 1; -- trailing comment\nSELECT 2 /* inline */ ;\n\n",
    "CREATE TABLE t (x int); -- c1\n-- c2\nINSERT INTO t VALUES (1);\n",
    "BEGIN;\nUPDATE t SET x = 1;\nCOMMIT;",
    "CREATE FUNCTION f() RETURNS trigger AS $$ BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql;",
    "CREATE OR REPLACE PROCEDURE p() BEGIN SELECT 1; SELECT 2; END; CALL p();",
    "DECLARE x int; BEGIN x := 1; END;",
    "SELECT * FROM t WHERE a = 'it''s; fine' AND b = E'\\\\;'; SELECT 2",
    "INSERT INTO t VALUES ('--'), ('/*'), ('*/');\n\nSELECT 3 -- end",
    "﻿SELECT 1; SELECT 'été'; SELECT 2\r\n",
    "SELECT 1\n;\n;\nSELECT 2",
    "IF EXISTS (SELECT 1) BEGIN SELECT 2; END; SELECT 3",
]

DEBUG_SQL = [
    "SELECT 1",
    (
        'SELECT "auth_user"."id", "auth_user"."username" FROM "auth_user" WHERE '
        '("auth_user"."is_active" AND "auth_user"."id" IN (SELECT U0."user_id" FROM '
        '"auth_user_groups" U0 WHERE U0."group_id" = 1)) ORDER BY "auth_user"."id" ASC LIMIT 21'
    ),
    (
        "select case when a = 1 then 'x' else 'y' end as c, count(*) over (partition by b) from t "
        "group by c having count(*) > 1 union all select 1, 2"
    ),
    "INSERT INTO t (a, b) VALUES (%s, %s) RETURNING t.id",
    "UPDATE t SET a = a + 1, b = COALESCE(b, '') WHERE id BETWEEN 1 AND 10",
    "with recursive r(n) as (select 1 union all select n + 1 from r where n < 5) select * from r",
    "SELECT " + "(" * 120 + "1" + ")" * 120,  # deeper than MAX_GROUPING_DEPTH: Django falls back
    "SELECT " + ", ".join(f"c{i}" for i in range(12000)) + " FROM t",  # over MAX_GROUPING_TOKENS
]

# CHECK clauses as MySQL's information_schema gives them (django/db/backends/mysql).
MYSQL_CHECKS = [
    ("`age` < 200", {"age"}),
    ("(`a` > 0) and (`b` <> _utf8mb4\\'x\\')", {"a", "b"}),
    ("json_valid(`data`)", {"data"}),
    ("((`x` + `y`) between 1 and 10)", {"x", "y", "z"}),
    ("`name` is not null or (`id` in (1,2,3))", {"name", "id"}),
]

RAW_TABLES = [
    (
        'CREATE TABLE "raw1" ("id" integer NOT NULL PRIMARY KEY AUTOINCREMENT, "ref" integer UNIQUE, '
        '"v" varchar(255) CHECK ("v" != \'test\'), "w" text COLLATE NOCASE, '
        'CONSTRAINT "raw1_uniq" UNIQUE ("ref", "v"), CONSTRAINT "raw1_chk" CHECK ("ref" > 0 AND "v" <> "w"))'
    ),
    (
        "CREATE TABLE raw2 (id int PRIMARY KEY NOT NULL, a int CHECK (a > 0) UNIQUE, "
        "b varchar(10) COLLATE RTRIM, c int REFERENCES raw1(id), UNIQUE (a, b), "
        "CHECK (a < 100 OR b IS NULL), CONSTRAINT named CHECK (length(b) > 2))"
    ),
    (
        "CREATE TABLE [raw 3] ([id col] integer PRIMARY KEY, `x` int, [y] int, FOREIGN KEY (x) "
        "REFERENCES raw2 (id) ON DELETE CASCADE, CONSTRAINT `xy` UNIQUE (`x`, [y]))"
    ),
    "CREATE TABLE raw4(\n    id INTEGER NOT NULL,\n    created DATE,\n    PRIMARY KEY(id)\n)",
    'CREATE INDEX "raw1_idx" ON "raw1" ("ref" DESC, "v" ASC, "w")',
    "CREATE UNIQUE INDEX raw2_idx ON raw2 (b DESC) WHERE a > 1",
    'CREATE INDEX "raw2_expr" ON "raw2" (lower("b"), "a" DESC)',
]


def write_app(root: Path) -> None:
    app = root / "consapp"
    (app / "migrations").mkdir(parents=True)
    (app / "__init__.py").write_text("")
    (app / "models.py").write_text(textwrap.dedent(APP_MODELS))
    (app / "migrations" / "__init__.py").write_text("")
    (app / "migrations" / "0001_initial.py").write_text(textwrap.dedent(MIGRATION_0001))
    (app / "migrations" / "0002_note.py").write_text(textwrap.dedent(MIGRATION_0002))
    sys.path.insert(0, str(root))


def configure(root: Path) -> None:
    import django
    from django.conf import settings

    settings.configure(
        DEBUG=True,
        SECRET_KEY="x",
        USE_TZ=True,
        DEFAULT_AUTO_FIELD="django.db.models.AutoField",
        DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": str(root / "db.sqlite3")}},
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django.contrib.sessions",
            "django.contrib.admin",
            "django.contrib.messages",
            "django.contrib.sites",
            "django.contrib.flatpages",
            "django.contrib.redirects",
            "consapp",
        ],
        # The SQLite migration of consapp 0002 rebuilds tables; keep MIGRATION_MODULES default.
    )
    django.setup()


def outcome(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - recorded: both runs must raise the same
        return {"raised": type(e).__name__, "message": str(e)}


def sqlmigrate(app: str, name: str, backwards: bool, buf: io.StringIO) -> Any:
    from django.core.management import call_command

    return call_command("sqlmigrate", app, name, backwards=backwards, stdout=buf)


def sqlmigrate_all() -> dict[str, Any]:
    from django.db import connection
    from django.db.migrations.loader import MigrationLoader

    loader = MigrationLoader(connection, ignore_no_migrations=True)
    out: dict[str, Any] = {}
    for app, name in sorted(loader.disk_migrations):
        for backwards in (False, True):
            buf = io.StringIO()
            r = outcome(sqlmigrate, app, name, backwards, buf)
            r = stable(r) if isinstance(r, str) else r  # call_command returns the SQL too
            out[f"{app}.{name}{' backwards' if backwards else ''}"] = [stable(buf.getvalue()), r]
    return out


# Values that change from run to run: timestamps, password hashes (random salts).
VOLATILE = [
    (re.compile(r"'\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d(?:\.\d+)?(?:\+00:00)?'"), "'2000-01-01 00:00:00'"),
    (re.compile(r"'(?:pbkdf2_sha256|md5|argon2)\$[^']*'"), "'<hash>'"),
]


def stable(sql: str) -> str:
    for pattern, repl in VOLATILE:
        sql = pattern.sub(repl, sql)
    return sql


def migrate_and_capture() -> list[str]:
    from django.core.management import call_command
    from django.db import connection, reset_queries

    reset_queries()
    call_command("migrate", verbosity=0, interactive=False)
    return [stable(q["sql"]) for q in connection.queries]


def orm_queries() -> list[str]:
    from consapp.models import Author, Book  # type: ignore[import-not-found]
    from django.contrib.auth.models import Group, Permission, User
    from django.db import connection, reset_queries
    from django.db.models import Count, Exists, OuterRef, Q, Subquery, Sum
    from django.db.models.functions import Coalesce, Upper

    reset_queries()
    g = Group.objects.create(name="g;1")
    u = User.objects.create_user("u1", "u1@example.com", "pw")
    u.groups.add(g)
    u.user_permissions.set(Permission.objects.filter(codename__startswith="add_")[:3])
    a = Author.objects.create(name="A -- b", email="a@example.com", age=30)
    b = Book.objects.create(title="T /* x */", author=a, data={"k": "v;w"})
    b.coauthors.add(a)
    list(User.objects.filter(Q(groups__name__icontains="g") | Q(is_staff=True)).distinct())
    list(User.objects.annotate(n=Count("groups")).filter(n__gt=0).values("username", "n"))
    list(Author.objects.annotate(pages=Coalesce(Sum("books__pages"), 0)).order_by("-pages"))
    list(Book.objects.select_related("author").prefetch_related("coauthors").filter(data__k="v;w"))
    sub = Book.objects.filter(author=OuterRef("pk")).order_by("-pages")
    list(Author.objects.annotate(top=Subquery(sub.values("title")[:1]), has=Exists(sub)))
    list(Author.objects.values(up=Upper("name")).union(Author.objects.values(up=Upper("email"))))
    Book.objects.filter(pages__lt=0).update(pages=1)
    Book.objects.filter(title__startswith="zz").delete()
    list(Permission.objects.filter(content_type__app_label__in=["auth", "consapp"])[:5])
    return [stable(q["sql"]) for q in connection.queries]


def introspection() -> dict[str, Any]:
    from django.core.management import call_command
    from django.db import connection

    out: dict[str, Any] = {}
    with connection.cursor() as cursor:
        for sql in RAW_TABLES:
            cursor.execute(sql)
        intro = connection.introspection
        for table in sorted(intro.table_names(cursor)):
            out[table] = {
                "constraints": jsonable(outcome(intro.get_constraints, cursor, table)),
                "description": jsonable(outcome(intro.get_table_description, cursor, table)),
                "relations": jsonable(outcome(intro.get_relations, cursor, table)),
                "pk": outcome(intro.get_primary_key_columns, cursor, table),
            }
    buf = io.StringIO()
    call_command("inspectdb", stdout=buf)
    out["inspectdb"] = buf.getvalue()
    return out


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted(map(str, x))
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return re.sub(r" at 0x[0-9a-f]+", "", repr(x))


def debug_sql_runner() -> str:
    """The text ``./manage.py test --debug-sql -v 2`` prints for a passing and a failing test."""
    from django.contrib.auth.models import User
    from django.test import TestCase
    from django.test.runner import DiscoverRunner

    class T(TestCase):
        def test_pass(self) -> None:
            User.objects.filter(username__in=["a", "b"]).exclude(is_active=False).count()

        def test_fail(self) -> None:
            list(User.objects.filter(groups__name="x").order_by("-date_joined")[:3])
            self.assertEqual(1, 2)

    runner = DiscoverRunner(debug_sql=True, verbosity=2, interactive=False)
    runner.setup_test_environment()
    old = runner.setup_databases()
    try:
        kwargs = runner.get_test_runner_kwargs()
        stream = io.StringIO()
        kwargs["stream"] = stream
        suite = unittest.TestSuite([T("test_pass"), T("test_fail")])
        unittest.TextTestRunner(**kwargs).run(suite)
    finally:
        runner.teardown_databases(old)
        runner.teardown_test_environment()
    text = stream.getvalue()
    text = re.sub(r"Ran (\d+) tests? in [\d.]+s", r"Ran \1 tests", text)
    return re.sub(r'File ".*?", line \d+', "File ...", text)


def mysql_checks() -> Any:
    try:
        import pymysql
    except ImportError:
        return "pymysql is not installed"
    pymysql.version_info = (2, 2, 7, "final", 0)  # what Django's mysql backend checks
    pymysql.install_as_MySQLdb()
    from django.db.backends.mysql.introspection import DatabaseIntrospection

    intro = DatabaseIntrospection(None)
    return [jsonable(outcome(intro._parse_constraint_columns, c, cols)) for c, cols in MYSQL_CHECKS]


def main() -> None:
    out_path = Path(sys.argv[1])
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_app(root)
        configure(root)
        import sqlparse.engine.grouping
        from django.db import connection

        import sqlparse

        res: dict[str, Any] = {
            "functions": {
                "sqlparse.format": sqlparse.format.__module__,
                "sqlparse.split": sqlparse.split.__module__,
                "sqlparse.parse": sqlparse.parse.__module__,
                "sqlparse.engine.grouping.group": sqlparse.engine.grouping.group.__module__,
            }
        }
        from django.core.management import call_command

        ops = connection.ops
        # Author and Book: a 0003 migration written by makemigrations.
        call_command("makemigrations", "consapp", name="models", verbosity=0)
        res["sqlmigrate"] = sqlmigrate_all()
        executed = migrate_and_capture()
        res["migrate_sql"] = executed
        executed += orm_queries()
        res["orm_sql"] = executed
        migration_script = ";\n".join(s for v in res["sqlmigrate"].values() for s in [v[0]])
        res["prepare_sql_script"] = [outcome(ops.prepare_sql_script, s) for s in [*SCRIPTS, migration_script]]
        res["format_debug_sql"] = [outcome(ops.format_debug_sql, s) for s in [*DEBUG_SQL, *executed]]
        res["introspection"] = introspection()
        res["debug_sql_runner"] = debug_sql_runner()
        res["mysql_checks"] = mysql_checks()
    res["calls"] = {k: len(v) for k, v in res.items() if isinstance(v, (list, dict))}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1, sort_keys=True, ensure_ascii=False))
    print(
        f"{out_path}: format via {res['functions']['sqlparse.format']}, "
        f"{res['calls']}, {time.perf_counter() - t0:.1f} s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    os.environ.pop("DJANGO_SETTINGS_MODULE", None)
    main()
