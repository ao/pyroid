"""A corpus of the SQL Django generates, and a parity check of pyroid.sqlparse on it.

``generate`` loads the models of Django's own test suite (every app under ``tests/`` that has
models, as ``runtests.py`` installs them) and compiles, for SQLite, PostgreSQL, MySQL, MariaDB
and Oracle: the schema (``schema_editor(collect_sql=True)``: CREATE TABLE / INDEX, constraints,
foreign keys), SELECT / UPDATE / DELETE / INSERT queries for every model, the flush and
sequence-reset SQL, and the migrations of the contrib apps and of ``tests/migrations``. No
database server is needed: the server versions are faked (``pg_version``,
``mysql_server_data``, ``oracle_version``) and nothing is executed. Statements are written as
JSON Lines ``{"backend", "source", "sql"}``, plus one ``"script"`` record per migration plan::

    python django_sql.py generate --tests /target/src/django-6.1.3/django-6.1.3/tests \\
        --out /target/djangosql.jsonl

``check`` runs each distinct statement through sqlparse 0.6.1 and pyroid.sqlparse and counts
the matches: ``lexer.tokenize`` (token types and values), ``parse`` (the whole tree: classes,
token types, values, parent links), ``engine.grouping.group`` on sqlparse's own ungrouped
statements (``StatementSplitter``), ``split``, and ``format`` with the default options, ``reindent=True``,
``keyword_case="upper"`` and ``strip_comments=True`` (results or exceptions)::

    python django_sql.py check /target/djangosql.jsonl [--workers 16]

Needs Django, sqlparse 0.6.1, pyroid, and for the other backends psycopg, PyMySQL and
python-oracledb (``consumers/setup.sh django`` installs them).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------------- generate

SKIP_DIRS = {"import_error_package", "test_runner_apps", "gis_tests"}
ALWAYS_INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sites",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.admin.apps.SimpleAdminConfig",
    "django.contrib.staticfiles",
    "django.contrib.flatpages",
    "django.contrib.redirects",
    "django.contrib.postgres",
]
CONTRIB_MIGRATIONS = ["admin", "auth", "contenttypes", "sessions", "sites", "flatpages", "redirects"]

FAKE_SERVERS: dict[str, dict[str, Any]] = {
    "postgresql": {"pg_version": 180000},
    "mysql": {
        "mysql_server_data": {
            "version": "8.4.3",
            "sql_mode": "STRICT_TRANS_TABLES,NO_ENGINE_SUBSTITUTION",
            "default_storage_engine": "InnoDB",
            "sql_auto_is_null": False,
            "lower_case_table_names": False,
            "has_zoneinfo_database": True,
        }
    },
    "mariadb": {
        "mysql_server_data": {
            "version": "11.4.4-MariaDB",
            "sql_mode": "STRICT_TRANS_TABLES",
            "default_storage_engine": "InnoDB",
            "sql_auto_is_null": False,
            "lower_case_table_names": False,
            "has_zoneinfo_database": True,
        }
    },
    "oracle": {"oracle_version": (23, 5)},
}


def test_apps(tests: Path) -> list[str]:
    return [
        d.name
        for d in sorted(tests.iterdir())
        if d.is_dir()
        and "." not in d.name
        and d.name not in SKIP_DIRS
        and (d / "__init__.py").exists()
        and ((d / "models.py").exists() or (d / "models").is_dir())
    ]


def configure(tests: Path) -> None:
    import django
    from django.conf import settings

    try:  # PyMySQL as MySQLdb, with the version Django's mysql backend requires
        import pymysql

        pymysql.version_info = (2, 2, 7, "final", 0)
        pymysql.install_as_MySQLdb()
    except ImportError:
        pass
    sys.path.insert(0, str(tests))
    databases: dict[str, Any] = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
    for alias, engine in [
        ("postgresql", "postgresql"),
        ("mysql", "mysql"),
        ("mariadb", "mysql"),
        ("oracle", "oracle"),
    ]:
        databases[alias] = {"ENGINE": f"django.db.backends.{engine}", "NAME": "corpus"}
    settings.configure(
        DEBUG=False,
        SECRET_KEY="corpus",
        USE_TZ=False,
        SITE_ID=1,
        DEFAULT_AUTO_FIELD="django.db.models.AutoField",
        DATABASES=databases,
        INSTALLED_APPS=ALWAYS_INSTALLED_APPS + test_apps(tests),
        MIGRATION_MODULES={},
    )
    django.setup()


def backends() -> Iterator[tuple[str, Any]]:
    from django.db import connections

    for alias in ["default", "postgresql", "mysql", "mariadb", "oracle"]:
        try:
            conn = connections[alias]
        except Exception as e:  # noqa: BLE001 - a missing driver
            print(f"skipping {alias}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        conn.__dict__.update(FAKE_SERVERS.get(alias, {}))
        fake_connection(alias, conn)
        yield ("sqlite" if alias == "default" else alias), conn


def fake_connection(alias: str, conn: Any) -> None:
    """Answer, without a server, the questions the compilers and schema editors ask one."""
    if alias == "oracle":  # LIKE operators: Django asks the server which ones it supports
        conn.__dict__["operators"] = conn._standard_operators
    if conn.vendor == "mysql":
        from django.db.backends.mysql.schema import DatabaseSchemaEditor
        from pymysql.converters import escape_item

        def quote_value(self: Any, value: Any) -> str:
            q = escape_item(value, "utf8mb4")
            return q.decode() if isinstance(q, bytes) else q

        DatabaseSchemaEditor.quote_value = quote_value  # type: ignore[method-assign]
        conn.introspection.get_storage_engine = lambda cursor, table: "InnoDB"
        conn.cursor = lambda: None  # only get_storage_engine's argument


def schema_sql(conn: Any, models: list[Any], errors: Counter[str]) -> list[str]:
    with conn.schema_editor(collect_sql=True, atomic=False) as editor:
        for model in models:
            try:
                editor.create_model(model)
            except Exception as e:  # noqa: BLE001 - counted, and the corpus goes on
                errors[f"create_model {type(e).__name__}"] += 1
    return list(editor.collected_sql)


def query_sql(alias: str, conn: Any, model: Any, errors: Counter[str]) -> Iterator[tuple[str, str]]:
    from django.db.models import Count, F, Max, Q
    from django.db.models.sql import DeleteQuery, InsertQuery, UpdateQuery

    opts = model._meta
    qs = model._default_manager.using(alias).all()
    fields = [f for f in opts.concrete_fields if not f.primary_key]
    fks = [f.name for f in fields if f.is_relation][:3]
    variants: list[tuple[str, Any]] = [
        ("all", lambda: qs),
        ("filter", lambda: qs.filter(pk=1)),
        ("in_order_slice", lambda: qs.filter(pk__in=[1, 2, 3]).order_by("-pk")[5:15]),
        ("annotate", lambda: qs.values("pk").annotate(n=Count("pk"), m=Max("pk"))),
        ("q", lambda: qs.filter(Q(pk__gt=1) | ~Q(pk__lt=10)).distinct()),
        ("select_related", lambda: qs.select_related(*fks) if fks else qs.none()),
        ("exists", lambda: qs.filter(pk=1).query.exists(limit=True)),
    ]
    for name, make in variants:
        try:
            q = make()
            query = q.query if hasattr(q, "query") else q
            sql, _ = query.get_compiler(using=alias).as_sql()
            if sql:
                yield f"select:{name}", sql
        except Exception as e:  # noqa: BLE001 - counted, and the corpus goes on
            errors[f"select:{name} {type(e).__name__}"] += 1
    try:
        uq = qs.filter(pk__gte=1).query.chain(UpdateQuery)
        if fields:
            uq.add_update_values({fields[0].name: F(fields[0].name)})
            yield "update", uq.get_compiler(using=alias).as_sql()[0]
    except Exception as e:  # noqa: BLE001 - counted, and the corpus goes on
        errors[f"update {type(e).__name__}"] += 1
    try:
        dq = qs.filter(pk__in=[1, 2]).query.chain(DeleteQuery)
        yield "delete", dq.get_compiler(using=alias).as_sql()[0][0]
    except Exception as e:  # noqa: BLE001 - counted, and the corpus goes on
        errors[f"delete {type(e).__name__}"] += 1
    try:
        iq = InsertQuery(model)
        ins = [f for f in opts.local_concrete_fields if not f.generated and f is not opts.auto_field]
        iq.insert_values(ins, [model()])
        for sql, _ in iq.get_compiler(using=alias).as_sql():
            yield "insert", sql
    except Exception as e:  # noqa: BLE001 - counted, and the corpus goes on
        errors[f"insert {type(e).__name__}"] += 1


def migration_sql(conn: Any, errors: Counter[str]) -> Iterator[tuple[str, list[str]]]:
    """``sqlmigrate`` for every migration of the contrib apps and of tests/migrations' modules."""
    from django.apps import apps
    from django.db.migrations.loader import MigrationLoader
    from django.test.utils import override_settings

    mig_dir = Path(apps.get_app_config("migrations").path)
    groups: list[tuple[str, dict[str, str]]] = [("contrib", {})] + [
        (d.name, {"migrations": f"migrations.{d.name}"})
        for d in sorted(mig_dir.iterdir())
        if d.is_dir() and d.name.startswith("test_migrations") and (d / "__init__.py").exists()
    ]
    for group, modules in groups:
        try:
            with override_settings(MIGRATION_MODULES=modules):
                loader = MigrationLoader(None, ignore_no_migrations=True)
                labels = CONTRIB_MIGRATIONS if group == "contrib" else ["migrations"]
                targets = [n for n in loader.graph.leaf_nodes() if n[0] in labels]
                plan: list[Any] = []
                for t in targets:
                    plan += [n for n in loader.graph.forwards_plan(t) if n not in plan and n[0] in labels]
        except Exception as e:  # noqa: BLE001 - counted, and the corpus goes on
            errors[f"migration-graph {type(e).__name__}"] += 1
            continue
        statements: list[str] = []
        state = None
        for key in plan:
            migration = loader.graph.nodes[key]
            if state is None:
                state = loader.project_state(key, at_end=False)
            try:
                with conn.schema_editor(collect_sql=True, atomic=False) as editor:
                    new_state = migration.apply(state.clone(), editor, collect_sql=True)
                statements += editor.collected_sql
                state = new_state
            except Exception as e:  # noqa: BLE001 - needs a live database: keep the state going
                errors[f"migration {type(e).__name__}"] += 1
                try:
                    state = migration.mutate_state(state, preserve=True)
                except Exception:  # noqa: BLE001 - a deliberately broken test migration: drop the rest
                    errors["migration-state"] += 1
                    break
        yield group, statements


def generate(tests: Path, out: Path) -> None:
    t0 = time.perf_counter()
    configure(tests)
    from django.apps import apps
    from django.core.management.color import no_style

    models = [
        m
        for m in apps.get_models(include_auto_created=True)
        if m._meta.managed and not m._meta.proxy and not m._meta.swapped
    ]
    counts: Counter[str] = Counter()
    errors: dict[str, Counter[str]] = defaultdict(Counter)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:

        def emit(backend: str, source: str, sql: str) -> None:
            f.write(json.dumps({"backend": backend, "source": source, "sql": sql}) + "\n")
            counts[backend] += 1

        for backend, conn in backends():
            alias = conn.alias
            err = errors[backend]
            for sql in schema_sql(conn, models, err):
                emit(backend, "schema", sql)
            for model in models:
                for kind, sql in query_sql(alias, conn, model, err):
                    emit(backend, f"{kind}:{model._meta.label}", sql)
            tables = sorted({m._meta.db_table for m in models})
            try:
                for sql in conn.ops.sql_flush(no_style(), tables, reset_sequences=True, allow_cascade=True):
                    emit(backend, "flush", sql)
                for sql in conn.ops.sequence_reset_sql(no_style(), models):
                    emit(backend, "sequence_reset", sql)
            except Exception as e:  # noqa: BLE001 - counted, and the corpus goes on
                err[f"flush {type(e).__name__}"] += 1
            for group, statements in migration_sql(conn, err):
                for sql in statements:
                    emit(backend, f"migration:{group}", sql)
                if statements:
                    emit(backend, f"script:{group}", "\n".join(statements))
    print(f"{out}: {sum(counts.values())} statements, {len(models)} models, {dict(counts)}")
    for backend, err in errors.items():
        if err:
            print(f"  {backend} errors (skipped): {dict(err.most_common(8))}")
    print(f"  {time.perf_counter() - t0:.1f} s")


# ------------------------------------------------------------------------------------- check

FORMATS: dict[str, dict[str, Any]] = {
    "format": {},
    "format_reindent": {"reindent": True},
    "format_upper": {"keyword_case": "upper"},
    "format_strip_comments": {"strip_comments": True},
}
OPS = ["tokenize", "parse", "group", "split", *FORMATS]


def shape(tok: Any) -> Any:
    if tok.is_group:
        if not all(t.parent is tok for t in tok.tokens):
            raise AssertionError(f"a child of {type(tok).__name__} has another parent")
        return [type(tok).__name__, [shape(t) for t in tok.tokens]]
    return [type(tok).__name__, str(tok.ttype), tok.value]


def run_op(mod: Any, op: str, sql: str) -> Any:
    try:
        if op == "tokenize":
            return [[str(t), v] for t, v in mod.lexer.tokenize(sql)]
        if op == "parse":
            return [shape(s) for s in mod.parse(sql)]
        if op == "group":  # sqlparse's ungrouped statements (Python tokens), grouped by `mod`
            from sqlparse.engine.statement_splitter import StatementSplitter

            import sqlparse

            stmts = StatementSplitter().process(sqlparse.lexer.tokenize(sql))
            return [shape(mod.engine.grouping.group(s)) for s in stmts]
        if op == "split":
            return mod.split(sql)
        return mod.format(sql, **FORMATS[op])
    except Exception as e:  # noqa: BLE001 - the exception is part of the compared behaviour
        return ["raised", type(e).__name__, str(e)]


def check_chunk(sqls: list[str]) -> dict[str, Any]:
    import pyroid.sqlparse as psp

    import sqlparse

    assert sqlparse.__version__ == "0.6.1", sqlparse.__version__
    assert not sqlparse.format.__module__.startswith("pyroid"), "run check without the patch"
    same: Counter[str] = Counter()
    raised: Counter[str] = Counter()
    secs: dict[str, list[float]] = {op: [0.0, 0.0] for op in OPS}
    diffs: dict[str, list[Any]] = defaultdict(list)
    for sql in sqls:
        for op in OPS:
            t = time.perf_counter()
            a = run_op(sqlparse, op, sql)
            t2 = time.perf_counter()
            b = run_op(psp, op, sql)
            t3 = time.perf_counter()
            secs[op][0] += t2 - t
            secs[op][1] += t3 - t2
            if a == b:
                same[op] += 1
                if isinstance(a, list) and a[:1] == ["raised"]:
                    raised[op] += 1
            elif len(diffs[op]) < 5:
                diffs[op].append(
                    {"sql": sql[:2000], "len": len(sql), "sqlparse": str(a)[:600], "pyroid": str(b)[:600]}
                )
            else:
                diffs[op].append(None)
    return {"n": len(sqls), "same": same, "raised": raised, "secs": secs, "diffs": diffs}


def check(path: Path, workers: int, shard: str | None = None) -> int:
    t0 = time.perf_counter()
    records = [json.loads(line) for line in path.open()]
    by_backend: dict[str, set[str]] = defaultdict(set)
    for r in records:
        by_backend[r["backend"]].add(r["sql"])
    sqls = sorted({r["sql"] for r in records}, key=len, reverse=True)  # long ones first
    if shard:  # "K/N": every Nth distinct statement, from the Kth
        k, n_shards = map(int, shard.split("/"))
        sqls = sqls[k::n_shards]
    chunks = [sqls[i :: max(1, workers * 8)] for i in range(max(1, workers * 8))]
    same: Counter[str] = Counter()
    raised: Counter[str] = Counter()
    secs = {op: [0.0, 0.0] for op in OPS}
    diffs: dict[str, list[Any]] = defaultdict(list)
    with ProcessPoolExecutor(workers) as pool:
        for res in pool.map(check_chunk, [c for c in chunks if c]):
            same.update(res["same"])
            raised.update(res["raised"])
            for op, (a, b) in res["secs"].items():
                secs[op][0] += a
                secs[op][1] += b
            for op, d in res["diffs"].items():
                diffs[op] += d
    n = len(sqls)
    sources = Counter(r["source"].split(":")[0] for r in records)
    print(f"sources: {dict(sources.most_common())}")
    print(
        f"{path}{' shard ' + shard if shard else ''}: {len(records)} records, {n} distinct statements checked "
        f"({', '.join(f'{b} {len(s)}' for b, s in sorted(by_backend.items()))}), "
        f"{sum(map(len, sqls))} characters"
    )
    print(f"{'op':24} {'same':>7} {'differ':>7} {'raised':>7}  sqlparse s  pyroid s")
    bad = 0
    for op in OPS:
        d = len(diffs[op])
        bad += d
        print(f"{op:24} {same[op]:7} {d:7} {raised[op]:7}  {secs[op][0]:10.2f} {secs[op][1]:9.2f}")
    for op in OPS:
        for ex in [x for x in diffs[op] if x][:3]:
            print(
                f"\nDIFF {op} (len {ex['len']}): {ex['sql'][:400]!r}\n  sqlparse: {ex['sqlparse']}\n  pyroid:   {ex['pyroid']}"
            )
    print(
        f"\n{time.perf_counter() - t0:.1f} s wall, {workers} workers (CPU seconds summed over workers above)"
    )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--tests", type=Path, required=True, help="Django's tests/ directory")
    g.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("corpus", type=Path)
    c.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    c.add_argument("--shard", help="K/N: check every Nth distinct statement, from the Kth")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.tests, args.out)
        return 0
    return check(args.corpus, args.workers, args.shard)


if __name__ == "__main__":
    sys.exit(main())
