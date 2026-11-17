#!/usr/bin/env python3
"""sqlparse 0.6.1 vs pyroid.sqlparse: per-call cost of format, split and parse, and batches.

    python bench/sqlparse/sqlparse_bench.py --out results.json            # the full run
    python bench/sqlparse/sqlparse_bench.py --quick                        # smoke test
    python bench/sqlparse/sqlparse_bench.py --corpus statements.jsonl      # also a corpus

Both libraries run in the same process, interleaved, and each timing is the best of
``--repeat`` rounds (the median is recorded too). Before a case is timed, both results must be
equal (formatted text, statement lists, and token trees field by field); a mismatch aborts the
run.

Cases: a one-line SELECT, a 4-line SELECT with joins, CASE, IN, BETWEEN, GROUP BY and HAVING,
an INSERT of 200 rows, a CREATE TABLE of 40 columns and a 120-statement migration script, each
through ``format()``, ``format(reindent=True)``, ``format(reindent=True,
keyword_case="upper")``, ``split()`` and ``parse()``; then ``format_batch`` and ``split_batch``
over 20,000 statements against a loop over sqlparse (pyroid on all cores and on one).
``--corpus`` adds the same calls over a JSONL file of ``{"sql": ...}`` lines (the real-SQL
corpus of ``bench/sqlparse/corpus.py``), as statements per second.

Prints a Markdown table; ``--out`` writes the timings and the machine (CPU, Python, versions and
the load average before and after; no hostname or instance id).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Run from anywhere: keep bench/ off sys.path (bench/sqlparse would shadow the library).
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import sqlparse

import pyroid
import pyroid.sqlparse as ps

SELECT_1 = "select id, name from users where id = 42"
SELECT_4 = (
    "SELECT a.id, a.name, COUNT(b.x) AS cnt, SUM(CASE WHEN b.y > 10 THEN 1 ELSE 0 END) AS big\n"
    "FROM accounts a LEFT JOIN orders b ON a.id = b.account_id\n"
    "WHERE a.created_at >= '2024-01-01' AND a.status IN ('open', 'pending') "
    "AND b.total BETWEEN 10 AND 100\n"
    "GROUP BY a.id, a.name HAVING COUNT(b.x) > 2 ORDER BY cnt DESC LIMIT 50;"
)
INSERT_200 = "INSERT INTO events (id, kind, payload, created_at) VALUES " + ", ".join(
    f"({i}, 'kind{i % 7}', '{{\"n\": {i}}}', '2024-01-{i % 28 + 1:02d}')" for i in range(200)
)
CREATE_40 = (
    "CREATE TABLE IF NOT EXISTS wide (\n"
    + ",\n".join(f"  col{i} {'integer' if i % 3 else 'varchar(255)'} NOT NULL DEFAULT 0" for i in range(40))
    + ",\n  PRIMARY KEY (col0)\n);"
)
MIGRATION = "\n".join(
    [
        f'ALTER TABLE "app_model{i}" ADD COLUMN "field{i}" varchar(100) DEFAULT \'\' NOT NULL;\n'
        f'CREATE INDEX "app_model{i}_field{i}_idx" ON "app_model{i}" ("field{i}");\n'
        f'UPDATE "app_model{i}" SET "field{i}" = \'x\' WHERE "id" IN (SELECT "id" FROM "app_model{i}" LIMIT 10);'
        for i in range(40)
    ]
)

DOCS: list[tuple[str, str]] = [
    ("1-line SELECT", SELECT_1),
    ("4-line SELECT", SELECT_4),
    ("INSERT, 200 rows", INSERT_200),
    ("CREATE TABLE, 40 columns", CREATE_40),
    ("migration, 120 statements", MIGRATION),
]

CALLS: list[tuple[str, Callable[[Any, str], Any]]] = [
    ("format()", lambda m, s: m.format(s)),
    ("format(reindent=True)", lambda m, s: m.format(s, reindent=True)),
    ("format(reindent, keyword_case)", lambda m, s: m.format(s, reindent=True, keyword_case="upper")),
    ("split()", lambda m, s: m.split(s)),
    ("parse()", lambda m, s: m.parse(s)),
]


def tree(t: Any) -> Any:
    return (
        type(t).__name__,
        t.ttype,
        t.value,
        t.normalized,
        t.is_keyword,
        t.is_group,
        t.is_whitespace,
        t.is_newline,
        tuple(tree(c) for c in t.tokens) if t.is_group else None,
    )


def comparable(x: Any) -> Any:
    if isinstance(x, tuple) and x and isinstance(x[0], sqlparse.sql.Statement):
        return [tree(s) for s in x]
    return x


def best(f: Callable[[], Any], repeat: int, budget: float) -> tuple[float, float]:
    """Best and median seconds per call over `repeat` rounds of about `budget` seconds."""
    t = time.perf_counter()
    f()
    one = max(time.perf_counter() - t, 1e-7)
    n = max(1, int(budget / one))
    rounds = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(n):
            f()
        rounds.append((time.perf_counter() - t) / n)
    return min(rounds), statistics.median(rounds)


def machine() -> dict[str, Any]:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "cpu part")):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "cpus": os.cpu_count(),
        "pyroid": pyroid.__version__,
        "sqlparse": sqlparse.__version__,
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }


def run_calls(rows: list[dict[str, Any]], repeat: int, budget: float) -> None:
    for doc_name, doc in DOCS:
        for call_name, call in CALLS:
            want = comparable(call(sqlparse, doc))
            got = comparable(call(ps, doc))
            if want != got:
                raise SystemExit(f"mismatch: {call_name} on {doc_name}")
            s_best, s_med = best(lambda c=call, d=doc: c(sqlparse, d), repeat, budget)
            p_best, p_med = best(lambda c=call, d=doc: c(ps, d), repeat, budget)
            rows.append(
                {
                    "group": "calls",
                    "doc": doc_name,
                    "call": call_name,
                    "sqlparse_us": s_best * 1e6,
                    "pyroid_us": p_best * 1e6,
                    "sqlparse_median_us": s_med * 1e6,
                    "pyroid_median_us": p_med * 1e6,
                    "speedup": s_best / p_best,
                }
            )
            print(f"| {doc_name} | `{call_name}` | {s_best * 1e6:,.1f} | {p_best * 1e6:,.2f} | {s_best / p_best:.1f}x |")


def run_batches(rows: list[dict[str, Any]], n: int, repeat: int) -> None:
    docs = [
        f"select a{i}, b{i % 13}, count(*) from t{i % 50} where c = {i} and d in (1, 2, 3) group by 1, 2"
        for i in range(n)
    ]
    cases: list[tuple[str, Callable[[], Any], Callable[[], Any], Callable[[], Any]]] = [
        (
            "format(reindent=True)",
            lambda: [sqlparse.format(d, reindent=True) for d in docs],
            lambda: ps.format_batch(docs, reindent=True),
            lambda: ps.format_batch(docs, reindent=True, threads=1),
        ),
        (
            "split()",
            lambda: [sqlparse.split(d) for d in docs],
            lambda: ps.split_batch(docs),
            lambda: ps.split_batch(docs, threads=1),
        ),
    ]
    for name, s_call, p_all, p_one in cases:
        if s_call() != p_all() or p_all() != p_one():
            raise SystemExit(f"mismatch: batch {name}")
        s_t = min(_time(s_call) for _ in range(max(1, repeat // 3)))
        a_t = min(_time(p_all) for _ in range(repeat))
        o_t = min(_time(p_one) for _ in range(repeat))
        rows.append(
            {
                "group": "batch",
                "call": name,
                "docs": n,
                "sqlparse_s": s_t,
                "pyroid_all_s": a_t,
                "pyroid_one_s": o_t,
                "speedup_all": s_t / a_t,
                "speedup_one": s_t / o_t,
            }
        )
        print(
            f"| {n:,} statements, `{name}` | {s_t:.2f} s | {a_t * 1e3:,.1f} ms | {o_t * 1e3:,.1f} ms | "
            f"{s_t / a_t:.0f}x | {s_t / o_t:.0f}x |"
        )


def _time(f: Callable[[], Any]) -> float:
    t = time.perf_counter()
    f()
    return time.perf_counter() - t


def run_corpus(rows: list[dict[str, Any]], path: Path, limit: int) -> None:
    docs: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            sql = json.loads(line).get("sql")
            if isinstance(sql, str) and len(sql) < 200_000:
                docs.append(sql)
            if len(docs) >= limit:
                break
    total = sum(len(d) for d in docs)
    for call_name, call in CALLS:
        try:
            t_s = _time(lambda c=call: [_quiet(c, sqlparse, d) for d in docs])
            t_p = _time(lambda c=call: [_quiet(c, ps, d) for d in docs])
        except Exception as e:  # noqa: BLE001
            print(f"corpus {call_name}: {type(e).__name__}: {e}")
            continue
        rows.append(
            {
                "group": "corpus",
                "call": call_name,
                "docs": len(docs),
                "bytes": total,
                "sqlparse_s": t_s,
                "pyroid_s": t_p,
                "speedup": t_s / t_p,
            }
        )
        print(f"| corpus ({len(docs):,} docs, {total / 1e6:.1f} MB), `{call_name}` | {t_s:.1f} s | {t_p:.2f} s | {t_s / t_p:.0f}x |")


def _quiet(call: Callable[[Any, str], Any], mod: Any, doc: str) -> Any:
    try:
        return call(mod, doc)
    except mod.exceptions.SQLParseError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--budget", type=float, default=0.3, help="seconds per round")
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--corpus-limit", type=int, default=50_000)
    args = ap.parse_args()
    if args.quick:
        args.repeat, args.budget, args.batch, args.corpus_limit = 2, 0.02, 500, 500
    before = machine()
    rows: list[dict[str, Any]] = []
    print("| Input | Call | sqlparse (µs) | pyroid (µs) | Speedup |\n|---|---|---:|---:|---:|")
    run_calls(rows, args.repeat, args.budget)
    print(
        "\n| Batch | sqlparse loop | pyroid, all cores | pyroid, 1 thread | Speedup | Speedup, 1 thread |"
        "\n|---|---:|---:|---:|---:|---:|"
    )
    run_batches(rows, args.batch, args.repeat)
    if args.corpus:
        print("\n| Corpus | sqlparse | pyroid | Speedup |\n|---|---:|---:|---:|")
        run_corpus(rows, args.corpus, args.corpus_limit)
    after = machine()
    if args.out:
        args.out.write_text(
            json.dumps({"machine": before, "loadavg_after": after["loadavg"], "rows": rows}, indent=1)
            + "\n"
        )


if __name__ == "__main__":
    main()
