"""Replay the sqlparse inputs a consumer's test run made (probe/sitecustomize.py's
``$PROBE_CAPTURE`` file) through sqlparse 0.6.1 and pyroid.sqlparse, and count the matches.

    python replay.py CAPTURE.jsonl [CAPTURE.jsonl ...]

For each distinct text that reached ``lexer.tokenize`` it compares the tokens, the parse tree
(classes, token types, values, parent links), ``engine.grouping.group`` on sqlparse's own
ungrouped statements and ``split``; for each ``format`` call, the result with the consumer's
options. Results that are exceptions compare by type and message. Exits with 1 on a difference.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyroid.sqlparse as psp
from sqlparse.engine.statement_splitter import StatementSplitter

import sqlparse


def shape(tok: Any) -> Any:
    if tok.is_group:
        if not all(t.parent is tok for t in tok.tokens):
            raise AssertionError(f"a child of {type(tok).__name__} has another parent")
        return [type(tok).__name__, [shape(t) for t in tok.tokens]]
    return [type(tok).__name__, str(tok.ttype), tok.value]


def run(mod: Any, op: str, sql: str, options: dict[str, Any] | None) -> Any:
    try:
        if op == "tokenize":
            return [[str(t), v] for t, v in mod.lexer.tokenize(sql)]
        if op == "parse":
            return [shape(s) for s in mod.parse(sql)]
        if op == "group":
            stmts = StatementSplitter().process(sqlparse.lexer.tokenize(sql))
            return [shape(mod.engine.grouping.group(s)) for s in stmts]
        if op == "split":
            return mod.split(sql)
        return mod.format(sql, **(options or {}))
    except Exception as e:  # noqa: BLE001 - the exception is part of the compared behaviour
        return ["raised", type(e).__name__, str(e)]


def main(paths: list[str]) -> int:
    assert not sqlparse.format.__module__.startswith("pyroid"), "replay without the patch"
    records = []
    for p in paths:
        records += [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    texts = sorted({r["sql"] for r in records if r["fn"] == "tokenize" and not r["sql"].startswith("<")})
    formats = {(r["sql"], json.dumps(r["options"], sort_keys=True)) for r in records if r["fn"] == "format"}
    same: Counter[str] = Counter()
    total: Counter[str] = Counter()
    shown = 0
    jobs = [(op, s, None) for s in texts for op in ("tokenize", "parse", "group", "split")]
    jobs += [("format", s, json.loads(o)) for s, o in sorted(formats)]
    for op, sql, options in jobs:
        total[op] += 1
        a, b = run(sqlparse, op, sql, options), run(psp, op, sql, options)
        if a == b:
            same[op] += 1
        elif shown < 10:
            shown += 1
            print(
                f"DIFF {op} {options or ''}: {sql[:300]!r}\n  sqlparse: {str(a)[:400]}\n  pyroid:   {str(b)[:400]}"
            )
    other = Counter(r["sql"] for r in records if r["sql"].startswith("<"))
    print(
        f"{', '.join(paths)}: {len(texts)} distinct texts, {len(formats)} distinct format calls"
        f"{f', non-str inputs {dict(other)}' if other else ''}"
    )
    for op in total:
        print(f"  {op:9} {same[op]:6} of {total[op]:6} same")
    return 0 if same == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
