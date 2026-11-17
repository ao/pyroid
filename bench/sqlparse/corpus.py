#!/usr/bin/env python3
"""Download real SQL and cut it into a JSONL corpus for the pyroid.sqlparse parity run.

    python bench/sqlparse/corpus.py /target/sqlcorpus                  # every source
    python bench/sqlparse/corpus.py /target/sqlcorpus --sources tpch,tpcds,sqlglot
    python bench/sqlparse/corpus.py /target/sqlcorpus --list           # the sources
    python bench/sqlparse/corpus.py /target/sqlcorpus --offline        # rebuild from downloads

Downloads go to ``OUT/raw/`` (kept between runs; ``--refresh`` fetches again). Archives are
read in memory, never extracted. ``OUT/SOURCES.json`` records every source: its URL, sha256,
size, licence, the files used, the units it gave, or why it was skipped. ``OUT/corpus.jsonl``
holds one unit per line, ``{"id", "source", "path", "kind", "text"}``, where ``kind`` is
``statement`` (one statement: a line-based split at ``;``, a sqllogictest record, a dataset
row) or ``script`` (a whole file, or a chunk of at most ``--chunk`` bytes cut at a statement
end). Identical texts are kept once. Run it on the devbox: the downloads total about 250 MB.

Needs the standard library only; pyarrow (if installed) adds the Parquet datasets.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

UA = "pyroid-sqlparse-corpus/1 (+https://github.com/ao/pyroid)"


@dataclass
class Source:
    """One download and how to cut it into units."""

    name: str
    url: str
    licence: str
    note: str = ""
    kind: str = "file"  # file | tar | github-dir | hf-json | hf-parquet
    members: tuple[str, ...] = ()  # tar: fnmatch patterns on the path inside the top directory
    cut: str = "sql"  # sql | sqllogictest | lines | blank-lines | create-context | parquet
    columns: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


GH = "https://codeload.github.com"
HF = "https://huggingface.co/datasets"
SOURCES = [
    Source(
        "tpch",
        f"{GH}/duckdb/duckdb/tar.gz/refs/heads/main",
        "MIT (duckdb); the query texts are TPC-H's (TPC EULA)",
        "TPC-H's 22 queries as shipped in duckdb's tpch extension",
        kind="tar",
        members=("extension/tpch/dbgen/queries/q*.sql",),
    ),
    Source(
        "tpcds",
        f"{GH}/duckdb/duckdb/tar.gz/refs/heads/main",
        "MIT (duckdb); the query texts are TPC-DS's (TPC EULA)",
        "TPC-DS's 99 queries as shipped in duckdb's tpcds extension",
        kind="tar",
        members=("extension/tpcds/dsdgen/queries/*.sql",),
    ),
    Source(
        "duckdb-tests",
        f"{GH}/duckdb/duckdb/tar.gz/refs/heads/main",
        "MIT",
        "duckdb's sqllogictest suite (test/sql/**/*.test): statement and query records",
        kind="tar",
        members=("test/sql/*.test", "test/sql/*.test_slow"),
        cut="sqllogictest",
    ),
    Source(
        "postgres-regress",
        f"{GH}/postgres/postgres/tar.gz/refs/heads/master",
        "PostgreSQL License",
        "PostgreSQL's regression tests and contrib tests (psql scripts)",
        kind="tar",
        members=("src/test/regress/sql/*.sql", "contrib/*/sql/*.sql", "src/pl/*/sql/*.sql"),
    ),
    Source(
        "sqlglot",
        f"{GH}/tobymao/sqlglot/tar.gz/refs/heads/main",
        "MIT",
        "sqlglot's fixtures: identity.sql (one statement per line, many dialects), pretty.sql, "
        "the optimizer fixtures",
        kind="tar",
        members=("tests/fixtures/*.sql",),
        cut="lines-or-sql",
    ),
    Source(
        "gitlab-structure",
        "https://gitlab.com/gitlab-org/gitlab/-/raw/master/db/structure.sql",
        "MIT (GitLab CE)",
        "GitLab's PostgreSQL schema dump",
    ),
    Source(
        "supabase-postgres",
        f"{GH}/supabase/postgres/tar.gz/refs/heads/develop",
        "PostgreSQL License",
        "Supabase's Postgres image: migrations and extension tests",
        kind="tar",
        members=("migrations/*.sql", "nix/tests/*.sql", "ansible/files/*.sql"),
    ),
    Source(
        "jaffle_shop",
        f"{GH}/dbt-labs/jaffle_shop/tar.gz/refs/heads/main",
        "Apache-2.0",
        "dbt's classic example project (Jinja-templated models)",
        kind="tar",
        members=("models/*.sql", "*.sql"),
    ),
    Source(
        "jaffle-shop",
        f"{GH}/dbt-labs/jaffle-shop/tar.gz/refs/heads/main",
        "see the repository (no LICENSE file at the root when fetched)",
        "dbt's current example project",
        kind="tar",
        members=("models/*.sql", "macros/*.sql", "analyses/*.sql"),
    ),
    Source(
        "dbt-utils",
        f"{GH}/dbt-labs/dbt-utils/tar.gz/refs/heads/main",
        "Apache-2.0",
        "dbt-utils macros and integration-test models",
        kind="tar",
        members=("macros/*.sql", "integration_tests/models/*.sql"),
    ),
    Source(
        "chinook",
        f"{GH}/lerocha/chinook-database/tar.gz/refs/heads/master",
        "MIT",
        "Chinook sample database scripts for several engines (DDL and many INSERTs)",
        kind="tar",
        members=("ChinookDatabase/DataSources/*.sql",),
    ),
    Source(
        "pagila",
        f"{GH}/devrimgunduz/pagila/tar.gz/refs/heads/master",
        "PostgreSQL License",
        "Pagila (Sakila for PostgreSQL): schema, data as COPY and as INSERT",
        kind="tar",
        members=("*.sql",),
    ),
    Source(
        "mastodon-views",
        "https://api.github.com/repos/mastodon/mastodon/contents/db/views?ref=main",
        "AGPL-3.0",
        "Mastodon's SQL views (db/views/*.sql); its migrations are Ruby",
        kind="github-dir",
    ),
    Source(
        "sqlparse-tests",
        "https://files.pythonhosted.org/packages/source/s/sqlparse/sqlparse-0.6.1.tar.gz",
        "BSD-3-Clause",
        "sqlparse 0.6.1's own test files",
        kind="tar",
        members=("tests/files/*.sql",),
    ),
    Source(
        "sql-create-context",
        f"{HF}/b-mc2/sql-create-context/resolve/main/sql_create_context_v4.json",
        "CC-BY-4.0",
        "WikiSQL/Spider-derived questions: the CREATE TABLE context and the answer query",
        kind="hf-json",
        cut="create-context",
    ),
    Source(
        "gretel-text-to-sql",
        f"{HF}/gretelai/synthetic_text_to_sql/resolve/main/synthetic_text_to_sql_train.snappy.parquet",
        "Apache-2.0",
        "synthetic text-to-SQL pairs: the sql and sql_context columns",
        kind="hf-parquet",
        cut="parquet",
        columns=("sql", "sql_context"),
    ),
    Source(
        "spider",
        f"{HF}/xlangai/spider/resolve/main/spider/train-00000-of-00001.parquet",
        "CC-BY-SA-4.0",
        "Spider's training queries (the query column)",
        kind="hf-parquet",
        cut="parquet",
        columns=("query",),
    ),
]
#: Looked at, not used (recorded in SOURCES.json).
NOT_USED = [
    {
        "name": "discourse",
        "url": "https://github.com/discourse/discourse",
        "reason": "its SQL lives in Ruby migrations and heredocs, not .sql files",
    },
    {
        "name": "metabase-sample-queries",
        "url": "https://github.com/metabase/metabase",
        "reason": "the sample database is an H2 binary; no corpus of query texts in the repository",
    },
    {
        "name": "stack-overflow",
        "url": "https://archive.org/details/stackexchange",
        "reason": "the CC BY-SA dump is a multi-GB XML archive with SQL inside HTML posts; over the size budget",
    },
    {
        "name": "public-query-logs",
        "url": "https://skyserver.sdss.org",
        "reason": "no permissively licensed SQL query-log dump reachable as a plain download",
    },
]


# ---- downloads -----------------------------------------------------------------------------------


def fetch(url: str, dest: str, refresh: bool, offline: bool) -> dict[str, Any]:
    """Download `url` to `dest` (kept if present); its sha256 and size."""
    if not os.path.exists(dest) or refresh:
        if offline:
            raise FileNotFoundError(f"{dest} is missing and --offline is set")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        last: Exception | None = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                os.replace(dest + ".part", dest)
                last = None
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                time.sleep(2 + 3 * attempt)
        if last is not None:
            raise last
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": os.path.getsize(dest)}


def decode(raw: bytes) -> tuple[str, str]:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes anything")


# ---- cutting text into units ----------------------------------------------------------------------


def split_statements(text: str) -> list[str]:
    """A line-based split: a statement ends on a line that ends with ``;`` (outside $$ bodies)."""
    out: list[str] = []
    cur: list[str] = []
    in_dollar = False
    for line in text.splitlines(keepends=True):
        cur.append(line)
        if line.count("$$") % 2:
            in_dollar = not in_dollar
        if not in_dollar and line.rstrip().endswith(";"):
            stmt = "".join(cur).strip()
            if stmt:
                out.append(stmt)
            cur = []
    rest = "".join(cur).strip()
    if rest:
        out.append(rest)
    return out


def _size(s: str) -> int:
    return len(s.encode("utf-8", "surrogatepass"))


def _lines_upto(stmt: str, limit: int) -> list[str]:
    """`stmt`, or cut at line ends into pieces of at most about `limit` bytes (COPY data)."""
    if _size(stmt) <= limit:
        return [stmt]
    out, cur, size = [], [], 0
    for line in stmt.splitlines(keepends=True):
        if cur and size + _size(line) > limit:
            out.append("".join(cur).rstrip("\n"))
            cur, size = [], 0
        cur.append(line)
        size += _size(line)
    if cur:
        out.append("".join(cur).rstrip("\n"))
    return out


def chunks(text: str, limit: int) -> list[str]:
    """`text` whole, or cut at statement ends (line ends inside a longer statement) into pieces
    of at most about `limit` bytes."""
    if _size(text) <= limit:
        return [text]
    out, cur, size = [], [], 0
    for stmt in (p for s in split_statements(text) for p in _lines_upto(s, limit)):
        n = _size(stmt) + 1
        if cur and size + n > limit:
            out.append("\n".join(cur))
            cur, size = [], 0
        cur.append(stmt)
        size += n
    if cur:
        out.append("\n".join(cur))
    return out


_SLT_HEAD = re.compile(r"^(statement|query)\b")


def sqllogictest(text: str) -> list[str]:
    """The SQL of each ``statement`` and ``query`` record of a sqllogictest file."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if _SLT_HEAD.match(lines[i]):
            i += 1
            sql: list[str] = []
            while i < len(lines) and lines[i].strip() not in ("", "----"):
                sql.append(lines[i])
                i += 1
            if sql:
                out.append("\n".join(sql))
            while i < len(lines) and lines[i].strip() != "":
                i += 1
        i += 1
    return out


# ---- sources --------------------------------------------------------------------------------------


def tar_files(path: str, patterns: tuple[str, ...]) -> Iterator[tuple[str, bytes]]:
    """(path below the top directory, bytes) of the regular files matching `patterns`."""
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/", 1)
            inner = parts[1] if len(parts) == 2 else parts[0]
            if any(fnmatch.fnmatch(inner, p) for p in patterns):
                f = tf.extractfile(m)
                if f is not None:
                    yield inner, f.read()


def license_head(path: str) -> str:
    """The first lines of the LICENSE file at the top of a tarball, if any."""
    try:
        for inner, raw in tar_files(path, ("LICENSE", "LICENSE.*", "COPYING", "COPYRIGHT")):
            if "/" not in inner:
                return " ".join(decode(raw)[0].split()[:12])
    except tarfile.TarError:
        pass
    return ""


def units_from_text(src: Source, path: str, text: str, chunk: int) -> Iterator[tuple[str, str, str]]:
    """(path, kind, text) units of one file."""
    if src.cut == "sqllogictest":
        for stmt in sqllogictest(text):
            yield path, "statement", stmt
        return
    if src.cut == "lines-or-sql" and path.endswith("identity.sql"):
        for line in text.splitlines():
            if line.strip() and not line.startswith("#"):
                yield path, "statement", line
        return
    for piece in chunks(text, chunk):
        yield path, "script", piece
    for stmt in split_statements(text):
        for piece in _lines_upto(stmt, chunk):
            yield path, "statement", piece


def build(src: Source, raw_dir: str, args: argparse.Namespace) -> tuple[dict[str, Any], list]:
    """Fetch `src` and cut it into units: (its SOURCES.json entry, [(path, kind, text)])."""
    entry: dict[str, Any] = {
        "name": src.name,
        "url": src.url,
        "licence": src.licence,
        "note": src.note,
    }
    units: list[tuple[str, str, str]] = []
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", src.url.split("://", 1)[1])[:150]
    dest = os.path.join(raw_dir, safe)
    if src.kind == "github-dir":
        listing = fetch(src.url, dest, args.refresh, args.offline)
        files = []
        with open(dest, encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            if it.get("type") == "file" and it["name"].endswith(".sql"):
                fd = os.path.join(raw_dir, safe + "_files", it["name"])
                info = fetch(it["download_url"], fd, args.refresh, args.offline)
                files.append({"path": it["name"], "url": it["download_url"], **info})
                with open(fd, "rb") as f:
                    units += units_from_text(src, it["name"], decode(f.read())[0], args.chunk)
        entry.update(listing, files=files)
        return entry, units
    info = fetch(src.url, dest, args.refresh, args.offline)
    entry.update(info)
    used: list[str] = []
    if src.kind == "file":
        with open(dest, "rb") as f:
            text, enc = decode(f.read())
        used.append(os.path.basename(src.url))
        units += units_from_text(src, used[0], text, args.chunk)
    elif src.kind == "tar":
        head = license_head(dest)
        if head:
            entry["licence_file_head"] = head
        for inner, raw in tar_files(dest, src.members):
            text, enc = decode(raw)
            used.append(inner if enc == "utf-8" else f"{inner} ({enc})")
            units += units_from_text(src, inner, text, args.chunk)
    elif src.kind == "hf-json":
        with open(dest, encoding="utf-8") as f:
            rows = json.load(f)
        used.append(os.path.basename(src.url))
        for i, row in enumerate(rows):
            units.append((f"row{i}/answer", "statement", row["answer"]))
            for j, stmt in enumerate(s for s in row["context"].split(";") if s.strip()):
                units.append((f"row{i}/context{j}", "statement", stmt.strip()))
    elif src.kind == "hf-parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError:
            entry["status"] = "downloaded, not cut: needs pyarrow"
            return entry, []
        table = pq.read_table(dest, columns=list(src.columns))
        used.append(os.path.basename(src.url.split("?")[0]))
        for col in src.columns:
            for i, v in enumerate(table.column(col).to_pylist()):
                if isinstance(v, str) and v.strip():
                    units.append((f"row{i}/{col}", "statement", v))
    entry["files_used"] = len(used)
    entry["files"] = used[:400]
    return entry, units


def sample(units: list, cap: int) -> list:
    """At most `cap` units, spread evenly over the list (deterministic)."""
    if cap <= 0 or len(units) <= cap:
        return units
    step = len(units) / cap
    return [units[int(i * step)] for i in range(cap)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", help="output directory (on the devbox, e.g. /target/sqlcorpus)")
    ap.add_argument("--sources", help="comma-separated source names (default: all)")
    ap.add_argument("--list", action="store_true", help="list the sources and exit")
    ap.add_argument("--refresh", action="store_true", help="download again")
    ap.add_argument("--offline", action="store_true", help="use the downloads in OUT/raw only")
    ap.add_argument("--chunk", type=int, default=64 * 1024, help="max script bytes (64 KiB)")
    ap.add_argument("--max-statements", type=int, default=40000, help="statements kept per source (40000)")
    ap.add_argument("--max-scripts", type=int, default=3000, help="scripts kept per source (3000)")
    args = ap.parse_args(argv)
    if args.list:
        for s in SOURCES:
            print(f"{s.name:20} {s.licence:30} {s.url}")
        return 0
    wanted = set(args.sources.split(",")) if args.sources else None
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = {"statement": 0, "script": 0, "duplicates": 0}
    tmp = os.path.join(args.out, "corpus.jsonl.part")
    with open(tmp, "w", encoding="utf-8") as out:
        for src in SOURCES:
            if wanted is not None and src.name not in wanted:
                continue
            t0 = time.monotonic()
            try:
                entry, units = build(src, raw_dir, args)
            except Exception as e:  # noqa: BLE001 (recorded, the run goes on)
                entries.append(
                    {
                        "name": src.name,
                        "url": src.url,
                        "licence": src.licence,
                        "status": f"failed: {type(e).__name__}: {e}",
                    }
                )
                print(f"{src.name}: failed: {e}", file=sys.stderr)
                continue
            by_kind: dict[str, list] = {"statement": [], "script": []}
            for u in units:
                by_kind[u[1]].append(u)
            kept = sample(by_kind["statement"], args.max_statements) + sample(
                by_kind["script"], args.max_scripts
            )
            counts = {"statement": 0, "script": 0, "duplicates": 0, "chars": 0}
            for i, (path, kind, text) in enumerate(kept):
                h = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
                if h in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(h)
                counts[kind] += 1
                counts["chars"] += len(text)
                rec = {"id": f"{src.name}/{i}", "source": src.name, "path": path, "kind": kind, "text": text}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            entry.setdefault("status", "ok")
            entry["units_found"] = {k: len(v) for k, v in by_kind.items()}
            entry["units_kept"] = counts
            entries.append(entry)
            for k in ("statement", "script", "duplicates"):
                total[k] += counts[k]
            print(
                f"{src.name}: {counts['statement']} statements, {counts['script']} scripts, "
                f"{counts['duplicates']} duplicates ({time.monotonic() - t0:.0f} s)",
                file=sys.stderr,
            )
    os.replace(tmp, os.path.join(args.out, "corpus.jsonl"))
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": total,
        "sources": entries,
        "not_used": NOT_USED,
    }
    with open(os.path.join(args.out, "SOURCES.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
