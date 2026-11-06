#!/usr/bin/env python3
"""The benchmark's per-doc digest of a ``pyroid clean`` run (kept + ``--excluded`` outputs).

    pyroid clean fineweb-dev-10k.parquet kept.parquet --recipe fineweb --excluded dropped.parquet
    python examples/cli/digest.py fineweb-dev-10k.parquet kept.parquet dropped.parquet \
        --expect ef314b8d6ae95cc63c6cfa8b1a3a76ee5bc101eb88a66729beb10ecb7226473b

One JSON line per input doc, in input order, ``[id, text, lang, score (%.6f), keep, reason, dup]``,
sha256: the format of ``bench/pipeline.py`` (datatrove's chain) and ``bench/chain.py``. ``keep`` =
passed every filter, ``dup`` = removed by MinHash (``None`` when the doc was dropped earlier).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from pyroid.integrations import _io
except ImportError:  # pragma: no cover
    raise SystemExit("needs pyroid with pyroid.integrations") from None


def read(path: str, columns: list[str] | None = None) -> pa.Table:
    files = _io.input_files(path)
    tables = list(_io.read_tables(files, 1 << 16))
    t = pa.concat_tables(tables) if tables else pq.read_schema(files[0]).empty_table()
    return t.select(columns) if columns else t


def digest(
    input_path: str, kept: str, excluded: str, id_col: str = "id", text_col: str = "text"
) -> str:
    ids = read(input_path, [id_col]).column(id_col).to_pylist()
    rec: dict[Any, list[Any]] = {}

    def put(t: pa.Table, kind: str) -> None:
        cols = [id_col, text_col, "language", "language_score"]
        reasons = (
            t.column("filter_reason").to_pylist() if kind == "excluded" else [None] * t.num_rows
        )
        rows = zip(*(t.column(c).to_pylist() for c in cols), strict=True)
        for (i, text, lang, score), reason in zip(rows, reasons, strict=True):
            if i in rec:
                raise SystemExit(f"doc {i!r} appears twice in the outputs")
            sc = None if score is None else f"{score:.6f}"
            if kind == "kept":
                rec[i] = [i, text, lang, sc, True, None, False]
            elif reason == "minhash":
                rec[i] = [i, text, lang, sc, True, None, True]
            else:
                rec[i] = [i, text, lang, sc, False, reason, None]

    put(read(kept), "kept")
    put(read(excluded), "excluded")
    missing = [i for i in ids if i not in rec]
    if missing or len(rec) != len(ids):
        raise SystemExit(f"{len(missing)} input docs are in neither output (e.g. {missing[:3]})")
    h = hashlib.sha256()
    for i in ids:
        h.update(json.dumps(rec[i], ensure_ascii=False).encode("utf-8", "surrogatepass") + b"\n")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input")
    ap.add_argument("kept")
    ap.add_argument("excluded")
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--expect")
    a = ap.parse_args()
    d = digest(a.input, a.kept, a.excluded, a.id_column, a.text_column)
    print(d)  # noqa: T201
    if a.expect and d != a.expect:
        print(f"DIGEST MISMATCH: expected {a.expect}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
