#!/usr/bin/env python3
"""Check pyroid.yaml against the PyYAML recordings of ``record_oracle.py``, file by file.

    PYTHONHASHSEED=0 python -P bench/yaml/check_oracle.py /target/data/corpus /target/data/oracle/corpus --jobs 16

For every file the recorder saw: ``pyroid.yaml.safe_load_all`` of the bytes must give the recorded
canonical JSON (documents, and the exception's class, ``str()`` and fields); for files that loaded,
``pyroid.yaml.safe_dump_all`` of the documents must give the recorded text for each option set; for
files that failed, the recorded ``str`` and stream outcomes must match too. Prints the counts and
the first differences, and writes them to ``OUT/pyroid-check.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _recorder() -> Any:
    spec = importlib.util.spec_from_file_location("record_oracle", HERE / "record_oracle.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_file(root: str, oracle: str, entry: dict[str, Any]) -> dict[str, Any]:
    import pyroid.yaml as py

    rec = _recorder()
    warnings.simplefilter("ignore")
    rel = entry["path"]
    data = (Path(root) / rel).read_bytes()
    out: dict[str, Any] = {"path": rel, "checks": 0, "diffs": []}

    def load(stream: Any) -> tuple[list[Any], Any]:
        docs: list[Any] = []
        try:
            for d in py.safe_load_all(stream):
                docs.append(d)  # noqa: PERF402 - keep the documents before an exception
        except Exception as e:  # noqa: BLE001
            return docs, rec.error_record(e)
        return docs, None

    docs, error = load(data)
    text, _ = rec.canonical(docs, error)
    want = (Path(oracle) / "load" / (rel + ".json")).read_text(encoding="utf-8")
    out["checks"] += 1
    if text != want:
        out["diffs"].append(("load", want[:300], text[:300]))
    if error is None:
        for name, opts in rec.OPTIONS.items():
            path = Path(oracle) / "dump" / name / rel
            if not path.exists():
                continue
            out["checks"] += 1
            try:
                got = py.safe_dump_all(docs, **opts).encode("utf-8", "surrogatepass")
            except Exception as e:  # noqa: BLE001
                got = repr(e).encode()
            if got != path.read_bytes():
                out["diffs"].append(
                    (
                        f"dump:{name}",
                        path.read_bytes()[:200].decode("utf-8", "replace"),
                        got[:200].decode("utf-8", "replace"),
                    )
                )
    else:
        variants = entry.get("variants") or {}
        for kind, recorded in variants.items():
            if kind == "str":
                try:
                    stream: Any = data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            elif kind == "stream":
                stream = io.BytesIO(data)
            else:
                continue
            d2, e2 = load(stream)
            out["checks"] += 1
            same = rec.canonical(d2, None)[0] == rec.canonical(docs, None)[0]
            if e2 != recorded.get("error") or same != recorded.get("same_docs", True):
                out["diffs"].append((kind, str(recorded)[:300], str(e2)[:300]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("oracle")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    index = [json.loads(line) for line in (Path(args.oracle) / "index.jsonl").read_text().splitlines()]
    if args.limit:
        index = index[: args.limit]
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(check_file, args.root, args.oracle, e) for e in index]
        results = [f.result() for f in futures]
    checks = sum(r["checks"] for r in results)
    bad = [r for r in results if r["diffs"]]
    print(f"{len(results)} files, {checks} checks, {len(bad)} files differ ({time.time() - t0:.0f} s)")
    for r in bad[:20]:
        print(r["path"])
        for kind, want, got in r["diffs"][:3]:
            print(f"  {kind}\n    want {want!r}\n    got  {got!r}")
    summary = {"files": len(results), "checks": checks, "differ": [r["path"] for r in bad], "first": bad[:50]}
    (Path(args.oracle) / "pyroid-check.json").write_text(json.dumps(summary, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
