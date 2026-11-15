#!/usr/bin/env python3
"""python-dateutil's parser vs pyroid.dateutil.parser on a corpus of real date strings.

    python bench/dateutil_bench.py CORPUS_DIR --out /target/du-bench.json [--per-source 20000]

CORPUS_DIR holds one ``<source>.txt`` per source, one string per line (as
``.scratch/corpus/fetch_corpus.py`` writes them). A seeded sample of ``--per-source`` strings is
taken from each source and mixed.

* Gate, before anything is timed: on the whole sample, pyroid's ``parse(s)`` and ``isoparse(s)``
  must give dateutil's result (value, type, tzinfo type and repr) or dateutil's exception (type
  and message). A mismatch aborts the run.
* Single calls: ``dateutil.parser.parse`` and pyroid's ``parse`` in a loop (a failing string
  counts, with its exception caught), best of ``--repeat`` runs, microseconds per string.
* Batches: ``parse_batch(strings, errors="coerce", threads=t)`` for t = 1, 8 and the whole pool,
  as a list and as a pyarrow array (``ignoretz=True``, so every row fits one column). For context:
  ``pandas.to_datetime(..., errors="coerce", utc=True)`` with ``format="mixed"`` (pandas' own
  dateutil-based path) and without ``format`` (it guesses one format from the first string).
  pandas' results are not dateutil's: these rows are timings only.
* ISO strings (the sample's strings that ``datetime.fromisoformat`` accepts): dateutil's and
  pyroid's ``isoparse``, ``isoparse_batch``, ``datetime.fromisoformat`` and, if installed,
  ``ciso8601.parse_datetime``.

Output: a JSON file (``--out``) with machine info and every timing, and a Markdown summary on
stdout. Numbers from a shared machine are indicative.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata as md
import json
import os
import platform
import random
import sys
import sysconfig
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load(corpus: Path, per_source: int, seed: int) -> tuple[list[str], dict[str, int]]:
    rnd = random.Random(seed)
    out: list[str] = []
    counts: dict[str, int] = {}
    for path in sorted(corpus.glob("*.txt")):
        lines = [x for x in path.read_text(encoding="utf-8", errors="replace").split("\n") if x]
        if not lines:
            continue
        pick = rnd.sample(lines, min(per_source, len(lines)))
        counts[path.stem] = len(pick)
        out.extend(pick)
    rnd.shuffle(out)
    return out, counts


def outcome(f: Callable[[str], Any], s: str) -> Any:
    try:
        r = f(s)
    except Exception as e:  # noqa: BLE001 - compared
        return ("err", type(e).__name__, str(e))
    return ("ok", r, repr(r), type(r.tzinfo).__name__)


def best(f: Callable[[], Any], repeat: int) -> float:
    times = []
    for _ in range(repeat):
        gc.collect()
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times)


def loop(f: Callable[[str], Any], xs: list[str]) -> Callable[[], None]:
    def run() -> None:
        for s in xs:
            try:
                f(s)
            except Exception:  # noqa: BLE001, S112 - failures are part of the workload
                continue

    return run


def machine() -> dict[str, Any]:
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    return {
        "python": sys.version.split()[0],
        "free_threaded": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled": gil,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "python-dateutil": md.version("python-dateutil"),
        "pyroid": md.version("pyroid"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--per-source", type=int, default=20000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import dateutil.parser as du
    import pyroid.dateutil.parser as pp

    xs, counts = load(args.corpus, args.per_source, args.seed)
    info = machine()
    report: dict[str, Any] = {"machine": info, "sources": counts, "strings": len(xs), "rows": []}
    default = datetime.datetime(2003, 9, 25)

    def du_parse(s: str) -> Any:
        return du.parse(s, default=default)

    def pp_parse(s: str) -> Any:
        return pp.parse(s, default=default)

    bad = [s for s in xs if outcome(du_parse, s) != outcome(pp_parse, s)]
    iso = []
    for s in xs:
        try:
            datetime.datetime.fromisoformat(s)
        except ValueError:
            continue
        iso.append(s)
    bad += [s for s in iso if outcome(du.isoparse, s) != outcome(pp.isoparse, s)]
    report["gate"] = {"compared": len(xs) + len(iso), "different": len(bad), "examples": bad[:20]}
    if bad:
        print(json.dumps(report["gate"], indent=1, default=str))
        if args.out:
            args.out.write_text(json.dumps(report, indent=1, default=str))
        return 1

    def row(name: str, seconds: float, n: int, base: float | None = None) -> None:
        us = seconds / n * 1e6
        r = {"name": name, "us_per_string": us, "strings": n}
        if base is not None:
            r["speedup"] = base / seconds
        report["rows"].append(r)
        extra = f"  {base / seconds:6.1f}x" if base is not None else ""
        print(f"| {name:58} | {us:9.3f} |{extra}", flush=True)

    print(f"{len(xs)} strings from {len(counts)} sources; {len(iso)} ISO strings; {info}")
    print("| call | us/string | vs dateutil |")
    rep = args.repeat
    base = best(loop(du_parse, xs), rep)
    row("dateutil.parser.parse, loop", base, len(xs))
    row("pyroid parse, loop", best(loop(pp_parse, xs), rep), len(xs), base)
    if not info["gil_enabled"]:
        # Free-threaded: dateutil itself on 8 threads (contiguous chunks), for comparison.
        from concurrent.futures import ThreadPoolExecutor

        chunks = [xs[i::8] for i in range(8)]

        def du_threads() -> None:
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda c: loop(du_parse, c)(), chunks))

        row("dateutil.parser.parse, 8 threads", best(du_threads, rep), len(xs), base)
    for t in (1, 8, None):
        f = lambda t=t: pp.parse_batch(xs, errors="coerce", default=default, threads=t)
        row(f"pyroid parse_batch, list, threads={t}", best(f, rep), len(xs), base)
    try:
        import pyarrow as pa

        arr = pa.array(xs)
        for t in (1, 8, None):
            f = lambda t=t: pp.parse_batch(arr, errors="coerce", default=default, ignoretz=True, threads=t)
            row(f"pyroid parse_batch, Arrow, threads={t}", best(f, rep), len(xs), base)
    except ImportError:
        pass
    try:
        import warnings

        import pandas as pd

        series = pd.Series(xs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for fmt in ("mixed", None):
                kw = {"format": fmt} if fmt else {}
                f = lambda kw=kw: pd.to_datetime(series, errors="coerce", utc=True, **kw)
                row(
                    f"pandas.to_datetime(format={fmt!r}) (not dateutil's results)",
                    best(f, rep),
                    len(xs),
                    base,
                )
    except ImportError:
        pass
    if iso:
        print("| ISO strings | us/string | vs dateutil isoparse |")
        ibase = best(loop(du.isoparse, iso), rep)
        row("dateutil.parser.isoparse, loop", ibase, len(iso))
        row("pyroid isoparse, loop", best(loop(pp.isoparse, iso), rep), len(iso), ibase)
        for t in (1, None):
            f = lambda t=t: pp.isoparse_batch(iso, errors="coerce", threads=t)
            row(f"pyroid isoparse_batch, list, threads={t}", best(f, rep), len(iso), ibase)
        fromiso = datetime.datetime.fromisoformat
        row(
            "datetime.fromisoformat, loop (not dateutil's results)",
            best(loop(fromiso, iso), rep),
            len(iso),
            ibase,
        )
        try:
            import ciso8601

            row(
                "ciso8601.parse_datetime, loop (not dateutil's results)",
                best(loop(ciso8601.parse_datetime, iso), rep),
                len(iso),
                ibase,
            )
        except ImportError:
            pass
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
