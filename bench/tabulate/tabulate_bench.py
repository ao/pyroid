#!/usr/bin/env python3
"""Time pyroid.tabulate against tabulate on generated tables.

    python bench/tabulate/tabulate_bench.py [--quick] [--json out.json]

Tables: small (5x3), medium (1,000x8), large (100,000x10), each numeric (ints and floats)
and text-heavy (words, some numeric strings), plus mixed; formats simple, pipe, grid, github,
html; and DataFrame.to_markdown() (tabulate's "pipe" with the index). Every pyroid result is
checked equal to tabulate's before it is timed. Reports the best of N runs of each.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from typing import Any

import tabulate as ref

import pyroid
import pyroid.tabulate as pt

WORDS = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor".split()


def make(n: int, cols: int, kind: str, seed: int = 1) -> list[list[Any]]:
    rnd = random.Random(seed)
    rows: list[list[Any]] = []
    for i in range(n):
        row: list[Any] = []
        for j in range(cols):
            if kind == "numeric":
                row.append(i * cols + j if j % 2 == 0 else rnd.random() * 10 ** rnd.randint(-3, 6))
            elif kind == "text":
                if j == 0:
                    row.append(f"item-{i}")
                elif j % 3 == 0:
                    row.append(str(rnd.randint(0, 10**6)))
                else:
                    row.append(" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(1, 4))))
            else:  # mixed
                row.append([f"name{i}", i, rnd.random() * 100, None if i % 7 == 0 else "x" * (i % 5), str(i)][j % 5])
        rows.append(row)
    return rows


def best(fn: Any, reps: int) -> float:
    t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json")
    args = ap.parse_args()
    sizes = [("small", 5, 3), ("medium", 1000, 8), ("large", 100_000, 10)]
    if args.quick:
        sizes = [("small", 5, 3), ("medium", 1000, 8), ("large", 20_000, 10)]
    formats = ["simple", "pipe", "grid", "github", "html"]
    results = []
    for size, n, cols in sizes:
        reps_ref = 200 if n <= 5 else (5 if n <= 1000 else 1)
        reps_new = 2000 if n <= 5 else (20 if n <= 1000 else 3)
        for kind in ("numeric", "text", "mixed"):
            data = make(n, cols, kind)
            headers = [f"col{j}" for j in range(cols)]
            for fmt in formats:
                want = ref.tabulate(data, headers=headers, tablefmt=fmt)
                got = pt.tabulate(data, headers=headers, tablefmt=fmt)
                assert got == want, (size, kind, fmt)
                t_ref = best(lambda: ref.tabulate(data, headers=headers, tablefmt=fmt), reps_ref)
                t_new = best(lambda: pt.tabulate(data, headers=headers, tablefmt=fmt), reps_new)
                results.append({"size": size, "rows": n, "cols": cols, "kind": kind, "format": fmt, "tabulate_s": t_ref, "pyroid_s": t_new})
                print(f"{size:6} {n:>7}x{cols:<2} {kind:7} {fmt:7} tabulate {t_ref*1e3:10.3f} ms  pyroid {t_new*1e3:9.3f} ms  {t_ref/t_new:6.1f}x", flush=True)
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if pd is not None:
        for size, n, cols in sizes:
            for kind in ("numeric", "mixed"):
                df = pd.DataFrame(make(n, cols, kind), columns=[f"c{j}" for j in range(cols)])
                want = df.to_markdown()
                got = pt.tabulate(df, headers="keys", tablefmt="pipe")
                assert got == want, (size, kind)
                reps_ref = 200 if n <= 5 else (5 if n <= 1000 else 1)
                reps_new = 2000 if n <= 5 else (20 if n <= 1000 else 3)
                t_ref = best(lambda: ref.tabulate(df, headers="keys", tablefmt="pipe", showindex=True), reps_ref)
                t_new = best(lambda: pt.tabulate(df, headers="keys", tablefmt="pipe", showindex=True), reps_new)
                results.append({"size": size, "rows": n, "cols": cols, "kind": f"dataframe-{kind}", "format": "to_markdown", "tabulate_s": t_ref, "pyroid_s": t_new})
                print(f"{size:6} {n:>7}x{cols:<2} df-{kind:4} to_markdown tabulate {t_ref*1e3:10.3f} ms  pyroid {t_new*1e3:9.3f} ms  {t_ref/t_new:6.1f}x", flush=True)
    if args.json:
        meta = {
            "python": sys.version.split()[0],
            "tabulate": ref.__version__,
            "pyroid": pyroid.__version__,
            "machine": platform.machine(),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
