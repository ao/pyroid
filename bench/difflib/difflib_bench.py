#!/usr/bin/env python3
"""The standard library's difflib vs pyroid.difflib.

    python bench/difflib/difflib_bench.py --lib OLD_STDLIB --lib NEW_STDLIB \\
        --places cities15000.txt --companies company_tickers.json --out results.json
    python bench/difflib/difflib_bench.py ... --quick           # smoke test

Both run in the same process, interleaved; each timing is the best of ``--repeat`` rounds (the
median is recorded too). Before a case is timed, both results must be identical (floats bit for
bit); a mismatch aborts the run.

Cases:

- ratio: ``SequenceMatcher(None, a, b).ratio()`` in a Python loop over 10,000 pairs of a company
  or place name and a misspelling of it, and over 1,000 pairs of 500-character paragraphs; and
  the same pairs through ``ratio_batch`` (one thread and all of them). rapidfuzz's
  ``fuzz.ratio`` is timed on the same pairs for context when it is installed (it computes a
  different, Levenshtein-based score).
- get_close_matches: a misspelled place name against 10,000 and 100,000 place names (GeoNames),
  and 1,000,000 (those names with numbered suffixes); ``close_matches_batch`` for 1,000 queries
  against 10,000 names.
- diffs: ``unified_diff``, ``context_diff``, ``ndiff`` and ``HtmlDiff().make_file`` of real files
  between two CPython releases (``--lib``): argparse.py and typing.py, and every ``*.py`` file of
  the standard library (``unified_diff`` only).

Prints a Markdown table; ``--out`` writes the timings and the machine (CPU, Python, versions and
the load average before and after; no hostname or instance id).
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import os
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyroid
import pyroid.difflib as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


def machine() -> dict[str, Any]:
    """The machine, without identifiers (no hostname or instance id)."""
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
        "machine": platform.machine(),
        "cpu": cpu,
        "cpus": os.cpu_count(),
        "pyroid": pyroid.__version__,
        "rapidfuzz": getattr(sys.modules.get("rapidfuzz"), "__version__", None),
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }


def best(f: Callable[[], Any], repeat: int) -> tuple[float, float]:
    times = []
    for _ in range(repeat):
        gc.collect()
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return min(times), statistics.median(times)


ROWS: list[dict[str, Any]] = []


def case(
    name: str,
    unit: str,
    per: int,
    std: Callable[[], Any],
    pyr: dict[str, Callable[[], Any]],
    repeat: int,
    other: dict[str, Callable[[], Any]] | None = None,
) -> None:
    expected = std()
    for label, f in pyr.items():
        got = f()
        if repr(got) != repr(expected):
            sys.exit(f"MISMATCH: {name} ({label})")
    timings: dict[str, Any] = {}
    s_best, s_med = best(std, repeat)
    timings["difflib"] = {"best": s_best / per, "median": s_med / per}
    for label, f in {**pyr, **(other or {})}.items():
        b, m = best(f, repeat)
        timings[label] = {"best": b / per, "median": m / per}
    row = {"case": name, "unit": unit, "per": per, "timings": timings}
    ROWS.append(row)
    scale = 1e6 if unit == "us" else 1e3
    cells = [f"{timings['difflib']['best'] * scale:,.2f} {unit}"]
    for label in {**pyr, **(other or {})}:
        t = timings[label]["best"]
        speed = s_best / per / t if label in pyr else None
        cells.append(f"{t * scale:,.2f} {unit}" + (f" ({speed:,.1f}x)" if speed else ""))
    labels = ", ".join(["difflib", *pyr, *(other or {})])
    print(f"| {name} | {' | '.join(cells)} |  <!-- {labels} -->", flush=True)


def misspell(rng: random.Random, name: str) -> str:
    i = rng.randrange(max(1, len(name)))
    kind = rng.randrange(4)
    if kind == 0:
        return name[:i] + name[i + 1 :]
    if kind == 1:
        return name[:i] + rng.choice("aeiorst") + name[i:]
    if kind == 2 and i + 1 < len(name):
        return name[:i] + name[i + 1] + name[i] + name[i + 2 :]
    return name.lower()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lib", type=Path, action="append", default=[])
    p.add_argument("--places", type=Path, required=True)
    p.add_argument("--companies", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    repeat = 2 if args.quick else args.repeat
    rng = random.Random(1)
    info = {"before": machine()}
    places = list(
        dict.fromkeys(
            r.split("\t")[1] for r in args.places.read_text(encoding="utf-8").splitlines() if r
        )
    )
    companies = [v["title"] for v in json.loads(args.companies.read_text(encoding="utf-8")).values()]
    names = rng.sample(places + companies, 10_000)
    typos = [misspell(rng, n) for n in names]
    n_pairs = 1000 if args.quick else len(names)
    a, b = typos[:n_pairs], names[:n_pairs]
    print("| case | difflib | pyroid ... |\n|---|---|---|")
    SM, PSM = difflib.SequenceMatcher, pd.SequenceMatcher
    other = {"rapidfuzz fuzz.ratio (another score)": lambda: [fuzz.ratio(x, y) for x, y in zip(a, b)]}
    case(
        "SequenceMatcher(None, a, b).ratio(), names",
        "us",
        n_pairs,
        lambda: [SM(None, x, y).ratio() for x, y in zip(a, b)],
        {
            "pyroid.difflib": lambda: [PSM(None, x, y).ratio() for x, y in zip(a, b)],
            "ratio_batch, 1 thread": lambda: pd.ratio_batch(a, b, threads=1),
            "ratio_batch": lambda: pd.ratio_batch(a, b),
        },
        repeat,
        other if fuzz else None,
    )
    words = " ".join(places[:20000]).split()
    paras = [" ".join(rng.choices(words, k=70))[:500] for _ in range(200 if args.quick else 1000)]
    edited = [misspell(rng, misspell(rng, x)).replace(" ", "  ", 3) for x in paras]
    other = {"rapidfuzz fuzz.ratio (another score)": lambda: [fuzz.ratio(x, y) for x, y in zip(paras, edited)]}
    case(
        "SequenceMatcher(None, a, b).ratio(), 500 characters",
        "us",
        len(paras),
        lambda: [SM(None, x, y).ratio() for x, y in zip(paras, edited)],
        {
            "pyroid.difflib": lambda: [PSM(None, x, y).ratio() for x, y in zip(paras, edited)],
            "ratio_batch, 1 thread": lambda: pd.ratio_batch(paras, edited, threads=1),
            "ratio_batch": lambda: pd.ratio_batch(paras, edited),
        },
        repeat,
        other if fuzz else None,
    )
    sizes = (10_000,) if args.quick else (10_000, 100_000, 1_000_000)
    for size in sizes:
        pool = places[:size] if size <= len(places) else [
            f"{n} {i}" for i in range(size // len(places) + 1) for n in places
        ][:size]
        query = misspell(rng, rng.choice(places[:1000]))
        r = 1 if size >= 1_000_000 else repeat
        case(
            f"get_close_matches, {size:,} place names",
            "ms",
            1,
            lambda pool=pool, query=query: difflib.get_close_matches(query, pool),
            {
                "pyroid.difflib, 1 thread": lambda pool=pool, query=query: pd.get_close_matches(query, pool, threads=1),
                "pyroid.difflib": lambda pool=pool, query=query: pd.get_close_matches(query, pool),
            },
            r,
        )
    queries = typos[:100 if args.quick else 1000]
    pool = places[:10_000]
    case(
        f"close_matches_batch, {len(queries):,} queries x 10,000 names",
        "ms",
        1,
        lambda: [difflib.get_close_matches(q, pool) for q in queries],
        {
            "close_matches_batch, 1 thread": lambda: pd.close_matches_batch(queries, pool, threads=1),
            "close_matches_batch": lambda: pd.close_matches_batch(queries, pool),
        },
        1,
    )
    if len(args.lib) == 2:
        old, new = args.lib
        for name in ("argparse.py", "typing.py"):
            x = (old / name).read_text(encoding="utf-8").splitlines(True)
            y = (new / name).read_text(encoding="utf-8").splitlines(True)
            label = f"{name} {old.name[6:]} -> {new.name[6:]} ({len(x):,} / {len(y):,} lines)"
            for fn in ("unified_diff", "context_diff", "ndiff"):
                case(
                    f"{fn}, {label}",
                    "ms",
                    1,
                    lambda fn=fn, x=x, y=y: list(getattr(difflib, fn)(x, y)),
                    {"pyroid.difflib": lambda fn=fn, x=x, y=y: list(getattr(pd, fn)(x, y))},
                    repeat,
                )

            def html(module: Any, x: list[str] = x, y: list[str] = y) -> str:
                module.HtmlDiff._default_prefix = 0
                return module.HtmlDiff().make_file(x, y)

            case(
                f"HtmlDiff().make_file, {label}",
                "ms",
                1,
                lambda html=html: html(difflib),
                {"pyroid.difflib": lambda html=html: html(pd)},
                repeat,
            )
        pairs = []
        for path in sorted(new.rglob("*.py")):
            rel = path.relative_to(new)
            if "site-packages" in rel.parts or (old / rel).is_file() is False:
                continue
            try:
                pairs.append(
                    (
                        (old / rel).read_text(encoding="utf-8").splitlines(True),
                        path.read_text(encoding="utf-8").splitlines(True),
                    )
                )
            except UnicodeDecodeError:
                continue
        total = sum(len(x) + len(y) for x, y in pairs)
        case(
            f"unified_diff, the standard library {old.name[6:]} -> {new.name[6:]} "
            f"({len(pairs):,} files, {total:,} lines)",
            "ms",
            1,
            lambda: [list(difflib.unified_diff(x, y)) for x, y in pairs],
            {"pyroid.difflib": lambda: [list(pd.unified_diff(x, y)) for x, y in pairs]},
            3 if not args.quick else 1,
        )
    info["after"] = machine()
    if args.out:
        args.out.write_text(json.dumps({"machine": info, "rows": ROWS}, indent=2) + "\n")


if __name__ == "__main__":
    main()
