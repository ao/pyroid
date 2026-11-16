#!/usr/bin/env python3
"""pyroid.difflib against the running interpreter's difflib on real data.

    python bench/difflib/real_data.py --lib OLD_STDLIB_DIR --lib NEW_STDLIB_DIR ... \\
        --places cities15000.txt --companies company_tickers.json --out counts.json

- Source files: every ``*.py`` file found under two consecutive ``--lib`` directories (CPython
  standard libraries of different releases) is diffed old -> new with ``unified_diff``,
  ``context_diff``, ``ndiff``, ``SequenceMatcher`` (``ratio``, ``get_opcodes``) and, for files
  under ``--html-max`` lines, ``HtmlDiff().make_table`` (full and with context).
- Names: ``--places`` (GeoNames ``cities15000.txt``, column 2) and ``--companies`` (the SEC's
  ``company_tickers.json``, ``title``). Each list gets ``--queries`` misspelled names (a deleted,
  inserted, swapped or replaced character, or a lower-cased name), matched against the list with
  ``get_close_matches`` and scored against the names they came from with ``ratio_batch``.

Every result must be identical (floats bit for bit); a difference aborts with the input. Prints
and writes (``--out``) the number of comparisons.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import pyroid.difflib as pd


def same(what: str, expected: Any, actual: Any, where: str) -> None:
    if expected != actual or repr(expected) != repr(actual):
        sys.exit(f"MISMATCH in {what} for {where}")


def read_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines(True)
    except (UnicodeDecodeError, OSError):
        return None


def source_pairs(libs: list[Path]) -> list[tuple[str, Path, Path]]:
    pairs = []
    for old, new in itertools.pairwise(libs):
        for path in sorted(new.rglob("*.py")):
            rel = path.relative_to(new)
            if "site-packages" in rel.parts or "test" in rel.parts[:1]:
                continue
            other = old / rel
            if other.is_file():
                pairs.append((f"{old.name}->{new.name}:{rel}", other, path))
    return pairs


def compare_sources(libs: list[Path], html_max: int, counts: dict[str, int]) -> None:
    for where, old, new in source_pairs(libs):
        a, b = read_lines(old), read_lines(new)
        if a is None or b is None:
            continue
        counts["file pairs"] += 1
        counts["lines"] += len(a) + len(b)
        if a != b:
            counts["changed file pairs"] += 1
        for name in ("unified_diff", "context_diff"):
            same(name, list(getattr(difflib, name)(a, b)), list(getattr(pd, name)(a, b)), where)
        same("ndiff", list(difflib.ndiff(a, b)), list(pd.ndiff(a, b)), where)
        s1, s2 = difflib.SequenceMatcher(None, a, b), pd.SequenceMatcher(None, a, b)
        same("ratio", s1.ratio().hex(), s2.ratio().hex(), where)
        same("get_opcodes", s1.get_opcodes(), s2.get_opcodes(), where)
        counts["diffs compared"] += 5
        if len(a) + len(b) <= html_max:
            for context in (False, True):
                difflib.HtmlDiff._default_prefix = pd.HtmlDiff._default_prefix = 0
                same(
                    "make_table",
                    difflib.HtmlDiff().make_table(a, b, context=context),
                    pd.HtmlDiff().make_table(a, b, context=context),
                    where,
                )
                counts["diffs compared"] += 1


def misspell(rng: random.Random, name: str) -> str:
    if len(name) < 2:
        return name + "x"
    i = rng.randrange(len(name))
    kind = rng.randrange(5)
    if kind == 0:
        return name[:i] + name[i + 1 :]
    if kind == 1:
        return name[:i] + rng.choice("aeiorst ") + name[i:]
    if kind == 2 and i + 1 < len(name):
        return name[:i] + name[i + 1] + name[i] + name[i + 2 :]
    if kind == 3:
        return name[:i] + rng.choice("aeiorst") + name[i + 1 :]
    return name.lower()


def compare_names(label: str, names: list[str], queries: int, counts: dict[str, int]) -> None:
    rng = random.Random(f"pyroid-difflib-{label}")
    names = list(dict.fromkeys(n for n in names if n))
    picked = rng.sample(names, min(queries, len(names)))
    qs = [misspell(rng, n) for n in picked]
    expected = [difflib.SequenceMatcher(None, q, n).ratio() for q, n in zip(qs, picked, strict=True)]
    same("ratio_batch", [r.hex() for r in expected], [r.hex() for r in pd.ratio_batch(qs, picked)], label)
    counts["name pairs scored"] += len(qs)
    pool = names[:20_000]
    for q in qs:
        for n, cutoff in ((3, 0.6), (10, 0.3)):
            same(
                "get_close_matches",
                difflib.get_close_matches(q, pool, n, cutoff),
                pd.get_close_matches(q, pool, n, cutoff),
                f"{label}: {q!r}",
            )
            counts["close-match queries"] += 1
    counts["names"] += len(names)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lib", type=Path, action="append", default=[], help="a stdlib directory (2+)")
    p.add_argument("--places", type=Path, help="GeoNames cities15000.txt")
    p.add_argument("--companies", type=Path, help="SEC company_tickers.json")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--html-max", type=int, default=3000)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    warnings.simplefilter("error", pd.FallbackWarning)
    counts: dict[str, int] = dict.fromkeys(
        [
            "file pairs",
            "changed file pairs",
            "lines",
            "diffs compared",
            "names",
            "name pairs scored",
            "close-match queries",
        ],
        0,
    )
    start = time.monotonic()
    if len(args.lib) >= 2:
        compare_sources(args.lib, args.html_max, counts)
    if args.places:
        rows = args.places.read_text(encoding="utf-8").splitlines()
        compare_names("places", [r.split("\t")[1] for r in rows if r], args.queries, counts)
    if args.companies:
        data = json.loads(args.companies.read_text(encoding="utf-8"))
        compare_names("companies", [v["title"] for v in data.values()], args.queries, counts)
    result = {"python": sys.version.split()[0], "seconds": round(time.monotonic() - start), **counts}
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
