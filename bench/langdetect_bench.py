#!/usr/bin/env python3
"""langdetect 1.0.9 vs pyroid.langdetect: single calls, batches and start-up.

    python bench/langdetect_bench.py --fineweb /data/fineweb/fineweb-dev-10k.parquet \\
        --fw2 /data/fineweb2 --out /target/langdetect-bench.json

Needs langdetect 1.0.9, pyarrow and pyroid. Three text sets, from FineWeb (English) and every
FineWeb-2 file in ``--fw2`` (round robin, so the sets are multilingual):

- ``short``: the first 50 characters of documents (titles, tweets, queries);
- ``medium``: 1,500 characters of documents at least that long;
- ``long``: 12,000 characters (documents joined), past langdetect's 10,000-character cut.

Before anything is timed, every text's ``detect_langs`` (languages and ``float.hex`` of every
probability, seed 0) must be equal in both libraries, and ``detect_batch`` must equal pyroid's
per-call results; a difference aborts the run. Then, interleaved, ``--rounds`` times: langdetect
in a loop, langdetect on a warmed fork ``Pool(8)``, pyroid in a loop, and pyroid's
``detect_batch`` with 1 and 8 threads; the table reports the median µs per text. Start-up (a fresh
interpreter's import and first ``detect``) and ``pyroid.patch("langdetect")`` (langdetect's own
``detect`` in a loop, patched) run in subprocesses.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import langdetect
import pyarrow.parquet as pq

import pyroid
import pyroid.langdetect as pl


def _docs(path: Path, n: int) -> list[str]:
    pf = pq.ParquetFile(path)
    out: list[str] = []
    for batch in pf.iter_batches(columns=["text"], batch_size=1024):
        out.extend(batch.column(0).to_pylist())
        if len(out) >= n:
            break
    return out[:n]


def text_sets(fineweb: Path, fw2: Path, n: int) -> dict[str, list[str]]:
    sources = [_docs(fineweb, n)] + [_docs(p, n // 4) for p in sorted(fw2.glob("*.parquet"))]
    mixed: list[str] = []
    for i in range(max(len(s) for s in sources)):
        mixed.extend(s[i] for s in sources if i < len(s))
    short = [d[:50] for d in mixed if d.strip()][:n]
    medium = [d[:1500] for d in mixed if len(d) >= 1500][: n // 2]
    long: list[str] = []
    buf = ""
    for d in mixed:
        buf += d + "\n"
        if len(buf) >= 12_000:
            long.append(buf[:12_000])
            buf = ""
        if len(long) >= n // 8:
            break
    return {"short": short, "medium": medium, "long": long}


def _langs(fn: Callable[[str], Any], text: str) -> Any:
    try:
        return [(x.lang, x.prob.hex()) for x in fn(text)]
    except langdetect.LangDetectException as exc:
        return ("err", str(exc))


def gate(sets: dict[str, list[str]]) -> None:
    for name, texts in sets.items():
        ours = [_langs(pl.detect_langs, t) for t in texts]
        theirs = [_langs(langdetect.detect_langs, t) for t in texts]
        bad = [i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
        if bad:
            sys.exit(f"{name}: {len(bad)} texts differ, e.g. {texts[bad[0]][:80]!r}")
        batch = pl.detect_langs_batch(texts, errors="none")
        mine = [None if r is None else [(x.lang, x.prob.hex()) for x in r] for r in batch]
        expect = [None if isinstance(r, tuple) else r for r in ours]
        if mine != expect:
            sys.exit(f"{name}: detect_langs_batch differs from detect_langs")
        print(f"gate {name}: {len(texts)} texts identical", flush=True)


def _ld_chunk(texts: list[str]) -> list[str | None]:
    out: list[str | None] = []
    for t in texts:
        try:
            out.append(langdetect.detect(t))
        except langdetect.LangDetectException:
            out.append(None)
    return out


def _loop(detect: Callable[[str], str], texts: list[str]) -> None:
    for t in texts:
        try:
            detect(t)
        except langdetect.LangDetectException:
            pass


def _per_text(fn: Callable[[], Any], n: int) -> float:
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) / n * 1e6


def startup(code: str, k: int = 5) -> float:
    times = []
    for _ in range(k):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1e3


def patched_loop(texts_path: Path) -> float:
    code = f"""
import json, time, warnings
warnings.simplefilter("ignore")
import pyroid; pyroid.patch("langdetect")
import langdetect
langdetect.DetectorFactory.seed = 0
texts = json.load(open({str(texts_path)!r}, encoding="utf-8"))
langdetect.detect(texts[0])
best = None
for _ in range(3):
    t0 = time.perf_counter()
    for t in texts:
        try: langdetect.detect(t)
        except langdetect.LangDetectException: pass
    dt = (time.perf_counter() - t0) / len(texts) * 1e6
    best = dt if best is None else min(best, dt)
print(best)
"""
    out = subprocess.run([sys.executable, "-I", "-c", code], check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fineweb", type=Path, required=True)
    ap.add_argument("--fw2", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--label", default="EC2 c8g.16xlarge devbox, shared, indicative")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    langdetect.DetectorFactory.seed = 0
    pl.DetectorFactory.seed = 0
    sets = text_sets(a.fineweb, a.fw2, a.n)
    gate(sets)
    rows: dict[str, dict[str, list[float]]] = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(8) as pool:
        pool.map(_ld_chunk, [["warm up"]] * 8)
        for _ in range(a.rounds):
            for name, texts in sets.items():
                n = len(texts)
                chunks = [texts[i::8] for i in range(8)]
                runs = {
                    "langdetect, one thread": lambda: _loop(langdetect.detect, texts),
                    "langdetect, 8 processes": lambda: pool.map(_ld_chunk, chunks),
                    "pyroid, one call per text": lambda: _loop(pl.detect, texts),
                    "pyroid detect_batch, threads=1": lambda: pl.detect_batch(texts, errors="none", threads=1),
                    "pyroid detect_batch, threads=8": lambda: pl.detect_batch(texts, errors="none", threads=8),
                }
                for label, fn in runs.items():
                    rows.setdefault(label, {}).setdefault(name, []).append(_per_text(fn, n))
    texts_path = a.out.with_suffix(".medium.json")
    texts_path.write_text(json.dumps(sets["medium"]), encoding="utf-8")
    patched = patched_loop(texts_path)
    start = {
        "langdetect": startup("import langdetect; langdetect.detect('hello world, how are you')"),
        "pyroid.langdetect": startup("import pyroid.langdetect as l; l.detect('hello world, how are you')"),
    }
    med = {label: {k: statistics.median(v) for k, v in by.items()} for label, by in rows.items()}
    base = med["langdetect, one thread"]
    print(f"\n{a.label}; CPython {platform.python_version()}; median of {a.rounds} rounds; µs per text\n")
    names = list(sets)
    print("| Run | " + " | ".join(f"{k} ({len(sets[k])} texts)" for k in names) + " |")
    print("|---|" + "---:|" * len(names))
    for label, by in med.items():
        cells = [f"{by[k]:,.1f}" + ("" if label == "langdetect, one thread" else f" ({base[k] / by[k]:.0f}×)") for k in names]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print(f"\npatched langdetect.detect, medium: {patched:.1f} µs ({base['medium'] / patched:.0f}×)")
    print("start-up to the first detect (ms):", {k: round(v) for k, v in start.items()})
    a.out.write_text(
        json.dumps(
            {
                "label": a.label,
                "python": sys.version,
                "machine": platform.machine(),
                "cpus": os.cpu_count(),
                "pyroid": pyroid.__version__,
                "sets": {k: len(v) for k, v in sets.items()},
                "chars": {k: round(statistics.mean(map(len, v))) for k, v in sets.items()},
                "median_us": med,
                "raw_us": rows,
                "patched_medium_us": patched,
                "startup_ms": start,
            },
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
