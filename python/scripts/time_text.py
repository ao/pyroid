"""Indicative timings for pyroid.text vs pure Python on synthetic documents.

Not the benchmark harness (that lives in ``bench/``): a quick check that the
core conventions are fast. Run it against a release wheel, for example::

    python python/scripts/time_text.py --docs 100000

Prints a Markdown table. Every number is the minimum of ``--reps`` runs.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import random
import sys
import sysconfig
import time
from collections.abc import Callable
from typing import Any

import pyroid
import pyroid.text as t

WORDS_ASCII = [
    *("the", "of", "and", "to", "in", "is", "for", "on", "that", "with", "as", "was"),
    *("by", "at", "from", "it", "this", "be", "are", "or", "data", "model", "text"),
    *("clean", "token", "corpus", "filter", "web", "page", "quality", "language", "train"),
]
WORDS_OTHER = [
    "café",
    "naïve",
    "über",
    "señor",
    "Привет",
    "мир",
    "日本語",
    "テキスト",
    "中文",
    "Ελληνικά",
    "😀",
    "“quoted”",
]
SEPARATORS = [" "] * 30 + ["  ", "\t", "\n", "\n\n", "\u00a0", "\u3000", " \r\n"]


def make_docs(n: int, seed: int) -> list[str]:
    """``n`` documents of ~2 KB: 70% pure ASCII, mixed whitespace, 10% clean."""
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        n_words = max(1, int(rng.lognormvariate(5.5, 0.6)))  # median ~245 words
        pool = WORDS_ASCII if rng.random() < 0.7 else WORDS_ASCII + WORDS_OTHER
        words = rng.choices(pool, k=n_words)
        if i % 10 == 0:  # already normalized: returned unchanged
            docs.append(" ".join(words))
            continue
        seps = rng.choices(SEPARATORS, k=n_words)
        docs.append(" " + "".join(w + s for w, s in zip(words, seps, strict=True)))
    return docs


def best(fn: Callable[[], Any], reps: int) -> float:
    """Minimum wall time of ``reps`` calls, in milliseconds."""
    times = []
    for _ in range(reps):
        gc.collect()
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times) * 1e3


def main() -> None:
    """Run the timings and print a Markdown table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=100_000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=1500, help="min_chars threshold")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.compute as pc

    docs = make_docs(args.docs, args.seed)
    mb = sum(len(d.encode()) for d in docs) / 1e6
    n = args.n

    # Correctness first: the timed calls must agree with the reference.
    assert t.normalize_whitespace(docs) == [" ".join(d.split()) for d in docs]
    assert t.min_chars(docs, n) == [len(d) >= n for d in docs]
    arr = pa.array(docs, type=pa.string())
    assert pa.array(t.normalize_whitespace(arr)).to_pylist() == [" ".join(d.split()) for d in docs]

    # Cold list extraction: fresh str objects have no cached UTF-8 yet.
    fresh = [d.encode().decode() for d in docs]
    start = time.perf_counter()
    t.min_chars(fresh, n)
    cold_ms = (time.perf_counter() - start) * 1e3
    del fresh

    rows: list[tuple[str, str, float]] = []

    def add(op: str, impl: str, fn: Callable[[], Any]) -> None:
        rows.append((op, impl, best(fn, args.reps)))

    nw, mc = "normalize_whitespace", "min_chars"
    add(nw, 'pure Python `" ".join(s.split())`', lambda: [" ".join(d.split()) for d in docs])
    add(nw, "pyroid list[str] -> list, threads=1", lambda: t.normalize_whitespace(docs, threads=1))
    add(nw, "pyroid list[str] -> list, threads=None", lambda: t.normalize_whitespace(docs))
    add(nw, "pyroid pa.string -> Column, threads=1", lambda: t.normalize_whitespace(arr, threads=1))
    add(nw, "pyroid pa.string -> Column, threads=None", lambda: t.normalize_whitespace(arr))
    add(mc, "pure Python `len(s) >= n`", lambda: [len(d) >= n for d in docs])
    add(mc, "pyarrow.compute utf8_length >= n", lambda: pc.greater_equal(pc.utf8_length(arr), n))
    add(mc, "pyroid list[str] -> list[bool], threads=1", lambda: t.min_chars(docs, n, threads=1))
    add(mc, "pyroid list[str] -> list[bool], threads=None", lambda: t.min_chars(docs, n))
    add(mc, "pyroid pa.string -> Column[bool], threads=1", lambda: t.min_chars(arr, n, threads=1))
    add(mc, "pyroid pa.string -> Column[bool], threads=None", lambda: t.min_chars(arr, n))

    ft = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    print(
        f"{args.docs:,} docs, {mb:.0f} MB UTF-8; Python {platform.python_version()}"
        f"{' (free-threaded)' if ft else ''}; pyroid {pyroid.__version__}; "
        f"pool {pyroid._pyroid._testing.pool_size()} threads; os.cpu_count()={os.cpu_count()}; "
        f"{platform.machine()}; min of {args.reps}. Docker/OrbStack VM, indicative."
    )
    print(f"Cold first call, list[str] extraction + min_chars: {cold_ms:.0f} ms\n")
    print("| op | implementation | ms |\n|---|---|---:|")
    for op, impl, ms in rows:
        print(f"| {op} | {impl} | {ms:,.1f} |")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
