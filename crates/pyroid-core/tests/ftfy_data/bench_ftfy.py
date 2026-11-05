"""Single-thread ftfy 6.3.1 (or pyroid.ftfy) `fix_text` timing on a parquet corpus.

Run in the dev container (never on the host):

    python bench_ftfy.py /data/fineweb/fineweb-100k.parquet 10000 [ftfy|pyroid.ftfy]

Prints the best of three passes; numbers are indicative (Docker/OrbStack VM).
"""

from __future__ import annotations

import importlib
import sys
import time

import pyarrow.parquet as pq


def main() -> None:
    path, n = sys.argv[1], int(sys.argv[2])
    impl = sys.argv[3] if len(sys.argv) > 3 else "ftfy"
    mod = importlib.import_module(impl)
    docs = pq.read_table(path, columns=["text"]).column("text").to_pylist()[:n]
    size = sum(len(d.encode("utf-8")) for d in docs)
    fix = mod.fix_text
    for d in docs[:200]:
        fix(d)
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for d in docs:
            fix(d)
        best = min(best, time.perf_counter() - t)
    print(
        f"{impl} {sys.version.split()[0]}: {len(docs)} docs, {size / 1e6:.1f} MB: "
        f"best {best:.3f} s = {best * 1e6 / len(docs):.1f} us/doc, {size / 1e6 / best:.2f} MB/s"
    )


if __name__ == "__main__":
    main()
