"""Thread scaling of `fix_text` on a free-threaded CPython (run in the dev container).

    python bench_threads.py /data/fineweb/fineweb-100k.parquet 20000 pyroid.ftfy 1 2 4 8

Each thread fixes a contiguous slice of the documents with the single-document
API; the GIL must stay disabled. Numbers are indicative (Docker/OrbStack VM).
"""

from __future__ import annotations

import importlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq


def main() -> None:
    path, n, impl = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    counts = [int(x) for x in sys.argv[4:]] or [1, 2, 4, 8]
    mod = importlib.import_module(impl)
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    docs = pq.read_table(path, columns=["text"]).column("text").to_pylist()[:n]
    fix = mod.fix_text
    for d in docs[:200]:
        fix(d)
    base = None
    for t in counts:
        chunks = [docs[i::t] for i in range(t)]
        best = float("inf")
        with ThreadPoolExecutor(t) as ex:
            for _ in range(3):
                start = time.perf_counter()
                list(ex.map(lambda c: [fix(d) for d in c], chunks))
                best = min(best, time.perf_counter() - start)
        base = base or best
        print(
            f"{impl} {sys.version.split()[0]} gil={gil} threads={t}: {best:.3f} s, "
            f"{best * 1e6 / len(docs):.2f} us/doc, speedup {base / best:.2f}x"
        )


if __name__ == "__main__":
    main()
