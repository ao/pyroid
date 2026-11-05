"""Scaling of `pyroid.ftfy.fix_text_batch` over `threads=` (run in the dev container).

    python bench_batch.py /data/fineweb/fineweb-100k.parquet 40000 1 2 4 8

Times a list[str] input and an Arrow (pyarrow) input, best of 5, and checks
that every thread count gives the same output. Numbers are indicative
(Docker/OrbStack VM).
"""

from __future__ import annotations

import sys
import time

import pyarrow.parquet as pq

import pyroid.ftfy as pf


def best(fn, repeat: int = 5) -> float:
    out = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - start)
    return out


def main() -> None:
    path, n = sys.argv[1], int(sys.argv[2])
    counts = [int(x) for x in sys.argv[3:]] or [1, 2, 4, 8]
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    column = pq.read_table(path, columns=["text"]).column("text").slice(0, n)
    docs = column.to_pylist()
    size = sum(len(d.encode("utf-8")) for d in docs)
    reference = pf.fix_text_batch(docs, threads=1)
    print(f"{sys.version.split()[0]} gil={gil}: {len(docs)} docs, {size / 1e6:.1f} MB")
    for kind, data in (("list", docs), ("arrow", column)):
        base = None
        for t in counts:
            out = pf.fix_text_batch(data, threads=t)
            got = out if kind == "list" else out.to_pylist()
            assert got == reference, f"threads={t} changed the output"
            secs = best(lambda data=data, t=t: pf.fix_text_batch(data, threads=t))
            base = base or secs
            print(
                f"  {kind:5} threads={t}: {secs:.3f} s, {secs * 1e6 / len(docs):.2f} us/doc, "
                f"{size / 1e6 / secs:.0f} MB/s, speedup {base / secs:.2f}x"
            )


if __name__ == "__main__":
    main()
