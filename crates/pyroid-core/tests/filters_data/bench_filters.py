"""Time pyroid.filters against datatrove 0.10.1 (indicative; run in Docker).

    python bench_filters.py single  [--docs 2000] [--repeat 5]   # one core, vs datatrove
    python bench_filters.py threads [--docs 10000] [--repeat 5]  # pyroid scaling, threads=1/2/4/8

`single` needs datatrove 0.10.1 + spaCy 3.8.18 in the env; `threads` needs only
pyroid (+ pyarrow for the Arrow rows). Documents: the first N of the pinned
FineWeb dev slice. Every pyroid row is checked against datatrove's (or
threads=1's) result before it is timed. Prints one Markdown table.
"""

# ruff: noqa: T201  (a benchmark script)

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

DEV = "/data/fineweb/fineweb-dev-10k.parquet"


def load(n: int) -> list[str]:
    import pyarrow.parquet as pq

    texts: list[str] = pq.read_table(DEV, columns=["text"]).column("text").to_pylist()[:n]
    return texts


def timeit(fn: Callable[[], Any], repeat: int) -> float:
    """Median wall seconds of `repeat` calls (after one warm-up)."""
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def datatrove_cases() -> dict[str, Any]:
    from datatrove.pipeline import filters as dtf

    return {
        "GopherRepetition": dtf.GopherRepetitionFilter(),
        "GopherQuality": dtf.GopherQualityFilter(),
        "C4Quality (default)": dtf.C4QualityFilter(),
        "C4Quality (FineWeb cfg)": dtf.C4QualityFilter(filter_no_terminal_punct=False),
        "FineWebQuality": dtf.FineWebQualityFilter(),
    }


def datatrove_run(filters: list[Any], texts: list[str]) -> list[tuple[bool, str | None, str]]:
    from datatrove.data import Document
    from datatrove.pipeline.filters.base_filter import get_filter_result

    out = []
    for i, t in enumerate(texts):
        doc = Document(text=t, id=str(i))
        keep, reason = True, None
        for f in filters:
            keep, reason = get_filter_result(f.filter(doc))
            if not keep:
                break
        out.append((bool(keep), reason, doc.text))
    return out


def single(n: int, repeat: int) -> None:
    import pyroid.filters as pf

    texts = load(n)
    dt = datatrove_cases()
    ours = {
        "GopherRepetition": pf.GopherRepetition(),
        "GopherQuality": pf.GopherQuality(),
        "C4Quality (default)": pf.C4Quality(),
        "C4Quality (FineWeb cfg)": pf.C4Quality(filter_no_terminal_punct=False),
        "FineWebQuality": pf.FineWebQuality(),
    }
    chain = ["GopherRepetition", "GopherQuality", "C4Quality (FineWeb cfg)", "FineWebQuality"]
    rows = [(name, [dt[name]], [ours[name]]) for name in dt]
    rows.append(("chain: FineWeb's 4 filters", [dt[c] for c in chain], [ours[c] for c in chain]))
    print(f"{n} docs, python {sys.version.split()[0]}, median of {repeat}; us/doc")
    print("| filter | datatrove | pyroid list[str] | pyroid Arrow | speedup (list) |")
    print("|---|---:|---:|---:|---:|")
    import pyarrow as pa

    col = pa.array(texts)
    for name, dfs, pfs in rows:
        want = datatrove_run(dfs, texts)
        got = pf.apply(texts, pfs, threads=1)
        assert list(zip(got.keep, got.reason, got.text, strict=True)) == want, name
        t_dt = timeit(lambda dfs=dfs: datatrove_run(dfs, texts), max(1, repeat // 2))
        t_list = timeit(lambda pfs=pfs: pf.apply(texts, pfs, threads=1), repeat)
        t_arrow = timeit(lambda pfs=pfs: pf.apply(col, pfs, threads=1), repeat)
        print(
            f"| {name} | {t_dt / n * 1e6:,.0f} | {t_list / n * 1e6:,.1f} | "
            f"{t_arrow / n * 1e6:,.1f} | {t_dt / t_list:,.0f}x |"
        )


def threads(n: int, repeat: int) -> None:
    import pyarrow as pa

    import pyroid.filters as pf

    texts = load(n)
    col = pa.array(texts)
    chain = [
        pf.GopherRepetition(),
        pf.GopherQuality(),
        pf.C4Quality(filter_no_terminal_punct=False),
        pf.FineWebQuality(),
    ]
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(
        f"{n} docs, python {sys.version.split()[0]} (GIL {'on' if gil else 'off'}), "
        f"{os.cpu_count()} CPUs, median of {repeat}; FineWeb's 4 filters via apply(); us/doc"
    )
    print("| threads | list[str] | speedup | Arrow | speedup |")
    print("|---:|---:|---:|---:|---:|")
    ref = pf.apply(texts, chain, threads=1)
    base: dict[str, float] = {}
    for t in (1, 2, 4, 8):
        got = pf.apply(texts, chain, threads=t)
        assert (got.keep, got.reason, got.text) == (ref.keep, ref.reason, ref.text)
        cells = []
        for kind, x in (("list", texts), ("arrow", col)):
            s = timeit(lambda x=x, t=t: pf.apply(x, chain, threads=t), repeat)
            base.setdefault(kind, s)
            cells.append(f"{s / n * 1e6:,.1f} | {base[kind] / s:.2f}x")
        print(f"| {t} | " + " | ".join(cells) + " |")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["single", "threads"])
    p.add_argument("--docs", type=int, default=None)
    p.add_argument("--repeat", type=int, default=5)
    a = p.parse_args()
    if a.mode == "single":
        single(a.docs or 2000, a.repeat)
    else:
        threads(a.docs or 10000, a.repeat)


if __name__ == "__main__":
    main()
