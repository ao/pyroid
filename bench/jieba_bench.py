#!/usr/bin/env python3
"""pyroid.jieba against jieba 0.42.1: segmentation of FineWeb-2 cmn_Hani documents, and loading.

    python jieba_bench.py --dump-texts /target/zh-texts.json        # in an env with pyarrow: the slice as JSON
    python jieba_bench.py --texts /target/zh-texts.json --threads 1,8 -o results/<machine>/<date>/jieba-<py>.json

Needs ``jieba==0.42.1`` and a pyroid wheel; nothing else (so it runs on free-threaded CPython). The texts are
the pinned ``fineweb2/cmn_Hani.parquet`` slice (250 documents), repeated ``--repeat`` times.

Rows, for ``lcut`` (accurate mode with the HMM, what datatrove and spaCy call) and ``lcut_for_search``:

* ``jieba x1``: one document at a time on the calling thread.
* ``jieba threads xN``: N Python threads, each segmenting a contiguous share of the documents (on a
  free-threaded build they run in parallel; with the GIL they take turns).
* ``jieba processes xN``: a fork ``multiprocessing.Pool(N)`` started before the clock, ``map`` with
  ``chunksize=8`` (how ``jieba.enable_parallel`` and datatrove's executor parallelise).
* ``pyroid loop x1``: ``pyroid.jieba.lcut(doc)`` per document, the drop-in loop.
* ``pyroid batch xN``: one ``pyroid.jieba.lcut(docs, threads=N)`` call.

``load`` rows time a fresh interpreter from start to the first segmented sentence (``import`` included),
minus the same interpreter running ``pass``; jieba with its pickle cache file present (the usual case) and
without it (the first run on a machine).

Gate: before anything is timed, pyroid's words must equal jieba's on every timed document, in both modes.
Numbers are wall-clock medians; label them with the machine (Docker/OrbStack VM runs are indicative).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

SLICE = "/data/fineweb2/cmn_Hani.parquet"


def dump_texts(out: str) -> None:
    import pyarrow.parquet as pq

    texts = pq.read_table(SLICE, columns=["text"]).column("text").to_pylist()
    Path(out).write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(texts)} texts to {out}")


def _median(fn: Any, runs: int) -> float:
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def _threads_map(fn: Any, texts: list[str], n: int) -> list[Any]:
    out: list[Any] = [None] * len(texts)
    step = (len(texts) + n - 1) // n

    def work(lo: int) -> None:
        for i in range(lo, min(lo + step, len(texts))):
            out[i] = fn(texts[i])

    ts = [threading.Thread(target=work, args=(lo,)) for lo in range(0, len(texts), step)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out


def _load_time(code: str, runs: int, env: dict[str, str] | None = None) -> float:
    def once(src: str) -> float:
        t = time.perf_counter()
        subprocess.run([sys.executable, "-W", "ignore", "-c", src], check=True, env=env)
        return time.perf_counter() - t

    base = statistics.median(once("pass") for _ in range(runs))
    return statistics.median(once(code) for _ in range(runs)) - base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-texts")
    ap.add_argument("--texts")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    if a.dump_texts:
        dump_texts(a.dump_texts)
        return
    import jieba
    import pyroid.jieba as pj

    jieba.setLogLevel(logging.WARNING)
    texts: list[str] = json.loads(Path(a.texts).read_text(encoding="utf-8")) * a.repeat
    threads = [int(x) for x in a.threads.split(",")]
    jieba.initialize()
    pj.initialize()
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    rows: dict[str, dict[str, Any]] = {}
    # jieba's module-level _lcut / _lcut_for_search pickle by name (what jieba's own _pcut maps).
    for mode, picklable in (("lcut", jieba._lcut), ("lcut_for_search", jieba._lcut_for_search)):
        jf, pf = getattr(jieba, mode), getattr(pj, mode)
        gate = [jf(t) for t in texts] == pf(texts)
        rows[mode] = {"gate": gate}
        if not gate:
            continue
        timings: dict[str, float] = {}
        timings["jieba x1"] = _median(lambda jf=jf: [jf(t) for t in texts], max(1, a.runs // 2))
        timings["pyroid loop x1"] = _median(lambda pf=pf: [pf(t) for t in texts], a.runs)
        for n in threads:
            timings[f"pyroid batch x{n}"] = _median(lambda n=n, pf=pf: pf(texts, threads=n), a.runs)
            if n > 1:
                timings[f"jieba threads x{n}"] = _median(
                    lambda n=n, jf=jf: _threads_map(jf, texts, n), max(1, a.runs // 2)
                )
                with mp.get_context("fork").Pool(n) as pool:
                    pool.map(picklable, texts[:n], chunksize=1)  # warm the workers
                    timings[f"jieba processes x{n}"] = _median(
                        lambda p=pool, f=picklable: p.map(f, texts, chunksize=8), max(1, a.runs // 2)
                    )
        rows[mode]["us_per_doc"] = {k: v / len(texts) * 1e6 for k, v in timings.items()}
    # Loading: a fresh interpreter, first sentence segmented.
    cache = Path(tempfile.gettempdir()) / "jieba.cache"
    first = 'import logging, jieba; jieba.setLogLevel(logging.WARNING); jieba.lcut("我来到北京清华大学")'
    load = {"pyroid": _load_time('import pyroid.jieba as j; j.lcut("我来到北京清华大学")', a.runs)}
    subprocess.run([sys.executable, "-W", "ignore", "-c", first], check=True)  # writes the cache
    load["jieba (cache file present)"] = _load_time(first, a.runs)
    no_cache = f"import os; os.path.exists({str(cache)!r}) and os.remove({str(cache)!r}); {first}"
    load["jieba (no cache file)"] = _load_time(no_cache, a.runs)
    out = {
        "what": "jieba 0.42.1 vs pyroid.jieba on FineWeb-2 cmn_Hani (250 docs x repeat)",
        "label": "Docker/OrbStack VM, indicative",
        "python": sys.version.split()[0],
        "gil_enabled": gil,
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "loadavg": os.getloadavg(),
        "docs": len(texts),
        "chars": sum(map(len, texts)),
        "rows": rows,
        "load_ms": {k: round(v * 1e3, 1) for k, v in load.items()},
    }
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
