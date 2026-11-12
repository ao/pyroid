#!/usr/bin/env python3
"""pyroid.nltk against NLTK 3.10.4: sent_tokenize and word_tokenize on FineWeb documents.

    python nltk_bench.py --dump-texts /target/en-texts.json               # in an env with pyarrow
    python nltk_bench.py --texts /target/en-texts.json --threads 1,8 -o results/<machine>/<date>/nltk-<py>.json

Needs ``nltk==3.10.4`` with its ``punkt_tab`` data (``NLTK_DATA``) and a pyroid wheel. The texts are the
first ``--docs`` documents of the pinned FineWeb dev slice.

Rows, for ``sent_tokenize`` and ``word_tokenize`` (English model):

* ``nltk x1``: one document at a time on the calling thread.
* ``nltk threads xN``: N Python threads, each tokenizing a contiguous share of the documents (in parallel
  only on a free-threaded build).
* ``nltk processes xN``: a fork ``multiprocessing.Pool(N)`` started before the clock, ``map`` with
  ``chunksize=8``.
* ``pyroid loop x1``: ``pyroid.nltk.<fn>(doc)`` per document, the drop-in loop.
* ``pyroid batch xN``: one ``pyroid.nltk.<fn>(docs, threads=N)`` call.

``load_ms`` times a fresh interpreter from start to the first tokenized sentence (``import`` and model
loading included), minus the same interpreter running ``pass``.

Gate: before anything is timed, pyroid's output must equal NLTK's on every timed document.
Numbers are wall-clock medians; label them with the machine (Docker/OrbStack VM runs are indicative).
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
import threading
import time
from pathlib import Path
from typing import Any

SLICE = "/data/fineweb/fineweb-dev-10k.parquet"


def dump_texts(out: str, n: int) -> None:
    import pyarrow.parquet as pq

    texts = pq.read_table(SLICE, columns=["text"]).column("text").to_pylist()[:n]
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


def _load_time(code: str, runs: int) -> float:
    def once(src: str) -> float:
        t = time.perf_counter()
        subprocess.run([sys.executable, "-W", "ignore", "-c", src], check=True)
        return time.perf_counter() - t

    base = statistics.median(once("pass") for _ in range(runs))
    return statistics.median(once(code) for _ in range(runs)) - base


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-texts")
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--texts")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    if a.dump_texts:
        dump_texts(a.dump_texts, a.docs)
        return
    import nltk
    import pyroid.nltk as pn

    texts: list[str] = json.loads(Path(a.texts).read_text(encoding="utf-8"))
    threads = [int(x) for x in a.threads.split(",")]
    nltk.sent_tokenize("Warm up. Load the model.")
    pn.sent_tokenize("Warm up. Load the model.")
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    rows: dict[str, dict[str, Any]] = {}
    for mode in ("sent_tokenize", "word_tokenize"):
        nf, pf = getattr(nltk, mode), getattr(pn, mode)
        gate = [nf(t) for t in texts] == pf(texts)
        rows[mode] = {"gate": gate}
        if not gate:
            continue
        timings: dict[str, float] = {}
        timings["nltk x1"] = _median(lambda nf=nf: [nf(t) for t in texts], max(1, a.runs // 2))
        timings["pyroid loop x1"] = _median(lambda pf=pf: [pf(t) for t in texts], a.runs)
        for n in threads:
            timings[f"pyroid batch x{n}"] = _median(lambda n=n, pf=pf: pf(texts, threads=n), a.runs)
            if n > 1:
                timings[f"nltk threads x{n}"] = _median(
                    lambda n=n, nf=nf: _threads_map(nf, texts, n), max(1, a.runs // 2)
                )
                with mp.get_context("fork").Pool(n) as pool:
                    pool.map(nf, texts[:n], chunksize=1)  # warm the workers (and load the model)
                    timings[f"nltk processes x{n}"] = _median(
                        lambda p=pool, f=nf: p.map(f, texts, chunksize=8), max(1, a.runs // 2)
                    )
        rows[mode]["us_per_doc"] = {k: round(v / len(texts) * 1e6, 1) for k, v in timings.items()}
    first = "Mr. Smith went to Washington. He liked it."
    load = {
        "pyroid": _load_time(f"import pyroid.nltk as n; n.sent_tokenize({first!r})", a.runs),
        "nltk": _load_time(f"import nltk; nltk.sent_tokenize({first!r})", a.runs),
    }
    out = {
        "what": "NLTK 3.10.4 vs pyroid.nltk on FineWeb (fineweb-dev-10k, first --docs documents)",
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
