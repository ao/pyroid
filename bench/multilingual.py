#!/usr/bin/env python3
"""Per-language benchmark: FineWeb-2's quality filters and MinHash signatures on each FineWeb-2 slice.

    python multilingual.py -o results/<machine>/<date>/multilingual.json            # every slice
    python multilingual.py --langs deu_Latn,cmn_Hani --repeat 4 --runs 5 --threads 1,8

Rows per language (``bench/manifest.json`` ``fineweb2/<lang>.parquet``, each slice repeated ``--repeat``
times so a batch is long enough to time):

* ``filters``: FineWeb-2's filtering step (``fineweb-2-pipeline.py`` step 5: GopherRepetition -> FineWebQuality
  -> GopherQuality with the language's own thresholds and stop words, ``pyroid_parity.specs``), chained as
  datatrove's executor chains them (first drop wins; the chained filters share datatrove's word cache).
* ``minhash``: stage-1 signatures with FineWeb-2's MinhashConfig (xxhash/64, 14 x 8, 5-grams) and the language.

datatrove runs one document at a time; at N > 1 workers it is a fork ``multiprocessing.Pool(N)`` started before
the clock (only the map is timed, which favours it). pyroid gets the whole batch in one call with
``threads=N``. Languages pyroid does not tokenize natively run with ``python_tokenizer=True`` (datatrove's
tokenizer in Python on the calling thread, then pyroid's filter maths); their rows say so.

Gate: before anything is timed, pyroid's keep/reason (filters) and signatures (minhash) must equal datatrove's
on exactly the timed documents, else the language is reported as a gate failure and not timed.
Numbers are wall-clock medians, Docker/OrbStack VM, indicative.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import tempfile
import time
import warnings
from typing import Any

from pyroid_parity import data as D
from pyroid_parity import multilingual as ML
from pyroid_parity import specs
from pyroid_parity.refs import minhash_config

FILTER_ORDER = ("gopher_repetition@fineweb2", "fineweb_quality@fineweb2", "gopher_quality@fineweb2")

_DT: dict[str, Any] = {}  # datatrove objects, built before forking


def _datatrove_filters(lang: str) -> list[Any]:
    from datatrove.pipeline import filters as F

    cases = specs.fineweb2_filter_cases(lang)
    return [getattr(F, cases[c][1])(**cases[c][2]) for c in FILTER_ORDER]


def _dt_chain(text: str) -> tuple[bool, str | None]:
    from datatrove.data import Document
    from datatrove.pipeline.filters.base_filter import get_filter_result

    doc = Document(text=text, id="0")
    for f in _DT["filters"]:
        keep, reason = get_filter_result(f.filter(doc))
        if not keep:
            return False, reason
    return True, None


def _dt_signature(text: str) -> list[int] | None:
    step = _DT["minhash"]
    sh = step.get_shingles(text)
    return None if sh.size == 0 else [int(x) for b in step.get_signature(sh) for x in b]


def _dt_map(fn: Any, texts: list[str], pool: Any) -> list[Any]:
    return [fn(t) for t in texts] if pool is None else pool.map(fn, texts, chunksize=8)


def _median_time(fn: Any, runs: int) -> float:
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def _pyroid_kwargs(lang: str) -> dict[str, Any]:
    from pyroid import filters as F

    try:
        F.apply(["Probe."], [F.GopherQuality(language=lang)])
    except NotImplementedError:
        return {"python_tokenizer": True}
    return {}


def bench_language(lang: str, repeat: int, runs: int, threads: list[int]) -> dict[str, Any]:
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from pyroid import FallbackWarning
    from pyroid import dedup as Dd
    from pyroid import filters as F

    if problem := ML.oracle_problem(lang):
        return {"lang": lang, "skipped": problem}
    texts = ML.fineweb2(lang).texts * repeat
    kw = _pyroid_kwargs(lang)
    cases = specs.fineweb2_filter_cases(lang)
    configs = {
        "gopher_repetition": F.GopherRepetition,
        "fineweb_quality": F.FineWebQuality,
        "gopher_quality": F.GopherQuality,
    }
    steps = [configs[cases[c][0]](**cases[c][2]) for c in FILTER_ORDER]
    mh = ML.minhash_config(lang)
    _DT["filters"] = _datatrove_filters(lang)
    _DT["minhash"] = MinhashDedupSignature(
        output_folder=tempfile.mkdtemp(prefix="pyroid-bench-ml-"),
        config=minhash_config({k: v for k, v in mh.items() if k != "language"}),
        language=lang,
    )
    warnings.simplefilter("ignore", FallbackWarning)

    def py_filters(t: int) -> list[tuple[bool, str | None]]:
        r = F.apply(texts, steps, threads=t, **kw)
        return list(zip(r.keep, r.reason, strict=True))

    def py_minhash(t: int) -> list[Any]:
        return [None if s is None else list(s) for s in Dd.minhash_signatures(texts, threads=t, **mh, **kw)]

    row: dict[str, Any] = {"lang": lang, "docs": len(texts), "python_tokenizer": bool(kw)}
    # Gates (on exactly the timed documents).
    want_f = [_dt_chain(t) for t in texts[: len(texts) // repeat]] * repeat
    want_m = [_dt_signature(t) for t in texts[: len(texts) // repeat]] * repeat
    row["gate"] = {
        "filters": py_filters(1) == want_f,
        "minhash": py_minhash(1) == want_m,
        "kept": sum(k for k, _ in want_f),
    }
    if not all(v for k, v in row["gate"].items() if k != "kept"):
        return row
    timings: dict[str, dict[str, float]] = {}
    for n in threads:
        pool = mp.get_context("fork").Pool(n) if n > 1 else None
        try:
            if pool is not None:  # warm the workers
                pool.map(_dt_chain, texts[:n], chunksize=1)
            timings[f"datatrove filters x{n}"] = {
                "s": _median_time(lambda p=pool: _dt_map(_dt_chain, texts, p), max(1, runs // 2))
            }
            timings[f"datatrove minhash x{n}"] = {
                "s": _median_time(lambda p=pool: _dt_map(_dt_signature, texts, p), max(1, runs // 2))
            }
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        timings[f"pyroid filters x{n}"] = {"s": _median_time(lambda n=n: py_filters(n), runs)}
        timings[f"pyroid minhash x{n}"] = {"s": _median_time(lambda n=n: py_minhash(n), runs)}
    for v in timings.values():
        v["us_per_doc"] = v["s"] / len(texts) * 1e6
    row["timings"] = timings
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default=",".join(specs.FINEWEB2_LANGUAGES))
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    threads = [int(x) for x in a.threads.split(",")]
    for lang in a.langs.split(","):
        D.path(ML.slice_rel(lang))  # pinned and verified
    rows = []
    for lang in a.langs.split(","):
        t0 = time.perf_counter()
        rows.append(bench_language(lang, a.repeat, a.runs, threads))
        sys.stderr.write(f"{lang}: {time.perf_counter() - t0:.0f} s\n")
    out = {
        "what": "FineWeb-2 filters (GopherRepetition -> FineWebQuality -> GopherQuality) + MinHash signatures",
        "label": "Docker/OrbStack VM, indicative",
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "loadavg": os.getloadavg(),
        "rows": rows,
    }
    text = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    print(
        "\n| language | path | stage | "
        + " | ".join(f"datatrove x{n} | pyroid x{n} | speedup x{n}" for n in threads)
        + " |"
    )
    for r in rows:
        if "timings" not in r:
            print(
                f"| {r['lang']} | - | - | {r.get('skipped') or 'gate failed: ' + json.dumps(r.get('gate'))} |"
            )
            continue
        path = "python tokenizer" if r["python_tokenizer"] else "native"
        for stage in ("filters", "minhash"):
            cells = []
            for n in threads:
                d = r["timings"][f"datatrove {stage} x{n}"]["us_per_doc"]
                p = r["timings"][f"pyroid {stage} x{n}"]["us_per_doc"]
                cells.append(f"{d:,.0f} | {p:,.0f} | {d / p:.1f}x")
            print(f"| {r['lang']} | {path} | {stage} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
