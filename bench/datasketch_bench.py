"""datasketch 2.0.0 vs pyroid.datasketch: gated MinHash signature and MinHashLSH timings on FineWeb.

    python bench/datasketch_bench.py --docs 10000 --out /target/datasketch-bench.json
    python bench/datasketch_bench.py --docs 2000 --quick              # smoke test

Needs numpy, pyarrow, ``datasketch==2.0.0`` and a pyroid wheel. Runs on any CPython the three install on;
on a free-threaded build the datasketch "8 threads" rows split the work over a thread pool (the GIL is off,
so numpy runs in parallel there), and the GIL must stay off for the whole run.

* Corpus: the first ``--docs`` docs of the pinned FineWeb dev slice (``/data/fineweb/fineweb-dev-10k.parquet``)
  or the 100k headline slice (``--corpus headline``). Shingles are word 5-grams,
  ``" ".join(words[i:i + 5]).encode("utf-8")`` over ``text.split()``, built untimed except in the
  "from text" rows.
* Equivalence gate, before anything is timed: pyroid's signatures equal datasketch's for every doc and scheme,
  and every doc's LSH query result set equals datasketch's (also for ``query_batch``). A mismatch aborts.
* Rows (best of ``--repeats`` wall-clock runs; 128 permutations, seed 1):
  - signatures: ``MinHash.bulk(shingles)`` per scheme; datasketch on 1 thread and over a pool of 8 threads,
    pyroid with ``threads=1`` and ``threads=8``; and from raw text: datasketch = Python shingling + bulk,
    pyroid = ``minhash_batch(texts, ngram=5)``.
  - LSH: ``MinHashLSH(threshold=0.9)`` construction (the ``(b, r)`` search), ``insert`` per doc (prebuilt
    LeanMinHash objects), ``query`` per doc, and pyroid's ``insert_batch`` / ``query_batch`` from the
    signature matrix at 1 and 8 threads.
* Output: a JSON file (``--out``, hostname scrubbed) and a Markdown table on stdout. Numbers measured in the
  Docker/OrbStack VM are indicative.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.metadata as md
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

NUM_PERM = 128
NGRAM = 5
SCHEMES = ("affine32", "legacy", "affine64")
DATA = Path(os.environ.get("PYROID_DATA", "/data"))
CORPORA = {"dev": "fineweb/fineweb-dev-10k.parquet", "headline": "fineweb/fineweb-100k.parquet"}


def shingles(text: str) -> list[bytes]:
    words = text.split()
    return [" ".join(words[i : i + NGRAM]).encode("utf-8") for i in range(len(words) - NGRAM + 1)]


def gil_enabled() -> bool:
    f = getattr(sys, "_is_gil_enabled", None)
    return True if f is None else bool(f())


def best(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    out = None
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out


def chunks(seq: list[Any], n: int) -> list[list[Any]]:
    k = -(-len(seq) // n)
    return [seq[i : i + k] for i in range(0, len(seq), k)]


def pooled(fn: Callable[[list[Any]], list[Any]], seq: list[Any], threads: int) -> list[Any]:
    with cf.ThreadPoolExecutor(threads) as ex:
        return [x for part in ex.map(fn, chunks(seq, threads)) for x in part]


def bulk(mod: Any, docs: list[list[bytes]], scheme: str, **kw: Any) -> list[Any]:
    return list(mod.MinHash.bulk(docs, num_perm=NUM_PERM, scheme=scheme, **kw))


def bulk_text(mod: Any, texts: list[str], scheme: str) -> list[Any]:
    return list(mod.MinHash.bulk([shingles(x) for x in texts], num_perm=NUM_PERM, scheme=scheme))


def build_lsh(mod: Any, keys: list[str], leans: list[Any], threshold: float) -> Any:
    lsh = mod.MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    for k, m in zip(keys, leans, strict=True):
        lsh.insert(k, m)
    return lsh


def build_lsh_batch(mod: Any, keys: list[str], sig: np.ndarray, threshold: float, threads: int) -> Any:
    lsh = mod.MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    lsh.insert_batch(keys, sig, threads=threads)
    return lsh


def query_all(lsh: Any, leans: list[Any]) -> list[Any]:
    return [lsh.query(m) for m in leans]


class Gate(AssertionError):
    pass


def check(ok: bool, what: str) -> None:
    if not ok:
        raise Gate(what)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="dev")
    ap.add_argument("--docs", type=int, default=10_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="1 repeat, affine32 only")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    repeats = 1 if args.quick else args.repeats
    schemes = ("affine32",) if args.quick else SCHEMES
    T = args.threads

    import datasketch as ds
    import pyarrow.parquet as pq
    import pyroid.datasketch as pd

    texts = (
        pq.read_table(DATA / CORPORA[args.corpus], columns=["text"]).column("text").to_pylist()[: args.docs]
    )
    keys = [f"doc{i}" for i in range(len(texts))]
    sh = [shingles(t) for t in texts]
    n_sh = sum(map(len, sh))
    rows: list[dict[str, Any]] = []
    gil0 = gil_enabled()

    def row(group: str, name: str, impl: str, seconds: float, n: int, threads: int) -> None:
        rows.append(
            {"group": group, "row": name, "impl": impl, "threads": threads, "seconds": seconds, "items": n}
        )
        print(f"  {group:<10} {name:<44} {seconds * 1e3:10.1f} ms  {n / seconds:12.0f}/s", flush=True)

    print(f"{len(texts)} docs, {n_sh} shingles; Python {platform.python_version()}, GIL {gil_enabled()}")

    # ---------------------------------------------------------------------------------------- signatures
    sigs: dict[str, np.ndarray] = {}
    for scheme in schemes:
        g = f"sig/{scheme}"
        t, ref = best(partial(bulk, ds, sh, scheme), repeats)
        row(g, "datasketch MinHash.bulk", "datasketch", t, len(sh), 1)
        want = np.array([m.hashvalues for m in ref])
        sigs[scheme] = want
        t, _ = best(partial(pooled, partial(bulk, ds, scheme=scheme), sh, T), repeats)
        row(g, f"datasketch MinHash.bulk, {T}-thread pool", "datasketch", t, len(sh), T)
        for th in (1, T):
            t, got = best(partial(bulk, pd, sh, scheme, threads=th), repeats)
            check(np.array_equal(np.array([m.hashvalues for m in got]), want), f"{g} bulk threads={th}")
            row(g, f"pyroid MinHash.bulk(threads={th})", "pyroid", t, len(sh), th)
        # From raw text: shingling is part of the work.
        g = f"text/{scheme}"
        t, _ = best(partial(bulk_text, ds, texts, scheme), repeats)
        row(g, "datasketch shingles + MinHash.bulk (from text)", "datasketch", t, len(sh), 1)
        t, _ = best(partial(pooled, partial(bulk_text, ds, scheme=scheme), texts, T), repeats)
        row(g, f"datasketch from text, {T}-thread pool", "datasketch", t, len(sh), T)
        for th in (1, T):
            batch = partial(pd.minhash_batch, num_perm=NUM_PERM, scheme=scheme, ngram=NGRAM, threads=th)
            t, got = best(partial(batch, texts), repeats)
            check(np.array_equal(got, want), f"{g} minhash_batch threads={th}")
            row(g, f"pyroid minhash_batch(texts, threads={th})", "pyroid", t, len(sh), th)

    # ---------------------------------------------------------------------------------------- LSH
    sig = sigs["affine32"]
    mods = (("datasketch", ds), ("pyroid", pd))
    for impl, mod in mods:
        t, _ = best(partial(mod.MinHashLSH, threshold=0.9, num_perm=NUM_PERM), max(repeats, 3))
        row("lsh/init", f"{impl} MinHashLSH(threshold=0.9)", impl, t, 1, 1)
    lean = {
        impl: [mod.LeanMinHash(seed=1, hashvalues=sig[i], scheme="affine32") for i in range(len(sig))]
        for impl, mod in mods
    }
    for threshold in (0.9, 0.5):
        g = f"lsh-insert/t{threshold}"
        idx: dict[str, Any] = {}
        for impl, mod in mods:
            t, idx[impl] = best(partial(build_lsh, mod, keys, lean[impl], threshold), repeats)
            row(g, f"{impl} insert per doc", impl, t, len(keys), 1)
        batch_idx = None
        for th in (1, T):
            t, batch_idx = best(partial(build_lsh_batch, pd, keys, sig, threshold, th), repeats)
            row(g, f"pyroid insert_batch(threads={th})", "pyroid", t, len(keys), th)
        g = f"lsh-query/t{threshold}"
        ref_q: list[set[str]] = []
        for impl, _mod in mods:
            t, res = best(partial(query_all, idx[impl], lean[impl]), repeats)
            row(g, f"{impl} query per doc", impl, t, len(keys), 1)
            if impl == "datasketch":
                ref_q = [set(r) for r in res]
            else:
                check(all(set(a) == b for a, b in zip(res, ref_q, strict=True)), f"{g} query")
        n_results = sum(map(len, ref_q))
        for th in (1, T):
            t, res = best(partial(batch_idx.query_batch, sig, threads=th), repeats)
            check(all(set(a) == b for a, b in zip(res, ref_q, strict=True)), f"{g} query_batch threads={th}")
            row(g, f"pyroid query_batch(threads={th})", "pyroid", t, len(keys), th)
        print(f"  ({g}: {n_results} results, {n_results / len(keys):.2f} per query)")

    check(gil_enabled() == gil0, "the GIL state changed during the run")
    if args.out:
        info = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "free_threaded": not gil0 if hasattr(sys, "_is_gil_enabled") else False,
            "machine": platform.machine(),
            "cpus": os.cpu_count(),
            "in_docker": Path("/.dockerenv").exists(),
            "versions": {d: md.version(d) for d in ("datasketch", "numpy", "pyroid")},
            "corpus": CORPORA[args.corpus],
            "docs": len(texts),
            "shingles": n_sh,
            "repeats": repeats,
        }
        args.out.write_text(json.dumps({"info": info, "rows": rows}, indent=1) + "\n", "utf-8")
    print_summary(rows)
    return 0


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n| group | row | threads | ms | items/s | vs datasketch (1 thread) |")
    print("|---|---|---:|---:|---:|---:|")
    base: dict[str, float] = {}
    for r in rows:
        if r["impl"] == "datasketch" and r["threads"] == 1 and r["group"] not in base:
            base[r["group"]] = r["seconds"]
    for r in rows:
        b = base.get(r["group"])
        speed = f"{b / r['seconds']:.1f}x" if b else ""
        print(
            f"| {r['group']} | {r['row']} | {r['threads']} | {r['seconds'] * 1e3:.1f} | "
            f"{r['items'] / r['seconds']:.0f} | {speed} |"
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Gate as e:
        print(f"equivalence gate failed: {e}", file=sys.stderr)
        raise SystemExit(1) from None
