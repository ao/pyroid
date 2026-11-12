#!/usr/bin/env python3
"""The FineWeb-style chain of ``pipeline.py``, run natively in pyroid: one process, Arrow in, ``threads=N``.

    python chain.py --corpus dev --threads 8 --digest            # timing summary + the per-doc digest
    hyperfine -N 'python chain.py --corpus dev --threads 8' ...   # timing runs (no digest)

Same stages, configs and order as ``pipeline.py``: ftfy.fix_text (default config) -> LID (lid.176.bin, keep iff
the top label is en with p > 0.65) -> GopherRepetition -> GopherQuality -> C4Quality(filter_no_terminal_punct=
False) -> FineWebQuality -> MinHash dedup (specs.MINHASH_CONFIG) over the survivors, in corpus order.

Equivalence gate: ``--digest`` computes ``pipeline.py``'s sha256 over one JSON line per doc
``[id, text, lang, score (%.6f), keep, reason, dup]``; it must equal the datatrove chain's digest on the same
docs (``--expect DIGEST``, or the e2e/equivalence.json of a harness run). Collecting the outputs for the
digest happens after the timed region, as in ``pipeline.py --no-digest`` timing runs.

Timed region (``wall_s.chain``): every stage, from the Arrow column read off the pinned parquet to the
dedup mask. ``text.replace("\\n", " ")`` for LID is pyarrow.compute.replace_substring on the column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from typing import Any

T_START = time.perf_counter()

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyroid
import pyroid.dedup
import pyroid.fasttext
import pyroid.filters as F
import pyroid.ftfy
from pyroid_parity import specs

import gates as G

FILTERS = ("gopher_repetition", "gopher_quality", "c4_quality", "fineweb_quality")
LID_MODEL = "lid.176.bin"


def load(corpus: str, n: int) -> tuple[list[str], Any]:
    """ids and the text column (Arrow, as read) of the first ``n`` docs (0 = all) of the pinned corpus."""
    tables = [pq.read_table(path, columns=["id", "text"]) for path, _ in G.segment_rows(corpus)]
    t = pa.concat_tables(tables)
    if n:
        t = t.slice(0, n)
    return t.column("id").to_pylist(), t.column("text")


def stage_of_reason() -> dict[str, str | None]:
    """datatrove reason -> filter stage, from the pinned filters reference; ``None`` when two filters use it
    (``digest`` then finds the stage by re-running that doc through growing prefixes of the chain)."""
    ref = pq.read_table(G.D.path("parity/filters-ref.parquet"), columns=["case", "reason"]).to_pylist()
    out: dict[str, str | None] = {}
    for r in ref:
        stage = r["case"].split("@")[0]
        if r["reason"] and stage in FILTERS and out.setdefault(r["reason"], stage) != stage:
            out[r["reason"]] = None
    return out


def stage_by_prefix(text: str) -> str:
    """The first filter of the chain that drops ``text`` (only for reasons two filters share)."""
    for k, stage in enumerate(FILTERS, 1):
        if not F.apply([text], CHAIN[:k]).keep[0]:
            return stage
    raise AssertionError("the chain dropped this doc but no prefix of it does")


def run(texts: Any, threads: int | None) -> dict[str, Any]:
    """The chain; returns every intermediate needed for the digest (no Python strs built here)."""
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    fixed = pa.chunked_array(pyroid.ftfy.fix_text_batch(texts, threads=threads))
    t["fix_text"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = MODEL[0]
    labels, probs = model.predict(pc.replace_substring(fixed, "\n", " "), k=1, threads=threads)
    lid_keep = [
        bool(lb) and lb[0] == "__label__en" and float(p[0]) > G.LID_FILTER["language_threshold"]
        for lb, p in zip(labels, probs, strict=True)
    ]
    t["lid"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    after_lid = fixed.filter(pa.array(lid_keep))
    res = F.apply(after_lid, CHAIN, threads=threads)
    t["filters"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    keep_f = pa.chunked_array(res.keep)
    survivors = pa.chunked_array(res.text).filter(keep_f)
    dedup_keep = pyroid.dedup.MinHashDedup(threads=threads, **specs.MINHASH_CONFIG).fit(survivors)
    t["minhash"] = time.perf_counter() - t0
    return {
        "seconds": t,
        "fixed": fixed,
        "labels": labels,
        "probs": probs,
        "lid_keep": lid_keep,
        "res": res,
        "keep_f": keep_f,
        "dedup_keep": dedup_keep,
    }


def digest(ids: list[str], out: dict[str, Any]) -> tuple[str, dict[str, int]]:
    """``pipeline.py``'s digest (score formatted %.6f) and the removal counts per stage."""
    by_reason = stage_of_reason()
    fixed = out["fixed"].to_pylist()
    res = out["res"]
    f_keep, f_reason, f_text = (pa.chunked_array(c).to_pylist() for c in (res.keep, res.reason, res.text))
    dedup_keep = pa.chunked_array(out["dedup_keep"]).to_pylist()
    h = hashlib.sha256()
    counts = dict.fromkeys(("lid", *FILTERS, "minhash"), 0)
    fi = di = 0
    for n, i in enumerate(ids):
        lb, p = out["labels"][n], out["probs"][n]
        lang, score = (lb[0].removeprefix("__label__"), f"{float(p[0]):.6f}") if lb else (None, None)
        if not out["lid_keep"][n]:
            row = [i, fixed[n], lang, score, False, "lid", None]
            counts["lid"] += 1
        else:
            if not f_keep[fi]:
                stage = by_reason[f_reason[fi]] or stage_by_prefix(fixed[n])
                row = [i, f_text[fi], lang, score, False, f"{stage}:{f_reason[fi]}", None]
                counts[stage] += 1
            else:
                dup = not dedup_keep[di]
                row = [i, f_text[fi], lang, score, True, None, dup]
                counts["minhash"] += dup
                di += 1
            fi += 1
        h.update(json.dumps(row, ensure_ascii=False).encode("utf-8", "surrogatepass") + b"\n")
    return h.hexdigest(), counts


MODEL: list[Any] = []
CHAIN = [
    F.GopherRepetition(),
    F.GopherQuality(),
    F.C4Quality(filter_no_terminal_punct=False),
    F.FineWebQuality(),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="dev", choices=sorted(G.SEGMENTS))
    ap.add_argument("--docs", type=int, default=0, help="first N docs (0 = all)")
    ap.add_argument("--threads", type=int, default=0, help="pyroid threads= (0 = None: the whole pool)")
    ap.add_argument("--digest", action="store_true", help="collect outputs after timing and print the digest")
    ap.add_argument("--expect", help="digest the datatrove chain gave on the same docs (exit 1 if different)")
    ap.add_argument("--summary-log", help="append the summary as one JSON line to this file")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    if getattr(sys, "_is_gil_enabled", lambda: False)() and "t" in sys.abiflags:
        raise SystemExit("free-threaded interpreter with the GIL re-enabled: thread numbers would be invalid")
    threads = a.threads or None
    t_load = time.perf_counter()
    ids, texts = load(a.corpus, a.docs)
    MODEL.append(pyroid.fasttext.load_model(G.lid_model_path(LID_MODEL)))
    load_s = time.perf_counter() - t_load
    t0 = time.perf_counter()
    out = run(texts, threads)
    wall = time.perf_counter() - t0
    ru = resource.getrusage(resource.RUSAGE_SELF)
    summary: dict[str, Any] = {
        "tool": "bench/chain.py",
        "label": a.label,
        "corpus": a.corpus,
        "docs": len(ids),
        "threads": a.threads or "pool",
        "pool_size_env": {k: os.environ.get(k) for k in ("PYROID_NUM_THREADS", "RAYON_NUM_THREADS")},
        "python": sys.version.split()[0],
        "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        "pyroid": pyroid.__version__,
        "wall_s": {
            "chain": round(wall, 3),
            **{k: round(v, 3) for k, v in out["seconds"].items()},
            "load_and_model": round(load_s, 3),
            "startup_imports": round(t_load - T_START, 3),
        },
        "cpu_s": round(ru.ru_utime + ru.ru_stime, 3),
        "maxrss_mib": round(ru.ru_maxrss / 1024, 1),
    }
    rc = 0
    if a.digest or a.expect:
        summary["digest"], summary["removed"] = digest(ids, out)
        if a.expect and summary["digest"] != a.expect:
            print(f"DIGEST MISMATCH: {summary['digest']} != {a.expect}", file=sys.stderr)
            rc = 1
        summary["digest_matches"] = None if not a.expect else summary["digest"] == a.expect
    text = json.dumps(summary, indent=1, sort_keys=True)
    if a.summary_log:
        with open(a.summary_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(text)
    return rc


if __name__ == "__main__":
    sys.exit(main())
