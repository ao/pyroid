#!/usr/bin/env python3
"""pyroid.metrics vs rouge-score 0.1.2 and sacrebleu 2.6.0: per pair, and corpus-level on 10k and 100k pairs.

Runs in the parity env (it has the oracles), against an installed pyroid wheel (build it with --release):

    DX_NAME=metrics .orchestration/dx.sh 'cd parity && UV_PROJECT_ENVIRONMENT=/target/venv-parity \
      uv run --frozen --no-default-groups --with /target/dist/pyroid-*.whl python ../bench/metrics.py \
      --out ../bench/results/<machine>/<date>-metrics'

Workloads (real data, pinned in bench/manifest.json):
- ROUGE: CNN/DM test, lead-3 vs highlights, ``RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])`` with and
  without the Porter stemmer. 100k pairs = the 11,490 test pairs repeated in order (neither library caches across
  pairs).
- BLEU / chrF: WMT21 newstest2021 segments in file order (8 language pairs, 141 systems), first reference. 10k and
  100k pairs are the first N segments, so every pair is distinct.

Rules (as the rest of bench/):
1. Equivalence first: every timed result is compared, exactly (``pyroid_parity.metrics_cases`` serializers),
   with the reference's result on the same pairs; a mismatch aborts the run.
2. References run at 1 process and at 8 processes (``multiprocessing`` fork pool, chunks of 1/32 of the
   pairs; corpus BLEU/chrF in 8 processes uses sacrebleu's own ``_extract_corpus_statistics`` per chunk, then one
   ``_aggregate_and_compute``). pyroid runs in one process at ``threads=1`` and ``threads=8``.
3. Wall time, median of ``--repeats`` runs (1 run for reference rows over 100k pairs), plus process CPU time.
   Numbers from the shared Docker/OrbStack VM are indicative only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pyroid_parity import metrics_cases as M

ROUGE_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
PROCS = 8
CHUNKS = 32


def log(msg: str) -> None:
    print(f"[metrics-bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------ the workloads

_DATA: dict[str, Any] = {}


def rouge_data(n: int) -> tuple[list[str], list[str]]:
    _, targets, preds = M.rouge_pairs("cnndm")
    reps = -(-n // len(targets))
    return (targets * reps)[:n], (preds * reps)[:n]


def mt_data(n: int) -> tuple[list[str], list[str]]:
    w = M.wmt21()
    assert n <= len(w["hyp"]), f"only {len(w['hyp'])} WMT21 segments"
    return w["hyp"][:n], [r[0] for r in w["refs"][:n]]


# Process-pool workers read the workload from module globals inherited at fork.
def _ref_rouge_chunk(bounds: tuple[int, int]) -> list[str]:
    from rouge_score import rouge_scorer

    lo, hi = bounds
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=_DATA["stem"])
    return [
        M.render_rouge(scorer.score(t, p)) for t, p in zip(_DATA["a"][lo:hi], _DATA["b"][lo:hi], strict=True)
    ]


def _ref_sentence_chunk(bounds: tuple[int, int]) -> list[str]:
    import sacrebleu

    lo, hi = bounds
    fn, render = (
        (sacrebleu.sentence_bleu, M.render_bleu)
        if _DATA["metric"] == "bleu"
        else (sacrebleu.sentence_chrf, M.render_chrf)
    )
    return [render(fn(h, [r])) for h, r in zip(_DATA["a"][lo:hi], _DATA["b"][lo:hi], strict=True)]


def _ref_stats_chunk(bounds: tuple[int, int]) -> list[list[Any]]:
    import sacrebleu

    lo, hi = bounds
    metric = sacrebleu.BLEU() if _DATA["metric"] == "bleu" else sacrebleu.CHRF()
    return metric._extract_corpus_statistics(_DATA["a"][lo:hi], [_DATA["b"][lo:hi]])


def _chunks(n: int) -> list[tuple[int, int]]:
    step = max(1, -(-n // CHUNKS))
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _pool_map(fn: Callable[[tuple[int, int]], list[Any]], n: int) -> list[Any]:
    with mp.get_context("fork").Pool(PROCS) as pool:
        return [x for part in pool.map(fn, _chunks(n)) for x in part]


def rows(kind: str, n: int, stem: bool) -> list[tuple[str, int, Callable[[], list[str]]]]:
    """``[(row name, workers, run)]``; ``run()`` returns the serialized results (one per pair, or one corpus)."""
    import pyroid.metrics.rouge_scorer as prs
    import pyroid.metrics.sacrebleu as psb

    if kind == "rouge":
        a, b = rouge_data(n)
        _DATA.update(a=a, b=b, stem=stem)
        scorer = prs.RougeScorer(ROUGE_TYPES, use_stemmer=stem)

        def ref1() -> list[str]:
            return _ref_rouge_chunk((0, n))

        return [
            ("rouge-score score() loop", 1, ref1),
            ("rouge-score score() loop", PROCS, lambda: _pool_map(_ref_rouge_chunk, n)),
            (
                "pyroid score() loop",
                1,
                lambda: [M.render_rouge(scorer.score(t, p)) for t, p in zip(a, b, strict=True)],
            ),
            (
                "pyroid score_batch",
                1,
                lambda: [M.render_rouge(r) for r in scorer.score_batch(a, b, threads=1)],
            ),
            (
                "pyroid score_batch",
                PROCS,
                lambda: [M.render_rouge(r) for r in scorer.score_batch(a, b, threads=PROCS)],
            ),
        ]
    metric, level = kind.split("-")  # bleu-sentence, bleu-corpus, chrf-sentence, chrf-corpus
    a, b = mt_data(n)
    _DATA.update(a=a, b=b, metric=metric)
    render = M.render_bleu if metric == "bleu" else M.render_chrf
    if level == "sentence":
        fn = psb.sentence_bleu if metric == "bleu" else psb.sentence_chrf
        batch = psb.sentence_bleu_batch if metric == "bleu" else psb.sentence_chrf_batch
        return [
            (f"sacrebleu sentence_{metric}() loop", 1, lambda: _ref_sentence_chunk((0, n))),
            (f"sacrebleu sentence_{metric}() loop", PROCS, lambda: _pool_map(_ref_sentence_chunk, n)),
            (
                f"pyroid sentence_{metric}() loop",
                1,
                lambda: [render(fn(h, [r])) for h, r in zip(a, b, strict=True)],
            ),
            (f"pyroid sentence_{metric}_batch", 1, lambda: [render(s) for s in batch(a, [b], threads=1)]),
            (
                f"pyroid sentence_{metric}_batch",
                PROCS,
                lambda: [render(s) for s in batch(a, [b], threads=PROCS)],
            ),
        ]

    def ref_corpus(procs: int) -> list[str]:
        import sacrebleu

        m = sacrebleu.BLEU() if metric == "bleu" else sacrebleu.CHRF()
        if procs == 1:
            score = m.corpus_score(a, [b])
        else:
            stats = _pool_map(_ref_stats_chunk, n)
            score = m._aggregate_and_compute(stats)
            m.num_refs = 1  # what corpus_score's reference caching sets (it ran in the workers here)
        return [render(score, str(m.get_signature()))]

    def pyroid_corpus(threads: int) -> list[str]:
        m = psb.BLEU() if metric == "bleu" else psb.CHRF()
        return [render(m.corpus_score(a, [b], threads=threads), str(m.get_signature()))]

    return [
        (f"sacrebleu {metric.upper()}.corpus_score", 1, lambda: ref_corpus(1)),
        (f"sacrebleu {metric.upper()}.corpus_score", PROCS, lambda: ref_corpus(PROCS)),
        (f"pyroid {metric.upper()}.corpus_score", 1, lambda: pyroid_corpus(1)),
        (f"pyroid {metric.upper()}.corpus_score", PROCS, lambda: pyroid_corpus(PROCS)),
    ]


# ------------------------------------------------------------------------------------------------ timing


def measure(run: Callable[[], list[str]], repeats: int) -> tuple[list[float], list[float], list[str]]:
    walls, cpus, out = [], [], []
    for _ in range(repeats):
        c0, t0 = time.process_time(), time.perf_counter()
        out = run()
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    return walls, cpus, out


def bench(kind: str, n: int, stem: bool, repeats: int, ref_repeats: int) -> list[dict[str, Any]]:
    label = f"{kind}{'+stem' if stem else ''} n={n}"
    want: list[str] | None = None
    results = []
    for name, workers, run in rows(kind, n, stem):
        reference = not name.startswith("pyroid")
        reps = ref_repeats if reference else repeats
        walls, cpus, out = measure(run, reps)
        if want is None:
            want = out  # the first row is the 1-process reference
        same = out == want
        if not same:
            bad = next(i for i, (g, w) in enumerate(zip(out, want, strict=False)) if g != w)
            sys.exit(
                f"EQUIVALENCE FAILURE {label} / {name} x{workers}: first diff at {bad}: {out[bad]!r} != {want[bad]!r}"
            )
        med = statistics.median(walls)
        log(f"{label:28} {name:34} x{workers}  {med * 1e3:10.1f} ms  ({len(out)} results, gate ok)")
        results.append(
            {
                "workload": kind,
                "stemmer": stem,
                "pairs": n,
                "row": name,
                "reference": reference,
                "workers": workers,
                "wall_s": walls,
                "wall_median_s": med,
                "per_pair_us": med / n * 1e6,
                "cpu_s_calling_process": cpus,
                "gate": "identical to the 1-process reference",
            }
        )
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="1000,10000,100000")
    ap.add_argument("--workloads", default="rouge,bleu-sentence,bleu-corpus,chrf-sentence,chrf-corpus")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--ref-repeats", type=int, default=3)
    a = ap.parse_args()
    import importlib.metadata as md

    import pyroid

    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        sys.exit("run on a GIL build: sacrebleu loads lxml, which re-enables the GIL on free-threaded ones")
    load_before = os.getloadavg()
    out: list[dict[str, Any]] = []
    for kind in a.workloads.split(","):
        for n in map(int, a.sizes.split(",")):
            for stem in (False, True) if kind == "rouge" else (False,):
                ref_reps = 1 if n >= 100_000 else a.ref_repeats
                out += bench(kind, n, stem, a.repeats if n < 100_000 else 3, ref_reps)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "label": "Docker/OrbStack VM, indicative",
        "python": sys.version,
        "cpus": os.cpu_count(),
        "pyroid": pyroid.__version__,
        "versions": {d: md.version(d) for d in ("rouge-score", "sacrebleu", "nltk", "numpy", "regex")},
        "data": {"rouge": M.CNNDM, "mt": M.WMT21},
        "rouge_types": ROUGE_TYPES,
        "reference_processes": PROCS,
        # Other sessions share the VM: the 1/5/15-minute load averages before and after the run.
        "loadavg_before": load_before,
        "loadavg_after": os.getloadavg(),
    }
    (a.out / "metrics.json").write_text(json.dumps({"meta": meta, "results": out}, indent=1) + "\n", "utf-8")
    log(f"wrote {a.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
