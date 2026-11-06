#!/usr/bin/env python3
"""Equivalence + timing check: the benchmark's datatrove FineWeb chain with pyroid's drop-in steps.

Runs exactly ``bench/pipeline.py``'s pipeline (datatrove's own ``LocalPipelineExecutor``,
tasks = workers = N, its readers/writers/exclusion writers and its MinHash stages 2-4), with
these steps swapped for ``pyroid.integrations.datatrove``'s: FTFYFormatter (ftfy's default
config, as the benchmark), LanguageFilter, GopherRepetitionFilter, GopherQualityFilter,
C4QualityFilter(filter_no_terminal_punct=False), FineWebQualityFilter, MinhashDedupSignature.
Prints the same summary and per-doc digest as pipeline.py; the digest must equal the stock
chain's (``--expect``).

Needs a pyroid checkout (for bench/) and the bench environment (datatrove 0.10.1, /data):

    cd bench && python ../examples/datatrove/verify.py --corpus dev --workers 8 \
        --out /target/pd-w8 \
        --expect ef314b8d6ae95cc63c6cfa8b1a3a76ee5bc101eb88a66729beb10ecb7226473b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parents[2] / "bench"
sys.path.insert(0, str(BENCH))

import gates as G  # noqa: E402, N812
import pipeline as P  # noqa: E402, N812
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter  # noqa: E402
from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets  # noqa: E402
from pyroid_parity import refs as R  # noqa: E402, N812

from pyroid.integrations import datatrove as PD  # noqa: E402, N812

STAGES = P.STAGES


def make_step(stage: str, work: Path, kw: dict[str, Any]) -> Any:
    excl = P.writer(work / "excluded" / stage) if stage != "fix_text" else None
    if (
        stage == "fix_text"
    ):  # ftfy's default TextFixerConfig, spelled with FTFYFormatter's arguments
        return PD.FTFYFormatter(
            fix_latin_ligatures=True,
            fix_character_width=True,
            uncurl_quotes=True,
            fix_line_breaks=True,
            normalization="NFC",
            **kw,
        )
    if stage == "lid":
        return PD.LanguageFilter(
            **G.LID_FILTER, exclusion_writer=excl, model_path=G.lid_model_path("lid.176.bin"), **kw
        )
    cls = {
        "gopher_repetition": PD.GopherRepetitionFilter,
        "gopher_quality": PD.GopherQualityFilter,
        "c4_quality": PD.C4QualityFilter,
        "fineweb_quality": PD.FineWebQualityFilter,
    }[stage]
    cfg = {"filter_no_terminal_punct": False} if stage == "c4_quality" else {}
    return cls(exclusion_writer=excl, **cfg, **kw)


def run(a: argparse.Namespace, segs: list[tuple[Path, int]], n: int, work: Path) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    kw = {"batch_size": a.batch_size, "threads": a.threads or None}
    shard = P.ShardReader(segs, n)
    per_doc = [s for s in STAGES if s != "minhash"]
    for d in ("kept", "final", "dup", *(f"excluded/{s}" for s in per_doc)):
        (work / d).mkdir(parents=True, exist_ok=True)
    steps = [shard, *(make_step(s, work, kw) for s in per_doc)]
    P.run_phase("filters", [*steps, P.writer(work / "kept")], a.workers, a.workers, work, phases)
    cfg = R.minhash_config()
    mh = work / "minhash"
    P.run_phase(
        "minhash_signature",
        [
            P.reader(work / "kept"),
            PD.MinhashDedupSignature(output_folder=str(mh / "sigs"), config=cfg, **kw),
        ],
        a.workers,
        a.workers,
        work,
        phases,
    )
    P.run_phase(
        "minhash_buckets",
        [
            MinhashDedupBuckets(
                input_folder=str(mh / "sigs"), output_folder=str(mh / "buckets"), config=cfg
            )
        ],
        cfg.num_buckets,
        a.workers,
        work,
        phases,
    )
    P.run_phase(
        "minhash_cluster",
        [
            MinhashDedupCluster(
                input_folder=str(mh / "buckets"), output_folder=str(mh / "remove_ids"), config=cfg
            )
        ],
        1,
        1,
        work,
        phases,
    )
    P.run_phase(
        "minhash_filter",
        [
            P.reader(work / "kept"),
            MinhashDedupFilter(
                input_folder=str(mh / "remove_ids"), exclusion_writer=P.writer(work / "dup")
            ),
            P.writer(work / "final"),
        ],
        a.workers,
        a.workers,
        work,
        phases,
    )
    return phases


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", default="dev", choices=sorted(G.SEGMENTS))
    ap.add_argument("--docs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--threads", type=int, default=1, help="pyroid threads= per step (0 = the whole pool)"
    )
    ap.add_argument("--batch-size", type=int, default=PD.DEFAULT_BATCH_SIZE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", help="the stock chain's digest (exit 1 if different)")
    ap.add_argument("--no-digest", action="store_true")
    ap.add_argument("--summary-log")
    ap.add_argument(
        "--keep-work", action="store_true", help="keep OUT/work (datatrove logs and stats)"
    )
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    segs = G.segment_rows(a.corpus)
    n = a.docs or sum(k for _, k in segs)
    out = Path(a.out)
    work = out / "work"
    shutil.rmtree(out, ignore_errors=True)
    work.mkdir(parents=True)
    t0 = time.perf_counter()
    phases = run(a, segs, n, work)
    wall = time.perf_counter() - t0
    ru = P.rusage()
    summary: dict[str, Any] = {
        "tool": "examples/datatrove/verify.py",
        "label": a.label,
        "corpus": a.corpus,
        "docs": n,
        "workers": a.workers,
        "pyroid_threads": a.threads or "pool",
        "batch_size": a.batch_size,
        "wall_s": round(wall, 3),
        "phases": phases,
        "cpu_s": ru["total"],
        "counts": P.counts(work, list(STAGES), n),
        "python": sys.version.split()[0],
    }
    rc = 0
    if not a.no_digest:
        ids = [i for i, _ in G.iter_slice(segs, 0, n)]
        recs = P.collect(work, list(STAGES), ids)
        summary["digest"], summary["digest_exact"] = P.digests(recs, ids)
        if a.expect:
            summary["digest_matches"] = summary["digest"] == a.expect
            rc = 0 if summary["digest_matches"] else 1
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    text = json.dumps(summary, indent=1, sort_keys=True)
    (out / "summary.json").write_text(text)
    if a.summary_log:
        with open(a.summary_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(text)  # noqa: T201
    return rc


if __name__ == "__main__":
    sys.exit(main())
