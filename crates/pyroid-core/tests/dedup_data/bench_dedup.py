"""Indicative timings for pyroid.dedup against datatrove, rensa and datasketch (Docker only).

Run inside the pyroid-dev container, never on the host, with an env that has the pyroid wheel,
pyarrow and (for the incumbent rows) the bench harness's dependencies:

    python bench_dedup.py signatures [--docs 1000] [--repeat 7] [--rows a,b,...]
        per-document signature cost, single thread, over the first --docs FineWeb dev docs
    python bench_dedup.py pipeline --corpus dev|headline [--threads 1,2,4,8] [--repeat 3]
        MinHashDedup.fit over a whole corpus: wall, CPU, peak RSS; one fresh process per run
    python bench_dedup.py datatrove-chain --docs N
        datatrove's MinHash stages in one process (1 task, 14 bucket tasks): wall and CPU

Every number is a median of --repeat runs after one warm-up run. Label results
"Docker/OrbStack VM, indicative". Rows other than ``pyroid tokens`` use pyroid's whitespace
stand-in tokenizer until the spaCy-exact one is wired in; their signatures are not datatrove's
for ~31% of documents.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
CORPORA = {
    "dev": DATA / "fineweb" / "fineweb-dev-10k.parquet",
    "headline": DATA / "fineweb" / "fineweb-100k.parquet",
}
TOKENS = Path(os.environ.get("PYROID_DEDUP_ORACLE", "/target/oracle")) / "tokens.parquet"
BENCH = Path(os.environ.get("PYROID_BENCH_DIR", "/work/bench"))


def log(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def texts_of(corpus: str, n: int | None = None) -> list[str]:
    import pyarrow.parquet as pq

    t: list[str] = pq.read_table(CORPORA[corpus], columns=["text"]).column("text").to_pylist()
    return t[:n] if n else t


def r2_shingles(text: str, n: int = 5) -> set[str]:
    """The bench harness's shingles for rensa/datasketch (``gates.r2_shingles``)."""
    w = text.lower().split()
    return {" ".join(w[i : i + n]) for i in range(max(1, len(w) - n + 1))}


def timed(fn: Callable[[], Any], repeat: int) -> list[float]:
    fn()
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def rows(docs: list[str]) -> dict[str, Callable[[], Any]]:
    import pyroid.dedup as dedup

    cfg = dedup.MinhashConfig(hash_config=dedup.HashConfig(64, "sha1"))
    native = dedup._native_config(cfg)
    out: dict[str, Callable[[], Any]] = {
        "pyroid": lambda: dedup.minhash_signatures(docs, cfg, threads=1),
        "pyroid xxhash": lambda: dedup.minhash_signatures(docs, threads=1),
        "pyroid simplify_text": lambda: dedup.simplify_text(docs, threads=1),
    }
    if TOKENS.exists():
        import pyarrow.parquet as pq

        toks = pq.read_table(TOKENS, columns=["tokens"]).column("tokens").to_pylist()[: len(docs)]
        out["pyroid tokens"] = lambda: dedup._native._signatures_from_tokens(
            toks, native, threads=1
        )
    try:
        from rensa import RMinHash

        out["rensa"] = lambda: RMinHash.from_token_sets(
            [list(r2_shingles(t)) for t in docs], 112, 42
        )
    except ImportError:
        log("rensa not installed: row skipped")
    try:
        from datasketch import MinHash

        out["datasketch"] = lambda: MinHash.bulk(
            [[s.encode("utf-8") for s in r2_shingles(t)] for t in docs], num_perm=112, seed=1
        )
    except ImportError:
        log("datasketch not installed: row skipped")
    try:
        import tempfile

        from datatrove.pipeline.dedup import MinhashDedupSignature
        from datatrove.pipeline.dedup.minhash import MinhashConfig
        from datatrove.utils.hashing import HashConfig

        step = MinhashDedupSignature(
            output_folder=tempfile.mkdtemp(prefix="pyroid-bench-"),
            config=MinhashConfig(hash_config=HashConfig(hash_fc="sha1", precision=64)),
        )

        def dt() -> list[Any]:
            out = []
            for t in docs:
                sh = step.get_shingles(t)
                out.append(None if sh.size == 0 else step.get_signature(sh))
            return out

        out["datatrove"] = dt
    except ImportError:
        log("datatrove not installed: row skipped")
    return out


def cmd_signatures(a: argparse.Namespace) -> None:
    docs = texts_of("dev", a.docs)
    table = rows(docs)
    wanted = a.rows.split(",") if a.rows else list(table)
    res = {}
    for name in wanted:
        reps = a.repeat if name != "datatrove" else max(3, a.repeat // 2)
        ts = timed(table[name], reps)
        per = [1e3 * t / len(docs) for t in ts]
        q = statistics.quantiles(per, n=4) if len(per) >= 2 else [per[0]] * 3
        res[name] = {"ms_per_doc": statistics.median(per), "q1": q[0], "q3": q[2], "runs": reps}
        log(f"{name:24s} {res[name]['ms_per_doc']:.4f} ms/doc [{q[0]:.4f}-{q[2]:.4f}]")
    sys.stdout.write(
        json.dumps({"docs": len(docs), "python": sys.version.split()[0], "rows": res}) + "\n"
    )


def cmd_pipeline_one(a: argparse.Namespace) -> None:
    import pyroid.dedup as dedup

    docs = texts_of(a.corpus)
    rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    cfg = dedup.MinhashConfig(hash_config=dedup.HashConfig(64, "sha1"))
    c0 = time.process_time()
    t0 = time.perf_counter()
    keep = dedup.MinHashDedup(cfg, threads=a.threads).fit(docs)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out = {
        "docs": len(docs),
        "threads": a.threads,
        "wall_s": wall,
        "cpu_s": cpu,
        "removed": len(docs) - sum(keep),
        "maxrss_mib_after_load": rss_before / 1024,
        "maxrss_mib": rss / 1024,
    }
    sys.stdout.write(json.dumps(out) + "\n")


def cmd_pipeline(a: argparse.Namespace) -> None:
    res = []
    for t in [int(x) for x in a.threads.split(",")]:
        runs = []
        for _ in range(a.repeat + 1):
            p = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "pipeline-one",
                    "--corpus",
                    a.corpus,
                    "--threads",
                    str(t),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            runs.append(json.loads(p.stdout))
        runs = runs[1:]  # the first run warms the page cache
        row = {
            "threads": t,
            "docs": runs[0]["docs"],
            "removed": runs[0]["removed"],
            "wall_s": statistics.median(r["wall_s"] for r in runs),
            "cpu_s": statistics.median(r["cpu_s"] for r in runs),
            "maxrss_mib": max(r["maxrss_mib"] for r in runs),
            "maxrss_mib_after_load": max(r["maxrss_mib_after_load"] for r in runs),
            "runs": len(runs),
        }
        res.append(row)
        log(json.dumps(row))
    sys.stdout.write(
        json.dumps({"corpus": a.corpus, "python": sys.version.split()[0], "rows": res}) + "\n"
    )


def cmd_datatrove_chain(a: argparse.Namespace) -> None:
    import tempfile

    sys.path.insert(0, str(BENCH.parent / "parity"))
    from pyroid_parity.refs import datatrove_minhash_removed

    docs = texts_of("dev", a.docs)
    c0 = time.process_time()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pyroid-chain-") as w:
        removed = datatrove_minhash_removed(docs, [str(i) for i in range(len(docs))], Path(w))
    out = {
        "docs": len(docs),
        "wall_s": time.perf_counter() - t0,
        "cpu_s": time.process_time() - c0,
        "removed": sum(removed),
        "maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    sys.stdout.write(json.dumps(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("signatures")
    s.add_argument("--docs", type=int, default=1000)
    s.add_argument("--repeat", type=int, default=7)
    s.add_argument("--rows", default="")
    p = sub.add_parser("pipeline")
    p.add_argument("--corpus", default="dev", choices=list(CORPORA))
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--repeat", type=int, default=3)
    o = sub.add_parser("pipeline-one")
    o.add_argument("--corpus", default="dev", choices=list(CORPORA))
    o.add_argument("--threads", type=int, default=1)
    c = sub.add_parser("datatrove-chain")
    c.add_argument("--docs", type=int, default=10_000)
    a = ap.parse_args()
    {
        "signatures": cmd_signatures,
        "pipeline": cmd_pipeline,
        "pipeline-one": cmd_pipeline_one,
        "datatrove-chain": cmd_datatrove_chain,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
