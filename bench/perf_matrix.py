#!/usr/bin/env python3
"""Candidate-only A/B matrix on the gated harness: every pyroid module x pool sizes, via ``stages.py``.

    python perf_matrix.py run --python /target/venv-b315t/bin/python --out /target/ab/base-315t
    python perf_matrix.py run ... --fresh             # new str objects per timed call (cold UTF-8 caches)
    python perf_matrix.py table /target/ab/base-315t  # median us/doc [IQR] per row and pool size
    python perf_matrix.py compare /target/ab/base-315t /target/ab/mimalloc-315t   # B vs A, >= 5% flagged

Each case is one ``stages.py`` call (``--rows`` keeps only the candidate rows, so the incumbents are not
re-timed): the manager process gates every row on the timed docs first, then pyperf times it. The pool size
is ``RAYON_NUM_THREADS`` (pyroid's pool reads it when ``PYROID_NUM_THREADS`` is unset); gates from the first
pool size are reused for the others (same docs, same rows). Run inside Docker only (see README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# case -> (stage, --impl, --rows)
CASES = {
    "ftfy": ("fix_text", "pyroid.ftfy", "fix_text_batch"),  # list[str] and Arrow rows
    "fasttext": ("lid", "pyroid.fasttext", "k=1 batch"),  # .bin and .ftz
    "filters": ("gopher_repetition", "pyroid.filters", "direct"),
    "dedup": ("minhash", "pyroid.dedup", "direct"),
    "translit": ("translit", "pyroid.translit", "pyroid.translit,!per doc"),  # list[str] and Arrow rows
}
# --per-doc: the single-str call in a Python loop; the pool size becomes BENCH_THREADS Python threads.
PER_DOC_ROWS = "per doc"


def run(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = a.cases.split(",") if a.cases else list(CASES)
    env = dict(os.environ)
    if a.fresh:
        env["BENCH_FRESH_STRS"] = "1"
    for case in cases:
        stage, impl, rows = CASES[case]
        if a.per_doc:
            rows = PER_DOC_ROWS
        gates: Path | None = None
        for t in a.threads:
            dest = out / f"{case}-t{t}.json"
            if dest.exists():
                dest.unlink()
            cmd = [
                a.python, str(HERE / "stages.py"), "--stage", stage, "--impl", impl, "--rows", rows,
                "--docs", str(a.docs), "--corpus", a.corpus, "--offset", str(a.offset), "-p", str(a.processes), "-n", str(a.values),
                "-w", "1", "--quiet", "-o", str(dest),
            ]  # fmt: skip
            if gates is not None:
                cmd += ["--reuse-gates", str(gates)]
            print(f"[perf_matrix] {case} t={t}", file=sys.stderr, flush=True)
            knob = {"BENCH_THREADS": str(t)} if a.per_doc else {"RAYON_NUM_THREADS": str(t)}
            subprocess.run(cmd, check=True, env={**env, **knob}, cwd=HERE)
            if gates is None:
                gates = dest.with_suffix(".gates.json")


def load(d: Path) -> dict[tuple[str, int], dict[str, float]]:
    """``{(row, pool size): {median, q1, q3, n}}`` in us/doc."""
    import pyperf

    res: dict[tuple[str, int], dict[str, float]] = {}
    for f in sorted(d.glob("*-t*.json")):
        if f.name.endswith(".gates.json"):
            continue
        t = int(f.stem.rsplit("-t", 1)[1])
        for b in pyperf.BenchmarkSuite.load(str(f)).get_benchmarks():
            md = b.get_metadata()
            per = 1e6 / int(md["docs"])
            vals = sorted(v * per for v in b.get_values())
            q = statistics.quantiles(vals, n=4) if len(vals) > 1 else [vals[0]] * 3
            row = md["impl_name"].replace(" 1.0.0.dev0", "")
            res[(row, t)] = {"median": statistics.median(vals), "q1": q[0], "q3": q[2], "n": len(vals)}
    return res


def table(a: argparse.Namespace) -> None:
    r = load(Path(a.dir))
    ts = sorted({t for _, t in r})
    rows = sorted({row for row, _ in r})
    print("| row | " + " | ".join(f"pool {t}: us/doc [IQR]" for t in ts) + " |")
    print("|---|" + "---:|" * len(ts))
    for row in rows:
        cells = []
        for t in ts:
            v = r.get((row, t))
            cells.append(f"{v['median']:.2f} [{v['q1']:.2f}-{v['q3']:.2f}]" if v else "-")
        print(f"| {row} | " + " | ".join(cells) + " |")


def compare(a: argparse.Namespace) -> None:
    ra, rb = load(Path(a.a)), load(Path(a.b))
    print(f"| row | pool | A us/doc | B us/doc | B vs A | (A={Path(a.a).name}, B={Path(a.b).name}) |")
    print("|---|---:|---:|---:|---:|---|")
    for key in sorted(ra.keys() & rb.keys()):
        x, y = ra[key]["median"], rb[key]["median"]
        gain = x / y - 1
        flag = "faster >=5%" if gain >= 0.05 else ("slower >=5%" if gain <= -0.05 else "")
        print(f"| {key[0]} | {key[1]} | {x:.2f} | {y:.2f} | {gain:+.1%} | {flag} |")
    if a.json:
        out = {f"{k[0]}|{k[1]}": {"a": ra[k]["median"], "b": rb[k]["median"]} for k in ra.keys() & rb.keys()}
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", "utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--out", required=True)
    r.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    r.add_argument("--cases", help=f"comma subset of {','.join(CASES)}")
    r.add_argument("--docs", type=int, default=10000)
    r.add_argument("--corpus", default="dev")
    r.add_argument("--offset", type=int, default=0, help="skip the first N docs (e.g. PGO training on other docs)")
    r.add_argument("--processes", type=int, default=5)
    r.add_argument("--values", type=int, default=3)
    r.add_argument("--fresh", action="store_true", help="BENCH_FRESH_STRS=1")
    r.add_argument("--per-doc", action="store_true", help="single-str rows; --threads = Python threads")
    t = sub.add_parser("table")
    t.add_argument("dir")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--json")
    a = p.parse_args()
    {"run": run, "table": table, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
