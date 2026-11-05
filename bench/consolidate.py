#!/usr/bin/env python3
"""Consolidated tables and charts for one run_consolidated.sh results folder.

    python consolidate.py results/<machine>/<date>-consolidated

Reads (all written by run_consolidated.sh, every row gated before it was timed):
  stages/*.json, e2e/hyperfine-*.json, e2e/equivalence.json   run_baselines.sh on CPython 3.13 (incumbents,
                                                               pyroid at RAYON_NUM_THREADS=1)
  ft315t/stages/*-p{1,8}.json                                  pyroid on CPython 3.15t at pools of 1 and 8
  ft315t/hyperfine-chain.json, ft315t/chain-digest.json        pyroid's native chain (chain.py)
Writes CONSOLIDATED.md, consolidated.json (rows in docs/data/benchmarks.json's schema) and
charts/consolidated-{e2e,stages}-{light,dark}.svg. Nothing is estimated: every number is a median of
measured values in those files.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
LABEL = "Docker/OrbStack VM on Apple Silicon, indicative"
# stage -> (module, call shown in tables, incumbent row substring, pyroid row substring)
HEADLINE = [
    ("fix_text", "ftfy", "fix_text_batch", "ftfy ", "fix_text_batch"),
    ("lid", "fasttext", "predict, lid.176.bin, k=1", "fasttext-predict", "lid.176.bin k=1 batch"),
    ("lid", "fasttext", "predict, lid.176.ftz, k=1", "fasttext-predict", "lid.176.ftz k=1 batch"),
    ("gopher_repetition", "filters", "GopherRepetition", "datatrove", "direct"),
    ("gopher_quality", "filters", "GopherQuality", "datatrove", "direct"),
    ("c4_quality", "filters", "C4Quality (default)", "datatrove", "direct"),
    ("c4_quality@fineweb", "filters", "C4Quality (FineWeb config)", "datatrove", "direct"),
    ("fineweb_quality", "filters", "FineWebQuality", "datatrove", "direct"),
    ("minhash", "dedup", "MinHash signatures, sha1/64", "MinhashDedupSignature", "minhash_signatures direct"),
    ("translit", "translit", "anyascii", "anyascii ", "anyascii"),
]


def quartiles(v: list[float]) -> tuple[float, float, float]:
    if len(v) == 1:
        return v[0], v[0], v[0]
    q = statistics.quantiles(v, n=4)
    return q[0], statistics.median(v), q[2]


def pyperf_rows(paths: list[Path], interp: str) -> list[dict[str, Any]]:
    import pyperf

    rows = []
    for p in paths:
        if p.name.endswith(".gates.json"):
            continue
        for b in pyperf.BenchmarkSuite.load(str(p)).get_benchmarks():
            md = b.get_metadata()
            if "stage" not in md or md.get("unit", "second") != "second":
                continue
            docs = int(md["docs"])
            q1, med, q3 = quartiles([v / docs * 1e6 for v in b.get_values()])
            pool = md.get("rayon_threads", "unset")
            rows.append({
                "stage": md["stage"], "name": md["impl_name"], "docs": docs, "med": med, "q1": q1, "q3": q3,
                "reference": bool(md.get("is_reference")), "interp": interp, "pool": pool,
                "input": md.get("input_kind", "list[str]"), "gate": md.get("gate", "missing"),
                "python": md.get("python_version", ""),
            })  # fmt: skip
    return rows


def hyperfine(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for r in json.loads(path.read_text("utf-8"))["results"]:
        ok = [m for m in r["measurements"] if m.get("exit_code", 0) == 0]
        if not ok:
            continue
        q1, med, q3 = quartiles([m["time_wall_clock"]["value"] for m in ok])
        cpu = statistics.median([m["time_cpu"]["value"] for m in ok if "time_cpu" in m] or [float("nan")])
        mem = max((m.get("memory_peak_resident", {}).get("value", 0) for m in ok), default=0)
        out.append({"name": r["name"], "med": med, "q1": q1, "q3": q3, "cpu": cpu, "rss_mib": mem / 2**20,
                    "runs": len(ok)})  # fmt: skip
    return out


def us(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def pick(rows: list[dict[str, Any]], stage: str, sub: str, ref: bool, **kw: Any) -> dict[str, Any] | None:
    c = [r for r in rows if r["stage"] == stage and r["reference"] == ref and sub in r["name"]]
    c = [r for r in c if all(r.get(k) == v for k, v in kw.items())]
    return max(c, key=lambda r: r["docs"]) if c else None


def e2e_label(name: str) -> str:
    """Readable e2e names: hyperfine's command names say what ran, not where."""
    n = name.rsplit("-w", 1)
    if name.startswith("datatrove") and len(n) == 2:
        procs = f"{n[1]} process{'es' if n[1] != '1' else ''}"
        return f"datatrove {name.split('-')[1]}, {procs} (CPython 3.13)"
    if name.startswith("pyroid-w"):
        procs = f"{n[1]} process{'es' if n[1] != '1' else ''}"
        return f"pyroid stages in datatrove's executor, {procs} (3.13)"
    if name.startswith("pyroid chain"):
        t = name.rsplit("threads=", 1)[1]
        return f"pyroid native chain, {t} thread{'s' if t != '1' else ''} (CPython 3.15t)"
    return name


def merged_hyperfine(paths: list[Path], dest: Path) -> Path:
    """One hyperfine JSON with every e2e result, renamed with e2e_label (for chart.py)."""
    results = []
    for p in paths:
        if p.exists():
            doc = json.loads(p.read_text("utf-8"))
            results += [{**r, "name": e2e_label(r.get("name") or r["command"])} for r in doc["results"]]
    dest.write_text(json.dumps({"schema_version": 2, "results": results}, indent=1) + "\n", "utf-8")
    return dest


def chart_row(md: dict[str, Any]) -> bool:
    """Rows for the consolidated stage chart: every incumbent, and each pyroid API once (no per-doc loops and
    no parity-adapter duplicates of the same batch call)."""
    if md.get("is_reference"):
        return True
    name, stage = md["impl_name"], md["stage"]
    if "per doc" in name:
        return False
    if stage == "lid":
        return "batch" in name
    if stage == "fix_text" or stage == "translit":
        return True
    return "direct" in name


def trimmed(paths: list[Path], tmp: Path) -> list[str]:
    """Copies of pyperf files holding only chart_row benchmarks (keeps the SVG small)."""
    import pyperf

    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for i, p in enumerate(paths):
        keep = [b for b in pyperf.BenchmarkSuite.load(str(p)).get_benchmarks() if chart_row(b.get_metadata())]
        if keep:
            dest = tmp / f"{i:03d}-{p.name}"
            dest.unlink(missing_ok=True)
            pyperf.BenchmarkSuite(keep).dump(str(dest))
            out.append(str(dest))
    return out


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    dt = pyperf_rows(sorted((out / "stages").glob("*-d*.json")), "3.13")
    ft_files = sorted((out / "ft315t" / "stages").glob("*.json"))
    ft = pyperf_rows(ft_files, "3.15t")
    e2e_json = merged_hyperfine(
        [out / "e2e" / "hyperfine-e2e.json", out / "ft315t" / "hyperfine-chain.json"],
        out / "e2e" / "hyperfine-consolidated.json",
    )
    e2e = hyperfine(e2e_json)
    stage_e2e = {r["name"]: r for r in hyperfine(out / "e2e" / "hyperfine-stages-w8.json")}
    eq = json.loads((out / "e2e" / "equivalence.json").read_text("utf-8"))
    chain = json.loads((out / "ft315t" / "chain-digest.json").read_text("utf-8"))
    machine = json.loads((out / "machine.json").read_text("utf-8")) if (out / "machine.json").exists() else {}

    md: list[str] = [f"# Consolidated benchmark ({LABEL})", ""]
    md += [(f"Machine: `{out.parent.name}`; run folder `{out.name}`. Corpus: FineWeb dev slice (first 10,000 docs "
            "of the pinned 100k headline slice). Every row passed its equivalence gate before it was timed "
            "(byte-identical ftfy/anyascii output, fastText labels with |dp| <= 1e-6, datatrove filter decisions "
            "and text, datatrove MinHash signatures)."), ""]  # fmt: skip
    if machine.get("git_sha"):
        md += [f"pyroid git sha: `{machine['git_sha']}`.", ""]

    base8 = next((r for r in e2e if r["name"].startswith("datatrove") and "8 processes" in r["name"]), None)
    base1 = next((r for r in e2e if r["name"].startswith("datatrove") and "1 process " in r["name"]), None)
    md += ["## End to end: the FineWeb chain (ftfy -> LID -> 4 filters -> MinHash dedup), 10k docs", ""]
    md += ["| run | wall s, median [IQR] | CPU s | peak RSS of one process, MiB | vs datatrove, 1 proc | vs datatrove, 8 procs |",
           "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in sorted(e2e, key=lambda r: r["med"]):
        speed = f"{base8['med'] / r['med']:.1f}x" if base8 else "-"
        speed1 = f"{base1['med'] / r['med']:.1f}x" if base1 else "-"
        md.append(f"| {r['name']} | {r['med']:.2f} [{r['q1']:.2f}-{r['q3']:.2f}] | {r['cpu']:.1f} | "
                  f"{r['rss_mib']:.0f} | {speed1} | {speed} |")  # fmt: skip
    md += ["", (f"Digests: datatrove at 1 and 8 workers and datatrove's executor with pyroid's stages: "
            f"{'identical' if eq.get('ok') else 'MISMATCH'} ({', '.join(eq.get('labels', []))}). pyroid's native "
            f"chain (chain.py, 3.15t): digest `{chain.get('digest', '?')[:12]}...`, "
            f"{'identical to datatrove' if chain.get('digest_matches') else 'NOT CHECKED or MISMATCH'}."), ""]  # fmt: skip

    md += ["## Per stage: microseconds per document, median [IQR] of pyperf values", "",
           ("Incumbents: one process, CPython 3.13, per-doc loop over their largest measured batch (10k docs for "
            "fix_text, LID and translit; 1k for the filters and MinHash). pyroid: one call over the same docs "
            "(list[str]); `pool` = RAYON_NUM_THREADS. One-stage e2e: "
            "datatrove's executor running only that stage at 8 processes (wall time per doc, start-up and parquet IO "
            "included)."), "",
           ("| stage | incumbent, 1 proc | incumbent, 8 procs (one-stage e2e) | pyroid 3.13, 1 thread | "
            "pyroid 3.15t, pool 1 | pyroid 3.15t, pool 8 | speedup, 1 thread vs 1 proc |"),
           "|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    docs_rows = []
    e2e_stage_of = {"fix_text": "fix_text", "lid": "lid", "gopher_repetition": "gopher_repetition",
                    "gopher_quality": "gopher_quality", "c4_quality@fineweb": "c4_quality",
                    "fineweb_quality": "fineweb_quality", "minhash": "minhash"}  # fmt: skip
    for stage, module, call, inc_sub, py_sub in HEADLINE:
        inc = pick(dt, stage, inc_sub, True)
        if stage == "lid" and inc:
            model = "lid.176.bin" if "bin" in py_sub else "lid.176.ftz"
            inc = pick(dt, stage, f"{model} k=1", True)
        # Same docs as the incumbent: its largest batch (first N docs of the dev slice).
        n = inc["docs"] if inc else 10000
        p313 = pick(dt, stage, py_sub, False, input="list[str]", docs=n)
        p1 = pick(ft, stage, py_sub, False, pool="1", input="list[str]", docs=n)
        p8 = pick(ft, stage, py_sub, False, pool="8", input="list[str]", docs=n)
        s8 = stage_e2e.get(f"{e2e_stage_of.get(stage, '')}-w8")
        s8_us = s8["med"] / 10000 * 1e6 if s8 else None
        cell = lambda r: f"{us(r['med'])} [{us(r['q1'])}-{us(r['q3'])}]" if r else "-"
        ratio = inc["med"] / p1["med"] if inc and p1 else None
        sp1 = "-" if ratio is None else (f"{ratio:.1f}\u00d7" if ratio < 10 else f"{ratio:.0f}\u00d7")
        label = f"{stage} ({call.split(', ')[1]})" if stage == "lid" else stage
        md.append(f"| {label}, {n:,} docs | {cell(inc)} | {us(s8_us) if s8_us else '-'} | {cell(p313)} | {cell(p1)} | "
                  f"{cell(p8)} | {sp1} |")  # fmt: skip
        if inc and p1:
            docs_rows.append({
                "module": module, "call": call, "input": f"FineWeb dev, first {n:,} docs; pyroid on CPython 3.15t, incumbent on 3.13", "incumbent": inc["name"].split(" ")[0] + " " + inc["name"].split(" ")[1],
                "incumbent_us": us(inc["med"]), "pyroid_us": us(p1["med"]),
                "pyroid_us_8_threads": us(p8["med"]) if p8 else "", "speedup": sp1, "check": "harness gate",
            })  # fmt: skip
    extra = [
        r
        for r in dt
        if r["stage"] == "minhash"
        and r["reference"]
        and ("rensa" in r["name"] or "datasketch" in r["name"])
        and r["docs"] == max(x["docs"] for x in dt if x["stage"] == "minhash")
    ]
    if extra:
        md += ["", ("MinHash estimators with other hash families (gated on near-duplicate precision/recall >= 0.90, "
               "not on identical signatures): ") + "; ".join(f"{r['name']}: {us(r['med'])} us/doc"
               for r in sorted(extra, key=lambda r: r["med"])) + "."]  # fmt: skip
    big = [
        r
        for r in ft
        if r["docs"] == 10000
        and r["input"] == "list[str]"
        and r["stage"] not in ("fix_text", "lid", "translit")
        and ("apply(" in r["name"] or "minhash_signatures direct" in r["name"])
    ]
    if big:
        md += [
            "",
            "pyroid on the filters and MinHash with 10k docs per call (3.15t, list[str]; larger batches use the pool "
            "better): "
            + "; ".join(
                f"{r['stage']} pool {r['pool']}: {us(r['med'])}"
                for r in sorted(big, key=lambda r: (r["stage"], int(r["pool"])))
            )
            + " us/doc.",
        ]
    arrow = [r for r in ft if r["input"] != "list[str]" and r["docs"] == 10000]
    if arrow:
        md += [
            "",
            "pyroid with Arrow input (3.15t, 10k docs): "
            + "; ".join(
                f"{r['name']}: {us(r['med'])} us/doc"
                for r in sorted(arrow, key=lambda r: (r["stage"], r["pool"]))
            )
            + ".",
        ]
    md += ["", "## Charts", "", "- `charts/consolidated-e2e-{light,dark}.svg`: the end-to-end table.",
           ("- `charts/consolidated-stages-{light,dark}.svg`: per stage at the incumbents' largest batch "
            "(10k fast stages, 1k filters/MinHash), with pyroid's 3.13 and 3.15t rows."), ""]  # fmt: skip
    (out / "CONSOLIDATED.md").write_text("\n".join(md) + "\n", "utf-8")
    (out / "consolidated.json").write_text(json.dumps({
        "about": "pyroid consolidated gated benchmark (bench/run_consolidated.sh, bench/consolidate.py). Units: "
                 "microseconds per document; pyroid_us is one thread (pool of 1) on CPython 3.15t, "
                 "pyroid_us_8_threads a pool of 8; incumbents run one process on CPython 3.13.",
        "label": LABEL, "source": str(out.relative_to(HERE.parent)) if out.is_relative_to(HERE.parent) else str(out),
        "rows": docs_rows, "e2e": e2e,
    }, indent=1, ensure_ascii=False) + "\n", "utf-8")  # fmt: skip

    charts = out / "charts"
    charts.mkdir(exist_ok=True)
    chart = [sys.executable, str(HERE / "chart.py")]
    e2e_files = [str(e2e_json)]
    subprocess.run([*chart, "e2e", *e2e_files, "--out", str(charts / "consolidated-e2e"), "--title",
                    (f"FineWeb chain, 10k docs: datatrove (3.13) vs pyroid (3.13 in datatrove's executor; 3.15t "
                     f"native chain). {LABEL}")], check=True)  # fmt: skip
    picked = [p for p in sorted((out / "stages").glob("*-d*[0-9].json"))]
    picked += [p for p in ft_files if not p.name.endswith(".gates.json")
               and ("-d1000-" in p.name or p.name.startswith(("fix_text", "lid", "translit")))]  # fmt: skip
    tmp = Path(tempfile.mkdtemp(prefix="pyroid-consolidate-"))
    stage_files = trimmed(picked, tmp)
    subprocess.run([*chart, "stages", *stage_files, "--out", str(charts / "consolidated-stages"), "--title",
                    (f"per-doc time at the incumbents' largest batch; pyroid rows without a suffix: CPython 3.13, "
                     f"1 thread; [3.15t, pool N]: free-threaded 3.15t. {LABEL}")], check=True)  # fmt: skip
    shutil.rmtree(tmp)
    print((out / "CONSOLIDATED.md").read_text("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
