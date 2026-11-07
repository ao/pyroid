#!/usr/bin/env python3
"""Turn a results dir into Markdown tables (paste-ready for the report).

    python summarize.py results/<machine>/<date> [--out results/<machine>/<date>/SUMMARY.md]

Reads (whatever exists): machine.json, gates.json, stages/*.json (pyperf timing), memory/*.json (pyperf
--track-memory), e2e/hyperfine-*.json (hyperfine 2.0), e2e/runs.jsonl (pipeline summaries of the timed runs),
e2e/equivalence.jsonl (pipeline summaries of the digest / check-refs runs).
Every number printed comes from those files; nothing is estimated.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

MIB = 1 << 20


def quart(v: list[float]) -> tuple[float, float, float]:
    v = sorted(v)
    if len(v) == 1:
        return v[0], v[0], v[0]
    q1, med, q3 = statistics.quantiles(v, n=4, method="inclusive")
    return q1, med, q3


def t(seconds: float) -> str:
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "\u00b5s"), (1e-9, "ns")):
        if seconds >= scale or unit == "ns":
            x = seconds / scale
            return f"{x:.3g} {unit}" if x < 1000 else f"{x:.0f} {unit}"
    return str(seconds)


def table(head: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def jsonl(p: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in p.read_text("utf-8").splitlines() if line.strip()] if p.exists() else []
    )


def header(d: Path) -> str:
    p = d / "machine.json"
    if not p.exists():
        return f"## Benchmark results: {d}\n\n(no machine.json)\n"
    m = json.loads(p.read_text("utf-8"))
    py = m["python"]
    cpu = m["cpu"]
    ls = cpu.get("lscpu", {})
    model = m.get("host_cpu") or (ls.get("Model name") if ls.get("Model name", "-") != "-" else None)
    model = model or f"{ls.get('Vendor ID', '?')} {ls.get('Architecture', '')} (model name hidden by the VM)"
    keys = m["packages"]["key"]
    vers = ", ".join(
        f"{k} {v}" for k, v in keys.items() if v and k not in ("psutil", "matplotlib", "pyroid-parity")
    )
    cloud = m.get("cloud") or {}
    if cloud.get("service") == "ec2":
        where = f"AWS EC2 {cloud.get('instance_type', '?')} (a VM on real hardware, not bare metal)"
    else:
        where = (
            "Docker/OrbStack VM, indicative (not README numbers)" if m.get("virtualized") else "bare metal"
        )
    after = m.get("after", {})
    lines = [
        f"## Benchmark results: {m['slug']}, {m['before']['timestamp_utc'][:10]}",
        "",
        (
            f"*{where}.* CPU: {model}, {cpu['cpu_count']} logical CPUs (affinity {cpu['affinity']}); "
            f"kernel {m['uname']['release']}; Python {py['version'].split()[0]}"
            f"{' free-threaded' if py['free_threaded_build'] else ''} (GIL {'on' if py['gil_enabled'] else 'off'}); "
            f"git {m['git']['sha'][:12]} ({m['git']['source']}); manifest sha256 {str(m.get('manifest_sha256'))[:12]}."
        ),
        "",
        (
            f"Load average (1/5/15 min) before: {_load(m['before'])}; after: {_load(after) if after else 'n/a'}. "
            "Other work may share this machine."
        ),
        "",
        f"Versions: {vers}.",
    ]
    return "\n".join(lines) + "\n"


def _load(snap: dict[str, Any]) -> str:
    return " / ".join(f"{x:.2f}" for x in snap.get("loadavg", []))


def gates(d: Path) -> str:
    p = d / "gates.json"
    if not p.exists():
        return ""
    g = json.loads(p.read_text("utf-8"))
    rows = []
    for r in g["gates"]:
        if "precision" in r:
            res = (
                f"P {r['precision']}, R {r['recall']}, mean abs err {r['mean_abs_err']} ({r['pairs']} pairs)"
            )
        else:
            res = f"{r['mismatches']} mismatches ({r['pinned']} pinned, {r['live']} live)"
        rows.append([r["gate"], r["impl"], r.get("docs", r.get("pairs")), res])
    status = "all passed" if g["ok"] else f"FAILED: {g['failed']}"
    return (
        f"### Equivalence gates ({g['corpus']}, {g['docs']} docs; {status})\n\n"
        + table(["gate", "implementation", "docs", "result"], rows)
        + "\n"
    )


def e2e(d: Path) -> str:
    runs = defaultdict(list)
    for r in jsonl(d / "e2e" / "runs.jsonl"):
        runs[r.get("label", "")].append(r)
    out = []
    for p in sorted((d / "e2e").glob("hyperfine-*.json")):
        doc = json.loads(p.read_text("utf-8"))
        rows = []
        for res in doc["results"]:
            ok = [m for m in res["measurements"] if m.get("exit_code", 0) == 0]
            if not ok:
                rows.append([res.get("name") or res["command"], "failed", "", "", "", "", "", "", ""])
                continue
            wall = [m["time_wall_clock"]["value"] for m in ok]
            q1, med, q3 = quart(wall)
            cpu = statistics.median(m["time_cpu"]["value"] for m in ok) if "time_cpu" in ok[0] else None
            rss = (
                statistics.median(m["memory_peak_resident"]["value"] for m in ok) / MIB
                if ("memory_peak_resident" in ok[0])
                else None
            )
            name = res.get("name") or res["command"]
            pr = runs.get(name, [])
            anon = [
                r["cgroup"].get("sampled_max_anon_mib") for r in pr if r["cgroup"].get("sampled_max_anon_mib")
            ]
            bound = [r["rss"]["upper_bound_mib"] for r in pr if r.get("rss")]
            sd = statistics.stdev(wall) if len(wall) > 1 else 0.0
            docs = pr[0]["docs"] if pr else None
            rows.append(
                [
                    name,
                    f"{med:.2f}",
                    f"{q1:.2f} to {q3:.2f}",
                    f"{statistics.mean(wall):.2f} ± {sd:.2f}",
                    f"{cpu:.1f}" if cpu is not None else "",
                    f"{cpu / med:.2f}" if cpu else "",
                    f"{rss:.0f}" if rss else "",
                    f"{statistics.median(anon):.0f}" if anon else "",
                    f"{statistics.median(bound):.0f}" if bound else "",
                    f"{med / docs * 1e3:.2f}" if docs else "",
                    len(ok),
                ]
            )
        out.append(
            f"### End to end: {p.name}\n\n"
            + table(
                [
                    "command",
                    "median wall (s)",
                    "IQR (s)",
                    "mean ± sd (s)",
                    "CPU-s (median)",
                    "CPU-s / wall",
                    "largest process RSS (MiB)",
                    "cgroup anon peak, sampled (MiB)",
                    "sum of worker maxrss (MiB, bound)",
                    "wall ms / doc",
                    "runs",
                ],
                rows,
            )
        )
    if out:
        out.append(
            "hyperfine `memory_peak_resident` is the largest single process, not the simultaneous total; "
            "the cgroup column is the container's anonymous memory sampled every 0.1 s during the run "
            "(it can miss short spikes; the median includes the warmup run), and the last memory column sums per-worker peaks (an upper bound)."
        )
    return "\n\n".join(out) + ("\n" if out else "")


def stages(d: Path) -> str:
    try:
        import pyperf
    except ImportError:
        return "(pyperf not importable: no stage tables)\n"
    mem: dict[str, float] = {}
    for p in sorted((d / "memory").glob("*.json")):
        if p.name.endswith(".gates.json"):
            continue
        for b in pyperf.BenchmarkSuite.load(str(p)).get_benchmarks():
            mem[b.get_name()] = statistics.median(b.get_values()) / MIB
    rows = []
    for p in sorted((d / "stages").glob("*.json")):
        if p.name.endswith(".gates.json"):
            continue
        for b in pyperf.BenchmarkSuite.load(str(p)).get_benchmarks():
            md = b.get_metadata()
            if md.get("unit", "second") != "second":
                continue
            v = b.get_values()
            q1, med, q3 = quart(v)
            n = int(md["docs"])
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            gate = md.get("gate", "")
            try:
                g = json.loads(gate)
                gate = (
                    f"{g.get('mismatches', 0)} diff"
                    if "mismatches" in g
                    else f"P {g['precision']} R {g['recall']}"
                )
            except (ValueError, KeyError, TypeError):
                pass
            rows.append(
                (
                    (_order(md["stage"]), n, med),
                    [
                        md["stage"],
                        md.get("impl_name", b.get_name()),
                        n,
                        t(med),
                        f"{t(q1)} to {t(q3)}",
                        t(sd),
                        t(med / n),
                        t(q1 / n) + " to " + t(q3 / n),
                        len(v),
                        f"{mem[b.get_name()]:.0f}" if b.get_name() in mem else "",
                        gate,
                    ],
                )
            )
    if not rows:
        return ""
    rows.sort(key=lambda r: r[0])
    return (
        "### Per stage (pyperf; one value = one call over the batch)\n\n"
        + table(
            [
                "stage",
                "implementation",
                "docs / call",
                "median / call",
                "IQR / call",
                "stdev",
                "per doc (median)",
                "per doc (IQR)",
                "values",
                "peak RSS MiB (track-memory)",
                "gate",
            ],
            [r[1] for r in rows],
        )
        + "\n"
    )


def _order(stage: str) -> int:
    order = [
        "fix_text",
        "lid",
        "gopher_repetition",
        "gopher_quality",
        "c4_quality",
        "c4_quality@fineweb",
        "fineweb_quality",
        "minhash",
    ]
    return order.index(stage) if stage in order else len(order)


def equivalence(d: Path) -> str:
    rows = []
    for r in jsonl(d / "e2e" / "equivalence.jsonl"):
        cr = r.get("check_refs") or {}
        cov = (
            f"{cr.get('fully_checked', 0)} full / {cr.get('partially_checked', 0)} partial / {cr.get('no_ref', 0)} no ref"
            if (cr)
            else ""
        )
        impl = ",".join(sorted({v["module"] for v in r["impl"].values()}))
        rows.append(
            [
                r.get("label", ""),
                r["corpus"],
                r["docs"],
                r["workers"],
                "+".join(r["stages"]) if len(r["stages"]) < 7 else "all",
                impl,
                (r.get("digest") or "")[:16],
                cr.get("mismatches", ""),
                cov,
            ]
        )
    if not rows:
        return ""
    verdict = d / "e2e" / "equivalence.json"
    extra = ""
    if verdict.exists():
        v = json.loads(verdict.read_text("utf-8"))
        extra = f"\n\nDigest check: {'identical' if v['ok'] else 'MISMATCH'} across {', '.join(v['labels'])}."
    return (
        "### Pipeline equivalence runs\n\n"
        + table(
            [
                "label",
                "corpus",
                "docs",
                "workers",
                "stages",
                "impl",
                "digest",
                "check-refs mismatches",
                "coverage",
            ],
            rows,
        )
        + extra
        + "\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = Path(a.results)
    parts = [header(d), gates(d), equivalence(d), e2e(d), stages(d)]
    text = "\n".join(p for p in parts if p)
    if a.out:
        Path(a.out).write_text(text, "utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
