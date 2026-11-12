#!/usr/bin/env python3
"""Markdown tables from one or more collected EC2 run folders (bench/results/<machine>/<date>/, as run.sh writes them).

    python bench/ec2/summary.py bench/results/aws-c7i.4xlarge/2026-11-06 bench/results/aws-c8g.4xlarge/2026-11-06

Stdlib only. Sections: machines (lscpu, IMDS), step durations, parity per interpreter and suite, reference
deltas (refs/refs-check.json), the end-to-end tables of the dev and headline runs (consolidated.json) and the
per-stage table of the dev run. Every number is copied from the run's own files.
"""

# ruff: noqa: PERF401  (row-building loops read better than comprehensions here)
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def kv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text("utf-8", "replace").splitlines():
        k, sep, v = line.partition(":")
        if sep and v.strip():
            out.setdefault(k.strip(), v.strip())
    return out


def load(path: Path) -> Any:
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def machines(runs: list[Path]) -> list[str]:
    keys = ["Model name", "BIOS Model name", "Vendor ID", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "Socket(s)",
            "CPU max MHz", "L2 cache", "L3 cache", "NUMA node(s)"]  # fmt: skip
    rows = []
    for r in runs:
        ls, imds, sysd = kv(r / "machine/lscpu.txt"), kv(r / "machine/imds.txt"), kv(r / "machine/system.txt")
        flags = ls.get("Flags", "").split()
        simd = [
            f
            for f in ("avx2", "fma", "avx512f", "avx512_bf16", "amx_tile", "asimd", "sve", "sve2")
            if f in flags
        ]
        osr = (
            dict(
                ln.partition("=")[::2]
                for ln in (r / "machine/os-release.txt").read_text().splitlines()
                if "=" in ln
            )
            .get("PRETTY_NAME", "")
            .strip('"')
        )
        uname = (r / "machine/uname.txt").read_text().split() if (r / "machine/uname.txt").exists() else []
        rows.append(
            [imds.get("instance-type", r.parent.name), imds.get("placement/availability-zone", "")]
            + [ls.get(k, "") for k in keys]
            + [" ".join(simd), osr, uname[2] if len(uname) > 2 else "", sysd.get("bench_cpus", "")]
        )
    head = ["instance", "AZ", *keys, "SIMD flags", "OS", "kernel", "BENCH_CPUS"]
    return table(head, rows)


def durations(runs: list[Path]) -> list[str]:
    steps: dict[str, dict[str, str]] = {}
    for r in runs:
        p = r / "durations.tsv"
        for line in p.read_text().splitlines() if p.exists() else []:
            name, secs, rc = [*line.split("\t"), "", ""][:3]
            steps.setdefault(name, {})[r.parent.name] = f"{int(secs) / 60:.1f}" + (
                "" if rc == "0" else f" (rc {rc})"
            )
    return table(["step (minutes)", *(r.parent.name for r in runs)],
                 [[s, *(v.get(r.parent.name, "") for r in runs)] for s, v in steps.items()])  # fmt: skip


def pytest_rc(r: Path, suite: str, py: str) -> str:
    """The pytest exit code: status.tsv for run.sh's suites, else the log's trailing ``exit N`` line."""
    st = r / "parity" / "status.tsv"
    for line in st.read_text().splitlines() if st.exists() else []:
        f = line.split("\t")
        if f[:2] == [py, suite]:
            return f[2]
    log = r / "parity" / f"{suite}-{py}.log"
    tail = log.read_text("utf-8", "replace").strip().splitlines()[-1:] if log.exists() else []
    return tail[0].removeprefix("exit ") if tail and tail[0].startswith("exit ") else "?"


def parity(runs: list[Path]) -> list[str]:
    """One row per report file; ``suite`` is the file's (``filters+aiohttp`` marks a supplementary rerun). Errors
    before a check is recorded (e.g. a missing oracle dependency) only show in the pytest exit code."""
    rows = []
    for r in runs:
        for p in sorted((r / "parity").glob("*-3.*.json")):
            if p.name.endswith(".merged.json"):
                continue
            rep = load(p)
            suite, py = p.stem.rsplit("-", 1)
            rc = pytest_rc(r, suite, py)
            checks = [c for s in rep.get("suites", {}).values() for c in s.get("checks", {}).values()]
            n = {k: sum(c["status"] == k for c in checks) for k in ("pass", "fail", "skip")}
            rows.append([r.parent.name, py, suite, rc, rep.get("status", "no report"), n["pass"], n["fail"],
                         n["skip"], f"{sum(c.get('same', 0) for c in checks):,}", sum(c.get("diff", 0) for c in checks),
                         "off" if rep.get("environment", {}).get("gil_enabled") is False else "on"])  # fmt: skip
    head = [
        "machine",
        "python",
        "suite",
        "pytest rc",
        "report",
        "checks pass",
        "fail",
        "skip",
        "same",
        "diff",
        "GIL",
    ]
    return table(head, rows)


def deltas(runs: list[Path]) -> list[str]:
    rows = []
    for r in runs:
        c = load(r / "refs/refs-check.json")
        if not c:
            rows.append([r.parent.name, "(no refs-check.json)", "", "", ""])
            continue
        for m in c.get("mismatched", []):
            d = m.get("delta") or {}
            rows.append([r.parent.name, m["rel"], "per-platform" if m.get("platform_specific") else "UNEXPECTED",
                         f"{m.get('old', {}).get('sha256', '?')[:12]} -> {m.get('new', {}).get('sha256', '?')[:12]}",
                         d.get("summary") or m.get("delta_error", "")])  # fmt: skip
        for m in c.get("missing", []):
            rows.append([r.parent.name, m["rel"], "MISSING", "", m.get("reason", "")])
        ok = [x["rel"] for x in c.get("checked", []) if x.get("reproduced")]
        rows.append(
            [
                r.parent.name,
                f"{len(ok)} reproduced byte-identically",
                "",
                "",
                ", ".join(Path(x).name for x in ok),
            ]
        )
    return table(["machine", "ref", "kind", "sha256 old -> new", "delta"], rows)


def e2e(runs: list[Path]) -> list[str]:
    rows = []
    for r in runs:
        for part in ("dev", "headline"):
            c = load(r / part / "consolidated.json")
            if not c:
                continue
            res = c.get("e2e", [])
            b1 = next(
                (x for x in res if x["name"].startswith("datatrove") and "1 process " in x["name"]), None
            )
            b8 = next(
                (x for x in res if x["name"].startswith("datatrove") and "8 processes" in x["name"]), None
            )
            for x in sorted(res, key=lambda x: x["med"]):
                rows.append([r.parent.name, part, x["name"], f"{x['med']:.2f}", f"{x['cpu']:.1f}", f"{x['rss_mib']:.0f}",
                             f"{b1['med'] / x['med']:.1f}x" if b1 else "", f"{b8['med'] / x['med']:.1f}x" if b8 else "",
                             x.get("runs", "")])  # fmt: skip
    return table(["machine", "corpus", "run", "wall s", "CPU s", "peak RSS MiB", "vs dt 1 proc", "vs dt 8 procs", "runs"],
                 rows)  # fmt: skip


def stages(runs: list[Path]) -> list[str]:
    rows = []
    for r in runs:
        c = load(r / "dev" / "consolidated.json")
        for x in (c or {}).get("rows", []):
            rows.append([r.parent.name, f"{x['module']}: {x['call']}", x["incumbent_us"], x["pyroid_us"],
                         x.get("pyroid_us_8_threads", ""), x["speedup"]])  # fmt: skip
    return table(
        ["machine", "stage", "incumbent us/doc", "pyroid 3.15t 1 thr", "8 thr", "speedup 1 thr"], rows
    )


def table(head: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return out + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]


def main() -> int:
    runs = [Path(a).resolve() for a in sys.argv[1:]]
    if not runs:
        print(__doc__)
        return 2
    for title, fn in [("Machines", machines), ("Step durations", durations), ("Parity", parity),
                      ("Reference deltas", deltas), ("End to end", e2e), ("Per stage (dev slice)", stages)]:  # fmt: skip
        print(f"## {title}\n")
        print("\n".join(fn(runs)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
