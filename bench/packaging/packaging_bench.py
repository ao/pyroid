#!/usr/bin/env python3
"""packaging 26.3 vs pyroid.packaging: per-call costs, sorts, filters, markers, requirements.

    python bench/packaging/packaging_bench.py --out results.json      # the full run
    python bench/packaging/packaging_bench.py --quick                  # smoke test

Both libraries run in the same process, interleaved, and each timing is the best of ``--repeat``
rounds (the median is recorded too). Before a case is timed, both results must be equal; a
mismatch aborts the run. Prints Markdown tables; ``--out`` writes the timings and the machine
(CPU, Python, versions and the load average; no hostname or instance id).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import packaging
import packaging.markers as pm
import packaging.requirements as pr
import packaging.specifiers as ps
import packaging.utils as pu
import packaging.version as pv

import pyroid
import pyroid.packaging as rp
import pyroid.packaging.markers as rm
import pyroid.packaging.requirements as rr
import pyroid.packaging.specifiers as rs
import pyroid.packaging.utils as ru
import pyroid.packaging.version as rv


def machine() -> dict[str, Any]:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "cpu part")):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "cpus": os.cpu_count(),
        "loadavg": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "packaging": packaging.__version__,
        "pyroid": pyroid.__version__,
    }


def best(fn: Callable[[], Any], number: int) -> float:
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(number):
        fn()
    return (time.perf_counter() - t0) / number


def calls_for(fn: Callable[[], Any], target: float = 0.05) -> int:
    t0 = time.perf_counter()
    fn()
    return max(1, int(target / max(time.perf_counter() - t0, 1e-7)))


def ab(label: str, fp: Callable[[], Any], fr: Callable[[], Any], repeat: int, scale: float, unit: str, same: Callable[[Any, Any], bool] | None = None) -> dict[str, Any]:
    a, b = fp(), fr()
    ok = same(a, b) if same else a == b
    if not ok:
        sys.exit(f"results differ for {label!r}:\n  packaging {a!r:.300}\n  pyroid    {b!r:.300}")
    na, nb = calls_for(fp), calls_for(fr)
    xs, ys = [], []
    for _ in range(repeat):
        xs.append(best(fp, na))
        ys.append(best(fr, nb))
    x, y = min(xs), min(ys)
    print(f"| {label} | {x * scale:,.3f} | {y * scale:,.3f} | {x / y:.1f}x |", flush=True)
    return {"case": label, "unit": unit, "packaging": x * scale, "pyroid": y * scale, "packaging_median": statistics.median(xs) * scale, "pyroid_median": statistics.median(ys) * scale, "speedup": x / y}


def versions(n: int, rng: random.Random) -> list[str]:
    tails = ["", "", "", "", "rc1", "a2", "b1", ".post1", ".dev3", "+local.7", ".post2.dev1"]
    return [f"{rng.randint(0, 30)}.{rng.randint(0, 40)}.{rng.randint(0, 99)}{rng.choice(tails)}" for _ in range(n)]


def strs(xs: Any) -> list[str]:
    return [str(x) for x in xs]


REQS = [
    "requests>=2.8.1",
    "numpy>=1.21; python_version >= '3.8'",
    "requests[security,socks]>=2.8.1,==2.8.*; python_version < '2.7'",
    "typing-extensions>=4.0; python_version < '3.11'",
    "urllib3<3,>=1.21.1",
    "pywin32>=227; sys_platform == 'win32' and platform_python_implementation != 'PyPy'",
    "pytest>=7; extra == 'test'",
    "name @ https://example.com/name-1.0.tar.gz ; os_name == 'posix'",
]
MARKERS = [
    "python_version >= '3.8'",
    "sys_platform == 'win32' and platform_python_implementation != 'PyPy'",
    "python_version < '3.11' or (os_name == 'nt' and extra == 'test')",
    "platform_machine in 'x86_64 aarch64' and python_full_version >= '3.9.1'",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--repeat", type=int, default=15)
    ap.add_argument("--sizes", default="100000,1000000")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    repeat = 3 if args.quick else args.repeat
    sizes = [10000] if args.quick else [int(s) for s in args.sizes.split(",")]
    rng = random.Random(20261116)
    out: dict[str, Any] = {"machine": machine(), "sections": {}}
    head = "| Case | packaging 26.3 | pyroid | Speedup |\n|---|---:|---:|---:|"

    print(f"\n## One call (µs)\n\n{head}")
    rows = []
    for s in ["1.2.3", "1.0.0rc1", "2!1.0.post1.dev3+abc.5"]:
        rows.append(ab(f"Version({s!r})", lambda s=s: str(pv.Version(s)), lambda s=s: str(rv.Version(s)), repeat, 1e6, "us"))
    a, b = pv.Version("1.2.3"), pv.Version("1.2.4")
    c, d = rv.Version("1.2.3"), rv.Version("1.2.4")
    rows.append(ab("a < b", lambda: a < b, lambda: c < d, repeat, 1e6, "us"))
    rows.append(ab("str(v)", lambda: str(a), lambda: str(c), repeat, 1e6, "us"))
    rows.append(ab("hash(Version('1.2.3'))", lambda: hash(pv.Version("1.2.3")), lambda: hash(rv.Version("1.2.3")), repeat, 1e6, "us"))
    rows.append(ab("SpecifierSet('>=1.0,<2,!=1.5.*')", lambda: str(ps.SpecifierSet(">=1.0,<2,!=1.5.*")), lambda: str(rs.SpecifierSet(">=1.0,<2,!=1.5.*")), repeat, 1e6, "us"))
    pss, rss = ps.SpecifierSet(">=1.0,<2,!=1.5.*"), rs.SpecifierSet(">=1.0,<2,!=1.5.*")
    rows.append(ab("ss.contains('1.6.2')", lambda: pss.contains("1.6.2"), lambda: rss.contains("1.6.2"), repeat, 1e6, "us"))
    rows.append(ab("ss.contains(Version)", lambda: pss.contains(a), lambda: rss.contains(c), repeat, 1e6, "us"))
    rows.append(ab("Specifier('~=1.4.2').contains('1.4.5')", lambda: ps.Specifier("~=1.4.2").contains("1.4.5"), lambda: rs.Specifier("~=1.4.2").contains("1.4.5"), repeat, 1e6, "us"))
    m = MARKERS[2]
    rows.append(ab("Marker(...) parse", lambda: str(pm.Marker(m)), lambda: str(rm.Marker(m)), repeat, 1e6, "us"))
    pmk, rmk = pm.Marker(m), rm.Marker(m)
    env = {"python_version": "3.12", "os_name": "posix", "extra": "test"}
    rows.append(ab("marker.evaluate()", lambda: pmk.evaluate(), lambda: rmk.evaluate(), repeat, 1e6, "us"))
    rows.append(ab("marker.evaluate(environment)", lambda: pmk.evaluate(env), lambda: rmk.evaluate(env), repeat, 1e6, "us"))
    for r in (REQS[0], REQS[2]):
        rows.append(ab(f"Requirement({r!r})", lambda r=r: str(pr.Requirement(r)), lambda r=r: str(rr.Requirement(r)), repeat, 1e6, "us"))
    rows.append(ab("canonicalize_name('Foo.Bar_baz')", lambda: pu.canonicalize_name("Foo.Bar_baz"), lambda: ru.canonicalize_name("Foo.Bar_baz"), repeat, 1e6, "us"))
    out["sections"]["single"] = rows

    for n in sizes:
        vs = versions(n, rng)
        r = max(3, repeat // (5 if n >= 1_000_000 else 1))
        print(f"\n## {n:,} versions (ms)\n\n{head}")
        rows = []
        rows.append(ab("parse every version", lambda vs=vs: strs(pv.Version(s) for s in vs), lambda vs=vs: strs(rv.Version(s) for s in vs), r, 1e3, "ms"))
        rows.append(ab("parse_batch", lambda vs=vs: strs(pv.Version(s) for s in vs), lambda vs=vs: strs(rp.parse_batch(vs)), r, 1e3, "ms"))
        rows.append(ab("sorted(strs, key=Version)", lambda vs=vs: sorted(vs, key=pv.Version), lambda vs=vs: sorted(vs, key=rv.Version), r, 1e3, "ms"))
        rows.append(ab("sort_versions(strs)", lambda vs=vs: sorted(vs, key=pv.Version), lambda vs=vs: rp.sort_versions(vs), r, 1e3, "ms"))
        objs_p, objs_r = [pv.Version(s) for s in vs], [rv.Version(s) for s in vs]
        rows.append(ab("sorted(Version objects)", lambda: strs(sorted(objs_p)), lambda: strs(sorted(objs_r)), r, 1e3, "ms"))
        rows.append(ab("set(Version objects)", lambda: len(set(objs_p)), lambda: len(set(objs_r)), r, 1e3, "ms"))
        rows.append(ab("SpecifierSet.filter", lambda vs=vs: list(pss.filter(vs)), lambda vs=vs: list(rss.filter(vs)), r, 1e3, "ms"))
        rows.append(ab("filter_batch", lambda vs=vs: list(pss.filter(vs)), lambda vs=vs: rp.filter_batch(rss, vs), r, 1e3, "ms"))
        rows.append(ab("contains_batch", lambda vs=vs: [pss.contains(s) for s in vs], lambda vs=vs: rp.contains_batch(rss, vs), r, 1e3, "ms"))
        out["sections"][f"versions_{n}"] = rows

    release_list = sorted(set(versions(2000, rng)), key=pv.Version)
    print(f"\n## A project's release list ({len(release_list):,} versions) and requirement lists (ms)\n\n{head}")
    rows = []
    for spec in (">=1.0,<2", ">=4.2,!=4.3.*,<30", "~=12.4"):
        p, q = ps.SpecifierSet(spec), rs.SpecifierSet(spec)
        rows.append(ab(f"SpecifierSet({spec!r}).filter(releases)", lambda p=p: list(p.filter(release_list)), lambda q=q: list(q.filter(release_list)), repeat, 1e3, "ms"))
    reqs = [rng.choice(REQS) for _ in range(10_000)]
    rows.append(ab("Requirement() x 10,000", lambda: strs(pr.Requirement(x) for x in reqs), lambda: strs(rr.Requirement(x) for x in reqs), max(3, repeat // 3), 1e3, "ms"))
    markers = [pm.Marker(x) for x in MARKERS] * 250
    rmarkers = [rm.Marker(x) for x in MARKERS] * 250
    rows.append(ab("Marker.evaluate x 1,000", lambda: [x.evaluate() for x in markers], lambda: [x.evaluate() for x in rmarkers], repeat, 1e3, "ms"))
    out["sections"]["lists"] = rows
    out["loadavg_after"] = machine()["loadavg"]
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
