#!/usr/bin/env python3
"""jmespath 1.1.0 vs pyroid.jmespath: per-call cost of small queries, and queries over records.

    python bench/jmespath/jmespath_bench.py --out results.json            # the full run
    python bench/jmespath/jmespath_bench.py --quick                        # smoke test

Both libraries run in the same process, interleaved, and each timing is the best of ``--repeat``
rounds (the median is recorded too). Before a case is timed, both results must be identical
(types, key order and which input objects come back); a mismatch aborts the run.

Cases:

- small: one boto3-style ``describe_instances`` response (3 instances): ``search()`` with the
  parse cached, and ``compile(e).search()`` on a compiled expression;
- uncached: ``search()`` on a new expression every call (each parses, as with more than 512
  distinct expressions in rotation);
- records: projections, filters, ``sort_by``, ``max_by``, ``sum``, multi-select hashes and
  ``contains`` over 10,000 and 1,000,000 records (``--sizes``).

Prints a Markdown table; ``--out`` writes the timings and the machine (CPU, Python, versions and
the load average before and after; no hostname or instance id).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


util = _load("_jmespath_util", ROOT / "python" / "tests" / "_jmespath_util.py")

import jmespath  # noqa: E402

import pyroid.jmespath as pj  # noqa: E402


def machine() -> dict[str, Any]:
    """The machine, without identifiers (no hostname or instance id)."""
    import os
    import platform

    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "cpu part")):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem = None
    try:
        mem = Path("/proc/meminfo").read_text().splitlines()[0].split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    import pyroid

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "cpu": cpu,
        "mem": mem,
        "loadavg": load,
        "jmespath": jmespath.__version__,
        "pyroid": pyroid.__version__,
    }


def response(rng: random.Random) -> dict[str, Any]:
    """A describe_instances-shaped response with 3 instances."""
    instances = []
    for i in range(3):
        instances.append(
            {
                "InstanceId": f"i-{rng.getrandbits(64):016x}",
                "InstanceType": rng.choice(["t3.micro", "m7g.large", "c7i.4xlarge"]),
                "LaunchTime": f"2026-0{i + 1}-0{i + 3}T10:00:00.000Z",
                "State": {"Code": 16, "Name": "running" if i != 1 else "stopped"},
                "PrivateIpAddress": f"10.0.0.{10 + i}",
                "Tags": [{"Key": "Name", "Value": f"web-{i}"}, {"Key": "env", "Value": "prod"}],
                "SecurityGroups": [{"GroupId": "sg-1", "GroupName": "default"}],
            }
        )
    return {"Reservations": [{"ReservationId": "r-1", "Instances": instances}]}


SMALL = [
    ("field", "Reservations"),
    ("path", "Reservations[0].Instances[0].State.Name"),
    ("projection", "Reservations[].Instances[].InstanceId"),
    ("filter", "Reservations[].Instances[?State.Name=='running'].InstanceId[]"),
    ("multi-select", "Reservations[].Instances[].{id: InstanceId, type: InstanceType}"),
    ("tag lookup", "Reservations[].Instances[].Tags[?Key=='Name'].Value[]"),
    ("sort_by", "sort_by(Reservations[].Instances[], &LaunchTime)[-1].InstanceId"),
    ("join", "join(', ', Reservations[].Instances[].InstanceId)"),
]


def records(n: int, rng: random.Random) -> list[dict[str, Any]]:
    cities = ["Paris", "Oslo", "Lima", "Kyoto", "Accra"]
    return [
        {
            "id": i,
            "name": f"user{i}",
            "age": rng.randint(18, 90),
            "score": rng.random() * 100,
            "tags": ["a", "b", "c"][: i % 4],
            "address": {"city": rng.choice(cities), "zip": f"{i:05d}"},
        }
        for i in range(n)
    ]


RECORDS = [
    ("projection", "[*].address.city"),
    ("filter", "[?age > `30`].name"),
    ("filter, and", "[?age > `30` && address.city == 'Oslo'].id"),
    ("multi-select", "[*].{n: name, c: address.city}"),
    ("sort_by", "sort_by(@, &age)[0].id"),
    ("max_by", "max_by(@, &score).id"),
    ("sum", "sum([*].age)"),
    ("contains", "[?contains(tags, 'a')] | length(@)"),
]


def best(fn: Callable[[], Any], repeat: int, number: int) -> tuple[float, float]:
    """(best, median) seconds per call over `repeat` rounds of `number` calls."""
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number)
    return min(times), statistics.median(times)


def calls_for(fn: Callable[[], Any], target: float = 0.05) -> int:
    t0 = time.perf_counter()
    fn()
    one = max(time.perf_counter() - t0, 1e-7)
    return max(1, int(target / one))


def gate(expression: str, data: Any) -> None:
    ids = util.index(data)
    a = util.outcome(jmespath.search, expression, data, None, ids)
    b = util.outcome(pj.search, expression, data, None, ids)
    if a != b:
        sys.exit(f"results differ for {expression!r}:\n  jmespath {a}\n  pyroid   {b}")


def ab(
    label: str,
    expression: str,
    fj: Callable[[], Any],
    fp: Callable[[], Any],
    repeat: int,
    scale: float = 1e6,
) -> dict[str, Any]:
    nj, np_ = calls_for(fj), calls_for(fp)
    # Interleave the two, so load on a shared machine hits both.
    rows_j, rows_p = [], []
    for _ in range(repeat):
        rows_j.append(best(fj, 1, nj)[0])
        rows_p.append(best(fp, 1, np_)[0])
    j, p = min(rows_j), min(rows_p)
    row = {
        "case": label,
        "expression": expression,
        "jmespath_us": j * 1e6,
        "pyroid_us": p * 1e6,
        "jmespath_median_us": statistics.median(rows_j) * 1e6,
        "pyroid_median_us": statistics.median(rows_p) * 1e6,
        "speedup": j / p,
    }
    print(f"| {label} | `{expression}` | {j * scale:,.2f} | {p * scale:,.2f} | {j / p:.1f}x |")
    sys.stdout.flush()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="write timings and machine info here (JSON)")
    ap.add_argument("--repeat", type=int, default=15)
    ap.add_argument("--sizes", default="10000,1000000", help="record counts")
    ap.add_argument("--quick", action="store_true", help="few rounds, 1,000 records")
    args = ap.parse_args()
    repeat = 3 if args.quick else args.repeat
    sizes = [1000] if args.quick else [int(s) for s in args.sizes.split(",")]
    rng = random.Random(20261115)
    out: dict[str, Any] = {"machine": machine(), "sections": {}}

    data = response(rng)
    print("\n## Small queries, one describe_instances response (µs per call)\n")
    print("| Case | Expression | jmespath | pyroid | Speedup |\n|---|---|---:|---:|---:|")
    rows = []
    for label, e in SMALL:
        gate(e, data)
        rows.append(ab(f"{label}, search()", e, lambda e=e: jmespath.search(e, data),
                       lambda e=e: pj.search(e, data), repeat))
        cj, cp = jmespath.compile(e), pj.compile(e)
        rows.append(ab(f"{label}, compiled", e, lambda cj=cj: cj.search(data),
                       lambda cp=cp: cp.search(data), repeat))
    out["sections"]["small"] = rows

    print("\n## A new expression every call: parse and search (µs per call)\n")
    print("| Case | Expression | jmespath | pyroid | Speedup |\n|---|---|---:|---:|---:|")
    template = "Reservations[].Instances[?State.Name=='running' && Tags[?Value=='web-{0}']].InstanceId"
    counter = iter(range(10**9))

    def fresh_j() -> Any:
        return jmespath.search(template.format(next(counter)), data)

    def fresh_p() -> Any:
        return pj.search(template.format(next(counter)), data)

    gate(template.format(0), data)
    out["sections"]["uncached"] = [ab("parse + search", template, fresh_j, fresh_p, repeat)]

    for n in sizes:
        recs = records(n, rng)
        r = max(3, repeat // (3 if n >= 100_000 else 1))
        print(f"\n## {n:,} records (ms per query)\n")
        print("| Case | Expression | jmespath | pyroid | Speedup |\n|---|---|---:|---:|---:|")
        rows = []
        for label, e in RECORDS:
            gate(e, recs)
            cj, cp = jmespath.compile(e), pj.compile(e)
            row = ab(
                label, e, lambda cj=cj: cj.search(recs), lambda cp=cp: cp.search(recs), r, 1e3
            )
            rows.append(row)
        out["sections"][f"records_{n}"] = rows
    out["loadavg_after"] = machine()["loadavg"]
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
