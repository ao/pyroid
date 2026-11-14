#!/usr/bin/env python3
"""Load and dump timings of PyYAML (pure and libyaml), ruamel.yaml and rapidyaml on the YAML corpus.

    python -P bench/yaml/baseline_bench.py /target/data/corpus \
        --index /target/data/oracle/corpus/index.jsonl --out /target/data/bench/baseline.json

Context numbers for pyroid.yaml's benchmarks, not a gate. Every (implementation, input) cell runs in
a fresh subprocess that reads its inputs into memory first; the time is the sum of per-file
``time.perf_counter`` deltas around one call, with the garbage collector left on (the prepared
inputs are frozen out of it with ``gc.freeze()``).

Implementations (``LOADS`` and ``DUMPS``):
  load  pyroid-safe_load_all  list(pyroid.yaml.safe_load_all(data))
        pyyaml-SafeLoader     list(yaml.load_all(data, Loader=yaml.SafeLoader))   (= yaml.safe_load_all)
        pyyaml-CSafeLoader    list(yaml.load_all(data, Loader=yaml.CSafeLoader))
        ruamel-safe-pure      list(YAML(typ="safe", pure=True).load_all(data)), allow_duplicate_keys
        ruamel-safe-c         the same with typ="safe" and ruamel.yaml.clib, when it is installed
        ryml-parse            ryml.parse_in_arena(data): rapidyaml's parse tree, no Python objects
  dump  pyroid-safe_dump_all  pyroid.yaml.safe_dump_all(docs)
        pyyaml-SafeDumper     yaml.safe_dump_all(docs)
        pyyaml-CSafeDumper    yaml.dump_all(docs, Dumper=yaml.CSafeDumper)
ruamel.yaml follows YAML 1.2 and rapidyaml builds no Python values, so their work differs from PyYAML's.
The dump inputs are the documents CSafeLoader loads (MB/s counts the YAML written). A file an
implementation raises on is left out of its totals and counted.

Inputs: ``corpus`` is every file outside ``synthetic/`` that SafeLoader loads (per the recorder's
index), as one total in MB/s; ``1mb``, ``10mb`` and ``100mb`` are ``synthetic/mixed-*.yaml``, best of
5, 3 and 1 runs. ``/proc/loadavg`` is recorded around every cell: on a shared machine the numbers
are indicative only.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

LOADS = [
    "pyroid-safe_load_all",
    "pyyaml-SafeLoader",
    "pyyaml-CSafeLoader",
    "ruamel-safe-pure",
    "ruamel-safe-c",
    "ryml-parse",
]
DUMPS = ["pyroid-safe_dump_all", "pyyaml-SafeDumper", "pyyaml-CSafeDumper"]
SIZES = {
    "1mb": ("synthetic/mixed-1mb.yaml", 5),
    "10mb": ("synthetic/mixed-10mb.yaml", 3),
    "100mb": ("synthetic/mixed-100mb.yaml", 1),
}


def make_call(impl: str):
    """Return the timed callable: it takes the file bytes (loads) or the loaded documents (dumps)."""
    import yaml

    if impl == "pyroid-safe_load_all":
        import pyroid.yaml as py

        return lambda d: list(py.safe_load_all(d))
    if impl == "pyroid-safe_dump_all":
        import pyroid.yaml as py

        return lambda docs: py.safe_dump_all(docs)
    if impl == "pyyaml-SafeLoader":
        return lambda d: list(yaml.load_all(d, Loader=yaml.SafeLoader))
    if impl == "pyyaml-CSafeLoader":
        return lambda d: list(yaml.load_all(d, Loader=yaml.CSafeLoader))
    if impl.startswith("ruamel-safe-"):
        from ruamel.yaml import YAML

        pure = impl.endswith("pure")
        if not pure:
            import _ruamel_yaml  # noqa: F401 - ruamel.yaml.clib; ImportError marks the row unavailable
        y = YAML(typ="safe", pure=pure)
        y.allow_duplicate_keys = True
        return lambda d: list(y.load_all(d))
    if impl == "ryml-parse":
        import ryml

        return lambda d: ryml.parse_in_arena(d)
    if impl == "pyyaml-SafeDumper":
        return lambda docs: yaml.safe_dump_all(docs)
    if impl == "pyyaml-CSafeDumper":
        return lambda docs: yaml.dump_all(docs, Dumper=yaml.CSafeDumper)
    raise ValueError(impl)


def load_docs(yaml: object, data: bytes) -> list:
    try:
        return list(yaml.load_all(data, Loader=yaml.CSafeLoader))
    except yaml.YAMLError:
        return list(yaml.load_all(data, Loader=yaml.SafeLoader))


def worker(spec: dict) -> dict:
    """Time one implementation over the listed files; print a JSON result."""
    import yaml

    impl, files, repeat = spec["impl"], spec["files"], spec["repeat"]
    progress = Path(spec["progress"])
    try:
        call = make_call(impl)
    except ImportError as e:
        return {"impl": impl, "unavailable": f"{type(e).__name__}: {e}"}
    inputs = [Path(f).read_bytes() for f in files]
    is_dump = impl in DUMPS
    if is_dump:
        inputs = [load_docs(yaml, d) for d in inputs]
    # Move the prepared inputs out of the collector's reach: the dump inputs are millions of
    # objects, and walking them on every collection would be timed as part of each call.
    gc.collect()
    gc.freeze()
    best = None
    for _ in range(repeat):
        total, ok_bytes, ok_files, out_bytes, failed = 0.0, 0, 0, 0, []
        for f, inp in zip(files, inputs, strict=True):
            progress.write_text(f)
            size = os.path.getsize(f)
            gc.collect()
            t0 = time.perf_counter()
            try:
                res = call(inp)
            except Exception as e:  # noqa: BLE001 - left out of this implementation's totals
                failed.append([f, type(e).__name__])
                continue
            dt = time.perf_counter() - t0
            total += dt
            ok_bytes += size
            ok_files += 1
            if is_dump:
                out_bytes += len(res.encode("utf-8", "surrogatepass"))
            del res
        if best is None or total < best["seconds"]:
            best = {"seconds": total, "bytes": ok_bytes, "files": ok_files, "failed": failed}
            if is_dump:
                best["output_bytes"] = out_bytes
    best["impl"] = impl
    best["repeat"] = repeat
    # Loads: MB of input per second; dumps: MB of YAML written per second.
    mb = best["output_bytes"] if is_dump else best["bytes"]
    best["mb_per_s"] = mb / 1e6 / best["seconds"] if best["seconds"] else None
    return best


def run_cell(impl: str, files: list[str], repeat: int, tmp: Path) -> dict:
    """Run a worker subprocess; skip and retry past a file that crashes the interpreter."""
    crashed: list[str] = []
    while True:
        todo = [f for f in files if f not in crashed]
        spec = {"impl": impl, "files": todo, "repeat": repeat, "progress": str(tmp / "progress")}
        (tmp / "spec.json").write_text(json.dumps(spec))
        la0 = Path("/proc/loadavg").read_text().split()[:3]
        r = subprocess.run(
            [sys.executable, "-P", "-s", __file__, "--worker", str(tmp / "spec.json")],
            capture_output=True,
            text=True,
            check=False,
        )
        la1 = Path("/proc/loadavg").read_text().split()[:3]
        if r.returncode == 0:
            res = json.loads(r.stdout.strip().splitlines()[-1])
            res["loadavg_before"], res["loadavg_after"] = la0, la1
            res["crashed"] = crashed
            return res
        bad = (tmp / "progress").read_text() if (tmp / "progress").exists() else None
        if bad is None or bad in crashed or len(crashed) >= 20:
            return {
                "impl": impl,
                "error": f"worker exit {r.returncode}: {r.stderr[-500:]}",
                "crashed": crashed,
            }
        crashed.append(bad)


def machine() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpus": os.cpu_count()}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("CPU part"):
                info["cpu_part"] = line.split(":")[1].strip()
                break
        mem = Path("/proc/meminfo").read_text().split("\n")[0]
        info["mem"] = mem.split(":")[1].strip()
    except OSError:
        pass
    for dist in ("pyyaml", "ruamel.yaml", "ruamel.yaml.clib", "rapidyaml"):
        try:
            info[dist] = md.version(dist)
        except md.PackageNotFoundError:
            info[dist] = None
    return info


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        print(json.dumps(worker(json.loads(Path(sys.argv[2]).read_text()))))
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--index", type=Path, required=True, help="record_oracle.py index.jsonl of the corpus")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--impl", action="append", help="only these implementations")
    ap.add_argument("--inputs", default="corpus,1mb,10mb,100mb")
    a = ap.parse_args()

    recs = [json.loads(line) for line in a.index.read_text().splitlines()]
    corpus = [
        str(a.corpus / r["path"])
        for r in recs
        if r.get("error") is None and "recorder_error" not in r and not r["path"].startswith("synthetic/")
    ]
    inputs = {"corpus": (corpus, 1)}
    for k, (rel, rep) in SIZES.items():
        inputs[k] = ([str(a.corpus / rel)], rep)
    impls = a.impl or LOADS + DUMPS
    tmp = a.out.parent / "bench-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    result = {"machine": machine(), "label": "indicative (shared machine)", "cells": []}
    for name in a.inputs.split(","):
        files, rep = inputs[name]
        for impl in impls:
            t0 = time.time()
            res = run_cell(impl, files, rep, tmp)
            res["input"] = name
            result["cells"].append(res)
            mbs = res.get("mb_per_s")
            print(
                f"{name:7} {impl:20} {mbs and round(mbs, 3)} MB/s  files {res.get('files')} "
                f"failed {len(res.get('failed', []))} crashed {len(res.get('crashed', []))} "
                f"load {res.get('loadavg_before')} wall {time.time() - t0:.0f}s {res.get('unavailable', '')}"
                f"{res.get('error', '')}",
                file=sys.stderr,
                flush=True,
            )
            a.out.write_text(json.dumps(result, indent=1) + "\n")
    print("| input | " + " | ".join(impls) + " |")
    print("|---|" + "---|" * len(impls))
    for name in a.inputs.split(","):
        row = []
        for impl in impls:
            c = next(c for c in result["cells"] if c["input"] == name and c["impl"] == impl)
            row.append("n/a" if c.get("mb_per_s") is None else f"{c['mb_per_s']:.2f}")
        print(f"| {name} | " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
