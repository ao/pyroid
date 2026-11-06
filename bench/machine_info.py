#!/usr/bin/env python3
"""Record the machine and environment a results dir was measured on (R2 section 5: disclose hardware).

    python machine_info.py --out results/<machine>/<date>/machine.json   # "before" snapshot
    python machine_info.py --update results/<machine>/<date>/machine.json   # add the "after" snapshot
    python machine_info.py --slug                                       # results dir name, e.g. orbstack-aarch64-12cpu

Other sessions may share the VM, so the load average and cgroup memory are recorded before AND after.
The git sha comes from ``git`` when the repo metadata is reachable, else from ``BENCH_GIT_SHA`` (a git worktree's
``.git`` file points outside the container mount, so pass ``BENCH_GIT_SHA=$(git rev-parse HEAD)`` from the host).
``HOST_CPU`` may name the physical CPU, which VMs hide (lscpu shows "Model name: -" under OrbStack).
AWS EC2 is recognised from DMI (``BENCH_DMI_DIR`` [/sys/devices/virtual/dmi/id]; on Nitro ``sys_vendor`` is
"Amazon EC2", ``product_name`` the instance type, ``board_asset_tag`` the instance id): ``cloud`` in machine.json
and the slug ``aws-<instance type>``, e.g. aws-c7i.4xlarge. ``os_release`` reads ``BENCH_OS_RELEASE``
[/etc/os-release]; inside a container that is the image's OS, so mount the host's file there to record the host's.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CGROUP = Path("/sys/fs/cgroup")
KEY_DISTS = [
    "ftfy",
    "datatrove",
    "spacy",
    "fasttext-predict",
    "datasketch",
    "rensa",
    "numpy",
    "pyarrow",
    "pyperf",
    "psutil",
    "matplotlib",
    "pyroid-parity",
    "pyroid",
    "regex",
    "tokenizers",
    "astral-dev-toolchain-hyperfine",
]


def run(cmd: list[str], timeout: float = 60) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"error: {e}"
    return (p.stdout + (p.stderr if p.returncode else "")).strip()


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def lscpu() -> dict[str, str]:
    out = run(["lscpu"]) or ""
    return {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in out.splitlines()) if v}


def cpuinfo_model() -> str | None:
    text = read(Path("/proc/cpuinfo")) or ""
    for key in ("model name", "Hardware", "CPU part"):
        m = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
        if m:
            return f"{key}: {m.group(1)}"
    return None if text else platform.processor() or None


def meminfo() -> dict[str, int]:
    out = {}
    for line in (read(Path("/proc/meminfo")) or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "SwapTotal"):
            out[k + "_kib"] = int(v.split()[0])
    return out


def cgroup() -> dict[str, Any]:
    keys = ["cpu.max", "cpuset.cpus.effective", "memory.max", "memory.current", "memory.peak"]
    out: dict[str, Any] = {k: read(CGROUP / k) for k in keys}
    stat = read(CGROUP / "memory.stat") or ""
    out["memory.stat"] = {
        k: int(v) for k, _, v in (ln.partition(" ") for ln in stat.splitlines()) if k in ("anon", "file")
    }
    return out


def git() -> dict[str, Any]:
    sha = run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    if sha and re.fullmatch(r"[0-9a-f]{40}", sha):
        status = run(["git", "-C", str(REPO), "status", "--porcelain"]) or ""
        return {
            "sha": sha,
            "source": "git",
            "dirty_files": len(status.splitlines()),
            "branch": run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"]),
        }
    env = os.environ.get("BENCH_GIT_SHA")
    return {
        "sha": env or "unknown",
        "source": "BENCH_GIT_SHA" if env else "unavailable",
        "dirty_files": os.environ.get("BENCH_GIT_DIRTY"),
    }


def packages() -> dict[str, Any]:
    key = {}
    for d in KEY_DISTS:
        try:
            key[d] = md.version(d)
        except md.PackageNotFoundError:
            key[d] = None
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions() if d.metadata["Name"])
    return {"key": key, "freeze": freeze}


def python_info() -> dict[str, Any]:
    cfg = sysconfig.get_config_var
    return {
        "version": sys.version,
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "free_threaded_build": bool(cfg("Py_GIL_DISABLED")),
        "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        "config_args": cfg("CONFIG_ARGS"),
        "cflags": cfg("CFLAGS"),
    }


def snapshot() -> dict[str, Any]:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": list(os.getloadavg()),
        "uptime_s": float((read(Path("/proc/uptime")) or "0").split()[0]),
        "cgroup": cgroup(),
        "meminfo": meminfo(),
    }


def cloud() -> dict[str, Any] | None:
    """AWS EC2 (Nitro) from DMI, else None."""
    dmi = Path(os.environ.get("BENCH_DMI_DIR", "/sys/devices/virtual/dmi/id"))
    if read(dmi / "sys_vendor") != "Amazon EC2":
        return None
    return {
        "provider": "aws",
        "service": "ec2",
        "instance_type": read(dmi / "product_name"),
        "instance_id": read(dmi / "board_asset_tag"),
    }


def os_release() -> dict[str, str] | None:
    """NAME, VERSION_ID and PRETTY_NAME from os-release(5)."""
    text = read(Path(os.environ.get("BENCH_OS_RELEASE", "/etc/os-release")))
    if text is None:
        return None
    out = {}
    for line in text.splitlines():
        k, _, v = line.partition("=")
        if k.strip() in ("NAME", "VERSION_ID", "PRETTY_NAME"):
            out[k.strip()] = v.strip().strip("\"'")
    return out


def slug() -> str:
    c = cloud()
    if c and c["instance_type"]:
        return "aws-" + re.sub(r"[^a-z0-9.]+", "-", c["instance_type"].lower()).strip("-")
    u = platform.uname()
    where = (
        "orbstack"
        if "orbstack" in u.release.lower()
        else ("docker" if Path("/.dockerenv").exists() else "host")
    )
    host = os.environ.get("HOST_CPU")
    tag = f"-{re.sub(r'[^a-z0-9]+', '-', host.lower()).strip('-')}" if host else ""
    return f"{where}{tag}-{u.machine}-{os.cpu_count()}cpu"


def collect() -> dict[str, Any]:
    u = platform.uname()
    manifest = REPO / "bench" / "manifest.json"
    ec2 = cloud()
    return {
        "slug": slug(),
        "uname": u._asdict(),
        "dockerenv": Path("/.dockerenv").exists(),
        "virtualized": "orbstack" in u.release.lower() or Path("/.dockerenv").exists() or ec2 is not None,
        "cloud": ec2,
        "os_release": os_release(),
        "host_cpu": os.environ.get("HOST_CPU"),
        "cpu": {
            "model": cpuinfo_model(),
            "lscpu": lscpu(),
            "cpu_count": os.cpu_count(),
            "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
        "python": python_info(),
        "packages": packages(),
        "tools": {
            "hyperfine": run(["hyperfine", "--version"]),
            "rustc": run(["rustc", "-V"]),
            "uv": run(["uv", "--version"]),
            "taskset": shutil.which("taskset"),
        },
        "git": git(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(("BENCH_", "RAYON_", "PYROID_"))},
        "pyperf_system_show": run([sys.executable, "-m", "pyperf", "system", "show"]),
        "before": snapshot(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out", help="write a new machine.json (with the 'before' snapshot)")
    g.add_argument("--update", help="add the 'after' snapshot to an existing machine.json")
    g.add_argument("--slug", action="store_true", help="print the machine slug used for results/<machine>/")
    a = ap.parse_args()
    if a.slug:
        print(slug())
        return 0
    if a.update:
        p = Path(a.update)
        doc = json.loads(p.read_text("utf-8"))
        doc["after"] = snapshot()
    else:
        p = Path(a.out)
        doc = collect()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {p}: {doc['slug']}, load {doc[('after' if a.update else 'before')]['loadavg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
