"""PyPI metadata corpus for pyroid.packaging's parity checks.

``fetch`` downloads, politely (a descriptive User-Agent, a global rate cap), the top projects of
hugovk's top-pypi-packages list and each one's PyPI JSON (``/pypi/<name>/json``): every release
version, every file's ``requires_python`` and the latest release's ``requires_dist``; and for the
first ``--per-release`` projects, the JSON of each of their last 10 releases (their own
``requires_dist``). ``compare`` checks pyroid.packaging against packaging 26.3 on all of it and
writes the counts. Keep the data on the devbox; it is not committed.

    python bench/packaging/pypi_corpus.py fetch --out /target/packaging-data --top 10000
    python bench/packaging/pypi_corpus.py compare --data /target/packaging-data --out counts.json
"""

# The lambdas in loops are called before the loop moves on; the lists are built for clarity.
# ruff: noqa: B023, PERF401

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TOP = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
AGENT = "pyroid-parity/1.0 (packaging parity corpus; https://github.com/ao/pyroid)"


class Limiter:
    def __init__(self, per_second: float) -> None:
        self.gap = 1.0 / per_second
        self.next = time.monotonic()
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next - now)
            self.next = max(now, self.next) + self.gap
        if delay:
            time.sleep(delay)


def get(url: str, limiter: Limiter, tries: int = 4) -> Any:
    for attempt in range(tries):
        limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(2**attempt)
    return None


def fetch(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    limiter = Limiter(args.rate)
    top = get(TOP, limiter)
    rows = top["rows"][: args.top]
    (out / "top.json").write_text(
        json.dumps({"source": TOP, "last_update": top.get("last_update"), "rows": rows})
    )

    def project(row: dict[str, Any]) -> dict[str, Any] | None:
        name = row["project"]
        data = get(f"https://pypi.org/pypi/{name}/json", limiter)
        if data is None:
            return None
        info = data.get("info") or {}
        releases = data.get("releases") or {}
        rp = sorted(
            {
                f.get("requires_python")
                for files in releases.values()
                for f in files
                if f.get("requires_python")
            }
        )
        return {
            "name": name,
            "downloads": row.get("download_count"),
            "releases": list(releases),
            "requires_python": rp,
            "requires_dist": info.get("requires_dist") or [],
            "latest": info.get("version"),
        }

    with cf.ThreadPoolExecutor(args.workers) as pool, (out / "projects.jsonl").open("w") as fh:
        for i, rec in enumerate(pool.map(project, rows)):
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
            if i % 500 == 0:
                print(f"projects: {i}/{len(rows)}", flush=True)

    projects = [json.loads(line) for line in (out / "projects.jsonl").open()]
    wanted = []
    for rec in projects[: args.per_release]:
        for v in rec["releases"][-10:]:
            wanted.append((rec["name"], v))

    def release(item: tuple[str, str]) -> dict[str, Any] | None:
        name, version = item
        data = get(f"https://pypi.org/pypi/{name}/{version}/json", limiter)
        if data is None:
            return None
        info = data.get("info") or {}
        return {
            "name": name,
            "version": version,
            "requires_dist": info.get("requires_dist") or [],
            "requires_python": info.get("requires_python"),
        }

    with cf.ThreadPoolExecutor(args.workers) as pool, (out / "releases.jsonl").open("w") as fh:
        for i, rec in enumerate(pool.map(release, wanted)):
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
            if i % 1000 == 0:
                print(f"releases: {i}/{len(wanted)}", flush=True)
    meta = {
        "top": args.top,
        "per_release": args.per_release,
        "fetched": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": "https://pypi.org/pypi/<name>/json",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


# --------------------------------------------------------------------------------------------


def outcome(f: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    try:
        return ("ok", f(*args, **kwargs))
    except Exception as e:  # noqa: BLE001
        return ("err", type(e).__name__, str(e))


def version_view(v: Any) -> tuple[Any, ...]:
    return (
        str(v),
        repr(v),
        v.epoch,
        v.release,
        v.pre,
        v.post,
        v.dev,
        v.local,
        v.public,
        v.base_version,
        v.is_prerelease,
        v._key,
        hash(v),
    )


CPYTHONS = [
    f"{a}.{b}.{c}"
    for a, b in [
        (2, 6),
        (2, 7),
        (3, 0),
        (3, 4),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (3, 10),
        (3, 11),
        (3, 12),
        (3, 13),
        (3, 14),
        (3, 15),
    ]
    for c in (0, 1, 5, 20)
]
CPYTHONS += ["3.13.0a1", "3.14.0rc1", "3.15.0b2", "3.12.1+", "3.14.0+local"]
ENVS: list[dict[str, Any]] = [
    {
        "python_version": "3.9",
        "python_full_version": "3.9.18",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "platform_release": "5.15.0",
        "implementation_version": "3.9.18",
    },
    {
        "python_version": "3.12",
        "python_full_version": "3.12.4",
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "os_name": "posix",
        "platform_machine": "arm64",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "platform_release": "23.5.0",
        "implementation_version": "3.12.4",
    },
    {
        "python_version": "3.14",
        "python_full_version": "3.14.0",
        "sys_platform": "win32",
        "platform_system": "Windows",
        "os_name": "nt",
        "platform_machine": "AMD64",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "platform_release": "10",
        "implementation_version": "3.14.0",
    },
    {
        "python_version": "3.10",
        "python_full_version": "3.10.14",
        "sys_platform": "linux",
        "platform_system": "Linux",
        "os_name": "posix",
        "platform_machine": "aarch64",
        "implementation_name": "pypy",
        "platform_python_implementation": "PyPy",
        "platform_release": "6.1.0",
        "implementation_version": "7.3.16",
    },
]


class Tally:
    def __init__(self) -> None:
        self.counts: dict[str, list[int]] = {}
        self.examples: dict[str, list[Any]] = {}

    def check(self, name: str, ours: Any, theirs: Any, what: Any) -> None:
        c = self.counts.setdefault(name, [0, 0])
        c[0] += 1
        if ours != theirs:
            c[1] += 1
            ex = self.examples.setdefault(name, [])
            if len(ex) < 20:
                ex.append([repr(what), repr(ours)[:300], repr(theirs)[:300]])


def compare(args: argparse.Namespace) -> None:
    import packaging.requirements as pr
    import packaging.specifiers as ps
    import packaging.utils as pu
    import packaging.version as pv
    import pyroid.packaging as rp
    import pyroid.packaging.requirements as rr
    import pyroid.packaging.specifiers as rs
    import pyroid.packaging.utils as ru
    import pyroid.packaging.version as rv

    import packaging

    assert packaging.__version__ == "26.3", packaging.__version__
    data = Path(args.data)
    projects = [json.loads(line) for line in (data / "projects.jsonl").open()]
    releases = (
        [json.loads(line) for line in (data / "releases.jsonl").open()]
        if (data / "releases.jsonl").exists()
        else []
    )
    t = Tally()
    by_name = {}
    # Versions: every release string, its parse, canonical forms, and each project's order.
    for rec in projects:
        texts = rec["releases"]
        mine, theirs = [], []
        for s in texts:
            a = outcome(lambda: version_view(rv.Version(s)))
            b = outcome(lambda: version_view(pv.Version(s)))
            t.check("version parse, str, properties, key, hash", a, b, s)
            t.check("canonicalize_version", ru.canonicalize_version(s), pu.canonicalize_version(s), s)
            if b[0] == "ok":
                mine.append(rv.Version(s))
                theirs.append(pv.Version(s))
        t.check(
            "sort of each project's releases",
            [str(v) for v in sorted(mine)],
            [str(v) for v in sorted(theirs)],
            rec["name"],
        )
        t.check(
            "sort_versions batch",
            [str(x) for x in rp.sort_versions([str(v) for v in theirs])],
            [str(v) for v in sorted(theirs)],
            rec["name"],
        )
        by_name[pu.canonicalize_name(rec["name"])] = [str(v) for v in theirs]
    every = [v for vs in by_name.values() for v in vs]
    t.check(
        "sort of all release versions",
        [str(v) for v in sorted(map(rv.Version, every))],
        [str(v) for v in sorted(map(pv.Version, every))],
        "all",
    )
    # Requires-Python: parse, contains over CPython versions, filter over the project's releases.
    for rec in projects:
        for value in rec["requires_python"]:
            a, b = outcome(rs.SpecifierSet, value), outcome(ps.SpecifierSet, value)
            t.check(
                "Requires-Python parse",
                (a[0], str(a[1]) if a[0] == "ok" else a),
                (b[0], str(b[1]) if b[0] == "ok" else b),
                value,
            )
            if a[0] != "ok" or b[0] != "ok":
                continue
            ra, rb = a[1], b[1]
            for py in CPYTHONS:
                for pre in (None, True, False):
                    t.check(
                        "Requires-Python contains (CPython versions, prereleases None/True/False)",
                        outcome(ra.contains, py, prereleases=pre),
                        outcome(rb.contains, py, prereleases=pre),
                        (value, py, pre),
                    )
            own = rec["releases"]
            t.check(
                "Requires-Python filter over the project's releases",
                outcome(lambda: list(ra.filter(own))),
                outcome(lambda: list(rb.filter(own))),
                value,
            )
            t.check(
                "contains_batch over CPython versions",
                rp.contains_batch(ra, CPYTHONS),
                [rb.contains(x) for x in CPYTHONS],
                value,
            )
    # Requires-Dist: parse, str, ==/hash across libraries, markers in 4 environments x 3 contexts,
    # each specifier's filter over the dependency's own release list.
    dists = [(rec["name"], d) for rec in projects for d in rec["requires_dist"]] + [
        (r["name"], d) for r in releases for d in r["requires_dist"]
    ]
    seen: set[str] = set()
    for _name, value in dists:
        if value in seen:
            continue
        seen.add(value)

        def view(cls: Any) -> Any:
            r = cls(value)
            return (
                str(r),
                repr(r),
                r.name,
                r.url,
                sorted(r.extras),
                str(r.specifier),
                None if r.marker is None else str(r.marker),
            )

        a, b = outcome(view, rr.Requirement), outcome(view, pr.Requirement)
        t.check("Requires-Dist parse and str", a, b, value)
        if a[0] != "ok" or b[0] != "ok":
            continue
        x, y = rr.Requirement(value), pr.Requirement(value)
        t.check(
            "Requirement == and hash across the two libraries",
            (x == y, y == x, hash(x) == hash(y)),
            (True, True, True),
            value,
        )
        if y.marker is not None:
            for env in ENVS:
                for extra in (None, "test", "dev"):
                    e = dict(env, extra=extra) if extra else env
                    for context in ("metadata", "requirement", "lock_file"):
                        t.check(
                            "marker evaluate (4 environments x extras x 3 contexts)",
                            outcome(x.marker.evaluate, e, context),
                            outcome(y.marker.evaluate, e, context),
                            (value, extra, context),
                        )
        dep = by_name.get(pu.canonicalize_name(y.name))
        if dep:
            for pre in (None, True, False):
                t.check(
                    "dependency specifier filter over the dependency's releases",
                    outcome(lambda: list(x.specifier.filter(dep, prereleases=pre))),
                    outcome(lambda: list(y.specifier.filter(dep, prereleases=pre))),
                    (value, pre),
                )
            t.check(
                "filter_batch over the dependency's releases",
                rp.filter_batch(x.specifier, dep),
                list(y.specifier.filter(dep)),
                value,
            )
    total = sum(c[0] for c in t.counts.values())
    diff = sum(c[1] for c in t.counts.values())
    report = {
        "packaging": packaging.__version__,
        "python": sys.version.split()[0],
        "projects": len(projects),
        "releases_json": len(releases),
        "requires_dist_unique": len(seen),
        "total": total,
        "different": diff,
        "checks": {k: {"compared": v[0], "different": v[1]} for k, v in t.counts.items()},
        "examples": t.examples,
    }
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", required=True)
    f.add_argument("--top", type=int, default=10000)
    f.add_argument("--per-release", type=int, default=1000)
    f.add_argument("--rate", type=float, default=20.0, help="requests per second, at most")
    f.add_argument("--workers", type=int, default=16)
    c = sub.add_parser("compare")
    c.add_argument("--data", required=True)
    c.add_argument("--out", required=True)
    a = p.parse_args()
    {"fetch": fetch, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
