"""Every record of whole Common Crawl files: warcio 1.8.2 (+ datatrove 0.10.1) vs a candidate, at scale.

    python -m pyroid_parity.warc_scale OUT.json FILE... [--paths-file LIST] [--impl pyroid.warc]
        [--workers N] [--configs content/file,content/datatrove,...] [--no-documents] [--limit N]

For each file (local WARC / WET / WAT / ARC, gzip or plain), each job runs twice, once per implementation, in the
same worker process: the record level (``warc_compare.iter_events`` under each config: by default
``content/file``, warcio over the raw file, and ``content/datatrove``, warcio over the GzipFile datatrove opens)
and the datatrove level (``iter_documents(path)`` and datatrove's ``process_record`` over the candidate's
iterator, both against datatrove's own read path). Traces are not kept: each record's events become one
digest pair (main events, ``io`` events), so memory stays flat on 1 GB files. At the first record whose digests
differ, both sides run again up to that record and the summary gets the first differing event.

Without ``--impl`` the candidate is warcio itself (a run of the harness on the machine, and its timing). Files
run in parallel (``--workers``, default: the CPU count); results are written as they finish, so a stopped run
still leaves a valid OUT.json. The summary holds per-file counts (records, documents, bytes, seconds per side),
per-job status and up to ``--max-diffs`` diff examples; it never holds page content beyond the trace prefixes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

from pyroid_parity import impl as I
from pyroid_parity import warc_compare as C
from pyroid_parity import warc_datatrove as T

DEFAULT_CONFIGS = ("content/file", "content/datatrove")
ALL_CONFIGS = {c.name: c for c in C.SYNTH_CONFIGS}

_STATE: dict[str, Any] = {}


def _init(impl_spec: str) -> None:
    T.ensure_magic()
    from pyroid_parity.adapters import warc_warcio

    _STATE["ref"] = warc_warcio.make()
    _STATE["cand"] = _STATE["ref"] if impl_spec == "warcio" else I.resolve("warc", impl_spec).obj
    _STATE["spec"] = impl_spec


class Digests:
    """A sink that keeps one (main, io) digest pair per chunk of events (per record / item)."""

    def __init__(self, keep: int | None = None) -> None:
        self.pairs: list[tuple[bytes, bytes]] = []
        self.keep = keep
        self.kept: list[C.Event] = []
        self.records = 0

    def __call__(self, events: list[C.Event]) -> None:
        main, io_ = C.split(events)
        hm, hi = hashlib.sha1(), hashlib.sha1()
        for e in main:
            hm.update(C.canon(e).encode("utf-8", "surrogatepass"))
        for e in io_:
            hi.update(C.canon(e).encode("utf-8", "surrogatepass"))
        if self.keep is not None and len(self.pairs) == self.keep:
            self.kept = list(events)
        self.pairs.append((hm.digest()[:8], hi.digest()[:8]))
        self.records += sum(1 for e in events if e and e[0] in ("record", "item"))


def _first_mismatch(a: list[tuple[bytes, bytes]], b: list[tuple[bytes, bytes]], part: int) -> int | None:
    for i in range(max(len(a), len(b))):
        x = a[i][part] if i < len(a) else None
        y = b[i][part] if i < len(b) else None
        if x != y:
            return i
    return None


def _run(kind: str, side: Any, path: Path, name: str, sink: Digests, limit: int) -> float:
    t = time.perf_counter()
    if kind == "records":
        C.iter_events(side.mod, path, ALL_CONFIGS[name], limit, side.exc_name, sink)
    elif name == "reference":
        T.events_of(lambda: T.reference_documents(path), T.doc_view, side.exc_name, limit, sink)
    elif name == "iter_documents":
        T.events_of(lambda: side.mod.iter_documents(path), T.doc_view, side.exc_name, limit, sink)
    else:
        T.events_of(lambda: T.via_process_record(side.mod, path), T.doc_view, side.exc_name, limit, sink)
    return time.perf_counter() - t


def one(task: tuple[str, str, str, int]) -> dict[str, Any]:
    """One (file, kind, job): both sides, digests compared, the first diff explained."""
    path_s, kind, name, limit = task
    path = Path(path_s)
    ref, cand = _STATE["ref"], _STATE["cand"]
    out: dict[str, Any] = {"file": path_s, "kind": kind, "job": name, "bytes": path.stat().st_size}
    if kind == "documents" and name == "iter_documents" and not getattr(cand, "has_documents", False):
        out["status"] = "skip: no iter_documents"
        return out
    ref_job = "reference" if kind == "documents" else name
    want, got = Digests(), Digests()
    try:
        out["seconds_ref"] = round(_run(kind, ref, path, ref_job, want, limit), 3)
        out["seconds_cand"] = round(_run(kind, cand, path, name, got, limit), 3)
    except Exception as e:  # noqa: BLE001  a harness failure is reported, not fatal
        out["status"] = f"error: {type(e).__name__}: {e}"
        return out
    out["records" if kind == "records" else "documents"] = want.records
    out["chunks"] = [len(want.pairs), len(got.pairs)]
    diffs = []
    for part, label in ((0, "main"), (1, "io")):
        k = _first_mismatch(got.pairs, want.pairs, part)
        if k is None:
            continue
        g, w = Digests(keep=k), Digests(keep=k)
        _run(kind, ref, path, ref_job, w, limit)
        _run(kind, cand, path, name, g, limit)
        gm, gi = C.split(g.kept)
        wm, wi = C.split(w.kept)
        d = C.first_diff(gm, wm) if part == 0 else C.first_diff(gi, wi)
        diffs.append({"channel": label, "chunk": k, "diff": d.short() if d else None})
    out["status"] = "same" if not diffs else "diff"
    out["diffs"] = diffs
    return out


def tasks_for(
    paths: list[str], configs: list[str], documents: bool, limit: int
) -> list[tuple[str, str, str, int]]:
    out = []
    for p in paths:
        gz = p.endswith(".gz")
        for name in configs:
            if ALL_CONFIGS[name].needs_gzip and not gz:
                continue
            out.append((p, "records", name, limit))
        if documents:
            out.append((p, "documents", "iter_documents", limit))
            out.append((p, "documents", "process_record", limit))
    return out


def summarize(results: list[dict[str, Any]], meta: dict[str, Any], max_diffs: int) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for r in results:
        key = r.get("status", "?").split(":")[0]
        by_status[key] = by_status.get(key, 0) + 1
    examples = [
        {"file": r["file"], "kind": r["kind"], "job": r["job"], **d}
        for r in results
        if r.get("status") == "diff"
        for d in r["diffs"]
    ][:max_diffs]
    records = sum(
        r.get("records", 0) for r in results if r["kind"] == "records" and r["job"] == "content/file"
    )
    docs = sum(r.get("documents", 0) for r in results if r["job"] == "iter_documents")
    return {
        **meta,
        "status": "fail" if by_status.get("diff") or by_status.get("error") else "pass",
        "counts": {
            "files": len({r["file"] for r in results}),
            "jobs": len(results),
            "by_status": by_status,
            "records": records,
            "documents": docs,
            "bytes": sum({r["file"]: r["bytes"] for r in results}.values()),
            "seconds_ref": round(sum(r.get("seconds_ref", 0) for r in results), 1),
            "seconds_cand": round(sum(r.get("seconds_cand", 0) for r in results), 1),
        },
        "first_diffs": examples,
        "jobs": [{k: v for k, v in r.items() if k != "diffs"} for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pyroid_parity.warc_scale", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("out")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--paths-file", help="a file with one local path per line")
    ap.add_argument("--impl", default="warcio", help="candidate module (default: warcio itself)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--configs", default=",".join(DEFAULT_CONFIGS), help=f"of {sorted(ALL_CONFIGS)}")
    ap.add_argument("--no-documents", action="store_true", help="skip the datatrove level")
    ap.add_argument("--limit", type=int, default=0, help="records (documents) per file, 0 = all")
    ap.add_argument("--max-diffs", type=int, default=20)
    a = ap.parse_args(argv)
    paths = list(a.files)
    if a.paths_file:
        paths += [ln.strip() for ln in Path(a.paths_file).read_text("utf-8").splitlines() if ln.strip()]
    if not paths:
        ap.error("no input files")
    configs = [c for c in a.configs.split(",") if c]
    if unknown := [c for c in configs if c not in ALL_CONFIGS]:
        ap.error(f"unknown configs {unknown}")
    tasks = tasks_for(paths, configs, not a.no_documents, a.limit)
    meta = {
        "tool": "pyroid_parity.warc_scale",
        "impl": a.impl,
        "configs": configs,
        "documents": not a.no_documents,
        "limit": a.limit,
        "workers": a.workers,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpus": os.cpu_count(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(a.out)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.impl,)) as ex:
        futs = {ex.submit(one, t): t for t in tasks}
        for n, f in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001  a crashed worker is a result too
                t = futs[f]
                r = {
                    "file": t[0],
                    "kind": t[1],
                    "job": t[2],
                    "bytes": 0,
                    "status": f"error: {type(e).__name__}: {e}",
                }
            results.append(r)
            print(
                f"[{n}/{len(tasks)}] {r.get('status')} {Path(r['file']).name} {r['kind']}/{r['job']} "
                f"ref {r.get('seconds_ref', '-')} s cand {r.get('seconds_cand', '-')} s",
                flush=True,
            )
            meta["seconds_wall"] = round(time.perf_counter() - t0, 1)
            out.write_text(json.dumps(summarize(results, meta, a.max_diffs), indent=1) + "\n", "utf-8")
    s = summarize(results, meta, a.max_diffs)
    out.write_text(json.dumps(s, indent=1) + "\n", "utf-8")
    print(json.dumps(s["counts"]), s["status"])
    return 0 if s["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
