"""Large-sample trafilatura validation: stream Common Crawl slices, compare, keep only the counts (not CI).

    python -m pyroid_parity.trafilatura_scale OUT_DIR [--impl MODULE] [--compat 1.11.0] [--pages 20000] [--crawls ...]

``--compat`` picks the trafilatura release (mode) checked: the installed trafilatura must be it (with the pins of
``specs.TRAFILATURA_ORACLES``), its option matrix is used, and a candidate that takes ``compat=`` gets it.

For each crawl, WARC files are picked with a seeded RNG (never the ones the pinned corpus uses), and from each a
slice of ``--slice-mb`` at a random offset, cut to whole gzip members (``trafilatura_cc``). The slice becomes
pages exactly as datatrove 0.10.1 hands them to Trafilatura, is compared under every option set, and is dropped:
nothing but the JSON summaries stays on disk, so the disk budget is one slice plus the summaries.

Without ``--impl`` (a reference self-check) trafilatura runs twice in different worker processes and the two
runs must agree. The second run checks the dedup state handling as well: each stateful set goes through the
slice in three parts in three processes, the module-level cache (``trafilatura.deduplication.LRU_TEST``) carried
from one to the next as its (key, count) items, while the first run uses ``deduplicate=True`` with that cache
emptied at the slice start, in one process. With ``--impl`` the candidate runs in this process
(``extract_batch`` if it has one, the dedup cache emptied with ``reset_caches()`` at each slice start) against
trafilatura in the workers.

Outputs in OUT_DIR: ``<crawl>.json`` per crawl (per option set: pages, same, diff, statuses, up to 5 examples;
wall seconds of the oracle and candidate runs), ``summary.json`` (all crawls), ``state.json`` (finished slices, so a rerun resumes). ``dedup_effect``
counts the pages where the stateful ``datatrove`` set differs from ``precision`` (the same options without dedup):
how much the cache state matters on that crawl.

Docker (local, indicative), one OUT_DIR per mode (a rerun with the same OUT_DIR resumes)::

    cd parity && PYROID_DATA=/target/data-parity uv run --frozen --no-default-groups --group trafilatura \\
        --group cc python -m pyroid_parity.trafilatura_scale /target/out-parity/scale --pages 20000 --workers 8
    cd parity && PYROID_DATA=/target/data-parity uv run --frozen --no-default-groups --group trafilatura111 \\
        --group cc python -m pyroid_parity.trafilatura_scale /target/out-parity/scale-111 --compat 1.11.0

With the candidate: ``uv run ... --with dist/pyroid-*.whl python -m pyroid_parity.trafilatura_scale OUT_DIR
--impl pyroid.trafilatura [--compat 1.11.0] --pages 100000 --workers $(nproc) --slice-mb 100``.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pyroid_parity import specs
from pyroid_parity import trafilatura_cc as CC
from pyroid_parity import trafilatura_compare as T

DEFAULT_CRAWLS = (
    "CC-MAIN-2026-43",
    "CC-MAIN-2025-08",
    "CC-MAIN-2023-14",
    "CC-MAIN-2021-04",
    "CC-MAIN-2019-04",
)
#: The pinned corpus's WARC files (bench/fetch.py): never sampled here.
PINNED_WARCS = {
    "crawl-data/CC-MAIN-2026-39/segments/1788492699475.64/warc/CC-MAIN-20260904131603-20260904161603-00100.warc.gz",
    "crawl-data/CC-MAIN-2023-50/segments/1700679099281.67/warc/CC-MAIN-20231128083443-20231128113443-00100.warc.gz",
    "crawl-data/CC-MAIN-2019-35/segments/1566027312025.20/warc/CC-MAIN-20190817203056-20190817225056-00100.warc.gz",
}


def option_sets(mode: str) -> dict[str, dict[str, Any]]:
    """A mode's corpus option sets, plus "precision" (datatrove's options without dedup) for the dedup effect."""
    return {
        **specs.TRAFILATURA_OPTION_SETS_BY_MODE[mode],
        "precision": {"favor_precision": True, "include_comments": False},
    }


OPTION_SETS = option_sets(specs.TRAFILATURA_VERSION)
CHUNK = 40
MAX_EXAMPLES = 5


def log(msg: str) -> None:
    print(f"[scale {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------- worker side


def _row(o: T.Outcome) -> tuple[str, int | None, str]:
    status = f"raise:{o.value[0]}" if o.kind == "raise" else ("None" if o.value is None else "str")
    return T.digest(o), T.text_length(o), status


def job_stateless(
    pages: list[str], mode: str, options: list[str], reverse: bool
) -> dict[str, list[tuple[Any, ...]]]:
    """Every stateless option set on a chunk of pages (``reverse``: visit the pages backwards, a second run)."""
    import trafilatura

    sets = option_sets(mode)
    order = list(range(len(pages)))[::-1] if reverse else list(range(len(pages)))
    out: dict[str, list[Any]] = {o: [None] * len(pages) for o in options}
    for i in order:
        for o in options:
            out[o][i] = _row(T.capture_call(trafilatura.extract, pages[i], **sets[o]))
    return out


def job_stateful_global(pages: list[str], mode: str, option: str) -> list[tuple[Any, ...]]:
    """A stateful set as datatrove runs it: ``deduplicate=True``, the module-level cache emptied first."""
    import trafilatura
    from trafilatura.meta import reset_caches

    kw = option_sets(mode)[option]
    reset_caches()
    return [_row(T.capture_call(trafilatura.extract, p, **kw)) for p in pages]


def job_stateful_carried(
    pages: list[str], mode: str, option: str, items: list[tuple[str, Any]]
) -> tuple[list[tuple[str, Any]], list[tuple[Any, ...]]]:
    """Part of a stateful run: the module-level cache restored from ``items`` (least recent first), the pages
    run with ``deduplicate=True``, and the cache's items returned for the next part (another process)."""
    import trafilatura
    from trafilatura import deduplication

    kw = option_sets(mode)[option]
    original = deduplication.LRU_TEST
    cache = deduplication.LRUCache(maxsize=original.maxsize)
    for key, value in items:
        cache.put(key, value)
    deduplication.LRU_TEST = cache  # read at call time by duplicate_test, in 2.3.1 and 1.11.0
    try:
        rows = [_row(T.capture_call(trafilatura.extract, p, **kw)) for p in pages]
    finally:
        deduplication.LRU_TEST = original
    return T._lru_items(cache), rows


# --------------------------------------------------------------------------------------------- driver side


class Tally:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    def add(self, option: str, same: bool, status: str, example: dict[str, Any] | None) -> None:
        t = self.data.setdefault(option, {"pages": 0, "same": 0, "diff": 0, "statuses": {}, "examples": []})
        t["pages"] += 1
        t["same" if same else "diff"] += 1
        t["statuses"][status] = t["statuses"].get(status, 0) + 1
        if not same and example is not None and len(t["examples"]) < MAX_EXAMPLES:
            t["examples"].append(example)


def warc_paths(crawl: str, out: Path) -> list[str]:
    cache = out / f"{crawl}.warc.paths.gz"
    if not cache.exists():
        url = f"{CC.CC_DATA}crawl-data/{crawl}/warc.paths.gz"
        with urllib.request.urlopen(url, timeout=120) as r:  # Common Crawl's public listing
            cache.write_bytes(r.read())
    return gzip.decompress(cache.read_bytes()).decode().split()


def warc_size(warc: str) -> int:
    req = urllib.request.Request(CC.CC_DATA + warc, method="HEAD", headers={"User-Agent": CC.USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return int(r.headers["Content-Length"])


def plan_slice(crawl: str, paths: list[str], rng: random.Random, slice_bytes: int) -> tuple[str, int]:
    while True:
        warc = paths[rng.randrange(len(paths))]
        if warc not in PINNED_WARCS:
            break
    size = warc_size(warc)
    offset = rng.randrange(0, max(1, size - slice_bytes))
    return warc, offset


def compare_slice(
    pool: ProcessPoolExecutor,
    candidate: Any,
    pages: list[tuple[str, str, str]],
    where: dict[str, Any],
    tally: Tally,
    mode: str = specs.TRAFILATURA_VERSION,
) -> None:
    htmls = [h for _, _, h in pages]
    n = len(htmls)
    sets = option_sets(mode)
    stateful = [o for o, kw in sets.items() if specs.trafilatura_stateful(kw)]
    stateless = [o for o in sets if o not in stateful]
    chunks = [(i, min(n, i + CHUNK)) for i in range(0, n, CHUNK)]

    t0 = time.perf_counter()
    first_g = {o: pool.submit(job_stateful_global, htmls, mode, o) for o in stateful}
    first_s = [pool.submit(job_stateless, htmls[a:b], mode, stateless, False) for a, b in chunks]
    second: dict[str, list[Any]] = {}
    if candidate is None:
        # Second run: other chunk boundaries, pages visited backwards, stateful sets carried across 3 processes.
        offs = [(i, min(n, i + CHUNK + 7)) for i in range(0, n, CHUNK + 7)]
        sec_s = [(a, pool.submit(job_stateless, htmls[a:b], mode, stateless, True)) for a, b in offs]
        thirds = [0, n // 3, 2 * n // 3, n]
        for o in stateful:
            items: list[tuple[str, Any]] = []
            rows: list[Any] = []
            for a, b in itertools.pairwise(thirds):
                items, part = pool.submit(job_stateful_carried, htmls[a:b], mode, o, items).result()
                rows += part
            second[o] = rows
        for o in stateless:
            second[o] = [None] * n
        for a, f in sec_s:
            res = f.result()
            for o in stateless:
                second[o][a : a + len(res[o])] = res[o]
    first: dict[str, list[Any]] = {o: f.result() for o, f in first_g.items()}
    for o in stateless:
        first[o] = []
    for f in first_s:
        res = f.result()
        for o in stateless:
            first[o] += res[o]
    oracle_s = time.perf_counter() - t0

    cand_s = 0.0
    if candidate is not None:
        for o, kw in sets.items():
            t1 = time.perf_counter()
            if specs.trafilatura_stateful(kw):
                candidate.reset_caches()
            if candidate.has_batch:
                outs = []
                for a, b in chunks:
                    batch = candidate.extract_batch(htmls[a:b], threads=None, **kw)
                    outs += [_row(T.Outcome("ok", T.normalize(v))) for v in batch]
            else:
                outs = [_row(T.capture_call(candidate.extract, h, **kw)) for h in htmls]
            cand_s += time.perf_counter() - t1
            second[o] = outs
    for o in sets:
        for k, ((doc_id, url, _), a, b) in enumerate(zip(pages, first[o], second[o], strict=True)):
            same = a[0] == b[0]
            ex = None if same else {**where, "page": k, "id": doc_id, "url": url, "first": a, "second": b}
            tally.add(o, same, a[2], ex)
    if "datatrove" in first and "precision" in first:
        effect = sum(1 for a, b in zip(first["datatrove"], first["precision"], strict=True) if a[0] != b[0])
        t = tally.data.setdefault("_dedup_effect", {"pages": 0, "differs": 0})
        t["pages"] += n
        t["differs"] += effect
    tally.data.setdefault("_seconds", {"oracle_wall": 0.0, "candidate_wall": 0.0})
    tally.data["_seconds"]["oracle_wall"] += oracle_s
    tally.data["_seconds"]["candidate_wall"] += cand_s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.trafilatura_scale", description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--impl", help="candidate module (default: trafilatura against itself)")
    ap.add_argument("--pages", type=int, default=20_000, help="pages in all, split evenly over the crawls")
    ap.add_argument("--crawls", default=",".join(DEFAULT_CRAWLS))
    ap.add_argument("--slice-mb", type=float, default=40.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument(
        "--compat",
        choices=specs.TRAFILATURA_MODES,
        default=os.environ.get(specs.TRAFILATURA_COMPAT_ENV) or specs.TRAFILATURA_VERSION,
        help="the trafilatura release checked (default: $PYROID_TRAFILATURA_COMPAT, else 2.3.1)",
    )
    a = ap.parse_args(argv)
    mode = a.compat
    import importlib.metadata as md

    for dist, want in specs.TRAFILATURA_ORACLES[mode].items():
        if (got := md.version(dist)) != want:
            group = specs.TRAFILATURA_GROUPS[mode]
            sys.exit(f"--compat {mode} needs {dist}=={want} (dependency group {group}), found {got}")
    os.environ[specs.TRAFILATURA_COMPAT_ENV] = mode  # for impl.resolve and the workers
    a.out.mkdir(parents=True, exist_ok=True)
    state_path = a.out / "state.json"
    state = json.loads(state_path.read_text("utf-8")) if state_path.exists() else {"slices": [], "mode": mode}
    if state.get("mode", specs.TRAFILATURA_VERSION) != mode:
        sys.exit(f"{a.out} holds a run of mode {state.get('mode')}: use another OUT_DIR for {mode}")
    done = {(s["crawl"], s["warc"], s["start"]) for s in state["slices"]}

    candidate = None
    if a.impl:
        from pyroid_parity import impl as I

        candidate = I.resolve("trafilatura", a.impl).obj
    crawls = a.crawls.split(",")
    per_crawl = -(-a.pages // len(crawls))
    pool = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("forkserver"))
    pages_of = CC.Pages()
    summary: dict[str, Any] = {}
    try:
        for crawl in crawls:
            path = a.out / f"{crawl}.json"
            result = json.loads(path.read_text("utf-8")) if path.exists() else {"crawl": crawl, "slices": []}
            tally = Tally()
            tally.data = result.get("options", {})
            rng = random.Random(f"{a.seed}:{crawl}")
            paths = warc_paths(crawl, a.out)
            have = sum(s["pages"] for s in result["slices"])
            while have < per_crawl:
                warc, offset = plan_slice(crawl, paths, rng, int(a.slice_mb * 1e6))
                if (crawl, warc, offset) in done:  # resumed: the RNG replays the same plan
                    continue
                s, raw = CC.align(crawl, warc, offset, int(a.slice_mb * 1e6))
                pages = list(pages_of(raw))
                del raw
                where = {"crawl": crawl, "warc": warc, "range": [s.start, s.end]}
                log(f"{crawl}: {len(pages)} pages from [{s.start:,}, {s.end:,}) of {warc.rsplit('/', 1)[1]}")
                compare_slice(pool, candidate, pages, where, tally, mode)
                rec = {
                    "crawl": crawl,
                    "warc": warc,
                    "start": offset,
                    "range": [s.start, s.end],
                    "sha256": s.sha256,
                }
                rec["pages"] = len(pages)
                result["slices"].append(rec)
                state["slices"].append(rec)
                have += len(pages)
                del pages
                result["options"] = tally.data
                path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", "utf-8")
                state_path.write_text(json.dumps(state, indent=1) + "\n", "utf-8")
                diffs = {o: t["diff"] for o, t in tally.data.items() if not o.startswith("_")}
                log(f"{crawl}: {have} pages so far; differences per option set: {diffs}")
            summary[crawl] = {
                "pages": have,
                "slices": len(result["slices"]),
                **{
                    o: {k: t[k] for k in ("same", "diff")}
                    for o, t in result["options"].items()
                    if "diff" in t
                },
                "dedup_effect": result["options"].get("_dedup_effect"),
            }
    finally:
        pool.shutdown(cancel_futures=True)
    total = {"pages": sum(v["pages"] for v in summary.values())}
    total["diff"] = sum(
        t["diff"] for v in summary.values() for t in v.values() if isinstance(t, dict) and "diff" in t
    )
    out = {
        "check": "candidate" if a.impl else "reference self-check (trafilatura twice)",
        "compat": mode,
        "candidate": a.impl,
        "crawls": summary,
        "total": total,
    }
    (a.out / "summary.json").write_text(json.dumps(out, indent=1) + "\n", "utf-8")
    log(f"done: {total}")
    return 1 if total["diff"] else 0


if __name__ == "__main__":
    sys.exit(main())
