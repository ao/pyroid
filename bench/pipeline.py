#!/usr/bin/env python3
"""End-to-end FineWeb-style cleaning on N worker processes (the end-to-end benchmark behind the README hero), with an equivalence digest.

    cd bench && uv run --frozen python pipeline.py --corpus dev --docs 500 --workers 4 --out /target/pipe/w4
    ... pipeline.py --stages lid --corpus parity --workers 8 --check-refs --out /target/pipe/lid   # one stage vs refs
    ... pipeline.py --corpus headline --workers 8 --no-digest --summary-log runs.jsonl --out ...  # hyperfine timing

Chain (datatrove ``examples/fineweb.py`` after text extraction, with ftfy in front):

    fix_text           ftfy.fix_text(text), ftfy's DEFAULT config (= parity/ftfy-ref). datatrove's own
                       FTFYFormatter uses a gentler non-default TextFixerConfig, so it is not used here.
    lid                LanguageFilter(languages=["en"], language_threshold=0.65, backend="ft176")
                       (``gates.PinnedLanguageFilter``: upstream code + 3 documented shims for fasttext-predict)
    gopher_repetition  GopherRepetitionFilter()
    gopher_quality     GopherQualityFilter()   (its defaults are fineweb.py's 50 / 100000 / 2 stop words)
    c4_quality         C4QualityFilter(filter_no_terminal_punct=False)   (rewrites text: drops lines)
    fineweb_quality    FineWebQualityFilter()
    minhash            MinhashDedupSignature -> MinhashDedupBuckets -> MinhashDedupCluster -> MinhashDedupFilter,
                       config = specs.MINHASH_CONFIG (sha1/64, 14 buckets x 8 hashes, word 5-grams)

Executor: every phase is datatrove's own ``LocalPipelineExecutor`` (tasks = workers = N, as fineweb.py chains
them): [read -> per-doc stages -> ParquetWriter(kept)] on N tasks; MinHash signatures on N tasks, buckets on
14 tasks, cluster on 1, filter on N. Our code is only (a) ``ShardReader`` (pinned parquet -> contiguous
rank shards, so (rank, index) order == corpus order and MinHash's "which copy survives" does not depend on
N), (b) the ftfy formatter, (c) writer/reader adapters with a fixed schema, (d) an RSS probe step, and
(e) collecting outputs into the digest. Removed docs go to each filter's ``exclusion_writer`` (datatrove's
mechanism; fineweb.py does the same), so every input doc ends up in exactly one output folder.

``start_method="fork"`` instead of datatrove's default "forkserver": forkserver workers are grandchildren
reaped by the forkserver, so their CPU time would be invisible to getrusage(RUSAGE_CHILDREN) and to hyperfine's
``time_cpu``. Writers are uncompressed parquet (fineweb.py writes gzip JSONL) to keep IO out of the comparison.

Summary JSON (stdout, ``OUT/summary.json``, and one line appended to ``--summary-log``): wall seconds per
phase, CPU seconds (getrusage self + children, user + sys), peak RSS (parent and largest child maxrss, the sum
of per-worker maxrss as an upper bound on the simultaneous total, and cgroup v2 memory.current / anon sampled
every ``--mem-interval`` s; ``memory.peak`` is container-lifetime and read-only in Docker, so it is only
reported at start and end), docs in/kept/removed per stage, and a sha256 ``digest`` over one JSON line per doc
in corpus order: ``[id, text, lang, score (%.6f), keep, reason, dup]`` (``digest_exact`` uses float.hex).
``text`` is the text as it leaves the pipeline (after ftfy and C4's line removal); ``reason`` is
``"<stage>:<datatrove reason>"`` (``"lid"`` has none); ``dup`` is None when MinHash did not run or the doc was
removed earlier. Equal digests across ``--workers`` (or impls) = identical per-doc results.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import resource
import shutil
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any

T_START = time.perf_counter()

import ftfy
from datatrove.data import Document
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.formatters.base import BaseFormatter
from pyroid_parity import impl as I
from pyroid_parity import refs as R
from pyroid_parity import specs

import gates as G

STAGES = (
    "fix_text",
    "lid",
    "gopher_repetition",
    "gopher_quality",
    "c4_quality",
    "fineweb_quality",
    "minhash",
)
FILTER_CASE = {
    "gopher_repetition": "gopher_repetition",
    "gopher_quality": "gopher_quality",
    "c4_quality": "c4_quality@fineweb",
    "fineweb_quality": "fineweb_quality",
}
SUITE = {"fix_text": "ftfy", "lid": "fasttext", "minhash": "dedup", **dict.fromkeys(FILTER_CASE, "filters")}
CGROUP = Path("/sys/fs/cgroup")
MIB = 1 << 20


# ------------------------------------------------------------------------------------------ pipeline steps


class ShardReader(PipelineStep):
    """Rows [rank * n / W, (rank + 1) * n / W) of the pinned corpus (empty texts kept, unlike datatrove readers)."""

    type = "READER"
    name = "pinned corpus, contiguous shards"

    def __init__(self, segments: list[tuple[str, int]], n_docs: int) -> None:
        super().__init__()
        self.segments = [(str(p), n) for p, n in segments]
        self.n_docs = n_docs

    def run(self, data: Any = None, rank: int = 0, world_size: int = 1) -> Any:
        lo, hi = rank * self.n_docs // world_size, (rank + 1) * self.n_docs // world_size
        for doc_id, text in G.iter_slice(self.segments, lo, hi):
            self.stat_update("documents")
            yield Document(text=text, id=doc_id)


class FixText(BaseFormatter):
    name = "ftfy.fix_text (default config)"

    def format(self, text: str) -> str:
        return ftfy.fix_text(text)


class CandidateFixText(PipelineStep):
    type = "FORMAT"
    name = "candidate fix_text_batch"

    def __init__(self, module: str, batch: int = 256) -> None:
        super().__init__()
        self.module, self.batch = module, batch

    def run(self, data: Any, rank: int = 0, world_size: int = 1) -> Any:
        obj = I.resolve("ftfy", self.module).obj
        for chunk in itertools.batched(data, self.batch):
            with self.track_time("batch"):
                out = obj.fix_text_batch([d.text for d in chunk])
            for d, t in zip(chunk, out, strict=True):
                d.text = t
                yield d


class CandidateLID(BaseFilter):
    """LanguageFilter(en, 0.65) semantics on a candidate fastText: keep iff top-1 == en and p > 0.65 (equivalent
    to datatrove's P(en) > 0.65 over all labels, since probabilities sum to <= 1)."""

    name = "candidate LID"

    def __init__(self, module: str, exclusion_writer: Any = None) -> None:
        super().__init__(exclusion_writer)
        self.module = module
        self._o: Any = None

    def filter(self, doc: Document) -> bool:
        if self._o is None:
            o = I.resolve("fasttext", self.module).obj
            self._o = (o, o.load(G.lid_model_path("lid.176.bin")))
        o, model = self._o
        label, score = o.predict(model, doc.text.replace("\n", " "), 1)[0]
        doc.metadata["language"] = label.removeprefix("__label__")
        doc.metadata["language_score"] = float(score)
        return doc.metadata["language"] == "en" and float(score) > G.LID_FILTER["language_threshold"]


class CandidateFilter(BaseFilter):
    name = "candidate filter"

    def __init__(self, module: str, case: str, exclusion_writer: Any = None, batch_size: int = 256) -> None:
        super().__init__(exclusion_writer, batch_size)
        self.module, self.case = module, case
        self._o: Any = None

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        return self.filter_batch([doc])[0]

    def filter_batch(self, batch: list[Document]) -> list[bool | tuple[bool, str]]:
        if self._o is None:
            self._o = I.resolve("filters", self.module).obj
        fn, _, cfg = specs.FILTER_CASES[self.case]
        out: list[bool | tuple[bool, str]] = []
        for doc, dec in zip(batch, self._o.run(fn, [d.text for d in batch], cfg), strict=True):
            if dec[0] != "ok":
                raise RuntimeError(f"{self.module}.{fn} raised {dec[1]} on doc {doc.id}")
            _, keep, reason, text_out = dec
            if text_out is not None:
                doc.text = text_out
            out.append(True if keep else ((False, reason) if reason else False))
        return out


class RssProbe(PipelineStep):
    """Last step of every phase: records this worker's maxrss after its task."""

    type = "PROBE"
    name = "rss probe"

    def __init__(self, phase: str, folder: Path) -> None:
        super().__init__()
        self.phase, self.folder = phase, str(folder)

    def run(self, data: Any = None, rank: int = 0, world_size: int = 1) -> Any:
        if data:
            yield from data
        rec = {
            "phase": self.phase,
            "rank": rank,
            "pid": os.getpid(),
            "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        (Path(self.folder) / f"{self.phase}-{rank:05d}.json").write_text(json.dumps(rec))


def record_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("id", pa.string()),
            ("text", pa.string()),
            ("lang", pa.string()),
            ("score", pa.float64()),
            ("reason", pa.string()),
        ]
    )


def to_record(_writer: Any, doc: Document) -> dict[str, Any]:
    md = doc.metadata
    return {
        "id": doc.id,
        "text": doc.text,
        "lang": md.get("language"),
        "score": md.get("language_score"),
        "reason": md.get("filter_reason"),
    }


def from_record(_reader: Any, data: dict[str, Any], _path: str, _id_in_file: Any) -> dict[str, Any]:
    md = {
        k: data[c] for k, c in (("language", "lang"), ("language_score", "score")) if data.get(c) is not None
    }
    return {"text": data["text"], "id": data["id"], "metadata": md}


def writer(folder: Path) -> Any:
    from datatrove.pipeline.writers import ParquetWriter

    return ParquetWriter(
        str(folder),
        output_filename="${rank}.parquet",
        compression=None,
        adapter=to_record,
        schema=record_schema(),
        max_file_size=-1,
        use_content_defined_chunking=False,
        write_page_index=False,
    )


def reader(folder: Path) -> Any:
    from datatrove.pipeline.readers import ParquetReader

    return ParquetReader(str(folder), adapter=from_record, glob_pattern="*.parquet")


# ------------------------------------------------------------------------------------------ measurement


def _cgroup_int(name: str) -> int | None:
    try:
        return int((CGROUP / name).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _cgroup_stat(key: str) -> int | None:
    try:
        for line in (CGROUP / "memory.stat").read_text().splitlines():
            k, _, v = line.partition(" ")
            if k == key:
                return int(v)
    except OSError:
        return None
    return None


def _sampler(conn: Any, interval: float) -> None:
    cur = anon = n = 0
    while not conn.poll(interval):
        cur = max(cur, _cgroup_int("memory.current") or 0)
        anon = max(anon, _cgroup_stat("anon") or 0)
        n += 1
    conn.send({"samples": n, "max_current": cur, "max_anon": anon})


class Sampler:
    """cgroup v2 memory.current / anon high-water marks, sampled in a forked process (no thread in the parent:
    the executor forks workers, and fork with live threads is unsafe; a process also leaves the GIL alone)."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.proc: Any = None
        if interval > 0 and (CGROUP / "memory.current").exists():
            ctx = mp.get_context("fork")
            self.conn, child = ctx.Pipe()
            self.proc = ctx.Process(target=_sampler, args=(child, interval), daemon=True)
            self.proc.start()

    def stop(self) -> dict[str, Any]:
        if self.proc is None:
            return {"samples": 0}
        self.conn.send("stop")
        res = self.conn.recv()
        self.proc.join()
        return {
            "interval_s": self.interval,
            "samples": res["samples"],
            "sampled_max_current_mib": round(res["max_current"] / MIB, 1),
            "sampled_max_anon_mib": round(res["max_anon"] / MIB, 1),
        }


def rusage() -> dict[str, float]:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "self_user": s.ru_utime,
        "self_sys": s.ru_stime,
        "children_user": c.ru_utime,
        "children_sys": c.ru_stime,
        "total": round(s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime, 3),
        "self_maxrss_mib": round(s.ru_maxrss / 1024, 1),
        "children_maxrss_mib": round(c.ru_maxrss / 1024, 1),
    }


def rss_upper_bound(folder: Path, parent_mib: float) -> dict[str, Any]:
    """Per phase: sum over distinct worker pids of their maxrss (+ the parent's if it was not a worker)."""
    per_phase: dict[str, dict[int, int]] = {}
    for f in folder.glob("*.json"):
        r = json.loads(f.read_text())
        pids = per_phase.setdefault(r["phase"], {})
        pids[r["pid"]] = max(pids.get(r["pid"], 0), r["maxrss_kib"])
    out = {}
    for phase, pids in per_phase.items():
        total = sum(pids.values()) / 1024 + (0 if os.getpid() in pids else parent_mib)
        out[phase] = {"workers": len(pids), "sum_maxrss_mib": round(total, 1)}
    return {
        "per_phase": out,
        "upper_bound_mib": max((v["sum_maxrss_mib"] for v in out.values()), default=0.0),
    }


# ---------------------------------------------------------------------------------------------- running


def make_step(stage: str, module: str, native: bool, work: Path) -> Any:
    excl = writer(work / "excluded" / stage) if stage != "fix_text" else None
    if stage == "fix_text":
        return FixText() if native else CandidateFixText(module)
    if stage == "lid":
        return (
            G.language_filter(exclusion_writer=excl)
            if native
            else CandidateLID(module, exclusion_writer=excl)
        )
    case = FILTER_CASE[stage]
    if native:
        from datatrove.pipeline import filters as F

        _, cls, cfg = specs.FILTER_CASES[case]
        return getattr(F, cls)(**cfg, exclusion_writer=excl)
    return CandidateFilter(module, case, exclusion_writer=excl)


def run_phase(
    name: str, steps: list[Any], tasks: int, workers: int, work: Path, phases: dict[str, Any]
) -> None:
    t = time.perf_counter()
    LocalPipelineExecutor(
        pipeline=[*steps, RssProbe(name, work / "rss")],
        tasks=tasks,
        workers=min(workers, tasks),
        logging_dir=str(work / "logs" / name),
        skip_completed=False,
        start_method="fork",
    ).run()
    phases[name] = {
        "seconds": round(time.perf_counter() - t, 3),
        "tasks": tasks,
        "workers": min(workers, tasks),
    }


def rows_in(folder: Path) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(folder.glob("*.parquet")))


def read_records(folder: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return [r for f in sorted(folder.glob("*.parquet")) for r in pq.read_table(f).to_pylist()]


def write_records(folder: Path, recs: list[dict[str, Any]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(recs, schema=record_schema()), folder / "00000.parquet", compression="none"
    )


def candidate_dedup(module: str, kept: list[dict[str, Any]], work: Path, phases: dict[str, Any]) -> None:
    """Candidate dedup contract is whole-corpus (``duplicates(texts) -> mask``): run it in this process."""
    t = time.perf_counter()
    mask = I.resolve("dedup", module).obj.duplicates([r["text"] for r in kept], specs.MINHASH_CONFIG)
    write_records(work / "final", [r for r, d in zip(kept, mask, strict=True) if not d])
    write_records(work / "dup", [r for r, d in zip(kept, mask, strict=True) if d])
    phases["minhash_candidate"] = {"seconds": round(time.perf_counter() - t, 3), "tasks": 1, "workers": 1}


def run_pipeline(
    a: argparse.Namespace,
    stages: list[str],
    chosen: dict[str, str],
    native: dict[str, bool],
    segs: list[tuple[Path, int]],
    n: int,
    work: Path,
) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    per_doc = [s for s in stages if s != "minhash"]
    shard = ShardReader(segs, n)
    for d in ("kept", "final", "dup", *(f"excluded/{s}" for s in per_doc)):
        (work / d).mkdir(parents=True, exist_ok=True)
    if per_doc:
        steps = [shard, *(make_step(s, chosen[SUITE[s]], native[SUITE[s]], work) for s in per_doc)]
        run_phase("filters", [*steps, writer(work / "kept")], a.workers, a.workers, work, phases)
    if "minhash" not in stages:
        return phases
    have_input = not per_doc or any((work / "kept").glob("*.parquet"))
    if not have_input:
        return phases
    if not native["dedup"]:
        kept = (
            read_records(work / "kept")
            if per_doc
            else [
                {"id": i, "text": t, "lang": None, "score": None, "reason": None}
                for i, t in G.iter_slice(segs, 0, n)
            ]
        )
        candidate_dedup(chosen["dedup"], kept, work, phases)
        return phases
    from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import MinhashDedupBuckets

    cfg = R.minhash_config()
    src = (lambda: reader(work / "kept")) if per_doc else (lambda: ShardReader(segs, n))
    mh = work / "minhash"
    run_phase(
        "minhash_signature",
        [src(), MinhashDedupSignature(output_folder=str(mh / "sigs"), config=cfg)],
        a.workers,
        a.workers,
        work,
        phases,
    )
    run_phase(
        "minhash_buckets",
        [MinhashDedupBuckets(input_folder=str(mh / "sigs"), output_folder=str(mh / "buckets"), config=cfg)],
        cfg.num_buckets,
        a.workers,
        work,
        phases,
    )
    run_phase(
        "minhash_cluster",
        [
            MinhashDedupCluster(
                input_folder=str(mh / "buckets"), output_folder=str(mh / "remove_ids"), config=cfg
            )
        ],
        1,
        1,
        work,
        phases,
    )
    run_phase(
        "minhash_filter",
        [
            src(),
            MinhashDedupFilter(input_folder=str(mh / "remove_ids"), exclusion_writer=writer(work / "dup")),
            writer(work / "final"),
        ],
        a.workers,
        a.workers,
        work,
        phases,
    )
    return phases


def counts(work: Path, stages: list[str], n: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cur = n
    for s in stages:
        if s == "fix_text":
            out[s] = {"in": cur, "kept": cur, "removed": 0}
        elif s == "minhash":
            final, dup = rows_in(work / "final"), rows_in(work / "dup")
            out[s] = {"in": final + dup, "kept": final, "removed": dup, "reader_skipped": cur - final - dup}
            cur = final
        else:
            removed = rows_in(work / "excluded" / s)
            out[s] = {"in": cur, "kept": cur - removed, "removed": removed}
            cur -= removed
    return {"per_stage": out, "docs_in": n, "docs_out": cur}


def collect(work: Path, stages: list[str], ids: list[str]) -> dict[str, dict[str, Any]]:
    recs: dict[str, dict[str, Any]] = {}

    def put(r: dict[str, Any], keep: bool, reason: str | None, dup: bool | None) -> None:
        if r["id"] in recs:
            raise RuntimeError(f"doc {r['id']} appears in two outputs")
        recs[r["id"]] = {**r, "keep": keep, "reason": reason, "dup": dup}

    for s in stages:
        if s not in ("fix_text", "minhash"):
            for r in read_records(work / "excluded" / s):
                put(r, False, f"{s}:{r['reason']}" if r["reason"] else s, None)
    if "minhash" in stages:
        for r in read_records(work / "final"):
            put(r, True, None, False)
        for r in read_records(work / "dup"):
            put(r, True, None, True)
    else:
        for r in read_records(work / "kept"):
            put(r, True, None, None)
    for i in ids:
        if i not in recs:  # e.g. an empty text kept by every filter is skipped by datatrove's readers
            recs[i] = {
                "id": i,
                "text": None,
                "lang": None,
                "score": None,
                "keep": None,
                "reason": "missing",
                "dup": None,
            }
    return recs


def digests(recs: dict[str, dict[str, Any]], ids: list[str]) -> tuple[str, str]:
    h6, hx = hashlib.sha256(), hashlib.sha256()
    for i in ids:
        r = recs[i]
        s = r["score"]
        for h, sc in ((h6, None if s is None else f"{s:.6f}"), (hx, None if s is None else float(s).hex())):
            row = [i, r["text"], r["lang"], sc, r["keep"], r["reason"], r["dup"]]
            h.update(json.dumps(row, ensure_ascii=False).encode("utf-8", "surrogatepass") + b"\n")
    return h6.hexdigest(), hx.hexdigest()


def check_refs(
    recs: dict[str, dict[str, Any]], ids: list[str], raws: list[str], stages: list[str], full_parity: bool
) -> dict[str, Any]:
    """Per-doc decisions vs the pinned refs wherever the refs determine them: refs were computed per stage on
    the RAW text, so a doc is checked up to the first stage whose input is no longer the raw text (ftfy
    changed it, or C4 rewrote it); the rest of its chain is counted as ``undetermined``. MinHash removals are
    checked only for ``--stages minhash`` on the full parity corpus (the refs dedup dev+adversarial jointly)."""
    fixed, lid, filt = G.ftfy_ref(), G.lid_ref(), G.filters_ref()
    removed = G.minhash_ref()[1] if (stages == ["minhash"] and full_parity) else None
    tol = G.LID_TOL
    st = {
        "docs": len(ids),
        "fully_checked": 0,
        "partially_checked": 0,
        "no_ref": 0,
        "mismatches": 0,
        "examples": [],
    }

    def bad(i: str, what: str, got: Any, want: Any) -> None:
        st["mismatches"] += 1
        if len(st["examples"]) < 5:
            st["examples"].append({"id": i, "what": what, "got": G._short(got), "want": G._short(want)})

    for i, raw in zip(ids, raws, strict=True):
        r = recs[i]
        if i not in fixed:
            st["no_ref"] += 1
            continue
        text, raw_ok, determined, ended = raw, True, True, False
        for s in stages:
            if s == "minhash":
                break
            if s == "fix_text":
                text = fixed[i]
                raw_ok = text == raw
                continue
            if not raw_ok:
                determined = False
                break
            if s == "lid":
                keep, lang, score = G.lid_filter_expected(*lid[("lid.176.bin", i)])
                if r["lang"] != lang or r["score"] is None or abs(r["score"] - score) > tol:
                    bad(i, "lid language/score", (r["lang"], r["score"]), (lang, score))
                if not keep:
                    ended = True
                    if r["reason"] != "lid" or r["text"] != text:
                        bad(i, "lid removal", (r["reason"], r["keep"]), ("lid", False))
                    break
                continue
            dec = filt[(FILTER_CASE[s], i)]
            if dec[0] != "ok":
                determined = False
                break
            _, keep, reason, text_out = dec
            if text_out is not None:
                text, raw_ok = text_out, False
            if not keep:
                ended = True
                want = f"{s}:{reason}" if reason else s
                if r["reason"] != want or r["text"] != text:
                    bad(i, f"{s} removal", (r["reason"], _sha(r["text"])), (want, _sha(text)))
                break
        if determined and not ended:
            if r["keep"] is not True or r["text"] != text:
                bad(i, "survivor", (r["keep"], r["reason"], _sha(r["text"])), (True, None, _sha(text)))
            if removed is not None and r["dup"] != removed[i]:
                bad(i, "minhash removed", r["dup"], removed[i])
        st["fully_checked" if determined else "partially_checked"] += 1
    return st


def _sha(t: str | None) -> str | None:
    return None if t is None else hashlib.sha256(t.encode("utf-8", "surrogatepass")).hexdigest()[:12]


def parse_stages(spec: str) -> list[str]:
    if spec == "all":
        return list(STAGES)
    want = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [s for s in want if s not in STAGES]
    if unknown or not want:
        raise SystemExit(f"--stages: unknown {unknown}; choose from {', '.join(STAGES)} or 'all'")
    return [s for s in STAGES if s in want]  # always chain order


def versions() -> dict[str, str | None]:
    return {
        d: I.version_of(m)
        for d, m in (
            ("ftfy", "ftfy"),
            ("datatrove", "datatrove"),
            ("spacy", "spacy"),
            ("fasttext-predict", "fasttext"),
            ("numpy", "numpy"),
            ("pyarrow", "pyarrow"),
        )
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="dev", choices=sorted(G.SEGMENTS))
    ap.add_argument("--docs", type=int, default=0, help="first N docs of the corpus (0 = all)")
    ap.add_argument("--workers", type=int, default=1, help="worker processes (tasks = workers)")
    ap.add_argument(
        "--stages", default="all", help=f"comma subset of {','.join(STAGES)} (run in chain order)"
    )
    ap.add_argument(
        "--impl",
        action="append",
        default=[],
        metavar="[SUITE=]MODULE",
        help="datatrove (default: all incumbents) or SUITE=MODULE candidates via pyroid_parity.impl",
    )
    ap.add_argument("--out", required=True, help="output dir: summary.json + work/ (cleared at start)")
    ap.add_argument("--no-digest", action="store_true", help="skip collecting outputs (timing runs)")
    ap.add_argument(
        "--check-refs", action="store_true", help="compare per-doc decisions with the pinned refs"
    )
    ap.add_argument("--summary-log", help="append the summary as one JSON line to this file")
    ap.add_argument(
        "--label", default="", help="free-form tag stored in the summary (e.g. the hyperfine -n name)"
    )
    ap.add_argument("--mem-interval", type=float, default=0.1, help="cgroup sampling interval s (0 = off)")
    ap.add_argument("--keep-work", action="store_true", help="keep OUT/work (default: deleted after the run)")
    ap.add_argument("--quiet", action="store_true", help="send datatrove's console logs to OUT/pipeline.log")
    a = ap.parse_args()
    if a.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if sysconfig.get_config_var("Py_GIL_DISABLED") and sys._is_gil_enabled():
        raise SystemExit("free-threaded interpreter with the GIL re-enabled by an extension: results invalid")
    stages = parse_stages(a.stages)
    chosen = I.parse_specs(a.impl)
    native = {s: chosen[s] == I.REFERENCE[s] for s in I.SUITES}
    out = Path(a.out)
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    segs = G.segment_rows(a.corpus, verify=True)  # pinned sha256, parent only
    total = sum(k for _, k in segs)
    n = a.docs or total
    if n > total:
        raise SystemExit(f"--docs {n}: corpus {a.corpus!r} has {total} docs")

    mem_start = _cgroup_int("memory.peak")
    sampler = Sampler(a.mem_interval)
    log_fd = None
    if a.quiet:
        log_fd = os.dup(2)
        with open(out / "pipeline.log", "w") as f:
            os.dup2(f.fileno(), 2)
    t0 = time.perf_counter()
    try:
        phases = run_pipeline(a, stages, chosen, native, segs, n, work)
    finally:
        if log_fd is not None:
            os.dup2(log_fd, 2)
            os.close(log_fd)
    wall_pipeline = time.perf_counter() - t0
    cpu_pipeline = rusage()
    mem = sampler.stop()
    summary: dict[str, Any] = {
        "tool": "bench/pipeline.py",
        "label": a.label,
        "corpus": a.corpus,
        "docs": n,
        "workers": a.workers,
        "stages": stages,
        "impl": {
            s: {"module": chosen[s], "native_datatrove_path": native[s], "version": I.version_of(chosen[s])}
            for s in sorted({SUITE[x] for x in stages})
        },
        "executor": "datatrove LocalPipelineExecutor(tasks=workers, workers, start_method='fork')",
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version.split()[0],
        "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
        "versions": versions(),
        "wall_s": {"pipeline": round(wall_pipeline, 3), "startup_imports": round(t0 - T_START, 3)},
        "phases": phases,
        "cpu_s": cpu_pipeline,
        "rss": rss_upper_bound(work / "rss", cpu_pipeline["self_maxrss_mib"]),
        "cgroup": {
            "memory_peak_start_mib": _mib(mem_start),
            "memory_peak_end_mib": _mib(_cgroup_int("memory.peak")),
            "anon_end_mib": _mib(_cgroup_stat("anon")),
            **mem,
            "note": "memory.peak is the container-lifetime high-water mark (read-only, not resettable)",
        },
        "counts": counts(work, stages, n),
    }
    if not a.no_digest or a.check_refs:
        t = time.perf_counter()
        ids, raws = [], []
        for i, txt in G.iter_slice(segs, 0, n):
            ids.append(i)
            raws.append(txt)
        recs = collect(work, stages, ids)
        summary["digest"], summary["digest_exact"] = digests(recs, ids)
        reasons: dict[str, int] = {}
        for r in recs.values():
            if r["reason"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        summary["reasons"] = dict(sorted(reasons.items()))
        summary["missing"] = reasons.get("missing", 0)
        if a.check_refs:
            summary["check_refs"] = check_refs(recs, ids, raws, stages, a.corpus == "parity" and n == total)
        summary["wall_s"]["collect"] = round(time.perf_counter() - t, 3)
    summary["wall_s"]["total"] = round(time.perf_counter() - T_START, 3)
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    text = json.dumps(summary, indent=1, sort_keys=True)
    (out / "summary.json").write_text(text + "\n", "utf-8")
    if a.summary_log:
        with open(a.summary_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
    print(text)
    cr = summary.get("check_refs")
    if cr and cr["mismatches"]:
        print(f"CHECK-REFS FAILED: {cr['mismatches']} mismatches, e.g. {cr['examples'][:2]}", file=sys.stderr)
        return 1
    if summary.get("missing"):
        print(f"WARNING: {summary['missing']} docs are in no output folder", file=sys.stderr)
    return 0


def _mib(v: int | None) -> float | None:
    return None if v is None else round(v / MIB, 1)


if __name__ == "__main__":
    import pipeline  # re-import under its module name so dill pickles the steps by reference for workers

    pipeline.T_START = T_START  # count interpreter start + imports from the first execution
    sys.exit(pipeline.main())
