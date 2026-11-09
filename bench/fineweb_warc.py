#!/usr/bin/env python3
"""datatrove's FineWeb pipeline from WARC files, end to end, four ways, with digests of the documents kept.

    python fineweb_warc.py stock        --warcs DIR --out OUT --workers N [--lid-model PATH] [--limit N]
    python fineweb_warc.py pyroid-steps --warcs DIR --out OUT --workers N --lid-model PATH [--limit N]
    python fineweb_warc.py native       --warcs DIR --out OUT --workers N --lid-model PATH [--limit N]
                                        [--timeouts-of RUN] [--reader pyroid|warcio]
    python fineweb_warc.py cli          --warcs DIR --out OUT --workers N --lid-model PATH [--limit N]
    python fineweb_warc.py reader       --warcs DIR --out OUT [--limit N] [--readers warcio,documents,batches]
    python fineweb_warc.py compare RUN [RUN ...] [--json FILE]

The pipeline is datatrove 0.10.1's ``examples/fineweb.py``: WarcReader -> URLFilter ->
Trafilatura(favour_precision=True) -> LanguageFilter(languages=["en"]) -> GopherRepetitionFilter ->
GopherQualityFilter -> C4QualityFilter(filter_no_terminal_punct=False) -> FineWebQualityFilter -> JsonlWriter,
each filter with fineweb.py's exclusion writer (gzip JSONL); then its four MinHash stages (sha1/64, 14 buckets x 8
hashes, 5-grams): signatures, buckets, clusters, and JsonlReader -> MinhashDedupFilter -> PIIFormatter ->
JsonlWriter. TokensCounter is left out (fineweb.py marks it optional). LanguageFilter gets languages=["en"]:
FineWeb kept English, and 0.10.1's default keeps every language.

Modes:

stock
    datatrove's own steps on ``LocalPipelineExecutor(tasks=<WARC files>, workers=N, start_method="fork")``. Run it
    in an env with ``datatrove[io,processing]==0.10.1`` as pip resolves it (trafilatura 1.11.0,
    fasttext-numpy2-wheel) plus spaCy (datatrove's English word tokenizer). Without ``--lid-model`` this is
    datatrove's LanguageFilter unchanged; with it, fastText is fasttext-predict on that model file (the shim of
    ``bench/gates.py``'s PinnedLanguageFilter, for machines that cannot build fasttext-numpy2-wheel).
pyroid-steps
    The same executor and stages, with pyroid.integrations.datatrove's drop-ins for URLFilter, Trafilatura
    (``compat=None``: the installed trafilatura's release), LanguageFilter, the four quality filters,
    MinhashDedupSignature and PIIFormatter; datatrove's reader, writers, MinhashDedupBuckets, MinhashDedupCluster
    and MinhashDedupFilter. A pyroid.FallbackWarning is an error here.
native
    No datatrove executor. A pool of N processes, one task per WARC file: the documents of
    ``pyroid.warc.iter_documents`` (``--reader warcio``: warcio over ``gzip.open`` and datatrove's own
    ``process_record``, the path datatrove takes), then the pieces of ``pyroid.integrations.recipes.FineWebWarc``:
    ``pyroid.filters.URLFilter``, ``TaskExtractor`` (``pyroid.trafilatura.extract_batch``, ``compat="1.11.0"``,
    favor_precision=True, include_comments=False, the dedup cache emptied per file) and ``FineWeb(ftfy_config=False)``
    for LID and the filters, writing the same kept and excluded files. Then, in this process, MinHash dedup over
    every file in file (rank) order, and PII replacement with a fresh replacer per file, as datatrove's per-task step.
cli
    ``pyroid clean FILE... OUT/cli.parquet --recipe fineweb-warc --workers N`` on the same files, in rank order. Its
    output is the final documents (after PII), so this mode has only the ``after_pii`` digest; ``compare`` checks the
    digests both runs have.
reader
    The reader alone, one file after another in this process: warcio over ``gzip.open`` + datatrove's
    ``process_record`` (``warcio``), ``pyroid.warc.iter_documents`` (``documents``) and ``pyroid.warc.iter_batches``
    (``batches``, the rows with a text). Per file and reader: wall and CPU seconds, documents, and a digest of every
    document's (id, url, date, text), which must be the same for every reader.

Every pipeline mode runs one task per WARC file, so the dedup cache, the MinHash order and the PII rotation are the same for
any N, and the digests must be too.

OUT/summary.json: wall seconds per phase and in total, CPU seconds (getrusage, self + children), peak RSS, seconds
per step summed over tasks, documents out of each stage, Trafilatura's timeouts and broken processes, and two
digests, sha256 over ``id \\0 text \\0`` (UTF-8) of the documents sorted by id: as they leave MinhashDedupFilter and
as they leave PIIFormatter. OUT/docs.tsv.gz has one line per final document (id, rank, 16 hex digits of the sha256
of each text) to locate differences. Seconds per step are wall time inside the step minus the time spent in the
steps before it (the ``Timed`` wrapper of P10's research/htmlx/py/fineweb_share.py, branch feat/htmlx); in native
mode, time per stage within each task.

datatrove's Trafilatura step runs extract() in a child process with a 1 s timeout per page: a page over it is
dropped and the child restarted, which also empties trafilatura's dedup cache, so stock's output depends on the
machine's speed. The pyroid modes have no timeout. stock records the pages it lost (OUT/events.jsonl), and
``native --timeouts-of STOCK_OUT`` drops the same pages and restarts its cache after each, so its digests can be
compared with that stock run exactly.

Outputs other than the final documents (OUT/deduped/) are deleted after the digests unless ``--keep-outputs``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gzip
import hashlib
import importlib.machinery
import json
import os
import resource
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Any

T = time.perf_counter
T_START = T()
GLOB = "*.warc.gz"
MARKER = ".fineweb_warc"
WATCH = ("timeout", "broken_process", "clean_error")
DIGESTS = ("after_minhash_filter", "after_pii")
# fineweb.py's exclusion folders, per stage (the LID one is split by language: 2_non_english/<lang>/<rank>.jsonl.gz).
EXCLUDED = {
    "url_filter": "removed/1_url",
    "lid": "2_non_english",
    "gopher_repetition": "removed/3_gopher_rep",
    "gopher_quality": "removed/4_gopher_qual",
    "c4_quality": "removed/5_c4",
    "fineweb_quality": "removed/6_fineweb_qual",
}
FILTER_STAGES = ("gopher_repetition", "gopher_quality", "c4_quality", "fineweb_quality")
STAGES = ("warc_reader", "url_filter", "trafilatura", "lid", *FILTER_STAGES, "minhash_filter", "pii")
VERSIONS = (
    "datatrove",
    "trafilatura",
    "lxml",
    "justext",
    "courlan",
    "htmldate",
    "warcio",
    "faust-cchardet",
    "python-magic",
    "tldextract",
    "pyahocorasick",
    "spacy",
    "fasttext-numpy2-wheel",
    "fasttext-predict",
    "fasteners",
    "orjson",
    "numpy",
    "pyarrow",
    "pyroid",
)

try:
    import magic  # noqa: F401

    LIBMAGIC = True
except ImportError:  # datatrove only needs libmagic for records without WARC-Identified-Payload-Type
    LIBMAGIC = False

    def _no_magic(*_a: object, **_k: object) -> str:
        raise RuntimeError("record without WARC-Identified-Payload-Type: libmagic needed")

    _stub = types.ModuleType("magic")
    _stub.__spec__ = importlib.machinery.ModuleSpec("magic", None)
    _stub.from_buffer = _no_magic  # type: ignore[attr-defined]
    sys.modules["magic"] = _stub


# ------------------------------------------------------------------------------------------------- helpers


def warc_files(folder: str) -> list[str]:
    """The WARC files in ``folder``, in the order datatrove's WarcReader shards them (sorted relative paths)."""
    import fnmatch

    names = sorted(
        f for f in os.listdir(folder) if fnmatch.fnmatch(f, GLOB) and os.path.isfile(Path(folder, f))
    )
    if not names:
        sys.exit(f"no {GLOB} files in {folder}")
    return names


def prepare_out(out: str) -> Path:
    p = Path(out)
    if p.exists() and any(p.iterdir()):
        if not (p / MARKER).exists():
            sys.exit(f"{out} is not empty and was not written by fineweb_warc.py; refusing to delete it")
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    (p / MARKER).write_text("")
    return p


def rusage() -> dict[str, float]:
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "total": round(s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime, 3),
        "self_user": round(s.ru_utime, 3),
        "self_sys": round(s.ru_stime, 3),
        "children_user": round(c.ru_utime, 3),
        "children_sys": round(c.ru_stime, 3),
    }


def loadavg() -> str | None:
    try:
        return Path("/proc/loadavg").read_text().strip()
    except OSError:
        return None


def versions() -> dict[str, str | None]:
    import importlib.metadata as md

    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for d in VERSIONS:
        try:
            out[d] = md.version(d)
        except md.PackageNotFoundError:
            out[d] = None
    return out


def default_label() -> str:
    if os.environ.get("BENCH_LABEL"):
        return os.environ["BENCH_LABEL"]
    return "Docker/OrbStack VM, indicative" if Path("/.dockerenv").exists() else "unlabelled"


def _utf8(s: str) -> bytes:
    return s.encode("utf-8", "surrogatepass")


def digest(docs: list[tuple[str, str]]) -> dict[str, Any]:
    """sha256 over id \\0 text \\0 of ``docs`` sorted by id."""
    h = hashlib.sha256()
    for i, t in sorted(docs):
        h.update(_utf8(i) + b"\0" + _utf8(t) + b"\0")
    return {"sha256": h.hexdigest(), "docs": len(docs)}


def short_hash(t: str) -> str:
    return hashlib.sha256(_utf8(t)).hexdigest()[:16]


def finish_digests(
    out: Path, after_filter: dict[int, list[tuple[str, str]]], after_pii: dict[int, list[tuple[str, str]]]
) -> dict[str, Any]:
    """Both digests, and OUT/docs.tsv.gz (id, rank, hash after the MinHash filter, hash after PII)."""
    flat_f = [d for r in sorted(after_filter) for d in after_filter[r]]
    flat_p = [d for r in sorted(after_pii) for d in after_pii[r]]
    res = {"after_minhash_filter": digest(flat_f), "after_pii": digest(flat_p)}
    pii = {i: t for i, t in flat_p}
    rank_of = {i: r for r in after_filter for i, _ in after_filter[r]}
    with gzip.open(out / "docs.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("id\trank\tafter_minhash_filter\tafter_pii\n")
        for i, t in sorted(flat_f):
            p = pii.get(i)
            f.write(f"{i}\t{rank_of[i]}\t{short_hash(t)}\t{'-' if p is None else short_hash(p)}\n")
    if {i for i, _ in flat_f} != set(pii):
        res["warning"] = "the documents after PII are not the documents after the MinHash filter"
    return res


def read_jsonl(path: Path) -> list[tuple[str, str]]:
    import orjson

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        return [(r["id"], r["text"]) for r in map(orjson.loads, f)]


def rank_files(folder: Path, suffix: str) -> dict[int, Path]:
    """``{rank: path}`` for datatrove-style ``RANK.<suffix>`` files."""
    out = {}
    for p in sorted(folder.glob(f"*{suffix}")) if folder.exists() else []:
        out[int(p.name[: -len(suffix)])] = p
    return out


def cleanup(out: Path, keep: bool) -> None:
    if keep:
        return
    for d in ("removed", "2_non_english", "output", "minhash", "tee"):
        shutil.rmtree(out / d, ignore_errors=True)


def print_steps(summary: dict[str, Any]) -> None:
    rows = summary["steps"]
    total = sum(r["seconds"] for r in rows if r["step"] != "tee") or float("nan")
    print(f"{'phase:step':<34} {'seconds':>10} {'share':>7} {'docs out':>10}")
    for r in rows:
        share = "" if r["step"] == "tee" else f"{100 * r['seconds'] / total:6.1f}%"
        print(f"{r['phase'] + ':' + r['step']:<34} {r['seconds']:>10.1f} {share:>7} {r['docs_out']:>10}")
    w = summary["wall_seconds"]
    print(
        f"wall {w['total']:.1f} s ({', '.join(f'{k} {v:.1f}' for k, v in w['phases'].items())}); "
        f"cpu {summary['cpu_seconds']['total']:.1f} s"
    )
    for k in DIGESTS:
        d = summary["digests"][k]
        if d is not None:
            print(f"digest {k}: {d['sha256']} ({d['docs']} docs)")


def write_summary(out: Path, summary: dict[str, Any]) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print_steps(summary)


def base_summary(a: argparse.Namespace, mode: str, files: list[str]) -> dict[str, Any]:
    return {
        "mode": mode,
        "label": a.label,
        "git_sha": os.environ.get("BENCH_GIT_SHA"),
        "workers": min(a.workers, len(files)),
        "tasks": len(files),
        "limit": a.limit,
        "warcs": [{"name": f, "bytes": os.path.getsize(Path(a.warcs, f))} for f in files],
        "cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "loadavg_start": loadavg(),
        "libmagic": LIBMAGIC,
        "versions": versions(),
    }


# ----------------------------------------------------------------------------------------- datatrove modes

from datatrove.pipeline.base import PipelineStep  # type: ignore[import-not-found]


class Timed(PipelineStep):  # type: ignore[misc]
    """Run ``inner`` and record the wall time spent in it, excluding the upstream steps.

    With ``watch``, also record which document each ``timeout``/``broken_process``/``clean_error`` of the inner
    step's stats belongs to: datatrove's extractor counts it before it pulls the next document, so it is the
    document pulled last.
    """

    type = "TIMED"
    name = "timed"

    def __init__(self, inner: Any, label: str, folder: str, phase: str, watch: bool = False) -> None:
        super().__init__()
        self.inner, self.label, self.folder, self.phase, self.watch = inner, label, folder, phase, watch
        self.name = f"timed {label}"

    def _counts(self) -> tuple[float, ...]:
        st = self.inner.stats.stats
        return tuple(st[k].total if k in st else 0 for k in WATCH)

    def run(self, data: Any = None, rank: int = 0, world_size: int = 1) -> Any:
        up = [0.0]
        last: list[Any] = [None]
        pos = [0]
        prev = [self._counts()]
        events: list[dict[str, Any]] = []

        def check() -> None:
            c = self._counts()
            if c != prev[0]:
                for kind, a, b in zip(WATCH, prev[0], c, strict=True):
                    events.extend(
                        [
                            {
                                "rank": rank,
                                "kind": kind,
                                "position": pos[0] - 1,
                                "id": None if last[0] is None else last[0].id,
                            }
                        ]
                        * int(b - a)
                    )
                prev[0] = c

        def upstream(it: Any) -> Any:
            it = iter(it)
            while True:
                if self.watch:
                    check()
                t0 = T()
                try:
                    x = next(it)
                except StopIteration:
                    up[0] += T() - t0
                    return
                up[0] += T() - t0
                last[0] = x
                pos[0] += 1
                yield x

        incl, n = 0.0, 0
        t0 = T()
        gen = self.inner.run(upstream(data) if data is not None else None, rank, world_size)
        incl += T() - t0  # steps whose run() is not a generator (MinHash stages) do all their work here
        try:
            while gen is not None:
                t0 = T()
                try:
                    doc = next(gen)
                except StopIteration:
                    incl += T() - t0
                    break
                incl += T() - t0
                n += 1
                yield doc
        finally:
            if self.watch:
                check()
            ru_s, ru_c = (
                resource.getrusage(resource.RUSAGE_SELF),
                resource.getrusage(resource.RUSAGE_CHILDREN),
            )
            stats = self.inner.stats.stats
            rec = {
                "phase": self.phase,
                "rank": rank,
                "pid": os.getpid(),
                "step": self.label,
                "seconds": incl - up[0],
                "docs_out": n,
                "counts": {
                    k: stats[k].total for k in (*WATCH, "extracted", "forwarded", "dropped") if k in stats
                },
                "cpu_self": ru_s.ru_utime + ru_s.ru_stime,
                "cpu_children": ru_c.ru_utime + ru_c.ru_stime,
                "maxrss_kib": ru_s.ru_maxrss,
                "children_maxrss_kib": ru_c.ru_maxrss,
            }
            d = Path(self.folder)
            (d / "timing").mkdir(parents=True, exist_ok=True)
            with open(d / "timing" / f"{self.phase}-{rank:05d}.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            if events:
                with open(d / "timing" / f"events-{rank:05d}.jsonl", "a") as f:
                    f.writelines(json.dumps(e) + "\n" for e in events)


class Tee(PipelineStep):  # type: ignore[misc]
    """Pass documents on and write their id and text to FOLDER/RANK.jsonl (for the digest)."""

    type = "TEE"
    name = "tee"

    def __init__(self, folder: str) -> None:
        super().__init__()
        self.folder = folder

    def run(self, data: Any = None, rank: int = 0, world_size: int = 1) -> Any:
        import orjson

        Path(self.folder).mkdir(parents=True, exist_ok=True)
        with open(f"{self.folder}/{rank:05d}.jsonl", "wb") as f:
            for doc in data:
                f.write(orjson.dumps({"id": doc.id, "text": doc.text}, option=orjson.OPT_APPEND_NEWLINE))
                yield doc


class LazyPIIFormatter(PipelineStep):  # type: ignore[misc]
    """pyroid's PIIFormatter drop-in, built inside each task.

    A workaround, used only when the drop-in cannot be deep-copied or pickled (the executor does both to the
    pipeline: a deepcopy per task with one worker, a pickle per task with more). One instance per task is what
    datatrove's executor gives each task anyway, so the replacement rotation is the same.
    """

    type = "FORMAT"
    name = "lazy PIIFormatter"

    def run(self, data: Any = None, rank: int = 0, world_size: int = 1) -> Any:
        from pyroid.integrations.datatrove import PIIFormatter

        step = PIIFormatter()
        self.stats = step.stats
        yield from step.run(data, rank, world_size)


def pii_step(pyroid_steps: bool) -> tuple[Any, str | None]:
    """The stage-4 PIIFormatter, and the workaround used for it (None if none)."""
    if not pyroid_steps:
        from datatrove.pipeline.formatters import PIIFormatter as DatatrovePII

        return DatatrovePII(), None
    import pickle

    from pyroid.integrations.datatrove import PIIFormatter

    step = PIIFormatter()
    try:
        copy.deepcopy(step)
        pickle.dumps(step)
    except TypeError as e:
        return LazyPIIFormatter(), f"PIIFormatter: {e}; built inside each task instead"
    return step, None


def stock_lid(model_path: str | None, exclusion_writer: Any) -> Any:
    """datatrove's LanguageFilter; with ``model_path``, on fasttext-predict and that file (see the docstring)."""
    from datatrove.pipeline.filters import LanguageFilter

    kw = {"languages": ["en"], "language_threshold": 0.65, "exclusion_writer": exclusion_writer}
    if model_path is None:
        return LanguageFilter(**kw)
    import numpy as np
    from datatrove.utils.lid import FT176LID

    class NumpyProbs:  # fasttext-numpy2-wheel returns numpy probabilities; datatrove calls .item() on them
        def __init__(self, m: Any) -> None:
            self.m = m

        def predict(
            self, text: str, k: int = 1, threshold: float = 0.0, on_unicode_error: str = "strict"
        ) -> Any:
            labels, probs = self.m.predict(text, k, threshold, on_unicode_error)
            return labels, np.asarray(probs)

    class LocalFT176(FT176LID):  # type: ignore[misc]
        def __init__(self, languages: Any, path: str, k: int = -1) -> None:
            super().__init__(languages, k)
            self.path = path

        @property
        def model(self) -> Any:
            if self._model is None:
                from fasttext.FastText import _FastText

                self._model = NumpyProbs(_FastText(self.path))
            return self._model

    class ShimLanguageFilter(LanguageFilter):  # type: ignore[misc]
        def __new__(cls, *a: Any, **k: Any) -> Any:  # skip the fasttext-numpy2-wheel dependency check
            return object.__new__(cls)

        def __init__(self, path: str, **k: Any) -> None:
            super().__init__(**k)
            self.model = LocalFT176(self.languages, path)

    return ShimLanguageFilter(model_path, **kw)


def base_steps(a: argparse.Namespace, out: Path, pyroid_steps: bool) -> list[tuple[str, Any]]:
    from datatrove.pipeline.readers import WarcReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter

    if pyroid_steps:
        from pyroid.integrations.datatrove import (
            C4QualityFilter,
            FineWebQualityFilter,
            GopherQualityFilter,
            GopherRepetitionFilter,
            LanguageFilter,
            Trafilatura,
            URLFilter,
        )
    else:
        from datatrove.pipeline.extractors import Trafilatura
        from datatrove.pipeline.filters import (
            C4QualityFilter,
            FineWebQualityFilter,
            GopherQualityFilter,
            GopherRepetitionFilter,
            URLFilter,
        )

    def rm(stage: str) -> Any:
        return JsonlWriter(f"{out}/{EXCLUDED[stage]}")

    lid_writer = JsonlWriter(f"{out}/{EXCLUDED['lid']}", output_filename="${language}/${rank}.jsonl.gz")
    if pyroid_steps:
        lid = LanguageFilter(
            languages=["en"], language_threshold=0.65, exclusion_writer=lid_writer, model_path=a.lid_model
        )
    else:
        lid = stock_lid(a.lid_model, lid_writer)
    return [
        (
            "warc_reader",
            WarcReader(a.warcs, glob_pattern=GLOB, limit=a.limit, default_metadata={"dump": a.dump}),
        ),
        ("url_filter", URLFilter(exclusion_writer=rm("url_filter"))),
        ("trafilatura", Trafilatura(favour_precision=True)),
        ("lid", lid),
        ("gopher_repetition", GopherRepetitionFilter(exclusion_writer=rm("gopher_repetition"))),
        ("gopher_quality", GopherQualityFilter(exclusion_writer=rm("gopher_quality"))),
        ("c4_quality", C4QualityFilter(filter_no_terminal_punct=False, exclusion_writer=rm("c4_quality"))),
        ("fineweb_quality", FineWebQualityFilter(exclusion_writer=rm("fineweb_quality"))),
        ("writer", JsonlWriter(f"{out}/output")),
    ]


def minhash_phases(
    out: Path, tasks: int, pyroid_steps: bool, pii: Any
) -> list[tuple[str, int, list[tuple[str, Any]]]]:
    from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter
    from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter
    from datatrove.utils.hashing import HashConfig

    if pyroid_steps:
        from pyroid.integrations.datatrove import MinhashDedupSignature
    else:
        from datatrove.pipeline.dedup import MinhashDedupSignature

    cfg = MinhashConfig(
        hash_config=HashConfig(hash_fc="sha1", precision=64), num_buckets=14, hashes_per_bucket=8, n_grams=5
    )
    mh = f"{out}/minhash"
    return [
        (
            "mh1",
            tasks,
            [
                ("reader", JsonlReader(f"{out}/output")),
                ("minhash_signature", MinhashDedupSignature(output_folder=f"{mh}/signatures", config=cfg)),
            ],
        ),
        (
            "mh2",
            cfg.num_buckets,
            [
                (
                    "minhash_buckets",
                    MinhashDedupBuckets(
                        input_folder=f"{mh}/signatures",
                        output_folder=f"{mh}/buckets",
                        config=MinhashConfig(hash_config=cfg.hash_config),
                    ),
                ),
            ],
        ),
        (
            "mh3",
            1,
            [
                (
                    "minhash_cluster",
                    MinhashDedupCluster(
                        input_folder=f"{mh}/buckets", output_folder=f"{mh}/remove_ids", config=cfg
                    ),
                ),
            ],
        ),
        (
            "mh4",
            tasks,
            [
                ("reader", JsonlReader(f"{out}/output")),
                ("minhash_filter", MinhashDedupFilter(input_folder=f"{mh}/remove_ids")),
                ("tee", Tee(f"{out}/tee/after_minhash_filter")),
                ("pii", pii),
                ("writer", JsonlWriter(f"{out}/deduped")),
            ],
        ),
    ]


def run_executor(
    phase: str, steps: list[tuple[str, Any]], tasks: int, workers: int, out: Path, watch: str | None = None
) -> float:
    from datatrove.executor import LocalPipelineExecutor

    pipeline = [Timed(s, label, str(out), phase, watch=label == watch) for label, s in steps]
    t0 = T()
    LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=tasks,
        workers=min(tasks, workers),
        logging_dir=str(out / "logs" / phase),
        skip_completed=False,
        start_method="fork",
    ).run()
    return T() - t0


def timing_records(out: Path) -> list[dict[str, Any]]:
    recs = []
    for f in sorted((out / "timing").glob("*-*.jsonl")):
        if not f.name.startswith("events-"):
            recs += [json.loads(line) for line in f.read_text().splitlines()]
    return recs


def summarize_timing(recs: list[dict[str, Any]], order: list[tuple[str, str]]) -> dict[str, Any]:
    """Seconds, documents out and counts per (phase, step), summed over tasks, in pipeline order."""
    steps = {
        k: {"phase": k[0], "step": k[1], "seconds": 0.0, "docs_out": 0, "tasks": 0, "counts": {}}
        for k in order
    }
    for r in recs:
        s = steps[(r["phase"], r["step"])]
        s["seconds"] += r["seconds"]
        s["docs_out"] += r["docs_out"]
        s["tasks"] += 1
        for k, v in r.get("counts", {}).items():
            s["counts"][k] = s["counts"].get(k, 0) + v
    rows = list(steps.values())
    for s in rows:
        s["seconds"] = round(s["seconds"], 3)
    pids: dict[int, int] = {}
    for r in recs:
        pids[r["pid"]] = max(pids.get(r["pid"], 0), r["maxrss_kib"])
    return {"steps": rows, "worker_maxrss_mib": round(max(pids.values(), default=0) / 1024, 1)}


def stage_docs(rows: list[dict[str, Any]]) -> dict[str, int]:
    by = {(r["phase"], r["step"]): r["docs_out"] for r in rows}
    docs = {s: by.get(("base", s), 0) for s in STAGES[:-2]}
    docs["minhash_filter"] = by.get(("mh4", "minhash_filter"), 0)
    docs["pii"] = by.get(("mh4", "pii"), 0)
    return docs


def cmd_datatrove(a: argparse.Namespace, mode: str) -> None:
    pyroid_steps = mode == "pyroid-steps"
    files = warc_files(a.warcs)
    out = prepare_out(a.out)
    summary = base_summary(a, mode, files)
    if pyroid_steps:
        import warnings

        import pyroid

        warnings.simplefilter("error", pyroid.FallbackWarning)
    steps = base_steps(a, out, pyroid_steps)
    if pyroid_steps:
        traf = dict(steps)["trafilatura"]
        summary["trafilatura_mode"] = traf._mode()
        if summary["trafilatura_mode"] is None:
            sys.exit(
                f"pyroid does not reproduce the installed trafilatura {summary['versions']['trafilatura']}"
            )
    else:
        summary["lid"] = (
            "datatrove LanguageFilter" if a.lid_model is None else f"shim: fasttext-predict, {a.lid_model}"
        )
    tasks = len(files)
    phases: dict[str, float] = {}
    pii, workaround = pii_step(pyroid_steps)
    if workaround:
        summary["workarounds"] = [workaround]
        print(f"workaround: {workaround}", file=sys.stderr)
    mh = minhash_phases(out, tasks, pyroid_steps, pii)
    order = [("base", label) for label, _ in steps] + [(ph, label) for ph, _, st in mh for label, _ in st]
    phases["base"] = run_executor(
        "base", steps, tasks, a.workers, out, watch=None if pyroid_steps else "trafilatura"
    )
    for phase, n, st in mh:
        phases[phase] = run_executor(phase, st, n, a.workers, out)
    total = T() - T_START
    cpu = rusage()
    summary["loadavg_end"] = loadavg()
    t = summarize_timing(timing_records(out), order)
    events = [
        json.loads(line)
        for f in sorted((out / "timing").glob("events-*.jsonl"))
        for line in f.read_text().splitlines()
    ]
    if events:
        (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    traf_counts = next(
        (r["counts"] for r in t["steps"] if r["phase"] == "base" and r["step"] == "trafilatura"), {}
    )
    after_filter = {
        r: read_jsonl(p) for r, p in rank_files(out / "tee" / "after_minhash_filter", ".jsonl").items()
    }
    after_pii = {r: read_jsonl(p) for r, p in rank_files(out / "deduped", ".jsonl.gz").items()}
    summary.update(
        {
            "wall_seconds": {"total": round(total, 3), "phases": {k: round(v, 3) for k, v in phases.items()}},
            "cpu_seconds": cpu,
            "maxrss_mib": {"parent": ru_self(), "largest_worker": t["worker_maxrss_mib"]},
            "steps": t["steps"],
            "docs": stage_docs(t["steps"]),
            "trafilatura": {
                k: int(traf_counts.get(k, 0)) for k in (*WATCH, "extracted", "forwarded", "dropped")
            },
            "events": len(events),
            "pyroid_imported": "pyroid" in sys.modules,
            "digests": finish_digests(out, after_filter, after_pii),
        }
    )
    cleanup(out, a.keep_outputs)
    write_summary(out, summary)


def ru_self() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# ------------------------------------------------------------------------------------------------ native

_TIMEOUT = object()  # a page dropped as datatrove's step drops a timeout (--timeouts-of)
_BROKEN = object()  # a page whose extraction raised (datatrove: broken_process, a new child, an empty cache)
_W: dict[str, Any] = {}  # per worker process: the URL filter and the recipe, built at first use
# What datatrove's Trafilatura step passes to trafilatura.extract, in the release datatrove 0.10.1 resolves.
EXTRACT_KW = {"compat": "1.11.0", "favor_precision": True, "include_comments": False}


class Writers:
    """gzip JSONL files under OUT, opened on first write (datatrove's JsonlWriter: orjson, gzip level 9)."""

    def __init__(self, out: Path) -> None:
        self.out, self.files, self.stack = out, {}, contextlib.ExitStack()

    def write(self, rel: str, doc: dict[str, Any]) -> None:
        import orjson

        f = self.files.get(rel)
        if f is None:
            p = self.out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            f = self.files[rel] = self.stack.enter_context(gzip.open(p, "wb"))  # noqa: SIM115 (closed in close())
        f.write(orjson.dumps(doc, option=orjson.OPT_APPEND_NEWLINE))

    def close(self) -> None:
        self.stack.close()


def _records(path: Path, limit: int) -> Any:
    """Documents as datatrove's WarcReader makes them, on warcio: process_record, the empty-text skip, the limit."""
    from datatrove.pipeline.readers.warc import process_record
    from warcio.archiveiterator import ArchiveIterator

    n = 0
    with gzip.open(path, "rb") as f:
        for record in ArchiveIterator(f):
            d = process_record(record)
            if not d or not d.get("text"):
                continue
            if limit != -1 and n >= limit:
                return
            n += 1
            yield d


def documents(path: Path, limit: int, reader: str) -> Any:
    """The documents of one WARC file: pyroid.warc (FineWebWarc.documents) or warcio (``--reader warcio``)."""
    if reader == "warcio":
        return _records(path, limit)
    from pyroid.integrations.recipes import FineWebWarc

    return FineWebWarc.documents(path, limit=limit)


def extractor(skip: dict[str, str]) -> Any:
    """``recipes.TaskExtractor`` over one task's pages, plus the pages a stock run lost (``--timeouts-of``)."""
    from pyroid.integrations.recipes import TaskExtractor

    class Extractor(TaskExtractor):  # type: ignore[misc]
        """datatrove's per-task Trafilatura step (TaskExtractor: the dedup cache, the error drops), and the
        timeout and dead-child drops of a stock run, emulated: the page is dropped and the cache restarted."""

        def __init__(self) -> None:
            super().__init__(threads=1, **EXTRACT_KW)
            self.skip = skip
            self.events: list[dict[str, Any]] = []

        def _batch(self, htmls: list[str]) -> list[Any]:
            return [_BROKEN if isinstance(r, Exception) else r for r in TaskExtractor.__call__(self, htmls)]

        def __call__(self, docs: list[dict[str, Any]], rank: int, pos: int) -> list[Any]:  # type: ignore[override]
            out: list[Any] = []
            i = 0
            while i < len(docs):
                j = next((k for k in range(i, len(docs)) if docs[k]["id"] in self.skip), len(docs))
                if j > i:
                    out += self._batch([d["text"] for d in docs[i:j]])
                if j < len(docs):
                    kind = self.skip[docs[j]["id"]]
                    out.append(_TIMEOUT)
                    self.events.append(
                        {
                            "rank": rank,
                            "kind": kind,
                            "position": pos + j,
                            "id": docs[j]["id"],
                            "emulated": True,
                        }
                    )
                    if kind != "clean_error":  # datatrove restarts the child after a timeout or a dead child
                        self.reset()
                    j += 1
                i = j
            for k, r in enumerate(out):
                if r is _BROKEN:
                    self.events.append(
                        {"rank": rank, "kind": "broken_process", "position": pos + k, "id": docs[k]["id"]}
                    )
            return out

    return Extractor()


def native_task(job: tuple[int, str, dict[str, Any]]) -> dict[str, Any]:
    rank, path, o = job
    out = Path(o["out"])
    secs = dict.fromkeys(("warc_reader", "url_filter", "trafilatura", "lid_and_filters", "writers"), 0.0)
    docs = dict.fromkeys(STAGES[:-2], 0)
    counts = dict.fromkeys(("timeout", "broken_process", "extracted", "forwarded", "dropped"), 0)
    w = Writers(out)
    meta0 = {"dump": o["dump"], "file_path": path}
    t0 = T()
    if "url" not in _W:
        from pyroid.integrations.recipes import FineWebWarc

        fw = FineWebWarc(lid_model=o["lid_model"], threads=1)
        _W["url"] = fw.url_filter
        _W["recipe"] = fw.fineweb  # FineWeb(ftfy_config=False): LID and the four filters
    # building the filter (datatrove's step loads its lists at the first document)
    secs["url_filter"] += T() - t0
    ext = extractor(o["skip"].get(str(rank), {}))
    seen = 0

    def process(batch: list[dict[str, Any]]) -> None:
        nonlocal seen
        docs["warc_reader"] += len(batch)
        t = T()
        reasons = _W["url"].reasons([d["url"] for d in batch], threads=1)
        secs["url_filter"] += T() - t
        kept = []
        t = T()
        for d, r in zip(batch, reasons, strict=True):
            if r is None:
                kept.append(d)
            else:
                w.write(
                    f"{EXCLUDED['url_filter']}/{rank:05d}.jsonl.gz",
                    {
                        "text": d["text"],
                        "id": d["id"],
                        "metadata": {**meta0, "url": d["url"], "date": d["date"], "filter_reason": r},
                    },
                )
        secs["writers"] += T() - t
        docs["url_filter"] += len(kept)
        t = T()
        results = ext(kept, rank, seen)
        secs["trafilatura"] += T() - t
        seen += len(kept)
        texts, live = [], []
        for d, r in zip(kept, results, strict=True):
            if r is _TIMEOUT:
                counts["timeout"] += 1
            elif r is _BROKEN:
                counts["broken_process"] += 1
            else:
                counts["extracted"] += 1
                if r:
                    counts["forwarded"] += 1
                    texts.append(r)
                    live.append(d)
                else:
                    counts["dropped"] += 1
        docs["trafilatura"] += len(live)
        if not live:
            return
        t = T()
        ann = _W["recipe"].annotate(texts, threads=1)
        keep, reason = ann.keep.to_pylist(), ann.reason.to_pylist()
        lang, score, final = ann.language.to_pylist(), ann.language_score.to_pylist(), ann.text.to_pylist()
        secs["lid_and_filters"] += T() - t
        t = T()
        for d, k, r, lg, sc, txt in zip(live, keep, reason, lang, score, final, strict=True):
            meta = {**meta0, "url": d["url"], "date": d["date"], "language": lg, "language_score": sc}
            if k:
                w.write(f"output/{rank:05d}.jsonl.gz", {"text": txt, "id": d["id"], "metadata": meta})
                continue
            stage, _, why = r.partition(":")
            rel = (
                f"{EXCLUDED['lid']}/{lg}/{rank:05d}.jsonl.gz"
                if stage == "lid"
                else f"{EXCLUDED[stage]}/{rank:05d}.jsonl.gz"
            )
            w.write(
                rel,
                {"text": txt, "id": d["id"], "metadata": {**meta, **({"filter_reason": why} if why else {})}},
            )
        secs["writers"] += T() - t
        n = len(live)
        dropped = [r.partition(":")[0] for r in reason if r is not None]
        for s in ("lid", *FILTER_STAGES):
            n -= dropped.count(s)
            docs[s] += n

    batch: list[dict[str, Any]] = []
    it = documents(Path(o["warcs"], path), o["limit"], o["reader"])
    while True:
        t = T()
        d = next(it, None)
        secs["warc_reader"] += T() - t
        if d is not None:
            batch.append(d)
        if batch and (d is None or len(batch) >= o["batch_size"]):
            process(batch)
            batch = []
        if d is None:
            break
    t = T()
    w.close()
    secs["writers"] += T() - t
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "rank": rank,
        "pid": os.getpid(),
        "seconds": secs,
        "docs": docs,
        "counts": counts,
        "events": ext.events,
        "maxrss_kib": ru.ru_maxrss,
    }


def load_skip(run: str | None) -> dict[str, dict[str, str]]:
    """``{rank: {id: kind}}`` of the pages a stock run's Trafilatura step lost (its events.jsonl)."""
    if run is None:
        return {}
    p = Path(run) / "events.jsonl"
    out: dict[str, dict[str, str]] = {}
    if p.exists():
        for line in p.read_text().splitlines():
            e = json.loads(line)
            out.setdefault(str(e["rank"]), {})[e["id"]] = e["kind"]
    return out


def cmd_native(a: argparse.Namespace) -> None:
    import multiprocessing as mp

    files = warc_files(a.warcs)
    out = prepare_out(a.out)
    summary = base_summary(a, "native", files)
    summary["reader"] = "pyroid.warc.iter_documents" if a.reader == "pyroid" else "warcio + process_record"
    skip = load_skip(a.timeouts_of)
    if a.timeouts_of:
        summary["timeouts_of"] = {"run": a.timeouts_of, "pages": sum(len(v) for v in skip.values())}
    o = {
        "out": str(out),
        "warcs": a.warcs,
        "lid_model": a.lid_model,
        "limit": a.limit,
        "dump": a.dump,
        "batch_size": a.batch_size,
        "skip": skip,
        "reader": a.reader,
    }
    workers = min(a.workers, len(files))
    phases: dict[str, float] = {}
    t0 = T()
    with mp.get_context("fork").Pool(workers) as pool:
        results = sorted(
            pool.imap_unordered(native_task, [(r, f, o) for r, f in enumerate(files)]),
            key=lambda x: x["rank"],
        )
        pool.close()
        pool.join()
    phases["base"] = T() - t0

    import orjson
    import pyroid.dedup
    import pyroid.filters as pf
    from pyroid.integrations.recipes import MINHASH_CONFIG

    t0 = T()
    sec = {"reader": 0.0, "minhash": 0.0, "pii": 0.0, "writer": 0.0}
    per_rank = []
    for rank, p in rank_files(out / "output", ".jsonl.gz").items():
        with gzip.open(p, "rb") as f:
            per_rank.append((rank, [orjson.loads(line) for line in f]))
    sec["reader"] = T() - t0
    t = T()
    dedup = pyroid.dedup.MinHashDedup(threads=workers, **MINHASH_CONFIG)
    for _, rows in per_rank:
        dedup.add([r["text"] for r in rows], threads=workers)
    mask = dedup.keep_mask()
    sec["minhash"] = T() - t
    phases["dedup"] = T() - t0
    t0 = T()
    after_filter: dict[int, list[tuple[str, str]]] = {}
    after_pii: dict[int, list[tuple[str, str]]] = {}
    at = 0
    for rank, rows in per_rank:
        kept = [r for r, k in zip(rows, mask[at : at + len(rows)], strict=True) if k]
        at += len(rows)
        after_filter[rank] = [(r["id"], r["text"]) for r in kept]
        t = T()
        texts = pf.PIIReplacer()([r["text"] for r in kept], threads=workers)  # a fresh rotation per task
        sec["pii"] += T() - t
        t = T()
        (out / "deduped").mkdir(exist_ok=True)
        with gzip.open(out / "deduped" / f"{rank:05d}.jsonl.gz", "wb") as f:
            for r, txt in zip(kept, texts, strict=True):
                r["text"] = txt
                f.write(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE))
        sec["writer"] += T() - t
        after_pii[rank] = [(r["id"], r["text"]) for r in kept]
    phases["pii"] = T() - t0
    total = T() - T_START
    cpu = rusage()
    summary["loadavg_end"] = loadavg()

    rows_out = []
    out_of = {
        "warc_reader": "warc_reader",
        "url_filter": "url_filter",
        "trafilatura": "trafilatura",
        "lid_and_filters": "fineweb_quality",
    }
    for step in ("warc_reader", "url_filter", "trafilatura", "lid_and_filters", "writers"):
        n_out = sum(r["docs"][out_of[step]] for r in results) if step in out_of else 0
        rows_out.append(
            {
                "phase": "base",
                "step": step,
                "seconds": round(sum(r["seconds"][step] for r in results), 3),
                "docs_out": n_out,
                "tasks": len(results),
            }
        )
    n_kept = sum(len(rows) for _, rows in per_rank)
    n_dedup = sum(len(v) for v in after_filter.values())
    for phase, step, n in (
        ("dedup", "reader", n_kept),
        ("dedup", "minhash", n_dedup),
        ("pii", "pii", n_dedup),
        ("pii", "writer", n_dedup),
    ):
        rows_out.append(
            {"phase": phase, "step": step, "seconds": round(sec[step], 3), "docs_out": n, "tasks": 1}
        )
    docs = {s: sum(r["docs"][s] for r in results) for s in STAGES[:-2]}
    docs["minhash_filter"] = n_dedup
    docs["pii"] = n_dedup
    events = [e for r in results for e in r["events"]]
    if events:
        (out / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    counts = {k: sum(r["counts"][k] for r in results) for k in results[0]["counts"]}
    summary.update(
        {
            "wall_seconds": {"total": round(total, 3), "phases": {k: round(v, 3) for k, v in phases.items()}},
            "cpu_seconds": cpu,
            "maxrss_mib": {
                "parent": ru_self(),
                "largest_worker": round(max(r["maxrss_kib"] for r in results) / 1024, 1),
            },
            "steps": rows_out,
            "docs": docs,
            "trafilatura": {
                "timeout": counts["timeout"],
                "broken_process": counts["broken_process"],
                "clean_error": 0,
                "extracted": counts["extracted"],
                "forwarded": counts["forwarded"],
                "dropped": counts["dropped"],
            },
            "events": len(events),
            "digests": finish_digests(out, after_filter, after_pii),
        }
    )
    cleanup(out, a.keep_outputs)
    write_summary(out, summary)


# --------------------------------------------------------------------------------------------------- cli


def cmd_cli(a: argparse.Namespace) -> None:
    """``pyroid clean --recipe fineweb-warc`` on the WARC files, in rank order, and its final documents' digest."""
    import subprocess

    import pyarrow.parquet as pq

    files = warc_files(a.warcs)
    out = prepare_out(a.out)
    summary = base_summary(a, "cli", files)
    stats_json = out / "cli-stats.json"
    cmd = [
        sys.executable,
        "-m",
        "pyroid",
        "clean",
        *(str(Path(a.warcs, f)) for f in files),
        str(out / "cli.parquet"),
        "--recipe",
        "fineweb-warc",
        "--lid-model",
        a.lid_model,
        "--workers",
        str(min(a.workers, len(files))),
        "--dump",
        a.dump,
        *(["--limit", str(a.limit)] if a.limit != -1 else []),
        "--stats-json",
        str(stats_json),
        "--tmp-dir",
        str(out / "tmp"),
    ]
    summary["command"] = [Path(cmd[0]).name, *cmd[1:]]
    t0 = T()
    p = subprocess.run(cmd, check=False)
    wall = T() - t0
    if p.returncode != 0:
        sys.exit(f"pyroid clean failed ({p.returncode})")
    cpu = rusage()
    summary["loadavg_end"] = loadavg()
    st = json.loads(stats_json.read_text())
    t = pq.read_table(out / "cli.parquet", columns=["id", "text", "file_path"])
    rank_of = {os.path.abspath(Path(a.warcs, f)): r for r, f in enumerate(files)}
    after_pii: dict[int, list[tuple[str, str]]] = {}
    for i, txt, fp in zip(*(t.column(c).to_pylist() for c in ("id", "text", "file_path")), strict=True):
        after_pii.setdefault(rank_of[fp], []).append((i, txt))
    flat = [d for r in sorted(after_pii) for d in after_pii[r]]
    removed: dict[str, int] = st.get("removed", {})

    def dropped(prefix: str) -> int:
        return sum(v for k, v in removed.items() if k == prefix or k.startswith(prefix + ":"))

    n = st["docs_in"]
    docs = {"warc_reader": n}
    for stage in ("url_filter", "trafilatura", "lid", *FILTER_STAGES, "minhash"):
        n -= dropped(stage)
        docs["minhash_filter" if stage == "minhash" else stage] = n
    docs["pii"] = st["docs_out"]
    with gzip.open(out / "docs.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("id\trank\tafter_minhash_filter\tafter_pii\n")
        for r in sorted(after_pii):
            for i, txt in after_pii[r]:
                f.write(f"{i}\t{r}\t-\t{short_hash(txt)}\n")
    summary.update(
        {
            "wall_seconds": {"total": round(T() - T_START, 3), "phases": {"cli": round(wall, 3)}},
            "cpu_seconds": cpu,
            "maxrss_mib": {"parent": ru_self(), "largest_worker": round(_children_maxrss() / 1024, 1)},
            "steps": [
                {"phase": "cli", "step": "pyroid clean", "seconds": round(wall, 3), "docs_out": len(flat)}
            ],
            "docs": docs,
            "cli_stats": st,
            "trafilatura": {
                "timeout": 0,
                "broken_process": removed.get("trafilatura:broken_process", 0),
                "clean_error": 0,
                "extracted": None,
                "forwarded": None,
                "dropped": removed.get("trafilatura", 0),
            },
            "events": 0,
            "digests": {"after_minhash_filter": None, "after_pii": digest(flat)},
        }
    )
    if not a.keep_outputs:
        shutil.rmtree(out / "tmp", ignore_errors=True)
    write_summary(out, summary)


def _children_maxrss() -> float:
    return float(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)


# ------------------------------------------------------------------------------------------------ reader

READERS = ("warcio", "documents", "batches")


def _read_warcio(path: Path, limit: int) -> Any:
    """datatrove's read path: warcio over gzip.open, datatrove's process_record."""
    from datatrove.pipeline.readers.warc import process_record
    from warcio.archiveiterator import ArchiveIterator

    with gzip.open(path, "rb") as f:
        for record in ArchiveIterator(f):
            d = process_record(record)
            if d:
                yield d["id"], d["url"], d["date"], d["text"]
                limit -= 1
                if limit == 0:
                    return


def _read_documents(path: Path, limit: int) -> Any:
    import pyroid.warc as pw

    for _, d in pw.iter_documents(path):
        yield d["id"], d["url"], d["date"], d["text"]
        limit -= 1
        if limit == 0:
            return


def _read_batches(path: Path, limit: int) -> Any:
    import pyroid.warc as pw

    for b in pw.iter_batches(path):
        cols = [b.column(c).to_pylist() for c in ("id", "url", "date", "text")]
        for row in zip(*cols, strict=True):
            if row[3] is not None:
                yield row
                limit -= 1
                if limit == 0:
                    return


def cmd_reader(a: argparse.Namespace) -> None:
    """Time each reader on each file, one after another, and check they give the same documents."""
    files = warc_files(a.warcs)
    out = prepare_out(a.out)
    summary = base_summary(a, "reader", files)
    readers = [r.strip() for r in a.readers.split(",") if r.strip()]
    bad = [r for r in readers if r not in READERS]
    if bad:
        sys.exit(f"unknown readers {bad}: choose from {', '.join(READERS)}")
    fn = {"warcio": _read_warcio, "documents": _read_documents, "batches": _read_batches}
    rows, ok = [], True
    totals = {r: {"wall": 0.0, "cpu": 0.0, "docs": 0} for r in readers}
    print(f"{'file':<58} {'reader':<10} {'wall s':>8} {'cpu s':>8} {'MB/s':>8} {'docs':>7} digest")
    for name in files:
        path = Path(a.warcs, name)
        size = path.stat().st_size
        digests = {}
        for r in readers:
            h = hashlib.sha256()
            n = 0
            error = None
            c0 = resource.getrusage(resource.RUSAGE_SELF)
            t0 = T()
            try:
                for fields in fn[r](path, a.limit):
                    for x in fields:
                        h.update(b"\1" if x is None else _utf8(x) + b"\0")
                    n += 1
            except Exception as e:  # noqa: BLE001 - a damaged file: every reader must stop at the same place
                error = f"{type(e).__name__}: {e}"
                h.update(_utf8(error))
            wall = T() - t0
            c1 = resource.getrusage(resource.RUSAGE_SELF)
            cpu = c1.ru_utime + c1.ru_stime - c0.ru_utime - c0.ru_stime
            digests[r] = h.hexdigest()
            totals[r]["wall"] += wall
            totals[r]["cpu"] += cpu
            totals[r]["docs"] += n
            rows.append(
                {
                    "file": name,
                    "bytes": size,
                    "reader": r,
                    "wall_seconds": round(wall, 3),
                    "cpu_seconds": round(cpu, 3),
                    "docs": n,
                    "error": error,
                    "sha256": digests[r],
                }
            )
            print(
                f"{name:<58} {r:<10} {wall:>8.2f} {cpu:>8.2f} {size / 1e6 / wall:>8.1f} {n:>7} {digests[r][:12]}"
            )
        if len(set(digests.values())) > 1:
            ok = False
            print(f"{name}: the readers differ: {digests}")
    summary.update(
        {
            "readers": readers,
            "files": rows,
            "totals": {r: {k: round(v, 3) for k, v in t.items()} for r, t in totals.items()},
            "identical": ok,
            "loadavg_end": loadavg(),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for r, t in totals.items():
        print(f"total {r:<10} wall {t['wall']:.1f} s, cpu {t['cpu']:.1f} s, {t['docs']} docs")
    print("identical" if ok else "DIFFERENT")
    if not ok:
        sys.exit(1)


# ----------------------------------------------------------------------------------------------- compare


def load_docs(run: Path) -> dict[str, tuple[str, str]]:
    with gzip.open(run / "docs.tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        return {i: (a, b) for i, _, a, b in (line.rstrip("\n").split("\t") for line in f)}


def doc_diff(ref: Path, run: Path) -> dict[str, Any]:
    a, b = load_docs(ref), load_docs(run)
    only_ref = sorted(set(a) - set(b))
    only_run = sorted(set(b) - set(a))
    both = sorted(set(a) & set(b))
    # "-": not recorded (a cli run has only the final text)
    filt = [i for i in both if a[i][0] != b[i][0] and "-" not in (a[i][0], b[i][0])]
    pii = [i for i in both if a[i][1] != b[i][1] and "-" not in (a[i][1], b[i][1])]
    return {
        "only_in_reference": len(only_ref),
        "only_in_run": len(only_run),
        "text_differs_after_minhash_filter": len(filt),
        "text_differs_after_pii": len(pii),
        "examples": {
            "only_in_reference": only_ref[:5],
            "only_in_run": only_run[:5],
            "text_differs_after_minhash_filter": filt[:5],
            "text_differs_after_pii": pii[:5],
        },
    }


def cmd_compare(a: argparse.Namespace) -> int:
    runs = []
    for r in a.runs:
        p = Path(r)
        if not (p / "summary.json").exists():
            print(f"{r}: no summary.json, left out")
            continue
        runs.append((p, json.loads((p / "summary.json").read_text())))
    if not runs:
        return 1
    ref_dir, ref = runs[0]
    rows, ok = [], True
    print(f"reference: {ref_dir.name}")
    print(
        f"{'run':<26} {'mode':<13} {'w':>2} {'wall s':>9} {'cpu s':>10} {'final':>7} {'lost':>5} "
        f"{'after_minhash_filter':<14} {'after_pii':<14} same"
    )
    for p, s in runs:
        # a digest a run does not have (a cli run: after_minhash_filter) is not compared
        same = {
            k: s["digests"][k]["sha256"] == ref["digests"][k]["sha256"]
            for k in DIGESTS
            if s["digests"].get(k) and ref["digests"].get(k)
        }
        lost = s["trafilatura"]["timeout"] + s["trafilatura"]["broken_process"]
        row = {
            "run": p.name,
            "mode": s["mode"],
            "workers": s["workers"],
            "wall_seconds": s["wall_seconds"]["total"],
            "cpu_seconds": s["cpu_seconds"]["total"],
            "final_docs": s["digests"]["after_pii"]["docs"],
            "trafilatura_lost": lost,
            "timeouts_of": s.get("timeouts_of"),
            "digests": {k: (s["digests"].get(k) or {}).get("sha256") for k in DIGESTS},
            "same_as_reference": same,
            "docs": s["docs"],
        }
        if not all(same.values()):
            ok = False
            row["diff"] = doc_diff(ref_dir, p)
        rows.append(row)
        print(
            f"{p.name:<26} {s['mode']:<13} {s['workers']:>2} {row['wall_seconds']:>9.1f} {row['cpu_seconds']:>10.1f} "
            f"{row['final_docs']:>7} {lost:>5} {(row['digests'][DIGESTS[0]] or '-')[:12]:<14} "
            f"{(row['digests'][DIGESTS[1]] or '-')[:12]:<14} {'yes' if all(same.values()) else 'NO'}"
        )
    for row in rows:
        if "diff" in row:
            print(f"{row['run']} vs {ref_dir.name}: {json.dumps(row['diff'])}")
            stages = [k for k in STAGES if row["docs"].get(k) != ref["docs"].get(k)]
            if stages:
                print(
                    f"  document counts differ from stage {stages[0]} on: "
                    + ", ".join(f"{k} {ref['docs'].get(k)} vs {row['docs'].get(k)}" for k in stages)
                )
    lossy = [r["run"] for r in rows if r["trafilatura_lost"] and r["mode"] == "stock"]
    if lossy and not ok:
        print(
            f"note: {', '.join(lossy)} lost pages to datatrove's Trafilatura timeout or a dead child process; compare "
            "them with `native --timeouts-of <stock run>` instead"
        )
    res = {"reference": ref_dir.name, "identical": ok, "runs": rows}
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    print("identical" if ok else "DIFFERENT")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    for mode in ("stock", "pyroid-steps", "native", "cli", "reader"):
        p = sub.add_parser(mode)
        p.add_argument("--warcs", required=True, help=f"folder of {GLOB} files, one task each")
        p.add_argument(
            "--out", required=True, help="output folder (replaced if an earlier run of this script made it)"
        )
        p.add_argument("--workers", type=int, default=1, help="processes [1]")
        p.add_argument("--lid-model", default=None, help="lid.176.bin (stock: use the fasttext-predict shim)")
        p.add_argument(
            "--limit", type=int, default=-1, help="documents per WARC file, as WarcReader's limit [-1]"
        )
        p.add_argument(
            "--dump", default="CC-MAIN-2026-39", help="the documents' dump metadata [CC-MAIN-2026-39]"
        )
        p.add_argument("--label", default=default_label(), help="where this ran [$BENCH_LABEL]")
        p.add_argument("--keep-outputs", action="store_true", help="keep every output folder")
        if mode == "native":
            p.add_argument("--batch-size", type=int, default=256, help="documents per pyroid call [256]")
            p.add_argument(
                "--timeouts-of",
                default=None,
                metavar="RUN",
                help="drop the pages this stock run's Trafilatura step lost, as it did",
            )
            p.add_argument(
                "--reader",
                choices=("pyroid", "warcio"),
                default="pyroid",
                help="pyroid.warc.iter_documents, or warcio + datatrove's process_record [pyroid]",
            )
        if mode == "reader":
            p.add_argument(
                "--readers", default=",".join(READERS), help=f"which readers, in order [{','.join(READERS)}]"
            )
    p = sub.add_parser("compare")
    p.add_argument("runs", nargs="+", help="run folders; the first is the reference")
    p.add_argument("--json", default=None, help="write the comparison here")
    a = ap.parse_args()
    if a.mode == "compare":
        return cmd_compare(a)
    if a.mode == "reader":
        cmd_reader(a)
        return 0
    if a.mode in ("native", "cli") and a.lid_model is None:
        sys.exit(f"{a.mode} needs --lid-model")
    if a.mode == "native":
        cmd_native(a)
    elif a.mode == "cli":
        cmd_cli(a)
    else:
        cmd_datatrove(a, a.mode)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
