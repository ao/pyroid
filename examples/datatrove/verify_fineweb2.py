#!/usr/bin/env python3
"""Equivalence + timing check per language: FineWeb-2 filters and MinHash in datatrove's executor.

For each FineWeb-2 slice (``bench/manifest.json`` ``fineweb2/<lang>.parquet``, 250 docs), runs the
same datatrove pipeline twice in datatrove's own ``LocalPipelineExecutor``: once with datatrove
0.10.1's steps (stock) and once with ``pyroid.integrations.datatrove``'s, then compares a per-doc
digest.

* input: the slice plus ``--copies - 1`` near-duplicate copies of every document (about one
  character in 300 deleted, so datatrove's word cache never hits and MinHash has duplicates to
  find), written as ``--workers`` Parquet shards and read with datatrove's ``ParquetReader``;
* ``filters`` (``--chains``), each step with an exclusion writer and ``language=<lang>``.
  ``fineweb2`` is FineWeb-2's step-5 chain with the language's own settings
  (``pyroid_parity.specs``): GopherRepetitionFilter -> FineWebQualityFilter ->
  GopherQualityFilter. The published slices already passed it, so it keeps nearly everything.
  ``defaults`` drops and rewrites documents: datatrove's defaults in the English FineWeb chain's
  order, GopherRepetitionFilter -> GopherQualityFilter (with the language's FineWeb-2 stop words)
  -> C4QualityFilter(filter_no_terminal_punct=False) -> FineWebQualityFilter (``defaults-no-c4``
  without C4);
* ``minhash``: FineWeb-2's MinhashConfig (xxhash/64, 14 x 8, 5-grams) with ``language=<lang>``:
  ``MinhashDedupSignature`` (stock or pyroid) -> datatrove's buckets -> cluster -> filter.

The digest is sha256 over ``[id, text, keep, reason, dup]`` per input document, in input order. If
the stock pipeline raises, the pyroid one must raise the same exception type and message in the
same executor phase; if that phase is not ``filters`` (where pyroid raises for the whole batch
and datatrove mid-batch), the outputs left behind must be identical too.

Languages pyroid does not tokenize natively (``pyroid.filters.languages()``) run datatrove's code
in both pipelines (with one ``FallbackWarning``), so they must match too, at datatrove's speed.

Needs the parity package and datatrove 0.10.1 with its tokenizers (spaCy; indic-nlp-library for
hin_Deva; jieba / SudachiPy for cmn_Hani / jpn_Jpan) and /data:

    python examples/datatrove/verify_fineweb2.py --out /target/fw2 --workers 8 --copies 20 \\
        --langs arb_Arab,deu_Latn,fra_Latn,hin_Deva,rus_Cyrl,spa_Latn --chains fineweb2,defaults

Prints (and with ``--summary-log`` appends) one JSON line per language and chain; exit 1 if a
digest differs. Timings are wall-clock for all five executor phases, stock and pyroid run back to
back.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline import filters as dt_filters
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.dedup import minhash as dt_minhash
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from pyroid_parity import multilingual as ML  # noqa: N812
from pyroid_parity import refs as R  # noqa: N812
from pyroid_parity import specs

import pyroid.filters
from pyroid.integrations import datatrove as PD  # noqa: N812

#: (stage, ``specs.fineweb2_filter_cases`` case) per chain; ``+stopwords``: FineWeb-2's stop words.
CHAINS = {
    "fineweb2": [
        ("gopher_repetition", "gopher_repetition@fineweb2"),
        ("fineweb_quality", "fineweb_quality@fineweb2"),
        ("gopher_quality", "gopher_quality@fineweb2"),
    ],
    "defaults": [
        ("gopher_repetition", "gopher_repetition"),
        ("gopher_quality", "gopher_quality+stopwords"),
        ("c4_quality", "c4_quality@fineweb"),
        ("fineweb_quality", "fineweb_quality"),
    ],
    # For hin_Deva: IndicNLP's sentence splitter makes datatrove's C4QualityFilter raise
    # ValueError("substring not found") on some documents (pyroid raises the same).
    "defaults-no-c4": [
        ("gopher_repetition", "gopher_repetition"),
        ("gopher_quality", "gopher_quality+stopwords"),
        ("fineweb_quality", "fineweb_quality"),
    ],
}


def corpus(lang: str, copies: int) -> tuple[list[str], list[str]]:
    """The slice, then ``copies - 1`` near-duplicate copies of it (ids ``<id>#<k>``)."""
    c = ML.fineweb2(lang)
    ids, texts = list(c.ids), list(c.texts)
    for k in range(1, copies):
        for i, t in zip(c.ids, c.texts, strict=True):
            ids.append(f"{i}#{k}")
            texts.append(ML._near_duplicate(t, random.Random(f"{lang}/{k}/{i}")))
    return ids, texts


def write_shards(folder: Path, ids: list[str], texts: list[str], shards: int) -> None:
    folder.mkdir(parents=True)
    n = len(ids)
    for r in range(shards):
        lo, hi = r * n // shards, (r + 1) * n // shards
        pq.write_table(
            pa.table({"id": ids[lo:hi], "text": texts[lo:hi]}), folder / f"{r:05d}.parquet"
        )


def to_record(_writer: Any, doc: Any) -> dict[str, Any]:
    return {"id": doc.id, "text": doc.text, "reason": doc.metadata.get("filter_reason")}


SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string()), ("reason", pa.string())])


def writer(folder: Path) -> Any:
    return ParquetWriter(
        str(folder), output_filename="${rank}.parquet", adapter=to_record, schema=SCHEMA
    )


def reader(folder: Path) -> Any:
    return ParquetReader(str(folder), glob_pattern="*.parquet")


class PhaseError(Exception):
    """An executor phase failed: which one, and the exception (with ``work`` paths masked)."""

    def __init__(self, phase: str, error: str) -> None:
        super().__init__(f"{phase}: {error}")
        self.phase, self.error = phase, error


def run_phase(name: str, steps: list[Any], tasks: int, workers: int, work: Path) -> float:
    t = time.perf_counter()
    try:
        LocalPipelineExecutor(
            pipeline=steps,
            tasks=tasks,
            workers=min(workers, tasks),
            logging_dir=str(work / "logs" / name),
            skip_completed=False,
            start_method="fork",
        ).run()
    except Exception as e:
        raise PhaseError(name, f"{type(e).__name__}: {e}".replace(str(work), "<work>")) from e
    return round(time.perf_counter() - t, 3)


def filter_step(mod: Any, lang: str, case: str, excluded: Path) -> Any:
    case, _, extra = case.partition("+")
    _, cls, kwargs = specs.fineweb2_filter_cases(lang)[case]
    if extra == "stopwords":
        kwargs = {**kwargs, "stop_words": list(specs.fineweb2_config(lang)["stopwords"])}
    return getattr(mod, cls)(**kwargs, exclusion_writer=writer(excluded))


def pipeline(
    impl: str, chain: str, lang: str, inp: Path, work: Path, workers: int
) -> dict[str, float]:
    mod: Any = PD if impl == "pyroid" else dt_filters
    steps = [filter_step(mod, lang, c, work / "excluded" / s) for s, c in CHAINS[chain]]
    phases = {
        "filters": run_phase(
            "filters", [reader(inp), *steps, writer(work / "kept")], workers, workers, work
        )
    }
    cfg = R.minhash_config(specs.MINHASH_CONFIG_FINEWEB2)
    sig_cls = PD.MinhashDedupSignature if impl == "pyroid" else dt_minhash.MinhashDedupSignature
    mh = work / "minhash"
    phases["minhash_signature"] = run_phase(
        "minhash_signature",
        [reader(work / "kept"), sig_cls(output_folder=str(mh / "sigs"), config=cfg, language=lang)],
        workers,
        workers,
        work,
    )
    phases["minhash_buckets"] = run_phase(
        "minhash_buckets",
        [
            dt_minhash.MinhashDedupBuckets(
                input_folder=str(mh / "sigs"), output_folder=str(mh / "buckets"), config=cfg
            )
        ],
        cfg.num_buckets,
        workers,
        work,
    )
    phases["minhash_cluster"] = run_phase(
        "minhash_cluster",
        [
            MinhashDedupCluster(
                input_folder=str(mh / "buckets"), output_folder=str(mh / "remove_ids"), config=cfg
            )
        ],
        1,
        1,
        work,
    )
    phases["minhash_filter"] = run_phase(
        "minhash_filter",
        [
            reader(work / "kept"),
            MinhashDedupFilter(
                input_folder=str(mh / "remove_ids"), exclusion_writer=writer(work / "dup")
            ),
            writer(work / "final"),
        ],
        workers,
        workers,
        work,
    )
    return phases


def records(folder: Path) -> list[dict[str, Any]]:
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    return [r for f in files for r in pq.read_table(f).to_pylist()]


def digest(work: Path, ids: list[str], stages: list[str]) -> tuple[str, dict[str, int]]:
    recs: dict[str, tuple[Any, ...]] = {}

    def put(r: dict[str, Any], keep: bool, reason: str | None, dup: bool) -> None:
        if r["id"] in recs:
            raise RuntimeError(f"doc {r['id']} appears in two outputs")
        recs[r["id"]] = (r["text"], keep, reason, dup)

    for s in stages:
        for r in records(work / "excluded" / s):
            put(r, False, f"{s}:{r['reason']}", False)
    for r in records(work / "final"):
        put(r, True, None, False)
    for r in records(work / "dup"):
        put(r, True, None, True)
    h = hashlib.sha256()
    for i in ids:
        row = [i, *recs.get(i, (None, None, "missing", None))]
        h.update(json.dumps(row, ensure_ascii=False).encode("utf-8", "surrogatepass") + b"\n")
    vals = list(recs.values())
    counts = {
        "kept": sum(1 for v in vals if v[1] and not v[3]),
        "dup": sum(1 for v in vals if v[3]),
        **{s: sum(1 for v in vals if v[2] and v[2].startswith(f"{s}:")) for s in stages},
        "missing": len(ids) - len(recs),
    }
    return h.hexdigest(), counts


def cpu_children() -> float:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--langs", default=",".join(specs.FINEWEB2_LANGUAGES))
    ap.add_argument("--chains", default="fineweb2,defaults", help=f"of {sorted(CHAINS)}")
    ap.add_argument("--copies", type=int, default=1, help="slice copies (1 = the slice only)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--impls", default="stock,pyroid", help="run order, e.g. pyroid,stock")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-log", help="append one JSON line per language and chain")
    a = ap.parse_args()
    out = Path(a.out)
    rc = 0
    for lang in a.langs.split(","):
        # The workers are forked: a FallbackWarning shows in their logs, not here.
        native = pyroid.filters.languages()[lang].native
        ids, texts = corpus(lang, a.copies)
        shutil.rmtree(out / lang, ignore_errors=True)
        inp = out / lang / "input"
        write_shards(inp, ids, texts, a.workers)
        for chain in a.chains.split(","):
            stages = [s for s, _ in CHAINS[chain]]
            row: dict[str, Any] = {
                "lang": lang,
                "chain": chain,
                "native": native,
                "docs": len(ids),
                "workers": a.workers,
                "loadavg_before": [round(x, 1) for x in os.getloadavg()],
            }
            for impl in a.impls.split(","):
                work = out / lang / f"{chain}-{impl}"
                cpu0, t0 = cpu_children(), time.perf_counter()
                res: dict[str, Any] = {}
                try:
                    res["phases"] = pipeline(impl, chain, lang, inp, work, a.workers)
                except PhaseError as e:  # compared with the other pipeline's
                    res["error"] = {"phase": e.phase, "error": e.error}
                res["wall_s"] = round(time.perf_counter() - t0, 3)
                res["cpu_s"] = round(cpu_children() - cpu0, 3)
                # A crashed filters phase leaves unfinished Parquet files: nothing to compare.
                crashed = res.get("error", {}).get("phase") == "filters"
                res["digest"], res["counts"] = (
                    (None, None) if crashed else digest(work, ids, stages)
                )
                row[impl] = res
                shutil.rmtree(work, ignore_errors=True)
            if "stock" in row and "pyroid" in row:
                st, py = row["stock"], row["pyroid"]
                err = st.get("error")
                row["identical"] = err == py.get("error") and st["digest"] == py["digest"]
                if err is None:
                    row["speedup_wall"] = round(st["wall_s"] / py["wall_s"], 2)
                rc |= 0 if row["identical"] else 1
            row["loadavg_after"] = [round(x, 1) for x in os.getloadavg()]
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)  # noqa: T201
            if a.summary_log:
                with open(a.summary_log, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        shutil.rmtree(inp, ignore_errors=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
