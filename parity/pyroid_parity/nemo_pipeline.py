"""Run NeMo Curator 1.4.0's own pipelines with stock stages and with pyroid's, and compare their outputs.

    python -m pyroid_parity.nemo_pipeline all                         # each pipeline: stock twice, integration, patch
    python -m pyroid_parity.nemo_pipeline all --variants stock stock  # the stock-vs-stock reproducibility check
    python -m pyroid_parity.nemo_pipeline run heuristic --variant integration --tag integration-1
    python -m pyroid_parity.nemo_pipeline compare heuristic/stock-1 heuristic/integration-1

Runs in the NeMo env (README, "The nemo suite") on Linux, one pipeline at a time, each in a fresh process with a
fresh Ray cluster (``RayClient(num_cpus=--cpus)``, default 4) and ``PYTHONHASHSEED=0`` set before the cluster
starts (RepeatingTopNGramsFilter's tie-breaking follows the hash seed of the worker processes).

Pipelines:
    heuristic  NeMo's ``config/text/heuristic_filter_english_pipeline.yaml`` composed with hydra and built by
               NeMo's ``create_pipeline_from_yaml``: its 28 ScoreFilter stages, with ParquetReader (one task per
               input file) in place of JsonlReader and no writer, so the output DocumentBatches are compared
               directly. Each ScoreFilter gets a ``score_field`` (``--no-scores`` keeps NeMo's ``null``), so the
               scores of the kept documents are compared too.
    fasttext   ParquetReader, ScoreFilter(FastTextLangId(lid.176.bin, min_langid_score=0.3),
               score_field="language"), Score(FastTextQualityFilter(the parity quality model),
               score_field="quality"), then modifier_benchmark.py's stages in its order: Modify with
               BoilerPlateStringModifier, MarkdownRemover, NewlineNormalizer, QuotationRemover,
               UnicodeReformatter, UrlRemover. The quality filter scores only (Score, not ScoreFilter): its
               keep_document draws from numpy's global RNG, which NeMo seeds in the constructor, in the driver.

Variants:
    stock        the config as it is
    integration  every ``_target_`` whose class ``pyroid.integrations.nemo_curator`` has, rewritten to it
    patch        the stock config, after ``pyroid.patch("nemo_curator")``

Input: dev 10k + adversarial 3,232 as parquet files (id, text) under ``<state>/in``, 8 and 3 files. Two
adversarial documents ("\\n" and "\\n\\n\\n") are left out of both pipelines: NeMo's NonAlphaNumericFilter keeps
them (its alphanumeric class includes "\\n"), then SymbolsToWordsFilter raises ZeroDivisionError on them, which
fails the whole stock pipeline. The excluded ids come from the pinned per-document references.

Each run writes ``<state>/runs/<pipeline>/<tag>/run.json`` (wall time, NeMo's ``_stage_perf`` per stage, the
classes the driver built, rewritten targets, columns and dtypes, kept-id and output digests) and ``rows.parquet``
(one row per output document: id and a digest of every cell). ``compare`` reports "identical" or the precise
difference: ids only in one run, columns, dtypes, and per column the rows whose values differ (with the values
when both runs kept ``output.parquet``). For the heuristic pipeline each run also checks its kept ids against the
per-document references (the documents that pass all 28 filters in order). ``$PYROID_NEMO_PIPELINE`` sets
``<state>`` (default /target/parity/pipeline).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import importlib.resources
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pyroid_parity import data as D
from pyroid_parity import nemo_cases as C

STATE = Path(os.environ.get("PYROID_NEMO_PIPELINE", "/target/parity/pipeline"))
PIPELINES = ("heuristic", "fasttext")
VARIANTS = ("stock", "integration", "patch")
INTEGRATION = "pyroid.integrations.nemo_curator"
READER = "nemo_curator.stages.text.io.reader.ParquetReader"
PARTS = {"dev": 8, "adversarial": 3}
VERSIONS = ("nemo-curator", "pyroid", "ray", "cosmos-xenna", "pandas", "pyarrow", "numpy", "fasttext", "ftfy")


def log(msg: str) -> None:
    print(f"[nemo-pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for d in VERSIONS:
        try:
            out[d] = md.version(d)
        except md.PackageNotFoundError:
            pass
    return out


# ------------------------------------------------------------------------------------------------ inputs


def heuristic_expected() -> tuple[list[str], list[str]]:
    """``(kept ids, excluded ids)`` of the heuristic pipeline, from the pinned per-document references.

    A document goes through the 28 filters in order: the first ``keep`` False drops it, the first exception
    would fail the stock pipeline (excluded from the inputs).
    """
    from pyroid_parity import nemo_compare as NC

    docs = NC.all_docs()
    rows = [NC.ref_rows(c.case) for c in C.HEURISTIC_PIPELINE]
    kept, excluded = [], []
    for i in docs.ids:
        for r in rows:
            row = r[i]
            if row["outcome"] == "raise":
                excluded.append(i)
                break
            if not row["keep"]:
                break
        else:
            kept.append(i)
    return kept, excluded


def ensure_inputs(state: Path) -> tuple[Path, list[str], list[str]]:
    """Write the input parquet files once; return ``(directory, input ids in order, excluded ids)``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pyroid_parity import nemo_compare as NC

    _, excluded = heuristic_expected()
    skip = set(excluded)
    docs = NC.all_docs()
    ids = [i for i in docs.ids if i not in skip]
    d = state / "in"
    marker = d / "inputs.json"
    want = {"ids_sha256": hashlib.sha256("\0".join(ids).encode()).hexdigest(), "excluded": excluded}
    if marker.exists() and json.loads(marker.read_text()) == want:
        return d, ids, excluded
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    n_dev = len(D.corpus("dev"))
    for name, lo, hi in (("dev", 0, n_dev), ("adversarial", n_dev, len(docs.ids))):
        rows = [(docs.ids[k], docs.texts[k]) for k in range(lo, hi) if docs.ids[k] not in skip]
        n = PARTS[name]
        size = -(-len(rows) // n)
        for p in range(n):
            part = rows[p * size : (p + 1) * size]
            table = pa.table({"id": [r[0] for r in part], "text": [r[1] for r in part]})
            pq.write_table(table, d / f"{name}-{p:02d}.parquet", compression="zstd")
    marker.write_text(json.dumps(want))
    log(f"wrote {sum(PARTS.values())} input files ({len(ids)} documents, {len(excluded)} excluded) to {d}")
    return d, ids, excluded


# ------------------------------------------------------------------------------------------------ configs


def heuristic_config(in_dir: Path, scores: bool) -> Any:
    """NeMo's heuristic YAML via hydra, with ParquetReader and no writer."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    # Through the distribution, not the module: under pyroid.patch("nemo_curator") the module may be a shim.
    conf = Path(str(md.distribution("nemo-curator").locate_file("nemo_curator/config/text")))
    with initialize_config_dir(config_dir=str(conf.resolve()), version_base=None):
        cfg = compose(
            config_name="heuristic_filter_english_pipeline",
            overrides=[f"input_path={in_dir}", f"output_path={STATE / 'unused-output'}"],
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        stages = list(cfg.stages)
        assert stages[0]["_target_"].endswith(".JsonlReader"), stages[0]
        assert stages[-1]["_target_"].endswith(".JsonlWriter"), stages[-1]
        stages[0] = OmegaConf.create(
            {"_target_": READER, "file_paths": str(in_dir), "files_per_partition": 1}
        )
        stages = stages[:-1]
        if scores:
            for k, s in enumerate(stages[1:]):
                name = s.filter_obj["_target_"].rsplit(".", 1)[1].removesuffix("Filter")
                n = s.filter_obj.get("n")
                s.score_field = f"s{k:02d}_{name}{n if n is not None else ''}"
        cfg.stages = stages
    return cfg


def fasttext_config(in_dir: Path) -> Any:
    from omegaconf import OmegaConf

    filters = "nemo_curator.stages.text.filters"
    mods = "nemo_curator.stages.text.modifiers"
    q = C.FASTTEXT_CASES["quality"].nemo_kwargs
    stages: list[dict[str, Any]] = [
        {"_target_": READER, "file_paths": str(in_dir), "files_per_partition": 1},
        {
            "_target_": f"{filters}.score_filter.ScoreFilter",
            "filter_obj": {
                "_target_": f"{filters}.fasttext.fasttext_filters.FastTextLangId",
                "model_path": str(D.path(C.LID_MODELS["lid.176.bin"])),
                "min_langid_score": 0.3,
            },
            "text_field": "text",
            "score_field": "language",
        },
        {
            "_target_": f"{filters}.score_filter.Score",
            "score_fn": {
                "_target_": f"{filters}.fasttext.fasttext_filters.FastTextQualityFilter",
                "model_path": str(D.path(C.QUALITY_MODEL)),
                **q,
            },
            "score_field": "quality",
            "text_field": "text",
        },
    ]
    modifiers = (
        "string.c4.BoilerPlateStringModifier",
        "string.markdown_remover.MarkdownRemover",
        "string.newline_normalizer.NewlineNormalizer",
        "string.quotation_remover.QuotationRemover",
        "unicode.unicode_reformatter.UnicodeReformatter",
        "string.url_remover.UrlRemover",
    )
    stages += [
        {"_target_": f"{mods}.modifier.Modify", "modifier_fn": {"_target_": f"{mods}.{m}"}} for m in modifiers
    ]
    return OmegaConf.create({"stages": stages})


def rewrite_targets(cfg: Any, module: Any) -> tuple[list[list[str]], list[str]]:
    """Point every ``_target_`` whose class ``module`` has at it, in place. Returns (rewritten, kept)."""
    rewritten, kept = [], []

    def walk(node: Any) -> None:
        if hasattr(node, "items"):
            for k, v in list(node.items()):
                if k == "_target_":
                    name = v.rsplit(".", 1)[1]
                    if hasattr(module, name):
                        node[k] = f"{module.__name__}.{name}"
                        rewritten.append([v, node[k]])
                    else:
                        kept.append(v)
                else:
                    walk(v)
        elif isinstance(node, (list, tuple)) or type(node).__name__ == "ListConfig":
            for v in node:
                walk(v)

    walk(cfg)
    return rewritten, kept


def _targets(cfg: Any) -> list[str]:
    out: list[str] = []

    def walk(node: Any) -> None:
        if hasattr(node, "items"):
            for k, v in node.items():
                out.append(v) if k == "_target_" else walk(v)
        elif isinstance(node, (list, tuple)) or type(node).__name__ == "ListConfig":
            for v in node:
                walk(v)

    walk(cfg)
    return out


def _classes(pipe: Any) -> list[dict[str, Any]]:
    """The classes the driver instantiated: each stage and the filters / modifiers it holds."""

    def name(o: Any) -> str:
        return f"{type(o).__module__}.{type(o).__qualname__}"

    out = []
    for s in pipe.stages:
        inner = [
            name(o)
            for attr in ("filter_obj", "score_fn", "filter_fn", "modifier_fn")
            for o in (getattr(s, attr, None) or [])
        ]
        out.append({"stage": name(s), "holds": inner})
    return out


# ------------------------------------------------------------------------------------------------ one run


def _cell(x: Any) -> str:
    if x is None:
        return "None:"
    return f"{type(x).__name__}:{x if isinstance(x, str) else repr(x)}"


def _digest(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).hexdigest()


def _run_dir(pipeline: str, tag: str) -> Path:
    return STATE / "runs" / pipeline / tag


def run(
    pipeline: str, variant: str, tag: str, executor: str, cpus: int, scores: bool, keep_output: bool
) -> dict[str, Any]:
    """One pipeline run in this process (``PYTHONHASHSEED=0`` already set): writes run.json and rows.parquet."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = _run_dir(pipeline, tag)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    record: dict[str, Any] = {
        "pipeline": pipeline,
        "variant": variant,
        "tag": tag,
        "executor": executor,
        "cpus": cpus,
        "hash_seed": os.environ.get(C.HASH_SEED_ENV),
        "platform": D.platform_key(),
        "versions": _versions(),
    }
    if variant == "patch":  # first, before anything imports nemo_curator
        try:
            pyroid = importlib.import_module("pyroid")
        except ImportError as e:
            return _skipped(out_dir, record, f"pyroid does not import: {e}")
        if not hasattr(pyroid, "patch"):
            return _skipped(out_dir, record, "pyroid has no patch()")
        try:
            record["patch"] = repr(pyroid.patch("nemo_curator"))
        except Exception as e:  # noqa: BLE001  a pyroid.patch that does not know nemo_curator
            return _skipped(out_dir, record, f'pyroid.patch("nemo_curator") raised {type(e).__name__}: {e}')
    in_dir, in_ids, excluded = ensure_inputs(STATE)
    record.update(n_in=len(in_ids), excluded=excluded)
    cfg = heuristic_config(in_dir, scores) if pipeline == "heuristic" else fasttext_config(in_dir)
    if pipeline == "heuristic":
        record["score_fields"] = scores
    if variant == "integration":
        try:
            module = importlib.import_module(INTEGRATION)
        except ImportError as e:
            return _skipped(out_dir, record, f"{INTEGRATION} does not import: {e}")
        record["rewritten"], record["not_rewritten"] = rewrite_targets(cfg, module)
    record["targets"] = _targets(cfg)

    from nemo_curator.config.run import create_pipeline_from_yaml
    from nemo_curator.core.client import RayClient

    pipe = create_pipeline_from_yaml(cfg, log_config=False)
    record["classes"] = _classes(pipe)
    # `ray start` (RayClient) and its workers must be this interpreter, so that they import the same packages.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    client = RayClient(num_cpus=cpus, include_dashboard=False)
    client.start()
    try:
        ex = _executor(executor)
        t0 = time.perf_counter()
        tasks = pipe.run(ex) or []
        record["wall_s"] = round(time.perf_counter() - t0, 3)
    finally:
        client.stop()
    record["n_tasks"] = len(tasks)
    record["stage_perf"] = _stage_perf(tasks)
    frames = [t.to_pandas() for t in tasks]
    dtypes: dict[str, list[str]] = {}
    for f in frames:
        for col, dt in f.dtypes.items():
            if str(dt) not in dtypes.setdefault(col, []):
                dtypes[col].append(str(dt))
    record["dtypes"] = dtypes
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    order = {i: k for k, i in enumerate(in_ids)}
    out_ids = df["id"].tolist() if len(df) else []
    df = df.iloc[sorted(range(len(df)), key=lambda k: order[out_ids[k]])].reset_index(drop=True)
    cols = list(df.columns)
    record["columns"] = cols
    record["n_out"] = len(df)
    values = {c: df[c].tolist() for c in cols}
    cells = {c: [_digest(_cell(v)) for v in values[c]] for c in cols}
    kept = values.get("id", [])
    record["kept_ids_sha256"] = hashlib.sha256("\0".join(kept).encode()).hexdigest()
    h = hashlib.sha256(json.dumps({c: dtypes.get(c) for c in cols}, sort_keys=True).encode())
    for k in range(len(df)):
        h.update("\0".join(cells[c][k] for c in cols).encode())
    record["output_sha256"] = h.hexdigest()
    if pipeline == "heuristic":
        expected, _ = heuristic_expected()
        expected = [i for i in expected if i in order]
        record["kept_as_references"] = kept == expected
        if kept != expected:
            only_run, only_ref = sorted(set(kept) - set(expected)), sorted(set(expected) - set(kept))
            record["kept_vs_references"] = {"only_run": only_run[:20], "only_references": only_ref[:20]}
    pq.write_table(pa.table({c: cells[c] for c in cols}), out_dir / "rows.parquet", compression="zstd")
    (out_dir / "kept.json").write_text(json.dumps(kept))
    if keep_output:
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False), out_dir / "output.parquet", compression="zstd"
        )
    (out_dir / "run.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    log(
        f"{pipeline}/{tag} ({variant}): {record['n_out']} of {record['n_in']} kept in {record['wall_s']} s, "
        f"output {record['output_sha256'][:16]}"
    )
    return record


def _skipped(out_dir: Path, record: dict[str, Any], why: str) -> dict[str, Any]:
    record["skipped"] = why
    (out_dir / "run.json").write_text(json.dumps(record, indent=1) + "\n")
    log(f"{record['pipeline']}/{record['tag']} ({record['variant']}): skipped, {why}")
    return record


def _executor(name: str) -> Any:
    if name == "xenna":
        from nemo_curator.backends.xenna import XennaExecutor

        return XennaExecutor(config={"execution_mode": "batch"})
    from nemo_curator.backends.ray_data import RayDataExecutor

    return RayDataExecutor()


def _stage_perf(tasks: list[Any]) -> list[dict[str, Any]]:
    """NeMo's per-stage StagePerfStats summed over the output tasks, in pipeline order."""
    per: dict[str, dict[str, Any]] = {}
    for t in tasks:
        for p in t._stage_perf:
            s = per.setdefault(
                p.stage_name,
                {
                    "stage": p.stage_name,
                    "tasks": 0,
                    "process_time": 0.0,
                    "actor_idle_time": 0.0,
                    "num_items_processed": 0,
                    "input_data_size_mb": 0.0,
                },
            )
            s["tasks"] += 1
            s["process_time"] += p.process_time
            s["actor_idle_time"] += p.actor_idle_time
            s["num_items_processed"] += p.num_items_processed
            s["input_data_size_mb"] += p.input_data_size_mb
    for s in per.values():
        for k in ("process_time", "actor_idle_time", "input_data_size_mb"):
            s[k] = round(s[k], 4)
    return list(per.values())


# --------------------------------------------------------------------------------------------- compare


def compare(a: str, b: str) -> dict[str, Any]:
    """Compare two runs (``<pipeline>/<tag>``); ``identical`` is True when kept ids, columns, dtypes and every
    cell match."""
    import pyarrow.parquet as pq

    ra, rb = (json.loads((STATE / "runs" / x / "run.json").read_text()) for x in (a, b))
    out: dict[str, Any] = {"a": a, "b": b}
    if ra.get("skipped") or rb.get("skipped"):
        out["identical"] = None
        out["skipped"] = ra.get("skipped") or rb.get("skipped")
        return out
    ta, tb = (pq.read_table(STATE / "runs" / x / "rows.parquet").to_pydict() for x in (a, b))
    out["columns"] = None if ra["columns"] == rb["columns"] else {"a": ra["columns"], "b": rb["columns"]}
    dt = {c: (ra["dtypes"].get(c), rb["dtypes"].get(c)) for c in set(ra["dtypes"]) | set(rb["dtypes"])}
    out["dtypes"] = {c: {"a": x, "b": y} for c, (x, y) in sorted(dt.items()) if x != y} or None
    out["n_out"] = [ra["n_out"], rb["n_out"]]
    # rows.parquet holds a digest per cell, in the order of kept.json.
    kept_a, kept_b = _kept(a), _kept(b)
    set_a, set_b = set(kept_a), set(kept_b)
    only_a, only_b = [i for i in kept_a if i not in set_b], [i for i in kept_b if i not in set_a]
    out["ids_only_in_a"] = {"count": len(only_a), "first": only_a[:10]} if only_a else None
    out["ids_only_in_b"] = {"count": len(only_b), "first": only_b[:10]} if only_b else None
    pos_b = {i: k for k, i in enumerate(kept_b)}
    cells: dict[str, Any] = {}
    for col in [c for c in ra["columns"] if c in rb["columns"]]:
        bad = [i for k, i in enumerate(kept_a) if i in pos_b and ta[col][k] != tb[col][pos_b[i]]]
        if bad:
            cells[col] = {"count": len(bad), "first": bad[:5], "values": _values(a, b, col, bad[:3])}
    out["cells"] = cells or None
    out["identical"] = ra["output_sha256"] == rb["output_sha256"] and not (
        out["columns"] or out["dtypes"] or only_a or only_b or cells
    )
    # Documents each stage processed, where the two runs differ: [stage a, stage b, items a, items b].
    out["stage_items"] = [
        [s["stage"], t["stage"], s["num_items_processed"], t["num_items_processed"]]
        for s, t in zip(ra["stage_perf"], rb["stage_perf"], strict=False)
        if s["num_items_processed"] != t["num_items_processed"] or s["stage"] != t["stage"]
    ] or None
    out["wall_s"] = [ra["wall_s"], rb["wall_s"]]
    return out


def _kept(run_name: str) -> list[str]:
    path = STATE / "runs" / run_name / "kept.json"
    return json.loads(path.read_text())


def _values(a: str, b: str, col: str, ids: list[str]) -> list[Any] | None:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pa_, pb_ = (STATE / "runs" / x / "output.parquet" for x in (a, b))
    if not (pa_.exists() and pb_.exists()):
        return None
    out = []
    for i in ids:
        row = []
        for p in (pa_, pb_):
            t = pq.read_table(p, columns=["id", col])
            row.append(t.filter(pc.equal(t["id"], i))[col].to_pylist())
        out.append({"id": i, "a": repr(row[0])[:300], "b": repr(row[1])[:300]})
    return out


# ------------------------------------------------------------------------------------------------- main


def _subprocess_run(pipeline: str, variant: str, tag: str, a: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "-m",
        "pyroid_parity.nemo_pipeline",
        "run",
        pipeline,
        "--variant",
        variant,
        "--tag",
        tag,
    ]
    cmd += ["--executor", a.executor, "--cpus", str(a.cpus)]
    cmd += ["--no-scores"] if not a.scores else []
    cmd += ["--keep-output"] if a.keep_output or a.cmd == "all" else []
    env = {**os.environ, C.HASH_SEED_ENV: C.HASH_SEED}
    subprocess.run(cmd, env=env, check=True, cwd=Path(__file__).resolve().parents[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m pyroid_parity.nemo_pipeline", description=__doc__.split("\n\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "all"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("pipeline", choices=PIPELINES)
            p.add_argument("--variant", choices=VARIANTS, default="stock")
            p.add_argument("--tag", required=True)
        else:
            p.add_argument("--pipelines", nargs="+", choices=PIPELINES, default=list(PIPELINES))
            p.add_argument(
                "--variants", nargs="+", choices=VARIANTS, default=["stock", "stock", "integration", "patch"]
            )
            p.add_argument("--keep", action="store_true", help="keep output.parquet of every run")
        p.add_argument("--executor", choices=("xenna", "ray_data"), default="xenna")
        p.add_argument("--cpus", type=int, default=4)
        p.add_argument(
            "--no-scores", dest="scores", action="store_false", help="heuristic: NeMo's score_field: null"
        )
        p.add_argument("--keep-output", action="store_true", help="also write output.parquet (the full rows)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args(argv)
    if a.cmd == "compare":
        print(json.dumps(compare(a.a, a.b), indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "run":
        if not C.hash_seed_ok():
            env = {**os.environ, C.HASH_SEED_ENV: C.HASH_SEED}
            return subprocess.run(
                [sys.executable, "-m", "pyroid_parity.nemo_pipeline", *(argv or sys.argv[1:])],
                env=env,
                check=False,
            ).returncode
        run(a.pipeline, a.variant, a.tag, a.executor, a.cpus, a.scores, a.keep_output)
        return 0
    return run_all(a)


def run_all(a: argparse.Namespace) -> int:
    """Each pipeline: every variant in turn (stock first), then each later run compared with the first stock."""
    summary: dict[str, Any] = {
        "versions": _versions(),
        "executor": a.executor,
        "cpus": a.cpus,
        "pipelines": {},
    }
    status = 0
    for pipeline in a.pipelines:
        tags: list[tuple[str, str]] = []
        seen: dict[str, int] = {}
        for v in a.variants:
            seen[v] = seen.get(v, 0) + 1
            tag = f"{v}-{seen[v]}"
            log(f"{pipeline}/{tag} ...")
            _subprocess_run(pipeline, v, tag, a)
            tags.append((v, tag))
        base = next(t for v, t in tags if v == "stock")
        results = {}
        for _, tag in tags:
            rec = json.loads((_run_dir(pipeline, tag) / "run.json").read_text())
            entry: dict[str, Any] = {
                k: rec.get(k)
                for k in (
                    "variant",
                    "wall_s",
                    "n_in",
                    "n_out",
                    "output_sha256",
                    "kept_as_references",
                    "skipped",
                )
            }
            if tag != base:
                entry["vs_" + base] = compare(f"{pipeline}/{base}", f"{pipeline}/{tag}")
                if entry["vs_" + base]["identical"] is False:
                    status = 1
            results[tag] = entry
        summary["pipelines"][pipeline] = results
    if not a.keep:
        for p in (STATE / "runs").glob("*/*/output.parquet"):
            p.unlink()
    out = STATE / "summary.json"
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    log(f"summary in {out}")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return status


if __name__ == "__main__":
    sys.exit(main())
