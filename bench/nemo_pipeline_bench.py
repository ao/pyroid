"""NeMo Curator's heuristic_filter_english_pipeline.yaml, stock vs pyroid's stages, at N CPUs.

The pipeline is NeMo's own YAML, composed with hydra as NeMo's ``score_filter_benchmark.py`` does,
with a ParquetReader (one task per input file) in place of the JsonlReader and no writer. The
``pyroid`` variant points every ``_target_`` that ``pyroid.integrations.nemo_curator`` has at it.
Each run is a fresh process with a fresh Ray cluster (``RayClient(num_cpus=N)``) and NeMo's
XennaExecutor in batch mode, with ``PYTHONHASHSEED=0`` (``RepeatingTopNGramsFilter`` breaks ties by
the hash seed, so only a fixed seed makes two runs comparable). The kept documents of the two
variants must be identical (ids and text); a mismatch is reported and fails the run.

    python bench/nemo_pipeline_bench.py split --docs 100000 --files 64
    python bench/nemo_pipeline_bench.py run --variant stock --cpus 8
    python bench/nemo_pipeline_bench.py all --cpus 1 8 --out results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
STATE = Path(os.environ.get("PYROID_NEMO_BENCH", "/target/bench-nemo"))
READER = "nemo_curator.stages.text.io.reader.ParquetReader"


def split(docs: int, files: int) -> None:
    import pyarrow.parquet as pq

    t = pq.read_table(DATA / "fineweb/fineweb-100k.parquet", columns=["id", "text"]).slice(0, docs)
    d = STATE / "in"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.parquet"):
        old.unlink()
    size = -(-len(t) // files)
    for p in range(files):
        pq.write_table(t.slice(p * size, size), d / f"part-{p:03d}.parquet", compression="zstd")
    print(f"{len(t)} documents in {files} files under {d}")


def config(variant: str) -> Any:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    conf = Path(str(md.distribution("nemo-curator").locate_file("nemo_curator/config/text")))
    with initialize_config_dir(config_dir=str(conf.resolve()), version_base=None):
        cfg = compose(
            config_name="heuristic_filter_english_pipeline",
            overrides=[f"input_path={STATE / 'in'}", f"output_path={STATE / 'unused'}"],
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        stages = list(cfg.stages)
        stages[0] = OmegaConf.create(
            {"_target_": READER, "file_paths": str(STATE / "in"), "files_per_partition": 1}
        )
        cfg.stages = stages[:-1]  # no writer
    if variant == "pyroid":
        import pyroid.integrations.nemo_curator as ours

        def walk(node: Any) -> None:
            if hasattr(node, "items"):
                for k, v in list(node.items()):
                    if k == "_target_" and hasattr(ours, v.rsplit(".", 1)[1]):
                        node[k] = f"pyroid.integrations.nemo_curator.{v.rsplit('.', 1)[1]}"
                    else:
                        walk(v)
            elif type(node).__name__ == "ListConfig" or isinstance(node, list):
                for v in node:
                    walk(v)

        walk(cfg)
    return cfg


def run_one(variant: str, cpus: int) -> dict[str, Any]:
    from nemo_curator.backends.xenna import XennaExecutor
    from nemo_curator.config.run import create_pipeline_from_yaml
    from nemo_curator.core.client import RayClient

    pipe = create_pipeline_from_yaml(config(variant), log_config=False)
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    client = RayClient(num_cpus=cpus, include_dashboard=False)
    client.start()
    try:
        t0 = time.perf_counter()
        # cpu_allocation_percentage 1.0: N CPUs give N workers (the default 0.95 leaves 1 CPU unusable).
        ex = XennaExecutor(config={"execution_mode": "batch", "cpu_allocation_percentage": 1.0})
        tasks = pipe.run(ex) or []
        wall = time.perf_counter() - t0
    finally:
        client.stop()
    frames = [t.to_pandas() for t in tasks]
    rows = sorted((r.id, r.text) for f in frames for r in f[["id", "text"]].itertuples())
    digest = hashlib.sha256("\0".join(f"{i}\1{t}" for i, t in rows).encode()).hexdigest()
    per: dict[str, float] = {}
    for t in tasks:
        for p in t._stage_perf:
            per[p.stage_name] = per.get(p.stage_name, 0.0) + p.process_time
    filters = sum(v for k, v in per.items() if "reader" not in k.lower() and "partition" not in k.lower())
    return {
        "variant": variant,
        "cpus": cpus,
        "wall_s": round(wall, 2),
        "stage_process_s": round(filters, 2),
        "kept": len(rows),
        "digest": digest,
        "per_stage_s": {k: round(v, 3) for k, v in per.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--docs", type=int, default=100000)
    s.add_argument("--files", type=int, default=64)
    r = sub.add_parser("run")
    r.add_argument("--variant", choices=("stock", "pyroid"), required=True)
    r.add_argument("--cpus", type=int, default=8)
    a = sub.add_parser("all")
    a.add_argument("--cpus", type=int, nargs="+", default=[1, 8])
    a.add_argument("--out")
    args = p.parse_args()
    if args.cmd == "split":
        split(args.docs, args.files)
    elif args.cmd == "run":
        print(json.dumps(run_one(args.variant, args.cpus)))
    else:
        results = []
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        for cpus in args.cpus:
            for variant in ("stock", "pyroid"):
                load = os.getloadavg()
                out = subprocess.run(
                    [sys.executable, __file__, "run", "--variant", variant, "--cpus", str(cpus)],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                rec = json.loads(out.stdout.strip().splitlines()[-1])
                rec["loadavg_before"] = load
                results.append(rec)
                print(
                    f"{cpus} cpus {variant:7} wall {rec['wall_s']:8.1f} s  stages {rec['stage_process_s']:8.1f} s  "
                    f"kept {rec['kept']}",
                    flush=True,
                )
            a_, b_ = results[-2], results[-1]
            if a_["digest"] != b_["digest"] or a_["kept"] != b_["kept"]:
                sys.exit(f"{cpus} cpus: pyroid's kept documents differ from NeMo's")
        meta = {
            "where": "Docker (OrbStack VM), indicative",
            "machine": platform.machine(),
            "python": platform.python_version(),
            "nproc": os.cpu_count(),
            "nemo_curator": md.version("nemo-curator"),
            "pyroid": md.version("pyroid"),
            "executor": "XennaExecutor(execution_mode=batch, cpu_allocation_percentage=1.0)",
            "input": f"FineWeb sample/10BT, {sum(1 for _ in (STATE / 'in').glob('*.parquet'))} parquet files",
        }
        if args.out:
            Path(args.out).write_text(json.dumps({"meta": meta, "runs": results}, indent=1))


if __name__ == "__main__":
    main()
