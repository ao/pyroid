#!/usr/bin/env python3
"""datatrove's FineWeb pipeline (``examples/fineweb.py``) after text extraction, on pyroid's steps.

The only change from a stock datatrove pipeline is the import block: ``--stock`` runs the same
pipeline on datatrove's own steps, for comparison. Input: Parquet or JSONL with a ``text`` column
(e.g. a FineWeb sample); output: ``out/output`` (kept), ``out/removed/<stage>`` (dropped, with
``filter_reason``), and the MinHash stages' files.

    pip install 'pyroid[datatrove]'
    python fineweb.py data/ out/ --tasks 8 --workers 8 --lid-model lid.176.bin
"""

from __future__ import annotations

import argparse
import os
import time

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.readers import JsonlReader, ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", help="a folder of .parquet or .jsonl[.gz] files")
    ap.add_argument("output")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lid-model", help="local lid.176.bin (default: download, as datatrove does)")
    ap.add_argument(
        "--stock", action="store_true", help="datatrove's own steps instead of pyroid's"
    )
    a = ap.parse_args()

    if a.stock:
        from datatrove.pipeline.dedup import MinhashDedupSignature
        from datatrove.pipeline.filters import (
            C4QualityFilter,
            FineWebQualityFilter,
            GopherQualityFilter,
            GopherRepetitionFilter,
            LanguageFilter,
        )

        lid_kw = {}
    else:
        from pyroid.integrations.datatrove import (
            C4QualityFilter,
            FineWebQualityFilter,
            GopherQualityFilter,
            GopherRepetitionFilter,
            LanguageFilter,
            MinhashDedupSignature,
        )

        lid_kw = {"model_path": a.lid_model}

    out = a.output.rstrip("/")
    parquet = any(n.endswith(".parquet") for n in os.listdir(a.input))
    reader = ParquetReader if parquet else JsonlReader
    t0 = time.perf_counter()
    LocalPipelineExecutor(
        pipeline=[
            reader(a.input),
            LanguageFilter(exclusion_writer=JsonlWriter(f"{out}/removed/1_non_english"), **lid_kw),
            GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{out}/removed/2_gopher_rep")),
            GopherQualityFilter(exclusion_writer=JsonlWriter(f"{out}/removed/3_gopher_qual")),
            C4QualityFilter(
                filter_no_terminal_punct=False, exclusion_writer=JsonlWriter(f"{out}/removed/4_c4")
            ),
            FineWebQualityFilter(exclusion_writer=JsonlWriter(f"{out}/removed/5_fineweb_qual")),
            JsonlWriter(f"{out}/filtered"),
        ],
        tasks=a.tasks,
        workers=a.workers,
        logging_dir=f"{out}/logs/filtering",
    ).run()

    # MinHash, as datatrove's examples/fineweb.py configures it.
    config = MinhashConfig(
        hash_config=HashConfig(hash_fc="sha1", precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
        n_grams=5,
    )
    stages = [
        (
            [
                JsonlReader(f"{out}/filtered"),
                MinhashDedupSignature(output_folder=f"{out}/minhash/signatures", config=config),
            ],
            a.tasks,
        ),
        (
            [
                MinhashDedupBuckets(
                    input_folder=f"{out}/minhash/signatures",
                    output_folder=f"{out}/minhash/buckets",
                    config=config,
                )
            ],
            config.num_buckets,
        ),
        (
            [
                MinhashDedupCluster(
                    input_folder=f"{out}/minhash/buckets",
                    output_folder=f"{out}/minhash/remove_ids",
                    config=config,
                )
            ],
            1,
        ),
        (
            [
                JsonlReader(f"{out}/filtered"),
                MinhashDedupFilter(
                    input_folder=f"{out}/minhash/remove_ids",
                    exclusion_writer=JsonlWriter(f"{out}/removed/6_minhash"),
                ),
                JsonlWriter(f"{out}/output"),
            ],
            a.tasks,
        ),
    ]
    for i, (pipeline, tasks) in enumerate(stages, 1):
        LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=tasks,
            workers=min(a.workers, tasks),
            logging_dir=f"{out}/logs/minhash{i}",
        ).run()
    print(f"{'datatrove' if a.stock else 'pyroid'} steps: {time.perf_counter() - t0:.1f} s")  # noqa: T201


if __name__ == "__main__":
    main()
