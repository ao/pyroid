"""Per-stage timings: NeMo Curator 1.4.0's CPU text stages vs pyroid.integrations.nemo_curator.

Each stage runs on the same DocumentBatches (FineWeb dev slice, ``--batch`` docs each), one worker,
in this process: NeMo's stage (``ScoreFilter`` / ``Modify``) with NeMo's filter or modifier, and
pyroid's stage with the same NeMo object (what pointing a YAML ``_target_`` at pyroid gives). The
outputs must be equal (whole DataFrames, dtypes included) before anything is timed; a mismatch
aborts. Times are the minimum over ``--repeat`` passes over all batches.

Needs NeMo Curator 1.4.0 and fasttext (for the stock FastTextLangId), so CPython 3.11 to 3.13::

    PYTHONHASHSEED=0 python bench/nemo_bench.py --docs 10000 --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyroid
import pyroid.integrations.nemo_curator as ours
from nemo_curator.stages.text import filters as nc_filters
from nemo_curator.stages.text import modifiers as nc_mod
from nemo_curator.stages.text.filters import heuristic as h
from nemo_curator.stages.text.filters.heuristic import repetition as r
from nemo_curator.tasks import DocumentBatch

DATA = os.environ.get("PYROID_DATA", "/data")


def heuristic_stages() -> list[tuple[str, Callable[[], Any]]]:
    """The filters of NeMo's heuristic_filter_english_pipeline.yaml, in its order and settings."""
    return [
        ("NonAlphaNumericFilter", lambda: h.NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=0.25)),
        ("SymbolsToWordsFilter", lambda: h.SymbolsToWordsFilter(max_symbol_to_word_ratio=0.1)),
        ("NumbersFilter", lambda: h.NumbersFilter(max_number_to_text_ratio=0.15)),
        ("UrlsFilter", lambda: h.UrlsFilter(max_url_to_text_ratio=0.2)),
        ("WhiteSpaceFilter", lambda: h.WhiteSpaceFilter(max_white_space_ratio=0.25)),
        ("ParenthesesFilter", lambda: h.ParenthesesFilter(max_parentheses_ratio=0.1)),
        ("BoilerPlateStringFilter", lambda: h.BoilerPlateStringFilter(True, 0.4)),
        ("RepeatedLinesFilter", lambda: r.RepeatedLinesFilter(max_repeated_line_fraction=0.7)),
        ("RepeatedParagraphsFilter", lambda: r.RepeatedParagraphsFilter(max_repeated_paragraphs_ratio=0.7)),
        ("RepeatedLinesByCharFilter", lambda: r.RepeatedLinesByCharFilter(max_repeated_lines_char_ratio=0.8)),
        (
            "RepeatedParagraphsByCharFilter",
            lambda: r.RepeatedParagraphsByCharFilter(max_repeated_paragraphs_char_ratio=0.8),
        ),
        ("WordCountFilter", lambda: h.WordCountFilter(min_words=50, max_words=100000)),
        ("PunctuationFilter", lambda: h.PunctuationFilter(max_num_sentences_without_endmark_ratio=0.85)),
        ("WordsWithoutAlphabetsFilter", lambda: h.WordsWithoutAlphabetsFilter(min_words_with_alphabets=0.8)),
        ("CommonEnglishWordsFilter", lambda: h.CommonEnglishWordsFilter(min_num_common_words=2)),
        ("MeanWordLengthFilter", lambda: h.MeanWordLengthFilter(3, 10)),
        ("LongWordFilter", lambda: h.LongWordFilter(max_word_length=1000)),
        ("EllipsisFilter", lambda: h.EllipsisFilter(max_num_lines_ending_with_ellipsis_ratio=0.3)),
        ("RepeatingTopNGramsFilter n=2", lambda: r.RepeatingTopNGramsFilter(2, 0.2)),
        ("RepeatingTopNGramsFilter n=3", lambda: r.RepeatingTopNGramsFilter(3, 0.18)),
        ("RepeatingTopNGramsFilter n=4", lambda: r.RepeatingTopNGramsFilter(4, 0.16)),
        *[
            (f"RepeatingDuplicateNGramsFilter n={n}", lambda n=n, c=c: r.RepeatingDuplicateNGramsFilter(n, c))
            for n, c in zip(range(5, 11), (0.15, 0.14, 0.13, 0.12, 0.11, 0.10), strict=True)
        ],
        ("BulletsFilter", lambda: h.BulletsFilter(max_bullet_lines_ratio=0.9)),
    ]


def modifier_stages() -> list[tuple[str, Callable[[], Any]]]:
    """NeMo's modifier benchmark stages (benchmarking/scripts/modifier_benchmark.py), in its order."""
    from nemo_curator.stages.text.modifiers.unicode import UnicodeReformatter

    return [
        ("BoilerPlateStringModifier", nc_mod.BoilerPlateStringModifier),
        ("MarkdownRemover", nc_mod.MarkdownRemover),
        ("NewlineNormalizer", nc_mod.NewlineNormalizer),
        ("QuotationRemover", nc_mod.QuotationRemover),
        ("UnicodeReformatter", UnicodeReformatter),
        ("UrlRemover", nc_mod.UrlRemover),
    ]


def batches(n_docs: int, size: int) -> list[pa.Table]:
    t = pq.read_table(f"{DATA}/fineweb/fineweb-dev-10k.parquet", columns=["id", "text"]).slice(0, n_docs)
    return [t.slice(i, size) for i in range(0, len(t), size)]


def run_all(stage: Any, tables: list[pa.Table]) -> list[pd.DataFrame]:
    stage.setup()
    return [stage.process(DocumentBatch(dataset_name="bench", data=t)).to_pandas() for t in tables]


def best(stage: Any, tables: list[pa.Table], repeat: int) -> float:
    stage.setup()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for t in tables:
            stage.process(DocumentBatch(dataset_name="bench", data=t))
        times.append(time.perf_counter() - t0)
    return min(times)


def check(name: str, a: list[pd.DataFrame], b: list[pd.DataFrame]) -> None:
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        try:
            assert x.dtypes.to_dict() == y.dtypes.to_dict()
            pd.testing.assert_frame_equal(x, y, check_exact=True)
        except AssertionError as e:
            sys.exit(f"{name}: batch {i} differs, not timing it: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs", type=int, default=10000)
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--repeat-stock", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--model", default=f"{DATA}/models/lid.176.bin")
    p.add_argument("--out")
    args = p.parse_args()
    load_start = os.getloadavg()
    tables = batches(args.docs, args.batch)
    rows = []

    def measure(name: str, stock: Any, mine: Any) -> None:
        check(name, run_all(stock, tables), run_all(mine, tables))
        s = best(stock, tables, args.repeat_stock)
        o = best(mine, tables, args.repeat)
        rows.append({"stage": name, "nemo_s": s, "pyroid_s": o, "speedup": s / o})
        print(f"{name:34} NeMo {s * 1e3:9.1f} ms   pyroid {o * 1e3:8.1f} ms   {s / o:6.1f}x", flush=True)

    for name, make in heuristic_stages():
        measure(name, nc_filters.ScoreFilter(make()), ours.ScoreFilter(make(), threads=args.threads))
    for name, make in modifier_stages():
        measure(name, nc_mod.Modify(make()), ours.Modify(make(), threads=args.threads))
    if os.path.exists(args.model):
        from nemo_curator.stages.text.filters.fasttext import FastTextLangId

        measure(
            "FastTextLangId (lid.176.bin)",
            nc_filters.ScoreFilter(FastTextLangId(args.model, 0.3)),
            ours.ScoreFilter(ours.FastTextLangId(args.model, 0.3), threads=args.threads),
        )
    # The 28 filter stages of the YAML as one chain of stages, batch by batch.
    chain_stock = [nc_filters.ScoreFilter(m()) for _, m in heuristic_stages()]
    chain_ours = [ours.ScoreFilter(m(), threads=args.threads) for _, m in heuristic_stages()]

    class Chain:
        def __init__(self, stages: list[Any]) -> None:
            self.stages = stages

        def setup(self) -> None:
            for s in self.stages:
                s.setup()

        def process(self, batch: Any) -> Any:
            for s in self.stages:
                batch = s.process(batch)
            return batch

    measure("heuristic_filter_english_pipeline", Chain(chain_stock), Chain(chain_ours))
    meta = {
        "where": "Docker (OrbStack VM), indicative",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "nproc": os.cpu_count(),
        "loadavg_start": load_start,
        "loadavg_end": os.getloadavg(),
        "pyroid": pyroid.__version__,
        "docs": args.docs,
        "batch": args.batch,
        "threads": args.threads,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
