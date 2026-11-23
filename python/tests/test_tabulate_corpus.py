"""pyroid.tabulate against tabulate 0.10.0 on real tables: 48 public CSV files.

The files (seaborn-data, vega-datasets and datahub datasets, pinned to commits) are listed with
their sha256 in ``parity/data/tabulate-corpus/manifest.json``; fetch them with
``python bench/tabulate/fetch_corpus.py DIR`` and point ``PYROID_TABULATE_CORPUS`` at ``DIR``.
Without them, or without tabulate 0.10.0, the module is skipped.

Each file (its first 3,000 rows) is drawn in every format with ``headers="firstrow"``, in a few
formats with other options, and as a pandas DataFrame through ``DataFrame.to_markdown()``.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "parity" / "data" / "tabulate-corpus" / "manifest.json"
CORPUS = Path(os.environ.get("PYROID_TABULATE_CORPUS", "/nonexistent"))
if not (MANIFEST.is_file() and CORPUS.is_dir()):
    pytest.skip("set PYROID_TABULATE_CORPUS to the fetched corpus", allow_module_level=True)

ref: Any = pytest.importorskip("tabulate")
import pyroid.tabulate as pt  # noqa: E402

if getattr(ref, "__pyroid__", False) or ref.__version__ != pt.TABULATE_VERSION:
    pytest.skip(f"needs the real tabulate {pt.TABULATE_VERSION}", allow_module_level=True)

FILES = sorted(json.loads(MANIFEST.read_text(encoding="utf-8"))["files"].items())
MAX_ROWS = 3000
VARIANTS: list[dict[str, Any]] = [
    {"floatfmt": ".2f", "intfmt": ","},
    {"numalign": "right", "stralign": "center"},
    {"disable_numparse": True},
    {"showindex": "always", "missingval": "-"},
    {"colalign": ["right", "left", "center"], "floatfmt": ".3e"},
]


def _rows(name: str, entry: dict[str, Any]) -> list[list[str]]:
    path = CORPUS / f"{name}.csv"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], name
    with path.open(encoding="utf-8", newline="") as f:
        return list(itertools.islice(csv.reader(f), MAX_ROWS + 1))


@pytest.mark.parametrize(("name", "entry"), FILES, ids=[n for n, _ in FILES])
def test_every_format(name: str, entry: dict[str, Any]) -> None:
    rows = _rows(name, entry)
    for fmt in pt.tabulate_formats:
        want = ref.tabulate(rows, headers="firstrow", tablefmt=fmt)
        got = pt.tabulate(rows, headers="firstrow", tablefmt=fmt)
        assert got == want, (name, fmt)
        assert type(got).__name__ == type(want).__name__
    for options in VARIANTS:
        for fmt in ("simple", "pipe", "grid"):
            want = ref.tabulate(rows, headers="firstrow", tablefmt=fmt, **options)
            got = pt.tabulate(rows, headers="firstrow", tablefmt=fmt, **options)
            assert got == want, (name, fmt, options)


@pytest.mark.parametrize(("name", "entry"), FILES, ids=[n for n, _ in FILES])
def test_dataframe_to_markdown(name: str, entry: dict[str, Any]) -> None:
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(CORPUS / f"{name}.csv", nrows=MAX_ROWS)
    for fmt in ("pipe", "github", "grid", "simple", "psql"):
        want = df.to_markdown(tablefmt=fmt)
        got = pt.tabulate(df, headers="keys", tablefmt=fmt, showindex=True)
        assert got == want, (name, fmt)
    assert pt.tabulate(df, headers="keys", showindex=False) == df.to_markdown(
        index=False, tablefmt="simple"
    )
