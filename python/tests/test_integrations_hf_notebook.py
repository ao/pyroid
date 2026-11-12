"""examples/notebooks/clean_fineweb_in_60_seconds.ipynb runs headless, offline, on local files.

Executes the notebook with nbclient in a fresh kernel (this interpreter), on the pinned
FineWeb dev slice and lid.176.bin, with the Hub switched off. Skips when nbclient or
ipykernel is missing, when the files are missing, or when the suite runs outside a
pyroid checkout (the notebook is not part of the wheel).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

nbclient = pytest.importorskip("nbclient")
nbformat = pytest.importorskip("nbformat")
pytest.importorskip("ipykernel")
pytest.importorskip("datasets")
pq = pytest.importorskip("pyarrow.parquet")

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "notebooks"
    / "clean_fineweb_in_60_seconds.ipynb"
)
LID_MODEL = Path(os.environ.get("PYROID_LID_MODEL", "/data/models/lid.176.bin"))
DEV_SLICE = Path(os.environ.get("PYROID_FINEWEB_DEV", "/data/fineweb/fineweb-dev-10k.parquet"))

pytestmark = [
    pytest.mark.skipif(not NOTEBOOK.exists(), reason="not a pyroid checkout"),
    pytest.mark.skipif(
        not (LID_MODEL.exists() and DEV_SLICE.exists()), reason=f"no {DEV_SLICE} or {LID_MODEL}"
    ),
]


def test_notebook_is_saved_without_outputs() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code
    assert all(c.outputs == [] and c.execution_count is None for c in code)
    assert "%pip install" in code[0].source
    assert '"pyroid[hf]"' in code[0].source


@pytest.mark.thread_unsafe
def test_notebook_runs_headless(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "clean.parquet"
    env = {
        "PYROID_NOTEBOOK_DATA": str(DEV_SLICE),
        "PYROID_LID_MODEL": str(LID_MODEL),
        "PYROID_NOTEBOOK_OUT": str(out),
        "HF_HOME": str(tmp_path / "hf-home"),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    for k, v in env.items():  # the kernel inherits this process's environment
        monkeypatch.setenv(k, v)
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()

    text = "\n".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs
    )
    assert "10,000 docs -> 8,407 kept (84.1%)" in text
    assert "--- gopher_quality: 1,524 dropped" in text
    assert "Requirement already satisfied" not in text  # the install cell was skipped
    assert pq.read_metadata(out).num_rows == 8407
