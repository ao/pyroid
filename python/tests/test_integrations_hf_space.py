"""examples/space/app.py: the Gradio demo's handlers and UI, never launched (no sockets).

Skips without gradio/datasets, outside a pyroid checkout, and (for the handlers) without
fastText's ``lid.176.bin`` (``$PYROID_LID_MODEL``, else ``/data/models/lid.176.bin``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

gr = pytest.importorskip("gradio")
pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

APP = Path(__file__).resolve().parents[2] / "examples" / "space" / "app.py"
LID_MODEL = Path(os.environ.get("PYROID_LID_MODEL", "/data/models/lid.176.bin"))
DEV_SLICE = Path(os.environ.get("PYROID_FINEWEB_DEV", "/data/fineweb/fineweb-dev-10k.parquet"))

pytestmark = pytest.mark.skipif(not APP.exists(), reason="not a pyroid checkout")
needs_model = pytest.mark.skipif(not LID_MODEL.exists(), reason=f"no {LID_MODEL}")

ENGLISH = (
    "The naÃ¯ve on the corner opened in 1998, and it has been busy ever since.\n"
    "Every morning a group of retired teachers meets there to read the newspapers and talk about "
    "the news of the day.\n"
    "In the afternoon the room fills with students who come to study after school, because the "
    "light is good and the chairs are comfortable.\n"
    "The owner says that the number of visitors has doubled in ten years, and that many of them "
    "come back every week.\n"
    "She hopes to keep the doors open in the evening as well, so that people who work during the "
    "day can also enjoy it.\n"
    "For now, the café closes at six o'clock, and the last visitors usually leave just before "
    "the doors are locked for the night."
)
OTHER_ENGLISH = (
    "The river that runs through the valley floods almost every spring.\n"
    "Farmers on both banks have learned to plant late, after the water has gone back down.\n"
    "Their grandparents built the houses on higher ground, and most of them are still standing "
    "today.\n"
    "Each year the town holds a small market when the fields are dry again, with music and food "
    "from the farms.\n"
    "Visitors come from the cities to buy fresh bread, honey and cheese, and to walk along the "
    "water in the evening.\n"
    "The market ends with a dinner in the square, where everyone sits at the same long tables."
)
FRENCH = (
    "La bibliothèque du village a ouvert une nouvelle salle de lecture au printemps dernier, et "
    "elle ne désemplit pas depuis. Chaque matin, des enseignants à la retraite s'y "
    "retrouve pour lire les journaux et discuter des nouvelles du jour."
)
DOCS = [ENGLISH, ENGLISH, "short", FRENCH, OTHER_ENGLISH]


@pytest.fixture(scope="module")
def app() -> Iterator[ModuleType]:
    """app.py, imported (not run) with the local LID model and Gradio's analytics off."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYROID_LID_MODEL", str(LID_MODEL))
        mp.setenv("GRADIO_ANALYTICS_ENABLED", "False")
        spec = importlib.util.spec_from_file_location("pyroid_space_app", APP)
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod


@pytest.fixture
def out_dir(app: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "out"
    monkeypatch.setattr(app, "OUT_DIR", d)
    return d


def stage_table(rows: list[list[str]]) -> dict[str, str]:
    return {stage: verdict for stage, verdict, _ in rows}


# ------------------------------------------------------------------------------------ paste


@needs_model
def test_clean_text_kept(app: ModuleType) -> None:
    verdict, text, rows = app.clean_text(ENGLISH)
    assert verdict.startswith("**Kept.**")
    assert text.startswith("The naïve on the corner")
    assert stage_table(rows) == {
        "ftfy": "changed",
        "lid": "pass",
        "gopher_repetition": "pass",
        "gopher_quality": "pass",
        "c4_quality": "pass",
        "fineweb_quality": "pass",
        "minhash": "not reached",
    }
    assert rows[1][2].startswith("en 0.")


@needs_model
def test_clean_text_dropped(app: ModuleType) -> None:
    verdict, _, rows = app.clean_text("short")
    assert verdict == "**Dropped** by `gopher_quality:gopher_short_doc`."
    table = stage_table(rows)
    assert (table["ftfy"], table["lid"], table["gopher_quality"]) == ("unchanged", "pass", "drop")
    assert table["c4_quality"] == table["fineweb_quality"] == "not reached"

    verdict, text, rows = app.clean_text(FRENCH)
    assert verdict == "**Dropped** by `lid`."
    assert text == FRENCH
    assert rows[1][1:] == ["drop", rows[1][2]]
    assert rows[1][2].startswith("fr ")
    assert all(v == "not reached" for s, v, _ in rows[2:])


def test_clean_text_rejects(app: ModuleType) -> None:
    for bad in (None, "", "  \n "):
        with pytest.raises(app.RejectedInputError, match="paste a document"):
            app.clean_text(bad)
    with pytest.raises(app.RejectedInputError, match="at most 100,000 characters"):
        app.clean_text("x" * (app.MAX_TEXT_CHARS + 1))
    ui = app._for_ui(app.clean_text)
    assert ui.__name__ == "clean_text"
    with pytest.raises(gr.Error, match="paste a document"):
        ui("")


@needs_model
@pytest.mark.skipif(not DEV_SLICE.exists(), reason=f"no {DEV_SLICE}")
def test_clean_text_agrees_with_the_recipe(
    app: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stage-by-stage view gives the recipe's decision, reason and text."""
    monkeypatch.setattr(app, "MAX_TEXT_CHARS", 10**7)  # some dev docs are longer than the cap
    texts = pq.read_table(DEV_SLICE, columns=["text"])["text"].slice(0, 300)
    a = app.RECIPE.annotate(texts)
    dropped = 0
    for doc, keep, reason, out in zip(
        texts.to_pylist(), a.keep.to_pylist(), a.reason.to_pylist(), a.text.to_pylist(), strict=True
    ):
        verdict, text, _ = app.clean_text(doc)
        assert verdict == (
            "**Kept.** This document passes every FineWeb stage but the dedup."
            if keep
            else f"**Dropped** by `{reason}`."
        )
        assert text == out
        dropped += not keep
    assert dropped > 10


# ----------------------------------------------------------------------------------- upload


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    if path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return path


@needs_model
@pytest.mark.parametrize("suffix", [".parquet", ".jsonl"])
def test_clean_file(app: ModuleType, out_dir: Path, tmp_path: Path, suffix: str) -> None:
    src = write(tmp_path / f"docs{suffix}", [{"id": i, "text": t} for i, t in enumerate(DOCS)])
    out, stats = app.clean_file(str(src))
    out_path = Path(out)
    assert out_path.parent == out_dir
    assert out_path.suffix == suffix
    assert re.match(r"docs-clean-[0-9a-f]{8}$", out_path.stem)
    if suffix == ".parquet":
        rows = pq.read_table(out_path).to_pylist()
    else:
        lines = out_path.read_text("utf-8").splitlines()
        assert "naïve" in lines[0]  # not \u-escaped
        rows = [json.loads(line) for line in lines]
    assert [r["id"] for r in rows] == [0, 4]
    assert rows[0]["text"].startswith("The naïve")
    assert rows[0]["language"] == "en"
    assert stats.startswith("**5 documents -> 2 kept (40.0%)**")
    for line in ("| lid | 1 |", "| gopher_quality | 1 |", "| minhash | 1 |", "| `minhash` | 1 |"):
        assert line in stats
    assert "Only the first" not in stats


@needs_model
def test_clean_file_row_cap(
    app: ModuleType, out_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "MAX_ROWS", 2)
    for suffix in (".parquet", ".jsonl"):
        src = write(tmp_path / f"docs{suffix}", [{"text": t} for t in DOCS])
        _, stats = app.clean_file(str(src))
        assert stats.startswith("**2 documents -> 1 kept (50.0%)**")
        assert "Only the first 2 of 5 rows were cleaned." in stats


def test_clean_file_rejects(
    app: ModuleType, out_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reject = app.RejectedInputError
    with pytest.raises(reject, match="upload a"):
        app.clean_file(None)
    with pytest.raises(reject, match=r"upload a \.parquet or \.jsonl"):
        app.clean_file(str(write(tmp_path / "docs.csv", [{"text": "x"}])))
    with pytest.raises(reject, match="no 'text' column"):
        app.clean_file(str(write(tmp_path / "a.parquet", [{"body": "x"}])))
    with pytest.raises(reject, match="holds int64, not strings"):
        app.clean_file(str(write(tmp_path / "b.jsonl", [{"text": 1}])))
    (tmp_path / "c.parquet").write_bytes(b"not parquet")
    with pytest.raises(reject, match="not a parquet file"):
        app.clean_file(str(tmp_path / "c.parquet"))
    (tmp_path / "d.jsonl").write_text('[{"text": "a JSON array, not lines"}', "utf-8")
    with pytest.raises(reject, match="not JSON Lines"):
        app.clean_file(str(tmp_path / "d.jsonl"))
    big = write(tmp_path / "e.parquet", [{"text": "x" * 1000}] * 10)
    monkeypatch.setattr(app, "MAX_DECODED_BYTES", 100)
    with pytest.raises(reject, match="decode to more than"):
        app.clean_file(str(big))
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(reject, match="at most 0 MiB"):
        app.clean_file(str(big))
    assert not out_dir.exists()  # nothing was written


def test_prune_deletes_old_outputs_only(app: ModuleType, tmp_path: Path) -> None:
    old, new, other = (tmp_path / n for n in ("a-clean-1.parquet", "b-clean-2.jsonl", "keep.txt"))
    for p in (old, new, other):
        p.write_text("x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    os.utime(other, (time.time() - 7200, time.time() - 7200))
    app._prune(tmp_path, 3600)
    assert not old.exists()
    assert new.exists()
    assert other.exists()


# -------------------------------------------------------------------------------------- UI


@pytest.mark.thread_unsafe
def test_build_constructs_without_launching(
    run_python: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    # A fresh interpreter: Gradio's Blocks leave asyncio loops unclosed (ResourceWarnings at
    # exit, which this suite turns into errors) and keep global state.
    code = f"""
    import importlib.util, os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    import gradio as gr

    def no_launch(*args, **kwargs):
        raise AssertionError("build() must not launch a server")

    gr.Blocks.launch = no_launch
    spec = importlib.util.spec_from_file_location("app", {str(APP)!r})
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    demo = app.build()
    assert isinstance(demo, gr.Blocks), type(demo)
    print(demo.title, "|", sorted(getattr(f.fn, "__name__", "") for f in demo.fns.values()))
    """
    r = run_python(code, env={"PYROID_LID_MODEL": str(LID_MODEL), "HF_DATASETS_OFFLINE": "1"})
    assert r.returncode == 0, r.stderr
    title, fns = r.stdout.strip().splitlines()[-1].split(" | ")
    assert title == "pyroid FineWeb cleaner"
    assert "'clean_text'" in fns
    assert "'clean_file'" in fns


def test_app_never_binds_wide_or_shares() -> None:
    """Gradio's defaults only: 127.0.0.1 locally, the platform's address on Spaces."""
    src = APP.read_text("utf-8")
    for bad in ("0.0.0.0", "server_name", "share=", "GRADIO_SERVER_NAME", "auth="):
        assert bad not in src, bad
    assert src.count(".launch(") == 1
    main = src[src.index('if __name__ == "__main__":') :]
    assert "build().launch(max_file_size=MAX_UPLOAD_BYTES)" in main
