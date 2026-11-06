"""pyroid FineWeb cleaner: a minimal Gradio demo of pyroid's FineWeb recipe.

- **Paste text**: one document through the chain, with each stage's verdict
  (ftfy, fastText language ID, the Gopher/C4/FineWeb quality filters).
- **Upload a file**: a parquet or JSONL file with a ``text`` column through the
  whole recipe, MinHash dedup included (``pyroid.integrations.hf.fineweb``);
  download the cleaned file, in the same format.

A public, anonymous demo, so every input is capped (see the ``MAX_*`` constants
and README.md). Run it locally with ``python app.py``: Gradio's defaults serve it
on http://127.0.0.1:7860 only (Hugging Face Spaces sets its own server address).
"""

from __future__ import annotations

import functools
import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import datasets
import gradio as gr
import pyarrow as pa
import pyarrow.json as pj
import pyarrow.parquet as pq

import pyroid.filters
from pyroid.integrations import hf
from pyroid.integrations.recipes import FineWeb

#: Pasted text: at most this many characters.
MAX_TEXT_CHARS = 100_000
#: Uploads: at most this many bytes (also enforced by Gradio at upload time).
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
#: Uploads: only the first rows are cleaned.
MAX_ROWS = 5_000
#: Uploads: at most this much decoded column data in the rows read (parquet compresses).
MAX_DECODED_BYTES = 128 * 1024 * 1024
#: Cleaned files (and Gradio's copies of uploads) older than this are deleted (checked hourly
#: by Gradio, and on each upload by this app).
FILE_TTL_S = 3600
#: pyroid's threads= (unset: every CPU).
THREADS = int(os.environ.get("PYROID_SPACE_THREADS", "0")) or None

OUT_DIR = Path(tempfile.gettempdir()) / "pyroid-space"
RECIPE = FineWeb(threads=THREADS)  # the LID model loads (or downloads, once) on first use
LID_THRESHOLD = RECIPE.language_threshold
NOT_REACHED = "not reached"


class RejectedInputError(ValueError):
    """An input over a cap or in the wrong shape (shown to the user)."""


# ------------------------------------------------------------------------------------ paste


def clean_text(text: str | None) -> tuple[str, str, list[list[str]]]:
    """One document through the chain, stage by stage.

    Returns:
        ``(verdict markdown, text as it leaves the chain, rows of [stage, verdict, detail])``.
    """
    if not text or not text.strip():
        raise RejectedInputError("paste a document first")
    if len(text) > MAX_TEXT_CHARS:
        raise RejectedInputError(f"at most {MAX_TEXT_CHARS:,} characters, got {len(text):,}")

    fixed = RECIPE.fix([text])[0].as_py()
    rows = [["ftfy", "changed" if fixed != text else "unchanged", "fix_text, ftfy's defaults"]]
    labels, probs = RECIPE.model.predict(fixed.replace("\n", " "), k=1)
    lang, score = labels[0].removeprefix("__label__"), float(probs[0])
    lid_ok = lang in (RECIPE.languages or [lang]) and score > LID_THRESHOLD
    rows.append(["lid", "pass" if lid_ok else "drop", f"{lang} {score:.3f} (keeps en > 0.65)"])

    dropped_by = None if lid_ok else "lid"
    current = fixed
    for stage, config in zip(RECIPE.stage_names, RECIPE.filters, strict=True):
        if dropped_by is not None:
            rows.append([stage, NOT_REACHED, ""])
            continue
        res = pyroid.filters.apply([current], [config])
        if res.keep[0]:
            changed = res.text[0] != current
            rows.append([stage, "pass", "removed some lines" if changed else ""])
            current = res.text[0]
        else:
            rows.append([stage, "drop", res.reason[0] or ""])
            dropped_by = f"{stage}:{res.reason[0]}"
    rows.append(["minhash", NOT_REACHED, "needs a corpus: upload a file"])

    if dropped_by is None:
        verdict = "**Kept.** This document passes every FineWeb stage but the dedup."
    else:
        verdict = f"**Dropped** by `{dropped_by}`."
    return verdict, current, rows


# ----------------------------------------------------------------------------------- upload


def _read_upload(path: Path) -> tuple[Any, int]:
    """The first ``MAX_ROWS`` rows of a parquet or JSONL file, and the file's row count."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            pf = pq.ParquetFile(path)
        except pa.ArrowException as e:
            raise RejectedInputError(f"not a parquet file: {e}") from None
        groups, rows, decoded = [], 0, 0
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            groups.append(i)
            rows += rg.num_rows
            decoded += rg.total_byte_size
            if rows >= MAX_ROWS:
                break
        if decoded > MAX_DECODED_BYTES:
            raise RejectedInputError(
                f"the first {MAX_ROWS:,} rows decode to more than {MAX_DECODED_BYTES >> 20} MiB"
            )
        return pf.read_row_groups(groups).slice(0, MAX_ROWS), pf.metadata.num_rows
    if suffix in (".jsonl", ".json"):
        try:
            table = pj.read_json(path)
        except pa.ArrowInvalid as e:
            raise RejectedInputError(f"not JSON Lines (one JSON object per line): {e}") from None
        return table.slice(0, MAX_ROWS), table.num_rows
    raise RejectedInputError("upload a .parquet or .jsonl file")


def _write_jsonl(ds: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for batch in ds.with_format("arrow").iter(batch_size=1000):
            for row in batch.to_pylist():
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _prune(directory: Path, max_age_s: float) -> None:
    """Delete this app's output files older than ``max_age_s``."""
    now = time.time()
    for p in directory.glob("*-clean-*"):
        try:
            if now - p.stat().st_mtime > max_age_s:
                p.unlink()
        except FileNotFoundError:
            pass


def clean_file(path: str | os.PathLike[str] | None) -> tuple[str, str]:
    """A parquet or JSONL file through the whole recipe.

    Returns:
        ``(path of the cleaned file, stats markdown)``.
    """
    if path is None:
        raise RejectedInputError("upload a .parquet or .jsonl file first")
    src = Path(path)
    size = src.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise RejectedInputError(
            f"at most {MAX_UPLOAD_BYTES >> 20} MiB, got {size / 2**20:.1f} MiB"
        )
    table, total_rows = _read_upload(src)
    if "text" not in table.column_names:
        raise RejectedInputError(f"no 'text' column (columns: {', '.join(table.column_names)})")
    t = table.schema.field("text").type
    if not (pa.types.is_string(t) or pa.types.is_large_string(t)):
        raise RejectedInputError(f"the 'text' column holds {t}, not strings")
    if table.num_rows == 0:
        raise RejectedInputError("the file has no rows")

    ds = datasets.Dataset.from_dict({n: table[n] for n in table.column_names})  # zero-copy
    res = hf.fineweb(
        ds,
        lid_model=RECIPE.model,
        threads=THREADS,
        keep_in_memory=True,  # nothing goes to the datasets cache
        new_fingerprint=uuid.uuid4().hex,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _prune(OUT_DIR, FILE_TTL_S)
    suffix = ".parquet" if src.suffix.lower() == ".parquet" else ".jsonl"
    out = OUT_DIR / f"{src.stem[:40]}-clean-{uuid.uuid4().hex[:8]}{suffix}"
    if suffix == ".parquet":
        res.dataset.to_parquet(str(out))
    else:
        _write_jsonl(res.dataset, out)
    return str(out), _stats_markdown(res.stats, total_rows)


def _stats_markdown(stats: hf.Stats, total_rows: int) -> str:
    pct = 100 * stats.kept / stats.docs
    lines = [
        f"**{stats.docs:,} documents -> {stats.kept:,} kept ({pct:.1f}%)** "
        f"in {stats.seconds['total']:.2f} s"
    ]
    if total_rows > stats.docs:
        lines.append(f"\nOnly the first {stats.docs:,} of {total_rows:,} rows were cleaned.")
    lines += ["", "| stage | dropped |", "|---|---:|"]
    lines += [f"| {stage} | {n:,} |" for stage, n in stats.dropped.items()]
    if stats.reasons:
        lines += ["", "| reason | dropped |", "|---|---:|"]
        lines += [f"| `{r}` | {n:,} |" for r, n in list(stats.reasons.items())[:10]]
    return "\n".join(lines)


# -------------------------------------------------------------------------------------- UI


def _for_ui(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Show rejected inputs as a Gradio error instead of a traceback."""

    @functools.wraps(fn)
    def run(*args: Any) -> Any:
        try:
            return fn(*args)
        except RejectedInputError as e:
            raise gr.Error(str(e)) from None

    return run


INTRO = f"""
# pyroid: FineWeb's cleaning recipe, in Rust

[pyroid](https://github.com/ao/pyroid) runs the stages of datatrove's FineWeb pipeline:
ftfy, fastText language ID (English, p > {LID_THRESHOLD}), the Gopher repetition, Gopher
quality, C4 and FineWeb quality filters, then MinHash dedup, with the same results.
Paste a document to see each stage's verdict, or upload a file (parquet or JSONL with a
`text` column, at most {MAX_UPLOAD_BYTES >> 20} MiB; the first {MAX_ROWS:,} rows are cleaned).
"""

EXAMPLES = [
    "The cafÃ© on the corner opened in 1998.\nIt serves soup and bread at lunch, and there is "
    "music on most evenings of the week.\nThe owner bought the building when it was an empty "
    "shop with broken windows, and she repaired the floor and the walls herself.\nMost of the "
    "people who eat there live nearby, and many of them know each other by name.\nOn Sundays "
    "the kitchen is closed, but the doors stay open for anyone who wants a cup of tea.",
    "Click here!!! Buy now!!! Best price!!!",
]


def build() -> gr.Blocks:
    """The Gradio app (not launched)."""
    with gr.Blocks(
        title="pyroid FineWeb cleaner",
        analytics_enabled=False,
        delete_cache=(FILE_TTL_S, FILE_TTL_S),
    ) as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Paste text"):
            text_in = gr.Textbox(label="Document", lines=10, max_length=MAX_TEXT_CHARS)
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text_in])
            text_btn = gr.Button("Clean", variant="primary")
            verdict = gr.Markdown()
            stages = gr.Dataframe(
                headers=["stage", "verdict", "detail"], label="Stages", interactive=False
            )
            text_out = gr.Textbox(label="Text as it leaves the chain", lines=10)
            text_btn.click(_for_ui(clean_text), [text_in], [verdict, text_out, stages])
        with gr.Tab("Upload a file"):
            file_in = gr.File(label="parquet or JSONL", file_types=[".parquet", ".jsonl", ".json"])
            file_btn = gr.Button("Clean file", variant="primary")
            stats = gr.Markdown()
            file_out = gr.File(label="Cleaned file")
            file_btn.click(_for_ui(clean_file), [file_in], [file_out, stats])
    demo.queue(default_concurrency_limit=1, max_size=16)
    return demo


if __name__ == "__main__":
    _ = RECIPE.model  # load (or download) the LID model before the first request
    # Gradio's defaults: 127.0.0.1 locally; Hugging Face Spaces sets the address itself.
    build().launch(max_file_size=MAX_UPLOAD_BYTES)
