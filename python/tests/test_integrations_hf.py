"""pyroid.integrations.hf: Hugging Face ``datasets`` map/filter helpers and the FineWeb driver.

The model-backed tests need fastText's ``lid.176.bin`` (``$PYROID_LID_MODEL``, else
``/data/models/lid.176.bin``); the equivalence tests also need the pinned 10k-doc
FineWeb dev slice (``$PYROID_FINEWEB_DEV``, else ``/data/fineweb/fineweb-dev-10k.parquet``).
Both skip when the files are missing. Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from packaging.version import Version

datasets = pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")

import pyroid.fasttext  # noqa: E402
from pyroid.integrations import hf  # noqa: E402

LID_MODEL = Path(os.environ.get("PYROID_LID_MODEL", "/data/models/lid.176.bin"))
DEV_SLICE = Path(os.environ.get("PYROID_FINEWEB_DEV", "/data/fineweb/fineweb-dev-10k.parquet"))

# bench/chain.py's per-doc digest of the datatrove FineWeb chain on the dev slice, and its counts.
DEV_DIGEST = "ef314b8d6ae95cc63c6cfa8b1a3a76ee5bc101eb88a66729beb10ecb7226473b"
DEV_DROPPED = {
    "lid": 4,
    "gopher_repetition": 24,
    "gopher_quality": 1524,
    "c4_quality": 14,
    "fineweb_quality": 27,
    "minhash": 0,
}
DEV_KEPT = 8407

needs_model = pytest.mark.skipif(not LID_MODEL.exists(), reason=f"no {LID_MODEL}")
needs_dev = pytest.mark.skipif(
    not (LID_MODEL.exists() and DEV_SLICE.exists()), reason=f"no {DEV_SLICE} or {LID_MODEL}"
)
# datasets forks its num_proc workers; CPython warns about fork() in a threaded process.
fork_ok = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")

MOJIBAKE = "The naÃ¯ve opens at nine."
FIXED = "The naïve opens at nine."

ENGLISH = (
    "The village library opened a new reading room last spring, and it has been busy ever since.\n"
    "Every morning a group of retired teachers meets there to read the newspapers and talk about "
    "the news of the day.\n"
    "In the afternoon the room fills with students who come to study after school, because the "
    "light is good and the chairs are comfortable.\n"
    "The librarian says that the number of visitors has doubled since the room was built, and "
    "that many of them had never used the library before.\n"
    "She hopes that the town will find the money to keep the room open in the evening as well, so "
    "that people who work during the day can also enjoy it.\n"
    "For now, the room closes at six o'clock, and the last visitors usually leave just before the "
    "doors are locked for the night."
)
ENGLISH_2 = (
    MOJIBAKE + "\n"
    "It serves soup and bread at lunch, and in the evening there is music on most days of "
    "the week.\n"
    "The owner bought the building ten years ago, when it was an empty shop with broken windows.\n"
    "She repaired the floor and the walls herself, and her brother built the long wooden tables "
    "that are still used today.\n"
    "Most of the people who eat there live in the neighbourhood, and many of them know each other "
    "by name.\n"
    "On Sundays the kitchen is closed, but the doors stay open for anyone who wants a cup of tea "
    "and a quiet place to sit."
)
FRENCH = (
    "La bibliothèque du village a ouvert une nouvelle salle de lecture au printemps dernier, et "
    "elle ne désemplit pas depuis. Chaque matin, des enseignants à la retraite s'y "
    "retrouve pour lire les journaux et discuter des nouvelles du jour."
)


@pytest.fixture(scope="module")
def model() -> pyroid.fasttext.Model:
    """The loaded LID model (one load for the module)."""
    if not LID_MODEL.exists():
        pytest.skip(f"no {LID_MODEL}")
    return pyroid.fasttext.load_model(LID_MODEL)


def table(texts: list[str | None]) -> Any:
    return pa.table({"text": pa.array(texts, pa.string()), "id": pa.array(range(len(texts)))})


def digest(annotated: Any) -> str:
    """bench/chain.py's digest, from :func:`hf.fineweb`'s annotated dataset.

    One JSON line per doc ``[id, text, lang, score %.6f, keep, reason, dup]`` where
    ``keep`` = survived the filters, ``dup`` = removed by MinHash (None if not kept).
    """
    cols = ["id", "text", "language", "language_score", "keep", "reason"]
    h = hashlib.sha256()
    for r in annotated.with_format("arrow")[:].select(cols).to_pylist():
        dup = r["reason"] == "minhash"
        survived = r["keep"] or dup
        score = None if r["language_score"] is None else f"{r['language_score']:.6f}"
        row = [
            r["id"],
            r["text"],
            r["language"],
            score,
            survived,
            None if survived else r["reason"],
            dup if survived else None,
        ]
        h.update(json.dumps(row, ensure_ascii=False).encode("utf-8", "surrogatepass") + b"\n")
    return h.hexdigest()


# ----------------------------------------------------------------------------------- fix_text


def test_fix_text_arrow_batch() -> None:
    t = table([MOJIBAKE, "fine", None])
    out = hf.fix_text()(t)
    assert isinstance(out, pa.Table)
    assert out.schema == t.schema
    assert out["text"].to_pylist() == [FIXED, "fine", None]
    # The other columns are the batch's own buffers.
    assert out["id"].chunk(0).buffers()[1].address == t["id"].chunk(0).buffers()[1].address


def test_fix_text_python_batch_keeps_objects_and_nones() -> None:
    texts = [MOJIBAKE, "fine", None]
    out = hf.fix_text()({"text": texts, "id": [0, 1, 2]})
    assert out == {"text": [FIXED, "fine", None]}  # only the changed column: datasets merges it
    assert out["text"][1] is texts[1]


def test_fix_text_large_string_and_options() -> None:
    t = pa.table({"body": pa.array([MOJIBAKE, "“quoted”"], pa.large_string())})
    out = hf.fix_text("body", uncurl_quotes=False)(t)
    assert out.schema.field("body").type == pa.large_string()
    assert out["body"].to_pylist() == [FIXED, "“quoted”"]


@pytest.mark.parametrize("fmt", ["arrow", None])
def test_fix_text_in_dataset_map(fmt: str | None) -> None:
    ds = datasets.Dataset.from_dict({"text": [MOJIBAKE, None, "ok"] * 50, "n": list(range(150))})
    out = ds.with_format(fmt).map(hf.fix_text(), batched=True, batch_size=32)
    assert out.format["type"] == fmt
    assert out.features == ds.features
    assert out.with_format(None)[:3] == {"text": [FIXED, None, "ok"], "n": [0, 1, 2]}


@pytest.mark.parametrize("fmt", ["arrow", None])
def test_helpers_in_a_streaming_dataset(fmt: str | None) -> None:
    if fmt == "arrow" and Version(datasets.__version__) < Version("5"):
        pytest.skip("datasets < 5 rejects any Arrow-formatted IterableDataset.filter")
    ds = datasets.Dataset.from_dict(
        {"text": [MOJIBAKE, None, "ok"] * 4, "k": [True, False, None] * 4}
    )
    it = ds.to_iterable_dataset(num_shards=2).with_format(fmt)
    it = it.map(hf.fix_text(), batched=True, batch_size=5)
    it = it.filter(hf.keep_mask("k"), batched=True, batch_size=5)
    rows = [
        r.to_pylist()[0] if isinstance(r, pa.Table) else r
        for r in it  # arrow format yields one-row tables
    ]
    assert rows == [{"text": FIXED, "k": True}] * 4


def test_wrong_batch_kind_and_missing_column() -> None:
    with pytest.raises(TypeError, match="input_columns"):
        hf.fix_text()(pa.chunked_array([["x"]]))
    with pytest.raises(KeyError, match="no column 'body'"):
        hf.fix_text("body")(table(["x"]))


# --------------------------------------------------------------------------- zero-copy evidence


def _python_peak(fn: Any, arg: Any) -> int:
    tracemalloc.start()
    try:
        fn(arg)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.thread_unsafe
def test_arrow_batches_build_no_python_strings() -> None:
    """The Arrow path allocates no Python objects per document text (tracemalloc sees
    Python's allocators only; pyroid's buffers are Rust's)."""
    docs = [f"Doc {i}: the naÃ¯ve has a long menu. " * 60 for i in range(2000)]
    t = table(docs)  # type: ignore[arg-type]
    text_bytes = t["text"].nbytes
    baseline = _python_peak(lambda x: x["text"].to_pylist(), t)  # what a Python batch costs
    assert baseline > text_bytes / 2
    for fn in (hf.fix_text(threads=2), hf.quality_mask(threads=2), hf.keep_mask("ok")):
        peak = _python_peak(fn, t.append_column("ok", pa.array([True] * len(docs))))
        assert peak < text_bytes / 50, (fn, peak, text_bytes)


# ------------------------------------------------------------------------------------- langid


@needs_model
def test_langid_arrow_and_python(model: pyroid.fasttext.Model) -> None:
    fn = hf.langid(model)
    texts: list[str | None] = [ENGLISH, FRENCH, None]
    out = fn(table(texts))
    assert out.column_names == ["text", "id", "language", "language_score"]
    assert out.schema.field("language_score").type == pa.float64()
    assert out["language"].to_pylist() == ["en", "fr", None]
    scores = out["language_score"].to_pylist()
    assert scores[0] > 0.65
    assert scores[1] > 0.65
    assert scores[2] is None
    py = fn({"text": texts})
    assert py == {"language": ["en", "fr", None], "language_score": scores}
    # Overwrites an existing column in place, as datatrove's LanguageFilter does.
    again = fn(out)
    assert again.column_names == out.column_names


@needs_model
def test_langid_top_k_lists(model: pyroid.fasttext.Model) -> None:
    out = hf.langid(model, k=3, language_column="langs", score_column="probs")(table([ENGLISH]))
    assert out.schema.field("langs").type == pa.list_(pa.string())
    langs, probs = out["langs"][0].as_py(), out["probs"][0].as_py()
    assert len(langs) == 3
    assert langs[0] == "en"
    assert probs == sorted(probs, reverse=True)


# ------------------------------------------------------------------------- quality / keep masks


@pytest.mark.parametrize("fmt", ["arrow", None])
def test_quality_mask_filter(fmt: str | None) -> None:
    ds = datasets.Dataset.from_dict({"text": [ENGLISH, "too short", None, ENGLISH + "\n"]})
    out = ds.with_format(fmt).filter(hf.quality_mask(), batched=True)
    assert out.with_format(None)["text"][:] == [ENGLISH, ENGLISH + "\n"]
    assert out.format["type"] == fmt


def test_quality_mask_custom_filters() -> None:
    import pyroid.filters

    t = table(["one two three", "a b"])
    keep = hf.quality_mask(filters=[pyroid.filters.GopherQuality(min_doc_words=3)])(t)
    assert keep.to_pylist() == [False, False]  # stop words / mean word length still apply
    assert hf.quality_mask(filters=[])(t).to_pylist() == [True, True]


@pytest.mark.parametrize("fmt", ["arrow", None])
def test_keep_mask_filter(fmt: str | None) -> None:
    ds = datasets.Dataset.from_dict({"t": ["a", "b", "c"], "keep": [True, None, False]})
    out = ds.with_format(fmt).filter(hf.keep_mask(), batched=True)
    assert out.with_format(None)["t"][:] == ["a"]


# --------------------------------------------------------------------------- pickling / hashing


@needs_model
def test_helpers_pickle_by_settings(model: pyroid.fasttext.Model) -> None:
    hasher = pytest.importorskip("datasets.fingerprint").Hasher

    fn = hf.langid(str(LID_MODEL))
    fn(table([ENGLISH]))  # loads the model
    # Round-trips our own object, as datasets does for num_proc workers (no untrusted data).
    clone = pickle.loads(pickle.dumps(fn))
    assert clone(table([FRENCH]))["language"].to_pylist() == ["fr"]
    assert hasher.hash(fn) == hasher.hash(hf.langid(str(LID_MODEL)))  # the model is not hashed
    assert hasher.hash(hf.fix_text()) == hasher.hash(hf.fix_text())
    assert hasher.hash(hf.fix_text()) != hasher.hash(hf.fix_text("body"))
    assert "lid.176.bin" in repr(fn)
    with pytest.raises(TypeError, match="pass the model's path"):
        pickle.dumps(hf.langid(model))
    assert "Model" in repr(hf.annotate(lid_model=model))
    ds = datasets.Dataset.from_dict({"text": [ENGLISH]})
    with pytest.raises(ValueError, match="num_proc > 1 needs lid_model as a path"):
        hf.fineweb(ds, lid_model=model, num_proc=2)


def test_fineweb_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        hf.fineweb(datasets.Dataset.from_dict({"text": []}))


# ------------------------------------------------------------------------------------- fineweb


@needs_model
def test_annotate_arrow_equals_python(model: pyroid.fasttext.Model) -> None:
    texts: list[str | None] = [ENGLISH, FRENCH, "short", None, MOJIBAKE + " " + ENGLISH]
    fn = hf.annotate(lid_model=model)
    arrow = fn(table(texts))
    assert arrow.column_names == ["text", "id", *hf.ANNOTATION_COLUMNS]
    py = fn({"text": texts})
    assert set(py) == {"text", *hf.ANNOTATION_COLUMNS}
    for name, values in py.items():
        assert arrow[name].to_pylist() == values, name
    assert py["keep"] == [True, False, False, False, True]
    assert py["reason"][:4] == [None, "lid", "gopher_quality:gopher_short_doc", "null"]
    assert py["language"][:4] == ["en", "fr", "en", None]
    assert py["text"][4].startswith("The naïve")


@needs_model
@pytest.mark.parametrize("fmt", [None, "arrow"])
def test_fineweb_small(model: pyroid.fasttext.Model, fmt: str | None, tmp_path: Path) -> None:
    texts = [ENGLISH, FRENCH, ENGLISH, "short", None, ENGLISH_2]
    ds = datasets.Dataset.from_dict({"text": texts, "id": [f"d{i}" for i in range(6)]})
    res = hf.fineweb(ds.with_format(fmt), lid_model=model, batch_size=2)
    s = res.stats
    assert (s.docs, s.kept) == (6, 2)
    assert s.dropped == {
        "lid": 1,
        "gopher_repetition": 0,
        "gopher_quality": 1,
        "c4_quality": 0,
        "fineweb_quality": 0,
        "minhash": 1,  # the second copy of ENGLISH
        "null": 1,
    }
    assert list(s.reasons) == ["gopher_quality:gopher_short_doc", "lid", "minhash", "null"]
    assert set(s.reasons.values()) == {1}
    assert set(s.seconds) == {"annotate", "dedup", "total"}
    assert str(s).startswith("6 docs -> 2 kept (33.3%)")
    assert res.dataset.format["type"] == fmt
    assert res.annotated.format["type"] == fmt
    kept = res.dataset.with_format(None)[:]
    assert kept["id"] == ["d0", "d5"]
    assert kept["text"][1].startswith("The naïve")
    assert kept["language"] == ["en", "en"]
    assert res.dataset.column_names == ["text", "id", "language", "language_score"]
    ann = res.annotated.with_format(None)[:]
    assert ann["keep"] == [True, False, False, False, False, True]
    assert ann["reason"] == [
        None,
        "lid",
        "minhash",
        "gopher_quality:gopher_short_doc",
        "null",
        None,
    ]
    res.dataset.to_parquet(str(tmp_path / "clean.parquet"))
    assert datasets.Dataset.from_parquet(str(tmp_path / "clean.parquet"), cache_dir=str(tmp_path))[
        "id"
    ][:] == ["d0", "d5"]


@needs_dev
@pytest.mark.thread_unsafe
def test_fineweb_dev_slice_matches_datatrove(tmp_path: Path) -> None:
    """hf.fineweb on the 10k-doc dev slice is digest-identical to datatrove's FineWeb chain."""
    ds = datasets.Dataset.from_parquet(str(DEV_SLICE), cache_dir=str(tmp_path))
    res = hf.fineweb(ds, lid_model=str(LID_MODEL))
    assert digest(res.annotated) == DEV_DIGEST
    assert res.stats.docs == len(ds) == 10_000
    assert res.stats.kept == len(res.dataset) == DEV_KEPT
    assert res.stats.dropped == DEV_DROPPED
    assert res.dataset.column_names == ds.column_names
    ann = res.annotated.with_format("arrow")[:]
    survivors = ann.filter(ann["keep"])
    assert res.dataset.with_format("arrow")[:]["text"].to_pylist() == survivors["text"].to_pylist()


@needs_dev
@fork_ok
@pytest.mark.thread_unsafe
def test_fineweb_dev_slice_num_proc(tmp_path: Path) -> None:
    """Worker processes and another batch size give the same result: dedup stays global."""
    ds = datasets.Dataset.from_parquet(str(DEV_SLICE), cache_dir=str(tmp_path))
    res = hf.fineweb(
        ds,
        lid_model=str(LID_MODEL),
        num_proc=2,
        threads=2,
        batch_size=700,
        load_from_cache_file=False,
    )
    assert digest(res.annotated) == DEV_DIGEST
    assert res.stats.dropped == DEV_DROPPED
