"""The ``pyroid`` command line: formats, streaming, subcommands, digest identity with the chain."""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

import pyroid  # noqa: E402
import pyroid.dedup  # noqa: E402
import pyroid.translit  # noqa: E402
from pyroid.cli import main  # noqa: E402

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
DEV = DATA / "fineweb" / "fineweb-dev-10k.parquet"
LID_BIN = DATA / "models" / "lid.176.bin"
LID_FTZ = DATA / "models" / "lid.176.ftz"
DIGEST_SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "cli" / "digest.py"
DEV_DIGEST = "ef314b8d6ae95cc63c6cfa8b1a3a76ee5bc101eb88a66729beb10ecb7226473b"

BASE = (
    "The quick brown fox jumps over the lazy dog while the farmer watches from the porch and "
    "thinks about the weather, the harvest and the long winter that is surely coming soon to "
    "the valley where his family has lived for many generations. "
)


def _table(texts: list[Any], **extra: list[Any]) -> Any:
    cols = {"id": [f"d{i}" for i in range(len(texts))], "text": texts, **extra}
    return pa.table(cols)


def _write(t: Any, path: Path) -> Path:
    if path.suffix == ".parquet":
        pq.write_table(t, path)
    else:
        opener: Any = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt", encoding="utf-8") as f:
            for r in t.to_pylist():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _read(path: Path) -> Any:
    from pyroid.integrations import _io

    tables = list(_io.read_tables(_io.input_files(path), 1 << 16))
    return pa.concat_tables(tables) if tables else None


# --------------------------------------------------------------------------------------- basics


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert pyroid.__version__ in capsys.readouterr().out


def test_python_dash_m(tmp_path: Path) -> None:
    src = _write(_table(["Ünïcödé", "ascii"]), tmp_path / "in.parquet")
    out = tmp_path / "out.jsonl"
    r = subprocess.run(
        [sys.executable, "-m", "pyroid", "translit", str(src), str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert [json.loads(x)["text"] for x in out.read_text().splitlines()] == ["Unicode", "ascii"]


#: The leading bytes of each compressed output (the suffixes `pyroid --help` lists).
MAGIC = {".gz": b"\x1f\x8b", ".bz2": b"BZh", ".zst": b"\x28\xb5\x2f\xfd", ".lz4": b'\x04"M\x18'}


@pytest.mark.parametrize("fmt_in", [".parquet", ".jsonl", ".jsonl.gz"])
@pytest.mark.parametrize(
    "fmt_out", [".parquet", ".jsonl", ".ndjson.gz", ".json.zst", ".jsonl.bz2", ".jsonl.lz4"]
)
def test_translit_formats(tmp_path: Path, fmt_in: str, fmt_out: str) -> None:
    texts = ["Ålborg — “quoted” ½", "日本語", "plain", "Ελληνικά"] * 5
    src = _write(_table(texts, n=list(range(len(texts)))), tmp_path / f"in{fmt_in}")
    out = tmp_path / f"out{fmt_out}"
    assert main(["translit", str(src), str(out), "--batch-size", "3"]) == 0
    assert out.read_bytes().startswith(MAGIC.get(out.suffix, b""))
    t = _read(out)
    assert t.column("text").to_pylist() == pyroid.translit.anyascii(texts)
    assert t.column("n").to_pylist() == list(range(len(texts)))
    assert t.column_names == ["id", "text", "n"]
    assert not list(tmp_path.glob("*.partial"))


def test_translit_output_column_and_unidecode(tmp_path: Path) -> None:
    texts = ["café", "\U0001f600 smile"]
    src = _write(_table(texts), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    args = ["translit", str(src), str(out), "--mode", "unidecode", "--errors", "replace"]
    assert main([*args, "--output-column", "ascii"]) == 0
    t = pq.read_table(out)
    assert t.column("text").to_pylist() == texts
    assert t.column("ascii").to_pylist() == [pyroid.translit.unidecode(x, "replace") for x in texts]


def test_directory_input_in_name_order(tmp_path: Path) -> None:
    d = tmp_path / "shards"
    d.mkdir()
    _write(_table(["b1", "b2"]), d / "b.parquet")
    _write(_table(["a1"]), d / "a.parquet")
    (d / "notes.txt").write_text("ignored")
    out = tmp_path / "out.parquet"
    assert main(["translit", str(d), str(out), "--batch-size", "2"]) == 0
    assert pq.read_table(out).column("text").to_pylist() == ["a1", "b1", "b2"]


def test_empty_input(tmp_path: Path) -> None:
    src = _write(
        _table([]).cast(pa.schema([("id", pa.string()), ("text", pa.string())])),
        tmp_path / "in.parquet",
    )
    out = tmp_path / "out.parquet"
    assert main(["dedup", str(src), str(out), "--method", "exact"]) == 0
    assert pq.read_table(out).num_rows == 0


def test_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = _write(_table(["x"]), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    with pytest.raises(SystemExit, match="no column 'body'"):
        main(["translit", str(src), str(out), "--text-column", "body"])
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))
    out.write_text("keep me")
    with pytest.raises(SystemExit, match="--overwrite"):
        main(["translit", str(src), str(out)])
    assert out.read_text() == "keep me"
    assert main(["translit", str(src), str(out), "--overwrite"]) == 0
    assert main(["translit", str(src), str(tmp_path / "out.csv")]) == 1
    err = capsys.readouterr().err
    assert "unknown format" in err
    assert "end OUT with / to make it a directory" in err
    assert main(["translit", str(tmp_path / "missing.parquet"), str(tmp_path / "o.parquet")]) == 1
    with pytest.raises(SystemExit):
        main(["clean", str(src), str(tmp_path / "o.parquet")])  # --recipe is required
    with pytest.raises(SystemExit, match=r"in\.parquet is both an input and an output"):
        main(["translit", str(src), str(src), "--overwrite"])


@pytest.mark.thread_unsafe(reason="monkeypatches pyroid.cli and the environment")
def test_missing_optional_dependency_is_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ImportError from a command exits 1 with one line that says what to install."""
    import pyroid.cli as cli

    src = _write(_table(["x"]), tmp_path / "in.parquet")
    args = ["translit", str(src), str(tmp_path / "o.parquet")]
    cases = [
        (
            ModuleNotFoundError("No module named 'cchardet'", name="cchardet"),
            "No module named 'cchardet' (pip install 'pyroid[warc]')",
        ),
        (
            ImportError("failed to find libmagic.  Check your installation"),
            "failed to find libmagic.  Check your installation (python-magic needs the libmagic "
            "library, e.g. apt install libmagic1)",
        ),
        (
            ModuleNotFoundError("No module named 'tldextract.cache'", name="tldextract.cache"),
            "No module named 'tldextract.cache' (pip install tldextract)",
        ),
        (ModuleNotFoundError("x needs y (pip install y)", name="y"), "x needs y (pip install y)"),
        (ModuleNotFoundError("No module named 'zzz'", name="zzz"), "No module named 'zzz'"),
    ]
    for err, line in cases:

        def body(a: Any, err: ImportError = err) -> int:
            raise err

        monkeypatch.setattr(cli, "_cmd_translit", body)
        assert main(args) == 1
        assert capsys.readouterr().err == f"pyroid translit: {line}\n"
    monkeypatch.setenv("PYROID_DEBUG", "1")
    with pytest.raises(ModuleNotFoundError, match="zzz"):
        main(args)


# ----------------------------------------------------------------------------------------- dedup


def test_dedup_exact(tmp_path: Path) -> None:
    texts = ["a", "b", "a", "c", "b", "A "]
    src = _write(_table(texts), tmp_path / "in.jsonl")
    out, excl = tmp_path / "out.parquet", tmp_path / "dup.parquet"
    stats = tmp_path / "stats.json"
    args = ["dedup", str(src), str(out), "--method", "exact", "--excluded", str(excl)]
    assert main([*args, "--batch-size", "2", "--stats-json", str(stats)]) == 0
    assert pq.read_table(out).column("text").to_pylist() == ["a", "b", "c", "A "]
    x = pq.read_table(excl)
    assert x.column("id").to_pylist() == ["d2", "d4"]
    assert x.column("filter_reason").to_pylist() == ["exact", "exact"]
    s = json.loads(stats.read_text())
    assert (s["docs_in"], s["docs_out"], s["removed"]) == (6, 4, {"exact": 2})
    out2 = tmp_path / "out2.parquet"
    assert main(["dedup", str(src), str(out2), "--method", "exact", "--normalize"]) == 0
    assert pq.read_table(out2).column("text").to_pylist() == ["a", "b", "c"]


def test_dedup_minhash_matches_library(tmp_path: Path) -> None:
    texts = []
    for i in range(40):
        doc = f"Document number {i} says: " + BASE * 2
        texts.append(doc)
        if i % 3 == 0:
            texts.append(doc.replace("lazy", "sleepy"))  # a near-duplicate
    d = tmp_path / "in"
    d.mkdir()
    _write(_table(texts[:30]), d / "0.parquet")
    _write(_table(texts[30:]), d / "1.parquet")
    out, excl = tmp_path / "out.parquet", tmp_path / "dup.parquet"
    cfg = ["--hash-fc", "sha1", "--num-buckets", "14", "--hashes-per-bucket", "8"]
    assert (
        main(["dedup", str(d), str(out), "--excluded", str(excl), "--batch-size", "7", *cfg]) == 0
    )
    mask = pyroid.dedup.minhash_dedup(texts, hash_fc="sha1", num_buckets=14, hashes_per_bucket=8)
    assert pq.read_table(out).column("text").to_pylist() == [
        t for t, k in zip(texts, mask, strict=True) if k
    ]
    assert pq.read_table(excl).num_rows == mask.count(False) > 0


# ---------------------------------------------------------------------------------------- langid


def _ftz() -> Path:
    for m in (LID_FTZ, LID_BIN):
        if m.exists():
            return m
    pytest.skip("no pinned lid.176 model")


def test_langid(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    import pyroid.fasttext

    model = _ftz()
    texts = [
        "This is an English sentence about the weather.",
        "Ceci est une phrase en français.",
        "Guten Morgen, wie geht es dir?\nMit zwei Zeilen.",
        "これは日本語の文です。",
    ]
    src = _write(_table(texts, language=["xx"] * 4), tmp_path / "in.parquet")
    out = tmp_path / "out.parquet"
    assert main(["langid", str(src), str(out), "--model", str(model), "--top-k", "3"]) == 0
    t = pq.read_table(out)
    m = pyroid.fasttext.load_model(model)
    want = [m.predict(x.replace("\n", " "), k=3) for x in texts]
    assert t.column_names == [
        "id",
        "text",
        "language",
        "language_score",
        "top_languages",
        "top_language_scores",
    ]
    assert t.column("language").to_pylist() == [lb[0].removeprefix("__label__") for lb, _ in want]
    assert t.column("language_score").to_pylist() == [float(p[0]) for _, p in want]
    assert t.column("top_languages").to_pylist() == [
        [x.removeprefix("__label__") for x in lb] for lb, _ in want
    ]
    kept, excl = tmp_path / "en.parquet", tmp_path / "other.jsonl"
    args = [
        "langid",
        str(src),
        str(kept),
        "--model",
        str(model),
        "--keep",
        "en,fr",
        "--min-score",
        "0.5",
    ]
    assert main([*args, "--excluded", str(excl)]) == 0
    assert pq.read_table(kept).column("language").to_pylist() == ["en", "fr"]
    rows = [json.loads(x) for x in excl.read_text().splitlines()]
    assert [r["filter_reason"] for r in rows] == ["lid", "lid"]


# ----------------------------------------------------------------------------------------- clean


def _digest_module() -> Any:
    if not DIGEST_SCRIPT.exists():
        pytest.skip("examples/cli/digest.py is not in this checkout")
    spec = importlib.util.spec_from_file_location("pyroid_cli_digest", DIGEST_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    ("batch_size", "threads", "out_ext"), [(10_000, "0", ".parquet"), (3001, "1", ".jsonl.gz")]
)
def test_clean_fineweb_is_digest_identical_to_the_chain(
    tmp_path: Path, batch_size: int, threads: str, out_ext: str
) -> None:
    pytest.importorskip("numpy")
    if not (DEV.exists() and LID_BIN.exists()):
        pytest.skip("needs the pinned dev slice and lid.176.bin under $PYROID_DATA")
    digest = _digest_module()
    out, excl = tmp_path / f"kept{out_ext}", tmp_path / "dropped.parquet"
    stats = tmp_path / "stats.json"
    args = ["clean", str(DEV), str(out), "--recipe", "fineweb", "--lid-model", str(LID_BIN)]
    args += ["--excluded", str(excl), "--batch-size", str(batch_size), "--threads", threads]
    assert main([*args, "--stats-json", str(stats)]) == 0
    assert digest.digest(str(DEV), str(out), str(excl)) == DEV_DIGEST
    s = json.loads(stats.read_text())
    assert (s["docs_in"], s["docs_out"]) == (10_000, 8_407)
    assert s["removed_by_stage"] == {
        "gopher_quality": 1524,
        "fineweb_quality": 27,
        "gopher_repetition": 24,
        "c4_quality": 14,
        "lid": 4,
    }
    t = _read(out)
    assert t.column_names == pq.read_schema(DEV).names  # language/_score replaced in place
    assert not list(tmp_path.glob(".pyroid-*"))  # the dedup spool is gone


def test_clean_with_no_survivor_keeps_the_columns(tmp_path: Path) -> None:
    """When the filters drop every row, the output still has the input's columns."""
    pytest.importorskip("numpy")
    if not LID_BIN.exists():
        pytest.skip("needs lid.176.bin")
    src = _write(_table(["too short"], url=["https://example.org/"]), tmp_path / "in.parquet")
    for out, extra in [("o.parquet", []), ("o2.parquet", ["--no-dedup"])]:
        args = ["clean", str(src), str(tmp_path / out), "--recipe", "fineweb", "--lid-model"]
        assert main([*args, str(LID_BIN), *extra]) == 0
        t = pq.read_table(tmp_path / out)
        assert t.num_rows == 0
        assert t.column_names == ["id", "text", "url", "language", "language_score"], extra


def test_clean_small_no_dedup_and_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("numpy")
    if not (DEV.exists() and LID_BIN.exists()):
        pytest.skip("needs the pinned dev slice and lid.176.bin")
    from pyroid.integrations.recipes import FineWeb

    sample = pq.read_table(DEV, columns=["text"]).slice(0, 20).column("text")
    good = sample.filter(FineWeb(lid_model=str(LID_BIN)).annotate(sample).keep)[0].as_py()
    texts = [good, "too short", None, "Ceci est un texte en français, " * 20, good]
    src = _write(_table(texts), tmp_path / "in.parquet")
    out, excl = tmp_path / "out.parquet", tmp_path / "x.parquet"
    args = ["clean", str(src), str(out), "--recipe", "fineweb", "--lid-model", str(LID_BIN)]
    assert main([*args, "--excluded", str(excl), "--stats"]) == 0
    err = capsys.readouterr().err
    assert "pyroid clean: 5 in" in err
    assert "dropped lid" in err
    reasons = dict(
        zip(
            pq.read_table(excl).column("id").to_pylist(),
            pq.read_table(excl).column("filter_reason").to_pylist(),
            strict=True,
        )
    )
    assert reasons["d2"] == "null"
    assert reasons["d3"] == "lid"
    assert reasons["d1"].startswith(("gopher_", "lid"))
    assert reasons.get("d4") == "minhash"  # an exact copy of d0
    out2 = tmp_path / "out2.parquet"
    assert main(["clean", str(src), str(out2), *args[3:], "--no-dedup"]) == 0
    assert pq.read_table(out2).column("id").to_pylist() == ["d0", "d4"]


# ---------------------------------------------------------------------- the dedup spool's cleanup
# Windows can't delete a file this process still has open or memory-mapped (PermissionError),
# where Linux and macOS can. `strict_unlink` makes os.unlink of pyroid's spools behave the Windows
# way on Linux (from /proc/self/fd and /proc/self/maps); on Windows the OS itself is the check.
# A whole `clean` run under the same check is in test_cli_scale.py.


def _held(path: str) -> list[str]:
    """How this process still holds ``path``: open descriptors and memory maps (Linux)."""
    real = os.path.realpath(path)
    held = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == real:
                held.append(f"fd {fd}")
        except OSError:  # the descriptor listdir itself used, already closed
            continue
    with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
        held += [f"mapped at {ln.split()[0]}" for ln in f if ln.rstrip("\n").endswith(" " + real)]
    return held


@pytest.fixture
def strict_unlink(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The spool paths os.unlink removed; it raises if one was still open or mapped."""
    linux = Path("/proc/self/maps").exists()
    if not linux and sys.platform != "win32":
        pytest.skip("needs /proc (Linux) to see open files and maps, or Windows itself")
    real_unlink = os.unlink
    removed: list[str] = []

    def unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        p = os.fspath(path)
        if p.endswith((".arrow", ".arrow.partial")) and os.path.exists(p):
            held = _held(p) if linux else []
            if held:
                raise PermissionError(f"{p} is still {', '.join(held)} (Windows refuses this)")
            removed.append(p)
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)
    return removed


def _removed_under(removed: list[str], where: Path) -> list[str]:
    """The spools in ``removed`` that were under ``where``.

    pytest-run-parallel's ``--iterations`` reruns a test with the same fixture values but a fresh
    ``tmp_path``, so ``strict_unlink`` also holds the earlier runs' spools.
    """
    return [p for p in removed if Path(p).is_relative_to(where)]


def _spool_tables() -> list[Any]:
    texts = [f"{BASE} {i}" for i in range(9)]
    return [_table(texts[:4]), _table(texts[4:5]), _table(texts[5:])]


@pytest.mark.thread_unsafe  # patches os.unlink
def test_spool_is_closed_before_it_is_removed(tmp_path: Path, strict_unlink: list[str]) -> None:
    from pyroid.cli import _Ipc, _read_ipc

    tables = _spool_tables()
    path = str(tmp_path / "kept.arrow")
    spool = _Ipc(path)
    for t in tables:
        spool.write(t)
    spool.close()
    back = list(_read_ipc(path))  # the tables are still referenced, and must not pin the file
    os.unlink(path)
    assert _removed_under(strict_unlink, tmp_path) == [path]
    assert not list(tmp_path.iterdir())
    assert pa.concat_tables(back).equals(pa.concat_tables(tables))
    # stopping early closes the file too
    spool = _Ipc(path)
    for t in tables:
        spool.write(t)
    spool.close()
    for _ in _read_ipc(path):
        break
    os.unlink(path)


@pytest.mark.thread_unsafe
def test_spool_abort_removes_the_partial_file(tmp_path: Path, strict_unlink: list[str]) -> None:
    from pyroid.cli import _Ipc

    path = str(tmp_path / "kept.arrow")
    spool = _Ipc(path)
    spool.write(_spool_tables()[0])  # the IPC writer and its file are still open
    spool.abort()
    spool.abort()  # idempotent
    assert _removed_under(strict_unlink, tmp_path) == [path + ".partial"]
    assert not list(tmp_path.iterdir())
