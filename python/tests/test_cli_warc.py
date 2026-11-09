"""``pyroid clean --recipe fineweb-warc``: datatrove's examples/fineweb.py from WARC files.

The WARC files are written here from FineWeb documents of the dev slice (as HTML pages, with
copies across files for MinHash, emails and IP addresses for PII, a German page, a blocked URL, a
page with no text and records datatrove skips). The command's final documents must equal those of
``bench/fineweb_warc.py native --reader warcio``, which reads the files with warcio and
datatrove's own ``process_record`` and runs MinHash in one process: the same ids and texts
(the bench's ``after_pii`` digest). Needs the dev slice and lid.176.bin under ``$PYROID_DATA``,
and datatrove, warcio, faust-cchardet, python-magic, tldextract and fasteners.
"""

from __future__ import annotations

import hashlib
import html as htmllib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("numpy")

from test_datatrove_warc import (  # noqa: E402
    html,
    metadata,
    request,
    response,
    warcinfo,
    write_warc,
)

from pyroid.cli import main  # noqa: E402

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
DEV = DATA / "fineweb" / "fineweb-dev-10k.parquet"
LID_BIN = DATA / "models" / "lid.176.bin"
BENCH = Path(__file__).resolve().parents[2] / "bench" / "fineweb_warc.py"
NEEDS = ("datatrove", "warcio", "cchardet", "magic", "tldextract", "fasteners", "orjson")

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses, worker processes and large models")


def _missing() -> list[str]:
    return [m for m in NEEDS if importlib.util.find_spec(m) is None]


needs_all = pytest.mark.skipif(
    bool(_missing()) or not DEV.exists() or not LID_BIN.exists(),
    reason=f"needs the dev slice and lid.176.bin under $PYROID_DATA, and {', '.join(NEEDS)}",
)

PII = "Write to jane.doe@example.org or reach the server at 93.184.216.34 for details."
GERMAN = (
    "Die Stadt liegt am Ufer des Flusses und ist seit dem Mittelalter ein wichtiger Ort fuer "
    "Kaufleute gewesen. Viele Menschen besuchen jedes Jahr die alte Kirche, das Rathaus und die "
    "schoenen Gassen der Altstadt, in denen es kleine Geschaefte und Cafes gibt."
)


def _page(text: str, title: str) -> bytes:
    paras = [htmllib.escape(line) for line in text.split("\n") if line.strip()]
    return html(htmllib.escape(title), *paras).encode()


def _corpus(folder: Path, n: int = 120) -> list[Path]:
    """Two WARC files of FineWeb pages (see the module docstring)."""
    rows = pq.read_table(DEV, columns=["text", "url"]).slice(0, 2 * n).to_pylist()
    files = []
    k = 0
    for f in range(2):
        recs = [warcinfo(k)]
        k += 1
        part = rows[f * n : (f + 1) * n]
        if f == 1:
            part = part + rows[:12]  # copies of file 0's first pages: MinHash keeps the first
        for i, r in enumerate(part):
            text = r["text"]
            if i % 17 == 3:
                text = text + "\n" + PII
            url = r["url"] if not (f == 1 and i >= n) else f"http://copy.example.com/{i}"
            recs.append(request(k, url))
            recs.append(response(k + 1, url, _page(text, f"Page {f}-{i}")))
            recs.append(metadata(k + 2, url))
            k += 3
        recs.append(response(k, "http://example.de/stadt", _page("\n".join([GERMAN] * 3), "Stadt")))
        recs.append(response(k + 1, "http://pornhub.com/video", _page(rows[f]["text"], "Blocked")))
        recs.append(response(k + 2, "http://example.com/empty", b"<html><body></body></html>"))
        recs.append(
            response(k + 3, "http://example.com/a.pdf", b"%PDF-1.4", payload_type="application/pdf")
        )
        k += 4
        files.append(write_warc(folder / f"part-{f}.warc.gz", recs))
    return files


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    if _missing() or not DEV.exists():
        pytest.skip("needs the dev slice and the reader's dependencies")
    return _corpus(tmp_path_factory.mktemp("warcs"))


@pytest.fixture(autouse=True)
def _magic(monkeypatch: pytest.MonkeyPatch) -> None:
    """libmagic may be missing: every record here names its payload type, so it is never asked."""
    try:
        import magic  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: F401
    except ImportError:
        stub = types.ModuleType("magic")
        stub.__spec__ = importlib.machinery.ModuleSpec("magic", None)
        monkeypatch.setitem(sys.modules, "magic", stub)


def _digest(docs: list[tuple[str, str]]) -> str:
    """bench/fineweb_warc.py's digest: sha256 over id \\0 text \\0 of the documents sorted by id."""
    h = hashlib.sha256()
    for i, t in sorted(docs):
        h.update(
            i.encode("utf-8", "surrogatepass") + b"\0" + t.encode("utf-8", "surrogatepass") + b"\0"
        )
    return h.hexdigest()


def _rows(path: Path) -> Any:
    if path.is_file():
        return pq.read_table(path)
    return pa.concat_tables([pq.read_table(p) for p in sorted(path.glob("*.parquet"))])


def _docs(t: Any) -> list[tuple[str, str]]:
    return list(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))


def _clean(*args: str) -> None:
    assert main(["clean", *args, "--recipe", "fineweb-warc", "--lid-model", str(LID_BIN)]) == 0


@needs_all
def test_same_documents_as_the_bench(corpus: list[Path], tmp_path: Path) -> None:
    out = tmp_path / "out.parquet"
    excl = tmp_path / "dropped.parquet"
    stats = tmp_path / "stats.json"
    _clean(
        *map(str, corpus),
        str(out),
        "--excluded",
        str(excl),
        "--stats-json",
        str(stats),
        "--dump",
        "CC-T",
    )
    t = _rows(out)
    assert t.column_names == [
        "text",
        "id",
        "dump",
        "url",
        "date",
        "file_path",
        "language",
        "language_score",
    ]
    got = list(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))
    assert len(got) > 50
    assert not any(PII.split()[2] in x for _, x in got)  # the email is replaced
    assert set(t.column("dump").to_pylist()) == {"CC-T"}
    assert set(t.column("language").to_pylist()) == {"en"}
    assert set(t.column("file_path").to_pylist()) == {str(p) for p in corpus}
    # the bench, reading with warcio and datatrove's process_record
    bench_out = tmp_path / "bench"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONWARNINGS",)}
    p = subprocess.run(
        [
            sys.executable,
            str(BENCH),
            "native",
            "--warcs",
            str(corpus[0].parent),
            "--out",
            str(bench_out),
            "--lid-model",
            str(LID_BIN),
            "--reader",
            "warcio",
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=600,
    )
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    summary = json.loads((bench_out / "summary.json").read_text())
    want = summary["digests"]["after_pii"]
    assert (len(got), _digest(got)) == (want["docs"], want["sha256"])
    # every kind of drop, with datatrove's reasons
    d = _rows(excl)
    reasons = set(d.column("filter_reason").to_pylist())
    assert {"url_filter:domain", "trafilatura", "lid", "minhash"} <= reasons, reasons
    s = json.loads(stats.read_text())
    assert s["docs_in"] == summary["docs"]["warc_reader"]
    assert s["docs_out"] == len(got) == summary["docs"]["pii"]
    assert s["removed"]["minhash"] == summary["docs"]["fineweb_quality"] - len(got)


def _magic_in_children() -> bool:
    """python-magic works in a fresh interpreter (the spawned --workers processes need it)."""
    p = subprocess.run([sys.executable, "-c", "import magic"], capture_output=True, check=False)
    return p.returncode == 0


@needs_all
def test_workers_directory_out_and_resume(corpus: list[Path], tmp_path: Path) -> None:
    if not _magic_in_children():
        pytest.skip("datatrove's reading imports magic in each worker: needs libmagic")
    one = tmp_path / "one.parquet"
    _clean(str(corpus[0].parent), str(one))
    out = tmp_path / "out"
    _clean(str(corpus[0].parent), str(out), "--workers", "2", "--threads", "1")
    assert sorted(p.name for p in out.glob("*.parquet")) == ["part-0.parquet", "part-1.parquet"]
    a, b = _rows(one), _rows(out)
    assert sorted(a.column("id").to_pylist()) == sorted(b.column("id").to_pylist())
    assert _digest(_docs(a)) == _digest(_docs(b))
    before = {p: p.stat().st_mtime_ns for p in out.glob("*.parquet")}
    _clean(str(corpus[0].parent), str(out))  # finished: nothing is redone
    assert {p: p.stat().st_mtime_ns for p in out.glob("*.parquet")} == before


@needs_all
def test_no_dedup_limit_and_pii(corpus: list[Path], tmp_path: Path) -> None:
    full = tmp_path / "full.parquet"
    _clean(str(corpus[1]), str(full), "--no-dedup")
    t = _rows(full)
    ids = t.column("id").to_pylist()
    assert len(ids) == len(set(ids))
    assert not any("jane.doe@example.org" in x for x in t.column("text").to_pylist())
    assert "dump" not in t.column_names
    small = tmp_path / "small.jsonl"
    _clean(str(corpus[1]), str(small), "--no-dedup", "--limit", "20")
    rows = [json.loads(line) for line in small.read_text().splitlines()]
    assert 0 < len(rows) <= 20
    assert [r["id"] for r in rows] == [i for i in ids if i in {r["id"] for r in rows}]


@needs_all
def test_pages_with_no_text_and_copies(corpus: list[Path], tmp_path: Path) -> None:
    excl = tmp_path / "excl"
    _clean(*map(str, corpus), str(tmp_path / "out"), "--excluded", str(excl))
    assert sorted(p.name for p in excl.glob("*.parquet")) == ["part-0.parquet", "part-1.parquet"]
    d = _rows(excl)
    rows = list(
        zip(d.column("url").to_pylist(), d.column("filter_reason").to_pylist(), strict=True)
    )
    copies = [u for u, r in rows if r == "minhash"]
    assert copies
    assert all(u.startswith("http://copy.example.com/") for u in copies)  # the later copy
    assert [u for u, r in rows if r == "trafilatura"] == ["http://example.com/empty"] * 2
    assert [u for u, r in rows if r == "lid"] == ["http://example.de/stadt"] * 2


def test_usage_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "in.parquet"
    pq.write_table(pa.table({"text": ["x"]}), src)
    with pytest.raises(SystemExit, match="apply to --recipe fineweb-warc"):
        main(
            ["clean", str(src), str(tmp_path / "o.parquet"), "--recipe", "fineweb", "--limit", "3"]
        )
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "notes.txt").write_text("not a WARC file")
    if not _missing():
        assert (
            main(
                [
                    "clean",
                    str(tmp_path / "w"),
                    str(tmp_path / "o.parquet"),
                    "--recipe",
                    "fineweb-warc",
                ]
            )
            == 1
        )
        assert "no WARC files" in capsys.readouterr().err


def test_help_lists_the_recipe(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["clean", "--help"])
    out = capsys.readouterr().out
    for word in ("fineweb-warc", "--dump", "--limit"):
        assert word in out


def test_warc_names() -> None:
    from pyroid import cli

    assert cli._warc_stem("a/x.warc.gz") == "a/x"
    assert cli._warc_stem("CC-MAIN-1.warc.wet.gz") == "CC-MAIN-1.warc"
    assert cli._is_warc("X.ARC.GZ")
    assert cli._is_warc("y.warc")
    assert not cli._is_warc("y.parquet")
