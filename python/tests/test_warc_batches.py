"""pyroid.warc.iter_batches: the rows against iter_documents and warcio, and the Arrow export."""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.warc as pw


def _sibling(name: str) -> Any:
    """A sibling test module, imported by path (the suite runs without python/tests on sys.path)."""
    key = f"_pyroid_test_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


tw = _sibling("test_warc")
td = _sibling("test_warc_documents")

COLUMNS = ["id", "url", "date", "mime_type", "content_type", "html", "text"]


@pytest.fixture(autouse=True)
def stub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    td.install_stubs(monkeypatch)


def rows(source: Any, **kw: Any) -> tuple[list[dict[str, Any]], Any]:
    """All rows (as dicts), and the exception that ended the batches."""
    out: list[dict[str, Any]] = []
    err = None
    it = pw.iter_batches(source, **kw)
    try:
        for b in it:
            assert b.num_rows > 0
            assert b.column_names == COLUMNS
            d = b.to_pydict()
            out += [dict(zip(COLUMNS, r, strict=True)) for r in zip(*d.values(), strict=True)]
    except Exception as e:
        err = tw.exc_summary(e)
    return out, err


def docs(source: Any, **kw: Any) -> tuple[list[tuple[int, Any]], Any]:
    out: list[tuple[int, Any]] = []
    err = None
    try:
        for ri, d in pw.iter_documents(source, **kw):
            out.append((ri, d))
    except Exception as e:
        err = tw.exc_summary(e)
    return out, err


def expected_rows(path: Path, gz: bool) -> list[dict[str, Any]]:
    """What warcio gives for the records iter_batches lists (datatrove's first checks)."""
    wa: Any = pytest.importorskip("warcio.archiveiterator")
    out = []
    with path.open("rb") as raw:
        f: Any = gzip.GzipFile(fileobj=raw, mode="rb") if gz else raw
        try:
            for rec in wa.ArchiveIterator(f):
                if rec.rec_type not in ("response", "conversion"):
                    continue
                mime = rec.rec_headers.get("WARC-Identified-Payload-Type", None)
                if mime is not None and not pw._mime_ok(mime, rec.rec_type == "conversion"):  # type: ignore[attr-defined]
                    continue
                h = rec.rec_headers
                http = rec.http_headers
                out.append(
                    {
                        "id": h.get("WARC-Record-ID"),
                        "mime_type": mime,
                        "content_type": http.get_header("content-type") if http else None,
                        "html": rec.content_stream().read(),
                    }
                )
        except Exception:
            pass
    return out


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("batch_size", [1, 2, 1000])
@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gz"])
@pytest.mark.parametrize("name", sorted(td.DOC_CASES))
def test_rows_match_documents_and_warcio(
    name: str, gz: bool, batch_size: int, tmp_path: Path
) -> None:
    data = td.DOC_CASES[name]
    path = tmp_path / ("in.warc.gz" if gz else "in.warc")
    path.write_bytes(gzip.compress(data, mtime=0) if gz else data)
    got, err = rows(path, batch_size=batch_size)
    want_docs, doc_err = docs(path)
    with_text = [r for r in got if r["text"] is not None]
    if doc_err is not None and doc_err[0] == "KeyError":
        # iter_documents raises KeyError at the record; the batch has null url/date there.
        assert any(r["url"] is None or r["date"] is None for r in got)
        assert err is None or err[0] != "KeyError"
    else:
        assert err == doc_err
        assert [{k: r[k] for k in ("text", "id", "url", "date")} for r in with_text] == [
            d for _, d in want_docs
        ]
    want = expected_rows(path, gz)
    if err is not None:  # a prefix: magic may raise where warcio reads on
        want = want[: len(got)]
    assert [{k: r[k] for k in ("id", "mime_type", "content_type", "html")} for r in got] == want


def test_batch_api(tmp_path: Path) -> None:
    path = tmp_path / "in.warc"
    path.write_bytes(td.DOC_CASES["kept_and_skipped"] + td.DOC_CASES["encodings"])
    batches = list(pw.iter_batches(path, batch_size=4))
    assert [b.num_rows for b in batches] == [4, 4, 1]
    b = batches[0]
    assert len(b) == 4
    assert b.num_columns == 7
    html = b.column("html")
    assert isinstance(html, pyroid.Column)
    assert html.type == "large_binary"
    assert html.name == "html"
    assert html.to_pylist()[0] == b"<html>one</html>"
    assert b.column(-1).name == "text"
    assert b.column(0).name == "id"
    assert b.column("text").type == "large_string"
    with pytest.raises(KeyError):
        b.column("nope")
    with pytest.raises(IndexError):
        b.column(7)
    with pytest.raises(TypeError):
        b.column(1.5)  # type: ignore[arg-type]
    assert repr(b).startswith("pyroid.warc.Batch(num_rows=4")
    with pytest.raises(ValueError, match="batch_size"):
        pw.iter_batches(path, batch_size=0)
    with pytest.raises(ValueError, match="inferred from its extension"):
        pw.iter_batches(tmp_path / "x.zst")


def test_pyarrow_and_polars_take_batches(tmp_path: Path) -> None:
    pa: Any = pytest.importorskip("pyarrow")
    path = tmp_path / "in.warc.gz"
    path.write_bytes(gzip.compress(td.DOC_CASES["kept_and_skipped"] + td.DOC_CASES["magic"]))
    batches = list(pw.iter_batches(path, batch_size=3))
    rb = pa.record_batch(batches[0])
    assert rb.schema.names == COLUMNS
    assert [str(t) for t in rb.schema.types] == ["large_string"] * 5 + [
        "large_binary",
        "large_string",
    ]
    assert rb.to_pydict() == batches[0].to_pydict()
    table = pa.table(batches[1])
    assert table.num_rows == batches[1].num_rows
    assert table.to_pydict() == batches[1].to_pydict()
    whole = pa.Table.from_batches([pa.record_batch(b) for b in batches])
    assert whole.num_rows == sum(b.num_rows for b in batches)
    assert pa.schema(batches[0]) == rb.schema
    col = batches[0].column("html")
    assert pa.array(col).type == pa.large_binary()
    assert pa.chunked_array(col).to_pylist() == col.to_pylist()
    del batches, col  # the exported buffers outlive the pyroid objects
    assert whole.column("id").null_count == 0
    pl: Any = pytest.importorskip("polars")
    df = pl.DataFrame(next(pw.iter_batches(path)))
    assert df.columns == COLUMNS
    assert df.height == whole.num_rows
