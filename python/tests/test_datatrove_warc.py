# mypy: disallow-subclassing-any=False
"""pyroid.integrations.datatrove.WarcReader against datatrove 0.10.1's own WarcReader.

Both readers get the same WARC files, written here (gzip members per record, as Common Crawl
writes them, or plain), on the local filesystem and on fsspec's memory filesystem, with every
``compression`` datatrove takes. The documents out (``text``, ``id``, ``metadata``), the
exception at the end (type, message, after the same documents), warcio's log warnings and every
statistic except time must be equal. The stock reader needs warcio, faust-cchardet and
python-magic; libmagic itself is not needed (every record but one names its payload type, and
that one runs with a stand-in ``magic`` module on both sides).

The helpers at the top (``response``, ``write_warc``, ...) build the WARC files for the other
warc integration tests too.
"""

from __future__ import annotations

import bz2
import copy
import gzip
import importlib.machinery
import importlib.metadata
import logging
import pickle
import sys
import types
import uuid
import zlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------------- WARC fixtures

DATE = "2026-09-20T01:02:03Z"


def record(rec_type: str, headers: Sequence[tuple[str, str]], block: bytes) -> bytes:
    """One WARC/1.0 record (``Content-Length`` added), with its two trailing CRLFs."""
    head = [("WARC-Type", rec_type), *headers, ("Content-Length", str(len(block)))]
    lines = b"".join(f"{k}: {v}\r\n".encode() for k, v in head)
    return b"WARC/1.0\r\n" + lines + b"\r\n" + block + b"\r\n\r\n"


def rid(n: int) -> str:
    return f"<urn:uuid:{uuid.UUID(int=n)}>"


def response(
    n: int,
    url: str,
    body: bytes,
    *,
    payload_type: str | None = "text/html",
    http: Sequence[tuple[str, str]] = (("Content-Type", "text/html; charset=utf-8"),),
    date: str = DATE,
    status: str = "200 OK",
) -> bytes:
    """A ``response`` record with an HTTP block (status line, headers, ``body``)."""
    block = f"HTTP/1.1 {status}\r\n".encode() + b"".join(f"{k}: {v}\r\n".encode() for k, v in http)
    headers = [
        ("WARC-Date", date),
        ("WARC-Record-ID", rid(n)),
        ("Content-Type", "application/http; msgtype=response"),
        ("WARC-Target-URI", url),
    ]
    if payload_type is not None:
        headers.append(("WARC-Identified-Payload-Type", payload_type))
    return record("response", headers, block + b"\r\n" + body)


def request(n: int, url: str) -> bytes:
    block = f"GET / HTTP/1.1\r\nHost: {url.split('/')[2]}\r\n\r\n".encode()
    headers = [
        ("WARC-Date", DATE),
        ("WARC-Record-ID", rid(n)),
        ("Content-Type", "application/http; msgtype=request"),
        ("WARC-Target-URI", url),
    ]
    return record("request", headers, block)


def warcinfo(n: int) -> bytes:
    block = b"software: pyroid tests\r\nformat: WARC File Format 1.0\r\n"
    headers = [
        ("WARC-Date", DATE),
        ("WARC-Record-ID", rid(n)),
        ("Content-Type", "application/warc-fields"),
    ]
    return record("warcinfo", headers, block)


def metadata(n: int, url: str) -> bytes:
    headers = [
        ("WARC-Date", DATE),
        ("WARC-Record-ID", rid(n)),
        ("Content-Type", "application/warc-fields"),
        ("WARC-Target-URI", url),
    ]
    return record("metadata", headers, b"fetchTimeMs: 12\r\n")


def conversion(n: int, url: str, text: str) -> bytes:
    """A WET ``conversion`` record (datatrove keeps ``text/plain`` ones)."""
    headers = [
        ("WARC-Date", DATE),
        ("WARC-Record-ID", rid(n)),
        ("WARC-Target-URI", url),
        ("WARC-Identified-Payload-Type", "text/plain"),
        ("Content-Type", "text/plain"),
    ]
    return record("conversion", headers, text.encode())


def chunked(data: bytes, size: int = 7) -> bytes:
    parts = [data[i : i + size] for i in range(0, len(data), size)]
    return b"".join(b"%x\r\n" % len(p) + p + b"\r\n" for p in parts) + b"0\r\n\r\n"


def html(title: str, *paragraphs: str) -> str:
    ps = "".join(f"<p>{p}</p>\n" for p in paragraphs)
    return f"<html><head><title>{title}</title></head><body><article>\n{ps}</article></body></html>"


def noise(n: int, size: int = 24_000) -> bytes:
    """A page of ``size`` bytes of seeded random words: it barely compresses."""
    import random

    rng = random.Random(n)
    words = [
        "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=rng.randint(2, 9)))
        for _ in range(size // 5)
    ]
    return html(f"Noise {n}", " ".join(words)[:size]).encode()


def big_records(groups: int) -> list[bytes]:
    """``groups`` times :func:`sample_records` and a large page (6 groups: over 128 KiB of gzip)."""
    out: list[bytes] = []
    for g in range(groups):
        out += sample_records(1000 * g)
        out.append(response(1000 * g + 999, f"http://example.com/noise/{g}", noise(g)))
    return out


def members(records: Sequence[bytes]) -> tuple[bytes, list[int]]:
    """The gzip file of ``records`` and where each member starts."""
    parts = [gzip.compress(r, mtime=0) for r in records]
    starts = [0]
    for p in parts:
        starts.append(starts[-1] + len(p))
    return b"".join(parts), starts


def write_warc(path: Path, records: Sequence[bytes], *, gz: bool = True) -> Path:
    """Write ``records``: one gzip member each (as Common Crawl), or plain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for r in records:
            f.write(gzip.compress(r, mtime=0) if gz else r)
    return path


def sample_records(start: int = 0) -> list[bytes]:
    """Every kind of record datatrove's ``process_record`` keeps or skips, and how it decodes."""
    n = start
    page = html("A page", "The first paragraph has words in it.", "And a second one, too.")
    latin1 = html("Caf\xe9", "Le caf\xe9 \xe9tait tr\xe8s bon, \xe0 la fran\xe7aise. " * 8)
    out = [
        warcinfo(n),
        request(n + 1, "http://example.com/"),
        response(n + 2, "http://example.com/", page.encode()),
        metadata(n + 3, "http://example.com/"),
        response(
            n + 4,
            "http://example.org/caf\xe9",
            latin1.encode("latin-1"),
            http=(("Content-Type", "text/html"),),
        ),
        response(
            n + 5, "http://example.org/doc.pdf", b"%PDF-1.4 ...", payload_type="application/pdf"
        ),
        response(n + 6, "http://example.net/empty", b""),
        response(
            n + 7,
            "http://example.net/chunked",
            chunked(gzip.compress(page.encode(), mtime=0)),
            http=(
                ("Content-Type", "text/html"),
                ("Transfer-Encoding", "chunked"),
                ("Content-Encoding", "gzip"),
            ),
        ),
        response(
            n + 8, "http://example.net/x.xhtml", page.encode(), payload_type="application/xhtml+xml"
        ),
        response(n + 9, "http://example.net/a page with spaces", page.encode()),
        conversion(n + 10, "http://example.com/wet", "Plain text from a WET file.\nSecond line."),
        response(
            n + 11,
            "http://example.net/deflate",
            zlib.compress(page.encode()),
            http=(("Content-Encoding", "deflate"),),
        ),
        response(n + 12, "http://example.net/plain", b"just text", payload_type="text/plain"),
    ]
    return out


# ------------------------------------------------------------------------------------ the tests

pytest.importorskip("datatrove")
pytest.importorskip("fsspec")


def _have(dist: str) -> bool:
    try:
        importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


if not all(_have(d) for d in ("warcio", "faust-cchardet", "python-magic")):
    pytest.skip(
        "needs warcio, faust-cchardet and python-magic for datatrove's own WarcReader",
        allow_module_level=True,
    )

from datatrove.pipeline.readers import WarcReader as StockWarcReader  # type: ignore[import-not-found,import-untyped,unused-ignore]  # noqa: E402

from pyroid.integrations import datatrove as pdt  # noqa: E402

pytestmark = pytest.mark.thread_unsafe(reason="sys.modules['magic'] and the fsspec memory store")


class _Magic:
    """A stand-in for python-magic (libmagic may be missing): HTML for anything with '<html'."""

    calls = 0

    @classmethod
    def from_buffer(cls, buf: bytes, mime: bool = False) -> str:
        cls.calls += 1
        return "text/html" if b"<html" in buf[:4096] else "application/octet-stream"


@pytest.fixture(autouse=True)
def magic(monkeypatch: pytest.MonkeyPatch) -> type[_Magic]:
    """``import magic`` gives the stand-in on both sides."""
    stub = types.ModuleType("magic")
    stub.__spec__ = importlib.machinery.ModuleSpec("magic", None)
    stub.from_buffer = _Magic.from_buffer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "magic", stub)
    _Magic.calls = 0
    return _Magic


def read(cls: Any, folder: Any, **kw: Any) -> dict[str, Any]:
    """Run a reader over ``folder``: its documents, the exception it ended with, its stats."""
    reader = cls(folder, **kw)
    docs: list[tuple[str, str, dict[str, Any]]] = []
    error = None
    try:  # list.extend keeps the documents before an exception
        docs.extend((d.text, d.id, dict(d.metadata)) for d in reader.run(rank=0, world_size=1))
    except Exception as e:  # compared below
        error = (type(e), str(e))
    stats = reader.stats.stats.to_dict()
    return {"docs": docs, "error": error, "stats": stats}


def both(folder: Any, caplog: pytest.LogCaptureFixture, **kw: Any) -> dict[str, Any]:
    """The stock and the pyroid reader on the same input must agree; return the result."""
    caplog.set_level(logging.WARNING, logger="warcio")
    caplog.clear()
    want = read(StockWarcReader, folder, **kw)
    want_log = [(r.name, r.getMessage()) for r in caplog.records]
    caplog.clear()
    got = read(pdt.WarcReader, folder, **kw)
    got_log = [(r.name, r.getMessage()) for r in caplog.records]
    assert got["docs"] == want["docs"]
    assert got["error"] == want["error"]
    assert got["stats"] == want["stats"]
    assert got_log == want_log
    return got


@pytest.mark.parametrize("gz", [True, False], ids=["gzip", "plain"])
@pytest.mark.parametrize("compression", ["infer", "gzip", None])
def test_local_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, gz: bool, compression: Any
) -> None:
    name = "a.warc.gz" if gz else "a.warc"
    write_warc(tmp_path / "in" / name, sample_records(), gz=gz)
    write_warc(tmp_path / "in" / f"b{name[1:]}", sample_records(100), gz=gz)
    got = both(str(tmp_path / "in"), caplog, compression=compression)
    if not gz and compression == "gzip":
        assert got["error"] is not None
        assert got["error"][0].__name__ == "BadGzipFile"
        return
    # compression=None on .gz: warcio (and pyroid) decompress the members themselves
    assert got["error"] is None
    texts = [t for t, _, _ in got["docs"]]
    assert len(got["docs"]) == 2 * 7, texts
    assert any("tr\xe8s bon" in t for t in texts)  # latin-1 page: cchardet's guess
    assert any(t.startswith("Plain text from a WET file") for t in texts)
    meta = got["docs"][0][2]
    assert set(meta) == {"url", "date", "file_path"}
    assert meta["file_path"] == str(tmp_path / "in" / name)
    assert any("Replacing spaces" in m for _, _, m in caplog.record_tuples)


def test_metadata_limit_skip_and_adapter(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    for i in range(3):
        write_warc(tmp_path / "in" / f"{i}.warc.gz", sample_records(100 * i))

    def adapter(
        self: Any, data: dict[str, Any], path: str, id_in_file: int | str
    ) -> dict[str, Any]:
        return {
            "text": data["text"],
            "id": f"{Path(path).name}#{id_in_file}",
            "metadata": {"u": data["url"]},
        }

    both(str(tmp_path / "in"), caplog, limit=11, skip=4, default_metadata={"dump": "CC-TEST"})
    got = both(str(tmp_path / "in"), caplog, adapter=adapter, glob_pattern="*1.warc.gz")
    assert [i for _, i, _ in got["docs"]][:2] == ["1.warc.gz#2", "1.warc.gz#4"]


def test_no_payload_type_asks_magic(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, magic: type[_Magic]
) -> None:
    page = html("Typed by libmagic", "No WARC-Identified-Payload-Type on this one.").encode()
    records = [
        response(1, "http://example.com/a", page, payload_type=None),
        response(2, "http://example.com/b", b"\x00\x01binary", payload_type=None),
    ]
    write_warc(tmp_path / "in" / "a.warc.gz", records)
    got = both(str(tmp_path / "in"), caplog)
    assert [i for _, i, _ in got["docs"]] == [rid(1)]
    assert magic.calls == 4  # two records, twice


@pytest.mark.parametrize("compression", ["infer", None])
def test_errors_at_the_same_document(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, compression: Any
) -> None:
    records = big_records(8)
    data, starts = members(records)
    k = len(records) - 20  # a member well past the first 128 KiB read
    mid = (starts[k] + starts[k + 1]) // 2
    cases = {
        "cut": data[:mid],  # truncated inside member k
        "crc": data[: starts[k + 1] - 6]
        + b"\0\0"
        + data[starts[k + 1] - 4 :],  # bad CRC of member k
        "garbage": data[: starts[k]] + b"not gzip at all" + data[starts[k] :],  # between members
        "deflate": data[: mid - 8] + b"\xff" * 16 + data[mid + 8 :],  # damaged deflate data
        "header": data[: starts[k] + 5],  # inside a gzip header
        "padding": data + b"\0" * 1000,  # zero padding after the last member
    }
    errors = {}
    for name, blob in cases.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "a.warc.gz").write_bytes(blob)
        got = both(str(tmp_path / name), caplog, compression=compression)
        errors[name] = got["error"] and got["error"][0].__name__
        if name != "padding":
            assert len(got["docs"]) > 0, name
    if compression == "infer":  # datatrove's GzipFile under warcio
        assert errors["crc"] == errors["garbage"] == "BadGzipFile"
        assert errors["padding"] is None  # GzipFile skips zero padding
    else:  # warcio's own gzip reader
        assert errors["crc"] == errors["padding"] == "ArchiveLoadFailed"
    one = tmp_path / "one" / "a.warc.gz"
    one.parent.mkdir()
    one.write_bytes(gzip.compress(b"".join(records), mtime=0))  # one member, many records
    got = both(str(one.parent), caplog, compression=compression)
    if compression is None:  # warcio's own gzip reader: not chunked gzip
        from warcio.exceptions import ArchiveLoadFailed  # type: ignore[import-not-found,import-untyped,unused-ignore]

        # warcio's class, as stock datatrove raises it (not pyroid.warc.ArchiveLoadFailed)
        assert got["error"] is not None
        assert got["error"][0] is ArchiveLoadFailed
        with pytest.raises(ArchiveLoadFailed, match="gzip"):
            list(pdt.WarcReader(str(one.parent), compression=None).run())


def test_other_compressions_use_datatrove_open(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    raw = b"".join(sample_records())
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "a.warc.bz2").write_bytes(bz2.compress(raw))
    got = both(str(tmp_path / "in"), caplog)
    assert len(got["docs"]) == 7
    reader = pdt.WarcReader(str(tmp_path / "in"))
    assert reader._plan("a.warc.bz2") == ("datatrove", None)
    assert reader._plan("a.warc.gz") == ("path", "gzip")
    assert reader._plan("a.warc") == ("path", None)
    assert pdt.WarcReader(str(tmp_path / "in"), compression="zstd")._plan("a.warc") == (
        "datatrove",
        None,
    )


@pytest.mark.parametrize("gz", [True, False], ids=["gzip", "plain"])
def test_memory_filesystem(caplog: pytest.LogCaptureFixture, gz: bool) -> None:
    from fsspec.implementations.memory import MemoryFileSystem  # type: ignore[import-not-found,import-untyped,unused-ignore]

    fs = MemoryFileSystem()
    root = f"/pyroid-warc-test-{uuid.uuid4().hex}"
    try:
        for i in range(2):
            data = b"".join(gzip.compress(r, mtime=0) if gz else r for r in sample_records(50 * i))
            fs.pipe(f"{root}/{i}.warc{'.gz' if gz else ''}", data)
        data, starts = members(big_records(8))
        fs.pipe(f"{root}/z-cut.warc.gz", data[: (starts[-30] + starts[-29]) // 2])
        got = both((f"memory://{root}", fs), caplog)
        assert len(got["docs"]) > 14
        assert got["error"] is not None
        assert got["error"][0] is EOFError
        assert got["docs"][0][2]["file_path"].startswith("memory://")
        reader = pdt.WarcReader((f"memory://{root}", fs))
        assert reader._plan("0.warc.gz") == ("file", "gzip")
    finally:
        fs.rm(root, recursive=True)


def test_pickle_and_deepcopy(tmp_path: Path) -> None:
    write_warc(tmp_path / "in" / "a.warc.gz", sample_records())
    reader = pdt.WarcReader(str(tmp_path / "in"), limit=3)
    # datatrove's executors deep-copy (one worker) or pickle (several) the pipeline: our own object
    for clone in (copy.deepcopy(reader), pickle.loads(pickle.dumps(reader))):
        assert isinstance(clone, StockWarcReader)
        assert len(list(clone.run())) == 3


def test_signature_and_requirements() -> None:
    assert pdt.WarcReader.__init__ is StockWarcReader.__init__  # the same arguments
    assert "WarcReader" in pdt.__all__
    assert issubclass(pdt.WarcReader, StockWarcReader)


@pytest.mark.parametrize("name", ["missing.warc.gz", "missing.warc"])
def test_missing_file(tmp_path: Path, name: str) -> None:
    errors = []
    for cls in (StockWarcReader, pdt.WarcReader):
        with pytest.raises(FileNotFoundError) as e:
            list(cls(str(tmp_path)).read_file(name))
        errors.append((str(e.value), e.value.filename, e.value.errno))
    assert errors[0] == errors[1]
