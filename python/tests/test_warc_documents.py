"""pyroid.warc.iter_documents against datatrove 0.10.1's WarcReader loop.

The reference is ``WarcReader.read_file`` without the Document step::

    f = open(path, "rb")                         # fsspec: wrapped in gzip.GzipFile for .gz
    for ri, record in enumerate(warcio.ArchiveIterator(f)):
        doc = process_record(record)             # datatrove.pipeline.readers.warc
        if doc:
            yield ri, doc

with datatrove's own ``process_record`` where datatrove is installed (else pyroid's copy of
it, still over warcio's reader). ``magic`` is a stub in both (libmagic is often missing), and
``cchardet`` is faust-cchardet where it has wheels, else a stub. Traces hold the documents, the
stderr/log messages between them, and the exception that ends the run.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
CRLF = b"\r\n"

pytestmark = pytest.mark.thread_unsafe  # sys.modules stubs, sys.stderr, the warcio logger


def _magic_stub() -> types.ModuleType:
    m = types.ModuleType("magic")

    def from_buffer(buf: bytes, mime: bool = False) -> str:
        assert mime is True
        if buf.startswith(b"TXT"):
            return "text/plain"
        if buf.startswith(b"BOOM"):
            raise RuntimeError("magic stub failure")
        return "text/html" if b"<" in buf[:200] else "application/octet-stream"

    m.from_buffer = from_buffer  # type: ignore[attr-defined]
    return m


def _cchardet_stub() -> types.ModuleType:
    m = types.ModuleType("cchardet")

    def detect(buf: bytes) -> dict[str, Any]:
        if buf.startswith(b"\xff\xfe"):
            return {"encoding": "UTF-16"}
        if b"\x81" in buf:
            return {"encoding": None}
        if b"\x82" in buf:
            return {"encoding": "NO-SUCH-CODEC"}
        return {"encoding": "WINDOWS-1252"}

    m.detect = detect  # type: ignore[attr-defined]
    return m


def install_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``magic`` stub always; faust-cchardet if it imports, else a stub."""
    monkeypatch.setitem(sys.modules, "magic", _magic_stub())
    try:
        importlib.import_module("cchardet")
    except ImportError:
        monkeypatch.setitem(sys.modules, "cchardet", _cchardet_stub())


@pytest.fixture(autouse=True)
def stub_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    install_stubs(monkeypatch)


def process_record() -> Callable[[Any], Any]:
    try:
        mod: Any = importlib.import_module("datatrove.pipeline.readers.warc")
    except ImportError:
        mod = pw
        return mod._process_record  # type: ignore[no-any-return]
    return mod.process_record  # type: ignore[no-any-return]


def run(events: list[tuple[Any, ...]], docs: Iterator[tuple[int, Any]]) -> None:
    try:
        for ri, d in docs:
            events.append(("doc", ri, list(d.items())))
    except Exception as e:
        events.append(("exc", *tw.exc_summary(e)))
        try:
            nxt = next(docs)
            events.append(("after", "doc", nxt[0]))
        except StopIteration:
            events.append(("after", "stop"))
        except Exception as e2:
            events.append(("after", *tw.exc_summary(e2)))


def reference(path: Path, gz: bool) -> list[tuple[Any, ...]]:
    wa: Any = pytest.importorskip("warcio.archiveiterator")
    pr = process_record()

    def docs() -> Iterator[tuple[int, dict[str, Any]]]:
        with path.open("rb") as raw:
            f: Any = gzip.GzipFile(fileobj=raw, mode="rb") if gz else raw
            for ri, record in enumerate(wa.ArchiveIterator(f)):
                d = pr(record)
                if not d:
                    continue
                yield ri, d

    events: list[tuple[Any, ...]] = []
    with tw.capture(events):
        run(events, docs())
    return events


def candidate(docs: Callable[[], Iterator[tuple[int, Any]]]) -> list[tuple[Any, ...]]:
    events: list[tuple[Any, ...]] = []
    with tw.capture(events):
        run(events, docs())
    return events


def check(data: bytes, tmp: Path, *, gz: bool) -> list[tuple[Any, ...]]:
    """iter_documents on ``data`` (a .warc.gz if ``gz``), through every source kind."""
    path = tmp / ("in.warc.gz" if gz else "in.warc")
    path.write_bytes(data)
    want = reference(path, gz)
    comp: Literal["gzip"] | None = "gzip" if gz else None
    assert candidate(lambda: pw.iter_documents(path)) == want
    assert candidate(lambda: pw.iter_documents(str(path), compression=comp)) == want
    with path.open("rb") as f:
        assert candidate(lambda: pw.iter_documents(f, compression=comp)) == want
    with path.open("rb") as f:
        assert candidate(lambda: pw.iter_documents(io.BytesIO(f.read()), compression=comp)) == want
    return want


def doc_record(
    content: bytes,
    *,
    rec_type: str = "response",
    payload_type: str | None = "text/html",
    uri: str | None = "http://example.com/",
    date: str | None = "2024-01-02T03:04:05Z",
    fields: tuple[tuple[str, str], ...] = (),
    http: bool = True,
    **kw: Any,
) -> bytes:
    block = tw.http_block(content, **kw) if http else content
    out: bytes = tw.warc_record(
        block, rec_type=rec_type, uri=uri, date=date, payload_type=payload_type, fields=fields
    )
    return out


DOC_CASES: dict[str, bytes] = {
    "kept_and_skipped": b"".join(
        [
            tw.warcinfo(),
            tw.request(),
            doc_record(b"<html>one</html>"),
            doc_record(b"<html>xhtml</html>", payload_type="application/xhtml+xml"),
            doc_record(b"png", payload_type="image/png"),
            doc_record(b"TXT plain", rec_type="conversion", payload_type="text/plain", http=False),
            doc_record(b"TXT plain response", payload_type="text/plain"),
            doc_record(b"metadata", rec_type="metadata", http=False, payload_type=None),
        ]
    ),
    "magic": b"".join(
        [
            doc_record(b"<html>no payload type</html>", payload_type=None),
            doc_record(b"binary \x00\x01", payload_type=None),
            doc_record(b"TXT conversion", rec_type="conversion", payload_type=None, http=False),
            doc_record(b"TXT response", payload_type=None),
        ]
    ),
    "magic_raises": doc_record(b"<p>ok</p>") + doc_record(b"BOOM", payload_type=None),
    "encodings": b"".join(
        [
            doc_record(b"<p>caf\xe9</p>"),
            doc_record(b"<p>\xff\xfeutf16</p>"),
            doc_record(b"<p>\x81 no guess</p>"),
            doc_record(b"<p>\x82 bad codec</p>"),
            doc_record("<p>ok ✓</p>".encode()),
            doc_record(b"<p>\xed\xa0\x80 surrogate</p>"),
        ]
    ),
    "content_encodings": b"".join(
        [
            doc_record(b"<p>gz</p>" * 40, encoding="gzip"),
            doc_record(b"<p>chunked</p>" * 30, chunked=True),
            doc_record(b"<p>deflate</p>" * 30, encoding="deflate"),
        ]
    ),
    "missing_uri": doc_record(
        b"TXT x", rec_type="conversion", uri=None, http=False, payload_type="text/plain"
    )
    + doc_record(b"<p>after</p>"),
    "missing_date": doc_record(b"<p>no date</p>", date=None) + doc_record(b"<p>after</p>"),
    "missing_both_undecodable": doc_record(
        b"<p>\x81</p>", rec_type="conversion", uri=None, date=None, http=False
    )
    + doc_record(b"<p>after</p>"),
    "missing_id": tw.warc_record(
        tw.http_block(b"<p>no id</p>"), record_id=None, payload_type="text/html"
    ),
    "empty_uri": doc_record(
        b"TXT x",
        rec_type="conversion",
        uri="",
        http=False,
        payload_type="text/plain",
        fields=(("uri", "http://fallback/"),),
    ),
    "arc": tw.arc_file(
        [
            ("http://example.com/", tw.http_block(b"<html>arc</html>")),
            ("http://example.com/2", tw.http_block(b"<html>arc 2</html>")),
        ]
    ),
    "bad_length": doc_record(b"<p>a</p>")
    + tw.warc_record(b"abc", length=b"1")
    + doc_record(b"<p>b</p>"),
    "spaces": doc_record(b"<p>s</p>", uri="http://example.com/a b"),
    "garbage": b"not a warc\n",
    "truncated": (doc_record(b"<p>a</p>") + doc_record(b"<p>b</p>" * 300))[:-200],
    "empty": b"",
}


@pytest.mark.parametrize("gz", [False, True], ids=["plain", "gz"])
@pytest.mark.parametrize("name", sorted(DOC_CASES))
def test_documents_match_datatrove(name: str, gz: bool, tmp_path: Path) -> None:
    data = DOC_CASES[name]
    check(gzip.compress(data, mtime=0) if gz else data, tmp_path, gz=gz)


@pytest.mark.parametrize("name", sorted(DOC_CASES))
def test_documents_over_gzip_members(name: str, tmp_path: Path) -> None:
    """A .warc.gz with one member per record, as Common Crawl writes them."""
    data = DOC_CASES[name]
    parts = data.split(b"WARC/1.0\r\n")
    members = [b"WARC/1.0\r\n" + p for p in parts[1:]] if len(parts) > 1 else [data]
    check(tw.gz_members(members), tmp_path, gz=True)
    # Read as plain (compression=None): warcio's own per-member decompression.
    path = tmp_path / "members.warc.gz"
    path.write_bytes(tw.gz_members(members))
    want = reference(path, False)
    assert candidate(lambda: pw.iter_documents(path, compression=None)) == want


GZIP_DAMAGE: dict[str, Callable[[bytes], bytes]] = {
    "truncated_member": lambda g: g[:-10],
    "truncated_trailer": lambda g: g[:-4],
    "truncated_header": lambda g: g + b"\x1f\x8b\x08",
    "bad_crc": lambda g: g[:-8] + bytes([g[-8] ^ 1]) + g[-7:],
    "bad_length": lambda g: g[:-4] + bytes([g[-4] ^ 1]) + g[-3:],
    "bad_magic": lambda g: g + b"PK\x03\x04 not gzip",
    "zero_padding": lambda g: g + b"\x00" * 50,
    "corrupt_deflate": lambda g: g[:40] + b"\xff" * 8 + g[48:],
    "garbage_member": lambda g: g + gzip.compress(b"not a warc record at all\n"),
}


@pytest.mark.parametrize("damage", sorted(GZIP_DAMAGE))
def test_gzip_damage_matches_datatrove(damage: str, tmp_path: Path) -> None:
    data = tw.gz_members(
        [tw.warcinfo(), doc_record(b"<p>first</p>" * 100), doc_record(b"<p>last</p>" * 3000)]
    )
    check(GZIP_DAMAGE[damage](data), tmp_path, gz=True)


def test_common_crawl_slice_matches_datatrove(tmp_path: Path) -> None:
    """The first records of the CC-MAIN-2026-39 prefix, if it is here (dev container)."""
    src = Path("/target/data/cc2639-00000-100m.warc.gz")
    if not src.is_file():
        pytest.skip(f"{src} is not here")
    with src.open("rb") as f:
        head = f.read(3 << 20)
    check(head, tmp_path, gz=True)  # cut mid-member: ends with GzipFile's EOFError


_docs_strategy = st.lists(
    st.builds(
        doc_record,
        st.one_of(
            st.binary(max_size=200),
            st.text(max_size=100).map(lambda s: ("<p>" + s + "</p>").encode()),
            st.sampled_from([b"<p>caf\xe9</p>", b"TXT x", b"BOOM", b"\x81", b""]),
        ),
        rec_type=st.sampled_from(["response", "response", "conversion", "resource", "request"]),
        payload_type=st.sampled_from(["text/html", None, "text/plain", "image/gif"]),
        uri=st.sampled_from(["http://e.com/", None, "", "http://e.com/ sp"]),
        date=st.sampled_from(["2024-01-01T00:00:00Z", None, ""]),
        http=st.booleans(),
        chunked=st.booleans(),
        encoding=st.sampled_from([None, None, "gzip", "deflate"]),
    ),
    min_size=1,
    max_size=6,
)


@settings(
    max_examples=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(records=_docs_strategy, layout=st.sampled_from(["plain", "gz", "members"]), cut=st.data())
def test_random_documents_match_datatrove(
    records: list[bytes], layout: str, cut: st.DataObject, tmp_path: Path
) -> None:
    data = b"".join(records)
    if layout == "gz":
        data = gzip.compress(data, mtime=0)
    elif layout == "members":
        data = tw.gz_members(records)
    if cut.draw(st.booleans()) and data:
        data = data[: cut.draw(st.integers(0, len(data) - 1))]
    check(data, tmp_path, gz=layout != "plain")


def test_missing_cchardet_raises_at_the_first_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """datatrove imports cchardet and magic at its first record; so does iter_documents."""
    path = tmp_path / "in.warc"
    path.write_bytes(DOC_CASES["kept_and_skipped"])
    monkeypatch.setitem(sys.modules, "cchardet", None)  # import raises ImportError
    want = reference(path, False)
    assert want[0][:2] in {("exc", "ImportError"), ("exc", "ModuleNotFoundError")}
    assert candidate(lambda: pw.iter_documents(path)) == want
    # No record, no import: an empty file reads without error in both.
    empty = tmp_path / "empty.warc"
    empty.write_bytes(b"")
    assert candidate(lambda: pw.iter_documents(empty)) == reference(empty, False) == []
    # A file that fails before its first record raises that, not the ImportError.
    bad = tmp_path / "bad.warc"
    bad.write_bytes(b"garbage\n")
    assert candidate(lambda: pw.iter_documents(bad))[0][1] == "ArchiveLoadFailed"


def test_missing_magic_raises_at_the_first_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "in.warc.gz"
    path.write_bytes(gzip.compress(DOC_CASES["kept_and_skipped"]))
    monkeypatch.setitem(sys.modules, "magic", None)
    want = reference(path, True)
    assert want[0][:2] in {("exc", "ImportError"), ("exc", "ModuleNotFoundError")}
    assert candidate(lambda: pw.iter_documents(path)) == want


def test_compression_arguments(tmp_path: Path) -> None:
    for name in ("x.warc.zst", "x.warc.bz2", "x.warc.XZ", "x.lz4", "x.zip"):
        with pytest.raises(ValueError, match="inferred from its extension"):
            pw.iter_documents(tmp_path / name)
    with pytest.raises(ValueError, match="compression must be 'infer', 'gzip' or None"):
        pw.iter_documents(tmp_path / "x.warc", compression="zstd")  # type: ignore[arg-type]
    # A .GZ name is gzip; the file opens at the first next(), as datatrove's generator.
    docs = pw.iter_documents(tmp_path / "missing.warc.GZ")
    with pytest.raises(FileNotFoundError):
        next(docs)
    p = tmp_path / "plain.warc.gz"
    p.write_bytes(DOC_CASES["kept_and_skipped"])  # not gzip at all
    with pytest.raises(gzip.BadGzipFile, match="Not a gzipped file"):
        list(pw.iter_documents(p))
    assert [ri for ri, _ in pw.iter_documents(p, compression=None)] == [2, 3, 5]


def test_iterator_cls_gives_the_exception_and_logger(tmp_path: Path) -> None:
    class FailedError(Exception):
        pass

    class It(pw.ArchiveIterator):
        archive_load_failed = FailedError

    p = tmp_path / "bad.warc"
    p.write_bytes(b"garbage\n")
    with pytest.raises(FailedError, match="Unknown archive format"):
        list(pw.iter_documents(p, iterator_cls=It))


def test_documents_read_ahead_keeps_messages_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tw.warc_record(b"abc", length=b"1")
    data = b"".join(doc_record(f"<p>{i}</p>".encode()) + (bad if i == 5 else b"") for i in range(9))
    p = tmp_path / "in.warc"
    p.write_bytes(data)
    seen = []
    for ri, _ in pw.iter_documents(p):
        seen.append((ri, "Record not followed by newline" in capsys.readouterr().err))
    assert [ri for ri, warned in seen if warned] == [7]  # the record after the bad one
