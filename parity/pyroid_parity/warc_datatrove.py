"""The warc suite's datatrove level: ``process_record`` dicts, ``iter_documents``, ``iter_batches``, WarcReader runs.

The reference is datatrove 0.10.1's ``WarcReader.read_file`` without the Document step::

    with data_folder.open(path, "rb", compression="infer") as f:     # a GzipFile for .gz
        for ri, record in enumerate(ArchiveIterator(f)):              # warcio 1.8.2
            d = process_record(record)                                # datatrove.pipeline.readers.warc
            if d:
                yield ri, d

``process_record`` decodes the content stream as UTF-8, else with cchardet's guess, and calls libmagic
(python-magic) for records without ``WARC-Identified-Payload-Type``. Where libmagic is missing, ``ensure_magic``
puts a stub in ``sys.modules`` that raises ``MagicMissing``: the reference and the candidate then stop at the same
record with the same exception, and the comparison still holds up to there (the pinned digests, made with
libmagic, do not, and those checks skip).

Every view yields events like ``warc_compare.iter_events``: one per document (``ri``, the dict's keys, ``text``
as length + sha256 + prefix, ``id`` / ``url`` / ``date`` as ``repr``), stderr / log messages per call, and the
exception that ends the run with what the next ``next()`` does.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.machinery
import os
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pyroid_parity.warc_compare import Event, Step, datatrove_open, exc_namer, flush

PAYLOAD_TYPES = ("text/html", "application/xhtml+xml")


class MagicMissing(RuntimeError):
    """Raised by the stub ``magic.from_buffer`` where libmagic is not installed."""


# Named as the stub module's, so that every implementation's trace names it the same (``magic.MagicMissing``).
MagicMissing.__module__ = "magic"


def ensure_magic() -> str | None:
    """libmagic's version if python-magic works here; otherwise install the stub and return None."""
    try:
        import magic
    except ImportError:
        stub = types.ModuleType("magic")
        stub.__spec__ = importlib.machinery.ModuleSpec("magic", None)

        def from_buffer(*_a: object, **_k: object) -> str:
            raise MagicMissing("libmagic is not installed: datatrove calls magic.from_buffer for this record")

        stub.from_buffer = from_buffer  # type: ignore[attr-defined]
        stub.MagicMissing = MagicMissing  # type: ignore[attr-defined]
        sys.modules["magic"] = stub
        return None
    if getattr(magic, "MagicMissing", None) is MagicMissing:
        return None
    return str(magic.version())


def text_view(t: Any) -> Any:
    if not isinstance(t, str):
        return ["not-str", repr(t)[:80]]
    b = t.encode("utf-8", "surrogatepass")
    return [len(t), hashlib.sha256(b).hexdigest()[:16], repr(t[:60])]


def doc_view(item: Any) -> list[Any]:
    if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict)):
        return ["not-(ri, dict)", repr(item)[:200]]
    ri, d = item
    rest = {k: repr(v) for k, v in d.items() if k != "text"}
    return [repr(ri), list(d), text_view(d.get("text")), rest]


def events_of(
    gen_fn: Callable[[], Iterator[Any]],
    view: Callable[[Any], list[Any]],
    exc_name: Callable[[BaseException], str],
    limit: int = 0,
    sink: Callable[[list[Event]], None] | None = None,
) -> list[Event]:
    """Run a generator to its end (or ``limit`` items), recording each item, message and the final exception.

    With ``sink``, events are handed over item by item (``warc_compare.iter_events``).
    """
    events: list[Event] = []
    step = Step(events, exc_name)
    ok, gen = step("call", lambda: iter(gen_fn()), lambda g: None)
    if not ok:
        return flush(events, sink)
    n = 0
    while True:
        ok, item = step("next", lambda: next(gen), lambda r: None)
        if not ok:
            if isinstance(item, StopIteration):
                events[step.last] = ["next", "stop"]
            else:
                step("next-after-raise", lambda: next(gen), lambda r: "item")
                if events[step.last][:3] == ["next-after-raise", "raise", "builtins.StopIteration"]:
                    events[step.last] = ["next-after-raise", "stop"]
            break
        events[step.last] = ["item", n, *view(item)]
        n += 1
        if sink is not None:
            sink(events)
            events.clear()
        if limit and n >= limit:
            break
    close = getattr(gen, "close", None)
    if close is not None:
        step("close", close)
    return flush(events, sink)


# ---------------------------------------------------------------------------------------- the reference


def reference_documents(source: Any, *, compression: str | None = "infer") -> Iterator[tuple[int, dict]]:
    """datatrove 0.10.1's ``WarcReader.read_file``, before ``get_document_from_dict``: (record index, dict).

    ``source`` is a path (opened as datatrove's DataFolder does) or a binary file object (``compression``
    "gzip": a ``gzip.GzipFile`` over it; None: read as it is; the object is not closed).
    """
    from datatrove.pipeline.readers.warc import process_record
    from warcio.archiveiterator import ArchiveIterator

    if isinstance(source, str | os.PathLike):
        with datatrove_open(Path(source), compression) as f:
            for ri, rec in enumerate(ArchiveIterator(f)):
                d = process_record(rec)
                if d:
                    yield ri, d
        return
    if compression not in ("gzip", None):
        raise ValueError(f"compression {compression!r} for a file object")
    f = gzip.GzipFile(fileobj=source) if compression == "gzip" else source
    for ri, rec in enumerate(ArchiveIterator(f)):
        d = process_record(rec)
        if d:
            yield ri, d


def via_process_record(mod: Any, path: Path) -> Iterator[tuple[int, dict]]:
    """datatrove's ``process_record`` on ``mod.ArchiveIterator``'s records over what WarcReader opens."""
    from datatrove.pipeline.readers.warc import process_record

    with datatrove_open(path) as f:
        for ri, rec in enumerate(mod.ArchiveIterator(f)):
            d = process_record(rec)
            if d:
                yield ri, d


def document_events(fn: Callable[[], Iterator[Any]], exc_name: Callable[[BaseException], str]) -> list[Event]:
    return events_of(fn, doc_view, exc_name)


# ------------------------------------------------------------------------------------------- batches


def reference_rows(path: Path) -> Iterator[list[Any]]:
    """What a row of ``iter_batches`` holds for each record datatrove reads the content of.

    Those are the response and conversion records whose ``WARC-Identified-Payload-Type`` is missing or one
    datatrove keeps. Columns: id, url, date (``process_record``'s, or the WARC headers' when it gives no
    document; None where its ``dict(headers)["uri"]`` / ``["archive-date"]`` fallback raises KeyError, as
    pyroid.warc documents), mime_type (the WARC payload type), content_type (the HTTP Content-Type), html (the
    content stream), text (``process_record``'s text, or None).
    """
    from datatrove.pipeline.readers.warc import process_record
    from warcio.archiveiterator import ArchiveIterator

    with datatrove_open(path) as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type not in ("response", "conversion"):
                continue
            mime = rec.rec_headers.get("WARC-Identified-Payload-Type", None)
            if mime is not None and not (
                mime in PAYLOAD_TYPES or (rec.rec_type == "conversion" and mime == "text/plain")
            ):
                continue
            html = rec.content_stream().read()
            ctype = rec.http_headers.get_header("Content-Type") if rec.http_headers else None
            proxy = _Replay(rec, html)
            d = process_record(proxy)
            yield [
                repr(d["id"] if d else rec.rec_headers.get("WARC-Record-ID")),
                repr(row_value(d["url"]) if d else fallback(rec.rec_headers, "WARC-Target-URI", "uri")),
                repr(row_value(d["date"]) if d else fallback(rec.rec_headers, "WARC-Date", "archive-date")),
                repr(mime),
                repr(ctype),
                bytes_view(html),
                None if d is None else text_view(d["text"]),
            ]


_MISSING = "\0pyroid-parity: datatrove raises KeyError here"


def fallback(h: Any, name: str, key: str) -> Any:
    """``process_record``'s url / date rule for a record it gives no document for: the header, else
    ``dict(headers)[key]``, else None (where datatrove would raise KeyError)."""
    return h.get(name, None) or dict(h.headers).get(key)


def row_value(v: Any) -> Any:
    """A row's url / date: None where datatrove's fallback raises KeyError."""
    return None if v == _MISSING else v


class _Headers:
    """A record's WARC headers as ``process_record`` reads them, with ``uri`` / ``archive-date`` for its
    ``dict(headers)[...]`` fallbacks when the record has none (``iter_batches`` documents nulls there)."""

    def __init__(self, h: Any) -> None:
        self._h = h
        have = dict(h.headers)
        self.headers = list(h.headers) + [(k, _MISSING) for k in ("uri", "archive-date") if k not in have]

    def get(self, name: str, default: Any = None) -> Any:
        return self._h.get(name, default)

    def __getitem__(self, name: str) -> Any:
        return self._h[name]


class _Replay:
    """A record whose content stream replays bytes already read (``process_record`` reads it once)."""

    def __init__(self, rec: Any, data: bytes) -> None:
        self.rec_type = rec.rec_type
        self.rec_headers = _Headers(rec.rec_headers)
        self.http_headers = rec.http_headers
        self._data = data

    def content_stream(self) -> Any:
        import io

        return io.BytesIO(self._data)


def bytes_view(b: Any) -> Any:
    if b is None:
        return None
    if not isinstance(b, bytes | bytearray | memoryview):
        return ["not-bytes", type(b).__name__]
    b = bytes(b)
    return [len(b), hashlib.sha256(b).hexdigest()[:16]]


BATCH_COLUMNS = ("id", "url", "date", "mime_type", "content_type", "html", "text")


def batch_events(
    mod: Any, path: Path, batch_size: int, exc_name: Callable[[BaseException], str]
) -> list[Event]:
    """``mod.iter_batches(path, batch_size=...)``: one event per batch (rows, schema), then one per row."""
    import pyarrow as pa

    def gen() -> Iterator[Any]:
        yield from mod.iter_batches(path, batch_size=batch_size)

    def view(b: Any) -> list[Any]:
        rb = pa.record_batch(b)
        rows = []
        cols = {
            name: rb.column(name).to_pylist() if name in rb.schema.names else None for name in BATCH_COLUMNS
        }
        for i in range(rb.num_rows):
            row = []
            for name in BATCH_COLUMNS:
                v = None if cols[name] is None else cols[name][i]
                if name == "html":
                    row.append(bytes_view(v))
                elif name == "text":
                    row.append(None if v is None else text_view(v))
                else:
                    row.append(repr(v))
            rows.append(row)
        return [rb.num_rows, [f"{f.name}:{f.type}" for f in rb.schema], rows]

    return events_of(gen, view, exc_name)


# ------------------------------------------------------------------------------------------- WarcReader


def reader_doc_view(doc: Any) -> list[Any]:
    """id, text, metadata (``file_path`` as its file name: runs in other processes use other temporary folders), media."""
    meta = dict(doc.metadata)
    if isinstance(meta.get("file_path"), str):
        meta["file_path"] = ".../" + meta["file_path"].rsplit("/", 1)[-1]
    return [repr(doc.id), text_view(doc.text), repr(meta), repr(doc.media)]


def reader_events(
    cls: type, folder: Path, exc_name: Callable[[BaseException], str], **kw: Any
) -> list[Event]:
    """``cls(folder, **kw).run()``: every Document (id, text, metadata, media) and the final exception."""

    def gen() -> Iterator[Any]:
        yield from cls(str(folder), **kw).run()

    return events_of(gen, reader_doc_view, exc_name)


def read_file_events(cls: type, path: Path, exc_name: Callable[[BaseException], str]) -> list[Event]:
    def gen() -> Iterator[Any]:
        yield from cls(str(path.parent)).read_file(path.name)

    return events_of(gen, reader_doc_view, exc_name)


def executor_output(cls: type, folder: Path, out: Path, glob: str | None = None) -> list[Any]:
    """A LocalPipelineExecutor run ``[cls(folder), <a JSON lines writer>]``: the written lines, or the exception.

    The writer is a small step here (datatrove's JsonlWriter needs orjson, which the warc group leaves out).
    """
    import json

    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.base import PipelineStep

    class Collect(PipelineStep):
        name = "collect"

        def run(self, data: Any, rank: int = 0, world_size: int = 1) -> Any:
            (out / "docs").mkdir(parents=True, exist_ok=True)
            with open(out / "docs" / f"{rank:05d}.jsonl", "w", encoding="utf-8") as f:
                for doc in data:
                    row = {"id": doc.id, "text": doc.text, "metadata": doc.metadata}
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    yield doc

    ex = LocalPipelineExecutor(
        pipeline=[cls(str(folder), glob_pattern=glob), Collect()],
        tasks=1,
        workers=1,
        logging_dir=str(out / "logs"),
        skip_completed=False,
    )
    try:
        ex.run()
    except Exception as e:  # noqa: BLE001  the outcome
        err: list[Any] = [type(e).__name__, str(e)]
    else:
        err = []
    lines = []
    for p in sorted((out / "docs").glob("*")):
        for line in p.read_text("utf-8").splitlines():
            d = json.loads(line)
            d.get("metadata", {}).pop("file_path", None)
            lines.append([d.get("id"), text_view(d.get("text")), json.dumps(d.get("metadata"))])
    return [lines, err]


def default_exc_name() -> Callable[[BaseException], str]:
    return exc_namer("datatrove")
