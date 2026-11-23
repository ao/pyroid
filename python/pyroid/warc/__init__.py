# Derived from warcio 1.8.2 (Apache-2.0), © Webrecorder, Ilya Kreymer et al.: archiveiterator.py;
# and datatrove 0.10.1 (Apache-2.0), © HuggingFace: pipeline/readers/warc.py (process_record);
# modified.
"""A drop-in for warcio 1.8.2's reader, and datatrove 0.10.1's WARC reading loop, in Rust.

See the stub (``__init__.pyi``) for the documentation.
"""

from __future__ import annotations

import contextlib
import functools
import gzip
import importlib
import logging
import os
import sys
from collections.abc import Iterator
from typing import Any, ClassVar, TypedDict

from pyroid import _versions
from pyroid._pyroid import warc as _native
from pyroid.warc._records import (
    ArchiveLoadFailed,
    ArcWarcRecord,
    DigestChecker,
    StatusAndHeaders,
    StatusAndHeadersParserException,
    _brotli_ok,
)

__all__ = [
    "BUFF_SIZE",
    "WARCIO_VERSION",
    "ARCIterator",
    "ArcWarcRecord",
    "ArchiveIterator",
    "ArchiveLoadFailed",
    "Batch",
    "ContentStream",
    "DigestChecker",
    "Document",
    "RawStream",
    "StatusAndHeaders",
    "StatusAndHeadersParserException",
    "WARCIterator",
    "iter_batches",
    "iter_documents",
]

#: The warcio release whose reader this module reproduces.
WARCIO_VERSION = "1.8.2"
#: warcio's default ``block_size``.
BUFF_SIZE: int = _native.BUFF_SIZE

_versions.check(__name__, "warcio", f"warcio=={WARCIO_VERSION}")
# iter_documents and iter_batches are datatrove 0.10.1's WarcReader loop (process_record).
_versions.check(__name__, "datatrove", "datatrove==0.10.1")

RawStream: Any = _native.RawStream
ContentStream: Any = _native.ContentStream
Batch: Any = _native.Batch

for _cls in (ArchiveLoadFailed, ArcWarcRecord, DigestChecker, StatusAndHeaders):
    _cls.__module__ = __name__
StatusAndHeadersParserException.__module__ = __name__


@functools.cache
def _facts() -> tuple[bool, str, bool, bool, int]:
    """The interpreter facts the core reproduces (see ``pyroid_core::warc::ReaderOptions``)."""
    import base64
    import binascii
    import unicodedata
    import zlib

    try:
        legacy_padding = base64.b64decode("AA==AA") == b"\x00"
    except binascii.Error:
        legacy_padding = False
    d = zlib.decompressobj(31)
    with contextlib.suppress(zlib.error):
        d.decompress(b"x" * 16)
    try:
        d.flush()
    except zlib.error:
        flush_raises = True
    else:
        flush_raises = False
    return (
        _brotli_ok(),
        unicodedata.unidata_version,
        legacy_padding,
        flush_raises,
        sys.version_info.minor,
    )


def _digest_code(check_digests: Any) -> int:
    if not check_digests:
        return 0
    if check_digests == "raise":
        return 3
    if check_digests == "log":
        return 2
    return 1


@functools.cache
def _b16_messages() -> tuple[str, str]:
    """Return ``base64.b16decode``'s errors for a bad digit and an odd length, as here."""
    import base64
    import binascii

    out = []
    for bad in ("zz", "a"):
        try:
            base64.b16decode(bad, casefold=True)
        except binascii.Error as e:
            out.append(str(e))
    return (out[0], out[1])


def _err_classes(cls: type[ArchiveIterator]) -> tuple[Any, Any, Any, tuple[str, str]]:
    return (cls.archive_load_failed, cls.parser_exception, cls.logger, _b16_messages())


def _classes(cls: type[ArchiveIterator], check_digests: Any) -> tuple[Any, ...]:
    rc, hc = cls.record_cls, cls.headers_cls
    # The private fast constructors only for the classes that define them (not subclasses).
    record = rc._from_native if "_from_native" in vars(rc) else rc
    headers = hc._from_native if "_from_native" in vars(hc) else hc
    return (
        record,
        record is not rc,
        headers,
        headers is not hc,
        cls.digest_checker_cls,
        check_digests,
        _err_classes(cls),
    )


def _is_path(source: object) -> bool:
    return isinstance(source, (str, os.PathLike))


class ArchiveIterator:
    """Iterate over the records of a WARC or ARC file (warcio 1.8.2's ``ArchiveIterator``).

    The file may be uncompressed or gzip-compressed with one member per record, as warcio
    reads them.

    Args:
        fileobj: A binary file object, or a path to open (read with the GIL released).
        no_record_parse: Do not parse the HTTP headers.
        verify_http: Raise :class:`StatusAndHeadersParserException` for an HTTP status line
            that does not start with a known protocol (or verb, for requests).
        arc2warc: Present ARC records as WARC records.
        ensure_http_headers: Give records without HTTP headers a ``200 OK`` set.
        block_size: The bytes asked for in each ``fh.read()``.
        check_digests: Check ``WARC-Block-Digest`` and ``WARC-Payload-Digest``: ``True``
            records the results in ``record.digest_checker``, ``"log"`` also writes problems
            to stderr, ``"raise"`` raises :class:`ArchiveLoadFailed` at the first.

    Raises:
        ArchiveLoadFailed: The file is not a WARC or ARC file, a record is malformed, or a
            gzip member holds several records.
        EOFError: A gzip member or the file ends inside a record's headers.

    Subclass to change the classes it creates (what ``pyroid.patch("warcio")`` does): the
    class attributes ``record_cls`` (called as warcio's ``ArcWarcRecord(format, rec_type,
    rec_headers, raw_stream, http_headers, content_type, length, payload_length=...,
    digest_checker=...)``), ``headers_cls`` (as ``StatusAndHeaders(statusline=..., headers=...,
    protocol=..., total_len=...)``), ``digest_checker_cls``, ``archive_load_failed``,
    ``parser_exception`` and ``logger`` (``warcio.recordloader``'s by default).
    """

    GZIP_ERR_MSG = """
    ERROR: non-chunked gzip file detected, gzip block continues
    beyond single record.

    This file is probably not a multi-member gzip but a single gzip file.

    To allow seek, a gzipped {1} must have each record compressed into
    a single gzip member and concatenated together.

    This file is likely still valid and can be fixed by running:

    warcio recompress <path/to/file> <path/to/new_file>

"""

    INC_RECORD = """\
    WARNING: Record not followed by newline, perhaps Content-Length is invalid
    Offset: {0}
    Remainder: {1}
"""

    record_cls: ClassVar[Any] = ArcWarcRecord
    headers_cls: ClassVar[Any] = StatusAndHeaders
    digest_checker_cls: ClassVar[Any] = DigestChecker
    archive_load_failed: ClassVar[Any] = ArchiveLoadFailed
    parser_exception: ClassVar[Any] = StatusAndHeadersParserException
    logger: ClassVar[logging.Logger] = logging.getLogger("warcio.recordloader")
    _known_format: ClassVar[str | None] = None

    def __init__(
        self,
        fileobj: Any,
        no_record_parse: bool = False,
        verify_http: bool = False,
        arc2warc: bool = False,
        ensure_http_headers: bool = False,
        block_size: int = BUFF_SIZE,
        check_digests: Any = False,
    ) -> None:
        cls = type(self)
        self.no_record_parse = no_record_parse
        self.ensure_http_headers = ensure_http_headers
        self.mixed_arc_warc = arc2warc
        self.check_digests = check_digests
        is_path = _is_path(fileobj)
        self._it = _native._Reader(
            os.fspath(fileobj) if is_path else fileobj,
            is_path,
            (bool(no_record_parse), bool(verify_http), bool(arc2warc), bool(ensure_http_headers)),
            block_size,
            _digest_code(check_digests),
            cls._known_format,
            _classes(cls, check_digests),
            _facts(),
            sys.get_int_max_str_digits(),
        )
        self.the_iter = self._it

    def __iter__(self) -> Iterator[Any]:
        it: Iterator[Any] = self._it
        return it

    def __next__(self) -> Any:
        return next(self._it)

    def close(self) -> None:
        """Drop the current record and flush the decompressor (the file stays open).

        Raises:
            zlib.error: The last gzip member was corrupt (CPython 3.13 and later).
        """
        self._it.close()

    def read_to_end(self, record: Any = None) -> None:
        """Read the rest of the current record and the blank lines after it."""
        self._it.read_to_end()

    def get_record_offset(self) -> int:
        """Return the offset of the current record in the file (reads it to its end)."""
        offset: int = self._it.get_record_offset()
        return offset

    def get_record_length(self) -> int:
        """Return the bytes of the current record in the file (reads it to its end).

        Compressed: its gzip member. Uncompressed: up to the blank lines after it.
        """
        length: int = self._it.get_record_length()
        return length

    @property
    def offset(self) -> int:
        """The offset where the next record starts, once the current one has been read."""
        offset: int = self._it.offset
        return offset

    @property
    def err_count(self) -> int:
        """Records so far not followed by a blank line (each also warned about on stderr)."""
        n: int = self._it.err_count
        return n

    @property
    def member_info(self) -> tuple[int, int] | None:
        """``(offset, length)`` of the current record once it has been read to its end."""
        info: tuple[int, int] | None = self._it.member_info
        return info

    @property
    def known_format(self) -> str | None:
        """The format of the records read so far (the first record decides)."""
        fmt: str | None = self._it.known_format
        return fmt

    @property
    def record(self) -> Any:
        """The current record (``None`` before the first and after the end)."""
        return self._it.record

    @property
    def fh(self) -> Any:
        """The file object read (``None`` for a path)."""
        return self._it.fh


class WARCIterator(ArchiveIterator):
    """:class:`ArchiveIterator` that reads WARC records only."""

    _known_format: ClassVar[str | None] = "warc"


class ARCIterator(ArchiveIterator):
    """:class:`ArchiveIterator` that reads ARC records only."""

    _known_format: ClassVar[str | None] = "arc"


# ---- datatrove's WarcReader.read_file loop ---------------------------------------------------


class Document(TypedDict):
    """A document from :func:`iter_documents` (datatrove's ``process_record`` dict)."""

    text: str
    id: str | None
    url: str
    date: str


#: fsspec's compressions by extension (``fsspec.utils.infer_compression``).
_COMPRESSIONS = {
    "gz": "gzip",
    "bz2": "bz2",
    "zip": "zip",
    "lzma": "lzma",
    "xz": "xz",
    "lz4": "lz4",
    "zst": "zstd",
    "sz": "snappy",
}
_MAX_DOCS = 64
_MAX_BYTES = 16 << 20


def _mode(source: Any, compression: Any) -> tuple[Any, bool, bool]:
    """``(source, is_path, gzip)`` for the native reader."""
    if compression not in ("infer", "gzip", None):
        raise ValueError(f"compression must be 'infer', 'gzip' or None, not {compression!r}")
    if not _is_path(source):
        return source, False, compression == "gzip"
    path = os.fspath(source)
    if compression != "infer":
        return path, True, compression == "gzip"
    name = path if isinstance(path, str) else os.fsdecode(path)
    inferred = _COMPRESSIONS.get(os.path.splitext(name)[-1].strip(".").lower())
    if inferred is not None and inferred != "gzip":
        raise ValueError(
            f"pyroid.warc reads gzip-compressed and uncompressed files; {name!r} has "
            f"compression {inferred!r} (inferred from its extension)"
        )
    return path, True, inferred == "gzip"


def _emit(msgs: list[tuple[bool, str]], logger: Any) -> None:
    for is_log, text in msgs:
        if is_log:
            logger.warning(text)
        else:
            sys.stderr.write(text)


def _mime_ok(mime_type: Any, conversion: bool) -> bool:
    return (
        mime_type == "text/html"
        or mime_type == "application/xhtml+xml"
        or (conversion and mime_type == "text/plain")
    )


def _process_record(record: Any) -> dict[str, Any] | None:
    """Run datatrove 0.10.1's ``process_record``."""
    cchardet = importlib.import_module("cchardet")
    magic = importlib.import_module("magic")

    # record type
    if record.rec_type != "response" and record.rec_type != "conversion":
        return None

    # content type filtering
    mime_type = record.rec_headers.get("WARC-Identified-Payload-Type", None)
    if mime_type is not None and (
        mime_type != "text/html"
        and mime_type != "application/xhtml+xml"
        and (record.rec_type != "conversion" or mime_type != "text/plain")
    ):
        return None

    content_bytes = record.content_stream().read()
    if mime_type is None:
        # fallback for older crawls without payload types
        mime_type = magic.from_buffer(content_bytes, mime=True)
        if (
            mime_type != "text/html"
            and mime_type != "application/xhtml+xml"
            and (record.rec_type != "conversion" or mime_type != "text/plain")
        ):
            return None

    # Decode the response bytes
    charset = "UTF-8"
    try:
        html = content_bytes.decode(charset)
    except UnicodeDecodeError:
        encoding_det = cchardet.detect(content_bytes)["encoding"]
        if not encoding_det or encoding_det == charset:
            return None
        charset = encoding_det

        try:
            html = content_bytes.decode(charset)
        except (UnicodeDecodeError, LookupError):
            return None

    id_ = record.rec_headers["WARC-Record-ID"]
    url = record.rec_headers.get("WARC-Target-URI", None)
    date = record.rec_headers.get("WARC-Date", None)
    # handle older formats
    if not url:
        url = dict(record.rec_headers.headers)["uri"]
    if not date:
        date = dict(record.rec_headers.headers)["archive-date"]

    return {"text": html, "id": id_, "url": url, "date": date}


def _record_loop(
    source: Any, is_path: bool, gz: bool, iterator_cls: type[ArchiveIterator]
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Run datatrove's loop itself, over CPython's ``gzip.GzipFile`` (when an import fails)."""
    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(source, "rb")) if is_path else source
        if gz:
            f = stack.enter_context(gzip.GzipFile(fileobj=f, mode="rb"))
        for ri, record in enumerate(iterator_cls(f)):
            extracted = _process_record(record)
            if not extracted:
                continue
            yield ri, extracted


def _finish(item: tuple[Any, ...], cchardet: Any, magic: Any) -> tuple[int, dict[str, Any]] | None:
    """The rest of ``process_record`` for a document the core could not finish."""
    ri, conversion, mime_type, content_bytes, _utf8, id_, url, date, key_error = item
    if mime_type is None:
        mime_type = magic.from_buffer(content_bytes, mime=True)
        if not _mime_ok(mime_type, conversion):
            return None
    charset = "UTF-8"
    try:
        html = content_bytes.decode(charset)
    except UnicodeDecodeError:
        encoding_det = cchardet.detect(content_bytes)["encoding"]
        if not encoding_det or encoding_det == charset:
            return None
        charset = encoding_det
        try:
            html = content_bytes.decode(charset)
        except (UnicodeDecodeError, LookupError):
            return None
    if key_error is not None:
        raise KeyError(key_error)
    return ri, {"text": html, "id": id_, "url": url, "date": date}


def _documents(
    source: Any, is_path: bool, gz: bool, iterator_cls: type[ArchiveIterator]
) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        cchardet = importlib.import_module("cchardet")
        magic = importlib.import_module("magic")
    except Exception:
        # datatrove imports both at its first record: its own loop raises at the same place.
        yield from _record_loop(source, is_path, gz, iterator_cls)
        return
    logger = iterator_cls.logger
    reader = _native._DocReader(
        source,
        is_path,
        gz,
        _err_classes(iterator_cls),
        _facts(),
        sys.get_int_max_str_digits(),
    )
    while (batch := reader.next_batch(_MAX_DOCS, _MAX_BYTES)) is not None:
        for item in batch:
            if type(item) is list:
                _emit(item, logger)
            elif len(item) == 2:
                yield item
            else:
                doc = _finish(item, cchardet, magic)
                if doc is not None:
                    yield doc


def iter_documents(
    source: Any,
    *,
    compression: str | None = "infer",
    iterator_cls: type[ArchiveIterator] = ArchiveIterator,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read the documents datatrove 0.10.1's ``WarcReader`` reads from a WARC file.

    This is ``for ri, record in enumerate(ArchiveIterator(f)): doc = process_record(record);
    if doc: yield ri, doc``, where ``f`` is what datatrove opens: ``response`` and
    ``conversion`` records whose ``WARC-Identified-Payload-Type`` is ``text/html`` or
    ``application/xhtml+xml`` (or ``text/plain`` for ``conversion``), or missing and then
    detected by ``magic.from_buffer``; the payload decoded as UTF-8, else as cchardet guesses
    (records that decode neither way are skipped). The same documents, the same exceptions at
    the same record (``KeyError`` for a record without a URL or date), and the same messages.

    ``ri`` is the record's index among all records of the file. ``cchardet``
    (faust-cchardet) and ``magic`` (python-magic) are imported at the first ``next()``, as
    datatrove imports them at its first record; if either import fails, datatrove's own loop
    runs on :class:`ArchiveIterator` over CPython's ``gzip.GzipFile``, so the ``ImportError``
    comes at the first record. The file is opened at the first ``next()``; up to 64 documents
    (16 MiB of payload) are read ahead, and messages and errors are delivered at the document
    they belong to.

    Args:
        source: A path, or a binary file object.
        compression: For a path, ``"infer"`` reads names ending in ``.gz`` (any case) as gzip
            and others as uncompressed, as fsspec does; ``"gzip"`` and ``None`` say which. A
            gzip file is read through an emulation of CPython's ``gzip.GzipFile`` (the bytes
            warcio sees, and the ``EOFError`` / ``gzip.BadGzipFile`` / ``zlib.error`` at the
            same point, as on this CPython version). A file object is read as is unless
            ``compression="gzip"``.
        iterator_cls: Where the ``archive_load_failed`` exception and the ``logger`` come
            from (and the iterator when an import fails), such as warcio's patched class.

    Raises:
        ValueError: ``compression`` is not ``"infer"``, ``"gzip"`` or ``None``, or the path's
            extension names another compression (``.zst``, ``.bz2``, ``.xz``, ...).
    """
    src, is_path, gz = _mode(source, compression)
    return _documents(src, is_path, gz, iterator_cls)


def _text(
    conversion: bool, mime_type: Any, content: bytes, utf8: bool, mods: dict[str, Any]
) -> str | bool | None:
    """A pending row's text: ``None`` (datatrove skips it), ``True`` (the content) or a str."""
    if mime_type is None:
        if "magic" not in mods:
            mods["magic"] = importlib.import_module("magic")
        if not _mime_ok(mods["magic"].from_buffer(content, mime=True), conversion):
            return None
    if utf8:
        return True
    if "cchardet" not in mods:
        mods["cchardet"] = importlib.import_module("cchardet")
    encoding_det = mods["cchardet"].detect(content)["encoding"]
    if not encoding_det or encoding_det == "UTF-8":
        return None
    try:
        text: str = content.decode(encoding_det)
    except (UnicodeDecodeError, LookupError):
        return None
    return text


def _batches(
    source: Any, is_path: bool, gz: bool, batch_size: int, iterator_cls: type[ArchiveIterator]
) -> Iterator[Any]:
    logger = iterator_cls.logger
    reader = _native._DocReader(
        source,
        is_path,
        gz,
        _err_classes(iterator_cls),
        _facts(),
        sys.get_int_max_str_digits(),
    )
    mods: dict[str, Any] = {}
    while (rows := reader.next_rows(batch_size)) is not None:
        _emit(rows.messages, logger)
        fills: dict[int, str | bool | None] = {}
        for row, conversion, mime_type, content, utf8 in rows.pending():
            try:
                fills[row] = _text(conversion, mime_type, content, utf8, mods)
            except BaseException:
                # magic or cchardet raised at this row: the rows before it, then the error.
                batch = rows.finish(fills, row)
                if batch.num_rows:
                    yield batch
                raise
        batch = rows.finish(fills)
        if batch.num_rows:
            yield batch


def iter_batches(
    source: Any,
    *,
    batch_size: int = 1000,
    compression: str | None = "infer",
    iterator_cls: type[ArchiveIterator] = ArchiveIterator,
) -> Iterator[Any]:
    """Read a WARC file's documents as Arrow record batches of up to ``batch_size`` rows.

    The rows are the records :func:`iter_documents` looks at: ``response`` and
    ``conversion`` records whose ``WARC-Identified-Payload-Type`` is one datatrove keeps or
    is missing. ``text`` is what datatrove's ``process_record`` decodes, and null where it
    skips the record (``magic.from_buffer`` finds another type, or the payload decodes neither
    as UTF-8 nor as cchardet guesses); ``url`` and ``date`` are null where datatrove raises
    ``KeyError``. ``magic`` and ``cchardet`` are imported only when a row needs them.
    ``source``, ``compression`` and ``iterator_cls`` are as for :func:`iter_documents`; the
    messages of a batch are emitted before it is yielded.

    Raises:
        ValueError: ``batch_size`` is not a positive int, or as :func:`iter_documents`.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int, not {batch_size!r}")
    src, is_path, gz = _mode(source, compression)
    return _batches(src, is_path, gz, batch_size, iterator_cls)
