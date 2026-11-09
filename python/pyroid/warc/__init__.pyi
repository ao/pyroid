# Derived from warcio 1.8.2 (Apache-2.0), © Webrecorder, Ilya Kreymer et al.; and datatrove 0.10.1
# (Apache-2.0), © HuggingFace; modified.
"""A drop-in for warcio 1.8.2's WARC/ARC reader, and datatrove 0.10.1's WARC loop, in Rust.

```python
import pyroid.warc as pw

with open("CC-MAIN-...warc.gz", "rb") as f:
    for record in pw.ArchiveIterator(f):          # as warcio.ArchiveIterator
        if record.rec_type == "response":
            html = record.content_stream().read()

for record in pw.ArchiveIterator("x.warc.gz"):    # a path: read with the GIL released
    ...

for ri, doc in pw.iter_documents("x.warc.gz"):    # datatrove's WarcReader.read_file loop
    doc["text"], doc["id"], doc["url"], doc["date"]

import pyarrow as pa
table = pa.Table.from_batches(pa.record_batch(b) for b in pw.iter_batches("x.warc.gz"))
```

The reader is warcio 1.8.2's, ported statement by statement: the same records, headers
(:class:`StatusAndHeaders`), payloads (``content_stream()``: chunked, gzip, deflate,
deflate_alt and, with brotlicffi >= 1.2.1.0 installed, br), offsets
(``get_record_offset``/``get_record_length``), digest checks (``check_digests``), exceptions
with the same messages (:class:`ArchiveLoadFailed`, ``EOFError``, ``zlib.error``,
``brotlicffi.error``, ...), the same ``sys.stderr`` writes and ``warcio.recordloader`` log
warnings, and the same ``read(n)`` and ``tell()`` calls on the file object. Gzip members are
inflated with zlib-rs; parsing and decoding run in Rust.

A file object is read with the GIL held, one ``fh.read(block_size)`` per block as warcio
does. A path (``str`` or ``os.PathLike``, a pyroid extension) is opened and read in Rust with
the GIL released. An iterator may be used from several threads (its calls run one at a time),
including on free-threaded Python; a file object whose ``read()`` calls back into its own
iterator gets a ``RuntimeError``.

Records are pyroid's ports of warcio's classes. ``pyroid.patch("warcio")`` makes warcio's own
``ArchiveIterator`` this one, with warcio's ``StatusAndHeaders``, ``ArchiveLoadFailed`` and an
``ArcWarcRecord`` subclass: :class:`ArchiveIterator` takes every class it creates, and its
logger, from class attributes (``record_cls``, ``headers_cls``, ``digest_checker_cls``,
``archive_load_failed``, ``parser_exception``, ``logger``).

Differences from warcio:

- ``iter(it)`` returns the native iterator (warcio returns a generator), and the arguments
  are read once: changing ``it.no_record_parse`` later has no effect, and ``known_format``
  is read-only. A ``block_size`` that is not an int raises ``TypeError`` when the iterator
  is made, not at the first read.
- ``check_digests`` knows md5, sha1, sha224, sha256, sha384 and sha512, under the names
  ``hashlib.new`` takes for them (``sha1``, ``SHA-1``, ``sha-256``, ...). Other algorithms
  (``sha3_256``, ``blake2b``, ...) report "unknown hash algorithm name" where hashlib would
  hash.
- ``content_stream()`` needs the record's ``raw_stream`` to be pyroid's.
- Brotli payloads: a damaged stream whose command runs past the end of a metablock that is
  not the last one decodes further than libbrotli (more output, or another error later),
  where brotlicffi raises ``brotlicffi.error``.
"""

from collections.abc import Callable, Iterator
from logging import Logger
from re import Pattern
from typing import Any, ClassVar, Final, Literal, TypedDict, final

from _typeshed import StrPath, SupportsRead

from pyroid._pyroid import Column

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
WARCIO_VERSION: Final = "1.8.2"
#: warcio's default ``block_size``: the bytes asked for in each ``fh.read()``.
BUFF_SIZE: Final[int]

class ArchiveLoadFailed(Exception):  # noqa: N818 - warcio's name
    """The archive could not be read (warcio's ``warcio.exceptions.ArchiveLoadFailed``).

    Raised for an unknown format ("Unknown archive format, first line: ..."), a WARC or ARC
    record that does not start as one, a gzip file that is not one member per record, and,
    with ``check_digests="raise"``, a digest mismatch.
    """

    msg: str
    def __init__(self, reason: object) -> None: ...

class StatusAndHeadersParserException(Exception):  # noqa: N818 - warcio's name
    """An HTTP status line that does not start with a known protocol (``verify_http=True``).

    Attributes:
        statusline: The line, as read.
    """

    statusline: Any
    def __init__(self, msg: str, statusline: Any) -> None: ...

class StatusAndHeaders:
    """A status line and an ordered list of ``(name, value)`` headers.

    warcio's ``StatusAndHeaders``: ``rec_headers`` (the WARC or ARC headers) and
    ``http_headers`` of a record. Lookups by name ignore case; ``headers[...]``,
    ``get``, ``in`` and item assignment act like a case-insensitive dict.

    Args:
        statusline: The status line after the protocol (``"200 OK"``; empty for WARC headers).
        headers: ``(name, value)`` pairs (or a mapping); bytes are decoded as latin-1.
        protocol: ``"WARC/1.0"``, ``"HTTP/1.1"``, ``"GET"``, ...
        total_len: The characters read for the block, line ends included.
        is_http_request: Split the protocol (the verb) off ``statusline``.
    """

    ENCODE_HEADER_RX: ClassVar[Pattern[str]]
    statusline: str
    headers: list[tuple[str, str]]
    protocol: str
    total_len: int
    headers_buff: bytes | None
    def __init__(
        self,
        statusline: str,
        headers: Any,
        protocol: str = "",
        total_len: int = 0,
        is_http_request: bool = False,
    ) -> None: ...
    def get_header(self, name: str, default_value: Any = None) -> Any:
        """Return the value of the first header named ``name`` (any case), or ``default_value``."""

    def add_header(self, name: str, value: str) -> None:
        """Append a header."""

    def replace_header(self, name: str, value: str) -> str | None:
        """Set the last header named ``name`` (any case) to ``value``, or append it.

        Returns:
            The old value, or ``None`` if the header was added.
        """

    def remove_header(self, name: str) -> bool:
        """Remove the last header named ``name`` (any case).

        Returns:
            Whether there was one.
        """

    def get_statuscode(self) -> str:
        """Return the status code: the status line up to the first space."""

    def validate_statusline(self, valid_statusline: str) -> bool:
        """Check that the status line starts with a positive integer; else replace it.

        Returns:
            Whether it did.
        """

    def add_range(self, start: int, part_len: int, total_len: int) -> StatusAndHeaders:
        """Make this a ``206 Partial Content`` response for ``part_len`` bytes at ``start``."""

    def compute_headers_buffer(self, header_filter: Callable[..., Any] | None = None) -> None:
        """Set ``headers_buff`` to :meth:`to_ascii_bytes`."""

    def __ne__(self, other: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __str__(self, exclude_list: Any = None) -> str: ...
    def __bool__(self) -> bool: ...
    def __nonzero__(self) -> bool: ...
    def to_str(self, filter_func: Callable[..., Any] | None = None) -> str:
        r"""Return the status line and headers, each line ending in ``\r\n``.

        Args:
            filter_func: Called with each ``(name, value)``; returns the pair to write, or a
                false value to drop it.
        """

    def to_bytes(
        self, filter_func: Callable[..., Any] | None = None, encoding: str = "utf-8"
    ) -> bytes:
        """Return :meth:`to_str` encoded, followed by the blank line."""

    def to_ascii_bytes(self, filter_func: Callable[..., Any] | None = None) -> bytes:
        """Return :meth:`to_bytes` in ASCII, %-encoding non-ASCII values first (RFC 8187)."""

    def percent_encode_non_ascii_headers(self, encoding: str = "UTF-8") -> None:
        """%-encode the header values that are not ASCII (RFC 8187, RFC 5987)."""

    def get(self, name: str, default_value: Any = None) -> Any:
        """Return the value of the first header named ``name`` (any case), or ``default_value``."""

    def __getitem__(self, name: str, default_value: Any = None) -> Any: ...
    def __setitem__(self, name: str, value: str) -> str | None: ...
    def __delitem__(self, name: str) -> bool: ...
    def __contains__(self, key: str) -> bool: ...

class DigestChecker:
    """The digest checks of one record (warcio's ``DigestChecker``).

    ``record.digest_checker`` is kept up to date as the record's stream is read.

    Args:
        kind: ``ArchiveIterator``'s ``check_digests``: ``"raise"`` raises
            :class:`ArchiveLoadFailed` at the first problem, ``"log"`` also writes each
            problem to ``sys.stderr``.
    """

    kind: Any
    _problem: list[str]
    _passed: bool | None
    def __init__(self, kind: Any = None) -> None: ...
    @property
    def passed(self) -> bool | None:
        """Whether every digest checked so far matched (``None``: none checked)."""

    @passed.setter
    def passed(self, value: bool | None) -> None: ...
    @property
    def problems(self) -> list[str]:
        """The problems found so far, such as ``"payload digest failed: sha1:..."``."""

    def problem(self, value: str, passed: bool = False) -> None:
        """Record a problem (and raise or log it, as ``kind`` says)."""

@final
class RawStream:
    """A record's bytes after its headers (``record.raw_stream``).

    warcio's ``LimitReader`` over the iterator's reader: it stops at the record's
    ``Content-Length``. Reading an old record reads the file where the iterator is now, as
    in warcio.
    """

    def read(self, length: int | None = None) -> bytes:
        """Read up to ``length`` bytes (the rest of the record when ``None``)."""

    def readline(self, length: int | None = None) -> bytes:
        r"""Read up to the next ``\n`` (at most ``length`` bytes)."""

    def tell(self) -> int:
        """Return the bytes of the record read so far."""

    def close(self) -> None:
        """Close the file object, then the iterator's decompressor (as warcio)."""

    @property
    def limit(self) -> int | None:
        """The bytes left in the record (``None`` when it has no ``Content-Length``)."""

@final
class ContentStream:
    """A record's payload, de-chunked and decompressed (``record.content_stream()``)."""

    def read(self, length: int | None = None) -> bytes:
        """Read up to ``length`` decoded bytes (all of them when ``None``)."""

    def readline(self, length: int | None = None) -> bytes:
        r"""Read up to the next ``\n`` (at most ``length`` bytes)."""

    def tell(self) -> int:
        """Return the decoded bytes produced so far."""

    def close(self) -> None:
        """Close the underlying stream and the file object, then the decompressor."""

class ArcWarcRecord:
    """A WARC or ARC record (warcio's ``ArcWarcRecord``).

    Attributes:
        format: ``"warc"`` or ``"arc"`` (``"warc"`` for ARC records read with ``arc2warc``).
        rec_type: ``WARC-Type`` (``"response"``, ``"request"``, ``"warcinfo"``, ...);
            ``"response"`` or ``"arc_header"`` for ARC records.
        rec_headers: The WARC (or ARC) headers.
        raw_stream: The bytes after the headers (the HTTP headers already read).
        http_headers: The HTTP headers of ``response``, ``request`` and ``revisit`` records
            with an ``http:`` or ``https:`` URI, else ``None``.
        content_type: The record's ``Content-Type``.
        length: ``Content-Length`` as an int (0 when it does not parse), ``None`` without one.
        payload_length: The bytes after the HTTP headers (-1 when they were not parsed).
        digest_checker: The record's :class:`DigestChecker`.
    """

    format: str
    rec_type: str | None
    rec_headers: StatusAndHeaders
    raw_stream: RawStream
    http_headers: StatusAndHeaders | None
    content_type: str | None
    length: int | None
    payload_length: int
    digest_checker: DigestChecker
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def content_stream(self) -> RawStream | ContentStream:
        """Return the payload, decoded per the HTTP headers.

        ``Transfer-Encoding: chunked`` is de-chunked, and a ``Content-Encoding`` of gzip,
        x-gzip, deflate, deflate_alt or (with brotlicffi >= 1.2.1.0) br decompressed. Returns
        ``raw_stream`` itself when there is nothing to decode, else a new decoder over it on
        each call, as warcio.
        """

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

    GZIP_ERR_MSG: ClassVar[str]
    INC_RECORD: ClassVar[str]
    record_cls: ClassVar[Any]
    headers_cls: ClassVar[Any]
    digest_checker_cls: ClassVar[Any]
    archive_load_failed: ClassVar[Any]
    parser_exception: ClassVar[Any]
    logger: ClassVar[Logger]
    _known_format: ClassVar[str | None]
    no_record_parse: bool
    ensure_http_headers: bool
    mixed_arc_warc: bool
    check_digests: Any
    the_iter: Iterator[ArcWarcRecord]
    def __init__(
        self,
        fileobj: SupportsRead[bytes] | StrPath,
        no_record_parse: bool = False,
        verify_http: bool = False,
        arc2warc: bool = False,
        ensure_http_headers: bool = False,
        block_size: int = 16384,
        check_digests: bool | Literal["log", "raise"] = False,
    ) -> None: ...
    def __iter__(self) -> Iterator[ArcWarcRecord]: ...
    def __next__(self) -> ArcWarcRecord: ...
    def close(self) -> None:
        """Drop the current record and flush the decompressor (the file stays open).

        Raises:
            zlib.error: The last gzip member was corrupt (CPython 3.13 and later).
        """

    def read_to_end(self, record: object = None) -> None:
        """Read the rest of the current record and the blank lines after it."""

    def get_record_offset(self) -> int:
        """Return the offset of the current record in the file (reads it to its end)."""

    def get_record_length(self) -> int:
        """Return the bytes of the current record in the file (reads it to its end).

        Compressed: its gzip member. Uncompressed: up to the blank lines after it.
        """

    @property
    def offset(self) -> int:
        """The offset where the next record starts, once the current one has been read."""

    @property
    def err_count(self) -> int:
        """Records so far not followed by a blank line (each also warned about on stderr)."""

    @property
    def member_info(self) -> tuple[int, int] | None:
        """``(offset, length)`` of the current record once it has been read to its end."""

    @property
    def known_format(self) -> Literal["warc", "arc"] | None:
        """The format of the records read so far (the first record decides)."""

    @property
    def record(self) -> ArcWarcRecord | None:
        """The current record (``None`` before the first and after the end)."""

    @property
    def fh(self) -> Any:
        """The file object read (``None`` for a path)."""

class WARCIterator(ArchiveIterator):
    """:class:`ArchiveIterator` that reads WARC records only."""

class ARCIterator(ArchiveIterator):
    """:class:`ArchiveIterator` that reads ARC records only."""

class Document(TypedDict):
    """A document from :func:`iter_documents` (datatrove's ``process_record`` dict)."""

    text: str
    id: str | None
    url: str
    date: str

def iter_documents(
    source: SupportsRead[bytes] | StrPath,
    *,
    compression: Literal["infer", "gzip"] | None = "infer",
    iterator_cls: type[ArchiveIterator] = ...,
) -> Iterator[tuple[int, Document]]:
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

@final
class Batch:
    """Documents of a WARC file as an Arrow record batch (no Arrow library needed).

    Columns: ``id`` (``WARC-Record-ID``), ``url``, ``date`` (``WARC-Target-URI`` and
    ``WARC-Date``, with datatrove's ARC fallbacks), ``mime_type``
    (``WARC-Identified-Payload-Type``), ``content_type`` (the HTTP ``Content-Type``), all
    ``large_string``; ``html`` (``large_binary``: the decoded payload); and ``text``
    (``large_string``: datatrove's decoding of it).

    ``pyarrow.record_batch(b)``, ``pyarrow.table(b)`` and ``polars.DataFrame(b)`` take it
    without a copy, through the Arrow PyCapsule interface.
    """

    @property
    def num_rows(self) -> int:
        """Rows in the batch."""

    @property
    def num_columns(self) -> int:
        """Columns in the batch."""

    @property
    def column_names(self) -> list[str]:
        """The column names, in order."""

    def column(self, key: str | int) -> Column[Any]:
        """Return a column by name or position, as a :class:`pyroid.Column` (no copy).

        Raises:
            KeyError: No column has that name.
            IndexError: The position is out of range.
        """

    def to_pydict(self) -> dict[str, list[Any]]:
        """Return ``{name: [values]}`` (``str``, ``bytes`` for ``html``, ``None`` for nulls)."""

    def __len__(self) -> int: ...
    def __arrow_c_schema__(self) -> object:
        """Export the schema, a struct of the columns (Arrow PyCapsule interface)."""

    def __arrow_c_array__(self, requested_schema: object | None = None) -> tuple[object, object]:
        """Export as one Arrow struct array (a record batch)."""

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
        """Export as an Arrow C stream of this one batch."""

def iter_batches(
    source: SupportsRead[bytes] | StrPath,
    *,
    batch_size: int = 1000,
    compression: Literal["infer", "gzip"] | None = "infer",
    iterator_cls: type[ArchiveIterator] = ...,
) -> Iterator[Batch]:
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
