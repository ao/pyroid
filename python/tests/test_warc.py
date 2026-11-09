"""pyroid.warc's ArchiveIterator against live warcio 1.8.2, on generated WARC and ARC files.

Each case runs the same script on both readers (iterate, read the records' streams in some
pattern, ask for offsets, digests and the iterator's attributes) and compares the traces:
records, bytes read, exceptions (type and message), the next ``next()`` after an exception,
and the ``sys.stderr`` writes and ``warcio.recordloader`` log warnings in order.

The generator (:func:`warc_record`, :func:`http_block`, :func:`arc_file`, :func:`gz_members`,
...) and :func:`trace` are reused by the other ``test_warc_*`` modules (imported by path).
"""

from __future__ import annotations

import base64
import contextlib
import gc
import gzip
import hashlib
import io
import logging
import re
import sys
import weakref
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid.warc as pw

CRLF = b"\r\n"

# ---- the generator ------------------------------------------------------------------------


def b32_digest(data: bytes, algo: str = "sha1") -> str:
    """``algo:BASE32`` as warcio's Digester formats it."""
    return algo + ":" + base64.b32encode(hashlib.new(algo, data).digest()).decode()


def encode_payload(body: bytes, encoding: str | None) -> bytes:
    """``body`` compressed for a ``Content-Encoding``."""
    if encoding in ("gzip", "x-gzip", "GZip"):
        return gzip.compress(body, mtime=0)
    if encoding == "deflate":
        return zlib.compress(body)
    if encoding == "deflate_alt":  # raw deflate, sent as "deflate" by some servers
        c = zlib.compressobj(wbits=-15)
        return c.compress(body) + c.flush()
    if encoding == "br":
        brotlicffi: Any = pytest.importorskip("brotlicffi")
        out: bytes = brotlicffi.compress(body)
        return out
    return body


def chunk(payload: bytes, size: int = 7, *, ext: bool = False) -> bytes:
    """HTTP chunked transfer encoding (``ext``: a chunk extension on each size line)."""
    out = b""
    for i in range(0, len(payload), max(size, 1)):
        piece = payload[i : i + max(size, 1)]
        out += f"{len(piece):x}".encode() + (b";x=1" if ext else b"") + CRLF + piece + CRLF
    return out + b"0" + CRLF + CRLF


def http_block(
    body: bytes,
    *,
    status: bytes = b"HTTP/1.1 200 OK",
    headers: Sequence[tuple[bytes, bytes]] = ((b"Content-Type", b"text/html; charset=utf-8"),),
    encoding: str | None = None,
    encoding_header: bytes | None = None,
    chunked: bool = False,
    chunk_size: int = 7,
) -> bytes:
    """An HTTP message (status line, headers, payload) for a WARC block."""
    payload = encode_payload(body, encoding)
    hdrs = list(headers)
    if encoding is not None:
        hdrs.append((b"Content-Encoding", encoding_header or encoding.encode()))
    if chunked:
        payload = chunk(payload, chunk_size)
        hdrs.append((b"Transfer-Encoding", b"chunked"))
    return status + CRLF + b"".join(n + b": " + v + CRLF for n, v in hdrs) + CRLF + payload


def warc_record(
    block: bytes,
    *,
    rec_type: str = "response",
    uri: str | None = "http://example.com/",
    version: bytes = b"WARC/1.0",
    record_id: str | None = "<urn:uuid:00000000-0000-4000-8000-000000000001>",
    date: str | None = "2024-01-02T03:04:05Z",
    length: bytes | None = None,
    payload_type: str | None = None,
    fields: Sequence[tuple[str, str]] = (),
    end: bytes = CRLF + CRLF,
) -> bytes:
    """A WARC record: ``version``, headers, ``block`` and ``end`` (normally ``\\r\\n\\r\\n``)."""
    hdrs: list[tuple[str, str]] = [("WARC-Type", rec_type)]
    if record_id is not None:
        hdrs.append(("WARC-Record-ID", record_id))
    if date is not None:
        hdrs.append(("WARC-Date", date))
    if uri is not None:
        hdrs.append(("WARC-Target-URI", uri))
    if payload_type is not None:
        hdrs.append(("WARC-Identified-Payload-Type", payload_type))
    hdrs.extend(fields)
    head = version + CRLF
    head += b"".join(n.encode() + b": " + v.encode("utf-8") + CRLF for n, v in hdrs)
    head += b"Content-Length: " + (str(len(block)).encode() if length is None else length) + CRLF
    return head + CRLF + block + end


def warcinfo(name: str = "test.warc.gz") -> bytes:
    body = b"software: pyroid tests\r\nformat: WARC File Format 1.0\r\n"
    return warc_record(
        body,
        rec_type="warcinfo",
        uri=None,
        fields=[("WARC-Filename", name), ("Content-Type", "application/warc-fields")],
    )


def response(
    body: bytes = b"<html><body>hello</body></html>",
    *,
    uri: str = "http://example.com/",
    payload_type: str | None = "text/html",
    digests: bool = False,
    **http: Any,
) -> bytes:
    block = http_block(body, **http)
    fields: list[tuple[str, str]] = [("Content-Type", "application/http; msgtype=response")]
    if digests:
        payload = block.split(CRLF + CRLF, 1)[1]
        fields += [("WARC-Block-Digest", b32_digest(block))]
        fields += [("WARC-Payload-Digest", b32_digest(payload))]
    return warc_record(block, uri=uri, payload_type=payload_type, fields=fields)


def request(uri: str = "http://example.com/") -> bytes:
    block = b"GET / HTTP/1.1\r\nHost: example.com\r\nUser-Agent: test\r\n\r\n"
    return warc_record(
        block,
        rec_type="request",
        uri=uri,
        fields=[("Content-Type", "application/http; msgtype=request")],
    )


def gz_members(records: Sequence[bytes]) -> bytes:
    """One gzip member per record (as Common Crawl writes them)."""
    return b"".join(gzip.compress(r, mtime=0) for r in records)


def arc_file(records: Sequence[tuple[str, bytes]], *, version_block: bool = True) -> bytes:
    """An ARC 1.0 file: the ``filedesc://`` header record, then ``(uri, block)`` records."""
    hb = b"1 0 pyroid\nURL IP-address Archive-date Content-type Archive-length\n"
    out = b""
    if version_block:
        out += f"filedesc://test.arc 0.0.0.0 20240102030405 text/plain {len(hb)}\n".encode()
        out += hb + b"\n"
    for uri, block in records:
        out += f"{uri} 192.0.2.1 20240102030405 text/html {len(block)}\n".encode() + block + b"\n"
    return out


def sample_warc() -> bytes:
    """A small, varied, valid WARC file (uncompressed)."""
    return b"".join(
        [
            warcinfo(),
            request(),
            response(digests=True),
            response(b"\xe4\xf6\xfc latin-1 page", uri="http://example.com/latin"),
            response(b"<p>chunked " * 20 + b"</p>", chunked=True, uri="http://example.com/c"),
            response(b"<p>gz</p>" * 50, encoding="gzip", uri="http://example.com/gz"),
            response(b"<p>df</p>" * 50, encoding="deflate", uri="http://example.com/df"),
            warc_record(b"plain text", rec_type="resource", uri="urn:x"),
            warc_record(b"", rec_type="revisit", uri="http://example.com/"),
            warc_record(
                b"text body",
                rec_type="conversion",
                uri="http://example.com/conv",
                payload_type="text/plain",
            ),
        ]
    )


# ---- tracing a reader ---------------------------------------------------------------------


class _Unseekable(io.RawIOBase):
    """A binary stream with ``read`` only (``tell`` raises): warcio's UnseekableYetTellable."""

    def __init__(self, data: bytes) -> None:
        self._b = io.BytesIO(data)

    def readable(self) -> bool:
        return True

    def read(self, n: int | None = -1) -> bytes:
        return self._b.read(n)

    def tell(self) -> int:
        raise OSError("not tellable")


def make_source(data: bytes, kind: str, tmp: Path | None = None) -> Any:
    """The file object (or path) a trace reads: bytesio, file, unseekable, path, gzipfile."""
    if kind == "bytesio":
        return io.BytesIO(data)
    if kind == "unseekable":
        return _Unseekable(data)
    if kind == "gzipfile":
        return gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb")
    assert tmp is not None
    p = tmp / "input.warc"
    p.write_bytes(data)
    if kind == "path":
        return str(p)
    return p.open("rb")


_UUID = re.compile(r"<urn:uuid:[0-9a-f-]{36}>")


def _norm(v: Any) -> Any:
    """Mask arc2warc's random record ids."""
    if isinstance(v, str):
        return _UUID.sub("<urn:uuid:*>", v)
    if isinstance(v, (list, tuple)):
        return type(v)(_norm(x) for x in v)
    return v


def headers_summary(h: Any) -> Any:
    if h is None:
        return None
    return (h.protocol, h.statusline, _norm(h.headers), h.total_len, bool(h))


def record_summary(rec: Any) -> tuple[Any, ...]:
    return (
        rec.format,
        rec.rec_type,
        headers_summary(rec.rec_headers),
        headers_summary(rec.http_headers),
        rec.content_type,
        rec.length,
        rec.payload_length,
    )


def exc_summary(e: BaseException) -> tuple[Any, ...]:
    return (type(e).__name__, str(e), getattr(e, "statusline", None))


@contextlib.contextmanager
def capture(events: list[tuple[Any, ...]]) -> Iterator[None]:
    """Put ``sys.stderr`` writes and ``warcio.recordloader`` warnings into ``events``."""

    class Err:
        def write(self, s: str) -> int:
            events.append(("stderr", s))
            return len(s)

        def flush(self) -> None:
            pass

    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            events.append(("log", record.name, record.levelname, record.getMessage()))

    logger = logging.getLogger("warcio.recordloader")
    handler = Handler()
    logger.addHandler(handler)
    old = sys.stderr
    sys.stderr = Err()
    try:
        yield
    finally:
        sys.stderr = old
        logger.removeHandler(handler)


#: What the trace does with each record (cycled).
PATTERNS = (
    "skip",
    "content",
    "content_chunks",
    "raw",
    "raw_lines",
    "offsets",
    "old_raw",
    "digest",
    "tell",
    "content_twice",
)


def _apply(pattern: str, it: Any, rec: Any, prev: Any, events: list[tuple[Any, ...]]) -> None:
    if pattern == "content":
        events.append(("read", rec.content_stream().read()))
    elif pattern == "content_chunks":
        s = rec.content_stream()
        events.append(("readline", s.readline()))
        while True:
            b = s.read(5)
            events.append(("read5", b))
            if not b:
                break
    elif pattern == "raw":
        events.append(("raw", rec.raw_stream.read()))
    elif pattern == "raw_lines":
        events.append(("rawline", rec.raw_stream.readline(), rec.raw_stream.readline(3)))
        events.append(("rawtell", rec.raw_stream.tell()))
    elif pattern == "offsets":
        events.append(("offset", it.get_record_offset(), it.get_record_length(), it.member_info))
    elif pattern == "old_raw" and prev is not None:
        events.append(("old", prev.raw_stream.read(10), prev.content_stream().read(10)))
    elif pattern == "digest":
        rec.content_stream().read()
        it.read_to_end()
        dc = rec.digest_checker
        events.append(("digest", dc.passed, list(dc.problems)))
    elif pattern == "tell":
        s = rec.content_stream()
        s.read(3)
        events.append(("tell", s.tell(), s is rec.raw_stream))
    elif pattern == "content_twice":
        a = rec.content_stream().read(4)
        b = rec.content_stream().read()
        events.append(("twice", a, b))


def trace(
    mod: Any,
    data: bytes,
    *,
    opts: dict[str, Any] | None = None,
    patterns: Sequence[str] = ("skip",),
    source: str = "bytesio",
    tmp: Path | None = None,
    messages: bool = True,
    cls: str = "ArchiveIterator",
) -> list[tuple[Any, ...]]:
    """What ``mod.<cls>`` does on ``data``: records, reads, exceptions and messages, in order."""
    events: list[tuple[Any, ...]] = []
    f = make_source(data, source, tmp)
    ctx = capture(events) if messages else contextlib.nullcontext()
    with ctx:
        it: Any = None
        try:
            it = getattr(mod, cls)(f, **(opts or {}))
            prev = None
            for i, rec in enumerate(it):
                events.append(("rec", record_summary(rec)))
                _apply(patterns[i % len(patterns)], it, rec, prev, events)
                events.append(("it", it.err_count, it.known_format))
                prev = rec
            events.append(("end", it.record is None))
        except Exception as e:
            events.append(("exc", *exc_summary(e)))
            if it is not None:
                try:
                    nxt = next(it)
                    events.append(("after", "record", record_summary(nxt)))
                except StopIteration:
                    events.append(("after", "stop"))
                except Exception as e2:
                    events.append(("after", *exc_summary(e2)))
        if it is not None:
            try:
                it.close()
                events.append(("closed",))
            except Exception as e:
                events.append(("close", *exc_summary(e)))
    if hasattr(f, "close"):
        f.close()
    return events


def compare(
    data: bytes,
    *,
    opts: dict[str, Any] | None = None,
    patterns: Sequence[str] = ("skip",),
    source: str = "bytesio",
    tmp: Path | None = None,
    messages: bool = True,
    cls: str = "ArchiveIterator",
) -> list[tuple[Any, ...]]:
    """Assert pyroid.warc and warcio trace ``data`` the same; return the trace."""
    wa: Any = pytest.importorskip("warcio.archiveiterator")
    ref_source = "file" if source == "path" else source
    want = trace(
        wa,
        data,
        opts=opts,
        patterns=patterns,
        source=ref_source,
        tmp=tmp,
        messages=messages,
        cls=cls,
    )
    got = trace(
        pw, data, opts=opts, patterns=patterns, source=source, tmp=tmp, messages=messages, cls=cls
    )
    assert got == want
    return got


# ---- explicit cases -----------------------------------------------------------------------

CASES: dict[str, bytes] = {
    "plain": sample_warc(),
    "gzip": gz_members(
        [warcinfo(), request(), response(digests=True), response(b"x" * 70000, chunked=True)]
    ),
    "empty": b"",
    "blank_lines": b"\r\n\r\n" + warcinfo(),
    "garbage": b"this is not a warc file\n",
    "gzip_garbage": gzip.compress(b"not a warc\n"),
    "gzip_one_member": gzip.compress(warcinfo() + request() + response(), mtime=0),
    "gzip_then_garbage": gz_members([warcinfo(), response()]) + b"trailing junk",
    "gzip_truncated": gz_members([warcinfo(), response(b"y" * 5000)])[:-30],
    "gzip_corrupt": (lambda d: d[:200] + bytes([d[200] ^ 0x55]) + d[201:])(
        gz_members([warcinfo(), response(b"z" * 5000), response()])
    ),
    "truncated_block": sample_warc()[:-500],
    "truncated_headers": warcinfo() + b"WARC/1.0\r\nWARC-Type: resp",
    "bad_length": warc_record(b"abcdef", length=b"3")
    + warc_record(b"xyz", rec_type="resource", uri="urn:y"),
    "long_length": warc_record(b"abc", length=b"500"),
    "garbage_length": warc_record(b"abc", length=b"twelve") + warcinfo(),
    "unicode_length": warc_record(b"abc", length="٣".encode()) + warcinfo(),
    "negative_length": warc_record(b"abc", length=b"-3") + warcinfo(),
    "huge_length": warc_record(b"abc", length=b"99999999999999999999999"),
    "no_length": b"WARC/1.0\r\nWARC-Type: resource\r\nWARC-Target-URI: urn:z\r\n\r\nbody\r\n\r\n",
    "spaces_uri": response(uri="http://example.com/a b"),
    "angle_uri": response(uri="<http://example.com/angle>"),
    "no_uri_response": warc_record(http_block(b"hi"), uri=None),
    "dns_uri": warc_record(b"1.2.3.4", uri="dns:example.com"),
    "warc_1_1": warc_record(b"x", rec_type="resource", version=b"WARC/1.1", uri="urn:a"),
    "warc_bad_version": warc_record(b"x", version=b"WARC/9.9"),
    "lowercase_headers": b"WARC/1.0\r\nwarc-type: resource\r\ncontent-length: 2\r\n\r\nab\r\n\r\n",
    "continuation": b"WARC/1.0\r\nWARC-Type: resource\r\nX-Long: a\r\n  b\r\n\tc\r\n"
    b"Content-Length: 2\r\n\r\nab\r\n\r\n",
    "latin1_header": b"WARC/1.0\r\nWARC-Type: resource\r\nX-Name: caf\xe9\r\n"
    b"Content-Length: 2\r\n\r\nab\r\n\r\n",
    "http_bad_status": warc_record(http_block(b"x", status=b"FOO 200 OK")),
    "http_no_headers": warc_record(b"\r\n\r\n", uri="http://example.com/"),
    "http_10": warc_record(http_block(b"x", status=b"HTTP/1.0 404 Not Found")),
    "revisit_empty": warc_record(b"", rec_type="revisit"),
    "revisit_headers": warc_record(
        http_block(b"", status=b"HTTP/1.1 304 Not Modified"), rec_type="revisit"
    ),
    "chunked_ext": response(b"hello world" * 9, chunked=True, chunk_size=5),
    "chunked_bad": warc_record(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nnot chunked at all"
    ),
    "chunked_truncated": warc_record(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n10\r\nshort"
    ),
    "chunked_uppercase": warc_record(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: Chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"
    ),
    "x_gzip": response(b"<p>xg</p>" * 30, encoding="x-gzip"),
    "gzip_header_case": response(b"<p>GZ</p>" * 30, encoding="GZip"),
    "deflate_alt": response(
        b"<p>raw deflate</p>" * 30, encoding="deflate_alt", encoding_header=b"deflate"
    ),
    "bad_gzip_payload": warc_record(
        b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n\r\n" + b"\x1f\x8b\x08\x00" + b"\xff" * 40
    ),
    "gzip_payload_corrupt_late": warc_record(
        b"HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\n\r\n"
        + (lambda g: g[:30] + b"\x00" * 10 + g[40:])(gzip.compress(b"q" * 90000))
    ),
    "unknown_encoding": response(b"compressed?", encoding="compress"),
    "chunked_gzip": response(b"<p>both</p>" * 40, encoding="gzip", chunked=True),
    "digests": b"".join(
        [
            response(digests=True),
            warc_record(
                http_block(b"bad"),
                fields=[("WARC-Payload-Digest", "sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")],
            ),
            warc_record(
                b"blk",
                rec_type="resource",
                uri="urn:b",
                fields=[("WARC-Block-Digest", "nosuchalgo:abc")],
            ),
            warc_record(
                b"blk",
                rec_type="resource",
                uri="urn:b",
                fields=[("WARC-Block-Digest", "sha1:" + hashlib.sha1(b"blk").hexdigest())],
            ),
            warc_record(
                b"blk",
                rec_type="resource",
                uri="urn:b",
                fields=[
                    (
                        "WARC-Block-Digest",
                        "sha256:" + base64.b64encode(hashlib.sha256(b"blk").digest()).decode(),
                    )
                ],
            ),
            warc_record(
                b"blk",
                rec_type="resource",
                uri="urn:b",
                fields=[("WARC-Block-Digest", "sha1:!!notbase16!!notbase16!!notbase16!!!!")],
            ),
            warc_record(
                http_block(b"seg"),
                fields=[("WARC-Segment-Number", "1"), ("WARC-Payload-Digest", "sha1:XYZ")],
            ),
        ]
    ),
    "digest_malformed": warc_record(
        b"blk", rec_type="resource", uri="urn:b", fields=[("WARC-Block-Digest", "nocolon")]
    ),
    "arc": arc_file(
        [
            ("http://example.com/", http_block(b"<html>arc</html>")),
            ("http://example.com/2", b"HTTP/1.0 200 OK\r\n\r\nbody"),
        ]
    ),
    "arc_no_filedesc": arc_file([("http://example.com/", b"raw")], version_block=False),
    "arc_bad": b"http://example.com/ 1.2.3.4 20240102030405 text/html\nbody\n",
    "arc_gz": gz_members([arc_file([("http://example.com/", http_block(b"<html>arc gz</html>"))])]),
    "mixed": warcinfo() + arc_file([("http://example.com/", b"raw")], version_block=False),
}


@pytest.mark.thread_unsafe  # sys.stderr and the warcio.recordloader logger
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_cases_match_warcio(name: str, pattern: str) -> None:
    compare(CASES[name], patterns=[pattern, "skip", "content"])


OPTION_SETS: list[dict[str, Any]] = [
    {"no_record_parse": True},
    {"verify_http": True},
    {"arc2warc": True},
    {"ensure_http_headers": True},
    {"block_size": 7},
    {"block_size": 1},
    {"block_size": 0},
    {"block_size": -1},
    {"block_size": 10**30},
    {"check_digests": True},
    {"check_digests": "log"},
    {"check_digests": "raise"},
    {"arc2warc": True, "ensure_http_headers": True, "block_size": 100},
]


@pytest.mark.thread_unsafe
@pytest.mark.parametrize(
    "opts", OPTION_SETS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items())
)
@pytest.mark.parametrize("name", sorted(CASES))
def test_options_match_warcio(name: str, opts: dict[str, Any]) -> None:
    compare(CASES[name], opts=opts, patterns=["digest", "content_chunks", "offsets", "raw"])


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("cls", ["WARCIterator", "ARCIterator"])
@pytest.mark.parametrize("name", ["plain", "gzip", "arc", "arc_gz", "mixed", "garbage"])
def test_format_iterators_match_warcio(name: str, cls: str) -> None:
    compare(CASES[name], cls=cls, patterns=["content", "offsets"])


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("source", ["file", "unseekable", "path"])
@pytest.mark.parametrize("name", ["plain", "gzip", "gzip_truncated", "bad_length", "arc"])
def test_sources_match_warcio(name: str, source: str, tmp_path: Path) -> None:
    compare(CASES[name], source=source, tmp=tmp_path, patterns=["offsets", "content", "old_raw"])


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("name", ["plain", "truncated_block", "bad_length", "chunked_bad"])
def test_over_gzip_gzipfile_matches_warcio(name: str) -> None:
    """datatrove's case: warcio reading a ``gzip.GzipFile`` (decompressed bytes)."""
    compare(gzip.compress(CASES[name]), source="gzipfile", patterns=["content", "offsets"])
    # A truncated gzip stream: GzipFile raises EOFError inside a record.
    compare(gzip.compress(CASES[name])[:-40], source="gzipfile", patterns=["content", "offsets"])


def test_brotli_payload_matches_warcio() -> None:
    pytest.importorskip("brotlicffi")
    data = response(b"<p>brotli</p>" * 100, encoding="br") + response(
        b"<p>chunked brotli</p>" * 50, encoding="br", chunked=True
    )
    compare(data, patterns=["content", "content_chunks"], messages=False)
    broken = warc_record(b"HTTP/1.1 200 OK\r\nContent-Encoding: br\r\n\r\n" + b"\xff\xfe" * 30)
    compare(broken, patterns=["content"], messages=False)


# ---- Hypothesis: random files, mutations, options and read patterns -------------------------

_bodies = st.one_of(
    st.binary(max_size=300),
    st.text(max_size=200).map(lambda s: s.encode("utf-8")),
    st.sampled_from([b"", b"<html>\xe9t\xe9</html>", b"\r\n\r\n", b"0\r\n\r\n", b"x" * 20000]),
)


@st.composite
def _records(draw: st.DrawFn) -> bytes:
    kind = draw(st.sampled_from(["response", "request", "warcinfo", "resource", "revisit", "conv"]))
    body = draw(_bodies)
    if kind == "response":
        encoding = draw(st.sampled_from([None, None, "gzip", "deflate", "deflate_alt", "compress"]))
        block = http_block(
            body,
            encoding=encoding,
            chunked=draw(st.booleans()),
            chunk_size=draw(st.integers(1, 40)),
            status=draw(st.sampled_from([b"HTTP/1.1 200 OK", b"HTTP/1.0 302", b"BAD"])),
        )
        uri = draw(
            st.sampled_from(["http://e.com/", "https://e.com/ x", "<http://e.com/>", "ftp:x"])
        )
    elif kind == "request":
        block, uri = b"GET / HTTP/1.1\r\nHost: e.com\r\n\r\n", "http://e.com/"
    elif kind == "revisit":
        block = draw(st.sampled_from([b"", http_block(b"", status=b"HTTP/1.1 304 NM")]))
        uri = "http://e.com/"
    else:
        block, uri = body, "urn:x"
    length = draw(
        st.sampled_from([None, None, None, b"0", b"7", b" 12 ", b"abc", b"+5", b"1_0", b"-2"])
    )
    fields: list[tuple[str, str]] = []
    digest = draw(st.sampled_from([None, "good", "bad", "unknown", "b16"]))
    if digest == "good":
        fields.append(("WARC-Block-Digest", b32_digest(block)))
    elif digest == "bad":
        fields.append(("WARC-Block-Digest", "sha1:" + "A" * 32))
    elif digest == "unknown":
        fields.append(("WARC-Payload-Digest", "md2:AAAA"))
    elif digest == "b16":
        fields.append(("WARC-Block-Digest", "sha1:" + hashlib.sha1(block).hexdigest()))
    end = draw(st.sampled_from([CRLF + CRLF, CRLF + CRLF, CRLF, b"", b"\n\n", b"junk\r\n"]))
    return warc_record(
        block,
        rec_type="conversion" if kind == "conv" else kind,
        uri=uri if kind != "warcinfo" else None,
        length=length,
        fields=fields,
        end=end,
    )


@st.composite
def _files(draw: st.DrawFn) -> bytes:
    records = draw(st.lists(_records(), min_size=1, max_size=5))
    layout = draw(st.sampled_from(["plain", "gzip", "gzip", "single_gzip", "arc"]))
    if layout == "arc":
        data = arc_file([(f"http://e.com/{i}", r) for i, r in enumerate(records)])
    elif layout == "gzip":
        data = gz_members(records)
    elif layout == "single_gzip":
        data = gzip.compress(b"".join(records), mtime=0)
    else:
        data = b"".join(records)
    mutation = draw(st.sampled_from(["none", "none", "truncate", "flip", "append"]))
    if mutation == "truncate" and data:
        data = data[: draw(st.integers(0, len(data) - 1))]
    elif mutation == "flip" and data:
        i = draw(st.integers(0, len(data) - 1))
        data = data[:i] + bytes([data[i] ^ draw(st.integers(1, 255))]) + data[i + 1 :]
    elif mutation == "append":
        data += draw(st.binary(min_size=1, max_size=30))
    return data


_options = st.fixed_dictionaries(
    {},
    optional={
        "no_record_parse": st.booleans(),
        "verify_http": st.booleans(),
        "arc2warc": st.booleans(),
        "ensure_http_headers": st.booleans(),
        "block_size": st.sampled_from([16384, 64, 5, 1, -1]),
        "check_digests": st.sampled_from([False, True, "log", "raise"]),
    },
)


@pytest.mark.thread_unsafe
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(data=_files(), opts=_options, patterns=st.lists(st.sampled_from(PATTERNS), min_size=1))
def test_random_files_match_warcio(data: bytes, opts: dict[str, Any], patterns: list[str]) -> None:
    compare(data, opts=opts, patterns=patterns)


# ---- the record API -----------------------------------------------------------------------


def test_record_classes_and_attributes() -> None:
    it = pw.ArchiveIterator(io.BytesIO(CASES["plain"]))
    assert iter(it) is it.the_iter
    first = next(it)
    assert isinstance(first, pw.ArcWarcRecord)
    assert isinstance(first.rec_headers, pw.StatusAndHeaders)
    assert isinstance(first.raw_stream, pw.RawStream)
    assert isinstance(first.digest_checker, pw.DigestChecker)
    assert it.record is first
    assert first.rec_headers["warc-type"] == "warcinfo"
    assert "WARC-Filename" in first.rec_headers
    assert first.content_stream() is first.raw_stream  # nothing to decode
    rec = next(it)
    assert rec.rec_type == "request"
    assert rec.http_headers is not None
    assert rec.http_headers.protocol == "GET"
    resp = next(it)
    assert resp.content_stream() is resp.raw_stream  # identity encoding, not chunked
    assert resp.content_stream().read() == b"<html><body>hello</body></html>"
    assert len(list(it)) == 7
    assert it.record is None
    assert it.known_format == "warc"
    assert it.fh is not None


def test_content_stream_decoders() -> None:
    decoded = []
    for rec in pw.ArchiveIterator(io.BytesIO(CASES["plain"])):
        uri = rec.rec_headers.get("WARC-Target-URI")
        if uri in ("http://example.com/c", "http://example.com/gz", "http://example.com/df"):
            s = rec.content_stream()
            assert isinstance(s, pw.ContentStream)
            assert s is not rec.content_stream()  # a new decoder each call
            decoded.append(s.read())
    assert [d[:3] for d in decoded] == [b"<p>"] * 3


def test_path_source_releases_and_reads(tmp_path: Path) -> None:
    p = tmp_path / "a.warc.gz"
    p.write_bytes(CASES["gzip"])
    with p.open("rb") as f:
        want = [record_summary(r) for r in pw.ArchiveIterator(f)]
    it = pw.ArchiveIterator(p)
    assert it.fh is None
    assert [record_summary(r) for r in it] == want
    assert [record_summary(r) for r in pw.ArchiveIterator(str(p))] == want


def test_missing_path_raises_like_open(tmp_path: Path) -> None:
    missing = tmp_path / "nope.warc"
    with pytest.raises(FileNotFoundError) as exc:
        pw.ArchiveIterator(missing)
    with pytest.raises(FileNotFoundError) as ref:
        open(missing, "rb")  # noqa: SIM115
    assert (exc.value.errno, exc.value.strerror, exc.value.filename) == (
        ref.value.errno,
        ref.value.strerror,
        str(missing),
    )


def test_block_size_type_is_checked_early() -> None:
    with pytest.raises(TypeError, match="'float' object cannot be interpreted as an integer"):
        pw.ArchiveIterator(io.BytesIO(b""), block_size=1.5)  # type: ignore[arg-type]


@pytest.mark.thread_unsafe
def test_source_exceptions_pass_through() -> None:
    class BoomError(Exception):
        pass

    class Failing(io.RawIOBase):
        def __init__(self) -> None:
            self.reads = 0

        def readable(self) -> bool:
            return True

        def read(self, n: int | None = -1) -> bytes:
            self.reads += 1
            if self.reads == 3:
                raise BoomError("third read")
            return CASES["plain"][(self.reads - 1) * 100 : self.reads * 100]

    it = pw.ArchiveIterator(Failing(), block_size=100)
    with pytest.raises(BoomError, match="third read"):
        list(it)
    assert next(it, None) is None  # a generator that raised is exhausted


def test_reentrant_file_object_raises_runtime_error() -> None:
    holder: list[Any] = []

    class Reentrant(io.RawIOBase):
        def readable(self) -> bool:
            return True

        def read(self, n: int | None = -1) -> bytes:
            next(holder[0])
            return b""

    it = pw.ArchiveIterator(Reentrant())
    holder.append(it)
    with pytest.raises(RuntimeError, match="called from inside one of its own calls"):
        next(it)


def test_messages_go_to_stderr_and_the_warcio_logger(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    capsys.readouterr()  # pytest-run-parallel's --iterations reuse the fixtures
    caplog.clear()
    data = CASES["bad_length"] + CASES["spaces_uri"]
    with caplog.at_level(logging.WARNING, logger="warcio.recordloader"):
        recs = list(pw.ArchiveIterator(io.BytesIO(data)))
    assert len(recs) == 3
    err = capsys.readouterr().err
    assert "WARNING: Record not followed by newline, perhaps Content-Length is invalid" in err
    assert [r.getMessage() for r in caplog.records] == [
        "Replacing spaces in invalid WARC-Target-URI: http://example.com/a b"
    ]
    assert caplog.records[0].name == "warcio.recordloader"


def test_class_attributes_choose_the_classes() -> None:
    calls: list[str] = []

    class Headers(pw.StatusAndHeaders):
        def __init__(self, statusline: str, headers: Any, protocol: str = "", total_len: int = 0):
            calls.append("headers")
            super().__init__(statusline, headers, protocol, total_len)

    class Record(pw.ArcWarcRecord):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append("record")
            super().__init__(*args, **kwargs)

    class FailedError(Exception):
        pass

    log = logging.getLogger("pyroid.tests.warc")

    class It(pw.ArchiveIterator):
        record_cls = Record
        headers_cls = Headers
        archive_load_failed = FailedError
        logger = log

    recs = list(It(io.BytesIO(CASES["plain"])))
    assert all(type(r) is Record and type(r.rec_headers) is Headers for r in recs)
    assert calls[:2] == ["headers", "record"]  # warcinfo: no HTTP headers
    with pytest.raises(FailedError, match="Unknown archive format"):
        list(It(io.BytesIO(b"nonsense\n")))


def test_patch_style_classes_with_warcio() -> None:
    """The recipe in .scratch/P12/bindings-api.md: warcio's own classes on pyroid's reader."""
    recordloader: Any = pytest.importorskip("warcio.recordloader")
    statusandheaders: Any = pytest.importorskip("warcio.statusandheaders")
    digestverifyingreader: Any = pytest.importorskip("warcio.digestverifyingreader")
    exceptions: Any = pytest.importorskip("warcio.exceptions")

    class Rec(recordloader.ArcWarcRecord):  # type: ignore[misc]
        content_stream = pw.ArcWarcRecord.content_stream

    class It(pw.ArchiveIterator):
        record_cls = Rec
        headers_cls = statusandheaders.StatusAndHeaders
        digest_checker_cls = digestverifyingreader.DigestChecker
        archive_load_failed = exceptions.ArchiveLoadFailed
        parser_exception = statusandheaders.StatusAndHeadersParserException
        logger = recordloader.logger

    class WIt(It):
        _known_format = "warc"

    want = trace(
        pytest.importorskip("warcio.archiveiterator"),
        CASES["plain"],
        messages=False,
        patterns=PATTERNS,
    )
    ns: Any = type("NS", (), {"ArchiveIterator": It})
    got = trace(ns, CASES["plain"], messages=False, patterns=PATTERNS)
    assert got == want
    rec = next(iter(It(io.BytesIO(CASES["plain"]), check_digests=True)))
    assert type(rec.rec_headers) is statusandheaders.StatusAndHeaders
    assert type(rec.digest_checker) is digestverifyingreader.DigestChecker
    with pytest.raises(exceptions.ArchiveLoadFailed, match="Invalid WARC record"):
        list(WIt(io.BytesIO(CASES["arc"])))


def test_digest_checker_follows_the_stream() -> None:
    it = pw.ArchiveIterator(io.BytesIO(CASES["digests"]), check_digests=True)
    first = next(it)
    assert first.digest_checker.passed is None  # nothing checked until the stream is read
    assert first.content_stream().read(3) == b"<ht"
    second = next(it)  # read_to_end on the first
    assert first.digest_checker.passed is True
    second.content_stream().read()
    assert second.digest_checker.passed is False
    # warcio's wording when the mismatch is found at the end of the stream (no colon).
    assert second.digest_checker.problems == [
        "payload digest failed sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    ]
    with pytest.raises(pw.ArchiveLoadFailed, match="payload digest failed"):
        list(pw.ArchiveIterator(io.BytesIO(CASES["digests"]), check_digests="raise"))


def test_iterator_cycle_is_collected() -> None:
    f = io.BytesIO(CASES["plain"])
    ref = weakref.ref(f)
    it = pw.ArchiveIterator(f)
    rec = next(it)
    del f, it, rec
    gc.collect()
    assert ref() is None


def test_status_and_headers_port() -> None:
    h = pw.StatusAndHeaders("200 OK", [(b"Content-Type", b"text/html"), ("X-A", "1")], "HTTP/1.1")
    assert h.headers == [("Content-Type", "text/html"), ("X-A", "1")]
    assert h.get_header("content-type") == "text/html"
    assert h.replace_header("x-a", "2") == "1"
    assert h["X-A"] == "2"
    assert h.remove_header("X-A")
    assert not h.remove_header("X-A")
    assert h.get_statuscode() == "200"
    assert h.to_str() == "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
    assert h.to_bytes() == b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
    h.add_header("X-U", "caf\xe9")
    h.add_header("X-D", 'attachment; filename="\xe9t\xe9.txt"')
    assert h.to_ascii_bytes().endswith(
        b"X-U: caf%C3%A9\r\nX-D: attachment; filename*=UTF-8''%C3%A9t%C3%A9.txt\r\n\r\n"
    )
    assert pw.StatusAndHeaders("GET / HTTP/1.1", [], is_http_request=True).protocol == "GET"
    assert not pw.StatusAndHeaders("", [])
    assert h == pw.StatusAndHeaders(h.statusline, list(h.headers), h.protocol)
    assert h != pw.StatusAndHeaders("404", [], h.protocol)
    assert repr(pw.StatusAndHeaders("1", [("a", "b")], "P")).startswith("StatusAndHeaders(")
    wsh: Any = pytest.importorskip("warcio.statusandheaders")
    ref = wsh.StatusAndHeaders("200 OK", [("Content-Type", "text/html")], "HTTP/1.1")
    ours = pw.StatusAndHeaders("200 OK", [("Content-Type", "text/html")], "HTTP/1.1")
    for name in ("to_str", "to_bytes", "to_ascii_bytes", "__repr__", "__bool__", "get_statuscode"):
        assert getattr(ours, name)() == getattr(ref, name)()
    assert ours.add_range(5, 10, 100).headers == ref.add_range(5, 10, 100).headers
    for value in ("caf\xe9", 'a; b="\xe9"', '\xe9; b="\xe9"', "x; \xe9", "plain"):
        mine = pw.StatusAndHeaders("200 OK", [("X", value)], "HTTP/1.1")
        theirs = wsh.StatusAndHeaders("200 OK", [("X", value)], "HTTP/1.1")
        outcome = []
        for obj in (mine, theirs):
            try:
                outcome.append(obj.to_ascii_bytes())
            except UnicodeEncodeError as e:
                outcome.append(str(e))
        assert outcome[0] == outcome[1], value


def test_version_and_exports() -> None:
    assert pw.WARCIO_VERSION == "1.8.2"
    assert pw.BUFF_SIZE == 16384
    for name in pw.__all__:
        assert hasattr(pw, name), name
    assert pw.ArchiveLoadFailed("x").msg == "x"
    assert pw.ArchiveLoadFailed.__module__ == "pyroid.warc"
