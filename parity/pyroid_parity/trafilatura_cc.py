"""Common Crawl HTML as datatrove 0.10.1 hands it to its Trafilatura step: WARC byte ranges to pages.

Used by ``bench/fetch.py`` (the pinned ``trafilatura/CC-MAIN-*.parquet`` corpus) and by
``pyroid_parity.trafilatura_scale``. A *slice* is a byte range of one WARC file that starts and ends on gzip member
boundaries (Common Crawl compresses every record as its own member), fetched with one HTTP range request.

Its pages go through exactly what datatrove's ``examples/fineweb.py`` runs before Trafilatura:
``WarcReader.read_file`` (``process_record``: response records whose ``WARC-Identified-Payload-Type`` is HTML,
decoded as UTF-8 or else with cchardet's guess; records without text are skipped) and ``URLFilter()``. One input
is pinned instead of live: URLFilter's ``TLDExtract`` reads tldextract's bundled public suffix list rather than
downloading the current one, so a rebuild does not depend on the day it runs. On the 14,976 records of the
CC-MAIN-2026-39 slice both lists gave the same decisions.

datatrove imports ``magic`` (python-magic) unconditionally but only calls it for records without a payload type
(crawls before 2018). Where libmagic is missing, a stub stands in that raises if it is ever called, so such a
record stops the build instead of being decoded differently.

Licence: the pages are third-party web content distributed by Common Crawl under its terms of use
(https://commoncrawl.org/terms-of-use). They are fetched into ``$PYROID_DATA``, never committed.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

CC_DATA = "https://data.commoncrawl.org/"
LICENSE = "Common Crawl terms of use (https://commoncrawl.org/terms-of-use); third-party web pages"
PIPELINE = (
    "datatrove 0.10.1 WarcReader.read_file (process_record) + URLFilter(), as in examples/fineweb.py; "
    "TLDExtract on tldextract's bundled public suffix list"
)
GZIP_MAGIC = b"\x1f\x8b\x08"
USER_AGENT = "pyroid-parity (https://github.com/ao/pyroid)"


@dataclass(frozen=True)
class Slice:
    """Bytes ``[start, end)`` of ``warc`` (a path under data.commoncrawl.org), both on gzip member boundaries."""

    crawl: str
    warc: str
    start: int
    end: int
    sha256: str | None = None

    @property
    def url(self) -> str:
        return CC_DATA + self.warc

    def describe(self) -> dict[str, object]:
        return {"warc": self.warc, "range": [self.start, self.end], "sha256": self.sha256}


def log(msg: str) -> None:
    print(f"[cc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    """Bytes ``[start, end)`` of ``url``, retrying with backoff (Common Crawl answers 503 when it is busy)."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}", "User-Agent": USER_AGENT})
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:  # pinned https URL
                if r.status != 206:
                    raise OSError(f"{url}: HTTP {r.status} to a range request")
                chunks = []
                while block := r.read(1 << 20):
                    chunks.append(block)
            data = b"".join(chunks)
            if len(data) != end - start:
                raise OSError(f"{url}: got {len(data)} bytes for a range of {end - start}")
            return data
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            log(f"range {start}-{end} of {url}: {e}; retrying in {delay:.0f} s")
            time.sleep(delay)
            delay = min(delay * 2, 120)
    raise AssertionError("unreachable")


def member_end(data: bytes, pos: int, chunk: int = 1 << 16) -> int | None:
    """End offset of the complete gzip member starting at ``pos``, or None (truncated or not a member).

    Fed in chunks through a memoryview, so a scan over many members stays linear in the size of ``data``.
    """
    view = memoryview(data)
    d = zlib.decompressobj(31)
    i = pos
    try:
        while not d.eof:
            if i >= len(data):
                return None
            piece = view[i : i + chunk]
            d.decompress(piece)
            i += len(piece)
    except zlib.error:
        return None
    return i - len(d.unused_data)


def first_record_member(data: bytes, max_scan: int = 64 << 20) -> int:
    """Offset of the first gzip member in ``data`` that holds a whole WARC record."""
    pos = data.find(GZIP_MAGIC)
    while 0 <= pos < max_scan:
        end = member_end(data, pos)
        if end is not None:
            head = zlib.decompressobj(31).decompress(data[pos:end], 16)
            if head.startswith(b"WARC/1."):
                return pos
        pos = data.find(GZIP_MAGIC, pos + 1)
    raise ValueError("no complete WARC record member found")


def last_complete_end(data: bytes, start: int = 0) -> tuple[int, int]:
    """(end offset of the last complete member at or after ``start``, number of members), like a cut of a
    partial multi-member file at its last complete member."""
    pos, n = start, 0
    while pos < len(data):
        end = member_end(data, pos)
        if end is None:
            break
        pos, n = end, n + 1
    return pos, n


def align(crawl: str, warc: str, offset: int, length: int) -> tuple[Slice, bytes]:
    """The slice of whole members inside ``[offset, offset + length)`` of ``warc``, and its bytes."""
    data = http_range(CC_DATA + warc, offset, offset + length)
    first = 0 if offset == 0 else first_record_member(data)
    end, _ = last_complete_end(data, first)
    if end <= first:
        raise ValueError(f"{warc} @{offset}: no complete member in {length} bytes")
    raw = data[first:end]
    return Slice(crawl, warc, offset + first, offset + end, hashlib.sha256(raw).hexdigest()), raw


def fetch(s: Slice) -> bytes:
    """The slice's bytes, checked: sha256 (when pinned) and member boundaries at both ends."""
    data = http_range(s.url, s.start, s.end)
    if s.sha256 is not None and hashlib.sha256(data).hexdigest() != s.sha256:
        raise ValueError(f"{s.warc} [{s.start}, {s.end}): sha256 differs from the pinned one")
    end, _ = last_complete_end(data)
    if end != len(data):
        raise ValueError(f"{s.warc} [{s.start}, {s.end}): does not end on a member boundary")
    return data


# --------------------------------------------------------------------------------- datatrove's reader + filter


def _ensure_magic() -> None:
    try:
        import magic  # noqa: F401  (python-magic with libmagic: datatrove calls it for records without a type)
    except ImportError:
        stub = types.ModuleType("magic")
        stub.__spec__ = importlib.machinery.ModuleSpec("magic", None)

        def from_buffer(*_a: object, **_k: object) -> str:
            raise RuntimeError(
                "a WARC record without WARC-Identified-Payload-Type: datatrove would call libmagic here, "
                "which is not installed"
            )

        stub.from_buffer = from_buffer  # type: ignore[attr-defined]
        sys.modules["magic"] = stub


class Pages:
    """Turn slice bytes into (id, url, html) the way datatrove's reader and URL filter do (one per process)."""

    def __init__(self) -> None:
        _ensure_magic()
        from datatrove.pipeline.filters import URLFilter
        from tldextract import TLDExtract

        self.url_filter = URLFilter()
        self.url_filter.tldextractor = TLDExtract(suffix_list_urls=())

    def __call__(self, data: bytes) -> Iterator[tuple[str, str, str]]:
        from datatrove.pipeline.readers import WarcReader

        with tempfile.TemporaryDirectory(prefix="pyroid-warc-") as d:
            (Path(d) / "slice.warc.gz").write_bytes(data)
            reader = WarcReader(d, glob_pattern="slice.warc.gz")
            docs = reader.read_file("slice.warc.gz")
            for doc in self.url_filter.run(docs):
                yield doc.id, doc.metadata["url"], doc.text
