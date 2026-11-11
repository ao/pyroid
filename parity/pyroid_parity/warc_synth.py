"""Seeded synthetic WARC and ARC files: the framing, compression, header and content variants warcio 1.8.2 meets.

    python -m pyroid_parity.warc_synth list          # case names, sizes and tags
    python -m pyroid_parity.warc_synth write DIR     # write every case to DIR (for repros)

``cases()`` is deterministic for a seed: the same names and bytes on every run, given the same zlib and brotlicffi
(the digest of the whole set is pinned with the reference digests, ``warc_compare.REF``). Nothing is committed:
files are generated in memory or into a temporary directory.

Families (the first component of a case name):

- ``frame``: plain, a gzip member per record, one gzip stream for the file, several records per member, empty
  members, zero padding and garbage between members, gzip header flags (FNAME, FCOMMENT, FEXTRA, FHCRC, reserved
  bits), stored deflate blocks, bad CRC / ISIZE, members followed by plain records and the reverse, blank-line
  framing between records (none, one, three, whitespace, junk), LF-only and CR line ends, empty and tiny files.
- ``trunc`` / ``flip``: truncation at many offsets (inside gzip headers, deflate data, trailers, WARC headers,
  blocks) and single bit flips, for plain and gzip files.
- ``length``: Content-Length values (whitespace, sign, ``_``, Unicode digits, hex, huge, negative, empty,
  missing, duplicated) and lengths that are short or long by a few bytes.
- ``header``: WARC header anomalies: name case, duplicates, continuation lines, missing colon, latin-1 and
  invalid UTF-8 bytes, Unicode whitespace, KELVIN SIGN in a name, version lines, missing WARC-Type /
  WARC-Target-URI / WARC-Record-ID / WARC-Date, ``<uri>`` and spaces, other schemes.
- ``http``: status lines (HTTP/1.0, 1.1, 2, lowercase, missing, HTTP/0.9 bodies), request lines (verbs, case
  folding with 'ſ', 'ﬆ', 'ı'), header oddities.
- ``chunked``: Transfer-Encoding chunked, valid and broken (bad sizes, extensions, missing CRLF, trailers,
  truncated chunks, 64-byte size lines, ``Chunked`` case).
- ``encoding``: Content-Encoding gzip / x-gzip / deflate (zlib and raw) / deflate_alt / br / unknown / stacked /
  mixed case, corrupt after the first block, with chunked framing.
- ``type``: record types (warcinfo, request, response, revisit with and without HTTP headers, resource, metadata,
  conversion, continuation, unknown).
- ``digest``: block and payload digests in base32 / base16 / base64 (incl. ``-_``), sha1 / sha256 / md5 / ...,
  wrong values, unknown algorithms, other hashlib spellings, segmented records.
- ``arc``: ARC v1 with and without ``filedesc://``, v2-style header lines, wrong field counts, timestamps,
  non-ASCII URLs, gzip, ARC and WARC mixed.
- ``charset``: HTML in many charsets (datatrove's cchardet path), payload types, records without a payload type
  (datatrove's libmagic path), empty payloads.
- ``big``: records larger than warcio's 16 KiB blocks and CPython's 128 KiB GzipFile reads, with damage past them.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import random
import re
import struct
import sys
import uuid
import zlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

CRLF = b"\r\n"
SEED = 20261110


@dataclass(frozen=True)
class Case:
    """One synthetic file. ``suffix`` decides what datatrove infers (``.gz`` = gzip)."""

    name: str
    data: bytes
    suffix: str = ".warc"
    tags: frozenset[str] = field(default_factory=frozenset)

    @property
    def filename(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.name) + self.suffix

    @property
    def gz(self) -> bool:
        return self.suffix.endswith(".gz")


# --------------------------------------------------------------------------------------------- record bytes

Header = tuple[bytes, bytes] | bytes  # (name, value) rendered "name: value", or a raw line


def header_lines(headers: Iterable[Header], eol: bytes = CRLF, sep: bytes = b": ") -> bytes:
    out = []
    for h in headers:
        out.append(h if isinstance(h, bytes) else h[0] + sep + h[1])
        out.append(eol)
    return b"".join(out)


def record_id(rng: random.Random) -> bytes:
    return b"<urn:uuid:%s>" % str(uuid.UUID(int=rng.getrandbits(128), version=4)).encode()


AUTO = object()


def warc_record(
    rec_type: bytes | None,
    block: bytes,
    *,
    rng: random.Random,
    uri: bytes | None = b"http://example.com/",
    length: bytes | object | None = AUTO,
    content_type: bytes | object | None = AUTO,
    payload_type: bytes | None = None,
    extra: Sequence[Header] = (),
    first: Sequence[Header] = (),
    version: bytes = b"WARC/1.0",
    eol: bytes = CRLF,
    tail: bytes = CRLF + CRLF,
    rid: bytes | object | None = AUTO,
    date: bytes | None = b"2026-01-02T03:04:05Z",
) -> bytes:
    """A WARC record: version line, headers (CC's order), blank line, block, ``tail``.

    ``None`` leaves a header out; ``length`` defaults to ``len(block)``; ``first`` goes before the usual headers,
    ``extra`` after them.
    """
    if content_type is AUTO:
        content_type = {
            b"response": b"application/http; msgtype=response",
            b"request": b"application/http; msgtype=request",
            b"revisit": b"application/http; msgtype=response",
            b"warcinfo": b"application/warc-fields",
            b"metadata": b"application/warc-fields",
            b"conversion": b"text/plain",
        }.get(rec_type or b"", b"application/octet-stream")
    hs: list[Header] = list(first)
    if rec_type is not None:
        hs.append((b"WARC-Type", rec_type))
    if date is not None:
        hs.append((b"WARC-Date", date))
    if rid is AUTO:
        rid = record_id(rng)
    if rid is not None:
        hs.append((b"WARC-Record-ID", rid))  # type: ignore[arg-type]
    if length is AUTO:
        length = str(len(block)).encode()
    if length is not None:
        hs.append((b"Content-Length", length))  # type: ignore[arg-type]
    if content_type is not None:
        hs.append((b"Content-Type", content_type))  # type: ignore[arg-type]
    if uri is not None:
        hs.append((b"WARC-Target-URI", uri))
    if payload_type is not None:
        hs.append((b"WARC-Identified-Payload-Type", payload_type))
    hs.extend(extra)
    return version + eol + header_lines(hs, eol) + eol + block + tail


def http_response(
    body: bytes,
    *,
    status: bytes = b"HTTP/1.1 200 OK",
    headers: Sequence[Header] = ((b"Content-Type", b"text/html; charset=utf-8"),),
    eol: bytes = CRLF,
    length_header: bool = True,
) -> bytes:
    hs = list(headers)
    if length_header:
        hs.append((b"Content-Length", str(len(body)).encode()))
    return status + eol + header_lines(hs, eol) + eol + body


def http_request(
    line: bytes = b"GET / HTTP/1.1",
    headers: Sequence[Header] = ((b"Host", b"example.com"),),
    body: bytes = b"",
) -> bytes:
    return line + CRLF + header_lines([*headers, (b"User-Agent", b"pyroid-parity")]) + CRLF + body


WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "is",
    "was",
    "for",
    "on",
    "that",
    "with",
    "as",
    "by",
    "at",
    "from",
    "his",
    "her",
    "an",
    "were",
    "are",
    "which",
    "this",
    "be",
    "or",
    "has",
    "had",
    "not",
    "but",
    "it",
    "its",
    "they",
    "one",
    "first",
    "new",
    "have",
    "their",
    "after",
    "also",
    "who",
    "been",
    "two",
    "more",
    "when",
    "there",
    "all",
    "than",
    "other",
    "into",
    "time",
    "page",
    "record",
    "archive",
    "crawl",
    "header",
    "content",
    "length",
    "block",
    "member",
    "stream",
    "offset",
    "digest",
    "payload",
]


def html_page(rng: random.Random, paragraphs: int = 4, extra: str = "") -> str:
    body = "".join(
        "<p>" + " ".join(rng.choice(WORDS) for _ in range(rng.randint(8, 40))) + ".</p>\n"
        for _ in range(paragraphs)
    )
    title = " ".join(rng.choice(WORDS) for _ in range(4))
    return (
        f"<!DOCTYPE html>\n<html><head><title>{title}</title></head>\n<body>\n{extra}{body}</body></html>\n"
    )


def response(
    rng: random.Random,
    body: bytes | None = None,
    *,
    http_headers: Sequence[Header] = ((b"Content-Type", b"text/html; charset=utf-8"),),
    status: bytes = b"HTTP/1.1 200 OK",
    payload_type: bytes | None = b"text/html",
    uri: bytes | None = b"http://example.com/",
    **kw: object,
) -> bytes:
    if body is None:
        body = html_page(rng).encode()
    block = http_response(body, status=status, headers=http_headers)
    return warc_record(b"response", block, rng=rng, uri=uri, payload_type=payload_type, **kw)  # type: ignore[arg-type]


def warcinfo(rng: random.Random, **kw: object) -> bytes:
    block = header_lines(
        [
            (b"software", b"pyroid-parity synthetic"),
            (b"format", b"WARC File Format 1.0"),
            (b"isPartOf", b"SYNTH-0001"),
        ]
    )
    return warc_record(
        b"warcinfo", block, rng=rng, uri=None, extra=[(b"WARC-Filename", b"synth.warc.gz")], **kw
    )  # type: ignore[arg-type]


def request(rng: random.Random, line: bytes = b"GET / HTTP/1.1", **kw: object) -> bytes:
    return warc_record(b"request", http_request(line), rng=rng, **kw)  # type: ignore[arg-type]


def metadata(rng: random.Random, **kw: object) -> bytes:
    block = header_lines([(b"fetchTimeMs", b"%d" % rng.randint(1, 9999)), (b"charset-detected", b"UTF-8")])
    return warc_record(b"metadata", block, rng=rng, **kw)  # type: ignore[arg-type]


def base_records(rng: random.Random, n_pages: int = 3) -> list[bytes]:
    """warcinfo, then request / response / metadata triples, like a Common Crawl WARC file."""
    out = [warcinfo(rng)]
    for i in range(n_pages):
        uri = b"http://example.com/page/%d" % i
        out += [request(rng, uri=uri), response(rng, uri=uri), metadata(rng, uri=uri)]
    return out


# --------------------------------------------------------------------------------------------- gzip framing


def gz(data: bytes, level: int = 6) -> bytes:
    """``gzip.compress(data, level, mtime=0)`` as CPython 3.13+ writes it, on every Python.

    3.12 keeps zlib's OS byte (3, Unix) in the header and 3.13 writes 255 (unknown), so the pinned
    inputs would otherwise depend on the interpreter.
    """
    out = bytearray(zlib.compress(data, level=level, wbits=31))
    out[9] = 255
    return bytes(out)


def per_record(records: Sequence[bytes], level: int = 6) -> bytes:
    return b"".join(gz(r, level) for r in records)


def grouped(records: Sequence[bytes], k: int) -> bytes:
    return b"".join(gz(b"".join(records[i : i + k])) for i in range(0, len(records), k))


def raw_deflate(data: bytes, level: int = 6) -> bytes:
    c = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
    return c.compress(data) + c.flush()


def stored_deflate(data: bytes, block: int = 1000) -> bytes:
    """Deflate with stored (uncompressed) blocks only: no compressor involved, bytes fixed by the format."""
    out = []
    chunks = [data[i : i + block] for i in range(0, len(data), block)] or [b""]
    for i, c in enumerate(chunks):
        final = 1 if i == len(chunks) - 1 else 0
        out.append(bytes([final]) + struct.pack("<HH", len(c), len(c) ^ 0xFFFF) + c)
    return b"".join(out)


def gzip_member(
    data: bytes,
    *,
    deflate: bytes | None = None,
    flags: int = 0,
    fname: bytes = b"",
    comment: bytes = b"",
    fextra: bytes = b"",
    hcrc: bool = False,
    mtime: int = 0,
    xfl: int = 0,
    os_byte: int = 255,
    crc: int | None = None,
    isize: int | None = None,
) -> bytes:
    """A gzip member built by hand (RFC 1952): any header flags, wrong CRC / ISIZE on request."""
    if fname:
        flags |= 0x08
    if comment:
        flags |= 0x10
    if fextra:
        flags |= 0x04
    if hcrc:
        flags |= 0x02
    head = b"\x1f\x8b\x08" + bytes([flags]) + struct.pack("<I", mtime) + bytes([xfl, os_byte])
    if flags & 0x04:
        head += struct.pack("<H", len(fextra)) + fextra
    if flags & 0x08:
        head += fname + b"\0"
    if flags & 0x10:
        head += comment + b"\0"
    if flags & 0x02:
        head += struct.pack("<H", zlib.crc32(head) & 0xFFFF)
    body = raw_deflate(data) if deflate is None else deflate
    c = zlib.crc32(data) if crc is None else crc
    n = len(data) & 0xFFFFFFFF if isize is None else isize
    return head + body + struct.pack("<II", c & 0xFFFFFFFF, n & 0xFFFFFFFF)


def member_bounds(data: bytes) -> list[tuple[int, int]]:
    """(start, end) of each complete gzip member in ``data``."""
    out, pos = [], 0
    while pos < len(data):
        d = zlib.decompressobj(31)
        try:
            d.decompress(data[pos:])
        except zlib.error:
            break
        if not d.eof:
            break
        end = len(data) - len(d.unused_data)
        out.append((pos, end))
        pos = end
    return out


# ------------------------------------------------------------------------------------------------ helpers


class Builder:
    """Collects cases; ``add`` checks names are unique."""

    def __init__(self) -> None:
        self.cases: list[Case] = []
        self.names: set[str] = set()

    def add(self, name: str, data: bytes, suffix: str = ".warc", *tags: str) -> None:
        if name in self.names:
            raise ValueError(f"duplicate synthetic case {name}")
        self.names.add(name)
        self.cases.append(Case(name, data, suffix, frozenset(tags)))

    def both(self, name: str, records: Sequence[bytes], *tags: str) -> None:
        """The same records plain and as one gzip member per record."""
        self.add(name, b"".join(records), ".warc", *tags)
        self.add(name + "/gz", per_record(records), ".warc.gz", *tags)


def brotli_module() -> object | None:
    try:
        import brotlicffi
    except ImportError:
        return None
    return brotlicffi


# ------------------------------------------------------------------------------------------------ families


def frame(b: Builder, rng: random.Random) -> None:
    recs = base_records(rng)
    b.add("frame/plain", b"".join(recs))
    b.add("frame/per-record", per_record(recs), ".warc.gz")
    for lvl in (0, 1, 9):
        b.add(f"frame/per-record-level{lvl}", per_record(recs, lvl), ".warc.gz")
    b.add("frame/single-stream", gz(b"".join(recs)), ".warc.gz")
    b.add("frame/single-stream-one-record", gz(recs[0]), ".warc.gz")
    b.add("frame/single-stream-plain-suffix", gz(b"".join(recs)), ".warc")
    for k in (2, 3, 5):
        b.add(f"frame/grouped-{k}", grouped(recs, k), ".warc.gz")
    b.add(
        "frame/stored-blocks", b"".join(gzip_member(r, deflate=stored_deflate(r)) for r in recs), ".warc.gz"
    )
    b.add(
        "frame/stored-blocks-small",
        b"".join(gzip_member(r, deflate=stored_deflate(r, 7)) for r in recs),
        ".warc.gz",
    )
    flag_variants: dict[str, dict[str, object]] = {
        "fname": {"fname": b"record.warc"},
        "fcomment": {"comment": b"a comment"},
        "fextra": {"fextra": b"AB\x02\x00xy"},
        "fhcrc": {"hcrc": True},
        "all-flags": {"fname": b"n", "comment": b"c", "fextra": b"XY\x00\x00", "hcrc": True},
        "reserved-0x20": {"flags": 0x20},
        "reserved-0x80": {"flags": 0x80},
        "mtime": {"mtime": 1_700_000_000},
        "xfl-os": {"xfl": 2, "os_byte": 3},
        "fname-latin1": {"fname": b"r\xe9cord.warc"},
    }
    for name, kw in flag_variants.items():
        b.add(f"frame/gzip-{name}", b"".join(gzip_member(r, **kw) for r in recs), ".warc.gz")  # type: ignore[arg-type]
        b.add(
            f"frame/gzip-{name}-second-only",
            gz(recs[0]) + gzip_member(recs[1], **kw) + per_record(recs[2:]),  # type: ignore[arg-type]
            ".warc.gz",
        )
    empty = gz(b"")
    b.add("frame/empty-member-first", empty + per_record(recs), ".warc.gz")
    b.add("frame/empty-member-between", gz(recs[0]) + empty + per_record(recs[1:]), ".warc.gz")
    b.add("frame/empty-member-last", per_record(recs) + empty, ".warc.gz")
    b.add("frame/empty-members-only", empty * 3, ".warc.gz")
    for pad in (1, 7, 512):
        b.add(f"frame/zero-pad-{pad}-between", gz(recs[0]) + b"\0" * pad + per_record(recs[1:]), ".warc.gz")
        b.add(f"frame/zero-pad-{pad}-end", per_record(recs) + b"\0" * pad, ".warc.gz")
    for junk_name, junk in {
        "text": b"junk",
        "newline": b"\n",
        "crlf": CRLF,
        "random": rng.randbytes(37),
        "gzip-magic": b"\x1f\x8b\x08\x00",
    }.items():
        b.add(f"frame/garbage-{junk_name}-between", gz(recs[0]) + junk + per_record(recs[1:]), ".warc.gz")
        b.add(f"frame/garbage-{junk_name}-end", per_record(recs) + junk, ".warc.gz")
        b.add(f"frame/garbage-{junk_name}-start", junk + per_record(recs), ".warc.gz")
    b.add("frame/gzip-then-plain", gz(recs[0]) + b"".join(recs[1:]), ".warc.gz")
    b.add("frame/gzip-then-plain-records", per_record(recs[:3]) + b"".join(recs[3:]), ".warc.gz")
    b.add("frame/plain-then-gzip", recs[0] + per_record(recs[1:]), ".warc.gz")
    b.add("frame/plain-then-gzip-plain-suffix", recs[0] + per_record(recs[1:]), ".warc")
    for i, what in ((0, "first"), (3, "middle"), (len(recs) - 1, "last")):
        mem = [gz(r) for r in recs]
        bad = bytearray(mem[i])
        bad[-8] ^= 0x01
        b.add(f"frame/bad-crc-{what}", b"".join(mem[:i]) + bytes(bad) + b"".join(mem[i + 1 :]), ".warc.gz")
        bad = bytearray(mem[i])
        bad[-4] ^= 0x01
        b.add(f"frame/bad-isize-{what}", b"".join(mem[:i]) + bytes(bad) + b"".join(mem[i + 1 :]), ".warc.gz")
    b.add("frame/bad-magic", b"\x1f\x8c" + per_record(recs)[2:], ".warc.gz")
    b.add("frame/bad-method", b"\x1f\x8b\x07" + per_record(recs)[3:], ".warc.gz")
    # Blank-line framing between plain records (each record ends in "\r\n\r\n" unless noted).
    bodies = [r.removesuffix(CRLF + CRLF) for r in recs]
    for name, sep in {
        "none": b"",
        "one": CRLF,
        "two": CRLF * 2,
        "three": CRLF * 3,
        "lf-lf": b"\n\n",
        "spaces": b"\r\n  \t\r\n",
        "junk-line": CRLF + b"junk line\r\n",
        "two-junk-lines": CRLF + b"junk\r\nmore junk\r\n\r\n",
        "nbsp-line": CRLF + " 　".encode() + CRLF,
        "ff-vt": CRLF + b"\x0c\x0b\x1c\r\n",
    }.items():
        b.add(f"frame/sep-{name}", sep.join(bodies) + sep, ".warc")
        b.add(f"frame/sep-{name}/gz", b"".join(gz(x + sep) for x in bodies), ".warc.gz")
    # Line ends.
    lf = [warc_record(b"response", http_response(b"<p>lf</p>", eol=b"\n"), rng=rng, eol=b"\n", tail=b"\n\n")]
    lf.append(
        warc_record(b"resource", b"text\n", rng=rng, eol=b"\n", tail=b"\n\n", content_type=b"text/plain")
    )
    b.both("frame/lf-only", lf)
    cr = [warc_record(b"resource", b"x", rng=rng, eol=b"\r", tail=b"\r\r")]
    b.both("frame/cr-only", cr)
    mixed = [warc_record(b"response", http_response(b"<p>m</p>", eol=b"\n"), rng=rng), lf[1]]
    b.both("frame/mixed-eol", mixed)
    # Tiny and empty files.
    for name, data in {
        "empty": b"",
        "crlf": CRLF,
        "blank-lines": CRLF * 5,
        "one-byte": b"W",
        "version-only": b"WARC/1.0",
        "version-line": b"WARC/1.0\r\n",
        "version-and-blank": b"WARC/1.0\r\n\r\n",
        "header-no-blank": b"WARC/1.0\r\nWARC-Type: resource\r\nContent-Length: 0\r\n",
        "nul-bytes": b"\0" * 64,
        "random": rng.randbytes(300),
        "html": b"<html><body>not a warc</body></html>\n",
    }.items():
        b.add(f"frame/tiny-{name}", data, ".warc")
        b.add(f"frame/tiny-{name}/gz", gz(data) if data else b"", ".warc.gz")


def trunc_flip(b: Builder, rng: random.Random) -> None:
    recs = base_records(rng, 2)
    plain = b"".join(recs)
    zipped = per_record(recs)
    bounds = member_bounds(zipped)
    offsets: set[int] = set()
    for s, e in bounds[:3]:
        offsets |= {s + 1, s + 3, s + 9, s + 10, s + 14, (s + e) // 2, e - 9, e - 8, e - 5, e - 1, e, e + 1}
    offsets |= {rng.randrange(1, len(zipped)) for _ in range(8)}
    for off in sorted(o for o in offsets if 0 < o < len(zipped)):
        b.add(f"trunc/gz-at-{off}", zipped[:off], ".warc.gz")
    single = gz(plain)
    for off in sorted({10, 20, len(single) // 3, len(single) // 2, len(single) - 8, len(single) - 1}):
        b.add(f"trunc/single-stream-at-{off}", single[:off], ".warc.gz")
    # Plain: inside the version line, the headers, the blank line, the block, the record tail.
    first = recs[0]
    hdr_end = first.index(b"\r\n\r\n")
    cuts = {3, 8, 9, 20, hdr_end, hdr_end + 2, hdr_end + 3, len(first) - 3, len(first) - 1, len(first)}
    resp_start = len(recs[0]) + len(recs[1])
    resp = recs[2]
    rh = resp.index(b"\r\n\r\n")
    cuts |= {
        resp_start + rh + 6,
        resp_start + rh + 30,
        resp_start + len(resp) // 2,
        resp_start + len(resp) - 2,
    }
    for off in sorted(c for c in cuts if 0 < c < len(plain)):
        b.add(f"trunc/plain-at-{off}", plain[:off], ".warc")
    for i in range(14):
        pos = rng.randrange(len(zipped))
        bit = 1 << rng.randrange(8)
        flipped = bytearray(zipped)
        flipped[pos] ^= bit
        b.add(f"flip/gz-{i}-at-{pos}", bytes(flipped), ".warc.gz")
    for i in range(10):
        pos = rng.randrange(len(plain))
        flipped = bytearray(plain)
        flipped[pos] ^= 1 << rng.randrange(8)
        b.add(f"flip/plain-{i}-at-{pos}", bytes(flipped), ".warc")


def lengths(b: Builder, rng: random.Random) -> None:
    def rec(length: object, body: bytes | None = None, **kw: object) -> list[bytes]:
        block = http_response(body or b"<p>length</p>")
        return [
            warc_record(b"response", block, rng=rng, length=length, payload_type=b"text/html", **kw),  # type: ignore[arg-type]
            metadata(rng),
        ]

    block_len = len(http_response(b"<p>length</p>"))
    n = str(block_len)
    arabic = "".join(chr(0x0660 + int(d)) for d in n)
    full = "".join(chr(0xFF10 + int(d)) for d in n)
    deva = "".join(chr(0x0966 + int(d)) for d in n)
    values: dict[str, bytes | None] = {
        "exact": n.encode(),
        "spaces": f"  {n}  ".encode(),
        "tab": f"\t{n}".encode(),
        "plus": f"+{n}".encode(),
        "underscore": f"{n[:1]}_{n[1:]}".encode(),
        "double-underscore": f"{n[:1]}__{n[1:]}".encode(),
        "leading-zeros": f"000{n}".encode(),
        "arabic-indic": arabic.encode(),
        "fullwidth": full.encode(),
        "devanagari": deva.encode(),
        "superscript": "²".encode(),
        "latin1-superscript": b"\xb2",
        "hex": b"0x" + f"{block_len:x}".encode(),
        "float": f"{n}.0".encode(),
        "exp": b"1e2",
        "negative": b"-5",
        "minus-zero": b"-0",
        "zero": b"0",
        "empty": b"",
        "letters": b"abc",
        "huge": b"9" * 30,
        "two-numbers": f"{n} {n}".encode(),
        "nbsp-suffix": n.encode() + " ".encode(),
        "missing": None,
    }
    for name, v in values.items():
        b.both(f"length/{name}", rec(v))
    for d in (-1, -2, -10, 1, 2, 10, 1000):
        b.both(f"length/off-by-{d:+d}", rec(str(block_len + d).encode()))
    b.both(
        "length/duplicate",
        rec(n.encode(), extra=[(b"Content-Length", b"3")]),
    )
    b.both(
        "length/duplicate-first-wrong",
        rec(None, first=[(b"Content-Length", b"3")], extra=[(b"Content-Length", n.encode())]),
    )
    b.both("length/lowercase-name", rec(None, extra=[(b"content-length", n.encode())]))
    b.both("length/uppercase-name", rec(None, extra=[(b"CONTENT-LENGTH", n.encode())]))
    b.both("length/missing-last-record", [metadata(rng), *rec(None)[:1]])
    big = http_response(b"x" * 40_000)
    b.both("length/missing-big", [warc_record(b"response", big, rng=rng, length=None), metadata(rng)])
    b.both(
        "length/short-in-big",
        [warc_record(b"response", big, rng=rng, length=b"30000"), metadata(rng)],
    )


def headers(b: Builder, rng: random.Random) -> None:
    page = http_response(b"<p>headers</p>")

    def one(name: str, **kw: object) -> None:
        recs = [
            warc_record(
                kw.pop("rec_type", b"response"),
                kw.pop("block", page),
                rng=rng,
                payload_type=b"text/html",
                **kw,
            ),
            metadata(rng),
        ]  # type: ignore[arg-type]
        b.both(f"header/{name}", recs)

    one("case-lower", first=[], extra=[(b"warc-type", b"request")])
    one("dup-type", extra=[(b"WARC-Type", b"request")])
    one("dup-uri", extra=[(b"WARC-Target-URI", b"http://other.example/")])
    one("continuation-space", extra=[(b"X-Long", b"part one"), b" part two", b"\tpart three"])
    one("continuation-uri", uri=b"http://example.com/very/long", extra=[b" /continued"])
    one("continuation-after-no-colon", extra=[b"NoColonHere", b" continued"])
    one("no-colon", extra=[b"NoColonHere"])
    one("empty-name", extra=[b": value"])
    one("empty-value", extra=[(b"X-Empty", b"")])
    one("space-before-colon", extra=[b"X-Space \t: value"])
    one("colon-in-value", extra=[(b"X-Time", b"12:34:56")])
    one("latin1-value", extra=[(b"X-Latin", b"caf\xe9")])
    one("latin1-uri", uri=b"http://example.com/caf\xe9")
    one("invalid-utf8-name", extra=[(b"X-\xff\xfe", b"v")])
    one("utf8-value", extra=[(b"X-Utf8", "café 中文".encode())])
    one("utf8-uri", uri="http://example.com/été".encode())
    one("bom-version", version=b"\xef\xbb\xbfWARC/1.0")
    for ws_name, ws in {
        "nbsp": " ",
        "ideographic": "　",
        "line-sep": " ",
        "nel-utf8": "\u0085",
        "en-quad": " ",
        "zwsp": "\u200b",
    }.items():
        one(f"ws-{ws_name}-value", extra=[(b"X-Ws", ("a" + ws).encode())])
        one(f"ws-{ws_name}-type", rec_type=("response" + ws).encode())
        one(f"ws-{ws_name}-uri", uri=("http://example.com/" + ws).encode())
        one(f"ws-{ws_name}-lead", uri=(ws + "http://example.com/").encode())
    for byte in (b"\x1c", b"\x1d", b"\x1e", b"\x1f", b"\x0b", b"\x0c", b"\x85", b"\xa0"):
        one(f"ws-byte-{byte.hex()}-length", length=str(len(page)).encode() + byte)
        one(f"ws-byte-{byte.hex()}-type", rec_type=b"response" + byte)
    one("kelvin-block-digest", extra=[("WARC-BlocK-Digest".encode(), b"sha1:AAAA")])
    one("kelvin-type-name", first=[("WARC-Kype".encode(), b"request")])
    one("dotted-i-name", extra=[("WARC-Target-URİ".encode(), b"http://dotted.example/")])
    for v in (
        b"WARC/1.1",
        b"WARC/0.17",
        b"WARC/0.18",
        b"warc/1.0",
        b"WARC/1.0 extra",
        b"WARC/1.0  ",
        b" WARC/1.0",
    ):
        one(
            f"version-{v.decode().strip().replace('/', '_').replace(' ', '-') or 'x'}{'-lead' if v[:1] == b' ' else ''}{'-trail' if v.endswith(b'  ') else ''}",
            version=v,
        )
    one("version-2.0", version=b"WARC/2.0")
    one("version-garbage", version=b"GARBAGE")
    one("no-type", rec_type=None)
    one("no-uri", uri=None)
    one("no-uri-zero-length", uri=None, block=b"", length=b"0")
    one("no-record-id", rid=None)
    one("no-date", date=None)
    one("empty-uri", uri=b"")
    one("empty-record-id", rid=b"")
    one("angle-uri", uri=b"<http://example.com/angle>")
    one("angle-uri-open", uri=b"<http://example.com/open")
    one("space-uri", uri=b"http://example.com/a b c")
    one("angle-space-uri", uri=b"<http://example.com/a b>")
    one("tab-uri", uri=b"http://example.com/a\tb")
    one("dns-uri", uri=b"dns:example.com")
    one("ftp-uri", uri=b"ftp://example.com/file")
    one("upper-scheme", uri=b"HTTP://EXAMPLE.COM/")
    one("https-uri", uri=b"https://example.com/")
    one("whois-uri", uri=b"whois://example.com/")
    one("bad-content-type", content_type=b"text/html")
    one("no-content-type", content_type=None)
    many = [(b"X-H%d" % i, b"v" * (i % 50)) for i in range(300)]
    one("many-headers", extra=many)
    one("long-header", extra=[(b"X-Long", b"z" * 70_000)])


def http(b: Builder, rng: random.Random) -> None:
    body = b"<html><body><p>http</p></body></html>"

    def resp(name: str, block: bytes, rec_type: bytes = b"response", **kw: object) -> None:
        recs = [warc_record(rec_type, block, rng=rng, payload_type=b"text/html", **kw), metadata(rng)]  # type: ignore[arg-type]
        b.both(f"http/{name}", recs)

    for name, status in {
        "1.0-200": b"HTTP/1.0 200 OK",
        "1.1-404": b"HTTP/1.1 404 Not Found",
        "no-reason": b"HTTP/1.1 200",
        "protocol-only": b"HTTP/1.1",
        "http2": b"HTTP/2 200",
        "http2.0": b"HTTP/2.0 200 OK",
        "lowercase": b"http/1.1 200 ok",
        "spaces": b"HTTP/1.1  200  OK  ",
        "icy": b"ICY 200 OK",
        "leading-space": b" HTTP/1.1 200 OK",
        "latin1-reason": b"HTTP/1.1 200 D\xe9j\xe0",
        "utf8-reason": "HTTP/1.1 200 Ça va".encode(),
        "huge-code": b"HTTP/1.1 99999999999999999999 X",
        "long-s": "HTTP/1.1 200 ſure".encode(),
    }.items():
        resp(f"status-{name}", http_response(body, status=status))
    resp("empty-status-line", CRLF + header_lines([(b"Content-Type", b"text/html")]) + CRLF + body)
    resp("http09", body)
    resp("http09-lines", b"line one\r\nline: two\r\n\r\n" + body)
    resp("headers-only", b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
    resp("headers-no-blank", b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n")
    resp("status-only-no-eol", b"HTTP/1.1 200 OK")
    resp("lf-headers", http_response(body, eol=b"\n"))
    resp(
        "header-continuation",
        b"HTTP/1.1 200 OK\r\nX-A: one\r\n two\r\nContent-Type: text/html\r\n\r\n" + body,
    )
    resp("header-no-colon", b"HTTP/1.1 200 OK\r\nBroken header\r\nContent-Type: text/html\r\n\r\n" + body)
    resp("header-latin1", b"HTTP/1.1 200 OK\r\nX-L: \xe9\xe8\r\nContent-Type: text/html\r\n\r\n" + body)
    resp(
        "ws-line-ends-headers", b"HTTP/1.1 200 OK\r\nX-A: 1\r\n   \r\nContent-Type: text/html\r\n\r\n" + body
    )
    resp(
        "content-type-dup",
        http_response(body, headers=[(b"Content-Type", b"text/plain"), (b"content-type", b"text/html")]),
    )
    for name, line in {
        "get": b"GET / HTTP/1.1",
        "get-lower": b"get / HTTP/1.1",
        "post": b"POST /form HTTP/1.0",
        "head": b"HEAD / HTTP/1.1",
        "brew": b"BREW /pot HTTP/1.1",
        "long-s-post": "poſt / HTTP/1.1".encode(),
        "st-ligature-post": "poﬆ / HTTP/1.1".encode(),
        "long-st-ligature": "poﬅ / HTTP/1.1".encode(),
        "dotless-i-options": "optıons * HTTP/1.1".encode(),
        "sharp-s": "ß / HTTP/1.1".encode(),
        "fi-ligature-patch": "PATCH /ﬁ HTTP/1.1".encode(),
        "no-space": b"GET",
        "empty": b"",
    }.items():
        rec = warc_record(b"request", http_request(line), rng=rng)
        b.both(f"http/request-{name}", [rec, metadata(rng)])
    resp(
        "revisit-headers",
        http_response(b"", headers=[(b"Content-Type", b"text/html")], length_header=False),
        rec_type=b"revisit",
    )
    resp("revisit-empty-block", b"", rec_type=b"revisit")
    resp("revisit-crlf-block", CRLF, rec_type=b"revisit")
    resp("resource-http-uri", body, rec_type=b"resource", content_type=b"text/html")


def chunk_body(data: bytes, sizes: Sequence[int], ext: bytes = b"", fmt: str = "{:x}") -> bytes:
    out, pos, i = [], 0, 0
    while pos < len(data):
        n = min(sizes[i % len(sizes)], len(data) - pos)
        out.append(fmt.format(n).encode() + ext + CRLF + data[pos : pos + n] + CRLF)
        pos += n
        i += 1
    out.append(b"0" + ext + CRLF + CRLF)
    return b"".join(out)


def chunked(b: Builder, rng: random.Random) -> None:
    body = html_page(rng, 6).encode()

    def with_te(name: str, payload: bytes, te: bytes = b"chunked", extra: Sequence[Header] = ()) -> None:
        block = http_response(
            payload,
            headers=[(b"Content-Type", b"text/html"), (b"Transfer-Encoding", te), *extra],
            length_header=False,
        )
        recs = [warc_record(b"response", block, rng=rng, payload_type=b"text/html"), metadata(rng)]
        b.both(f"chunked/{name}", recs)

    with_te("valid", chunk_body(body, [100]))
    with_te("valid-one", chunk_body(body, [len(body)]))
    with_te("valid-mixed-sizes", chunk_body(body, [1, 7, 64, 300]))
    with_te("extension", chunk_body(body, [200], ext=b";name=value"))
    with_te("extension-quoted", chunk_body(body, [200], ext=b';a="b;c"'))
    with_te("uppercase-hex", chunk_body(body, [255, 171], fmt="{:X}"))
    with_te("leading-zeros", chunk_body(body, [300], fmt="000{:x}"))
    with_te("space-after-size", chunk_body(body, [300], fmt="{:x} "))
    with_te("plus-size", chunk_body(body, [300], fmt="+{:x}"))
    with_te("underscore-size", b"1_0\r\n" + body[:16] + CRLF + b"0\r\n\r\n")
    with_te("bad-hex", b"zz\r\n" + body)
    with_te("lf-only-size", b"10\n" + body[:16] + b"\n0\n\n")
    with_te(
        "missing-crlf-after-data", b"10\r\n" + body[:16] + b"XX10\r\n" + body[16:32] + CRLF + b"0\r\n\r\n"
    )
    with_te("no-final-crlf", chunk_body(body, [500])[:-2])
    with_te("trailer", chunk_body(body, [500])[:-2] + b"X-Trailer: yes\r\n\r\n")
    with_te("truncated-chunk", b"%x\r\n" % (len(body) + 500) + body)
    with_te("truncated-size-line", chunk_body(body, [500])[:-5])
    with_te("huge-size", b"%x\r\n" % (2**31 + 1) + body)
    with_te("max-size", b"%x\r\n" % (2**31) + body)
    with_te("long-size-line", b"0" * 70 + b"10\r\n" + body[:16] + CRLF + b"0\r\n\r\n")
    with_te("size-line-63", b"0" * 59 + b"10\r\n" + body[:16] + CRLF + b"0\r\n\r\n")
    with_te("zero-only", b"0\r\n\r\n")
    with_te("empty", b"")
    with_te("not-chunked", body)
    with_te("case-Chunked", chunk_body(body, [100]), te=b"Chunked")
    with_te("gzip-chunked-list", chunk_body(body, [100]), te=b"gzip, chunked")
    with_te(
        "dup-te-first-identity",
        chunk_body(body, [100]),
        te=b"identity",
        extra=[(b"Transfer-Encoding", b"chunked")],
    )
    with_te("negative-size", b"-10\r\n" + body)
    big = html_page(rng, 200).encode()
    with_te("big", chunk_body(big, [4096, 10000, 1]))


def encodings(b: Builder, rng: random.Random) -> None:
    body = html_page(rng, 10).encode()
    big = "".join(html_page(rng, 50) for _ in range(12)).encode()  # compresses to well over 16 KiB
    noise = rng.randbytes(60_000)
    big_noisy = big[:20_000] + base64.b64encode(noise) + big[20_000:]
    br = brotli_module()

    def enc(name: str, payload: bytes, ce: bytes | None, te: bytes | None = None, *tags: str) -> None:
        hs: list[Header] = [(b"Content-Type", b"text/html; charset=utf-8")]
        if ce is not None:
            hs.append((b"Content-Encoding", ce))
        if te is not None:
            hs.append((b"Transfer-Encoding", te))
        block = http_response(payload, headers=hs, length_header=te is None)
        recs = [warc_record(b"response", block, rng=rng, payload_type=b"text/html"), metadata(rng)]
        b.add(f"encoding/{name}", b"".join(recs), ".warc", "encoding", *tags)
        b.add(f"encoding/{name}/gz", per_record(recs), ".warc.gz", "encoding", *tags)

    deflate_zlib = zlib.compress(body)
    deflate_raw = raw_deflate(body)
    gzipped = gz(body)
    enc("identity", body, None)
    enc("gzip", gzipped, b"gzip")
    enc("x-gzip", gzipped, b"x-gzip")
    enc("X-GZIP-upper", gzipped, b"X-GZIP")
    enc("GZip-mixed", gzipped, b"GZip")
    enc("gzip-space", gzipped, b" gzip ")
    enc("deflate-zlib", deflate_zlib, b"deflate")
    enc("deflate-raw", deflate_raw, b"deflate")
    enc("deflate_alt", deflate_raw, b"deflate_alt")
    enc("deflate_alt-zlib", deflate_zlib, b"deflate_alt")
    enc("deflate-gzip-data", gzipped, b"deflate")
    enc("gzip-raw-data", deflate_raw, b"gzip")
    enc("gzip-plain-data", body, b"gzip")
    enc("deflate-plain-data", body, b"deflate")
    enc("unknown-compress", gzipped, b"compress")
    enc("unknown-zstd", gzipped, b"zstd")
    enc("identity-header", body, b"identity")
    enc("stacked-gzip-br", gzipped, b"gzip, br")
    enc("stacked-gzip-gzip", gz(gzipped), b"gzip")
    enc("empty-value", body, b"")
    enc("gzip-empty-payload", b"", b"gzip")
    enc("gzip-empty-member", gz(b""), b"gzip")
    enc("gzip-two-members", gz(body[:500]) + gz(body[500:]), b"gzip")
    enc("gzip-trailing-garbage", gzipped + b"garbage", b"gzip")
    enc("gzip-truncated", gzipped[: len(gzipped) // 2], b"gzip")
    enc("gzip-bad-crc", gzipped[:-8] + b"\0\0\0\0" + gzipped[-4:], b"gzip")
    enc("gzip-chunked", chunk_body(gzipped, [50]), b"gzip", b"chunked")
    enc("deflate-raw-chunked", chunk_body(deflate_raw, [33]), b"deflate", b"chunked")
    enc("gzip-chunked-broken", b"zz\r\n" + gzipped, b"gzip", b"chunked")
    big_gz = gz(big_noisy, 1)
    flipped = bytearray(big_gz)
    flipped[len(big_gz) * 2 // 3] ^= 0x55
    enc("gzip-big", big_gz, b"gzip")
    enc("gzip-big-corrupt-late", bytes(flipped), b"gzip")
    enc("gzip-big-corrupt-early", big_gz[:30] + bytes([big_gz[30] ^ 0xFF]) + big_gz[31:], b"gzip")
    enc("gzip-big-truncated", big_gz[: len(big_gz) * 3 // 4], b"gzip")
    big_raw = raw_deflate(big_noisy, 1)
    enc("deflate-raw-big", big_raw, b"deflate")
    flipped_raw = bytearray(big_raw)
    flipped_raw[len(big_raw) * 2 // 3] ^= 0x55
    enc("deflate-raw-big-corrupt-late", bytes(flipped_raw), b"deflate")
    enc("gzip-big-chunked", chunk_body(big_gz, [5000, 17]), b"gzip", b"chunked")
    if br is not None:
        brd = br.compress(body)  # type: ignore[attr-defined]
        enc("br", brd, b"br", None, "br")
        enc("BR-upper", brd, b"BR", None, "br")
        enc("br-chunked", chunk_body(brd, [40]), b"br", b"chunked", "br")
        enc("br-truncated", brd[: len(brd) // 2], b"br", None, "br")
        enc("br-garbage", b"\xff" * 40 + brd, b"br", None, "br")
        enc("br-trailing", brd + b"tail", b"br", None, "br")
        enc("br-plain-data", body, b"br", None, "br")
        big_br = br.compress(big_noisy, quality=1)  # type: ignore[attr-defined]
        enc("br-big", big_br, b"br", None, "br")
        bad = bytearray(big_br)
        bad[len(big_br) * 2 // 3] ^= 0x55
        enc("br-big-corrupt-late", bytes(bad), b"br", None, "br")


def types(b: Builder, rng: random.Random) -> None:
    body = html_page(rng).encode()
    recs = [
        warcinfo(rng),
        request(rng),
        response(rng, body),
        warc_record(
            b"revisit",
            http_response(b"", length_header=False),
            rng=rng,
            extra=[(b"WARC-Profile", b"http://netpreserve.org/warc/1.0/revisit/identical-payload-digest")],
        ),
        warc_record(b"revisit", b"", rng=rng),
        warc_record(b"resource", body, rng=rng, content_type=b"text/html", payload_type=b"text/html"),
        warc_record(b"resource", body, rng=rng, uri=b"urn:x:resource", content_type=b"text/html"),
        metadata(rng),
        warc_record(b"conversion", b"Plain text of a page.\n", rng=rng, payload_type=None),
        warc_record(b"conversion", b"Plain text, typed.\n", rng=rng, payload_type=b"text/plain"),
        warc_record(b"conversion", b"<p>html conversion</p>", rng=rng, payload_type=b"text/html"),
        warc_record(
            b"continuation",
            b"second segment",
            rng=rng,
            extra=[
                (b"WARC-Segment-Number", b"2"),
                (b"WARC-Segment-Origin-ID", b"<urn:uuid:00000000-0000-4000-8000-000000000000>"),
                (b"WARC-Segment-Total-Length", b"40"),
            ],
        ),
        warc_record(b"unknown-type", b"???", rng=rng),
        warc_record(b"RESPONSE", http_response(body), rng=rng, payload_type=b"text/html"),
        warc_record(b"response", b"", rng=rng, payload_type=b"text/html", length=b"0"),
        warc_record(
            b"response", http_response(body), rng=rng, uri=b"dns:example.com", payload_type=b"text/html"
        ),
    ]
    b.both("type/all", recs)
    for i, r in enumerate(recs):
        b.both(f"type/one-{i:02d}", [r])


def _digest(algo: str, data: bytes, enc: str) -> str:
    raw = hashlib.new(algo.split("~")[0], data).digest()
    if enc == "b32":
        return base64.b32encode(raw).decode()
    if enc == "b16":
        return base64.b16encode(raw).decode()
    if enc == "b16-lower":
        return base64.b16encode(raw).decode().lower()
    if enc == "b64":
        return base64.b64encode(raw).decode()
    if enc == "b64-url":
        return base64.urlsafe_b64encode(raw).decode()
    raise ValueError(enc)


def digests(b: Builder, rng: random.Random) -> None:
    def rec(
        block_d: str | None,
        payload_d: str | None,
        *,
        rec_type: bytes = b"response",
        body: bytes | None = None,
        http: bool = True,
        extra: Sequence[Header] = (),
    ) -> bytes:
        payload = body if body is not None else html_page(rng).encode()
        block = http_response(payload) if http else payload
        ex: list[Header] = list(extra)
        if block_d is not None:
            ex.append((b"WARC-Block-Digest", block_d.encode()))
        if payload_d is not None:
            ex.append((b"WARC-Payload-Digest", payload_d.encode()))
        return warc_record(rec_type, block, rng=rng, payload_type=b"text/html", extra=ex)

    def good(algo: str, enc: str, spelling: str | None = None, rec_type: bytes = b"response") -> bytes:
        payload = html_page(rng).encode()
        block = http_response(payload)
        name = spelling or algo
        return warc_record(
            rec_type,
            block,
            rng=rng,
            payload_type=b"text/html",
            extra=[
                (b"WARC-Block-Digest", f"{name}:{_digest(algo, block, enc)}".encode()),
                (b"WARC-Payload-Digest", f"{name}:{_digest(algo, payload, enc)}".encode()),
            ],
        )

    for algo in ("sha1", "sha256", "md5", "sha224", "sha384", "sha512"):
        for enc in ("b32", "b16", "b16-lower", "b64", "b64-url"):
            b.both(f"digest/{algo}-{enc}", [good(algo, enc), metadata(rng)], "digest")
    # base64 with "-" or "_": find a payload whose sha1 has them.
    for i in range(200):
        payload = b"<p>%d</p>" % i
        v = base64.urlsafe_b64encode(hashlib.sha1(payload).digest()).decode()
        if "-" in v or "_" in v:
            block = http_response(payload)
            r = warc_record(
                b"response",
                block,
                rng=rng,
                extra=[
                    (b"WARC-Payload-Digest", f"sha1:{v}".encode()),
                    (
                        b"WARC-Block-Digest",
                        f"sha1:{base64.urlsafe_b64encode(hashlib.sha1(block).digest()).decode()}".encode(),
                    ),
                ],
            )
            b.both("digest/b64-url-alphabet", [r], "digest")
            break
    # hashlib spellings: these name one of the six algorithms pyroid ports.
    for spelling, algo in {
        "SHA1": "sha1",
        "sha-1": "sha1",
        "SHA-1": "sha1",
        "Sha1": "sha1",
        "SHA256": "sha256",
        "sha-256": "sha256",
        "SHA2-256": "sha256",
        "sha2-512": "sha512",
        "MD5": "md5",
        "2.16.840.1.101.3.4.2.1": "sha256",
        "1.3.14.3.2.26": "sha1",
    }.items():
        b.both(f"digest/spelling-{spelling}", [good(algo, "b32", spelling)], "digest")
    # Algorithms hashlib knows here but pyroid does not port: documented to be "unknown hash algorithm".
    for algo in ("sha512_256", "sha3_256", "blake2b", "blake2s", "md5-sha1", "ripemd160", "sm3", "shake_128"):
        try:
            hashlib.new(algo)
        except ValueError:
            continue
        if algo.startswith("shake"):
            r = rec(f"{algo}:AAAA", None)
        else:
            r = good(algo, "b32")
        b.both(f"digest/other-{algo}", [r, metadata(rng)], "digest", "digest-other")
    good_rec = good("sha1", "b32")
    bad_block = re.sub(
        rb"(WARC-Block-Digest: sha1:)(.)",
        lambda m: m.group(1) + (b"B" if m.group(2) != b"B" else b"C"),
        good_rec,
    )
    bad_payload = re.sub(
        rb"(WARC-Payload-Digest: sha1:)(.)",
        lambda m: m.group(1) + (b"B" if m.group(2) != b"B" else b"C"),
        good_rec,
    )
    b.both("digest/wrong-block", [bad_block, metadata(rng)], "digest")
    b.both("digest/wrong-payload", [bad_payload, metadata(rng)], "digest")
    b.both(
        "digest/wrong-both",
        [
            re.sub(
                rb"(Digest: sha1:)(.)",
                lambda m: m.group(1) + (b"B" if m.group(2) != b"B" else b"C"),
                good_rec,
            )
        ],
        "digest",
    )
    for name, (bd, pd) in {
        "unknown-algo-block": ("foo:AAAA", None),
        "unknown-algo-payload": (None, "foo:AAAA"),
        "no-colon-block": ("sha1AAAA", None),
        "no-colon-payload": (None, "sha1AAAA"),
        "empty-value": ("sha1:", "sha1:"),
        "long-non-hex": ("sha1:" + "Z" * 40, None),
        "long-hex": ("sha1:" + "A" * 40, None),
        "short-bad-padding": ("sha1:AAA", None),
        "short-b64": ("sha1:AAAA", None),
        "empty-algo": (":AAAA", None),
        "lowercase-b32": ("sha1:" + "a" * 32, None),
        "unicode-value": ("sha1:" + "Ä" * 32, None),
    }.items():
        b.both(f"digest/{name}", [rec(bd, pd), metadata(rng)], "digest")
    payload = b"<p>zero</p>"
    zero_payload = warc_record(
        b"response",
        http_response(b"", length_header=False),
        rng=rng,
        extra=[(b"WARC-Payload-Digest", f"sha1:{_digest('sha1', b'', 'b32')}".encode())],
    )
    b.both("digest/empty-payload", [zero_payload], "digest")
    b.both(
        "digest/empty-payload-wrong",
        [
            warc_record(
                b"response",
                http_response(b"", length_header=False),
                rng=rng,
                extra=[(b"WARC-Payload-Digest", b"sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")],
            )
        ],
        "digest",
    )
    b.both(
        "digest/zero-length",
        [
            warc_record(
                b"resource",
                b"",
                rng=rng,
                extra=[(b"WARC-Block-Digest", f"sha1:{_digest('sha1', b'', 'b32')}".encode())],
            )
        ],
        "digest",
    )
    b.both("digest/revisit", [good("sha1", "b32", rec_type=b"revisit")], "digest")
    b.both(
        "digest/resource",
        [
            warc_record(
                b"resource",
                payload,
                rng=rng,
                extra=[
                    (b"WARC-Block-Digest", f"sha1:{_digest('sha1', payload, 'b32')}".encode()),
                    (b"WARC-Payload-Digest", f"sha1:{_digest('sha1', payload, 'b32')}".encode()),
                ],
            )
        ],
        "digest",
    )
    b.both(
        "digest/segmented",
        [
            warc_record(
                b"response",
                http_response(payload),
                rng=rng,
                extra=[
                    (b"WARC-Segment-Number", b"1"),
                    (b"WARC-Payload-Digest", b"sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
                    (b"WARC-Block-Digest", f"sha1:{_digest('sha1', http_response(payload), 'b32')}".encode()),
                ],
            )
        ],
        "digest",
    )
    chunked_payload = chunk_body(payload * 20, [37])
    cblock = http_response(chunked_payload, headers=[(b"Transfer-Encoding", b"chunked")], length_header=False)
    b.both(
        "digest/chunked",
        [
            warc_record(
                b"response",
                cblock,
                rng=rng,
                extra=[
                    (b"WARC-Payload-Digest", f"sha1:{_digest('sha1', chunked_payload, 'b32')}".encode()),
                    (b"WARC-Block-Digest", f"sha1:{_digest('sha1', cblock, 'b32')}".encode()),
                ],
            )
        ],
        "digest",
    )
    b.both("digest/wrong-then-good", [bad_block, good("sha1", "b32"), good("sha256", "b16")], "digest")


def _arc_line(url: bytes, ip: bytes, date: bytes, ctype: bytes, length: int | bytes) -> bytes:
    n = length if isinstance(length, bytes) else str(length).encode()
    return b" ".join([url, ip, date, ctype, n]) + b"\n"


def arc(b: Builder, rng: random.Random) -> None:
    def filedesc(name: bytes = b"synth.arc", date: bytes = b"20140101000000") -> bytes:
        body = b"1 0 Alexa Internet\nURL IP-address Archive-date Content-type Archive-length\n\n"
        return _arc_line(b"filedesc://" + name, b"0.0.0.0", date, b"text/plain", len(body)) + body + b"\n"

    def arc_rec(
        url: bytes,
        body: bytes | None = None,
        *,
        date: bytes = b"20140102030405",
        length: int | bytes | None = None,
        ctype: bytes = b"text/html",
        ip: bytes = b"93.184.216.34",
    ) -> bytes:
        block = http_response(
            body if body is not None else html_page(rng).encode(), status=b"HTTP/1.0 200 OK"
        )
        return _arc_line(url, ip, date, ctype, len(block) if length is None else length) + block + b"\n"

    recs = [
        filedesc(),
        arc_rec(b"http://example.com/"),
        arc_rec(b"http://example.com/two"),
        arc_rec(b"dns:example.com", b"93.184.216.34"),
    ]
    b.add("arc/v1", b"".join(recs), ".arc", "arc")
    b.add("arc/v1/gz", per_record(recs), ".arc.gz", "arc")
    b.add("arc/no-filedesc", b"".join(recs[1:]), ".arc", "arc")
    b.add("arc/no-filedesc/gz", per_record(recs[1:]), ".arc.gz", "arc")
    b.add("arc/single-stream", gz(b"".join(recs)), ".arc.gz", "arc")
    for name, r in {
        "url-spaces": arc_rec(b"http://example.com/a b c"),
        "url-latin1": arc_rec(b"http://example.com/caf\xe9"),
        "url-utf8": arc_rec("http://example.com/été/中".encode()),
        "date-12": arc_rec(b"http://example.com/", date=b"201401020304"),
        "date-8": arc_rec(b"http://example.com/", date=b"20140102"),
        "date-4": arc_rec(b"http://example.com/", date=b"2014"),
        "date-16": arc_rec(b"http://example.com/", date=b"2014010203040599"),
        "date-letters": arc_rec(b"http://example.com/", date=b"2014ab02030405"),
        "date-month-13": arc_rec(b"http://example.com/", date=b"20141302030405"),
        "date-arabic": arc_rec(b"http://example.com/", date="٢٠١٤".encode() + b"0102030405"),
        "length-short": arc_rec(b"http://example.com/", length=10),
        "length-long": arc_rec(b"http://example.com/", length=100_000),
        "length-text": arc_rec(b"http://example.com/", length=b"abc"),
        "length-negative": arc_rec(b"http://example.com/", length=b"-3"),
        "no-http": _arc_line(b"http://example.com/raw", b"1.2.3.4", b"20140102030405", b"text/plain", 5)
        + b"hello\n",
        "four-fields": b"http://example.com/ 1.2.3.4 20140102030405 5\nhello\n",
        "v2-line": b"http://example.com/ 1.2.3.4 20140102030405 text/html 200 ABCDEF - - 0 file.arc 5\nhello\n",
        "crlf-line": _arc_line(b"http://example.com/", b"1.2.3.4", b"20140102030405", b"text/html", 5)[:-1]
        + b"\r\nhello\n",
        "no-trailing-newline": _arc_line(
            b"http://example.com/", b"1.2.3.4", b"20140102030405", b"text/html", 5
        )
        + b"hello",
        "ctype-with-params": arc_rec(b"http://example.com/", ctype=b"text/html;charset=utf-8"),
    }.items():
        b.add(f"arc/{name}", recs[0] + r + recs[1], ".arc", "arc")
        b.add(f"arc/{name}/gz", per_record([recs[0], r, recs[1]]), ".arc.gz", "arc")
    b.add("arc/filedesc-only", recs[0], ".arc", "arc")
    b.add("arc/filedesc-utf8-name", filedesc("né".encode()) + recs[1], ".arc", "arc")
    w = base_records(rng, 1)
    b.add("arc/arc-then-warc", b"".join(recs[:2]) + b"".join(w), ".arc", "arc")
    b.add("arc/arc-then-warc/gz", per_record(recs[:2] + w), ".arc.gz", "arc")
    b.add("arc/warc-then-arc", b"".join(w) + b"".join(recs[1:3]), ".warc", "arc")
    b.add("arc/warc-then-arc/gz", per_record(w + recs[1:3]), ".warc.gz", "arc")


CHARSETS = (
    ("utf-8", "Café crème brûlée, naïve façade — “quotes” €"),
    ("utf-8-sig", "Byte order mark été"),
    ("cp1252", "Café “smart” — quotes … € éèêë"),
    ("latin-1", "Grüße aus München, ça va très bien, ñandú"),
    ("iso-8859-2", "Příliš žluťoučký kůň úpěl ďábelské ódy"),
    ("cp1251", "Съешь же ещё этих мягких французских булок"),
    ("koi8-r", "В чащах юга жил бы цитрус? Да, но фальшивый экземпляр!"),
    ("iso-8859-7", "Ξεσκεπάζω την ψυχοφθόρα βδελυγμία"),
    ("shift_jis", "いろはにほへとちりぬるを 日本語の文章です"),
    ("euc-jp", "日本語のウェブページ、文字コードは EUC-JP"),
    ("gbk", "中文网页，简体字编码，我们的网站"),
    ("big5", "繁體中文網頁，台灣的網站"),
    ("euc-kr", "한국어 웹 페이지입니다. 다람쥐 헌 쳇바퀴에"),
    ("cp1256", "هذه صفحة عربية على الإنترنت"),
    ("cp874", "ภาษาไทยบนเว็บ"),
    ("utf-16", "UTF-16 with a byte order mark"),
    ("utf-16-le", "UTF-16 little endian, no BOM"),
)


def charsets(b: Builder, rng: random.Random) -> None:
    for enc, text in CHARSETS:
        page = html_page(rng, 3, f"<h1>{text}</h1>\n<p>{text} {text}</p>\n").encode(enc)
        r = response(rng, page, http_headers=[(b"Content-Type", b"text/html")])
        b.both(f"charset/{enc}", [r, metadata(rng)], "charset")
    for i in range(6):
        raw = rng.randbytes(rng.choice([20, 200, 2000]))
        b.both(f"charset/random-{i}", [response(rng, raw), metadata(rng)], "charset")
    b.both(
        "charset/invalid-utf8-mostly-ascii",
        [response(rng, html_page(rng).encode() + b"\xff\xfe\xfd")],
        "charset",
    )
    b.both("charset/lone-continuation", [response(rng, b"<p>a\x80b</p>")], "charset")
    b.both("charset/surrogate-utf8", [response(rng, b"<p>\xed\xa0\x80</p>")], "charset")
    b.both("charset/overlong-utf8", [response(rng, b"<p>\xc0\xaf</p>")], "charset")
    b.both("charset/empty-body", [response(rng, b"")], "charset")
    b.both("charset/nul-body", [response(rng, b"\0" * 100)], "charset")
    for pt in (
        b"text/html",
        b"application/xhtml+xml",
        b"text/plain",
        b"TEXT/HTML",
        b"text/html; charset=utf-8",
        b"application/pdf",
        b"",
    ):
        b.both(
            f"charset/payload-type-{pt.decode().replace('/', '_').replace(';', '').replace(' ', '') or 'empty'}",
            [response(rng, payload_type=pt)],
            "charset",
        )
    # No WARC-Identified-Payload-Type: datatrove asks libmagic.
    for name, body in {
        "html": html_page(rng).encode(),
        "xhtml": b'<?xml version="1.0" encoding="UTF-8"?>\n<html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title></head><body><p>x</p></body></html>\n',
        "text": b"Just some plain text, several words long.\n" * 5,
        "json": b'{"a": 1, "b": [1, 2, 3]}\n',
        "pdf": b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n",
        "png": b"\x89PNG\r\n\x1a\n" + b"\0" * 40,
        "gzip": gz(b"compressed"),
        "empty": b"",
        "latin1-html": html_page(rng, 2, "<p>café crème</p>").encode("latin-1"),
        "html-fragment": b"<div><p>fragment</p></div>",
    }.items():
        r = response(rng, body, payload_type=None)
        b.both(f"charset/magic-{name}", [r, metadata(rng)], "charset", "magic")
        conv = warc_record(b"conversion", body, rng=rng, payload_type=None)
        b.both(f"charset/magic-conversion-{name}", [conv], "charset", "magic")
    b.both(
        "charset/no-uri-fallback",
        [warc_record(b"response", http_response(b"<p>x</p>"), rng=rng, uri=b"", payload_type=b"text/html")],
        "charset",
    )
    b.both("charset/no-date-fallback", [response(rng, date=None)], "charset")
    b.both("charset/no-record-id", [response(rng, rid=None)], "charset")
    b.both(
        "charset/zero-length-response",
        [warc_record(b"response", b"", rng=rng, uri=None, payload_type=b"text/html", length=b"0")],
        "charset",
    )


def big(b: Builder, rng: random.Random) -> None:
    pages = []
    for i in range(40):
        body = "".join(html_page(rng, 20) for _ in range(rng.choice([1, 5, 30]))).encode()
        if i % 7 == 3:
            body += base64.b64encode(rng.randbytes(150_000))
        pages.append(response(rng, body, uri=b"http://example.com/big/%d" % i))
    recs = [warcinfo(rng), *pages]
    zipped = per_record(recs)
    plain = b"".join(recs)
    b.add("big/plain", plain, ".warc", "big")
    b.add("big/gz", zipped, ".warc.gz", "big")
    b.add("big/single-stream", gz(plain, 1), ".warc.gz", "big")
    bounds = member_bounds(zipped)
    # Cuts and damage around CPython's 128 KiB GzipFile reads and warcio's 16 KiB blocks.
    marks = sorted(
        {k * 131072 + d for k in (1, 2, 3, 5) for d in (-1, 0, 1, 4000)}
        | {e for _, e in bounds[5:8]}
        | {e - 4 for _, e in bounds[5:8]}
    )
    for m in marks:
        if 0 < m < len(zipped):
            b.add(f"big/trunc-at-{m}", zipped[:m], ".warc.gz", "big")
    for m in marks[::3]:
        if 0 < m < len(zipped):
            bad = bytearray(zipped)
            bad[m] ^= 0x10
            b.add(f"big/flip-at-{m}", bytes(bad), ".warc.gz", "big")
    _, e = bounds[-1]
    bad = bytearray(zipped)
    bad[e - 6] ^= 0x01
    b.add("big/bad-crc-last", bytes(bad), ".warc.gz", "big")
    b.add("big/zero-pad-tail", zipped + b"\0" * 1000, ".warc.gz", "big")
    b.add("big/garbage-tail", zipped + b"tail garbage", ".warc.gz", "big")
    b.add(
        "big/garbage-middle", zipped[: bounds[20][0]] + b"junk" + zipped[bounds[20][0] :], ".warc.gz", "big"
    )
    for m in (100_000, 300_000, 700_000):
        if m < len(plain):
            b.add(f"big/plain-trunc-at-{m}", plain[:m], ".warc", "big")


FAMILIES: dict[str, Callable[[Builder, random.Random], None]] = {
    "frame": frame,
    "trunc": trunc_flip,
    "length": lengths,
    "header": headers,
    "http": http,
    "chunked": chunked,
    "encoding": encodings,
    "type": types,
    "digest": digests,
    "arc": arc,
    "charset": charsets,
    "big": big,
}


@cache
def cases(seed: int = SEED) -> tuple[Case, ...]:
    """Every synthetic case, in a fixed order (each family has its own RNG stream)."""
    b = Builder()
    for i, (_name, fn) in enumerate(FAMILIES.items()):
        fn(b, random.Random(seed * 1000 + i))
    return tuple(b.cases)


def inputs_sha256(cs: Sequence[Case]) -> str:
    """sha256 over ``name\\0suffix\\0data`` of each case: the generator's output, pinned with the references."""
    h = hashlib.sha256()
    for c in cs:
        h.update(c.name.encode() + b"\0" + c.suffix.encode() + b"\0")
        h.update(hashlib.sha256(c.data).digest())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.warc_synth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    w = sub.add_parser("write")
    w.add_argument("dir")
    w.add_argument("--only", default="", help="regular expression on case names")
    a = ap.parse_args(argv)
    cs = cases()
    if a.cmd == "list":
        for c in cs:
            print(f"{c.name:60} {len(c.data):9,} {c.suffix:8} {','.join(sorted(c.tags))}")
        fams: dict[str, int] = {}
        for c in cs:
            fams[c.name.split("/")[0]] = fams.get(c.name.split("/")[0], 0) + 1
        print(
            f"{len(cs)} cases, {sum(len(c.data) for c in cs) / 1e6:.1f} MB: {fams}; sha256 {inputs_sha256(cs)}"
        )
        return 0
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    rx = re.compile(a.only)
    n = 0
    for c in cs:
        if rx.search(c.name):
            (out / c.filename).write_bytes(c.data)
            n += 1
    print(f"wrote {n} files to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
