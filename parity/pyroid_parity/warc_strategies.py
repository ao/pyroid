"""Hypothesis strategies for WARC streams: records from a small grammar, then framing and damage.

A record is a WARC version line, a header list (CC's usual headers, each one possibly dropped, re-cased,
duplicated, continued or given an odd value), and a block (an HTTP response or request with chunked framing
and Content-Encoding variants, or a bare payload). A file is 1-5 records, framed plain, as one gzip member per
record, as one gzip stream or in groups, with separators between records, then optionally truncated, given a
bit flip or garbage at a point. Bytes come from ``pyroid_parity.warc_synth``'s builders.
"""

from __future__ import annotations

import random
import zlib

from hypothesis import strategies as st

from pyroid_parity import warc_synth as S

REC_TYPES = (b"response", b"request", b"revisit", b"resource", b"metadata", b"warcinfo", b"conversion", b"x")
URIS = (
    b"http://example.com/",
    b"https://example.com/a?b=c",
    b"<http://example.com/angle>",
    b"http://example.com/a b",
    b"dns:example.com",
    b"HTTP://EXAMPLE.COM/",
    b"",
    "http://example.com/é".encode(),
    b"http://example.com/\xe9",
)
ODD_VALUES = (b"", b" ", b"\xff", b"\xe9", b" ", "　".encode(), b"\x1c", b"<x>", b"a:b", b"\t")
LENGTH_MODES = ("auto", "short", "long", "missing", "negative", "text", "spaces", "unicode", "huge")
VERSIONS = (b"WARC/1.0", b"WARC/1.1", b"warc/1.0", b"WARC/0.18", b"WARC/2.0", b"HTTP/1.1", b"")
STATUS = (
    b"HTTP/1.1 200 OK",
    b"HTTP/1.0 404 X",
    b"HTTP/2 200",
    b"http/1.1 200",
    b"HTTP/1.1",
    b"",
    b"ICY 200 OK",
)
ENCODINGS = (None, b"gzip", b"x-gzip", b"deflate", b"deflate_alt", b"br", b"compress", b"GZIP")
PAYLOADS = (
    b"",
    b"<html><body><p>hi</p></body></html>",
    "<p>café</p>".encode(),
    "<p>café</p>".encode("latin-1"),
    "<p>日本語</p>".encode("shift_jis"),
    b"\x00\xff" * 10,
)


@st.composite
def payloads(draw: st.DrawFn) -> bytes:
    base = draw(st.sampled_from(PAYLOADS))
    extra = draw(st.binary(max_size=40))
    repeat = draw(st.sampled_from((1, 1, 5, 300)))
    return base * repeat + extra


@st.composite
def http_blocks(draw: st.DrawFn, rng: random.Random) -> bytes:
    body = draw(payloads())
    enc = draw(st.sampled_from(ENCODINGS))
    if enc in (b"gzip", b"x-gzip", b"GZIP"):
        body = S.gz(body)
    elif enc == b"deflate":
        body = draw(st.sampled_from((zlib.compress(body), S.raw_deflate(body))))
    elif enc == b"deflate_alt":
        body = S.raw_deflate(body)
    elif enc == b"br":
        br = S.brotli_module()
        if br is not None:
            body = br.compress(body)  # type: ignore[attr-defined]
    hs: list[S.Header] = [(b"Content-Type", draw(st.sampled_from((b"text/html", b"text/plain", b"x"))))]
    if enc is not None:
        hs.append((b"Content-Encoding", enc))
    chunked = draw(st.booleans())
    if chunked:
        sizes = draw(st.lists(st.integers(1, 300), min_size=1, max_size=4))
        body = S.chunk_body(body, sizes, ext=draw(st.sampled_from((b"", b";x=y"))))
        hs.append((b"Transfer-Encoding", draw(st.sampled_from((b"chunked", b"chunked", b"Chunked")))))
        if draw(st.booleans()):
            cut = draw(st.integers(0, len(body)))
            body = body[:cut] + draw(st.binary(max_size=8))
    if draw(st.booleans()):
        line = draw(
            st.sampled_from((b"GET / HTTP/1.1", b"post /x HTTP/1.0", "poſt / HTTP/1.1".encode(), b"BREW / X"))
        )
        return S.http_request(line, body=body)
    return S.http_response(body, status=draw(st.sampled_from(STATUS)), headers=hs, length_header=not chunked)


@st.composite
def records(draw: st.DrawFn) -> bytes:
    rng = random.Random(draw(st.integers(0, 2**32)))
    rec_type = draw(st.sampled_from((*REC_TYPES, None)))
    block = draw(st.one_of(http_blocks(rng), payloads()))
    mode = draw(st.sampled_from(LENGTH_MODES))
    n = len(block)
    length: object = {
        "auto": S.AUTO,
        "short": str(max(0, n - draw(st.integers(1, 5)))).encode(),
        "long": str(n + draw(st.integers(1, 50))).encode(),
        "missing": None,
        "negative": b"-1",
        "text": b"abc",
        "spaces": f"  {n} ".encode(),
        "unicode": "".join(chr(0x0660 + int(d)) for d in str(n)).encode(),
        "huge": b"9" * 25,
    }[mode]
    extra: list[S.Header] = []
    for _ in range(draw(st.integers(0, 3))):
        name = draw(
            st.sampled_from(
                (b"X-A", b"warc-type", b"Content-Length", b"WARC-Target-URI", b"WARC-Payload-Digest")
            )
        )
        extra.append((name, draw(st.sampled_from(ODD_VALUES))))
    if draw(st.booleans()):
        extra.append(draw(st.sampled_from((b" continued", b"\tcontinued", b"no colon", b": empty name"))))
    payload_type = draw(st.sampled_from((None, b"text/html", b"application/xhtml+xml", b"text/plain", b"x")))
    eol = draw(st.sampled_from((S.CRLF, S.CRLF, b"\n")))
    tail = draw(st.sampled_from((S.CRLF * 2, S.CRLF * 2, b"", S.CRLF, b"\n\n", b"junk\r\n")))
    return S.warc_record(
        rec_type,
        block,
        rng=rng,
        uri=draw(st.sampled_from((*URIS, None))),
        length=length,
        payload_type=payload_type,
        extra=extra,
        version=draw(st.sampled_from(VERSIONS)) if draw(st.integers(0, 9)) == 0 else b"WARC/1.0",
        eol=eol,
        tail=tail,
        rid=S.AUTO if draw(st.integers(0, 9)) else None,
    )


@st.composite
def warc_files(draw: st.DrawFn) -> tuple[bytes, str]:
    """(file bytes, suffix): records, framing, damage."""
    recs = draw(st.lists(records(), min_size=1, max_size=5))
    framing = draw(st.sampled_from(("plain", "per-record", "per-record", "single", "pairs", "mixed")))
    if framing == "plain":
        data, suffix = b"".join(recs), ".warc"
    elif framing == "per-record":
        data, suffix = S.per_record(recs), ".warc.gz"
    elif framing == "single":
        data, suffix = S.gz(b"".join(recs)), ".warc.gz"
    elif framing == "pairs":
        data, suffix = S.grouped(recs, 2), ".warc.gz"
    else:
        k = draw(st.integers(0, len(recs)))
        data, suffix = S.per_record(recs[:k]) + b"".join(recs[k:]), ".warc.gz"
    damage = draw(st.sampled_from(("none", "none", "truncate", "flip", "insert", "zeros")))
    if data and damage != "none":
        pos = draw(st.integers(0, len(data) - 1))
        if damage == "truncate":
            data = data[:pos]
        elif damage == "flip":
            b = bytearray(data)
            b[pos] ^= 1 << draw(st.integers(0, 7))
            data = bytes(b)
        elif damage == "insert":
            data = data[:pos] + draw(st.binary(min_size=1, max_size=16)) + data[pos:]
        else:
            data = data[:pos] + b"\0" * draw(st.integers(1, 600)) + data[pos:]
    return data, suffix
