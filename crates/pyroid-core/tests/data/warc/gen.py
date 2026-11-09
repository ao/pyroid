"""Write the synthetic fixtures in synth/ and expected.json.gz: what warcio 1.8.2 gives for
each fixture, with several ArchiveIterator options and read plans (tests/warc_oracle.rs).

Run in the dev container with warcio 1.8.2 installed (CPython 3.13):
    /target/venv313/bin/python -I crates/pyroid-core/tests/data/warc/gen.py
"""

import gzip
import importlib.metadata
import hashlib
import io
import json
import logging
import os
import sys
import zlib

from warcio.archiveiterator import ArchiveIterator, ARCIterator, WARCIterator

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.join(HERE, "synth")


# --- synthetic fixtures ----------------------------------------------------------------


def warc_record(rec_type, uri, block, extra=(), length=None, version=b"WARC/1.0"):
    h = [version, b"WARC-Type: " + rec_type]
    if uri is not None:
        h.append(b"WARC-Target-URI: " + uri)
    h.append(b"WARC-Record-ID: <urn:uuid:%s>" % hashlib.md5(block + rec_type).hexdigest().encode())
    h.append(b"WARC-Date: 2026-01-01T00:00:00Z")
    h.extend(extra)
    h.append(b"Content-Length: " + (str(len(block)).encode() if length is None else length))
    return b"\r\n".join(h) + b"\r\n\r\n" + block + b"\r\n\r\n"


def http_response(body, headers=(), status=b"HTTP/1.1 200 OK"):
    return status + b"\r\n" + b"".join(h + b"\r\n" for h in headers) + b"\r\n" + body


def gz(b, level=6):
    return gzip.compress(b, level, mtime=0)


def raw_deflate(b):
    c = zlib.compressobj(6, zlib.DEFLATED, -15)
    return c.compress(b) + c.flush()


def chunked(b, sizes, ext=b""):
    out, i = b"", 0
    for s in sizes:
        part = b[i : i + s]
        i += s
        out += b"%x%s\r\n" % (len(part), ext) + part + b"\r\n"
    return out + b"0\r\n\r\n"


HTML = b"<html><head><title>t</title></head><body>" + b"caf\xc3\xa9 " * 300 + b"</body></html>"


def synth():
    os.makedirs(SYNTH, exist_ok=True)
    files = {}
    resp = lambda body, hdrs=(), **kw: warc_record(  # noqa: E731
        b"response", b"http://example.com/", http_response(body, hdrs), **kw
    )
    enc = [
        resp(gz(HTML), [b"Content-Encoding: gzip"]),
        resp(gz(HTML), [b"Content-Encoding: X-GZIP"]),
        resp(zlib.compress(HTML), [b"Content-Encoding: deflate"]),
        resp(raw_deflate(HTML), [b"Content-Encoding: deflate"]),
        resp(raw_deflate(HTML), [b"Content-Encoding: deflate_alt"]),
        resp(b"\x1b\x00\x00junk", [b"Content-Encoding: br"]),
        resp(b"not gzip at all", [b"Content-Encoding: gzip"]),
        resp(gz(HTML) + b"trailing", [b"Content-Encoding: gzip"]),
        resp(gz(HTML)[:-20] + b"x" * 20, [b"Content-Encoding: gzip"]),
        resp(chunked(HTML, [10, 100, 1000, 5000]), [b"Transfer-Encoding: chunked"]),
        resp(chunked(gz(HTML), [7, 50, 4000]), [b"Transfer-Encoding: chunked", b"Content-Encoding: gzip"]),
        resp(chunked(HTML, [10, 20], b";ext=1"), [b"Transfer-Encoding: chunked"]),
        resp(chunked(HTML, [10, 20])[:-9] + b"zz", [b"Transfer-Encoding: chunked"]),
        resp(b"5\r\nabcde\r\n3\r\nxy", [b"Transfer-Encoding: chunked"]),
        resp(b"5\r\nabcdeXX3\r\nxyz\r\n0\r\n\r\n", [b"Transfer-Encoding: chunked"]),
        resp(b"0x5\r\nabcde\r\n -0 \r\n\r\n", [b"Transfer-Encoding: chunked"]),
        resp(b"-5\r\nabcde\r\n0\r\nxx", [b"Transfer-Encoding: chunked"]),
        resp(HTML, [b"Transfer-Encoding: Chunked"]),
        resp(b"\r\n" + HTML, [b"Transfer-Encoding: chunked"]),
        resp(HTML, [b"Content-Encoding:", b"Transfer-Encoding: chunked"]),
    ]
    files["encodings.warc"] = b"".join(enc)
    files["encodings.warc.gz"] = b"".join(gz(r) for r in enc)

    lengths = []
    for cl in [b" 12 ", b"+12", b"1_2", "١٢".encode(), b"-5", b"abc", b"0", b"99999999999999999999999", b"1" * 5000, b"12\x1c"]:
        lengths.append(warc_record(b"resource", b"urn:x", b"0123456789ab" * 3, length=cl))
    lengths.append(warc_record(b"metadata", b"urn:y", b"hello"))
    files["lengths.warc"] = b"".join(lengths)
    files["no-length.warc"] = (
        warc_record(b"warcinfo", None, b"info")
        + b"WARC/1.0\r\nWARC-Type: resource\r\nWARC-Target-URI: urn:z\r\n\r\nall the rest\r\n\r\n"
        + warc_record(b"resource", b"urn:w", b"never seen")
    )
    files["no-length.warc.gz"] = gz(
        b"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: http://a/\r\n\r\nHTTP/1.0 200 OK\r\n\r\nbody\r\n\r\n"
    ) + gz(warc_record(b"resource", b"urn:w", b"seen"))

    hdr = [
        warc_record(
            b"response",
            b"<http://example.com/a b>",
            http_response(b"x", [b"X-Long: one", b"  two", b"\tthree", b"NoColon", b" cont", b"Lat: caf\xe9", b"Utf: caf\xc3\xa9"]),
            extra=[b"WARC-Target-URI: <http://example.com/second>", b"Multi: a", b"   b", "WARC-BlocK-Digest: sha1:x".encode()],
        ),
        warc_record(b"response", b"http://example.com/", http_response(b"y"), version=b"warc/1.1"),
        warc_record(b"request", b"https://example.com/", b"POST /x HTTP/1.1\r\nHost: e\r\n\r\nbody"),
        warc_record(b"request", b"https://example.com/", "POﬆ /x HTTP/1.1\r\n\r\n".encode()),
        warc_record(b"revisit", b"http://example.com/", b""),
        warc_record(b"response", b"ftp://example.com/", b"ftp data"),
        warc_record(b"response", b"http://example.com/", b"\r\nnot headers"),
        warc_record(b"response", b"http://example.com/", b"   \r\nHTTP/1.1 200\r\n\r\n"),
    ]
    files["headers.warc"] = b"".join(hdr)
    files["missing-uri.warc"] = warc_record(b"warcinfo", None, b"i") + warc_record(b"response", None, http_response(b"z"))
    files["revisit-eof.warc"] = warc_record(b"warcinfo", None, b"i") + (
        b"WARC/1.0\r\nWARC-Type: revisit\r\nWARC-Target-URI: http://a/\r\nContent-Length: 50\r\n\r\n"
    )
    files["verify-http.warc"] = warc_record(b"response", b"http://example.com/", b"NOTHTTP 200\r\n\r\n")
    files["blank-start.warc"] = b"\r\n" + warc_record(b"resource", b"urn:x", b"data")
    files["garbage.bin"] = bytes(range(256)) * 3
    files["garbage-line.warc"] = b"some text here \x00\x7f\xc3\xa9\xe2\x80\x8b and 'quotes'\r\nmore\r\n"
    files["after-garbage.warc"] = warc_record(b"resource", b"urn:x", b"data") + b"GARBAGE LINE\r\n"
    files["garbage-after.warc.gz"] = gz(warc_record(b"resource", b"urn:x", b"data")) + b"NOT GZIP\r\nWARC/1.0\r\n"
    members = [gz(warc_record(b"resource", b"urn:%d" % i, HTML[: 100 * i + 1])) for i in range(5)]
    files["empty-member.warc.gz"] = members[0] + gz(b"") + members[1] + members[2]
    corrupt = bytearray(gz(warc_record(b"resource", b"urn:big", bytes(range(256)) * 400)))
    corrupt[200] ^= 0xFF
    files["corrupt-member.warc.gz"] = members[0] + bytes(corrupt) + members[3]
    big = gz(warc_record(b"resource", b"urn:big", bytes(range(256)) * 400))
    files["truncated.warc.gz"] = members[0] + members[1] + big[: len(big) // 2]
    files["crc.warc.gz"] = members[0] + big[:-8] + b"\0\0\0\0" + big[-4:] + members[1]
    files["two-in-one.warc.gz"] = gz(warc_record(b"resource", b"urn:a", b"a") + warc_record(b"resource", b"urn:b", b"b"))

    sha1 = lambda b: hashlib.sha1(b).digest()  # noqa: E731
    import base64

    blk = http_response(b"payload")
    def drec(bd, pd, block=blk, rec_type=b"response"):
        extra = []
        if bd is not None:
            extra.append(b"WARC-Block-Digest: " + bd)
        if pd is not None:
            extra.append(b"WARC-Payload-Digest: " + pd)
        return warc_record(rec_type, b"http://example.com/", block, extra=extra)

    b32 = lambda d: base64.b32encode(d)  # noqa: E731
    b16 = lambda d: base64.b16encode(d)  # noqa: E731
    b64 = lambda d: base64.b64encode(d)  # noqa: E731
    files["digests.warc"] = b"".join(
        [
            drec(b"sha1:" + b32(sha1(blk)), b"sha1:" + b32(sha1(b"payload"))),
            drec(b"sha1:" + b16(sha1(blk)).lower(), b"SHA-1:" + b64(sha1(b"payload"))),
            drec(b"sha256:" + b32(hashlib.sha256(blk).digest()), b"sha2-256:" + b64(hashlib.sha256(b"payload").digest()).replace(b"+", b"-").replace(b"/", b"_")),
            drec(b"md5:" + b16(hashlib.md5(blk).digest()), None),
            drec(b"sha512:" + b64(hashlib.sha512(blk).digest()), b"sha384:" + b32(hashlib.sha384(b"payload").digest())),
            drec(b"sha224:" + b16(hashlib.sha224(blk).digest()), b"sha1:WRONG"),
            drec(b"sha3_256:abc", b"nocolon"),
            drec(b"sha1:" + b32(sha1(blk))[:-1] + b"A", b"sha1:@@@@"),
            drec(b"sha1:" + b"Z" * 41, None),
            drec(b"sha1:" + b"A" * 33, None),
            drec(b"sha1:" + b"QQ==QUJD", None),
            drec(b"sha1:" + "é".encode() * 40, None),
            drec(b"sha1:" + b32(sha1(blk)), b"sha1:" + b32(sha1(b"payload")), rec_type=b"revisit"),
            drec(b"sha1:" + b32(sha1(b"HTTP/1.1 200 OK\r\n\r\n")), b"sha1:" + b32(sha1(b"")), block=b"HTTP/1.1 200 OK\r\n\r\n"),
            drec(b"", b""),
        ]
    )
    arc = (
        b"filedesc://synth.arc 0.0.0.0 20260101000000 text/plain 76\n"
        b"1 0 Synth\nURL IP-address Archive-date Content-type Archive-length\n\n"
        b"\n"
        b"http://example.com/ 1.2.3.4 2026010112 text/html 30\n"
        b"HTTP/1.0 200 OK\r\n\r\n0123456789AB\n"
        b"http://example.com/two 1.2.3.4 20260101123456 text/html 4\n"
        b"abcd\n"
        b"bad line\n"
    )
    files["synth.arc"] = arc

    def raw_record(fields, block):
        h = b"WARC/1.0\r\n" + b"".join(k + b": " + v + b"\r\n" for k, v in fields)
        return h + b"Content-Length: %d\r\n\r\n" % len(block) + block + b"\r\n\r\n"

    def ipt(rec_type, mime, block, uri=b"http://example.com/p", rid=b"<urn:uuid:1>", date=b"2026-01-02T00:00:00Z", extra=()):
        f = [(b"WARC-Type", rec_type)]
        if uri is not None:
            f.append((b"WARC-Target-URI", uri))
        if rid is not None:
            f.append((b"WARC-Record-ID", rid))
        if date is not None:
            f.append((b"WARC-Date", date))
        if mime is not None:
            f.append((b"WARC-Identified-Payload-Type", mime))
        return raw_record(f + list(extra), block)

    page = http_response(HTML, [b"Content-Type: text/html; charset=utf-8"])
    dt = [
        raw_record([(b"WARC-Type", b"warcinfo"), (b"WARC-Record-ID", b"<urn:uuid:0>")], b"software: synth\r\n"),
        ipt(b"request", None, b"GET /p HTTP/1.1\r\nHost: example.com\r\n\r\n"),
        ipt(b"response", b"text/html", page),
        ipt(b"response", b"application/xhtml+xml", http_response(gz(HTML), [b"Content-Encoding: gzip"]), rid=None),
        ipt(b"response", b"text/plain", http_response(b"plain text")),
        ipt(b"conversion", b"text/plain", b"converted text", uri=None, date=None, extra=[(b"uri", b"http://a/"), (b"archive-date", b"2020"), (b"uri", b"http://b/")]),
        ipt(b"response", b"text/html", http_response("café ".encode("latin-1") * 50)),
        ipt(b"response", b"image/png", http_response(b"\x89PNG")),
        ipt(b"response", None, http_response(b"<html>no payload type</html>")),
        ipt(b"metadata", b"text/html", b"meta"),
        ipt(b"response", b"text/html", http_response(chunked(HTML, [100, 2000]), [b"Transfer-Encoding: chunked"]), date=b"", extra=[(b"archive-date", b"1999")]),
        ipt(b"conversion", b"text/html", b"<p>conv</p>", uri=b"", extra=[(b"uri", b"urn:fallback")]),
        ipt(b"response", b"text/html", http_response(b"\xff\xfe\x00bad")),
        ipt(b"resource", b"text/html", b"<html>resource</html>"),
    ]
    files["datatrove.warc"] = b"".join(dt)
    files["datatrove.warc.gz"] = b"".join(gz(r) for r in dt)
    files["datatrove-uri.warc"] = dt[2] + ipt(b"conversion", b"text/html", b"<p>x</p>", uri=None) + dt[2]
    files["datatrove-date.warc"] = dt[2] + ipt(b"conversion", b"text/html", b"<p>x</p>", date=None) + dt[2]
    for name, data in files.items():
        with open(os.path.join(SYNTH, name), "wb") as f:
            f.write(data)
    return sorted("synth/" + n for n in files)


# --- running warcio --------------------------------------------------------------------


class Capture(logging.Handler):
    def __init__(self, events, counter):
        super().__init__()
        self.events, self.counter = events, counter

    def emit(self, record):
        self.events.append(["log", self.counter[0], record.getMessage()])


class Err:
    def __init__(self, events, counter):
        self.events, self.counter = events, counter

    def write(self, s):
        self.events.append(["stderr", self.counter[0], s])

    def flush(self):
        pass


def summary(b):
    if len(b) <= 64:
        return {"hex": b.hex()}
    return {"len": len(b), "sha1": hashlib.sha1(b).hexdigest()}


def sah(h):
    if h is None:
        return None
    return {"protocol": h.protocol, "statusline": h.statusline, "headers": [list(x) for x in h.headers], "total_len": h.total_len}


CHUNKS = [1, 2, 3, 5, 8, 13, 16384]


def run_plan(rec, plan):
    if plan in ("none", "offsets"):
        return None
    if plan == "stale":
        plan = "content"
    if plan == "raw":
        return rec.raw_stream.read()
    s = rec.content_stream()
    if plan == "content":
        return s.read()
    out = b""
    if plan == "chunks":
        i = 0
        while True:
            b = s.read(CHUNKS[i % len(CHUNKS)])
            i += 1
            if not b:
                return out
            out += b
    if plan == "lines":
        while True:
            b = s.readline()
            if not b:
                return out
            out += b
    if plan == "lines7":
        while True:
            b = s.readline(7)
            if not b:
                return out
            out += b
    raise ValueError(plan)


def run(path, opts, plan, cls="ArchiveIterator", offset=0):
    events, counter = [], [0]
    handler = Capture(events, counter)
    log = logging.getLogger("warcio.recordloader")
    log.addHandler(handler)
    log.propagate = False
    old_err, sys.stderr = sys.stderr, Err(events, counter)
    records, error = [], None
    it = prev = None
    try:
        with open(os.path.join(HERE, path), "rb") as fh:
            fh.seek(offset)
            it = {"ArchiveIterator": ArchiveIterator, "WARCIterator": WARCIterator, "ARCIterator": ARCIterator}[cls](fh, **opts)
            for rec in it:
                counter[0] += 1
                r = {
                    "format": rec.format,
                    "rec_type": rec.rec_type,
                    "rec_headers": sah(rec.rec_headers),
                    "http_headers": sah(rec.http_headers),
                    "content_type": rec.content_type,
                    "length": None if rec.length is None else str(rec.length),
                    "payload_length": str(rec.payload_length),
                }
                if plan == "offsets":
                    r["early"] = [it.get_record_offset(), it.get_record_length()]
                if plan == "stale" and prev is not None:
                    # reads of the previous record's streams after the iterator moved on
                    r["stale"] = [summary(prev.raw_stream.read(50)), summary(prev.content_stream().read(20)), prev.raw_stream.tell()]
                prev = rec
                try:
                    data = run_plan(rec, plan)
                    if data is not None:
                        r["content"] = summary(data)
                    if plan != "none":
                        r["offset"] = [it.get_record_offset(), it.get_record_length()]
                finally:
                    r["passed"] = rec.digest_checker.passed
                    r["problems"] = list(rec.digest_checker.problems)
                    records.append(r)
    except Exception as e:  # noqa: BLE001
        error = [type(e).__name__, str(e)]
    finally:
        sys.stderr = old_err
        log.removeHandler(handler)
    return {
        "records": records,
        "events": events,
        "error": error,
        "err_count": it.err_count if it is not None else 0,
        "final_offset": it.offset if it is not None else 0,
    }


def run_datatrove(path):
    """datatrove 0.10.1's WarcReader.read_file loop over open(path, "rb").

    libmagic is not in the dev image: magic.from_buffer answers a type datatrove drops, so
    records without WARC-Identified-Payload-Type are skipped (the Rust test leaves those to
    the bindings).
    """
    import types

    sys.modules["magic"] = types.SimpleNamespace(from_buffer=lambda b, mime=True: "application/x-unknown")
    from datatrove.pipeline.readers.warc import process_record

    old_err, sys.stderr = sys.stderr, io.StringIO()
    docs, error = [], None
    try:
        with open(os.path.join(HERE, path), "rb") as fh:
            for ri, record in enumerate(ArchiveIterator(fh)):
                d = process_record(record)
                if d:
                    text = d["text"].encode("utf-8", "surrogatepass")
                    docs.append([ri, d["id"], d["url"], d["date"], hashlib.sha1(text).hexdigest()])
    except Exception as e:  # noqa: BLE001
        error = [type(e).__name__, str(e)]
    finally:
        sys.stderr = old_err
    return {"docs": docs, "error": error}


def main():
    fixtures = sorted(f for f in os.listdir(HERE) if f.startswith(("example", "bad", "post")))
    fixtures += synth()
    cases = []
    plans = ["none", "content", "chunks", "lines", "lines7", "raw", "offsets", "stale"]
    for f in fixtures:
        for plan in plans:
            cases.append({"file": f, "opts": {}, "plan": plan})
        for bs in [1, 7, 100, 1000]:
            cases.append({"file": f, "opts": {"block_size": bs}, "plan": "chunks"})
        for cd in [True, "log", "raise"]:
            cases.append({"file": f, "opts": {"check_digests": cd}, "plan": "content"})
            cases.append({"file": f, "opts": {"check_digests": cd}, "plan": "none"})
        cases.append({"file": f, "opts": {"ensure_http_headers": True}, "plan": "content"})
        cases.append({"file": f, "opts": {"no_record_parse": True}, "plan": "content"})
        cases.append({"file": f, "opts": {"verify_http": True}, "plan": "content"})
        if "arc" in f:
            cases.append({"file": f, "opts": {"arc2warc": True}, "plan": "content"})
            cases.append({"file": f, "opts": {}, "plan": "content", "cls": "ARCIterator"})
        cases.append({"file": f, "opts": {}, "plan": "content", "cls": "WARCIterator"})
    cases.append({"file": "example.warc.gz", "opts": {}, "plan": "content", "offset": 10})
    cases.append({"file": "example-digest.warc", "opts": {"check_digests": True}, "plan": "none", "offset": 922})
    for c in cases:
        c.update(run(c["file"], c["opts"], c["plan"], c.get("cls", "ArchiveIterator"), c.get("offset", 0)))
    docs = [dict(file=f, **run_datatrove(f)) for f in fixtures]
    out = {
        "warcio": importlib.metadata.version("warcio"),
        "python": sys.version.split()[0],
        "cases": cases,
        "datatrove": docs,
    }
    data = json.dumps(out, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    with open(os.path.join(HERE, "expected.json.gz"), "wb") as f:
        f.write(gzip.compress(data, 9, mtime=0))
    print(len(cases), "cases")


main()
