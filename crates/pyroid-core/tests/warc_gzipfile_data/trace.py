"""Record CPython's gzip.GzipFile and brotlicffi traces for tests/warc_gzipfile*.rs.

gzip (once per Python; CASES is the directory dump_cases wrote):
    python -I trace.py gzip CASES PART.json [DETAIL_DIR]
    python -I trace.py merge gzip.json PART.json...
brotli (brotlicffi 1.2.1.0 installed; KNOWN lists case names, one per line, that the port
cannot match and that are kept out of the group digests):
    python -I trace.py brotli brotli.json [DETAIL_DIR [KNOWN]]

A gzip trace is every read result (length and bytes, or exception type and str()), tell() after
each call and mtime at the end; a brotli trace is every decompress() result, then flush() and
is_finished(). Both are hashed with SHA-1 as the tests do. The fixtures keep one digest per
group of cases; DETAIL_DIR gets the per-case digests (GZIP_TRACE_DETAIL in the tests).
"""

import base64
import collections
import gzip
import hashlib
import json
import os
import random
import sys
import sysconfig

MAX_CALLS = 2_000_000


def short(digest):
    """The fixtures' short digest: the first 8 bytes as a decimal number (no letters for typos)."""
    return str(int.from_bytes(digest[:8], "big"))


class Src:
    """The test's source: whole reads, at most k bytes per read, or OSError after k calls."""

    def __init__(self, data, spec):
        self.data = data
        self.pos = 0
        self.calls = 0
        kind, _, arg = spec.partition(":")
        self.kind = kind
        self.arg = int(arg) if arg else 0

    def read(self, n=-1):
        self.calls += 1
        if self.kind == "fail" and self.calls > self.arg:
            raise OSError("boom")
        if n is None or n < 0:
            n = len(self.data) - self.pos
        k = min(n, self.arg) if self.kind == "short" else n
        b = self.data[self.pos : self.pos + k]
        self.pos += len(b)
        return b


def run_gzip(data, spec, n):
    f = gzip.GzipFile(fileobj=Src(data, spec))
    h = hashlib.sha1()
    excs = empties = calls = nbytes = 0
    exc = collections.Counter()
    for _ in range(MAX_CALLS):
        calls += 1
        try:
            b = f.read() if n < 0 else f.read(n)
        except Exception as e:
            t = type(e).__name__
            h.update(b"E" + t.encode() + b"\0" + str(e).encode() + b"\0")
            exc[t] += 1
            excs += 1
            stop = excs >= 3
        else:
            h.update(b"O" + len(b).to_bytes(8, "little"))
            h.update(b)
            nbytes += len(b)
            empties += not b
            stop = empties >= 2
        h.update(b"T" + f.tell().to_bytes(8, "little"))
        if stop:
            break
    m = f.mtime
    h.update(b"N" if m is None else b"M" + m.to_bytes(8, "little"))
    return h.digest(), calls, nbytes, exc


def label():
    v = sys.version_info
    return f"{v.major}.{v.minor}" + ("t" if sysconfig.get_config_var("Py_GIL_DISABLED") else "")


def cmd_gzip(cases_dir, part, detail_dir=None):
    with open(os.path.join(cases_dir, "index.json")) as fh:
        index = json.load(fh)
    with open(os.path.join(cases_dir, "inputs.bin"), "rb") as fh:
        blob = fh.read()
    groups = {}
    per_case = []
    calls = nbytes = 0
    exc = collections.Counter()
    for c in index:
        data = blob[c["off"] : c["off"] + c["len"]]
        d, k, b, e = run_gzip(data, c["src"], c["n"])
        groups.setdefault(c["group"], hashlib.sha1()).update(d)
        per_case.append(d.hex())
        calls += k
        nbytes += b
        exc.update(e)
    out = {
        "python": label(),
        "version": sys.version.split()[0],
        "cases": len(index),
        "calls": calls,
        "bytes": nbytes,
        "exceptions": dict(sorted(exc.items())),
        "groups": {g: short(h.digest()) for g, h in sorted(groups.items())},
    }
    with open(part, "w") as fh:
        json.dump(out, fh, indent=1)
    if detail_dir:
        with open(os.path.join(detail_dir, f"detail-{label()}.json"), "w") as fh:
            json.dump({"cases": per_case}, fh)
    print(f"{label()}: {len(index)} cases, {calls} calls, {nbytes} bytes, {dict(exc)}")  # noqa: T201 (a dev script)


def cmd_merge(out, parts):
    pythons = {}
    cases = None
    for p in parts:
        with open(p) as fh:
            d = json.load(fh)
        cases = d["cases"] if cases is None else cases
        assert d["cases"] == cases, p
        pythons[d["python"]] = {
            k: d[k] for k in ("version", "calls", "bytes", "exceptions", "groups")
        }
    with open(out, "w") as fh:
        json.dump({"cases": cases, "pythons": dict(sorted(pythons.items()))}, fh, indent=1)
        fh.write("\n")


WORDS = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "data",
    "record",
    "archive",
    "crawl",
    "page",
    "html",
    "body",
    "div",
    "class",
    "content",
    "news",
    "with",
    "from",
    "this",
    "that",
    "for",
    "are",
    "was",
    "which",
    "their",
    "text",
    "link",
    "href",
    "https",
    "example",
    "org",
    "com",
    "index",
    "main",
]


def text(rnd, n):
    lines = []
    size = 0
    while size < n:
        line = " ".join(rnd.choice(WORDS) for _ in range(rnd.randrange(3, 12)))
        if rnd.random() < 0.3:
            line = f'<a href="https://example.org/{rnd.randrange(100)}">{line}</a>'
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines).encode()


def brotli_bases():
    import brotlicffi

    rnd = random.Random(20261110)
    c = brotlicffi.Compressor(quality=4)
    streamed = c.compress(text(rnd, 1500)) + c.flush() + c.compress(text(rnd, 1500)) + c.finish()
    return [
        brotlicffi.compress(text(rnd, 3000), quality=5),
        brotlicffi.compress(text(rnd, 12000), quality=11, lgwin=22),
        brotlicffi.compress(rnd.randbytes(600), quality=1),
        brotlicffi.compress(b""),
        brotlicffi.compress(text(rnd, 2000), quality=0),
        streamed,
        brotlicffi.compress(text(rnd, 4000), mode=brotlicffi.MODE_TEXT, quality=9, lgwin=10),
    ]


def brotli_cases(bases):
    """The same list as brotli_cases in warc_gzipfile.rs: (name, input, chunk size; 0 = whole)."""
    out = []
    for bi, b in enumerate(bases):
        n = len(b)
        for cs in (0, 1, 7, 100, 16384):
            if cs == 1 and n > 4096:
                continue
            out.append((f"b{bi}/valid/{cs}", b, cs))
        step = max(1, n // 150)
        out.extend((f"b{bi}/trunc/{i}", b[:i], 0) for i in range(0, n, step))
        for i in range(0, n, step):
            m = bytearray(b)
            m[i] ^= 1 << ((i * 5 + 3) % 8)
            m = bytes(m)
            if i % (step * 4) == 0:
                out.append((f"b{bi}/flip/{i}/100", m, 100))
            out.append((f"b{bi}/flip/{i}", m, 0))
        for ti, t in enumerate((b"\x00", b"x", b, b"\x00" * 100)):
            out.extend((f"b{bi}/trail{ti}/{cs}", b + t, cs) for cs in (0, 100))
    garbage = (b"", b"\x00", b"hello world", bytes(range(0, 256, 4)))
    out.extend((f"garbage{gi}", g, 0) for gi, g in enumerate(garbage))
    return out


def run_brotli(data, chunk):
    import brotlicffi

    d = brotlicffi.Decompressor()
    h = hashlib.sha1()
    log = []
    chunks = (
        [data]
        if chunk == 0 or not data
        else [data[i : i + chunk] for i in range(0, len(data), chunk)]
    )
    for c in chunks:
        try:
            r = d.decompress(c)
        except brotlicffi.error as e:
            h.update(b"E" + str(e).encode() + b"\0")
            log.append(str(e))
        else:
            h.update(b"O" + len(r).to_bytes(8, "little"))
            h.update(r)
            log.append(f"ok {len(r)}")
    h.update(b"F" + len(d.flush()).to_bytes(8, "little"))
    h.update(b"I1" if d.is_finished() else b"I0")
    return h.digest(), log


def cmd_brotli(out, detail_dir=None, known_file=None):
    import brotlicffi

    known_names = set()
    if known_file:
        with open(known_file) as fh:
            known_names = {line.strip() for line in fh if line.strip()}
    bases = brotli_bases()
    cases = brotli_cases(bases)
    groups = {}
    known = {}
    per_case = []
    logs = []
    for name, data, chunk in cases:
        d, log = run_brotli(data, chunk)
        per_case.append(d.hex())
        logs.append(log[-4:])
        if name in known_names:
            known[name] = {"brotlicffi": log[-1], "digest": short(d)}
            continue
        groups.setdefault("/".join(name.split("/")[:2]), hashlib.sha1()).update(d)
    assert len(known) == len(known_names), known_names - set(known)
    with open(out, "w") as fh:
        json.dump(
            {
                "brotlicffi": brotlicffi.__version__,
                "bases": [base64.b64encode(b).decode() for b in bases],
                "cases": len(cases),
                "groups": {g: short(h.digest()) for g, h in sorted(groups.items())},
                "known": known,
            },
            fh,
            indent=1,
        )
        fh.write("\n")
    if detail_dir:
        with open(os.path.join(detail_dir, "detail-brotli.json"), "w") as fh:
            json.dump({"cases": per_case, "logs": logs}, fh)
    print(f"brotli: {len(cases)} cases ({len(known)} known), bases {[len(b) for b in bases]} bytes")  # noqa: T201 (a dev script)


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "gzip":
        cmd_gzip(*args)
    elif cmd == "merge":
        cmd_merge(args[0], args[1:])
    elif cmd == "brotli":
        cmd_brotli(*args)
    else:
        sys.exit(__doc__)
