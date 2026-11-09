"""Trafilatura 2.3.1 reference outputs for the pyroid.trafilatura parity checks.

    python -I ref.py RAW.bin OUTDIR MODE[,MODE...] [--bare]

RAW.bin is the prototype's .bin format (u32 LE count, then u32 LE length + surrogatepass UTF-8 per
document). For each MODE, documents run in order in this process; modes with deduplicate=True start
from an empty module-level cache (LRU_TEST.clear()), as a fresh datatrove task does.

Writes OUTDIR/MODE.jsonl ({"text": str | null} per document, trafilatura.extract) or, with --bare,
OUTDIR/MODE.bare.jsonl (bare_extraction: null or {"text", "raw_text", "comments", "body",
"commentsbody"}, the trees as lxml.etree.tostring(..., encoding="unicode")).
"""

import hashlib
import json
import struct
import sys
import time
import unicodedata

import trafilatura
from lxml import etree
from trafilatura.deduplication import LRU_TEST

MODES = {
    # datatrove's call: Trafilatura(favour_precision=True) in datatrove 0.10.1
    "datatrove": {"favor_precision": True, "include_comments": False, "deduplicate": True},
    "nodedup": {"favor_precision": True, "include_comments": False},
    "fast": {"favor_precision": True, "include_comments": False, "fast": True},
    "fastdedup": {"favor_precision": True, "include_comments": False, "fast": True, "deduplicate": True},
    # trafilatura's own defaults and the option sets standalone users call
    "default": {},
    "default_fast": {"fast": True},
    "recall": {"favor_recall": True},
    "precision_comments": {"favor_precision": True, "include_comments": True},
    "default_dedup": {"deduplicate": True},
    "notables": {"include_tables": False},
}


def read_bin(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<I", f.read(4))
        out = []
        for _ in range(n):
            (k,) = struct.unpack("<I", f.read(4))
            out.append(f.read(k).decode("utf-8", "surrogatepass"))
    return out


def digest(rec):
    "sha1 over the fields: b'N' for None, else b'S' + UTF-8, each followed by NUL; None is 'None'."
    if rec is None:
        return "None"
    h = hashlib.sha1()
    for k in ("text", "raw_text", "comments", "body", "commentsbody"):
        if k in rec:
            v = rec[k]
            h.update(b"N\0" if v is None else b"S" + v.encode("utf-8", "surrogatepass") + b"\0")
    return h.hexdigest()


def tostr(e):
    return None if e is None else etree.tostring(e, encoding="unicode")


def main():
    raw, outdir, modes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    bare = "--bare" in sys.argv[4:]
    hashed = "--hash" in sys.argv[4:]
    docs = read_bin(raw)
    for mode in modes:
        kw = MODES[mode]
        LRU_TEST.clear()
        t0 = time.perf_counter()
        suffix = (".bare" if bare else "") + (".sha1" if hashed else ".jsonl")
        with open(f"{outdir}/{mode}{suffix}", "w", encoding="utf-8") as f:
            for html in docs:
                if bare:
                    d = trafilatura.bare_extraction(html, **kw)
                    rec = None
                    if d is not None:
                        rec = {
                            "text": d.text,
                            "raw_text": d.raw_text,
                            "comments": d.comments,
                            "body": tostr(d.body),
                            "commentsbody": tostr(d.commentsbody),
                        }
                    f.write((digest(rec) if hashed else json.dumps(rec)) + "\n")
                else:
                    t = trafilatura.extract(html, **kw)
                    f.write((digest({"text": t}) if hashed else json.dumps({"text": t})) + "\n")
        dt = time.perf_counter() - t0
        print(f"{sys.version.split()[0]} (Unicode {unicodedata.unidata_version}) {mode}{' bare' if bare else ''}: "
              f"{len(docs)} docs, {dt:.1f} s, {1000 * dt / len(docs):.2f} ms/doc", flush=True)


if __name__ == "__main__":
    main()
