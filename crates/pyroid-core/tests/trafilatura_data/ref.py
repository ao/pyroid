"""Trafilatura reference outputs for the pyroid.trafilatura parity checks (2.3.1 or 1.11.0,
whichever is installed).

    python -I ref.py RAW.bin OUTDIR MODE[,MODE...] [--bare] [--hash]

RAW.bin is the prototype's .bin format (u32 LE count, then u32 LE length + surrogatepass UTF-8 per
document). For each MODE, documents run in order in this process; modes with deduplicate=True start
from an empty module-level cache (LRU_TEST.clear()), as a fresh datatrove task does.

Writes OUTDIR/MODE.jsonl ({"text": str | null} per document, trafilatura.extract) or, with --bare,
OUTDIR/MODE.bare.jsonl (bare_extraction: null or {"text", "raw_text", "comments", "body",
"commentsbody"}, the trees as lxml.etree.tostring(..., encoding="unicode")). With --hash, one
sha1 digest per document instead (MODE.sha1 / MODE.bare.sha1). A call that raises gives
{"error": "Type: message"} (with --hash: the line "!Type: message").

1.11.0 has no `fast` argument: the modes pass it as `no_fallback`, and bare_extraction runs with
as_dict=False.
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

V1 = trafilatura.__version__.startswith("1.")

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


def kwargs(mode, bare):
    kw = dict(MODES[mode])
    if V1:
        if "fast" in kw:
            kw["no_fallback"] = kw.pop("fast")
        if bare:
            kw["as_dict"] = False
    return kw


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
    if "error" in rec:
        return "!" + rec["error"]
    h = hashlib.sha1()
    for k in ("text", "raw_text", "comments", "body", "commentsbody"):
        if k in rec:
            v = rec[k]
            h.update(b"N\0" if v is None else b"S" + v.encode("utf-8", "surrogatepass") + b"\0")
    return h.hexdigest()


def tostr(e):
    return None if e is None else etree.tostring(e, encoding="unicode")


def run(html, kw, bare):
    try:
        if bare:
            d = trafilatura.bare_extraction(html, **kw)
            if d is None:
                return None
            return {
                "text": d.text,
                "raw_text": d.raw_text,
                "comments": d.comments,
                "body": tostr(d.body),
                "commentsbody": tostr(d.commentsbody),
            }
        return {"text": trafilatura.extract(html, **kw)}
    except Exception as e:  # recorded: the port reports the same exception
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    raw, outdir, modes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    bare = "--bare" in sys.argv[4:]
    hashed = "--hash" in sys.argv[4:]
    docs = read_bin(raw)
    for mode in modes:
        kw = kwargs(mode, bare)
        LRU_TEST.clear()
        t0 = time.perf_counter()
        suffix = (".bare" if bare else "") + (".sha1" if hashed else ".jsonl")
        with open(f"{outdir}/{mode}{suffix}", "w", encoding="utf-8") as f:
            for html in docs:
                rec = run(html, kw, bare)
                f.write((digest(rec) if hashed else json.dumps(rec)) + "\n")
        dt = time.perf_counter() - t0
        print(f"trafilatura {trafilatura.__version__}, {sys.version.split()[0]} (Unicode "
              f"{unicodedata.unidata_version}) {mode}{' bare' if bare else ''}: "
              f"{len(docs)} docs, {dt:.1f} s, {1000 * dt / len(docs):.2f} ms/doc", flush=True)


if __name__ == "__main__":
    main()
