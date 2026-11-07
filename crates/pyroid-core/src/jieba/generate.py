"""Generate ``data/hmm.bin``, and find the ``dict.txt`` behind ``data/dict.trie`` (jieba 0.42.1).

Run in Docker, in an environment with ``jieba==0.42.1``::

    python crates/pyroid-core/src/jieba/generate.py   # writes data/hmm.bin, prints dict.txt's path
    JIEBA_DICT_TXT=<path> cargo test -p pyroid-core --lib jieba::tests::regenerate -- --ignored

The second command rebuilds ``data/dict.trie`` from ``dict.txt`` (the trie format is
Rust's, so the Rust test writes it).

``hmm.bin`` holds ``jieba.finalseg``'s probabilities bit for bit (little-endian): ``b"JBHM"``,
u32 version 1, ``prob_start`` (4 f64, states B E M S), ``prob_trans`` (4x4 f64, ``MIN_FLOAT``
where jieba has no entry), then per state a u32 count and that many distinct ``prob_emit``
values, then for each code point U+4E00..U+9FD5 (``finalseg.re_han``, the only characters the
Viterbi sees) four u16 indices into those values (0: no entry, ``MIN_FLOAT``).
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import jieba
from jieba.finalseg import MIN_FLOAT
from jieba.finalseg.prob_emit import P as EMIT
from jieba.finalseg.prob_start import P as START
from jieba.finalseg.prob_trans import P as TRANS

VERSION = "0.42.1"
#: sha256 of jieba 0.42.1's files (PyPI sdist jieba-0.42.1.tar.gz, sha256 055ca12f...928db2).
SHA256 = {
    "dict.txt": "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8",
}
STATES = "BEMS"
HAN = range(0x4E00, 0x9FD5 + 1)
OUT = Path(__file__).resolve().parent / "data" / "hmm.bin"


def f64(v: float) -> bytes:
    return struct.pack("<d", v)


def main() -> int:
    if jieba.__version__ != VERSION:
        sys.exit(f"need jieba {VERSION}, found {jieba.__version__}")
    root = Path(jieba.__file__).resolve().parent
    for name, want in SHA256.items():
        got = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if got != want:
            sys.exit(f"{name}: sha256 {got} != {want}")
    # Every character the emission table drops is outside re_han (none is used by the Viterbi).
    outside = {c for s in STATES for c in EMIT[s] if ord(c) not in HAN}
    out = bytearray(b"JBHM") + struct.pack("<I", 1)
    out += b"".join(f64(START[s]) for s in STATES)
    out += b"".join(f64(TRANS[a].get(b, MIN_FLOAT)) for a in STATES for b in STATES)
    index: dict[str, dict[bytes, int]] = {}
    for s in STATES:
        values = sorted({f64(v) for v in EMIT[s].values()})
        index[s] = {v: i + 1 for i, v in enumerate(values)}
        out += struct.pack("<I", len(values)) + b"".join(values)
    for cp in HAN:
        for s in STATES:
            v = EMIT[s].get(chr(cp))
            out += struct.pack("<H", 0 if v is None else index[s][f64(v)])
    OUT.write_bytes(out)
    sizes = {s: len(index[s]) for s in STATES}
    print(f"wrote {OUT} ({len(out)} bytes): distinct values {sizes}, dropped {sorted(outside)!r}")  # noqa: T201
    print(f"dict.txt: {root / 'dict.txt'}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
