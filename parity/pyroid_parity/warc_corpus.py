"""The warc suite's pinned inputs: Common Crawl WARC/WET/WAT/ARC slices and warcio 1.8.2's own test files.

    python -m pyroid_parity.warc_corpus fetch [--pin]   # download into $PYROID_DATA/warc/ (bench/fetch.py --only warc)
    python -m pyroid_parity.warc_corpus resolve         # print the member-aligned ranges for CC_PLAN (maintainers)
    python -m pyroid_parity.warc_corpus stats           # records per pinned file, as warcio reads them

Common Crawl slices are byte ranges of one file under data.commoncrawl.org, fetched with one HTTP range request
and pinned by sha256 in ``bench/manifest.json`` (``trafilatura_cc.http_range``). Common Crawl compresses every
record as its own gzip member; ``cut`` says how a slice relates to those members:

- ``members``: starts and ends on member boundaries (a valid multi-member file),
- ``tail``: starts on a boundary and ends inside a member (a truncated download),
- ``head``: starts inside a member (garbage before the first whole member) and ends on a boundary.

The crawls run from 2012 (ARC files) and 2013 (the first WARC crawl) to 2026. Up to CC-MAIN-2016-40 records have no
``WARC-Identified-Payload-Type`` header (nor do WET records), so datatrove's ``process_record`` sends them to libmagic. WET
files hold ``conversion`` records, WAT files JSON ``metadata`` records.

warcio's test files (https://github.com/webrecorder/warcio/tree/v1.8.2/test/data, Apache-2.0) are fetched from
the tagged commit and checked against their git blob sha1s. Nothing is committed: everything lives under
``$PYROID_DATA/warc/``. Licences: Common Crawl's terms of use (third-party web content), Apache-2.0 for warcio's.

Derived files (plain, recompressed, re-chunked versions of the pinned ones) are written at test time by
``derived()``; they are deterministic functions of the pinned bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

from pyroid_parity import data as D
from pyroid_parity import trafilatura_cc as CC

WARCIO_VERSION = "1.8.2"
DATATROVE_VERSION = "0.10.1"
CC_LICENSE = CC.LICENSE
WARCIO_LICENSE = "Apache-2.0 (warcio's test files, https://github.com/webrecorder/warcio)"
WARCIO_REPO = "webrecorder/warcio"
WARCIO_COMMIT = "b9036802318d20a2dec8d58e56a8dacb543fa3ef"  # tag v1.8.2
PREFIX = "warc/"


@dataclass(frozen=True)
class CCFile:
    """Bytes ``[start, end)`` of ``path`` (under data.commoncrawl.org), stored as ``$PYROID_DATA/<rel>``."""

    rel: str
    crawl: str
    kind: str  # warc | wet | wat | arc
    path: str
    start: int
    end: int
    cut: str  # members | tail | head

    @property
    def url(self) -> str:
        return CC.CC_DATA + self.path

    def source(self) -> dict[str, object]:
        return {
            "crawl": self.crawl,
            "kind": self.kind,
            "url": self.url,
            "range": [self.start, self.end],
            "cut": self.cut,
            "license": CC_LICENSE,
        }


def _seg(crawl: str, seg: str, kind: str, name: str) -> str:
    return f"crawl-data/{crawl}/segments/{seg}/{kind}/{name}"


# File 00001 of a segment of each crawl. Before 2017 the file names carry the crawler's host name.
_P13 = ("1368696381249", "CC-MAIN-20130516092621-00001-ip-10-60-113-184.ec2.internal")
_P1348 = ("1386163035819", "CC-MAIN-20131204131715-00001-ip-10-33-133-15.ec2.internal")
_P14 = ("1397609521512.15", "CC-MAIN-20140416005201-00001-ip-10-147-4-33.ec2.internal")
_P15 = ("1435375094690.4", "CC-MAIN-20150627031814-00039-ip-10-179-60-89.ec2.internal")
# A whole file of 431 bytes: the crawl has many such (one warcinfo record and nothing else).
EMPTY_2015 = _seg(
    "CC-MAIN-2015-27",
    "1435375090820.0",
    "warc",
    "CC-MAIN-20150627031810-00001-ip-10-179-60-89.ec2.internal.warc.gz",
)
_P16 = ("1474738659496.36", "CC-MAIN-20160924173739-00001-ip-10-143-35-109.ec2.internal")
_CRAWLS: dict[str, tuple[str, str]] = {
    "CC-MAIN-2013-20": _P13,
    "CC-MAIN-2013-48": _P1348,
    "CC-MAIN-2014-15": _P14,
    "CC-MAIN-2015-27": _P15,
    "CC-MAIN-2016-40": _P16,
    "CC-MAIN-2017-30": ("1500549423183.57", "CC-MAIN-20170720121902-20170720141902-00001"),
    "CC-MAIN-2018-30": ("1531676588961.14", "CC-MAIN-20180715183800-20180715203800-00001"),
    "CC-MAIN-2018-51": ("1544376823009.19", "CC-MAIN-20181209185547-20181209211547-00001"),
    "CC-MAIN-2019-35": ("1566027312025.20", "CC-MAIN-20190817203056-20190817225056-00001"),
    "CC-MAIN-2020-50": ("1606141163411.0", "CC-MAIN-20201123153826-20201123183826-00001"),
    "CC-MAIN-2021-49": ("1637964358074.14", "CC-MAIN-20211126224056-20211127014056-00001"),
    "CC-MAIN-2022-49": ("1669446706285.92", "CC-MAIN-20221126080725-20221126110725-00001"),
    "CC-MAIN-2023-50": ("1700679099281.67", "CC-MAIN-20231128083443-20231128113443-00001"),
    "CC-MAIN-2024-51": ("1733066035857.0", "CC-MAIN-20241201162023-20241201192023-00001"),
    "CC-MAIN-2025-51": ("1764871306713.64", "CC-MAIN-20251204191828-20251204221828-00001"),
    "CC-MAIN-2026-43": ("1791170053823.21", "CC-MAIN-20261005105623-20261005135623-00001"),
}
_SUFFIX = {"warc": ".warc.gz", "wet": ".warc.wet.gz", "wat": ".warc.wat.gz"}
# The 2012 crawl (ARC files, one gzip member per record); its file names come from the segment's metadata.
ARC_2012 = "parse-output/segment/1346823845675/1346864469604_0.arc.gz"


def cc_path(crawl: str, kind: str) -> str:
    if kind == "arc":
        return ARC_2012
    seg, stem = _CRAWLS[crawl]
    return _seg(crawl, seg, kind, stem + _SUFFIX[kind])


# What to fetch: (rel, crawl, kind, offset, length, cut). ``resolve`` turns this into CC_FILES' exact ranges
# (members: the last whole member inside [offset, offset + length); head/tail: the raw range).
MB = 1 << 20
CC_PLAN: tuple[tuple[str, str, str, int, int, str], ...] = (
    *((f"warc/cc/{c}.warc.gz", c, "warc", 0, 3 * MB, "members") for c in _CRAWLS),
    ("warc/cc/CC-MAIN-2017-30-mid.warc.gz", "CC-MAIN-2017-30", "warc", 351_160_313, 3 * MB, "members"),
    ("warc/cc/CC-MAIN-2021-49-mid.warc.gz", "CC-MAIN-2021-49", "warc", 492_131_831, 3 * MB, "members"),
    ("warc/cc/CC-MAIN-2025-51-mid.warc.gz", "CC-MAIN-2025-51", "warc", 351_474_270, 3 * MB, "members"),
    ("warc/cc/CC-MAIN-2015-27-empty.warc.gz", "CC-MAIN-2015-27", "warc", 0, 431, "members"),
    ("warc/cc/CC-MAIN-2013-48-tail.warc.gz", "CC-MAIN-2013-48", "warc", 0, 2 * MB, "tail"),
    ("warc/cc/CC-MAIN-2022-49-tail.warc.gz", "CC-MAIN-2022-49", "warc", 470_673_244, 2 * MB, "tail"),
    ("warc/cc/CC-MAIN-2024-51-head.warc.gz", "CC-MAIN-2024-51", "warc", 401_245_087, MB // 2, "head"),
    *(
        (f"warc/cc/{c}.warc.wet.gz", c, "wet", 0, 2 * MB, "members")
        for c in (
            "CC-MAIN-2013-20",
            "CC-MAIN-2016-40",
            "CC-MAIN-2019-35",
            "CC-MAIN-2023-50",
            "CC-MAIN-2026-43",
        )
    ),
    ("warc/cc/CC-MAIN-2026-43-tail.warc.wet.gz", "CC-MAIN-2026-43", "wet", 0, 3 * MB // 2, "tail"),
    *(
        (f"warc/cc/{c}.warc.wat.gz", c, "wat", 0, 2 * MB, "members")
        for c in ("CC-MAIN-2013-48", "CC-MAIN-2018-30", "CC-MAIN-2022-49", "CC-MAIN-2026-43")
    ),
    ("warc/cc/CC-MAIN-2012.arc.gz", "CC-MAIN-2012", "arc", 0, 3 * MB, "members"),
)

# Rows whose file is not the crawl's usual one.
_PATHS = {"warc/cc/CC-MAIN-2015-27-empty.warc.gz": EMPTY_2015}

# The resolved ranges (python -m pyroid_parity.warc_corpus resolve); sha256s are in bench/manifest.json.
CC_RANGES: dict[str, tuple[int, int]] = {
    "warc/cc/CC-MAIN-2013-20.warc.gz": (0, 3_145_455),
    "warc/cc/CC-MAIN-2013-48.warc.gz": (0, 3_145_322),
    "warc/cc/CC-MAIN-2014-15.warc.gz": (0, 3_139_705),
    "warc/cc/CC-MAIN-2015-27.warc.gz": (0, 3_131_085),
    "warc/cc/CC-MAIN-2016-40.warc.gz": (0, 3_130_069),
    "warc/cc/CC-MAIN-2017-30.warc.gz": (0, 3_145_471),
    "warc/cc/CC-MAIN-2018-30.warc.gz": (0, 3_142_087),
    "warc/cc/CC-MAIN-2018-51.warc.gz": (0, 3_131_767),
    "warc/cc/CC-MAIN-2019-35.warc.gz": (0, 3_076_371),
    "warc/cc/CC-MAIN-2020-50.warc.gz": (0, 3_141_670),
    "warc/cc/CC-MAIN-2021-49.warc.gz": (0, 3_139_531),
    "warc/cc/CC-MAIN-2022-49.warc.gz": (0, 3_086_273),
    "warc/cc/CC-MAIN-2023-50.warc.gz": (0, 3_018_771),
    "warc/cc/CC-MAIN-2024-51.warc.gz": (0, 3_144_386),
    "warc/cc/CC-MAIN-2025-51.warc.gz": (0, 3_132_790),
    "warc/cc/CC-MAIN-2026-43.warc.gz": (0, 2_959_691),
    "warc/cc/CC-MAIN-2017-30-mid.warc.gz": (351_163_653, 354_289_015),
    "warc/cc/CC-MAIN-2021-49-mid.warc.gz": (492_141_861, 495_055_886),
    "warc/cc/CC-MAIN-2025-51-mid.warc.gz": (351_503_007, 354_540_711),
    "warc/cc/CC-MAIN-2015-27-empty.warc.gz": (0, 431),
    "warc/cc/CC-MAIN-2013-48-tail.warc.gz": (0, 2_097_152),
    "warc/cc/CC-MAIN-2022-49-tail.warc.gz": (470_690_814, 472_770_396),
    "warc/cc/CC-MAIN-2024-51-head.warc.gz": (401_245_087, 401_479_269),
    "warc/cc/CC-MAIN-2013-20.warc.wet.gz": (0, 2_097_128),
    "warc/cc/CC-MAIN-2016-40.warc.wet.gz": (0, 2_095_705),
    "warc/cc/CC-MAIN-2019-35.warc.wet.gz": (0, 2_093_426),
    "warc/cc/CC-MAIN-2023-50.warc.wet.gz": (0, 2_094_949),
    "warc/cc/CC-MAIN-2026-43.warc.wet.gz": (0, 2_096_290),
    "warc/cc/CC-MAIN-2026-43-tail.warc.wet.gz": (0, 1_572_864),
    "warc/cc/CC-MAIN-2013-48.warc.wat.gz": (0, 2_093_301),
    "warc/cc/CC-MAIN-2018-30.warc.wat.gz": (0, 2_091_266),
    "warc/cc/CC-MAIN-2022-49.warc.wat.gz": (0, 2_095_292),
    "warc/cc/CC-MAIN-2026-43.warc.wat.gz": (0, 2_096_660),
    "warc/cc/CC-MAIN-2012.arc.gz": (0, 3_145_261),
}


def cc_files() -> list[CCFile]:
    """The pinned Common Crawl files, in CC_PLAN order (only those whose range is resolved)."""
    out = []
    for rel, crawl, kind, _off, _len, cut in CC_PLAN:
        if rel in CC_RANGES:
            start, end = CC_RANGES[rel]
            out.append(CCFile(rel, crawl, kind, _PATHS.get(rel) or cc_path(crawl, kind), start, end, cut))
    return out


def resolve_one(
    rel: str, crawl: str, kind: str, offset: int, length: int, cut: str
) -> tuple[int, int, bytes]:
    """The range CC_RANGES pins for one CC_PLAN row, and its bytes (downloads ``length`` bytes)."""
    path = _PATHS.get(rel) or cc_path(crawl, kind)
    data = CC.http_range(CC.CC_DATA + path, offset, offset + length)
    if cut == "members":
        first = 0 if offset == 0 else _first_member(data)
        end, n = CC.last_complete_end(data, first)
        if n == 0:
            raise ValueError(f"{path} @{offset}: no whole member")
        return offset + first, offset + end, data[first:end]
    if cut == "tail":
        first = 0 if offset == 0 else _first_member(data)
        end, _ = CC.last_complete_end(data, first)
        if end == len(data):
            raise ValueError(f"{path} @{offset}: the range ends on a member boundary; pick another length")
        return offset + first, offset + len(data), data[first:]
    if cut == "head":
        first = _first_member(data)
        if first == 0:
            raise ValueError(f"{path} @{offset}: the range starts on a member boundary")
        end, _ = CC.last_complete_end(data, first)
        return offset, offset + end, data[:end]
    raise ValueError(cut)


def _first_member(data: bytes) -> int:
    """Offset of the first whole member that decompresses to a WARC or ARC record."""
    pos = data.find(CC.GZIP_MAGIC)
    while pos >= 0:
        end = CC.member_end(data, pos)
        if end is not None:
            return pos
        pos = data.find(CC.GZIP_MAGIC, pos + 1)
    raise ValueError("no whole gzip member")


# ------------------------------------------------------------------------------------- warcio's test files

# test/data at v1.8.2: name -> git blob sha1 (from the tree listing of WARCIO_COMMIT).
WARCIO_FILES: dict[str, str] = {
    "bad.arc": "9de4160070adf25bf3dda8d79944c5dbe66dce2a",
    "example-bad-non-chunked.warc.gz": "0c81fbaa298920befb660433650ba33beb841c76",
    "example-bad.warc.gz.bad": "95d2c415b4d792cd7fab3f0141dd224c7d9b08f6",
    "example-digest.warc": "ebda679663bdeeeed548f0077d35fd444eb2ee17",
    "example-iana.org-chunked.warc": "f544e6a300b551cebf19c5d6a56fdd0ee1583eec",
    "example-resource.warc.gz": "91737eeebb0e62e5308753786cfe9373f5653f2f",
    "example-space-in-target-uri.warc.gz": "8a63c18962330cd08f31399c36bfa90ecc4873f6",
    "example-space-in-url.arc": "da248a83219a382534f1673b9af1ee4c58b795c1",
    "example-trunc.warc": "6a1e7351529d61b3c626edc6a63e6ca0667d1845",
    "example-wget-bad-target-uri.warc.gz": "8193195023377a3f0692303ebaa0f912db07f146",
    "example-wrong-chunks.warc.gz": "4ec5cafd167deffb1e0e5c997c45fb2a10cb5cd4",
    "example.arc": "2024e9c25fc636ce59e88b63145fae770faecebd",
    "example.arc.gz": "bc959cf183aefc78d5e7fd630bbb906494ee190d",
    "example.warc": "3d0f5ed822eddc34153575f50d672960ea0fd65f",
    "example.warc.gz": "ed255396bf2f34d5e740e574d86f0f752864c9eb",
    "post-test.warc.gz": "b9cc1f48102154696b7d95c36ade930003556a8f",
}
WARCIO_DIR = f"warc/warcio-{WARCIO_VERSION}/"


def warcio_rel(name: str) -> str:
    return WARCIO_DIR + name


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ------------------------------------------------------------------------------------------- fetch and pin


def _write(rel: str, data: bytes) -> Path:
    out = D.DATA / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out


def fetch(pin: bool = False, log=CC.log) -> None:
    """Download every pinned warc input that is missing; with ``pin``, (re)write their manifest entries."""
    manifest = D.load_manifest()["files"]
    for f in cc_files():
        p = D.DATA / f.rel
        if not p.exists():
            log(f"{f.crawl} {f.kind}: bytes [{f.start:,}, {f.end:,}) of {f.path.rsplit('/', 1)[1]}")
            data = CC.http_range(f.url, f.start, f.end)
            want = manifest.get(f.rel, {}).get("sha256")
            if want and hashlib.sha256(data).hexdigest() != want:
                sys.exit(f"{f.rel}: sha256 differs from the manifest")
            _write(f.rel, data)
        if pin:
            D.pin(f.rel, source=f.source())
    base = f"https://raw.githubusercontent.com/{WARCIO_REPO}/{WARCIO_COMMIT}/test/data/"
    for name, sha1 in WARCIO_FILES.items():
        rel = warcio_rel(name)
        if not (D.DATA / rel).exists():
            with urllib.request.urlopen(base + urllib.parse.quote(name), timeout=120) as r:  # pinned commit
                data = r.read()
            if git_blob_sha1(data) != sha1:
                sys.exit(f"{name}: git blob sha1 differs from warcio's tree at {WARCIO_COMMIT}")
            _write(rel, data)
        if pin:
            D.pin(
                rel,
                source={
                    "repo": WARCIO_REPO,
                    "tag": f"v{WARCIO_VERSION}",
                    "commit": WARCIO_COMMIT,
                    "path": f"test/data/{name}",
                    "git_sha1": sha1,
                    "license": WARCIO_LICENSE,
                },
            )


# ------------------------------------------------------------------------------------------- the inputs


@dataclass(frozen=True)
class Input:
    """One pinned or derived input file: ``rel`` under $PYROID_DATA (pinned) or a name (derived)."""

    name: str
    path: Path
    kind: str  # warc | wet | wat | arc | warcio
    pinned: bool
    crawl: str = ""
    cut: str = "members"

    @property
    def gz(self) -> bool:
        return self.path.name.endswith(".gz")


def pinned_inputs(prefix: str = PREFIX) -> list[Input]:
    """Every pinned warc input present and verified here (unverifiable ones are left out: see ``missing``)."""
    out = [
        Input(f.rel, D.path(f.rel), f.kind, True, f.crawl, f.cut) for f in cc_files() if D.available(f.rel)
    ]
    for name in WARCIO_FILES:
        rel = warcio_rel(name)
        if D.available(rel):
            out.append(Input(rel, D.path(rel), "warcio", True))
    return [i for i in out if i.name.startswith(prefix)]


def missing() -> list[str]:
    rels = [f.rel for f in cc_files()] + [warcio_rel(n) for n in WARCIO_FILES]
    return [r for r in rels if not D.available(r)]


def gunzip_members(data: bytes) -> bytes:
    """All members of a multi-member gzip file, decompressed (stops quietly at a truncated tail)."""
    out, view, pos = [], memoryview(data), 0
    while pos < len(data):
        d = zlib.decompressobj(31)
        try:
            out.append(d.decompress(view[pos:]))
        except zlib.error:
            break
        if not d.eof:
            break
        pos = len(data) - len(d.unused_data)
    return b"".join(out)


def split_members(data: bytes) -> list[bytes]:
    """The decompressed members of a multi-member gzip file (stops at a truncated tail)."""
    out, view, pos = [], memoryview(data), 0
    while pos < len(data):
        d = zlib.decompressobj(31)
        try:
            out.append(d.decompress(view[pos:]))
        except zlib.error:
            break
        if not d.eof:
            break
        pos = len(data) - len(d.unused_data)
    return out


def gz(data: bytes, level: int = 6) -> bytes:
    """``gzip.compress`` with mtime 0, as CPython 3.13+ writes it on every Python (deterministic bytes
    for a given zlib; 3.12 keeps zlib's OS byte 3 in the header, 3.13 writes 255)."""
    out = bytearray(zlib.compress(data, level=level, wbits=31))
    out[9] = 255
    return bytes(out)


def derived(inp: Input, out_dir: Path) -> list[Input]:
    """Variants of a pinned whole-member CC file: plain, one gzip stream, two records per member, gzip again.

    Written into ``out_dir``. Only for ``cut == "members"`` gzip inputs.
    """
    if not inp.gz or inp.cut != "members":
        return []
    data = inp.path.read_bytes()
    members = split_members(data)
    stem = Path(inp.name).name.removesuffix(".gz")
    variants = {
        f"{stem}": b"".join(members),
        f"{stem}.single-stream.gz": gz(b"".join(members)),
        f"{stem}.pairs.gz": b"".join(gz(b"".join(members[i : i + 2])) for i in range(0, len(members), 2)),
        f"{stem}.level1.gz": b"".join(gz(m, 1) for m in members),
    }
    out = []
    for name, blob in variants.items():
        p = out_dir / name
        if not p.exists():
            p.write_bytes(blob)
        out.append(Input(f"derived:{name}", p, inp.kind, False, inp.crawl, "members"))
    return out


# ------------------------------------------------------------------------------------------------- CLI


def _stats() -> None:
    from warcio.archiveiterator import ArchiveIterator

    for inp in pinned_inputs():
        types: dict[str, int] = {}
        err = ""
        try:
            with inp.path.open("rb") as f:
                for rec in ArchiveIterator(f, arc2warc=False):
                    types[rec.rec_type] = types.get(rec.rec_type, 0) + 1
        except Exception as e:  # noqa: BLE001  report, do not stop
            err = f"{type(e).__name__}: {str(e).strip()[:60]}"
        n = sum(types.values())
        size = inp.path.stat().st_size
        print(f"{inp.name:52} {size / 1e6:6.2f} MB {n:6} records {types} {err}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pyroid_parity.warc_corpus")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--pin", action="store_true")
    sub.add_parser("resolve")
    sub.add_parser("stats")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        fetch(a.pin)
        return 0
    if a.cmd == "resolve":
        for rel, crawl, kind, off, length, cut in CC_PLAN:
            if rel in CC_RANGES:
                start, end = CC_RANGES[rel]
            else:
                start, end, blob = resolve_one(rel, crawl, kind, off, length, cut)
                _write(rel, blob)
            print(f'    "{rel}": ({start:_}, {end:_}),', flush=True)
        return 0
    _stats()
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CCFile", "Input", "cc_files", "derived", "fetch", "gunzip_members", "io", "pinned_inputs"]
