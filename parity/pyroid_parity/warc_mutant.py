"""A stand-in candidate with warcio 1.8.2's API, to prove the warc suite: ``--impl warc=pyroid_parity.warc_mutant``.

    python -m pyroid_parity.warc_mutant prove [N]      # every defect on: exit 1 unless every check it can fail fails
    python -m pyroid_parity.warc_mutant faithful [N]   # no defect: exit 1 unless no check fails

``$PYROID_WARC_MUTANT`` picks the defects (comma-separated names of ``DEFECTS``, ``all`` (the default) or
``none``). With ``none`` this module is warcio behind the suite's contract, but not *as* the reference: the live
comparisons, ``iter_documents``, ``iter_batches`` and the datatrove step all run, so a faithful run checks the
harness end to end before pyroid.warc exists. ``N`` is ``--parity-limit`` (records per Common Crawl slice).

Each defect is one a port could ship:
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warcio.archiveiterator import ArchiveIterator as _ArchiveIterator
from warcio.digestverifyingreader import DigestChecker as _DigestChecker
from warcio.exceptions import ArchiveLoadFailed
from warcio.recordloader import ArcWarcRecord
from warcio.statusandheaders import StatusAndHeaders as _StatusAndHeaders
from warcio.utils import BUFF_SIZE

DEFECTS = {
    "content": "content_stream().read() loses the last byte of what it returns (records-*, documents-*)",
    "length": "get_record_length() is one too large (records-*)",
    "quiet": "the 'Record not followed by newline' warning is not written to stderr (records-*)",
    "block-size": "the iterator reads its source in blocks twice as large (io)",
    "drop-last": "iter_documents loses the last document of a file (iter-documents-*, hypothesis-documents)",
    "fileobj": "iter_documents ignores compression= for file objects (iter-documents-fileobj)",
    "batches": "iter_batches leaves mime_type empty, loses its last row and ends quietly on errors (batches-*)",
    "reader": "the datatrove step adds a metadata key (warcreader, warcreader-executor)",
    "headers": "StatusAndHeaders.get_header is case-sensitive (api-status-and-headers)",
    "checker": "DigestChecker('log') writes nothing (api-digest-checker)",
    "version": "WARCIO_VERSION is 1.8.1 (api-names)",
    "signature": "iter_documents takes compression positionally (api-signature)",
    "other-digests": "a digest with an algorithm outside the port counts as passed (digest-other-algorithms)",
}
__doc__ += "\n".join(f"* ``{k}``: {v}" for k, v in DEFECTS.items()) + "\n"
# Checks no defect here can fail: data integrity, oracle versions, the self-check-only ones, the pinned ones (they
# run only where warcio is missing, and this module needs warcio).
EXEMPT_PREFIXES = ("inputs", "oracle", "mutant")
EXEMPT_SUFFIX = "-pinned"


def _active() -> set[str]:
    raw = os.environ.get("PYROID_WARC_MUTANT", "all")
    if raw == "all":
        return set(DEFECTS)
    if raw in ("", "none"):
        return set()
    names = set(raw.split(","))
    if unknown := names - DEFECTS.keys():
        raise ValueError(f"PYROID_WARC_MUTANT: unknown defects {sorted(unknown)}")
    return names


ACTIVE = _active()
WARCIO_VERSION = "1.8.1" if "version" in ACTIVE else "1.8.2"


class _Short:
    def __init__(self, s: Any) -> None:
        self.s = s

    def read(self, length: Any = None) -> bytes:
        b = self.s.read(length)
        return b[:-1] if len(b) > 1 else b

    def readline(self, length: Any = None) -> bytes:
        return self.s.readline(length)


class ArchiveIterator(_ArchiveIterator):
    def __init__(
        self,
        fileobj: Any,
        no_record_parse: bool = False,
        verify_http: bool = False,
        arc2warc: bool = False,
        ensure_http_headers: bool = False,
        block_size: int = BUFF_SIZE,
        check_digests: Any = False,
    ) -> None:
        if "block-size" in ACTIVE:
            block_size *= 2
        super().__init__(
            fileobj, no_record_parse, verify_http, arc2warc, ensure_http_headers, block_size, check_digests
        )

    def read_to_end(self, record: Any = None) -> Any:
        fresh = self.member_info is None
        r = super().read_to_end(record)
        if "length" in ACTIVE and fresh and self.member_info:
            self.member_info = (self.member_info[0], self.member_info[1] + 1)
        return r

    def _consume_blanklines(self) -> Any:
        if "quiet" not in ACTIVE:
            return super()._consume_blanklines()
        old = sys.stderr
        sys.stderr = io.StringIO()
        try:
            return super()._consume_blanklines()
        finally:
            sys.stderr = old

    def _next_record(self, next_line: Any) -> Any:
        rec = super()._next_record(next_line)
        if "content" in ACTIVE:
            real = rec.content_stream
            rec.content_stream = lambda: _Short(real())
        if "other-digests" in ACTIVE and rec.digest_checker is not None:
            dc = rec.digest_checker
            for name in ("WARC-Block-Digest", "WARC-Payload-Digest"):
                v = rec.rec_headers.get_header(name) or ""
                if v.split(":")[0].lower() in (
                    "sha3_256",
                    "blake2b",
                    "blake2s",
                    "sha512_256",
                    "md5-sha1",
                    "sm3",
                    "ripemd160",
                ):
                    dc._passed = True
        return rec


class WARCIterator(ArchiveIterator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.known_format = "warc"


class ARCIterator(ArchiveIterator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.known_format = "arc"


class StatusAndHeaders(_StatusAndHeaders):
    def get_header(self, name: str, default_value: Any = None) -> Any:
        if "headers" not in ACTIVE:
            return super().get_header(name, default_value)
        for n, v in self.headers:
            if n == name:
                return v
        return default_value

    get = get_header
    __getitem__ = get_header


class DigestChecker(_DigestChecker):
    def problem(self, value: str, passed: bool = False) -> None:
        if "checker" in ACTIVE and self.kind == "log":
            self._problem.append(value)
            self._passed = passed
            return
        super().problem(value, passed)


def _documents(source: Any, compression: Any) -> Iterator[tuple[int, dict]]:
    import gzip

    from datatrove.pipeline.readers.warc import process_record

    from pyroid_parity.warc_compare import datatrove_open

    if isinstance(source, str | os.PathLike):
        with datatrove_open(Path(source), compression) as f:
            for ri, rec in enumerate(ArchiveIterator(f)):
                d = process_record(rec)
                if d:
                    yield ri, d
        return
    if "fileobj" in ACTIVE:
        compression = None
    f = gzip.GzipFile(fileobj=source) if compression == "gzip" else source
    for ri, rec in enumerate(ArchiveIterator(f)):
        d = process_record(rec)
        if d:
            yield ri, d


def _drop_last(it: Iterator[Any]) -> Iterator[Any]:
    prev = None
    have = False
    for x in it:
        if have:
            yield prev
        prev, have = x, True


if "signature" in ACTIVE:

    def iter_documents(source: Any, compression: Any = "infer") -> Iterator[tuple[int, dict]]:  # type: ignore[misc]
        g = _documents(source, compression)
        return _drop_last(g) if "drop-last" in ACTIVE else g

else:

    def iter_documents(source: Any, *, compression: Any = "infer") -> Iterator[tuple[int, dict]]:
        g = _documents(source, compression)
        return _drop_last(g) if "drop-last" in ACTIVE else g


def iter_batches(source: Any, *, batch_size: int = 1000, compression: Any = "infer") -> Iterator[Any]:
    """Rows as ``pyroid_parity.warc_datatrove.reference_rows`` describes them, as pyarrow RecordBatches."""
    if "batches" not in ACTIVE:
        yield from _batches(source, batch_size, compression)
        return
    import pyarrow as pa

    last = None
    with contextlib.suppress(Exception):  # the defect: errors end the batches quietly
        for b in _batches(source, batch_size, compression):
            if last is not None:
                yield last
            last = b
    if last is not None and last.num_rows > 1:
        yield pa.record_batch(last).slice(0, last.num_rows - 1)


def _batches(source: Any, batch_size: int, compression: Any) -> Iterator[Any]:
    import pyarrow as pa

    from pyroid_parity import warc_datatrove as T

    rows: dict[str, list[Any]] = {k: [] for k in T.BATCH_COLUMNS}
    schema = pa.schema(
        [
            ("id", pa.large_string()),
            ("url", pa.large_string()),
            ("date", pa.large_string()),
            ("mime_type", pa.large_string()),
            ("content_type", pa.large_string()),
            ("html", pa.large_binary()),
            ("text", pa.large_string()),
        ]
    )

    def flush() -> Any:
        b = pa.record_batch([pa.array(rows[k], schema.field(k).type) for k in T.BATCH_COLUMNS], schema=schema)
        for v in rows.values():
            v.clear()
        return b

    from datatrove.pipeline.readers.warc import process_record

    from pyroid_parity.warc_compare import datatrove_open

    with datatrove_open(Path(source), compression) as f:
        for rec in ArchiveIterator(f):
            if rec.rec_type not in ("response", "conversion"):
                continue
            mime = rec.rec_headers.get("WARC-Identified-Payload-Type", None)
            if mime is not None and not (
                mime in T.PAYLOAD_TYPES or (rec.rec_type == "conversion" and mime == "text/plain")
            ):
                continue
            html = rec.content_stream().read()
            d = process_record(T._Replay(rec, html))
            rows["id"].append(d["id"] if d else rec.rec_headers.get("WARC-Record-ID"))
            rows["url"].append(
                T.row_value(d["url"]) if d else T.fallback(rec.rec_headers, "WARC-Target-URI", "uri")
            )
            rows["date"].append(
                T.row_value(d["date"]) if d else T.fallback(rec.rec_headers, "WARC-Date", "archive-date")
            )
            rows["mime_type"].append(None if "batches" in ACTIVE else mime)
            rows["content_type"].append(
                rec.http_headers.get_header("Content-Type") if rec.http_headers else None
            )
            rows["html"].append(html)
            rows["text"].append(d["text"] if d else None)
            if len(rows["id"]) >= batch_size:
                yield flush()
    if rows["id"]:
        yield flush()


def _reader() -> Any:
    try:
        from datatrove.pipeline.readers import WarcReader as _WarcReader
    except ImportError:
        return None

    class WarcReader(_WarcReader):
        def read_file(self, filepath: str) -> Any:
            for doc in super().read_file(filepath):
                if "reader" in ACTIVE:
                    doc.metadata["mutant"] = True
                yield doc

    return WarcReader


datatrove_reader = _reader()

__all__ = [
    "WARCIO_VERSION",
    "ARCIterator",
    "ArcWarcRecord",
    "ArchiveIterator",
    "ArchiveLoadFailed",
    "DigestChecker",
    "StatusAndHeaders",
    "WARCIterator",
    "datatrove_reader",
    "iter_batches",
    "iter_documents",
]


# ------------------------------------------------------------------------------------------------ proofs


@dataclass
class Proof:
    returncode: int
    checks: dict[str, Any]
    output: str

    def failing(self) -> set[str]:
        return {n for n, c in self.checks.items() if c["status"] == "fail"}


def run(defects: str, limit: int, extra: tuple[str, ...] = ()) -> Proof:
    parity = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="pyroid-warc-mutant-") as d:
        report = Path(d) / "report.json"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_warc.py",
            "-q",
            "-p",
            "no:cacheprovider",
            "--impl",
            "warc=pyroid_parity.warc_mutant",
            "--parity-limit",
            str(limit),
            "--parity-report",
            str(report),
            *extra,
        ]
        env = {**os.environ, "PYROID_WARC_MUTANT": defects}
        p = subprocess.run(cmd, cwd=parity, env=env, capture_output=True, text=True, check=False)
        checks = json.loads(report.read_text("utf-8"))["suites"]["warc"]["checks"] if report.exists() else {}
    return Proof(p.returncode, checks, p.stdout[-4000:] + p.stderr[-2000:])


def exempt(check: str) -> bool:
    return check.startswith(EXEMPT_PREFIXES) or check.endswith(EXEMPT_SUFFIX)


def prove(limit: int = 20) -> tuple[Proof, set[str]]:
    """Every defect on: the checks that should fail and did not."""
    p = run("all", limit, ("-k", "not big"))
    should = {n for n in p.checks if not exempt(n) and p.checks[n]["status"] != "skip"}
    return p, should - p.failing()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "prove"
    limit = int(args[1]) if len(args) > 1 else 20
    if cmd == "faithful":
        p = run("none", limit)
        bad = p.failing()
        for n, c in sorted(p.checks.items()):
            print(f"{c['status']:5} {n:34} same={c['same']} diff={c['diff']}")
        print("faithful run:", "no check failed" if not bad else f"failing {sorted(bad)}")
        return 1 if bad or p.returncode not in (0,) else 0
    p, missed = prove(limit)
    for n, c in sorted(p.checks.items()):
        print(f"{c['status']:5} {n:34} same={c['same']} diff={c['diff']}")
    print("mutant:", "every check that can fail failed" if not missed else f"not caught: {sorted(missed)}")
    return 1 if missed or p.returncode == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
