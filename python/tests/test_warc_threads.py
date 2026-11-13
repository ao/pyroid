"""pyroid.warc under threads (free-threaded builds included) and on a large input.

Each iterator serializes its own calls behind a lock that is taken without blocking the
interpreter; separate iterators run in parallel (a path is read with the GIL released).
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import pyroid.warc as pw


def _sibling(name: str) -> Any:
    """A sibling test module, imported by path (the suite runs without python/tests on sys.path)."""
    key = f"_pyroid_test_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
        assert spec is not None
        assert spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


tw = _sibling("test_warc")
td = _sibling("test_warc_documents")


def big_warc(n: int) -> bytes:
    """``n`` response records (with HTTP headers, some chunked or gzip), one gzip member each."""
    recs = []
    for i in range(n):
        body = f"<html><p>document {i}</p>{'x' * (i % 300)}</html>".encode()
        recs.append(
            tw.response(
                body,
                uri=f"http://example.com/{i}",
                chunked=i % 7 == 0,
                encoding="gzip" if i % 11 == 0 else None,
                digests=i % 5 == 0,
            )
        )
    out: bytes = tw.gz_members(recs)
    return out


def digest(it: Any) -> str:
    h = hashlib.sha256()
    for rec in it:
        h.update(repr(tw.record_summary(rec)).encode())
        h.update(rec.content_stream().read())
    return h.hexdigest()


@pytest.fixture(scope="module")
def warc_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    p = tmp_path_factory.mktemp("warc") / "big.warc.gz"
    p.write_bytes(big_warc(2000))
    return p


def test_separate_iterators_in_threads(warc_path: Path) -> None:
    want = digest(pw.ArchiveIterator(io.BytesIO(warc_path.read_bytes())))
    data = warc_path.read_bytes()

    def one(i: int) -> str:
        src: Any = warc_path if i % 2 else io.BytesIO(data)
        return digest(pw.ArchiveIterator(src, check_digests=i % 3 == 0))

    with ThreadPoolExecutor(8) as ex:
        assert list(ex.map(one, range(16))) == [want] * 16


def test_one_iterator_shared_by_threads(warc_path: Path) -> None:
    it = pw.ArchiveIterator(warc_path)
    seen: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(8)

    def work() -> None:
        start.wait()
        for rec in it:
            uri = rec.rec_headers.get_header("WARC-Target-URI")
            rec.raw_stream.read(5)  # reads interleave with other threads' next()
            with lock:
                seen.append(uri)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(seen) == sorted(f"http://example.com/{i}" for i in range(2000))


def test_records_read_from_other_threads(warc_path: Path) -> None:
    """Old records' streams read while the iterator advances (the lock order)."""
    it = pw.ArchiveIterator(io.BytesIO(warc_path.read_bytes()))
    recs: list[Any] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            for r in list(recs[-5:]):
                r.raw_stream.read(3)
                r.content_stream().read(3)
                r.raw_stream.tell()

    t = threading.Thread(target=reader)
    t.start()
    try:
        for rec in it:
            recs.append(rec)
            it.get_record_offset()
    finally:
        stop.set()
        t.join()
    assert len(recs) == 2000


@pytest.mark.thread_unsafe  # the magic / cchardet stubs in sys.modules
def test_documents_and_batches_in_threads(warc_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    td.install_stubs(monkeypatch)
    want = [(ri, d["url"]) for ri, d in pw.iter_documents(warc_path)]
    assert len(want) == 2000

    def one(i: int) -> Any:
        if i % 2:
            return [(ri, d["url"]) for ri, d in pw.iter_documents(warc_path)]
        return sum(b.num_rows for b in pw.iter_batches(warc_path, batch_size=300))

    with ThreadPoolExecutor(6) as ex:
        out = list(ex.map(one, range(12)))
    assert out == [2000 if i % 2 == 0 else want for i in range(12)]


@pytest.mark.thread_unsafe  # heavy memory
@pytest.mark.timeout(120)
def test_large_input_completes(tmp_path: Path) -> None:
    """20,000 records (about 13 MB uncompressed) through every API, under a timeout."""
    data = big_warc(20_000)
    p = tmp_path / "large.warc.gz"
    p.write_bytes(data)
    n = 0
    for rec in pw.ArchiveIterator(p):
        n += 1
        if n % 10 == 0:
            rec.content_stream().read()
    assert n == 20_000
    with p.open("rb") as f:
        assert sum(1 for _ in pw.ArchiveIterator(f)) == 20_000
    # Not gzip.decompress: it copies the rest of the input for every member, which is
    # quadratic over 20,000 members and took over 120 s on a musllinux-aarch64 runner.
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
        raw = g.read()
    assert sum(1 for _ in pw.ArchiveIterator(io.BytesIO(raw), check_digests=True)) == 20_000


def test_gil_stays_disabled() -> None:
    if not hasattr(sys, "_is_gil_enabled"):
        pytest.skip("no sys._is_gil_enabled")
    if sysconfig_gil_disabled():
        assert not sys._is_gil_enabled()


def sysconfig_gil_disabled() -> bool:
    import sysconfig

    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
