"""pyroid.patch("warcio"): pyroid's iterators on warcio's own modules, and warcio's back after.

With warcio installed, ``patch("warcio")`` sets subclasses of pyroid.warc's iterators as
``ArchiveIterator`` (on ``warcio`` and ``warcio.archiveiterator``), ``WARCIterator`` and
``ARCIterator`` (on ``warcio.archiveiterator``). Their records are warcio's own classes, and
everything they give (records, headers, content, offsets, errors, log warnings) must equal
warcio's on the same files; ``unpatch`` puts the exact original classes back. Without warcio the
names come from shims holding pyroid.warc's classes. Patching is process-wide state, so every
test runs in a fresh interpreter.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from pyroid.integrations import _patch


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


_dw = _sibling("test_datatrove_warc")
big_records, response, sample_records, write_warc = (
    _dw.big_records,
    _dw.response,
    _dw.sample_records,
    _dw.write_warc,
)

pytestmark = pytest.mark.thread_unsafe(reason="subprocesses and process-wide import state")

WARCIO_VERSION = "1.8.2"


def _warcio_ok() -> bool:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("warcio") == WARCIO_VERSION
    except PackageNotFoundError:
        return False


needs_warcio = pytest.mark.skipif(not _warcio_ok(), reason=f"needs warcio {WARCIO_VERSION}")

# Environment that would leak a patch (or a warning filter) into the children.
_CLEAN = ("PYTHONPATH", "PYROID_PATCH", "_PYROID_PATCH_WARNED", "PYTHONWARNINGS", "TEST_HIDE")

# Prepended to every child: TEST_HIDE=warcio makes an installed warcio look uninstalled.
PRELUDE = """\
import importlib, importlib.abc, json, logging, os, sys, warnings
HIDE = set(filter(None, os.environ.get("TEST_HIDE", "").split(",")))
class _Hide(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in HIDE:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
if HIDE:
    sys.meta_path.insert(0, _Hide())
LOG = []
class _Keep(logging.Handler):
    def emit(self, rec):
        LOG.append((rec.name, rec.getMessage()))
logging.getLogger("warcio").addHandler(_Keep())
logging.getLogger("warcio").setLevel(logging.WARNING)

def trace(cls, path, **kw):
    # Everything a warcio user sees of each record, and how the iteration ends.
    out = []
    with open(path, "rb") as f:
        it = cls(f, **kw)
        try:
            for r in it:
                h = r.http_headers
                out.append([
                    type(r).__mro__[-2].__name__, r.rec_type, r.format, r.length,
                    r.rec_headers.headers, r.rec_headers.protocol, r.rec_headers.statusline,
                    None if h is None else [h.protocol, h.statusline, h.headers],
                    r.content_type, r.content_stream().read().decode("latin-1"),
                    it.get_record_offset(), it.get_record_length(),
                ])
        except Exception as e:
            out.append(["error", type(e).__module__, type(e).__name__, str(e)])
    out.append(["log", LOG[:]])
    del LOG[:]
    return out
"""


def child(code: str, **env: str) -> subprocess.CompletedProcess[str]:
    e = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    e.update(env)
    return subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=e,
        check=False,
        timeout=240,
    )


def ok(p: subprocess.CompletedProcess[str]) -> Any:
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def warcs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    d = tmp_path_factory.mktemp("warcs")
    page = b"<html><body><p>Hello.</p></body></html>"
    files = {
        "gzip": write_warc(d / "a.warc.gz", big_records(2)),
        "plain": write_warc(d / "b.warc", sample_records(), gz=False),
        # one gzip member for several records: warcio raises ArchiveLoadFailed
        "one-member": d / "c.warc.gz",
        "truncated": d / "d.warc.gz",
        "unknown-te": write_warc(
            d / "e.warc.gz",
            [response(1, "http://example.com/", page, http=(("Transfer-Encoding", "gzip"),))],
        ),
    }
    files["one-member"].write_bytes(gzip.compress(b"".join(sample_records()), mtime=0))
    data = files["gzip"].read_bytes()
    files["truncated"].write_bytes(data[: len(data) - 500])
    return {k: str(v) for k, v in files.items()}


def test_supported_and_help() -> None:
    assert "warcio" in _patch.SUPPORTED
    assert _patch._WARCIO_SWAPS == {
        "warcio": ("ArchiveIterator",),
        "warcio.archiveiterator": ("ArchiveIterator", "WARCIterator", "ARCIterator"),
    }
    assert "warcio" in _patch._RUN_HELP
    assert "warcio" in (_patch.__doc__ or "")


@needs_warcio
def test_patch_swaps_and_unpatch_restores(warcs: dict[str, str]) -> None:
    got = ok(
        child(
            f"""
            import warcio, warcio.archiveiterator as A, warcio.recordloader as L
            import warcio.statusandheaders as S, warcio.exceptions as E
            import pyroid, pyroid.warc as pw
            from pyroid.integrations import PatchWarning
            files = {json.dumps(warcs)}
            orig = (warcio.ArchiveIterator, A.ArchiveIterator, A.WARCIterator, A.ARCIterator)
            want = {{k: trace(A.ArchiveIterator, p) for k, p in files.items()}}
            want_arc = trace(A.ARCIterator, files["plain"])
            want_warc = trace(A.WARCIterator, files["plain"], arc2warc=True)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pyroid.patch("warcio")
                pyroid.patch("warcio")  # a no-op
            assert [x.category for x in w] == [PatchWarning], [str(x.message) for x in w]
            msg = str(w[0].message)
            assert "the installed warcio now uses pyroid's drop-ins" in msg, msg
            new = (warcio.ArchiveIterator, A.ArchiveIterator, A.WARCIterator, A.ARCIterator)
            assert all(issubclass(c, pw.ArchiveIterator) for c in new)
            assert warcio.ArchiveIterator is A.ArchiveIterator
            assert all(a is not b for a, b in zip(orig, new))
            assert "warcio" not in str(type(sys.modules["warcio"]))  # no shim: warcio's module
            with open(files["gzip"], "rb") as f:
                rec = next(iter(A.ArchiveIterator(f)))
                assert isinstance(rec, L.ArcWarcRecord), type(rec).__mro__
                assert type(rec.rec_headers) is S.StatusAndHeaders
            got = {{k: trace(A.ArchiveIterator, p) for k, p in files.items()}}
            got_arc = trace(A.ARCIterator, files["plain"])
            got_warc = trace(A.WARCIterator, files["plain"], arc2warc=True)
            # datatrove's WarcReader imports ArchiveIterator from warcio.archiveiterator per file
            classes = [A.ArchiveIterator.__module__, A.ArchiveIterator.__qualname__]
            pyroid.unpatch("warcio")
            back = (warcio.ArchiveIterator, A.ArchiveIterator, A.WARCIterator, A.ARCIterator)
            assert all(a is b for a, b in zip(orig, back))
            print(json.dumps({{"want": want, "got": got, "classes": classes,
                              "arc": [want_arc, got_arc], "warc": [want_warc, got_warc]}}))
            """
        )
    )
    assert got["classes"] == ["pyroid.integrations._patch", "ArchiveIterator"]
    for name in warcs:
        assert got["got"][name] == got["want"][name], name
    assert got["arc"][0] == got["arc"][1]
    assert got["warc"][0] == got["warc"][1]
    ends = {k: v[-2][:3] for k, v in got["got"].items() if v[-2][0] == "error"}
    assert ends["one-member"] == ["error", "warcio.exceptions", "ArchiveLoadFailed"]
    assert any("Replacing spaces" in m for _, m in got["got"]["plain"][-1][1])


@needs_warcio
def test_datatrove_reads_through_the_patch(warcs: dict[str, str]) -> None:
    pytest.importorskip("datatrove")
    got = ok(
        child(
            f"""
            import types, importlib.machinery
            stub = types.ModuleType("magic")
            stub.__spec__ = importlib.machinery.ModuleSpec("magic", None)
            stub.from_buffer = lambda b, mime=False: "text/html"
            sys.modules["magic"] = stub
            import pyroid
            from datatrove.pipeline.readers import WarcReader
            folder = os.path.dirname({json.dumps(warcs["gzip"])})
            def docs():
                r = WarcReader(folder, glob_pattern="a.warc.gz")
                return [[d.id, d.text, d.metadata] for d in r.run()]
            want = docs()
            warnings.simplefilter("ignore")
            pyroid.patch("warcio")
            got = docs()
            print(json.dumps({{"want": want, "got": got}}))
            """
        )
    )
    assert got["got"] == got["want"]
    assert len(got["got"]) == 2 * 7 + 2


@needs_warcio
def test_names_imported_before_the_patch_keep_warcio(warcs: dict[str, str]) -> None:
    got = ok(
        child(
            """
            from warcio.archiveiterator import ArchiveIterator as Before
            import pyroid, pyroid.integrations, pyroid.warc as pw
            warnings.simplefilter("ignore")
            with pyroid.integrations.patching("warcio"):
                from warcio.archiveiterator import ArchiveIterator as During
                inside = pyroid.integrations.patched()
            from warcio.archiveiterator import ArchiveIterator as After
            print(json.dumps([issubclass(During, pw.ArchiveIterator), Before is After,
                              issubclass(Before, pw.ArchiveIterator), inside]))
            """
        )
    )
    assert got == [True, True, False, ["warcio"]]


def test_patch_without_warcio(warcs: dict[str, str]) -> None:
    got = ok(
        child(
            f"""
            import pyroid, pyroid.warc as pw
            warnings.simplefilter("ignore")
            pyroid.patch("warcio")
            import warcio
            from warcio.archiveiterator import ArchiveIterator, WARCIterator, ARCIterator
            from warcio.archiveiterator import ArchiveLoadFailed
            same = [ArchiveIterator is pw.ArchiveIterator, WARCIterator is pw.WARCIterator,
                    ARCIterator is pw.ARCIterator, ArchiveLoadFailed is pw.ArchiveLoadFailed,
                    warcio.ArchiveIterator is pw.ArchiveIterator,
                    warcio.StatusAndHeaders is pw.StatusAndHeaders, warcio.__pyroid__]
            try:
                warcio.WARCWriter
                missing = None
            except AttributeError as e:
                missing = str(e)
            n = len(trace(warcio.ArchiveIterator, {json.dumps(warcs["gzip"])}))
            pyroid.unpatch("warcio")
            gone = "warcio" not in sys.modules and "warcio.archiveiterator" not in sys.modules
            print(json.dumps({{"same": same, "missing": missing, "records": n, "gone": gone}}))
            """,
            TEST_HIDE="warcio",
        )
    )
    assert all(got["same"]), got["same"]
    assert "pyroid's warcio shim" in got["missing"]
    assert "warcio is not installed" in got["missing"]
    assert got["records"] > 20
    assert got["gone"]


def test_pyroid_run_patches_warcio(tmp_path: Path, warcs: dict[str, str]) -> None:
    script = tmp_path / "s.py"
    script.write_text(
        "import json, warcio.archiveiterator as A, pyroid.warc as pw\n"
        "print(json.dumps(issubclass(A.ArchiveIterator, pw.ArchiveIterator)))\n"
    )
    e = {k: v for k, v in os.environ.items() if k not in _CLEAN}
    p = subprocess.run(
        [sys.executable, "-m", "pyroid", "run", "-p", "warcio", str(script)],
        capture_output=True,
        text=True,
        env=e,
        check=False,
        timeout=240,
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1]) is True
    h = subprocess.run(
        [sys.executable, "-m", "pyroid", "run", "--help"], capture_output=True, text=True, env=e
    )
    assert "warcio" in h.stdout
