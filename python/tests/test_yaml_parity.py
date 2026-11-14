"""pyroid.yaml against PyYAML 6.0.3's pure-Python SafeLoader and SafeDumper.

Every comparison covers the loaded values (types, key order, shared objects), or the exception
class and its ``str()`` (message and marks), and dumped text byte for byte.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def _helpers() -> Any:
    """python/tests/_yaml_util.py (CI imports tests with --import-mode=importlib)."""
    name = "_yaml_util"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("_yaml_util.py"))
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_u = _helpers()
Chunked = _u.Chunked
data = _u.data
dump_options = _u.dump_options
encode = _u.encode
load_outcome = _u.load_outcome
outcome = _u.outcome
yaml_texts = _u.yaml_texts

yaml: Any = pytest.importorskip("yaml")
import pyroid.yaml as py  # noqa: E402

# The results are PyYAML's either way; the slow-path notice (e.g. dumping a str with lone
# surrogates) says which code path ran. test_yaml.py checks the native paths.
pytestmark = pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")

DATA = Path(__file__).resolve().parents[2] / "parity" / "data" / "pyyaml-6.0.3" / "data"
EXTENSIONS = (".data", ".canonical", ".loader-error", ".single-loader-error", ".detect", ".unicode",
              ".code", ".structure", ".tokens", ".events", ".error", ".stream-error", ".path")
FILES = sorted(p for p in DATA.glob("*") if p.suffix in EXTENSIONS) if DATA.is_dir() else []


def ref_load(stream: Any) -> Any:
    return yaml.load(stream, Loader=yaml.SafeLoader)


def ref_load_all(stream: Any) -> Any:
    return yaml.load_all(stream, Loader=yaml.SafeLoader)


def inputs(raw: bytes) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = [("bytes", raw)]
    try:
        out.append(("str", raw.decode("utf-8")))
    except UnicodeDecodeError:
        pass
    return out


@pytest.mark.skipif(not FILES, reason="needs parity/data/pyyaml-6.0.3 (a source checkout)")
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_pyyaml_test_data_loads(path: Path) -> None:
    raw = path.read_bytes()
    for _, value in inputs(raw):
        assert load_outcome(py.safe_load, value) == load_outcome(ref_load, value)
        assert load_outcome(py.safe_load_all, value) == load_outcome(ref_load_all, value)
    # File objects, read 4096 bytes at a time by both, and in small pieces.
    for make in (lambda: io.BytesIO(raw), lambda: Chunked(raw, 7), lambda: Chunked(raw.decode("latin-1"), 3)):
        assert load_outcome(py.safe_load, make()) == load_outcome(ref_load, make())
        assert load_outcome(py.safe_load_all, make()) == load_outcome(ref_load_all, make())


@pytest.mark.skipif(not FILES, reason="needs parity/data/pyyaml-6.0.3 (a source checkout)")
@pytest.mark.parametrize("path", [p for p in FILES if p.suffix in (".data", ".canonical")], ids=lambda p: p.name)
def test_pyyaml_test_data_dumps(path: Path) -> None:
    try:
        docs = list(yaml.load_all(path.read_bytes(), Loader=yaml.SafeLoader))
    except yaml.YAMLError:
        pytest.skip("not loadable")
    for opts in ({}, {"default_flow_style": None}, {"canonical": True}, {"allow_unicode": True, "width": 20},
                 {"sort_keys": False, "indent": 4, "explicit_start": True}, {"default_style": '"'}):
        assert outcome(py.safe_dump_all, docs, **opts) == outcome(yaml.safe_dump_all, docs, **opts)
        assert outcome(py.dump_all, docs, **opts) == outcome(yaml.dump_all, docs, **opts)


@settings(suppress_health_check=list(HealthCheck), deadline=None)
@given(yaml_texts(yaml.safe_dump), st.booleans())
def test_random_yaml_loads(text: str, as_bytes: bool) -> None:
    value: Any = text.encode("utf-8") if as_bytes else text
    assert load_outcome(py.safe_load, value) == load_outcome(ref_load, value)
    assert load_outcome(py.safe_load_all, value) == load_outcome(ref_load_all, value)


@settings(suppress_health_check=list(HealthCheck), deadline=None)
@given(data, dump_options, st.booleans())
def test_random_dumps(doc: Any, opts: dict[str, Any], share: bool) -> None:
    if share:
        doc = [doc, doc, [doc]]
    assert outcome(py.safe_dump, doc, **opts) == outcome(yaml.safe_dump, doc, **opts)
    assert outcome(py.dump, doc, **opts) == outcome(yaml.dump, doc, **opts)
    assert outcome(py.safe_dump_all, [doc, doc], **opts) == outcome(yaml.safe_dump_all, [doc, doc], **opts)


@settings(suppress_health_check=list(HealthCheck), deadline=None)
@given(data)
def test_round_trip(doc: Any) -> None:
    text = py.safe_dump(doc)
    assert load_outcome(py.safe_load, text) == load_outcome(ref_load, text)


@pytest.mark.parametrize(
    "text",
    [
        "a: 1\nb: [x, 'y']\n",
        "- !!int ' 0x1F '\n- !!float '١٫٥'\n",
        "!!bool maybe",
        "!!int ''",
        "!!int 0b",
        "!!timestamp nope",
        "!!timestamp 2001-13-01",
        "2001-02-30",
        "2001-12-14 25:00:00",
        "!!binary 'aGVsbG8'",
        "!!binary é",
        "!!str {=: x}",
        "!!null [a]",
        "!!python/none ''",
        "- &a {<<: *a, x: 1}\n",
        "- &v =\n- {*v : 1}\n",
        "- {&v = : 1}\n- [*v]\n",
        "!!set {a, b, a}",
        "!!omap [a: 1, b: 2]",
        "!!pairs [a: 1, a: 2]",
        "!!omap {a: 1}",
        "[.nan, .NaN]",
        "{.nan: 1, .NaN: 2}",
        "{1: a, 1.0: b, true: c}",
        "[[1]]: x",
        "\"\\ud800\\udc00\"",
        "\"\\U00110000\"",
        "\"\\Uffffffff\"",
        "%YAML 2.0\n--- x",
        "%YAML 1.1\n%YAML 1.1\n--- x",
        "%TAG ! a\n%TAG ! b\n--- x",
        "--- a\n--- b",
        "@",
        "a\x07",
        "\ufeffa: 1",
        "a: \ufeffb",
    ],
)
def test_edge_cases(text: str) -> None:
    for value in (text, text.encode("utf-8", "surrogatepass")):
        assert load_outcome(py.safe_load, value) == load_outcome(ref_load, value)
        assert load_outcome(py.safe_load_all, value) == load_outcome(ref_load_all, value)


@pytest.mark.parametrize(
    "raw",
    [
        b"a: \xff",
        b"a: 1\n" * 2000 + b"b: \xe4\x41",
        b"a: [" + b"x, " * 3000 + b"\x80",
        b"\xef\xbb\xbfa: 1",
        "a: é\n".encode("utf-16-le"),
        b"\xff\xfe" + "a: é\n".encode("utf-16-le"),
        b"\xfe\xff" + "a: é\n".encode("utf-16-be"),
        b"\xff\xfe" + "a: 1\n".encode("utf-16-le") + b"\x00",
        b"\xff\xfe" + b"\x00\xd8a\x00",
        b"x: 1\n" * 1000 + b"\x07",
        b"[" * 10 + b"\x07",
        b"a: 'x" + b"y" * 5000,
    ],
)
def test_reader_errors(raw: bytes) -> None:
    assert load_outcome(py.safe_load, raw) == load_outcome(ref_load, raw)
    for size in (4096, 1, 2, 3, 4000, 5000):
        assert load_outcome(py.safe_load_all, Chunked(raw, size)) == load_outcome(ref_load_all, Chunked(raw, size))
    text = raw.decode("latin-1")
    for size in (4096, 7):
        assert load_outcome(py.safe_load, Chunked(text, size)) == load_outcome(ref_load, Chunked(text, size))


@pytest.mark.parametrize("loader", ["SafeLoader", "CSafeLoader", "FullLoader", "Loader", "UnsafeLoader"])
def test_stock_loaders(loader: str) -> None:
    cls = getattr(yaml, loader, None)
    if cls is None:
        pytest.skip(f"no {loader}")
    text = "a: [1, 2.5, yes, 2001-12-14]\nb: !!set {x}\n"
    assert encode(py.load(text, Loader=cls)) == encode(yaml.load(text, Loader=yaml.SafeLoader))


def test_str_mark_name_for_files(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("a: b: c\n", encoding="utf-8")
    with p.open() as f1, p.open() as f2:
        assert outcome(py.safe_load, f1) == outcome(ref_load, f2)
    with p.open("rb") as f1, p.open("rb") as f2:
        assert outcome(py.safe_load, f1) == outcome(ref_load, f2)
