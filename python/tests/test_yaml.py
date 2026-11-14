"""pyroid.yaml: the API, fallbacks, PyYAML-less mode, threads and hostile inputs."""

from __future__ import annotations

import datetime
import importlib
import io
import math
import sys
import threading
import warnings
from typing import Any

import pytest

import pyroid
import pyroid.yaml as py

yaml: Any
try:
    yaml = importlib.import_module("yaml")
except ImportError:  # pragma: no cover - the PyYAML-less tests run in a subprocess
    yaml = None

needs_pyyaml = pytest.mark.skipif(yaml is None, reason="needs PyYAML")


def test_basic_load_and_dump() -> None:
    doc = py.safe_load(
        "a: [1, 2.5, yes, ~, 2001-12-14, 2001-12-14t21:59:43.10-05:00]\n"
        "b: !!binary aGVsbG8=\nc: .nan\n"
    )
    assert doc["a"][:4] == [1, 2.5, True, None]
    assert doc["a"][4] == datetime.date(2001, 12, 14)
    stamp = doc["a"][5]
    assert stamp.utcoffset() == datetime.timedelta(hours=-5)
    assert stamp.microsecond == 100000
    assert doc["b"] == b"hello"
    assert math.isnan(doc["c"])
    assert py.safe_dump({"b": [1, 2], "a": "x"}) == "a: x\nb:\n- 1\n- 2\n"
    assert py.safe_dump([1, "two"], default_flow_style=True) == "[1, two]\n"
    assert py.safe_dump("é", encoding="utf-8") == b'"\\xE9"\n'
    assert py.safe_dump("é", allow_unicode=True) == "é\n...\n"
    assert list(py.safe_load_all("--- 1\n--- 2\n")) == [1, 2]


def test_nan_is_one_object() -> None:
    a, b = py.safe_load("[.nan, .NaN]")
    assert a is b
    assert py.safe_load("{.nan: 1, .NaN: 2}") == {a: 2}


def test_aliases_share_objects() -> None:
    doc = py.safe_load("a: &x [1]\nb: *x\nc: &y {}\nd: *y\n")
    assert doc["a"] is doc["b"]
    assert doc["c"] is doc["d"]
    rec = py.safe_load("&a [*a]")
    assert rec[0] is rec
    out = py.safe_dump([doc["a"], doc["a"]])
    assert out == "- &id001\n  - 1\n- *id001\n"


def test_merge_keys() -> None:
    doc = py.safe_load("base: &b {x: 1, y: 2}\nd:\n  <<: *b\n  y: 3\n")
    assert doc["d"] == {"x": 1, "y": 3}


def test_errors_have_marks() -> None:
    with pytest.raises(py.ScannerError) as info:
        py.safe_load("a: b: c")
    err = info.value
    assert err.problem == "mapping values are not allowed here"
    assert err.problem_mark is not None
    assert (err.problem_mark.line, err.problem_mark.column, err.problem_mark.index) == (0, 4, 4)
    assert err.problem_mark.name == "<unicode string>"
    assert err.problem_mark.buffer == "a: b: c\0"
    assert str(err).endswith("    a: b: c\n        ^")
    assert isinstance(err, py.MarkedYAMLError)
    assert isinstance(err, py.YAMLError)


def test_reader_error() -> None:
    with pytest.raises(py.ReaderError) as info:
        py.safe_load(b"a: \xff")
    assert (
        str(info.value)
        == 'unacceptable character #x00ff: invalid start byte\n  in "<byte string>", position 3'
    )
    with pytest.raises(py.ReaderError) as info:
        py.safe_load("a\x07")
    assert info.value.position == 1
    with pytest.raises(py.ReaderError):
        py.safe_load("a\ud800")


@needs_pyyaml
def test_exception_classes_are_pyyaml_classes() -> None:
    assert py.YAMLError is yaml.YAMLError
    assert py.ScannerError is yaml.scanner.ScannerError
    assert py.Mark is yaml.error.Mark
    assert py.SafeLoader is yaml.SafeLoader
    with pytest.raises(yaml.YAMLError):
        py.safe_load("[")


@needs_pyyaml
@pytest.mark.thread_unsafe
def test_customised_loader_falls_back() -> None:
    getattr(py, "_WARNED").clear()  # noqa: B009 - private, not in the stub

    class Loader(yaml.SafeLoader):  # type: ignore[misc]
        pass

    def upper(loader: Any, node: Any) -> str:
        return str(loader.construct_scalar(node)).upper()

    Loader.add_constructor("!up", upper)
    with pytest.warns(pyroid.FallbackWarning):
        assert py.load("!up abc", Loader=Loader) == "ABC"
    # A plain subclass runs natively.

    class Plain(yaml.SafeLoader):  # type: ignore[misc]
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        assert py.load("a: 1", Loader=Plain) == {"a": 1}


@needs_pyyaml
@pytest.mark.thread_unsafe
def test_full_loader_python_tags_fall_back() -> None:
    getattr(py, "_WARNED").clear()  # noqa: B009 - private, not in the stub
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        assert py.load("a: [1, x]", Loader=yaml.FullLoader) == {"a": [1, "x"]}
    with pytest.warns(pyroid.FallbackWarning):
        assert py.load("!!python/tuple [1, 2]", Loader=yaml.FullLoader) == (1, 2)
    # A file object is replayed to PyYAML (warned once per reason, already shown above).
    stream = io.StringIO("a: !!python/tuple [1]\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert py.load(stream, Loader=yaml.UnsafeLoader) == {"a": (1,)}


@needs_pyyaml
@pytest.mark.thread_unsafe
def test_customised_dumper_falls_back() -> None:
    getattr(py, "_WARNED").clear()  # noqa: B009 - private, not in the stub

    class Dumper(yaml.SafeDumper):  # type: ignore[misc]
        def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
            return super().increase_indent(flow, False)

    data = {"a": [1, 2]}
    with pytest.warns(pyroid.FallbackWarning):
        assert py.dump(data, Dumper=Dumper) == yaml.dump(data, Dumper=Dumper)


@needs_pyyaml
@pytest.mark.thread_unsafe
def test_full_dumper_objects_fall_back() -> None:
    import collections

    getattr(py, "_WARNED").clear()  # noqa: B009 - private, not in the stub
    data = collections.OrderedDict(a=1)
    with pytest.warns(pyroid.FallbackWarning):
        assert py.dump(data) == yaml.dump(data)
    assert py.dump((1, 2)) == "!!python/tuple\n- 1\n- 2\n"


def test_safe_dumper_rejects_objects() -> None:
    with pytest.raises(py.RepresenterError) as info:
        py.safe_dump(object())
    assert info.value.args[0] == "cannot represent an object"


def test_dump_to_streams() -> None:
    s = io.StringIO()
    assert py.safe_dump_all([1, 2], s) is None
    assert s.getvalue() == "1\n--- 2\n...\n"
    b = io.BytesIO()
    py.safe_dump("x", b, encoding="utf-16")
    # Python's utf-16 codec adds a BOM to every write, as PyYAML's writes show.
    assert b.getvalue().count(b"\xff\xfe") > 1
    if yaml is not None:
        ref = io.BytesIO()
        yaml.safe_dump("x", ref, encoding="utf-16")
        assert b.getvalue() == ref.getvalue()


def test_deep_nesting_raises_recursion_error() -> None:
    with pytest.raises(RecursionError):
        py.safe_load("[" * 100_000 + "]" * 100_000)
    nested: Any = []
    for _ in range(100_000):
        nested = [nested]
    with pytest.raises(RecursionError):
        py.safe_dump(nested)


def test_billion_laughs_shares_objects() -> None:
    lines = ["a0: &a0 [x]"]
    for i in range(1, 30):
        refs = ", ".join([f"*a{i - 1}"] * 9)
        lines.append(f"a{i}: &a{i} [{refs}]")
    doc = py.safe_load("\n".join(lines) + "\n")
    assert doc["a29"][0] is doc["a28"]


def test_huge_scalars() -> None:
    big = "x" * 20_000_000
    assert py.safe_load(big) == big
    assert py.safe_load("'" + big + "'") == big
    assert py.safe_load("1" * 3000) == int("1" * 3000)
    with pytest.raises(ValueError, match="Exceeds the limit"):
        py.safe_load("1" * 5000)


def test_threads_give_the_same_results() -> None:
    text = "a: [1, 2, {b: c}]\nd: 2001-12-14\n" * 200
    want = py.safe_load(text)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(20):
                assert py.safe_load(text) == want
                assert py.safe_load(py.safe_dump(want)) == want
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


@pytest.mark.thread_unsafe
def test_without_pyyaml(run_python: Any) -> None:
    code = r"""
import sys
sys.modules["yaml"] = None
import pyroid.yaml as py
assert py.YAMLError.__module__ == "pyroid.yaml"
print(py.safe_load("a: [1, x]"))
try:
    py.safe_load("a: b: c")
except py.ScannerError as e:
    print(repr(str(e)))
print(repr(py.safe_dump({"a": [1, 2]})))
try:
    py.load("a", Loader=py.BaseLoader)
except Exception as e:
    print(type(e).__name__)
"""
    r = run_python(code)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "{'a': [1, 'x']}"
    assert lines[1] == repr(
        "mapping values are not allowed here\n"
        '  in "<unicode string>", line 1, column 5:\n'
        "    a: b: c\n        ^"
    )
    assert lines[2] == repr("a:\n- 1\n- 2\n")
    assert lines[3] == "PyroidError"


@needs_pyyaml
@pytest.mark.thread_unsafe
def test_patch_swaps_and_restores() -> None:
    import pyroid.integrations as integrations

    before = yaml.safe_load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", integrations.PatchWarning)
        integrations.patch("yaml")
    try:
        assert yaml.safe_load is py.safe_load
        assert yaml.dump is py.dump
        assert yaml.safe_load("a: 1") == {"a": 1}
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load("[")
    finally:
        integrations.unpatch("yaml")
    assert yaml.safe_load is before


@pytest.mark.thread_unsafe
def test_patch_without_pyyaml(run_python: Any) -> None:
    code = r"""
import sys, warnings
sys.modules["yaml"] = None
import pyroid.integrations as integrations
del sys.modules["yaml"]
import importlib.machinery
class Block:
    def find_spec(self, name, path=None, target=None):
        if name == "yaml" or name.startswith("yaml."):
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, Block())
warnings.simplefilter("ignore")
integrations.patch("yaml")
import yaml
from yaml.scanner import ScannerError
print(yaml.safe_load("a: 1"), yaml.__version__, yaml.__pyroid__)
try:
    yaml.safe_load("a: b: c")
except ScannerError:
    print("ScannerError")
"""
    r = run_python(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["{'a': 1} 6.0.3 True", "ScannerError"]


def test_module_getattr() -> None:
    if yaml is None:
        with pytest.raises(AttributeError, match="PyYAML provides it"):
            py.scan  # noqa: B018
    else:
        assert py.scan is yaml.scan
    assert sys.modules["pyroid.yaml"] is py
