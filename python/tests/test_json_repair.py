# Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
"""pyroid.json_repair: the API surface, options, errors, delegation and the batch function.

Results are compared with json_repair 0.64.9 where it is installed (the test group pins it);
test_json_repair_parity.py and test_json_repair_upstream.py hold the differential suites.
"""

from __future__ import annotations

import io
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

import pyroid
import pyroid.json_repair as jr
from pyroid._pyroid import _testing

ref: Any = pytest.importorskip("json_repair")

BROKEN = [
    "{'a': 1, b: [1, 2,}",
    '{"a": "unterminated',
    "[1, 2, 3",
    'Here you go: ```json\n{"x": [1, 2]}\n``` hope it helps',
    '{"a": True, "b": None, "c": False}',
    '{"text": "He said "hi" to me"}',
    '{"a": 1 /* comment */, # another\n "b": 2}',
    "{'nested': {'deep': [1, {'x': 'y'}]}, 'n': -1.5e3}",
    '{"a": 1}{"a": 2}',
    '"a": 1, "b": 2}',
    "",
    "no json here",
    '{"emoji": "😀", "cjk": "日本", "esc": "\\u00e9\\n"}',
    "(1, 2)",
    '[{"a": 1, "a": 2, "a": 3}]',
]
VALID = ['{"a": 1, "b": [true, false, null], "c": "x"}', "[1.0, 2e5, -0.0, 12345678901234567890]"]


def same(a: Any, b: Any) -> bool:
    """Type-exact equality (int vs float vs bool, NaN, dict key order)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict):
        return list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    return bool(a == b)


@pytest.mark.parametrize("doc", BROKEN + VALID)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"skip_json_loads": True},
        {"stream_stable": True},
        {"ensure_ascii": False},
        {"indent": 2},
        {"indent": "\t", "sort_keys": True},
        {"separators": (",", ":")},
    ],
)
def test_repair_json_string_matches_json_repair(doc: str, kwargs: dict[str, Any]) -> None:
    assert jr.repair_json(doc, **kwargs) == ref.repair_json(doc, **kwargs)


@pytest.mark.parametrize("doc", BROKEN + VALID)
def test_objects_and_logging_match_json_repair(doc: str) -> None:
    assert same(jr.loads(doc), ref.loads(doc))
    assert same(jr.repair_json(doc, return_objects=True), ref.repair_json(doc, return_objects=True))
    got = jr.repair_json(doc, logging=True)
    want = ref.repair_json(doc, logging=True)
    assert isinstance(got, tuple)
    assert same(got[0], want[0])
    assert got[1] == want[1]
    assert same(jr.loads(doc, logging=True), ref.loads(doc, logging=True))


def test_types_of_numbers_and_literals() -> None:
    big = "123456789012345678901234567890"
    value = jr.loads(f'{{"i": 1, "f": 1.0, "e": 1e2, "big": {big}, "t": true}}')
    assert isinstance(value, dict)
    assert type(value["i"]) is int
    assert type(value["f"]) is float
    assert type(value["e"]) is float
    assert value["big"] == 123456789012345678901234567890
    assert value["t"] is True
    nan = jr.loads("[NaN, Infinity, -Infinity]")
    assert isinstance(nan, list)
    assert math.isnan(nan[0])
    assert nan[1:] == [math.inf, -math.inf]
    assert jr.repair_json("[NaN]") == "[NaN]"
    # Duplicate keys: last value, first position (dict semantics), as json.loads.
    assert list(jr.loads('{"a": 1, "b": 2, "a": 3}').items()) == [("a", 3), ("b", 2)]  # type: ignore[union-attr]


def test_empty_result_is_empty_string() -> None:
    assert jr.repair_json("") == ""
    assert jr.repair_json("nothing") == ""
    assert jr.repair_json('"unterminated') == ""
    assert jr.loads("nothing") == ""


@pytest.mark.parametrize("strict", [False, True])
def test_valid_empty_string_is_json(strict: bool) -> None:
    # json_repair 0.64.6: valid JSON holding "" is written as JSON; only a failed repair gives "".
    assert jr.repair_json('""', strict=strict) == '""'
    assert jr.repair_json('""', strict=strict, indent=2, sort_keys=True) == '""'
    assert jr.repair_json('""', strict=strict, cls=json.JSONEncoder) == '""'
    assert jr.repair_json('""', strict=strict, return_objects=True) == ""
    assert jr.repair_json('""', strict=strict, logging=True) == ("", [])
    assert jr.repair_json('""', skip_json_loads=True) == ""
    assert jr.repair_json_batch(['""', "x"]) == ['""', ""]


@pytest.mark.parametrize(
    ("doc", "expected", "log"),
    [
        # json_repair 0.64.8: consecutive commas after an object value's number.
        ('{"a": 1,, "b": 2}', {"a": 1, "b": 2}, "Found consecutive commas after an object number"),
        ('{"a":1,,2,"b":3}', {"a": 1, "b": 3}, "Discarded stray number between object fields"),
        ('{"a": 1,,2}', {"a": 1}, "Found consecutive commas after an object number"),
        # json_repair 0.64.8: a dangling exponent is dropped.
        ("[1e, 2]", [1, 2], "Discarded dangling exponent from number"),
        (
            '{"n": 1.5E-, "next": 2}',
            {"n": 1.5, "next": 2},
            "Discarded dangling exponent from number",
        ),
    ],
)
def test_number_repairs(doc: str, expected: Any, log: str) -> None:
    for skip in (False, True):
        value, entries = jr.repair_json(doc, logging=True, skip_json_loads=skip)
        assert value == expected
        assert log in [e["text"] for e in entries]
        assert same(
            jr.loads(doc, logging=True, skip_json_loads=skip),
            ref.loads(doc, logging=True, skip_json_loads=skip),
        )
        with pytest.raises(ValueError, match="strict mode") as info:
            jr.repair_json(doc, strict=True, skip_json_loads=skip)
        with pytest.raises(ValueError, match=f"^{re.escape(str(info.value))}$"):
            ref.repair_json(doc, strict=True, skip_json_loads=skip)


class _Pipe(io.StringIO):
    """A text stream that cannot seek (json_repair 0.64.6 reads those whole, as a str)."""

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        raise OSError("not seekable")


@pytest.mark.parametrize("doc", ['{"key": 1}', '{"key": }', 'text [1, 2] {"a": 1} more'])
def test_load_non_seekable_stream(doc: str) -> None:
    for kw in ({}, {"skip_json_loads": True}, {"logging": True}):
        fd, ref_fd = _Pipe("ignored" + doc), _Pipe("ignored" + doc)
        fd.read(7)
        ref_fd.read(7)
        assert same(jr.load(fd, **kw), ref.load(ref_fd, **kw))
        assert same(jr.load(_Pipe(doc), **kw), jr.repair_json(doc, return_objects=True, **kw))


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ('{"a": 1, "a": 2}', None),
        ('{"a" 1}', "Missing ':' after key in strict mode while parsing object."),
        ("[1] [2]", "Multiple top-level JSON elements found in strict mode."),
        ('[{"a": 1, "a": 2}]', "Duplicate key found in strict mode while parsing object."),
        ('{"": 1}', None),
    ],
)
def test_strict_mode_errors_match(doc: str, message: str | None) -> None:
    def outcome(f: Any) -> Any:
        try:
            return ("ok", f(doc, strict=True, skip_json_loads=True))
        except ValueError as e:
            return ("ValueError", str(e))

    got, want = outcome(jr.repair_json), outcome(ref.repair_json)
    assert got == want
    if message is not None:
        assert got == ("ValueError", message)


def test_argument_checks_match_json_repair() -> None:
    with pytest.raises(ValueError, match="requires schema"):
        jr.repair_json("{}", schema_repair_mode="salvage")
    with pytest.raises(ValueError, match="cannot be used together"):
        jr.repair_json("{}", schema={"type": "object"}, strict=True)
    with pytest.raises(ValueError, match="must be one of: standard, salvage"):
        jr.repair_json("{}", schema_repair_mode="other")  # type: ignore[call-overload]


def test_dumps_kwargs() -> None:
    doc = "{'b': [1, {'c': NaN}], 'a': 'é'}"
    cases: list[dict[str, Any]] = [
        {"indent": 0},
        {"indent": -1},
        {"indent": True},
        {"separators": ("; ", "= ")},
        {"skipkeys": True, "check_circular": False, "default": str},
        {"cls": json.JSONEncoder},
    ]
    for kwargs in cases:
        assert jr.repair_json(doc, **kwargs) == ref.repair_json(doc, **kwargs), kwargs
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        jr.repair_json("[1, NaN]", allow_nan=False)
    with pytest.raises(ValueError, match="not JSON compliant") as info:
        ref.repair_json("[1, NaN]", allow_nan=False)
    with pytest.raises(ValueError, match=f"^{info.value}$"):
        jr.repair_json("[1, NaN]", allow_nan=False)
    with pytest.raises(TypeError):
        jr.repair_json(doc, no_such_argument=1)
    # Ignored when an object is returned, as in json_repair.
    assert jr.repair_json(doc, return_objects=True, no_such_argument=1) is not None


def test_load_and_from_file(tmp_path: Path) -> None:
    doc = "{'a': [1, 2,], b: 'x'}"
    path = tmp_path / "broken.json"
    path.write_text(doc)
    assert same(jr.from_file(path), ref.from_file(path))
    assert same(jr.from_file(str(path), logging=True), ref.from_file(str(path), logging=True))
    assert same(jr.load(io.StringIO(doc)), ref.load(io.StringIO(doc)))
    assert same(jr.load(io.StringIO('{"a": 1}')), {"a": 1})
    assert jr.repair_json(json_fd=io.StringIO(doc)) == ref.repair_json(json_fd=io.StringIO(doc))


def test_nesting_error() -> None:
    assert issubclass(jr.NestingError, pyroid.PyroidError)
    assert issubclass(jr.NestingError, ValueError)
    assert issubclass(jr.NestingError, RecursionError)
    assert jr.NestingError.__module__ == "pyroid.json_repair"
    depth = jr.MAX_DEPTH
    # Inner empty arrays are dropped, as json_repair does (it stops at ~330 levels).
    assert jr.repair_json("[" * 200) == ref.repair_json("[" * 200) == "[]"
    assert jr.repair_json("[" * (depth - 1)) == "[]"
    assert jr.repair_json("[1, " * (depth - 1)).startswith("[1, [1, [1, ")
    with pytest.raises(jr.NestingError, match="exceeds the supported parser recursion depth"):
        jr.repair_json("[" * (depth + 1))
    nested = '{"a": ' * (depth + 1)
    if sys.version_info >= (3, 12):
        with pytest.raises(ValueError, match="exceeds the supported parser recursion depth"):
            jr.loads(nested)
    else:
        # 3.11's json.loads, json_repair's first step, raises RecursionError on it first.
        for f in (jr.loads, ref.loads):
            with pytest.raises(RecursionError, match="while decoding a JSON object"):
                f(nested)
    # A valid JSON string result comes from the Rust decoder, which has no depth limit.
    deep = "[" * 100_000 + "]" * 100_000
    assert jr.repair_json(deep) == deep.replace("][", "], [")

    # Values come from json.loads, as in json_repair (a RecursionError on some Pythons).
    def outcome(f: Any) -> str:
        try:
            f(deep)
        except RecursionError:
            return "RecursionError"
        return "ok"

    assert outcome(jr.loads) == outcome(ref.loads)


def test_surrogates_match_json_repair() -> None:
    docs = ['{"a": "\ud800"}', "{'a': 'x\udfff', b: '\\ud83d'}", '["\\ud83d\\ude00", "\\udc00"]']
    options: list[dict[str, Any]] = [{}, {"ensure_ascii": False}, {"skip_json_loads": True}]
    for doc in docs:
        for kwargs in options:
            assert jr.repair_json(doc, **kwargs) == ref.repair_json(doc, **kwargs)
        assert same(jr.loads(doc), ref.loads(doc))


def _reset_fallback_warnings() -> None:
    """Forget which fallbacks warned already (they warn once per process)."""
    vars(jr)["_WARNED"].clear()


@pytest.mark.thread_unsafe
def test_reserved_code_points_are_delegated() -> None:
    doc = "{'a': '\U0010fffe'}"
    _reset_fallback_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert jr.repair_json(doc) == ref.repair_json(doc)
        assert jr.loads('"\U0010f800"') == ref.loads('"\U0010f800"')
        assert jr.load(io.StringIO('["\U0010ffff"]')) == ["\U0010ffff"]
        assert jr.repair_json('"\\udbff\\udfff"') == ref.repair_json('"\\udbff\\udfff"')
    categories = [w.category for w in caught]
    assert pyroid.FallbackWarning in categories
    assert all(c is pyroid.FallbackWarning for c in categories)


@pytest.mark.thread_unsafe
def test_schema_is_delegated() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
    _reset_fallback_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got = jr.repair_json("{'a': 1}", schema=schema)
        except ValueError as e:  # json_repair needs jsonschema for this
            got = str(e)
        try:
            want = ref.repair_json("{'a': 1}", schema=schema)
        except ValueError as e:
            want = str(e)
    assert got == want
    assert [w.category for w in caught if w.category is pyroid.FallbackWarning]


@pytest.mark.thread_unsafe
def test_delegation_without_json_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "json_repair", None)
    with pytest.raises(pyroid.PyroidError, match="json_repair is not installed"):
        jr.repair_json("{}", schema={"type": "object"})
    # The Rust port itself needs nothing.
    assert jr.repair_json("{'a': 1}") == '{"a": 1}'


@pytest.mark.thread_unsafe
def test_int_max_str_digits_follows_sys() -> None:
    doc = '{"a": ' + "1" * 5000 + ', "b": }'
    old = sys.get_int_max_str_digits()
    try:
        for limit in [0, 4300, 6000]:
            sys.set_int_max_str_digits(limit)
            assert jr.repair_json(doc) == ref.repair_json(doc)
            assert jr.repair_json(doc[:-8] + "}") == ref.repair_json(doc[:-8] + "}")
    finally:
        sys.set_int_max_str_digits(old)


# --- repair_json_batch ---------------------------------------------------------------


def test_batch_kinds_and_identity() -> None:
    docs = [*BROKEN, *VALID, '{"a": 1}']
    want = [ref.repair_json(d) for d in docs]
    out = jr.repair_json_batch(docs)
    assert type(out) is list
    assert out == want
    # A document whose repair equals it comes back as the same object.
    assert out[-1] is docs[-1]
    out_t = jr.repair_json_batch(tuple(docs))
    assert type(out_t) is tuple
    assert list(out_t) == want
    assert jr.repair_json_batch(docs[0]) == want[0]
    assert jr.repair_json_batch(docs, indent=2) == [ref.repair_json(d, indent=2) for d in docs]
    assert jr.repair_json_batch(docs, cls=json.JSONEncoder) == want
    assert jr.repair_json_batch(docs, skip_json_loads=True) == [
        ref.repair_json(d, skip_json_loads=True) for d in docs
    ]


@pytest.mark.parametrize(("large", "chunks"), [(False, 1), (True, 3)])
def test_batch_arrow(large: bool, chunks: int) -> None:
    docs: list[str | None] = [*BROKEN, None, *VALID]
    col = jr.repair_json_batch(_testing.column(docs, large=large, chunks=chunks, name="raw"))
    assert col.to_pylist() == [None if d is None else ref.repair_json(d) for d in docs]
    assert col.type == ("large_string" if large else "string")
    assert col.name == "raw"


def test_batch_threads_and_large_batch() -> None:
    docs = [f"{{'id': {i}, 'text': 'item {i}', tags: [1, 2,"[: 10 + i % 25] for i in range(10_000)]
    want = jr.repair_json_batch(docs, threads=1)
    assert want == [ref.repair_json(d) for d in docs[:500]] + want[500:]
    for threads in [None, 2, 8]:
        assert jr.repair_json_batch(docs, threads=threads) == want


def test_batch_errors() -> None:
    with pytest.raises(ValueError, match="Missing ':'") as info:
        jr.repair_json_batch(["{}", '{"a" 1}'], strict=True, skip_json_loads=True)
    assert any("index 1" in note for note in info.value.__notes__)
    with pytest.raises(pyroid.InputError, match="index 1") as err:
        jr.repair_json_batch(["{}", 3])  # type: ignore[list-item]
    assert "while processing 'texts'" in err.value.__notes__
    with pytest.raises(jr.NestingError):
        jr.repair_json_batch(["[1]", "[" * (jr.MAX_DEPTH + 1)])
    with pytest.raises(ValueError, match="threads"):
        jr.repair_json_batch(["{}"], threads=0)


def test_batch_surrogates() -> None:
    docs = ['{"a": "\ud800"}', "{'a': 'x'}", "\udcff"]
    assert jr.repair_json_batch(docs) == [ref.repair_json(d) for d in docs]


@pytest.mark.thread_unsafe
def test_batch_delegates_reserved_items() -> None:
    docs = ["{'a': 1}", "{'a': '\U0010fffe'}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert jr.repair_json_batch(docs) == [ref.repair_json(d) for d in docs]


def test_constants() -> None:
    assert jr.JSON_REPAIR_VERSION == "0.64.9"
    assert jr.DELEGATED == ("schema",)
    assert jr.MAX_DEPTH == 1000
    assert jr.repair_json.__module__ == "pyroid.json_repair"
