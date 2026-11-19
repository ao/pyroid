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
from pyroid._pyroid import json_repair as _native_jr

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


def _floats(value: Any) -> list[tuple[str, str | None]]:
    """The floats in `value`, in order, each with the json.decoder constant it is (if any)."""
    constants = {
        id(json.decoder.NaN): "NaN",  # type: ignore[attr-defined]
        id(json.decoder.PosInf): "PosInf",  # type: ignore[attr-defined]
        id(json.decoder.NegInf): "NegInf",  # type: ignore[attr-defined]
    }
    out: list[tuple[str, str | None]] = []
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, float):
            out.append((v.hex(), constants.get(id(v))))
        elif isinstance(v, dict):
            stack.extend(reversed(list(v.values())))
        elif isinstance(v, (list, tuple)):
            stack.extend(reversed(v))
    return out


@pytest.mark.parametrize(
    "doc",
    [
        "[NaN,NaN]]",
        "[NaN, NaN] x",
        "[1, NaN, NaN] ]",
        '{"a": NaN, "b": NaN}}',
        "[Infinity, -Infinity, 1e400, -1e400, NaN] and some text",
        '"a": NaN, "b": [Infinity, 1e999]}',  # the missing-opening-brace repair decodes it whole
        "[NaN]",
        "[NaN, 1",
        "{'a': NaN, 'b': [1e400]",
    ],
)
def test_json_constants_are_jsons_shared_objects(doc: str) -> None:
    # NaN, Infinity and -Infinity in a stretch of valid JSON come from json's decoder, which
    # returns one shared object for each; a number that overflows (1e400) is a new float.
    options: list[dict[str, Any]] = [{}, {"logging": True}, {"skip_json_loads": True}]
    for kwargs in options:
        kwargs = {"return_objects": True, **kwargs}
        got, want = jr.repair_json(doc, **kwargs), ref.repair_json(doc, **kwargs)
        assert same(got, want)
        assert _floats(got) == _floats(want), kwargs
    # Containers check identity before equality: with the shared NaN, results compare equal.
    assert (jr.loads(doc) == ref.loads(doc)) is (ref.loads(doc) == ref.loads(doc))
    assert (jr.loads(doc) == jr.loads(doc)) is (ref.loads(doc) == ref.loads(doc))


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
    load: Any = jr.load  # the option sets below cross the stubs' overloads
    repair: Any = jr.repair_json
    for kw in ({}, {"skip_json_loads": True}, {"logging": True}):
        fd, ref_fd = _Pipe("ignored" + doc), _Pipe("ignored" + doc)
        fd.read(7)
        ref_fd.read(7)
        assert same(load(fd, **kw), ref.load(ref_fd, **kw))
        assert same(load(_Pipe(doc), **kw), repair(doc, return_objects=True, **kw))


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


@pytest.mark.parametrize("unit", ["[", '{"a":', '{"a":[', "[{", "{[", "[[{", "[{["])
def test_max_depth_counts_levels_in_any_mix(unit: str) -> None:
    # An empty object re-read as an array is one level: "[{" * 334 used to raise at 668.
    per = unit.count("[") + unit.count("{")
    n = jr.MAX_DEPTH // per
    assert isinstance(jr.repair_json(unit * n), str)
    with pytest.raises(jr.NestingError):
        jr.repair_json(unit * (n + 1))


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
    col = _testing.column(["{}", "{'a': '\U0010fffe'"])
    with pytest.raises(pyroid.PyroidError, match="json_repair is not installed") as info:
        jr.repair_json_batch(col)
    assert any("row 1 of the Arrow input" in n for n in info.value.__notes__)


def _old_json_repair(calls: list[dict[str, Any]]) -> Any:
    """A stand-in for json-repair 0.30.0, whose repair_json has 7 arguments and no **kwargs."""
    import types

    module = types.ModuleType("json_repair")

    def repair_json(
        json_str: Any = "",
        return_objects: bool = False,
        skip_json_loads: bool = False,
        logging: bool = False,
        json_fd: Any = None,
        ensure_ascii: bool = True,
        chunk_length: int = 0,
    ) -> Any:
        calls.append(dict(locals()))
        return "old"

    module.repair_json = repair_json  # type: ignore[attr-defined]
    return module


@pytest.mark.thread_unsafe
def test_delegation_to_an_older_json_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    # Delegated calls passed 10 positional arguments, a TypeError below json-repair 0.58.
    untyped: Any = jr  # bytes input, which the stubs do not admit
    calls: list[dict[str, Any]] = []
    monkeypatch.setitem(sys.modules, "json_repair", _old_json_repair(calls))
    _reset_fallback_warnings()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert untyped.repair_json(b"{'a': 1") == "old"
        assert calls[-1]["json_str"] == b"{'a': 1"
        assert calls[-1]["ensure_ascii"] is True
        assert untyped.loads(b"[1, 2") == "old"
        assert calls[-1]["return_objects"] is True
        assert untyped.repair_json('["\U0010f800"', ensure_ascii=False) == "old"
        assert calls[-1]["ensure_ascii"] is False
        # Arguments left at their default are not passed, even when named.
        assert (
            untyped.loads(b"[1", strict=False, schema=None, schema_repair_mode="standard") == "old"
        )
        assert untyped.repair_json_batch(["{'a': '\U0010fffe'", "{}"]) == ["old", "{}"]
        assert calls[-1]["json_str"] == "{'a': '\U0010fffe'"
    assert len(calls) == 5
    for kwargs, release in [
        ({"schema": {"type": "object"}}, "0.56.0"),
        ({"strict": True}, "0.54"),
        ({"stream_stable": True}, "0.44.0"),
        ({"indent": 2}, "0.47.1"),
    ]:
        key = next(iter(kwargs))
        with pytest.raises(
            pyroid.PyroidError, match=rf"has no {key}= argument .*{re.escape(release)}"
        ):
            untyped.repair_json(b"{", **kwargs)
    with pytest.raises(pyroid.PyroidError, match="has no strict= argument"):
        untyped.repair_json_batch(["{'a': '\U0010fffe'"], strict=True)
    assert len(calls) == 5


@pytest.mark.thread_unsafe
def test_delegation_to_a_json_repair_without_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    # json-repair 0.47.1 to 0.53 take **json_dumps_args but not strict, schema or
    # schema_repair_mode: those must not reach json.dumps.
    import types

    untyped: Any = jr  # bytes input, which the stubs do not admit
    calls: list[dict[str, Any]] = []
    module = types.ModuleType("json_repair")

    def repair_json(
        json_str: Any = "",
        return_objects: bool = False,
        skip_json_loads: bool = False,
        logging: bool = False,
        json_fd: Any = None,
        chunk_length: int = 0,
        stream_stable: bool = False,
        **json_dumps_args: Any,
    ) -> Any:
        calls.append({**locals(), **json_dumps_args})
        json.dumps(None, **json_dumps_args)
        return "0.50"

    module.repair_json = repair_json  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "json_repair", module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert untyped.repair_json(b"{", indent=2, stream_stable=True) == "0.50"
        assert calls[-1]["indent"] == 2
        assert calls[-1]["stream_stable"] is True
        assert not {"strict", "schema", "schema_repair_mode"} & calls[-1].keys()
        with pytest.raises(pyroid.PyroidError, match=r"has no strict= argument .*0\.54"):
            untyped.repair_json(b"{", strict=True)
    assert len(calls) == 1


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


# --- the json module and str subclasses ----------------------------------------------


def _outcome(f: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return ("ok", f(*args, **kwargs))
    except Exception as e:
        return ("raise", type(e).__name__, str(e))


def _calls(lib: Any) -> list[Any]:
    """Results of the calls json_repair makes json.loads and json.dumps matter for."""
    docs = ['{"a": 1}', "{a: 1, 'b': [1, 2,", "", "nothing", '[1, 2] {"a": "b"}']
    out = []
    for doc in docs:
        out.append(_outcome(lib.repair_json, doc))
        out.append(_outcome(lib.repair_json, doc, indent=2))
        out.append(_outcome(lib.repair_json, doc, skip_json_loads=True))
        out.append(_outcome(lib.repair_json, doc, return_objects=True))
        out.append(_outcome(lib.loads, doc))
        out.append(_outcome(lib.repair_json, json_fd=io.StringIO(doc)))
        out.append(_outcome(lib.load, io.StringIO(doc)))
    return out


_BATCH_OPTIONS: list[dict[str, Any]] = [{}, {"indent": 2}, {"skip_json_loads": True}]


def _batch_calls() -> list[Any]:
    """repair_json_batch over the same documents, as a list, a tuple, a str and Arrow."""
    docs = ['{"a": 1}', "{a: 1, 'b': [1, 2,", "", "nothing", '[1, 2] {"a": "b"}']
    out = []
    for kwargs in _BATCH_OPTIONS:
        out.append(_outcome(jr.repair_json_batch, docs, **kwargs))
        out.append(_outcome(jr.repair_json_batch, tuple(docs), **kwargs))
        out.append(_outcome(jr.repair_json_batch, docs[1], **kwargs))
        col = _outcome(jr.repair_json_batch, _testing.column(docs, chunks=2), **kwargs)
        out.append((col[0], col[1].to_pylist()) if col[0] == "ok" else col)
    return out


def _ref_batch_calls() -> list[Any]:
    docs = ['{"a": 1}', "{a: 1, 'b': [1, 2,", "", "nothing", '[1, 2] {"a": "b"}']
    out = []
    for kwargs in _BATCH_OPTIONS:
        want = [ref.repair_json(d, **kwargs) for d in docs]
        out += [("ok", want), ("ok", tuple(want)), ("ok", want[1]), ("ok", want)]
    return out


@pytest.mark.thread_unsafe(reason="replaces json module functions")
@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("dumps", lambda orig: lambda *a, **k: "DUMPED:" + orig(*a, **k)),
        ("dumps", lambda orig: lambda *a, **k: orig(*a, **{**k, "separators": (",", ":")})),
        ("loads", lambda orig: lambda *a, **k: {"patched": True}),
        ("loads", lambda orig: lambda s, *a, **k: orig(s.replace("1", "7"), *a, **k)),
    ],
    ids=["dumps-prefix", "dumps-compact", "loads-constant", "loads-edit"],
)
def test_replaced_json_functions_are_used_as_json_repair_uses_them(
    monkeypatch: pytest.MonkeyPatch, name: str, replacement: Any
) -> None:
    # json_repair calls json.loads and json.dumps through the module at each call, so a program
    # that swaps them (an orjson-backed shim, say) gets their results. String results came from
    # the Rust port's own decoder and encoder, which ignored the swap.
    monkeypatch.setattr(json, name, replacement(getattr(json, name)))
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)  # the Python route, not json_repair
        assert _calls(jr) == _calls(ref)
        assert _batch_calls() == _ref_batch_calls()


@pytest.mark.thread_unsafe(reason="replaces json module functions")
def test_replaced_json_decoder_and_load_are_delegated(monkeypatch: pytest.MonkeyPatch) -> None:
    # json_repair decodes stretches of valid JSON inside broken input with json.JSONDecoder, and
    # a file with json.load: the port cannot call them, so such calls go to json_repair.
    class Decoder(json.JSONDecoder):
        def raw_decode(self, s: str, idx: int = 0) -> tuple[Any, int]:
            value, end = super().raw_decode(s, idx)
            return ["decoded", value], end

    orig_load = json.load
    _reset_fallback_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        monkeypatch.setattr(json, "JSONDecoder", Decoder)
        assert _calls(jr) == _calls(ref)
        assert _batch_calls() == _ref_batch_calls()
        monkeypatch.undo()
        monkeypatch.setattr(json, "load", lambda fd, **k: ["load", orig_load(fd, **k)])
        assert _calls(jr) == _calls(ref)
    reasons = {str(w.message).split(" does not handle ")[1].split(";")[0] for w in caught}
    assert reasons == {"a replaced json.JSONDecoder", "a replaced json.load"}
    assert {w.category for w in caught} == {pyroid.FallbackWarning}


def test_json_functions_replaced_before_the_import_are_used(tmp_path: Path) -> None:
    # Replaced before pyroid.json_repair was imported: the import cannot take them as json's own.
    import subprocess

    script = tmp_path / "early.py"
    script.write_text(
        "import json\n"
        "orig = json.dumps\n"
        "json.dumps = lambda *a, **k: 'DUMPED:' + orig(*a, **k)\n"
        "import pyroid.json_repair as jr\n"
        "import json_repair as ref\n"
        "for doc in ['{\"a\": 1}', '{a: 1', '']:\n"
        "    assert jr.repair_json(doc) == ref.repair_json(doc), doc\n"
        "    assert jr.repair_json_batch([doc]) == [ref.repair_json(doc)], doc\n"
        "print('ok')\n"
    )
    run = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False, timeout=120
    )
    assert run.stdout.strip() == "ok", run.stderr


class _Upper(str):
    """Characters read through ``__getitem__`` come back upper-case (json_repair reads them so)."""

    __slots__ = ()

    def __getitem__(self, key: Any) -> str:
        part = str.__getitem__(self, key)
        return part.upper() if isinstance(key, int) else part


class _Plain(str):
    __slots__ = ()


@pytest.mark.thread_unsafe(reason="global one-time-warning state")
def test_str_subclass_overriding_what_json_repair_calls_is_delegated() -> None:
    class Find(str):
        def find(self, *args: Any) -> int:
            return -1

    class Length(str):
        def __len__(self) -> int:
            return 3

    docs = ["{a: 'b'}", '{"a": "b"}', "{'key': \"value\", x: [1, 2", "[NaN] trailing"]
    _reset_fallback_warnings()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cls in (_Upper, Find, Length):
            for doc in docs:
                for kwargs in [{}, {"return_objects": True}, {"skip_json_loads": True}]:
                    got = _outcome(jr.repair_json, cls(doc), **kwargs)
                    assert got == _outcome(ref.repair_json, cls(doc), **kwargs), (cls, doc)
            want = [ref.repair_json(cls(d)) for d in docs]
            assert jr.repair_json_batch([cls(d) for d in docs]) == want
            assert jr.repair_json_batch(tuple(cls(d) for d in docs)) == tuple(want)
            assert jr.repair_json_batch(cls(docs[0])) == want[0]
    reasons = {str(w.message).split(" does not handle ")[1].split(";")[0] for w in caught}
    assert reasons == {
        "a str subclass overriding __getitem__()",
        "a str subclass overriding find()",
        "a str subclass overriding __len__()",
    }
    # A subclass that overrides nothing json_repair calls stays on the Rust path.
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        for doc in docs:
            assert jr.repair_json(_Plain(doc)) == ref.repair_json(_Plain(doc))
        assert jr.repair_json_batch([_Plain(d) for d in docs]) == [ref.repair_json(d) for d in docs]


def test_str_subclasses_seen_by_the_rust_path() -> None:
    assert _native_jr._overridden_str_hook("abc") is None
    assert _native_jr._overridden_str_hook(_Plain("abc")) is None
    assert _native_jr._overridden_str_hook(_Upper("abc")) == "__getitem__"

    class Deeper(_Upper):
        pass

    assert _native_jr._overridden_str_hook(Deeper("abc")) == "__getitem__"

    class WithDict(str):
        pass

    s = WithDict("abc")
    s.startswith = lambda *a: True  # type: ignore[method-assign]
    assert _native_jr._overridden_str_hook(s) == "startswith"
    np = pytest.importorskip("numpy")
    assert _native_jr._overridden_str_hook(np.str_("abc")) is None


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


def _first_error(docs: Any, **kwargs: Any) -> Any:
    """What a loop over json_repair.repair_json raises first: (index, type, message)."""
    for i, doc in enumerate(docs):
        if doc is None:
            continue
        try:
            ref.repair_json(doc, **kwargs)
        except Exception as e:
            return i, type(e), str(e)
    return None


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@pytest.mark.parametrize(
    "docs",
    [
        ["", "\U0010ffff{[1, 2", "[1e, 2"],  # P67: the delegated document at index 1 fails first
        ["[1e, 2", "\U0010ffff{[1, 2"],
        ["{}", "\U0010ffff[1]", "{'a' 1}", "[1e, 2", "\U0010ffff{[1, 2"],
        ["\U0010ffff{[1, 2", "\U0010ffff[1e, 2"],
        [_Upper("{a: 1}"), _Upper("[1e, 2"), "[1e, 2"],
    ],
)
def test_batch_raises_the_first_failing_document(docs: list[str]) -> None:
    first = _first_error(docs, strict=True)
    assert first is not None
    index, error, message = first
    inputs: list[Any] = [docs, tuple(docs)]
    if not any(type(d) is _Upper for d in docs):
        inputs.append(_testing.column(docs, chunks=2))  # Arrow has no str subclasses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        for texts in inputs:
            with pytest.raises(error) as info:
                jr.repair_json_batch(texts, strict=True)
            assert str(info.value) == message, type(texts)
            where = "the document at index" if isinstance(texts, (list, tuple)) else "row"
            assert f"pyroid: while repairing {where} {index}" in " ".join(
                getattr(info.value, "__notes__", [])
            ), type(texts)
        with pytest.raises(error, match=re.escape(message)):
            jr.repair_json_batch(docs[index], strict=True)


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
def test_batch_python_route_keeps_the_order_and_takes_arrow() -> None:
    # json.dumps arguments the port does not reproduce: one repair_json call per document, which
    # now also works for Arrow input, with the same notes.
    docs: list[str | None] = ["{'a': 1", None, "[1, 2,", '{"b": 2}']
    kwargs: dict[str, Any] = {"cls": json.JSONEncoder, "indent": 1}
    want = [None if d is None else ref.repair_json(d, **kwargs) for d in docs]
    col = jr.repair_json_batch(_testing.column(docs, chunks=2), **kwargs)
    assert col.to_pylist() == want
    assert jr.repair_json_batch([d for d in docs if d], **kwargs) == [w for w in want if w]
    bad = ["{}", "[1e, 2", "{'a' 1}"]
    for texts in (bad, _testing.column(bad)):
        with pytest.raises(ValueError, match="Dangling exponent") as info:
            jr.repair_json_batch(texts, strict=True, **kwargs)
        assert any(" 1" in n for n in info.value.__notes__)


def test_batch_surrogates() -> None:
    docs = ['{"a": "\ud800"}', "{'a': 'x'}", "\udcff"]
    assert jr.repair_json_batch(docs) == [ref.repair_json(d) for d in docs]


@pytest.mark.parametrize("large", [False, True])
def test_batch_arrow_result_with_lone_surrogate(large: bool) -> None:
    # json_repair keeps the escape \ud800 (and an escaped pair it parses itself) as lone
    # surrogates; with ensure_ascii=False its str holds them, which Arrow cannot. The port
    # stores them in U+10F800..U+10FFFF, which must never reach the column.
    docs = ['{"ok": 1,}', '{"emoji": "\\ud83d\\ude00", "ok": true,}', '"\\ud800"']
    col = _testing.column(docs, large=large, chunks=2)
    want = [ref.repair_json(d, ensure_ascii=False) for d in docs]
    assert jr.repair_json_batch(docs, ensure_ascii=False) == want
    with pytest.raises(UnicodeEncodeError) as info:
        jr.repair_json_batch(col, ensure_ascii=False)
    assert info.value.object == want[1]
    assert any("row 1 of the Arrow input" in n for n in info.value.__notes__)
    for kw in [{}, {"ensure_ascii": True}]:
        assert jr.repair_json_batch(col, **kw).to_pylist() == [ref.repair_json(d) for d in docs]
    # A result without one is unaffected.
    assert jr.repair_json_batch(_testing.column(docs[:1]), ensure_ascii=False).to_pylist() == [
        want[0]
    ]


def test_batch_arrow_error_names_the_row() -> None:
    col = _testing.column(["{}", None, "[1]", '{"a" 1}'], chunks=2)
    with pytest.raises(ValueError, match="Missing ':'") as info:
        jr.repair_json_batch(col, strict=True, skip_json_loads=True)
    assert any("row 3 of the Arrow input" in n for n in info.value.__notes__)


@pytest.mark.thread_unsafe
def test_batch_delegates_reserved_items() -> None:
    docs = ["{'a': 1}", "{'a': '\U0010fffe'}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        assert jr.repair_json_batch(docs) == [ref.repair_json(d) for d in docs]


@pytest.mark.thread_unsafe
@pytest.mark.parametrize("large", [False, True])
def test_batch_arrow_delegates_only_the_reserved_rows(large: bool) -> None:
    # It raised PyroidError for the whole column; a list delegated just those documents.
    docs: list[str | None] = [
        "{'a': 1",
        "{'a': '\U0010fffe'",
        None,
        '"\\udbff\\udfff"',
        "[1, 2,",
    ]
    want = [None if d is None else ref.repair_json(d) for d in docs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        col = jr.repair_json_batch(_testing.column(docs, large=large, chunks=2))
        assert col.to_pylist() == want
        got = jr.repair_json_batch(
            _testing.column(docs, chunks=2), ensure_ascii=False, indent=1
        ).to_pylist()
    assert got == [
        None if d is None else ref.repair_json(d, ensure_ascii=False, indent=1) for d in docs
    ]


def test_constants() -> None:
    assert jr.JSON_REPAIR_VERSION == "0.64.9"
    assert jr.DELEGATED == ("schema",)
    assert jr.MAX_DEPTH == 1000
    assert jr.repair_json.__module__ == "pyroid.json_repair"
