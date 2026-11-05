"""Arrow interop: importing string columns, the ``pyroid.Column`` result type,
and exporting it to other Arrow libraries.

The first half needs no Arrow library (``_testing.column`` builds Columns in
Rust); the pyarrow and polars halves skip when those are not installed.
"""

import gc
import re
import struct
import types
import typing
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import pyroid
import pyroid.text as t
from pyroid._pyroid import _testing


def ref(s: str | None) -> str | None:
    return None if s is None else " ".join(s.split())


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


def _min_chars_1(x: Any) -> object:
    return t.min_chars(x, 1)


#: Every entry point that takes documents (transform, predicate, map_values).
FUNCS: list[Callable[[Any], object]] = [t.normalize_whitespace, _min_chars_1, _testing.char_lengths]


VALUES: list[str | None] = [
    "  a  b ",
    None,
    "",
    "ok",
    "\xe9\t\xfc",
    "a much longer document than twelve bytes",  # stored out of line in string_view
    None,
    "  \U0001f600  ",
    "x" * 100,
    "nul\x00 inside",  # NUL is an ordinary character in Arrow and Python
]


# ---- Column, without any Arrow library ---------------------------------------


@pytest.mark.parametrize("large", [False, True], ids=["string", "large_string"])
@pytest.mark.parametrize("chunks", [1, 3])
def test_column_api(large: bool, chunks: int) -> None:
    col = _testing.column(VALUES, large=large, chunks=chunks, name="text")
    assert isinstance(col, pyroid.Column)
    assert len(col) == len(VALUES)
    assert col.type == ("large_string" if large else "string")
    assert col.name == "text"
    assert col.null_count == 2
    assert col.num_chunks == chunks
    assert col.to_pylist() == VALUES
    kind = "large_string" if large else "string"
    want = f"pyroid.Column<{kind}>(len={len(VALUES)}, null_count=2, num_chunks={chunks})"
    assert repr(col) == want


def test_column_is_generic_at_runtime_but_not_constructible() -> None:
    alias: Any = pyroid.Column[str]
    assert isinstance(alias, types.GenericAlias)
    assert typing.get_origin(alias) is pyroid.Column
    assert typing.get_args(pyroid.Column[bool]) == (bool,)
    ctor: Any = pyroid.Column
    with pytest.raises(TypeError):
        ctor()
    with pytest.raises(TypeError):
        type("Sub", (pyroid.Column,), {})


def test_column_is_read_only() -> None:
    col: Any = _testing.column(["a"])
    with pytest.raises(AttributeError):
        col.name = "other"


def test_capsules() -> None:
    col = _testing.column(["a", None], chunks=2)
    stream = col.__arrow_c_stream__()
    assert type(stream).__name__ == "PyCapsule"
    assert col.__arrow_c_stream__(None) is not stream
    schema, array = col.__arrow_c_array__()
    assert type(schema).__name__ == "PyCapsule"
    assert type(array).__name__ == "PyCapsule"
    # Unconsumed capsules release their structs when they are collected.
    del stream, schema, array
    gc.collect()
    assert col.to_pylist() == ["a", None]


@pytest.mark.parametrize("large", [False, True], ids=["string", "large_string"])
@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_transform_and_predicate_on_columns(large: bool, chunks: int) -> None:
    col = _testing.column(VALUES, large=large, chunks=chunks, name="body")
    out = t.normalize_whitespace(col)
    assert isinstance(out, pyroid.Column)
    assert out.type == ("large_string" if large else "string")
    assert out.name == "body"
    assert out.num_chunks == col.num_chunks
    assert out.null_count == 2
    assert out.to_pylist() == [ref(v) for v in VALUES]
    mask = t.min_chars(col, 2)
    assert isinstance(mask, pyroid.Column)
    assert mask.type == "bool"
    assert mask.name == "body"
    assert mask.num_chunks == col.num_chunks
    assert mask.null_count == 2
    assert mask.to_pylist() == [None if v is None else len(v) >= 2 for v in VALUES]
    assert _testing.char_lengths(col) == [None if v is None else len(v) for v in VALUES]


def test_nulls_and_empty_strings_are_distinct() -> None:
    col = _testing.column(["", None, " "])
    assert t.normalize_whitespace(col).to_pylist() == ["", None, ""]
    assert t.min_chars(col, 0).to_pylist() == [True, None, True]


def test_empty_column() -> None:
    col = _testing.column([])
    assert len(col) == 0
    assert col.num_chunks == 0
    out = t.normalize_whitespace(col)
    assert len(out) == 0
    assert out.to_pylist() == []
    assert out.type == "string"
    assert t.min_chars(col, 1).to_pylist() == []
    _, array = out.__arrow_c_array__()  # exporting zero chunks still works
    assert type(array).__name__ == "PyCapsule"


def test_all_null_column() -> None:
    col = _testing.column([None] * 100, chunks=3)
    assert t.normalize_whitespace(col).to_pylist() == [None] * 100
    assert t.min_chars(col, 0).null_count == 100


def test_chaining_columns() -> None:
    col = _testing.column(VALUES, chunks=3)
    once = t.normalize_whitespace(col)
    twice = t.normalize_whitespace(once)
    assert twice.to_pylist() == once.to_pylist()
    assert twice.num_chunks == once.num_chunks
    assert t.min_chars(once, 3).to_pylist() == [
        None if v is None else len(ref(v) or "") >= 3 for v in VALUES
    ]
    # The same Column can be consumed any number of times.
    assert t.normalize_whitespace(col).to_pylist() == once.to_pylist()


def test_bool_column_is_not_text() -> None:
    mask = t.min_chars(_testing.column(["a"]), 1)
    with pytest.raises(pyroid.InputError, match="got bool"):
        t.normalize_whitespace(mask)


@pytest.mark.timeout(60)
def test_large_multi_chunk_column_uses_the_pool() -> None:
    values = [None if i % 7 == 0 else f" doc  {i} " for i in range(20_000)]
    col = _testing.column(values, chunks=4)
    out = t.normalize_whitespace(col)
    assert out.num_chunks == 4
    assert out.to_pylist() == [ref(v) for v in values]
    assert t.min_chars(col, 9).to_pylist() == [None if v is None else len(v) >= 9 for v in values]


def test_output_outlives_input() -> None:
    col = _testing.column(["a  b"] * 10)
    out = t.normalize_whitespace(col)
    del col
    gc.collect()
    assert out.to_pylist() == ["a b"] * 10


# ---- custom PyCapsule producers ----------------------------------------------


class StreamOnly:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.inner.__arrow_c_stream__(requested_schema)


class ArrayOnly:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __arrow_c_array__(self, requested_schema: object = None) -> tuple[object, object]:
        result: tuple[object, object] = self.inner.__arrow_c_array__(requested_schema)
        return result


class Both(StreamOnly):
    def __arrow_c_array__(self, requested_schema: object = None) -> tuple[object, object]:
        raise AssertionError("__arrow_c_stream__ is preferred")


def test_stream_producer() -> None:
    col = _testing.column(VALUES, chunks=3)
    out = t.normalize_whitespace(StreamOnly(col))
    assert out.num_chunks == 3
    assert out.to_pylist() == [ref(v) for v in VALUES]


def test_array_producer_concatenates() -> None:
    col = _testing.column(VALUES, large=True, chunks=3, name="t")
    out = t.normalize_whitespace(ArrayOnly(col))
    assert out.num_chunks == 1
    assert out.type == "large_string"
    assert out.name == "t"
    assert out.to_pylist() == [ref(v) for v in VALUES]


def test_stream_is_preferred_over_array() -> None:
    out = t.normalize_whitespace(Both(_testing.column(["a  b"])))
    assert out.to_pylist() == ["a b"]


class BadStream:
    def __init__(self, result: object) -> None:
        self.result = result

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.result


class BadArray:
    def __init__(self, result: object) -> None:
        self.result = result

    def __arrow_c_array__(self, requested_schema: object = None) -> object:
        return self.result


@pytest.mark.parametrize(
    "producer",
    [
        BadStream(42),
        BadStream(_testing.column(["a"]).__arrow_c_array__()[0]),  # wrong capsule name
        BadArray(42),
        BadArray((1, 2)),
        BadArray(_testing.column(["a"]).__arrow_c_array__()[::-1]),  # swapped
    ],
    ids=["stream-int", "stream-schema-capsule", "array-int", "array-ints", "array-swapped"],
)
def test_malformed_producer_is_an_input_error(producer: object) -> None:
    with pytest.raises(pyroid.InputError):
        FUNCS[0](producer)


class StreamFails(ArrayOnly):
    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        raise RuntimeError("stream export failed")


def test_array_protocol_is_the_fallback_when_the_stream_raises() -> None:
    out = t.normalize_whitespace(StreamFails(_testing.column(["a  b", None], chunks=2)))
    assert out.to_pylist() == ["a b", None]


def test_stream_error_is_reported_when_both_protocols_fail() -> None:
    with pytest.raises(RuntimeError, match="stream export failed"):
        t.normalize_whitespace(StreamFails(BadArray(42)))


def test_producer_exceptions_propagate() -> None:
    class Boom:
        def __arrow_c_stream__(self, requested_schema: object = None) -> object:
            raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        t.normalize_whitespace(Boom())


# ---- pyarrow ---------------------------------------------------------------

ARROW_TYPES = ["string", "large_string", "string_view"]
arrow_types = pytest.mark.parametrize("type_name", ARROW_TYPES)


def _out_type(type_name: str) -> str:
    return "large_string" if type_name == "large_string" else "string"


@arrow_types
def test_pyarrow_array(pa: ModuleType, type_name: str) -> None:
    arr = pa.array(VALUES, type=getattr(pa, type_name)())
    out = t.normalize_whitespace(arr)
    assert out.type == _out_type(type_name)
    assert out.to_pylist() == [ref(v) for v in VALUES]
    assert out.null_count == 2
    mask = t.min_chars(arr, 3)
    assert mask.type == "bool"
    assert mask.to_pylist() == [None if v is None else len(v) >= 3 for v in VALUES]
    assert _testing.char_lengths(arr) == [None if v is None else len(v) for v in VALUES]


@arrow_types
@pytest.mark.parametrize(("start", "stop"), [(3, None), (1, 4), (5, 9), (9, None)])
def test_pyarrow_slices(pa: ModuleType, type_name: str, start: int, stop: int | None) -> None:
    # Offsets that are not multiples of 8 exercise the validity-bitmap offset.
    values = VALUES * 3
    arr = pa.array(values, type=getattr(pa, type_name)())
    length = None if stop is None else stop - start
    sliced = arr.slice(start, length)
    assert sliced.offset == start
    want = values[start:stop]
    assert t.normalize_whitespace(sliced).to_pylist() == [ref(v) for v in want]
    assert t.min_chars(sliced, 3).to_pylist() == [None if v is None else len(v) >= 3 for v in want]


@arrow_types
def test_pyarrow_chunked(pa: ModuleType, type_name: str) -> None:
    ty = getattr(pa, type_name)()
    chunks: list[list[str | None]] = [VALUES[:3], [], VALUES[3:], [None], []]
    arr = pa.chunked_array(chunks, type=ty)
    out = t.normalize_whitespace(arr)
    assert out.num_chunks == 3  # empty chunks are dropped
    assert out.type == _out_type(type_name)
    flat = [v for c in chunks for v in c]
    assert out.to_pylist() == [ref(v) for v in flat]
    back = pa.chunked_array(out)
    assert [len(c) for c in back.chunks] == [3, len(VALUES) - 3, 1]
    mask = t.min_chars(arr, 1)
    assert mask.num_chunks == 3
    assert pa.chunked_array(mask).type == pa.bool_()


@arrow_types
def test_pyarrow_sliced_chunked(pa: ModuleType, type_name: str) -> None:
    arr = pa.chunked_array([VALUES, VALUES], type=getattr(pa, type_name)()).slice(4, 10)
    want = (VALUES + VALUES)[4:14]
    assert t.normalize_whitespace(arr).to_pylist() == [ref(v) for v in want]


@arrow_types
def test_pyarrow_empty(pa: ModuleType, type_name: str) -> None:
    ty = getattr(pa, type_name)()
    for empty in (pa.array([], type=ty), pa.chunked_array([], type=ty)):
        out = t.normalize_whitespace(empty)
        assert len(out) == 0
        assert out.num_chunks == 0
        assert out.type == _out_type(type_name)
        assert pa.chunked_array(out).type == pa.array(out).type
        assert len(t.min_chars(empty, 1)) == 0


def test_pyarrow_string_view_long_and_short(pa: ModuleType) -> None:
    values = ["short", "twelve_bytes", "thirteen byte", " padded  long value here ", "", None]
    arr = pa.array(values, type=pa.string_view())
    assert t.normalize_whitespace(arr).to_pylist() == [ref(v) for v in values]
    assert t.normalize_whitespace(arr.slice(2)).to_pylist() == [ref(v) for v in values[2:]]


@pytest.mark.parametrize("type_name", ["string", "large_string"])
def test_pyarrow_export_round_trip(pa: ModuleType, type_name: str) -> None:
    ty = getattr(pa, type_name)()
    col = t.normalize_whitespace(pa.chunked_array([VALUES[:4], VALUES[4:]], type=ty))
    assert col.num_chunks == 2
    whole = pa.array(col)  # __arrow_c_array__: concatenates
    assert pa.array(col).equals(whole)  # exporting again gives the same data
    assert isinstance(whole, pa.Array)
    assert whole.type == ty
    whole.validate(full=True)
    assert whole.to_pylist() == [ref(v) for v in VALUES]
    chunked = pa.chunked_array(col)  # __arrow_c_stream__: keeps the chunks
    assert chunked.num_chunks == 2
    assert chunked.type == ty
    chunked.validate(full=True)
    assert chunked.to_pylist() == whole.to_pylist()
    mask = t.min_chars(pa.chunked_array([VALUES[:4], VALUES[4:]], type=ty), 2)
    bools = pa.array(mask)
    bools.validate(full=True)
    assert bools.type == pa.bool_()
    assert bools.null_count == 2
    assert bools.to_pylist() == [None if v is None else len(v) >= 2 for v in VALUES]


def test_pyarrow_exports_outlive_the_column(pa: ModuleType) -> None:
    col = t.normalize_whitespace(pa.array(["a  b", None, "c"]))
    whole, chunked = pa.array(col), pa.chunked_array(col)
    del col
    gc.collect()
    assert whole.to_pylist() == ["a b", None, "c"]
    assert chunked.to_pylist() == ["a b", None, "c"]


def test_pyarrow_column_without_pyarrow_round_trip(pa: ModuleType) -> None:
    col = _testing.column(VALUES, chunks=3)
    arr = pa.chunked_array(col)
    arr.validate(full=True)
    assert arr.num_chunks == 3
    assert t.normalize_whitespace(arr).to_pylist() == [ref(v) for v in VALUES]


def test_pyarrow_field_name_is_carried_over(pa: ModuleType) -> None:
    field = pa.field("text", pa.string())
    arr = pa.array(["a  b", None])

    class Named:
        def __arrow_c_array__(self, requested_schema: object = None) -> tuple[object, object]:
            schema: object = field.__arrow_c_schema__()
            array: object = arr.__arrow_c_array__()[1]
            return schema, array

    out = t.normalize_whitespace(Named())
    assert out.name == "text"
    assert t.min_chars(Named(), 1).name == "text"
    assert pa.array(out).to_pylist() == ["a b", None]


def test_pyarrow_examples_from_modules_md(pa: ModuleType) -> None:
    col = t.normalize_whitespace(pa.array(["a  b", None]))
    assert repr(col) == "pyroid.Column<string>(len=2, null_count=1, num_chunks=1)"
    assert t.min_chars(pa.chunked_array([["abc"], [None]]), 2).to_pylist() == [True, None]


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda pa: pa.array(["a", "b", "a"]).dictionary_encode(), "dictionary-encoded"),
        (lambda pa: pa.array([1, 2]), "got int64"),
        (lambda pa: pa.array([1.5]), "got float64"),
        (lambda pa: pa.array([True]), "got bool"),
        (lambda pa: pa.array([None, None]), "got null"),
        (lambda pa: pa.array([b"a"], pa.binary()), "got binary; decode"),
        (lambda pa: pa.array([b"a"], pa.large_binary()), "got large_binary; decode"),
        (lambda pa: pa.array([b"a"], pa.binary_view()), "got binary_view; decode"),
        (lambda pa: pa.array([["a"]]), "got list"),
        (lambda pa: pa.array([{"text": "a"}]), "table['text']"),
        (lambda pa: pa.table({"text": ["a"]}), "table['text']"),
        (lambda pa: pa.record_batch({"text": ["a"]}), "table['text']"),
        (lambda pa: pa.chunked_array([[1]]), "got int64"),
    ],
    ids=[
        "dictionary",
        "int64",
        "float64",
        "bool",
        "null",
        "binary",
        "large_binary",
        "binary_view",
        "list",
        "struct",
        "table",
        "record_batch",
        "chunked-int64",
    ],
)
def test_pyarrow_wrong_types(pa: ModuleType, make: Any, message: str) -> None:
    obj = make(pa)
    for fn in FUNCS:
        with pytest.raises(pyroid.InputError, match=re.escape(message)) as info:
            fn(obj)
        assert isinstance(info.value, TypeError)
        assert "expected an Arrow string column" in str(info.value) or "dictionary" in message


# ---- malformed Arrow data ----------------------------------------------------


def _string_from_buffers(pa: ModuleType, offsets: list[int], data: bytes) -> Any:
    raw = struct.pack(f"<{len(offsets)}i", *offsets)
    return pa.Array.from_buffers(
        pa.string(), len(offsets) - 1, [None, pa.py_buffer(raw), pa.py_buffer(data)]
    )


def test_invalid_utf8_from_buffers(pa: ModuleType) -> None:
    bad = _string_from_buffers(pa, [0, 2, 4, 5], b"ok\xc3\x28z")
    with pytest.raises(UnicodeDecodeError) as info:
        t.normalize_whitespace(pa.chunked_array([pa.array(["fine"]), bad]))
    assert "row 1 of chunk 1" in _notes(info.value)
    assert info.value.object == b"\xc3\x28"


def test_decreasing_offsets(pa: ModuleType) -> None:
    bad = _string_from_buffers(pa, [0, 3, 1, 3], b"abc")
    with pytest.raises(ValueError, match="invalid Arrow data") as info:
        t.normalize_whitespace(bad)
    assert "offsets decrease" in str(info.value)
    assert not isinstance(info.value, TypeError)


def _binary_as(pa: ModuleType, values: list[bytes | None], type_name: str) -> Any:
    """Arrow string data that is not valid UTF-8 (a binary array viewed as strings)."""
    binary = {"string": "binary", "large_string": "large_binary", "string_view": "binary_view"}
    arr = pa.array(values, type=getattr(pa, binary[type_name])())
    return arr.view(getattr(pa, type_name)())


@arrow_types
@pytest.mark.parametrize(
    "bad", [b"\xff", b"a\xc3", b"\xed\xa0\x80", b"long enough to be out of line \xfe"]
)
def test_invalid_utf8(pa: ModuleType, type_name: str, bad: bytes) -> None:
    arr = _binary_as(pa, [b"ok", bad, b"fine"], type_name)
    for fn in FUNCS:
        with pytest.raises(UnicodeDecodeError) as info:
            fn(arr)
        assert "row 1 of chunk 0" in _notes(info.value)


@arrow_types
def test_invalid_utf8_in_a_null_slot_is_never_read(pa: ModuleType, type_name: str) -> None:
    arr = _binary_as(pa, [b"a  b", b"\xff", b"c"], type_name)
    masked = pa.Array.from_buffers(
        arr.type,
        len(arr),
        [pa.py_buffer(bytes([0b101])), *arr.buffers()[1:]],
        null_count=1,
    )
    assert t.normalize_whitespace(masked).to_pylist() == ["a b", None, "c"]


@pytest.mark.timeout(60)
def test_invalid_utf8_deep_in_a_pool_sized_chunk(pa: ModuleType) -> None:
    values: list[bytes | None] = [b"ok"] * 10_000
    values[5_000] = b"\x80"
    good = pa.array(["x"] * 10, pa.string())
    arr = pa.chunked_array([good, _binary_as(pa, values, "string")])
    with pytest.raises(UnicodeDecodeError) as info:
        t.normalize_whitespace(arr)
    assert "row 5000 of chunk 1" in _notes(info.value)


def test_invalid_utf8_chunk_number_counts_empty_chunks(pa: ModuleType) -> None:
    bad = _binary_as(pa, [b"ok", b"\xff"], "string")
    arr = pa.chunked_array([pa.array(["x"]), pa.array([], pa.string()), bad])
    assert arr.chunk(2).equals(bad)
    with pytest.raises(UnicodeDecodeError) as info:
        t.normalize_whitespace(arr)
    assert "row 1 of chunk 2" in _notes(info.value)


# ---- polars ----------------------------------------------------------------


def test_polars_series(pl: ModuleType) -> None:
    s = pl.Series("text", ["a  b", None, "  c", "ok"])
    out = t.normalize_whitespace(s)
    assert isinstance(out, pyroid.Column)
    assert out.name == "text"
    assert out.type in {"string", "large_string"}
    assert out.to_pylist() == ["a b", None, "c", "ok"]
    mask = t.min_chars(s, 2)
    assert mask.name == "text"
    assert mask.to_pylist() == [True, None, True, True]
    back = pl.Series(out)
    assert back.name == "text"  # polars takes the name from the stream
    assert back.to_list() == ["a b", None, "c", "ok"]
    assert pl.Series(mask).to_list() == [True, None, True, True]


def test_polars_multi_chunk(pl: ModuleType) -> None:
    s = pl.concat([pl.Series("t", ["a  b"]), pl.Series("t", [None, " c "])], rechunk=False)
    assert s.n_chunks() == 2
    out = t.normalize_whitespace(s)
    assert out.num_chunks == 2
    assert out.to_pylist() == ["a b", None, "c"]


def test_polars_column_without_pyarrow_round_trip(pl: ModuleType) -> None:
    col = _testing.column(VALUES, chunks=3, name="body")
    s = pl.Series(col)
    assert s.name == "body"
    assert s.to_list() == VALUES


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda pl: pl.Series(["a", "b"], dtype=pl.Categorical), "dictionary-encoded"),
        (lambda pl: pl.DataFrame({"text": ["a"]}), "table['text']"),
        (lambda pl: pl.Series([1, 2]), "got int64"),
    ],
    ids=["categorical", "dataframe", "int64"],
)
def test_polars_wrong_types(pl: ModuleType, make: Any, message: str) -> None:
    with pytest.raises(pyroid.InputError, match=re.escape(message)):
        t.normalize_whitespace(make(pl))
