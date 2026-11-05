"""Argument extraction: wrong types, bad values, keyword rules.

Wrong types raise ``pyroid.InputError`` (a ``PyroidError`` and a
``TypeError``); bad values raise ``ValueError``; PyO3 adds the note
``while processing '<arg>'`` to argument errors.
"""

import array
import collections
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import pyroid
import pyroid.text as t
from pyroid._pyroid import _testing


def _min_chars_1(x: Any, **kw: Any) -> object:
    return t.min_chars(x, 1, **kw)


#: Every entry point that takes documents (transform, predicate, map_values).
FUNCS: list[Callable[..., object]] = [t.normalize_whitespace, _min_chars_1, _testing.char_lengths]
FUNC_IDS = ["normalize_whitespace", "min_chars", "char_lengths"]
funcs = pytest.mark.parametrize("fn", FUNCS, ids=FUNC_IDS)


def _notes(exc: BaseException) -> list[str]:
    return list(getattr(exc, "__notes__", []))


def _gen() -> Iterator[str]:
    yield "a"


class _NotText:
    pass


# ---- the documents argument ------------------------------------------------


@funcs
@pytest.mark.parametrize(
    ("x", "type_name"),
    [
        (1, "int"),
        (1.5, "float"),
        (None, "NoneType"),
        (True, "bool"),
        ({"a": "b"}, "dict"),
        ({"a"}, "set"),
        (frozenset({"a"}), "frozenset"),
        (range(3), "range"),
        (iter(["a"]), "list_iterator"),
        (collections.deque(["a"]), "collections.deque"),
        (array.array("b", b"ab"), "array.array"),  # a buffer, but not bytes
        (_NotText(), "_NotText"),
    ],
    ids=[
        "int",
        "float",
        "None",
        "bool",
        "dict",
        "set",
        "frozenset",
        "range",
        "iterator",
        "deque",
        "array",
        "object",
    ],
)
def test_wrong_input_type(fn: Callable[..., object], x: object, type_name: str) -> None:
    with pytest.raises(pyroid.InputError) as info:
        fn(x)
    err = info.value
    assert type(err) is pyroid.InputError
    assert isinstance(err, TypeError)
    assert isinstance(err, pyroid.PyroidError)
    msg = str(err)
    assert msg.startswith("expected str, list[str], tuple[str, ...] or an Arrow string"), msg
    assert type_name in msg.split(", got ", 1)[1]  # may be module-qualified
    assert "while processing 'x'" in _notes(err)


HINT = "; convert other iterables with list(...)"


@pytest.mark.parametrize(
    ("x", "iterable"),
    [
        (1, False),
        (None, False),
        (_NotText(), False),
        ({"a": "b"}, True),
        ({"a"}, True),
        (range(3), True),
        (iter(["a"]), True),
        (collections.deque(["a"]), True),
    ],
    ids=["int", "None", "object", "dict", "set", "range", "iterator", "deque"],
)
def test_list_hint_only_for_iterables(x: object, iterable: bool) -> None:
    raw: Any = t.normalize_whitespace
    with pytest.raises(pyroid.InputError) as info:
        raw(x)
    assert str(info.value).endswith(HINT) is iterable, str(info.value)


@funcs
def test_generator_suggests_list(fn: Callable[..., object]) -> None:
    with pytest.raises(pyroid.InputError, match=re.escape("got generator; convert")):
        fn(_gen())


@funcs
@pytest.mark.parametrize(
    "x",
    [b"a b", bytearray(b"a b"), memoryview(b"a b")],
    ids=["bytes", "bytearray", "memoryview"],
)
def test_bytes_like_input(fn: Callable[..., object], x: object) -> None:
    with pytest.raises(pyroid.InputError, match="decode bytes to str first") as info:
        fn(x)
    assert f"got {type(x).__name__}" in str(info.value)


@funcs
@pytest.mark.parametrize("kind", [list, tuple], ids=["list", "tuple"])
@pytest.mark.parametrize(
    ("item", "type_name"),
    [
        (None, "NoneType"),
        (1, "int"),
        (b"b", "bytes"),
        (["b"], "list"),
        (("b",), "tuple"),
    ],
    ids=["None", "int", "bytes", "list", "tuple"],
)
@pytest.mark.parametrize("index", [0, 3, 70], ids=["first", "inline", "pool-sized"])
def test_wrong_item_type_names_the_index(
    fn: Callable[..., object],
    kind: Callable[[list[object]], object],
    item: object,
    type_name: str,
    index: int,
) -> None:
    docs: list[object] = ["ok"] * 100
    docs[index] = item
    with pytest.raises(pyroid.InputError) as info:
        fn(kind(docs))
    assert str(info.value) == f"expected str at index {index}, got {type_name}"
    assert isinstance(info.value, TypeError)
    assert "while processing 'x'" in _notes(info.value)


def test_first_bad_item_is_reported() -> None:
    bad: Any = ["a", 1, None]
    with pytest.raises(pyroid.InputError, match="at index 1, got int"):
        t.normalize_whitespace(bad)


def test_none_in_list_is_an_error_not_a_null() -> None:
    bad: Any = ["a", None]
    with pytest.raises(pyroid.InputError, match="at index 1, got NoneType"):
        t.normalize_whitespace(bad)


def test_input_error_can_be_caught_as_type_error() -> None:
    bad: Any = 42
    with pytest.raises(TypeError):
        t.normalize_whitespace(bad)


# ---- lone surrogates (examples; the property tests are in test_text_props) ----


@funcs
@pytest.mark.parametrize("size", [1, 5000], ids=["short", "detached"])
def test_lone_surrogate_in_str(fn: Callable[..., object], size: int) -> None:
    doc = "a" * size + chr(0xD800)
    with pytest.raises(UnicodeEncodeError) as info:
        fn(doc)
    notes = _notes(info.value)
    assert any("the str contains a lone surrogate" in n for n in notes), notes
    assert "while processing 'x'" in notes


@funcs
@pytest.mark.parametrize("kind", [list, tuple], ids=["list", "tuple"])
@pytest.mark.parametrize("index", [0, 70])
def test_lone_surrogate_in_batch(
    fn: Callable[..., object], kind: Callable[[list[str]], object], index: int
) -> None:
    docs = ["fine"] * 100
    docs[index] = "x" + chr(0xDFFF) + "y"
    with pytest.raises(UnicodeEncodeError) as info:
        fn(kind(docs))
    notes = _notes(info.value)
    assert any(f"the str at index {index} contains a lone surrogate" in n for n in notes), notes


# ---- threads= ----------------------------------------------------------------


@funcs
@pytest.mark.parametrize("threads", [None, 1, 2, 3, 64, 10_000])
def test_valid_threads(fn: Callable[..., object], threads: int | None) -> None:
    docs = [" a  b "] * 200
    assert fn(docs, threads=threads) == fn(docs)


@funcs
@pytest.mark.parametrize("threads", [0, -1, -(2**40)])
def test_threads_must_be_positive(fn: Callable[..., object], threads: int) -> None:
    with pytest.raises(ValueError, match="threads must be None or a positive int") as info:
        fn(["a"], threads=threads)
    assert not isinstance(info.value, TypeError)
    assert "while processing 'threads'" in _notes(info.value)


@funcs
@pytest.mark.parametrize("threads", [True, False, 1.5, 2.0, "2", [2], b"2"], ids=repr)
def test_threads_wrong_type(fn: Callable[..., object], threads: object) -> None:
    with pytest.raises(pyroid.InputError, match="threads must be None or a positive int") as info:
        fn(["a"], threads=threads)
    assert f"got {type(threads).__name__}" in str(info.value)
    assert "while processing 'threads'" in _notes(info.value)


def test_threads_is_keyword_only() -> None:
    nw: Any = t.normalize_whitespace
    mc: Any = t.min_chars
    with pytest.raises(TypeError, match="positional"):
        nw(["a"], 2)
    with pytest.raises(TypeError, match="positional"):
        mc(["a"], 1, 2)
    with pytest.raises(TypeError, match="positional"):
        _testing.char_lengths(["a"], 2)


def test_x_is_positional_only() -> None:
    raw: Any = t.normalize_whitespace
    with pytest.raises(TypeError):
        raw(x="a")
    raw_min: Any = t.min_chars
    with pytest.raises(TypeError):
        raw_min(x="a", n=1)


def test_huge_threads_is_a_positive_int() -> None:
    assert t.normalize_whitespace(["a  b"] * 100, threads=2**64) == ["a b"] * 100


# ---- min_chars(n) ------------------------------------------------------------


@pytest.mark.parametrize("n", [-1, -(2**40)])
def test_negative_n(n: int) -> None:
    with pytest.raises(ValueError, match="expected a non-negative int") as info:
        t.min_chars(["abc"], n)
    assert not isinstance(info.value, TypeError)
    assert "while processing 'n'" in _notes(info.value)


@pytest.mark.parametrize("n", ["3", True, False, 1.5, 3.0, None, [3]], ids=repr)
def test_n_wrong_type(n: object) -> None:
    raw: Any = t.min_chars
    with pytest.raises(pyroid.InputError, match="expected a non-negative int") as info:
        raw(["abc"], n)
    assert f"got {type(n).__name__}" in str(info.value)
    assert "while processing 'n'" in _notes(info.value)


def test_n_is_required() -> None:
    raw: Any = t.min_chars
    with pytest.raises(TypeError, match="'n'"):
        raw(["abc"])


def test_huge_n_is_never_met() -> None:
    assert t.min_chars(["abc", ""], 2**64) == [False, False]


def test_huge_negative_n_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        t.min_chars(["abc"], -(2**64))
