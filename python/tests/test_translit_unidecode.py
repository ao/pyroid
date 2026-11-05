"""Differential tests: ``pyroid.translit.unidecode`` has Unidecode 1.4.0's ``errors=`` semantics.

Unidecode (GPL) is used here **only as a black-box oracle**: the tests call its public
functions and its ``UnidecodeError`` at test time and compare what they return, raise and
warn. Nothing in this file is derived from Unidecode's code or tables, and no Unidecode
traceback is ever shown (every call into it is caught). The module skips unless Unidecode
1.4.0 is installed; it is not a dependency of pyroid or of its test group.

The two libraries transliterate differently (pyroid uses anyascii's data), so the comparison
runs over characters on which they agree, computed at test time from both libraries:

- AGREE: ASCII, plus the non-ASCII characters both map to the same string in strict mode;
- BOTH-MISSING: the characters both reject in strict mode (unassigned and private-use code
  points, planes 15 and 16, ...);
- lone surrogates (str, list and tuple input only: an Arrow string cannot hold one).

Over that alphabet every ``errors`` mode, ``replace_str`` value (non-str ones too) and
invalid ``errors`` value must give the same result, or the same exception (class, message,
``args``, ``index``), with the same warnings in the same order. A batch must match a loop of
Unidecode calls, the error carrying a note that names the failing document. The last tests
check pyroid-only behaviour where the libraries differ on purpose (see ``translit.pyi``).

Set HYPOTHESIS_PROFILE=ci, or PYROID_TRANSLIT_EXAMPLES=5000 for a deep run.
"""

from __future__ import annotations

import array
import collections
import dataclasses
import datetime
import decimal
import enum
import functools
import importlib.metadata
import os
import pickle
import random
import string
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.translit as pt
from pyroid._pyroid import _testing

# Untyped on purpose: the mypy hook does not install Unidecode.
oracle: Any = pytest.importorskip("unidecode")
if importlib.metadata.version("Unidecode") != "1.4.0":
    pytest.skip(
        f"needs Unidecode 1.4.0, found {importlib.metadata.version('Unidecode')}",
        allow_module_level=True,
    )

_EXAMPLES = os.environ.get("PYROID_TRANSLIT_EXAMPLES")
examples = (
    settings(suppress_health_check=[HealthCheck.too_slow], max_examples=int(_EXAMPLES))
    if _EXAMPLES
    else settings(suppress_health_check=[HealthCheck.too_slow])
)

# Calls with deliberately ill-typed arguments go through this alias.
unidecode_any: Callable[..., Any] = pt.unidecode

ARROW_SURROGATE_NOTE = (
    "pyroid: replace_str holds a lone surrogate, which an Arrow string column cannot hold"
)

# ============================================================== calling and comparing


@dataclasses.dataclass
class Outcome:
    """What one call did: its result or exception, and the warnings it emitted, in order."""

    result: object
    error: Exception | None
    warned: list[tuple[type[Warning], str]]


def call(fn: Callable[..., object], /, *args: object, **kwargs: object) -> Outcome:
    """Call ``fn`` and record every warning. An exception is caught and kept without its
    traceback, so nothing from inside Unidecode is ever printed."""
    result: object = None
    error: Exception | None = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            error = e.with_traceback(None)
    return Outcome(result, error, [(w.category, str(w.message)) for w in caught])


_NO_INDEX = "<no index attribute>"


def describe(
    exc: BaseException, *, oracle_side: bool = False, note: str | None = None
) -> tuple[object, ...]:
    """What must match between the two exceptions. Unidecode's ``UnidecodeError`` stands for
    pyroid's; ``note`` is the note pyroid adds for a batch."""
    cls: type[BaseException] = type(exc)
    if oracle_side and cls is oracle.UnidecodeError:
        cls = pt.UnidecodeError
    notes = getattr(exc, "__notes__", None)
    if note is not None:
        notes = [*(notes or []), note]
    return (cls, str(exc), exc.args, getattr(exc, "index", _NO_INDEX), notes)


def assert_same(mine: Outcome, theirs: Outcome, *, note: str | None = None) -> None:
    """pyroid's outcome is Unidecode's: the same result (and type), or the same exception
    (plus ``note``), and the same warnings."""
    if theirs.error is None:
        if mine.error is not None:
            pytest.fail(
                f"pyroid raised {describe(mine.error)!r}; Unidecode returned {theirs.result!r}"
            )
        assert type(mine.result) is type(theirs.result)
        assert mine.result == theirs.result
    else:
        if mine.error is None:
            theirs_error = describe(theirs.error, oracle_side=True)
            pytest.fail(f"pyroid returned {mine.result!r}; Unidecode raised {theirs_error!r}")
        assert describe(mine.error) == describe(theirs.error, oracle_side=True, note=note)
    assert mine.warned == theirs.warned


class _Absent(enum.Enum):
    """An argument left out of the call, so that its default applies."""

    ABSENT = enum.auto()


ABSENT = _Absent.ABSENT


def arguments(
    string: object, errors: object, replace_str: object, *, keywords: bool
) -> tuple[tuple[object, ...], dict[str, object]]:
    """``(args, kwargs)`` for a call, positional or by keyword, leaving out ABSENT ones."""
    if keywords:
        named = {"string": string, "errors": errors, "replace_str": replace_str}
        return (), {k: v for k, v in named.items() if v is not ABSENT}
    args: list[object] = [string]
    kwargs: dict[str, object] = {}
    if errors is not ABSENT:
        args.append(errors)
        if replace_str is not ABSENT:
            args.append(replace_str)
    elif replace_str is not ABSENT:
        kwargs["replace_str"] = replace_str
    return tuple(args), kwargs


def oracle_loop(
    docs: Sequence[str | None], errors: object, replace_str: object
) -> tuple[Outcome, int | None]:
    """Unidecode over each document in turn, as a user's loop would (None stays None).

    Returns the outcome (a list) and the index of the document that raised, if one did.
    """
    at: list[int] = []

    def loop() -> list[str | None]:
        out: list[str | None] = []
        for i, doc in enumerate(docs):
            if doc is None:
                out.append(None)
                continue
            at[:] = [i]
            args, kwargs = arguments(doc, errors, replace_str, keywords=False)
            out.append(oracle.unidecode(*args, **kwargs))
        at.clear()
        return out

    return call(loop), (at[0] if at else None)


def has_surrogate(s: str) -> bool:
    return any(0xD800 <= ord(c) <= 0xDFFF for c in s)


# ============================================================== the comparison alphabet


def _candidates() -> list[int]:
    """Every code point up to U+2FFF, a sample of the rest, and known missing ones."""
    cps = set(range(0x80, 0x3000))
    cps.update(range(0x3000, 0x20000, 7))
    cps.update(range(0x20000, 0x110000, 89))
    cps.update([0x378, 0xE000, 0xF8FF, 0xFDD0, 0xFDEF, 0xF0000, 0xFFFFD, 0x100000, 0x10FFFD])
    return sorted(cp for cp in cps if not 0xD800 <= cp <= 0xDFFF)


@functools.cache
def alphabets() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(AGREE, BOTH-MISSING) non-ASCII characters, from both libraries in strict mode."""
    agree: list[str] = []
    both_missing: list[str] = []
    for cp in _candidates():
        c = chr(cp)
        mine, theirs = call(pt.unidecode, c, "strict"), call(oracle.unidecode, c, "strict")
        if mine.warned or theirs.warned:
            continue
        if mine.error is None and theirs.error is None and mine.result == theirs.result:
            agree.append(c)
        elif isinstance(mine.error, pt.UnidecodeError) and isinstance(
            theirs.error, oracle.UnidecodeError
        ):
            both_missing.append(c)
    return tuple(agree), tuple(both_missing)


def _text(*, missing: bool, surrogates: bool) -> st.SearchStrategy[str]:
    agree, both_missing = alphabets()
    parts: list[st.SearchStrategy[str]] = [
        st.characters(max_codepoint=0x7F),
        st.sampled_from(agree),
    ]
    if missing:
        parts.append(st.sampled_from(both_missing))
    if surrogates:
        parts.append(st.characters(categories=["Cs"]))
    return st.text(st.one_of(*parts), max_size=12)


# Deferred: the alphabet is computed on first use, not at collection time.
CLEAN_TEXT = st.deferred(lambda: _text(missing=False, surrogates=False))
ARROW_TEXT = st.one_of(CLEAN_TEXT, st.deferred(lambda: _text(missing=True, surrogates=False)))
TEXT = st.one_of(ARROW_TEXT, st.deferred(lambda: _text(missing=True, surrogates=True)))


def _long(pieces: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    """Longer strings, with ASCII runs of 8+ bytes (the kernels' word-at-a-time path)."""
    ascii_runs = st.text(st.characters(max_codepoint=0x7F), min_size=8, max_size=40)
    return st.lists(st.one_of(pieces, ascii_runs), max_size=6).map("".join)


ANY_TEXT = st.one_of(TEXT, _long(TEXT))
ANY_ARROW_TEXT = st.one_of(ARROW_TEXT, _long(ARROW_TEXT))


class _Tagged(str):
    """A str subclass with its own ``repr`` (Unidecode's messages use ``repr()``)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<tagged {str.__repr__(self)}>"


class _Mode(enum.StrEnum):
    STRICT = "strict"
    REPLACE = "replace"
    BOGUS = "bogus"


VALID_ERRORS = ("ignore", "replace", "strict", "preserve")
INVALID_ERRORS: tuple[object, ...] = (
    *("bogus", "", "IGNORE", "Strict", "replace ", "\u00e9", "\ud800"),
    *(None, 0, 1, 1.5, b"strict", ("strict",), ["ignore"], _Tagged("bogus"), _Mode.BOGUS),
)
ERRORS = st.one_of(
    st.sampled_from(VALID_ERRORS),
    st.sampled_from([ABSENT, *VALID_ERRORS, _Tagged("strict"), _Mode.STRICT, _Mode.REPLACE]),
    st.sampled_from(INVALID_ERRORS),
)
STR_REPLACEMENTS = (
    *("?", "", "XX", "\x00", "\u00e9", "\u4e2d\u6587", "\U0001f600"),
    *("\ud800", "a\udfffb", _Tagged("<>")),
)
NON_STR_REPLACEMENTS: tuple[object, ...] = (None, 5, 1.5, b"?", ["?"])
REPLACE = st.one_of(
    st.sampled_from(STR_REPLACEMENTS), st.sampled_from([ABSENT, *NON_STR_REPLACEMENTS])
)
FUNCTIONS = ("unidecode", "unidecode_expect_ascii", "unidecode_expect_nonascii")
THREADS = st.sampled_from([None, 1, 2])


@pytest.mark.thread_unsafe(reason="Unidecode loads its tables lazily; warnings state")
def test_the_comparison_alphabet_is_not_trivial() -> None:
    agree, both_missing = alphabets()
    assert len(agree) > 2000
    assert "\u00e9" in agree
    assert len(both_missing) > 1000
    assert {"\u0378", "\ue000", "\uf8ff", "\U000f0000", "\U0010fffd"} <= set(both_missing)


# ============================================================== str


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@examples
@given(
    string=ANY_TEXT,
    errors=ERRORS,
    replace_str=REPLACE,
    keywords=st.booleans(),
    name=st.sampled_from(FUNCTIONS),
)
@example(string="a\ue000b", errors="strict", replace_str=ABSENT, keywords=False, name="unidecode")
@example(string="abc", errors="bogus", replace_str=None, keywords=False, name="unidecode")
@example(string="\u00e9", errors=None, replace_str=ABSENT, keywords=True, name="unidecode")
@example(string="a\ue000", errors=None, replace_str=ABSENT, keywords=True, name="unidecode")
@example(string="ab\ue000", errors="replace", replace_str=None, keywords=False, name="unidecode")
@example(string="a\ud800b\udc00", errors="strict", replace_str="?", keywords=True, name="unidecode")
@example(string="a\ud800b\udc00", errors="replace", replace_str=5, keywords=False, name="unidecode")
@example(
    string="a\ud800b\udc00", errors="bogus", replace_str=ABSENT, keywords=False, name="unidecode"
)
@example(string="a\U000f0000", errors="preserve", replace_str="?", keywords=False, name="unidecode")
@example(string="a\ue000", errors="replace", replace_str="\ud800", keywords=False, name="unidecode")
@example(
    # A surrogate pair in a str is two lone surrogates.
    string="\ud83d\ude00",
    errors=ABSENT,
    replace_str=ABSENT,
    keywords=False,
    name="unidecode_expect_ascii",
)
def test_str_matches_unidecode(
    string: str, errors: object, replace_str: object, keywords: bool, name: str
) -> None:
    args, kwargs = arguments(string, errors, replace_str, keywords=keywords)
    mine = call(getattr(pt, name), *args, **kwargs)
    theirs = call(getattr(oracle, name), *args, **kwargs)
    assert_same(mine, theirs)
    if string.isascii() and theirs.result is string:
        assert mine.result is string


class _Sub(str):
    __slots__ = ()


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@pytest.mark.parametrize("errors", VALID_ERRORS)
@pytest.mark.parametrize(
    "text",
    ["", "abc", "\u00e9", "\ue000", "a\ue000b", "\u00e9\ue000"],
    ids=["empty", "ascii", "mapped", "missing", "ascii-and-missing", "mapped-and-missing"],
)
def test_str_subclass_input(text: str, errors: str) -> None:
    """Unidecode returns an ASCII str as it is (even a subclass), anything else as a str."""
    s = _Sub(text)
    mine, theirs = call(pt.unidecode, s, errors), call(oracle.unidecode, s, errors)
    assert_same(mine, theirs)
    if theirs.error is None:
        assert (theirs.result is s) == text.isascii()
        assert (mine.result is s) == text.isascii()


class _Outer:
    class Inner:
        """Nested: ``str.join`` names its type ``Inner``, not ``_Outer.Inner``."""


# str.join's TypeError names the type by its C-level tp_name: "datetime.date" for a static
# type, the bare __name__ for a class defined in Python.
NON_STR_TYPES: list[object] = [
    *(None, 5, 1.5, True, b"?", bytearray(b"?"), ["?"], ("?",), object(), _Outer.Inner()),
    *(datetime.date(2020, 1, 1), decimal.Decimal(1), collections.OrderedDict(), array.array("b")),
]


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@pytest.mark.parametrize("replace_str", NON_STR_TYPES, ids=lambda v: type(v).__qualname__)
@pytest.mark.parametrize(
    "text",
    ["\ue000", "ab\u00e9\ue000\ue000", "a\ud800b\ue000\udfff"],
    ids=["missing", "mapped-then-missing", "with-surrogates"],
)
def test_non_str_replace_str_type_error(text: str, replace_str: object) -> None:
    mine = call(pt.unidecode, text, "replace", replace_str)
    theirs = call(oracle.unidecode, text, "replace", replace_str)
    assert_same(mine, theirs)


class _SurrogateRepr:
    def __repr__(self) -> str:
        return "<mode \u00e9\ud800>"


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@pytest.mark.parametrize(
    "errors",
    [_SurrogateRepr(), _Tagged("bogus"), "\ud800", "\u00e9\n", None, b"x"],
    ids=["repr-with-surrogate", "str-subclass-repr", "surrogate", "non-ascii", "None", "bytes"],
)
@pytest.mark.parametrize(
    "text", ["\ue000", "a\ud800", "abc", "\u00e9"], ids=["missing", "surrogate", "ascii", "mapped"]
)
def test_invalid_errors_message(text: str, errors: object) -> None:
    """An invalid ``errors`` value is reported with its ``repr()``, once a character needs it."""
    assert_same(call(pt.unidecode, text, errors), call(oracle.unidecode, text, errors))


# ============================================================== batches


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@examples
@given(
    docs=st.lists(ANY_TEXT, max_size=6),
    errors=ERRORS,
    replace_str=REPLACE,
    kind=st.sampled_from([list, tuple]),
    keywords=st.booleans(),
    threads=THREADS,
)
@example(
    docs=["ab", "\u00e9", "x\ue000", "\ud800y"],
    errors="strict",
    replace_str=ABSENT,
    kind=list,
    keywords=False,
    threads=None,
)
@example(
    docs=["\ud800", "ab", "\udfff\ue000"],
    errors="replace",
    replace_str=None,
    kind=tuple,
    keywords=True,
    threads=1,
)
def test_list_and_tuple_match_a_loop(
    docs: list[str],
    errors: object,
    replace_str: object,
    kind: type[list[str] | tuple[str, ...]],
    keywords: bool,
    threads: int | None,
) -> None:
    args, kwargs = arguments(kind(docs), errors, replace_str, keywords=keywords)
    mine = call(pt.unidecode, *args, threads=threads, **kwargs)
    theirs, failed = oracle_loop(docs, errors, replace_str)
    if theirs.error is None:
        assert isinstance(theirs.result, list)
        theirs.result = kind(theirs.result)
    note = None if failed is None else f"pyroid: in the str at index {failed}"
    assert_same(mine, theirs, note=note)
    if mine.error is None:
        assert isinstance(mine.result, kind)
        for out, doc in zip(mine.result, docs, strict=True):
            assert type(out) is str
            if doc.isascii():
                assert out is doc


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@examples
@given(
    docs=st.lists(st.none() | ANY_ARROW_TEXT, max_size=6),
    errors=ERRORS,
    replace_str=REPLACE,
    large=st.booleans(),
    chunks=st.integers(1, 3),
    threads=THREADS,
)
@example(
    docs=["ab", None, "\u00e9", "x\ue000"],
    errors="strict",
    replace_str=ABSENT,
    large=False,
    chunks=2,
    threads=None,
)
def test_arrow_matches_a_loop(
    docs: list[str | None],
    errors: object,
    replace_str: object,
    large: bool,
    chunks: int,
    threads: int | None,
) -> None:
    col = _testing.column(docs, large=large, chunks=chunks)
    args, kwargs = arguments(col, errors, replace_str, keywords=False)
    mine = call(pt.unidecode, *args, threads=threads, **kwargs)
    if errors == "replace" and isinstance(replace_str, str) and has_surrogate(replace_str):
        # The output could hold a lone surrogate, which an Arrow string cannot.
        assert isinstance(mine.error, UnicodeEncodeError)
        assert mine.error.__notes__ == [ARROW_SURROGATE_NOTE]
        assert mine.warned == []
        return
    theirs, failed = oracle_loop(docs, errors, replace_str)
    if isinstance(mine.result, pyroid.Column):
        mine.result = mine.result.to_pylist()
    note = None if failed is None else f"pyroid: in row {failed} of the Arrow input"
    assert_same(mine, theirs, note=note)


def _large_batch(kind: str) -> list[str | None]:
    """300 documents without missing characters, then 200 with them (and, for str input,
    lone surrogates), so that the parallel path has work before the first failure."""
    agree, both_missing = alphabets()
    rng = random.Random(2026)
    clean = [*string.ascii_letters, " ", *agree]
    pools = [clean, list(both_missing)]
    if kind != "arrow":
        pools.append([chr(cp) for cp in range(0xD800, 0xE000, 61)])

    def dirty() -> str:
        return "".join(rng.choice(rng.choice(pools)) for _ in range(rng.randrange(1, 20)))

    docs: list[str | None] = ["".join(rng.choices(clean, k=rng.randrange(20))) for _ in range(300)]
    docs += [dirty() for _ in range(200)]
    if kind == "arrow":
        docs = [None if i % 11 == 5 else d for i, d in enumerate(docs)]
    return docs


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@pytest.mark.parametrize("threads", [None, 1, 4])
@pytest.mark.parametrize("kind", ["list", "tuple", "arrow"])
@pytest.mark.parametrize(
    ("errors", "replace_str"),
    [
        ("ignore", "?"),
        ("replace", "<>"),
        ("preserve", "?"),
        ("strict", "?"),
        ("bogus", "?"),
        ("replace", None),
    ],
)
def test_large_batch_matches_a_loop(
    kind: str, threads: int | None, errors: str, replace_str: str | None
) -> None:
    docs = _large_batch(kind)
    batch: object
    if kind == "arrow":
        batch = _testing.column(docs, chunks=3)
    else:
        batch = list(docs) if kind == "list" else tuple(docs)
    mine = call(pt.unidecode, batch, errors, replace_str, threads=threads)
    theirs, failed = oracle_loop(docs, errors, replace_str)
    if isinstance(mine.result, pyroid.Column):
        mine.result = mine.result.to_pylist()
    elif isinstance(mine.result, tuple):
        mine.result = list(mine.result)
    if failed is None:
        note = None
    elif kind == "arrow":
        note = f"pyroid: in row {failed} of the Arrow input"
    else:
        note = f"pyroid: in the str at index {failed}"
    assert_same(mine, theirs, note=note)
    if errors in ("ignore", "preserve") and kind != "arrow":
        assert len(mine.warned) > 50  # one per lone surrogate, in document order


# ============================================================== pyroid-only behaviour

# anyascii maps these to "" on purpose. Unidecode treats some of them as missing (a
# documented difference), so only pyroid's behaviour is checked.
DELIBERATELY_EMPTY = {
    "U+0301-combining-acute": "\u0301",
    "U+200D-zwj": "\u200d",
    "U+FE0F-vs16": "\ufe0f",
    "U+0640-tatweel": "\u0640",
    "U+1F3FB-skin-tone": "\U0001f3fb",
}
EVERY_MODE: list[tuple[object, object]] = [
    ("ignore", "?"),
    ("replace", "X"),
    ("replace", None),
    ("strict", "?"),
    ("preserve", "?"),
    ("bogus", "?"),
]


@pytest.mark.parametrize("char", DELIBERATELY_EMPTY.values(), ids=DELIBERATELY_EMPTY.keys())
@pytest.mark.parametrize(("errors", "replace_str"), EVERY_MODE)
def test_deliberately_empty_mappings_never_count_as_missing(
    char: str, errors: object, replace_str: object
) -> None:
    assert pt.anyascii(char) == ""
    text = f"a{char}b{char}"
    assert unidecode_any(text, errors, replace_str) == "ab"
    assert unidecode_any([text, char], errors, replace_str) == ["ab", ""]
    assert unidecode_any((char,), errors, replace_str) == ("",)
    col = unidecode_any(_testing.column([text, None, char]), errors, replace_str)
    assert col.to_pylist() == ["ab", None, ""]


@given(
    text=st.text(st.sampled_from([*DELIBERATELY_EMPTY.values(), "a", " ", "\x00"])),
    mode=st.sampled_from(EVERY_MODE),
)
def test_text_of_deliberately_empty_mappings_never_fails(
    text: str, mode: tuple[object, object]
) -> None:
    assert unidecode_any(text, *mode) == "".join(c for c in text if c.isascii())


# The pickle round trips below only load bytes the same test has just dumped.


def test_unidecode_error_class() -> None:
    assert issubclass(pt.UnidecodeError, ValueError)
    assert issubclass(pt.UnidecodeError, pyroid.PyroidError)
    for err in (
        pt.UnidecodeError("msg", index=4),
        pt.UnidecodeError("msg", 4),
        pt.UnidecodeError("msg"),
    ):
        back = pickle.loads(pickle.dumps(err))
        assert type(back) is pt.UnidecodeError
        assert (back.args, back.index, str(back)) == (err.args, err.index, str(err))


def test_raised_unidecode_error_pickles_with_index_and_note() -> None:
    with pytest.raises(pt.UnidecodeError) as info:
        pt.unidecode(["ok", "ab\ue000"], "strict")
    back = pickle.loads(pickle.dumps(info.value))
    assert type(back) is pt.UnidecodeError
    assert back.index == 2
    assert back.args == ("no replacement found for character '\\ue000' in position 2",)
    assert back.__notes__ == ["pyroid: in the str at index 1"]


def _construct_and_pickle(cls: Any, args: tuple[object, ...], kwargs: dict[str, object]) -> Any:
    err = cls(*args, **kwargs)
    back = pickle.loads(pickle.dumps(err))
    return (err.args, err.index, str(err), back.args, back.index, isinstance(err, ValueError))


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("msg",), {}),
        (("msg", 4), {}),
        (("msg",), {"index": 4}),
        ((), {"message": "m", "index": 0}),
        (("msg", None), {}),
        ((), {}),
    ],
)
def test_unidecode_error_constructor_matches(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> None:
    mine = call(_construct_and_pickle, pt.UnidecodeError, args, kwargs)
    theirs = call(_construct_and_pickle, oracle.UnidecodeError, args, kwargs)
    assert_same(mine, theirs)


@pytest.mark.parametrize("replace_str", ["\ud800", "a\udfffb"])
@pytest.mark.parametrize("docs", [["ab", None, "x\ue000"], ["ab"], []])
def test_arrow_with_a_surrogate_replace_str_raises(
    docs: list[str | None], replace_str: str
) -> None:
    with pytest.raises(UnicodeEncodeError) as info:
        pt.unidecode(_testing.column(docs), "replace", replace_str)
    assert info.value.__notes__ == [ARROW_SURROGATE_NOTE]
    # The other modes never use replace_str.
    for errors in ("ignore", "preserve"):
        out = pt.unidecode(_testing.column(docs), errors, replace_str)
        assert out.to_pylist() == pt.unidecode(_testing.column(docs), errors).to_pylist()


@pytest.mark.thread_unsafe(reason="warnings.catch_warnings")
def test_surrogate_warning_is_attributed_to_the_caller() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pt.unidecode(["a\ud800", "\udfff"])
    assert [(w.category, w.filename) for w in caught] == [(RuntimeWarning, __file__)] * 2
