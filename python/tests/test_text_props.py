"""Property tests: pyroid.text against its pure-Python reference.

``normalize_whitespace(s) == " ".join(s.split())`` and
``min_chars(s, n) == (len(s) >= n)`` for every surrogate-free str, in every
container kind (str, list, tuple, Arrow with and without pyarrow). Lone
surrogates raise the documented ``UnicodeEncodeError``.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import pyroid.text as t
from pyroid._pyroid import _testing

SURROGATES = range(0xD800, 0xE000)


def ref(s: str) -> str:
    return " ".join(s.split())


# Whitespace-heavy text: every str.isspace() character, plus characters that
# look like whitespace but are not (ZWSP, BOM, NUL, word joiner), multi-byte
# letters, and anything else Hypothesis likes (never surrogates).
_TRICKY = [chr(c) for c in range(0x3001) if chr(c).isspace()] + [
    "\u200b",
    "\ufeff",
    "\x00",
    "\u2060",
    "\u180e",
    "a",
    "b",
    "\xe9",
    "\u65e5",
    "\U0001f600",
]
_CHARS = st.one_of(st.sampled_from(_TRICKY), st.characters(exclude_categories=["Cs"]))
docs_st = st.text(_CHARS, max_size=40)
doc_lists = st.lists(docs_st, max_size=150)  # crosses the 64-doc pool threshold
optional_doc_lists = st.lists(st.none() | docs_st, max_size=150)
lone_surrogates = st.characters(categories=["Cs"])


@given(docs_st)
def test_str_matches_reference(s: str) -> None:
    out = t.normalize_whitespace(s)
    assert out == ref(s)
    if out == s:
        assert out is s
    assert _testing.char_lengths(s) == [len(s)]


@given(docs_st, st.integers(0, 50))
def test_min_chars_str(s: str, n: int) -> None:
    assert t.min_chars(s, n) is (len(s) >= n)


@given(doc_lists)
def test_list_matches_reference(docs: list[str]) -> None:
    out = t.normalize_whitespace(docs)
    assert out == [ref(d) for d in docs]
    for o, d in zip(out, docs, strict=True):
        if o == d:
            assert o is d


@given(doc_lists.map(tuple))
def test_tuple_matches_reference(docs: tuple[str, ...]) -> None:
    out = t.normalize_whitespace(docs)
    assert out == tuple(ref(d) for d in docs)
    for o, d in zip(out, docs, strict=True):
        if o == d:
            assert o is d


@given(doc_lists, st.integers(0, 50), st.sampled_from([None, 1, 2, 3]))
def test_min_chars_batches(docs: list[str], n: int, threads: int | None) -> None:
    want = [len(d) >= n for d in docs]
    assert t.min_chars(docs, n, threads=threads) == want
    assert t.min_chars(tuple(docs), n, threads=threads) == tuple(want)
    assert _testing.char_lengths(docs, threads=threads) == [len(d) for d in docs]


@given(optional_doc_lists, st.booleans(), st.integers(1, 5), st.integers(0, 50))
def test_column_without_pyarrow(values: list[str | None], large: bool, chunks: int, n: int) -> None:
    col = _testing.column(values, large=large, chunks=chunks)
    out = t.normalize_whitespace(col)
    assert out.type == ("large_string" if large else "string")
    assert out.to_pylist() == [None if v is None else ref(v) for v in values]
    assert out.null_count == values.count(None)
    mask = t.min_chars(col, n)
    assert mask.type == "bool"
    assert mask.to_pylist() == [None if v is None else len(v) >= n for v in values]
    assert _testing.char_lengths(col) == [None if v is None else len(v) for v in values]


_ARROW_TYPES = ["string", "large_string", "string_view"]


@given(
    optional_doc_lists,
    st.sampled_from(_ARROW_TYPES),
    st.integers(0, 10),
    st.integers(0, 50),
)
def test_pyarrow_matches_reference(
    pa: ModuleType, values: list[str | None], type_name: str, offset: int, n: int
) -> None:
    arr: Any = pa.array(values, type=getattr(pa, type_name)()).slice(offset)
    values = values[offset:]
    out = t.normalize_whitespace(arr)
    assert out.type == ("large_string" if type_name == "large_string" else "string")
    assert out.to_pylist() == [None if v is None else ref(v) for v in values]
    mask = t.min_chars(arr, n)
    assert mask.to_pylist() == [None if v is None else len(v) >= n for v in values]


@given(st.lists(optional_doc_lists, max_size=5), st.sampled_from(_ARROW_TYPES))
def test_pyarrow_chunked_matches_reference(
    pa: ModuleType, chunks: list[list[str | None]], type_name: str
) -> None:
    arr: Any = pa.chunked_array(chunks, type=getattr(pa, type_name)())
    values = [v for c in chunks for v in c]
    out = t.normalize_whitespace(arr)
    assert out.num_chunks == sum(1 for c in chunks if c)
    assert out.to_pylist() == [None if v is None else ref(v) for v in values]
    back = pa.chunked_array(out)
    assert [len(c) for c in back.chunks] == [len(c) for c in chunks if c]


# ---- lone surrogates -------------------------------------------------------


def _min_chars_1(x: Any) -> object:
    return t.min_chars(x, 1)


#: Every entry point that extracts documents (transform, predicate, map_values).
FUNCS: dict[str, Callable[[Any], object]] = {
    "normalize_whitespace": t.normalize_whitespace,
    "min_chars": _min_chars_1,
    "char_lengths": _testing.char_lengths,
}


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


@st.composite
def with_surrogate(draw: st.DrawFn) -> str:
    """A str with at least one lone surrogate somewhere in it."""
    return draw(docs_st) + draw(lone_surrogates) + draw(st.text(_CHARS | lone_surrogates))


@given(with_surrogate(), st.sampled_from(sorted(FUNCS)))
def test_surrogate_in_a_str(doc: str, fn: str) -> None:
    with pytest.raises(UnicodeEncodeError) as info:
        FUNCS[fn](doc)
    notes = _notes(info.value)
    assert "lone surrogate" in notes
    assert "index" not in notes


@given(doc_lists, st.data(), st.booleans())
def test_surrogate_in_a_batch_names_the_index(
    docs: list[str], data: st.DataObject, as_tuple: bool
) -> None:
    i = data.draw(st.integers(0, len(docs)), label="index")
    bad = data.draw(with_surrogate(), label="bad")
    batch: list[str] | tuple[str, ...] = [*docs[:i], bad, *docs[i:]]
    if as_tuple:
        batch = tuple(batch)
    for call in FUNCS.values():
        with pytest.raises(UnicodeEncodeError) as info:
            call(batch)
        notes = _notes(info.value)
        assert "lone surrogate" in notes
        assert f"index {i}" in notes


def test_surrogate_pair_code_points_are_still_lone() -> None:
    # A UTF-16 surrogate pair spelled as two code points is not U+1F600 in a
    # Python str: both halves are lone surrogates, with no UTF-8 form.
    pair = "\ud83d\ude00"
    assert len(pair) == 2
    with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
        t.normalize_whitespace(pair)
    assert t.normalize_whitespace("\U0001f600") == "\U0001f600"


@pytest.mark.parametrize("cp", [0xD800, 0xDBFF, 0xDC00, 0xDFFF])
def test_surrogate_bounds(cp: int) -> None:
    with pytest.raises(UnicodeEncodeError) as info:
        t.normalize_whitespace(["ok", "x" * 5000 + chr(cp)])
    assert "index 1" in _notes(info.value)


# ---- exhaustive: every non-surrogate code point -----------------------------


@pytest.mark.thread_unsafe(reason="~1.1M documents: too much memory per thread")
@pytest.mark.timeout(120)
def test_every_code_point_between_words() -> None:
    docs = [f"a{c}{c}b" for c in map(chr, range(0x110000)) if ord(c) not in SURROGATES]
    out = t.normalize_whitespace(docs)
    assert len(out) == len(docs) == 0x110000 - len(SURROGATES)
    want = [ref(d) for d in docs]
    mismatches = [hex(ord(d[1])) for d, o, w in zip(docs, out, want, strict=True) if o != w]
    assert mismatches == []
    changed = [d for d, o in zip(docs, out, strict=True) if o is not d]
    assert len(changed) == sum(1 for d in docs if d[1].isspace())
