"""pyroid.translit.anyascii against the ``anyascii`` package 0.3.3 (ISC), its reference.

- **Every code point** 0..=0x10FFFF: one per str (in one list call, and one call per str),
  and concatenated (in chunks of a few thousand code points, between ASCII runs of every
  length, and as one str), through the str, list, tuple and Arrow paths. Lone surrogates,
  which only a Python str can hold, are dropped by both; a batch holding one takes pyroid's
  WTF-8 path, so the surrogate-free batches are checked separately too.
- **Hypothesis** text, lone surrogates mixed in, in every container kind (Arrow without
  surrogates).
- **Real text**: FineWeb-2 (one parquet file per language) and FineWeb, if ``/data`` has them.
- ``unidecode()`` in its default mode (``errors="ignore"``) gives anyascii's output
  everywhere: both leave out characters without a mapping and lone surrogates.
"""

import warnings
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.translit as tr
from pyroid._pyroid import _testing

_anyascii: Any = pytest.importorskip("anyascii")
if _anyascii.__version__ != "0.3.3":
    pytest.skip(f"needs anyascii 0.3.3, found {_anyascii.__version__}", allow_module_level=True)
ref: Callable[[str], str] = _anyascii.anyascii

SURROGATES = range(0xD800, 0xE000)
N_CODE_POINTS = 0x110000
FINEWEB2 = Path("/data/fineweb2")
FINEWEB = Path("/data/fineweb/fineweb-dev-10k.parquet")

examples = settings(suppress_health_check=[HealthCheck.too_slow])


def code_points(*, surrogates: bool) -> list[int]:
    """Every code point, with or without the lone surrogates."""
    return [cp for cp in range(N_CODE_POINTS) if surrogates or cp not in SURROGATES]


def assert_same(
    got: Sequence[str | None], want: Sequence[str | None], inputs: Sequence[str | None]
) -> None:
    """``got == want``, reporting the first mismatches by input."""
    assert len(got) == len(want) == len(inputs)
    if list(got) == list(want):
        return
    bad = [
        (i, x, g, w) for i, (x, g, w) in enumerate(zip(inputs, got, want, strict=True)) if g != w
    ]
    shown = "\n".join(
        f"  [{i}] {_describe(x)}: pyroid {g!r:.200}, anyascii {w!r:.200}" for i, x, g, w in bad[:20]
    )
    pytest.fail(f"{len(bad)} of {len(got)} differ from anyascii:\n{shown}")


def _describe(x: str | None) -> str:
    if x is None:
        return "None"
    if len(x) == 1:
        return f"U+{ord(x):04X}"
    return f"{x!r:.80} (len {len(x)})"


def chunked(items: Sequence[str], size: int) -> Iterator[str]:
    for i in range(0, len(items), size):
        yield "".join(items[i : i + size])


@pytest.fixture(scope="module")
def singles() -> list[str]:
    """chr(cp) for every code point, lone surrogates included."""
    return [chr(cp) for cp in range(N_CODE_POINTS)]


@pytest.fixture(scope="module")
def singles_want(singles: list[str]) -> list[str]:
    return [ref(s) for s in singles]


def _scalars(singles: list[str]) -> list[str]:
    return [s for s in singles if ord(s) not in SURROGATES]


def test_reference_version() -> None:
    assert _anyascii.__version__ == tr.ANYASCII_VERSION == "0.3.3"


# ---- every code point ------------------------------------------------------


@pytest.mark.thread_unsafe(reason="1.1M strs, several times over")
def test_every_code_point_in_one_list_call(singles: list[str], singles_want: list[str]) -> None:
    # The batch holds lone surrogates, so all of it takes the WTF-8 path.
    assert_same(tr.anyascii(singles), singles_want, singles)
    assert_same(tr.anyascii(tuple(singles)), singles_want, singles)


@pytest.mark.thread_unsafe(reason="1.1M strs, several times over")
def test_every_scalar_value_in_one_list_call(singles: list[str], singles_want: list[str]) -> None:
    docs = _scalars(singles)
    want = _scalars_want(singles, singles_want)
    assert_same(tr.anyascii(docs), want, docs)
    assert_same(tr.anyascii(docs, threads=1), want, docs)


def _scalars_want(singles: list[str], singles_want: list[str]) -> list[str]:
    return [w for s, w in zip(singles, singles_want, strict=True) if ord(s) not in SURROGATES]


@pytest.mark.thread_unsafe(reason="1.1M calls")
def test_every_code_point_one_str_per_call(singles: list[str], singles_want: list[str]) -> None:
    assert_same([tr.anyascii(s) for s in singles], singles_want, singles)


@pytest.mark.thread_unsafe(reason="1.1M strs, several times over")
@pytest.mark.parametrize("surrogates", [False, True], ids=["scalars", "with-surrogates"])
@pytest.mark.parametrize("size", [4096, 4093, 61])
def test_every_code_point_in_chunks(singles: list[str], surrogates: bool, size: int) -> None:
    # 4096 code points per str: block-aligned; 4093 and 61: chunk edges fall
    # everywhere in anyascii's 256-code-point blocks and in the ASCII range.
    items = singles if surrogates else _scalars(singles)
    docs = list(chunked(items, size))
    want = [ref(d) for d in docs]
    assert_same(tr.anyascii(docs), want, docs)
    assert_same(list(tr.anyascii(tuple(docs))), want, docs)
    assert_same([tr.anyascii(d) for d in docs], want, docs)


@pytest.mark.thread_unsafe(reason="1.1M strs, several times over")
@pytest.mark.parametrize("surrogates", [False, True], ids=["scalars", "with-surrogates"])
def test_every_code_point_between_ascii_runs(singles: list[str], surrogates: bool) -> None:
    # ASCII runs of 0..16 bytes put each character at every offset of the
    # 8-byte ASCII scan, before and after it.
    items = singles if surrogates else _scalars(singles)
    padded = [s + "x" * (i % 17) for i, s in enumerate(items)]
    docs = list(chunked(padded, 1000))
    want = [ref(d) for d in docs]
    assert_same(tr.anyascii(docs), want, docs)
    assert_same([tr.anyascii(d) for d in docs], want, docs)


@pytest.mark.thread_unsafe(reason="a 4 MB str, several times over")
@pytest.mark.parametrize("surrogates", [False, True], ids=["scalars", "with-surrogates"])
def test_every_code_point_in_one_str(singles: list[str], surrogates: bool) -> None:
    doc = "".join(singles if surrogates else _scalars(singles))
    want = ref(doc)
    assert tr.anyascii(doc) == want
    assert tr.anyascii([doc, doc]) == [want, want]


@pytest.mark.thread_unsafe(reason="1.1M strs, several times over")
@pytest.mark.parametrize("large", [False, True])
def test_every_scalar_value_through_arrow_without_pyarrow(
    singles: list[str], singles_want: list[str], large: bool
) -> None:
    docs = _scalars(singles)
    want = _scalars_want(singles, singles_want)
    out = tr.anyascii(_testing.column(docs, large=large, chunks=7))
    assert out.type == ("large_string" if large else "string")
    assert_same(out.to_pylist(), want, docs)
    blocks = list(chunked(docs, 4093))
    out = tr.anyascii(_testing.column(blocks, large=large, chunks=3))
    assert_same(out.to_pylist(), [ref(b) for b in blocks], blocks)


@pytest.mark.thread_unsafe(reason="1.1M strs, several times over")
@pytest.mark.parametrize("type_name", ["string", "large_string", "string_view"])
def test_every_scalar_value_through_pyarrow(
    pa: Any, singles: list[str], singles_want: list[str], type_name: str
) -> None:
    docs = _scalars(singles)
    want = _scalars_want(singles, singles_want)
    arr = pa.array(docs, getattr(pa, type_name)())
    assert_same(tr.anyascii(arr).to_pylist(), want, docs)
    # A multi-chunk column of slices (offset > 0), and the same code points concatenated.
    sliced = pa.chunked_array([arr.slice(1, 300_000), arr.slice(300_001)])
    assert_same(tr.anyascii(sliced).to_pylist(), want[1:], docs[1:])
    blocks = list(chunked(docs, 4096))
    out = tr.anyascii(pa.array(blocks, getattr(pa, type_name)()))
    assert_same(out.to_pylist(), [ref(b) for b in blocks], blocks)


# ---- Hypothesis ------------------------------------------------------------

# Characters anyascii maps to "" on purpose or has no mapping for, multi-char
# sequences (ZWJ emoji, flags, decomposed accents), long transliterations, and
# ASCII runs around them.
PIECES = [
    "a\u0301",
    "\U0001f44d\U0001f3fd",
    "\U0001f468\u200d\U0001f469\u200d\U0001f467",
    "\U0001f1fa\U0001f1f8",
    "\u2764\ufe0f",
    "\u5317\u4eac",
    "\ud55c\uad6d\uc5b4",
    "\ue000",
    "\U000f0000",
    "\uffff",
    "\x85",
    "\ufeff",
    "\u0640",
    "\ufdfa",
    "\u0627\u0644",
    "\u216b",
    "\U00018000",
    "\U00020000",
    "\U0002ebe0",
    "\u200b",
    "\U0001f600",
    "xxxxxxxxx",
    " ",
]
chars = st.one_of(
    st.characters(),
    st.characters(categories=["Cs"]),
    st.characters(max_codepoint=0x7F),
)
pieces = st.one_of(st.sampled_from(PIECES), st.text(chars, max_size=10))
texts = st.lists(pieces, max_size=8).map("".join)
utf8_chars = st.characters(codec="utf-8")
utf8_texts = st.lists(
    st.sampled_from(PIECES) | st.text(utf8_chars, max_size=10),
    max_size=8,
).map("".join)


@examples
@given(docs=st.lists(texts, max_size=12), kind=st.sampled_from(["str", "list", "tuple"]))
def test_matches_anyascii(docs: list[str], kind: str) -> None:
    want = [ref(d) for d in docs]
    if kind == "str":
        for d, w in zip(docs, want, strict=True):
            assert tr.anyascii(d) == w
    elif kind == "list":
        assert tr.anyascii(docs) == want
        if docs:  # the same documents past the pool threshold
            reps = 65 // len(docs) + 1
            assert tr.anyascii(docs * reps) == want * reps
    else:
        assert tr.anyascii(tuple(docs)) == tuple(want)


@examples
@given(
    docs=st.lists(st.none() | utf8_texts, max_size=12),
    large=st.booleans(),
    chunks=st.integers(1, 3),
)
def test_arrow_matches_anyascii(docs: list[str | None], large: bool, chunks: int) -> None:
    out = tr.anyascii(_testing.column(docs, large=large, chunks=chunks))
    assert out.to_pylist() == [None if d is None else ref(d) for d in docs]


@examples
@given(
    docs=st.lists(st.none() | utf8_texts, max_size=12),
    type_name=st.sampled_from(["string", "large_string", "string_view"]),
)
def test_pyarrow_matches_anyascii(pa: Any, docs: list[str | None], type_name: str) -> None:
    arr = pa.array(["skip", *docs], getattr(pa, type_name)())[1:]
    out = tr.anyascii(arr)
    assert out.to_pylist() == [None if d is None else ref(d) for d in docs]


# ---- FineWeb-2 and FineWeb -------------------------------------------------


def _corpora() -> list[Any]:
    params = [pytest.param(p, id=p.stem) for p in sorted(FINEWEB2.glob("*.parquet"))]
    if not params:
        params.append(
            pytest.param(
                FINEWEB2, id="fineweb2", marks=pytest.mark.skip(reason=f"no {FINEWEB2}/*.parquet")
            )
        )
    marks = [] if FINEWEB.exists() else [pytest.mark.skip(reason=f"no {FINEWEB}")]
    params.append(pytest.param(FINEWEB, id="fineweb", marks=marks))
    return params


@pytest.mark.thread_unsafe(reason="whole corpus files in memory")
@pytest.mark.timeout(240)
@pytest.mark.parametrize("path", _corpora())
def test_corpus_matches_anyascii(pa: Any, path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    col = pq.read_table(path, columns=["text"]).column("text")
    docs: list[str | None] = col.to_pylist()
    strs = [d for d in docs if d is not None]
    assert strs, path
    want = [ref(d) for d in strs]
    assert_same(tr.anyascii(strs), want, strs)
    assert_same(tr.unidecode(strs), want, strs)
    want_col = [None if d is None else ref(d) for d in docs]
    out = tr.anyascii(col)
    assert isinstance(out, pyroid.Column)
    assert_same(out.to_pylist(), want_col, docs)
    assert_same(tr.unidecode(col).to_pylist(), want_col, docs)
    assert_same(tr.anyascii(col.cast(pa.large_string())).to_pylist(), want_col, docs)
    assert tr.is_ascii(strs) == [d.isascii() for d in strs]


# ---- unidecode(errors="ignore") is anyascii --------------------------------


@pytest.mark.thread_unsafe(reason="1.1M strs, several times over")
def test_unidecode_every_scalar_value(singles: list[str], singles_want: list[str]) -> None:
    docs = _scalars(singles)
    want = _scalars_want(singles, singles_want)
    assert_same(tr.unidecode(docs), want, docs)
    assert_same(tr.unidecode(docs, "ignore"), want, docs)
    assert_same(tr.unidecode_expect_ascii(docs), want, docs)
    assert_same(tr.unidecode_expect_nonascii(docs), want, docs)
    assert_same(tr.unidecode(_testing.column(docs, chunks=5)).to_pylist(), want, docs)
    blocks = list(chunked(docs, 4093))
    assert_same(tr.unidecode(blocks), [ref(b) for b in blocks], blocks)


@pytest.mark.thread_unsafe(reason="warnings filter; 1.1M strs")
def test_unidecode_every_code_point(singles: list[str], singles_want: list[str]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # one per lone surrogate
        assert_same(tr.unidecode(singles), singles_want, singles)
        assert_same(
            [tr.unidecode(s) for s in singles[0xD700:0xE100]],
            singles_want[0xD700:0xE100],
            singles[0xD700:0xE100],
        )
        doc = "".join(singles)
        assert tr.unidecode(doc) == ref(doc)


@pytest.mark.thread_unsafe(reason="warnings filter")
@examples
@given(docs=st.lists(texts, max_size=12), kind=st.sampled_from(["str", "list", "tuple"]))
def test_unidecode_default_matches_anyascii(docs: list[str], kind: str) -> None:
    want = [ref(d) for d in docs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # one per lone surrogate
        if kind == "str":
            for d, w in zip(docs, want, strict=True):
                assert tr.unidecode(d) == w
                assert tr.unidecode(d, errors="ignore") == w
        elif kind == "list":
            assert tr.unidecode(docs) == want
        else:
            assert tr.unidecode(tuple(docs)) == tuple(want)


@examples
@given(docs=st.lists(st.none() | utf8_texts, max_size=12), large=st.booleans())
def test_unidecode_default_matches_anyascii_arrow(docs: list[str | None], large: bool) -> None:
    out = tr.unidecode(_testing.column(docs, large=large, chunks=2 if len(docs) > 1 else 1))
    assert out.to_pylist() == [None if d is None else ref(d) for d in docs]
