# Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
"""pyroid.ftfy.fix_text_batch: every container kind, Arrow, threads, surrogates.

Each result must equal ``ftfy.fix_text(doc)`` for the same options.
"""

# The test data is full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import os
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
from pyroid import ftfy as candidate
from pyroid._pyroid import _testing

ftfy: Any = pytest.importorskip("ftfy")
if ftfy.__version__ != "6.3.1":
    pytest.skip(f"needs ftfy 6.3.1, found {ftfy.__version__}", allow_module_level=True)

_EXAMPLES = os.environ.get("PYROID_FTFY_EXAMPLES")
examples = (
    settings(suppress_health_check=[HealthCheck.too_slow], max_examples=int(_EXAMPLES))
    if _EXAMPLES
    else settings(suppress_health_check=[HealthCheck.too_slow])
)

DOCS = [
    "plain ascii",
    "Ã©tÃ© â€œquotedâ€\x9d",
    "&lt;b&gt; &amp;amp;",
    "ﬁne ＷＩＤＥ",
    "line one\r\nline two ",
    "\x1b[31mred\x1b[0m\x00",
    "",
    "日本語はそのまま",
    "é composed",
    "voilÃ le travail",
]
WITH_SURROGATES = ["ok", "😀 pair", "lone \udc00", "Ã© and \ud800"]


def expected(docs: list[str], **kwargs: Any) -> list[str]:
    return [ftfy.fix_text(d, **kwargs) for d in docs]


def test_str_list_tuple_kinds() -> None:
    assert candidate.fix_text_batch(DOCS[1]) == ftfy.fix_text(DOCS[1])
    out_list = candidate.fix_text_batch(DOCS)
    assert type(out_list) is list
    assert out_list == expected(DOCS)
    out_tuple = candidate.fix_text_batch(tuple(DOCS))
    assert type(out_tuple) is tuple
    assert list(out_tuple) == expected(DOCS)
    assert candidate.fix_text_batch([]) == []
    assert candidate.fix_text_batch(()) == ()


def test_unchanged_documents_are_the_same_objects() -> None:
    docs = ["unchanged " + str(i) for i in range(100)] + ["Ã©"]
    out = candidate.fix_text_batch(docs)
    assert all(o is d for o, d in zip(out[:-1], docs[:-1], strict=True))
    assert out[-1] == "é"


def test_options_and_configs() -> None:
    kwargs: dict[str, Any] = {"uncurl_quotes": False, "normalization": "NFKC"}
    assert candidate.fix_text_batch(DOCS, **kwargs) == expected(DOCS, **kwargs)
    cfg = candidate.TextFixerConfig(fix_encoding=False)
    assert candidate.fix_text_batch(DOCS, cfg) == [ftfy.fix_text(d, cfg) for d in DOCS]
    ftfy_cfg = ftfy.TextFixerConfig(unescape_html=False)
    assert candidate.fix_text_batch(DOCS, ftfy_cfg) == [ftfy.fix_text(d, ftfy_cfg) for d in DOCS]


def test_lone_surrogates_are_fixed_like_ftfy() -> None:
    options: list[dict[str, Any]] = [{}, {"fix_surrogates": False}]
    for kwargs in options:
        assert candidate.fix_text_batch(WITH_SURROGATES, **kwargs) == expected(
            WITH_SURROGATES, **kwargs
        )
        assert candidate.fix_text_batch(tuple(WITH_SURROGATES), **kwargs) == tuple(
            expected(WITH_SURROGATES, **kwargs)
        )
        for doc in WITH_SURROGATES:
            assert candidate.fix_text_batch(doc, **kwargs) == ftfy.fix_text(doc, **kwargs)
    out = candidate.fix_text_batch(WITH_SURROGATES)
    assert out[0] is WITH_SURROGATES[0]


@pytest.mark.parametrize("bad", [b"bytes", 3, None, ["ok", 3], ("ok", b"x"), {"a"}])
def test_wrong_types_raise_input_error(bad: Any) -> None:
    with pytest.raises(pyroid.InputError):
        candidate.fix_text_batch(bad)
    surrogate_and_bad: list[Any] = ["\udc00", 3]
    with pytest.raises(pyroid.InputError, match="index 1"):
        candidate.fix_text_batch(surrogate_and_bad)


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    values: list[str | None] = [*DOCS[:5], None, *DOCS[5:]]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    out = candidate.fix_text_batch(col)
    assert isinstance(out, pyroid.Column)
    assert out.type == ("large_string" if large else "string")
    assert out.name == "text"
    assert out.null_count == 1
    assert out.to_pylist() == [None if v is None else ftfy.fix_text(v) for v in values]


def test_arrow_with_pyarrow(pa: Any) -> None:
    values: list[str | None] = [*DOCS, None]
    want = [None if v is None else ftfy.fix_text(v) for v in values]
    arrays = [
        pa.array(values),
        pa.array(values, pa.large_string()),
        pa.array(values, pa.string_view()),
        pa.chunked_array([values[:4], values[4:]]),
        pa.array(["skip", *values])[1:],  # offset > 0
    ]
    for arr in arrays:
        out = candidate.fix_text_batch(arr)
        assert pa.chunked_array(out).to_pylist() == want
    with pytest.raises(pyroid.InputError):
        candidate.fix_text_batch(pa.array([1, 2]))


def test_arrow_with_unsupported_option_raises() -> None:
    with pytest.raises(pyroid.PyroidError, match="Arrow"):
        candidate.fix_text_batch(_testing.column(["x"]), max_decode_length=5.0)


@pytest.mark.parametrize("threads", [None, 1, 2, 4, 8])
def test_threads_give_identical_results(threads: int | None) -> None:
    docs = DOCS * 50 + WITH_SURROGATES * 10
    assert candidate.fix_text_batch(docs, threads=threads) == expected(docs)


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_thread_counts(bad: int) -> None:
    with pytest.raises(ValueError, match="threads"):
        candidate.fix_text_batch(DOCS, threads=bad)


@pytest.mark.thread_unsafe(reason="10k documents through ftfy: too slow per thread")
@pytest.mark.timeout(240)
def test_ten_thousand_documents() -> None:
    docs = [f"{d} #{i}" for i in range(1000) for d in DOCS]
    assert len(docs) == 10_000
    assert candidate.fix_text_batch(docs) == expected(docs)
    assert candidate.fix_text_batch(_testing.column(docs, chunks=4)).to_pylist() == expected(docs)


pieces = st.one_of(
    st.sampled_from([*DOCS, "\x85", "\x92", "&#x1F600;", "Ã ", "\r"]),
    st.text(st.characters(codec="utf-8"), max_size=12),
    st.characters(categories=["Cs"]),
)
texts = st.lists(pieces, max_size=6).map("".join)


@examples
@given(docs=st.lists(texts, max_size=12), kind=st.sampled_from(["str", "list", "tuple"]))
def test_matches_ftfy(docs: list[str], kind: str) -> None:
    if kind == "str":
        for d in docs:
            assert candidate.fix_text_batch(d) == ftfy.fix_text(d)
    elif kind == "list":
        assert candidate.fix_text_batch(docs) == expected(docs)
    else:
        assert candidate.fix_text_batch(tuple(docs)) == tuple(expected(docs))


@examples
@given(docs=st.lists(st.none() | st.text(st.characters(codec="utf-8"), max_size=20), max_size=12))
def test_arrow_matches_ftfy(docs: list[str | None]) -> None:
    out = candidate.fix_text_batch(_testing.column(docs, chunks=2 if len(docs) > 1 else 1))
    assert out.to_pylist() == [None if d is None else ftfy.fix_text(d) for d in docs]
