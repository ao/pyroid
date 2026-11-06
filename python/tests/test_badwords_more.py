"""pyroid.filters C4 bad words: datatrove 0.10.1's C4BadWordsFilter, same decisions and draws.

The reference is datatrove's own regex construction over caller-supplied word lists, so most tests
need neither datatrove nor a download. Tests on datatrove's real (LDNOOBW, CC BY 4.0) lists run
when they are already in pyroid's cache or ``PYROID_TEST_DOWNLOADS=1`` allows fetching them.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.filters as pf
from pyroid._pyroid import _testing

REASON = "document_removed_with_badwords"


def ref_found(words: list[str], text: str, boundaries: bool = True) -> bool:
    """datatrove's ``_get_badwords`` regex, searched in ``text.lower()``."""
    alts = "|".join(re.escape(w) for w in words)
    regex = re.compile(rf"(?:\W|^)({alts})(?:\W|$)") if boundaries else re.compile(alts)
    return regex.search(text.lower()) is not None


WORDS = ["bad", "two words", "x-y", "ünï", "σς", "@$$", "c++", "坏", "a"]
PIECES = [
    *WORDS,
    *" .,-_\n\t!",
    "BAD",
    "Bad",
    "İ",
    "Σ",
    "ΑΣ",
    "é",
    "́",
    "ß",
    "ǅ",
    "x",
    "9",
    "_",
    "\u2028",
    "\u00a0",
]
TEXT = st.lists(st.sampled_from(PIECES), max_size=12).map("".join)
WORD_LISTS = st.lists(
    st.one_of(st.sampled_from(WORDS), st.lists(st.sampled_from(PIECES), max_size=3).map("".join)),
    min_size=1,
    max_size=6,
)


@settings(max_examples=400, deadline=None, suppress_health_check=list(HealthCheck))
@given(WORD_LISTS, st.lists(TEXT, min_size=1, max_size=6), st.booleans())
def test_found_matches_datatrove_regex(words: list[str], texts: list[str], cjk: bool) -> None:
    lang = "zh" if cjk else "en"
    f = pf.BadWordsFilter(words={lang: words}, default_language=lang)
    assert f.found(texts) == [ref_found(words, t, boundaries=not cjk) for t in texts]


def test_examples() -> None:
    f = pf.BadWordsFilter(words={"en": ["bad", "two words"], "ja": ["悪い"]})
    texts = ["This is BAD.", "badly", "_bad", "Two  Words", "two words", "とても悪い", ""]
    assert f.found(texts) == [True, False, False, False, True, False, False]
    assert f.found(texts, languages="ja") == [False] * 5 + [True, False]
    assert pf.c4_badwords(texts, words={"en": ["bad"]}) == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    assert pf.c4_badwords_reasons("bad", words={"en": ["bad"]}) == REASON
    assert pf.c4_badwords_reasons("good", words={"en": ["bad"]}) is None


def test_empty_word_like_datatrove() -> None:
    texts = ["", " ", "ab", "a b", "a  b", " ab", "ab "]
    for cjk in (False, True):
        lang = "th" if cjk else "en"
        f = pf.BadWordsFilter(words={lang: [""]}, default_language=lang)
        assert f.found(texts) == [ref_found([""], t, boundaries=not cjk) for t in texts]


def _numpy_draws(seed: int, n: int) -> list[float]:
    np = pytest.importorskip("numpy")
    g = np.random.default_rng(seed)
    return [float(g.uniform()) for _ in range(n)]


@pytest.mark.parametrize("seed", [0, 1, 42, 2**32 + 5, 2**200 + 7, 123456789012345])
def test_draws_are_numpys(seed: int) -> None:
    draws = _numpy_draws(seed, 40)
    f = pf.BadWordsFilter(words={"en": ["bad"]}, keep_fraction=0.5, seed=seed)
    texts = ["bad", "good", "bad"] * 10  # 20 flagged documents
    keep = f.keep(texts[:7]) + f.keep(texts[7:])
    flagged = iter(draws)
    want = [next(flagged) < 0.5 if t == "bad" else True for t in texts]
    assert keep == want


def test_rng_state_is_numpys_and_assignable() -> None:
    np = pytest.importorskip("numpy")
    g = np.random.default_rng(7)
    f = pf.BadWordsFilter(seed=7, words={"en": ["bad"]}, keep_fraction=0.25)
    s = g.bit_generator.state["state"]
    assert f.rng_state == (s["state"], s["inc"])
    f.keep(["bad"] * 5)
    for _ in range(5):
        g.uniform()
    s = g.bit_generator.state["state"]
    assert f.rng_state == (s["state"], s["inc"])
    f.rng_state = (s["state"], s["inc"])
    assert f.keep(["bad"]) == [g.uniform() < 0.25]


def test_call_gives_keep_and_reasons_from_one_pass() -> None:
    f = pf.BadWordsFilter(words={"en": ["bad"]}, keep_fraction=0.5, seed=3)
    g = pf.BadWordsFilter(words={"en": ["bad"]}, keep_fraction=0.5, seed=3)
    texts = ["bad"] * 20
    res = f(texts)
    assert res.keep == g.keep(texts)
    assert res.reason == [None if k else REASON for k in res.keep]
    assert f.rng_state == g.rng_state


def test_no_draws_without_keep_fraction_or_hits() -> None:
    f = pf.BadWordsFilter(words={"en": ["bad"]}, seed=1)
    before = f.rng_state
    assert f.keep(["bad", "good"]) == [False, True]
    assert f.rng_state == before
    f = pf.BadWordsFilter(words={"en": ["bad"]}, seed=1, keep_fraction=0.5)
    f.keep(["good"] * 3)
    f.found(["bad"] * 3)
    assert f.rng_state == before


def test_languages_per_document_and_missing() -> None:
    f = pf.BadWordsFilter(words={"en": ["bad"], "de": ["schlecht"]})
    texts = ["bad", "schlecht", "bad", "schlecht"]
    assert f.keep(texts, languages=["en", "de", None, "en"]) == [False, False, False, True]
    with pytest.raises(ValueError, match='There is not badwords list available for "xx"'):
        f.keep(texts, languages=["en", "xx", "en", "en"])
    lax = pf.BadWordsFilter(words={"en": ["bad"]}, fail_on_missing_language=False)
    assert lax.keep(texts, languages=["xx", "xx", "en", "yy"]) == [True, True, False, True]
    assert lax.found(["bad"], languages="xx") == [False]
    with pytest.raises(ValueError, match="one entry per document"):
        f.keep(texts, languages=["en"])


@pytest.mark.parametrize("kind", [list, tuple])
def test_container_kinds(kind: Callable[..., Any]) -> None:
    f = pf.BadWordsFilter(words={"en": ["bad"]})
    out = f.keep(kind(["bad", "ok"]))
    assert type(out) is kind
    assert list(out) == [False, True]
    assert f.keep("bad") is False
    assert f.reasons("ok") is None


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 2])
def test_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    values = ["bad", None, "ok", "so bad", None, "x"]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    f = pf.BadWordsFilter(words={"en": ["bad"]})
    keep, reason = f(col)
    assert keep.type == "bool"
    assert keep.name == "text"
    assert keep.to_pylist() == [False, None, True, False, None, True]
    assert reason.to_pylist() == [REASON, None, None, REASON, None, None]


def test_arrow_with_pyarrow(pa: Any) -> None:
    values = ["bad", None, "ok"]
    f = pf.BadWordsFilter(words={"en": ["bad"]})
    for arr in (pa.array(values, pa.string_view()), pa.chunked_array([values[:1], values[1:]])):
        assert f.keep(arr).to_pylist() == [False, None, True]


def test_errors() -> None:
    with pytest.raises(pyroid.InputError):
        pf.c4_badwords(b"x", words={"en": []})  # type: ignore[call-overload]
    with pytest.raises(UnicodeEncodeError):
        pf.c4_badwords(["\ud800"], words={"en": ["x"]})
    with pytest.raises(TypeError):
        pf.BadWordsFilter(seed="1", words={"en": []})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        pf.BadWordsFilter(seed=-1, words={"en": []})
    with pytest.raises(ValueError, match="There is not badwords list"):
        pf.load_c4_badwords("xx")


def test_load_without_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pf.load_c4_badwords("en", cache_dir=tmp_path, download=False)
    (tmp_path / "c4_badwords").mkdir()
    (tmp_path / "c4_badwords" / "en").write_text("not the list\n")
    with pytest.raises(OSError, match="sha256"):
        pf.load_c4_badwords("en", cache_dir=tmp_path, download=False)


@pytest.mark.parametrize("threads", [None, 1, 3])
def test_threads_equivalence(threads: int | None) -> None:
    docs = [f"doc {i} {'bad' if i % 7 == 0 else 'fine'} words" for i in range(400)]
    f = pf.BadWordsFilter(words={"en": ["bad"]}, keep_fraction=0.3, seed=5)
    g = pf.BadWordsFilter(words={"en": ["bad"]}, keep_fraction=0.3, seed=5)
    assert f.keep(docs, threads=threads) == g.keep(docs, threads=1)


@pytest.mark.thread_unsafe(reason="10k documents")
@pytest.mark.timeout(120)
def test_ten_thousand_documents() -> None:
    docs = [f"line {i} " + ("bad things " if i % 3 else "good things ") * 30 for i in range(10_000)]
    keep = pf.c4_badwords(docs, words={"en": ["bad", "things and"]})
    assert keep[:4] == [not ref_found(["bad", "things and"], d) for d in docs[:4]]


# ---------------------------------------------------------------- datatrove's lists and datatrove

DATA = Path(os.environ.get("PYROID_DATA", "/data"))


def _have_lists() -> bool:
    if os.environ.get("PYROID_TEST_DOWNLOADS") == "1":
        return True
    try:
        pf.load_c4_badwords("en", download=False)
    except OSError:
        return False
    return True


def _datatrove_c4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, lists: dict[str, list[str]]
) -> Any:
    """datatrove's c4_filters module, its downloads redirected to word files written here."""
    c4 = pytest.importorskip("datatrove.pipeline.filters.c4_filters")
    by_url = {}
    for lang, words in lists.items():
        p = tmp_path / lang
        p.write_text("".join(f"{w}\n" for w in words))
        by_url[c4._BADWORDS_URL + lang if lang != "en" else c4._EN_BADWORDS_URL] = str(p)
    monkeypatch.setattr(c4, "cached_asset_path_or_download", lambda url, **_: by_url[url])
    return c4


@pytest.mark.thread_unsafe(reason="monkeypatches datatrove")
@settings(max_examples=100, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    st.lists(TEXT, min_size=1, max_size=8),
    st.lists(st.sampled_from(["en", "de", "ja", "xx", None]), min_size=8, max_size=8),
    st.sampled_from([0.0, 0.3, 1.0]),
    st.integers(0, 2**70),
)
def test_drop_in_matches_live_datatrove(
    texts: list[str], langs: list[str | None], keep_fraction: float, seed: int
) -> None:
    import tempfile

    pytest.importorskip("datatrove")
    from datatrove.data import Document  # type: ignore[import-not-found,import-untyped,unused-ignore]

    from pyroid.integrations import datatrove as pdt

    with tempfile.TemporaryDirectory() as d, pytest.MonkeyPatch.context() as mp:
        c4 = _datatrove_c4(
            mp, Path(d), {"en": ["bad", "two words"], "de": ["schlecht", "ünï"], "ja": ["坏", "x"]}
        )
        kw = {"keep_fraction": keep_fraction, "seed": seed, "fail_on_missing_language": False}

        def docs() -> list[Any]:
            return [
                Document(text=t, id=str(i), metadata={} if lang is None else {"language": lang})
                for i, (t, lang) in enumerate(zip(texts, langs, strict=False))
            ]

        stock = c4.C4BadWordsFilter(**kw)
        mine = pdt.C4BadWordsFilter(**kw, batch_size=3)
        want = [stock.filter(doc) for doc in docs()]
        got = [r for b in range(0, len(texts), 3) for r in mine.filter_batch(docs()[b : b + 3])]
        assert got == want
        assert mine.uniform() == stock.uniform()
        assert mine.stats.stats.to_dict() == stock.stats.stats.to_dict()


@pytest.mark.skipif(
    not _have_lists(), reason="datatrove's lists are not cached (PYROID_TEST_DOWNLOADS=1)"
)
@pytest.mark.thread_unsafe(reason="whole corpus through datatrove")
@pytest.mark.timeout(900)
def test_corpus_matches_live_datatrove() -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    c4 = pytest.importorskip("datatrove.pipeline.filters.c4_filters")
    from datatrove.data import Document  # type: ignore[import-not-found,import-untyped,unused-ignore]

    files = [DATA / "fineweb" / "fineweb-dev-10k.parquet", DATA / "parity" / "adversarial.parquet"]
    if not all(f.exists() for f in files):
        pytest.skip("pinned corpora not available")
    texts = [
        t for f in files for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()
    ]
    cases: list[dict[str, Any]] = [{}, {"keep_fraction": 0.3, "seed": 7}]
    for kw in cases:
        stock = c4.C4BadWordsFilter(**kw)
        want = [stock.filter(Document(text=t, id=str(i))) is True for i, t in enumerate(texts)]
        got = pf.BadWordsFilter(**kw).keep(texts)
        assert sum(g != w for g, w in zip(got, want, strict=True)) == 0


def test_module_surface() -> None:
    names = {
        "C4_BADWORDS_LANGUAGES",
        "BadWordsFilter",
        "BadWordsResult",
        "C4BadWords",
        "c4_badwords",
        "c4_badwords_reasons",
        "load_c4_badwords",
    }
    assert names <= set(pf.__all__)
    assert len(pf.C4_BADWORDS_LANGUAGES) == 28
    for f in (pf.c4_badwords, pf.c4_badwords_reasons, pf.load_c4_badwords):
        assert f.__module__ == "pyroid.filters"
        assert f.__doc__
