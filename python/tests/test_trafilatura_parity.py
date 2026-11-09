"""pyroid.trafilatura against trafilatura 2.3.1 itself (lxml 6.1.3, jusText 3.0.2).

Every comparison is of outcomes: the returned text (``None`` included) or the exception's type and
message, and for ``bare_extraction`` every ``Document`` field (trees as
``etree.tostring(..., encoding="unicode")``). Inputs: pages from a Hypothesis tag grammar
(boilerplate classes, entities, junk markup, lone surrogates, control characters), the shared
small pages, and Common Crawl pages from ``$PYROID_TRAFILATURA_PAGES`` (a ``raw.bin`` of the dev
split: u32 count, then u32 length + UTF-8 per page; skipped when absent), drawn and mutated by
Hypothesis. Dedup option sets run as sequences from empty caches, and the caches must end equal.

Importing trafilatura imports lxml, which re-enables the GIL on free-threaded CPython: this module
is skipped there (the pinned outcomes in test_trafilatura.py run on every build).
"""

from __future__ import annotations

import configparser
import hashlib
import importlib.metadata
import importlib.util
import inspect
import os
import struct
import sys
import warnings
from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()
if FREE_THREADED:
    pytest.skip(
        "importing trafilatura re-enables the GIL (lxml); parity runs on GIL builds",
        allow_module_level=True,
    )
if importlib.util.find_spec("trafilatura") is None:
    pytest.skip("needs trafilatura 2.3.1 (the test-gil group)", allow_module_level=True)
_WANT = {"trafilatura": "2.3.1", "lxml": "6.1.3", "justext": "3.0.2"}
_FOUND = {d: importlib.metadata.version(d) for d in _WANT}
if _FOUND != _WANT:
    pytest.skip(f"needs {_WANT}, found {_FOUND}", allow_module_level=True)

with warnings.catch_warnings():
    # urllib3 warns (DependencyWarning) when trafilatura probes for SOCKS support.
    warnings.simplefilter("ignore")
    import trafilatura  # type: ignore[import-not-found,import-untyped,unused-ignore]
    import trafilatura.deduplication  # type: ignore[import-not-found,import-untyped,unused-ignore]
    import trafilatura.meta  # type: ignore[import-not-found,import-untyped,unused-ignore]
    import trafilatura.settings  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from lxml import etree  # type: ignore[import-untyped,unused-ignore]
    from lxml import html as lxml_html  # type: ignore[import-untyped,unused-ignore]

from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import pyroid  # noqa: E402
import pyroid.trafilatura as pt  # noqa: E402

oracle: Any = trafilatura


def load_pages() -> Any:
    name = "_pyroid_test_trafilatura_pages"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("trafilatura_pages.py")
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


tp = load_pages()
OPTION_SETS: dict[str, dict[str, Any]] = tp.OPTION_SETS
BARE_SETS: dict[str, dict[str, Any]] = {
    "default": {},
    "precision": {"favor_precision": True, "include_comments": False},
    "recall_dedup": {"favor_recall": True, "deduplicate": True},
}
RAW = Path(os.environ.get("PYROID_TRAFILATURA_PAGES", "/target/cc/dev/raw.bin"))

pytestmark = pytest.mark.thread_unsafe(reason="the module-level dedup caches")


def reset() -> None:
    pt.reset_caches()
    oracle.meta.reset_caches()


def normalize(result: Any) -> Any:
    """A comparable form of an extract / bare_extraction result."""
    if result is None or isinstance(result, str):
        return result
    if isinstance(result, list):
        return [normalize(r) for r in result]
    items: Iterable[tuple[Any, Any]]
    if isinstance(result, dict):
        items = result.items()
    else:
        items = ((s, getattr(result, s, None)) for s in oracle.settings.Document.__slots__)
    out = {}
    for key, value in items:
        if type(value).__module__.startswith("lxml."):
            value = etree.tostring(value, encoding="unicode")
        out[key] = value
    return out


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """The normalized result or exception, and the warning categories (FallbackWarning left out)."""
    return counted(fn, *args, **kwargs)[0]


def counted(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], int]:
    """``outcome(fn, ...)`` and the number of FallbackWarnings the call raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got: tuple[Any, ...] = ("ok", normalize(fn(*args, **kwargs)))
        except Exception as e:
            got = ("raise", type(e).__name__, str(e))
    cats = sorted({w.category.__name__ for w in caught if w.category is not pyroid.FallbackWarning})
    return (*got, cats), sum(w.category is pyroid.FallbackWarning for w in caught)


def same(fn: str, html: Any, kwargs: dict[str, Any]) -> None:
    stateful = bool(kwargs.get("deduplicate"))
    if stateful:
        reset()
    got = outcome(getattr(pt, fn), html, **kwargs)
    want = outcome(getattr(oracle, fn), html, **kwargs)
    assert got == want, f"{fn}({html[:300]!r}, **{kwargs})"
    if stateful:
        assert list(pt.LRU_TEST.cache.items()) == list(oracle.deduplication.LRU_TEST.cache.items())


# ------------------------------------------------------------------------------------- API mirror


def _shape(fn: Callable[..., Any]) -> list[tuple[str, str, Any]]:
    out = []
    for p in inspect.signature(fn).parameters.values():
        d = p.default
        if isinstance(d, configparser.ConfigParser):
            d = ("config", {k: dict(v) for k, v in d.items()})
        out.append((p.name, p.kind.name, d))
    return out


@pytest.mark.parametrize("name", ["extract", "bare_extraction", "extract_with_metadata"])
def test_signatures_match(name: str) -> None:
    ours, theirs = getattr(pt, name), getattr(oracle, name)
    assert _shape(ours) == [*_shape(theirs), ("compat", "KEYWORD_ONLY", "2.3.1")]
    assert _shape(cast(Any, pt)._api("2.3.1")[name]) == _shape(theirs)
    assert inspect.signature(theirs).parameters["config"].default is oracle.settings.DEFAULT_CONFIG


def test_settings_match() -> None:
    real = oracle.settings.DEFAULT_CONFIG
    assert {k: dict(v) for k, v in pt.DEFAULT_CONFIG.items()} == {
        k: dict(v) for k, v in real.items()
    }
    assert pt.LRU_SIZE == oracle.settings.LRU_SIZE
    assert oracle.deduplication.LRU_TEST.maxsize == pt.LRU_TEST.maxsize


def test_document_is_trafilaturas() -> None:
    doc = pt.bare_extraction(tp.PAGES["article"])
    assert isinstance(doc, oracle.settings.Document)
    assert isinstance(doc.body, etree._Element)
    assert isinstance(pt.Document(title="x"), oracle.settings.Document)
    assert pt.Document.from_dict({"text": "y"}).text == "y"


def test_lru_cache_matches_trafilaturas() -> None:
    ours, theirs = pt.LRUCache(3), oracle.deduplication.LRUCache(3)
    for key in "abcabdaeab":
        assert ours.increment(key) == theirs.increment(key)
        assert ours.get("a") == theirs.get("a")
        assert list(ours.cache.items()) == list(theirs.cache.items())


# ------------------------------------------------------------------------------- pinned and pages


@pytest.mark.parametrize("page", list(tp.PAGES))
def test_pinned_outcomes_are_trafilaturas(page: str) -> None:
    got = []
    for kwargs in OPTION_SETS.values():
        oracle.meta.reset_caches()
        try:
            o: tuple[Any, ...] = ("ok", oracle.extract(tp.PAGES[page], **kwargs))
        except Exception as e:
            o = ("raise", type(e).__name__, str(e))
        got.append(hashlib.sha256(repr(o).encode("utf-8", "surrogatepass")).hexdigest()[:16])
    assert got == tp.PINNED[page].split()


@pytest.mark.parametrize("page", list(tp.PAGES))
@pytest.mark.parametrize("option", list(OPTION_SETS))
def test_pages(page: str, option: str) -> None:
    same("extract", tp.PAGES[page], OPTION_SETS[option])


@pytest.mark.parametrize("page", list(tp.PAGES))
@pytest.mark.parametrize("option", list(BARE_SETS))
def test_pages_bare(page: str, option: str) -> None:
    same("bare_extraction", tp.PAGES[page], BARE_SETS[option])
    same("bare_extraction", tp.PAGES[page], {**BARE_SETS[option], "as_dict": True})


# ------------------------------------------------------------------------------------ Hypothesis


@pytest.mark.parametrize("option", list(OPTION_SETS))
@given(html=tp.documents())
@example(html=tp.PAGES["article"] + "<p>&amp;#" + "9" * 4301 + "</p>")
@example(html="<html><body><p>" + "𐏿 " * 90 + "</p></body></html>")
def test_hypothesis_extract(option: str, html: str) -> None:
    same("extract", html, OPTION_SETS[option])


@pytest.mark.parametrize("option", list(BARE_SETS))
@given(html=tp.documents())
def test_hypothesis_bare_extraction(option: str, html: str) -> None:
    same("bare_extraction", html, BARE_SETS[option])


@given(
    htmls=st.lists(st.one_of(tp.documents(), st.sampled_from(list(tp.PAGES.values()))), max_size=12)
)
@settings(max_examples=30)
def test_hypothesis_dedup_sequences(htmls: list[str]) -> None:
    reset()
    want = [oracle.extract(h, deduplicate=True) for h in htmls]
    assert pt.extract_batch(htmls, deduplicate=True) == want
    assert list(pt.LRU_TEST.cache.items()) == list(oracle.deduplication.LRU_TEST.cache.items())


@cache
def _raw_pages() -> tuple[str, ...]:
    data = RAW.read_bytes()
    (n,) = struct.unpack_from("<I", data, 0)
    out, off = [], 4
    for _ in range(n):
        (ln,) = struct.unpack_from("<I", data, off)
        off += 4
        out.append(data[off : off + ln].decode("utf-8", "surrogatepass"))
        off += ln
    return tuple(out)


needs_pages = pytest.mark.skipif(not RAW.is_file(), reason=f"no {RAW} (PYROID_TRAFILATURA_PAGES)")


@st.composite
def real_pages(draw: st.DrawFn) -> str:
    """A Common Crawl page, as it is or cut, spliced or with junk inserted."""
    pages = _raw_pages()
    page = pages[draw(st.integers(0, len(pages) - 1))]
    how = draw(st.sampled_from(["as is", "as is", "cut", "splice", "insert"]))
    if how == "as is" or not page:
        return page
    at = draw(st.integers(0, len(page)))
    if how == "cut":
        return page[:at]
    if how == "splice":
        other = pages[draw(st.integers(0, len(pages) - 1))]
        return page[:at] + other[draw(st.integers(0, len(other))) :]
    return page[:at] + cast(str, draw(tp.nodes)) + page[at:]


@needs_pages
@pytest.mark.parametrize("option", list(OPTION_SETS))
@given(html=real_pages())
@settings(max_examples=25)
def test_hypothesis_real_pages(option: str, html: str) -> None:
    same("extract", html, OPTION_SETS[option])


@needs_pages
@pytest.mark.timeout(900)
def test_real_pages_in_order() -> None:
    """The first 300 pages of the split through one dedup cache, datatrove's call."""
    pages = list(_raw_pages()[:300])
    kwargs: dict[str, Any] = {
        "favor_precision": True,
        "include_comments": False,
        "deduplicate": True,
    }
    reset()
    want = [oracle.extract(p, **kwargs) for p in pages]
    for threads in (1, 4, None):
        pt.reset_caches()
        assert pt.extract_batch(pages, threads=threads, **kwargs) == want
    assert list(pt.LRU_TEST.cache.items()) == list(oracle.deduplication.LRU_TEST.cache.items())
    bare = [normalize(pt.bare_extraction(p)) for p in pages[:100]]
    assert bare == [normalize(oracle.bare_extraction(p)) for p in pages[:100]]


# ----------------------------------------------------------------------------- edges and inputs

EDGE: dict[str, dict[str, Any]] = {
    "url": {"url": "https://example.org/article/1"},
    "record_id": {"record_id": "0001"},
    "tei_validation": {"tei_validation": True},
    "no_fallback": {"no_fallback": True},
    "max_tree_size": {"max_tree_size": 100},
    "precision_and_recall": {"favor_precision": True, "favor_recall": True},
    "include_formatting=False": {"include_formatting": False},
    "output_format=txt": {"output_format": "txt"},
    "date_extraction_params": {"date_extraction_params": {"extensive_search": False}},
    "empty_blacklists": {"url_blacklist": set(), "author_blacklist": set()},
    "options_not_extractor": {"options": {"precision": True}},
    "config_none": {"config": None},
}


@pytest.mark.parametrize("case", list(EDGE))
def test_edge_options(case: str) -> None:
    for page in ("article", "forum", "short"):
        same("extract", tp.PAGES[page], EDGE[case])


def _config(**values: str) -> configparser.ConfigParser:
    cfg = oracle.settings.use_config()
    for key, value in values.items():
        cfg.set("DEFAULT", key, value)
    return cast(configparser.ConfigParser, cfg)


@pytest.mark.parametrize(
    "values",
    [
        {"MIN_EXTRACTED_SIZE": "10", "MIN_OUTPUT_SIZE": "5"},
        {"MIN_OUTPUT_SIZE": "1000", "MIN_OUTPUT_COMM_SIZE": "1000"},
        {"MAX_TREE_SIZE": "3"},
        {"MAX_TREE_SIZE": "²"},
        {"MIN_DUPLCHECK_SIZE": "5", "MAX_REPETITIONS": "0"},
        {"MIN_EXTRACTED_SIZE": "x"},
        {"EXTENSIVE_DATE_SEARCH": "maybe"},
    ],
)
def test_config(values: dict[str, str], tmp_path: Path) -> None:
    for page in ("article", "table", "repeated"):
        same("extract", tp.PAGES[page], {"config": _config(**values)})
        same("extract", tp.PAGES[page], {"config": _config(**values), "deduplicate": True})
        same("bare_extraction", tp.PAGES[page], {"config": _config(**values)})
    path = tmp_path / "settings.cfg"
    with path.open("w", encoding="utf-8") as f:
        _config(**values).write(f)
    same("extract", tp.PAGES["article"], {"settingsfile": str(path)})


def test_inputs() -> None:
    page = tp.PAGES["article"]
    for data in (
        page.encode(),
        page.encode("utf-16"),
        page.replace("Café", "Caf\xe9").encode("latin-1", "replace"),
        b"\xef\xbb\xbf" + page.encode(),
        page.encode()[:-7] + b"\xff",
    ):
        same("extract", data, {})
        same("bare_extraction", data, {})
    tree = lxml_html.fromstring(page)
    for kwargs in ({}, {"favor_precision": True}):
        got = outcome(pt.extract, tree, **kwargs)
        assert got == outcome(oracle.extract, lxml_html.fromstring(page), **kwargs)


# --------------------------------------------------------------------------------- delegation


DELEGATION_PAGES = ("article", "table")


@pytest.mark.parametrize("case", list(tp.DELEGATED_CALLS))
def test_delegation_warns_once_and_matches(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_WARNED", set())
    fn, kwargs = tp.DELEGATED_CALLS[case]
    counts = []
    for name in DELEGATION_PAGES:
        page = tp.PAGES[name]
        if fn == "extract_batch":
            got, n = counted(pt.extract_batch, [page], **kwargs)
            want = outcome(oracle.extract, page, **kwargs)
            want = (want[0], [want[1]], *want[2:]) if want[0] == "ok" else want
        else:
            got, n = counted(getattr(pt, fn), page, **kwargs)
            want = outcome(getattr(oracle, fn), page, **kwargs)
        assert got == want, name
        counts.append(n)
    assert counts == [1, 0]  # once per process, function and reason


@pytest.mark.parametrize(
    "build",
    [
        lambda: {"deduplicate": oracle.deduplication.LRUCache(2)},
        lambda: {"options": oracle.settings.Extractor(precision=True, comments=False)},
        lambda: {"prune_xpath": ["//p[1]", "//table"]},
        lambda: {"$input": "tree"},
    ],
    ids=["trafilatura_lru", "extractor", "prune_xpath_list", "tree"],
)
def test_oracle_objects_are_delegated(
    build: Callable[[], dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pt, "_WARNED", set())
    counts = []
    for name in ("article", "repeated", "repeated"):
        kw, theirs = build(), build()
        page: Any = tp.PAGES[name]
        mine = theirs_page = page
        if kw.pop("$input", None):
            del theirs["$input"]
            mine, theirs_page = lxml_html.fromstring(page), lxml_html.fromstring(page)
        got, n = counted(pt.extract, mine, **kw)
        assert got == outcome(oracle.extract, theirs_page, **theirs), name
        counts.append(n)
    assert counts == [1, 0, 0]


def test_delegated_calls_lend_the_pyroid_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_WARNED", set())
    pages = [tp.PAGES["repeated"], tp.PAGES["article"], tp.PAGES["repeated"]] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        # native and delegated calls in turn, through one pyroid and one trafilatura cache
        ours, theirs = pt.LRUCache(64), oracle.deduplication.LRUCache(64)
        for i, page in enumerate(pages):
            kw: dict[str, Any] = {"include_links": True} if i % 2 else {}
            want = oracle.extract(page, deduplicate=theirs, **kw)
            assert pt.extract(page, deduplicate=ours, **kw) == want
            assert list(ours.cache.items()) == list(theirs.cache.items())
        # deduplicate=True: the module caches stay in step too
        reset()
        for i, page in enumerate(pages):
            kw = {"include_links": True} if i % 2 else {}
            want = oracle.extract(page, deduplicate=True, **kw)
            assert pt.extract(page, deduplicate=True, **kw) == want
    assert list(pt.LRU_TEST.cache.items()) == list(oracle.deduplication.LRU_TEST.cache.items())


@pytest.mark.parametrize(
    "html",
    [
        "<!DOCTYPE html><h1>The the the.</h1>",  # the body is the <head> itself, with a tail
        "<html><body><h2>Title &amp; more</h2>\n tail &lt;text&gt; </body></html>",
    ],
)
def test_bare_trees_keep_their_tail(html: str) -> None:
    same("bare_extraction", html, {})
    same("bare_extraction", html, {"favor_precision": True})
