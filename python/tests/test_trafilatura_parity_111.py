"""pyroid.trafilatura with compat="1.11.0" against trafilatura 1.11.0 itself.

The trafilatura 1.11.0 stack datatrove 0.10.1 installs: lxml 6.1.3, jusText 3.0.2 (the module
skips unless those are installed, and on free-threaded builds, where lxml re-enables the GIL).
Outcomes are compared as in test_trafilatura_parity.py: the text or the exception's type and
message, ``bare_extraction``'s fields (trees as XML), the warning categories, and the module
dedup cache after each dedup sequence. Both sides get the same keyword arguments (1.11.0's:
``no_fallback``, no ``fast``); ``bare_extraction`` always gets ``as_dict`` explicitly, since
1.11.0's default is ``True`` and pyroid's ``False``.
"""

from __future__ import annotations

import configparser
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
    pytest.skip("needs trafilatura 1.11.0", allow_module_level=True)
_WANT = {"trafilatura": "1.11.0", "lxml": "6.1.3", "justext": "3.0.2"}
_FOUND = {d: importlib.metadata.version(d) for d in _WANT}
if _FOUND != _WANT:
    pytest.skip(f"needs {_WANT}, found {_FOUND}", allow_module_level=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # urllib3's DependencyWarning (SOCKS) when trafilatura loads
    import trafilatura  # type: ignore[import-untyped,unused-ignore]
    import trafilatura.deduplication  # type: ignore[import-untyped,unused-ignore]
    import trafilatura.meta  # type: ignore[import-untyped,unused-ignore]
    import trafilatura.settings  # type: ignore[import-untyped,unused-ignore]
    from lxml import etree  # type: ignore[import-untyped,unused-ignore]
    from lxml import html as lxml_html  # type: ignore[import-untyped,unused-ignore]

from hypothesis import example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import pyroid  # noqa: E402
import pyroid.trafilatura as pt  # noqa: E402

oracle: Any = trafilatura
V111 = "1.11.0"
P: Any = pt  # private helpers (_api, _lru_items, _WARNED) are not in the stubs


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
OPTION_SETS: dict[str, dict[str, Any]] = tp.OPTION_SETS_111
BARE_SETS: dict[str, dict[str, Any]] = {
    "default": {"as_dict": False},
    "precision": {"favor_precision": True, "include_comments": False, "as_dict": False},
    "recall_dedup_dict": {"favor_recall": True, "deduplicate": True, "as_dict": True},
}
RAW = Path(os.environ.get("PYROID_TRAFILATURA_PAGES", "/target/cc/dev/raw.bin"))

pytestmark = pytest.mark.thread_unsafe(reason="the module-level dedup caches")


def reset() -> None:
    pt.reset_caches()
    oracle.meta.reset_caches()


def caches_equal() -> bool:
    return bool(list(pt.LRU_TEST.cache.items()) == P._lru_items(oracle.deduplication.LRU_TEST))


def normalize(result: Any) -> Any:
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
    return counted(fn, *args, **kwargs)[0]


def counted(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], int]:
    """The normalized result or exception, the warning categories, and the FallbackWarnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got: tuple[Any, ...] = ("ok", normalize(fn(*args, **kwargs)))
        except Exception as e:
            got = ("raise", type(e).__name__, str(e))
    cats = sorted({w.category.__name__ for w in caught if w.category is not pyroid.FallbackWarning})
    return (*got, cats), sum(w.category is pyroid.FallbackWarning for w in caught)


def ours(fn: str) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(pt, fn)(*args, compat=V111, **kwargs)

    return call


def same(fn: str, html: Any, kwargs: dict[str, Any]) -> None:
    stateful = bool(kwargs.get("deduplicate"))
    if stateful:
        reset()
    got = outcome(ours(fn), html, **kwargs)
    want = outcome(getattr(oracle, fn), html, **kwargs)
    assert got == want, f"{fn}({html[:300]!r}, **{kwargs})"
    if stateful:
        assert caches_equal()


# ------------------------------------------------------------------------------------- API mirror


def _shape(fn: Callable[..., Any]) -> list[tuple[str, str, Any]]:
    return [
        (p.name, p.kind.name, id(p.default) if p.name == "config" else p.default)
        for p in inspect.signature(fn).parameters.values()
    ]


@pytest.mark.parametrize("name", ["extract", "bare_extraction"])
def test_shim_signatures_are_trafilaturas(name: str) -> None:
    api = P._api(V111, oracle.settings.DEFAULT_CONFIG)
    assert _shape(api[name]) == _shape(getattr(oracle, name))


def test_settings_match() -> None:
    real = oracle.settings.DEFAULT_CONFIG
    assert {k: dict(v) for k, v in pt.DEFAULT_CONFIGS[V111].items()} == {
        k: dict(v) for k, v in real.items()
    }
    assert pt.LRU_SIZE == oracle.settings.LRU_SIZE


def test_document_is_trafilaturas() -> None:
    doc = pt.bare_extraction(tp.PAGES["article"], as_dict=False, compat=V111)
    assert isinstance(doc, oracle.settings.Document)
    assert isinstance(doc.body, etree._Element)


@pytest.mark.parametrize("page", list(tp.PAGES))
def test_pinned_outcomes_are_trafilaturas(page: str) -> None:
    import hashlib

    got = []
    for kwargs in OPTION_SETS.values():
        oracle.meta.reset_caches()
        try:
            o: tuple[Any, ...] = ("ok", oracle.extract(tp.PAGES[page], **kwargs))
        except Exception as e:
            o = ("raise", type(e).__name__, str(e))
        got.append(hashlib.sha256(repr(o).encode("utf-8", "surrogatepass")).hexdigest()[:16])
    assert got == tp.PINNED_111[page].split()


@pytest.mark.parametrize("page", list(tp.PAGES))
@pytest.mark.parametrize("option", list(OPTION_SETS))
def test_pages(page: str, option: str) -> None:
    same("extract", tp.PAGES[page], OPTION_SETS[option])


@pytest.mark.parametrize("page", list(tp.PAGES))
@pytest.mark.parametrize("option", list(BARE_SETS))
def test_pages_bare(page: str, option: str) -> None:
    same("bare_extraction", tp.PAGES[page], BARE_SETS[option])


# ------------------------------------------------------------------------------------ Hypothesis


@pytest.mark.parametrize("option", list(OPTION_SETS))
@given(html=tp.documents())
@example(html=tp.PAGES["article"] + "<p>&amp;#" + "9" * 4301 + "</p>")
# a control character after a stray </html>: 1.11.0's readability raises on it
@example(
    html="<html><body>the<article><p>" + tp.PARAGRAPH + "</p><div>the<div></html></div>"
    "<div></div>\x08</div></article></body></html>"
)
@example(
    html="<html><body><table><tr><td colspan='x'>a</td></tr></table>" + "w " * 99 + "</body></html>"
)
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
    assert pt.extract_batch(htmls, deduplicate=True, compat=V111) == want
    assert caches_equal()


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
    """The first 300 pages through one dedup cache: datatrove 0.10.1's call to 1.11.0."""
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
        assert pt.extract_batch(pages, threads=threads, compat=V111, **kwargs) == want
    assert caches_equal()
    bare = [normalize(pt.bare_extraction(p, compat=V111)) for p in pages[:100]]
    assert bare == [normalize(oracle.bare_extraction(p)) for p in pages[:100]]  # dicts


# ----------------------------------------------------------------------------- edges and inputs

EDGE: dict[str, dict[str, Any]] = {
    "url": {"url": "https://example.org/article/1"},
    "record_id": {"record_id": "0001"},
    "tei_validation": {"tei_validation": True},
    "no_fallback": {"no_fallback": True},
    "fast_ignored": {"fast": True},
    "max_tree_size_1": {"max_tree_size": 1},
    "max_tree_size_100": {"max_tree_size": 100},
    "max_tree_size_true": {"max_tree_size": True},
    "max_tree_size_negative": {"max_tree_size": -3},
    "precision_and_recall": {"favor_precision": True, "favor_recall": True},
    "include_formatting=False": {"include_formatting": False},
    "output_format=txt": {"output_format": "txt"},
    "date_extraction_params": {"date_extraction_params": {"extensive_search": False}},
    "author_blacklist": {"author_blacklist": {"Jane Doe"}},
    "empty_url_blacklist": {"url_blacklist": set()},
    "options_not_extractor": {"options": {"precision": True}},
    "config_none": {"config": None},
    "settingsfile_ignored": {"settingsfile": "/nonexistent/settings.cfg"},
    "settingsfile_missing": {"settingsfile": "/nonexistent/settings.cfg", "config": None},
}


@pytest.mark.parametrize("case", list(EDGE))
def test_edge_options(case: str) -> None:
    for page in ("article", "forum", "short", "empty"):
        same("extract", tp.PAGES[page], EDGE[case])
        bare = {k: v for k, v in EDGE[case].items() if k not in ("record_id", "tei_validation")}
        bare.pop("settingsfile", None)
        same("bare_extraction", tp.PAGES[page], {**bare, "as_dict": False})


def test_bare_extraction_errors() -> None:
    for html in ("", tp.PAGES["fragment"], 123, None, b"\xff\xfe"):
        same("bare_extraction", html, {"as_dict": False})
        same("bare_extraction", html, {"as_dict": False, "options": {"x": 1}})
        same("bare_extraction", html, {"as_dict": False, "config": None})
    same("bare_extraction", tp.PAGES["article"], {"as_dict": False, "config": None})


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
        {"MIN_DUPLCHECK_SIZE": "5", "MAX_REPETITIONS": "0"},
        {"MIN_EXTRACTED_COMM_SIZE": "x"},
        {"MIN_EXTRACTED_SIZE": "x"},
        {"EXTENSIVE_DATE_SEARCH": "maybe"},
    ],
)
def test_config(values: dict[str, str], tmp_path: Path) -> None:
    for page in ("article", "table", "repeated", "empty"):
        same("extract", tp.PAGES[page], {"config": _config(**values)})
        same("extract", tp.PAGES[page], {"config": _config(**values), "deduplicate": True})
        same("bare_extraction", tp.PAGES[page], {"config": _config(**values), "as_dict": False})
    path = tmp_path / "settings.cfg"
    with path.open("w", encoding="utf-8") as f:
        _config(**values).write(f)
    same("extract", tp.PAGES["article"], {"settingsfile": str(path), "config": None})


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
        same("bare_extraction", data, {"as_dict": False})
    for kwargs in ({}, {"favor_precision": True}):
        got = outcome(ours("extract"), lxml_html.fromstring(page), **kwargs)
        assert got == outcome(oracle.extract, lxml_html.fromstring(page), **kwargs)


# --------------------------------------------------------------------------------- delegation


@pytest.mark.parametrize("case", list(tp.DELEGATED_CALLS_111))
def test_delegation_warns_once_and_matches(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_WARNED", set())
    fn, kwargs = tp.DELEGATED_CALLS_111[case]
    if fn == "bare_extraction":
        kwargs = {**kwargs, "as_dict": False}
    counts = []
    for name in ("article", "table"):
        page = tp.PAGES[name]
        if fn == "extract_batch":
            got, n = counted(ours("extract_batch"), [page], **kwargs)
            want = outcome(oracle.extract, page, **kwargs)
            want = (want[0], [want[1]], *want[2:]) if want[0] == "ok" else want
        else:
            got, n = counted(ours(fn), page, **kwargs)
            want = outcome(getattr(oracle, fn), page, **kwargs)
        assert got == want, name
        counts.append(n)
    assert counts == [1, 0]  # once per process, function and reason


def test_oracle_objects_are_delegated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pt, "_WARNED", set())
    counts = []
    for name in ("article", "repeated", "repeated"):
        options = oracle.settings.Extractor(precision=True, comments=False)
        got, n = counted(ours("extract"), tp.PAGES[name], options=options)
        want = outcome(oracle.extract, tp.PAGES[name], options=options)
        assert got == want, name
        counts.append(n)
    assert counts == [1, 0, 0]


def test_delegated_calls_lend_the_module_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native and delegated deduplicate=True calls in turn: one cache, as in one 1.11.0 run."""
    monkeypatch.setattr(pt, "_WARNED", set())
    pages = [tp.PAGES["repeated"], tp.PAGES["article"], tp.PAGES["repeated"]] * 2
    reset()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        for i, page in enumerate(pages):
            kw: dict[str, Any] = {"include_links": True} if i % 2 else {}
            want = oracle.extract(page, deduplicate=True, **kw)
            assert pt.extract(page, deduplicate=True, compat=V111, **kw) == want
            assert caches_equal()
    assert pt.LRU_TEST.cache


def test_defaults_without_as_dict_and_unknown_keywords() -> None:
    for page in ("article", "empty", "short"):
        same("bare_extraction", tp.PAGES[page], {})  # 1.11.0's as_dict=True
        same("extract", tp.PAGES[page], {"bogus": 1})
        same("extract", tp.PAGES[page], {"json_output": True})
