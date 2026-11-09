"""pyroid.trafilatura without the oracle: API, pinned results, dedup, batches, threads, errors.

Live parity against trafilatura 2.3.1 is in test_trafilatura_parity.py (GIL builds with
trafilatura installed); the patch shim and the datatrove step have their own modules. Everything
here runs on every build, free-threaded ones included, with or without trafilatura installed:
delegation without trafilatura is checked by hiding it (``sys.modules["trafilatura"] = None``).
"""

from __future__ import annotations

import configparser
import copy
import hashlib
import importlib.util
import inspect
import pickle
import sys
import threading
import types
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

import pyroid
import pyroid.trafilatura as pt

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()


def load_pages() -> Any:
    """python/tests/trafilatura_pages.py (shared pages and strategies), loaded by path."""
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
PAGES: dict[str, str] = tp.PAGES
OPTION_SETS: dict[str, dict[str, Any]] = tp.OPTION_SETS

ARTICLE_TEXT = (
    "River notes\nTrafilatura keeps the main text of a page and drops the navigation, the footer "
    "and the other boilerplate around it. This paragraph is long enough to count as content, and "
    "it mentions Café, naïve, Zürich and 日本語 so that the encodings stay honest.\nA second "
    "paragraph with bold, italics and a link, which keeps going for a while so the length checks "
    "pass without any doubt at all.\n- first item of a list\n- second item\nA quoted passage."
)
ARTICLE_COMMENTS = (
    "\nComments\nFirst! A reader comment that is long enough to be kept as a comment by the "
    "extractor."
)

PINNED: dict[str, str] = tp.PINNED


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:
        return ("raise", type(e).__name__, str(e))


def digest(o: tuple[Any, ...]) -> str:
    return hashlib.sha256(repr(o).encode("utf-8", "surrogatepass")).hexdigest()[:16]


@pytest.fixture
def no_trafilatura(monkeypatch: pytest.MonkeyPatch) -> None:
    """trafilatura looks uninstalled (``import trafilatura`` raises ImportError)."""
    monkeypatch.setitem(sys.modules, "trafilatura", None)
    monkeypatch.setattr(pt, "_WARNED", set())


# ---------------------------------------------------------------------------------------- API


def test_constants() -> None:
    assert pt.TRAFILATURA_VERSION == "2.3.1"
    assert pt.TRAFILATURA_VERSIONS == ("2.3.1",)
    assert dict(pt.REPRODUCED_VERSIONS["2.3.1"]) == {
        "trafilatura": "2.3.1",
        "lxml": "6.1.3",
        "libxml2": "2.14.6",
        "justext": "3.0.2",
    }
    assert pt.DELEGATED == ("extract_with_metadata",)
    assert pt.LRU_SIZE == 4096
    assert pt.LRU_TEST.maxsize == 4096
    assert isinstance(pt.LRU_TEST, pt.LRUCache)
    assert set(pt.__all__) <= set(dir(pt)) | {"Document"}
    cfg = pt.DEFAULT_CONFIG["DEFAULT"]
    assert cfg["MIN_EXTRACTED_SIZE"] == "250"
    assert cfg["MIN_OUTPUT_SIZE"] == cfg["MIN_OUTPUT_COMM_SIZE"] == "1"
    assert cfg["MIN_DUPLCHECK_SIZE"] == "100"
    assert cfg["MAX_REPETITIONS"] == "2"
    assert cfg["MAX_TREE_SIZE"] == ""
    assert cfg["MAX_FILE_SIZE"] == "20000000"


# trafilatura 2.3.1's signatures: (name, default); every parameter positional-or-keyword.
_COMMON = [("fast", False), ("no_fallback", False), ("favor_precision", False)]
_COMMON += [("favor_recall", False), ("include_comments", True)]
_FILTERS = [("include_tables", True), ("include_images", False), ("include_formatting", None)]
_FILTERS += [("include_links", False), ("deduplicate", False), ("date_extraction_params", None)]
_META = [("with_metadata", False), ("only_with_metadata", False), ("max_tree_size", None)]
_LISTS = [("url_blacklist", None), ("author_blacklist", None)]
SIGNATURES: dict[str, list[tuple[str, Any]]] = {
    "extract": [
        ("filecontent", inspect.Parameter.empty),
        ("url", None),
        ("record_id", None),
        *_COMMON,
        ("output_format", "txt"),
        ("tei_validation", False),
        ("target_language", None),
        *_FILTERS,
        *_META,
        *_LISTS,
        ("settingsfile", None),
        ("prune_xpath", None),
        ("config", "DEFAULT_CONFIG"),
        ("options", None),
    ],
    "bare_extraction": [
        ("filecontent", inspect.Parameter.empty),
        ("url", None),
        *_COMMON,
        ("output_format", "python"),
        ("target_language", None),
        *_FILTERS,
        *_META,
        *_LISTS,
        ("as_dict", False),
        ("prune_xpath", None),
        ("config", "DEFAULT_CONFIG"),
        ("options", None),
    ],
    "extract_with_metadata": [
        ("filecontent", inspect.Parameter.empty),
        ("url", None),
        ("record_id", None),
        ("fast", False),
        ("favor_precision", False),
        ("favor_recall", False),
        ("include_comments", True),
        ("output_format", "txt"),
        ("tei_validation", False),
        ("target_language", None),
        *_FILTERS,
        *_LISTS,
        ("settingsfile", None),
        ("prune_xpath", None),
        ("config", "DEFAULT_CONFIG"),
        ("options", None),
    ],
}


def shape(fn: Callable[..., Any]) -> list[tuple[str, str, Any]]:
    out = []
    for p in inspect.signature(fn).parameters.values():
        d = p.default
        if isinstance(d, configparser.ConfigParser):
            d = "DEFAULT_CONFIG"
        out.append((p.name, p.kind.name, d))
    return out


@pytest.mark.parametrize("name", list(SIGNATURES))
def test_signatures(name: str) -> None:
    want = [(n, "POSITIONAL_OR_KEYWORD", d) for n, d in SIGNATURES[name]]
    got = shape(getattr(pt, name))
    # trafilatura's parameters, then pyroid's one addition
    assert got == [*want, ("compat", "KEYWORD_ONLY", "2.3.1")]
    assert inspect.signature(getattr(pt, name)).parameters["config"].default is pt.DEFAULT_CONFIG
    # the per-version functions the patch shim installs: trafilatura's signature exactly
    assert shape(cast(Any, pt)._api("2.3.1")[name]) == want


def test_extract_batch_signature() -> None:
    got = [
        (p.name, p.kind.name, p.default)
        for p in inspect.signature(pt.extract_batch).parameters.values()
    ]
    assert got == [
        ("htmls", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("threads", "KEYWORD_ONLY", None),
        ("compat", "KEYWORD_ONLY", "2.3.1"),
        ("kwargs", "VAR_KEYWORD", inspect.Parameter.empty),
    ]
    assert str(inspect.signature(pt.reset_caches)) == "()"  # as trafilatura.meta.reset_caches


def test_compat_values() -> None:
    page = PAGES["article"]
    assert pt.extract(page, compat="2.3.1") == pt.extract(page)
    for bad in ("1.11.0", "2.3", 231, None):
        with pytest.raises(ValueError, match=r"compat must be one of '2\.3\.1'"):
            pt.extract(page, compat=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="compat must be one of"):
            pt.bare_extraction(page, compat=bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="compat must be one of"):
            pt.extract_batch([page], compat=bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------- pinned results


@pytest.mark.thread_unsafe(reason="deduplicate=True uses the module-level cache")
@pytest.mark.parametrize("page", list(PAGES))
def test_pinned_outcomes(page: str) -> None:
    want = PINNED[page].split()
    got = []
    for kwargs in OPTION_SETS.values():
        pt.reset_caches()
        got.append(digest(outcome(pt.extract, PAGES[page], **kwargs)))
    assert got == want


def test_article_text() -> None:
    assert pt.extract(PAGES["article"]) == ARTICLE_TEXT + ARTICLE_COMMENTS
    text = pt.extract(PAGES["article"], favor_precision=True, include_comments=False)
    assert text == ARTICLE_TEXT
    # positional arguments as in trafilatura's signature
    assert pt.extract(PAGES["article"], None, None, False, False, True, False, False) == text


def test_utf8_bytes_are_native() -> None:
    for page in PAGES.values():
        try:
            data = page.encode("utf-8")
        except UnicodeEncodeError:
            continue  # lone surrogates
        assert pt.extract(data) == pt.extract(page)
        bare = pt.bare_extraction(data)
        assert (bare is None) == (pt.bare_extraction(page) is None)


def test_other_input_types_give_none() -> None:
    # trafilatura's load_html raises TypeError("incompatible input type"), which it catches.
    for obj in (None, 123, 1.5, ["<p>x</p>"], bytearray(b"<html></html>"), object()):
        assert pt.extract(obj) is None
        assert pt.bare_extraction(obj) is None


def test_str_subclass() -> None:
    class S(str):
        pass

    assert pt.extract(S(PAGES["article"])) == pt.extract(PAGES["article"])


def test_long_numeric_reference_raises_like_cpython() -> None:
    page = f"<html><body><p>{tp.PARAGRAPH} &amp;#{'1' * 4400};</p></body></html>"
    msg = (
        "Exceeds the limit (4300 digits) for integer string conversion: value has 4400 digits; "
        "use sys.set_int_max_str_digits() to increase the limit"
    )
    with pytest.raises(ValueError, match="Exceeds the limit") as info:
        pt.extract(page)
    assert str(info.value) == msg
    with pytest.raises(ValueError, match="Exceeds the limit") as info:
        pt.extract_batch([PAGES["article"], page])
    assert str(info.value) == msg
    assert any("index 1" in n for n in getattr(info.value, "__notes__", []))


def test_max_tree_size_raises() -> None:
    with pytest.raises(ValueError, match=r'"max_tree_size" will be removed, use settings\.cfg'):
        pt.extract(PAGES["article"], max_tree_size=100)
    with pytest.raises(ValueError, match="max_tree_size"):
        pt.bare_extraction(PAGES["article"], max_tree_size=1)
    assert pt.extract(PAGES["article"], max_tree_size=0) == pt.extract(PAGES["article"])


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_deprecations_warn_at_the_caller() -> None:
    with pytest.warns(DeprecationWarning, match='"no_fallback" will be removed') as rec:
        fast = pt.extract(PAGES["article"], no_fallback=True)
    assert fast == pt.extract(PAGES["article"], fast=True)
    assert [Path(w.filename).name for w in rec] == [Path(__file__).name]
    with pytest.warns(DeprecationWarning, match='"as_dict" will be removed') as rec:
        d = pt.bare_extraction(PAGES["article"], as_dict=True)
    assert isinstance(d, dict)
    assert d["text"] == ARTICLE_TEXT
    assert [Path(w.filename).name for w in rec] == [Path(__file__).name]


def test_config_and_settingsfile(tmp_path: Path) -> None:
    page = PAGES["article"]
    cfg = configparser.ConfigParser()
    cfg.read_string(
        "[DEFAULT]\n" + "\n".join(f"{k} = {v}" for k, v in pt.DEFAULT_CONFIG["DEFAULT"].items())
    )
    assert pt.extract(page, config=cfg) == pt.extract(page)
    cfg.set("DEFAULT", "MIN_OUTPUT_SIZE", "100000")
    cfg.set("DEFAULT", "MIN_OUTPUT_COMM_SIZE", "100000")
    assert pt.extract(page, config=cfg) is None
    path = tmp_path / "settings.cfg"
    path.write_text("[DEFAULT]\nMIN_OUTPUT_SIZE = 100000\nMIN_OUTPUT_COMM_SIZE = 100000\n")
    assert pt.extract(page, settingsfile=str(path)) is None
    with pytest.raises(FileNotFoundError, match="The given config file does not exist"):
        pt.extract(page, settingsfile=str(tmp_path / "missing.cfg"))
    cfg.set("DEFAULT", "MIN_OUTPUT_SIZE", "lots")
    with pytest.raises(ValueError, match="invalid literal for int"):
        pt.extract(page, config=cfg)
    # config=None is trafilatura's DEFAULT_CONFIG too
    assert pt.extract(page, config=None) == pt.extract(page)  # type: ignore[arg-type]


def test_url_and_record_id_do_not_change_txt() -> None:
    page = PAGES["article"]
    assert pt.extract(page, url="https://example.org/a", record_id="0001") == pt.extract(page)
    # trafilatura's Extractor calls url.encode(): a non-str url raises AttributeError
    with pytest.raises(AttributeError, match="encode"):
        pt.extract(page, url=123)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------ bare_extraction


def test_bare_extraction_fields() -> None:
    doc = pt.bare_extraction(PAGES["article"])
    assert doc is not None
    assert type(doc).__name__ == "Document"
    assert type(doc) is pt.Document
    assert doc.text == ARTICLE_TEXT
    assert doc.text is doc.raw_text
    assert doc.comments == ARTICLE_COMMENTS[1:]
    for slot in ("title", "author", "url", "date", "fingerprint", "id", "language", "image"):
        assert getattr(doc, slot) is None
    as_dict = doc.as_dict()
    assert list(as_dict) == [
        "title", "author", "url", "hostname", "description", "sitename", "date", "categories",
        "tags", "fingerprint", "id", "license", "body", "comments", "commentsbody", "raw_text",
        "text", "language", "image", "pagetype", "filedate",
    ]  # fmt: skip
    no_comments = pt.bare_extraction(PAGES["article"], include_comments=False)
    assert no_comments is not None
    assert no_comments.comments is None
    assert pt.bare_extraction(PAGES["empty"]) is None
    assert pt.bare_extraction(PAGES["fragment"]) is None


def _xml(tree: Any) -> str:
    if isinstance(tree, str):
        return tree
    from lxml import etree  # type: ignore[import-untyped,unused-ignore]

    return cast(str, etree.tostring(tree, encoding="unicode"))


def test_bare_extraction_trees() -> None:
    doc = pt.bare_extraction(PAGES["article"])
    assert doc is not None
    body = _xml(doc.body)
    assert body.startswith('<body><head rend="h1">River notes</head><p>Trafilatura keeps')
    assert '<list rend="ul"><item>first item of a list</item>' in body
    assert _xml(doc.commentsbody).startswith("<body><head>Comments</head><p>First! A reader")
    doc = pt.bare_extraction(PAGES["article"], include_comments=False)
    assert doc is not None
    assert _xml(doc.commentsbody) == "<body/>"
    doc.body = "replaced"
    assert doc.body == "replaced"


@pytest.mark.thread_unsafe(reason="subprocess")
def test_bare_extraction_without_lxml(run_python: Callable[..., Any]) -> None:
    proc = run_python(
        """
        import sys
        sys.modules["lxml"] = sys.modules["trafilatura"] = None
        import pyroid.trafilatura as pt
        page = "<html><body><article><p>" + "Some words. " * 30 + "</p></article></body></html>"
        doc = pt.bare_extraction(page)
        assert type(doc).__mro__[1].__name__ == "_DocumentFields", type(doc).__mro__
        assert doc.body.startswith("<body><p>Some words.") and doc.body.endswith("</p></body>")
        assert doc.commentsbody == "<body/>"
        assert doc.as_dict()["body"] == doc.body
        d = pt.Document(title="t")
        assert d.title == "t" and d.body == "<body/>" and d.id is None
        assert pt.Document.from_dict({"text": "x"}).text == "x"
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_document_pickles() -> None:
    doc = pt.bare_extraction(PAGES["table"])
    assert doc is not None
    if importlib.util.find_spec("lxml") is not None:
        assert not isinstance(doc.commentsbody, str)  # parsed: it pickles as its XML
    back = pickle.loads(pickle.dumps(doc))  # our own object: nothing untrusted is unpickled
    assert type(back) is type(doc)
    assert back.text == doc.text
    assert _xml(back.body) == _xml(doc.body)


# -------------------------------------------------------------------------------------- LRUCache


class ModelLRU:
    """trafilatura 2.3.1's LRUCache, line for line (the reference for the native one)."""

    def __init__(self, maxsize: int = 128) -> None:
        from collections import OrderedDict

        self.maxsize = maxsize
        self.cache: OrderedDict[str, int] = OrderedDict()

    def get(self, key: str) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return -1

    def put(self, key: str, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

    def increment(self, key: str) -> int:
        previous = self.cache.pop(key, -1)
        self.put(key, max(previous, 0) + 1)
        return previous

    def clear(self) -> None:
        self.cache.clear()


keys = st.sampled_from(["a", "b", "c", "d", "e", "", "é", "日本"])
ops = st.lists(
    st.one_of(
        st.tuples(st.just("get"), keys),
        st.tuples(st.just("increment"), keys),
        st.tuples(st.just("put"), keys, st.integers(-5, 5)),
        st.tuples(st.just("clear")),
    ),
    max_size=60,
)


@given(maxsize=st.integers(1, 5), steps=ops)
def test_lru_cache_matches_trafilatura(maxsize: int, steps: list[tuple[Any, ...]]) -> None:
    ours, model = pt.LRUCache(maxsize), ModelLRU(maxsize)
    for op, *args in steps:
        assert getattr(ours, op)(*args) == getattr(model, op)(*args), (op, args)
        assert list(ours.cache.items()) == list(model.cache.items())
    assert ours.maxsize == maxsize


def test_lru_cache_api() -> None:
    c = pt.LRUCache()
    assert c.maxsize == 128
    assert bool(c) is True
    assert not c.cache
    assert str(inspect.signature(pt.LRUCache)) == "(maxsize=128)"
    for bad in (0, -1):
        with pytest.raises(ValueError, match="maxsize must be at least 1"):
            pt.LRUCache(bad)
    with pytest.raises(pyroid.InputError):
        pt.LRUCache("3")  # type: ignore[arg-type]
    assert pt.LRUCache(10**30).maxsize >= 2**63 - 1
    c.increment("x")
    c.put("y", 7)
    snapshot = c.cache
    assert isinstance(snapshot, types.MappingProxyType)
    assert dict(snapshot) == {"x": 1, "y": 7}
    with pytest.raises(TypeError):
        snapshot["z"] = 1  # type: ignore[index]
    assert "maxsize=128, 2 entries" in repr(c)
    with pytest.raises(pyroid.InputError, match="keys are str"):
        c.get(1)  # type: ignore[arg-type]
    with pytest.raises(UnicodeEncodeError):
        c.get("\ud800")


def test_lru_cache_pickle_and_copy() -> None:
    c = pt.LRUCache(3)
    for k in "abcd":
        c.increment(k)
    c.increment("c")
    # our own freshly built object: nothing untrusted is unpickled
    for back in (pickle.loads(pickle.dumps(c)), copy.copy(c), copy.deepcopy(c)):
        assert type(back) is pt.LRUCache
        assert back is not c
        assert back.maxsize == 3
        assert list(back.cache.items()) == [("b", 1), ("d", 1), ("c", 2)]
    back.increment("z")
    assert "z" not in c.cache  # a copy, not a view


def test_lru_cache_threads() -> None:
    c = pt.LRUCache(1000)
    keys = [f"k{i}" for i in range(50)]

    def work() -> None:
        for _ in range(200):
            for k in keys:
                c.increment(k)

    workers = [threading.Thread(target=work) for _ in range(8)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert dict(c.cache) == dict.fromkeys(keys, 1600)


# ------------------------------------------------------------------------------------ dedup


@pytest.mark.thread_unsafe(reason="the module-level dedup cache")
def test_module_cache_and_reset() -> None:
    page = PAGES["repeated"]
    pt.reset_caches()
    first = pt.extract(page, deduplicate=True)
    assert first is not None
    assert pt.LRU_TEST.cache
    results = [pt.extract(page, deduplicate=True) for _ in range(3)]
    assert results.count(None) >= 1  # duplicates are dropped
    pt.reset_caches()
    assert not pt.LRU_TEST.cache
    assert pt.extract(page, deduplicate=True) == first
    # any truthy value means the module cache, as in trafilatura
    pt.reset_caches()
    assert pt.extract(page, deduplicate=1) == first
    assert pt.LRU_TEST.cache
    pt.reset_caches()


@pytest.mark.thread_unsafe(reason="the module-level dedup cache")
def test_own_cache_is_separate() -> None:
    pt.reset_caches()
    page = PAGES["article"]
    mine = pt.LRUCache(64)
    seq = [pt.extract(page, deduplicate=mine) for _ in range(4)]
    assert not pt.LRU_TEST.cache
    assert len(mine.cache) > 0
    pt.reset_caches()
    assert seq == [pt.extract(page, deduplicate=True) for _ in range(4)]
    pt.reset_caches()


def _batch_pages() -> list[str]:
    pages = list(PAGES.values())
    return [pages[i % len(pages)] + f"<!-- {i} -->" * (i % 3) for i in range(300)]


@pytest.mark.thread_unsafe(reason="the module-level dedup cache")
@pytest.mark.parametrize("kwargs", list(OPTION_SETS.values()), ids=list(OPTION_SETS))
@pytest.mark.parametrize("threads", [None, 1, 2, 3])
def test_batch_equals_loop(kwargs: dict[str, Any], threads: int | None) -> None:
    pages = _batch_pages()
    pt.reset_caches()
    loop = [pt.extract(p, **kwargs) for p in pages]
    loop_cache = list(pt.LRU_TEST.cache.items())
    pt.reset_caches()
    assert pt.extract_batch(pages, threads=threads, **kwargs) == loop
    assert list(pt.LRU_TEST.cache.items()) == loop_cache
    assert pt.extract_batch(tuple(pages[:5]), threads=threads, **kwargs) is not None
    pt.reset_caches()


@pytest.mark.parametrize("threads", [None, 1, 4])
def test_batch_own_cache_equals_loop(threads: int | None) -> None:
    pages = _batch_pages()
    a, b = pt.LRUCache(50), pt.LRUCache(50)
    loop = [pt.extract(p, deduplicate=a, favor_precision=True) for p in pages]
    assert pt.extract_batch(pages, threads=threads, deduplicate=b, favor_precision=True) == loop
    assert list(a.cache.items()) == list(b.cache.items())


def test_batch_mixed_items_run_the_loop() -> None:
    page = PAGES["article"]
    items: list[Any] = [page, page.encode(), None, 3, page]
    want = [pt.extract(x) for x in items]
    assert pt.extract_batch(items) == want


def test_batch_errors() -> None:
    with pytest.raises(pyroid.InputError, match="list or tuple"):
        pt.extract_batch("<html></html>")
    with pytest.raises(pyroid.InputError, match="list or tuple"):
        pt.extract_batch(iter([PAGES["article"]]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="extract\\(\\) got an unexpected keyword argument 'nope'"):
        pt.extract_batch([PAGES["article"]], nope=1)
    with pytest.raises(ValueError, match="threads must be None or a positive int, got 0"):
        pt.extract_batch([PAGES["article"]], threads=0)
    with pytest.raises(pyroid.InputError, match="threads must be"):
        pt.extract_batch([PAGES["article"]], threads=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_tree_size"):
        pt.extract_batch([PAGES["article"]], max_tree_size=3)
    assert pt.extract_batch([]) == []


@pytest.mark.timeout(120)
def test_10k_pages_complete() -> None:
    page = "<html><body><article><p>{} " + "word " * 60 + "</p></article></body></html>"
    pages = [page.format(i) for i in range(10_000)]
    out = pt.extract_batch(pages)
    assert len(out) == 10_000
    assert out[123] is not None
    assert out[123].startswith("123 word")
    assert pt.extract_batch(pages[:2000], threads=1) == out[:2000]


def test_concurrent_extractions_through_one_cache() -> None:
    cache = pt.LRUCache(4096)
    pages = _batch_pages()[:60]

    def work() -> None:
        for p in pages:
            pt.extract(p, deduplicate=cache)

    workers = [threading.Thread(target=work) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    # the results depend on the interleaving; the cache stays whole
    counts = dict(cache.cache)
    assert counts
    assert len(counts) <= 4096
    assert all(v >= 1 for v in counts.values())


# --------------------------------------------------------------------- delegation without oracle

DELEGATED_CALLS: dict[str, tuple[str, dict[str, Any]]] = tp.DELEGATED_CALLS


@pytest.mark.thread_unsafe(reason="hides trafilatura in sys.modules")
@pytest.mark.parametrize("case", list(DELEGATED_CALLS))
def test_delegation_without_trafilatura_raises(case: str, no_trafilatura: None) -> None:
    fn, kwargs = DELEGATED_CALLS[case]
    arg: Any = [PAGES["article"]] if fn == "extract_batch" else PAGES["article"]
    with pytest.raises(pyroid.PyroidError, match=r"pip install trafilatura==2\.3\.1") as info:
        getattr(pt, fn)(arg, **kwargs)
    assert "not installed" in str(info.value)


@pytest.mark.thread_unsafe(reason="hides trafilatura in sys.modules")
def test_input_delegation_without_trafilatura_raises(no_trafilatura: None) -> None:
    page = PAGES["article"]
    for data in (page.encode("latin-1", "replace") + b"\xff", zlib.compress(page.encode())):
        with pytest.raises(pyroid.PyroidError, match="bytes"):
            pt.extract(data)

    class Response:
        data = page.encode()

    with pytest.raises(pyroid.PyroidError, match="a response object input"):
        pt.extract(Response())
    cfg = configparser.ConfigParser()
    cfg.read_dict({"DEFAULT": dict(pt.DEFAULT_CONFIG["DEFAULT"])})
    cfg.set("DEFAULT", "MIN_DUPLCHECK_SIZE", "-1")
    with pytest.raises(pyroid.PyroidError, match="negative size setting"):
        pt.extract(page, config=cfg)


@pytest.mark.thread_unsafe(reason="replaces trafilatura in sys.modules")
def test_delegation_to_another_version_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("trafilatura")
    fake.__version__ = "1.11.0"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trafilatura", fake)
    with pytest.raises(pyroid.PyroidError, match=r"installed trafilatura is 1\.11\.0, not 2\.3\.1"):
        pt.extract(PAGES["article"], output_format="xml")


@pytest.mark.thread_unsafe(reason="sys.set_int_max_str_digits is process-wide")
def test_other_int_digit_limits_are_delegated(no_trafilatura: None) -> None:
    old = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(0)
    try:
        with pytest.raises(pyroid.PyroidError, match="set_int_max_str_digits"):
            pt.extract(PAGES["article"])
    finally:
        sys.set_int_max_str_digits(old)


def test_extractor_like_options_are_ignored() -> None:
    # trafilatura ignores an `options` that is not an Extractor (it builds its own).
    page = PAGES["article"]
    assert pt.extract(page, options={"precision": True}) == pt.extract(page)


# ------------------------------------------------------------------------------- free-threading


@pytest.mark.thread_unsafe(reason="subprocess")
def test_import_and_extract_keep_the_gil_disabled(run_python: Callable[..., Any]) -> None:
    proc = run_python(
        """
        import sys
        sys.modules["trafilatura"] = None  # bare_extraction's Document would import it
        import pyroid.trafilatura as pt
        page = "<html><body><article><p>" + "Some words. " * 30 + "</p></article></body></html>"
        assert pt.extract(page, deduplicate=True)
        assert pt.extract_batch([page] * 300, favor_precision=True)[0]
        assert pt.extract_batch([page] * 300, deduplicate=pt.LRUCache(10))[0]
        assert pt.bare_extraction(page).text
        assert "lxml" not in sys.modules
        print(getattr(sys, "_is_gil_enabled", lambda: True)())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(not FREE_THREADED)
