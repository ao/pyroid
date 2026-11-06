# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
"""pyroid.filters.URLFilter against datatrove 0.10.1's URLFilter and tldextract 5.4.0 itself.

- Live differential tests (Hypothesis): generated URLs (schemes, userinfo, ports, IPv4 and
  IPv6 literals, scope ids, unicode dots, IDN and punycode labels including invalid punycode,
  upper and Unicode case, Public Suffix List rules with their wildcards and exceptions,
  whitespace, random text) against tldextract's ``extract_str`` (registered domain and FQDN,
  with the default, private-domain and extra-suffix tries), its ``lenient_netloc`` and
  ``looks_like_ipv6``, ``str.lower()``, and datatrove's ``URLFilter.filter()`` with its
  integrated lists and with generated small lists (blank and odd entries, thresholds).
- The drop-in step against datatrove's own, through ``run()``.
- Containers, Arrow nulls, threads, errors, a 10k-URL batch.

The reference for one URL is what datatrove's step returns for
``Document(text="", id="0", metadata={"url": url})`` (its reason, or the type name of the
exception it raises). ``PYROID_URL_EXAMPLES=n`` sets the Hypothesis examples per test.
"""

# Look-alike characters and other scripts on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import functools
import math
import os
import pickle
import random
import unicodedata
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from hypothesis import HealthCheck, given, note, settings
from hypothesis import strategies as st

import pyroid
import pyroid.filters as pf
from pyroid._pyroid import _testing
from pyroid._pyroid import filters as native

tldextract = pytest.importorskip("tldextract")

pytestmark = pytest.mark.thread_unsafe(reason="datatrove, tldextract and large shared fixtures")

_HEALTH = [HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much]
_EXAMPLES = os.environ.get("PYROID_URL_EXAMPLES")
examples = (
    settings(suppress_health_check=_HEALTH, max_examples=int(_EXAMPLES))
    if _EXAMPLES
    else settings(suppress_health_check=_HEALTH)
)
few = settings(suppress_health_check=_HEALTH, max_examples=int(_EXAMPLES or 25))

DATA = Path(os.environ.get("PYROID_PARITY_DATA", "/data"))
DEV = DATA / "fineweb" / "fineweb-dev-10k.parquet"
VERSION = unicodedata.unidata_version

# ---- tldextract ------------------------------------------------------------------------


@functools.cache
def _tld() -> Any:
    """One TLDExtract for every reference and pyroid filter: the very same suffix lists."""
    t = tldextract.TLDExtract()
    t._get_tld_extractor()
    return t


@functools.cache
def _tld_private() -> Any:
    t = tldextract.TLDExtract(include_psl_private_domains=True)
    t._get_tld_extractor()
    return t


@functools.cache
def _tld_extra() -> Any:
    # Odd extra suffixes: a root wildcard, its exception, an empty label, upper case, a
    # bracket, and an A-label next to its U-label.
    t = tldextract.TLDExtract(
        extra_suffixes=["*.example", "!keep.example", ".dot", "UPPER", "a.b]", "xn--p1ai", "*"]
    )
    t._get_tld_extractor()
    return t


def _ref_parts(tld: Any, url: str) -> tuple[str, str]:
    r = tld(url)
    return r.top_domain_under_public_suffix, r.fqdn


@functools.cache
def _rules() -> tuple[str, ...]:
    ex = _tld()._get_tld_extractor()
    return tuple(sorted(set(ex.public_tlds) | set(ex.private_tlds)))


@functools.cache
def _filter_for(which: str) -> pf.URLFilter:
    tld = {"default": _tld, "private": _tld_private, "extra": _tld_extra}[which]()
    return pf.URLFilter(use_integrated_lists=False, extra_domains=["x"], tld_extractor=tld)


# ---- URL strategies ----------------------------------------------------------------------

LABELS = [
    "com", "co", "uk", "org", "jp", "kawasaki", "city", "ck", "www", "!www", "*", "kobe",
    "br", "nom", "blogspot", "github", "io", "рф", "xn--p1ai", "XN--P1AI", "公司",
    "xn--55qx5d", "cn", "中国", "xn--fiqs8s", "ελ", "xn--qxam", "ΕΛ", "k12", "ak", "us", "sch",
    "Com", "CO", "İ", "ı", "\u212aZ", "s3", "amazonaws", "localhost", "example", "keep", "a",
    "", "xn--", "xn---bbk", "xn--bbk", "xn--zz", "xn--a-", "xn--abc-def", "xn--ls8h",
    "xn--p1aj", "xn--e1afmkfd", "пример", "ΑΣ", "dot", "UPPER", "a.b]", "b]",
]  # fmt: skip
DOTS = [".", ".", ".", "。", "．", "｡"]
SPACES = ["", "", "", " ", "\t", "\u3000", "\x85", "\x1c"]
SCHEMES = [
    "", "http://", "https://", "HTTP://", "ftp://", "//", "a+b.c-d://", "a_b://", "://",
    "http:/", "mailto:", "1http://", "é://", "http:///",
]  # fmt: skip
USERINFO = ["", "", "", "user@", "u:p@", "a@b@", "@", "x:@"]
PORTS = ["", "", "", ":80", ":", ":abc", "::"]
PATHS = [
    "", "", "/", "/a/b", "?q=1", "#frag", "/porn", "/sex-cam", "/x.html", "/Hot-Chick",
    "/2girls1cup", "?x=/y", "/日本/", "/a b",
]  # fmt: skip

_no_surrogates = st.characters(exclude_categories=["Cs"])
_label_text = st.text(_no_surrogates, max_size=6)
_ascii_label = st.from_regex(r"[a-zA-Z0-9-]{1,10}", fullmatch=True)
_puny = st.one_of(
    st.text(_no_surrogates, min_size=1, max_size=5).map(
        lambda s: "xn--" + s.encode("punycode").decode("ascii")
    ),
    st.from_regex(r"[xX][nN]--[a-zA-Z0-9-]{0,8}", fullmatch=True),
)


@st.composite
def suffixes(draw: st.DrawFn) -> str:
    """A Public Suffix List rule as a URL might spell it (IDN and wildcard rules oversampled)."""
    rules = _rules()
    special = [r for r in rules if not r.isascii() or "*" in r or "!" in r]
    out = []
    for lab in draw(st.sampled_from(rules) | st.sampled_from(special)).split("."):
        if lab == "*":
            lab = draw(st.sampled_from(["*", "x", "city", "www", "日本"]) | _ascii_label)
        elif lab.startswith("!"):
            lab = draw(st.sampled_from([lab, lab[1:]]))
        if not lab.isascii() and draw(st.booleans()):
            lab = "xn--" + lab.encode("punycode").decode("ascii")
        if draw(st.integers(0, 4)) == 0:
            lab = lab.upper()
        out.append(lab)
    return ".".join(out)


_hextet = st.from_regex(r"[0-9a-fA-F]{0,5}", fullmatch=True)
_v6_part = st.one_of(
    _hextet, st.just(""), st.sampled_from(["1.2.3.4", "256.0.0.1", "01.2.3.4", "1.2.3", "0.0.0.0"])
)
_ipv6 = st.builds(
    lambda parts, scope: ":".join(parts) + scope,
    st.lists(_v6_part, min_size=1, max_size=10),
    st.sampled_from(["", "", "%eth0", "%", "%a%b", "%é", "/64"]),
)


@st.composite
def hosts(draw: st.DrawFn) -> str:
    kind = draw(st.integers(0, 9))
    if kind == 0:
        return draw(st.sampled_from(["1.2.3.4", "256.1.1.1", "01.2.3.4", "1.2.3", "１.2.3.4"]))
    if kind == 1:
        v6 = draw(st.sampled_from(["::1", "fe80::1%eth0", "::ffff:1.2.3.4", "1:2:3:4:5:6:7:8"]))
        inner: str = draw(st.sampled_from([v6, draw(_ipv6)]))
        return "[" + inner + draw(st.sampled_from(["]", ""]))
    labels = draw(
        st.lists(st.one_of(st.sampled_from(LABELS), _label_text, _ascii_label, _puny), max_size=3)
    )
    if kind < 8:
        labels.append(draw(suffixes()))
    host = ""
    for lab in labels:
        host += (draw(st.sampled_from(DOTS)) if host else "") + lab
    return (
        draw(st.sampled_from(SPACES))
        + host
        + draw(st.sampled_from(["", "", ".", "。.", "..", " ."]))
        + draw(st.sampled_from(SPACES))
    )


@st.composite
def urls(draw: st.DrawFn) -> str:
    if draw(st.integers(0, 9)) == 0:
        return draw(st.text(_no_surrogates, max_size=30))
    return (
        draw(st.sampled_from(SCHEMES))
        + draw(st.sampled_from(USERINFO))
        + draw(hosts())
        + draw(st.sampled_from(PORTS))
        + draw(st.sampled_from(PATHS))
    )


# ---- datatrove ---------------------------------------------------------------------------


def _datatrove() -> Any:
    pytest.importorskip("datatrove")
    pytest.importorskip("ahocorasick")
    pytest.importorskip("fasteners")
    from datatrove.pipeline.filters.url_filter import URLFilter  # type: ignore[import-not-found,import-untyped,unused-ignore]

    return URLFilter


def _dt_outcome(dt: Any, url: Any) -> tuple[str, Any]:
    from datatrove.data import Document  # type: ignore[import-not-found,import-untyped,unused-ignore]

    with warnings.catch_warnings():
        # tldextract 5.4's registered_domain warns per document (hidden by default filters).
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            r = dt.filter(Document(text="", id="0", metadata={"url": url}))
        except Exception as e:
            return ("raise", type(e).__name__)
    return ("ok", None if r is True else r[1])


def _py_outcomes(f: pf.URLFilter, items: list[str]) -> list[tuple[str, Any]]:
    try:
        return [("ok", r) for r in f.reasons(items)]
    except Exception:
        out: list[tuple[str, Any]] = []
        for u in items:
            try:
                out.append(("ok", f.reasons(u)))
            except Exception as e:
                out.append(("raise", type(e).__name__))
        return out


def _build(make: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return ("ok", make())
    except Exception as e:
        return ("raise", type(e).__name__)


@functools.cache
def _integrated() -> tuple[Any, pf.URLFilter]:
    """datatrove's URLFilter with its integrated lists, and pyroid's on the same tldextract."""
    dt = _datatrove()()
    dt.tldextractor = _tld()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # first tar extraction (3.12-3.13)
        dt.download_data()
    return dt, pf.URLFilter(tld_extractor=_tld())


@functools.cache
def _blocked_samples() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Some of the integrated blocked domains and URLs."""
    lists = pf.datatrove_url_lists()
    rng = random.Random(0)
    with open(lists.domains, encoding="utf-8") as f:
        domains = rng.sample(f.read().split("\n"), 400)
    with open(lists.urls, encoding="utf-8") as f:
        blocked = rng.sample(f.read().split("\n"), 200)
    return tuple(domains), tuple(blocked)


# ---- tldextract parity -------------------------------------------------------------------


@examples
@given(st.lists(urls(), min_size=1, max_size=20), st.sampled_from(["default", "private", "extra"]))
def test_domains_match_tldextract(items: list[str], which: str) -> None:
    tld = {"default": _tld, "private": _tld_private, "extra": _tld_extra}[which]()
    rd, fqdn = _filter_for(which).domains(items)
    assert list(zip(rd, fqdn, strict=True)) == [_ref_parts(tld, u) for u in items]


@pytest.mark.parametrize(
    "url",
    [
        "http://forums.bbc.co.uk/x",
        "http://a.b.ck",
        "http://www.ck",
        "http://a.www.ck",
        "http://x.city.kawasaki.jp",
        "http://x.foo.kawasaki.jp",
        "http://*.ck",
        "http://a.!www.ck",
        "http://xn--e1afmkfd.xn--p1ai",
        "http://пример.РФ",
        "http://a.XN--P1AI/",
        "http://a.xn--p1aj",
        "http://x.blogspot.com",
        "http://blogspot.com",
        "http://a..blogspot.com",
        "http://a。b．c｡com.。",
        "http://[::1]:80/",
        "http://[fe80::1%eth0]/",
        "http://u:p@Example.COM:8080/a?b#c",
        "http://127.0.0.1",
        "http://\u212aZ.\u212aZ",
        "http:// a.com ./",
        "http://..a.com",
        "x.keep.example",
        "y.z.example",
        "a.dot",
        "a.b]",
        "",
    ],
)
@pytest.mark.parametrize("which", ["default", "private", "extra"])
def test_domains_examples(url: str, which: str) -> None:
    tld = {"default": _tld, "private": _tld_private, "extra": _tld_extra}[which]()
    rd, fqdn = _filter_for(which).domains(url)
    assert (rd, fqdn) == _ref_parts(tld, url)


@examples
@given(st.one_of(urls(), st.text(_no_surrogates)))
def test_lenient_netloc_matches(url: str) -> None:
    from tldextract.remote import lenient_netloc  # type: ignore[import-not-found,unused-ignore]

    assert native._lenient_netloc(url) == lenient_netloc(url)


@examples
@given(st.one_of(_ipv6, st.text(st.sampled_from("0123456789abcdefABCDEF:.%/gé１"), max_size=50)))
def test_looks_like_ipv6_matches(s: str) -> None:
    from tldextract.remote import looks_like_ipv6  # type: ignore[import-not-found,unused-ignore]

    assert native._looks_like_ipv6(s) == looks_like_ipv6(s)


@examples
@given(st.text(_no_surrogates, max_size=12))
def test_lower_matches(s: str) -> None:
    assert native._py_lower(s, VERSION) == s.lower()


def test_lower_examples() -> None:
    for s in ["ΑΣ", " ΑΣ ", "ΣΑ", "Σ", "aΣ.", "İ", "\u212a", "ẞ", "\u3000Σ", "AΣ\u0301b", "ǅ"]:
        assert native._py_lower(s, VERSION) == s.lower(), s


def test_deferred_lookups_ask_tldextract() -> None:
    """With an idna lacking the canonical A-label check, unknown xn-- labels go to tldextract."""
    f = _filter_for("default")
    strings, self_ids, aliases, nodes, edges, defer_xn = cast(Any, pf)._suffix_trie(_tld())
    assert not defer_xn  # the installed idna has the check
    calls: list[str] = []

    def resolver(url: str) -> tuple[str, str]:
        calls.append(url)
        return _ref_parts(_tld(), url)

    deferring = native._URLFilter(
        (strings, self_ids, aliases, nodes, edges, True),
        (["xn--abc.com", "x.xn--abd.com"], []),
        ([], []),
        ([], [], [], cast(Any, pf)._NEVER),
        VERSION,
        resolver,
    )
    items = ["http://xn--abc.com/", "http://x.xn--abd.com/", "http://a.xn--p1ai", "http://a.com"]
    assert deferring.domains(items) == f.domains(items)
    assert calls == items[:2]
    calls.clear()
    assert deferring.reasons(items) == ["domain", "subdomain", None, None]
    assert calls == items[:2]


def test_trie_is_the_extractors() -> None:
    f = _filter_for("default")
    nodes, strings, labels = cast(Any, f)._native.trie_size
    ex = _tld()._get_tld_extractor()

    def count(n: Any) -> int:
        return 1 + sum(count(c) for c in n.matches.values())

    assert nodes == count(ex.tlds_excl_private_trie)
    assert labels > strings  # the A-labels of the IDN suffixes
    assert f.tld_extractor is _tld()


# ---- datatrove parity: integrated lists --------------------------------------------------


@examples
@given(st.data())
def test_integrated_lists_match_datatrove(data: st.DataObject) -> None:
    dt, f = _integrated()
    domains, blocked = _blocked_samples()
    items = data.draw(
        st.lists(
            st.one_of(
                urls(),
                st.builds(
                    lambda s, sub, d, p: s + sub + d + p,
                    st.sampled_from(SCHEMES),
                    st.sampled_from(["", "www.", "a.b.", "WWW."]),
                    st.sampled_from(domains),
                    st.sampled_from(PATHS),
                ),
                st.sampled_from(blocked),
            ),
            min_size=1,
            max_size=20,
        ).map(lambda xs: [x for x in xs if x] or ["http://x.com"])
    )
    want = [_dt_outcome(dt, u) for u in items]
    note(repr(list(zip(items, want, strict=True))))
    assert _py_outcomes(f, items) == want


def test_integrated_lists_dev_slice() -> None:
    """The FineWeb dev slice's URLs (all 10,000) when the data is there."""
    if not DEV.exists():
        pytest.skip(f"{DEV} not found")
    pq = pytest.importorskip("pyarrow.parquet")
    dt, f = _integrated()
    items = pq.read_table(DEV, columns=["url"]).column("url").to_pylist()
    want = [_dt_outcome(dt, u)[1] for u in items]
    assert f.reasons(items) == want
    assert {"domain", "subdomain"} <= set(want)


def test_integrated_list_sizes() -> None:
    _, f = _integrated()
    assert f.num_domains == 4_558_940
    assert f.num_urls == 19_587
    lists = pf.datatrove_url_lists()
    assert all(Path(p).is_file() for p in lists)


# ---- datatrove parity: generated lists ---------------------------------------------------

ENTRIES = [
    " ", "#x", " #y", "--", "\t", "porn", "sex", "cam", "Wiki", "html", "com", "www", "blog",
    "a", "日本", "x y", "2girls1cup", "xn--p1ai", "example.com", "www.example.com",
    "bbc.co.uk", "ck", "a.b.ck", "www.ck", "http://a.com/x", " http://a.com/y ", "\x85b.com",
    "localhost", "[::1]", "Example.COM", "#", "##", "1", "co.uk",
]  # fmt: skip
_lists = st.one_of(
    st.none(),
    st.lists(st.sampled_from(ENTRIES), max_size=6),
    st.lists(st.sampled_from([*ENTRIES, ""]), max_size=4),  # "" raises IndexError
    st.sampled_from(["abc", ""]),  # a str is iterated character by character
)
THRESHOLDS = [0, 1, 2, 3, 1.5, -1, math.inf, -math.inf, math.nan, True, 10**30]


@st.composite
def list_urls(draw: st.DrawFn) -> str:
    if draw(st.booleans()):
        return draw(urls())
    words = draw(st.lists(st.sampled_from(ENTRIES), max_size=4))
    sep = draw(st.sampled_from(["/", "-", ".", "_", "", "%20"]))
    host = draw(st.sampled_from(["a.com", "x.example.com", "www.bbc.co.uk", "a.b.ck", "[::1]"]))
    return (
        draw(st.sampled_from(SCHEMES))
        + host
        + "/"
        + sep.join(words)
        + draw(st.sampled_from(["", "/"]))
    )


@few
@given(
    st.fixed_dictionaries(
        {},
        optional={
            "soft_word_threshold": st.sampled_from(THRESHOLDS),
            "extra_domains": _lists,
            "extra_urls": _lists,
            "banned_words": _lists,
            "banned_subwords": _lists,
            "soft_banned_words": _lists,
        },
    ),
    st.lists(list_urls(), min_size=1, max_size=15),
)
def test_lists_match_datatrove(kw: dict[str, Any], items: list[str]) -> None:
    items = [u for u in items if u] or ["http://a.com/"]
    kw = {**kw, "use_integrated_lists": False}
    note(repr(kw))
    built_dt = _build(lambda: _datatrove()(**kw))
    built_py = _build(lambda: pf.URLFilter(tld_extractor=_tld(), **kw))
    if built_dt[0] == "raise":
        assert built_py == built_dt
        return
    assert built_py[0] == "ok", built_py
    dt, f = built_dt[1], built_py[1]
    dt.tldextractor = _tld()
    assert _py_outcomes(f, items) == [_dt_outcome(dt, u) for u in items]


def test_empty_subword_only_raises_like_datatrove() -> None:
    dt = _datatrove()(banned_subwords=["--", " "], use_integrated_lists=False)
    f = pf.URLFilter(banned_subwords=["--", " "], use_integrated_lists=False, tld_extractor=_tld())
    assert _dt_outcome(dt, "http://a.com/") == ("raise", "AttributeError")
    with pytest.raises(AttributeError, match="Not an Aho-Corasick automaton yet") as e:
        f.reasons(["http://a.com/", "http://b.com/"])
    assert any("index 0" in n for n in getattr(e.value, "__notes__", []))
    # Earlier checks still decide first, as in datatrove.
    f2 = pf.URLFilter(
        banned_subwords=["--"],
        banned_words=["porn"],
        use_integrated_lists=False,
        tld_extractor=_tld(),
    )
    assert f2.reasons("http://a.com/porn") == "hard_blacklisted"


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_thresholds(threshold: Any) -> None:
    kw: dict[str, Any] = {"soft_banned_words": ["sex", "cam", "xxx"], "use_integrated_lists": False}
    dt = _datatrove()(soft_word_threshold=threshold, **kw)
    f = pf.URLFilter(soft_word_threshold=threshold, tld_extractor=_tld(), **kw)
    items = [
        "http://a.com/",
        "http://a.com/sex",
        "http://a.com/sex-cam",
        "http://a.com/sex/cam/xxx",
    ]
    assert _py_outcomes(f, items) == [_dt_outcome(dt, u) for u in items]


def test_threshold_type() -> None:
    with pytest.raises(TypeError, match="soft_word_threshold"):
        pf.URLFilter(
            soft_word_threshold="2",  # type: ignore[arg-type]
            use_integrated_lists=False,
            tld_extractor=_tld(),
        )
    with pytest.raises(TypeError, match="soft_word_threshold"):
        pf.URLFilter(
            soft_word_threshold=None,  # type: ignore[arg-type]
            use_integrated_lists=False,
            tld_extractor=_tld(),
        )


# ---- API: containers, nulls, threads, errors --------------------------------------------


def _small() -> pf.URLFilter:
    return pf.URLFilter(
        extra_domains=["bad.com", "www.ok2.com"],
        extra_urls=["http://ok.com/x"],
        banned_words=["porn"],
        banned_subwords=["girlsgone"],
        soft_banned_words=["sex", "cam"],
        use_integrated_lists=False,
        tld_extractor=_tld(),
    )


SAMPLE = [
    "http://bad.com/",
    "http://www.ok2.com/",
    "http://ok.com/x",
    "http://ok.com/porn",
    "http://ok.com/sex/cam",
    "http://ok.com/Girls-Gone-Wild",
    "http://ok.com/",
]
WANT = [
    "domain",
    "subdomain",
    "url",
    "hard_blacklisted",
    "soft_blacklisted",
    "blacklisted_subword",
    None,
]


def test_containers(pa: Any) -> None:
    f = _small()
    assert f.reasons(SAMPLE) == WANT
    assert f.reasons(tuple(SAMPLE)) == tuple(WANT)
    assert f.reasons(SAMPLE[0]) == "domain"
    assert f.reasons(SAMPLE[-1]) is None
    assert f.keep(SAMPLE) == [r is None for r in WANT]
    assert f.keep(tuple(SAMPLE)) == tuple(r is None for r in WANT)
    assert f.keep(SAMPLE[-1]) is True
    rd, fqdn = f.domains(SAMPLE[:2])
    assert (rd, fqdn) == (["bad.com", "ok2.com"], ["bad.com", "www.ok2.com"])
    assert f.domains(tuple(SAMPLE[:1])) == (("bad.com",), ("bad.com",))
    assert f.domains(SAMPLE[1]) == ("ok2.com", "www.ok2.com")
    for ty in (pa.string(), pa.large_string(), pa.string_view()):
        arr = pa.array([*SAMPLE, None], type=ty)
        col = f.reasons(arr)
        assert col.type == "string"
        assert col.to_pylist() == [*WANT, None]
        assert f.keep(arr).to_pylist() == [*(r is None for r in WANT), None]
        rd_col, fq_col = f.domains(arr)
        assert rd_col.type == ("large_string" if ty == pa.large_string() else "string")
        assert rd_col.to_pylist()[:2] == ["bad.com", "ok2.com"]
        assert fq_col.to_pylist()[-1] is None
    chunked = pa.chunked_array([SAMPLE[:3], [None], SAMPLE[3:]]).slice(1)
    assert f.reasons(chunked).to_pylist() == [*WANT[1:3], None, *WANT[3:]]
    named = pa.chunked_array([SAMPLE]).cast(pa.string())
    assert pa.chunked_array(f.keep(pa.table({"u": named}).column("u"))).to_pylist() == [
        r is None for r in WANT
    ]
    col = _testing.column([*SAMPLE, None], large=True, chunks=3)
    assert f.reasons(col).to_pylist() == [*WANT, None]


def test_functions_and_cache() -> None:
    cfg = pf.URLFilterConfig(
        extra_domains=["bad.com"], banned_words=["porn"], use_integrated_lists=False
    )
    assert pf.url_filter_reasons(SAMPLE[:4], cfg) == ["domain", None, None, "hard_blacklisted"]
    assert pf.url_filter(SAMPLE[:4], cfg) == [False, True, True, False]
    a = cast(Any, pf)._url_filter(cfg, {})
    assert (
        cast(Any, pf)._url_filter(cfg._replace(extra_domains=("bad.com",)), {}) is a
    )  # same parsed lists
    assert cast(Any, pf)._url_filter(cfg, {"banned_words": None}) is not a
    f = _small()
    assert pf.url_filter_reasons(SAMPLE, f) == WANT
    with pytest.raises(TypeError, match="not both"):
        pf.url_filter(SAMPLE, f, banned_words=["x"])
    with pytest.raises(TypeError):
        pf.url_filter(SAMPLE, pf.GopherQuality())  # type: ignore[call-overload]
    with pytest.raises(TypeError):
        pf.URLFilter(nope=1)  # type: ignore[call-arg]


def test_threads_equivalence() -> None:
    _, f = _integrated()
    rng = random.Random(1)
    domains, blocked = _blocked_samples()
    items = [rng.choice([*domains, *blocked, *SAMPLE]) for _ in range(3000)]
    items = [("http://" if i % 3 else "https://www.") + u for i, u in enumerate(items)]
    want = f.reasons(items, threads=1)
    for t in (None, 2, 8):
        assert f.reasons(items, threads=t) == want
        assert f.keep(items, threads=t) == [r is None for r in want]
    assert f.domains(items, threads=8) == f.domains(items, threads=1)


@pytest.mark.timeout(120)
def test_10k_urls() -> None:
    _, f = _integrated()
    if DEV.exists():
        pq = pytest.importorskip("pyarrow.parquet")
        col = pq.read_table(DEV, columns=["url"]).column("url")
        out = f.reasons(col)
        assert len(out) == 10_000
        assert out.null_count < 10_000
    else:
        items = [f"http://site{i}.example.com/page/{i}" for i in range(10_000)]
        assert f.keep(items) == [True] * 10_000


def test_errors() -> None:
    f = _small()
    with pytest.raises(ValueError, match="empty URL") as e:
        f.reasons(["http://a.com", ""])
    assert any("index 1" in n for n in getattr(e.value, "__notes__", []))
    with pytest.raises(ValueError, match="empty URL"):
        f.keep("")
    assert f.domains("") == ("", "")
    with pytest.raises(pyroid.InputError):
        f.reasons(["http://a.com", 3])  # type: ignore[list-item]
    with pytest.raises(pyroid.InputError):
        f.reasons(b"http://a.com")  # type: ignore[call-overload]
    with pytest.raises(UnicodeEncodeError):
        f.reasons(["http://a.com/\ud800"])
    with pytest.raises(IndexError):
        pf.URLFilter(extra_domains=["a.com", ""], use_integrated_lists=False, tld_extractor=_tld())
    with pytest.raises(TypeError):
        pf.URLFilter(banned_words=[3], use_integrated_lists=False, tld_extractor=_tld())  # type: ignore[list-item]
    with pytest.raises(TypeError, match="URLFilterConfig"):
        pf.URLFilter(pf.GopherQuality())  # type: ignore[arg-type]


def test_surrogate_entries_are_kept_for_the_python_path() -> None:
    f = pf.URLFilter(
        extra_domains=["a.com", "\ud800.com"], use_integrated_lists=False, tld_extractor=_tld()
    )
    assert f.num_domains == 2
    assert cast(Any, f)._python_reason("http://\ud800.com/") == "domain"
    assert cast(Any, f)._python_reason("http://x.a.com/\ud800") == "domain"
    assert cast(Any, f)._python_reason("http://b.com/\ud800") is None


# ---- the drop-in datatrove step ----------------------------------------------------------


def _pdt() -> Any:
    _datatrove()
    from pyroid.integrations import datatrove as pdt

    return pdt


def _docs(items: list[Any]) -> list[Any]:
    from datatrove.data import Document  # type: ignore[import-not-found,import-untyped,unused-ignore]

    return [Document(text=f"doc {i}", id=str(i), metadata={"url": u}) for i, u in enumerate(items)]


def _run(step: Any, items: list[Any], tmp: Path) -> tuple[list[str], dict[str, Any], list[str]]:
    pytest.importorskip("orjson")  # datatrove's JsonlWriter
    from datatrove.pipeline.writers import JsonlWriter  # type: ignore[import-not-found,import-untyped,unused-ignore]

    step.exclusion_writer = JsonlWriter(str(tmp), compression=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        kept = [d.id for d in step.run(iter(_docs(items)))]
    stats = {k: v for k, v in step.stats.stats.to_dict().items() if k != "batches"}
    excluded = [line for p in sorted(tmp.glob("*.jsonl")) for line in p.read_text().splitlines()]
    return kept, stats, excluded


def _drop_in_items() -> list[Any]:
    domains, blocked = _blocked_samples()
    items: list[Any] = [*SAMPLE, *blocked[:20], *("http://" + d + "/x" for d in domains[:40])]
    items += ["http://a.b.ck/sex/cam", "http://[::1]/porn", "http://xn--e1afmkfd.xn--p1ai/"]
    if DEV.exists():
        pq = pytest.importorskip("pyarrow.parquet")
        items += pq.read_table(DEV, columns=["url"]).slice(0, 300).column("url").to_pylist()
    return items


@pytest.mark.parametrize("batch_size", [1, 7, 1000])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"use_integrated_lists": False, "extra_domains": ["ok.com"], "banned_words": ["porn"]},
        {
            "soft_word_threshold": 1,
            "soft_banned_words": ["x", "com"],
            "extra_urls": [" http://ok.com/x "],
        },
    ],
)
def test_drop_in_matches_datatrove_run(
    kwargs: dict[str, Any], batch_size: int, tmp_path: Path
) -> None:
    pdt = _pdt()
    items = _drop_in_items()
    stock = _datatrove()(**kwargs)
    stock.tldextractor = _tld()
    ours = pdt.URLFilter(**kwargs, batch_size=batch_size)
    ours.tldextractor = _tld()
    want = _run(stock, items, tmp_path / "stock")
    got = _run(ours, items, tmp_path / "ours")
    assert got == want
    assert want[2]  # something was excluded, with its filter_reason


def test_drop_in_signature_and_pickle() -> None:
    import inspect

    pdt = _pdt()
    theirs = _datatrove()
    base = list(inspect.signature(theirs).parameters.values())
    params = list(inspect.signature(pdt.URLFilter).parameters.values())
    assert params[: len(base)] == base
    assert [p.name for p in params[len(base) :]] == ["batch_size", "threads"]
    step = pdt.URLFilter(use_integrated_lists=False, extra_domains=["bad.com", "www.ok2.com"])
    assert step.filter(_docs(["http://www.ok2.com/"])[0]) == (False, "subdomain")
    # Pickled after use (our own object): the native filter is rebuilt in the copy.
    copy = pickle.loads(pickle.dumps(step))
    assert copy._pyroid is None
    assert copy.filter(_docs(["http://bad.com/"])[0]) == (False, "domain")


@pytest.mark.parametrize("url", [None, "", 0, b"http://a.com", 5, "http://a.com/\udfff"])
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_drop_in_odd_urls(url: Any) -> None:
    pdt = _pdt()
    dt = _datatrove()(use_integrated_lists=False, banned_words=["porn"])
    dt.tldextractor = _tld()
    step = pdt.URLFilter(use_integrated_lists=False, banned_words=["porn"])
    step.tldextractor = _tld()
    want = _dt_outcome(dt, url)
    try:
        got: tuple[str, Any] = ("ok", step.filter(_docs([url])[0]))
    except Exception as e:
        got = ("raise", type(e).__name__)
    if got[0] == "ok":
        got = ("ok", None if got[1] is True else got[1][1])
    assert got == want


@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_drop_in_surrogate_urls() -> None:
    """URLs with lone surrogates (e.g. from JSON escapes) take the Python path, as datatrove."""
    pdt = _pdt()
    kw: dict[str, Any] = {
        "use_integrated_lists": False,
        "extra_domains": ["\ud800.com", "bad.com", "sub.\ud801.com"],
        "extra_urls": ["http://a.com/\udc00"],
        "banned_words": ["porn"],
        "banned_subwords": ["girlsgone"],
    }
    dt = _datatrove()(**kw)
    dt.tldextractor = _tld()
    step = pdt.URLFilter(**kw, batch_size=1000)
    step.tldextractor = _tld()
    items = [
        "http://\ud800.com/",
        "http://x.bad.com/\ud800",
        "http://sub.\ud801.com/",
        "http://a.com/\udc00",
        "http://a.com/\udc00porn",
        "http://a.com/porn\udc00",
        "http://a.com/girls\ud800gone",
        "http://a.com/",
        "http://a.com/porn",
    ]
    docs = _docs(items)
    want = [_dt_outcome(dt, u)[1] for u in items]
    got = [None if r is True else r[1] for r in step.filter_batch(docs)]
    assert got == want
    assert {"domain", "subdomain", "url", "hard_blacklisted", "blacklisted_subword", None} == set(
        want
    )


def test_drop_in_without_asserts(run_python: Any) -> None:
    """python -O: datatrove's assert is gone and an empty URL is filtered like any other."""
    _pdt()
    code = """
        import warnings; warnings.simplefilter("ignore")
        from datatrove.data import Document
        from datatrove.pipeline.filters.url_filter import URLFilter as DT
        from pyroid.integrations.datatrove import URLFilter
        kw = dict(use_integrated_lists=False, banned_words=["--"], extra_domains=["x"])
        for url in ["", "http://a.com", "a"]:
            doc = lambda: Document(text="", id="0", metadata={"url": url})
            print(repr(DT(**kw).filter(doc())) + "|" + repr(URLFilter(**kw).filter(doc())))
    """
    out = run_python(code, args=["-O"])
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0] == "(False, 'hard_blacklisted')|(False, 'hard_blacklisted')"
    assert len(lines) == 3
    for line in lines:
        a, b = line.split("|")
        assert a == b, line


def test_executor_end_to_end(tmp_path: Path) -> None:
    """URLFilter first, as in FineWeb, in datatrove's LocalPipelineExecutor: same JSONL output."""
    pdt = _pdt()
    pytest.importorskip("orjson")  # datatrove's JsonlReader and JsonlWriter
    import json

    from datatrove.executor import LocalPipelineExecutor  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.readers import JsonlReader  # type: ignore[import-not-found,import-untyped,unused-ignore]
    from datatrove.pipeline.writers import JsonlWriter  # type: ignore[import-not-found,import-untyped,unused-ignore]

    src = tmp_path / "in"
    src.mkdir()
    items = _drop_in_items()
    with open(src / "0.jsonl", "w", encoding="utf-8") as fh:
        for i, u in enumerate(items):
            fh.write(json.dumps({"text": f"doc {i}", "id": str(i), "metadata": {"url": u}}) + "\n")

    def run(cls: Any, name: str) -> dict[str, list[str]]:
        steps = [
            JsonlReader(str(src)),
            cls(exclusion_writer=JsonlWriter(str(tmp_path / name / "x"))),
            JsonlWriter(str(tmp_path / name / "out")),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            LocalPipelineExecutor(
                pipeline=steps, tasks=1, workers=1, logging_dir=str(tmp_path / name / "logs")
            ).run()
        return {d: _read_jsonl_dir(tmp_path / name / d) for d in ("out", "x")}

    stock = run(_datatrove(), "stock")
    assert run(pdt.URLFilter, "ours") == stock
    assert stock["out"]
    assert stock["x"]


def _read_jsonl_dir(d: Path) -> list[str]:
    import gzip

    out: list[str] = []
    for p in sorted(d.glob("*.jsonl*")):
        opener: Any = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            out += [line.rstrip("\n") for line in f]
    return out


def test_module_namespace() -> None:
    for name in ("URLFilter", "URLFilterConfig", "url_filter", "url_filter_reasons"):
        assert name in pf.__all__
    assert pf.TLDEXTRACT_VERSION == tldextract.__version__ == "5.4.0"
    assert repr(_small()).startswith("<pyroid.filters.URLFilter: 2 domains")


def test_fallback_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each slow path warns once: lone-surrogate URLs (drop-in), deferred xn-- lookups."""
    pdt = _pdt()
    monkeypatch.setattr(pf, "_url_fallbacks", set())
    step = pdt.URLFilter(use_integrated_lists=False, banned_words=["porn"])
    step.tldextractor = _tld()
    with pytest.warns(pyroid.FallbackWarning, match="lone surrogates"):
        assert step.filter(_docs(["http://a.com/porn\ud800"])[0]) == (False, "hard_blacklisted")
    real = cast(Any, pf)._suffix_trie
    monkeypatch.setattr(pf, "_suffix_trie", lambda tld: (*real(tld)[:-1], True))
    with pytest.warns(pyroid.FallbackWarning, match="canonical A-label"):
        f = pf.URLFilter(
            extra_domains=["xn--abc.com"], use_integrated_lists=False, tld_extractor=_tld()
        )
    assert f.reasons(["http://xn--abc.com/", "http://a.com"]) == ["domain", None]
