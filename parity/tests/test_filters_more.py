"""More datatrove steps vs the live oracle: PIIFormatter, C4BadWordsFilter, SymbolLinesFormatter.

datatrove 0.10.1 runs here, on the pinned dev slice and adversarial set (no pinned references yet: the
oracle is cheap enough to run live). The checks are recorded under the ``filters`` suite as
``<step>/<config>/<corpus>``; each needs datatrove (the parity ``oracle`` group) and skips without it.
C4BadWordsFilter downloads datatrove's LDNOOBW lists (CC BY 4.0) on first use, as datatrove does.
"""

from __future__ import annotations

from typing import Any

import pytest

from pyroid_parity.plugin import need_corpus, reference

pytestmark = pytest.mark.filters

CORPORA = ["dev", "adversarial"]

# adversarial.parquet's FineWeb-2 slices -> the C4 bad-word list datatrove would use for them.
BADWORDS_LANG = {
    "arb_Arab": "ar",
    "cmn_Hani": "zh",
    "deu_Latn": "de",
    "fra_Latn": "fr",
    "hin_Deva": "hi",
    "jpn_Jpan": "ja",
    "rus_Cyrl": "ru",
    "spa_Latn": "es",
}

PII_CASES: dict[str, dict[str, Any]] = {
    "default": {},
    "all_ips": {"only_remove_public_ips": False},
    "custom": {"email_replacement": ("a@b.c", "d@e.f", "g@h.i"), "ip_replacement": "9.9.9.9"},
}
BADWORDS_CASES: dict[str, dict[str, Any]] = {
    "default": {},
    "keep_0.3_seed7": {"keep_fraction": 0.3, "seed": 7},
}
SYMBOL_CASES: dict[str, dict[str, Any]] = {
    "default": {},
    "fineweb2": {"symbols_to_remove": ["|"], "replace_char": "\n"},
}


def _oracle(module: str) -> Any:
    return pytest.importorskip(module, reason="datatrove (parity oracle group) is not installed")


@pytest.fixture(scope="module", autouse=True)
def _meta() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        reference("filters", "datatrove", version("datatrove"), None)
    except PackageNotFoundError:
        pass


def _record(parity: Any, check: str, corpus: Any, got: list[Any], want: list[Any]) -> None:
    assert len(got) == len(want) == len(corpus)
    bad = 0
    for doc_id, text, g, w in zip(corpus.ids, corpus.texts, got, want, strict=True):
        if not parity.record("filters", check, g == w, case=doc_id, input=text, got=g, want=w):
            bad += 1
    assert bad == 0, f"{check}: {bad}/{len(corpus)} documents differ"


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("case", list(PII_CASES))
def test_pii(case: str, corpus: str, parity: Any) -> None:
    dt = _oracle("datatrove.pipeline.formatters.pii")
    import pyroid.filters as pf

    docs = need_corpus(corpus)
    kw = PII_CASES[case]
    ref = dt.PIIFormatter(**kw)
    want = [ref.format(t) for t in docs.texts]
    r = pf.PIIReplacer(**kw)
    got = [t for i in range(0, len(docs), 1000) for t in r(docs.texts[i : i + 1000])]
    _record(parity, f"pii/{case}/{corpus}", docs, got, want)


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("case", list(BADWORDS_CASES))
def test_c4_badwords(case: str, corpus: str, parity: Any) -> None:
    c4 = _oracle("datatrove.pipeline.filters.c4_filters")
    import pyroid.filters as pf
    from datatrove.data import Document

    docs = need_corpus(corpus)
    langs = [
        BADWORDS_LANG.get(m.split(":", 1)[1], "en") if m.startswith("nonenglish:") else "en"
        for m in (docs.meta if corpus == "adversarial" else [""] * len(docs))
    ]
    kw = BADWORDS_CASES[case]
    ref = c4.C4BadWordsFilter(**kw)
    want = []
    for i, (t, lang) in enumerate(zip(docs.texts, langs, strict=True)):
        r = ref.filter(Document(text=t, id=str(i), metadata={"language": lang}))
        want.append((True, None) if r is True else (False, r[1]))
    f = pf.BadWordsFilter(**kw)
    got: list[Any] = []
    for i in range(0, len(docs), 1000):
        keep, reason = f(docs.texts[i : i + 1000], languages=langs[i : i + 1000])
        got += list(zip(keep, reason, strict=True))
    _record(parity, f"c4_badwords/{case}/{corpus}", docs, got, want)


@pytest.mark.parametrize("corpus", CORPORA)
@pytest.mark.parametrize("case", list(SYMBOL_CASES))
def test_symbol_lines(case: str, corpus: str, parity: Any) -> None:
    sym = _oracle("datatrove.pipeline.formatters.symbol_lines_remover")
    import pyroid.filters as pf

    docs = need_corpus(corpus)
    kw = SYMBOL_CASES[case]
    ref = sym.SymbolLinesFormatter(**kw)
    want = [ref.format(t) for t in docs.texts]
    got = pf.symbol_lines(docs.texts, **kw)
    _record(parity, f"symbol_lines/{case}/{corpus}", docs, got, want)


URL_CASES: dict[str, dict[str, Any]] = {
    "integrated_lists": {},
    "custom_lists": {
        "use_integrated_lists": False,
        "extra_domains": ["example.com", "blogspot.com"],
        "banned_words": ["news", "blog"],
        "banned_subwords": ["shop"],
        "soft_banned_words": ["com", "www", "org"],
        "soft_word_threshold": 2,
    },
}


@pytest.mark.parametrize("case", list(URL_CASES))
def test_url_filter(case: str, parity: Any) -> None:
    uf = _oracle("datatrove.pipeline.filters.url_filter")
    import pyroid.filters as pf
    from datatrove.data import Document

    from pyroid_parity import data as D

    docs = need_corpus("dev")
    urls = D.read_parquet_columns(D.path(D.CORPORA["dev"]), ["url"])["url"]
    kw = URL_CASES[case]
    ref = uf.URLFilter(**kw)
    want = []
    for i, u in enumerate(urls):
        r = ref.filter(Document(text="", id=str(i), metadata={"url": u}))
        want.append(None if r is True else r[1])
    got = pf.URLFilter(pf.URLFilterConfig(**kw), tld_extractor=ref.tldextractor).reasons(urls)
    _record(parity, f"url_filter/{case}/dev", docs, list(got), want)
