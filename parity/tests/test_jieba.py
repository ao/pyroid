"""jieba parity: the ``--impl`` candidate vs live jieba 0.42.1 on real Chinese text.

    pytest tests/test_jieba.py                                 # self-check (candidate = jieba)
    pytest tests/test_jieba.py --impl pyroid.jieba             # pyroid vs jieba
    pytest tests/test_jieba.py --impl pyroid.jieba --parity-limit 50

Every document is cut whole, in every mode, and compared to jieba's outcome: the list of words (or of
``(word, start, end)`` for ``tokenize``), or the exception type and message.

Corpora:
    fineweb2            fineweb2/cmn_Hani.parquet: the 250 FineWeb-2 Mandarin documents
    adversarial-zh      parity/adversarial.parquet rows of kind nonenglish:cmn_Hani (250)
    adversarial-other   the other rows of parity/adversarial.parquet (other languages, mojibake, controls,
                        near-duplicates, edge cases: jieba cuts any text)

Checks (suite ``jieba`` in parity-report.json):
    <corpus>-<mode>     one comparison per document; modes: lcut, lcut_no_hmm, cut_all, search,
                        search_no_hmm, tokenize, tokenize_no_hmm, tokenize_search, tokenize_search_no_hmm
    batch-<corpus>-<mode>  the candidate's batch call (``lcut(list_of_docs)``, ...) vs jieba per document;
                        skipped for a candidate whose ``lcut`` takes no list (jieba itself)
    userdict-<corpus>   fresh ``Tokenizer()``s with jieba's sample user dictionary loaded and words added
                        (lcut and lcut_no_hmm per document)
    freq-<corpus>       ``FREQ.get(word)`` for every distinct word of the corpus's cut, ``len(FREQ)``, ``total``

Without ``--impl`` the candidate is jieba itself, so the checks only show that jieba is deterministic here.
jieba is an sdist-only pure-Python package; where it is not installed every check is skipped with the reason.
"""

from __future__ import annotations

import importlib.metadata as md
import io
import logging
import warnings
from collections.abc import Callable
from functools import cache
from typing import Any

import pytest

from pyroid_parity import data as D
from pyroid_parity import multilingual as ML
from pyroid_parity.plugin import _impl, need, need_corpus, oracle_pin, reference
from pyroid_parity.report import Collector

SUITE = "jieba"
JIEBA_VERSION = "0.42.1"
ZH_KIND = "nonenglish:cmn_Hani"
CORPORA = ("fineweb2", "adversarial-zh", "adversarial-other")

with warnings.catch_warnings():
    # jieba 0.42.1 has invalid escape sequences (SyntaxWarning on first compile) and logs to stderr.
    warnings.simplefilter("ignore")
    try:
        import jieba
    except ImportError as e:
        jieba = None
        _NO_ORACLE = f"jieba (the oracle) is not installed here: {e}"
    else:
        jieba.setLogLevel(logging.WARNING)
        _NO_ORACLE = (
            None
            if jieba.__version__ == oracle_pin("jieba", JIEBA_VERSION)
            else f"needs jieba {oracle_pin('jieba', JIEBA_VERSION)}, found {jieba.__version__}"
        )

MODES: dict[str, Callable[[Any, Any], Any]] = {
    "lcut": lambda m, x: m.lcut(x),
    "lcut_no_hmm": lambda m, x: m.lcut(x, HMM=False),
    "cut_all": lambda m, x: m.lcut(x, cut_all=True),
    "search": lambda m, x: m.lcut_for_search(x),
    "search_no_hmm": lambda m, x: m.lcut_for_search(x, HMM=False),
    "tokenize": lambda m, x: list(m.tokenize(x)),
    "tokenize_no_hmm": lambda m, x: list(m.tokenize(x, HMM=False)),
    "tokenize_search": lambda m, x: list(m.tokenize(x, mode="search")),
    "tokenize_search_no_hmm": lambda m, x: list(m.tokenize(x, mode="search", HMM=False)),
}

# jieba 0.42.1 test/userdict.txt (its BOM and last line of spaces included), and words added on top.
SAMPLE_USERDICT = (
    "\ufeff云计算 5\n李小福 2 nr\n创新办 3 i\neasy_install 3 eng\n好用 300\n韩玉赏鉴 3 nz\n"
    "八一双鹿 3 nz\n台中\n凱特琳 nz\nEdu Trust认证 2000\n      "
).encode()
ADDED_WORDS: list[tuple[str, int | None]] = [
    ("石墨烯", None),
    ("凱特琳", None),
    ("人工智能", 100000),
    ("的确", 1),
    ("中国人", 3),
    ("我们", 2),
    ("数据集", None),
]


def outcome(fn: Callable[..., Any], *args: Any) -> tuple[Any, ...]:
    try:
        return ("ok", fn(*args))
    except Exception as e:  # noqa: BLE001  the exception is part of the outcome
        return ("raise", type(e).__name__, str(e))


@pytest.fixture(scope="session")
def jieba_impl(request: pytest.FixtureRequest) -> Any:
    return _impl(request, SUITE)


@pytest.fixture(scope="module", autouse=True)
def _oracle() -> None:
    reference(SUITE, "jieba", md.version("jieba") if jieba is not None else None, None)


def oracle(parity: Collector, check: str) -> Any:
    if _NO_ORACLE:
        parity.skip(SUITE, check, _NO_ORACLE)
        pytest.skip(_NO_ORACLE)
    return jieba


@cache
def _docs(name: str, limit: int) -> tuple[list[str], list[str]]:
    """``(ids, texts)`` of a corpus (pinned files, sha256-verified)."""
    if name == "fineweb2":
        need(ML.slice_rel("cmn_Hani"))
        c = ML.fineweb2("cmn_Hani")
        ids, texts = c.ids, c.texts
    else:
        c = need_corpus("adversarial")
        zh = name == "adversarial-zh"
        rows = [(i, t) for i, t, k in zip(c.ids, c.texts, c.meta, strict=True) if (k == ZH_KIND) == zh]
        ids, texts = [i for i, _ in rows], [t for _, t in rows]
    if limit:
        ids, texts = ids[:limit], texts[:limit]
    return list(ids), list(texts)


def _rel(name: str) -> str:
    return ML.slice_rel("cmn_Hani") if name == "fineweb2" else D.CORPORA["adversarial"]


#: Distinct words of jieba's cut_all and search output per corpus (for the freq checks).
_WORDS: dict[tuple[str, int], set[str]] = {}


def _reference(name: str, mode: str, limit: int) -> list[tuple[Any, ...]]:
    _, texts = _docs(name, limit)
    f = MODES[mode]
    want = [outcome(f, jieba, t) for t in texts]
    if mode in ("cut_all", "search"):
        _WORDS.setdefault((name, limit), set()).update(w for r in want if r[0] == "ok" for w in r[1])
    return want


def test_oracle_is_pinned_jieba(parity: Collector) -> None:
    j = oracle(parity, "oracle-version")
    want = oracle_pin("jieba", JIEBA_VERSION)
    same = parity.record(SUITE, "oracle-version", j.__version__ == want, got=j.__version__, want=want)
    assert same


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("name", CORPORA)
def test_corpus(jieba_impl: Any, parity: Collector, parity_limit: int, name: str, mode: str) -> None:
    """Per document, then (for a candidate with batch input) one batch call over the corpus."""
    check, batch_check = f"{name}-{mode}", f"batch-{name}-{mode}"
    oracle(parity, check)
    ids, texts = _docs(name, parity_limit)
    parity.uses(SUITE, _rel(name))
    want = _reference(name, mode, parity_limit)
    f = MODES[mode]
    cand = jieba_impl.obj.mod
    bad = []
    for doc_id, text, w in zip(ids, texts, want, strict=True):
        got = outcome(f, cand, text)
        if not parity.record(SUITE, check, got == w, case=doc_id, input=text, got=got, want=w):
            bad.append(doc_id)
    batch_bad = []
    if not jieba_impl.obj.has_batch:
        parity.skip(SUITE, batch_check, f"{jieba_impl.module}.lcut takes no list of documents")
    elif any(w[0] == "raise" for w in want):
        parity.skip(SUITE, batch_check, "a document raises in jieba; compared per document only")
    else:
        got_batch = f(cand, texts)
        assert len(got_batch) == len(texts)
        for doc_id, text, g, w in zip(ids, texts, got_batch, want, strict=True):
            if not parity.record(SUITE, batch_check, g == w[1], case=doc_id, input=text, got=g, want=w[1]):
                batch_bad.append(doc_id)
    assert not bad, f"{check}: {len(bad)} of {len(ids)} documents differ, e.g. {bad[:5]}"
    assert not batch_bad, f"{batch_check}: {len(batch_bad)} of {len(ids)} differ, e.g. {batch_bad[:5]}"


def _with_user_words(mod: Any) -> Any:
    t = mod.Tokenizer()
    t.load_userdict(io.BytesIO(SAMPLE_USERDICT))
    for word, freq in ADDED_WORDS:
        t.add_word(word, freq)
    t.suggest_freq("台中", True)
    return t


@pytest.mark.parametrize("name", CORPORA)
def test_userdict(jieba_impl: Any, parity: Collector, parity_limit: int, name: str) -> None:
    """User words change the routes; none has frequency 0, so Force_Split_Words stays empty."""
    check = f"userdict-{name}"
    j = oracle(parity, check)
    ids, texts = _docs(name, parity_limit)
    cand, ref = _with_user_words(jieba_impl.obj.mod), _with_user_words(j)
    same_state = (cand.total, len(cand.FREQ), sorted(cand.user_word_tag_tab.items())) == (
        ref.total,
        len(ref.FREQ),
        sorted(ref.user_word_tag_tab.items()),
    )
    assert parity.record(SUITE, check, same_state, case="state", got=cand.total, want=ref.total)
    bad = []
    for doc_id, text in zip(ids, texts, strict=True):
        for mode in ("lcut", "lcut_no_hmm"):
            got, want = outcome(MODES[mode], cand, text), outcome(MODES[mode], ref, text)
            if not parity.record(
                SUITE, check, got == want, case=f"{doc_id}:{mode}", input=text, got=got, want=want
            ):
                bad.append(f"{doc_id}:{mode}")
    assert not bad, f"{check}: {len(bad)} cuts differ, e.g. {bad[:5]}"


@pytest.mark.parametrize("name", CORPORA)
def test_freq(jieba_impl: Any, parity: Collector, parity_limit: int, name: str) -> None:
    check = f"freq-{name}"
    j = oracle(parity, check)
    cand = jieba_impl.obj.mod
    cand.dt.check_initialized()
    j.dt.check_initialized()
    got, want = (len(cand.dt.FREQ), cand.dt.total), (len(j.dt.FREQ), j.dt.total)
    assert parity.record(SUITE, check, got == want, case="len-total", got=got, want=want)
    if (name, parity_limit) not in _WORDS:
        for mode in ("cut_all", "search"):
            _reference(name, mode, parity_limit)
    bad = []
    for w in sorted(_WORDS[(name, parity_limit)]):
        g, r = cand.get_FREQ(w), j.get_FREQ(w)
        if not parity.record(SUITE, check, g == r, case=w, got=g, want=r):
            bad.append(w)
    assert not bad, f"{check}: {len(bad)} words differ, e.g. {bad[:5]}"
