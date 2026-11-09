"""NeMo Curator 1.4.0 parity: ``--impl pyroid.nemo`` and its NeMo integration against NeMo's own outputs.

    pytest tests/test_nemo.py                                    # self-check: NeMo against its pinned outputs
    pytest tests/test_nemo.py --impl pyroid.nemo                 # pyroid.nemo + pyroid.integrations.nemo_curator
    PYTHONHASHSEED=0 pytest tests/test_nemo.py --impl pyroid.nemo --parity-limit 500

The pinned references (``nemo_cases.nemo_ref``: ``parity/nemo-ref.parquet`` on linux-aarch64) hold NeMo's
output for every case of ``pyroid_parity.nemo_cases`` on dev 10k + adversarial 3,232 (``pyroid_parity.nemo_record``).
They let the suite run where NeMo does not install (it needs Python < 3.14): on 3.14t/3.15t the pinned checks run
and every live check skips, saying so in the report.

Checks (suite ``nemo``; ``<case>`` is a key of ``nemo_cases.ALL_CASES``):
    <case>                 one str call per document: the score's type and ``repr`` (filters), the output's
                           digest and length (modifiers), or the exception type
    batch-<case>           list and Arrow calls: dtype (float64 / int64) and every element; with a document NeMo
                           raises on, the call must raise its exception type, and the others are checked apart
    lang_id@<model>, quality   fasttext_lang_id + lang_id_strings, fasttext_quality: NeMo's score strings / scores
    integration-classes    the integration has every class, subclassing NeMo's, with NeMo's constructor signature
    integration-<case>     the integration's class per document: score_document and keep_document, or
                           modify_document (quality: keep_document drawn in the recorded order)
    stage-<case>           live: ScoreFilter / Score / Filter / Modify built from the integration's and NeMo's
                           classes, in combinations, vs NeMo's stock stages on the same DocumentBatch (the first
                           1000 documents of dev and of adversarial, and the edge cases): columns, dtypes, index,
                           values; a batch with a document NeMo raises on must raise the same type
    hypothesis-<case>      live: NeMo vs the candidate's functions on generated text (str and batch calls)
    hypothesis-integration-<case>   live: NeMo vs the integration's class
    hypothesis-surrogates-<case>, hypothesis-surrogates-integration-<case>   the same on documents with lone
                           surrogates (a str can hold them, a reader's Arrow column cannot), apart, so that a
                           difference there does not hide one on ordinary text
    config-edges           live: options at the edges (n=0, an unknown position, lang in upper case, ...)
    oracle-version         live: the installed NeMo is 1.4.0

Hash seed: RepeatingTopNGramsFilter breaks ties by set iteration order, so its scores follow the string hash
seed. The references were recorded with PYTHONHASHSEED=0 (the tie order is then the same on CPython 3.12 to
3.15t, GIL and free-threaded). When the test process has another seed, the pinned checks of those cases run in a
subprocess with it set (``nemo_compare.isolated``); live checks compare NeMo and the candidate in one process,
so any seed works for them.

Skips: without NeMo every live check and every integration check; without ``<top>.integrations.nemo_curator``
the integration and stage checks; a check whose function the candidate lacks; ``lang="zh"`` cases where the
candidate raises NotImplementedError. Without ``--impl`` the candidate is NeMo itself (its own classes as the
"integration"), which shows the references reproduce here; that self-check reads the first 1000 documents of
each corpus unless ``--parity-limit`` says otherwise (``--parity-limit 0``: all). A limit always keeps the
adversarial set's 32 edge cases, the documents NeMo raises on.
"""

from __future__ import annotations

import importlib.metadata as md
from functools import cache
from typing import Any

import pytest
from hypothesis import given

from pyroid_parity import data as D
from pyroid_parity import nemo_cases as C
from pyroid_parity import nemo_compare as NC
from pyroid_parity import nemo_strategies as S
from pyroid_parity.plugin import effective_limit, need, reference
from pyroid_parity.report import Collector

SUITE = C.SUITE
REF = C.nemo_ref(D.platform_key())
FILTERS = list(C.FILTER_CASES)
MODIFIERS = list(C.MODIFIER_CASES)
FASTTEXT = list(C.FASTTEXT_CASES)
ALL = list(C.ALL_CASES)

try:
    import nemo_curator  # noqa: F401
except ImportError as e:
    NO_NEMO: str | None = f"nemo_curator is not installed here (NeMo Curator 1.4.0 needs Python < 3.14): {e}"
else:
    v = md.version("nemo-curator")
    NO_NEMO = None if v == C.NEMO_VERSION else f"needs nemo-curator {C.NEMO_VERSION}, found {v}"


@pytest.fixture(scope="module", autouse=True)
def _meta(parity: Collector) -> None:
    entry = D.load_manifest()["files"].get(REF, {})
    reference(SUITE, "nemo-curator", entry.get("oracle", {}).get("nemo-curator", C.NEMO_VERSION), None)
    parity.uses(
        SUITE, REF, D.CORPORA["dev"], D.CORPORA["adversarial"], *C.LID_MODELS.values(), C.QUALITY_MODEL
    )


@pytest.fixture(scope="session")
def nemo_limit(request: pytest.FixtureRequest, nemo_impl: Any) -> int:
    """``--parity-limit``; without it a candidate gets every document, the self-check 1000 per corpus."""
    return effective_limit(request.config, nemo_impl)


def live(parity: Collector, check: str) -> None:
    if NO_NEMO:
        parity.skip(SUITE, check, NO_NEMO)
        pytest.skip(NO_NEMO)


def finish(parity: Collector, tally: NC.Tally, checks: list[str]) -> None:
    """Record ``tally``; skip if none of ``checks`` compared anything; fail on any difference."""
    tally.replay(parity)
    ran = [k for k in checks if k in tally.checks]
    if not ran:
        pytest.skip("; ".join(f"{k}: {tally.skips.get(k, 'not run')}" for k in checks))
    bad = [tally.summary(k) for k in ran if tally.diff(k)]
    assert not bad, "\n".join(bad)


def pinned(impl: Any, case: str, what: list[str], limit: int) -> NC.Tally:
    need(REF)
    if C.ALL_CASES[case].hash_seeded and not C.hash_seed_ok():
        return NC.isolated(impl.module, case, what, limit)
    return NC.run(impl.obj, case, what, limit)


# ------------------------------------------------------------------------------------------- pinned


@pytest.mark.parametrize("case", ALL)
def test_pinned(case: str, nemo_impl: Any, parity: Collector, nemo_limit: int) -> None:
    """The candidate's function for ``case`` against NeMo's pinned output, per document and in batches."""
    checks = [case] if C.kind_of(case) == "fasttext" else [case, f"batch-{case}"]
    finish(parity, pinned(nemo_impl, case, ["fn"], nemo_limit), checks)


@pytest.mark.parametrize("case", ALL)
def test_integration(case: str, nemo_impl: Any, parity: Collector, nemo_limit: int) -> None:
    """The integration's class for ``case`` against NeMo's pinned output, per document."""
    check = f"integration-{case}"
    if nemo_impl.obj.integration is None:
        parity.skip(SUITE, check, nemo_impl.obj.integration_reason)
        pytest.skip(nemo_impl.obj.integration_reason)
    finish(parity, pinned(nemo_impl, case, ["integration"], nemo_limit), [check])


def test_integration_classes(nemo_impl: Any, parity: Collector) -> None:
    t = NC.Tally()
    NC.integration_classes(nemo_impl.obj, t)
    finish(parity, t, ["integration-classes"])


# ---------------------------------------------------------------------------------------------- live


def test_oracle_version(parity: Collector) -> None:
    live(parity, "oracle-version")
    got = md.version("nemo-curator")
    assert parity.record(SUITE, "oracle-version", got == C.NEMO_VERSION, got=got, want=C.NEMO_VERSION)


@pytest.mark.parametrize("case", ALL)
def test_stage(case: str, nemo_impl: Any, parity: Collector, nemo_limit: int) -> None:
    check = f"stage-{case}"
    live(parity, check)
    if nemo_impl.obj.integration is None:
        parity.skip(SUITE, check, nemo_impl.obj.integration_reason)
        pytest.skip(nemo_impl.obj.integration_reason)
    if isinstance(C.ALL_CASES[case], C.FastTextCase):
        need(C.ALL_CASES[case].model)
    finish(parity, NC.run(nemo_impl.obj, case, ["stage"], nemo_limit), [check])


@cache
def _nemo(case: str) -> Any:
    c = C.ALL_CASES[case]
    if isinstance(c, C.FastTextCase):
        obj = c.make(model_path=str(D.path(c.model)))
        obj.load_model()
        return obj
    return c.make()


def _candidate_obj(impl: Any, case: str) -> Any:
    """The integration's instance for ``case``, or None."""
    c = C.ALL_CASES[case]
    cls = getattr(impl.obj.integration, c.nemo, None) if impl.obj.integration is not None else None
    if cls is None:
        return None
    if isinstance(c, C.FastTextCase):
        obj = c.make(cls, model_path=str(D.path(c.model)))
        obj.load_model()
        return obj
    return c.make(cls)


def _integration_or_skip(impl: Any, parity: Collector, case: str, check: str) -> Any:
    obj = _candidate_obj(impl, case)
    if obj is None:
        why = impl.obj.integration_reason or f"{impl.obj.integration_name} has no {C.ALL_CASES[case].nemo}"
        parity.skip(SUITE, check, why)
    return obj


#: Generated text: ordinary documents, and documents with lone surrogates (checked apart).
TEXTS = {"text": ("hypothesis-", S.batch), "surrogates": ("hypothesis-surrogates-", S.batch_surrogates)}


@pytest.mark.parametrize("text", list(TEXTS))
@pytest.mark.parametrize("case", FILTERS)
def test_hypothesis_filter(case: str, text: str, nemo_impl: Any, parity: Collector) -> None:
    prefix, strategy = TEXTS[text]
    check, icheck = f"{prefix}{case}", f"{prefix}integration-{case}"
    live(parity, check)
    c = C.FILTER_CASES[case]
    cand = nemo_impl.obj
    fn = cand.fn(c.fn)
    if fn is None:
        parity.skip(SUITE, check, cand.missing(c.fn))
    integ = _integration_or_skip(nemo_impl, parity, case, icheck)
    if fn is None and integ is None:
        pytest.skip(cand.missing(c.fn))
    ref = _nemo(case)

    @given(texts=strategy)
    def prop(texts: list[str]) -> None:
        t = NC.Tally()
        want = [NC.score_outcome(ref.score_document, x, {}) for x in texts]
        if fn is not None:
            got = [NC.score_outcome(fn, x, c.kwargs) for x in texts]
            if any(NC._zh_unsupported(c, g) for g in got):
                parity.skip(SUITE, check, f"{cand.name}.{c.fn} raises NotImplementedError for lang='zh'")
            else:
                for x, g, w in zip(texts, got, want, strict=True):
                    t.add(check, g == w, case="str", input=x, got=g, want=w)
                NC._batch_scores(
                    fn,
                    c.kwargs,
                    "list",
                    list(texts),
                    [f"#{k}" for k in range(len(texts))],
                    texts,
                    want,
                    t,
                    check,
                )
        if integ is not None:
            for x, w in zip(texts, want, strict=True):
                try:
                    gv = integ.score_document(x)
                except Exception as e:  # noqa: BLE001
                    g: tuple[Any, ...] = ("raise", type(e).__name__)
                else:
                    g = ("ok", *C.score_cell(gv))
                t.add(icheck, g == w, case="score_document", input=x, got=g, want=w)
                if g[0] == "ok" and w[0] == "ok":
                    gk, wk = integ.keep_document(gv), ref.keep_document(ref.score_document(x))
                    t.add(icheck, gk == wk, case="keep_document", input=x, got=gk, want=wk)
        t.replay(parity)
        assert not (t.diff(check) or t.diff(icheck)), [t.summary(k) for k in (check, icheck) if t.diff(k)]

    prop()


@pytest.mark.parametrize("text", list(TEXTS))
@pytest.mark.parametrize("case", MODIFIERS)
def test_hypothesis_modifier(case: str, text: str, nemo_impl: Any, parity: Collector) -> None:
    prefix, strategy = TEXTS[text]
    check, icheck = f"{prefix}{case}", f"{prefix}integration-{case}"
    live(parity, check)
    c = C.MODIFIER_CASES[case]
    cand = nemo_impl.obj
    fn, fnb = cand.fn(c.fn), cand.fn_batch(c.fn)
    if fn is None:
        parity.skip(SUITE, check, cand.missing(c.fn))
    integ = _integration_or_skip(nemo_impl, parity, case, icheck)
    if fn is None and integ is None:
        pytest.skip(cand.missing(c.fn))
    ref = _nemo(case)

    @given(texts=strategy)
    def prop(texts: list[str]) -> None:
        t = NC.Tally()
        want = [NC.text_outcome(ref.modify_document, x, {}) for x in texts]
        if fn is not None:
            for x, w in zip(texts, want, strict=True):
                g = NC.text_outcome(fn, x, c.kwargs)
                t.add(check, g == w, case="str", input=x, got=g, want=w)
            if all(w[0] == "ok" for w in want):
                g = NC.text_outcome(lambda xs, **kw: "\0".join(fnb(xs, **kw)), list(texts), c.kwargs)
                w = NC.text_outcome(lambda xs: "\0".join(ref.modify_document(x) for x in xs), list(texts), {})
                t.add(check, g == w, case="list", input=texts, got=g, want=w)
        if integ is not None:
            for x, w in zip(texts, want, strict=True):
                g = NC.text_outcome(integ.modify_document, x, {})
                t.add(icheck, g == w, case="modify_document", input=x, got=g, want=w)
        t.replay(parity)
        assert not (t.diff(check) or t.diff(icheck)), [t.summary(k) for k in (check, icheck) if t.diff(k)]

    prop()


@pytest.mark.parametrize("case", ["lang_id@lid.176.ftz", "quality"])
def test_hypothesis_fasttext(case: str, nemo_impl: Any, parity: Collector) -> None:
    """Lone surrogates are left out: NeMo's own call fails there in fastText's argument conversion (TypeError)."""
    check, icheck = f"hypothesis-{case}", f"hypothesis-integration-{case}"
    live(parity, check)
    c = C.FASTTEXT_CASES[case]
    path = str(need(c.model))
    cand = nemo_impl.obj
    needs = ("fasttext_lang_id", "lang_id_strings") if c.nemo == "FastTextLangId" else ("fasttext_quality",)
    missing = [f for f in needs if cand.fn(f) is None]
    model = None
    if missing:
        parity.skip(SUITE, check, cand.missing(missing[0]))
    else:
        try:
            model = cand.fasttext_model(path)
        except LookupError as e:
            parity.skip(SUITE, check, str(e))
    integ = _integration_or_skip(nemo_impl, parity, case, icheck)
    if model is None and integ is None:
        pytest.skip(f"{cand.name}: no fastText functions and no integration class")
    ref = _nemo(case)

    @given(texts=S.batch)
    def prop(texts: list[str]) -> None:
        t = NC.Tally()
        want = [NC.score_outcome(ref.score_document, x, {}) for x in texts]
        if model is not None and all(w[0] == "ok" for w in want):
            if c.nemo == "FastTextLangId":
                scores, langs = cand.fn("fasttext_lang_id")(model, texts)
                got = list(cand.fn("lang_id_strings")(scores, langs))
                expect = [w[2] for w in want]
            else:
                arr = cand.fn("fasttext_quality")(model, texts, label=c.nemo_kwargs["label"])
                got = [NC._element(arr, k) for k in range(len(arr))]
                expect = [("float64", w[2]) for w in want]
            for x, g, w in zip(texts, got, expect, strict=True):
                t.add(check, g == w, case="batch", input=x, got=g, want=w)
        if integ is not None:
            for x, w in zip(texts, want, strict=True):
                g = NC.score_outcome(integ.score_document, x, {})
                t.add(icheck, g == w, case="score_document", input=x, got=g, want=w)
        t.replay(parity)
        assert not (t.diff(check) or t.diff(icheck)), [t.summary(k) for k in (check, icheck) if t.diff(k)]

    prop()


# Options at the edges: (function, keyword arguments, NeMo class, constructor arguments).
CONFIG_EDGES: list[tuple[str, dict[str, Any], str, dict[str, Any]]] = [
    ("repeating_top_ngrams", {"n": 0}, "RepeatingTopNGramsFilter", {"n": 0}),
    ("repeating_duplicate_ngrams", {"n": 0}, "RepeatingDuplicateNGramsFilter", {"n": 0}),
    ("repeating_top_ngrams", {"n": 50}, "RepeatingTopNGramsFilter", {"n": 50}),
    (
        "substring",
        {"substring": "a", "position": "middle"},
        "SubstringFilter",
        {"substring": "a", "position": "middle"},
    ),
    ("word_count", {"lang": "EN"}, "WordCountFilter", {"lang": "EN"}),
    ("word_count", {"lang": "ZH"}, "WordCountFilter", {"lang": "ZH"}),
    (
        "common_english_words",
        {"min_num_common_words": 0},
        "CommonEnglishWordsFilter",
        {"min_num_common_words": 0},
    ),
    ("remove_lines", {"patterns": []}, "LineRemover", {"patterns": []}),
    ("slice_text", {"left": "", "right": ""}, "Slicer", {"left": "", "right": ""}),
    (
        "slice_text",
        {"left": -5, "right": None, "strip": False},
        "Slicer",
        {"left": -5, "right": None, "strip": False},
    ),
    ("fasttext_label", {"label": ""}, "FastTextLabelModifier", {"label": ""}),
]
EDGE_TEXTS = ["", " ", "\n", "the cat and the dog", "a\nb\n\nc d", "x" * 10, "the the the the", "日本語 中文"]


def test_config_edges(nemo_impl: Any, parity: Collector) -> None:
    check = "config-edges"
    live(parity, check)
    cand = nemo_impl.obj
    bad = []
    for fn_name, kw, cls, nemo_kw in CONFIG_EDGES:
        fn = cand.fn(fn_name)
        if fn is None:
            continue
        modifier = fn_name in NC.MODIFIER_FN_CLASS
        try:
            ref = C.nemo_class(cls)(**nemo_kw)
            method = ref.modify_document if modifier else ref.score_document
        except Exception as e:  # noqa: BLE001  NeMo raises in the constructor: every call must raise that type
            method = e
        for text in EDGE_TEXTS:
            one = NC.text_outcome if modifier else NC.score_outcome
            w = ("raise", type(method).__name__) if isinstance(method, Exception) else one(method, text, {})
            g = one(fn, text, kw)
            if g == ("raise", "NotImplementedError") and str(kw.get("lang", "")).lower() == "zh":
                continue
            case = f"{fn_name}({kw})"
            if not parity.record(SUITE, check, g == w, case=case, input=text, got=g, want=w):
                bad.append((case, text, g, w))
    if check not in {k for _, k in parity.checks}:
        parity.skip(SUITE, check, f"{cand.name} has none of the functions")
        pytest.skip(f"{cand.name} has none of the functions")
    assert not bad, bad[:5]
