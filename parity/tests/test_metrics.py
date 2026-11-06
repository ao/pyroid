"""metrics: rouge-score 0.1.2 and sacrebleu 2.6.0 scores, exactly.

Pinned checks compare every result string of ``parity/metrics-ref.parquet`` (see ``pyroid_parity.metrics_cases``):
ROUGE on CNN/DM (lead-3 vs highlights, whole articles) and SummEval (16 systems, 11 references), BLEU/chrF/chrF++
per WMT21 system (141 corpora) and per segment (163k). Live checks fuzz the candidate against rouge-score and
sacrebleu themselves with Hypothesis: Unicode, empty strings, whitespace, very long inputs, multi-reference BLEU.

sacrebleu is never imported on a free-threaded build: it loads lxml, which re-enables the GIL; those live checks
skip there and say so (the pinned checks still run).
"""

from __future__ import annotations

import sys
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyroid_parity import data as D
from pyroid_parity import metrics_cases as M
from pyroid_parity.plugin import effective_limit, need, reference

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()
SUITE = "metrics"


@pytest.fixture(scope="module")
def ref(parity):
    for rel in (M.REF, M.CNNDM, M.SUMMEVAL, M.WMT21):
        need(rel)
    o = D.load_manifest()["files"].get(M.REF, {}).get("oracle", {})
    version = ", ".join(f"{d} {o.get(d)}" for d in M.ORACLES)
    reference(SUITE, "rouge-score + sacrebleu", version, 0.0)
    parity.uses(SUITE, M.REF, M.CNNDM, M.SUMMEVAL, M.WMT21)
    return M.reference_table()


def _compare(
    parity, check: str, keys: list[str], got: list[str], want: dict[str, list[str]], inputs: Any
) -> None:
    assert want["key"][: len(keys)] == keys, f"{check}: reference keys are not in corpus order"
    bad = []
    for i, (k, g, w) in enumerate(zip(keys, got, want["out"], strict=False)):
        same = g == w
        if not same:
            bad.append(k)
        parity.record(SUITE, check, same, case=k, input=inputs(i), got=g, want=w)
    assert len(got) == len(keys)
    assert not bad, f"{check}: {len(bad)}/{len(keys)} results differ: first {bad[:5]}"


def _limit(request, impl) -> int:
    return effective_limit(request.config, impl)


# ------------------------------------------------------------------------------------------------ pinned


@pytest.mark.parametrize("batch", [False, True], ids=["score", "score_batch"])
@pytest.mark.parametrize("case", list(M.ROUGE_CASES))
def test_rouge(case, batch, ref, metrics_impl, parity, request):
    data, _, how = M.ROUGE_CASES[case]
    if batch and (how == "multi" or not hasattr(metrics_impl.obj.rouge_scorer.RougeScorer, "score_batch")):
        pytest.skip("no batch entry point for this case")
    keys, targets, preds = M.rouge_pairs(data)
    n = _limit(request, metrics_impl) or len(keys)
    got = metrics_impl.obj.rouge(case, targets[:n], preds[:n], batch=batch)
    check = case + ("[batch]" if batch else "")
    _compare(parity, check, keys[:n], got, ref[case], lambda i: (targets[i], preds[i]))


@pytest.mark.parametrize("case", list(M.CORPUS_CASES))
def test_corpus(case, ref, metrics_impl, parity, request):
    items = [s for s in M.systems() if M.corpus_case_applies(case, s[0])]
    lim = _limit(request, metrics_impl)
    if lim:  # self-check / quick runs: a few systems
        items = items[: max(1, lim // 250)]
    got = [metrics_impl.obj.corpus(case, key, h, r) for key, h, r in items]
    _compare(parity, case, [s[0] for s in items], got, ref[case], lambda i: items[i][0])


@pytest.mark.parametrize("batch", [False, True], ids=["call", "batch"])
@pytest.mark.parametrize("case", list(M.SENTENCE_CASES))
def test_sentence(case, batch, ref, metrics_impl, parity, request):
    fn_name = M.SENTENCE_CASES[case][0]
    if batch and not hasattr(metrics_impl.obj.sacrebleu, f"{fn_name}_batch"):
        pytest.skip(f"no {fn_name}_batch")
    w = M.wmt21()
    keys = M.segment_keys()
    n = _limit(request, metrics_impl) or len(keys)
    got = metrics_impl.obj.sentence(case, w["hyp"][:n], w["refs"][:n], batch=batch)
    check = case + ("[batch]" if batch else "")
    _compare(parity, check, keys[:n], got, ref[case], lambda i: (w["hyp"][i], w["refs"][i]))


def test_reference_covers_every_case(ref):
    """Every case defined in metrics_cases has pinned outputs, and real data behind it."""
    assert set(ref) == {*M.ROUGE_CASES, *M.CORPUS_CASES, *M.SENTENCE_CASES}
    assert len(ref["sentence_bleu"]["key"]) == 163_000
    assert len(ref["bleu"]["key"]) == 141
    # The data exercises the interesting paths: zero-overlap BLEU, int-zero ROUGE-L results, exceptions none.
    assert any(o.startswith("0.0|") for o in ref["sentence_bleu"]["out"])
    assert not any(o.startswith("raise:") for c in ref.values() for o in c["out"])


# ------------------------------------------------------------------------------------------------- live

# Text that reaches every branch of the tokenizers: ASCII words and digits, punctuation and symbols (13a's
# classes, Unicode P*/S* for intl), CJK (zh), case pairs whose lowercase is ASCII (U+0130, U+212A), every str.split()
# whitespace, the 13a escapes and the newline handling of rougeLsum and 13a.
_WS = "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0        　"
_PIECES = st.one_of(
    st.sampled_from(["the", "cat", "sat", "running", "happily", "ponies", "caresses", "generalization", "a"]),
    st.from_regex(r"[a-zA-Z0-9]{1,12}", fullmatch=True),
    st.sampled_from(list(".,;:!?-'\"()[]{}<>/&@#$%^*+=~`|\\_")),
    st.sampled_from(
        ["&quot;", "&amp;", "&lt;", "&gt;", "<skipped>", "-\n", "3.14", "1,000", "U.S.", "e-mail"]
    ),
    st.sampled_from(["İstanbul", "Kelvin", "Stra\xdfe", "ΣΟΣ", "ＡＢ"]),
    st.characters(min_codepoint=0x4E00, max_codepoint=0x4FFF),
    st.characters(min_codepoint=0x3000, max_codepoint=0x30FF),
    st.characters(categories=["P", "S", "N", "Zs", "Mn"]),
    st.sampled_from(list(_WS)),
    st.text(st.characters(exclude_categories=["Cs"]), max_size=6),
)
texts = st.lists(_PIECES, max_size=40).map(" ".join) | st.lists(_PIECES, max_size=40).map("".join)
long_texts = st.lists(_PIECES, min_size=200, max_size=600).map(" ".join)
ROUGE_TYPES = [f"rouge{n}" for n in range(1, 10)] + ["rougeL", "rougeLsum"]


@pytest.fixture(scope="module")
def oracle_rouge():
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        pytest.skip(f"rouge-score not installed: {e}")
    return rouge_scorer


@pytest.fixture(scope="module")
def oracle_sacrebleu(parity):
    if FREE_THREADED:
        parity.skip(SUITE, "live-sacrebleu", "sacrebleu imports lxml, which re-enables the GIL on this build")
        pytest.skip("sacrebleu would re-enable the GIL on this free-threaded build (lxml)")
    try:
        import sacrebleu
    except ImportError as e:
        pytest.skip(f"sacrebleu not installed: {e}")
    return sacrebleu


def _rouge_live(metrics_impl, oracle_rouge, parity, check, kwargs, target, prediction):
    want = M.outcome(lambda: M.ser_rouge(oracle_rouge.RougeScorer(**kwargs).score(target, prediction)))
    got = M.outcome(
        lambda: M.ser_rouge(metrics_impl.obj.rouge_scorer.RougeScorer(**kwargs).score(target, prediction))
    )
    assert parity.record(SUITE, check, got == want, input=(target, prediction, kwargs), got=got, want=want), (
        got,
        want,
    )


@given(
    target=texts,
    prediction=texts,
    stem=st.booleans(),
    types=st.lists(st.sampled_from(ROUGE_TYPES), min_size=1),
)
def test_live_rouge(target, prediction, stem, types, metrics_impl, oracle_rouge, parity):
    kwargs = {"rouge_types": types, "use_stemmer": stem}
    _rouge_live(metrics_impl, oracle_rouge, parity, "live-rouge", kwargs, target, prediction)


@given(target=long_texts, prediction=long_texts)
def test_live_rouge_long(target, prediction, metrics_impl, oracle_rouge, parity):
    kwargs = {"rouge_types": ["rouge1", "rouge4", "rougeL", "rougeLsum"], "use_stemmer": True}
    _rouge_live(metrics_impl, oracle_rouge, parity, "live-rouge-long", kwargs, target, prediction)


def _sb_live(metrics_impl, oracle, parity, check, call):
    want = M.outcome(lambda: call(oracle))
    got = M.outcome(lambda: call(metrics_impl.obj.sacrebleu))
    assert parity.record(SUITE, check, got == want, input=check, got=got, want=want), (got, want)


BLEU_KW = st.fixed_dictionaries(
    {},
    optional={
        "tokenize": st.sampled_from(["13a", "intl", "char", "none", "zh"]),
        "lowercase": st.booleans(),
        "smooth_method": st.sampled_from(["none", "floor", "add-k", "exp"]),
        "effective_order": st.booleans(),
        "max_ngram_order": st.integers(1, 6),
    },
)
CHRF_KW = st.fixed_dictionaries(
    {},
    optional={
        "char_order": st.integers(1, 8),
        "word_order": st.integers(0, 3),
        "beta": st.integers(0, 4),
        "lowercase": st.booleans(),
        "whitespace": st.booleans(),
        "eps_smoothing": st.booleans(),
    },
)
# Corpora of 1-6 segments with 1-3 reference streams; references may have None holes (not the first stream).
corpora = st.integers(1, 6).flatmap(
    lambda n: st.tuples(
        st.lists(texts, min_size=n, max_size=n),
        st.integers(1, 3).flatmap(
            lambda k: st.tuples(
                st.lists(texts, min_size=n, max_size=n),
                st.lists(st.lists(st.none() | texts, min_size=n, max_size=n), min_size=k - 1, max_size=k - 1),
            ).map(lambda t: [t[0], *t[1]])
        ),
    )
)


@given(corpus=corpora, kw=BLEU_KW)
def test_live_corpus_bleu(corpus, kw, metrics_impl, oracle_sacrebleu, parity):
    hyps, refs = corpus

    def call(sb):
        m = sb.BLEU(**kw)
        return M.ser_bleu(m.corpus_score(hyps, refs), str(m.get_signature()))

    _sb_live(metrics_impl, oracle_sacrebleu, parity, "live-corpus_bleu", call)


@given(corpus=corpora, kw=CHRF_KW)
def test_live_corpus_chrf(corpus, kw, metrics_impl, oracle_sacrebleu, parity):
    hyps, refs = corpus

    def call(sb):
        m = sb.CHRF(**kw)
        return M.ser_chrf(m.corpus_score(hyps, refs), str(m.get_signature()))

    _sb_live(metrics_impl, oracle_sacrebleu, parity, "live-corpus_chrf", call)


@given(hyp=texts, refs=st.lists(texts, min_size=1, max_size=4), kw=BLEU_KW)
def test_live_sentence_bleu(hyp, refs, kw, metrics_impl, oracle_sacrebleu, parity):
    kw = dict(kw)
    if "effective_order" in kw:
        kw["use_effective_order"] = kw.pop("effective_order")
    kw.pop("max_ngram_order", None)  # not a sentence_bleu argument
    _sb_live(
        metrics_impl,
        oracle_sacrebleu,
        parity,
        "live-sentence_bleu",
        lambda sb: M.ser_bleu(sb.sentence_bleu(hyp, refs, **kw)),
    )


@given(hyp=texts | long_texts, refs=st.lists(texts | long_texts, min_size=1, max_size=3), kw=CHRF_KW)
def test_live_sentence_chrf(hyp, refs, kw, metrics_impl, oracle_sacrebleu, parity):
    kw = dict(kw)
    kw.pop("lowercase", None)  # sentence_chrf has no lowercase argument
    if "whitespace" in kw:
        kw["remove_whitespace"] = not kw.pop("whitespace")
    _sb_live(
        metrics_impl,
        oracle_sacrebleu,
        parity,
        "live-sentence_chrf",
        lambda sb: M.ser_chrf(sb.sentence_chrf(hyp, refs, **kw)),
    )


def test_surrogates_raise_unicode_encode_error(metrics_impl):
    """The documented difference: a lone surrogate has no UTF-8 form, so the Rust port raises; the
    incumbents (pure Python) accept it."""
    if metrics_impl.is_reference:
        pytest.skip("candidate-only check")
    scorer = metrics_impl.obj.rouge_scorer.RougeScorer(["rouge1"])
    with pytest.raises(UnicodeEncodeError):
        scorer.score("a \ud800 b", "a b")
    with pytest.raises(UnicodeEncodeError):
        metrics_impl.obj.sacrebleu.sentence_bleu("a \ud800 b", ["a b"])
