"""pyroid.metrics.sacrebleu against sacrebleu 2.6.0: scores, statistics, strings, errors, logs.

The reference is the real sacrebleu. Importing it re-enables the GIL on free-threaded builds
(its ``dataset`` module imports lxml), so on those builds the reference tests are skipped and
everything else (tokenizer whitespace, threads, batch shapes, errors, the no-GIL check) runs.
"""

from __future__ import annotations

import contextlib
import functools
import glob
import importlib
import itertools
import logging
import os
import random
import sys
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.metrics.sacrebleu as sb
from pyroid._pyroid import metrics as _native

FREE_THREADED = not getattr(sys, "_is_gil_enabled", lambda: True)()

TOKENIZERS = ["13a", "intl", "zh", "char", "none"]
SMOOTH = ["none", "floor", "add-k", "exp"]

# Python's str.isspace() code points, plus two that look like whitespace but are not.
WHITESPACE = [
    "\t", "\n", "\x0b", "\x0c", "\r", "\x1c", "\x1d", "\x1e", "\x1f", " ", "\x85", "\xa0",
    "\u1680", "\u2000", "\u2005", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000",
    "\u200b", "\u180e",
]  # fmt: skip
SPECIAL = [
    "&quot;", "&amp;", "&lt;", "&gt;", "&amp;quot;", "<skipped>", "-\n", "\n", " .", ".", ",",
    "1,000", "3.5", "2.", "-", "1-2", "'", "(hi)", "e.g.", "U.S.", "...", "!?", "$5", "a,b",
]  # fmt: skip
WORDS = [
    "the", "The", "cat", "sat", "on", "mat", "a", "dog", "猫", "坐在", "垫子上", "日本語の",
    "über", "ÉCOLE", "İstanbul", "ΣΊΣΥΦΟΣ", "straße", "ﬁ", "١٢٣", "²", "€", "«", "»", "—",
    "¿qué?", "😀", "é", "x", "Ⅻ", "・", "\uff0c", "。", "〜", "ｱ", "㈱", "\u2001", "‰",
]  # fmt: skip

piece = st.one_of(
    st.sampled_from(WORDS),
    st.sampled_from(SPECIAL),
    st.sampled_from(WHITESPACE),
    st.characters(exclude_categories=["Cs"]),
)
text = st.lists(piece, max_size=24).map("".join)


@st.composite
def corpora(draw: st.DrawFn, max_segments: int = 5) -> tuple[list[str], list[list[str | None]]]:
    """Hypotheses and 1-3 reference streams, often sharing words, with ``None`` holes."""
    n = draw(st.integers(1, max_segments))
    hyps = draw(st.lists(text, min_size=n, max_size=n))
    n_streams = draw(st.integers(1, 3))
    streams: list[list[str | None]] = []
    for k in range(n_streams):
        stream: list[str | None] = []
        for i, h in enumerate(hyps):
            kind = draw(st.sampled_from(["same", "edit", "other", "none"]))
            if kind == "none" and not (k == 0 and i == 0):
                stream.append(None)
            elif kind == "same":
                stream.append(h)
            elif kind == "edit":
                words = h.split(" ")
                stream.append(" ".join(draw(st.permutations(words))[: draw(st.integers(0, 9))]))
            else:
                stream.append(draw(text))
        streams.append(stream)
    return hyps, streams


@pytest.fixture(scope="module")
def ref() -> Any:
    """The real sacrebleu (GIL builds only), or skip."""
    if FREE_THREADED:
        pytest.skip(
            "importing sacrebleu re-enables the GIL (lxml); reference parity runs on GIL builds"
        )
    return pytest.importorskip("sacrebleu")


def outcome(f: Callable[[], Any]) -> tuple[Any, ...]:
    """A comparable summary of a call: its result's strings and fields, or its exception."""
    try:
        r = f()
    except Exception as e:
        return ("raised", type(e).__name__, str(e))
    return ("ok", *summary(r))


def summary(r: Any) -> tuple[Any, ...]:
    """Everything observable on a score (or a list of them)."""
    if isinstance(r, list):
        return tuple(summary(x) for x in r)
    fields = ["score", "counts", "totals", "precisions", "bp", "sys_len", "ref_len", "ratio"]
    fields += ["prec_str", "char_order", "word_order", "beta", "name", "_verbose"]
    values = tuple(repr(getattr(r, f, None)) for f in fields)
    return (*values, str(r), repr(r), r.format(width=4), r.format(score_only=True))


def same_metric(a: Any, b: Any, *args: Any, **kwargs: Any) -> None:
    """corpus_score on a sacrebleu metric and ours: same outcome, signature and format."""
    out_a = outcome(lambda: a.corpus_score(*args, **kwargs))
    out_b = outcome(lambda: b.corpus_score(*args, **kwargs))
    assert out_b == out_a
    if out_a[0] == "ok":
        sig_a, sig_b = a.get_signature(), b.get_signature()
        assert str(sig_b) == str(sig_a)
        assert sig_b.format(short=True) == sig_a.format(short=True)
        sa, sb_ = a.corpus_score(*args, **kwargs), b.corpus_score(*args, **kwargs)
        assert sb_.format(signature=str(sig_b)) == sa.format(signature=str(sig_a))
        assert sb_.format(signature=str(sig_b), is_json=True) == sa.format(
            signature=str(sig_a), is_json=True
        )


# ---------------------------------------------------------------------------
# Reference parity (Hypothesis)
# ---------------------------------------------------------------------------

bleu_options = st.fixed_dictionaries(
    {
        "tokenize": st.sampled_from(TOKENIZERS),
        "smooth_method": st.sampled_from(SMOOTH),
        "smooth_value": st.sampled_from([None, 0.0, 0.01, 0.5, 1, 2, 3.0]),
        "lowercase": st.booleans(),
        "force": st.booleans(),
        "effective_order": st.booleans(),
        "max_ngram_order": st.sampled_from([1, 2, 3, 4, 4, 5, 6, 9]),
    }
)

chrf_options = st.fixed_dictionaries(
    {
        "char_order": st.sampled_from([0, 1, 2, 6, 6, 7, 9]),
        "word_order": st.sampled_from([0, 0, 1, 2, 3, 5]),
        "beta": st.sampled_from([0, 1, 2, 3, 0.5, 1.5]),
        "lowercase": st.booleans(),
        "whitespace": st.booleans(),
        "eps_smoothing": st.booleans(),
    }
).filter(lambda o: o["char_order"] + o["word_order"] > 0)

SLOW = settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


@SLOW
@given(corpus=corpora(), opts=bleu_options)
def test_bleu_corpus_parity(ref: Any, corpus: Any, opts: dict[str, Any]) -> None:
    hyps, refs = corpus
    same_metric(ref.BLEU(**opts), sb.BLEU(**opts), hyps, refs)
    # The reference cache gives the same results.
    a, b = ref.BLEU(**opts, references=refs), sb.BLEU(**opts, references=refs)
    same_metric(a, b, hyps, None)


@SLOW
@given(corpus=corpora(), opts=chrf_options)
def test_chrf_corpus_parity(ref: Any, corpus: Any, opts: dict[str, Any]) -> None:
    hyps, refs = corpus
    same_metric(ref.CHRF(**opts), sb.CHRF(**opts), hyps, refs)
    a, b = ref.CHRF(**opts, references=refs), sb.CHRF(**opts, references=refs)
    same_metric(a, b, hyps, None)


@SLOW
@given(
    hyp=text,
    refs=st.lists(st.one_of(text, st.none()), min_size=1, max_size=4),
    tokenize=st.sampled_from(TOKENIZERS),
    smooth=st.sampled_from(SMOOTH),
    eff=st.booleans(),
)
def test_sentence_bleu_parity(
    ref: Any, hyp: str, refs: list[str | None], tokenize: str, smooth: str, eff: bool
) -> None:
    kw: dict[str, Any] = {"tokenize": tokenize, "smooth_method": smooth, "use_effective_order": eff}
    assert outcome(lambda: sb.sentence_bleu(hyp, refs, **kw)) == outcome(
        lambda: ref.sentence_bleu(hyp, refs, **kw)
    )


@SLOW
@given(
    hyp=text,
    refs=st.lists(st.one_of(text, st.none()), min_size=1, max_size=4),
    word_order=st.sampled_from([0, 2]),
    remove_whitespace=st.booleans(),
    eps=st.booleans(),
)
def test_sentence_chrf_parity(
    ref: Any, hyp: str, refs: list[str | None], word_order: int, remove_whitespace: bool, eps: bool
) -> None:
    kw: dict[str, Any] = {
        "word_order": word_order,
        "remove_whitespace": remove_whitespace,
        "eps_smoothing": eps,
    }
    assert outcome(lambda: sb.sentence_chrf(hyp, refs, **kw)) == outcome(
        lambda: ref.sentence_chrf(hyp, refs, **kw)
    )


@SLOW
@given(corpus=corpora(max_segments=8), tokenize=st.sampled_from(TOKENIZERS), eff=st.booleans())
def test_sentence_bleu_batch_is_sentence_bleu(
    ref: Any, corpus: Any, tokenize: str, eff: bool
) -> None:
    hyps, refs = corpus
    kw: dict[str, Any] = {"tokenize": tokenize, "use_effective_order": eff}
    want = [
        outcome(functools.partial(ref.sentence_bleu, h, [r[i] for r in refs], **kw))
        for i, h in enumerate(hyps)
    ]
    if all(w[0] == "ok" for w in want):
        got = sb.sentence_bleu_batch(hyps, refs, **kw)
        assert [("ok", *summary(g)) for g in got] == want
    else:
        first = next(w for w in want if w[0] == "raised")
        assert outcome(lambda: sb.sentence_bleu_batch(hyps, refs, **kw)) == first


@SLOW
@given(corpus=corpora(max_segments=8), word_order=st.sampled_from([0, 2]))
def test_sentence_chrf_batch_is_sentence_chrf(ref: Any, corpus: Any, word_order: int) -> None:
    hyps, refs = corpus
    want = [
        outcome(
            functools.partial(ref.sentence_chrf, h, [r[i] for r in refs], word_order=word_order)
        )
        for i, h in enumerate(hyps)
    ]
    if all(w[0] == "ok" for w in want):
        got = sb.sentence_chrf_batch(hyps, refs, word_order=word_order)
        assert [("ok", *summary(g)) for g in got] == want
    else:
        first = next(w for w in want if w[0] == "raised")
        assert outcome(lambda: sb.sentence_chrf_batch(hyps, refs, word_order=word_order)) == first


@SLOW
@given(line=text, tokenize=st.sampled_from(TOKENIZERS))
def test_tokenizers_parity(ref: Any, line: str, tokenize: str) -> None:
    theirs = importlib.import_module("sacrebleu.metrics.bleu")._get_tokenizer(tokenize)()
    ours = sb.BLEU(tokenize=tokenize).tokenizer
    assert ours(line) == theirs(line)
    assert ours.signature() == theirs.signature()


def test_chrf_best_reference_selection(ref: Any) -> None:
    """Many references with close f-scores: the same one wins (first on ties)."""
    rng = random.Random(7)
    alphabet = "abcde fgh"
    for _ in range(300):
        hyp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25)))
        refs = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 25))) for _ in range(6)]
        refs += refs[:2]  # exact ties
        for opts in ({}, {"word_order": 2}, {"beta": 1.5, "eps_smoothing": True}):
            a = ref.CHRF(**opts)._extract_corpus_statistics([hyp], [[r] for r in refs])
            b = sb.CHRF(**opts)._extract_corpus_statistics([hyp], [[r] for r in refs])
            assert b == a


EDGE_CASES: list[Callable[[Any], Any]] = [
    lambda m: m.corpus_bleu([], [[]]),
    lambda m: m.corpus_bleu(["a"], []),
    lambda m: m.corpus_bleu(["a b", "c"], ["a b", "c d"]),  # flat refs: zipped as chars
    lambda m: m.corpus_bleu(["a b"], [["a b", "c d"]]),
    lambda m: m.corpus_bleu(["a b", "c"], [["a b"]]),
    lambda m: m.corpus_bleu(["a b", "c"], [["a b", None]]),
    lambda m: m.corpus_bleu(["a b", ""], [["a b", None]]),
    lambda m: m.corpus_bleu(["a"], [["a"], []]),
    lambda m: m.corpus_bleu(("a .",) * 120, [["a ."] * 120]),
    lambda m: m.corpus_bleu(["x"], "abc"),
    lambda m: m.corpus_bleu("abc", [["a", "b", "c"]]),
    lambda m: m.corpus_bleu(["a", None], [["a", "b"]]),
    lambda m: m.corpus_bleu([1], [["a"]]),
    lambda m: m.corpus_bleu(["a"], [[1]]),
    lambda m: m.corpus_bleu(["a"], 5),
    lambda m: m.corpus_bleu((x for x in ["a"]), [["a"]]),
    lambda m: m.corpus_bleu(
        ["a b c d e"], [["a b c x e"]], smooth_method="add-k", smooth_value=0.5
    ),
    lambda m: m.corpus_bleu(["a b c d e"], [["a b c x e"]], smooth_method="floor"),
    lambda m: m.corpus_bleu(["Ⅻ. ½"], [["ⅻ. ½"]], lowercase=True, tokenize="intl"),
    lambda m: m.raw_corpus_bleu(["a b c"], [["a b d"]]),
    lambda m: m.raw_corpus_bleu(["a b c"], [["a b d"]], 0.3),
    lambda m: m.sentence_bleu("a", []),
    lambda m: m.sentence_bleu(1, ["a"]),
    lambda m: m.sentence_bleu("a", "a"),
    lambda m: m.sentence_bleu("a", [1]),
    lambda m: m.sentence_bleu("a b", [None]),
    lambda m: m.sentence_bleu("", [None]),
    lambda m: m.corpus_chrf(["a b", "c"], [["a b", None]]),
    lambda m: m.corpus_chrf(["a b", "c"], [[None, "c"]]),
    lambda m: m.sentence_chrf("a b", [None]),
    lambda m: m.corpus_chrf(["a b c"], [["a b d"]], beta=1.5),
    lambda m: m.corpus_chrf(["a b c"], [["a b d"]], beta=0),
    lambda m: m.BLEU(max_ngram_order=0).corpus_score(["a b"], [["a b"]]),
    lambda m: m.BLEU(max_ngram_order=-1).corpus_score(["a b"], [["a b"]]),
    lambda m: m.BLEU(max_ngram_order=300).corpus_score(["a b"], [["a b"]]),
    lambda m: m.BLEU(max_ngram_order=2.0).corpus_score(["a b"], [["a b"]]),
    lambda m: m.CHRF(char_order=0, word_order=0).corpus_score(["a b"], [["a b"]]),
    lambda m: m.CHRF(char_order=0, word_order=0, eps_smoothing=True).corpus_score(["a"], [["a"]]),
    lambda m: m.CHRF(char_order=-1, word_order=2).corpus_score(["a b"], [["a b"]]),
    lambda m: m.CHRF(char_order=0, word_order=2).corpus_score(["a b"], [["a b"]]),
    lambda m: m.CHRF(beta=float("inf")).corpus_score(["a b"], [["a c"]]),
    lambda m: m.CHRF(beta=True).corpus_score(["a b"], [["a c"]]),
    lambda m: m.BLEU(smooth_method="foo"),
    lambda m: m.BLEU(tokenize="foo"),
    lambda m: m.BLEU().get_signature(),
    lambda m: m.BLEU().corpus_score(["a"], None),
    lambda m: m.BLEU(references=[["a"], []]).corpus_score(["a"], None),
    lambda m: m.CHRF().corpus_score(["a"], None),
    lambda m: m.BLEU(trg_lang="zh").corpus_score(["我爱北京"], [["我爱上海"]]),
]


@pytest.mark.parametrize("case", EDGE_CASES, ids=[f"case{i:02d}" for i in range(len(EDGE_CASES))])
def test_edge_cases(ref: Any, case: Callable[[Any], Any]) -> None:
    assert outcome(lambda: case(sb)) == outcome(lambda: case(ref))


def test_signatures(ref: Any) -> None:
    hyps = ["a b c", "d e"]
    refs: list[list[str | None]] = [["a b c", "d"], [None, "d e f"]]
    bleu_kws: list[dict[str, Any]] = [
        {},
        {"lowercase": True, "tokenize": "intl", "smooth_method": "floor"},
        {"smooth_method": "add-k", "smooth_value": 2, "effective_order": True},
        {"smooth_method": "floor", "smooth_value": 0.25, "tokenize": "char"},
    ]
    for kw in bleu_kws:
        same_metric(ref.BLEU(**kw), sb.BLEU(**kw), hyps, refs)
    chrf_kws: list[dict[str, Any]] = [
        {},
        {"word_order": 2},
        {"lowercase": True, "whitespace": True, "eps_smoothing": True},
    ]
    for kw in chrf_kws:
        same_metric(ref.CHRF(**kw), sb.CHRF(**kw), hyps, refs)
    m = sb.BLEU()
    m.corpus_score(hyps, [refs[0]])
    assert str(m.get_signature()) == "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0"


@pytest.mark.thread_unsafe
def test_logged_warnings(ref: Any, caplog: pytest.LogCaptureFixture) -> None:
    def logs(f: Callable[[], Any]) -> list[tuple[str, int, str]]:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="sacrebleu"), contextlib.suppress(Exception):
            f()
        return [(r.name, r.levelno, r.getMessage()) for r in caplog.records]

    hyps = ["a b ."] * 99 + ["c ."]
    cases: list[Callable[[Any], Any]] = [
        lambda m: m.corpus_bleu(hyps, [hyps]),
        lambda m: m.corpus_bleu(hyps[:99], [hyps]),
        lambda m: m.corpus_bleu(hyps, [hyps], force=True),
        lambda m: m.corpus_bleu(hyps, [hyps], lowercase=True),
        lambda m: m.BLEU(references=[hyps]).corpus_score(hyps, None),
        lambda m: m.BLEU().sentence_score("a", ["a"]),
        lambda m: m.BLEU(effective_order=True).sentence_score("a", ["a"]),
        lambda m: m.sentence_bleu("a", ["a"], use_effective_order=False),
        lambda m: m.BLEU(tokenize="13a", trg_lang="zh"),
        lambda m: m.BLEU(tokenize="13a", trg_lang="ja"),
        lambda m: m.BLEU(tokenize="none", trg_lang="ko"),
        lambda m: m.BLEU(tokenize="zh", trg_lang="zh"),
        lambda m: m.corpus_chrf(hyps, [hyps]),
    ]
    for case in cases:
        theirs = logs(functools.partial(case, ref))
        assert logs(functools.partial(case, sb)) == theirs
    assert len(logs(lambda: sb.corpus_bleu(hyps, [hyps]))) == 3
    # The batch logs the effective_order recommendation once.
    batch = logs(
        lambda: sb.sentence_bleu_batch(["a", "b"], [["a", "b"]], use_effective_order=False)
    )
    assert batch == logs(lambda: ref.sentence_bleu("a", ["a"], use_effective_order=False))


def test_custom_python_tokenizer(ref: Any) -> None:
    """A tokenizer object that is not one of pyroid's runs in Python (as sacrebleu would)."""
    hyps, refs = ["Hello, world .", "It's 3.5 km."] * 60, [["Hello world!", "It is 3.5km."] * 60]
    ours = sb.BLEU()
    tok13a = importlib.import_module("sacrebleu.tokenizers.tokenizer_13a").Tokenizer13a()
    ours.__dict__["tokenizer"] = tok13a
    assert outcome(lambda: ours.corpus_score(hyps, refs)) == outcome(
        lambda: ref.corpus_bleu(hyps, refs)
    )


def test_tokenizer_tables_every_code_point(ref: Any) -> None:
    """intl's \\p{P}/\\p{S}/\\p{N} and zh's _is_chinese_char on every code point."""
    intl = importlib.import_module(
        "sacrebleu.tokenizers.tokenizer_intl"
    ).TokenizerV14International()
    zh = importlib.import_module("sacrebleu.tokenizers.tokenizer_zh").TokenizerZh()
    cps = [c for c in range(0x110000) if not 0xD800 <= c < 0xE000]
    for i in range(0, len(cps), 2048):
        chars = [chr(c) for c in cps[i : i + 2048]]
        # x?x, 1?1 and 1.? tell punctuation, symbols and numbers apart.
        line = " ".join(f"x{c}x 1{c}1 1.{c}" for c in chars)
        assert _native.sb_tokenize("intl", line) == intl(line)
        line = " ".join(f"a{c}b" for c in chars)
        assert _native.sb_tokenize("zh", line) == zh(line)
    assert _native.sb_table_versions() == ("2.6.0", "2026.9.29", "18.0.0")


@pytest.mark.thread_unsafe
def test_bootstrap_is_delegated(ref: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("numpy")
    monkeypatch.setattr(sb, "_WARNED", set())
    hyps, refs = ["a b c d", "e f g", "h i"] * 10, [["a b c x", "e f", "h i j"] * 10]
    ours = sb.BLEU()
    with pytest.warns(pyroid.FallbackWarning, match="n_bootstrap > 1"):
        got = ours.corpus_score(hyps, refs, n_bootstrap=50)
    theirs = ref.BLEU()
    want = theirs.corpus_score(hyps, refs, n_bootstrap=50)
    assert str(got) == str(want)
    assert str(ours.get_signature()) == str(theirs.get_signature())
    c_ours, c_theirs = sb.CHRF(word_order=2), ref.CHRF(word_order=2)
    with pytest.warns(pyroid.FallbackWarning):
        got_c = c_ours.corpus_score(hyps, refs, n_bootstrap=20)
    assert str(got_c) == str(c_theirs.corpus_score(hyps, refs, n_bootstrap=20))


@pytest.mark.thread_unsafe
def test_ter_and_datasets_are_delegated(ref: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sb, "_WARNED", set())
    hyps, refs = ["the cat sat", "a dog"], [["the cat sat down", "the dog"]]
    with pytest.warns(pyroid.FallbackWarning, match="corpus_ter"):
        assert str(sb.corpus_ter(hyps, refs)) == str(ref.corpus_ter(hyps, refs))
    with pytest.warns(pyroid.FallbackWarning, match="sentence_ter"):
        assert str(sb.sentence_ter("a b", ["a c"])) == str(ref.sentence_ter("a b", ["a c"]))
    with pytest.warns(pyroid.FallbackWarning, match=r"\.TER:"):
        ter = sb.TER(case_sensitive=True)
    ref_ter = ref.TER(case_sensitive=True)
    assert str(ter.corpus_score(hyps, refs)) == str(ref_ter.corpus_score(hyps, refs))
    assert str(ter.get_signature()) == str(ref_ter.get_signature())
    assert ter.case_sensitive is True
    with pytest.warns(pyroid.FallbackWarning, match="DATASETS"):
        assert list(sb.DATASETS) == list(ref.DATASETS)
    assert "wmt20" in sb.DATASETS
    with pytest.warns(pyroid.FallbackWarning, match="get_available_testsets"):
        assert sb.get_available_testsets() == ref.get_available_testsets()
    with pytest.warns(pyroid.FallbackWarning, match="get_langpairs_for_testset"):
        assert sb.get_langpairs_for_testset("wmt20") == ref.get_langpairs_for_testset("wmt20")
    # Once per function and reason.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sb.corpus_ter(hyps, refs)
    assert sb.SACREBLEU_DIR == ref.SACREBLEU_DIR


@pytest.mark.thread_unsafe
def test_delegated_tokenizer_warns_then_runs_sacrebleu(
    ref: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sb, "_WARNED", set())

    def make(m: Any) -> tuple[str, str]:
        try:
            return ("ok", m.BLEU(trg_lang="ja").tokenizer_signature)
        except Exception as e:
            return ("raised", type(e).__name__)

    with pytest.warns(pyroid.FallbackWarning, match="tokenize=.ja-mecab."):
        ours = make(sb)
    assert ours == make(ref)


# ---------------------------------------------------------------------------
# No reference needed (all builds)
# ---------------------------------------------------------------------------


def test_python_whitespace_on_this_interpreter() -> None:
    """The kernel's str.split() whitespace is this interpreter's, on every code point."""
    cps = [c for c in range(0x80, 0x110000) if not 0xD800 <= c < 0xE000]
    for i in range(0, len(cps), 4096):
        line = "".join(f"x{chr(c)}" for c in cps[i : i + 4096])
        # 13a only touches ASCII punctuation: on this line it is ' '.join(line.split()).
        assert _native.sb_tokenize("13a", line) == " ".join(line.split())
    for c in range(0x80):
        line = f"x{chr(c)}y"
        assert sb.sentence_bleu(line, [line], tokenize="none").sys_len == len(line.split())
        assert sb.sentence_bleu(line + chr(c), ["x"], tokenize="char").sys_len == len(
            "".join((line + chr(c)).rstrip().split())
        )


def test_char_ngrams_are_code_points() -> None:
    score = sb.sentence_chrf("日本語😀", ["日本語😀"], char_order=6)
    assert score.score == 100.0
    stats = sb.CHRF(char_order=2)._extract_corpus_statistics(["é😀x"], [["é😀y"]])
    assert stats == [[3, 3, 2, 2, 2, 1]]


def test_lone_surrogates_raise() -> None:
    with pytest.raises(UnicodeEncodeError) as e:
        sb.corpus_bleu(["ok", "bad\ud800"], [["ok", "x"]])
    assert any("hypotheses[1]" in n for n in getattr(e.value, "__notes__", []))
    with pytest.raises(UnicodeEncodeError) as e:
        sb.corpus_chrf(["ok"], [["ok"], ["\udfff"]])
    assert any("references[1][0]" in n for n in getattr(e.value, "__notes__", []))
    with pytest.raises(UnicodeEncodeError):
        sb.sentence_bleu("a", ["\ud800"])


def test_wrong_item_types_raise_input_error() -> None:
    with pytest.raises(pyroid.InputError, match=r"hypotheses\[1\]") as info:
        sb.corpus_bleu(["a", 5], [["a", "b"]])  # type: ignore[list-item]
    assert "while processing 'hypotheses'" in info.value.__notes__
    with pytest.raises(pyroid.InputError, match=r"references\[0\]\[1\]") as info:
        sb.corpus_chrf(["a", "b"], [["a", b"b"]])  # type: ignore[list-item]
    assert "while processing 'references'" in info.value.__notes__
    with pytest.raises(pyroid.InputError, match="pass \\[refs\\]"):
        sb.sentence_bleu_batch(["a", "b"], ["a", "b"])
    with pytest.raises(ValueError, match="2 hypotheses"):
        sb.sentence_chrf_batch(["a", "b"], [["a"]])
    with pytest.raises(ValueError, match="at least one"):
        sb.sentence_bleu_batch(["a"], [])
    with pytest.raises(TypeError, match="`hyps` should be a sequence"):
        sb.corpus_bleu(5, [["a"]])  # type: ignore[arg-type]


def _summaries(scores: list[Any]) -> list[Any]:
    return [None if s is None else summary(s) for s in scores]


@pytest.mark.parametrize("which", ["bleu", "chrf"])
@pytest.mark.parametrize("kw", [{}, {"lowercase": True}, {"tokenize": "none"}, {"eps": True}])
def test_sentence_batch_input_kinds(pa: Any, which: str, kw: dict[str, Any]) -> None:
    # str, list, tuple of str or Arrow in; always a list out; Arrow nulls give None.
    if which == "bleu":
        if "eps" in kw:
            kw = {"smooth_method": "floor"}
        batch: Callable[..., Any] = sb.sentence_bleu_batch
        one: Callable[..., Any] = sb.sentence_bleu
    else:
        kw = {"eps_smoothing": True} if "eps" in kw else {"word_order": 2}
        batch, one = sb.sentence_chrf_batch, sb.sentence_chrf
    hyps: list[str | None] = ["The cat sat .", None, "a b c d", "", None, "x Y z"]
    refs: list[list[str | None]] = [
        ["the cat sat .", "b", None, "", "q", "x y"],
        ["a cat", None, "a b c", "", None, None],
    ]
    want = [
        None if h is None else summary(one(h, [r[i] for r in refs if r[i] is not None], **kw))
        for i, h in enumerate(hyps)
    ]
    full = [i for i, h in enumerate(hyps) if h is not None]
    dense = [hyps[i] for i in full]
    dense_refs = [[r[i] for i in full] for r in refs]
    assert _summaries(batch(dense, dense_refs, **kw)) == [want[i] for i in full]
    assert _summaries(batch(tuple(dense), [tuple(r) for r in dense_refs], **kw)) == [
        want[i] for i in full
    ]
    for h_col, r_cols in [
        (pa.array(hyps, pa.string()), [pa.array(r, pa.string()) for r in refs]),
        (pa.chunked_array([hyps[:3], hyps[3:]], pa.large_string()), [refs[0], pa.array(refs[1])]),
    ]:
        got = batch(h_col, r_cols, threads=2, **kw)
        assert type(got) is list
        assert _summaries(got) == want
    assert batch(pa.array([None, None], pa.string()), [["a", "b"]], **kw) == [None, None]
    with pytest.raises(pyroid.InputError, match="hypotheses must be a sequence"):
        batch("a b", [["a b"]], **kw)


@pytest.mark.parametrize("kw", [{}, {"lowercase": True}, {"tokenize": "none"}])
def test_sentence_batch_none_hypothesis_raises(kw: dict[str, Any]) -> None:
    for batch in (sb.sentence_bleu_batch, sb.sentence_chrf_batch):
        args = {} if batch is sb.sentence_chrf_batch else kw
        with pytest.raises(pyroid.InputError, match=r"hypotheses\[1\] must be a str") as info:
            batch(["a", None], [["a", "b"]], **args)  # type: ignore[list-item]
        assert isinstance(info.value, TypeError)
        assert "while processing 'hypotheses'" in getattr(info.value, "__notes__", [])


def test_changed_orders_after_caching_raise() -> None:
    m = sb.BLEU(references=[["a b"]])
    m.max_ngram_order = 2
    with pytest.raises(ValueError, match="max_ngram_order changed"):
        m.corpus_score(["a b"], None)
    c = sb.CHRF(references=[["a b"]])
    c.char_order = 3
    c.order = 3
    with pytest.raises(ValueError, match="char_order or word_order changed"):
        c.corpus_score(["a b"], None)


def _corpus(n: int, seed: int = 1) -> tuple[list[str], list[list[str | None]]]:
    rng = random.Random(seed)
    vocab = [*WORDS, *SPECIAL, "of", "and", "to", "in", "is", "it", "that", "for", "was"]
    hyps = [" ".join(rng.choice(vocab) for _ in range(rng.randint(0, 30))) for _ in range(n)]
    refs: list[list[str | None]] = [
        [" ".join(h.split()[::-1][: rng.randint(0, 30)]) + " ." for h in hyps],
        [None if rng.random() < 0.3 else " ".join(rng.sample(vocab, 12)) for _ in hyps],
    ]
    return hyps, refs


@pytest.mark.parametrize("threads", [None, 1, 2, 3, 8])
def test_threads_give_the_same_results(threads: int | None) -> None:
    hyps, refs = _corpus(800)
    one = sb.corpus_bleu(hyps, refs, threads=1)
    assert summary(sb.corpus_bleu(hyps, refs, threads=threads)) == summary(one)
    for tok in ("intl", "zh", "char"):
        assert summary(sb.corpus_bleu(hyps, refs, tokenize=tok, threads=threads)) == summary(
            sb.corpus_bleu(hyps, refs, tokenize=tok, threads=1)
        )
    assert summary(sb.corpus_chrf(hyps, refs, word_order=2, threads=threads)) == summary(
        sb.corpus_chrf(hyps, refs, word_order=2, threads=1)
    )
    assert summary(sb.sentence_bleu_batch(hyps, refs, threads=threads)) == summary(
        sb.sentence_bleu_batch(hyps, refs, threads=1)
    )
    cached = sb.CHRF(references=refs)
    assert summary(cached.corpus_score(hyps, None, threads=threads)) == summary(
        sb.corpus_chrf(hyps, refs, threads=1)
    )


def test_bad_threads() -> None:
    with pytest.raises(ValueError, match="threads"):
        sb.corpus_bleu(["a"] * 100, [["a"] * 100], threads=0)
    with pytest.raises(pyroid.InputError, match="threads"):
        sb.corpus_chrf(["a"], [["a"]], threads="2")  # type: ignore[arg-type]


@pytest.mark.timeout(120)
def test_ten_thousand_segments(request: pytest.FixtureRequest) -> None:
    hyps, refs = _corpus(10_000, seed=3)
    bleu = sb.corpus_bleu(hyps, refs)
    chrf = sb.corpus_chrf(hyps, refs, word_order=2)
    batch = sb.sentence_bleu_batch(hyps, refs)
    assert len(batch) == 10_000
    if not FREE_THREADED:
        ref = request.getfixturevalue("ref")
        assert summary(bleu) == summary(ref.corpus_bleu(hyps, refs))
        assert summary(chrf) == summary(ref.corpus_chrf(hyps, refs, word_order=2))
        for i in range(0, 10_000, 97):
            want = ref.sentence_bleu(hyps[i], [r[i] for r in refs])
            assert summary(batch[i]) == summary(want)


def _real_sentences() -> Iterator[tuple[str, list[str]]]:
    """(language, sentences) from the shared FineWeb-2 / FineWeb dev slices, if mounted."""
    pq = pytest.importorskip("pyarrow.parquet")
    files = sorted(glob.glob("/data/fineweb2/*.parquet"))
    files += [f for f in ["/data/fineweb/fineweb-dev-10k.parquet"] if os.path.exists(f)]
    if not files:
        pytest.skip("/data (FineWeb slices) is not mounted")
    for f in files:
        batch = next(pq.ParquetFile(f).iter_batches(batch_size=120, columns=["text"]))
        texts = [t for t in batch.column(0).to_pylist() if t]
        sentences = [s for t in texts for s in t.replace("。", "。\n").split("\n") if s.strip()]
        yield os.path.basename(f).split(".")[0], sentences[:400]


@pytest.mark.timeout(300)
def test_real_text(ref: Any) -> None:
    """Sentence i against i+1 (and i+2 as a second reference) over 9 languages."""
    for lang, sents in _real_sentences():
        hyps, r1, r2 = sents[:-2], sents[1:-1], sents[2:]
        refs: list[Sequence[str | None]] = [r1, [s if i % 3 else None for i, s in enumerate(r2)]]
        toks = ["13a", "intl"] + (["zh", "char"] if lang in ("cmn_Hani", "jpn_Jpan") else [])
        for tok in toks:
            assert summary(sb.corpus_bleu(hyps, refs, tokenize=tok)) == summary(
                ref.corpus_bleu(hyps, refs, tokenize=tok)
            ), (lang, tok)
        for kw in ({}, {"word_order": 2}, {"remove_whitespace": False, "eps_smoothing": True}):
            assert summary(sb.corpus_chrf(hyps, refs, **kw)) == summary(
                ref.corpus_chrf(hyps, refs, **kw)
            ), (lang, kw)
        batch = sb.sentence_bleu_batch(hyps, refs)
        chrfs = sb.sentence_chrf_batch(hyps, refs, word_order=2)
        for i in itertools.islice(range(len(hyps)), 0, None, 7):
            seg_refs = [r[i] for r in refs]
            assert summary(batch[i]) == summary(ref.sentence_bleu(hyps[i], seg_refs)), (lang, i)
            assert summary(chrfs[i]) == summary(
                ref.sentence_chrf(hyps[i], seg_refs, word_order=2)
            ), (lang, i)


@pytest.mark.thread_unsafe
def test_missing_sacrebleu_raises_pyroid_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sacrebleu", None)
    with pytest.raises(pyroid.PyroidError, match=r"pip install sacrebleu==2\.6\.0"):
        sb.corpus_ter(["a"], [["a"]])
    with pytest.raises(pyroid.PyroidError, match="tokenize='spm'"):
        sb.BLEU(tokenize="spm")
    with pytest.raises(pyroid.PyroidError, match="n_bootstrap"):
        sb.BLEU().corpus_score(["a"], [["a"]], n_bootstrap=2)
    with pytest.raises(pyroid.PyroidError, match="DATASETS"):
        len(sb.DATASETS)


@pytest.mark.thread_unsafe
def test_import_keeps_the_gil_disabled(run_python: Callable[..., Any]) -> None:
    proc = run_python(
        """
        import sys
        import pyroid.metrics.sacrebleu as sb
        sb.corpus_bleu(["a b"] * 200, [["a b"] * 200])
        sb.sentence_chrf_batch(["a b"] * 200, [["a c"] * 200])
        assert "sacrebleu" not in sys.modules
        print(getattr(sys, "_is_gil_enabled", lambda: True)())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(not FREE_THREADED)


def test_api_surface() -> None:
    assert sb.__version__ == "2.6.0"
    assert list(sb.BLEU.TOKENIZERS) == [
        "none", "zh", "13a", "intl", "char", "ja-mecab", "ko-mecab", "spm", "flores101",
        "flores200", "spBLEU-1K",
    ]  # fmt: skip
    assert sb.BLEU.SMOOTH_DEFAULTS == {"none": None, "floor": 0.1, "add-k": 1, "exp": None}
    assert (sb.CHRF.CHAR_ORDER, sb.CHRF.WORD_ORDER, sb.CHRF.BETA) == (6, 0, 2)
    assert sb.BLEU().tokenizer("Hello, world!") == "Hello , world !"
    assert sb.BLEU(tokenize="none").tokenizer_signature == "none"
    m = sb.BLEU(references=[["a b", "c"], ["a", None]])
    assert m.num_refs == -1
    assert str(sb.corpus_bleu(["a b"], [["a b"]])).startswith("BLEU = 0.00 100.0/100.0/0.0/0.0")
    assert sb.extract_word_ngrams(["a", "b", "a", "b"], 2) == {"a b": 2, "b a": 1}
    assert sb.extract_char_ngrams("xy x", 2) == {"xy": 1, "yx": 1}
    assert sb.extract_char_ngrams("xy x", 2, include_whitespace=True) == {"xy": 1, "y ": 1, " x": 1}
    assert set(sb.DELEGATED) <= set(sb.__all__)
