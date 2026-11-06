# Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
"""pyroid.metrics.rouge_scorer / scoring / tokenizers must equal rouge-score 0.1.2 exactly.

Differential tests against the real rouge-score (and nltk 3.10.4's Porter
stemmer): every score is compared with ``==`` and by ``repr`` (which pins float
bits and int-vs-float zeros), key order and result types included. The cases
of rouge-score's own ``*_test.py`` are ported where they don't need its test
data files. Set PYROID_ROUGE_EXAMPLES=5000 for a deeper Hypothesis run.
"""

from __future__ import annotations

import os
import pickle
import random
import sys
import sysconfig
import warnings
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import pyroid
from pyroid._pyroid import _testing
from pyroid._pyroid import metrics as native
from pyroid.metrics import rouge_scorer, scoring, tokenizers
from pyroid.metrics import tokenize as ptokenize

_EXAMPLES = os.environ.get("PYROID_ROUGE_EXAMPLES")
examples = (
    settings(suppress_health_check=[HealthCheck.too_slow], max_examples=int(_EXAMPLES))
    if _EXAMPLES
    else settings(suppress_health_check=[HealthCheck.too_slow])
)

ALL_TYPES = [f"rouge{n}" for n in range(1, 10)] + ["rougeL", "rougeLsum"]


@pytest.fixture(scope="module")
def ref() -> Any:
    """rouge-score's rouge_scorer module (the reference), or skip."""
    return pytest.importorskip("rouge_score.rouge_scorer")


@pytest.fixture(scope="module")
def ref_tokenizers() -> Any:
    return pytest.importorskip("rouge_score.tokenizers")


@pytest.fixture(scope="module")
def porter() -> Any:
    """nltk's PorterStemmer (nltk 3.10.4), or skip."""
    nltk = pytest.importorskip("nltk")
    if nltk.__version__ != "3.10.4":
        pytest.skip(f"needs nltk 3.10.4, found {nltk.__version__}")
    return pytest.importorskip("nltk.stem.porter").PorterStemmer()


@pytest.fixture(scope="module")
def punkt() -> Any:
    """nltk with its punkt_tab sentence data, or skip."""
    nltk = pytest.importorskip("nltk")
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        pytest.skip("nltk's punkt_tab data is not installed (nltk.download('punkt_tab'))")
    return nltk


def canon(result: dict[Any, Any]) -> list[tuple[Any, str, tuple[str, ...], str]]:
    """Keys in order, value types, element types and reprs (float bits, int zeros)."""
    return [
        (k, type(v).__name__, tuple(type(x).__name__ for x in v), repr(v))
        for k, v in result.items()
    ]


def assert_same(ours: dict[Any, Any], theirs: dict[Any, Any]) -> None:
    assert ours == theirs
    assert canon(ours) == canon(theirs)
    assert all(type(v) is scoring.Score for v in ours.values())


def outcome(fn: Callable[..., Any], *args: Any) -> tuple[Any, ...]:
    """("ok", canonical result) or ("raise", exception type, message)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pyroid.FallbackWarning)
            r = fn(*args)
    except Exception as e:
        return ("raise", type(e), str(e))
    return ("ok", canon(r) if isinstance(r, dict) else r)


WORDS = [
    "the", "The", "THE", "cat", "cats", "Cats", "running", "runs", "ran", "happy", "happily",
    "generalization", "generalizations", "agreed", "feed", "\u0130stanbul", "\u212aelvin",
    "ß", "straße", "a1", "1990s", "42", "dying", "skies", "news", "y", "yy", "yyyy",
    "caresses", "ponies", "a", "an", "relational", "hopefulness", "controlling",
    "électricité", "日本語",
]  # fmt: skip
SEPS = [
    " ", "  ", "\n", "\r\n", "\r", "\t", ", ", ". ", "!", "-", "\u00a0", "\u2009", "\u0130",
    "\u212a", "\n\n", " \n ", "\u0307", "_",
]  # fmt: skip
word_text = st.lists(st.one_of(st.sampled_from(WORDS), st.sampled_from(SEPS)), max_size=40).map(
    "".join
)
texts = st.one_of(
    word_text,
    st.text(max_size=80),
    st.text(st.sampled_from("abcAB \u0130K\u212a\n\r ß.y1"), max_size=60),
)


def long_doc(seed: int, n: int) -> str:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(rng.choice(WORDS))
        out.append(rng.choice(SEPS[:8]))
    return "".join(out)


# ---------------------------------------------------------------------------
# Parity with rouge-score
# ---------------------------------------------------------------------------


@examples
@given(target=texts, prediction=texts, stem=st.booleans())
@example(target="", prediction="", stem=False)
@example(target=" \n\t", prediction="!!!", stem=True)
@example(target="\u0130STANBUL \u212aelvin \u212a", prediction="i stanbul kelvin k", stem=False)
@example(target="a\n\nb c\n", prediction="c b\na", stem=True)
def test_all_types_match(ref: Any, target: str, prediction: str, stem: bool) -> None:
    ours = rouge_scorer.RougeScorer(ALL_TYPES, use_stemmer=stem).score(target, prediction)
    theirs = ref.RougeScorer(ALL_TYPES, use_stemmer=stem).score(target, prediction)
    assert_same(ours, theirs)


@examples
@given(
    target=texts,
    prediction=texts,
    stem=st.booleans(),
    types=st.lists(st.sampled_from(ALL_TYPES), min_size=1, max_size=4),
)
def test_type_subsets_match(
    ref: Any, target: str, prediction: str, stem: bool, types: list[str]
) -> None:
    # Single types exercise rouge-score's rougeLsum-only shortcut; repeats and order matter.
    ours = rouge_scorer.RougeScorer(types, use_stemmer=stem).score(target, prediction)
    theirs = ref.RougeScorer(types, use_stemmer=stem).score(target, prediction)
    assert_same(ours, theirs)


@pytest.mark.parametrize("stem", [False, True])
@pytest.mark.parametrize("seed", [1, 2])
def test_long_documents_match(ref: Any, stem: bool, seed: int) -> None:
    target, prediction = long_doc(seed, 700), long_doc(seed + 100, 500)
    types = ["rouge1", "rouge2", "rouge4", "rougeL", "rougeLsum"]
    ours = rouge_scorer.RougeScorer(types, use_stemmer=stem).score(target, prediction)
    assert_same(ours, ref.RougeScorer(types, use_stemmer=stem).score(target, prediction))
    # One line each: rougeLsum's backtracking over a full 700 x 500 table.
    one = (target.replace("\n", " "), prediction.replace("\n", " "))
    ours = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=stem).score(*one)
    assert_same(ours, ref.RougeScorer(["rougeLsum"], use_stemmer=stem).score(*one))


TYPE_CASES: list[Any] = [
    ["rouge1\n"],
    ["rouge0"],
    ["rouge0\n"],
    ["rouge10"],
    ["rougen"],
    ["rougeL\n"],
    ["rougel"],
    ["Rouge1"],
    [""],
    [],
    ["rouge1", "rouge1"],
    ["rougeL", "rouge2", "rougeL"],
    ["rouge1", "rouge0", "rougeX"],
    ["rougeX", "rouge0"],
    ["rougeLsum", "rougeX"],
    ("rouge1", "rougeLsum"),
    "rouge1",
    "rougeL",
    [b"rouge1"],
    [b"rougeL"],
    [b"rouge0"],
    [1],
    [None],
    ["rouge1", 2],
    ["rouge\ud800"],
    ["rougeLsum"],
    ("rougeLsum",),
    [["rougeL"]],
]


@pytest.mark.parametrize("types", TYPE_CASES, ids=repr)
def test_rouge_types_and_errors_match(ref: Any, types: Any) -> None:
    t, p = "testing one two\nthree", "testing two"
    ours = outcome(rouge_scorer.RougeScorer(types).score, t, p)
    theirs = outcome(ref.RougeScorer(types).score, t, p)
    assert ours == theirs


class StrSub(str):
    __slots__ = ()


INPUT_CASES: list[tuple[Any, Any]] = [
    (b"Hello \xc4\xb0x WORLD", "hello i x world"),
    (b"caf\xc3\xa9 \xe2\x84\xaa", b"cafe k"),
    (StrSub("Hello \u0130 there"), "hello i there"),
    (None, "x"),
    ("x", 3),
    (bytearray(b"x"), "x"),
    (b"\xff", "x"),
]


@pytest.mark.parametrize("types", [["rouge1", "rougeL", "rougeLsum"], ["rougeLsum"], ["rouge2"]])
@pytest.mark.parametrize(("target", "prediction"), INPUT_CASES, ids=repr)
def test_non_str_inputs_match(ref: Any, types: list[str], target: Any, prediction: Any) -> None:
    for stem in (False, True):
        ours = outcome(rouge_scorer.RougeScorer(types, use_stemmer=stem).score, target, prediction)
        theirs = outcome(ref.RougeScorer(types, use_stemmer=stem).score, target, prediction)
        assert ours == theirs


def test_rouge_types_are_read_at_every_call(ref: Any) -> None:
    ours, theirs = rouge_scorer.RougeScorer(["rouge1"]), ref.RougeScorer(["rouge1"])
    ours.rouge_types = theirs.rouge_types = ["rougeL", "rouge3"]
    assert_same(ours.score("a b c d", "a c d"), theirs.score("a b c d", "a c d"))


# ---------------------------------------------------------------------------
# rouge-score's own unit tests (rouge_scorer_test.py), ported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rouge_type", ["rougen", "rouge0", "rouge10"])
def test_invalid_rouge_types(rouge_type: str) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        rouge_scorer.RougeScorer([rouge_type]).score("testing one two", "testing")


@pytest.mark.parametrize("rouge_type", ["rouge1", "rouge9", "rougeL", "rougeLsum"])
def test_valid_rouge_types(rouge_type: str) -> None:
    result = rouge_scorer.RougeScorer([rouge_type]).score("testing one two", "testing")
    assert list(result) == [rouge_type]


def test_rouge1_and_multi() -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    for result in (
        scorer.score("testing one two", "testing"),
        scorer.score_multi(["testing one two"], "testing"),
    ):
        assert result["rouge1"] == (1.0, 1 / 3, 2 * (1 / 3) / (1 + 1 / 3))
        assert result["rouge1"].fmeasure == pytest.approx(0.5)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    result = scorer.score_multi(["first text", "first something"], "text first")
    assert result["rouge1"].fmeasure == 1
    assert result["rouge2"].fmeasure == 0
    assert result["rougeL"].fmeasure == 0.5


@pytest.mark.parametrize("rouge_type", ["rouge1", "rouge2", "rougeL", "rougeLsum"])
def test_rouge_empty(rouge_type: str) -> None:
    result = rouge_scorer.RougeScorer([rouge_type]).score("testing one two", "")
    assert result[rouge_type] == (0, 0, 0)


def test_rouge2_and_rougel() -> None:
    assert rouge_scorer.RougeScorer(["rouge2"]).score("testing one two", "testing one")[
        "rouge2"
    ] == pytest.approx((1, 1 / 2, 2 / 3))
    lcs = rouge_scorer.RougeScorer(["rougeL"])
    for prediction in ("testing one", "testing two"):
        assert lcs.score("testing one two", prediction)["rougeL"] == pytest.approx(
            (1, 2 / 3, 4 / 5)
        )
    result = rouge_scorer.RougeScorer(["rouge1", "rougeL"]).score("testing one two", "testing one")
    assert list(result) == ["rouge1", "rougeL"]
    assert result["rouge1"] == result["rougeL"] == pytest.approx((1, 2 / 3, 4 / 5))


def test_rouge_lsum() -> None:
    scorer = rouge_scorer.RougeScorer(["rougeLsum"])
    result = scorer.score("w1 w2 w3 w4 w5", "w1 w2 w6 w7 w8\nw1 w3 w8 w9 w5")
    assert result["rougeLsum"].recall == pytest.approx(0.8)
    assert result["rougeLsum"].precision == pytest.approx(0.4)
    assert result["rougeLsum"].fmeasure == pytest.approx(0.5333, abs=1e-3)
    for t, p in [("w1 w2 w3 w4 w5", ""), ("", "w1"), ("w1 w2 w3 w4 w5", "/")]:
        assert scorer.score(t, p)["rougeLsum"] == (0, 0, 0)


def test_rouge_lsum_sentence_splitting() -> None:
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
    target = "First sentence.\nSecond Sentence."
    prediction = "Second sentence.\nFirst Sentence."
    assert scorer.score(target, prediction)["rougeLsum"].fmeasure == pytest.approx(1.0)
    # Without newlines, summaries are treated as single sentences.
    target, prediction = target.replace("\n", " "), prediction.replace("\n", " ")
    assert scorer.score(target, prediction)["rougeLsum"].fmeasure == pytest.approx(0.5)


@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_rouge_lsum_sentence_splitting_nltk(punkt: Any) -> None:
    # Split summaries into sentences using nltk.
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True, split_summaries=True)
    result = scorer.score("First sentence. Second Sentence.", "Second sentence. First Sentence.")
    assert result["rougeLsum"].fmeasure == pytest.approx(1.0)


def test_default_tokenizer_init() -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1"], tokenizer=tokenizers.DefaultTokenizer())
    assert scorer.score("this is a test", "this is a test")["rouge1"].fmeasure == 1.0
    assert isinstance(scorer, scoring.BaseScorer)


# ---------------------------------------------------------------------------
# Custom tokenizers
# ---------------------------------------------------------------------------


class Whitespace:
    """Duck-typed tokenizer (not a Tokenizer subclass)."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


class Chars(tokenizers.Tokenizer):
    """Arbitrary Unicode tokens, one per character."""

    def tokenize(self, text: str) -> list[str]:
        return list(text)


class AsTuple:
    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(text.lower().split())


class Recording:
    """Records every call, to compare where rouge-score calls the tokenizer."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize(self, text: str) -> list[str]:
        self.calls.append(text)
        return text.split()


class Falsy:
    def __bool__(self) -> bool:
        return False

    def tokenize(self, text: str) -> list[str]:
        raise AssertionError("a falsy tokenizer is replaced by the default one")


@examples
@given(target=texts, prediction=texts)
def test_custom_tokenizers_match(ref: Any, target: str, prediction: str) -> None:
    for tok in (Whitespace(), Chars(), AsTuple()):
        ours = rouge_scorer.RougeScorer(ALL_TYPES, tokenizer=tok).score(target, prediction)
        theirs = ref.RougeScorer(ALL_TYPES, tokenizer=tok).score(target, prediction)
        assert_same(ours, theirs)


@pytest.mark.parametrize(
    "types",
    [
        ["rougeLsum"],
        ["rouge1", "rougeLsum", "rougeLsum"],
        ["rougeX"],
        ["rouge1", "rougeX", "rougeLsum"],
        ["rougeLsum", "rouge0"],
        [],
    ],
)
def test_custom_tokenizer_call_order_matches(ref: Any, types: list[str]) -> None:
    t, p = "a b\nc d\n\ne", "b a\nd"
    mine, theirs = Recording(), Recording()
    assert outcome(rouge_scorer.RougeScorer(types, tokenizer=mine).score, t, p) == outcome(
        ref.RougeScorer(types, tokenizer=theirs).score, t, p
    )
    assert mine.calls == theirs.calls


def test_falsy_tokenizer_means_default(ref: Any) -> None:
    ours = rouge_scorer.RougeScorer(["rouge1"], tokenizer=Falsy()).score("A b", "a B")
    assert_same(ours, ref.RougeScorer(["rouge1"], tokenizer=Falsy()).score("A b", "a B"))


class Lengths:
    """Non-str tokens: scored by the real rouge-score."""

    def tokenize(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]


@pytest.mark.thread_unsafe(reason="global one-time-warning state")
def test_non_str_tokens_delegate_once(ref: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rouge_scorer, "_WARNED", set())
    scorer = rouge_scorer.RougeScorer(ALL_TYPES, tokenizer=Lengths())  # type: ignore[arg-type]
    t, p = "aa b ccc\ndd", "b aa\nccc dd e"
    with pytest.warns(pyroid.FallbackWarning, match="not lists of str") as record:
        ours = scorer.score(t, p)
    assert len(record) == 1
    assert record[0].filename == __file__
    assert_same(ours, ref.RougeScorer(ALL_TYPES, tokenizer=Lengths()).score(t, p))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scorer.score(t, p)  # warned once only


@pytest.mark.thread_unsafe(reason="monkeypatches sys.modules")
def test_non_str_tokens_without_rouge_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rouge_score", None)
    with pytest.raises(pyroid.PyroidError, match=r"pip install rouge-score==0\.1\.2"):
        rouge_scorer.RougeScorer(["rouge1"], tokenizer=Lengths()).score(  # type: ignore[arg-type]
            "a b", "a"
        )


@pytest.mark.parametrize("bad", [["a", 1], ["\ud800"], "ab", [b"a"]], ids=repr)
def test_token_jobs_return_none_for_unsupported_tokens(bad: Any) -> None:
    assert native.rouge_score_tokens([(1, bad, ["a"]), (0, ["a"], ["a"])]) == [
        None,
        scoring.Score(1.0, 1.0, 1.0),
    ]
    assert native.rouge_score_tokens([(-1, [bad], [["a"]])]) == [None]


# ---------------------------------------------------------------------------
# Scoring internals against rouge-score's functions (LCS tie-breaking)
# ---------------------------------------------------------------------------

small_tokens = st.lists(st.sampled_from(["a", "b", "c", "ab", "ß", ""]), max_size=12)


@examples
@given(
    refs=st.lists(small_tokens, max_size=5),
    cands=st.lists(small_tokens, max_size=5),
    n=st.integers(1, 9),
)
@example(
    refs=[["1", "2", "3", "4", "5"]],
    cands=[["1", "2", "6", "7", "8"], ["1", "3", "8", "9", "5"]],
    n=1,
)
@example(refs=[["reference"]], cands=[[]], n=1)
def test_token_scoring_matches_rouge_score_functions(
    ref: Any, refs: list[list[str]], cands: list[list[str]], n: int
) -> None:
    flat_r = [t for s in refs for t in s]
    flat_c = [t for s in cands for t in s]
    got = native.rouge_score_tokens([(-1, refs, cands), (0, flat_r, flat_c), (n, flat_r, flat_c)])
    want = [
        ref._summary_level_lcs(refs, cands),
        ref._score_lcs(flat_r, flat_c),
        ref._score_ngrams(ref._create_ngrams(flat_r, n), ref._create_ngrams(flat_c, n)),
    ]
    assert [repr(tuple(s)) for s in got] == [repr(tuple(s)) for s in want]


# ---------------------------------------------------------------------------
# score_multi
# ---------------------------------------------------------------------------


@examples
@given(targets=st.lists(texts, min_size=1, max_size=4), prediction=texts, stem=st.booleans())
def test_score_multi_matches(ref: Any, targets: list[str], prediction: str, stem: bool) -> None:
    types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    ours = rouge_scorer.RougeScorer(types, use_stemmer=stem).score_multi(targets, prediction)
    theirs = ref.RougeScorer(types, use_stemmer=stem).score_multi(targets, prediction)
    assert_same(ours, theirs)


def test_score_multi_edge_cases(ref: Any) -> None:
    for types, targets in [(["rouge1"], []), ([], []), (["rougeX"], []), (["rougeX"], ["a"])]:
        assert outcome(rouge_scorer.RougeScorer(types).score_multi, targets, "a") == outcome(
            ref.RougeScorer(types).score_multi, targets, "a"
        )
    # Ties: the first maximal target wins, as np.argmax picks it.
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"])
    theirs = ref.RougeScorer(["rouge1", "rougeL"])
    targets = ["b a", "a b", "a b c"]
    assert_same(scorer.score_multi(targets, "a b"), theirs.score_multi(targets, "a b"))
    assert_same(scorer.score_multi(iter(targets), "a b"), scorer.score_multi(targets, "a b"))


# ---------------------------------------------------------------------------
# score_batch
# ---------------------------------------------------------------------------


def pairs(n: int, seed: int = 0) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    return [long_doc(rng.randrange(10**6), rng.randrange(0, 40)) for _ in range(n)], [
        long_doc(rng.randrange(10**6), rng.randrange(0, 40)) for _ in range(n)
    ]


@pytest.mark.parametrize("stem", [False, True])
def test_score_batch_equals_score(stem: bool) -> None:
    targets, predictions = pairs(300)
    scorer = rouge_scorer.RougeScorer(ALL_TYPES, use_stemmer=stem)
    want = [scorer.score(t, p) for t, p in zip(targets, predictions, strict=True)]
    for threads in (None, 1, 2, 4):
        got = scorer.score_batch(targets, predictions, threads=threads)
        assert [canon(d) for d in got] == [canon(d) for d in want]
    got = scorer.score_batch(tuple(targets), tuple(predictions))
    assert [canon(d) for d in got] == [canon(d) for d in want]


@examples
@given(batch=st.lists(st.tuples(texts, texts), max_size=8), stem=st.booleans())
def test_score_batch_matches_rouge_score(
    ref: Any, batch: list[tuple[str, str]], stem: bool
) -> None:
    targets, predictions = [t for t, _ in batch], [p for _, p in batch]
    got = rouge_scorer.RougeScorer(ALL_TYPES, use_stemmer=stem).score_batch(targets, predictions)
    theirs = ref.RougeScorer(ALL_TYPES, use_stemmer=stem)
    assert [canon(d) for d in got] == [
        canon(theirs.score(t, p)) for t, p in zip(targets, predictions, strict=True)
    ]


@pytest.mark.thread_unsafe(reason="10k pairs: heavy per thread")
@pytest.mark.timeout(120)
def test_score_batch_10k_pairs() -> None:
    targets, predictions = pairs(10_000, seed=5)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    got = scorer.score_batch(targets, predictions)
    assert len(got) == 10_000
    assert got == scorer.score_batch(targets, predictions, threads=1)
    for i in range(0, 10_000, 97):
        assert canon(got[i]) == canon(scorer.score(targets[i], predictions[i]))


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_score_batch_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    targets, predictions = pairs(100, seed=3)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL", "rougeLsum"])
    want = [canon(scorer.score(t, p)) for t, p in zip(targets, predictions, strict=True)]
    col_t = _testing.column(targets, large=large, chunks=chunks)
    col_p = _testing.column(predictions, large=not large, chunks=2)
    assert [canon(d) for d in scorer.score_batch(col_t, col_p)] == want
    assert [canon(d) for d in scorer.score_batch(col_t, predictions, threads=1)] == want


def test_score_batch_pyarrow(pa: Any) -> None:
    targets, predictions = pairs(120, seed=4)
    scorer = rouge_scorer.RougeScorer(["rouge2", "rougeLsum"], use_stemmer=True)
    want = [canon(scorer.score(t, p)) for t, p in zip(targets, predictions, strict=True)]
    variants = [
        pa.array(targets),
        pa.array(targets, type=pa.large_string()),
        pa.chunked_array([targets[:50], targets[50:]]),
        pa.array(["pad", *targets]).slice(1),
    ]
    if hasattr(pa, "string_view"):
        variants.append(pa.array(targets, type=pa.string_view()))
    for col in variants:
        assert [canon(d) for d in scorer.score_batch(col, pa.array(predictions))] == want


def test_score_batch_arrow_nulls(pa: Any) -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    with pytest.raises(ValueError, match=r"predictions\[1\] is null"):
        scorer.score_batch(["a", "b"], pa.array(["a", None]))
    with pytest.raises(ValueError, match=r"targets\[0\] is null"):
        scorer.score_batch(_testing.column([None]), ["x"])


def test_score_batch_errors() -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    with pytest.raises(ValueError, match="same length, got 2 and 1"):
        scorer.score_batch(["a", "b"], ["a"])
    with pytest.raises(ValueError, match="same length"):
        rouge_scorer.RougeScorer(["rouge1"], tokenizer=Whitespace()).score_batch(["a"], [])
    with pytest.raises(pyroid.InputError, match="not a single str"):
        scorer.score_batch("abc", ["a"])
    with pytest.raises(pyroid.InputError):
        scorer.score_batch(["a"], 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="threads"):
        scorer.score_batch(["a"], ["a"], threads=0)
    with pytest.raises(ValueError, match="threads"):
        rouge_scorer.RougeScorer(["rouge1"], tokenizer=Whitespace()).score_batch(
            ["a"], ["a"], threads=0
        )
    with pytest.raises(ValueError, match="Invalid rouge type: rougeX"):
        rouge_scorer.RougeScorer(["rouge1", "rougeX"]).score_batch([], [])
    assert scorer.score_batch([], []) == []


@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")  # Chars() on bytes gives ints
def test_score_batch_slow_paths_equal_score(ref: Any) -> None:
    # Non-str items, custom tokenizers and unusual rouge_types go pair by pair.
    targets: list[Any] = [b"Hello \xc4\xb0x", "a b c", StrSub("A b")]
    predictions: list[Any] = ["hello i x", b"c b a", "a B"]
    configs: list[tuple[Any, dict[str, Any]]] = [
        (ALL_TYPES, {}),
        (ALL_TYPES, {"tokenizer": Chars()}),
        (("rouge1", "rougeL"), {}),
        ([b"rouge2"], {}),
    ]
    for types, kwargs in configs:
        scorer = rouge_scorer.RougeScorer(types, **kwargs)
        theirs = ref.RougeScorer(types, **kwargs)
        got = scorer.score_batch(targets, predictions, threads=2)
        assert [canon(d) for d in got] == [
            canon(theirs.score(t, p)) for t, p in zip(targets, predictions, strict=True)
        ]


@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_split_summaries_match(ref: Any, punkt: Any, pa: Any) -> None:
    targets = [
        "Mr. Smith went to Washington. He liked it! Did he? Yes.",
        "No split here",
        "One. Two.\nThree... four",
        "",
    ]
    predictions = [
        "Smith went. He did not like it. Yes.",
        "split here",
        "Three. Two. One.",
        "x. y.",
    ]
    for types in (["rougeLsum"], ["rouge1", "rougeLsum", "rougeL"], ["rouge2"]):
        ours = rouge_scorer.RougeScorer(types, use_stemmer=True, split_summaries=True)
        theirs = ref.RougeScorer(types, use_stemmer=True, split_summaries=True)
        want = [canon(theirs.score(t, p)) for t, p in zip(targets, predictions, strict=True)]
        assert [canon(ours.score(t, p)) for t, p in zip(targets, predictions, strict=True)] == want
        assert [canon(d) for d in ours.score_batch(targets, predictions)] == want
        assert [canon(d) for d in ours.score_batch(pa.array(targets), predictions)] == want


@pytest.mark.thread_unsafe(reason="global one-time-warning state")
def test_split_summaries_warns_once(punkt: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rouge_scorer, "_WARNED", set())
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], split_summaries=True)
    with pytest.warns(pyroid.FallbackWarning, match="nltk.sent_tokenize") as record:
        scorer.score("A b. C d.", "C d. A b.")
    assert len(record) == 1
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scorer.score("A b. C d.", "C d. A b.")
        # No nltk needed without a rougeLsum.
        rouge_scorer.RougeScorer(["rouge1"], split_summaries=True).score("a. b.", "a.")


@pytest.mark.thread_unsafe(reason="monkeypatches sys.modules")
def test_split_summaries_without_nltk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "nltk", None)
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], split_summaries=True)
    with pytest.raises(pyroid.PyroidError, match="punkt_tab"):
        scorer.score("a. b.", "a.")
    assert list(rouge_scorer.RougeScorer(["rouge1"], split_summaries=True).score("a", "a"))


@pytest.mark.thread_unsafe(reason="monkeypatches nltk")
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_split_summaries_lookup_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    nltk = pytest.importorskip("nltk")

    def missing(text: str, language: str = "english") -> list[str]:
        raise LookupError("Resource 'punkt_tab' not found.")

    monkeypatch.setattr(nltk, "sent_tokenize", missing)
    with pytest.raises(LookupError, match="punkt_tab"):
        rouge_scorer.RougeScorer(["rougeLsum", "rougeX"], split_summaries=True).score("a", "b")
    # An invalid type before the rougeLsum raises first, as in rouge-score.
    with pytest.raises(ValueError, match="Invalid rouge type"):
        rouge_scorer.RougeScorer(["rougeX", "rougeLsum"], split_summaries=True).score("a", "b")


# ---------------------------------------------------------------------------
# Tokenizers and the Porter stemmer
# ---------------------------------------------------------------------------


@examples
@given(text=texts)
def test_default_tokenizer_matches(ref_tokenizers: Any, text: str) -> None:
    for stem in (False, True):
        assert tokenizers.DefaultTokenizer(stem).tokenize(text) == ref_tokenizers.DefaultTokenizer(
            stem
        ).tokenize(text)


def test_tokenizers_module() -> None:
    assert isinstance(tokenizers.DefaultTokenizer(), tokenizers.Tokenizer)
    assert tokenizers.DefaultTokenizer(use_stemmer=False).tokenize("this is a test") == [
        "this",
        "is",
        "a",
        "test",
    ]
    assert tokenizers.DefaultTokenizer(use_stemmer=True).tokenize("the friends had a meeting") == [
        "the",
        "friend",
        "had",
        "a",
        "meet",
    ]
    with pytest.raises(TypeError):
        tokenizers.Tokenizer()  # type: ignore[abstract]


def test_tokenize_module(porter: Any) -> None:
    rt = pytest.importorskip("rouge_score.tokenize")
    assert ptokenize.tokenize("one Two three", None) == ["one", "two", "three"]
    assert ptokenize.tokenize("one\n Two \nthree", None) == ["one", "two", "three"]
    for text in [
        "The friends had a meeting",
        "\u0130STANBUL \u212aelvins",
        b"Caf\xc3\xa9s ran",
        "",
    ]:
        assert ptokenize.tokenize(text, None) == rt.tokenize(text, None)  # type: ignore[arg-type]
        assert ptokenize.tokenize(text, porter) == rt.tokenize(text, porter)  # type: ignore[arg-type]
    assert ptokenize.VALID_TOKEN_RE.pattern == rt.VALID_TOKEN_PATTERN


SUFFIXES = sorted(
    {
        "sses", "ies", "ss", "s", "eed", "ed", "ing", "at", "bl", "iz", "y", "ational", "tional",
        "enci", "anci", "izer", "bli", "alli", "entli", "eli", "ousli", "ization", "ation",
        "ator", "alism", "iveness", "fulness", "ousness", "aliti", "iviti", "biliti", "fulli",
        "logi", "icate", "ative", "alize", "iciti", "ical", "ful", "ness", "al", "ance", "ence",
        "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "sion", "tion", "ou",
        "ism", "ate", "iti", "ous", "ive", "ize", "e", "ll", "ied", "ying",
    }
)  # fmt: skip
stem_words = st.builds(
    lambda stem, sufs: stem + "".join(sufs),
    st.text(st.sampled_from("aeiouybcdlnrstzgw1"), max_size=6),
    st.lists(st.sampled_from(SUFFIXES), max_size=3),
)


@settings(
    suppress_health_check=[HealthCheck.too_slow],
    max_examples=int(_EXAMPLES) if _EXAMPLES else 1000,
)
@given(word=st.one_of(stem_words, st.text(st.sampled_from("abcdeilnorstuyz0"), max_size=12)))
@example(word="")
@example(word="y" * 50)
@example(word="generalizations")
def test_porter_matches_nltk(porter: Any, word: str) -> None:
    assert native.rouge_stem(word) == porter.stem(word)


# ---------------------------------------------------------------------------
# Results, errors, surrogates
# ---------------------------------------------------------------------------


def test_score_class(ref: Any) -> None:
    rscoring = pytest.importorskip("rouge_score.scoring")
    s = rouge_scorer.RougeScorer(["rouge1"]).score("a b", "a")["rouge1"]
    assert repr(s) == "Score(precision=1.0, recall=0.5, fmeasure=0.6666666666666666)"
    assert s == rscoring.Score(1.0, 0.5, 2 / 3) == (1.0, 0.5, 2 / 3)
    assert s._fields == ("precision", "recall", "fmeasure")
    assert type(s).__module__ == "pyroid.metrics.scoring"
    # Round-trips our own in-process object (trusted data): it pickles by reference.
    assert pickle.loads(pickle.dumps(s)) == s
    assert s._replace(recall=1.0) == (1.0, 1.0, 2 / 3)
    zero = rouge_scorer.RougeScorer(["rougeL"]).score("", "a")["rougeL"]
    assert repr(zero) == "Score(precision=0, recall=0, fmeasure=0)"
    assert scoring.fmeasure(0, 0) == 0.0
    assert scoring.fmeasure(0.5, 1.0) == rscoring.fmeasure(0.5, 1.0)
    with pytest.raises(TypeError):
        scoring.BaseScorer()  # type: ignore[abstract]


def test_lone_surrogates_raise(ref: Any) -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    with pytest.raises(UnicodeEncodeError) as info:
        scorer.score("a \ud800 b", "a")
    assert any("lone surrogate" in n for n in getattr(info.value, "__notes__", []))
    with pytest.raises(UnicodeEncodeError):
        scorer.score("a", "\udfff")
    with pytest.raises(UnicodeEncodeError):
        scorer.score_batch(["a", "b\ud800"], ["a", "b"])
    with pytest.raises(UnicodeEncodeError):
        tokenizers.DefaultTokenizer().tokenize("\ud800")
    # A custom tokenizer may produce them: rouge-score scores those tokens.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        result = rouge_scorer.RougeScorer(["rouge1"], tokenizer=Chars()).score("a\ud800", "\ud800")
    assert result["rouge1"] == (1.0, 0.5, 2 / 3)


# ---------------------------------------------------------------------------
# BootstrapAggregator
# ---------------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="numpy's global random state")
def test_bootstrap_aggregator_matches(ref: Any) -> None:
    np = pytest.importorskip("numpy")
    rscoring = pytest.importorskip("rouge_score.scoring")
    targets, predictions = pairs(60, seed=9)
    types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    ours = rouge_scorer.RougeScorer(types, use_stemmer=True).score_batch(targets, predictions)
    reference = ref.RougeScorer(types, use_stemmer=True)
    theirs = [reference.score(t, p) for t, p in zip(targets, predictions, strict=True)]
    configs: list[dict[str, Any]] = [{}, {"confidence_interval": 0.9, "n_samples": 50}]
    for kwargs in configs:
        a = scoring.BootstrapAggregator(**kwargs)
        b = rscoring.BootstrapAggregator(**kwargs)
        for d in ours:
            a.add_scores(d)
        for d in theirs:
            b.add_scores(d)
        np.random.seed(123)
        got = a.aggregate()
        np.random.seed(123)
        want = b.aggregate()
        assert repr(got) == repr(want)
        assert type(got["rouge1"]) is scoring.AggregateScore
        assert type(got["rouge1"].mid) is scoring.Score


@pytest.mark.thread_unsafe(reason="numpy's global random state")
def test_bootstrap_aggregator_rouge_score_cases() -> None:
    np = pytest.importorskip("numpy")
    np.random.seed(123)
    for ci, want in [
        (0.9, ((1 / 3, 2 / 3, 3 / 3), (1 / 9, 4 / 9, 7 / 9), (1 / 6, 3 / 6, 5 / 6))),
        (0.0, ((2 / 3,) * 3, (4 / 9,) * 3, (3 / 6,) * 3)),
    ]:
        aggregator = scoring.BootstrapAggregator(confidence_interval=ci)
        aggregator.add_scores({"rouge1": scoring.Score(precision=1, recall=1 / 3, fmeasure=1 / 2)})
        aggregator.add_scores({"rouge1": scoring.Score(precision=0, recall=0, fmeasure=0)})
        aggregator.add_scores({"rouge1": scoring.Score(precision=1, recall=1, fmeasure=1)})
        result = aggregator.aggregate()["rouge1"]
        for measure, (low, mid, high) in zip(range(3), want, strict=True):
            assert result.low[measure] == pytest.approx(low, abs=1e-8)
            assert result.mid[measure] == pytest.approx(mid, abs=1e-8)
            assert result.high[measure] == pytest.approx(high, abs=1e-8)
    with pytest.raises(ValueError, match="confidence_interval"):
        scoring.BootstrapAggregator(confidence_interval=1.5)
    with pytest.raises(ValueError, match="n_samples"):
        scoring.BootstrapAggregator(n_samples=0)


@pytest.mark.thread_unsafe(reason="monkeypatches sys.modules")
def test_bootstrap_aggregator_without_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "numpy", None)
    aggregator = scoring.BootstrapAggregator()
    aggregator.add_scores({"rouge1": scoring.Score(1, 1, 1)})
    with pytest.raises(ImportError, match="pip install numpy"):
        aggregator.aggregate()


# ---------------------------------------------------------------------------
# Free-threading
# ---------------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="subprocess")
@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs a free-threaded build"
)
def test_gil_stays_disabled(run_python: Callable[..., Any]) -> None:
    proc = run_python(
        """
        import sys
        from pyroid.metrics import rouge_scorer, scoring, tokenizers
        s = rouge_scorer.RougeScorer(["rouge1", "rougeLsum"], use_stemmer=True)
        s.score("a b", "a")
        s.score_batch(["a b"] * 100, ["b"] * 100)
        print(sys._is_gil_enabled())
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"
