"""Parity of ``pyroid.fasttext`` with fastText on the lid.176 language-ID models.

Everything runs on ``lid.176.bin`` (hierarchical softmax, dim 16) and on the
quantized, pruned ``lid.176.ftz``:

1. **Pinned references.** ``parity/fasttext-ref.parquet`` holds fasttext-predict
   0.9.2.4's ``model.predict(text.replace("\\n", " "), k=5, threshold=0.0)`` for
   the 13,232 docs of the FineWeb dev slice and the adversarial set. pyroid must
   return the same labels with probabilities within 1e-6 through per-str calls,
   one batched list call and one Arrow call (``threads=None``), and the three
   paths must agree bit for bit. Needs no oracle, so it runs on 3.15t too.
2. **Live oracle**, wherever ``fasttext`` imports (fasttext-predict up to 3.13,
   fasttext-numpy2-wheel on 3.12). Hypothesis tries any Unicode, lone surrogates,
   empty and whitespace-only lines, lines of 100k+ chars, embedded newlines, NUL
   and the other separators, ``__label__`` tokens, ``</s>``, k in {1, 5, -1}
   (plus the invalid 0 and -2) and thresholds {0, 0.01, 0.5, 1, 2, nan}. The
   results must be the same labels with probabilities within 1e-6, or the same
   exception type and message. For TypeError only the type is compared: pyroid
   names the lone surrogate instead of pybind11's "incompatible function
   arguments". Against fasttext-predict, the reference build, labels match
   exactly. fasttext-numpy2-wheel rounds differently (up to 1.8e-7), so against
   it two labels may swap where their probabilities tie within 1e-6
   (``near_tie_only``).
3. **The rest of the inference API** against fasttext-numpy2-wheel, the build
   with upstream's whole Python API (fasttext-predict only has ``predict``).

Divergences found by this suite (newline-before-surrogate precedence, k outside
int32) are fixed; the tests below pin them.

The data lives on the shared ``/data`` volume (``$PYROID_DATA``) and is pinned
by sha256, as in the parity harness's ``bench/manifest.json``. Without the
models (e.g. in CI) the module is skipped.
"""

import hashlib
import importlib
import importlib.metadata as md
import logging
import math
import os
import re
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from hypothesis import currently_in_test_context, event, example, given
from hypothesis import strategies as st

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
MODELS = {"lid.176.bin": "models/lid.176.bin", "lid.176.ftz": "models/lid.176.ftz"}
CORPORA = {"dev": "fineweb/fineweb-dev-10k.parquet", "adversarial": "parity/adversarial.parquet"}
CORPUS_DOCS = {"dev": 10_000, "adversarial": 3_232}
REF = "parity/fasttext-ref.parquet"
# sha256 of every file read here (bench/manifest.json on the harness branch).
PINNED = {
    "models/lid.176.bin": "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e",
    "models/lid.176.ftz": "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83",
    "fineweb/fineweb-dev-10k.parquet": (
        "4d6081f32ffcd96e1ebfeabf943e1ec55a138fff96ff8937c83bf855600b5536"
    ),
    "parity/adversarial.parquet": (
        "f3cb139fc03150f7620a2ebd5066e5d873e79b7b3b3addbc40458f221b842448"
    ),
    "parity/fasttext-ref.parquet": (
        "209adb5e1f1243dd307a139e0b06580638d9309f22ff93cc88e43ec753085030"
    ),
}

if not all((DATA / rel).is_file() for rel in MODELS.values()):
    pytest.skip(
        f"no lid.176 models in {DATA / 'models'} (the shared data volume; set PYROID_DATA)",
        allow_module_level=True,
    )

# Imported after the skip: CI has no models, and may not have numpy either.
import numpy as np  # noqa: E402

import pyroid.fasttext as pft  # noqa: E402

ATOL = 1e-6
MODEL_NAMES = list(MODELS)
LOG = logging.getLogger(__name__)
NEWLINE_MESSAGE = "predict processes one line at a time (remove '\\n')"
GET_LINE_NEWLINE_MESSAGE = "get_line processes one line at a time (remove '\\n')"

# (labels, probabilities as Python floats)
Prediction = tuple[tuple[str, ...], tuple[float, ...]]
# ("ok", labels, probabilities) or ("raise", exception type name, message)
Outcome = tuple[str, Any, Any]


def _optional(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


PA = _optional("pyarrow")
# The distribution that provides `import fasttext`, if any. fasttext-predict built the
# pinned references (and its -v2 fork reproduces them bit for bit).
ORACLE_BUILD = next(iter(md.packages_distributions().get("fasttext", [])), None)
REFERENCE_BUILDS = {"fasttext-predict", "fasttext-predict-v2"}


@cache
def pinned(rel: str) -> Path:
    """The file under ``DATA``, after checking its sha256; skip if it is missing."""
    path = DATA / rel
    if not path.is_file():
        pytest.skip(f"{path} is missing (see parity/README.md on the harness branch)")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != PINNED[rel]:
        pytest.fail(f"{path}: sha256 {digest} is not the pinned {PINNED[rel]}")
    return path


class Corpus(NamedTuple):
    ids: list[str]
    lines: list[str]  # text.replace("\n", " "), the call the references were made with


class Agreement(NamedTuple):
    n: int
    label_mismatches: int
    bit_exact: int
    max_dp: float  # over the docs whose labels match


def agreement(got: Sequence[Prediction], want: Sequence[Prediction]) -> Agreement:
    mismatches = bit_exact = 0
    max_dp = 0.0
    for (got_labels, got_probs), (want_labels, want_probs) in zip(got, want, strict=True):
        if got_labels != want_labels:
            mismatches += 1
            continue
        bit_exact += got_probs == want_probs
        max_dp = max([max_dp, *(abs(g - w) for g, w in zip(got_probs, want_probs, strict=True))])
    return Agreement(len(got), mismatches, bit_exact, max_dp)


def first_difference(a: Sequence[object], b: Sequence[object]) -> int | None:
    assert len(a) == len(b)
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)


def batch_predictions(result: tuple[Any, Any]) -> list[Prediction | None]:
    """``predict(batch)`` as Predictions; checks the documented types (``None`` = Arrow null)."""
    labels, probs = result
    assert isinstance(labels, list)
    assert isinstance(probs, list)
    assert len(labels) == len(probs)
    out: list[Prediction | None] = []
    for lbl, p in zip(labels, probs, strict=True):
        if lbl is None:
            assert p is None
            out.append(None)
            continue
        assert isinstance(lbl, list)
        assert isinstance(p, np.ndarray)
        assert p.dtype == np.float32
        assert p.ndim == 1
        out.append((tuple(lbl), tuple(p.tolist())))
    return out


def outcome(predict: Callable[..., Any], text: Any, k: int, threshold: float) -> Outcome:
    try:
        labels, probs = predict(text, k=k, threshold=threshold)
    except Exception as e:  # exception types and messages are part of parity
        return ("raise", type(e).__name__, str(e))
    return ("ok", tuple(labels), tuple(float(p) for p in probs))


def _event(value: str) -> None:
    """A Hypothesis event (``--hypothesis-show-statistics``), outside tests a no-op."""
    if currently_in_test_context():
        event(value)


def _tally(want: Outcome) -> None:
    """Count what the oracle did."""
    if want[0] == "ok":
        n = len(want[1])
        _event(f"oracle: {'0' if n == 0 else '1-5' if n <= 5 else '6+'} labels")
    else:
        _event(f"oracle: {want[1]}" + ("" if want[1] == "TypeError" else f" {want[2]!r}"))


def near_tie_only(
    got: Outcome, want: Outcome, full_got: Outcome, full_want: Outcome, k: int, threshold: float
) -> str | None:
    """Why two label lists differ by more than near-ties, or None if they don't.

    ``full_*`` are both sides' probabilities of every label (k=-1, threshold=-1, which
    turns off the hierarchical-softmax pruning). Each side's list must be exactly what
    its own distribution gives: sorted, the same probabilities, and every label it
    leaves out at or below its k-th probability or its pruning cut-off (threshold +
    1e-5). With both distributions within ATOL of each other, the lists can then only
    differ where probabilities tie within ATOL.
    """
    if full_got[0] != "ok" or full_want[0] != "ok":
        return f"the full distributions: {full_got} / {full_want}"
    sides = {
        "pyroid": (got, dict(zip(*full_got[1:], strict=True))),
        "oracle": (want, dict(zip(*full_want[1:], strict=True))),
    }
    p_got, p_want = sides["pyroid"][1], sides["oracle"][1]
    if p_got.keys() != p_want.keys():
        return "the full distributions have different labels"
    if (worst := max(abs(p_got[x] - p_want[x]) for x in p_got)) > ATOL:
        return f"the full distributions differ by {worst:.3g}"
    for name, ((_, labels, probs), p) in sides.items():
        if list(probs) != sorted(probs, reverse=True):
            return f"{name}: not sorted by probability"
        if any(p[x] != q for x, q in zip(labels, probs, strict=True)):
            return f"{name}: the list disagrees with its own full distribution"
        if 0 < k == len(labels):
            cutoff = probs[-1]  # the k-th; a tie may go either way
        elif threshold >= 0:  # (False for nan) hierarchical softmax prunes below this
            cutoff = (threshold + 1e-5) * (1 + 1e-4)  # slack: partial path scores
        else:
            cutoff = -math.inf
        if left_out := [x for x in p.keys() - set(labels) if p[x] > cutoff]:
            return f"{name} leaves out {left_out}, above its cut-off {cutoff:.3g}"
    if any(abs(g - w) > ATOL for g, w in zip(got[2], want[2], strict=False)):
        return "the sorted probabilities differ by more than ATOL"
    return None


def check_predict(m: pft.Model, oracle: Any, text: Any, k: int, threshold: float) -> Outcome:
    """``predict(text, k, threshold)``: pyroid against the oracle; returns pyroid's outcome.

    Against fasttext-predict (the reference build) labels must match exactly. Another
    build (fasttext-numpy2-wheel) rounds differently, by up to 1.8e-7 on the pinned docs,
    which can reorder labels that tie within that; there the lists may differ at
    near-ties only (see near_tie_only).
    """
    got = outcome(m.predict, text, k, threshold)
    want = outcome(oracle.predict, text, k, threshold)
    _tally(want)
    if got[0] != "ok" or want[0] != "ok":
        assert got[:2] == want[:2], f"{got} != oracle {want}"
        if want[1] == "TypeError":  # pybind11: "predict(): incompatible function arguments ..."
            assert "lone surrogate" in got[2]
        else:
            assert got[2] == want[2]
        return got
    if got[1] != want[1]:
        assert ORACLE_BUILD not in REFERENCE_BUILDS, f"labels {got[1]} != oracle {want[1]}"
        full_got = outcome(m.predict, text, -1, -1.0)
        full_want = outcome(oracle.predict, text, -1, -1.0)
        why = near_tie_only(got, want, full_got, full_want, k, threshold)
        assert why is None, f"labels {got[1]} != oracle {want[1]} ({ORACLE_BUILD}): {why}"
        _event(f"labels differ at a near-tie ({ORACLE_BUILD})")
        return got
    dp = max((abs(g - w) for g, w in zip(got[2], want[2], strict=True)), default=0.0)
    assert dp <= ATOL, f"|dp| = {dp:.3g} > {ATOL}: {got[2]} != oracle {want[2]}"
    return got


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def models() -> dict[str, pft.Model]:
    return {name: pft.load_model(pinned(rel)) for name, rel in MODELS.items()}


@pytest.fixture(scope="module")
def corpora() -> dict[str, Corpus]:
    pq = pytest.importorskip("pyarrow.parquet")
    out = {}
    for name, rel in CORPORA.items():
        table = pq.read_table(pinned(rel), columns=["id", "text"])
        ids: list[str] = table.column("id").to_pylist()
        texts: list[str] = table.column("text").to_pylist()
        assert len(ids) == len(set(ids)) == CORPUS_DOCS[name]
        out[name] = Corpus(ids, [t.replace("\n", " ") for t in texts])
    return out


@pytest.fixture(scope="module")
def refs() -> dict[str, dict[str, Prediction]]:
    pq = pytest.importorskip("pyarrow.parquet")
    t = pq.read_table(pinned(REF)).to_pydict()
    out: dict[str, dict[str, Prediction]] = {m: {} for m in MODELS}
    for m, doc_id, labels, probs in zip(t["model"], t["id"], t["labels"], t["probs"], strict=True):
        out[m][doc_id] = (tuple(labels), tuple(probs))
    assert {m: len(v) for m, v in out.items()} == {m: sum(CORPUS_DOCS.values()) for m in MODELS}
    return out


@pytest.fixture(scope="module")
def oracles() -> dict[str, Any]:
    """The installed fastText build's models (fasttext-predict or fasttext-numpy2-wheel)."""
    ft = _optional("fasttext")
    if ft is None or not hasattr(ft, "load_model"):
        pytest.skip("no live oracle: no `fasttext` (fasttext-predict / fasttext-numpy2-wheel) here")
    return {name: ft.load_model(str(pinned(rel))) for name, rel in MODELS.items()}


@pytest.fixture(scope="module")
def full_oracles(oracles: dict[str, Any]) -> dict[str, Any]:
    """The oracle models, if the build has upstream's whole API (fasttext-numpy2-wheel)."""
    if not all(hasattr(o, "get_sentence_vector") for o in oracles.values()):
        pytest.skip("needs fasttext-numpy2-wheel: this `fasttext` only has load_model and predict")
    return oracles


# ---------------------------------------------------------------- 1. pinned references


@pytest.mark.thread_unsafe(reason="heavy: 13k docs through three prediction paths")
@pytest.mark.parametrize("corpus", list(CORPORA))
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_pinned_references(
    model: str,
    corpus: str,
    models: dict[str, pft.Model],
    corpora: dict[str, Corpus],
    refs: dict[str, dict[str, Prediction]],
) -> None:
    pa = pytest.importorskip("pyarrow")
    m = models[model]
    ids, lines = corpora[corpus]
    want = [refs[model][doc_id] for doc_id in ids]

    per_str: list[Prediction | None] = []
    for line in lines:
        labels, probs = m.predict(line, k=5, threshold=0.0)
        assert isinstance(labels, tuple)
        assert probs.dtype == np.float64
        per_str.append((labels, tuple(probs.tolist())))
    batched = batch_predictions(m.predict(lines, k=5, threshold=0.0))
    arrow = batch_predictions(
        m.predict(pa.array(lines, type=pa.string()), k=5, threshold=0.0, threads=None)
    )
    # Bit for bit: float32 probabilities widen exactly to the per-str float64 ones.
    assert first_difference(batched, per_str) is None, "list batch != per-str calls"
    assert first_difference(arrow, per_str) is None, "Arrow batch != per-str calls"

    got = [p for p in per_str if p is not None]
    stats = agreement(got, want)
    # Shown with --log-cli-level=INFO.
    LOG.info("%s/%s vs pinned refs: %s", model, corpus, stats)
    bad = [doc_id for doc_id, g, w in zip(ids, got, want, strict=True) if g[0] != w[0]]
    assert stats.label_mismatches == 0, f"{stats}; first ids: {bad[:5]}"
    assert stats.max_dp <= ATOL, stats


# ---------------------------------------------------------------- 2. live oracle: predict

# fastText's word separators (Dictionary::readWord); "\n" ends the line, and predict rejects it.
SEPARATORS = (" ", "\t", "\v", "\f", "\r", "\x00")
# Whitespace to Python, ordinary characters to fastText.
NOT_SEPARATORS = ("\xa0", "\x85", "\x1c", "\u2028", "\u3000", "\u200b", "\ufeff")
TOKENS = (
    "</s>",
    "<",
    ">",
    "__label__",
    "__label__en",
    "__label__fr",
    "__label__zz",
    "__label__en__label__fr",
    "the",
    "hello",
    "Bonjour",
    "\u041f\u0440\u0438\u0432\u0435\u0442",  # Russian
    "\u65e5\u672c\u8a9e",  # Japanese
    "\u0645\u0631\u062d\u0628\u0627",  # Arabic
    "\u0928\u092e\u0938\u094d\u0924\u0947",  # Hindi
    "e\u0301",
    "\U0001f600",
    "a" * 300,
)
KS = (1, 5, -1)
BAD_KS = (0, -2)
THRESHOLDS = (0.0, 0.01, 0.5, 1.0, 2.0, math.nan)

_char = st.characters(exclude_characters="\n", exclude_categories=["Cs"])
_piece = st.one_of(
    st.text(_char, max_size=20), st.sampled_from(SEPARATORS + NOT_SEPARATORS + TOKENS)
)
valid_lines = st.lists(_piece, max_size=30).map("".join)


def _insert(line: str, extra: list[tuple[int, str]]) -> str:
    for pos, ch in extra:
        i = pos % (len(line) + 1)
        line = line[:i] + ch + line[i:]
    return line


def _tile(unit: str, size: int) -> str:
    return (unit * (size // len(unit) + 1))[:size]


# A line holds lone surrogates or newlines, never both: which error wins when both are
# present is a known divergence (test_newline_is_rejected_before_surrogates).
lines = st.one_of(
    valid_lines,
    st.builds(
        _insert,
        valid_lines,
        st.lists(
            st.tuples(st.integers(0), st.characters(categories=["Cs"])), min_size=1, max_size=3
        ),
    ),
    st.builds(
        _insert,
        valid_lines,
        st.lists(st.tuples(st.integers(0), st.just("\n")), min_size=1, max_size=3),
    ),
)
_nonempty_piece = st.one_of(
    st.text(_char, min_size=1, max_size=20), st.sampled_from(SEPARATORS + NOT_SEPARATORS + TOKENS)
)
long_lines = st.builds(
    _tile,
    st.lists(_nonempty_piece, min_size=1, max_size=6).map("".join),
    st.integers(100_000, 200_000),
)


@pytest.mark.parametrize("model", MODEL_NAMES)
@given(text=lines, k=st.sampled_from(KS + BAD_KS), threshold=st.sampled_from(THRESHOLDS))
@example(text="", k=1, threshold=0.0)
@example(text="   ", k=5, threshold=0.0)
@example(text=" \t\v\f\r\x00", k=-1, threshold=0.0)
@example(text="a\x00b", k=5, threshold=0.0)
@example(text="line one\nline two", k=1, threshold=0.0)
@example(text="\udc00 lone surrogate", k=1, threshold=0.0)
@example(text="\ud83d\ude00 surrogate pair as two code points", k=5, threshold=0.0)
@example(text="__label__en __label__fr", k=5, threshold=0.0)
@example(text="</s> </s>", k=5, threshold=0.0)
@example(text="hello", k=-1, threshold=0.0)
@example(text="hello", k=-1, threshold=math.nan)
@example(text="hello", k=5, threshold=1.0)
@example(text="hello", k=5, threshold=2.0)
@example(text="hello", k=0, threshold=0.0)
@example(text="hello", k=-2, threshold=0.0)
@example(text="a\nb", k=0, threshold=0.0)
@example(text="\udc00", k=0, threshold=0.0)
# Found by the ci profile: fasttext-numpy2-wheel ties os with kk (pyroid and fasttext-predict
# put os 4e-11 lower); a near-tie, accepted against that build only.
@example(
    text="\xe6\xf7#\xbc|\x85\ufeff hello\x10\xf6\U0001f600\x1ce\u0301=\x10p\x08the",
    k=-1,
    threshold=0.0,
)
def test_predict_vs_live_oracle(
    model: str,
    text: str,
    k: int,
    threshold: float,
    models: dict[str, pft.Model],
    oracles: dict[str, Any],
) -> None:
    check_predict(models[model], oracles[model], text, k, threshold)


@pytest.mark.parametrize("model", MODEL_NAMES)
@given(text=long_lines, k=st.sampled_from(KS), threshold=st.sampled_from(THRESHOLDS))
@example(text="x" * 100_000, k=5, threshold=0.0)
@example(text="the " * 25_000, k=-1, threshold=0.0)
@example(text="\x00" * 100_000, k=5, threshold=0.0)
def test_long_lines_vs_live_oracle(
    model: str,
    text: str,
    k: int,
    threshold: float,
    models: dict[str, pft.Model],
    oracles: dict[str, Any],
) -> None:
    assert len(text) >= 100_000
    check_predict(models[model], oracles[model], text, k, threshold)


@pytest.mark.parametrize("model", MODEL_NAMES)
@given(
    texts=st.lists(valid_lines, max_size=100),  # crosses the 64-doc pool threshold
    k=st.sampled_from(KS),
    threshold=st.sampled_from(THRESHOLDS),
)
def test_batches_vs_live_oracle(
    model: str,
    texts: list[str],
    k: int,
    threshold: float,
    models: dict[str, pft.Model],
    oracles: dict[str, Any],
) -> None:
    m, oracle = models[model], oracles[model]
    per_str = [check_predict(m, oracle, t, k, threshold) for t in texts]
    expected = [(o[1], o[2]) for o in per_str]
    assert batch_predictions(m.predict(texts, k=k, threshold=threshold)) == expected
    assert batch_predictions(m.predict(tuple(texts), k=k, threshold=threshold)) == expected
    if PA is not None:
        col = PA.array([*texts, None], type=PA.string())
        assert batch_predictions(m.predict(col, k=k, threshold=threshold)) == [*expected, None]


@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize(
    ("texts", "k"),
    [
        pytest.param(["ok", "a\nb"], 1, id="newline"),
        pytest.param(["ok", "\udc00"], 1, id="surrogate"),
        pytest.param(["ok", "fine"], 0, id="k0"),
        pytest.param(["a\nb", "ok"], 0, id="newline-k0"),
        pytest.param(["ok", "\udc00"], -2, id="surrogate-k-2"),
    ],
)
def test_batch_errors_vs_live_oracle(
    model: str, texts: list[str], k: int, models: dict[str, pft.Model], oracles: dict[str, Any]
) -> None:
    """Errors of a list match fastText's list path (which raises before predicting)."""
    assert check_predict(models[model], oracles[model], texts, k, 0.0)[0] == "raise"


# Which error fastText raises when a line has both a "\n" and a lone surrogate: its
# Python layer looks for "\n" before pybind11 encodes the str.
NEWLINE_FIRST = [
    pytest.param("predict", "a\n\udc00", NEWLINE_MESSAGE, id="predict-str"),
    pytest.param("predict", ["\udc00", "a\nb"], NEWLINE_MESSAGE, id="predict-list"),
    pytest.param("get_sentence_vector", "\udc00\n", NEWLINE_MESSAGE, id="sentence-vector"),
    pytest.param("get_line", "a\n\udc00", GET_LINE_NEWLINE_MESSAGE, id="get-line-str"),
    pytest.param("get_line", ["\udc00", "a\nb"], GET_LINE_NEWLINE_MESSAGE, id="get-line-list"),
]


@pytest.mark.parametrize(("method", "text", "message"), NEWLINE_FIRST)
def test_oracle_rejects_newline_before_surrogates(
    method: str, text: Any, message: str, oracles: dict[str, Any]
) -> None:
    """Pins the oracle behaviour that the pyroid test below checks."""
    oracle = oracles["lid.176.bin"]
    if not hasattr(oracle, method):
        pytest.skip(f"this fasttext build has no {method}")
    with pytest.raises(ValueError, match=re.escape(message)):
        getattr(oracle, method)(text)


@pytest.mark.parametrize(("method", "text", "message"), NEWLINE_FIRST)
def test_newline_is_rejected_before_surrogates(
    method: str, text: Any, message: str, models: dict[str, pft.Model]
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        getattr(models["lid.176.bin"], method)(text)


K_OUTSIDE_INT32 = [2**31, 2**40, -(2**31) - 1]


@pytest.mark.parametrize("k", K_OUTSIDE_INT32)
def test_oracle_rejects_k_outside_int32(k: int, oracles: dict[str, Any]) -> None:
    """Pins the oracle behaviour that the pyroid test below checks."""
    with pytest.raises(TypeError):
        oracles["lid.176.bin"].predict("hello", k=k)


@pytest.mark.parametrize("k", K_OUTSIDE_INT32)
def test_k_outside_int32_is_a_type_error(k: int, models: dict[str, pft.Model]) -> None:
    with pytest.raises(TypeError):
        models["lid.176.bin"].predict("hello", k=k)


# ---------------------------------------------------------------- 3. metadata, full API


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_metadata(model: str, models: dict[str, pft.Model]) -> None:
    m = models[model]
    labels = m.get_labels()
    assert len(labels) == len(set(labels)) == 176
    assert all(lbl.startswith("__label__") for lbl in labels)
    assert m.labels == labels
    assert m.get_dimension() == 16
    assert m.is_quantized() is model.endswith(".ftz")


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_metadata_vs_live_oracle(
    model: str, models: dict[str, pft.Model], oracles: dict[str, Any]
) -> None:
    m, oracle = models[model], oracles[model]
    # Every build has predict: a negative threshold disables the hierarchical-softmax
    # pruning (log(threshold + 1e-5) is NaN), so k=-1 returns every label.
    labels, _ = oracle.predict("", k=-1, threshold=-1.0)
    assert sorted(labels) == sorted(m.get_labels())
    assert m.predict("", k=-1, threshold=-1.0)[0] == tuple(labels)
    for name in ("get_dimension", "is_quantized", "get_labels"):
        if hasattr(oracle, name):
            assert getattr(m, name)() == getattr(oracle, name)(), name


def _assert_vector(got: Any, want: Any) -> None:
    assert isinstance(got, np.ndarray)
    assert got.dtype == want.dtype == np.float32
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=0, atol=ATOL)


@pytest.mark.thread_unsafe(reason="heavy memory: copies of the 2M x 16 input matrix")
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_dictionary_and_matrices_vs_full_api(
    model: str, models: dict[str, pft.Model], full_oracles: dict[str, Any]
) -> None:
    m, oracle = models[model], full_oracles[model]
    assert m.get_dimension() == oracle.get_dimension()
    assert m.is_quantized() == oracle.is_quantized()
    for name in ("get_words", "get_labels"):
        assert getattr(m, name)() == getattr(oracle, name)(), name
        got, got_freq = getattr(m, name)(include_freq=True)
        want, want_freq = getattr(oracle, name)(include_freq=True)
        assert got == want, name
        assert got_freq.dtype == want_freq.dtype == np.int64
        np.testing.assert_array_equal(got_freq, want_freq)
    assert m.words == oracle.words
    assert m.labels == oracle.labels

    rows = list(range(0, len(m.words), 997))
    for name in ("get_input_matrix", "get_output_matrix"):
        if m.is_quantized():
            with pytest.raises(ValueError, match=r"^Can't get quantized Matrix$"):
                getattr(m, name)()
            with pytest.raises(ValueError, match=r"^Can't get quantized Matrix$"):
                getattr(oracle, name)()
            continue
        got_matrix, want_matrix = getattr(m, name)(), getattr(oracle, name)()
        assert got_matrix.dtype == want_matrix.dtype == np.float32
        assert got_matrix.shape == want_matrix.shape
        np.testing.assert_array_equal(got_matrix, want_matrix)
        if name == "get_input_matrix":
            rows.append(got_matrix.shape[0] - 1)
        del got_matrix, want_matrix
    # The oracle has no bounds check here (it crashes): in-range rows only.
    for i in rows:
        _assert_vector(m.get_input_vector(i), oracle.get_input_vector(i))


SAMPLE_WORDS = (
    "",
    " ",
    "</s>",
    "<",
    ">",
    "<he",
    "the",
    "The",
    "hello",
    "__label__en",
    "__label__zz",
    "a b",
    "a\tb",
    "a\nb",
    "\x00",
    "e\u0301",
    "\u00e9",
    "\U0001f600",
    "\u65e5\u672c\u8a9e",
    "x" * 5000,
)


def _assert_same_word(m: pft.Model, oracle: Any, word: str) -> None:
    _assert_vector(m.get_word_vector(word), oracle.get_word_vector(word))
    _assert_vector(m[word], oracle[word])
    got_sub, got_ids = m.get_subwords(word)
    want_sub, want_ids = oracle.get_subwords(word)
    assert got_sub == want_sub
    # Upstream's np.array([]) is float64 when every n-gram of the word was pruned (.ftz).
    assert got_ids.dtype == np.int64
    assert want_ids.dtype == np.int64 or want_ids.size == 0
    np.testing.assert_array_equal(got_ids, want_ids)
    assert m.get_word_id(word) == oracle.get_word_id(word)
    assert m.get_label_id(word) == oracle.get_label_id(word)
    assert m.get_subword_id(word) == oracle.get_subword_id(word)
    assert (word in m) is (word in oracle)


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_words_vs_full_api(
    model: str, models: dict[str, pft.Model], full_oracles: dict[str, Any]
) -> None:
    m, oracle = models[model], full_oracles[model]
    for word in [*SAMPLE_WORDS, *m.words[::97], *m.labels]:
        _assert_same_word(m, oracle, word)


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_sentences_vs_full_api(
    model: str,
    models: dict[str, pft.Model],
    full_oracles: dict[str, Any],
    corpora: dict[str, Corpus],
) -> None:
    m, oracle = models[model], full_oracles[model]
    sample = [*corpora["dev"].lines[:500], *corpora["adversarial"].lines]
    vectors = []
    for line in sample:
        got = m.get_sentence_vector(line)
        _assert_vector(got, oracle.get_sentence_vector(line))
        vectors.append(got)
        assert tuple(m.get_line(line)) == tuple(oracle.get_line(line))
    batch = m.get_sentence_vector(sample)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch, np.stack(vectors))
    assert tuple(m.get_line(sample)) == tuple(oracle.get_line(sample))


def _call(fn: Callable[[str], Any], text: str) -> tuple[str, Any]:
    try:
        return ("ok", fn(text))
    except Exception as e:  # exception types and messages are part of parity
        return ("raise", (type(e).__name__, str(e)))


def _assert_same_call(got: tuple[str, Any], want: tuple[str, Any]) -> None:
    assert got[0] == want[0], (got, want)
    if got[0] == "ok":
        return
    assert got[1][0] == want[1][0], (got, want)
    if want[1][0] == "TypeError":  # pybind11's "incompatible function arguments"
        assert "lone surrogate" in got[1][1], got
    else:
        assert got[1][1] == want[1][1], (got, want)


@pytest.mark.parametrize("model", MODEL_NAMES)
@given(text=lines)
@example(text="")
@example(text="a\nb")
@example(text="\udc00")
@example(text="__label__en hello </s>\x00world\t__label__fr")
def test_text_api_vs_full_api(
    model: str, text: str, models: dict[str, pft.Model], full_oracles: dict[str, Any]
) -> None:
    m, oracle = models[model], full_oracles[model]
    got, want = _call(m.get_sentence_vector, text), _call(oracle.get_sentence_vector, text)
    _assert_same_call(got, want)
    if got[0] == "ok":
        _assert_vector(got[1], want[1])
    got, want = _call(m.get_line, text), _call(oracle.get_line, text)
    _assert_same_call(got, want)
    if got[0] == "ok":
        assert tuple(got[1]) == tuple(want[1])
    for word in [text, *text.split(" ")[:20]]:
        got = _call(m.get_word_vector, word)
        _assert_same_call(got, _call(oracle.get_word_vector, word))
        if got[0] == "ok":
            _assert_same_word(m, oracle, word)
