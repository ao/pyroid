"""fastText inference: identical top-k labels, probabilities within 1e-6, same exceptions.

Pinned outputs (``specs.lid_ref``: ``parity/fasttext-ref.parquet`` on linux-aarch64, one file per other platform)
come from fasttext-predict 0.9.2.4 on CPython 3.13 for lid.176.bin and lid.176.ftz, top-5 on
``text.replace("\\n", " ")`` (datatrove's LID preprocessing). The oracle's floats depend on the CPU it runs on, so each
platform compares against its own oracle's output. They let the suite run on interpreters the oracle doesn't install
on (3.14, 3.14t, 3.15t); the live-oracle Hypothesis checks run wherever ``fasttext`` (fasttext-predict) imports.
"""

from __future__ import annotations

import importlib
import math

import pyarrow.parquet as pq
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from pyroid_parity import data as D
from pyroid_parity import specs
from pyroid_parity.plugin import need, need_corpus, reference

REF = specs.lid_ref(D.platform_key())
ATOL = 1e-6
MODELS = list(specs.LID_MODELS)


@pytest.fixture(scope="module", autouse=True)
def _meta(parity):
    entry = D.load_manifest()["files"].get(REF, {})
    reference("fasttext", "fasttext-predict", entry.get("oracle", {}).get("fasttext-predict"), ATOL)
    parity.uses("fasttext", REF, *specs.LID_MODELS.values(), D.CORPORA["dev"], D.CORPORA["adversarial"])


@pytest.fixture(scope="module")
def candidate_models(fasttext_impl):
    return {m: fasttext_impl.obj.load(str(need(rel))) for m, rel in specs.LID_MODELS.items()}


@pytest.fixture(scope="module")
def oracle_models():
    try:
        ft = importlib.import_module("fasttext")
    except ImportError:
        pytest.skip("live oracle (fasttext-predict) is not installable on this interpreter")
    return {m: ft.load_model(str(need(rel))) for m, rel in specs.LID_MODELS.items()}


def close(got: list[tuple[str, float]], want: list[tuple[str, float]]) -> bool:
    return len(got) == len(want) and all(
        g[0] == w[0] and math.isclose(g[1], w[1], rel_tol=0, abs_tol=ATOL)
        for g, w in zip(got, want, strict=True)
    )


@pytest.mark.parametrize("corpus", ["dev", "adversarial"])
@pytest.mark.parametrize("model", MODELS)
def test_pinned_predictions(model, corpus, fasttext_impl, candidate_models, parity, parity_limit):
    docs = need_corpus(corpus, parity_limit)
    t = pq.read_table(need(REF), filters=[("model", "=", model)]).to_pydict()
    want = {
        i: list(zip(lbl, p, strict=True)) for i, lbl, p in zip(t["id"], t["labels"], t["probs"], strict=True)
    }
    m = candidate_models[model]
    bad, worst = [], 0.0
    for doc_id, text in docs:
        got = fasttext_impl.obj.predict(m, text.replace("\n", " "), specs.LID_TOPK)
        same = close(got, want[doc_id])
        if same:
            worst = max([worst, *(abs(g[1] - w[1]) for g, w in zip(got, want[doc_id], strict=True))])
        else:
            bad.append(doc_id)
        parity.record(
            "fasttext", f"{model}/{corpus}", same, case=doc_id, input=text[:200], got=got, want=want[doc_id]
        )
    assert not bad, f"{len(bad)}/{len(docs)} predictions differ beyond {ATOL}: first {bad[:5]}"
    print(f"{model}/{corpus}: max |dp| = {worst:.3g}")


def outcome(fn, *a):
    try:
        return ("ok", fn(*a))
    except Exception as e:  # noqa: BLE001  exception types are part of parity
        return ("raise", type(e).__name__)


one_line = st.text(st.characters(exclude_characters="\n", exclude_categories=("Cs",)), max_size=300)
with_surrogates = st.text(st.characters(exclude_characters="\n"), max_size=50)  # includes lone surrogates


@pytest.mark.parametrize("model", MODELS)
@given(text=st.one_of(one_line, with_surrogates), k=st.integers(1, 5))
@example(text="", k=1)
@example(text="   ", k=3)
@example(text="\udc00 lone surrogate", k=1)
@example(text="a\x00b", k=2)
def test_live_oracle(model, text, k, fasttext_impl, candidate_models, oracle_models, parity):
    ref_model = oracle_models[model]

    def oracle(t, kk):
        labels, probs = ref_model.predict(t, k=kk, threshold=0.0)
        return [(str(lbl), float(p)) for lbl, p in zip(labels, probs, strict=True)]

    got = outcome(fasttext_impl.obj.predict, candidate_models[model], text, k)
    want = outcome(oracle, text, k)
    same = got[0] == want[0] and (close(got[1], want[1]) if got[0] == "ok" else got[1] == want[1])
    parity.record("fasttext", f"hypothesis-{model}", same, case=None, input=text, got=got, want=want)
    assert same, (got, want)


@pytest.mark.parametrize("model", MODELS)
def test_newline_raises_like_fasttext(model, fasttext_impl, candidate_models, parity):
    """fastText's Python API rejects multi-line input with ValueError; datatrove relies on replacing "\\n"."""
    got = outcome(fasttext_impl.obj.predict, candidate_models[model], "line one\nline two", 1)
    parity.record(
        "fasttext",
        "newline-raises",
        got == ("raise", "ValueError"),
        input="line one\\nline two",
        got=got,
        want=("raise", "ValueError"),
    )
    assert got == ("raise", "ValueError")
