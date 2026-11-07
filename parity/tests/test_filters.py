"""datatrove quality filters: same keep/drop decision, same reason string, same rewritten text (C4), same exception.

The pinned decisions in ``parity/filters-ref.parquet`` come from datatrove 0.10.1 (English words via spaCy
``blank("en")``) on the dev slice and the adversarial set.
"""

from __future__ import annotations

from typing import Any

import pyarrow.parquet as pq
import pytest

from pyroid_parity import specs
from pyroid_parity.plugin import effective_limit, need, need_corpus, reference

REF = "parity/filters-ref.parquet"


def _ref_rows(case: str) -> dict[str, tuple[Any, ...]]:
    t = pq.read_table(need(REF), filters=[("case", "=", case)]).to_pydict()
    out = {}
    for i, o, k, r, txt in zip(t["id"], t["outcome"], t["keep"], t["reason"], t["text_out"], strict=True):
        out[i] = ("ok", k, r, txt) if o == "ok" else ("raise", r)
    return out


@pytest.fixture(scope="module", autouse=True)
def _ref_meta(parity):
    import json

    from pyroid_parity import data as D

    entry = D.load_manifest()["files"].get(REF, {})
    reference("filters", "datatrove", entry.get("oracle", {}).get("datatrove"), None)
    parity.uses("filters", REF, D.CORPORA["dev"], D.CORPORA["adversarial"])
    return json.dumps(entry.get("oracle", {}))


@pytest.mark.parametrize("corpus", ["dev", "adversarial"])
@pytest.mark.parametrize("case", list(specs.FILTER_CASES))
def test_filter_decisions(case, corpus, filters_impl, parity, request):
    name, _, config = specs.FILTER_CASES[case]
    docs = need_corpus(corpus, effective_limit(request.config, filters_impl))
    want = _ref_rows(case)
    got = filters_impl.obj.run(name, docs.texts, dict(config))
    assert len(got) == len(docs), f"{name} returned {len(got)} decisions for {len(docs)} docs"
    bad = []
    for doc_id, text, g in zip(docs.ids, docs.texts, got, strict=True):
        w = want[doc_id]
        same = tuple(g) == w
        if not same:
            bad.append(doc_id)
        parity.record("filters", f"{case}/{corpus}", same, case=doc_id, input=text, got=g, want=w)
    assert not bad, f"{len(bad)}/{len(docs)} decisions differ for {case} on {corpus}: first {bad[:5]}"


def test_reason_strings_are_datatrove_names():
    """Guard the reason vocabulary pyroid must emit (from the pinned decisions, not hard-coded)."""
    t = pq.read_table(need(REF), columns=["case", "reason", "outcome"]).to_pydict()
    reasons = {
        (c, r) for c, r, o in zip(t["case"], t["reason"], t["outcome"], strict=True) if o == "ok" and r
    }
    assert {r for c, r in reasons if c == "gopher_quality"} >= {
        "gopher_short_doc",
        "gopher_below_alpha_threshold",
    }
    assert all(c in specs.FILTER_CASES for c, _ in reasons)
