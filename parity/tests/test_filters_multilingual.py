"""datatrove quality filters per language: the same decisions, reasons and C4 text on every FineWeb-2 slice.

Each slice runs with its own datatrove language code (so its own word tokenizer), under FineWeb-2's pipeline
settings for that language (``specs.fineweb2_filter_cases``) and datatrove's defaults. The reference is the
pinned ``parity/filters-ml-ref.parquet`` if pinned, else datatrove live (see ``pyroid_parity.multilingual``).
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pyarrow.parquet as pq
import pytest

from pyroid_parity import data as D
from pyroid_parity import impl as I
from pyroid_parity import multilingual as ML
from pyroid_parity import specs
from pyroid_parity.plugin import effective_limit, need, reference

pytestmark = pytest.mark.filters

CASES = list(specs.fineweb2_filter_cases(specs.FINEWEB2_LANGUAGES[0]))


@cache
def _pinned() -> dict[tuple[str, str, str], tuple[Any, ...]] | None:
    if not D.available(ML.REFS["filters"]):
        return None
    t = pq.read_table(D.path(ML.REFS["filters"])).to_pydict()
    out = {}
    for lang, case, i, o, k, r, txt in zip(
        t["lang"], t["case"], t["id"], t["outcome"], t["keep"], t["reason"], t["text_out"], strict=True
    ):
        out[lang, case, i] = ("ok", k, r, txt) if o == "ok" else ("raise", r)
    return out


@cache
def _live() -> Any:
    return I.resolve("filters", "datatrove").obj


def _want(lang: str, case: str, docs: D.Corpus) -> list[tuple[Any, ...]]:
    pinned = _pinned()
    if pinned is not None and (lang, case, docs.ids[0]) in pinned:
        return [pinned[lang, case, i] for i in docs.ids]
    try:
        import datatrove  # noqa: F401
    except ImportError:
        pytest.skip(f"{ML.REFS['filters']} is not pinned and datatrove is not installed")
    if problem := ML.oracle_problem(lang):
        pytest.skip(problem)
    name, _, config = specs.fineweb2_filter_cases(lang)[case]
    return [tuple(d) for d in _live().run(name, docs.texts, dict(config))]


@pytest.fixture(scope="module", autouse=True)
def _ref_meta(parity):
    entry = D.load_manifest()["files"].get(ML.REFS["filters"], {})
    reference("filters", "datatrove", entry.get("oracle", {}).get("datatrove"), None)
    parity.uses("filters", *[ML.slice_rel(lang) for lang in specs.FINEWEB2_LANGUAGES])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("lang", specs.FINEWEB2_LANGUAGES)
def test_fineweb2_decisions(lang, case, filters_impl, parity, request):
    need(ML.slice_rel(lang))
    docs = ML.fineweb2(lang)
    limit = effective_limit(request.config, filters_impl)
    if limit:
        docs = docs.head(limit)
    want = _want(lang, case, docs)
    name, _, config = specs.fineweb2_filter_cases(lang)[case]
    got = filters_impl.obj.run(name, docs.texts, dict(config))
    assert len(got) == len(docs)
    bad = []
    for doc_id, text, g, w in zip(docs.ids, docs.texts, got, want, strict=True):
        same = tuple(g) == tuple(w)
        if not same:
            bad.append(doc_id)
        parity.record("filters", f"{case}/{lang}", same, case=doc_id, input=text, got=g, want=w)
    assert not bad, f"{len(bad)}/{len(docs)} decisions differ for {case} on {lang}: first {bad[:5]}"


def test_fineweb2_settings_are_fineweb2s():
    """Guard the transcription: FineWeb-2's German thresholds and first stop words (configs/deu_Latn.yml)."""
    c = specs.fineweb2_config("deu_Latn")
    assert c["dup_line_frac"] == 0.263 and c["max_avg_word_length"] == 15
    assert c["stopwords"][:4] == ["der", "und", "die", "in"]
