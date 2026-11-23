"""MinHash per language: identical datatrove signatures and removed sets on every FineWeb-2 slice.

FineWeb-2's own MinhashConfig (xxhash/64, 14 x 8, 5-grams) with each slice's datatrove language code, so
shingles come from that language's word tokenizer. Every fifth document gets a near-duplicate (a copy with
about one character in 300 deleted) so the removed set is not empty. The reference is the pinned
``parity/minhash-ml-ref.parquet`` if pinned, else datatrove live (see ``pyroid_parity.multilingual``).
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

pytestmark = pytest.mark.dedup


@cache
def _pinned() -> dict[str, dict[str, Any]] | None:
    if not D.available(ML.REFS["dedup"]):
        return None
    t = pq.read_table(D.path(ML.REFS["dedup"])).to_pydict()
    out: dict[str, dict[str, Any]] = {}
    for lang, i, s, r in zip(t["lang"], t["id"], t["signature"], t["removed"], strict=True):
        out.setdefault(lang, {})[i] = (s, r)
    return out


def _reference(lang: str, docs: D.Corpus) -> tuple[list[Any], list[bool]]:
    pinned = (_pinned() or {}).get(lang)
    if pinned is not None:
        return [pinned[i][0] for i in docs.ids], [pinned[i][1] for i in docs.ids]
    try:
        import datatrove  # noqa: F401
    except ImportError:
        pytest.skip(f"{ML.REFS['dedup']} is not pinned and datatrove is not installed")
    if problem := ML.oracle_problem(lang):
        pytest.skip(problem)
    return _live(lang)


@cache
def _live(lang: str) -> tuple[list[Any], list[bool]]:
    ref = I.resolve("dedup", "datatrove").obj
    docs = ML.with_near_duplicates(lang)
    cfg = ML.minhash_config(lang)
    return ref.signatures(docs.texts, cfg), ref.duplicates(docs.texts, cfg)


@pytest.fixture(scope="module", autouse=True)
def _ref_meta(parity):
    entry = D.load_manifest()["files"].get(ML.REFS["dedup"], {})
    reference("dedup", "datatrove", entry.get("oracle", {}).get("datatrove"), None)
    parity.uses("dedup", *[ML.slice_rel(lang) for lang in specs.FINEWEB2_LANGUAGES])


@pytest.mark.parametrize("lang", specs.FINEWEB2_LANGUAGES)
def test_fineweb2_signatures(lang, dedup_impl, parity, request):
    need(ML.slice_rel(lang))
    docs = ML.with_near_duplicates(lang)
    want, _ = _reference(lang, docs)
    n = effective_limit(request.config, dedup_impl) or len(docs)
    got = dedup_impl.obj.signatures(docs.texts[:n], ML.minhash_config(lang))
    bad = []
    for i in range(n):
        same = got[i] == want[i]
        if not same:
            bad.append(docs.ids[i])
        parity.record(
            "dedup",
            f"signatures/{lang}",
            same,
            case=docs.ids[i],
            input=docs.texts[i],
            got=None if got[i] is None else got[i][:4],
            want=None if want[i] is None else want[i][:4],
        )
    assert not bad, f"{len(bad)}/{n} {lang} signatures differ: first {bad[:5]}"


@pytest.mark.parametrize("lang", specs.FINEWEB2_LANGUAGES)
def test_fineweb2_removed_set(lang, dedup_impl, parity, request):
    if effective_limit(request.config, dedup_impl):
        parity.skip("dedup", f"removed/{lang}", "set-level check needs the full slice (--parity-limit 0)")
        pytest.skip("removed-set parity needs the full slice: run with --parity-limit 0")
    need(ML.slice_rel(lang))
    docs = ML.with_near_duplicates(lang)
    _, want = _reference(lang, docs)
    got = dedup_impl.obj.duplicates(docs.texts, ML.minhash_config(lang))
    bad = [docs.ids[i] for i, (g, w) in enumerate(zip(got, want, strict=True)) if g != w]
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        parity.record("dedup", f"removed/{lang}", g == w, case=docs.ids[i], input=docs.meta[i], got=g, want=w)
    assert not bad, f"{len(bad)} {lang} docs differ in removed status (ref removes {sum(want)}): {bad[:5]}"
    assert sum(want) >= len(docs) // (ML.NEAR_DUP_EVERY + 1) // 2, "the near-duplicates should be found"
