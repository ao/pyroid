"""MinHash dedup: identical datatrove signatures (FineWeb config) and the identical removed-document set.

``parity/minhash-ref.parquet`` = datatrove 0.10.1 ``MinhashDedupSignature`` maths per doc plus the real
Signature -> Buckets -> Cluster executor chain over dev (10k) + adversarial (incl. injected near-duplicates).
The removed set is a property of the whole corpus, so that check only runs on the full corpus.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from pyroid_parity import data as D
from pyroid_parity import specs
from pyroid_parity.plugin import effective_limit, need, need_corpus, reference

REF = "parity/minhash-ref.parquet"


@pytest.fixture(scope="module")
def ref(parity):
    t = pq.read_table(need(REF)).to_pydict()
    entry = D.load_manifest()["files"].get(REF, {})
    reference("dedup", "datatrove", entry.get("oracle", {}).get("datatrove"), None)
    parity.uses("dedup", REF, D.CORPORA["dev"], D.CORPORA["adversarial"])
    return t


@pytest.fixture(scope="module")
def corpus():
    dev, adv = need_corpus("dev"), need_corpus("adversarial")
    return D.Corpus("dev+adversarial", dev.ids + adv.ids, dev.texts + adv.texts, dev.meta + adv.meta)


def test_corpus_order_matches_reference(ref, corpus):
    assert ref["id"] == corpus.ids


def test_signatures(ref, corpus, dedup_impl, parity, request):
    n = effective_limit(request.config, dedup_impl) or len(corpus)
    got = dedup_impl.obj.signatures(corpus.texts[:n], dict(specs.MINHASH_CONFIG))
    assert len(got) == n
    bad = []
    for i in range(n):
        same = got[i] == ref["signature"][i]
        if not same:
            bad.append(corpus.ids[i])
        parity.record(
            "dedup",
            "signatures",
            same,
            case=corpus.ids[i],
            input=corpus.texts[i],
            got=None if got[i] is None else got[i][:4],
            want=None if ref["signature"][i] is None else ref["signature"][i][:4],
        )
    assert not bad, f"{len(bad)}/{n} signatures differ: first {bad[:5]}"


def test_removed_set(ref, corpus, dedup_impl, parity, request):
    if effective_limit(request.config, dedup_impl):
        parity.skip("dedup", "removed", "set-level check needs the full corpus (run with --parity-limit 0)")
        pytest.skip("removed-set parity needs the full corpus: run with --parity-limit 0")
    got = dedup_impl.obj.duplicates(corpus.texts, dict(specs.MINHASH_CONFIG))
    assert len(got) == len(corpus)
    bad = []
    for i, (g, w) in enumerate(zip(got, ref["removed"], strict=True)):
        if g != w:
            bad.append(corpus.ids[i])
        parity.record("dedup", "removed", g == w, case=corpus.ids[i], input=corpus.meta[i], got=g, want=w)
    assert not bad, f"{len(bad)} docs differ in removed status (ref removes {sum(ref['removed'])}): {bad[:5]}"


def test_reference_has_near_duplicates_to_find(ref, corpus):
    """The gate is meaningless if nothing is removed: FineWeb is already deduplicated, so the injected
    near-duplicates in the adversarial set must account for removals."""
    removed = {i for i, r in zip(ref["id"], ref["removed"], strict=True) if r}
    assert len(removed) > 100
    assert sum(i.startswith("adv-dup-") for i in removed) > 100
