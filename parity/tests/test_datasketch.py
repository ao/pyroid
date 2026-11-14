"""datasketch parity: the ``--impl`` candidate vs datasketch 2.0.0 (live), on a large realistic LSH workload.

    pytest tests/test_datasketch.py                                   # self-check (candidate = datasketch)
    pytest tests/test_datasketch.py --impl pyroid.datasketch          # pyroid vs datasketch, all docs
    pytest tests/test_datasketch.py --impl pyroid.datasketch --parity-limit 5000

Corpus: the 100k FineWeb headline slice followed by the 3,232-doc adversarial set (its 600 near-duplicates of dev
docs give the LSH real collisions). Shingles are word 5-grams, ``" ".join(words[i:i + 5]).encode("utf-8")`` over
``text.split()``; a doc with fewer than 5 words has none. Keys are the doc ids. The reference is datasketch itself,
run live on the same inputs; every comparison is exact (hash values, bytes, ``(b, r)``, query result *sets*).

Checks (suite ``datasketch`` in parity-report.json):
    signatures-<scheme>      ``MinHash.bulk`` hash values (values and dtype), 128 permutations, seed 1, for
                             affine32 (default), affine64 and legacy
    minhash-batch-<scheme>   the candidate's ``minhash_batch(texts, ngram=5)`` rows == datasketch's signatures
    lean-bytes-<scheme>      ``LeanMinHash.serialize`` bytes per doc (one byte order per scheme: @, >, <)
    lsh-params               ``MinHashLSH(threshold, num_perm, weights)`` (b, r) over a threshold grid
    lsh-<config>             every doc's ``MinHashLSH.query`` result set, per index configuration below
    lsh-batch-<config>       the candidate's ``query_batch`` == datasketch's per-doc ``query`` (if provided)
    lsh-remove               after removing every 10th key: every doc's result set, and ``key in lsh``
    lsh-merge                two half indexes merged == datasketch's merged index (sampled queries)
    forest-topk              ``MinHashLSHForest(l=8)`` top-10 key sets, every 10th doc
    ensemble                 ``MinHashLSHEnsemble(threshold=0.8, num_part=16)`` containment result sets (first
                             20k docs with shingles, every 10th queried)
"""

from __future__ import annotations

import importlib.metadata as md
from typing import Any

import datasketch
import numpy as np
import pytest

from pyroid_parity import data as D
from pyroid_parity.plugin import _impl, effective_limit, need_corpus, oracle_pin, reference
from pyroid_parity.report import Collector

SUITE = "datasketch"
DATASKETCH_VERSION = "2.0.0"
NUM_PERM = 128
NGRAM = 5
SCHEMES = ("affine32", "affine64", "legacy")
BYTE_ORDER = {"affine32": "@", "affine64": ">", "legacy": "<"}
CHUNK = 1000
# (name, scheme, MinHashLSH kwargs): the default threshold, a low threshold with many candidates, explicit
# narrow bands, and the legacy scheme.
LSH_CONFIGS = (
    ("t0.9", "affine32", {"threshold": 0.9}),
    ("t0.5", "affine32", {"threshold": 0.5}),
    ("b32r4", "affine32", {"params": (32, 4)}),
    ("legacy-t0.8", "legacy", {"threshold": 0.8}),
)


@pytest.fixture(scope="session")
def datasketch_impl(request: pytest.FixtureRequest) -> Any:
    return _impl(request, SUITE)


@pytest.fixture(scope="module", autouse=True)
def _oracle(parity: Collector) -> None:
    reference(SUITE, "datasketch", md.version("datasketch"), None)
    parity.uses(SUITE, D.CORPORA["headline"], D.CORPORA["adversarial"])


def shingles(text: str) -> list[bytes]:
    words = text.split()
    return [" ".join(words[i : i + NGRAM]).encode("utf-8") for i in range(len(words) - NGRAM + 1)]


class Workload:
    """The corpus, its shingle-set sizes, and datasketch's and the candidate's signatures per scheme."""

    def __init__(self, ids: list[str], texts: list[str], cand: Any) -> None:
        self.ids, self.texts, self.cand = ids, texts, cand
        n = len(ids)
        self.sizes = np.zeros(n, dtype=np.int64)
        self.ref: dict[str, np.ndarray] = {}
        self.got: dict[str, np.ndarray] = {}
        dtypes = {"affine32": np.uint32, "affine64": np.uint64, "legacy": np.uint64}
        for s in SCHEMES:
            self.ref[s] = np.empty((n, NUM_PERM), dtype=dtypes[s])
            self.got[s] = np.empty((n, NUM_PERM), dtype=dtypes[s])
        self.got_dtype_ok = dict.fromkeys(SCHEMES, True)
        for start in range(0, n, CHUNK):
            sh = [shingles(t) for t in texts[start : start + CHUNK]]
            self.sizes[start : start + len(sh)] = [len(set(x)) for x in sh]
            for s in SCHEMES:
                ref = datasketch.MinHash.bulk(sh, num_perm=NUM_PERM, scheme=s)
                got = cand.MinHash.bulk(sh, num_perm=NUM_PERM, scheme=s)
                self.ref[s][start : start + len(sh)] = [m.hashvalues for m in ref]
                self.got[s][start : start + len(sh)] = [m.hashvalues for m in got]
                if any(m.hashvalues.dtype != dtypes[s] for m in got):
                    self.got_dtype_ok[s] = False

    def lean(self, mod: Any, scheme: str, i: int) -> Any:
        return mod.LeanMinHash(seed=1, hashvalues=self.ref[scheme][i], scheme=scheme)


@pytest.fixture(scope="module")
def work(datasketch_impl: Any, request: pytest.FixtureRequest) -> Workload:
    head, adv = need_corpus("headline"), need_corpus("adversarial")
    ids, texts = head.ids + adv.ids, head.texts + adv.texts
    n = effective_limit(request.config, datasketch_impl) or len(ids)
    return Workload(ids[:n], texts[:n], datasketch_impl.obj.mod)


def test_live_oracle_is_pinned_datasketch() -> None:
    assert md.version("datasketch") == oracle_pin("datasketch", DATASKETCH_VERSION)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_signatures(work: Workload, parity: Collector, scheme: str) -> None:
    ref, got = work.ref[scheme], work.got[scheme]
    eq = (ref == got).all(axis=1)
    bad = [work.ids[i] for i in np.flatnonzero(~eq)]
    for i in range(len(work.ids)):
        parity.record(
            SUITE,
            f"signatures-{scheme}",
            bool(eq[i]) and work.got_dtype_ok[scheme],
            case=work.ids[i],
            input=work.texts[i],
            got=got[i, :4].tolist(),
            want=ref[i, :4].tolist(),
        )
    assert work.got_dtype_ok[scheme], "hash value dtype differs"
    assert not bad, f"{len(bad)}/{len(eq)} signatures differ: first {bad[:5]}"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_minhash_batch(work: Workload, parity: Collector, scheme: str) -> None:
    batch = getattr(work.cand, "minhash_batch", None)
    if batch is None:
        parity.skip(SUITE, f"minhash-batch-{scheme}", "the candidate has no minhash_batch")
        pytest.skip("the candidate has no minhash_batch")
    ref = work.ref[scheme]
    bad = 0
    for start in range(0, len(work.ids), 10 * CHUNK):
        got = batch(work.texts[start : start + 10 * CHUNK], num_perm=NUM_PERM, scheme=scheme, ngram=NGRAM)
        want = ref[start : start + len(got)]
        same = (got == want).all(axis=1) & (got.dtype == want.dtype)
        for j, ok in enumerate(same.tolist()):
            parity.record(
                SUITE,
                f"minhash-batch-{scheme}",
                ok,
                case=work.ids[start + j],
                got=got[j, :4].tolist(),
                want=want[j, :4].tolist(),
            )
        bad += int((~same).sum())
    assert not bad, f"{bad} minhash_batch rows differ"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_lean_bytes(work: Workload, parity: Collector, scheme: str) -> None:
    order = BYTE_ORDER[scheme]
    bad = []
    for i in range(len(work.ids)):
        ref, got = work.lean(datasketch, scheme, i), work.lean(work.cand, scheme, i)
        rb, gb = bytearray(ref.bytesize(order)), bytearray(got.bytesize(order))
        ref.serialize(rb, order)
        got.serialize(gb, order)
        back = work.cand.LeanMinHash.deserialize(rb, order)
        same = rb == gb and np.array_equal(back.hashvalues, ref.hashvalues) and back.scheme == scheme
        if not same:
            bad.append(work.ids[i])
        parity.record(
            SUITE, f"lean-bytes-{scheme}", same, case=work.ids[i], got=bytes(gb[:24]), want=bytes(rb[:24])
        )
    assert not bad, f"{len(bad)} LeanMinHash byte strings differ: first {bad[:5]}"


def _br(lsh: Any) -> tuple[int, int]:
    return (lsh.b, lsh.r)


def test_lsh_params(datasketch_impl: Any, parity: Collector) -> None:
    cand = datasketch_impl.obj.mod
    bad = []
    for num_perm in (16, 64, 128, 256):
        for weights in ((0.5, 0.5), (0.2, 0.8), (0.8, 0.2)):
            for t in np.round(np.arange(0.05, 1.0001, 0.05), 2).tolist():
                kw = {"threshold": t, "num_perm": num_perm, "weights": weights}
                try:
                    want: Any = _br(datasketch.MinHashLSH(**kw))
                except ValueError as e:
                    want = ("ValueError", str(e))
                try:
                    got: Any = _br(cand.MinHashLSH(**kw))
                except ValueError as e:
                    got = ("ValueError", str(e))
                if got != want:
                    bad.append((kw, got, want))
                parity.record(SUITE, "lsh-params", got == want, case=repr(kw), got=got, want=want)
    assert not bad, f"{len(bad)} (b, r) differ: {bad[:5]}"


def build_lsh(work: Workload, name: str) -> tuple[Any, Any, str]:
    _, scheme, kw = next(c for c in LSH_CONFIGS if c[0] == name)
    ref = datasketch.MinHashLSH(num_perm=NUM_PERM, **kw)
    got = work.cand.MinHashLSH(num_perm=NUM_PERM, **kw)
    with ref.insertion_session() as rs, got.insertion_session() as gs:
        for i, key in enumerate(work.ids):
            rs.insert(key, work.lean(datasketch, scheme, i))
            gs.insert(key, work.lean(work.cand, scheme, i))
    return ref, got, scheme


def _compare_queries(
    parity: Collector, check: str, work: Workload, ref: Any, got: Any, scheme: str, step: int = 1
) -> int:
    bad = 0
    for i in range(0, len(work.ids), step):
        r = work.lean(datasketch, scheme, i)
        g = work.lean(work.cand, scheme, i)
        want, have = set(ref.query(r)), set(got.query(g))
        same = want == have
        bad += not same
        parity.record(SUITE, check, same, case=work.ids[i], got=sorted(have)[:5], want=sorted(want)[:5])
    return bad


@pytest.mark.parametrize("name", [c[0] for c in LSH_CONFIGS])
def test_lsh_query(work: Workload, parity: Collector, name: str) -> None:
    ref, got, scheme = build_lsh(work, name)
    assert (got.b, got.r) == (ref.b, ref.r)
    bad = _compare_queries(parity, f"lsh-{name}", work, ref, got, scheme)
    if not hasattr(got, "query_batch"):
        parity.skip(SUITE, f"lsh-batch-{name}", "the candidate's MinHashLSH has no query_batch")
    else:
        for i, have in enumerate(got.query_batch(work.ref[scheme])):
            want = set(ref.query(work.lean(datasketch, scheme, i)))
            same = want == set(have) and len(have) == len(set(have))
            bad += not same
            parity.record(
                SUITE,
                f"lsh-batch-{name}",
                same,
                case=work.ids[i],
                got=sorted(have)[:5],
                want=sorted(want)[:5],
            )
    assert not bad, f"{bad} query result sets differ"


def test_lsh_remove(work: Workload, parity: Collector) -> None:
    kw = {"threshold": 0.5}
    ref = datasketch.MinHashLSH(num_perm=NUM_PERM, **kw)
    got = work.cand.MinHashLSH(num_perm=NUM_PERM, **kw)
    for i, key in enumerate(work.ids):
        ref.insert(key, work.lean(datasketch, "affine32", i))
        got.insert(key, work.lean(work.cand, "affine32", i))
    for key in work.ids[::10]:
        ref.remove(key)
        got.remove(key)
    for key in work.ids[:1000]:
        parity.record(SUITE, "lsh-remove", (key in ref) == (key in got), case=key)
    bad = _compare_queries(parity, "lsh-remove", work, ref, got, "affine32")
    assert not bad, f"{bad} result sets differ after removals"


def test_lsh_merge(work: Workload, parity: Collector) -> None:
    half = len(work.ids) // 2
    pair = []
    for mod in (datasketch, work.cand):
        a, b = (
            mod.MinHashLSH(num_perm=NUM_PERM, threshold=0.5),
            mod.MinHashLSH(num_perm=NUM_PERM, threshold=0.5),
        )
        for i, key in enumerate(work.ids):
            (a if i < half else b).insert(key, work.lean(mod, "affine32", i))
        a.merge(b)
        pair.append(a)
    bad = _compare_queries(parity, "lsh-merge", work, pair[0], pair[1], "affine32", step=10)
    assert not bad, f"{bad} result sets differ after merge"


def test_forest_topk(work: Workload, parity: Collector) -> None:
    forests = []
    for mod in (datasketch, work.cand):
        f = mod.MinHashLSHForest(num_perm=NUM_PERM, l=8)
        for i, key in enumerate(work.ids):
            f.add(key, work.lean(mod, "affine32", i))
        f.index()
        forests.append(f)
    bad = 0
    for i in range(0, len(work.ids), 10):
        want = set(forests[0].query(work.lean(datasketch, "affine32", i), 10))
        have = set(forests[1].query(work.lean(work.cand, "affine32", i), 10))
        bad += want != have
        parity.record(
            SUITE, "forest-topk", want == have, case=work.ids[i], got=sorted(have)[:5], want=sorted(want)[:5]
        )
    assert not bad, f"{bad} top-k sets differ"


def test_ensemble(work: Workload, parity: Collector) -> None:
    rows = [i for i in range(len(work.ids)) if work.sizes[i] > 0][:20_000]
    ens = []
    for mod in (datasketch, work.cand):
        e = mod.MinHashLSHEnsemble(threshold=0.8, num_perm=NUM_PERM, num_part=16)
        e.index([(work.ids[i], work.lean(mod, "affine32", i), int(work.sizes[i])) for i in rows])
        ens.append(e)
    bad = 0
    for i in rows[::10]:
        size = int(work.sizes[i])
        want = set(ens[0].query(work.lean(datasketch, "affine32", i), size))
        have = set(ens[1].query(work.lean(work.cand, "affine32", i), size))
        bad += want != have
        parity.record(
            SUITE, "ensemble", want == have, case=work.ids[i], got=sorted(have)[:5], want=sorted(want)[:5]
        )
    assert not bad, f"{bad} containment result sets differ"
