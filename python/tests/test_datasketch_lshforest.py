# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
"""pyroid.datasketch.MinHashLSHForest against datasketch 2.0.0.

Part 1 is datasketch's own test/test_lshforest.py run on pyroid's class.
Part 2 compares results with a live datasketch forest: query key sets
(including the early exit at ``k``), the exact sequence ``_query`` yields,
keys added after ``index()``, the views, and pickling, for the three MinHash
schemes and WeightedMinHash.
"""

from __future__ import annotations

import pickle
import random
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyroid.datasketch import MinHash as PyroidMinHash
from pyroid.datasketch.lshforest import MinHashLSHForest

ds = pytest.importorskip("datasketch")
if ds.__version__ != "2.0.0":
    pytest.skip(f"needs datasketch 2.0.0, found {ds.__version__}", allow_module_level=True)

MinHash = ds.MinHash
WeightedMinHashGenerator = ds.WeightedMinHashGenerator
SCHEMES = ("affine32", "affine64", "legacy")


# ---------------------------------------------------------------------------
# Part 1: datasketch 2.0.0's test/test_lshforest.py, on pyroid's forest.
# ---------------------------------------------------------------------------


def _setup() -> tuple[MinHashLSHForest, dict[str, Any]]:
    d = "abcdefghijklmnopqrstuvwxyz"
    data = {}
    forest = MinHashLSHForest()
    for i in range(len(d) - 2):
        key = d[i]
        m = MinHash()
        for s in d[i : i + 3]:
            m.update(s.encode("utf8"))
        data[key] = m
        forest.add(key, m)
    assert forest.is_empty()
    for t in forest.hashtables:
        assert len(t) >= 1
        items: list[Hashable] = []
        for h in t:
            items.extend(t[h])
        for key in data:
            assert key in items
    for key in data:
        assert key in forest
    for added in forest.keys:
        for i, h in enumerate(forest.keys[added]):
            assert added in forest.hashtables[i][h]
    with pytest.raises(ValueError, match="The given key has already been added"):
        forest.add("a", data["a"])
    forest.index()
    assert not forest.is_empty()
    with pytest.raises(ValueError, match="The given key has already been added"):
        forest.add("a", data["a"])
    return forest, data


def test_band_key_lengths() -> None:
    for l in range(2, 128 + 1, 16):  # noqa: E741
        forest = MinHashLSHForest(num_perm=128, l=l)
        m = MinHash()
        m.update(b"abcdefg")
        m.update(b"1234567")
        forest.add("m", m)
        sizes = [len(h) for ht in forest.hashtables for h in ht]
        assert all(sizes[0] == s for s in sizes)


def test_init() -> None:
    assert MinHashLSHForest().is_empty()


def test_query() -> None:
    forest, data = _setup()
    for key in data:
        assert key in forest.query(data[key], 10)


def test_get_minhash_hashvalues() -> None:
    forest, data = _setup()
    for key in data:
        minhash_ori = data[key]
        hashvalues = forest.get_minhash_hashvalues(key)
        minhash_retrieved = MinHash(
            seed=minhash_ori.seed, hashvalues=hashvalues, scheme=minhash_ori.scheme
        )
        retrieved_hashvalues = minhash_retrieved.hashvalues
        assert len(hashvalues) == len(retrieved_hashvalues)
        assert minhash_retrieved.jaccard(minhash_ori) == 1.0
        for i in range(len(retrieved_hashvalues)):
            assert hashvalues[i] == retrieved_hashvalues[i]


def test_get_minhash_hashvalues_rejects_corrupt_segments() -> None:
    forest, data = _setup()
    key = next(iter(data))
    segments = forest.keys[key]
    forest.keys[key] = [segments[0][:-3], *segments[1:]]
    with pytest.raises(ValueError, match="Stored hash segment of"):
        forest.get_minhash_hashvalues(key)
    forest.keys[key] = [*segments[:-1], segments[-1][: len(segments[-1]) // 2]]
    with pytest.raises(ValueError, match="inconsistent sizes"):
        forest.get_minhash_hashvalues(key)
    forest.keys[key] = segments


def test_pickle() -> None:
    forest, _ = _setup()
    forest2 = pickle.loads(pickle.dumps(forest))
    assert forest.hashtables == forest2.hashtables
    assert forest.keys == forest2.keys
    assert forest.l == forest2.l
    assert forest.k == forest2.k
    assert forest.hashranges == forest2.hashranges


def _weighted_setup() -> tuple[MinHashLSHForest, dict[int, Any]]:
    freqs = np.random.default_rng(0).integers(1, 100, (100, 100))
    data = {}
    forest = MinHashLSHForest()
    mg = WeightedMinHashGenerator(100, sample_size=128)
    for i in range(len(freqs)):
        m = mg.minhash(freqs[i])
        forest.add(i, m)
        data[i] = m
    assert forest.is_empty()
    for t in forest.hashtables:
        assert len(t) >= 1
        items: list[Hashable] = []
        for h in t:
            items.extend(t[h])
        for key in data:
            assert key in items
    for key in data:
        assert key in forest
    for added in forest.keys:
        for i, h in enumerate(forest.keys[added]):
            assert added in forest.hashtables[i][h]
    with pytest.raises(ValueError, match="The given key has already been added"):
        forest.add(5, data[5])
    forest.index()
    assert not forest.is_empty()
    with pytest.raises(ValueError, match="The given key has already been added"):
        forest.add(5, data[5])
    return forest, data


def test_weighted_band_key_lengths() -> None:
    mg = WeightedMinHashGenerator(100, sample_size=128)
    rng = np.random.default_rng(2)
    for l in range(2, mg.sample_size + 1, 16):  # noqa: E741
        m = mg.minhash(rng.integers(1, 99999999, 100))
        forest = MinHashLSHForest(num_perm=128, l=l)
        forest.add("m", m)
        sizes = [len(h) for ht in forest.hashtables for h in ht]
        assert all(sizes[0] == s for s in sizes)


def test_weighted_query() -> None:
    forest, data = _weighted_setup()
    for key in data:
        assert key in forest.query(data[key], 10)


def test_weighted_pickle() -> None:
    forest, _ = _weighted_setup()
    forest2 = pickle.loads(pickle.dumps(forest))
    assert forest.hashtables == forest2.hashtables
    assert forest.keys == forest2.keys
    assert forest.l == forest2.l
    assert forest.k == forest2.k
    assert forest.hashranges == forest2.hashranges


# ---------------------------------------------------------------------------
# Part 2: parity with a live datasketch forest.
# ---------------------------------------------------------------------------

NUM_PERM = 16
VOCAB = [f"t{i}".encode() for i in range(7)]
KEYS: list[Hashable] = [0, 1, 1.0, True, 2, "a", "b", b"x", (1, 2), None, *range(10, 40)]
_SKETCHES: dict[tuple[str, tuple[bytes, ...]], Any] = {}


def sketch(scheme: str, tokens: tuple[bytes, ...]) -> Any:
    key = (scheme, tokens)
    m = _SKETCHES.get(key)
    if m is None:
        m = MinHash(num_perm=NUM_PERM, scheme=scheme, seed=4)
        m.update_batch(list(tokens))
        _SKETCHES[key] = m
    return m


def outcome(f: Callable[..., Any], *args: Any) -> tuple[str, Any]:
    try:
        r = f(*args)
    except Exception as e:
        return (type(e).__name__, str(e))
    return ("ok", frozenset(r) if isinstance(r, list) else r)


def assert_same_state(a: MinHashLSHForest, b: Any) -> None:
    assert list(a.keys) == list(b.keys)
    assert a.keys == b.keys
    assert len(a.hashtables) == len(b.hashtables)
    for ta, tb in zip(a.hashtables, b.hashtables, strict=True):
        assert ta == dict(tb)
        assert len(ta) == len(tb)
    assert [list(s) for s in a.sorted_hashtables] == b.sorted_hashtables
    assert a.is_empty() == b.is_empty()
    assert (a.l, a.k, a.hashranges, a._minhash_scheme) == (
        b.l,
        b.k,
        b.hashranges,
        b._minhash_scheme,
    )


def check_queries(a: MinHashLSHForest, b: Any, m: Any) -> None:
    for k in (1, 2, 3, 5, 8, 100):
        assert outcome(a.query, m, k) == outcome(b.query, m, k), k
    for r in range(1, a.k + 1):
        # The exact sequence, duplicates included.
        assert list(a._query(m, r, a.l)) == list(b._query(m, r, b.l))


def run_ops(a: MinHashLSHForest, b: Any, ops: list[tuple[Any, ...]], scheme: str) -> None:
    for op in ops:
        if op[0] == "add":
            m = sketch(scheme, op[2])
            assert outcome(a.add, op[1], m) == outcome(b.add, op[1], m), op
        elif op[0] == "index":
            a.index()
            b.index()
        elif op[0] == "query":
            check_queries(a, b, sketch(scheme, op[1]))
        elif op[0] == "contains":
            assert (op[1] in a) == (op[1] in b)
    assert_same_state(a, b)


token_sets = st.lists(st.sampled_from(VOCAB), min_size=1, max_size=3, unique=True).map(
    lambda t: tuple(sorted(t))
)
ops_strategy = st.lists(
    st.one_of(
        st.tuples(st.just("add"), st.sampled_from(KEYS), token_sets),
        st.tuples(st.just("index")),
        st.tuples(st.just("query"), token_sets),
        st.tuples(st.just("contains"), st.sampled_from(KEYS)),
    ),
    max_size=40,
)


@given(ops=ops_strategy, l=st.sampled_from([1, 2, 4, 5, 8, 16]), scheme=st.sampled_from(SCHEMES))
def test_ops_match_datasketch(ops: list[tuple[Any, ...]], l: int, scheme: str) -> None:  # noqa: E741
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=l), ds.MinHashLSHForest(num_perm=NUM_PERM, l=l)
    run_ops(a, b, ops, scheme)
    a2, b2 = pickle.loads(pickle.dumps(a)), pickle.loads(pickle.dumps(b))
    assert_same_state(a2, b2)
    run_ops(a2, b2, [*ops, ("index",), *ops], scheme)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", range(3))
def test_seeded_workloads_match_datasketch(scheme: str, seed: int) -> None:
    rng = random.Random(seed)
    l = [2, 4, 8][seed]  # noqa: E741
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=l), ds.MinHashLSHForest(num_perm=NUM_PERM, l=l)
    ops: list[tuple[Any, ...]] = []
    for i in range(500):
        tokens = tuple(sorted(rng.sample(VOCAB, rng.randint(1, 3))))
        if i % 97 == 50:
            ops.append(("index",))
        elif rng.random() < 0.6:
            ops.append(("add", rng.choice(KEYS + list(range(100, 400))), tokens))
        else:
            ops.append(("query", tokens))
    run_ops(a, b, ops, scheme)


def test_weighted_forest_matches_datasketch() -> None:
    mg = WeightedMinHashGenerator(6, sample_size=NUM_PERM, seed=3)
    rng = np.random.default_rng(3)
    pool = [mg.minhash(rng.integers(1, 3, 6).astype(float)) for _ in range(10)]
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=4), ds.MinHashLSHForest(num_perm=NUM_PERM, l=4)
    for i in range(60):
        m = pool[i % len(pool)]
        assert outcome(a.add, i, m) == outcome(b.add, i, m)
        if i == 30:
            a.index()
            b.index()
    for m in pool:
        check_queries(a, b, m)
    assert_same_state(a, b)


def test_keys_added_after_index_are_found_through_indexed_band_keys() -> None:
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=4), ds.MinHashLSHForest(num_perm=NUM_PERM, l=4)
    m1, m2 = sketch("affine32", (b"t1",)), sketch("affine32", (b"t2",))
    for forest in (a, b):
        forest.add("first", m1)
        forest.index()
        forest.add("same", m1)  # its band keys were indexed
        forest.add("other", m2)  # not indexed
    assert set(a.query(m1, 10)) == set(b.query(m1, 10)) == {"first", "same"}
    assert set(a.query(m2, 10)) == set(b.query(m2, 10))
    assert_same_state(a, b)


def test_keys_view_is_a_dict() -> None:
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=4), ds.MinHashLSHForest(num_perm=NUM_PERM, l=4)
    m = sketch("affine64", (b"t1", b"t2"))
    for forest in (a, b):
        forest.add("k", m)
        forest.index()
        segments = forest.keys["k"]
        forest.keys["new"] = segments  # a keys entry without buckets
        del forest.keys["k"]
        forest.add("k", m)  # allowed again: listed twice in its buckets
    assert_same_state(a, b)
    assert a.query(m, 10) == ["k"] == b.query(m, 10)
    assert list(a._query(m, a.k, a.l)) == list(b._query(m, b.k, b.l))
    assert a.keys.get("missing", 5) == 5
    assert "new" in a.keys
    assert "new" in a
    assert (a.get_minhash_hashvalues("new") == b.get_minhash_hashvalues("new")).all()
    with pytest.raises(KeyError):
        del a.keys["missing"]
    with pytest.raises(KeyError, match="does not exist in the LSHForest"):
        a.get_minhash_hashvalues("missing")
    with pytest.raises(TypeError, match="lists of bytes"):
        a.keys["bad"] = ["not bytes"]  # type: ignore[list-item]
    del a.keys["new"]
    assert "new" not in a
    assert len(a.keys) == 1
    table, ordered = a.hashtables[0], a.sorted_hashtables[0]
    h = ordered[0]
    assert table[h] == ["k", "k"]
    assert table[b"nope"] == []
    assert table.get(b"nope") is None
    assert b"nope" not in table
    assert h in table
    assert table.get("str") is None  # type: ignore[call-overload]
    assert ordered[-1] == ordered[len(ordered) - 1]
    assert ordered[:5] == list(ordered)[:5]
    with pytest.raises(IndexError):
        ordered[len(ordered)]


def test_odd_hashvalues_and_k_match_datasketch() -> None:
    class Fake:
        def __init__(self, hv: Any) -> None:
            self.hashvalues = hv

        def __len__(self) -> int:
            return NUM_PERM

    hv = sketch("affine64", (b"t1", b"t3")).hashvalues
    odd = [
        hv.astype(hv.dtype.newbyteorder(">")),
        np.repeat(hv, 2)[::2],
        hv[:9],
        hv.astype(np.float32),
    ]
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=4), ds.MinHashLSHForest(num_perm=NUM_PERM, l=4)
    for i, values in enumerate(odd):
        for forest in (a, b):
            forest.add(i, Fake(values))
            forest.add(10 + i, sketch("affine64", (VOCAB[i],)))
    a.index()
    b.index()
    for values in [*odd, hv]:
        check_queries(a, b, Fake(values))
        for k in (2.5, 1.0, True):
            assert outcome(a.query, Fake(values), k) == outcome(b.query, Fake(values), k)
    bad = Fake(hv.tolist())
    assert outcome(a.add, "x", bad) == outcome(b.add, "x", bad)
    assert outcome(a.query, bad, 3) == outcome(b.query, bad, 3)
    assert_same_state(a, b)


@pytest.mark.parametrize(
    ("num_perm", "l"), [(128, 0), (0, 8), (8, 16), (16, 2.0), (16.0, 4), (16, True)]
)
def test_constructor_matches(num_perm: Any, l: Any) -> None:  # noqa: E741
    a = outcome(MinHashLSHForest, num_perm, l)
    b = outcome(ds.MinHashLSHForest, num_perm, l)
    assert a[0] == b[0]
    if a[0] != "ok":
        assert a == b


def test_query_errors_match() -> None:
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=4), ds.MinHashLSHForest(num_perm=NUM_PERM, l=4)
    m, short = sketch("affine32", (b"t1",)), MinHash(num_perm=8)
    for forest in (a, b):
        forest.add("x", m)
        forest.index()
    for args in [(m, 0), (m, -1), (short, 3), (sketch("legacy", (b"t1",)), 3)]:
        assert outcome(a.query, *args) == outcome(b.query, *args)
    for r, bb in [(0, 1), (5, 1), (1, 0), (1, 5)]:
        gen_a, gen_b = a._query(m, r, bb), b._query(m, r, bb)  # lazy, as generators
        assert outcome(list, gen_a) == outcome(list, gen_b)
    assert outcome(a.add, "y", short) == outcome(b.add, "y", short)
    assert outcome(a.add, ["unhashable"], m)[0] == outcome(b.add, ["unhashable"], m)[0]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_pyroid_minhash_sketches(scheme: str) -> None:
    rng = random.Random(6)
    docs = [rng.sample(VOCAB, rng.randint(1, 3)) for _ in range(150)]
    ours = PyroidMinHash.bulk(docs, num_perm=NUM_PERM, scheme=scheme, seed=4)
    theirs = MinHash.bulk(docs, num_perm=NUM_PERM, scheme=scheme, seed=4)
    a, b = MinHashLSHForest(num_perm=NUM_PERM, l=4), ds.MinHashLSHForest(num_perm=NUM_PERM, l=4)
    for i, (m, n) in enumerate(zip(ours, theirs, strict=True)):
        assert outcome(a.add, i % 120, m) == outcome(b.add, i % 120, n)
        if i == 100:
            a.index()
            b.index()
    for m, n in zip(ours, theirs, strict=True):
        for k in (1, 3, 10):
            assert set(a.query(m, k)) == set(b.query(n, k))
    assert_same_state(a, b)
