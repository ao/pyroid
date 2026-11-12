# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
"""pyroid.datasketch.MinHashLSH against datasketch 2.0.0.

Part 1 is datasketch's own test/test_lsh.py run on pyroid's class (the Redis
cases are delegated to datasketch). Part 2 compares whole index states with a
live datasketch index over seeded and Hypothesis workloads, for the three
MinHash schemes, WeightedMinHash, prepickle, check_duplication=False, merge
and pickling. Sketches come from datasketch.MinHash until pyroid's lands; the
index only reads ``hashvalues``, ``len()`` and ``scheme``.
"""

from __future__ import annotations

import copy
import pickle
import random
import sys
import threading
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pyroid
from pyroid.datasketch import MinHash as PyroidMinHash
from pyroid.datasketch import _common, minhash_batch
from pyroid.datasketch.lsh import (
    MinHashLSH,
    _false_negative_probability,
    _false_positive_probability,
    _optimal_param,
)
from pyroid.datasketch.storage import (
    DictListStorage,
    DictListStorageView,
    DictSetStorage,
    DictSetStorageView,
    OrderedStorage,
    UnorderedStorage,
    _random_name,
    ordered_storage,
    unordered_storage,
)

ds = pytest.importorskip("datasketch")
if ds.__version__ != "2.0.0":
    pytest.skip(f"needs datasketch 2.0.0, found {ds.__version__}", allow_module_level=True)

MinHash = ds.MinHash
WeightedMinHashGenerator = ds.WeightedMinHashGenerator
SCHEMES = ("affine32", "affine64", "legacy")


# ---------------------------------------------------------------------------
# Part 1: datasketch 2.0.0's test/test_lsh.py, on pyroid's MinHashLSH.
# ---------------------------------------------------------------------------


def fake_redis(**kwargs: Any) -> Any:
    mockredis = pytest.importorskip("mockredis")
    redis = mockredis.mock_redis_client(**kwargs)
    redis.connection_pool = None
    redis.response_callbacks = None
    return redis


@contextmanager
def redis_patch() -> Iterator[None]:
    pytest.importorskip("mockredis")
    pytest.importorskip("redis")
    with patch("redis.Redis", fake_redis):
        yield


REDIS = {"type": "redis", "redis": {"host": "localhost", "port": 6379}}


def _mh16(data: bytes, num_perm: int = 16) -> Any:
    m = MinHash(num_perm)
    m.update(data)
    return m


def test_init() -> None:
    lsh = MinHashLSH(threshold=0.8)
    assert lsh.is_empty()
    b1, r1 = lsh.b, lsh.r
    lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
    b2, r2 = lsh.b, lsh.r
    assert b1 < b2
    assert r1 > r2


def test_band_key_lengths() -> None:
    for _l in range(2, 128 + 1, 16):
        lsh = MinHashLSH(num_perm=128)
        m = MinHash()
        m.update(b"abcdefg")
        m.update(b"1234567")
        lsh.insert("m", m)
        sizes = [len(h) for ht in lsh.hashtables for h in ht]
        assert all(sizes[0] == s for s in sizes)


@pytest.mark.thread_unsafe(reason="1023 indexes and MinHashes of up to 4096 permutations")
def test_unpacking() -> None:
    for b in range(2, 1024 + 1):
        lsh = MinHashLSH(num_perm=b * 4, params=(b, 4))
        m = MinHash(num_perm=b * 4)
        m.update(b"abcdefg")
        m.update(b"1234567")
        lsh.insert("m", m)
        sizes = [len(h) for ht in lsh.hashtables for h in ht]
        assert all(sizes[0] == s for s in sizes)


def test_insert() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    m1, m2 = _mh16(b"a"), _mh16(b"b")
    lsh.insert("a", m1)
    lsh.insert("b", m2)
    for t in lsh.hashtables:
        assert len(t) >= 1
        items: list[Hashable] = []
        for h in t:
            items.extend(t[h])
        assert "a" in items
        assert "b" in items
    assert "a" in lsh
    assert "b" in lsh
    for i, h in enumerate(lsh.keys["a"]):
        assert "a" in lsh.hashtables[i][h]
    with pytest.raises(ValueError, match="Expecting minhash with length 16, got 18"):
        lsh.insert("c", MinHash(18))


def test_query() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    m1, m2 = _mh16(b"a"), _mh16(b"b")
    lsh.insert("a", m1)
    lsh.insert("b", m2)
    assert "a" in lsh.query(m1)
    assert "b" in lsh.query(m2)
    with pytest.raises(ValueError, match="Expecting minhash with length 16, got 18"):
        lsh.query(MinHash(18))


def test_query_buffer() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    m1, m2 = _mh16(b"a"), _mh16(b"b")
    lsh.insert("a", m1)
    lsh.insert("b", m2)
    lsh.add_to_query_buffer(m1)
    assert "a" in lsh.collect_query_buffer()
    lsh.add_to_query_buffer(m2)
    assert "b" in lsh.collect_query_buffer()
    with pytest.raises(ValueError, match="Expecting minhash with length 16, got 18"):
        lsh.add_to_query_buffer(MinHash(18))


def test_query_buffer_matches_query_candidates() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=32)
    docs = []
    for tokens in ([b"a", b"b", b"c"], [b"a", b"b", b"d"], [b"x", b"y", b"z"]):
        minhash = MinHash(num_perm=32)
        for token in tokens:
            minhash.update(token)
        docs.append(minhash)
    for key, minhash in enumerate(docs):
        lsh.insert(key, minhash)
    lsh.add_to_query_buffer(docs[0])
    buffered_result = set(lsh.collect_query_buffer())
    direct_result = set(lsh.query(docs[0]))
    assert buffered_result == direct_result
    assert buffered_result == {0, 1}


def test_remove() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    lsh.insert("a", _mh16(b"a"))
    lsh.insert("b", _mh16(b"b"))
    lsh.remove("a")
    assert "a" not in lsh.keys
    for table in lsh.hashtables:
        for h in table:
            assert len(table[h]) > 0
            assert "a" not in table[h]
    with pytest.raises(ValueError, match="The given key does not exist"):
        lsh.remove("c")


def test_pickle() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    m1, m2 = _mh16(b"a"), _mh16(b"b")
    lsh.insert("a", m1)
    lsh.insert("b", m2)
    lsh2 = pickle.loads(pickle.dumps(lsh))
    assert "a" in lsh2.query(m1)
    assert "b" in lsh2.query(m2)


@pytest.mark.thread_unsafe(reason="patches redis.Redis")
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_insert_redis() -> None:
    with redis_patch():
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        assert type(lsh) is ds.MinHashLSH
        m1, m2 = _mh16(b"a"), _mh16(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        for t in lsh.hashtables:
            assert len(t) >= 1
            items: list[Any] = []
            for h in t:
                items.extend(t[h])
            assert pickle.dumps("a") in items
            assert pickle.dumps("b") in items
        assert "a" in lsh
        assert "b" in lsh
        for i, h in enumerate(lsh.keys[pickle.dumps("a")]):
            assert pickle.dumps("a") in lsh.hashtables[i][h]
        with pytest.raises(ValueError, match="Expecting minhash with length 16, got 18"):
            lsh.insert("c", MinHash(18))


@pytest.mark.thread_unsafe(reason="patches redis.Redis")
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_query_redis() -> None:
    with redis_patch():
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        m1, m2 = _mh16(b"a"), _mh16(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        assert "a" in lsh.query(m1)
        assert "b" in lsh.query(m2)
        with pytest.raises(ValueError, match="Expecting minhash with length 16, got 18"):
            lsh.query(MinHash(18))


@pytest.mark.thread_unsafe(reason="patches redis.Redis")
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_query_buffer_redis() -> None:
    with redis_patch():
        lsh = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        m1, m2 = _mh16(b"a"), _mh16(b"b")
        lsh.insert("a", m1)
        lsh.insert("b", m2)
        lsh.query(m1)
        lsh.add_to_query_buffer(m1)
        assert "a" in lsh.collect_query_buffer()
        lsh.add_to_query_buffer(m2)
        assert "b" in lsh.collect_query_buffer()
        with pytest.raises(ValueError, match="Expecting minhash with length 16, got 18"):
            lsh.add_to_query_buffer(MinHash(18))


def test_insertion_session() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    data = [("a", _mh16(b"a")), ("b", _mh16(b"b"))]
    with lsh.insertion_session() as session:
        for key, minhash in data:
            session.insert(key, minhash)
    for t in lsh.hashtables:
        assert len(t) >= 1
        items: list[Hashable] = []
        for h in t:
            items.extend(t[h])
        assert "a" in items
        assert "b" in items
    assert "a" in lsh
    assert "b" in lsh
    for i, h in enumerate(lsh.keys["a"]):
        assert "a" in lsh.hashtables[i][h]


def test_deletion_session() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    lsh.insert("a", _mh16(b"a"))
    lsh.insert("b", _mh16(b"b"))
    lsh.insert("c", _mh16(b"c"))
    with lsh.deletion_session() as session:
        for key in ["a", "b"]:
            session.remove(key)
    assert "a" not in lsh.keys
    assert "b" not in lsh.keys
    assert "c" in lsh.keys
    for table in lsh.hashtables:
        for h in table:
            assert "a" not in table[h]
            assert "b" not in table[h]


def test_get_counts() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=16)
    lsh.insert("a", _mh16(b"a"))
    lsh.insert("b", _mh16(b"b"))
    counts = lsh.get_counts()
    assert len(counts) == lsh.b
    for table in counts:
        assert sum(table.values()) == 2


def test_merge() -> None:
    lsh1 = MinHashLSH(threshold=0.5, num_perm=16)
    m1, m2 = _mh16(b"a"), _mh16(b"b")
    lsh1.insert("a", m1)
    lsh1.insert("b", m2)
    lsh2 = MinHashLSH(threshold=0.5, num_perm=16)
    lsh2.insert("c", m1)
    lsh2.insert("d", m2)
    lsh1.merge(lsh2)
    for t in lsh1.hashtables:
        assert len(t) >= 1
        items: list[Hashable] = []
        for h in t:
            items.extend(t[h])
        assert "c" in items
        assert "d" in items
    for key in "abcd":
        assert key in lsh1
    for i, h in enumerate(lsh1.keys["c"]):
        assert "c" in lsh1.hashtables[i][h]
    with pytest.raises(ValueError, match="The keys are overlapping"):
        lsh1.merge(lsh2, check_overlap=True)
    lsh3 = MinHashLSH(threshold=0.5, num_perm=16)
    lsh3.insert("a", _mh16(b"e"))
    with pytest.raises(ValueError, match="The keys are overlapping"):
        lsh1.merge(lsh3, check_overlap=True)
    lsh1.merge(lsh3)
    lsh4 = MinHashLSH(threshold=0.5, num_perm=16)
    lsh4.insert("a", _mh16(b"e"))
    lsh1.merge(lsh4, check_overlap=False)


@pytest.mark.thread_unsafe(reason="patches redis.Redis")
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_merge_redis() -> None:
    with redis_patch():
        lsh1 = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        lsh2 = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        lsh1.insert("a", _mh16(b"a"))
        lsh1.insert("b", _mh16(b"b"))
        lsh2.insert("c", _mh16(b"c"))
        lsh2.insert("d", _mh16(b"d"))
        lsh1.merge(lsh2)
        for t in lsh1.hashtables:
            assert len(t) >= 1
            items: list[Any] = []
            for h in t:
                items.extend(t[h])
            assert pickle.dumps("c") in items
            assert pickle.dumps("d") in items
        for key in "abcd":
            assert key in lsh1
        for i, h in enumerate(lsh1.keys[pickle.dumps("c")]):
            assert pickle.dumps("c") in lsh1.hashtables[i][h]
        with pytest.raises(ValueError, match="The keys are overlapping"):
            lsh1.merge(lsh2, check_overlap=True)
        lsh3 = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        lsh3.insert("a", _mh16(b"e"))
        with pytest.raises(ValueError, match="The keys are overlapping"):
            lsh1.merge(lsh3, check_overlap=True)
        lsh4 = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        lsh4.insert("a", _mh16(b"e"))
        lsh1.merge(lsh4, check_overlap=False)


@pytest.mark.thread_unsafe(reason="patches redis.Redis")
@pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")
def test_redis_deletion_session() -> None:
    with redis_patch():
        lsh: Any = MinHashLSH(threshold=0.5, num_perm=16, storage_config=REDIS)
        for storage in [lsh.keys, *lsh.hashtables]:
            storage._buffer = storage._redis.pipeline()
        lsh.insert("a", _mh16(b"a"))
        lsh.insert("b", _mh16(b"b"))
        lsh.insert("c", _mh16(b"c"))
        with (
            patch.object(type(lsh.keys), "__init__", lambda self, config, name: None),
            lsh.deletion_session() as session,
        ):
            for key in ["a", "b"]:
                session.remove(key)
        assert "a" not in lsh
        assert "b" not in lsh
        assert "c" in lsh
        for table in lsh.hashtables:
            for h in table:
                items = table[h]
                assert b"a" not in items
                assert b"b" not in items


def _weighted(dim: int, sample_size: int) -> Callable[[], Any]:
    mg = WeightedMinHashGenerator(dim, sample_size)
    rng = np.random.default_rng(len(threading.current_thread().name) + dim)
    return lambda: mg.minhash(rng.uniform(1, 10, dim))


def test_weighted_init() -> None:
    lsh = MinHashLSH(threshold=0.8)
    assert lsh.is_empty()
    b1, r1 = lsh.b, lsh.r
    lsh = MinHashLSH(threshold=0.8, weights=(0.2, 0.8))
    assert b1 < lsh.b
    assert r1 > lsh.r


def test_weighted_band_key_lengths() -> None:
    mg = WeightedMinHashGenerator(100, sample_size=128)
    rng = np.random.default_rng(1)
    for _l in range(2, mg.sample_size + 1, 16):
        m = mg.minhash(rng.integers(1, 99999999, 100))
        lsh = MinHashLSH(num_perm=128)
        lsh.insert("m", m)
        sizes = [len(h) for ht in lsh.hashtables for h in ht]
        assert all(sizes[0] == s for s in sizes)


def test_weighted_insert() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=4)
    new = _weighted(10, 4)
    lsh.insert("a", new())
    lsh.insert("b", new())
    for t in lsh.hashtables:
        assert len(t) >= 1
        items: list[Hashable] = []
        for h in t:
            items.extend(t[h])
        assert "a" in items
        assert "b" in items
    assert "a" in lsh
    assert "b" in lsh
    for i, h in enumerate(lsh.keys["a"]):
        assert "a" in lsh.hashtables[i][h]
    with pytest.raises(ValueError, match="Expecting minhash with length 4, got 5"):
        lsh.insert("c", _weighted(10, 5)())


def test_weighted_query() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=4)
    new = _weighted(10, 4)
    m1, m2 = new(), new()
    lsh.insert("a", m1)
    lsh.insert("b", m2)
    assert "a" in lsh.query(m1)
    assert "b" in lsh.query(m2)
    with pytest.raises(ValueError, match="Expecting minhash with length 4, got 5"):
        lsh.query(_weighted(10, 5)())


def test_weighted_remove() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=4)
    new = _weighted(10, 4)
    lsh.insert("a", new())
    lsh.insert("b", new())
    lsh.remove("a")
    assert "a" not in lsh.keys
    for table in lsh.hashtables:
        for h in table:
            assert len(table[h]) > 0
            assert "a" not in table[h]
    with pytest.raises(ValueError, match="The given key does not exist"):
        lsh.remove("c")


def test_weighted_pickle() -> None:
    lsh = MinHashLSH(threshold=0.5, num_perm=4)
    new = _weighted(10, 4)
    m1, m2 = new(), new()
    lsh.insert("a", m1)
    lsh.insert("b", m2)
    lsh2 = pickle.loads(pickle.dumps(lsh))
    assert "a" in lsh2.query(m1)
    assert "b" in lsh2.query(m2)


# ---------------------------------------------------------------------------
# Part 2: whole-state parity with a live datasketch index.
# ---------------------------------------------------------------------------

NUM_PERM = 16
PARAMS = [(4, 4), (8, 2), (5, 3), (3, 5), (2, 8), (16, 1)]
VOCAB = [f"t{i}".encode() for i in range(6)]
# Keys that compare equal (1 == 1.0 == True), and other hashables.
KEYS: list[Hashable] = [0, 1, 1.0, True, 2, "a", "b", b"x", (1, 2), frozenset({3}), None]

_SKETCHES: dict[tuple[str, tuple[bytes, ...]], Any] = {}


def sketch(scheme: str, tokens: tuple[bytes, ...]) -> Any:
    """A datasketch MinHash of ``tokens`` (cached: datasketch's constructor is slow)."""
    key = (scheme, tokens)
    m = _SKETCHES.get(key)
    if m is None:
        m = MinHash(num_perm=NUM_PERM, scheme=scheme, seed=3)
        m.update_batch(list(tokens))
        _SKETCHES[key] = m
    return m


def pair(**kwargs: Any) -> tuple[MinHashLSH, Any]:
    return MinHashLSH(**kwargs), ds.MinHashLSH(**kwargs)


def outcome(f: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[str, Any]:
    """The result of ``f`` (sets for lists, as results are unordered), or the exception."""
    try:
        r = f(*args, **kwargs)
    except Exception as e:
        return (type(e).__name__, str(e))
    if isinstance(r, list) and not any(isinstance(x, dict) for x in r):
        return ("ok", frozenset(r))
    return ("ok", r)


def same_error_type(a: tuple[str, Any], b: tuple[str, Any]) -> bool:
    # Unhashable-key messages come from different dict paths, so compare types only.
    return a == b or (a[0] == b[0] == "TypeError" and "unhashable" in str(a[1]) + str(b[1]))


def assert_same_state(a: MinHashLSH, b: Any) -> None:
    assert list(a.keys) == list(b.keys)
    assert len(a.keys) == len(b.keys) == a.keys.size()
    for key in b.keys:
        assert a.keys[key] == b.keys[key]
        assert key in a.keys
        assert a.keys.itemcounts()[key] == len(b.keys[key])
    assert len(a.hashtables) == len(b.hashtables)
    for ta, tb in zip(a.hashtables, b.hashtables, strict=True):
        assert len(ta) == len(tb)
        assert {h: ta[h] for h in ta} == {h: tb[h] for h in tb}
    assert a.get_counts() == b.get_counts()
    assert a.is_empty() == b.is_empty()
    assert a._minhash_scheme == b._minhash_scheme


def run_ops(a: MinHashLSH, b: Any, ops: list[tuple[Any, ...]], scheme: str) -> None:
    for op in ops:
        kind = op[0]
        if kind == "insert":
            _, key, tokens, check = op
            m = sketch(scheme, tokens)
            ra = outcome(a.insert, key, m, check_duplication=check)
            rb = outcome(b.insert, key, m, check_duplication=check)
        elif kind == "remove":
            ra, rb = outcome(a.remove, op[1]), outcome(b.remove, op[1])
        elif kind == "query":
            m = sketch(scheme, op[1])
            ra, rb = outcome(a.query, m), outcome(b.query, m)
            assert outcome(a._query_b, m, op[2]) == outcome(b._query_b, m, op[2])
            a.add_to_query_buffer(m)
            b.add_to_query_buffer(m)
            assert outcome(a.collect_query_buffer) == outcome(b.collect_query_buffer)
        elif kind == "contains":
            ra, rb = outcome(a.__contains__, op[1]), outcome(b.__contains__, op[1])
        elif kind == "subset":
            ra = outcome(a.get_subset_counts, *op[1])
            rb = outcome(b.get_subset_counts, *op[1])
        else:
            raise AssertionError(kind)
        assert same_error_type(ra, rb), (op, ra, rb)
    assert_same_state(a, b)


token_sets = st.lists(st.sampled_from(VOCAB), min_size=1, max_size=4, unique=True).map(
    lambda t: tuple(sorted(t))
)
keys = st.sampled_from(KEYS)
ops_strategy = st.lists(
    st.one_of(
        st.tuples(st.just("insert"), keys, token_sets, st.booleans()),
        st.tuples(st.just("remove"), keys),
        st.tuples(st.just("query"), token_sets, st.integers(-2, 9)),
        st.tuples(st.just("contains"), keys),
        st.tuples(st.just("subset"), st.lists(keys, max_size=3)),
    ),
    max_size=30,
)


@pytest.mark.thread_unsafe(
    reason="MinHashLSH() draws from the global random module (as datasketch does), which "
    "Hypothesis reports when another thread is drawing examples"
)
@given(
    ops=ops_strategy,
    params=st.sampled_from(PARAMS),
    scheme=st.sampled_from(SCHEMES),
    prepickle=st.booleans(),
)
def test_ops_match_datasketch(
    ops: list[tuple[Any, ...]], params: tuple[int, int], scheme: str, prepickle: bool
) -> None:
    a, b = pair(num_perm=NUM_PERM, params=params, prepickle=prepickle)
    run_ops(a, b, ops, scheme)
    a2, b2 = pickle.loads(pickle.dumps(a)), pickle.loads(pickle.dumps(b))
    assert_same_state(a2, b2)
    run_ops(a2, b2, ops[::-1], scheme)


def random_ops(rng: random.Random, n: int) -> list[tuple[Any, ...]]:
    ops: list[tuple[Any, ...]] = []
    for _ in range(n):
        tokens = tuple(sorted(rng.sample(VOCAB, rng.randint(1, 4))))
        roll = rng.random()
        key = rng.choice(KEYS + list(range(40)))
        if roll < 0.5:
            ops.append(("insert", key, tokens, rng.random() < 0.7))
        elif roll < 0.7:
            ops.append(("remove", key))
        elif roll < 0.95:
            ops.append(("query", tokens, rng.randint(1, 8)))
        else:
            ops.append(("contains", key))
    return ops


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("seed", range(4))
def test_seeded_workloads_match_datasketch(scheme: str, seed: int) -> None:
    rng = random.Random(seed)
    params = PARAMS[seed % len(PARAMS)]
    a, b = pair(num_perm=NUM_PERM, params=params, prepickle=seed % 2 == 1)
    run_ops(a, b, random_ops(rng, 600), scheme)
    # Merges: into a non-empty index, with overlaps, and into itself.
    c, d = pair(num_perm=NUM_PERM, params=params, prepickle=seed % 2 == 1)
    run_ops(c, d, random_ops(rng, 200), scheme)
    for check in (True, False):
        assert same_error_type(outcome(a.merge, c, check), outcome(b.merge, d, check))
        assert_same_state(a, b)
    assert outcome(a.merge, a) == outcome(b.merge, b)
    assert_same_state(a, b)
    run_ops(a, b, random_ops(rng, 200), scheme)


def test_default_threshold_params_match() -> None:
    configs: list[dict[str, Any]] = [
        {},
        {"threshold": 0.5},
        {"threshold": 0.8, "weights": (0.2, 0.8)},
    ]
    for kwargs in configs:
        a, b = pair(num_perm=64, **kwargs)
        assert (a.b, a.r, a.h, a.hashranges) == (b.b, b.r, b.h, b.hashranges)


def test_weighted_minhash_states_match() -> None:
    mg = WeightedMinHashGenerator(8, sample_size=NUM_PERM, seed=5)
    rng = np.random.default_rng(5)
    pool = [mg.minhash(rng.integers(1, 3, 8).astype(float)) for _ in range(12)]
    a, b = pair(num_perm=NUM_PERM, params=(4, 4))
    for i in range(80):
        m = pool[int(rng.integers(len(pool)))]
        key = int(rng.integers(30))
        assert same_error_type(outcome(a.insert, key, m), outcome(b.insert, key, m))
        if i % 5 == 0:
            assert outcome(a.remove, key + 1) == outcome(b.remove, key + 1)
    for m in pool:
        assert outcome(a.query, m) == outcome(b.query, m)
    assert_same_state(a, b)
    assert a._minhash_scheme is None


class Fake:
    """A MinHash-like object: ``hashvalues``, ``len()`` and maybe ``scheme``."""

    def __init__(self, hashvalues: Any, n: int | None = None, scheme: str | None = None) -> None:
        self.hashvalues = hashvalues
        self._n = len(hashvalues) if n is None else n
        if scheme is not None:
            self.scheme = scheme

    def __len__(self) -> int:
        return self._n


class SubArray(np.ndarray):
    pass


def odd_hashvalues() -> list[Any]:
    hv = sketch("affine64", (b"t1", b"t2")).hashvalues
    return [
        hv.astype(hv.dtype.newbyteorder(">")),  # non-native byte order
        np.repeat(hv, 2)[::2],  # not contiguous
        hv.view(SubArray),  # an ndarray subclass
        hv.astype(np.float64),
        hv.astype(np.int8),
        hv.astype(np.bool_),
        hv[:10],  # short (len() says 16)
        hv.reshape(8, 2),  # rows of two values
        np.asfortranarray(hv.reshape(8, 2)),
        np.zeros((16, 0), dtype=np.uint64),
        hv.tolist(),  # not an array: datasketch's AttributeError
        np.uint64(5),  # 0-d
    ]


@pytest.mark.parametrize("which", range(12))
def test_odd_hashvalues_match_datasketch(which: int) -> None:
    hv = odd_hashvalues()[which]
    m = Fake(hv, n=NUM_PERM)
    for params in [(4, 4), (5, 3)]:
        a, b = pair(num_perm=NUM_PERM, params=params)
        assert same_error_type(outcome(a.insert, "k", m), outcome(b.insert, "k", m))
        assert same_error_type(
            outcome(a.insert, "j", m, check_duplication=False),
            outcome(b.insert, "j", m, check_duplication=False),
        )
        assert outcome(a.query, m) == outcome(b.query, m)
        assert outcome(a._query_b, m, 2) == outcome(b._query_b, m, 2)
        assert_same_state(a, b)


def test_band_size_that_is_not_an_int() -> None:
    m = sketch("affine32", (b"t1",))
    for params in [(4, 2.0), (4, np.int64(2)), (4, True), (4, 0), (4, -1)]:
        a, b = pair(num_perm=NUM_PERM, params=params)
        assert same_error_type(outcome(a.insert, "k", m), outcome(b.insert, "k", m))
        assert outcome(a.query, m) == outcome(b.query, m)
        assert_same_state(a, b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 1.5},
        {"threshold": -0.1},
        {"num_perm": 1},
        {"weights": (0.2, 0.9)},
        {"weights": (1.5, -0.5)},
        {"num_perm": 16, "params": (5, 4)},
        {"num_perm": 16, "params": (1, 4)},
        {"num_perm": 16, "params": (2.0, 4)},
        {"num_perm": 16, "params": (4, 4), "storage_config": {"type": "mongo"}},
        {"num_perm": 16, "params": (4, 4), "storage_config": {"basename": b"x"}},
        {"num_perm": 16, "params": (4, 4), "storage_config": {"type": "dict", "basename": "s"}},
        {"num_perm": 70000, "params": (65537, 1)},
    ],
)
def test_constructor_errors_match(kwargs: dict[str, Any]) -> None:
    ra = outcome(MinHashLSH, **kwargs)
    rb = outcome(ds.MinHashLSH, **kwargs)
    assert ra[0] == rb[0] != "ok"
    assert ra == rb


@pytest.mark.thread_unsafe(reason="global random state")
def test_constructor_draws_a_name_like_datasketch() -> None:
    for config in [None, {"type": "dict", "basename": b"given"}]:
        random.seed(7)
        MinHashLSH(num_perm=16, params=(4, 4), storage_config=config)
        x = random.random()
        random.seed(7)
        ds.MinHashLSH(num_perm=16, params=(4, 4), storage_config=config)
        assert random.random() == x
    random.seed(3)
    name = _random_name(11)
    random.seed(3)
    assert name == ds.storage._random_name(11)


@pytest.mark.thread_unsafe(reason="resets the once-per-process warning registry")
def test_hashfunc_and_redis_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_common, "_WARNED", set())
    with pytest.warns(pyroid.FallbackWarning, match="MinHashLSH: hashfunc"):
        lsh = MinHashLSH(num_perm=16, params=(4, 4), hashfunc=lambda b: b)
    assert type(lsh) is ds.MinHashLSH
    lsh.insert("a", _mh16(b"a"))
    assert lsh.query(_mh16(b"a")) == ["a"]
    with pytest.warns(pyroid.FallbackWarning, match="'redis'"), redis_patch():
        assert type(MinHashLSH(num_perm=16, params=(4, 4), storage_config=REDIS)) is ds.MinHashLSH


@pytest.mark.thread_unsafe(reason="hides datasketch from sys.modules")
def test_without_datasketch(monkeypatch: pytest.MonkeyPatch) -> None:
    m = _mh16(b"a")
    want = ds.MinHashLSH(threshold=0.5, num_perm=16)
    monkeypatch.setitem(sys.modules, "datasketch", None)
    with pytest.raises(ValueError, match=r"^Unknown storage type: mongo$"):
        MinHashLSH(num_perm=16, params=(4, 4), storage_config={"type": "mongo"})
    lsh = MinHashLSH(num_perm=16, params=(4, 4))
    lsh.insert("a", m)
    assert lsh.query(m) == ["a"]
    searched = MinHashLSH(threshold=0.5, num_perm=16)
    assert (searched.b, searched.r) == (want.b, want.r)
    monkeypatch.setattr(_common, "_WARNED", set())
    with pytest.raises(pyroid.PyroidError, match="datasketch is not installed"):
        MinHashLSH(num_perm=16, params=(4, 4), storage_config=REDIS)


def test_dangling_keys_after_extend_and_remove() -> None:
    # insert(k, m1); insert(k, m2, check_duplication=False); remove(k) leaves k in
    # m2's buckets; inserting an equal key again shares them.
    m1, m2, m3 = (sketch("affine32", (t,)) for t in VOCAB[:3])
    a, b = pair(num_perm=NUM_PERM, params=(4, 4))
    for lsh in (a, b):
        lsh.insert(1, m1)
        lsh.insert(1.0, m2, check_duplication=False)
        assert len(lsh.keys[1]) == 8
        lsh.remove(True)
        assert 1 not in lsh
        assert lsh.query(m2) == [1]
        lsh.insert(1, m3)
        lsh.insert("x", m2)
    assert_same_state(a, b)
    assert set(a.query(m2)) == set(b.query(m2)) == {1, "x"}


def test_storage_views() -> None:
    lsh = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    m = sketch("affine32", (b"t1",))
    lsh.insert("a", m)
    keys, table = lsh.keys, lsh.hashtables[0]
    assert isinstance(keys, DictListStorageView)
    assert isinstance(keys, OrderedStorage)
    assert isinstance(table, DictSetStorageView)
    assert isinstance(table, UnorderedStorage)
    assert keys.keys() == ["a"]
    assert list(keys) == ["a"]
    assert len(keys) == 1
    assert keys.get("missing") == []
    assert keys["missing"] == []
    assert keys.has_key("a")
    assert "a" in keys
    assert "b" not in keys
    assert keys.getmany("a", "b") == [keys["a"], []]
    assert keys.status() == {"keyspace_size": 1}
    assert keys.itemcounts() == {"a": 4}
    h = keys["a"][0]
    assert table.keys() == [h]
    assert table[h] == {"a"}
    assert table.get(b"nope") == set()
    assert table.get("not bytes") == set()
    assert not table.has_key(1)
    with pytest.raises(TypeError):
        table.get([1])
    with pytest.raises(TypeError):
        keys.get([1])  # type: ignore[arg-type]
    snapshot = table[h]
    snapshot.add("b")
    assert table[h] == {"a"}, "get() returns a copy"
    for f in (keys.insert, keys.remove, table.remove_val):
        with pytest.raises(NotImplementedError, match="read-only"):
            f("a", h)
    lsh.buffer_size = 10
    assert lsh.buffer_size == keys.buffer_size == table.buffer_size == 10


def test_storage_module_matches_datasketch() -> None:
    for ours, theirs in [
        (ordered_storage, ds.storage.ordered_storage),
        (unordered_storage, ds.storage.unordered_storage),
    ]:
        s, t = ours({"type": "dict"}), theirs({"type": "dict"})
        for store in (s, t):
            store.insert("k", 1, 2, 2)
            store.insert("j", 3)
            store.remove_val("k", 2)
        assert (list(s), s.itemcounts(), s.get("k"), s.size()) == (
            list(t),
            t.itemcounts(),
            t.get("k"),
            t.size(),
        )
        assert outcome(ours, {"type": "x"}) == outcome(theirs, {"type": "x"})
        assert type(s).__name__ == type(t).__name__
    assert isinstance(ordered_storage({"type": "dict"}), DictListStorage)
    assert isinstance(unordered_storage({"type": "dict"}), DictSetStorage)


# ---------------------------------------------------------------------------
# insert_batch / query_batch (pyroid extensions).
# ---------------------------------------------------------------------------


def docs_for(scheme: str, n: int, seed: int) -> list[Any]:
    rng = random.Random(seed)
    return [sketch(scheme, tuple(sorted(rng.sample(VOCAB, rng.randint(1, 4))))) for _ in range(n)]


def per_item(lsh: MinHashLSH, keys: list[Hashable], docs: list[Any], check: bool = True) -> None:
    for key, m in zip(keys, docs, strict=True):
        lsh.insert(key, m, check_duplication=check)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("threads", [None, 1, 4])
@pytest.mark.parametrize("prepickle", [False, True])
def test_batches_equal_per_item(scheme: str, threads: int | None, prepickle: bool) -> None:
    docs = docs_for(scheme, 300, 1)
    keys: list[Hashable] = [f"k{i}" for i in range(300)]
    one = MinHashLSH(num_perm=NUM_PERM, params=(5, 3), prepickle=prepickle)
    per_item(one, keys, docs)
    many = MinHashLSH(num_perm=NUM_PERM, params=(5, 3), prepickle=prepickle)
    many.insert_batch(keys, docs, threads=threads)
    assert_same_state(many, one)
    arr = MinHashLSH(num_perm=NUM_PERM, params=(5, 3), prepickle=prepickle)
    arr.insert_batch(iter(keys), np.stack([m.hashvalues for m in docs]), threads=threads)
    assert {k: arr.keys[k] for k in arr.keys} == {k: one.keys[k] for k in one.keys}
    want = [set(one.query(m)) for m in docs]
    assert [set(r) for r in many.query_batch(docs, threads=threads)] == want
    stacked = np.stack([m.hashvalues for m in docs])
    for values in (stacked, np.asfortranarray(stacked)):
        assert [set(r) for r in many.query_batch(values, threads=threads)] == want
    # Band keys are the raw bytes reversed (datasketch's byteswap()), so a
    # big-endian copy has other band keys: it must match per-item queries of it.
    swapped = stacked.astype(stacked.dtype.newbyteorder(">"))
    want_swapped = [set(one.query(Fake(row, scheme=scheme))) for row in swapped]
    assert [set(r) for r in many.query_batch(swapped, threads=threads)] == want_swapped
    theirs = ds.MinHashLSH(num_perm=NUM_PERM, params=(5, 3), prepickle=prepickle)
    per_item(theirs, keys, docs)
    assert [set(theirs.query(Fake(row, scheme=scheme))) for row in swapped] == want_swapped


def test_batch_with_repeated_keys_and_no_duplicate_check() -> None:
    docs = docs_for("affine32", 50, 2)
    keys: list[Hashable] = [i % 7 for i in range(50)]
    a, b = pair(num_perm=NUM_PERM, params=(4, 4))
    a.insert_batch(keys, docs, check_duplication=False)
    per_item(b, keys, docs, check=False)
    assert_same_state(a, b)


def test_batch_errors_insert_nothing() -> None:
    docs = docs_for("affine32", 10, 3)
    lsh = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    lsh.insert("old", docs[0])
    cases: list[tuple[list[Any], list[Any], type[Exception], str]] = [
        (["x", "old"], docs[:2], ValueError, "The given key already exists"),
        (["x", "y", "x"], docs[:3], ValueError, "The given key already exists"),
        (
            ["x", "y"],
            [docs[0], MinHash(num_perm=8)],
            ValueError,
            "Expecting minhash with length 16, got 8",
        ),
        (["x", "y"], [docs[0], sketch("legacy", (b"t1",))], ValueError, "does not match scheme"),
        (["x", ["unhashable"]], docs[:2], TypeError, "unhashable"),
        (["x", "y"], [docs[0], Fake([1] * 16)], AttributeError, "byteswap"),
        (["x"], docs[:2], ValueError, "differ in length"),
    ]
    for keys, items, error, match in cases:
        with pytest.raises(error, match=match) as info:
            lsh.insert_batch(keys, items)
        assert list(lsh.keys) == ["old"], (keys, items)
        if "length (" not in str(info.value):
            notes = getattr(info.value, "__notes__", [])
            assert notes == [] or notes[-1].startswith("pyroid: item "), notes
    with pytest.raises(ValueError, match="2-D"):
        lsh.insert_batch(["x"], docs[0].hashvalues)
    with pytest.raises(TypeError, match="numpy array of signatures"):
        lsh.insert_batch(["x"], np.zeros((1, 16), dtype=np.bool_))
    with pytest.raises(ValueError, match="Expecting minhash with length 16, got 8"):
        lsh.query_batch(np.zeros((2, 8), dtype=np.uint32))
    with pytest.raises(ValueError, match="threads must be None or a positive int"):
        lsh.query_batch(docs, threads=0)
    assert lsh._minhash_scheme == "affine32"
    assert list(lsh.keys) == ["old"]


def test_batch_learns_the_scheme_like_per_item() -> None:
    lsh = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    lsh.insert_batch(["a", "b"], [sketch("affine64", (b"t1",)), sketch("affine64", (b"t2",))])
    assert lsh._minhash_scheme == "affine64"
    with pytest.raises(ValueError, match="does not match scheme"):
        lsh.query_batch([sketch("affine32", (b"t1",))])
    arr = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    arr.insert_batch(["a"], np.zeros((1, 16), dtype=np.uint32))
    assert arr._minhash_scheme is None  # rows have no scheme, as WeightedMinHash


@pytest.mark.thread_unsafe(reason="10k signatures of 128 values")
def test_large_batch() -> None:
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2**32, size=(10_000, 128), dtype=np.uint64)
    values[5000:] = values[:5000]  # every signature twice
    lsh = MinHashLSH(threshold=0.9, num_perm=128)
    lsh.insert_batch(range(10_000), values)
    found = lsh.query_batch(values)
    assert all(set(found[i]) == {i % 5000, i % 5000 + 5000} for i in range(10_000))
    lsh.remove(0)
    assert lsh.query_batch(values[:1], threads=1) == [[5000]]


@pytest.mark.thread_unsafe(reason="starts its own threads")
def test_concurrent_writers_and_readers() -> None:
    docs = docs_for("affine32", 400, 4)
    lsh = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    errors: list[BaseException] = []

    def work(t: int) -> None:
        try:
            part = range(t * 100, t * 100 + 100)
            if t % 2:
                lsh.insert_batch(list(part), [docs[i] for i in part])
            else:
                for i in part:
                    lsh.insert(i, docs[i])
                    lsh.query(docs[i])
            lsh.query_batch(docs)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert errors == []
    ref = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    per_item(ref, list(range(400)), docs)
    assert sorted(lsh.keys) == list(range(400))
    for ta, tb in zip(lsh.hashtables, ref.hashtables, strict=True):
        assert {h: ta[h] for h in ta} == {h: tb[h] for h in tb}


def test_reentrant_key_raises_instead_of_deadlocking() -> None:
    lsh = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    m = sketch("affine32", (b"t1",))

    class Evil:
        def __hash__(self) -> int:
            lsh.query(m)
            return 1

    with pytest.raises(RuntimeError, match="called back into"):
        lsh.insert(Evil(), m)
    lsh.insert("fine", m)
    assert lsh.query(m) == ["fine"]


def test_pickle_round_trip_keeps_everything() -> None:
    a, b = pair(num_perm=NUM_PERM, params=(4, 4), prepickle=True)
    run_ops(a, b, random_ops(random.Random(9), 300), "affine64")
    a2 = pickle.loads(pickle.dumps(a, protocol=pickle.HIGHEST_PROTOCOL))
    assert type(a2._index) is type(a._index)
    assert copy.copy(a)._index is a._index
    assert_same_state(copy.deepcopy(a), b)
    assert a2.hashtables[0]._index is a2._index is a2.keys._index
    assert_same_state(a2, b)
    with pytest.raises(ValueError, match="invalid pickled index state"):
        a2._index.__setstate__((b"junk", []))
    state = a._index.__getstate__()
    with pytest.raises(ValueError, match="keys for"):
        a2._index.__setstate__((state[0], state[1][:-1]))
    assert_same_state(a2, b)


def test_param_search_matches_datasketch() -> None:
    dsl = ds.lsh
    for threshold, num_perm, weights in [
        (0.9, 128, (0.5, 0.5)),
        (0.5, 64, (0.3, 0.7)),
        (0.1, 16, (0.9, 0.1)),
        (1.0, 2, (0.5, 0.5)),
    ]:
        assert _optimal_param(threshold, num_perm, *weights) == dsl._optimal_param(
            threshold, num_perm, *weights
        )
    for b, r in [(1, 1), (20, 5), (3, 7)]:
        for ours, theirs in [
            (_false_positive_probability, dsl._false_positive_probability),
            (_false_negative_probability, dsl._false_negative_probability),
        ]:
            assert ours(0.6, b, r) == pytest.approx(theirs(0.6, b, r), rel=1e-12, abs=1e-15)
    assert outcome(_optimal_param, 0.5, 16.0, 0.5, 0.5) == outcome(
        dsl._optimal_param, 0.5, 16.0, 0.5, 0.5
    )
    assert _optimal_param(0.5, 0, 0.5, 0.5) == dsl._optimal_param(0.5, 0, 0.5, 0.5) == (0, 0)
    assert outcome(MinHashLSH, num_perm=16.0) == outcome(ds.MinHashLSH, num_perm=16.0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_pyroid_minhash_sketches(scheme: str) -> None:
    rng = random.Random(5)
    docs: list[list[bytes | str]] = [list(rng.sample(VOCAB, rng.randint(1, 4))) for _ in range(200)]
    ours = PyroidMinHash.bulk(docs, num_perm=NUM_PERM, scheme=scheme, seed=3)
    theirs = MinHash.bulk(docs, num_perm=NUM_PERM, scheme=scheme, seed=3)
    a, b = pair(num_perm=NUM_PERM, params=(4, 4))
    for i, (m, n) in enumerate(zip(ours, theirs, strict=True)):
        assert same_error_type(outcome(a.insert, i % 150, m), outcome(b.insert, i % 150, n))
    assert_same_state(a, b)
    want = [set(b.query(n)) for n in theirs]
    assert [set(a.query(m)) for m in ours] == want
    c = MinHashLSH(num_perm=NUM_PERM, params=(4, 4))
    c.insert_batch(range(150), ours[:150])
    signatures = minhash_batch(docs, num_perm=NUM_PERM, seed=3, scheme=cast(Any, scheme))
    assert [set(r) for r in c.query_batch(signatures)] == want
    assert [set(r) for r in c.query_batch(ours)] == want
