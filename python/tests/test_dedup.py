# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
"""pyroid.dedup on the installed wheel: kinds, Arrow, threads, errors, streaming, and stages 1-4
against a line-by-line Python transcription of datatrove 0.10.1's MinHash code.

The transcription (``ref_*`` below) follows ``datatrove/pipeline/dedup/minhash.py``; it takes the
permutation parameters from pyroid (they are checked against numpy in
``crates/pyroid-core/tests/dedup_minhash.rs``) and the words as given, so it is independent of
the tokenizer. ``test_dedup_parity.py`` and ``test_dedup_datatrove.py`` compare against datatrove
itself.
"""

from __future__ import annotations

import hashlib
import random
import struct
import sys
import sysconfig
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.dedup as dedup
from pyroid._pyroid import _testing

P = (1 << 61) - 1
M64 = (1 << 64) - 1
SMALL = dedup.MinhashConfig(
    n_grams=2, num_buckets=4, hashes_per_bucket=2, hash_config=dedup.HashConfig(64, "sha1")
)
FINEWEB = dedup.MinhashConfig(hash_config=dedup.HashConfig(64, "sha1"))
# Words spaCy splits exactly like str.split() after simplify_text (no exceptions, no affixes).
WORDS = [
    "alpha", "beta", "gamma", "delta", "omega", "river", "stone", "cloud",
    "paper", "green", "seven", "north", "quiet", "table", "light", "zebra",
]  # fmt: skip
nice = settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


# ------------------------------------------------------------------ datatrove transcription


def ref_hash(cfg: dedup.MinhashConfig, s: str) -> int:
    assert cfg.hash_config.hash_fc == "sha1"
    d = hashlib.sha1(s.encode("utf-8")).digest()
    if cfg.hash_config.precision == 32:
        return int(struct.unpack("<I", d[:4])[0])
    return int(struct.unpack("<Q", d[:8])[0])


def ref_signature(cfg: dedup.MinhashConfig, words: list[str]) -> list[int] | None:
    """``get_shingles`` + ``get_signature`` (flattened); ``None`` without shingles."""
    n = cfg.n_grams
    shingles = [ref_hash(cfg, " ".join(words[i : i + n])) for i in range(len(words) - n + 1)]
    if not shingles:
        return None
    a, b = dedup._native_config(cfg).parameters()
    mask = 0xFFFFFFFF if cfg.hash_config.precision == 32 else M64
    return [
        min((((s * ak + bk) & M64) % P) & mask for s in shingles)
        for ak, bk in zip(a, b, strict=True)
    ]


def ref_pairs(
    cfg: dedup.MinhashConfig, sigs: list[list[int] | None]
) -> list[list[tuple[int, int]]]:
    """Stage 1's sorted bucket files + stage 2's consecutive-equal pairs, per bucket."""
    r = cfg.hashes_per_bucket
    stop = (1 << 32) - 1 if cfg.hash_config.precision == 32 else P  # read_sigs' max_hash
    out = []
    for b in range(cfg.num_buckets):
        recs = sorted((tuple(s[b * r : (b + 1) * r]), i) for i, s in enumerate(sigs) if s)
        pairs: list[tuple[int, int]] = []
        last = None
        for sig, i in recs:
            if sig[0] >= stop:
                break
            if last is not None and last[0] == sig:
                pairs.append((last[1], i))
            last = (sig, i)
        out.append(pairs)
    return out


def ref_cluster(
    n: int, pairs: list[list[tuple[int, int]]]
) -> tuple[list[bool], list[int], list[int]]:
    """``MinhashDedupCluster.run``: keep mask, cluster ids, cluster sizes."""
    union_set: dict[int, int] = {}
    set_size: dict[int, int] = {}

    def parent(x: int) -> int:
        if x not in union_set or union_set[x] == x:
            union_set[x] = x
            return x
        union_set[x] = parent(union_set[x])
        return union_set[x]

    def union(v_a: int, v_b: int) -> None:
        root_a, root_b = parent(v_a), parent(v_b)
        if root_a != root_b:
            size_a, size_b = set_size.get(root_a, 1), set_size.get(root_b, 1)
            if size_a < size_b:
                root_a, root_b = root_b, root_a
            union_set[root_b] = root_a
            set_size[root_a] = size_a + size_b
            set_size.pop(root_b, None)

    for bucket in pairs:
        for a, b in bucket:
            union(a, b)
    keep, ids, sizes = [True] * n, [-1] * n, [1] * n
    cluster_ids: dict[int, int] = {}
    for node in sorted(union_set):
        p = parent(node)
        keep[node] = node == p
        ids[node] = cluster_ids.setdefault(p, len(cluster_ids))
        sizes[node] = set_size[p]
    return keep, ids, sizes


# ------------------------------------------------------------------ strategies


@st.composite
def families(draw: st.DrawFn) -> list[list[str]]:
    """Several near-duplicate families: edits, reorderings, drops, inserts, exact copies."""
    docs: list[list[str]] = []
    for _ in range(draw(st.integers(1, 4))):
        base = draw(st.lists(st.sampled_from(WORDS), max_size=30))
        docs.append(base)
        for _ in range(draw(st.integers(0, 6))):
            v = list(draw(st.sampled_from(docs)))
            op = draw(st.sampled_from(["edit", "swap", "drop", "insert", "copy"]))
            if v and op != "copy":
                i = draw(st.integers(0, len(v) - 1))
                if op == "edit":
                    v[i] = draw(st.sampled_from(WORDS))
                elif op == "swap":
                    j = draw(st.integers(0, len(v) - 1))
                    v[i], v[j] = v[j], v[i]
                elif op == "drop":
                    del v[i]
                else:
                    v.insert(i, draw(st.sampled_from(WORDS)))
            docs.append(v)
    return draw(st.permutations(docs))


configs = st.builds(
    dedup.MinhashConfig,
    n_grams=st.integers(1, 4),
    num_buckets=st.integers(1, 6),
    hashes_per_bucket=st.integers(1, 3),
    seed=st.sampled_from([0, 1, 7, 2**32 - 1]),
    hash_config=st.builds(dedup.HashConfig, st.sampled_from([32, 64]), st.just("sha1")),
)


# ------------------------------------------------------------------ stages vs the transcription


@nice
@given(docs=families(), cfg=configs)
def test_stages_match_datatrove_transcription(
    docs: list[list[str]], cfg: dedup.MinhashConfig
) -> None:
    sigs = [ref_signature(cfg, d) for d in docs]
    pairs = ref_pairs(cfg, sigs)
    keep, ids, sizes = ref_cluster(len(docs), pairs)

    d = dedup.MinHashDedup(cfg)
    assert d._state._add_tokens(docs) == (0, len(docs))
    assert [d.signature(i) for i in range(len(docs))] == sigs
    assert [list(map(tuple, d._state._bucket_pairs(b))) for b in range(cfg.num_buckets)] == pairs
    assert d.keep_mask() == keep
    assert d.cluster_ids() == ids
    assert d.cluster_sizes() == sizes
    assert d.duplicates() == [i for i, k in enumerate(keep) if not k]
    reps = d.representatives()
    for c in d.clusters():
        assert keep[c[0]]
        assert not any(keep[i] for i in c[1:])
        assert c[1:] == sorted(c[1:])
        assert all(reps[i] == c[0] for i in c)


@nice
@given(docs=families())
def test_full_pipeline_on_plain_words(docs: list[list[str]]) -> None:
    texts = [" ".join(d) for d in docs]
    want = [ref_signature(SMALL, d) for d in docs]
    assert dedup.minhash_signatures(texts, SMALL) == want
    keep, _, _ = ref_cluster(len(docs), ref_pairs(SMALL, want))
    assert dedup.minhash_dedup(texts, SMALL) == keep


def test_identical_documents_keep_the_first() -> None:
    doc = "the quiet river runs past the old stone mill"
    assert dedup.minhash_dedup([doc, "x", doc, doc], FINEWEB) == [True, True, False, False]


def test_kept_document_can_be_a_later_one() -> None:
    """Union by size: the root of a larger side beats an earlier document of the pair."""
    cfg = dedup.MinhashConfig(
        n_grams=1, num_buckets=6, hashes_per_bucket=1, hash_config=SMALL.hash_config
    )
    found = 0
    for seed in range(200):
        rng = random.Random(seed)
        docs = [rng.choices(WORDS[:6], k=rng.randint(1, 3)) for _ in range(12)]
        d = dedup.MinHashDedup(cfg)
        d._state._add_tokens(docs)
        sigs = [ref_signature(cfg, x) for x in docs]
        keep, _, _ = ref_cluster(len(docs), ref_pairs(cfg, sigs))
        assert d.keep_mask() == keep
        found += any(c[0] != min(c) for c in d.clusters())
    assert found > 0


def test_short_documents_have_no_signature_and_are_kept() -> None:
    assert dedup.minhash_signatures("one two three four", FINEWEB) is None
    assert dedup.minhash_signatures(["", "a b c d e"], FINEWEB)[0] is None
    assert dedup.minhash_dedup(["a b", "a b", "", ""], FINEWEB) == [True] * 4


# ------------------------------------------------------------------ API: kinds, Arrow, config


DOCS = [
    "the quiet river runs past the old stone mill tonight",
    "short",
    "the quiet river runs past the old stone mill tonight",
    "an entirely different document about green paper clouds",
    "the quiet river runs past the old stone mill tonight",
]
KEEP = [True, True, False, True, False]


def test_container_kinds() -> None:
    assert dedup.minhash_dedup(DOCS, FINEWEB) == KEEP
    out = dedup.minhash_dedup(tuple(DOCS), FINEWEB)
    assert type(out) is tuple
    assert list(out) == KEEP
    assert dedup.minhash_dedup(DOCS[0], FINEWEB) is True
    assert dedup.exact_dedup(DOCS) == KEEP
    assert dedup.exact_dedup(tuple(DOCS)) == tuple(KEEP)
    assert dedup.exact_dedup("x") is True
    sigs = dedup.minhash_signatures(DOCS, FINEWEB)
    assert type(sigs) is list
    assert sigs[1] is None
    assert sigs[0] is not None
    assert len(sigs[0]) == 112
    assert dedup.minhash_signatures(tuple(DOCS), FINEWEB) == sigs
    assert dedup.minhash_signatures(DOCS[0], FINEWEB) == sigs[0]


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 2, 5])
def test_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    values: list[str | None] = [*DOCS, None, DOCS[0]]
    col = _testing.column(values, large=large, chunks=chunks, name="text")
    for out in (dedup.minhash_dedup(col, FINEWEB), dedup.exact_dedup(col)):
        assert isinstance(out, pyroid.Column)
        assert out.type == "bool"
        assert out.name == "text"
        assert out.num_chunks == col.num_chunks
        assert out.to_pylist() == [*KEEP, None, False]
    sigs = dedup.minhash_signatures(col, FINEWEB)
    assert sigs[5] is None
    assert sigs[:5] == dedup.minhash_signatures(DOCS, FINEWEB)


def test_pyarrow_inputs(pa: Any) -> None:
    arr = pa.chunked_array([DOCS[:2], [*DOCS[2:], None]], type=pa.string_view())
    out = pa.chunked_array(dedup.minhash_dedup(arr, FINEWEB))
    assert out.to_pylist() == [*KEEP, None]
    sliced = pa.array(DOCS, pa.large_string())[1:]
    assert dedup.exact_dedup(sliced).to_pylist() == [True, True, True, False]
    assert dedup.simplify_text(pa.array(["A  b"])).to_pylist() == dedup.simplify_text(["A  b"])


def test_simplify_text_kinds_and_identity() -> None:
    plain = "already simple"
    out = dedup.simplify_text([plain, "x"])
    assert out[0] is plain
    assert dedup.simplify_text(plain) is plain
    assert type(dedup.simplify_text((plain,))) is tuple


def test_config_forms_agree() -> None:
    want = dedup.minhash_signatures(DOCS, FINEWEB)
    datatrove_like = SimpleNamespace(
        n_grams=5,
        num_buckets=14,
        hashes_per_bucket=8,
        seed=1,
        norm_config=SimpleNamespace(**vars(dedup.TextNormConfig())),
        hash_config=SimpleNamespace(precision=64, hash_fc="sha1"),
    )
    assert dedup.minhash_signatures(DOCS, datatrove_like) == want
    assert dedup.minhash_signatures(DOCS, hash_fc="sha1") == want
    assert dedup.minhash_signatures(DOCS, {"hash_config": {"hash_fc": "sha1"}}) == want
    assert dedup.minhash_signatures(DOCS, dedup.MinhashConfig(), hash_fc="sha1") == want
    assert dedup.minhash_signatures(DOCS) != want  # the default is xxhash
    assert dedup.MinHashDedup(hash_fc="sha1").config == FINEWEB


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"n_grams": 0}, ValueError),
        ({"num_buckets": 0}, ValueError),
        ({"hashes_per_bucket": -1}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": 2**32}, ValueError),
        ({"seed": 1.5}, TypeError),
        ({"n_grams": True}, TypeError),
        ({"precision": 16}, ValueError),
        ({"hash_fc": "md5"}, ValueError),
        ({"bogus": 1}, TypeError),
    ],
)
def test_bad_configs(kwargs: dict[str, Any], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        dedup.minhash_signatures("a b c d e", **kwargs)
    with pytest.raises(exc):
        dedup.MinHashDedup(**kwargs)


def test_bad_inputs() -> None:
    bads: list[Any] = [b"bytes", 3, ["ok", 3], None]
    for bad in bads:
        with pytest.raises(pyroid.InputError):
            dedup.minhash_dedup(bad)
        with pytest.raises(pyroid.InputError):
            dedup.exact_dedup(bad)
    with pytest.raises(TypeError):
        dedup.minhash_signatures(DOCS, config=42)
    with pytest.raises(pyroid.InputError):
        dedup.MinHashDedup()._state._add_tokens([["a", 1]])


@given(st.text(st.characters(categories=["Cs"]), min_size=1))
def test_lone_surrogates_raise(s: str) -> None:
    for f in (
        dedup.minhash_signatures,
        dedup.minhash_dedup,
        dedup.exact_dedup,
        dedup.simplify_text,
    ):
        with pytest.raises(UnicodeEncodeError):
            f(["fine", "x " + s])


# ------------------------------------------------------------------ streaming


def test_batches_equal_one_fit() -> None:
    docs = [" ".join(WORDS[i % 7 : i % 7 + 6]) for i in range(300)]
    whole = dedup.MinHashDedup(SMALL)
    keep = whole.fit(docs)
    d = dedup.MinHashDedup(SMALL)
    assert d.add(docs[:100]) == range(100)
    assert d.add(tuple(docs[100:250])) == range(100, 250)
    assert d.add(_testing.column(docs[250:], chunks=3)) == range(250, 300)
    assert len(d) == 300
    assert d.keep_mask() == keep
    assert d.clusters() == whole.clusters()
    assert d.cluster_ids() == whole.cluster_ids()
    # Decisions are global: a later batch can change earlier ones.
    d.add(docs)
    assert len(d) == 600
    assert d.keep_mask()[:300] == keep
    assert not any(d.keep_mask()[300:])
    d.reset()
    assert len(d) == 0
    assert d.keep_mask() == []
    with pytest.raises(IndexError):
        d.signature(0)


def test_fit_replaces_the_index() -> None:
    d = dedup.MinHashDedup(FINEWEB)
    d.add(DOCS)
    assert d.fit(DOCS[:2]) == KEEP[:2]
    assert len(d) == 2


def test_exact_streaming_equals_one_call() -> None:
    docs = [f"doc {i % 37}" for i in range(500)]
    want = dedup.exact_dedup(docs)
    assert want == [i < 37 for i in range(500)]
    e = dedup.ExactDedup()
    got = [*e.add(docs[:123]), *e.add(tuple(docs[123:400])), *e.add(docs[400:])]
    assert got == want
    assert len(e) == 37
    e.reset()
    assert len(e) == 0
    assert e.add(docs[:1]) == [True]


def test_exact_normalize_options() -> None:
    docs = ["Hello,   World!", "hello world", "HELLO WORLD?", "other"]
    assert dedup.exact_dedup(docs) == [True, True, True, True]
    assert dedup.exact_dedup(docs, normalize=True) == [True, False, False, True]
    no_punct = dedup.TextNormConfig(remove_punctuation=False)
    assert dedup.exact_dedup(docs, normalize=no_punct) == [True, True, True, True]
    assert dedup.exact_dedup(docs[1:3], normalize={"lowercase": True}) == [True, False]
    assert dedup.ExactDedup(normalize=True).add(docs) == [True, False, False, True]


# ------------------------------------------------------------------ threads


@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_threads_give_identical_results(threads: int | None) -> None:
    docs = [" ".join(WORDS[(i * 7) % 11 : (i * 7) % 11 + 5]) + f" n{i % 50}" for i in range(2000)]
    assert dedup.minhash_signatures(docs, SMALL, threads=threads) == dedup.minhash_signatures(
        docs, SMALL, threads=1
    )
    assert dedup.minhash_dedup(docs, SMALL, threads=threads) == dedup.minhash_dedup(
        docs, SMALL, threads=1
    )
    assert dedup.exact_dedup(docs, threads=threads) == dedup.exact_dedup(docs, threads=1)


@pytest.mark.parametrize("bad", [0, -1])
def test_bad_thread_counts(bad: int) -> None:
    with pytest.raises(ValueError, match="threads"):
        dedup.minhash_dedup(DOCS, threads=bad)
    with pytest.raises(ValueError, match="threads"):
        dedup.MinHashDedup(threads=bad)


@pytest.mark.thread_unsafe(reason="100k signatures: heavy per thread")
@pytest.mark.timeout(120)
def test_large_batch_completes() -> None:
    def word(k: int) -> str:  # digits would all simplify to "0"
        return "".join(chr(97 + (k // 26**p) % 26) for p in range(4))

    distinct = [" ".join(word(k * 8 + j) for j in range(8)) for k in range(1000)]
    docs = [distinct[i % 1000] for i in range(100_000)]
    keep = dedup.minhash_dedup(docs, FINEWEB)
    assert keep == [i < 1000 for i in range(100_000)]


@pytest.mark.thread_unsafe(reason="spawns its own threads")
def test_concurrent_adds() -> None:
    """Several threads add to one index; the cluster structure (not the roots) is order-free."""
    docs = [
        f"{WORDS[i % 16]} {WORDS[(i * 3) % 16]} common tail text here {i % 40}" for i in range(4000)
    ]
    d = dedup.MinHashDedup(FINEWEB)
    parts = [docs[i::8] for i in range(8)]
    ts = [threading.Thread(target=d.add, args=(p,)) for p in parts]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(d) == len(docs)
    assert len(d.duplicates()) == len(docs) - sum(dedup.MinHashDedup(FINEWEB).fit(docs))


@pytest.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="free-threaded only")
def test_gil_stays_disabled() -> None:
    dedup.minhash_dedup(DOCS, FINEWEB)
    dedup.ExactDedup().add(DOCS)
    assert not sys._is_gil_enabled()  # type: ignore[attr-defined,unused-ignore]
