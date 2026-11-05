"""pyroid.dedup against live datatrove 0.10.1 (only where datatrove and spaCy are installed).

Run in the parity oracle env with the pyroid wheel installed, e.g.::

    /target/venv-parity/bin/python -m pytest --import-mode=importlib \
        python/tests/test_dedup_datatrove.py

Words come from datatrove's own tokenizer (spaCy) and are fed to pyroid's stages, so stages 1-4 are
checked exactly whatever pyroid's tokenizer is; ``simplify_text`` is checked on raw Unicode text.
"""

# The inputs are full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid.dedup as dedup

pytest.importorskip("spacy")
dt_minhash: Any = pytest.importorskip("datatrove.pipeline.dedup.minhash")
dt_text: Any = pytest.importorskip("datatrove.utils.text")
dt_hashing: Any = pytest.importorskip("datatrove.utils.hashing")
Document: Any = pytest.importorskip("datatrove.data").Document

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
slow = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

HASHES = [("sha1", 64), ("sha1", 32), ("xxhash", 64), ("xxhash", 32)]
WORDS = [
    "The", "quiet", "river", "don't", "can't", "I'm", "£100", "3.5", "12,000", "café", "naïve",
    "ΣΟΦΙΑ", "straße", "U.S.", "e-mail", "Monday", "(see", "below)", "—", "50%", "#tag", "x²",
    "日本語", "😀", "İstanbul", "co-op", "rock'n'roll", "www.example.com", "a/b", "‘quoted’",
]  # fmt: skip


def dt_config(hash_fc: str, precision: int, **kw: Any) -> Any:
    return dt_minhash.MinhashConfig(
        hash_config=dt_hashing.HashConfig(hash_fc=hash_fc, precision=precision), **kw
    )


def pyroid_config(hash_fc: str, precision: int, **kw: Any) -> dedup.MinhashConfig:
    return dedup.MinhashConfig(hash_config=dedup.HashConfig(precision, hash_fc), **kw)


@st.composite
def text_families(draw: st.DrawFn) -> list[str]:
    """Near-duplicate documents with case, whitespace, punctuation and word edits."""
    docs: list[str] = []
    for _ in range(draw(st.integers(1, 3))):
        words = draw(st.lists(st.sampled_from(WORDS), max_size=25))
        docs.append(" ".join(words))
        for _ in range(draw(st.integers(0, 4))):
            w = list(words)
            op = draw(st.sampled_from(["upper", "space", "punct", "edit", "swap"]))
            if w:
                i = draw(st.integers(0, len(w) - 1))
                if op == "upper":
                    w[i] = w[i].upper()
                elif op == "punct":
                    w[i] += draw(st.sampled_from([",", "!", "...", "?!", ";", "»"]))
                elif op == "edit":
                    w[i] = draw(st.sampled_from(WORDS))
                elif op == "swap":
                    j = draw(st.integers(0, len(w) - 1))
                    w[i], w[j] = w[j], w[i]
            sep = draw(st.sampled_from([" ", "  ", "\n", " \t ", " "])) if op == "space" else " "
            docs.append(sep.join(w))
    return draw(st.permutations(docs))


def dt_tokens(step: Any, text: str) -> list[str]:
    words: list[str] = step.word_tokenizer.word_tokenize(
        dt_text.simplify_text(text, step.config.norm_config)
    )
    return words


def dt_signature(step: Any, text: str) -> list[int] | None:
    sh = step.get_shingles(text)
    return None if sh.size == 0 else [int(x) for b in step.get_signature(sh) for x in b]


def dt_stages(texts: list[str], cfg: Any) -> dict[str, Any]:
    """datatrove's stages 1-3 run in process; signatures, removed ids, cluster ids and sizes."""
    with tempfile.TemporaryDirectory(prefix="pyroid-dt-") as w:
        sig = dt_minhash.MinhashDedupSignature(output_folder=f"{w}/s", config=cfg)
        sig.run(iter([Document(text=t, id=str(i)) for i, t in enumerate(texts)]), 0, 1)
        for b in range(cfg.num_buckets):
            dt_minhash.MinhashDedupBuckets(
                input_folder=f"{w}/s", output_folder=f"{w}/b", config=cfg
            ).run(None, b, cfg.num_buckets)
        dt_minhash.MinhashDedupCluster(
            input_folder=f"{w}/b",
            output_folder=f"{w}/r",
            config=cfg,
            save_cluster_id=True,
            save_cluster_size=True,
        ).run(None, 0, 1)

        def read(name: str, fmt: str) -> list[Any]:
            p = Path(w) / "r" / name
            return list(struct.iter_unpack(fmt, p.read_bytes())) if p.exists() else []

        ids, sizes = [-1] * len(texts), [1] * len(texts)
        for d, c in read("000000.clusters", "<2I"):
            ids[d] = c
        for d, s in read("000000.sizes", "<2I"):
            sizes[d] = s
        return {
            "tokens": [dt_tokens(sig, t) for t in texts],
            "signatures": [dt_signature(sig, t) for t in texts],
            "removed": [d for (d,) in read("000000.remove", "<I")],
            "cluster_ids": ids,
            "cluster_sizes": sizes,
        }


@slow
@given(texts=text_families(), hc=st.sampled_from(HASHES), n=st.sampled_from([1, 2, 5]))
def test_stages_match_live_datatrove(texts: list[str], hc: tuple[str, int], n: int) -> None:
    kw: dict[str, Any] = {"n_grams": n, "num_buckets": 6, "hashes_per_bucket": 2}
    want = dt_stages(texts, dt_config(*hc, **kw))
    d = dedup.MinHashDedup(pyroid_config(*hc, **kw))
    d._state._add_tokens(want["tokens"])
    assert [d.signature(i) for i in range(len(texts))] == want["signatures"]
    assert d.duplicates() == want["removed"]
    assert d.cluster_ids() == want["cluster_ids"]
    assert d.cluster_sizes() == want["cluster_sizes"]


@pytest.mark.parametrize("hc", HASHES)
def test_default_bands_on_families(hc: tuple[str, int]) -> None:
    """datatrove's default 14x8 bands, on one fixed batch of families."""
    texts = [
        "The quiet river runs past the old stone mill, slowly and without a sound.",
        "the quiet river runs past the old stone mill slowly and without a sound",
        "THE QUIET RIVER RUNS PAST THE OLD STONE MILL; SLOWLY AND WITHOUT A SOUND!",
        "The quiet river runs past the old stone mill, slowly and without any sound.",
        "Something else entirely: a café in İstanbul serves 12,000 cups for £100.",
        "Something else entirely — a cafe in Istanbul serves 9,999 cups for £100!",
    ]
    want = dt_stages(texts, dt_config(*hc))
    d = dedup.MinHashDedup(pyroid_config(*hc))
    d._state._add_tokens(want["tokens"])
    assert d.duplicates() == want["removed"]
    assert d.cluster_ids() == want["cluster_ids"]
    assert want["removed"]  # the families really are near-duplicates


norm_configs = st.builds(
    dedup.TextNormConfig,
    lowercase=st.booleans(),
    norm_whitespace=st.booleans(),
    remove_punctuation=st.booleans(),
    norm_unicode_diacritics=st.booleans(),
    norm_numbers=st.booleans(),
    norm_weekdays=st.booleans(),
    norm_monthnames=st.booleans(),
)
unicode_text = st.text(
    st.one_of(
        st.characters(exclude_categories=["Cs"]),
        st.sampled_from("ΣσςİIi0123456789١٢٣.,،٫ \t\n ́̈!?;:'\"’—«»"),
    ),
    max_size=60,
)


@settings(max_examples=500, deadline=None)
@given(text=unicode_text, cfg=norm_configs)
def test_simplify_text_matches_live_datatrove(text: str, cfg: dedup.TextNormConfig) -> None:
    want = dt_text.simplify_text(text, dt_text.TextNormConfig(**vars(cfg)))
    assert dedup.simplify_text(text, cfg) == want
