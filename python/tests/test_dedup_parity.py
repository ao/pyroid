"""pyroid.dedup against recorded datatrove 0.10.1 decisions on the FineWeb dev slice.

Inputs (skipped when absent):

- ``/data/parity/minhash-ref.parquet``: the harness's pinned datatrove signatures and removed set
  (sha1/64, 14x8, 5-grams) over FineWeb dev (10k) + the adversarial set (13,232 docs).
- ``$PYROID_DEDUP_ORACLE`` (default ``/target/oracle``): files written by
  ``crates/pyroid-core/tests/dedup_data/oracle.py`` in the oracle env: ``tokens.parquet``
  (datatrove's ``simplify_text`` output and spaCy word tokens per doc) and
  ``stages-<hash>-<bits>.json`` (stage-2 pairs per bucket, removed ids, cluster ids and sizes).

The spaCy tokens let stages 1-4 be checked exactly before pyroid's own spaCy-exact tokenizer exists.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import pyroid.dedup as dedup
from pyroid._pyroid import dedup as native

pq = pytest.importorskip("pyarrow.parquet")

DATA = Path(os.environ.get("PYROID_DATA", "/data"))
ORACLE = Path(os.environ.get("PYROID_DEDUP_ORACLE", "/target/oracle"))
REF = DATA / "parity" / "minhash-ref.parquet"
CORPORA = [DATA / "fineweb" / "fineweb-dev-10k.parquet", DATA / "parity" / "adversarial.parquet"]
FINEWEB = dedup.MinhashConfig(hash_config=dedup.HashConfig(precision=64, hash_fc="sha1"))


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"{p} not found (see the module docstring)")
    return p


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return pq.read_table(_need(REF)).to_pydict()  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def corpus(ref: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    texts: list[str] = []
    for c in CORPORA:
        t = pq.read_table(_need(c), columns=["id", "text"]).to_pydict()
        ids += t["id"]
        texts += t["text"]
    assert ids == ref["id"]
    return texts


@pytest.fixture(scope="module")
def tokens(ref: dict[str, Any]) -> dict[str, Any]:
    t = pq.read_table(_need(ORACLE / "tokens.parquet")).to_pydict()
    assert t["id"] == ref["id"]
    return t  # type: ignore[no-any-return]


def _native_cfg(cfg: dedup.MinhashConfig) -> Any:
    return dedup._native_config(cfg)


def test_signatures_from_spacy_tokens(ref: dict[str, Any], tokens: dict[str, Any]) -> None:
    got = native._signatures_from_tokens(tokens["tokens"], _native_cfg(FINEWEB))
    bad = [i for i, (g, w) in enumerate(zip(got, ref["signature"], strict=True)) if g != w]
    assert not bad, f"{len(bad)}/{len(got)} signatures differ, first {bad[:5]}"
    assert sum(s is None for s in got) == sum(s is None for s in ref["signature"]) > 0


def test_removed_set_from_spacy_tokens(ref: dict[str, Any], tokens: dict[str, Any]) -> None:
    d = dedup.MinHashDedup(FINEWEB)
    assert d._state._add_tokens(tokens["tokens"]) == (0, len(ref["id"]))
    want = [i for i, r in enumerate(ref["removed"]) if r]
    assert d.duplicates() == want
    assert d.keep_mask() == [not r for r in ref["removed"]]
    assert len(want) == 445


def test_simplify_text_matches_datatrove(corpus: list[str], tokens: dict[str, Any]) -> None:
    got = dedup.simplify_text(corpus)
    bad = [i for i, (g, w) in enumerate(zip(got, tokens["simplified"], strict=True)) if g != w]
    assert not bad, f"{len(bad)} docs differ, first {bad[:5]}"


def test_full_pipeline_where_spacy_equals_split(
    ref: dict[str, Any], corpus: list[str], tokens: dict[str, Any]
) -> None:
    """End to end from raw text, where spaCy's tokens equal the whitespace stand-in's."""
    got = dedup.minhash_signatures(corpus, FINEWEB)
    same_tokens = [
        w == s.split() for w, s in zip(tokens["tokens"], tokens["simplified"], strict=True)
    ]
    checked = [i for i, ok in enumerate(same_tokens) if ok]
    bad = [i for i in checked if got[i] != ref["signature"][i]]
    assert not bad, f"{len(bad)}/{len(checked)} signatures differ, first {bad[:5]}"
    assert len(checked) > 9000


def _stage_files() -> list[Path]:
    return sorted(ORACLE.glob("stages-*.json")) if ORACLE.exists() else []


@pytest.mark.parametrize("path", _stage_files(), ids=lambda p: p.stem)
def test_stages_match_datatrove(path: Path, tokens: dict[str, Any]) -> None:
    rec = json.loads(path.read_text())
    cfg = dedup.MinhashConfig(
        hash_config=dedup.HashConfig(
            precision=rec["config"]["precision"], hash_fc=rec["config"]["hash_fc"]
        )
    )
    d = dedup.MinHashDedup(cfg)
    d._state._add_tokens(tokens["tokens"])
    assert len(d) == rec["n"]
    for b, want in enumerate(rec["pairs"]):
        assert [list(p) for p in d._state._bucket_pairs(b)] == want, f"bucket {b}"
    assert d.duplicates() == rec["removed"]
    ids = [-1] * rec["n"]
    for doc, cid in rec["cluster_ids"]:
        ids[doc] = cid
    assert d.cluster_ids() == ids
    sizes = [1] * rec["n"]
    for doc, size in rec["cluster_sizes"]:
        sizes[doc] = size
    assert d.cluster_sizes() == sizes
    # The kept document of each cluster is the one not in datatrove's .remove file.
    removed = set(rec["removed"])
    for c in d.clusters():
        assert c[0] not in removed
        assert all(x in removed for x in c[1:])
