"""datatrove's sentence dedup (stages 1-3) vs pyroid.dedup.sentence_dedup, live, on the dev slice.

datatrove 0.10.1 runs its three steps through ``run()`` (one task, one finder) in a temporary folder;
the kept documents and their texts must be the same. Recorded under the ``dedup`` suite as
``sentence_dedup/<config>/dev``; needs datatrove and spaCy (the parity ``oracle`` group).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pyroid_parity.plugin import need_corpus

pytestmark = pytest.mark.dedup

CASES: dict[str, dict[str, Any]] = {
    # datatrove's examples/sentence_deduplication.py settings, with its default hash (xxhash/64).
    "example_xxhash64": {"hash_fc": "xxhash", "precision": 64},
    "lines_sha1_32": {"split_sentences": False, "hash_fc": "sha1", "precision": 32},
}


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("case", list(CASES))
def test_sentence_dedup(case: str, parity: Any, tmp_path: Path) -> None:
    sd = pytest.importorskip("datatrove.pipeline.dedup.sentence_dedup", reason="needs the oracle group")
    pytest.importorskip("spacy", reason="needs the oracle group")
    import pyroid.dedup as pd
    from datatrove.data import Document
    from datatrove.utils.hashing import HashConfig

    docs = need_corpus("dev")
    kw = dict(CASES[case])
    hash_config = HashConfig(precision=kw.pop("precision"), hash_fc=kw.pop("hash_fc"))
    cfg = sd.SentDedupConfig(
        n_sentences=3, only_dedup_in_index=True, min_doc_words=50, hash_config=hash_config, **kw
    )

    def stream() -> Any:
        return (Document(text=t, id=str(i)) for i, t in enumerate(docs.texts))

    sd.SentenceDedupSignature(output_folder=str(tmp_path / "sig"), config=cfg).run(stream(), 0, 1)
    sd.SentenceFindDedups(
        data_folder=str(tmp_path / "sig"), output_folder=str(tmp_path / "dup"), config=cfg
    ).run(None, 0, 1)
    kept = {
        d.id: d.text
        for d in sd.SentenceDedupFilter(data_folder=str(tmp_path / "dup"), config=cfg).run(stream(), 0, 1)
    }
    want = [kept.get(str(i)) for i in range(len(docs))]

    res = pd.sentence_dedup(docs.texts, cfg)
    got = [t if k else None for t, k in zip(res.text, res.keep, strict=True)]
    bad = 0
    for doc_id, text, g, w in zip(docs.ids, docs.texts, got, want, strict=True):
        if not parity.record(
            "dedup", f"sentence_dedup/{case}/dev", g == w, case=doc_id, input=text, got=g, want=w
        ):
            bad += 1
    assert bad == 0, f"{bad}/{len(docs)} documents differ"


FW2_LANGS = ["deu_Latn", "fra_Latn", "rus_Cyrl", "spa_Latn", "arb_Arab", "hin_Deva"]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("language", FW2_LANGS)
def test_sentence_dedup_fineweb2(language: str, parity: Any, tmp_path: Path) -> None:
    """The example config on a FineWeb-2 slice, tokenized for its language (pyroid's native tokenizers)."""
    sd = pytest.importorskip("datatrove.pipeline.dedup.sentence_dedup", reason="needs the oracle group")
    pytest.importorskip("spacy", reason="needs the oracle group")
    if language == "hin_Deva":
        pytest.importorskip("indicnlp", reason="needs indic-nlp-library")
    import pyroid.dedup as pd
    from datatrove.data import Document

    from pyroid_parity import data as D
    from pyroid_parity.plugin import need

    texts = D.read_parquet_columns(need(f"fineweb2/{language}.parquet"), ["text"])["text"]
    # Repeat a share of each document's sentences elsewhere so removals are exercised.
    texts = texts + [t[: len(t) // 2] for t in texts[::3]]
    cfg = sd.SentDedupConfig(n_sentences=3, min_doc_words=50)

    def stream() -> Any:
        return (Document(text=t, id=str(i)) for i, t in enumerate(texts))

    sd.SentenceDedupSignature(output_folder=str(tmp_path / "sig"), config=cfg, language=language).run(
        stream(), 0, 1
    )
    sd.SentenceFindDedups(
        data_folder=str(tmp_path / "sig"), output_folder=str(tmp_path / "dup"), config=cfg
    ).run(None, 0, 1)
    step = sd.SentenceDedupFilter(data_folder=str(tmp_path / "dup"), config=cfg, language=language)
    kept = {d.id: d.text for d in step.run(stream(), 0, 1)}
    want = [kept.get(str(i)) for i in range(len(texts))]
    res = pd.sentence_dedup(texts, cfg, language=language)
    got = [t if k else None for t, k in zip(res.text, res.keep, strict=True)]
    bad = 0
    for i, (text, g, w) in enumerate(zip(texts, got, want, strict=True)):
        if not parity.record(
            "dedup", f"sentence_dedup/{language}/fineweb2", g == w, case=str(i), input=text, got=g, want=w
        ):
            bad += 1
    assert bad == 0, f"{bad}/{len(texts)} documents differ"
    assert sum(w is None or w != t for w, t in zip(want, texts, strict=True)) > 0  # something was removed
