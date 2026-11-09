"""Per-language parity data: the FineWeb-2 slices, near-duplicate variants of them, and the reference.

The reference is datatrove 0.10.1 itself with each slice's own language. It comes from the pinned
``parity/filters-ml-ref.parquet`` / ``parity/minhash-ml-ref.parquet`` when they are pinned in the manifest
(``python -m pyroid_parity.refs filters_ml dedup_ml --pin``), else from datatrove live when it (and the
language's word tokenizer: spaCy for ar/de/fr/ru/es, jieba for zh, SudachiPy for ja, indic-nlp-library for hi,
pyvi for vi, PyThaiNLP and python-crfsuite for th) imports, else the check is skipped with the reason.
"""

from __future__ import annotations

import random
from functools import cache
from typing import Any

from pyroid_parity import data as D
from pyroid_parity import specs

REFS = {"filters": "parity/filters-ml-ref.parquet", "dedup": "parity/minhash-ml-ref.parquet"}
#: Every NEAR_DUP_EVERY-th document of a slice gets a near-duplicate appended (for the removed-set check).
NEAR_DUP_EVERY = 5


def slice_rel(lang: str) -> str:
    return f"fineweb2/{lang}.parquet"


@cache
def fineweb2(lang: str) -> D.Corpus:
    """The pinned FineWeb-2 slice for ``lang`` (ids are ``<lang>:<FineWeb-2 id>``)."""
    p = D.path(slice_rel(lang))
    cols = D.read_parquet_columns(p, ["id", "text"])
    ids = [f"{lang}:{i}" for i in cols["id"]]
    return D.Corpus(lang, ids, cols["text"], [lang] * len(ids))


def _near_duplicate(text: str, rng: random.Random) -> str:
    """``text`` with about one character in 300 deleted (whatever the script, shingles stay mostly shared)."""
    chars = list(text)
    for _ in range(max(1, len(chars) // 300)):
        if len(chars) > 1:
            del chars[rng.randrange(len(chars))]
    return "".join(chars)


@cache
def with_near_duplicates(lang: str) -> D.Corpus:
    """The slice plus a near-duplicate of every ``NEAR_DUP_EVERY``-th document (appended, ids ``...#dup``)."""
    c = fineweb2(lang)
    ids, texts, meta = list(c.ids), list(c.texts), list(c.meta)
    for n in range(0, len(c), NEAR_DUP_EVERY):
        rng = random.Random(f"{lang}/{n}")
        ids.append(c.ids[n] + "#dup")
        texts.append(_near_duplicate(c.texts[n], rng))
        meta.append("near-duplicate")
    return D.Corpus(f"{lang}+dups", ids, texts, meta)


def minhash_config(lang: str) -> dict[str, Any]:
    return {**specs.MINHASH_CONFIG_FINEWEB2, "language": lang}


def oracle_problem(lang: str) -> str | None:
    """Why datatrove cannot tokenize ``lang`` here (words and sentences), or None if it can."""
    try:
        from datatrove.utils.text import split_into_words
        from datatrove.utils.word_tokenizers import load_word_tokenizer

        split_into_words("Probe 1.", lang)
        load_word_tokenizer(lang).span_tokenize("Probe 1. Probe 2.")  # PyThaiNLP's needs python-crfsuite
    except Exception as e:  # noqa: BLE001  missing backend packages surface as many types
        return f"datatrove cannot tokenize {lang} here: {type(e).__name__}: {e}"
    return None
