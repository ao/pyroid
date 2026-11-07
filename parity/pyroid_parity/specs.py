"""Shared constants: filter cases, MinHash config, model files. Each module's suite is written against these."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

# case id -> (contract function name, datatrove class, kwargs). "@fineweb" = datatrove examples/fineweb.py config.
FILTER_CASES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "gopher_repetition": ("gopher_repetition", "GopherRepetitionFilter", {}),
    "gopher_quality": ("gopher_quality", "GopherQualityFilter", {}),
    "c4_quality": ("c4_quality", "C4QualityFilter", {}),
    "c4_quality@fineweb": ("c4_quality", "C4QualityFilter", {"filter_no_terminal_punct": False}),
    "fineweb_quality": ("fineweb_quality", "FineWebQualityFilter", {}),
}
# Filters that rewrite doc.text in datatrove (C4 drops lines); parity covers the rewritten text too.
REWRITES_TEXT = {"c4_quality"}

# datatrove examples/fineweb.py MinhashConfig(HashConfig("sha1", 64), num_buckets=14, hashes_per_bucket=8, n_grams=5)
MINHASH_CONFIG: dict[str, Any] = {
    "n_grams": 5,
    "num_buckets": 14,
    "hashes_per_bucket": 8,
    "seed": 1,
    "hash_fc": "sha1",
    "precision": 64,
}

LID_MODELS = {"lid.176.bin": "models/lid.176.bin", "lid.176.ftz": "models/lid.176.ftz"}
LID_TOPK = 5


def lid_ref(platform: str) -> str:
    """The fastText reference file for a ``data.platform_key()``.

    fasttext-predict's probabilities depend on the CPU it was built for: GCC contracts the quantized-matrix
    loops into fused multiply-adds on aarch64 but not on baseline x86-64, so ``lid.176.ftz`` differs by up to
    6e-7 (``lid.176.bin`` is identical). Each platform therefore has its own pinned file; linux-aarch64 keeps
    the original name.
    """
    return (
        "parity/fasttext-ref.parquet"
        if platform == "linux-aarch64"
        else f"parity/fasttext-ref.{platform}.parquet"
    )


# json_repair suite: the pinned version and the option sets every corpus / Hypothesis input is repaired under
# (keyword arguments of ``repair_json``; ``ensure_ascii`` and ``indent`` are passed on to ``json.dumps``).
JSON_REPAIR_VERSION = "0.64.5"
JSON_REPAIR_OPTION_SETS: dict[str, dict[str, Any]] = {
    "default": {},
    "return_objects": {"return_objects": True},
    "logging": {"logging": True},
    "skip_json_loads": {"skip_json_loads": True},
    "ensure_ascii_false": {"ensure_ascii": False},
    "stream_stable": {"stream_stable": True},
    "strict": {"strict": True},
    "indent2": {"indent": 2},
}


# --------------------------------------------------------------------------------------------- FineWeb-2
# The FineWeb-2 slices pinned in bench/manifest.json (``fineweb2/<code>.parquet``, 250 docs each). A FineWeb-2
# subset's name is its datatrove language code: datatrove tokenizes it with the word tokenizer that
# ``tokenizer_assignment.csv`` assigns that code (spaCy ar/de/fr/ru/es; jieba zh, SudachiPy ja, IndicNLP hi).
FINEWEB2_LANGUAGES = (
    "arb_Arab",
    "cmn_Hani",
    "deu_Latn",
    "fra_Latn",
    "hin_Deva",
    "jpn_Jpan",
    "rus_Cyrl",
    "spa_Latn",
)

FINEWEB2_CONFIGS_FILE = Path(__file__).resolve().parents[1] / "data" / "fineweb2-configs" / "configs.json"


@cache
def fineweb2_config(lang: str) -> dict[str, Any]:
    """FineWeb-2's own per-language filter settings (``configs/<lang>.yml``, Apache-2.0, HuggingFace)."""
    return dict(json.loads(FINEWEB2_CONFIGS_FILE.read_text(encoding="utf-8"))[lang])


def fineweb2_filter_cases(lang: str) -> dict[str, tuple[str, str, dict[str, Any]]]:
    """Filter cases for one language: FineWeb-2's pipeline (``fineweb-2-pipeline.py`` step 5, with the
    language's thresholds and stop words) and datatrove's defaults, all with ``language=lang``."""
    c = fineweb2_config(lang)

    def ngrams(key: str) -> list[list[float]]:
        return [[int(n), float(f)] for n, f in c[key]]

    return {
        "gopher_repetition@fineweb2": (
            "gopher_repetition",
            "GopherRepetitionFilter",
            {
                "language": lang,
                "dup_para_frac": 0,
                "dup_line_char_frac": 0,
                "dup_para_char_frac": 0,
                "dup_line_frac": c["dup_line_frac"],
                "top_n_grams": ngrams("top_n_grams"),
                "dup_n_grams": ngrams("dup_n_grams"),
            },
        ),
        "fineweb_quality@fineweb2": (
            "fineweb_quality",
            "FineWebQualityFilter",
            {
                "language": lang,
                "short_line_thr": 999,
                "char_duplicates_ratio": 0.1,
                "line_punct_thr": c["line_punct_thr"],
                "new_line_ratio": c["new_line_ratio"],
            },
        ),
        "gopher_quality@fineweb2": (
            "gopher_quality",
            "GopherQualityFilter",
            {
                "language": lang,
                "max_avg_word_length": c["max_avg_word_length"],
                "min_avg_word_length": c["min_avg_word_length"],
                "stop_words": list(c["stopwords"]),
                "max_non_alpha_words_ratio": c["max_non_alpha_words_ratio"],
                "min_stop_words": 2,
            },
        ),
        **{case: (fn, cls, {**cfg, "language": lang}) for case, (fn, cls, cfg) in FILTER_CASES.items()},
    }


# FineWeb-2's MinhashConfig (fineweb-2-pipeline.py: HashConfig("xxhash", 64), 14 buckets x 8, 5-grams); the
# language is MinhashDedupSignature's, passed as ``language`` next to these keys.
MINHASH_CONFIG_FINEWEB2: dict[str, Any] = {**MINHASH_CONFIG, "hash_fc": "xxhash", "precision": 64}
