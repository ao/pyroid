"""Shared constants: filter cases, MinHash config, model files. Product tracks implement against these."""

from __future__ import annotations

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
