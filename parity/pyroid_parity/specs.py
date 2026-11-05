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
    "n_grams": 5, "num_buckets": 14, "hashes_per_bucket": 8, "seed": 1, "hash_fc": "sha1", "precision": 64,
}

LID_MODELS = {"lid.176.bin": "models/lid.176.bin", "lid.176.ftz": "models/lid.176.ftz"}
LID_TOPK = 5

