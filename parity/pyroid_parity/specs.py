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
JSON_REPAIR_VERSION = "0.64.9"
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
# ``tokenizer_assignment.csv`` assigns that code (spaCy ar/de/fr/ru/es; jieba zh, SudachiPy ja, IndicNLP hi,
# pyvi vi, PyThaiNLP th).
FINEWEB2_LANGUAGES = (
    "arb_Arab",
    "cmn_Hani",
    "deu_Latn",
    "fra_Latn",
    "hin_Deva",
    "jpn_Jpan",
    "rus_Cyrl",
    "spa_Latn",
    "tha_Thai",
    "vie_Latn",
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


# ------------------------------------------------------------------------------------------------ trafilatura
# The trafilatura suite's oracle (parity/pyproject.toml, group `trafilatura`): pyroid.trafilatura is an exact port
# of this triple, so the suite refuses to compare against any other.
TRAFILATURA_VERSION = "2.3.1"
TRAFILATURA_ORACLE = {"trafilatura": TRAFILATURA_VERSION, "lxml": "6.1.3", "justext": "3.0.2"}

# The pinned real-page corpus (bench/fetch.py --only trafilatura), in this order. Each file is one datatrove task:
# a dedup option set starts each file with an empty cache.
TRAFILATURA_CORPORA = (
    "trafilatura/CC-MAIN-2026-39.parquet",
    "trafilatura/CC-MAIN-2023-50.parquet",
    "trafilatura/CC-MAIN-2019-35.parquet",
)

# The corpus option matrix: name -> extract(html, **kwargs), output "txt". "datatrove" is datatrove 0.10.1's call
# (its Trafilatura step: favour_precision=True, deduplicate=True). Sets with deduplicate=True are stateful: the
# pages of a file run in order through one dedup cache (Trafilatura's module-level LRU_TEST), emptied first.
TRAFILATURA_OPTION_SETS: dict[str, dict[str, Any]] = {
    "datatrove": {"favor_precision": True, "include_comments": False, "deduplicate": True},
    "default": {},
    "fast": {"fast": True},
    "favor_recall": {"favor_recall": True},
    "precision_comments": {"favor_precision": True, "include_comments": True},
    "no_tables": {"include_tables": False},
    "dedup": {"deduplicate": True},
}
# bare_extraction(html, **kwargs) on the corpus, compared field by field (pyroid_parity.trafilatura_compare).
TRAFILATURA_BARE_SETS: dict[str, dict[str, Any]] = {
    "precision": {"favor_precision": True, "include_comments": False},
    "default": {},
}


def trafilatura_stateful(kwargs: dict[str, Any]) -> bool:
    """Whether a call with these kwargs reads and updates the dedup cache."""
    return kwargs.get("deduplicate") is True


# Options pyroid.trafilatura does not port. A call with any of them must go to trafilatura itself: pyroid warns
# with exactly one pyroid.FallbackWarning (per process, function and reason) and returns trafilatura's result.
# Values starting with "$" are built at run time by pyroid_parity.trafilatura_delegate (objects that need
# trafilatura or lxml).
TRAFILATURA_DELEGATED: dict[str, dict[str, Any]] = {
    "output_format=xml": {"output_format": "xml"},
    "output_format=xmltei": {"output_format": "xmltei"},
    "output_format=json": {"output_format": "json"},
    "output_format=markdown": {"output_format": "markdown"},
    "output_format=csv": {"output_format": "csv"},
    "output_format=html": {"output_format": "html"},
    "include_formatting": {"include_formatting": True},
    "include_links": {"include_links": True},
    "include_images": {"include_images": True},
    "with_metadata": {"with_metadata": True},
    "only_with_metadata": {"only_with_metadata": True},
    "url_blacklist": {"url_blacklist": {"https://example.org/blocked"}},
    "author_blacklist": {"author_blacklist": {"Jane Doe"}},
    "bare:with_metadata": {"$fn": "bare_extraction", "with_metadata": True},
    "bare:output_format=xml": {"$fn": "bare_extraction", "output_format": "xml"},
}
# Options outside the corpus matrix that pyroid may port or hand to trafilatura: the result must be
# trafilatura's, with at most one FallbackWarning.
TRAFILATURA_EDGE_OPTIONS: dict[str, dict[str, Any]] = {
    "target_language": {"target_language": "en"},
    "prune_xpath": {"prune_xpath": "//p[1]"},
    "prune_xpath_list": {"prune_xpath": ["//p[1]", "//table"]},
    "config": {"config": "$config:MIN_EXTRACTED_SIZE=10,MIN_OUTPUT_SIZE=5"},
    "settingsfile": {"settingsfile": "$settingsfile:MIN_OUTPUT_SIZE=1000"},
    "options": {"options": "$extractor:precision"},
    "dedup_cache_object": {"deduplicate": "$lrucache:2"},
    "tree_input": {"$input": "tree"},
    "bytes_input": {"$input": "utf-8-bytes"},
    "latin1_bytes_input": {"$input": "latin-1-bytes"},
    "url": {"url": "https://example.org/article/1"},
    "record_id": {"record_id": "0001"},
    "tei_validation": {"tei_validation": True},
    "no_fallback": {"no_fallback": True},
    "max_tree_size": {"max_tree_size": 100},
    "precision_and_recall": {"favor_precision": True, "favor_recall": True},
    "include_formatting=False": {"include_formatting": False},
    "output_format=txt": {"output_format": "txt"},
    "output_format=python": {"output_format": "python"},
    "output_format=bogus": {"output_format": "bogus"},
    "bare:as_dict": {"$fn": "bare_extraction", "as_dict": True},
}

# trafilatura 2.3.1's public signatures: (name, default) of every parameter, all positional-or-keyword, in order.
# "<required>" has no default; "DEFAULT_CONFIG" is trafilatura.settings.DEFAULT_CONFIG (a candidate may use None).
_TRAF_COMMON = (
    ("fast", False),
    ("no_fallback", False),
    ("favor_precision", False),
    ("favor_recall", False),
    ("include_comments", True),
)
_TRAF_FILTERS = (
    ("include_tables", True),
    ("include_images", False),
    ("include_formatting", None),
    ("include_links", False),
    ("deduplicate", False),
    ("date_extraction_params", None),
    ("with_metadata", False),
    ("only_with_metadata", False),
    ("max_tree_size", None),
    ("url_blacklist", None),
    ("author_blacklist", None),
)
TRAFILATURA_SIGNATURES: dict[str, tuple[tuple[str, Any], ...]] = {
    "extract": (
        ("filecontent", "<required>"),
        ("url", None),
        ("record_id", None),
        *_TRAF_COMMON,
        ("output_format", "txt"),
        ("tei_validation", False),
        ("target_language", None),
        *_TRAF_FILTERS,
        ("settingsfile", None),
        ("prune_xpath", None),
        ("config", "DEFAULT_CONFIG"),
        ("options", None),
    ),
    "bare_extraction": (
        ("filecontent", "<required>"),
        ("url", None),
        *_TRAF_COMMON,
        ("output_format", "python"),
        ("target_language", None),
        *_TRAF_FILTERS,
        ("as_dict", False),
        ("prune_xpath", None),
        ("config", "DEFAULT_CONFIG"),
        ("options", None),
    ),
}


# ----------------------------------------------------------------------------- trafilatura: one mode per release
# pyroid.trafilatura reproduces several trafilatura releases, chosen per call with ``compat=``. The suite runs one
# mode at a time (``--trafilatura-compat``, or $PYROID_TRAFILATURA_COMPAT): that release's references, option
# matrix and recorded upstream calls, against that release as the live oracle (its own dependency group).
TRAFILATURA_MODES = ("2.3.1", "1.11.0")
TRAFILATURA_COMPAT_ENV = "PYROID_TRAFILATURA_COMPAT"
TRAFILATURA_ORACLES: dict[str, dict[str, str]] = {
    "2.3.1": TRAFILATURA_ORACLE,
    # what a fresh `pip install datatrove[processing]==0.10.1` resolves
    "1.11.0": {
        "trafilatura": "1.11.0",
        "lxml": "6.1.3",
        "justext": "3.0.2",
        "lxml-html-clean": "0.4.6",
        "charset-normalizer": "3.5.2",
        "courlan": "1.4.0",
        "htmldate": "1.11.0",
    },
}
TRAFILATURA_GROUPS = {"2.3.1": "trafilatura", "1.11.0": "trafilatura111"}


def trafilatura_mode() -> str:
    """The mode this process checks: $PYROID_TRAFILATURA_COMPAT, else the default (2.3.1)."""
    import os

    mode = os.environ.get(TRAFILATURA_COMPAT_ENV) or TRAFILATURA_VERSION
    if mode not in TRAFILATURA_MODES:
        raise ValueError(f"{TRAFILATURA_COMPAT_ENV}={mode!r}: not one of {TRAFILATURA_MODES}")
    return mode


# 1.11.0 has no `fast` (its `no_fallback` is the same switch) and takes the rest of 2.3.1's matrix as it is.
# datatrove's stateful call comes first in both.
TRAFILATURA_OPTION_SETS_BY_MODE: dict[str, dict[str, dict[str, Any]]] = {
    "2.3.1": TRAFILATURA_OPTION_SETS,
    "1.11.0": {
        "datatrove": {"favor_precision": True, "include_comments": False, "deduplicate": True},
        "default": {},
        "no_fallback": {"no_fallback": True},
        "favor_recall": {"favor_recall": True},
        "precision_comments": {"favor_precision": True, "include_comments": True},
        "no_tables": {"include_tables": False},
        "dedup": {"deduplicate": True},
    },
}
TRAFILATURA_BARE_SETS_BY_MODE: dict[str, dict[str, dict[str, Any]]] = {
    "2.3.1": TRAFILATURA_BARE_SETS,
    "1.11.0": TRAFILATURA_BARE_SETS,  # bare_extraction returns a dict there (as_dict=True by default)
}

# trafilatura 1.11.0's signatures, in the same form; "**kwargs" is a trailing VAR_KEYWORD parameter (1.11.0's
# extract takes unknown keywords and ignores them, except the old *_output switches, which raise NameError).
_TRAF111_FILTERS = (
    ("include_tables", True),
    ("include_images", False),
    ("include_formatting", False),
    ("include_links", False),
    ("deduplicate", False),
    ("date_extraction_params", None),
    ("with_metadata", False),
    ("only_with_metadata", False),
    ("max_tree_size", None),
    ("url_blacklist", None),
    ("author_blacklist", None),
)
_TRAF111_COMMON = (("no_fallback", False), ("favor_precision", False), ("favor_recall", False))
TRAFILATURA_SIGNATURES_BY_MODE: dict[str, dict[str, tuple[tuple[str, Any], ...]]] = {
    "2.3.1": TRAFILATURA_SIGNATURES,
    "1.11.0": {
        "extract": (
            ("filecontent", "<required>"),
            ("url", None),
            ("record_id", None),
            *_TRAF111_COMMON,
            ("include_comments", True),
            ("output_format", "txt"),
            ("tei_validation", False),
            ("target_language", None),
            *_TRAF111_FILTERS,
            ("settingsfile", None),
            ("prune_xpath", None),
            ("config", "DEFAULT_CONFIG"),
            ("options", None),
            ("**kwargs", None),
        ),
        "bare_extraction": (
            ("filecontent", "<required>"),
            ("url", None),
            *_TRAF111_COMMON,
            ("include_comments", True),
            ("output_format", "python"),
            ("target_language", None),
            *_TRAF111_FILTERS,
            ("as_dict", True),
            ("prune_xpath", None),
            ("config", "DEFAULT_CONFIG"),
            ("options", None),
        ),
    },
}

# Edge cases only one mode has: 1.11.0's extract ignores `fast` (pyroid has the parameter, so the 1.11.0 mode
# must too), its bare_extraction rejects it.
TRAFILATURA_EDGE_EXTRA: dict[str, dict[str, dict[str, Any]]] = {
    "2.3.1": {},
    "1.11.0": {
        "fast_ignored": {"fast": True},
        "bare:fast": {"$fn": "bare_extraction", "fast": True},
    },
}


# Delegated cases that are not delegated in a mode, because there they do not change the txt output (they are
# edge cases instead: the release's result, at most one FallbackWarning).
TRAFILATURA_NOT_DELEGATED: dict[str, dict[str, str]] = {
    "2.3.1": {},
    "1.11.0": {
        "author_blacklist": "1.11.0 applies it only with metadata, and it does not switch metadata on"
    },
}


def trafilatura_cases(mode: str, group: str) -> dict[str, dict[str, Any]]:
    """The delegated or edge cases of a mode: those whose options the mode's functions have, plus its extras."""
    sigs = TRAFILATURA_SIGNATURES_BY_MODE[mode]

    def present(cases: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out = {}
        for name, kw in cases.items():
            params = {n for n, _ in sigs[kw.get("$fn", "extract")]}
            if all(k.startswith("$") or k in params for k in kw):
                out[name] = kw
        return out

    not_delegated = TRAFILATURA_NOT_DELEGATED[mode]
    delegated = {n: kw for n, kw in present(TRAFILATURA_DELEGATED).items() if n not in not_delegated}
    if group == "delegated":
        return delegated
    moved = {n: TRAFILATURA_DELEGATED[n] for n in not_delegated}
    return {**present(TRAFILATURA_EDGE_OPTIONS), **moved, **TRAFILATURA_EDGE_EXTRA[mode]}


def trafilatura_files(mode: str) -> dict[str, str]:
    """The pinned files of a mode: refs, recorded upstream calls and blobs, the upstream tests."""
    if mode == TRAFILATURA_VERSION:  # the first mode's names predate the others
        prefix = "parity/trafilatura"
    else:
        prefix = f"parity/trafilatura-{mode}"
    return {
        "ref": f"{prefix}-ref.parquet",
        "calls": f"{prefix}-upstream.parquet",
        "blobs": f"{prefix}-upstream-blobs.parquet",
        "tests": f"trafilatura/upstream-tests-{mode}.parquet",
    }
