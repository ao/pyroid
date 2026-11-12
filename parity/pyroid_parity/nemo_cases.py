"""The ``nemo`` suite's cases: NeMo Curator 1.4.0 filters, modifiers and fastText stages, with their configs.

Every case names the ``pyroid.nemo`` function that reproduces it (the suite contract, ``pyroid_parity.impl``),
the NeMo class and the constructor arguments. ``pyroid_parity.nemo_record`` records NeMo's outputs for every
case on the pinned corpora; ``tests/test_nemo.py`` compares a candidate against them.

The first 28 filter cases are the ScoreFilter stages of NeMo's ``config/text/heuristic_filter_english_pipeline.yaml``,
in order and with its arguments (``HEURISTIC_PIPELINE``); the recorder checks that list against the YAML. The
other cases are configs the YAML does not use.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass, field
from typing import Any

NEMO_VERSION = "1.4.0"
# What the references were recorded with (the oracle env, /target/parity/venv in the README's recipe).
ORACLE = {
    "nemo-curator": "1.4.0",
    "fasttext": "0.9.3",
    "ftfy": "6.3.1",
    "regex": "2026.11.9",
    "pandas": "3.1.1",
    "numpy": "2.5.4",
    "jieba": "0.42.1",
}
SUITE = "nemo"
#: RepeatingTopNGramsFilter breaks ties by set iteration order, which follows the string hash seed. The
#: references are recorded with this seed, and checks against them run with it (in a subprocess if needed).
HASH_SEED = "0"
HASH_SEED_ENV = "PYTHONHASHSEED"

NEMO_MODULES = {
    **dict.fromkeys(
        (
            "NonAlphaNumericFilter",
            "SymbolsToWordsFilter",
            "NumbersFilter",
            "UrlsFilter",
            "BulletsFilter",
            "WhiteSpaceFilter",
            "ParenthesesFilter",
            "LongWordFilter",
            "WordCountFilter",
            "BoilerPlateStringFilter",
            "MeanWordLengthFilter",
            "PunctuationFilter",
            "EllipsisFilter",
            "CommonEnglishWordsFilter",
            "WordsWithoutAlphabetsFilter",
            "PornographicUrlsFilter",
            "SubstringFilter",
        ),
        "nemo_curator.stages.text.filters.heuristic.string",
    ),
    **dict.fromkeys(
        (
            "RepeatedLinesFilter",
            "RepeatedParagraphsFilter",
            "RepeatedLinesByCharFilter",
            "RepeatedParagraphsByCharFilter",
            "RepeatingTopNGramsFilter",
            "RepeatingDuplicateNGramsFilter",
        ),
        "nemo_curator.stages.text.filters.heuristic.repetition",
    ),
    **dict.fromkeys(("FastTextLangId", "FastTextQualityFilter"), "nemo_curator.stages.text.filters.fasttext"),
    "BoilerPlateStringModifier": "nemo_curator.stages.text.modifiers.string",
    "MarkdownRemover": "nemo_curator.stages.text.modifiers.string",
    "NewlineNormalizer": "nemo_curator.stages.text.modifiers.string",
    "QuotationRemover": "nemo_curator.stages.text.modifiers.string",
    "UrlRemover": "nemo_curator.stages.text.modifiers.string",
    "LineRemover": "nemo_curator.stages.text.modifiers.string",
    "Slicer": "nemo_curator.stages.text.modifiers.string",
    "UnicodeReformatter": "nemo_curator.stages.text.modifiers.unicode",
    "FastTextLabelModifier": "nemo_curator.stages.text.modifiers.fasttext",
    "ScoreFilter": "nemo_curator.stages.text.filters",
    "Score": "nemo_curator.stages.text.filters",
    "Filter": "nemo_curator.stages.text.filters",
    "Modify": "nemo_curator.stages.text.modifiers",
}
HEURISTIC_CLASSES = tuple(k for k, v in NEMO_MODULES.items() if ".heuristic." in v)
FASTTEXT_CLASSES = ("FastTextLangId", "FastTextQualityFilter")
MODIFIER_CLASSES = tuple(k for k, v in NEMO_MODULES.items() if ".modifiers." in v and k != "Modify")
STAGE_CLASSES = ("ScoreFilter", "Score", "Filter", "Modify")
#: Every class ``pyroid.integrations.nemo_curator`` must provide, with NeMo's name and constructor signature.
INTEGRATION_CLASSES = (*STAGE_CLASSES, *HEURISTIC_CLASSES, *FASTTEXT_CLASSES, *MODIFIER_CLASSES)


def nemo_class(name: str) -> type:
    """NeMo's class ``name`` (imports nemo_curator)."""
    return getattr(importlib.import_module(NEMO_MODULES[name]), name)


# ------------------------------------------------------------------------------------------------- filters


@dataclass(frozen=True)
class FilterCase:
    """One DocumentFilter config. ``fn(text, **kwargs)`` is the candidate call, ``cls(**nemo_kwargs)`` NeMo's."""

    case: str
    fn: str
    nemo: str
    nemo_kwargs: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    #: "all" = dev + adversarial; "zh" = the adversarial set's Mandarin and edge-case documents.
    docs: str = "all"

    @property
    def hash_seeded(self) -> bool:
        """The score of a tied document depends on the process's string hash seed."""
        return self.nemo == "RepeatingTopNGramsFilter"

    def make(self, cls: type | None = None) -> Any:
        return (cls or nemo_class(self.nemo))(**self.nemo_kwargs)


def _f(case: str, fn: str, nemo: str, nemo_kwargs: dict[str, Any] | None = None, **kw: Any) -> FilterCase:
    return FilterCase(case, fn, nemo, nemo_kwargs or {}, **kw)


# heuristic_filter_english_pipeline.yaml, stage by stage (the reader and writer left out).
HEURISTIC_PIPELINE: list[FilterCase] = [
    _f(
        "non_alpha_numeric",
        "non_alpha_numeric",
        "NonAlphaNumericFilter",
        {"max_non_alpha_numeric_to_text_ratio": 0.25},
    ),
    _f("symbols_to_words", "symbols_to_words", "SymbolsToWordsFilter", {"max_symbol_to_word_ratio": 0.1}),
    _f("numbers", "numbers", "NumbersFilter", {"max_number_to_text_ratio": 0.15}),
    _f("urls", "urls", "UrlsFilter", {"max_url_to_text_ratio": 0.2}),
    _f("white_space", "white_space", "WhiteSpaceFilter", {"max_white_space_ratio": 0.25}),
    _f("parentheses", "parentheses", "ParenthesesFilter", {"max_parentheses_ratio": 0.1}),
    _f(
        "boilerplate_string",
        "boilerplate_string",
        "BoilerPlateStringFilter",
        {"remove_if_at_top_or_bottom": True, "max_boilerplate_string_ratio": 0.4},
    ),
    _f("repeated_lines", "repeated_lines", "RepeatedLinesFilter", {"max_repeated_line_fraction": 0.7}),
    _f(
        "repeated_paragraphs",
        "repeated_paragraphs",
        "RepeatedParagraphsFilter",
        {"max_repeated_paragraphs_ratio": 0.7},
    ),
    _f(
        "repeated_lines_by_char",
        "repeated_lines_by_char",
        "RepeatedLinesByCharFilter",
        {"max_repeated_lines_char_ratio": 0.8},
    ),
    _f(
        "repeated_paragraphs_by_char",
        "repeated_paragraphs_by_char",
        "RepeatedParagraphsByCharFilter",
        {"max_repeated_paragraphs_char_ratio": 0.8},
    ),
    _f("word_count", "word_count", "WordCountFilter", {"min_words": 50, "max_words": 100000}),
    _f(
        "punctuation",
        "punctuation",
        "PunctuationFilter",
        {"max_num_sentences_without_endmark_ratio": 0.85},
    ),
    _f(
        "words_without_alphabets",
        "words_without_alphabets",
        "WordsWithoutAlphabetsFilter",
        {"min_words_with_alphabets": 0.8},
    ),
    _f(
        "common_english_words",
        "common_english_words",
        "CommonEnglishWordsFilter",
        {"min_num_common_words": 2, "stop_at_false": True},
        kwargs={"min_num_common_words": 2, "stop_at_false": True},
    ),
    _f(
        "mean_word_length",
        "mean_word_length",
        "MeanWordLengthFilter",
        {"max_mean_word_length": 10, "min_mean_word_length": 3},
    ),
    _f("long_word", "long_word", "LongWordFilter", {"max_word_length": 1000}),
    _f(
        "ellipsis",
        "ellipsis",
        "EllipsisFilter",
        {"max_num_lines_ending_with_ellipsis_ratio": 0.3},
    ),
    *(
        _f(
            f"repeating_top_ngrams@{n}",
            "repeating_top_ngrams",
            "RepeatingTopNGramsFilter",
            {"n": n, "max_repeating_ngram_ratio": r},
            kwargs={"n": n},
        )
        for n, r in ((2, 0.2), (3, 0.18), (4, 0.16))
    ),
    *(
        _f(
            f"repeating_duplicate_ngrams@{n}",
            "repeating_duplicate_ngrams",
            "RepeatingDuplicateNGramsFilter",
            {"n": n, "max_repeating_duplicate_ngram_ratio": r},
            kwargs={"n": n},
        )
        for n, r in ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10))
    ),
    _f("bullets", "bullets", "BulletsFilter", {"max_bullet_lines_ratio": 0.9}),
]

# Configs the YAML does not use: class defaults, other n, other thresholds that change the score, lang="zh".
_ZH = {"lang": "zh"}
OTHER_FILTERS: list[FilterCase] = [
    _f("word_count@min80", "word_count", "WordCountFilter", {"min_words": 80}),
    _f(
        "common_english_words@no_stop",
        "common_english_words",
        "CommonEnglishWordsFilter",
        {"stop_at_false": False},
        kwargs={"stop_at_false": False},
    ),
    _f(
        "common_english_words@min5",
        "common_english_words",
        "CommonEnglishWordsFilter",
        {"min_num_common_words": 5},
        kwargs={"min_num_common_words": 5},
    ),
    _f(
        "repeating_top_ngrams@1",
        "repeating_top_ngrams",
        "RepeatingTopNGramsFilter",
        {"n": 1},
        kwargs={"n": 1},
    ),
    *(
        _f(
            f"repeating_duplicate_ngrams@{n}",
            "repeating_duplicate_ngrams",
            "RepeatingDuplicateNGramsFilter",
            {"n": n},
            kwargs={"n": n},
        )
        for n in (1, 2, 3, 4)
    ),
    _f("pornographic_urls", "pornographic_urls", "PornographicUrlsFilter"),
    *(
        _f(
            f"substring@{pos}:{name}",
            "substring",
            "SubstringFilter",
            {"substring": sub, "position": pos},
            kwargs={"substring": sub, "position": pos},
        )
        for name, sub, pos in (
            ("The", "The", "prefix"),
            ("period", ".", "suffix"),
            ("http", "http://", "any"),
            ("empty", "", "prefix"),
            ("newline", "\n", "suffix"),
            ("cookies", "cookies", "any"),
        )
    ),
    # lang="zh": NeMo splits words with jieba.cut (HMM on); on the Mandarin and edge-case documents only.
    *(
        _f(f"{fn}@zh", fn, cls, dict(_ZH), kwargs=dict(_ZH), docs="zh")
        for fn, cls in (
            ("symbols_to_words", "SymbolsToWordsFilter"),
            ("word_count", "WordCountFilter"),
            ("words_without_alphabets", "WordsWithoutAlphabetsFilter"),
            ("mean_word_length", "MeanWordLengthFilter"),
            ("long_word", "LongWordFilter"),
        )
    ),
    _f(
        "repeating_top_ngrams@2-zh",
        "repeating_top_ngrams",
        "RepeatingTopNGramsFilter",
        {"n": 2, **_ZH},
        kwargs={"n": 2, **_ZH},
        docs="zh",
    ),
    _f(
        "repeating_duplicate_ngrams@5-zh",
        "repeating_duplicate_ngrams",
        "RepeatingDuplicateNGramsFilter",
        {"n": 5, **_ZH},
        kwargs={"n": 5, **_ZH},
        docs="zh",
    ),
]
FILTER_CASES: dict[str, FilterCase] = {c.case: c for c in (*HEURISTIC_PIPELINE, *OTHER_FILTERS)}
assert len(FILTER_CASES) == len(HEURISTIC_PIPELINE) + len(OTHER_FILTERS)

# ----------------------------------------------------------------------------------------------- modifiers

#: UnicodeReformatter's defaults, which differ from ftfy's (no NFC, no ligature/width/quote/line-break fixes).
NEMO_UNICODE_DEFAULTS: dict[str, Any] = {
    "unescape_html": "auto",
    "remove_terminal_escapes": True,
    "fix_encoding": True,
    "restore_byte_a0": True,
    "replace_lossy_sequences": True,
    "decode_inconsistent_utf8": True,
    "fix_c1_controls": True,
    "fix_latin_ligatures": False,
    "fix_character_width": False,
    "uncurl_quotes": False,
    "fix_line_breaks": False,
    "fix_surrogates": True,
    "remove_control_chars": True,
    "normalization": None,
    "max_decode_length": 1000000,
}


@dataclass(frozen=True)
class ModifierCase:
    """One DocumentModifier config. ``fn`` is a ``pyroid.nemo`` function, or ``ftfy.fix_text`` (resolved in the
    candidate's package) for UnicodeReformatter. ``nemo_kwargs["config"]``, when present, holds the keyword
    arguments of an ftfy ``TextFixerConfig`` passed as UnicodeReformatter's ``config``."""

    case: str
    fn: str
    nemo: str
    nemo_kwargs: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    docs: str = "all"
    hash_seeded = False

    def make(self, cls: type | None = None) -> Any:
        kw = dict(self.nemo_kwargs)
        if "config" in kw:
            from ftfy import TextFixerConfig

            kw["config"] = TextFixerConfig(**kw["config"])
        return (cls or nemo_class(self.nemo))(**kw)


_NFKC = {"normalization": "NFKC", "uncurl_quotes": True, "fix_line_breaks": True, "unescape_html": False}
#: Lines that occur often in the dev slice, plus the empty line.
LINE_PATTERNS = [
    "",
    "* * *",
    "Post a Comment",
    "View Single Post",
    "Click here to view",
    "Back to top",
    "Read More »",
    "Table of Contents",
    "Terms & Conditions",
    "Frequently Asked Questions",
]
_SLICES: list[tuple[str, dict[str, Any]]] = [
    ("default", {}),
    ("int", {"left": 10, "right": -10}),
    ("negative-left", {"left": -200}),
    ("empty-range", {"left": 50, "right": 20}),
    ("strings", {"left": "the", "right": ".", "include_left": False, "include_right": True}),
    ("strings-inclusive-nostrip", {"left": "\n", "right": "\n", "strip": False}),
    ("strings-exclusive", {"left": "\n", "right": "\n", "include_left": False, "include_right": False}),
    ("int-left-string-right", {"left": 5, "right": "the"}),
    ("none-left", {"left": None, "right": 100, "strip": False}),
]
MODIFIER_CASES: dict[str, ModifierCase] = {
    c.case: c
    for c in [
        ModifierCase(
            "unicode_reformatter", "ftfy.fix_text", "UnicodeReformatter", {}, dict(NEMO_UNICODE_DEFAULTS)
        ),
        ModifierCase(
            "unicode_reformatter@ftfy_defaults", "ftfy.fix_text", "UnicodeReformatter", {"config": {}}
        ),
        ModifierCase(
            "unicode_reformatter@nfkc",
            "ftfy.fix_text",
            "UnicodeReformatter",
            dict(_NFKC),
            {**NEMO_UNICODE_DEFAULTS, **_NFKC},
        ),
        ModifierCase(
            "boilerplate_string_modifier", "remove_boilerplate_strings", "BoilerPlateStringModifier"
        ),
        ModifierCase(
            "boilerplate_string_modifier@anywhere",
            "remove_boilerplate_strings",
            "BoilerPlateStringModifier",
            {"remove_if_at_top_or_bottom": False},
            {"remove_if_at_top_or_bottom": False},
        ),
        ModifierCase("markdown_remover", "remove_markdown", "MarkdownRemover"),
        ModifierCase("newline_normalizer", "normalize_newlines", "NewlineNormalizer"),
        ModifierCase("quotation_remover", "remove_quotation", "QuotationRemover"),
        ModifierCase("url_remover", "remove_urls", "UrlRemover"),
        ModifierCase(
            "line_remover",
            "remove_lines",
            "LineRemover",
            {"patterns": LINE_PATTERNS},
            {"patterns": LINE_PATTERNS},
        ),
        *(ModifierCase(f"slicer@{name}", "slice_text", "Slicer", dict(kw), dict(kw)) for name, kw in _SLICES),
        ModifierCase(
            "fasttext_label_modifier",
            "fasttext_label",
            "FastTextLabelModifier",
            {"label": "__label__hq"},
            {"label": "__label__hq"},
        ),
    ]
}

# ------------------------------------------------------------------------------------------------ fastText

LID_MODELS = {"lid.176.bin": "models/lid.176.bin", "lid.176.ftz": "models/lid.176.ftz"}
QUALITY_MODEL = "models/nemo-quality-dev10k.ftz"
QUALITY_LABEL = "__label__hq"
#: fasttext 0.9.3 ``train_supervised`` arguments of the quality model (one thread: the model is reproducible).
#: On linux-aarch64 this fasttext build stops with "Encountered NaN" for dim < 32, and for dim 32 with
#: minCount > 1 or word bigrams, on this data; dim 32 with unigrams trains (52 MB), so the model is quantized.
QUALITY_RECIPE: dict[str, Any] = {
    "lr": 0.1,
    "epoch": 5,
    "dim": 32,
    "loss": "softmax",
    "thread": 1,
    "seed": 0,
    "verbose": 0,
}
#: ``model.quantize`` arguments (no retraining, so the .ftz is a function of the .bin): 1.7 MB.
QUALITY_QUANTIZE: dict[str, Any] = {"cutoff": 50000, "qnorm": True, "retrain": False}
QUALITY_LABEL_RULE = (
    "__label__hq if the dev document passes all 28 ScoreFilter stages of heuristic_filter_english_pipeline.yaml "
    "(recorded keep, no exception), else __label__lq; one line per dev document, in order, prepared with "
    "FastTextLabelModifier(label).modify_document(text)"
)


@dataclass(frozen=True)
class FastTextCase:
    case: str
    nemo: str
    model: str
    nemo_kwargs: dict[str, Any] = field(default_factory=dict)
    docs: str = "all"
    hash_seeded = False

    def make(self, cls: type | None = None, model_path: str | None = None) -> Any:
        return (cls or nemo_class(self.nemo))(model_path=model_path, **self.nemo_kwargs)


FASTTEXT_CASES: dict[str, FastTextCase] = {
    c.case: c
    for c in [
        FastTextCase(
            "lang_id@lid.176.bin", "FastTextLangId", LID_MODELS["lid.176.bin"], {"min_langid_score": 0.3}
        ),
        FastTextCase(
            "lang_id@lid.176.ftz", "FastTextLangId", LID_MODELS["lid.176.ftz"], {"min_langid_score": 0.3}
        ),
        # Same scores as lang_id@lid.176.bin; keep_document also matches the language.
        FastTextCase(
            "lang_id@lid.176.bin-en",
            "FastTextLangId",
            LID_MODELS["lid.176.bin"],
            {"min_langid_score": 0.3, "lang": "en"},
        ),
        FastTextCase(
            "quality",
            "FastTextQualityFilter",
            QUALITY_MODEL,
            {"label": QUALITY_LABEL, "alpha": 3, "seed": 42},
        ),
    ]
}
#: ``keep`` of the quality case: one FastTextQualityFilter(seed=42), keep_document called once per document of
#: dev + adversarial in that order (numpy's global RNG, seeded once in the constructor).
QUALITY_KEEP_ORDER = "dev then adversarial, one draw per document"

ALL_CASES: dict[str, FilterCase | ModifierCase | FastTextCase] = {
    **FILTER_CASES,
    **MODIFIER_CASES,
    **FASTTEXT_CASES,
}
assert len(ALL_CASES) == len(FILTER_CASES) + len(MODIFIER_CASES) + len(FASTTEXT_CASES)


def kind_of(case: str) -> str:
    """``filter``, ``modifier`` or ``fasttext``."""
    if case in FILTER_CASES:
        return "filter"
    return "modifier" if case in MODIFIER_CASES else "fasttext"


# ---------------------------------------------------------------------------------------------- references


def nemo_ref(platform: str) -> str:
    """The NeMo reference file for a ``data.platform_key()``.

    It holds fastText scores (fasttext 0.9.3's floats can depend on the CPU, as fasttext-predict's do), so each
    platform has its own pinned file, like ``specs.lid_ref``; linux-aarch64 keeps the plain name.
    """
    return "parity/nemo-ref.parquet" if platform == "linux-aarch64" else f"parity/nemo-ref.{platform}.parquet"


ZH_KINDS = ("nonenglish:cmn_Hani", "edge")


def doc_indices(docs: str, meta: list[str]) -> list[int]:
    """Indices into dev + adversarial (``refs.all_docs()``) of a case's documents."""
    if docs == "all":
        return list(range(len(meta)))
    if docs == "zh":
        return [i for i, k in enumerate(meta) if k in ZH_KINDS]
    raise ValueError(docs)


def hash_seed_ok() -> bool:
    """This process hashes str with the seed the references were recorded with."""
    return os.environ.get(HASH_SEED_ENV) == HASH_SEED


def digest(text: str) -> str:
    """128-bit hex digest of a modifier's output."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def score_cell(value: Any) -> tuple[str, str]:
    """``(type name, exact text)`` of a score: ``repr`` for numbers, the string itself for FastTextLangId's."""
    return type(value).__name__, value if isinstance(value, str) else repr(value)


#: Columns of the reference table. ``score`` is ``score_cell``'s text; ``digest`` / ``length`` / ``changed``
#: describe a modifier's output; ``error`` / ``message`` the exception when ``outcome`` is "raise".
REF_COLUMNS = (
    "case",
    "id",
    "outcome",
    "type",
    "score",
    "keep",
    "error",
    "message",
    "digest",
    "length",
    "changed",
)
