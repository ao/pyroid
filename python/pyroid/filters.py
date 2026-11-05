# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
"""datatrove's quality filters in Rust, with the same decisions (``pyroid.filters``).

Each filter has datatrove's parameter names and defaults, as keyword
arguments or as a config object (``GopherQuality(...)`` etc.). Types and docs
are in ``filters.pyi``.
"""

from __future__ import annotations

import unicodedata
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

from pyroid._pyroid import filters as _native

__all__ = [
    "DATATROVE_VERSION",
    "STOP_WORDS",
    "C4Quality",
    "FilterResult",
    "FineWebQuality",
    "GopherQuality",
    "GopherRepetition",
    "apply",
    "c4_lines",
    "c4_quality",
    "c4_quality_reasons",
    "english_languages",
    "fineweb_quality",
    "fineweb_quality_reasons",
    "gopher_quality",
    "gopher_quality_reasons",
    "gopher_repetition",
    "gopher_repetition_reasons",
]

#: The datatrove release whose decisions this module reproduces.
DATATROVE_VERSION = "0.10.1"

#: datatrove's default Gopher stop words.
STOP_WORDS = ("the", "be", "to", "of", "and", "that", "have", "with")


class GopherQuality(NamedTuple):
    """``GopherQualityFilter``'s parameters and defaults (falsy disables a check)."""

    min_doc_words: float | None = 50
    max_doc_words: float | None = 100000
    min_avg_word_length: float | None = 3
    max_avg_word_length: float | None = 10
    max_symbol_word_ratio: float | None = 0.1
    max_bullet_lines_ratio: float | None = 0.9
    max_ellipsis_lines_ratio: float | None = 0.3
    max_non_alpha_words_ratio: float | None = 0.8
    min_stop_words: float | None = 2
    stop_words: Iterable[str] | None = None
    language: str = "eng"


class GopherRepetition(NamedTuple):
    """``GopherRepetitionFilter``'s parameters and defaults (falsy disables a check)."""

    dup_line_frac: float | None = 0.3
    dup_para_frac: float | None = 0.3
    dup_line_char_frac: float | None = 0.2
    dup_para_char_frac: float | None = 0.2
    top_n_grams: Sequence[tuple[int, float]] = ((2, 0.2), (3, 0.18), (4, 0.16))
    dup_n_grams: Sequence[tuple[int, float]] = (
        (5, 0.15),
        (6, 0.14),
        (7, 0.13),
        (8, 0.12),
        (9, 0.11),
        (10, 0.10),
    )
    language: str = "eng"


class C4Quality(NamedTuple):
    """``C4QualityFilter``'s parameters and defaults (``-1`` disables a count)."""

    split_paragraph: bool = True
    remove_citations: bool = True
    filter_no_terminal_punct: bool = True
    min_num_sentences: int = 5
    min_words_per_line: int = 3
    max_word_length: int = 1000
    filter_lorem_ipsum: bool = True
    filter_javascript: bool = True
    filter_curly_bracket: bool = True
    filter_policy: bool = True
    language: str = "eng"


class FineWebQuality(NamedTuple):
    """``FineWebQualityFilter``'s parameters and defaults."""

    line_punct_thr: float = 0.12
    line_punct_exclude_zero: bool = False
    stop_chars: str | Iterable[str] | None = None
    short_line_thr: float = 0.67
    short_line_length: int = 30
    char_duplicates_ratio: float = 0.01
    new_line_ratio: float = 0.3
    language: str = "eng"


Config = GopherQuality | GopherRepetition | C4Quality | FineWebQuality


class FilterResult(NamedTuple):
    """What :func:`apply` returns, each in the caller's container kind."""

    keep: Any
    reason: Any
    text: Any


_KIND: dict[type, str] = {
    GopherQuality: "gopher_quality",
    GopherRepetition: "gopher_repetition",
    C4Quality: "c4_quality",
    FineWebQuality: "fineweb_quality",
}

# datatrove filter class -> (config class, {config field: datatrove attribute})
_DATATROVE: dict[str, tuple[type, dict[str, str]]] = {
    "GopherQualityFilter": (GopherQuality, {}),
    "GopherRepetitionFilter": (GopherRepetition, {}),
    "C4QualityFilter": (C4Quality, {}),
    "FineWebQualityFilter": (FineWebQuality, {"short_line_thr": "short_line_threshold"}),
}

_SUPPORTED_UNICODE = ("14.0.0", "15.0.0", "15.1.0", "16.0.0", "17.0.0")
_warned_unicode = False


def _unicode_version() -> str:
    global _warned_unicode
    v = unicodedata.unidata_version
    if v in _SUPPORTED_UNICODE:
        return v
    if not _warned_unicode:
        _warned_unicode = True
        warnings.warn(
            f"pyroid.filters has no tables for Unicode {v}; using 17.0.0 "
            "(str.isalpha and punctuation categories may differ for new characters)",
            RuntimeWarning,
            stacklevel=3,
        )
    return "17.0.0"


def _num(name: str, v: object) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    raise TypeError(f"{name} must be a number, not {type(v).__name__}")


def _gate(name: str, v: object) -> float | None:
    """A threshold datatrove only applies when truthy (``if self.x and ...``)."""
    if not v:
        return None
    return _num(name, v)


def _ngrams(name: str, v: Iterable[tuple[int, float]]) -> list[tuple[int, float]]:
    out = []
    for item in v:
        n, frac = item
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"{name}: n must be an int, not {type(n).__name__}")
        if n < 1:
            raise ValueError(f"{name}: n must be at least 1, got {n}")
        out.append((n, _num(name, frac)))
    return out


def _language(v: object) -> str:
    if not isinstance(v, str):
        raise TypeError(f"language must be a str, not {type(v).__name__}")
    return v


def _params(cfg: Config) -> tuple[str, str, dict[str, Any]]:
    """Validate a config into ``(kind, language, params)`` for the native side."""
    kind = _KIND[type(cfg)]
    if isinstance(cfg, GopherQuality):
        stop = STOP_WORDS if cfg.stop_words is None else cfg.stop_words
        # Only str elements can ever match a word.
        params: dict[str, Any] = {
            f: _gate(f, getattr(cfg, f)) for f in cfg._fields if f not in ("stop_words", "language")
        }
        params["stop_words"] = sorted({w for w in set(stop) if isinstance(w, str)})
    elif isinstance(cfg, GopherRepetition):
        params = {f: _gate(f, getattr(cfg, f)) for f in cfg._fields[:4]}
        params["top_n_grams"] = _ngrams("top_n_grams", cfg.top_n_grams)
        params["dup_n_grams"] = _ngrams("dup_n_grams", cfg.dup_n_grams)
    elif isinstance(cfg, C4Quality):
        counts = ("min_num_sentences", "min_words_per_line", "max_word_length")
        params = {
            f: (_num(f, getattr(cfg, f)) if f in counts else bool(getattr(cfg, f)))
            for f in cfg._fields
            if f != "language"
        }
    else:
        stop_chars = cfg.stop_chars
        if isinstance(stop_chars, str):
            stop_chars = (stop_chars,)
        params = {
            "line_punct_thr": _num("line_punct_thr", cfg.line_punct_thr),
            "line_punct_exclude_zero": bool(cfg.line_punct_exclude_zero),
            "stop_chars": None if stop_chars is None else [str(s) for s in stop_chars],
            "short_line_thr": _num("short_line_thr", cfg.short_line_thr),
            "short_line_length": _num("short_line_length", cfg.short_line_length),
            "char_duplicates_ratio": _num("char_duplicates_ratio", cfg.char_duplicates_ratio),
            "new_line_ratio": _num("new_line_ratio", cfg.new_line_ratio),
        }
    return kind, _language(cfg.language), params


def _freeze(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


_cache: dict[Any, Any] = {}


def _pipeline(configs: Sequence[Config]) -> Any:
    steps = [_params(c) for c in configs]
    version = _unicode_version()
    key = (
        version,
        tuple(
            (k, lang, tuple(sorted((n, _freeze(v)) for n, v in p.items()))) for k, lang, p in steps
        ),
    )
    p = _cache.get(key)
    if p is None:
        p = _native._Pipeline(steps, version)
        if len(_cache) >= 64:
            _cache.clear()
        _cache[key] = p
    return p


def _config(cls: type, config: Any, kwargs: dict[str, Any]) -> Any:
    if config is None:
        return cls(**kwargs)
    if type(config).__name__ in _DATATROVE and not isinstance(config, tuple):
        config = _from_datatrove(config)
    if not isinstance(config, cls):
        raise TypeError(f"config must be a {cls.__name__}, not {type(config).__name__}")
    return config._replace(**kwargs) if kwargs else config


def _from_datatrove(obj: Any) -> Config:
    """A config from a datatrove filter instance (its attributes)."""
    name = type(obj).__name__
    if name not in _DATATROVE:
        raise TypeError(f"not a supported datatrove filter: {name}")
    cls, renames = _DATATROVE[name]
    values = {f: getattr(obj, renames.get(f, f)) for f in cls._fields}
    cfg: Config = cls(**values)
    return cfg


def _as_config(f: Any) -> Config:
    if isinstance(f, (GopherQuality, GopherRepetition, C4Quality, FineWebQuality)):
        return f
    return _from_datatrove(f)


def _run(
    texts: Any, cls: type, config: Any, kwargs: dict[str, Any], what: str, threads: Any
) -> Any:
    cfg = _config(cls, config, kwargs)
    return _native.run(texts, _pipeline([cfg]), what, threads=threads)


def gopher_quality(texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any) -> Any:
    """``GopherQualityFilter``: True for each kept document (see ``filters.pyi``)."""
    return _run(texts, GopherQuality, config, kwargs, "keep", threads)


def gopher_quality_reasons(
    texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any
) -> Any:
    """``GopherQualityFilter``: datatrove's drop reason per document, ``None`` if kept."""
    return _run(texts, GopherQuality, config, kwargs, "reason", threads)


def gopher_repetition(
    texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any
) -> Any:
    """``GopherRepetitionFilter``: True for each kept document."""
    return _run(texts, GopherRepetition, config, kwargs, "keep", threads)


def gopher_repetition_reasons(
    texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any
) -> Any:
    """``GopherRepetitionFilter``: drop reason per document, ``None`` if kept."""
    return _run(texts, GopherRepetition, config, kwargs, "reason", threads)


def c4_quality(texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any) -> Any:
    """``C4QualityFilter``: True for each kept document."""
    return _run(texts, C4Quality, config, kwargs, "keep", threads)


def c4_quality_reasons(
    texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any
) -> Any:
    """``C4QualityFilter``: drop reason per document, ``None`` if kept."""
    return _run(texts, C4Quality, config, kwargs, "reason", threads)


def c4_lines(texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any) -> Any:
    """``C4QualityFilter`` line removal: each document's ``doc.text`` afterwards."""
    return _run(texts, C4Quality, config, kwargs, "text", threads)


def fineweb_quality(
    texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any
) -> Any:
    """``FineWebQualityFilter``: True for each kept document."""
    return _run(texts, FineWebQuality, config, kwargs, "keep", threads)


def fineweb_quality_reasons(
    texts: Any, config: Any = None, /, *, threads: Any = None, **kwargs: Any
) -> Any:
    """``FineWebQualityFilter``: drop reason per document, ``None`` if kept."""
    return _run(texts, FineWebQuality, config, kwargs, "reason", threads)


def apply(texts: Any, filters: Iterable[Any], /, *, threads: Any = None) -> FilterResult:
    """Run several filters in one pass, chained as datatrove's executor does."""
    configs = [_as_config(f) for f in filters]
    keep, reason, text = _native.run(texts, _pipeline(configs), "all", threads=threads)
    return FilterResult(keep, reason, text)


def english_languages() -> list[str]:
    """The datatrove language codes pyroid tokenizes (``SpaCyTokenizer("en")``)."""
    return list(_native.english_languages())
