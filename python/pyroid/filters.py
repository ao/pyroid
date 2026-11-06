# Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
"""datatrove's quality filters in Rust, with the same decisions (``pyroid.filters``).

Each filter has datatrove's parameter names and defaults, as keyword
arguments or as a config object (``GopherQuality(...)`` etc.). Types and docs
are in ``filters.pyi``.

Languages pyroid does not tokenize natively can opt in to datatrove's own
tokenizer, run in Python (``python_tokenizer=True``, or ``language=`` a word
tokenizer object): the words come from Python, the filter math stays in Rust.
``pyroid.dedup`` shares that machinery (the private helpers below).
"""

from __future__ import annotations

import importlib.metadata
import sys
import threading
import unicodedata
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple

from pyroid._pyroid import FallbackWarning, PyroidError
from pyroid._pyroid import filters as _native

__all__ = [
    "DATATROVE_VERSION",
    "STOP_WORDS",
    "C4Quality",
    "FilterResult",
    "FineWebQuality",
    "GopherQuality",
    "GopherRepetition",
    "Language",
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
    "languages",
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
    language: Any = "eng"


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
    language: Any = "eng"


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
    language: Any = "eng"


class FineWebQuality(NamedTuple):
    """``FineWebQualityFilter``'s parameters and defaults."""

    line_punct_thr: float = 0.12
    line_punct_exclude_zero: bool = False
    stop_chars: str | Iterable[str] | None = None
    short_line_thr: float = 0.67
    short_line_length: int = 30
    char_duplicates_ratio: float = 0.01
    new_line_ratio: float = 0.3
    language: Any = "eng"


Config = GopherQuality | GopherRepetition | C4Quality | FineWebQuality
_CONFIG_TYPES = (GopherQuality, GopherRepetition, C4Quality, FineWebQuality)


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
        out.append((n, _num(f"{name} fraction", frac)))
    return out


def _language(v: object, c4: bool = False) -> Any:
    """A datatrove language code, or a word tokenizer object (datatrove's or duck-typed)."""
    if isinstance(v, str):
        return v
    needs = ("word_tokenize", "span_tokenize") if c4 else ("word_tokenize",)
    if all(callable(getattr(v, m, None)) for m in needs):
        return v
    raise TypeError(
        "language must be a datatrove language code (str) or a word tokenizer (an object with "
        f"{' and '.join(needs)}, such as a datatrove WordTokenizer), not {type(v).__name__}"
    )


def _params(cfg: Config) -> tuple[str, Any, dict[str, Any]]:
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
        if stop_chars is not None and not isinstance(stop_chars, (str, Iterable)):
            raise TypeError(
                f"stop_chars must be a str or an iterable of str, not {type(stop_chars).__name__}"
            )
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
    return kind, _language(cfg.language, isinstance(cfg, C4Quality)), params


def _freeze(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    return v


_cache: dict[Any, Any] = {}


def _native_pipeline(steps: Sequence[tuple[str, str | None, dict[str, Any]]]) -> Any:
    """The native pipeline of validated steps (``language=None``: words from Python)."""
    version = _unicode_version()
    key = (
        version,
        tuple(
            (k, lang, tuple(sorted((n, _freeze(v)) for n, v in p.items()))) for k, lang, p in steps
        ),
    )
    p = _cache.get(key)
    if p is None:
        for _, lang, _ in steps:
            if lang is not None:
                _native_language(lang)
        p = _native._Pipeline(list(steps), version)
        if len(_cache) >= 64:
            _cache.clear()
        _cache[key] = p
    return p


def _pipeline(configs: Sequence[Config]) -> Any:
    return _native_pipeline([_params(c) for c in configs])


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


# ---- languages -------------------------------------------------------------------


class Language(NamedTuple):
    """How datatrove 0.10.1 tokenizes a language code, and whether pyroid does natively."""

    tokenizer: str
    arg: str
    native: bool


_languages_cache: dict[str, dict[str, Language]] = {}


def languages() -> dict[str, Language]:
    """Every datatrove language code with its tokenizer (see ``filters.pyi``)."""
    version = _unicode_version()
    table = _languages_cache.get(version)
    if table is None:
        table = {
            code: Language(cls, arg, native)
            for code, cls, arg, native in _native.languages(version)
        }
        _languages_cache[version] = table
    return dict(table)


def english_languages() -> list[str]:
    """The datatrove language codes whose tokenizer is ``SpaCyTokenizer("en")``."""
    return list(_native.english_languages())


#: What each datatrove tokenizer needs: (description, pip packages), by (class, arg)
#: first, then by class.
_BACKEND_LIBS: dict[tuple[str, str], tuple[str, str]] = {
    ("SpaCyTokenizer", "zh"): ("spaCy's Chinese tokenizer with jieba", "spacy==3.8.18 jieba"),
    ("SpaCyTokenizer", "ja"): (
        "spaCy's Japanese tokenizer with SudachiPy",
        "spacy==3.8.18 sudachipy sudachidict-core",
    ),
    ("SpaCyTokenizer", "vi"): ("spaCy's Vietnamese tokenizer with pyvi", "spacy==3.8.18 pyvi"),
}
_CLASS_LIBS: dict[str, tuple[str, str]] = {
    "SpaCyTokenizer": ("spaCy", "spacy==3.8.18"),
    "StanzaTokenizer": ("Stanza", "stanza"),
    "IndicNLPTokenizer": ("the Indic NLP Library", "indic-nlp-library"),
    "KiwiTokenizer": ("Kiwi", "kiwipiepy"),
    "ThaiTokenizer": ("PyThaiNLP", "pythainlp"),
    "KhmerTokenizer": ("khmer-nltk", "khmer-nltk"),
    "LaoTokenizer": ("LaoNLP", "laonlp"),
    "TibetanTokenizer": ("botok", "botok"),
    "BurmeseTokenizer": ("pyidaungsu", "pyidaungsu-numpy2"),
    "WhitespaceTokenizer": ("spaCy's multi-language tokenizer", "spacy==3.8.18"),
}
#: datatrove plus what its base install lacks for the tokenizers and filters.
_DATATROVE_PIP = f"datatrove=={DATATROVE_VERSION} regex"


def _backend(code: str) -> tuple[str, str, bool]:
    """The datatrove tokenizer of ``code``: ``(class, arg, native)``; ValueError as datatrove."""
    b = _native.backend(code, _unicode_version())
    if b is None:
        raise ValueError(
            f"Language '{code}' doesn't have a tokenizer assigned. Pass in a WordTokenizer "
            "directly or update tokenizer_assignment.csv"
        )
    return b  # type: ignore[no-any-return]


def _describe(cls: str, arg: str) -> str:
    """``SpaCyTokenizer("zh")`` (spaCy's Chinese tokenizer with jieba)."""
    what = _BACKEND_LIBS.get((cls, arg), _CLASS_LIBS.get(cls, (cls, "")))[0]
    call = f'{cls}("{arg}")' if arg else f"{cls}()"
    return f"{call} ({what})"


def _pip(cls: str, arg: str) -> str:
    lib = _BACKEND_LIBS.get((cls, arg), _CLASS_LIBS.get(cls, ("", "")))[1]
    return f"pip install {_DATATROVE_PIP} {lib}".rstrip()


_NATIVE_OK: set[tuple[str, str]] = set()


def _native_language(code: str, math: str = "the filter math still runs in Rust") -> None:
    """Raise unless pyroid tokenizes ``code`` natively (datatrove's ValueError if unknown)."""
    version = _unicode_version()
    if (code, version) in _NATIVE_OK:
        return
    cls, arg, native = _backend(code)
    if not native:
        raise NotImplementedError(
            f"pyroid has no native word tokenizer for language {code!r}: datatrove tokenizes it "
            f"with {_describe(cls, arg)}. Pass python_tokenizer=True to tokenize it with "
            "datatrove's own tokenizer in Python, on the calling thread "
            f"(needs `{_pip(cls, arg)}`); "
            f"{math}."
        )
    _NATIVE_OK.add((code, version))


# ---- the Python-tokenizer path ---------------------------------------------------

_WARNED: set[str] = set()
_WARN_LOCK = threading.Lock()


def _caller_stacklevel() -> int:
    """``stacklevel`` for warnings.warn that points at the first frame outside pyroid."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and str(frame.f_globals.get("__name__", "")).startswith(
        "pyroid."
    ):
        frame = frame.f_back
        level += 1
    return level


def _warn_once(key: str, message: str) -> None:
    with _WARN_LOCK:
        if key in _WARNED:
            return
        _WARNED.add(key)
    warnings.warn(message, FallbackWarning, stacklevel=_caller_stacklevel())


_DT_MODULES: dict[str, Any] = {}


def _datatrove(module: str, what: str, pip: str) -> Any:
    """Import a datatrove module, or raise PyroidError saying what to install."""
    mod = _DT_MODULES.get(module)
    if mod is not None:
        return mod
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        raise PyroidError(
            f"{what} needs datatrove {DATATROVE_VERSION}, which is not installed (`{pip}`): {e}"
        ) from e
    try:
        version = importlib.metadata.version("datatrove")
    except importlib.metadata.PackageNotFoundError:  # importable, but not installed (a checkout)
        version = DATATROVE_VERSION
    if version != DATATROVE_VERSION:
        _warn_version(version)
    _DT_MODULES[module] = mod
    return mod


def _warn_version(version: str) -> None:
    with _WARN_LOCK:
        if "datatrove-version" in _WARNED:
            return
        _WARNED.add("datatrove-version")
    warnings.warn(
        f"pyroid is using datatrove {version}'s tokenizers and filters, not "
        f"{DATATROVE_VERSION}'s: results can differ from datatrove {DATATROVE_VERSION}",
        RuntimeWarning,
        stacklevel=_caller_stacklevel(),
    )


class _PyTokenizer(NamedTuple):
    """Words from Python: datatrove's tokenizer for a code, or the caller's object."""

    key: Any  # the language code, or the tokenizer object
    tok: Any  # what word_tokenize is called on
    describe: str  # "language 'x' with datatrove's ..." (for warnings and errors)
    native: bool  # pyroid has a native tokenizer for this code


def _load_tokenizer(language: Any, where: str) -> _PyTokenizer:
    """The Python tokenizer for ``language`` (a code or an object), loaded now."""
    if not isinstance(language, str):
        name = type(language).__qualname__
        return _PyTokenizer(language, language, f"with the word tokenizer {name}", False)
    cls, arg, native = _backend(language)
    describe = f"language {language!r} with datatrove's {_describe(cls, arg)}"
    pip = _pip(cls, arg)
    wt = _datatrove(
        "datatrove.utils.word_tokenizers", f"{where}: tokenizing {describe} in Python", pip
    )
    try:
        tok = wt.load_word_tokenizer(language)
        if isinstance(tok, wt.SpaCyTokenizer):
            tok.tokenizer  # noqa: B018  # spaCy imports a language's segmenter on first use
    except ImportError as e:
        raise PyroidError(
            f"{where}: tokenizing {describe} in Python needs packages that are not installed "
            f"(`{pip}`): {e}"
        ) from e
    return _PyTokenizer(language, tok, describe, native)


def _tokenize(
    tok: _PyTokenizer, texts: Sequence[str], where: str, index: Sequence[int] | None = None
) -> list[list[str]]:
    """``word_tokenize`` of each text, on the calling thread (``index``: for error notes)."""
    f: Callable[[str], Any] = tok.tok.word_tokenize
    out: list[list[str]] = []
    for j, t in enumerate(texts):
        try:
            words = f(t)
        except ImportError as e:
            raise PyroidError(
                f"{where}: tokenizing {tok.describe} needs packages that are not installed: {e}"
            ) from e
        except Exception as e:
            e.add_note(f"{where}: raised by the word tokenizer on document {_at(index, j)}")
            raise
        out.append(words if type(words) is list else list(words))
    return out


def _at(index: Sequence[int] | None, j: int) -> int:
    return j if index is None else index[j]


def _warn_tokenizer(func: str, tok: _PyTokenizer, how: str) -> None:
    native = (
        " pyroid tokenizes this language natively: without python_tokenizer=True it runs at "
        "Rust speed."
        if tok.native
        else ""
    )
    key = tok.key if isinstance(tok.key, str) else f"object:{type(tok.key).__qualname__}"
    _warn_once(
        f"{func}:{how}:{key}",
        f"pyroid.{func}: tokenizing {tok.describe} in Python, one document at a time on "
        f"the calling thread; {how}. Expect the tokenizer's speed, not pyroid's.{native} "
        "This warning is shown once.",
    )


_ADAPTER: list[Any] = []  # the WordTokenizer subclass for duck-typed tokenizers, made once
_ADAPTED: dict[int, Any] = {}


def _as_datatrove_tokenizer(language: Any, wt: Any) -> Any:
    """``language`` as datatrove's filters take it: a code or a ``WordTokenizer``."""
    if isinstance(language, (str, wt.WordTokenizer)):
        return language
    hit = _ADAPTED.get(id(language))
    if hit is not None and hit.inner is language:
        return hit
    if not _ADAPTER:

        class _Adapter(wt.WordTokenizer):  # type: ignore[misc, name-defined]
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def word_tokenize(self, text: str) -> Any:
                return self.inner.word_tokenize(text)

            def sent_tokenize(self, text: str) -> Any:
                return self.inner.sent_tokenize(text)

            def span_tokenize(self, text: str) -> Any:
                return self.inner.span_tokenize(text)

        _ADAPTER.append(_Adapter)
    adapter = _ADAPTER[0](language)
    _ADAPTED[id(language)] = adapter
    return adapter


_DT_FILTERS = {
    GopherQuality: ("gopher_quality_filter", "GopherQualityFilter"),
    GopherRepetition: ("gopher_repetition_filter", "GopherRepetitionFilter"),
    C4Quality: ("c4_filters", "C4QualityFilter"),
    FineWebQuality: ("fineweb_quality_filter", "FineWebQualityFilter"),
}


class _Datatrove:
    """datatrove's own filters, for the steps (or documents) pyroid hands to it."""

    def __init__(self, where: str, pip: str) -> None:
        self.where = where
        self.pip = pip
        self.filters: dict[int, Any] = {}

    def filter(self, i: int, cfg: Config) -> Any:
        f = self.filters.get(i)
        if f is None:
            module, name = _DT_FILTERS[type(cfg)]
            why = f"{self.where}: running datatrove's {name}"
            mod = _datatrove(f"datatrove.pipeline.filters.{module}", why, self.pip)
            wt = _datatrove("datatrove.utils.word_tokenizers", why, self.pip)
            kwargs = cfg._asdict()
            kwargs["language"] = _as_datatrove_tokenizer(cfg.language, wt)
            f = getattr(mod, name)(**kwargs)
            self.filters[i] = f
        return f

    def run(self, steps: Sequence[tuple[int, Config]], text: str, doc_id: int) -> tuple[Any, str]:
        """``(reason, text)`` after ``steps``: reason ``None`` if kept, else datatrove's."""
        data = _datatrove("datatrove.data", self.where, self.pip)
        base = _datatrove("datatrove.pipeline.filters.base_filter", self.where, self.pip)
        doc = data.Document(text=text, id=str(doc_id))
        for i, cfg in steps:
            f = self.filter(i, cfg)
            try:
                keep, reason = base.get_filter_result(f.filter(doc))
            except Exception as e:  # datatrove's exception, as datatrove raises it
                e.add_note(
                    f"{self.where}: raised by datatrove's {type(f).__name__} on document {doc_id}"
                )
                raise
            if not keep:
                return reason, doc.text
        return None, doc.text


def _run_python(
    texts: Any,
    configs: Sequence[Config],
    what: str,
    threads: Any,
    python_tokenizer: bool,
    func: str,
) -> Any:
    """The filters with words from Python tokenizers, in datatrove's step order.

    Consecutive non-C4 steps with the same tokenizer run as one native pass: first without
    words (documents a check drops before it needs words are done, untokenized, as in
    datatrove), then, for the others, with ``word_tokenize`` of their text. A C4 step on
    this path is datatrove's own ``C4QualityFilter`` (it rewrites the text, so later steps
    tokenize the new text). Consecutive native steps run as one native pipeline.
    """
    steps = [_params(c) for c in configs]
    # Each step's words: a native language code, or a Python tokenizer.
    tokenizers: dict[Any, _PyTokenizer] = {}
    sources: list[Any] = []
    for kind, lang, _ in steps:
        if isinstance(lang, str) and not python_tokenizer:
            _native_language(lang)
            sources.append(lang)
            continue
        key = lang if isinstance(lang, str) else ("object", id(lang))
        if key not in tokenizers:
            tokenizers[key] = _load_tokenizer(lang, f"pyroid.{func}")
        tok = tokenizers[key]
        sources.append(tok)
        if kind == "c4_quality":
            _warn_tokenizer(func, tok, "C4QualityFilter runs as datatrove's own filter, in Python")
        else:
            _warn_tokenizer(func, tok, "the filter math runs in Rust")
    pip = _DATATROVE_PIP
    for tok in tokenizers.values():
        if isinstance(tok.key, str):
            cls, arg, _ = _backend(tok.key)
            pip = _pip(cls, arg)
            break
    dt = _Datatrove(f"pyroid.{func}", pip)
    for i, ((kind, _, _), src) in enumerate(zip(steps, sources, strict=True)):
        if kind == "c4_quality" and not isinstance(src, str):
            dt.filter(i, configs[i])  # datatrove's C4QualityFilter, made now (or PyroidError)

    # Segments: ("native", [step indices]), ("words", [steps], tokenizer) or ("c4", [step]).
    segments: list[tuple[str, list[int], Any]] = []
    for i, ((kind, _, _), src) in enumerate(zip(steps, sources, strict=True)):
        if isinstance(src, str):
            seg_kind, seg_tok = "native", None
        elif kind == "c4_quality":
            seg_kind, seg_tok = "c4", src
        else:
            seg_kind, seg_tok = "words", src
        last = segments[-1] if segments else None
        if last is not None and seg_kind != "c4" and last[0] == seg_kind and last[2] is seg_tok:
            last[1].append(i)
        else:
            segments.append((seg_kind, [i], seg_tok))

    docs, shape = _native._to_list(texts, threads=threads)
    n = len(docs)
    cur: list[Any] = list(docs)
    reasons: list[str | None] = [None] * n
    alive = [d is not None for d in docs]
    for seg_kind, seg_steps, tok in segments:
        idx = [i for i in range(n) if alive[i]]
        if not idx:
            break
        sub = [cur[i] for i in idx]
        if seg_kind == "c4":
            (s,) = seg_steps
            status: list[Any] = []
            new: list[Any] = []
            for j, t in enumerate(sub):
                reason, out = dt.run([(s, configs[s])], t, idx[j])
                status.append(reason)
                new.append(None if out == t else out)
        else:
            if seg_kind == "native":
                p = _native_pipeline([steps[s] for s in seg_steps])
            else:
                p = _native_pipeline([(steps[s][0], None, steps[s][2]) for s in seg_steps])
            status, new = _native._run_list(sub, p, None, idx, threads=threads)
            need = [j for j, st in enumerate(status) if type(st) is int]
            if need:
                at = [idx[j] for j in need]
                words: list[Any] = _tokenize(tok, [sub[j] for j in need], f"pyroid.{func}", at)
                st2, _ = _native._run_list([sub[j] for j in need], p, words, at, threads=threads)
                for j, st in zip(need, st2, strict=True):
                    status[j] = st
        for j, i in enumerate(idx):
            if new[j] is not None:
                cur[i] = new[j]
            if status[j] is not None:
                alive[i] = False
                reasons[i] = status[j]
    keep = [None if d is None else alive[i] for i, d in enumerate(docs)]
    return _native._pack(shape, what, keep, reasons, cur, threads=threads)


def _run(
    texts: Any,
    cls: type,
    config: Any,
    kwargs: dict[str, Any],
    what: str,
    threads: Any,
    python_tokenizer: bool,
    func: str,
) -> Any:
    cfg = _config(cls, config, kwargs)
    if not python_tokenizer and isinstance(cfg.language, str):
        return _native.run(texts, _pipeline([cfg]), what, threads=threads)
    return _run_python(texts, [cfg], what, threads, python_tokenizer, f"filters.{func}")


def gopher_quality(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``GopherQualityFilter``: True for each kept document (see ``filters.pyi``)."""
    return _run(
        texts, GopherQuality, config, kwargs, "keep", threads, python_tokenizer, "gopher_quality"
    )


def gopher_quality_reasons(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``GopherQualityFilter``: datatrove's drop reason per document, ``None`` if kept."""
    return _run(
        texts,
        GopherQuality,
        config,
        kwargs,
        "reason",
        threads,
        python_tokenizer,
        "gopher_quality_reasons",
    )


def gopher_repetition(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``GopherRepetitionFilter``: True for each kept document."""
    return _run(
        texts,
        GopherRepetition,
        config,
        kwargs,
        "keep",
        threads,
        python_tokenizer,
        "gopher_repetition",
    )


def gopher_repetition_reasons(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``GopherRepetitionFilter``: drop reason per document, ``None`` if kept."""
    return _run(
        texts,
        GopherRepetition,
        config,
        kwargs,
        "reason",
        threads,
        python_tokenizer,
        "gopher_repetition_reasons",
    )


def c4_quality(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``C4QualityFilter``: True for each kept document."""
    return _run(texts, C4Quality, config, kwargs, "keep", threads, python_tokenizer, "c4_quality")


def c4_quality_reasons(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``C4QualityFilter``: drop reason per document, ``None`` if kept."""
    return _run(
        texts, C4Quality, config, kwargs, "reason", threads, python_tokenizer, "c4_quality_reasons"
    )


def c4_lines(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``C4QualityFilter`` line removal: each document's ``doc.text`` afterwards."""
    return _run(texts, C4Quality, config, kwargs, "text", threads, python_tokenizer, "c4_lines")


def fineweb_quality(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``FineWebQualityFilter``: True for each kept document."""
    return _run(
        texts, FineWebQuality, config, kwargs, "keep", threads, python_tokenizer, "fineweb_quality"
    )


def fineweb_quality_reasons(
    texts: Any,
    config: Any = None,
    /,
    *,
    threads: Any = None,
    python_tokenizer: bool = False,
    **kwargs: Any,
) -> Any:
    """``FineWebQualityFilter``: drop reason per document, ``None`` if kept."""
    return _run(
        texts,
        FineWebQuality,
        config,
        kwargs,
        "reason",
        threads,
        python_tokenizer,
        "fineweb_quality_reasons",
    )


def apply(
    texts: Any, filters: Iterable[Any], /, *, threads: Any = None, python_tokenizer: bool = False
) -> FilterResult:
    """Run several filters in one pass, chained as datatrove's executor does."""
    if isinstance(filters, _CONFIG_TYPES) or type(filters).__name__ in _DATATROVE:
        filters = [filters]  # one filter, not an iterable of them
    configs = [_as_config(f) for f in filters]
    if not python_tokenizer and all(isinstance(c.language, str) for c in configs):
        keep, reason, text = _native.run(texts, _pipeline(configs), "all", threads=threads)
    else:
        keep, reason, text = _run_python(
            texts, configs, "all", threads, python_tokenizer, "filters.apply"
        )
    return FilterResult(keep, reason, text)
